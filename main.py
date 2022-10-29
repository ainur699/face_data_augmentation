import os
from glob import glob
from tqdm import tqdm
import argparse
import random

import cv2
import dlib
import torch
import math
import numpy as np

from FLAME import FLAME
from landmark_detection import get_landmarks, draw_face_landmarks
from fitting import Fitting
from renderer import render_delta, render_mesh
import util

config = {
        # FLAME
        'flame_model_path': './data/generic_model.pkl',  # acquire it from FLAME project page
        'flame_lmk_embedding_path': './data/landmark_embedding.npy',
        'shape_predictor': './data/shape_predictor_68_face_landmarks.dat',
        'aug_shape_params': './data/aug_shape_params.pt',
        'camera_params': 3,
        'shape_params': 100,
        'expression_params': 50,
        'pose_params': 6,
        'tex_params': 50,

        'batch_size': 1,
        'image_size': 512,
        'e_lr': 0.01,
        'e_wd': 0.0001,

        # weights of losses and reg terms
        'w_lmks': 1,
        'w_shape_reg': 1e-4,
        'w_expr_reg': 1e-4,
        'w_pose_reg': 0,

        'rigid_loss_target': 0.02,
        'loss_target': 0.012,
    }

def extrapolate_delta(delta, mask, eye_dist):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inp_mask = 1 - mask
    inp_mask = cv2.dilate(inp_mask, kernel)

    delta_x = cv2.inpaint(delta[..., 0].astype(np.float32), inp_mask, 5, cv2.INPAINT_TELEA)
    delta_y = cv2.inpaint(delta[..., 1].astype(np.float32), inp_mask, 5, cv2.INPAINT_TELEA)
    delta = np.stack([delta_x, delta_y], axis=2)

    dist = cv2.distanceTransform(inp_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    d = 0.75 * eye_dist
    dist = np.clip(dist, 0, d)
    warp_decrease = 0.5 + 0.5 * np.cos(math.pi * dist / d)

    return warp_decrease[..., None] * delta

def square_image(img):
    h, w, _ = img.shape
    m = max(h, w)

    pad_left = (m - w) // 2
    pad_right = m - w - pad_left
    pad_top = (m - h) // 2
    pad_bottom = m - h - pad_top

    pad = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    img = np.pad(img, pad)

    return img, pad

def get_face_mask(pt, size):
    pt = pt[0, :, :2].cpu().numpy()
    pt = (size * (0.5 * pt + 0.5)).astype(np.float32)
    pt_top = 2 * pt[29] - pt[8]

    contour = np.concatenate([pt[:17], pt_top[None]], axis=0)
    e = cv2.fitEllipse(contour)

    center = (int(e[0][0]), int(e[0][1]))
    # 0.45 instead of 0.5 to make mask a bit smaller
    axes = (int(0.45 * e[1][0]), int(0.45 * e[1][1]))
    angle = int(e[2])

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask = cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)

    return mask

def load_landmarks(path):
    landmarks = np.load(path) if os.path.exists(path) else None

    return landmarks

def augment_landmarks(pts, mapxy):
    h, w, _ = mapxy.shape
    grid = np.stack(np.meshgrid(range(w), range(h)), axis=2)

    dst_pst = []

    for pt in pts:
        src = np.zeros((h, w), dtype=np.uint8)
        src = cv2.circle(src, (int(pt[0]), int(pt[1])), 2, 255, -1)
        dst = cv2.remap(src, mapxy, None, interpolation=cv2.INTER_LINEAR)

        dst_pt = (dst[..., None] * grid).sum(axis=(0,1)) / dst.sum(axis=(0,1))
        dst_pst.append(dst_pt)

    dst_pst = np.array(dst_pst, dtype=np.float32)
    return dst_pst

def load_mask(path):
    mask = cv2.imread(path)

    return mask

def augment_mask(mask, mapxy):
    mask = cv2.remap(mask, mapxy, None, interpolation=cv2.INTER_LINEAR)

    return mask

def create_augs(config, img, landmarks, mask, aug_shape_params, predictor, flame, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    img, pad = square_image(img)
    
    # detect landmarks
    pts = get_landmarks(img, predictor)
    if pts is None:
        return

    # optimize 3DMM and save params
    param_path = os.path.join(dst_root, 'params.pt')
    if os.path.exists(param_path):
        params = torch.load(param_path)
    else:
        fitting = Fitting(config, flame, device=device)
        params = fitting.run(img, pts, savefolder=None)
        torch.save(params, param_path)

    vertices_org, _, _ = flame(shape_params=params['shape'], expression_params=params['exp'], pose_params=params['pose'])
    vertices_org = util.batch_orth_proj(vertices_org, params['cam']);
    vertices_org = vertices_org[0].detach().cpu().numpy().squeeze()

    for i, shape in enumerate(aug_shape_params):
        vertices_aug, landmarks2d, _ = flame(shape_params=shape.to(device), expression_params=params['exp'], pose_params=params['pose'])
        vertices_aug = util.batch_orth_proj(vertices_aug, params['cam']);
        landmarks2d = util.batch_orth_proj(landmarks2d, params['cam']);
        landmarks2d[..., 1:] = -landmarks2d[..., 1:]

        # render 3DMM delta
        h, w, _ = img.shape
        delta, delta_mask = render_delta(vertices_org, vertices_aug, flame.faces, (w, h))

        face_mask = get_face_mask(landmarks2d, (w, h))
        face_mask = delta_mask * face_mask

        eye_dist = cv2.norm(pts[36] - pts[45])
        delta = extrapolate_delta(delta, face_mask, eye_dist)

        # calculate map
        grid = np.stack(np.meshgrid(range(w), range(h)), axis=2)
        mapxy = (grid + delta).astype(np.float32)
        mapxy = cv2.blur(mapxy, (11, 11))

        aug_img = cv2.remap(img, mapxy, None, interpolation=cv2.INTER_LINEAR)

        aug_img = aug_img[pad[0][0]:h-pad[0][1], pad[1][0]:w-pad[1][1]]
        cv2.imwrite(os.path.join(dst_root, f'aug_{i}.png'), aug_img)

        mapxy = mapxy - np.array((pad[1][0], pad[0][0]), dtype=np.float32)
        mapxy = mapxy[pad[0][0]:h-pad[0][1], pad[1][0]:w-pad[1][1]]

        if landmarks is not None:
            aug_landmarks = augment_landmarks(landmarks, mapxy)
            np.save(os.path.join(dst_root, f'aug_landmarks_{i}.npy'), aug_landmarks)

        if mask is not None:
            aug_mask = augment_mask(mask, mapxy)
            cv2.imwrite(os.path.join(dst_root, f'aug_mask_{i}.png'), aug_mask)
 
        if debug and mask is not None and landmarks is not None:
            visual = 0.4*aug_img + 0.6*aug_mask
            visual = draw_face_landmarks(visual, aug_landmarks)
            cv2.imwrite(os.path.join(dst_root, f'visual_{i}.png'), visual)

            params['shape'] = shape.to(device)
            render = render_mesh(flame, params, (img.shape[1], img.shape[0]))
            cv2.imwrite(os.path.join(dst_root, f'mesh_{i}.png'), render)

if __name__ == '__main__':
    device = "cuda"
    config = util.dict2obj(config)
    debug = False # render extra images to check correctness of method

    parser = argparse.ArgumentParser(description='Face Image Augmentation')
    parser.add_argument('--source_dir', type=str, default='./images', help='directory with source images')
    parser.add_argument('--mask_dir', type=str, default='./images/masks', help='directory with masks')
    parser.add_argument('--landmarks_dir', type=str, default='./images/landmarks', help='directory with landmarks')
    parser.add_argument('--dst_dir', type=str, default='./results', help='directory where to save results')
    parser.add_argument('--aug_num', type=int, default=10, help='number of generated augmentations')

    args = parser.parse_args()
    
    predictor = dlib.shape_predictor(config.shape_predictor)
    flame = FLAME(config).to(device)
    aug_shape_params = torch.load(config.aug_shape_params)

    img_paths = glob(os.path.join(args.source_dir, '*.*'))

    for image_path in tqdm(img_paths):
        dst_root = os.path.join(args.dst_dir, os.path.splitext(os.path.basename(image_path))[0])
        
        # load image
        img = cv2.imread(image_path)
        if img is None:
            continue

        # load landmark
        if args.landmarks_dir is not None:
            landmarks_path = glob(os.path.join(args.landmarks_dir, os.path.splitext(os.path.basename(image_path))[0]+'.*'))
            landmarks = load_landmarks(landmarks_path[0]) if landmarks_path else None

        # load mask
        if args.mask_dir is not None:
            mask_path = glob(os.path.join(args.mask_dir, os.path.splitext(os.path.basename(image_path))[0]+'.*'))
            mask = load_mask(mask_path[0]) if mask_path else None

        # choose random shape augmentations
        augs = random.sample(aug_shape_params, min(args.aug_num, len(aug_shape_params)))

        create_augs(config, img, landmarks, mask, augs, predictor, flame, dst_root)