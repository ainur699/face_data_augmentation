import cv2
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from itertools import count
import util
from renderer import render_mesh


class Fitting(object):
    def __init__(self, config, flame, device='cuda'):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.config = config
        self.device = device
        
        self.flame = flame

    def optimize(self, images, landmarks, savefolder=None):
        bz = images.shape[0]
        shape = nn.Parameter(torch.zeros(bz, self.config.shape_params).float().to(self.device))
        exp = nn.Parameter(torch.zeros(bz, self.config.expression_params).float().to(self.device))
        pose = nn.Parameter(torch.zeros(bz, self.config.pose_params).float().to(self.device))
        cam = torch.zeros(bz, self.config.camera_params); cam[:, 0] = 5.
        cam = nn.Parameter(cam.float().to(self.device))

        e_opt_rigid = torch.optim.Adam(
            [pose, cam],
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_opt = torch.optim.Adam(
            [shape, exp, pose, cam], 
            lr=self.config.e_lr,
            weight_decay=self.config.e_wd
        )
        e_scheduler = torch.optim.lr_scheduler.MultiStepLR(e_opt, [100], gamma=0.1)

        gt_landmark = landmarks

        # rigid fitting of pose and camera with 51 static face landmarks,
        # this is due to the non-differentiable attribute of contour landmarks trajectory
        for k in count():
            losses = {}
            _, landmarks2d, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, 17:, :2], gt_landmark[:, 17:, :2]) * self.config.w_lmks

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt_rigid.zero_grad()
            all_loss.backward()
            e_opt_rigid.step()

            # visualize
            if savefolder and k % 10 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
                
                grid = torch.cat(list(grids.values()), 2)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                cv2.imwrite('{}/optim_landmarks_{}.png'.format(savefolder, k), grid_image)

                params = {'shape': shape.detach(), 'exp': exp.detach(), 'pose': pose.detach(), 'cam': cam.detach()}
                render = render_mesh(self.flame, params, (self.config.image_size, self.config.image_size), with_landmarks=False)
                cv2.imwrite('{}/optim_mesh_{}.png'.format(savefolder, k), render)

            if all_loss < self.config.rigid_loss_target or k > 300:
                break

        # non-rigid fitting of all the parameters with 68 face landmarks, photometric loss and regularization terms.
        for l in count(k):
            losses = {}
            _, landmarks2d, _ = self.flame(shape_params=shape, expression_params=exp, pose_params=pose)
            landmarks2d = util.batch_orth_proj(landmarks2d, cam);
            landmarks2d[..., 1:] = - landmarks2d[..., 1:]

            losses['landmark'] = util.l2_distance(landmarks2d[:, :, :2], gt_landmark[:, :, :2]) * self.config.w_lmks
            losses['shape_reg'] = (torch.sum(shape ** 2) / 2) * self.config.w_shape_reg  # *1e-4
            losses['expression_reg'] = (torch.sum(exp ** 2) / 2) * self.config.w_expr_reg  # *1e-4
            losses['pose_reg'] = (torch.sum(pose ** 2) / 2) * self.config.w_pose_reg

            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss
            e_opt.zero_grad()
            all_loss.backward()
            e_opt.step()
            e_scheduler.step()

            # visualize
            if savefolder and l % 10 == 0:
                grids = {}
                visind = range(bz)  # [0]
                grids['images'] = torchvision.utils.make_grid(images[visind]).detach().cpu()
                grids['landmarks_gt'] = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks[visind]))
                grids['landmarks2d'] = torchvision.utils.make_grid(util.tensor_vis_landmarks(images[visind], landmarks2d[visind]))
            
                grid = torch.cat(list(grids.values()), 2)
                grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
                grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
                cv2.imwrite('{}/optim_landmarks_{}.png'.format(savefolder, l), grid_image)

                params = {'shape': shape.detach(), 'exp': exp.detach(), 'pose': pose.detach(), 'cam': cam.detach()}
                render = render_mesh(self.flame, params, (self.config.image_size, self.config.image_size), with_landmarks=False)
                cv2.imwrite('{}/optim_mesh_{}.png'.format(savefolder, l), render)

            if all_loss < self.config.loss_target or l > k + 300:
                break

        single_params = {
            'shape': shape.detach(),
            'exp': exp.detach(),
            'pose': pose.detach(),
            'cam': cam.detach(),
        }
        return single_params

    def run(self, image, landmark, savefolder=None):
        # The implementation is potentially able to optimize with images(batch_size>1),
        # here we show the example with a single image fitting
        images = []
        landmarks = []

        image = image.astype(np.float32) / 255.
        image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
        images.append(torch.from_numpy(image[None, :, :, :]).to(self.device))

        landmark = landmark.astype(np.float32)
        landmark[:, 0] = landmark[:, 0] / float(image.shape[2]) * 2 - 1
        landmark[:, 1] = landmark[:, 1] / float(image.shape[1]) * 2 - 1
        landmarks.append(torch.from_numpy(landmark)[None, :, :].float().to(self.device))

        images = torch.cat(images, dim=0)
        images = F.interpolate(images, [self.image_size, self.image_size])

        landmarks = torch.cat(landmarks, dim=0)

        # optimize
        single_params = self.optimize(images, landmarks, savefolder)

        return single_params