import numpy as np
import cv2
import pyrender
import trimesh
import util


def render_delta(vertices_org, vertices_aug, faces, size):
    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5), bg_color=(0.0, 0.0, 0.5))

    vertices_aug = vertices_aug[0].detach().cpu().numpy().squeeze()
    vertex_delta = vertices_org - vertices_aug
    vertex_delta = 0.5 + vertex_delta
    vertex_delta[:, 2] = 0.0
    tri_mesh = trimesh.Trimesh(vertices_aug, faces, vertex_colors=vertex_delta)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh)

    cam = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 1.0
    scene.add(cam, pose=cam_pose)

    r = pyrender.OffscreenRenderer(size[0], size[1])
    delta, _ = r.render(scene, flags=pyrender.RenderFlags.FLAT)
    
    mask = np.array(delta[:, :, 2] == 0, dtype=np.uint8)
    delta = delta / 255 - 0.5
    delta = (0.5*size[0], -0.5*size[1]) * delta[..., :2]

    return delta, mask

def render_mesh(flame, params, size, with_landmarks=False):
    scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0), ambient_light=(0.1, 0.1, 0.1))

    vertices, landmarks2d, _ = flame(shape_params=params['shape'], expression_params=params['exp'], pose_params=params['pose'])

    vertices = util.batch_orth_proj(vertices, params['cam']);
    vertices = vertices[0].detach().cpu().numpy().squeeze()
    tri_mesh = trimesh.Trimesh(vertices, flame.faces)
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    scene.add(mesh)

    if with_landmarks:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.0, 0.0, 1.0, 1.0]
        landmarks2d = util.batch_orth_proj(landmarks2d, params['cam']);
        landmarks2d = landmarks2d[0].detach().cpu().numpy().squeeze()
        tfs = np.tile(np.eye(4), (len(landmarks2d), 1, 1))
        tfs[:, :3, 3] = landmarks2d
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

    cam = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = 1.0
    scene.add(cam, pose=cam_pose)

    light = pyrender.light.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
    scene.add(light, pose=cam_pose)

    r = pyrender.OffscreenRenderer(size[0], size[1])
    render, _ = r.render(scene)
    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)

    return render