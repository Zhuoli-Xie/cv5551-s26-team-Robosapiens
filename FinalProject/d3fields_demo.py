"""
D3Fields Demo — standalone script converted from D3Fields_demo_colab.ipynb

Builds D3Fields representation from multi-view RGBD data, then runs
batch_eval to obtain DINO features, instance masks, and colors on mesh vertices.

Usage:
    python d3fields_demo.py
"""

import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import cv2
import numpy as np
import torch

from fusion import Fusion, create_init_grid

# ─── Configuration ───────────────────────────────────────────────────────────
t = 0
num_cam = 3
step = 0.004

boundaries = {
    'x_lower': -0.2,
    'x_upper': 0.4,
    'y_lower': -0.4,
    'y_upper': 0.3,
    'z_lower': -0.2,
    'z_upper': 0.02,
}

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mugs')
pca_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_model', 'mug.pkl')
query_texts = ['mug']
query_thresholds = [0.3]
device = 'cuda'

# ─── Load data ───────────────────────────────────────────────────────────────
print('Loading data...')
colors = np.stack([
    cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png'))
    for i in range(num_cam)
], axis=0)  # [N, H, W, C]

depths = np.stack([
    cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'), cv2.IMREAD_ANYDEPTH)
    for i in range(num_cam)
], axis=0) / 1000.0  # [N, H, W]

extrinsics = np.stack([
    np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy'))
    for i in range(num_cam)
])

cam_param = np.stack([
    np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy'))
    for i in range(num_cam)
])

intrinsics = np.zeros((num_cam, 3, 3))
intrinsics[:, 0, 0] = cam_param[:, 0]
intrinsics[:, 1, 1] = cam_param[:, 1]
intrinsics[:, 0, 2] = cam_param[:, 2]
intrinsics[:, 1, 2] = cam_param[:, 3]
intrinsics[:, 2, 2] = 1

obs = {
    'color': colors,
    'depth': depths,
    'pose': extrinsics[:, :3],  # (N, 3, 4)
    'K': intrinsics,
}

# ─── Build D3Fields ──────────────────────────────────────────────────────────
print('Creating Fusion (skip_xmem=True to save GPU memory)...')
fusion = Fusion(num_cam=num_cam, feat_backbone='dinov2', skip_xmem=True)

print('Updating observation...')
fusion.update(obs)

print('Running grounded SAM for instance masks...')
fusion.text_queries_for_inst_mask_no_track(query_texts, query_thresholds, boundaries=boundaries)

# ─── Evaluate on grid → extract mesh ─────────────────────────────────────────
print('Evaluating distance field on grid...')
init_grid, grid_shape = create_init_grid(boundaries, step)
init_grid = init_grid.to(device=device, dtype=torch.float32)

with torch.no_grad():
    out = fusion.batch_eval(init_grid, return_names=[])

print('Extracting mesh...')
vertices, triangles = fusion.extract_mesh(init_grid, out, grid_shape)

# ─── Evaluate features / mask / color on mesh vertices ────────────────────────
print('Evaluating DINO features, mask, and color on mesh vertices...')
vertices_tensor = torch.from_numpy(vertices).to(device, dtype=torch.float32)

with torch.no_grad():
    out = fusion.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask', 'color_tensor'])

# ─── Extract mug-only point cloud ─────────────────────────────────────────────
mask = out['mask'].detach().cpu().numpy()            # (N, num_instance) one-hot
instance_id = mask.argmax(axis=-1)                   # 0 = background, 1+ = objects
mug_mask = instance_id > 0                           # all non-background vertices

mug_points = vertices[mug_mask]                      # (M, 3)
mug_colors = out['color_tensor'][mug_mask].detach().cpu().numpy()  # (M, 3) float 0-1
mug_feats  = out['dino_feats'][mug_mask].detach().cpu().numpy()    # (M, feat_dim)

print(f'\nMug point cloud: {mug_points.shape[0]} points')
print(f'  points:   {mug_points.shape}')
print(f'  colors:   {mug_colors.shape}')
print(f'  features: {mug_feats.shape}')

# Save as colored .ply
import open3d as o3d

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vis_output')
os.makedirs(output_dir, exist_ok=True)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(mug_points)
pcd.colors = o3d.utility.Vector3dVector(mug_colors[..., ::-1].copy())  # BGR→RGB
out_path = os.path.join(output_dir, 'mug_pointcloud.ply')
o3d.io.write_point_cloud(out_path, pcd)
print(f'Saved to {out_path}')

# Also save features alongside for downstream use
np.savez(os.path.join(output_dir, 'mug_data.npz'),
         points=mug_points, colors=mug_colors, dino_feats=mug_feats)
print(f'Saved points + features to {os.path.join(output_dir, "mug_data.npz")}')
