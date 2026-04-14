"""Scene data loading: cameras, depth, masks, gripper poses, point clouds."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from .fusion import Fusion
from utils.draw_utils import aggr_point_cloud_from_data


def _load_depth(path):
    """Load depth from .npz (float32, metres) or 16-bit .png (uint16, millimetres)."""
    path = str(path)
    npz_path = path.replace('.png', '.npz')
    if os.path.exists(npz_path):
        return np.load(npz_path)['depth']
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Depth file not found: {path} (tried .npz and .png)")
    return img.astype(np.float32) / 1000.0


def count_cameras(data_path):
    """Auto-detect number of camera_* directories."""
    data_path = Path(data_path)
    i = 0
    while (data_path / f"camera_{i}").is_dir():
        i += 1
    return i


def load_scene_data(data_path, t, num_cam=None):
    """Load color, depth, intrinsics, extrinsics for a given timestep.

    Returns:
        obs       : dict with keys 'color', 'depth', 'pose', 'K'
        colors    : (N, H, W, 3) uint8 BGR
        depths    : (N, H, W) float32 metres
        intrinsics: (N, 3, 3) float64
        extrinsics: (N, 4, 4) float64
        num_cam   : int
    """
    if num_cam is None:
        num_cam = count_cameras(data_path)
        print(f"Auto-detected {num_cam} camera(s) in {data_path}")

    colors = np.stack([
        cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png'))
        for i in range(num_cam)], axis=0)
    depths = np.stack([
        _load_depth(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'))
        for i in range(num_cam)], axis=0)

    extrinsics = np.stack([
        np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy'))
        for i in range(num_cam)])
    cam_param = np.stack([
        np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy'))
        for i in range(num_cam)])

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
    return obs, colors, depths, intrinsics, extrinsics, num_cam


def load_masks(data_path, t, num_cam):
    """Load pre-computed masks.

    Returns:
        masks: (N, H, W) bool array -- True where object is detected.
    """
    mask_list = []
    for i in range(num_cam):
        mask_path = os.path.join(data_path, f'camera_{i}', 'mask', f'{t}.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(
                f"Mask not found: {mask_path}\n"
                "Run detection pipeline first to generate masks.")
        mask_list.append(mask > 0)
    return np.stack(mask_list, axis=0)


def load_gripper_pose(path):
    """Load a gripper pose from a 4x4 text file or npy."""
    path = str(path)
    if path.endswith('.npy'):
        return np.load(path)
    return np.loadtxt(path)


def load_scene(data_path, timestep, device, feat_backbone='dinov2'):
    """Load scene data + pre-computed masks, build Fusion with D3Fields.

    Returns:
        fusion, colors, depths, intrinsics, extrinsics, masks, num_cam
    """
    obs, colors, depths, intrinsics, extrinsics, num_cam = \
        load_scene_data(data_path, timestep)

    print(f"Loading pre-computed masks from {data_path}...")
    masks = load_masks(data_path, timestep, num_cam)

    print(f"Building D3Fields (Fusion) with {num_cam} cameras "
          f"(backbone={feat_backbone})...")
    fusion = Fusion(num_cam=num_cam, feat_backbone=feat_backbone,
                    device=device, skip_xmem=True)
    fusion.update(obs)

    return fusion, colors, depths, intrinsics, extrinsics, masks, num_cam


def build_object_pcd(colors, depths, intrinsics, extrinsics, boundaries,
                     masks=None, z_max=None):
    """Aggregate multi-view point cloud within workspace boundaries.

    Args:
        masks: optional (N, H, W) bool array to filter points
        z_max: if set, discard points with z > z_max
    """
    pcd_o3d = aggr_point_cloud_from_data(
        colors[..., ::-1], depths, intrinsics, extrinsics,
        downsample=True, masks=masks, boundaries=boundaries)
    _, inlier_idx = pcd_o3d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_o3d = pcd_o3d.select_by_index(inlier_idx)
    # _, inlier_idx = pcd_o3d.remove_radius_outlier(nb_points=5, radius=0.02)
    # pcd_o3d = pcd_o3d.select_by_index(inlier_idx)
    # if z_max is not None:
    #     pts = np.asarray(pcd_o3d.points)
    #     pcd_o3d = pcd_o3d.select_by_index(np.where(pts[:, 2] <= z_max)[0])
    return np.asarray(pcd_o3d.points)
