"""
Gripper geometry + contact set construction.

Gripper geometry
----------------
All points are in the gripper-local frame where:
  - Origin = TCP (wrist mounting point)
  - Z      = approach / insertion direction
  - Y      = finger opening direction

Adjust the three constants at the top when the physical gripper changes.

Contact set
-----------
Two contact sets are built — one per fingertip — then concatenated into a
single (Q, f_star) that the optimizer works with.  The left/right split is
returned separately for visualization or per-finger weighting.
"""

import numpy as np
import open3d as o3d
import torch
from utils.my_utils import fps_np


# ── Tune these for the actual hardware ──────────────────────────────────────
FINGER_HALF_WIDTH = 0.035   # half of max stroke (70 mm / 2)
FINGER_LENGTH     = 0.046   # finger protrusion along Z
TCP_OFFSET        = 0.067   # distance from wrist origin to fingertip plane
# ────────────────────────────────────────────────────────────────────────────

FINGER_BASE_Z = TCP_OFFSET - FINGER_LENGTH

# Six key points in gripper-local frame
#   0: palm centre   1: left tip   2: right tip
#   3: left base     4: right base  5: wrist origin
GRIPPER_POINTS_LOCAL = np.array([
    [0,  0,                 FINGER_BASE_Z],  # 0
    [0,  FINGER_HALF_WIDTH, TCP_OFFSET],     # 1 left  fingertip
    [0, -FINGER_HALF_WIDTH, TCP_OFFSET],     # 2 right fingertip
    [0,  FINGER_HALF_WIDTH, FINGER_BASE_Z],  # 3 left  finger base
    [0, -FINGER_HALF_WIDTH, FINGER_BASE_Z],  # 4 right finger base
    [0,  0,                 0],              # 5 wrist origin
], dtype=np.float64)

GRIPPER_EDGES = [
    (1, 3), (2, 4),  # fingers
    (3, 4),          # palm bar
    (3, 5), (4, 5),  # palm to wrist
    (0, 5),          # stem
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def fingertip_positions_world(gripper_pose):
    """Left and right fingertip positions in world frame.

    Args:
        gripper_pose: (4, 4) np array

    Returns:
        left_tip, right_tip: each (3,) np array
    """
    R, t = gripper_pose[:3, :3], gripper_pose[:3, 3]
    return R @ GRIPPER_POINTS_LOCAL[1] + t, R @ GRIPPER_POINTS_LOCAL[2] + t


def gripper_points_world(gripper_pose):
    """All six gripper key points in world frame.

    Returns:
        pts: (6, 3) np array
    """
    R, t = gripper_pose[:3, :3], gripper_pose[:3, 3]
    return (R @ GRIPPER_POINTS_LOCAL.T).T + t


# ---------------------------------------------------------------------------
# Contact set construction
# ---------------------------------------------------------------------------

def _contact_set_near_point(fusion, object_pcd, anchor_world, n_points,
                             distance_thresh, gripper_pose_inv, device, label):
    """Sample n_points near anchor_world, query descriptors, convert to local frame."""
    dists = np.linalg.norm(object_pcd - anchor_world, axis=1)
    near_pts = object_pcd[dists < distance_thresh]

    if near_pts.shape[0] >= n_points:
        near_pts, _, _ = fps_np(near_pts, n_points)
        print(f"  [{label}] {n_points} pts within {distance_thresh:.3f} m of fingertip")
    else:
        print(f"  [{label}] Only {near_pts.shape[0]} pts within "
              f"{distance_thresh:.3f} m — using closest {n_points} surface pts")
        near_pts = object_pcd[np.argsort(dists)[:n_points]]

    pts_tensor = torch.from_numpy(near_pts).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = fusion.batch_eval(pts_tensor, return_names=['dino_feats'])
    f_star = out['dino_feats'].cpu().numpy()

    ones = np.ones((near_pts.shape[0], 1))
    Q = (gripper_pose_inv @ np.hstack([near_pts, ones]).T).T[:, :3]

    return Q, f_star, near_pts


def build_contact_set(fusion_ref, object_pcd, gripper_pose, n_points,
                      distance_thresh=0.008, device='cuda'):
    """
    Build two contact sets (one per fingertip) and return them concatenated.

    Args:
        fusion_ref      : Fusion object for the reference scene
        object_pcd      : (M, 3) object surface points in world frame
        gripper_pose    : (4, 4) demo gripper pose in world frame
        n_points        : points per fingertip (total = 2 * n_points)
        distance_thresh : search radius around each fingertip (metres)
        device          : torch device

    Returns:
        Q           : (2N, 3)        query points in gripper-local frame
        f_star      : (2N, feat_dim) reference descriptors
        contact_pts : (2N, 3)        contact points in world frame
        Q_left      : (N, 3)         left-fingertip subset (local frame)
        Q_right     : (N, 3)         right-fingertip subset (local frame)
    """
    left_tip, right_tip = fingertip_positions_world(gripper_pose)
    T_inv = np.linalg.inv(gripper_pose)

    Q_L, f_L, pts_L = _contact_set_near_point(
        fusion_ref, object_pcd, left_tip,  n_points,
        distance_thresh, T_inv, device, "left fingertip")

    Q_R, f_R, pts_R = _contact_set_near_point(
        fusion_ref, object_pcd, right_tip, n_points,
        distance_thresh, T_inv, device, "right fingertip")

    Q           = np.concatenate([Q_L, Q_R], axis=0)
    f_star      = np.concatenate([f_L, f_R], axis=0)
    contact_pts = np.concatenate([pts_L, pts_R], axis=0)

    return Q, f_star, contact_pts, Q_L, Q_R


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_contact_set(object_pcd, contact_pts, gripper_pose,
                          title="Contact Set"):
    """
    Open3D visualization to verify the contact set.

    - Object point cloud : gray
    - Gripper wireframe  : gray lines
    - Contact points     : blue
    - Fingertips (L + R) : red
    """
    geoms = []

    # Object point cloud — gray
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_pcd)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])
    geoms.append(pcd)

    # Contact points — blue (as spheres)
    sphere_radius = 0.001 # adjust as needed
    for p in contact_pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=10)
        s.translate(p)
        s.paint_uniform_color([0.15, 0.4, 1.0])
        geoms.append(s)

    # Gripper wireframe — gray lines
    pts_w = gripper_points_world(gripper_pose)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_w)
    line_set.lines = o3d.utility.Vector2iVector(GRIPPER_EDGES)
    line_set.paint_uniform_color([0.5, 0.5, 0.5])
    geoms.append(line_set)

    # Fingertips — red spheres
    left_tip, right_tip = fingertip_positions_world(gripper_pose)
    for tip in (left_tip, right_tip):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(tip)
        sphere.paint_uniform_color([1.0, 0.15, 0.15])
        geoms.append(sphere)

    o3d.visualization.draw_geometries(geoms, window_name=title,
                                      width=1024, height=768)
