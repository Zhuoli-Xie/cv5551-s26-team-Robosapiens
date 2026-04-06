"""
Grasp Pose Search via D3Fields Descriptor Matching.

Given a demonstration grasp (contact set + descriptors), finds the optimal
SE(3) gripper pose in a new scene by minimizing:

    C(T) = Σ_i [ w_f * ||f_new(T·q_i) - f_i*||^2  +  w_d * d_new(T·q_i) ]

where q_i are query points in gripper-local frame, f_i* are reference
descriptors, and d_new is the signed distance field in the new scene.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import cv2
import numpy as np
import open3d as o3d
import rerun as rr
import torch
from fusion import Fusion
from utils.draw_utils import aggr_point_cloud_from_data
from utils.my_utils import fps_np


# ---------------------------------------------------------------------------
# SE(3) parameterization helpers
# ---------------------------------------------------------------------------

def axis_angle_to_rotation_matrix(aa):
    """Convert axis-angle (3,) tensor to 3x3 rotation matrix via Rodrigues."""
    theta = torch.norm(aa)
    if theta < 1e-8:
        return torch.eye(3, device=aa.device, dtype=aa.dtype)
    k = aa / theta
    K = torch.zeros(3, 3, device=aa.device, dtype=aa.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]
    K[1, 0] = k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1]; K[2, 1] = k[0]
    R = torch.eye(3, device=aa.device, dtype=aa.dtype) + \
        torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def pose_from_params(aa, t):
    """Build 4x4 SE(3) matrix from axis-angle (3,) and translation (3,)."""
    T = torch.eye(4, device=aa.device, dtype=aa.dtype)
    T[:3, :3] = axis_angle_to_rotation_matrix(aa)
    T[:3, 3] = t
    return T


def rotation_matrix_to_axis_angle(R):
    """Convert 3x3 rotation matrix to axis-angle (3,) tensor."""
    theta = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    if theta < 1e-8:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    k = torch.stack([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2 * torch.sin(theta))
    return k * theta


# ---------------------------------------------------------------------------
# Contact set construction
# ---------------------------------------------------------------------------

def build_object_pcd(colors, depths, intrinsics, extrinsics, boundaries,
                     masks=None, z_max=None):
    """Aggregate multi-view point cloud within workspace boundaries.

    Args:
        masks: optional (N, H, W) bool array to filter points (e.g. instance mask)
        z_max: if set, discard points with z > z_max
    """
    pcd_o3d = aggr_point_cloud_from_data(
        colors[..., ::-1], depths, intrinsics, extrinsics,
        downsample=True, masks=masks, boundaries=boundaries)
    # Statistical filter (gentle) → keeps thin structures like handles
    _, inlier_idx = pcd_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)
    pcd_o3d = pcd_o3d.select_by_index(inlier_idx)
    # Radius filter → removes isolated noise without eroding geometry
    _, inlier_idx = pcd_o3d.remove_radius_outlier(nb_points=5, radius=0.02)
    pcd_o3d = pcd_o3d.select_by_index(inlier_idx)
    # Z cutoff
    if z_max is not None:
        pts = np.asarray(pcd_o3d.points)
        pcd_o3d = pcd_o3d.select_by_index(np.where(pts[:, 2] <= z_max)[0])
    return np.asarray(pcd_o3d.points)


def build_contact_set(fusion_ref, object_pcd, gripper_pose, n_points,
                      distance_thresh=0.03, device='cuda'):
    """
    Construct the contact set D = {(q_i, f_i*)}.

    Strategy:
    - If enough points lie within `distance_thresh` of the gripper, use those
      (real contact region).
    - Otherwise, FPS-sample from the full object surface to get
      spatially diverse points with distinct descriptors.

    Returns:
        Q          : (N, 3) np array, query points in gripper-local frame
        F_star     : (N, feat_dim) np array, reference descriptors
        P_contact  : (N, 3) np array, contact points in world frame
    """
    gripper_pos = gripper_pose[:3, 3]
    dists = np.linalg.norm(object_pcd - gripper_pos, axis=1)
    contact_mask = dists < distance_thresh
    contact_pcd = object_pcd[contact_mask]

    if contact_pcd.shape[0] >= n_points:
        # Enough nearby points — use the actual contact region
        contact_pcd, _, _ = fps_np(contact_pcd, n_points)
        print(f"  Using {n_points} contact-region points "
              f"(threshold={distance_thresh:.3f}m)")
    else:
        # Gripper far from object — use FPS over full surface for diverse
        # descriptors that provide strong constraints.
        print(f"  Gripper far from object ({contact_pcd.shape[0]} pts within "
              f"{distance_thresh:.3f}m). Using FPS over full surface.")
        contact_pcd, _, _ = fps_np(object_pcd, n_points)

    # Query descriptors at contact points
    pts_tensor = torch.from_numpy(contact_pcd).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = fusion_ref.batch_eval(pts_tensor, return_names=['dino_feats'])
    f_star = out['dino_feats'].cpu().numpy()  # (N, feat_dim)
    dist = out['dist'].cpu().numpy()          # (N,)

    # Transform to gripper-local frame: q_i = T_demo^{-1} · p_i
    T_inv = np.linalg.inv(gripper_pose)
    ones = np.ones((contact_pcd.shape[0], 1))
    contact_homo = np.hstack([contact_pcd, ones])  # (N, 4)
    Q = (T_inv @ contact_homo.T).T[:, :3]          # (N, 3)

    return Q, f_star, contact_pcd, dist


# ---------------------------------------------------------------------------
# Coarse matching: DINO feature NN → rigid alignment
# ---------------------------------------------------------------------------

def coarse_match_init_pose(fusion_new, object_pcd_new, contact_pts_world,
                           f_star, gripper_pose, device='cuda'):
    """
    Compute an initial SE(3) pose via DINO feature nearest-neighbour matching.

    Steps:
        1. Query DINO descriptors on the new scene's object point cloud.
        2. For each reference contact descriptor f_i*, find the NN in the
           new scene → correspondence point c_i.
        3. Solve the rigid alignment  contact_pts_world → c  via SVD
           (Procrustes), giving T_coarse.
        4. Recover the gripper pose as  T_init = T_coarse · T_demo.

    Returns:
        init_pose : (4, 4) np array, coarse gripper pose in the new scene
        corr_pts  : (N, 3) np array, matched points in the new scene
    """
    # 1. Query DINO features on new object surface (batched)
    pts_tensor = torch.from_numpy(object_pcd_new).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = fusion_new.batch_eval(pts_tensor, return_names=['dino_feats'])
    f_new_all = out['dino_feats']  # (M, feat_dim)

    # 2. NN matching: for each contact descriptor find closest in new scene
    f_star_t = torch.from_numpy(f_star).to(device, dtype=torch.float32)
    # Pairwise cosine-like L2: (N, M)
    dists = torch.cdist(f_star_t, f_new_all)  # (N, M)
    nn_idx = dists.argmin(dim=1).cpu().numpy()  # (N,)
    corr_pts = object_pcd_new[nn_idx]  # (N, 3)

    # 3. SVD rigid alignment: contact_pts_world → corr_pts
    #    Find R, t such that corr ≈ R @ contact + t
    src = contact_pts_world  # (N, 3)
    dst = corr_pts           # (N, 3)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    # Correct reflection
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    t_align = dst_mean - R_align @ src_mean

    T_coarse = np.eye(4)
    T_coarse[:3, :3] = R_align
    T_coarse[:3, 3] = t_align

    # 4. Recover gripper pose: T_init = T_coarse · T_demo
    init_pose = T_coarse @ gripper_pose

    print(f"  Coarse match: median NN feature dist = "
          f"{dists.min(dim=1).values.median().item():.4f}")
    print(f"  Coarse alignment residual (mean): "
          f"{np.linalg.norm(dst - (R_align @ src.T).T - t_align, axis=1).mean():.4f} m")

    nn_dists = dists.min(dim=1).values.cpu().numpy()  # (N,)
    return init_pose, corr_pts, nn_dists


# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

def compute_cost(fusion_new, Q, f_star, aa, t, w_f=1.0, w_d=0.1):
    """
    Compute C(T) = Σ_i [ w_f * ||f_new(x_i) - f_i*||^2  +  w_d * d_new(x_i) ]

    Args:
        fusion_new : Fusion object with new scene loaded
        Q          : (N, 3) tensor, query points in gripper-local frame
        f_star     : (N, feat_dim) tensor, reference descriptors
        aa         : (3,) tensor, axis-angle rotation parameters (requires_grad)
        t          : (3,) tensor, translation parameters (requires_grad)
        w_f        : float, feature matching weight
        w_d        : float, signed distance weight

    Returns:
        cost       : scalar tensor (differentiable)
    """
    T = pose_from_params(aa, t)
    R = T[:3, :3]
    trans = T[:3, 3]

    # x_i(T) = T · q_i  (apply rotation and translation)
    x = (R @ Q.T).T + trans  # (N, 3)

    # Query the new D3Fields
    out = fusion_new.eval(x, return_names=['dino_feats'])
    f_new = out['dino_feats']       # (N, feat_dim)
    d_new = out['dist']             # (N,)
    valid = out['valid_mask']       # (N,) bool

    # Debug: check gradient flow and validity
    if not hasattr(compute_cost, '_dbg_done'):
        n_valid = valid.sum().item()
        n_total = valid.shape[0]
        print(f"  [DEBUG] valid points: {n_valid}/{n_total}")
        print(f"  [DEBUG] x min: {x.min(0).values.detach().cpu().numpy()}")
        print(f"  [DEBUG] x max: {x.max(0).values.detach().cpu().numpy()}")
        print(f"  [DEBUG] f_new range: [{f_new.min().item():.4f}, {f_new.max().item():.4f}]")
        print(f"  [DEBUG] d_new range: [{d_new.min().item():.4f}, {d_new.max().item():.4f}]")
        print(f"  [DEBUG] x requires_grad: {x.requires_grad}")
        print(f"  [DEBUG] f_new requires_grad: {f_new.requires_grad}")
        print(f"  [DEBUG] d_new requires_grad: {d_new.requires_grad}")
        compute_cost._dbg_done = True

    # Feature matching cost
    feat_cost = torch.sum((f_new - f_star) ** 2, dim=1)  # (N,)

    # Total cost
    cost = w_f * feat_cost.sum() + w_d * d_new.sum()
    return cost


# ---------------------------------------------------------------------------
# Multi-start optimization
# ---------------------------------------------------------------------------

def optimize_grasp_pose(fusion_new, Q_np, f_star_np, init_pose,
                        w_f=1.0, w_d=0.1,
                        n_restarts=10, n_iters=200, lr=1e-3,
                        perturb_rot=0.3, perturb_trans=0.03,
                        device='cuda', verbose=True):
    """
    Optimize grasp pose T* with multiple random restarts.

    Args:
        fusion_new   : Fusion object with new scene loaded
        Q_np         : (N, 3) np array, query points in gripper-local frame
        f_star_np    : (N, feat_dim) np array, reference descriptors
        init_pose    : (4, 4) np array, initial guess for gripper pose
        w_f, w_d     : cost weights
        n_restarts   : number of random restarts
        n_iters      : optimization iterations per restart
        lr           : learning rate
        perturb_rot  : std of rotation perturbation (radians) for restarts
        perturb_trans: std of translation perturbation (meters) for restarts
        device       : torch device

    Returns:
        best_pose       : (4, 4) np array, optimal gripper pose
        best_cost       : float, minimum cost achieved
        all_results     : list of (pose, cost) tuples for all restarts
        best_trajectory : list of (pose, cost) for every iteration of the best restart
    """
    Q = torch.from_numpy(Q_np).to(device, dtype=torch.float32)
    f_star = torch.from_numpy(f_star_np).to(device, dtype=torch.float32)

    # Extract initial rotation and translation
    R_init = torch.from_numpy(init_pose[:3, :3]).to(device, dtype=torch.float32)
    t_init = torch.from_numpy(init_pose[:3, 3]).to(device, dtype=torch.float32)
    aa_init = rotation_matrix_to_axis_angle(R_init)

    best_cost = float('inf')
    best_pose = None
    best_trajectory = []  # full per-iteration trajectory of best restart
    all_results = []

    for restart_i in range(n_restarts):
        # Perturb the initial guess (first restart uses the exact initial guess)
        if restart_i == 0:
            aa_param = aa_init.clone().detach().requires_grad_(True)
            t_param = t_init.clone().detach().requires_grad_(True)
        else:
            aa_perturb = aa_init + torch.randn(3, device=device) * perturb_rot
            t_perturb = t_init + torch.randn(3, device=device) * perturb_trans
            aa_param = aa_perturb.clone().detach().requires_grad_(True)
            t_param = t_perturb.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([aa_param, t_param], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)
        traj = []

        # Record initial pose
        with torch.no_grad():
            init_snap = pose_from_params(aa_param, t_param).detach().cpu().numpy()
        traj.append((init_snap, float('inf')))

        for it in range(n_iters):
            optimizer.zero_grad()
            cost = compute_cost(fusion_new, Q, f_star, aa_param, t_param,
                                w_f=w_f, w_d=w_d)
            cost.backward()
            optimizer.step()
            scheduler.step()

            if verbose and it % 50 == 0:
                print(f"  restart {restart_i}, iter {it}: cost = {cost.item():.6f}")

            # Record every iteration
            with torch.no_grad():
                snap_pose = pose_from_params(aa_param, t_param).cpu().numpy()
            traj.append((snap_pose, cost.item()))

        final_cost = cost.item()
        final_pose = pose_from_params(aa_param, t_param).detach().cpu().numpy()
        all_results.append((final_pose, final_cost))

        if verbose:
            print(f"  restart {restart_i} final cost: {final_cost:.6f}")

        if final_cost < best_cost:
            best_cost = final_cost
            best_pose = final_pose
            best_trajectory = traj

    return best_pose, best_cost, all_results, best_trajectory


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_scene_data(data_path, num_cam, t):
    """Load color, depth, intrinsics, extrinsics for a given timestep."""
    colors = np.stack([
        cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png'))
        for i in range(num_cam)], axis=0)
    depths = np.stack([
        cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'),
                   cv2.IMREAD_ANYDEPTH)
        for i in range(num_cam)], axis=0) / 1000.0

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
    return obs, colors, depths, intrinsics, extrinsics


def load_gripper_pose(data_path, t):
    """Load the gripper base pose at timestep t."""
    return np.loadtxt(os.path.join(data_path, 'pose', 'robotiq_base', f'{t}.txt'))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def pose_error(T_pred, T_gt):
    """Compute rotation (deg) and translation (m) error between two SE(3) poses."""
    R_err = T_pred[:3, :3] @ T_gt[:3, :3].T
    cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    rot_err_deg = np.degrees(np.arccos(cos_angle))
    trans_err = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
    return rot_err_deg, trans_err


# ---------------------------------------------------------------------------
# Quick Open3D visualization (blocking, for debug)
# ---------------------------------------------------------------------------

def visualize_o3d(point_clouds, poses=None, title="Debug", frame_size=0.05):
    """Show point clouds and optional SE(3) poses in an Open3D window.

    Args:
        point_clouds: list of (points, color) tuples.
                      points: (N,3) np array, color: (3,) float [0-1].
        poses:        list of (4,4) np arrays (coordinate frames to draw).
        title:        window title.
        frame_size:   axis length for coordinate frames.
    """
    geometries = []
    for pts, color in point_clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    for pose in (poses or []):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose)
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1024, height=768)


# ---------------------------------------------------------------------------
# Visualization (Open3D for static views, Rerun for optimization trajectory)
# ---------------------------------------------------------------------------

_rr_initialized = False


def _ensure_rr(app_id="grasp_pose_search", save_path="vis_output/grasp_opt.rrd"):
    """Initialize Rerun once, saving to an .rrd file (no viewer spawned)."""
    global _rr_initialized
    if not _rr_initialized:
        rr.init(app_id)
        rr.save(save_path)
        _rr_initialized = True


def _log_gripper_axes(entity_path, pose, size=0.05):
    """Log three line segments (RGB = XYZ) representing a coordinate frame."""
    origin = pose[:3, 3]
    axes = pose[:3, :3] * size  # columns are axis directions
    points = np.stack([
        np.stack([origin, origin + axes[:, 0]]),  # X red
        np.stack([origin, origin + axes[:, 1]]),  # Y green
        np.stack([origin, origin + axes[:, 2]]),  # Z blue
    ])  # (3, 2, 3)
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    # Log as three separate line strips
    rr.log(entity_path, rr.LineStrips3D(points, colors=colors, radii=0.002))



def visualize_dino_matching(object_pcd_np, contact_pts, corr_pts,
                            nn_dists=None, title="DINO Matching"):
    """
    Visualize DINO feature correspondences in Open3D: source contact points,
    matched target points, and lines connecting each pair.

    Lines are colored by NN distance (green=close, red=far) when nn_dists
    is provided.
    """
    geometries = []

    # Object point cloud (grey)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    obj_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(obj_pcd)

    # Source contact points (blue)
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(contact_pts)
    src_pcd.paint_uniform_color([0.2, 0.4, 1.0])
    geometries.append(src_pcd)

    # Matched correspondence points (green)
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(corr_pts)
    dst_pcd.paint_uniform_color([0.2, 1.0, 0.4])
    geometries.append(dst_pcd)

    # Correspondence lines
    n = contact_pts.shape[0]
    line_points = np.concatenate([contact_pts, corr_pts], axis=0)  # (2N, 3)
    line_indices = [[i, i + n] for i in range(n)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)

    if nn_dists is not None:
        d = nn_dists.copy()
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)
        # Green (close) → Red (far)
        colors = np.zeros((n, 3))
        colors[:, 0] = d_norm
        colors[:, 1] = 1.0 - d_norm
    else:
        colors = np.full((n, 3), [1.0, 0.8, 0.2])

    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1024, height=768)


def visualize_pose(object_pcd_np, Q, pose, contact_pts_world,
                   title="Pose", frame_size=0.05):
    """Show a single gripper pose in an Open3D window."""
    geometries = []

    # Object point cloud (grey)
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    obj_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(obj_pcd)

    # Gripper coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    frame.transform(pose)
    geometries.append(frame)

    # Transformed query points: x_i = T · q_i (red)
    ones = np.ones((Q.shape[0], 1))
    Q_homo = np.hstack([Q, ones])
    x_world = (pose @ Q_homo.T).T[:, :3]
    q_pcd = o3d.geometry.PointCloud()
    q_pcd.points = o3d.utility.Vector3dVector(x_world)
    q_pcd.paint_uniform_color([1.0, 0.2, 0.2])
    geometries.append(q_pcd)

    # Reference contact points (blue)
    c_pcd = o3d.geometry.PointCloud()
    c_pcd.points = o3d.utility.Vector3dVector(contact_pts_world)
    c_pcd.paint_uniform_color([0.2, 0.4, 1.0])
    geometries.append(c_pcd)

    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1024, height=768)


def visualize_optimization(object_pcd_np, Q, trajectory, contact_pts_world,
                           frame_size=0.05):
    """Log the optimization trajectory to Rerun as a time sequence."""
    _ensure_rr()

    prefix = "optimization"

    # Static: object point cloud (grey)
    rr.log(f"{prefix}/object_pcd", rr.Points3D(
        object_pcd_np,
        colors=np.full((object_pcd_np.shape[0], 3), 180, dtype=np.uint8),
        radii=0.002), static=True)

    # Static: reference contact points (blue)
    blue = np.full((contact_pts_world.shape[0], 3), [50, 100, 255],
                   dtype=np.uint8)
    rr.log(f"{prefix}/ref_contact_pts",
           rr.Points3D(contact_pts_world, colors=blue, radii=0.003),
           static=True)

    ones = np.ones((Q.shape[0], 1))
    Q_homo = np.hstack([Q, ones])

    trail_pts_list = []

    for step_i, (pose_i, cost_i) in enumerate(trajectory):
        rr.set_time("opt_step", sequence=step_i)

        # Cost scalar
        rr.log(f"{prefix}/cost", rr.Scalars(cost_i))

        # Gripper frame
        _log_gripper_axes(f"{prefix}/gripper_frame", pose_i, size=frame_size)

        # Transformed query points (red)
        x_world = (pose_i @ Q_homo.T).T[:, :3]
        red = np.full((x_world.shape[0], 3), [255, 50, 50], dtype=np.uint8)
        rr.log(f"{prefix}/query_pts",
               rr.Points3D(x_world, colors=red, radii=0.004))

        # Trail (green, fading older points)
        trail_pts_list.append(x_world)
        trail_all = np.concatenate(trail_pts_list, axis=0)
        n_trail = trail_all.shape[0]
        green_vals = np.linspace(80, 230, n_trail).astype(np.uint8)
        trail_colors = np.zeros((n_trail, 3), dtype=np.uint8)
        trail_colors[:, 1] = green_vals
        rr.log(f"{prefix}/trail",
               rr.Points3D(trail_all, colors=trail_colors, radii=0.002))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    class Args: pass
    args = Args()
    args.data_path = 'data/mugs'
    args.num_cam = 3
    args.t_ref = 0
    args.t_new = 1
    args.n_contact = 64
    args.contact_thresh = 0.08
    args.w_f = 1.0
    args.w_d = 0.1
    args.n_restarts = 30
    args.n_iters = 300
    args.lr = 1e-3
    args.perturb_rot = 0.6
    args.perturb_trans = 0.2
    args.device = 'cuda'
    args.query_text = 'mug'
    args.query_threshold = 0.3

    boundaries = {
        'x_lower': -0.4, 'x_upper': 0.4,
        'y_lower': -0.4, 'y_upper': 0.3,
        'z_lower': -0.2, 'z_upper': 0.02,
    }

    # --- Step 1: Load reference scene and build D3Fields ---
    print("Loading reference scene...")
    obs_ref, colors_ref, depths_ref, intrinsics, extrinsics = \
        load_scene_data(args.data_path, args.num_cam, args.t_ref)

    fusion_ref = Fusion(num_cam=args.num_cam, feat_backbone='dinov2',
                        device=args.device, skip_xmem=True)
    fusion_ref.update(obs_ref)

    print("Running instance segmentation for reference scene...")
    fusion_ref.text_queries_for_inst_mask_no_track(
        [args.query_text], [args.query_threshold], boundaries=boundaries)

    # Build object-only mask: non-background instances (id > 0)
    ref_mask = fusion_ref.curr_obs_torch['mask']  # (N, H, W, num_inst) one-hot
    if isinstance(ref_mask, torch.Tensor):
        ref_mask = ref_mask.detach().cpu().numpy()
    # Sum all non-background channels → bool mask per camera
    ref_obj_mask = ref_mask[..., 1:].sum(axis=-1) > 0  # (N, H, W) bool

    # --- Step 2: Build object point cloud and contact set ---
    print("Building contact set...")
    T_demo = np.array([
        [-1.336891040490915950e-05, 9.975693974887754889e-01, 6.967996136016127440e-02, 2.304401291852276545e-01],
        [ 9.999999999106338189e-01, 1.333179035347721586e-05, 9.977665313340433606e-07, 40.346664004007867626e-03],
        [ 6.638272079945162218e-08, 6.967996136727330758e-02,-9.975693975778581191e-01, 0.086479689436886442e-01],
        [ 0.0,                      0.0,                      0.0,                      1.0],
    ])

    object_pcd = build_object_pcd(colors_ref, depths_ref, intrinsics,
                                  extrinsics, boundaries, masks=ref_obj_mask, z_max=-0.015)
    print(f"  Object point cloud: {object_pcd.shape[0]} points")

    # Visualize object point cloud and T_demo before building contact set
    # print("Showing object point cloud + T_demo (close window to continue)...")
    # visualize_o3d([(object_pcd, [0.7, 0.7, 0.7])], poses=[T_demo],
    #               title="Object PCD + T_demo", frame_size=0.08)
    
    Q, f_star, contact_pts, dist = build_contact_set(
        fusion_ref, object_pcd, T_demo,
        n_points=args.n_contact,
        distance_thresh=args.contact_thresh,
        device=args.device)
    print(f"  Contact set: {Q.shape[0]} points, "
          f"descriptor dim: {f_star.shape[1]}")


    # --- Step 3: Load new scene (or reuse ref for self-test) ---
    if args.t_ref == args.t_new:
        print("Self-test mode: using same scene for ref and target.")
        fusion_new = fusion_ref
        colors_new, depths_new = colors_ref, depths_ref
    else:
        print(f"Loading target scene at t={args.t_new}...")
        obs_new, colors_new, depths_new, _, _ = load_scene_data(
            args.data_path, args.num_cam, args.t_new)
        fusion_new = Fusion(num_cam=args.num_cam, feat_backbone='dinov2',
                            device=args.device, skip_xmem=True)
        fusion_new.update(obs_new)
        print("Running instance segmentation for target scene...")
        fusion_new.text_queries_for_inst_mask_no_track(
            [args.query_text], [args.query_threshold], boundaries=boundaries)

    # Build new scene object point cloud
    new_mask = fusion_new.curr_obs_torch['mask']
    if isinstance(new_mask, torch.Tensor):
        new_mask = new_mask.detach().cpu().numpy()
    new_obj_mask = new_mask[..., 1:].sum(axis=-1) > 0
    object_pcd_new = build_object_pcd(colors_new, depths_new, intrinsics,
                                      extrinsics, boundaries, masks=new_obj_mask, z_max=-0.015)
    print(f"  New scene object point cloud: {object_pcd_new.shape[0]} points")

    # --- Step 4: Coarse DINO matching → initial pose ---
    print("\n=== Coarse DINO feature matching ===")
    init_pose, corr_pts, nn_dists = coarse_match_init_pose(
        fusion_new, object_pcd_new, contact_pts, f_star,
        T_demo, device=args.device)

    # Visualize DINO correspondences (contact_pts → corr_pts)
    # visualize_dino_matching(object_pcd_new, contact_pts, corr_pts,
    #                         nn_dists=nn_dists, title="DINO Matching")
    # Visualize coarse initial pose from DINO matching
    # visualize_pose(object_pcd_new, Q, init_pose, corr_pts,
    #                title="Coarse Init Pose")

    # --- Step 5: Fine optimization from coarse init ---
    print("\n=== Fine optimization ===")
    best_pose, best_cost, all_results, best_trajectory = optimize_grasp_pose(
        fusion_new, Q, f_star, init_pose,
        w_f=args.w_f, w_d=args.w_d,
        n_restarts=args.n_restarts, n_iters=args.n_iters, lr=args.lr,
        perturb_rot=args.perturb_rot, perturb_trans=args.perturb_trans,
        device=args.device)
    print(f"Best cost: {best_cost:.6f}")

    # --- Step 6: Evaluate ---
    # T_gt = T_demo
    # rot_err, trans_err = pose_error(best_pose, T_gt)

    # print("\n" + "=" * 60)
    # print("Optimization Results")
    # print("=" * 60)
    # print(f"Best cost:         {best_cost:.6f}")
    # print(f"Final error:       {rot_err:.2f} deg, "
    #       f"{trans_err * 1000:.2f} mm")
    # print(f"\nGround truth pose:\n{T_gt}")
    # print(f"\nOptimized pose:\n{best_pose}")
    # print("=" * 60)

    # Print all restart results
    # costs = [c for _, c in all_results]
    # print(f"\nAll restart costs: "
    #       f"min={min(costs):.6f}, max={max(costs):.6f}, "
    #       f"mean={np.mean(costs):.6f}")

    # for i, (pose_i, cost_i) in enumerate(all_results):
    #     r_e, t_e = pose_error(pose_i, T_gt)
    #     print(f"  restart {i}: cost={cost_i:.4f}, "
    #           f"rot_err={r_e:.2f} deg, trans_err={t_e * 1000:.2f} mm")

    # --- Step 7: Visualize ---
    print("\n=== Visualization (close each Open3D window to proceed) ===")

    # Optimization trajectory (Rerun)
    visualize_optimization(object_pcd_new, Q, best_trajectory, contact_pts)

    # Final poses
    visualize_pose(object_pcd_new, Q, best_pose, contact_pts,
                   title="Optimized")

    print("\nRerun optimization trajectory saved. View with:  rerun vis_output/grasp_opt.rrd")
    return best_pose, best_cost


if __name__ == '__main__':
    main()
