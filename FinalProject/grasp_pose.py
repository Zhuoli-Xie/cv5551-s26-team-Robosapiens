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

def build_object_pcd(colors, depths, intrinsics, extrinsics, boundaries):
    """Aggregate multi-view point cloud within workspace boundaries."""
    pcd_o3d = aggr_point_cloud_from_data(
        colors[..., ::-1], depths, intrinsics, extrinsics,
        downsample=True, boundaries=boundaries)
    pcd_o3d.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)
    return np.asarray(pcd_o3d.points)


def build_contact_set(fusion_ref, object_pcd, gripper_pose, n_points,
                      distance_thresh=0.02, device='cuda'):
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

    # Transform to gripper-local frame: q_i = T_demo^{-1} · p_i
    T_inv = np.linalg.inv(gripper_pose)
    ones = np.ones((contact_pcd.shape[0], 1))
    contact_homo = np.hstack([contact_pcd, ones])  # (N, 4)
    Q = (T_inv @ contact_homo.T).T[:, :3]          # (N, 3)

    return Q, f_star, contact_pcd


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
        best_pose    : (4, 4) np array, optimal gripper pose
        best_cost    : float, minimum cost achieved
        all_results  : list of (pose, cost) tuples for all restarts
    """
    Q = torch.from_numpy(Q_np).to(device, dtype=torch.float32)
    f_star = torch.from_numpy(f_star_np).to(device, dtype=torch.float32)

    # Extract initial rotation and translation
    R_init = torch.from_numpy(init_pose[:3, :3]).to(device, dtype=torch.float32)
    t_init = torch.from_numpy(init_pose[:3, 3]).to(device, dtype=torch.float32)
    aa_init = rotation_matrix_to_axis_angle(R_init)

    best_cost = float('inf')
    best_pose = None
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

        for it in range(n_iters):
            optimizer.zero_grad()
            cost = compute_cost(fusion_new, Q, f_star, aa_param, t_param,
                                w_f=w_f, w_d=w_d)
            cost.backward()
            optimizer.step()
            scheduler.step()

            if verbose and it % 50 == 0:
                print(f"  restart {restart_i}, iter {it}: cost = {cost.item():.6f}")

        final_cost = cost.item()
        final_pose = pose_from_params(aa_param, t_param).detach().cpu().numpy()
        all_results.append((final_pose, final_cost))

        if verbose:
            print(f"  restart {restart_i} final cost: {final_cost:.6f}")

        if final_cost < best_cost:
            best_cost = final_cost
            best_pose = final_pose

    return best_pose, best_cost, all_results


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
# Visualization
# ---------------------------------------------------------------------------

def _make_gripper_frame(pose, size=0.05):
    """Create an Open3D coordinate frame at the given SE(3) pose."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(pose)
    return frame


def _make_contact_spheres(pts_world, color, radius=0.003):
    """Create small spheres at each contact point location."""
    spheres = o3d.geometry.TriangleMesh()
    for pt in pts_world:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=8)
        s.translate(pt)
        s.paint_uniform_color(color)
        spheres += s
    return spheres


def visualize_pose(object_pcd_np, Q, pose, contact_pts_world,
                   title="Pose", frame_size=0.05):
    """
    Show a single gripper pose in context.

    Displays:
    - Grey object point cloud
    - RGB coordinate frame at the gripper pose
    - Coloured spheres at the transformed contact query points (T · q_i)

    Args:
        object_pcd_np    : (M, 3) np array, object surface points
        Q                : (N, 3) np array, query points in gripper-local frame
        pose             : (4, 4) np array, gripper SE(3) pose to display
        contact_pts_world: (N, 3) np array, original demo contact points
                           (shown faintly for reference)
        title            : window title string
        frame_size       : length of the coordinate-frame axes (metres)
    """
    # Object point cloud (grey)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Gripper coordinate frame
    frame = _make_gripper_frame(pose, size=frame_size)

    # Transformed query points: x_i = T · q_i
    ones = np.ones((Q.shape[0], 1))
    Q_homo = np.hstack([Q, ones])
    x_world = (pose @ Q_homo.T).T[:, :3]
    query_spheres = _make_contact_spheres(x_world, color=[1.0, 0.2, 0.2],
                                          radius=0.004)

    # Original demo contact points (translucent blue, for spatial reference)
    ref_spheres = _make_contact_spheres(contact_pts_world,
                                        color=[0.2, 0.4, 1.0], radius=0.003)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    vis.add_geometry(pcd)
    vis.add_geometry(frame)
    vis.add_geometry(query_spheres)
    vis.add_geometry(ref_spheres)

    # Sensible default viewpoint: look at object centroid
    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center())
    ctr.set_zoom(0.6)

    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Grasp pose search via D3Fields')
    parser.add_argument('--data_path', type=str,
                        default='data/2023-09-15-13-21-56-171587',
                        help='Path to scene data (used for both ref and new scene)')
    parser.add_argument('--num_cam', type=int, default=4)
    parser.add_argument('--t_ref', type=int, default=50,
                        help='Reference timestep (demo)')
    parser.add_argument('--t_new', type=int, default=50,
                        help='New timestep (target). Same as t_ref for self-test.')
    parser.add_argument('--n_contact', type=int, default=64,
                        help='Number of contact query points')
    parser.add_argument('--contact_thresh', type=float, default=0.08,
                        help='Distance threshold (m) for contact region')
    parser.add_argument('--w_f', type=float, default=1.0,
                        help='Feature matching weight')
    parser.add_argument('--w_d', type=float, default=0.1,
                        help='Signed distance weight')
    parser.add_argument('--n_restarts', type=int, default=8,
                        help='Number of optimization restarts')
    parser.add_argument('--n_iters', type=int, default=200,
                        help='Iterations per restart')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--perturb_rot', type=float, default=0.3,
                        help='Rotation perturbation std (rad)')
    parser.add_argument('--perturb_trans', type=float, default=0.03,
                        help='Translation perturbation std (m)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

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
                        device=args.device)
    fusion_ref.update(obs_ref)

    # --- Step 2: Build object point cloud and contact set ---
    print("Building contact set...")
    T_demo = load_gripper_pose(args.data_path, args.t_ref)
    object_pcd = build_object_pcd(colors_ref, depths_ref, intrinsics,
                                  extrinsics, boundaries)
    print(f"  Object point cloud: {object_pcd.shape[0]} points")

    Q, f_star, contact_pts = build_contact_set(
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
    else:
        print(f"Loading target scene at t={args.t_new}...")
        obs_new, _, _, _, _ = load_scene_data(args.data_path, args.num_cam,
                                              args.t_new)
        fusion_new = Fusion(num_cam=args.num_cam, feat_backbone='dinov2',
                            device=args.device)
        fusion_new.update(obs_new)

    # --- Step 4: Perturbation recovery test ---
    # Apply a known perturbation to T_demo, then verify optimization
    # recovers the original pose.
    T_gt = load_gripper_pose(args.data_path, args.t_new)
    np.random.seed(42)
    delta_rot = args.perturb_rot * np.random.randn(3)
    delta_trans = args.perturb_trans * np.random.randn(3)
    R_perturb = axis_angle_to_rotation_matrix(
        torch.tensor(delta_rot, dtype=torch.float32)).numpy()
    init_pose = T_gt.copy()
    init_pose[:3, :3] = R_perturb @ T_gt[:3, :3]
    init_pose[:3, 3] = T_gt[:3, 3] + delta_trans

    init_rot_err, init_trans_err = pose_error(init_pose, T_gt)
    print(f"\nInitial perturbation: {init_rot_err:.2f} deg, "
          f"{init_trans_err * 1000:.2f} mm")

    # --- Step 5: Optimize ---
    print("Starting grasp pose optimization...")
    best_pose, best_cost, all_results = optimize_grasp_pose(
        fusion_new, Q, f_star, init_pose,
        w_f=args.w_f, w_d=args.w_d,
        n_restarts=args.n_restarts, n_iters=args.n_iters, lr=args.lr,
        perturb_rot=args.perturb_rot, perturb_trans=args.perturb_trans,
        device=args.device)

    # --- Step 6: Evaluate ---
    rot_err, trans_err = pose_error(best_pose, T_gt)

    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Initial error:     {init_rot_err:.2f} deg, "
          f"{init_trans_err * 1000:.2f} mm")
    print(f"Best cost:         {best_cost:.6f}")
    print(f"Final error:       {rot_err:.2f} deg, "
          f"{trans_err * 1000:.2f} mm")
    print(f"\nGround truth pose:\n{T_gt}")
    print(f"\nOptimized pose:\n{best_pose}")
    print("=" * 60)

    # Print all restart results
    costs = [c for _, c in all_results]
    print(f"\nAll restart costs: "
          f"min={min(costs):.6f}, max={max(costs):.6f}, "
          f"mean={np.mean(costs):.6f}")

    for i, (pose_i, cost_i) in enumerate(all_results):
        r_e, t_e = pose_error(pose_i, T_gt)
        print(f"  restart {i}: cost={cost_i:.4f}, "
              f"rot_err={r_e:.2f} deg, trans_err={t_e * 1000:.2f} mm")

    # --- Step 7: Visualize reference and inferred poses ---
    print("\nShowing reference pose (close window to continue)...")
    visualize_pose(object_pcd, Q, T_gt, contact_pts,
                   title="Reference Pose (ground truth)")

    print("Showing inferred pose (close window to finish)...")
    visualize_pose(object_pcd, Q, best_pose, contact_pts,
                   title="Inferred Pose (optimized)")

    return best_pose, best_cost


if __name__ == '__main__':
    main()
