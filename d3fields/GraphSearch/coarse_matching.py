"""Coarse matching: DINO feature NN -> rigid alignment (SVD Procrustes)."""

import numpy as np
import torch


def coarse_match_init_pose(fusion_new, object_pcd_new, contact_pts_world,
                           f_star, gripper_pose, device='cuda'):
    """
    Compute an initial SE(3) pose via DINO feature nearest-neighbour matching.

    Steps:
        1. Query DINO descriptors on the new scene's object point cloud.
        2. For each reference contact descriptor f_i*, find the NN in the
           new scene -> correspondence point c_i.
        3. Solve the rigid alignment  contact_pts_world -> c  via SVD
           (Procrustes), giving T_coarse.
        4. Recover the gripper pose as  T_init = T_coarse * T_demo.

    Returns:
        init_pose : (4, 4) np array, coarse gripper pose in the new scene
        corr_pts  : (N, 3) np array, matched points in the new scene
        nn_dists  : (N,) np array, NN feature distances
    """
    # 1. Query DINO features on new object surface
    pts_tensor = torch.from_numpy(object_pcd_new).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = fusion_new.batch_eval(pts_tensor, return_names=['dino_feats'])
    f_new_all = out['dino_feats']

    # 2. NN matching
    f_star_t = torch.from_numpy(f_star).to(device, dtype=torch.float32)
    dists = torch.cdist(f_star_t, f_new_all)
    nn_idx = dists.argmin(dim=1).cpu().numpy()
    corr_pts = object_pcd_new[nn_idx]

    # 3. SVD rigid alignment: contact_pts_world -> corr_pts
    src = contact_pts_world
    dst = corr_pts
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    t_align = dst_mean - R_align @ src_mean

    T_coarse = np.eye(4)
    T_coarse[:3, :3] = R_align
    T_coarse[:3, 3] = t_align

    # 4. Recover gripper pose
    init_pose = T_coarse @ gripper_pose

    print(f"  Coarse match: median NN feature dist = "
          f"{dists.min(dim=1).values.median().item():.4f}")
    print(f"  Coarse alignment residual (mean): "
          f"{np.linalg.norm(dst - (R_align @ src.T).T - t_align, axis=1).mean():.4f} m")

    nn_dists = dists.min(dim=1).values.cpu().numpy()
    return init_pose, corr_pts, nn_dists
