"""Multi-start SE(3) grasp pose optimization."""

import torch

from .se3_utils import axis_angle_to_rotation_matrix, pose_from_params, rotation_matrix_to_axis_angle
from .cost_functions import compute_cost


def optimize_grasp_pose(fusion_new, Q_np, f_star_np, init_pose,
                        w_f=1.0, w_d=0.1, w_n=0.5,
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
    import numpy as np

    Q = torch.from_numpy(Q_np).to(device, dtype=torch.float32)
    f_star = torch.from_numpy(f_star_np).to(device, dtype=torch.float32)

    R_init = torch.from_numpy(init_pose[:3, :3]).to(device, dtype=torch.float32)
    t_init = torch.from_numpy(init_pose[:3, 3]).to(device, dtype=torch.float32)
    aa_init = rotation_matrix_to_axis_angle(R_init)

    best_cost = float('inf')
    best_pose = None
    best_trajectory = []
    all_results = []

    for restart_i in range(n_restarts):
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

        with torch.no_grad():
            init_snap = pose_from_params(aa_param, t_param).detach().cpu().numpy()
        traj.append((init_snap, float('inf')))

        for it in range(n_iters):
            optimizer.zero_grad()
            R = axis_angle_to_rotation_matrix(aa_param)
            approach_local = torch.tensor([0.0, 0.0, 1.0], device=device)
            cost = compute_cost(fusion_new, Q, f_star, R, t_param,
                                w_f=w_f, w_d=w_d, w_n=w_n,
                                approach_local=approach_local)
            cost.backward()
            optimizer.step()
            scheduler.step()

            if verbose and it % 50 == 0:
                print(f"  restart {restart_i}, iter {it}: cost = {cost.item():.6f}")

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
