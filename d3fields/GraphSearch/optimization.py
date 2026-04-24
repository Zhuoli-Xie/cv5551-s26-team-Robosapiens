"""Multi-start SE(3) grasp pose optimization.

Minimizes compute_cost(Q, f*, R, t) over the gripper pose with Adam, using a
6D rotation parameterization (Zhou et al. 2019) — continuous, gimbal-lock free,
and autograd-friendly. Runs N restarts (first from the coarse init, the rest
perturbed in axis-angle space) and falls back to the coarse init if none wins.
"""

import torch

from .se3_utils import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    pose_from_params,
    axis_angle_to_rotation_matrix,
)
from .cost_functions import compute_cost


def optimize_grasp_pose(fusion_new, Q_np, f_star_np, init_pose,
                        w_f=1.0, w_d=0.1, w_n=0.5,
                        n_restarts=10, n_iters=200,
                        lr_rot=1e-2, lr_trans=1e-3,
                        perturb_rot=0.3, perturb_trans=0.03,
                        device='cuda', verbose=True):
    """Optimize grasp pose T* with multiple random restarts.

    Returns:
        best_pose       : (4, 4) np.ndarray
        best_cost       : float
        all_results     : list of (pose, cost) per restart
        best_trajectory : list of (pose, cost) per iter of the best restart
    """
    Q = torch.from_numpy(Q_np).to(device, dtype=torch.float32)
    f_star = torch.from_numpy(f_star_np).to(device, dtype=torch.float32)

    R_init = torch.from_numpy(init_pose[:3, :3]).to(device, dtype=torch.float32)
    t_init = torch.from_numpy(init_pose[:3, 3]).to(device, dtype=torch.float32)
    d6_init = matrix_to_rotation_6d(R_init)

    # Approach axis in gripper-local frame; w_n aligns this with the surface normal.
    approach_local = torch.tensor([0.0, 0.0, 1.0], device=device)

    def eval_cost(R, t):
        return compute_cost(fusion_new, Q, f_star, R, t,
                            w_f=w_f, w_d=w_d, w_n=w_n,
                            approach_local=approach_local)

    with torch.no_grad():
        init_cost = eval_cost(R_init, t_init).item()
    if verbose:
        print(f"  cost at coarse init pose: {init_cost:.6f}")

    best_pose, best_cost, best_trajectory = None, float('inf'), []
    all_results = []

    for restart_i in range(n_restarts):
        # Restart 0 starts from the coarse init; later restarts perturb in
        # axis-angle space (std = perturb_rot radians) then re-encode as 6D.
        if restart_i == 0:
            d6_start, t_start = d6_init, t_init
        else:
            aa = torch.randn(3, device=device) * perturb_rot
            R_perturb = axis_angle_to_rotation_matrix(aa) @ R_init
            d6_start = matrix_to_rotation_6d(R_perturb)
            t_start = t_init + torch.randn(3, device=device) * perturb_trans

        d6_param = d6_start.clone().detach().requires_grad_(True)
        t_param = t_start.clone().detach().requires_grad_(True)

        # lr_rot is typically ~10x lr_trans: Gram-Schmidt in the 6D->R mapping
        # absorbs the radial component of each update, shrinking effective rot LR.
        optimizer = torch.optim.Adam([
            {'params': [d6_param], 'lr': lr_rot},
            {'params': [t_param],  'lr': lr_trans},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

        traj = []
        for it in range(n_iters):
            optimizer.zero_grad()
            cost = eval_cost(rotation_6d_to_matrix(d6_param), t_param)
            cost.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                snap_pose = pose_from_params(d6_param, t_param).cpu().numpy()
            traj.append((snap_pose, cost.item()))

            # if verbose and it % 50 == 0:
            #     print(f"  restart {restart_i}, iter {it}: cost = {cost.item():.6f}")

        final_pose, final_cost = traj[-1]
        all_results.append((final_pose, final_cost))
        if verbose:
            print(f"  restart {restart_i} final cost: {final_cost:.6f}")

        if final_cost < best_cost:
            best_pose, best_cost, best_trajectory = final_pose, final_cost, traj

    # Safety net: fine opt can diverge when feature matches are weak — if no
    # restart beat the coarse init, return the coarse pose instead.
    if best_cost >= init_cost:
        if verbose:
            print(f"  fine opt did not improve on coarse init "
                  f"({best_cost:.6f} >= {init_cost:.6f}); returning coarse pose.")
        best_pose = init_pose.copy()
        best_cost = init_cost
        best_trajectory = [(init_pose.copy(), init_cost)]

    return best_pose, best_cost, all_results, best_trajectory
