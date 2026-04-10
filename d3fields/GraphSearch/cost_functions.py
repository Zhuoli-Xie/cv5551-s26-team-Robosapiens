"""
Cost function components for grasp pose optimization.

Each cost term is a standalone function with signature:
    term(fusion, x, **kwargs) -> scalar tensor

This makes it easy to add, remove, or replace individual terms
without touching the rest of the optimization pipeline.
"""

import torch


# ---------------------------------------------------------------------------
# SDF normal estimation (shared utility for normal-based costs)
# ---------------------------------------------------------------------------

def estimate_normals_fd(fusion, x, eps=0.002):
    """
    Estimate surface normals via central finite differences on the SDF.

    Args:
        fusion: Fusion object with .eval()
        x     : (N, 3) tensor, query points
        eps   : step size

    Returns:
        normals: (N, 3) tensor, unit normals
    """
    N = x.shape[0]
    offsets = torch.zeros(6, 3, device=x.device, dtype=x.dtype)
    offsets[0, 0] =  eps
    offsets[1, 0] = -eps
    offsets[2, 1] =  eps
    offsets[3, 1] = -eps
    offsets[4, 2] =  eps
    offsets[5, 2] = -eps

    x_perturbed = x.unsqueeze(0) + offsets.unsqueeze(1)  # (6, N, 3)
    x_perturbed = x_perturbed.reshape(-1, 3)

    out = fusion.eval(x_perturbed, return_names=[])
    d = out['dist'].reshape(6, N)

    normals = torch.stack([
        d[0] - d[1],
        d[2] - d[3],
        d[4] - d[5],
    ], dim=1) / (2 * eps)

    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    return normals


# ---------------------------------------------------------------------------
# Individual cost terms
# ---------------------------------------------------------------------------

def feature_cost(f_new, f_star):
    """L2 feature matching cost: sum_i ||f_new(x_i) - f_i*||^2.

    Args:
        f_new  : (N, D) tensor, descriptors at transformed query points
        f_star : (N, D) tensor, reference descriptors

    Returns:
        scalar tensor
    """
    return torch.sum((f_new - f_star) ** 2, dim=1).sum()


def distance_cost(d_new):
    """Signed distance cost: sum_i d_new(x_i).

    Encourages query points to lie on the object surface.

    Args:
        d_new: (N,) tensor, SDF values at transformed query points

    Returns:
        scalar tensor
    """
    return d_new.sum()


def normal_alignment_cost(fusion, x, approach_world):
    """Normal alignment cost: sum_i (1 + n(x_i) . a)^2.

    Penalizes gripper approach directions that are not anti-parallel
    to the surface normal (i.e., encourages the gripper to approach
    perpendicular to the surface).

    Args:
        fusion         : Fusion object (for SDF normal estimation)
        x              : (N, 3) tensor, transformed query points
        approach_world : (3,) tensor, gripper approach direction in world frame

    Returns:
        scalar tensor
    """
    normals = estimate_normals_fd(fusion, x)
    alignment = torch.sum(normals * approach_world.unsqueeze(0), dim=1)
    return ((1.0 + alignment) ** 2).sum()


# ---------------------------------------------------------------------------
# Combined cost
# ---------------------------------------------------------------------------

def compute_cost(fusion_new, Q, f_star, R, trans,
                 w_f=1.0, w_d=0.1, w_n=0.5,
                 approach_local=None):
    """
    Total cost for a candidate pose defined by rotation R and translation t.

    C(T) = w_f * feature_cost + w_d * distance_cost + w_n * normal_alignment_cost

    Args:
        fusion_new     : Fusion object for the target scene
        Q              : (N, 3) tensor, query points in gripper-local frame
        f_star         : (N, D) tensor, reference descriptors
        R              : (3, 3) tensor, rotation matrix
        trans          : (3,) tensor, translation vector
        w_f, w_d, w_n  : cost weights
        approach_local : (3,) tensor, gripper approach direction in local frame
                         (default Z-axis). Set None to disable normal cost.

    Returns:
        scalar tensor, total cost
    """
    x = (R @ Q.T).T + trans  # (N, 3)

    out = fusion_new.eval(x, return_names=['dino_feats'])
    f_new = out['dino_feats']
    d_new = out['dist']

    cost = w_f * feature_cost(f_new, f_star)
    cost = cost + w_d * distance_cost(d_new)

    if w_n > 0 and approach_local is not None:
        approach_world = R @ approach_local
        cost = cost + w_n * normal_alignment_cost(fusion_new, x, approach_world)

    return cost
