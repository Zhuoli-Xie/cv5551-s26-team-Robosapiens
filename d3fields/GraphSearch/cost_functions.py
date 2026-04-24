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
        normals      : (N, 3) tensor, unit normals
        normals_valid: (N,)   bool tensor — True when all 6 FD probes hit a
                              valid camera view. Points with any invalid probe
                              would otherwise inherit the 1e3 sentinel and
                              poison the normal direction.
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
    v = out['valid_mask'].reshape(6, N)

    normals = torch.stack([
        d[0] - d[1],
        d[2] - d[3],
        d[4] - d[5],
    ], dim=1) / (2 * eps)

    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)
    normals_valid = v.all(dim=0)
    return normals, normals_valid


# ---------------------------------------------------------------------------
# Individual cost terms
# ---------------------------------------------------------------------------

def feature_cost(f_new, f_star, valid_mask=None):
    """Cosine-distance feature matching cost: sum_i (1 - cos(f_new_i, f_i*)).

    Matches the coarse stage's objective so fine optimization refines rather
    than fights the SVD Procrustes init. Raw L2 on DINO features is dominated
    by magnitude variation across patch tokens and produces spurious minima.

    Invalid-mask points are zeroed out (Fusion returns a zero feature vector
    for them, which would otherwise contribute cos_sim=0 → cost=1 each).
    """
    f_new_n = torch.nn.functional.normalize(f_new, dim=1, eps=1e-8)
    f_star_n = torch.nn.functional.normalize(f_star, dim=1, eps=1e-8)
    cos_sim = (f_new_n * f_star_n).sum(dim=1)
    per_point = 1.0 - cos_sim
    if valid_mask is not None:
        per_point = per_point * valid_mask.float()
    return per_point.sum()


def distance_cost(d_new, valid_mask=None):
    """Squared-SDF surface-fit cost: sum_i d_new(x_i)^2.

    Two-sided penalty that pulls query points onto the surface from either
    side. The signed-sum form let the optimizer drive points deep into the
    object to drop the cost without improving the fit.

    Invalid points carry a 1e3 sentinel from Fusion — masking them out keeps
    the cost on a real scale (≤ mu² per valid point) instead of the sentinel
    drowning everything at ~1e6 per invalid point.
    """
    per_point = d_new ** 2
    if valid_mask is not None:
        per_point = per_point * valid_mask.float()
    return per_point.sum()


def normal_alignment_cost(fusion, x, approach_world, valid_mask=None):
    """Normal alignment cost: sum_i (1 + n(x_i) . a)^2.

    Penalizes gripper approach directions that are not anti-parallel
    to the surface normal (i.e., encourages the gripper to approach
    perpendicular to the surface).

    The FD probe returns its own validity mask; we AND it with the
    caller-supplied mask so only fully-valid points contribute.

    Args:
        fusion         : Fusion object (for SDF normal estimation)
        x              : (N, 3) tensor, transformed query points
        approach_world : (3,) tensor, gripper approach direction in world frame
        valid_mask     : (N,) bool tensor or None; center-point validity

    Returns:
        scalar tensor
    """
    normals, normals_valid = estimate_normals_fd(fusion, x)
    if valid_mask is not None:
        normals_valid = normals_valid & valid_mask
    alignment = torch.sum(normals * approach_world.unsqueeze(0), dim=1)
    per_point = (1.0 + alignment) ** 2
    per_point = per_point * normals_valid.float()
    return per_point.sum()


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

    return_names = ['dino_feats'] if w_f > 0 else []
    out = fusion_new.eval(x, return_names=return_names)
    d_new = out['dist']
    valid = out['valid_mask']

    cost = torch.zeros((), device=x.device, dtype=x.dtype)

    if w_f > 0:
        cost = cost + w_f * feature_cost(out['dino_feats'], f_star, valid_mask=valid)

    if w_d > 0:
        cost = cost + w_d * distance_cost(d_new, valid_mask=valid)

    if w_n > 0 and approach_local is not None:
        approach_world = R @ approach_local
        cost = cost + w_n * normal_alignment_cost(fusion_new, x, approach_world,
                                                   valid_mask=valid)

    return cost
