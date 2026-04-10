"""SE(3) parameterization helpers (axis-angle <-> rotation matrix)."""

import torch


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
