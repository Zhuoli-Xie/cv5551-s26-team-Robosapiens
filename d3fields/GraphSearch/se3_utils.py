"""SE(3) parameterization helpers.

Rotations are parameterized with the continuous 6D representation from
Zhou et al., "On the Continuity of Rotation Representations in Neural
Networks" (CVPR 2019): two 3-vectors are mapped to a valid rotation
matrix via Gram-Schmidt, giving continuous gradients everywhere (no
singularity at θ=π, no double cover).

Axis-angle (Rodrigues) is kept as a small utility for sampling random
rotation perturbations in radians during multi-start restarts.
"""

import torch


def rotation_6d_to_matrix(d6):
    """Gram-Schmidt from 6D (6,) to rotation matrix (3, 3).

    Columns 1-2 of the output span the plane of the first two input
    vectors; column 3 is their right-handed normal.
    """
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / (torch.norm(a1) + 1e-8)
    b2 = a2 - torch.dot(b1, a2) * b1
    b2 = b2 / (torch.norm(b2) + 1e-8)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=1)


def matrix_to_rotation_6d(R):
    """Pack a rotation matrix (3, 3) into a 6D tensor (first two columns)."""
    return torch.cat([R[:, 0], R[:, 1]])


def pose_from_params(d6, t):
    """Build 4x4 SE(3) matrix from 6D rotation (6,) and translation (3,)."""
    T = torch.eye(4, device=d6.device, dtype=d6.dtype)
    T[:3, :3] = rotation_6d_to_matrix(d6)
    T[:3, 3] = t
    return T


def axis_angle_to_rotation_matrix(aa):
    """Rodrigues formula — used only for sampling radian-scale perturbations."""
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
