"""Pose error metrics."""

import numpy as np


def pose_error(T_pred, T_gt):
    """Compute rotation (deg) and translation (m) error between two SE(3) poses."""
    R_err = T_pred[:3, :3] @ T_gt[:3, :3].T
    cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    rot_err_deg = np.degrees(np.arccos(cos_angle))
    trans_err = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
    return rot_err_deg, trans_err
