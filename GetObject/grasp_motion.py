"""
Grasp / place motions for Lite6 + parallel gripper.

object_pose: 4x4 homogeneous transform, object frame in robot base frame; translation in meters.
Orientation sent to the arm is roll–pitch–yaw in degrees, from the pose's rotation matrix
(scipy extrinsic ``'xyz'`` Euler angles, same component convention as the previous yaw-only path).
Pre-grasp / lift still offsets **base-frame +Z** by ``pre_height_mm`` (not along tool approach).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

# Same as RealRobotChallenge/checkpoint1.py (TCP length only)
DEFAULT_GRIPPER_LENGTH_MM = 0.067 * 1000
DEFAULT_PRE_HEIGHT_MM = 50
DEFAULT_FAST_PRE_HEIGHT_MM = 40
DEFAULT_FAST_SPEED_MM_S = 200


def _pose_to_cartesian_mm_rpy_deg(object_pose: np.ndarray) -> tuple[float, float, float, float, float, float]:
    x = float(object_pose[0, 3] * 1000.0)
    y = float(object_pose[1, 3] * 1000.0)
    z = float(object_pose[2, 3] * 1000.0)
    rot = Rotation.from_matrix(object_pose[:3, :3].astype(np.float64))
    roll, pitch, yaw = rot.as_euler("xyz", degrees=True)
    return x, y, z, float(roll), float(pitch), float(yaw)


def grasp_at_pose(
    arm: XArmAPI,
    object_pose: np.ndarray,
    *,
    pre_height_mm: float = DEFAULT_PRE_HEIGHT_MM,
    gripper_open_sleep_s: float = 0.5,
    step_sleep_s: float = 0.3,
    gripper_close_sleep_s: float = 0.5,
) -> None:
    """Pick sequence: open gripper, pre-grasp, descend, close, lift to pre height."""
    x, y, z, roll, pitch, yaw = _pose_to_cartesian_mm_rpy_deg(object_pose)

    arm.open_lite6_gripper()
    time.sleep(gripper_open_sleep_s)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, wait=True)
    time.sleep(step_sleep_s)

    arm.set_position(x, y, z, roll, pitch, yaw, wait=True)
    time.sleep(step_sleep_s)

    arm.close_lite6_gripper()
    time.sleep(gripper_close_sleep_s)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, wait=True)
    time.sleep(step_sleep_s)


def place_at_pose(
    arm: XArmAPI,
    object_pose: np.ndarray,
    *,
    pre_height_mm: float = DEFAULT_PRE_HEIGHT_MM,
    step_sleep_s: float = 0.3,
    gripper_open_sleep_s: float = 0.5,
) -> None:
    """Place sequence: pre-place, descend, open gripper, retract."""
    x, y, z, roll, pitch, yaw = _pose_to_cartesian_mm_rpy_deg(object_pose)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, wait=True)
    time.sleep(step_sleep_s)

    arm.set_position(x, y, z, roll, pitch, yaw, wait=True)
    time.sleep(step_sleep_s)

    arm.open_lite6_gripper()
    time.sleep(gripper_open_sleep_s)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, wait=True)
    time.sleep(step_sleep_s)


def fast_grasp_at_pose(
    arm: XArmAPI,
    object_pose: np.ndarray,
    *,
    pre_height_mm: float = DEFAULT_FAST_PRE_HEIGHT_MM,
    speed_mm_s: float = DEFAULT_FAST_SPEED_MM_S,
) -> None:
    """Faster pick (challenge1-style timing/speed)."""
    x, y, z, roll, pitch, yaw = _pose_to_cartesian_mm_rpy_deg(object_pose)

    arm.open_lite6_gripper()
    time.sleep(0.3)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, speed=speed_mm_s, wait=True)
    arm.set_position(x, y, z, roll, pitch, yaw, speed=speed_mm_s, wait=True)

    arm.close_lite6_gripper()
    time.sleep(0.3)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, speed=speed_mm_s, wait=True)


def fast_place_at_pose(
    arm: XArmAPI,
    object_pose: np.ndarray,
    *,
    pre_height_mm: float = DEFAULT_FAST_PRE_HEIGHT_MM,
    speed_mm_s: float = DEFAULT_FAST_SPEED_MM_S,
    place_descent_speed_mm_s: float = 50.0,
) -> None:
    """Faster place with slow final descent."""
    x, y, z, roll, pitch, yaw = _pose_to_cartesian_mm_rpy_deg(object_pose)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, speed=speed_mm_s, wait=True)
    arm.set_position(x, y, z, roll, pitch, yaw, speed=place_descent_speed_mm_s, wait=True)

    arm.open_lite6_gripper()
    time.sleep(0.3)

    arm.set_position(x, y, z + pre_height_mm, roll, pitch, yaw, speed=speed_mm_s, wait=True)
