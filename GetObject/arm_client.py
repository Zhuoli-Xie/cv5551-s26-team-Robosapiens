"""Connect and configure Lite6 for grasp scripts (matches RealRobotChallenge checkpoints)."""

from __future__ import annotations

import time

from xarm.wrapper import XArmAPI

from grasp_motion import DEFAULT_GRIPPER_LENGTH_MM


def connect_lite6(
    robot_ip: str,
    *,
    gripper_length_mm: float = DEFAULT_GRIPPER_LENGTH_MM,
    go_home: bool = True,
    home_speed: float | None = None,
) -> XArmAPI:
    """
    Enable motion, set TCP offset for parallel gripper length, mode 0, state 0.

    If home_speed is set, move_gohome uses that speed (mm/s), same as challenge1.
    """
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, gripper_length_mm, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    if go_home:
        if home_speed is not None:
            arm.move_gohome(speed=home_speed, wait=True)
        else:
            arm.move_gohome(wait=True)
        time.sleep(0.5)
    return arm


def disconnect_safe(arm: XArmAPI, *, go_home: bool = True) -> None:
    if go_home:
        arm.move_gohome(wait=True)
        time.sleep(0.3)
    arm.disconnect()
