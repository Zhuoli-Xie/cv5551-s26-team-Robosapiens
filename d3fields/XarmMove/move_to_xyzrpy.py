#!/usr/bin/env python3
"""
Move Lite6 TCP to a Cartesian pose (same as RealRobotChallenge/checkpoint2 ``set_position``).

Edit the CONFIG block below, then run:
  python GetObject/move_to_xyzrpy.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_GET_DIR = Path(__file__).resolve().parent
if str(_GET_DIR) not in sys.path:
    sys.path.insert(0, str(_GET_DIR))

from arm_client import connect_lite6, disconnect_safe
from grasp_motion import DEFAULT_GRIPPER_LENGTH_MM

# ---------------------------------------------------------------------------
# CONFIG — 只改这里
# ---------------------------------------------------------------------------

ROBOT_IP = "192.168.1.182"

# x, y, z: millimeters (robot base frame)
X_MM = 230.7
Y_MM = -297.2
Z_MM = 122.8

# roll, pitch, yaw: degrees unless RPY_IS_RADIANS is True
ROLL = 180.0
PITCH = 0.0
YAW = 0.0
RPY_IS_RADIANS = False  # True 时与 checkpoint2 的 BASKET_POSE 后三项一致（弧度）

# TCP Z offset (mm). True = 不补偿夹爪长度（与示教时一致时可开）
GRIPPER_LENGTH_MM = DEFAULT_GRIPPER_LENGTH_MM
NO_TCP_OFFSET = False

# Motion
SPEED_MM_S = None  # e.g. 200，或 None 用控制器默认
GO_HOME_BEFORE = True
GO_HOME_AFTER = True

# ---------------------------------------------------------------------------


def main() -> None:
    roll = ROLL
    pitch = PITCH
    yaw = YAW
    if RPY_IS_RADIANS:
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

    tcp_len = 0.0 if NO_TCP_OFFSET else GRIPPER_LENGTH_MM

    arm = connect_lite6(
        ROBOT_IP,
        gripper_length_mm=tcp_len,
        go_home=GO_HOME_BEFORE,
    )

    try:
        kw = {"wait": True}
        if SPEED_MM_S is not None:
            kw["speed"] = SPEED_MM_S
        arm.set_position(X_MM, Y_MM, Z_MM, roll, pitch, yaw, **kw)
    finally:
        disconnect_safe(arm, go_home=GO_HOME_AFTER)


if __name__ == "__main__":
    main()
