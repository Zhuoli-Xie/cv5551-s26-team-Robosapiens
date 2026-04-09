#!/usr/bin/env python3
"""
Move Lite6 TCP to a Cartesian pose you provide (same idea as RealRobotChallenge/checkpoint2:
``arm.set_position(x, y, z, roll, pitch, yaw, ...)``).

Convention (matches typical xArm / checkpoint2 usage):
  - x, y, z: **millimeters** (robot base frame)
  - roll, pitch, yaw: **degrees** by default; pass ``--rpy-rad`` if your numbers are radians
    (checkpoint2's ``BASKET_POSE`` stores rpy in radians).

Example:
  python GetObject/move_to_xyzrpy.py --robot-ip 192.168.1.182 \\
      --x 230.7 --y -297.2 --z 122.8 --roll 180 --pitch 0 --yaw 0

Radians:
  python GetObject/move_to_xyzrpy.py --robot-ip 192.168.1.182 \\
      --x 230.7 --y -297.2 --z 122.8 --roll 3.14159 --pitch 0 --yaw 0 --rpy-rad
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_GET_DIR = Path(__file__).resolve().parent
if str(_GET_DIR) not in sys.path:
    sys.path.insert(0, str(_GET_DIR))

from arm_client import connect_lite6, disconnect_safe
from grasp_motion import DEFAULT_GRIPPER_LENGTH_MM


def main() -> None:
    p = argparse.ArgumentParser(description="Move arm to given xyz (mm) + roll pitch yaw.")
    p.add_argument("--robot-ip", type=str, required=True)
    p.add_argument("--x", type=float, required=True, help="mm, base frame")
    p.add_argument("--y", type=float, required=True, help="mm")
    p.add_argument("--z", type=float, required=True, help="mm")
    p.add_argument("--roll", type=float, required=True)
    p.add_argument("--pitch", type=float, required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument(
        "--rpy-rad",
        action="store_true",
        help="Interpret roll/pitch/yaw as radians (convert to degrees for set_position).",
    )
    p.add_argument(
        "--gripper-length-mm",
        type=float,
        default=DEFAULT_GRIPPER_LENGTH_MM,
        help="TCP Z offset; set 0 if your taught pose assumed no finger length offset.",
    )
    p.add_argument(
        "--no-tcp-offset",
        action="store_true",
        help="Same as --gripper-length-mm 0.",
    )
    p.add_argument("--speed", type=float, default=None, help="Optional mm/s for set_position.")
    p.add_argument("--no-home-start", action="store_true", help="Do not move_gohome before motion.")
    p.add_argument("--no-home-end", action="store_true", help="Do not move_gohome before disconnect.")
    args = p.parse_args()

    roll = args.roll
    pitch = args.pitch
    yaw = args.yaw
    if args.rpy_rad:
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

    tcp_len = 0.0 if args.no_tcp_offset else args.gripper_length_mm

    arm = connect_lite6(
        args.robot_ip,
        gripper_length_mm=tcp_len,
        go_home=not args.no_home_start,
    )

    try:
        kw = {"wait": True}
        if args.speed is not None:
            kw["speed"] = args.speed
        arm.set_position(args.x, args.y, args.z, roll, pitch, yaw, **kw)
    finally:
        disconnect_safe(arm, go_home=not args.no_home_end)


if __name__ == "__main__":
    main()
