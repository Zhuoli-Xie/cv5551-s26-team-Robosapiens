#!/usr/bin/env python3
"""
Execute grasp (and optional place) using a known 4x4 object pose in the robot base frame.

Pose source: .npy file with shape (4, 4), float64/float32, translation in meters
(same convention as RealRobotChallenge checkpoint1).

Usage (from repository root, with GetObject on PYTHONPATH or run as below):
  python GetObject/grasp_from_pose.py --pose object_pose.npy --robot-ip 192.168.1.183

Or from inside GetObject:
  cd GetObject && python grasp_from_pose.py --pose ../poses/cube.npy --robot-ip 192.168.1.183
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow imports when executed as a script (sibling modules in this folder).
_GET_DIR = Path(__file__).resolve().parent
if str(_GET_DIR) not in sys.path:
    sys.path.insert(0, str(_GET_DIR))

from arm_client import connect_lite6, disconnect_safe
from grasp_motion import (
    DEFAULT_GRIPPER_LENGTH_MM,
    fast_grasp_at_pose,
    fast_place_at_pose,
    grasp_at_pose,
    place_at_pose,
)


def _load_pose(path: Path) -> np.ndarray:
    pose = np.load(str(path))
    if pose.shape != (4, 4):
        raise ValueError(f"Expected pose shape (4, 4), got {pose.shape}")
    return pose.astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Grasp object at given 4x4 pose (base frame, meters).")
    p.add_argument("--pose", type=Path, required=True, help="Path to .npy 4x4 homogeneous pose.")
    p.add_argument("--robot-ip", type=str, default="192.168.1.183", help="Lite6 controller IP.")
    p.add_argument(
        "--gripper-length-mm",
        type=float,
        default=DEFAULT_GRIPPER_LENGTH_MM,
        help="TCP Z offset for gripper length (same as checkpoint1).",
    )
    p.add_argument("--fast", action="store_true", help="Use fast_grasp / fast_place (challenge1 style).")
    p.add_argument("--place", action="store_true", help="After grasp, place back at the same pose.")
    p.add_argument("--no-home-start", action="store_true", help="Skip initial move_gohome.")
    p.add_argument("--no-home-end", action="store_true", help="Skip final move_gohome on disconnect.")
    args = p.parse_args()

    object_pose = _load_pose(args.pose)

    arm = connect_lite6(
        args.robot_ip,
        gripper_length_mm=args.gripper_length_mm,
        go_home=not args.no_home_start,
        home_speed=200 if args.fast else None,
    )

    try:
        if args.fast:
            fast_grasp_at_pose(arm, object_pose)
            if args.place:
                fast_place_at_pose(arm, object_pose)
        else:
            grasp_at_pose(arm, object_pose)
            if args.place:
                place_at_pose(arm, object_pose)
    finally:
        disconnect_safe(arm, go_home=not args.no_home_end)


if __name__ == "__main__":
    main()
