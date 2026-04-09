#!/usr/bin/env python3
"""
Detect cube AprilTag (id 4) + table tags, then grasp / place — same pipeline as
RealRobotChallenge/checkpoint1 main(), using GetObject grasp helpers.

Requires running with RealRobotChallenge on sys.path (ZED + pupil_apriltags + checkpoints).

From repository root:
  python GetObject/grasp_apriltag_cube.py --robot-ip 192.168.1.183
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CHALLENGE = _REPO_ROOT / "RealRobotChallenge"
_GET_DIR = Path(__file__).resolve().parent

for _d in (_CHALLENGE, _GET_DIR):
    s = str(_d)
    if s not in sys.path:
        sys.path.insert(0, s)

from arm_client import connect_lite6, disconnect_safe
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import GRIPPER_LENGTH, get_transform_cube
from grasp_motion import grasp_at_pose, place_at_pose
from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera


def main() -> None:
    parser = argparse.ArgumentParser(description="AprilTag cube pose -> grasp (checkpoint1 flow).")
    parser.add_argument("--robot-ip", type=str, default="192.168.1.183")
    parser.add_argument("--no-place", action="store_true", help="Only grasp and hold (still lifts).")
    args = parser.parse_args()

    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    arm = connect_lite6(args.robot_ip, gripper_length_mm=GRIPPER_LENGTH, go_home=True)

    try:
        cv_image = zed.image
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed camera–robot transform (table AprilTags).")
            return

        result = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot)
        if result is None:
            print("Cube tag not detected.")
            return

        t_robot_cube, t_cam_cube = result

        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow("Verify cube pose (press k to grasp)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Verify cube pose (press k to grasp)", 1280, 720)
        cv2.imshow("Verify cube pose (press k to grasp)", cv_image)
        key = cv2.waitKey(0)

        if key == ord("k"):
            cv2.destroyAllWindows()
            grasp_at_pose(arm, t_robot_cube)
            if not args.no_place:
                place_at_pose(arm, t_robot_cube)
        else:
            cv2.destroyAllWindows()
            print("Aborted (did not press k).")

    finally:
        disconnect_safe(arm, go_home=True)
        zed.close()


if __name__ == "__main__":
    main()
