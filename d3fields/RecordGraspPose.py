"""
Record xArm Lite6 gripper poses interactively. saved 4x4 matrix is in metres.

Run:  python RecordGraspPose.py --robot-ip <IP> -o <output_dir>

Press ENTER to save current gripper pose, 'q' to quit.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from xarm.wrapper import XArmAPI

GRIPPER_LENGTH = 0.067 * 1000  # mm, TCP offset


def _parse_args():
    p = argparse.ArgumentParser(description="Record gripper poses from xArm Lite6.")
    p.add_argument("--robot-ip", default="192.168.1.182", help="xArm controller IP")
    p.add_argument("-o", "--output", type=Path, default=Path("demos"),
                   help="Root directory for sessions (default: demos)")
    return p.parse_args()


def _init_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    return arm


def _read_robot_state(arm: XArmAPI) -> dict:
    code_p, pos = arm.get_position(is_radian=False)
    code_j, angles = arm.get_servo_angle(is_radian=False)
    return {
        "code_position": int(code_p),
        "tcp_pose_mm_deg": list(pos) if pos is not None else None,
        "code_joints": int(code_j),
        "joints_deg": list(angles) if angles is not None else None,
    }


def _tcp_to_4x4(tcp_pose_mm_deg: list[float]) -> np.ndarray:
    """Convert xArm TCP pose [x,y,z,rx,ry,rz] (mm, deg) to 4x4 matrix (metres)."""
    x, y, z, rx, ry, rz = tcp_pose_mm_deg
    rvec = np.radians([rx, ry, rz]).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]
    return T


def main() -> int:
    args = _parse_args()

    session_dir = args.output / "target_grasp_pose"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to robot at {args.robot_ip}...")
    arm = _init_arm(args.robot_ip)

    meta = {
        "created": datetime.now().isoformat(),
        "robot_ip": args.robot_ip,
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    capture_idx = 0
    print(f"Session directory: {session_dir.resolve()}")
    print("Press ENTER to save pose, 'q' to quit.\n")

    try:
        while True:
            raw = input(f"[{capture_idx}] ENTER to capture (q to quit) > ").strip()
            if raw.lower() == 'q':
                break

            state = _read_robot_state(arm)
            tcp = state["tcp_pose_mm_deg"]
            if tcp is None:
                print("  Failed to read TCP pose.")
                continue

            T = _tcp_to_4x4(tcp)

            # Save pose
            pose_dir = session_dir / f"pose_{capture_idx:04d}"
            pose_dir.mkdir(exist_ok=True)
            np.save(str(pose_dir / "gripper_pose.npy"), T)
            np.savetxt(str(pose_dir / "gripper_pose.txt"), T)
            (pose_dir / "robot_state.json").write_text(
                json.dumps({"time": datetime.now().isoformat(), **state}, indent=2),
                encoding="utf-8",
            )

            print(f"  Saved pose_{capture_idx:04d}/gripper_pose.npy")
            print(f"  TCP (mm, deg): {[f'{v:.2f}' for v in tcp]}")
            print(f"  Translation (m): [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]")
            capture_idx += 1

    finally:
        arm.disconnect()
        print(f"\nDone. {capture_idx} pose(s) saved to {session_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
