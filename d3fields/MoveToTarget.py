"""
MoveToTarget.py

Move the xArm Lite6 end-effector to a target pose stored as a 4x4
transform in base_link, translation in metres — i.e. an `ee_pose.npy`
file produced by RecordGraspPose.py.

The matrix is converted back to the xArm pose format
[x, y, z, roll, pitch, yaw] (mm, deg, RPY = extrinsic XYZ Euler) and
sent with `arm.set_position`. No TCP offset is applied, so the
controller reaches the recorded flange (end-effector) pose directly.

Usage:
    python MoveToTarget.py path/to/ee_pose.npy --robot-ip <IP>
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from xarm.wrapper import XArmAPI


# ---------------------------------------------------------------------------
# Pose conversion (inverse of RecordGraspPose._ee_pose_to_4x4)
# ---------------------------------------------------------------------------

def _matrix_to_xarm_pose(T: np.ndarray) -> tuple[float, float, float,
                                                 float, float, float]:
    """4x4 (m) → xArm pose [x,y,z (mm), rx,ry,rz (deg)].

    Decomposes R into extrinsic XYZ Euler angles (matches xArm RPY:
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)).
    """
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {T.shape}")

    R = T[:3, :3]
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock at pitch ≈ ±90°
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    x_mm, y_mm, z_mm = (T[:3, 3] * 1000.0).tolist()
    return (float(x_mm), float(y_mm), float(z_mm),
            float(np.degrees(roll)),
            float(np.degrees(pitch)),
            float(np.degrees(yaw)))


# ---------------------------------------------------------------------------
# Robot setup
# ---------------------------------------------------------------------------

def _init_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, 0, 0, 0, 0])  # match RecordGraspPose.py
    arm.set_mode(0)
    arm.set_state(0)
    return arm


def _fmt_pose(pose) -> str:
    return ", ".join(f"{v:8.3f}" for v in pose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move xArm EE to a saved 4x4 pose, or send it home.")
    p.add_argument("pose_path", type=Path, nargs="?", default=None,
                   help="Path to ee_pose.npy (4x4 in base_link, metres). "
                        "Omit when using --gohome.")
    p.add_argument("--gohome", action="store_true",
                   help="Move the arm to the controller's home pose and exit. "
                        "When set, pose_path is ignored.")
    p.add_argument("--robot-ip", default="192.168.1.168",
                   help="xArm controller IP")
    p.add_argument("--speed", type=float, default=100.0,
                   help="TCP linear speed in mm/s (default: 100)")
    p.add_argument("--mvacc", type=float, default=1000.0,
                   help="TCP acceleration in mm/s^2 (default: 1000)")
    p.add_argument("-y", "--yes", action="store_true",
                   help="Skip the confirmation prompt before moving")
    args = p.parse_args()
    if not args.gohome and args.pose_path is None:
        p.error("pose_path is required unless --gohome is set")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_gohome(arm: XArmAPI, confirm: bool) -> int:
    code_c, current = arm.get_position(is_radian=False)
    if code_c == 0 and current is not None:
        print(f"  Current EE pose:")
        print(f"    {_fmt_pose(current)}")

    if confirm:
        ans = input("Move robot to home pose? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    print("Moving to home position...")
    code = arm.move_gohome(wait=True)
    time.sleep(0.5)
    print(f"move_gohome returned code={code}")
    return 0 if code == 0 else 1


def _run_target(arm: XArmAPI, T: np.ndarray, speed: float, mvacc: float,
                confirm: bool) -> int:
    target = _matrix_to_xarm_pose(T)
    print(f"  Target EE pose [x,y,z (mm), rx,ry,rz (deg)]:")
    print(f"    {_fmt_pose(target)}")

    code_c, current = arm.get_position(is_radian=False)
    if code_c == 0 and current is not None:
        print(f"  Current EE pose:")
        print(f"    {_fmt_pose(current)}")

    if confirm:
        ans = input("Move robot to target pose? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return 0

    x, y, z, rx, ry, rz = target
    code = arm.set_position(x=x, y=y, z=z, roll=rx, pitch=ry, yaw=rz,
                            speed=speed, mvacc=mvacc,
                            wait=True, is_radian=False)
    print(f"set_position returned code={code}")
    if code != 0:
        print("  Motion failed. Look up the code in the xArm error table.")
        return 1

    code_f, final = arm.get_position(is_radian=False)
    if code_f == 0 and final is not None:
        print(f"  Reached EE pose:")
        print(f"    {_fmt_pose(final)}")
    return 0


def main() -> int:
    args = _parse_args()

    if not args.gohome:
        T = np.load(str(args.pose_path))
        print(f"Loaded {args.pose_path}")
    else:
        T = None

    print(f"Connecting to robot at {args.robot_ip}...")
    arm = _init_arm(args.robot_ip)

    try:
        if args.gohome:
            return _run_gohome(arm, confirm=not args.yes)
        return _run_target(arm, T, args.speed, args.mvacc,
                           confirm=not args.yes)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
