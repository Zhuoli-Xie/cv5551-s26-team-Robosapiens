"""
Record xArm Lite6 gripper poses interactively. Saved 4x4 matrix is in metres.

Run:  python RecordGraspPose.py --robot-ip <IP> -o <output_dir>

Controls
--------
s      – save camera snapshot (first press only; requires --camera-ids)
Enter  – save current gripper pose
q      – quit
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from xarm.wrapper import XArmAPI

try:
    from utils.zed_camera import ZedCamera
    _ZED_AVAILABLE = True
except ImportError:
    _ZED_AVAILABLE = False

GRIPPER_LENGTH = 0.047 * 1000  # mm, TCP offset


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _depth_vis(depth_m: np.ndarray) -> np.ndarray:
    """Convert float32 depth (metres) to an INFERNO colourmap image."""
    valid = np.isfinite(depth_m)
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = depth_m[valid].min(), depth_m[valid].max()
        norm = np.zeros_like(depth_m)
        if hi > lo:
            norm[valid] = (depth_m[valid] - lo) / (hi - lo)
        grey = (norm * 255).clip(0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
        vis[~valid] = 0
    return vis


def _save_camera(cam, cam_dir: Path, extrinsic: np.ndarray | None = None,
                 timestep: int = 0) -> None:
    """Save RGB, depth (npz + vis png), intrinsics, and optionally extrinsic."""
    (cam_dir / "color").mkdir(parents=True, exist_ok=True)
    (cam_dir / "depth").mkdir(parents=True, exist_ok=True)

    img = cam.image
    cv2.imwrite(str(cam_dir / "color" / f"{timestep}.png"), img)

    depth_m: np.ndarray = cam.depth
    np.savez_compressed(str(cam_dir / "depth" / f"{timestep}.npz"), depth=depth_m)
    cv2.imwrite(str(cam_dir / "depth" / f"{timestep}_vis.png"), _depth_vis(depth_m))

    if extrinsic is not None:
        np.save(str(cam_dir / "camera_extrinsics.npy"), extrinsic)

    K = cam.camera_intrinsic
    np.save(str(cam_dir / "camera_params.npy"),
            np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))

    valid = depth_m[np.isfinite(depth_m)]
    print(f"  {cam_dir.name}: color {img.shape[:2]}, "
          f"depth [{valid.min():.3f}, {valid.max():.3f}] m")


def _save_all_cameras(cameras: list, output_dir: Path,
                      calib_dir: Path | None) -> None:
    """Snapshot all cameras; load extrinsics from calib_dir if provided."""
    for i, cam in enumerate(cameras):
        extrinsic = None
        if calib_dir is not None:
            extrinsic_path = calib_dir / f"camera_{i}" / "camera_extrinsics.npy"
            if extrinsic_path.exists():
                extrinsic = np.load(str(extrinsic_path))
            else:
                print(f"  Warning: extrinsics not found at {extrinsic_path}")
        _save_camera(cam, output_dir / f"camera_{i}", extrinsic=extrinsic)
    print("  Camera snapshot saved.")


# ---------------------------------------------------------------------------
# Robot helpers
# ---------------------------------------------------------------------------

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


def _save_pose(arm: XArmAPI, session_dir: Path, idx: int) -> bool:
    """Read and save the current gripper pose. Returns True on success."""
    state = _read_robot_state(arm)
    tcp = state["tcp_pose_mm_deg"]
    if tcp is None:
        print("  Failed to read TCP pose.")
        return False

    T = _tcp_to_4x4(tcp)
    pose_dir = session_dir / f"pose_{idx:04d}"
    pose_dir.mkdir(exist_ok=True)
    np.save(str(pose_dir / "gripper_pose.npy"), T)
    np.savetxt(str(pose_dir / "gripper_pose.txt"), T)
    (pose_dir / "robot_state.json").write_text(
        json.dumps({"time": datetime.now().isoformat(), **state}, indent=2),
        encoding="utf-8",
    )
    print(f"  Saved pose_{idx:04d}/gripper_pose.npy")
    print(f"  TCP (mm, deg): {[f'{v:.2f}' for v in tcp]}")
    print(f"  Translation (m): [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}]")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record gripper poses from xArm Lite6.")
    p.add_argument("--robot-ip", default="192.168.1.168", help="xArm controller IP")
    p.add_argument("-o", "--output", type=Path, default=Path("demos"),
                   help="Root directory for sessions (default: demos)")
    p.add_argument("--camera-ids", type=int, nargs="+", default=None,
                   help="ZED camera USB indices; enables camera capture on 's' key")
    p.add_argument("--calib-dir", type=Path, default=None,
                   help="Directory with pre-computed camera_X/camera_extrinsics.npy")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # Camera setup — auto-detect if --camera-ids not given
    cameras: list = []
    if _ZED_AVAILABLE:
        if args.camera_ids is not None:
            camera_ids = args.camera_ids
        else:
            import pyzed.sl as sl
            camera_ids = sorted(
                d.id for d in sl.Camera.get_device_list()
                if d.camera_state == sl.CAMERA_STATE.AVAILABLE
            )
        if camera_ids:
            cameras = [ZedCamera(camera_id=cid) for cid in camera_ids]
            print(f"Initialised {len(cameras)} ZED camera(s): {camera_ids}")
        else:
            print("No ZED cameras detected; camera capture disabled.")
    else:
        print("pyzed not available; camera capture disabled.")

    # Robot setup
    print(f"Connecting to robot at {args.robot_ip}...")
    arm = _init_arm(args.robot_ip)

    session_dir = args.output / "target_grasp_pose"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "meta.json").write_text(
        json.dumps({"created": datetime.now().isoformat(),
                    "robot_ip": args.robot_ip}, indent=2),
        encoding="utf-8",
    )

    cameras_saved = False
    pose_idx = 0

    print(f"Session directory: {session_dir.resolve()}")
    if cameras:
        print("s=save cameras | ENTER=save pose | q=quit\n")
    else:
        print("ENTER=save pose | q=quit\n")

    try:
        while True:
            raw = input(f"[{pose_idx}] > ").strip().lower()

            if raw == 'q':
                break

            elif raw == 's':
                if not cameras:
                    print("  No cameras configured (use --camera-ids).")
                elif cameras_saved:
                    print("  Cameras already saved.")
                else:
                    _save_all_cameras(cameras, args.output, args.calib_dir)
                    cameras_saved = True

            elif raw == '':
                if _save_pose(arm, session_dir, pose_idx):
                    pose_idx += 1

    finally:
        for cam in cameras:
            cam.close()
        arm.disconnect()
        print(f"\nDone. {pose_idx} pose(s) saved to {session_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
