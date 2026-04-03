"""
Record synchronized ZED RGB + point clouds and xArm Lite6 state; merge clouds in a common frame.

Run from this directory:  python record_demo.py --robot-ip <IP>

Keys (OpenCV window):
  SPACE  Capture current frame(s) + robot state to disk and in-memory merge buffer.
  m      Merge all buffered captures, voxel-downsample, save merged_session.ply.
  c      Clear merge buffer (does not delete files on disk).
  q      Quit.

Dual ZED: pass --zed-serial twice (order matches --T-robot-cam npy files).
Without calibration .npy files, merge uses identity transforms (all points stay in each camera's
left optical frame — fill in 4x4 ``T_base_cam`` files when you have hand-eye or extrinsics).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera

# Same TCP offset convention as RealRobotChallenge/checkpoint1.py (mm)
GRIPPER_LENGTH = 0.067 * 1000


def _parse_args():
    p = argparse.ArgumentParser(description="Record demos: ZED + robot state; merge point clouds.")
    p.add_argument(
        "--robot-ip",
        default="192.168.1.183",
        help="xArm controller IP (empty with --no-robot)",
    )
    p.add_argument("--no-robot", action="store_true", help="Camera-only; skip XArm connection.")
    p.add_argument(
        "--zed-serial",
        action="append",
        type=int,
        default=None,
        help="ZED serial number; specify twice for two cameras (first = cam0). Omit for one default device.",
    )
    p.add_argument(
        "--T-robot-cam",
        action="append",
        default=None,
        help="Path to 4x4 numpy .npy: base_T_leftcam for each ZED, same order as --zed-serial. "
        "Omitted cameras use identity.",
    )
    p.add_argument("--output", type=Path, default=Path("demos"), help="Root directory for sessions.")
    p.add_argument(
        "--voxel",
        type=float,
        default=0.01,
        help="Voxel size (meters) for merged cloud downsampling.",
    )
    return p.parse_args()


def _load_T_path(path_str: str | None) -> numpy.ndarray:
    if not path_str:
        return numpy.eye(4, dtype=numpy.float64)
    p = Path(path_str)
    if not p.is_file():
        print(f"Warning: missing T file {p}, using identity.")
        return numpy.eye(4, dtype=numpy.float64)
    T = numpy.load(p)
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 in {p}, got {T.shape}")
    return T.astype(numpy.float64)


def _point_cloud_hw_to_xyz_xyz(pc: numpy.ndarray) -> numpy.ndarray:
    """ZED XYZ measure: HxWx4 float32; return Nx3 finite points in meters."""
    if pc is None:
        return numpy.zeros((0, 3), dtype=numpy.float64)
    flat = pc.reshape(-1, pc.shape[-1])[:, :3].astype(numpy.float64)
    ok = numpy.isfinite(flat).all(axis=1) & (numpy.linalg.norm(flat, axis=1) > 1e-6)
    return flat[ok]


def _apply_transform_points(xyz: numpy.ndarray, T_base_cam: numpy.ndarray) -> numpy.ndarray:
    if xyz.size == 0:
        return xyz
    R = T_base_cam[:3, :3]
    t = T_base_cam[:3, 3]
    return (xyz @ R.T) + t


def _bgr_from_zed_image(img: numpy.ndarray) -> numpy.ndarray:
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _read_robot_state(arm: XArmAPI | None) -> dict:
    if arm is None:
        return {"connected": False}
    code_p, pos = arm.get_position(is_radian=False)
    code_j, angles = arm.get_servo_angle(is_radian=False)
    return {
        "connected": True,
        "code_position": int(code_p),
        "tcp_pose_mm_deg": list(pos) if pos is not None else None,
        "code_joints": int(code_j),
        "joints_deg": list(angles) if angles is not None else None,
    }


def _init_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    return arm


def _open_cameras(serials: list[int] | None) -> list[ZedCamera]:
    if not serials:
        return [ZedCamera()]
    return [ZedCamera(serial_number=s) for s in serials]


def _resolve_transforms(n_cam: int, T_paths: list[str] | None) -> list[numpy.ndarray]:
    paths = T_paths or []
    Ts = [_load_T_path(paths[i]) if i < len(paths) else numpy.eye(4) for i in range(n_cam)]
    return Ts


def merge_buffered_captures(
    buffer: list[dict],
    T_base_cams: list[numpy.ndarray],
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Each capture dict has keys zed0_xyz, zed1_xyz, ... (Nx3) in camera frames."""
    if not buffer:
        raise ValueError("Merge buffer is empty.")
    all_pts = []
    for cap in buffer:
        for i, T in enumerate(T_base_cams):
            key = f"zed{i}_xyz"
            if key not in cap:
                continue
            xyz = cap[key]
            all_pts.append(_apply_transform_points(xyz, T))
    if not all_pts:
        raise ValueError("No point data in buffer.")
    merged = numpy.concatenate(all_pts, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged)
    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def main() -> int:
    args = _parse_args()
    serials = args.zed_serial
    if serials is not None and len(serials) > 2:
        print("Using first two --zed-serial values only.")
        serials = serials[:2]

    out_root: Path = args.output
    session_dir = out_root / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    arm: XArmAPI | None = None
    if not args.no_robot:
        if not args.robot_ip:
            print("Robot IP missing; use --robot-ip or --no-robot.", file=sys.stderr)
            return 1
        arm = _init_arm(args.robot_ip)

    try:
        cams = _open_cameras(serials)
    except SystemExit:
        return 1

    T_base_cams = _resolve_transforms(len(cams), args.T_robot_cam)

    meta = {
        "created": datetime.now().isoformat(),
        "n_cameras": len(cams),
        "zed_serials": serials if serials else ["default"],
        "T_robot_cam_files": args.T_robot_cam or [],
        "robot_ip": args.robot_ip if not args.no_robot else None,
        "intrinsics": [c.camera_intrinsic.tolist() for c in cams],
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    merge_buffer: list[dict] = []
    capture_idx = 0

    print(__doc__)
    print(f"Session directory: {session_dir.resolve()}")

    try:
        while True:
            frames = [c.get_synced_frame() for c in cams]
            if any(im is None for im, _ in frames):
                cv2.waitKey(10)
                continue

            preview = numpy.hstack([_bgr_from_zed_image(im) for im, _ in frames])
            cv2.imshow("record_demo (SPACE=capture, m=merge, q=quit)", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("c"):
                merge_buffer.clear()
                print("Merge buffer cleared.")
                continue
            if key == ord("m"):
                try:
                    pcd = merge_buffered_captures(merge_buffer, T_base_cams, args.voxel)
                    ply_path = session_dir / "merged_session.ply"
                    o3d.io.write_point_cloud(str(ply_path), pcd)
                    print(f"Wrote merged cloud ({len(pcd.points)} points) -> {ply_path}")
                except ValueError as e:
                    print(f"Merge skipped: {e}")
                continue
            if key != ord(" "):
                continue

            cap_dir = session_dir / f"capture_{capture_idx:04d}"
            cap_dir.mkdir(parents=True, exist_ok=True)
            wall_t = datetime.now().isoformat()
            robot_state = _read_robot_state(arm)
            (cap_dir / "robot_state.json").write_text(
                json.dumps({"time": wall_t, **robot_state}, indent=2),
                encoding="utf-8",
            )

            cap_entry: dict = {"time": wall_t}
            for i, ((im, pc), cam) in enumerate(zip(frames, cams)):
                bgr = _bgr_from_zed_image(im)
                cv2.imwrite(str(cap_dir / f"zed{i}_image.png"), bgr)
                numpy.save(cap_dir / f"zed{i}_points.npy", pc)
                numpy.save(cap_dir / f"zed{i}_K.npy", cam.camera_intrinsic)
                cap_entry[f"zed{i}_xyz"] = _point_cloud_hw_to_xyz_xyz(pc)

            merge_buffer.append(cap_entry)
            print(f"Saved capture_{capture_idx:04d} ({len(merge_buffer)} in merge buffer)")
            capture_idx += 1

    finally:
        cv2.destroyAllWindows()
        for c in cams:
            c.close()
        if arm is not None:
            arm.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
