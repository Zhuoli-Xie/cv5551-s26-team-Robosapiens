"""
dual_camera_calibration.py

Calibrates two ZED cameras against the robot base frame using AprilTags
(same method as checkpoint0), then derives the inter-camera transform.

Coordinate convention
---------------------
T_cam_robot  :  transforms a point in the ROBOT frame into the CAMERA frame
                    p_cam = T_cam_robot @ p_robot

T_cam1_cam2  :  transforms a point in CAM2 frame into CAM1 frame
                    Derived as: T_cam1_robot @ inv(T_cam2_robot)

Usage
-----
    python dual_camera_calibration.py            # calibrate, save, show axes
    python dual_camera_calibration.py --test     # load saved + verify (no cameras)
    python dual_camera_calibration.py --both     # calibrate + verify
    python dual_camera_calibration.py --test --interactive   # verify + point tool
    python dual_camera_calibration.py --id1 1 --id2 0       # swap USB order
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes

# ---------------------------------------------------------------------------
# AprilTag / PnP config  (identical to checkpoint0)
# ---------------------------------------------------------------------------

TAG_SIZE = 0.08  # metres

TAG_CENTER_COORDINATES = [
    [0.38,  0.4],   # tag 0
    [0.38, -0.4],   # tag 1
    [0.0,   0.4],   # tag 2
    [0.0,  -0.4],   # tag 3
]


def get_pnp_pairs(tags):
    """Return (world_points [N,3], image_points [N,2]) for tag IDs 0-3."""
    world_points = np.empty([0, 3])
    image_points = np.empty([0, 2])

    for tag in tags:
        if tag.tag_id > 3:
            continue
        cx, cy = TAG_CENTER_COORDINATES[tag.tag_id]
        hs = TAG_SIZE / 2
        corners_world = [
            [cx - hs, cy + hs, 0],  # bottom-left  -> corners[0]
            [cx - hs, cy - hs, 0],  # bottom-right -> corners[1]
            [cx + hs, cy - hs, 0],  # top-right    -> corners[2]
            [cx + hs, cy + hs, 0],  # top-left     -> corners[3]
        ]
        for wp, ip in zip(corners_world, tag.corners):
            world_points = np.vstack([world_points, wp])
            image_points = np.vstack([image_points, ip])

    return world_points, image_points


def get_transform_camera_robot(observation, camera_intrinsic, label="camera"):
    """Estimate T_cam_robot (4x4) from AprilTag detections. Returns None on failure."""
    detector = Detector(families='tag36h11')

    gray = observation
    if len(observation.shape) > 2:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)

    tags = detector.detect(gray, estimate_tag_pose=False)
    print(f"[{label}] Tags detected: {len(tags)}")

    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print(f"[{label}] Only {world_points.shape[0]} corners found — need at least 4.")
        return None

    success, rvec, tvec = cv2.solvePnP(
        world_points.astype(np.float64),
        image_points.astype(np.float64),
        camera_intrinsic,
        None,
    )
    if not success:
        print(f"[{label}] solvePnP failed.")
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    projected, _ = cv2.projectPoints(
        world_points.astype(np.float64), rvec, tvec, camera_intrinsic, None
    )
    errors = np.linalg.norm(image_points - projected.reshape(-1, 2), axis=1)
    print(f"[{label}] Reprojection — mean: {errors.mean():.2f} px, max: {errors.max():.2f} px")

    return T


# ---------------------------------------------------------------------------
# Core calibration routine
# ---------------------------------------------------------------------------

def calibrate_two_cameras(cam1: ZedCamera, cam2: ZedCamera):
    """Capture frames from both cameras and compute all three transforms."""
    print("\n=== Capturing frames ===")
    img1 = cam1.image
    img2 = cam2.image

    print("\n=== Calibrating CAM1 -> Robot ===")
    T1 = get_transform_camera_robot(img1, cam1.camera_intrinsic, label="cam1")

    print("\n=== Calibrating CAM2 -> Robot ===")
    T2 = get_transform_camera_robot(img2, cam2.camera_intrinsic, label="cam2")

    if T1 is None or T2 is None:
        return None

    T_cam1_cam2 = T1 @ np.linalg.inv(T2)  # cam2 -> robot -> cam1

    return {
        "T_cam1_robot": T1,
        "T_cam2_robot": T2,
        "T_cam1_cam2":  T_cam1_cam2,
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

SAVE_PATH = Path("calibration_results.npz")


def save_calibration(results: dict, path: Path = SAVE_PATH):
    np.savez(path, **results)
    print(f"\nCalibration saved -> {path.resolve()}")


def load_calibration(path: Path = SAVE_PATH) -> dict:
    data = np.load(path)
    results = {k: data[k] for k in data.files}
    print(f"Calibration loaded from {path.resolve()}")
    return results


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _fmt_matrix(label, M):
    print(f"\n{label}")
    print(np.array2string(M, formatter={"float_kind": lambda x: f"{x:10.5f}"}))


def verify_transforms(results: dict):
    T1  = results["T_cam1_robot"]
    T2  = results["T_cam2_robot"]
    T12 = results["T_cam1_cam2"]

    print("\n" + "="*60)
    print("CALIBRATION VERIFICATION REPORT")
    print("="*60)

    _fmt_matrix("T_cam1_robot  (robot -> cam1)", T1)
    _fmt_matrix("T_cam2_robot  (robot -> cam2)", T2)
    _fmt_matrix("T_cam1_cam2   (cam2  -> cam1)", T12)

    # Rotation matrix validity
    print("\n-- Rotation matrix checks --")
    for label, T in [("cam1_robot", T1), ("cam2_robot", T2), ("cam1_cam2", T12)]:
        R = T[:3, :3]
        det = np.linalg.det(R)
        orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
        ok = abs(det - 1) < 1e-4 and orth_err < 1e-4
        print(f"  {label:15s}  det(R)={det:.6f}  orth_err={orth_err:.2e}  [{'OK' if ok else 'FAIL'}]")

    # T12 consistency
    T12_check = T1 @ np.linalg.inv(T2)
    err = np.max(np.abs(T12 - T12_check))
    print(f"\n-- T_cam1_cam2 consistency: max_err={err:.2e}  [{'OK' if err < 1e-8 else 'FAIL'}]")

    # Translation magnitudes
    print("\n-- Translation magnitudes --")
    print(f"  |t_cam1_robot| = {np.linalg.norm(T1[:3,3]):.3f} m")
    print(f"  |t_cam2_robot| = {np.linalg.norm(T2[:3,3]):.3f} m")
    print(f"  |t_cam1_cam2|  = {np.linalg.norm(T12[:3,3]):.3f} m  (inter-camera baseline)")

    # Synthetic point round-trip through known tag positions
    print("\n-- Synthetic point round-trip --")
    test_points = np.array([
        [0.38,  0.4,  0.0, 1.0],
        [0.38, -0.4,  0.0, 1.0],
        [0.0,   0.4,  0.0, 1.0],
        [0.0,  -0.4,  0.0, 1.0],
        [0.19,  0.0,  0.1, 1.0],
    ])
    print(f"  {'Robot point (m)':35s}  {'cam1 direct vs cam2->cam1':>28s}")
    all_ok = True
    for p in test_points:
        p_cam1_direct = T1  @ p
        p_cam1_via2   = T12 @ (T2 @ p)
        diff_mm = np.linalg.norm(p_cam1_direct[:3] - p_cam1_via2[:3]) * 1000
        ok = diff_mm < 0.1
        if not ok:
            all_ok = False
        coord = f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
        print(f"  {coord:35s}  {diff_mm:8.4f} mm  [{'OK' if ok else 'WARN'}]")

    print("\n" + "="*60)
    print(f"Verification complete -- {'ALL OK' if all_ok else 'CHECK WARNINGS ABOVE'}")
    print("="*60)


def interactive_point_test(results: dict):
    """Type a point in any frame, see it transformed into all others."""
    T1      = results["T_cam1_robot"]
    T2      = results["T_cam2_robot"]
    T12     = results["T_cam1_cam2"]
    T1_inv  = np.linalg.inv(T1)
    T2_inv  = np.linalg.inv(T2)
    T12_inv = np.linalg.inv(T12)

    print("\n=== Interactive Point Transformer ===")
    print("Enter:  <frame> <X> <Y> <Z>  (metres)")
    print("Frames: robot | cam1 | cam2")
    print("Type 'q' to quit.\n")

    while True:
        raw = input("> ").strip()
        if raw.lower() == 'q':
            break
        parts = raw.split()
        if len(parts) != 4 or parts[0] not in ("robot", "cam1", "cam2"):
            print("  Usage:  robot 0.38 0.4 0.0")
            continue
        try:
            xyz = np.array([float(x) for x in parts[1:]] + [1.0])
        except ValueError:
            print("  Invalid numbers.")
            continue

        frame = parts[0]
        if frame == "robot":
            p_robot = xyz
            p_cam1  = T1     @ p_robot
            p_cam2  = T2     @ p_robot
        elif frame == "cam1":
            p_cam1  = xyz
            p_robot = T1_inv  @ p_cam1
            p_cam2  = T12_inv @ p_cam1
        else:
            p_cam2  = xyz
            p_robot = T2_inv @ p_cam2
            p_cam1  = T12    @ p_cam2

        def fmt(v):
            return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}) m"

        print(f"  robot : {fmt(p_robot)}")
        print(f"  cam1  : {fmt(p_cam1)}")
        print(f"  cam2  : {fmt(p_cam2)}\n")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_calibration(cam1: ZedCamera, cam2: ZedCamera, results: dict):
    """Draw robot-frame axes on each camera's live image to visually confirm calibration."""
    T1 = results["T_cam1_robot"]
    T2 = results["T_cam2_robot"]

    img1 = cam1.image.copy()
    img2 = cam2.image.copy()

    draw_pose_axes(img1, cam1.camera_intrinsic, T1, size=TAG_SIZE)
    draw_pose_axes(img2, cam2.camera_intrinsic, T2, size=TAG_SIZE)

    for name, img in [("CAM1 -- robot origin", img1), ("CAM2 -- robot origin", img2)]:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 960, 540)
        cv2.imshow(name, img)

    print("\nVisualization open -- press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dual ZED camera calibration via AprilTags")
    p.add_argument("--test",        action="store_true",
                   help="Load saved calibration and verify (no cameras needed)")
    p.add_argument("--both",        action="store_true",
                   help="Calibrate, save, then verify")
    p.add_argument("--vis",         action="store_true",
                   help="Show axis visualisation after calibration (default: on)")
    p.add_argument("--no-vis",      action="store_true",
                   help="Skip visualisation")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive point transformer")
    p.add_argument("--id1",  type=int, default=0,
                   help="USB device index for cam1 (default: 0)")
    p.add_argument("--id2",  type=int, default=1,
                   help="USB device index for cam2 (default: 1)")
    p.add_argument("--save", type=str, default=str(SAVE_PATH),
                   help=f"Path to save/load calibration (default: {SAVE_PATH})")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = Path(args.save)

    if args.test:
        results = load_calibration(save_path)
        verify_transforms(results)
        if args.interactive:
            interactive_point_test(results)
        return

    print(f"Initializing cam1 (id={args.id1}) and cam2 (id={args.id2})...")
    cam1 = ZedCamera(camera_id=args.id1)
    cam2 = ZedCamera(camera_id=args.id2)

    try:
        results = calibrate_two_cameras(cam1, cam2)
        if results is None:
            print("\nCalibration failed. Make sure the arena poster is visible to both cameras.")
            sys.exit(1)

        save_calibration(results, save_path)

        show_vis = not args.no_vis  # show by default unless --no-vis
        if show_vis:
            visualize_calibration(cam1, cam2, results)

        if args.both:
            verify_transforms(results)

        if args.interactive:
            interactive_point_test(results)

    finally:
        cam1.close()
        cam2.close()


if __name__ == "__main__":
    main()
