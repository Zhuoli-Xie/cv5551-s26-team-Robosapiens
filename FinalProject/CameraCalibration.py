"""
dual_camera_calibration.py

Calibrates two RGBD cameras against a shared robot base frame using
AprilTags (checkpoint0 method), then derives the inter-camera transform.

Coordinate convention
---------------------
T_cam_robot  :  transforms a point expressed in the ROBOT frame
                into the CAMERA frame.
                i.e.  p_cam = T_cam_robot @ p_robot

T_cam1_cam2  :  transforms a point expressed in CAM2 frame
                into CAM1 frame.
                Derived as:  T_cam1_robot @ inv(T_cam2_robot)

Usage
-----
    python dual_camera_calibration.py            # calibrate + save
    python dual_camera_calibration.py --test     # load saved + run tests
    python dual_camera_calibration.py --both     # calibrate, save, then test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

# ---------------------------------------------------------------------------
# Re-use the checkpoint0 helpers exactly as written
# ---------------------------------------------------------------------------
TAG_SIZE = 0.08  # meters

# top-left, top-right, bottom-left, bottom-right (robot XY, Z=0)
TAG_CENTER_COORDINATES = [
    [0.38,  0.4],
    [0.38, -0.4],
    [0.0,   0.4],
    [0.0,  -0.4],
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
            [cx - hs, cy + hs, 0],  # bottom-left  → corners[0]
            [cx - hs, cy - hs, 0],  # bottom-right → corners[1]
            [cx + hs, cy - hs, 0],  # top-right    → corners[2]
            [cx + hs, cy + hs, 0],  # top-left     → corners[3]
        ]
        for wp, ip in zip(corners_world, tag.corners):
            world_points = np.vstack([world_points, wp])
            image_points = np.vstack([image_points, ip])

    return world_points, image_points


def get_transform_camera_robot(observation, camera_intrinsic, label="camera"):
    """
    Estimate T_cam_robot (4×4) from AprilTag detections.

    Parameters
    ----------
    observation      : numpy.ndarray  BGR/BGRA or grayscale image
    camera_intrinsic : numpy.ndarray  3×3 intrinsic matrix
    label            : str            name used in log messages

    Returns
    -------
    transform_mat : numpy.ndarray (4×4) or None on failure
    """
    detector = Detector(families='tag36h11')

    gray = observation
    if len(observation.shape) > 2:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)

    tags = detector.detect(gray, estimate_tag_pose=False)
    print(f"[{label}] Tags detected: {len(tags)}")

    world_points, image_points = get_pnp_pairs(tags)
    if world_points.shape[0] < 4:
        print(f"[{label}] Insufficient corners ({world_points.shape[0]} < 4). Calibration failed.")
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

    print(f"[{label}] Calibration OK. Reprojection check below.")
    _reprojection_report(world_points, image_points, camera_intrinsic, rvec, tvec, label)

    return T


def _reprojection_report(world_pts, img_pts, K, rvec, tvec, label):
    """Print mean reprojection error as a quick sanity check."""
    projected, _ = cv2.projectPoints(
        world_pts.astype(np.float64), rvec, tvec, K, None
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(img_pts - projected, axis=1)
    print(f"  Reprojection error — mean: {errors.mean():.2f} px, "
          f"max: {errors.max():.2f} px  ({len(errors)} corners)")


# ---------------------------------------------------------------------------
# ZED camera wrapper — swap this out for your actual camera class
# ---------------------------------------------------------------------------

class ZedCamera:
    """
    Thin wrapper around the ZED SDK.
    Replace with your real camera driver if needed.
    Expects pyzed (sl module) to be installed.
    """

    def __init__(self, camera_id=0):
        import pyzed.sl as sl

        self._sl = sl
        self._zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.set_from_camera_id(camera_id)

        err = self._zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {err}")

        # Warm up
        image_mat = sl.Mat()
        for _ in range(5):
            self._zed.grab()
        self._zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        self._image_mat = image_mat

        calib = (
            self._zed.get_camera_information()
            .camera_configuration.calibration_parameters.left_cam
        )
        self.camera_intrinsic = np.array([
            [calib.fx, 0,        calib.cx],
            [0,        calib.fy, calib.cy],
            [0,        0,        1       ],
        ])

    @property
    def image(self):
        self._zed.grab()
        self._zed.retrieve_image(self._image_mat, self._sl.VIEW.LEFT)
        return self._image_mat.get_data()  # BGRA numpy array

    def close(self):
        self._zed.close()


# ---------------------------------------------------------------------------
# Core calibration routine
# ---------------------------------------------------------------------------

def calibrate_two_cameras(cam1: ZedCamera, cam2: ZedCamera):
    """
    Capture one frame from each camera and compute:
        T_cam1_robot  (4×4)
        T_cam2_robot  (4×4)
        T_cam1_cam2   (4×4)  — transforms cam2 pts into cam1 frame

    Returns a dict with all three matrices (as numpy arrays), or None on failure.
    """
    print("\n=== Capturing frames ===")
    img1 = cam1.image
    img2 = cam2.image

    print("\n=== Calibrating CAM1 → Robot ===")
    T1 = get_transform_camera_robot(img1, cam1.camera_intrinsic, label="cam1")

    print("\n=== Calibrating CAM2 → Robot ===")
    T2 = get_transform_camera_robot(img2, cam2.camera_intrinsic, label="cam2")

    if T1 is None or T2 is None:
        return None

    # Derive inter-camera transform
    # T_cam1_cam2: takes a point in cam2 frame → robot frame → cam1 frame
    T_cam1_cam2 = T1 @ np.linalg.inv(T2)

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
    np.savez(path,
             T_cam1_robot=results["T_cam1_robot"],
             T_cam2_robot=results["T_cam2_robot"],
             T_cam1_cam2=results["T_cam1_cam2"])
    print(f"\nCalibration saved → {path.resolve()}")


def load_calibration(path: Path = SAVE_PATH) -> dict:
    data = np.load(path)
    results = {k: data[k] for k in data.files}
    print(f"Calibration loaded from {path.resolve()}")
    return results


# ---------------------------------------------------------------------------
# Verification / testing
# ---------------------------------------------------------------------------

def print_matrix(label, M):
    print(f"\n{label}")
    print(np.array2string(M, formatter={"float_kind": lambda x: f"{x:10.5f}"}))


def verify_transforms(results: dict):
    """
    Run a suite of sanity checks on the calibration results.
    """
    T1  = results["T_cam1_robot"]
    T2  = results["T_cam2_robot"]
    T12 = results["T_cam1_cam2"]

    print("\n" + "="*60)
    print("CALIBRATION VERIFICATION REPORT")
    print("="*60)

    # ── 1. Print all matrices ────────────────────────────────────────────────
    print_matrix("T_cam1_robot  (robot → cam1)", T1)
    print_matrix("T_cam2_robot  (robot → cam2)", T2)
    print_matrix("T_cam1_cam2   (cam2  → cam1)", T12)

    # ── 2. Rotation matrices should be proper (det ≈ 1, R @ Rᵀ ≈ I) ─────────
    print("\n── Rotation matrix checks ──")
    for label, T in [("cam1_robot", T1), ("cam2_robot", T2), ("cam1_cam2", T12)]:
        R = T[:3, :3]
        det = np.linalg.det(R)
        orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
        status = "OK" if abs(det - 1) < 1e-4 and orth_err < 1e-4 else "FAIL"
        print(f"  {label:15s}  det(R)={det:.6f}  orthogonality_err={orth_err:.2e}  [{status}]")

    # ── 3. Consistency: T12 should equal T1 @ inv(T2) ───────────────────────
    T12_check = T1 @ np.linalg.inv(T2)
    consistency_err = np.max(np.abs(T12 - T12_check))
    status = "OK" if consistency_err < 1e-8 else "FAIL"
    print(f"\n── T_cam1_cam2 consistency: max_err={consistency_err:.2e}  [{status}]")

    # ── 4. Round-trip: cam1 → robot → cam1 should be identity ───────────────
    print("\n── Round-trip checks ──")
    rt1 = T1 @ np.linalg.inv(T1)
    rt2 = T2 @ np.linalg.inv(T2)
    rt12 = T12 @ np.linalg.inv(T12)
    for label, rt in [("cam1_robot", rt1), ("cam2_robot", rt2), ("cam1_cam2", rt12)]:
        err = np.max(np.abs(rt - np.eye(4)))
        status = "OK" if err < 1e-10 else "FAIL"
        print(f"  T_{label} @ inv(T_{label}) ≈ I  max_err={err:.2e}  [{status}]")

    # ── 5. Translational plausibility ───────────────────────────────────────
    print("\n── Translation plausibility ──")
    t1 = np.linalg.norm(T1[:3, 3])
    t2 = np.linalg.norm(T2[:3, 3])
    t12 = np.linalg.norm(T12[:3, 3])
    print(f"  |t_cam1_robot| = {t1:.3f} m  (camera distance from robot origin)")
    print(f"  |t_cam2_robot| = {t2:.3f} m")
    print(f"  |t_cam1_cam2|  = {t12:.3f} m  (inter-camera baseline)")

    # ── 6. Synthetic point round-trip ────────────────────────────────────────
    print("\n── Synthetic point round-trip ──")
    # A point in the robot frame (e.g. one of the tag centers)
    test_points_robot = np.array([
        [0.38,  0.4,  0.0, 1.0],
        [0.38, -0.4,  0.0, 1.0],
        [0.19,  0.0,  0.05, 1.0],  # midpoint, slightly above table
    ])

    print(f"  {'Robot point (m)':40s}  {'Δ cam1 via direct vs via cam2 (mm)':>36s}")
    for p_rob in test_points_robot:
        p_cam1_direct = T1 @ p_rob          # robot → cam1 directly
        p_cam2        = T2 @ p_rob           # robot → cam2
        p_cam1_via2   = T12 @ p_cam2         # cam2  → cam1

        diff_mm = np.linalg.norm(p_cam1_direct[:3] - p_cam1_via2[:3]) * 1000
        status = "OK" if diff_mm < 0.1 else "WARN"
        coord = f"({p_rob[0]:.2f}, {p_rob[1]:.2f}, {p_rob[2]:.2f})"
        print(f"  {coord:40s}  {diff_mm:8.4f} mm  [{status}]")

    print("\n" + "="*60)
    print("Verification complete.")
    print("="*60)


def interactive_point_test(results: dict):
    """
    Let you type a 3-D point in any frame and see where it lands in the others.
    Useful for manual sanity checking.
    """
    T1  = results["T_cam1_robot"]
    T2  = results["T_cam2_robot"]
    T12 = results["T_cam1_cam2"]

    T1_inv  = np.linalg.inv(T1)   # cam1  → robot
    T2_inv  = np.linalg.inv(T2)   # cam2  → robot
    T12_inv = np.linalg.inv(T12)  # cam1  → cam2

    print("\n=== Interactive Point Transformer ===")
    print("Enter a point and its frame, then see it in all other frames.")
    print("Type 'q' to quit.\n")

    frames = {"robot": 0, "cam1": 1, "cam2": 2}

    while True:
        raw = input("Frame [robot/cam1/cam2] and X Y Z (m) > ").strip()
        if raw.lower() == 'q':
            break
        parts = raw.split()
        if len(parts) != 4 or parts[0] not in frames:
            print("  Usage:  cam1 0.38 0.4 0.0")
            continue

        frame = parts[0]
        try:
            xyz = np.array([float(p) for p in parts[1:]] + [1.0])
        except ValueError:
            print("  Invalid numbers.")
            continue

        if frame == "robot":
            p_robot = xyz
            p_cam1  = T1  @ p_robot
            p_cam2  = T2  @ p_robot
        elif frame == "cam1":
            p_cam1  = xyz
            p_robot = T1_inv  @ p_cam1
            p_cam2  = T12_inv @ p_cam1
        else:  # cam2
            p_cam2  = xyz
            p_robot = T2_inv @ p_cam2
            p_cam1  = T12   @ p_cam2

        def fmt(v):
            return f"({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}) m"

        print(f"  → robot : {fmt(p_robot)}")
        print(f"  → cam1  : {fmt(p_cam1)}")
        print(f"  → cam2  : {fmt(p_cam2)}\n")


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_calibration(cam1: ZedCamera, cam2: ZedCamera, results: dict):
    """
    Draw projected robot-frame axes on each camera's live image so you can
    visually confirm the calibration looks right.
    """
    from utils.vis_utils import draw_pose_axes  # re-use the checkpoint0 helper

    T1 = results["T_cam1_robot"]
    T2 = results["T_cam2_robot"]

    img1 = cam1.image.copy()
    img2 = cam2.image.copy()

    draw_pose_axes(img1, cam1.camera_intrinsic, T1, size=TAG_SIZE)
    draw_pose_axes(img2, cam2.camera_intrinsic, T2, size=TAG_SIZE)

    cv2.namedWindow("CAM1 — robot origin", cv2.WINDOW_NORMAL)
    cv2.namedWindow("CAM2 — robot origin", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CAM1 — robot origin", 960, 540)
    cv2.resizeWindow("CAM2 — robot origin", 960, 540)
    cv2.imshow("CAM1 — robot origin", img1)
    cv2.imshow("CAM2 — robot origin", img2)
    print("\nVisualization open. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dual-camera calibration via AprilTags")
    p.add_argument("--test",  action="store_true",
                   help="Load saved calibration and run verification only (no cameras needed)")
    p.add_argument("--both",  action="store_true",
                   help="Calibrate, save, then immediately run verification")
    p.add_argument("--vis",   action="store_true",
                   help="Show live axis visualisation after calibration")
    p.add_argument("--interactive", action="store_true",
                   help="Launch interactive point transformer after verification")
    p.add_argument("--id1", type=int, default=0,
                   help="USB device index of camera 1 (default: 0)")
    p.add_argument("--id2", type=int, default=1,
                   help="USB device index of camera 2 (default: 1)")
    p.add_argument("--save", type=str, default=str(SAVE_PATH),
                   help=f"Path to save/load calibration  (default: {SAVE_PATH})")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = Path(args.save)

    # ── Test-only mode: no cameras required ─────────────────────────────────
    if args.test:
        results = load_calibration(save_path)
        verify_transforms(results)
        if args.interactive:
            interactive_point_test(results)
        return

    # ── Calibration mode ─────────────────────────────────────────────────────
    print("Initializing cameras...")
    cam1 = ZedCamera(camera_id=args.id1)
    cam2 = ZedCamera(camera_id=args.id2)

    try:
        results = calibrate_two_cameras(cam1, cam2)
        if results is None:
            print("\nCalibration failed. Check that the arena poster is visible to both cameras.")
            sys.exit(1)

        save_calibration(results, save_path)

        if args.vis:
            visualize_calibration(cam1, cam2, results)

        if args.both or args.interactive:
            verify_transforms(results)

        if args.interactive:
            interactive_point_test(results)

    finally:
        cam1.close()
        cam2.close()


if __name__ == "__main__":
    main()

// python dual_camera_calibration.py --both
