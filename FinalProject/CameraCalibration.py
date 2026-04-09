"""
dual_camera_calibration.py

Calibrates two ZED cameras against the robot base frame using AprilTags,
derives the inter-camera transform, and saves results in JSON format.

Usage
-----
    python dual_camera_calibration.py           # calibrate + save
    python dual_camera_calibration.py --load    # load saved + interactive test
    python dual_camera_calibration.py --id1 1 --id2 0  # swap USB order
"""

import argparse
import sys
import json
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera
from utils.vis_utils import draw_pose_axes

# ---------------------------------------------------------------------------
# AprilTag config
# ---------------------------------------------------------------------------

TAG_SIZE = 0.08  # metres

TAG_CENTER_COORDINATES = [
    [0.38,  0.4],   # tag 0
    [0.38, -0.4],   # tag 1
    [0.0,   0.4],   # tag 2
    [0.0,  -0.4],   # tag 3
]


def get_pnp_pairs(tags):
    world_points = np.empty([0, 3])
    image_points = np.empty([0, 2])

    for tag in tags:
        if tag.tag_id > 3:
            continue
        cx, cy = TAG_CENTER_COORDINATES[tag.tag_id]
        hs = TAG_SIZE / 2
        corners_world = [
            [cx - hs, cy + hs, 0],
            [cx - hs, cy - hs, 0],
            [cx + hs, cy - hs, 0],
            [cx + hs, cy + hs, 0],
        ]
        for wp, ip in zip(corners_world, tag.corners):
            world_points = np.vstack([world_points, wp])
            image_points = np.vstack([image_points, ip])

    return world_points, image_points


def get_transform_camera_robot(observation, camera_intrinsic, label="camera"):
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
# Calibration
# ---------------------------------------------------------------------------

def calibrate_two_cameras(cam1: ZedCamera, cam2: ZedCamera):
    print("\n=== Capturing frames ===")
    img1 = cam1.image
    img2 = cam2.image

    print("\n=== Calibrating CAM1 -> Robot ===")
    T1 = get_transform_camera_robot(img1, cam1.camera_intrinsic, label="cam1")

    print("\n=== Calibrating CAM2 -> Robot ===")
    T2 = get_transform_camera_robot(img2, cam2.camera_intrinsic, label="cam2")

    if T1 is None or T2 is None:
        return None

    return {
        "T_cam1_robot": T1,
        "T_cam2_robot": T2,
        "T_cam1_cam2":  T1 @ np.linalg.inv(T2),
        "intrinsics_cam1": cam1.camera_intrinsic,
        "intrinsics_cam2": cam2.camera_intrinsic,
    }


# ---------------------------------------------------------------------------
# Save / load in JSON format for D^3 Fields
# ---------------------------------------------------------------------------

SAVE_PATH = Path("calibration_results.json")


def save_calibration(results: dict, path: Path = SAVE_PATH):
    """
    Save calibration results in JSON format compatible with D^3 Fields.
    Format:
    {
        "cam1": {
            "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], 
                          [r31, r32, r33, tz], [0, 0, 0, 1]]
        },
        "cam2": { ... }
    }
    """
    # Convert numpy arrays to lists for JSON serialization
    calibration_data = {
        "cam1": {
            "intrinsics": results["intrinsics_cam1"].tolist(),
            "extrinsics": results["T_cam1_robot"].tolist(),  # World to camera
        },
        "cam2": {
            "intrinsics": results["intrinsics_cam2"].tolist(),
            "extrinsics": results["T_cam2_robot"].tolist(),  # World to camera
        },
        # Additional metadata for reference (optional)
        "metadata": {
            "T_cam1_cam2": results["T_cam1_cam2"].tolist(),
            "tag_size_m": TAG_SIZE,
            "tag_positions": TAG_CENTER_COORDINATES
        }
    }
    
    with open(path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"Calibration saved -> {path.resolve()}")
    print(f"  - Camera 1 intrinsics: {results['intrinsics_cam1'].shape}")
    print(f"  - Camera 1 extrinsics: 4x4 matrix (world → camera)")
    print(f"  - Camera 2 intrinsics: {results['intrinsics_cam2'].shape}")
    print(f"  - Camera 2 extrinsics: 4x4 matrix (world → camera)")


def load_calibration(path: Path = SAVE_PATH) -> dict:
    """
    Load calibration from JSON file.
    Returns dictionary with numpy arrays restored.
    """
    with open(path, 'r') as f:
        calibration_data = json.load(f)
    
    # Convert lists back to numpy arrays
    results = {
        "intrinsics_cam1": np.array(calibration_data["cam1"]["intrinsics"]),
        "extrinsics_cam1": np.array(calibration_data["cam1"]["extrinsics"]),
        "intrinsics_cam2": np.array(calibration_data["cam2"]["intrinsics"]),
        "extrinsics_cam2": np.array(calibration_data["cam2"]["extrinsics"]),
        "T_cam1_cam2": np.array(calibration_data["metadata"]["T_cam1_cam2"]),
    }
    
    print(f"Calibration loaded from {path.resolve()}")
    print(f"  - Camera 1: intrinsics {results['intrinsics_cam1'].shape}, extrinsics {results['extrinsics_cam1'].shape}")
    print(f"  - Camera 2: intrinsics {results['intrinsics_cam2'].shape}, extrinsics {results['extrinsics_cam2'].shape}")
    
    return results


# ---------------------------------------------------------------------------
# Depth map saving helper
# ---------------------------------------------------------------------------

def save_depth_maps(cam1: ZedCamera, cam2: ZedCamera, output_dir: Path = Path(".")):
    """
    Save depth maps from both cameras as .npy files for D^3 Fields.
    """
    print("\n=== Saving depth maps ===")
    
    # Get depth maps from ZED cameras
    depth1 = cam1.depth  # Assuming ZedCamera has depth property
    depth2 = cam2.depth
    
    # Save as numpy arrays
    np.save(output_dir / "cam1_depth.npy", depth1)
    np.save(output_dir / "cam2_depth.npy", depth2)
    
    # Also save RGB images for reference
    cv2.imwrite(str(output_dir / "cam1.png"), cam1.image)
    cv2.imwrite(str(output_dir / "cam2.png"), cam2.image)
    
    print(f"  - cam1_depth.npy: {depth1.shape}, range [{depth1.min():.3f}, {depth1.max():.3f}]")
    print(f"  - cam2_depth.npy: {depth2.shape}, range [{depth2.min():.3f}, {depth2.max():.3f}]")
    print(f"  - cam1.png saved")
    print(f"  - cam2.png saved")


# ---------------------------------------------------------------------------
# Interactive point tester (optional)
# ---------------------------------------------------------------------------

def interactive_point_test(cam1: ZedCamera, cam2: ZedCamera, results: dict):
    T1 = results["extrinsics_cam1"] if "extrinsics_cam1" in results else results.get("T_cam1_robot")
    T2 = results["extrinsics_cam2"] if "extrinsics_cam2" in results else results.get("T_cam2_robot")
    
    if T1 is None or T2 is None:
        print("No extrinsics found in results")
        return

    print("\n=== Interactive Point Tester ===")
    print("Enter a point in the ROBOT frame and it will be projected onto both camera feeds.")
    print("Type 'q' to quit.\n")

    cv2.namedWindow("CAMERAS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CAMERAS", 1280, 540)

    while True:
        raw = input("Robot X Y Z (m) > ").strip()
        if raw.lower() == 'q':
            break
        parts = raw.split()
        if len(parts) != 3:
            print("  Usage:  0.38 0.4 0.0")
            continue
        try:
            xyz = np.array([float(p) for p in parts] + [1.0])
        except ValueError:
            print("  Invalid numbers.")
            continue

        img1 = cam1.image
        img2 = cam2.image

        ann1, px1 = project_point_onto_image(img1, xyz, T1, cam1.camera_intrinsic)
        ann2, px2 = project_point_onto_image(img2, xyz, T2, cam2.camera_intrinsic)

        if px1:
            print(f"  CAM1 pixel: {px1}")
        else:
            print("  CAM1: point is behind camera")

        if px2:
            print(f"  CAM2 pixel: {px2}")
        else:
            print("  CAM2: point is behind camera")

        # Resize to same height
        h = 540
        w1 = int(ann1.shape[1] * h / ann1.shape[0])
        w2 = int(ann2.shape[1] * h / ann2.shape[0])

        ann1_resized = cv2.resize(ann1, (w1, h))
        ann2_resized = cv2.resize(ann2, (w2, h))

        combined = np.hstack((ann1_resized, ann2_resized))

        cv2.imshow("CAMERAS", combined)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def project_point_onto_image(image, point_robot, T_cam_robot, camera_intrinsic):
    """
    Draw a crosshair on a copy of image at the pixel corresponding to
    point_robot (homogeneous 4-vec in robot frame).
    Returns the annotated image and the projected (u, v) pixel.
    """
    p_cam = T_cam_robot @ point_robot          # robot -> camera
    x, y, z = p_cam[:3]

    if z <= 0:
        return image.copy(), None              # point is behind camera

    K = camera_intrinsic
    u = int(K[0, 0] * x / z + K[0, 2])
    v = int(K[1, 1] * y / z + K[1, 2])

    out = image.copy()
    r = 20
    cv2.line(out, (u - r, v), (u + r, v), (0, 255, 0), 2)
    cv2.line(out, (u, v - r), (u, v + r), (0, 255, 0), 2)
    cv2.circle(out, (u, v), r, (0, 255, 0), 2)
    cv2.putText(out, f"({x:.3f}, {y:.3f}, {z:.3f})",
                (u + 25, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out, (u, v)


# ---------------------------------------------------------------------------
# Updated load_camera_calibration for D^3 Fields
# ---------------------------------------------------------------------------

def create_camera_parameters_from_calibration(calib_path: Path = SAVE_PATH):
    """
    Helper function to load calibration and create CameraParameters objects
    for use with D^3 Fields code.
    """
    results = load_calibration(calib_path)
    
    # Create camera parameter objects matching your D^3 Fields structure
    class CameraParameters:
        def __init__(self, name, image_path, depth_path, intrinsics, extrinsics):
            self.name = name
            self.image_path = image_path      # RGB image
            self.depth_path = depth_path      # Depth map
            self.intrinsics = intrinsics      # 3x3 camera matrix
            self.extrinsics = extrinsics      # 4x4 world to camera transform
            self.width = None
            self.height = None
    
    cam1 = CameraParameters(
        name="cam1",
        image_path="cam1.png",
        depth_path="cam1_depth.npy",
        intrinsics=results["intrinsics_cam1"],
        extrinsics=results["extrinsics_cam1"]
    )
    
    cam2 = CameraParameters(
        name="cam2",
        image_path="cam2.png",
        depth_path="cam2_depth.npy",
        intrinsics=results["intrinsics_cam2"],
        extrinsics=results["extrinsics_cam2"]
    )
    
    return [cam1, cam2]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dual ZED camera calibration via AprilTags")
    p.add_argument("--load", action="store_true",
                   help="Skip calibration, load saved results and go straight to interactive test")
    p.add_argument("--id1",  type=int, default=0, help="USB index for cam1 (default: 0)")
    p.add_argument("--id2",  type=int, default=1, help="USB index for cam2 (default: 1)")
    p.add_argument("--save", type=str, default=str(SAVE_PATH),
                   help=f"Path to save/load calibration (default: {SAVE_PATH})")
    p.add_argument("--save-depth", action="store_true",
                   help="Save depth maps and RGB images for D^3 Fields")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = Path(args.save)

    print(f"Initializing cam1 (id={args.id1}) and cam2 (id={args.id2})...")
    cam1 = ZedCamera(camera_id=args.id1)
    cam2 = ZedCamera(camera_id=args.id2)

    try:
        if args.load:
            results = load_calibration(save_path)
        else:
            results = calibrate_two_cameras(cam1, cam2)
            if results is None:
                print("\nCalibration failed. Make sure the arena poster is visible to both cameras.")
                sys.exit(1)
            save_calibration(results, save_path)
        
        # Save depth maps if requested
        if args.save_depth:
            save_depth_maps(cam1, cam2, save_path.parent)
        
        interactive_point_test(cam1, cam2, results)

    finally:
        cam1.close()
        cam2.close()


if __name__ == "__main__":
    main()
