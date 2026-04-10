"""
CameraCalibration.py

Calibrates ZED cameras against the robot base frame using AprilTags,
then saves RGB, depth, intrinsics, and extrinsics in the directory
layout expected by the D^3 Fields pipeline (grasp_pose_search.py).

Output structure
----------------
    <output_dir>/
        camera_0/
            color/0.png              # BGR uint8
            depth/0.png              # uint16, millimetres
            camera_extrinsics.npy    # (4, 4) float64  T_cam_robot
            camera_params.npy        # [fx, fy, cx, cy]
        camera_1/
            ...

Usage
-----
    python CameraCalibration.py -o data/my_scene                 # auto-detect all cameras
    python CameraCalibration.py -o data/my_scene --ids 0 1       # specific cameras only
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from pupil_apriltags import Detector

from utils.zed_camera import ZedCamera


# ---------------------------------------------------------------------------
# AprilTag config
# ---------------------------------------------------------------------------

TAG_SIZE = 0.08  # metres

# (x, y) position of each tag centre in the robot base frame [metres]
TAG_CENTERS = [
    [0.38,  0.4],   # tag 0
    [0.38, -0.4],   # tag 1
    [0.0,   0.4],   # tag 2
    [0.0,  -0.4],   # tag 3
]


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _build_pnp_pairs(tags):
    """Return world and image point arrays from detected AprilTags."""
    world_pts, image_pts = [], []
    for tag in tags:
        if tag.tag_id >= len(TAG_CENTERS):
            continue
        tx, ty = TAG_CENTERS[tag.tag_id]
        hs = TAG_SIZE / 2
        corners_world = [
            [tx - hs, ty + hs, 0],
            [tx - hs, ty - hs, 0],
            [tx + hs, ty - hs, 0],
            [tx + hs, ty + hs, 0],
        ]
        for wp, ip in zip(corners_world, tag.corners):
            world_pts.append(wp)
            image_pts.append(ip)
    return np.array(world_pts, dtype=np.float64), np.array(image_pts, dtype=np.float64)


def calibrate(image, intrinsic, detector, label="cam"):
    """Compute the 4x4 camera-to-robot extrinsic from a single image.

    Returns the (4, 4) matrix, or None on failure.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tags = detector.detect(gray, estimate_tag_pose=False)
    print(f"[{label}] Tags detected: {len(tags)}")

    world_pts, image_pts = _build_pnp_pairs(tags)
    if len(world_pts) < 4:
        print(f"[{label}] Only {len(world_pts)} corners — need >= 4.")
        return None

    ok, rvec, tvec = cv2.solvePnP(world_pts, image_pts, intrinsic, None)
    if not ok:
        print(f"[{label}] solvePnP failed.")
        return None

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, intrinsic, None)
    errs = np.linalg.norm(image_pts - proj.reshape(-1, 2), axis=1)
    print(f"[{label}] Reprojection — mean: {errs.mean():.2f} px, max: {errs.max():.2f} px")
    return T


# ---------------------------------------------------------------------------
# Live preview
# ---------------------------------------------------------------------------

_TAG_COLOURS = {
    True:  (0,  220,  0),   # green  – tag(s) visible
    False: (0, 165, 255),   # orange – no tags yet
}
_PREVIEW_HEIGHT = 540       # each camera panel is resized to this height


def _draw_tags(frame, tags):
    """Draw tag corners and IDs onto *frame* in-place."""
    colour = _TAG_COLOURS[len(tags) >= 1]
    for tag in tags:
        pts = tag.corners.astype(int)
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, colour, 2)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(frame, f"id={tag.tag_id}", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
    return frame


def _make_panel(frame, label, n_tags, n_corners):
    """Resize frame to _PREVIEW_HEIGHT and add a status bar at the top."""
    h, w = frame.shape[:2]
    new_w = int(w * _PREVIEW_HEIGHT / h)
    panel = cv2.resize(frame, (new_w, _PREVIEW_HEIGHT))

    ready = n_corners >= 4
    bar_colour = (0, 180, 0) if ready else (0, 0, 200)
    cv2.rectangle(panel, (0, 0), (new_w, 30), bar_colour, -1)

    status = f"{label}  |  tags: {n_tags}  corners: {n_corners}"
    if ready:
        status += "  [READY]"
    cv2.putText(panel, status, (6, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def preview_cameras(cameras, detector):
    """Show a live tiled view of all cameras with AprilTag overlays.

    Controls
    --------
    Enter / Space  – accept current positions and proceed to calibration
    Q / Esc        – quit the program
    """
    print("\n[preview] Live view started.")
    print("[preview]  ENTER = proceed to calibration   |   Q = quit\n")

    window = "Camera Preview  (ENTER=calibrate  Q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        panels = []
        for i, cam in enumerate(cameras):
            frame = cam.image.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray, estimate_tag_pose=False)
            _, img_pts = _build_pnp_pairs(tags)
            _draw_tags(frame, tags)
            panels.append(_make_panel(frame, f"cam{i}", len(tags), len(img_pts)))

        cv2.imshow(window, np.hstack(panels))

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 32):    # Enter or Space
            print("[preview] Proceeding to calibration.")
            break
        elif key in (ord('q'), ord('Q'), 27):   # Q or Esc
            cv2.destroyAllWindows()
            print("[preview] Aborted by user.")
            sys.exit(0)

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Save in D^3 Fields format
# ---------------------------------------------------------------------------

def _depth_vis(depth_m):
    """Convert float32 depth (metres) to a colourmap PNG for inspection.

    Valid pixels are normalised to the scene's [min, max] range and mapped
    with COLORMAP_INFERNO.  Invalid (NaN/Inf) pixels are rendered black.
    """
    valid = np.isfinite(depth_m)
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if valid.any():
        lo, hi = depth_m[valid].min(), depth_m[valid].max()
        norm = np.zeros_like(depth_m)
        if hi > lo:
            norm[valid] = (depth_m[valid] - lo) / (hi - lo)
        grey = (norm * 255).clip(0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
        vis[~valid] = 0  # black out invalid pixels
    return vis


def save_camera(cam, extrinsic, cam_dir, timestep=0):
    """Save RGB, depth (npz + vis png), intrinsics, and extrinsic for one camera."""
    cam_dir = Path(cam_dir)
    (cam_dir / "color").mkdir(parents=True, exist_ok=True)
    (cam_dir / "depth").mkdir(parents=True, exist_ok=True)

    img = cam.image
    cv2.imwrite(str(cam_dir / "color" / f"{timestep}.png"), img)

    depth_m = cam.depth  # float32, metres
    np.savez_compressed(str(cam_dir / "depth" / f"{timestep}.npz"), depth=depth_m)
    cv2.imwrite(str(cam_dir / "depth" / f"{timestep}_vis.png"), _depth_vis(depth_m))

    np.save(str(cam_dir / "camera_extrinsics.npy"), extrinsic)

    K = cam.camera_intrinsic
    np.save(str(cam_dir / "camera_params.npy"),
            np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))

    valid = depth_m[np.isfinite(depth_m)]
    print(f"  {cam_dir.name}: color {img.shape[:2]}, "
          f"depth [{valid.min():.3f}, {valid.max():.3f}] m")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ZED camera calibration + data capture")
    p.add_argument("-o", "--output-dir", type=str, required=True,
                   help="Directory to save captured scene data")
    p.add_argument("--ids", type=int, nargs="+", default=None,
                   help="ZED camera USB indices (default: auto-detect all)")
    p.add_argument("-t", "--timestep", type=int, default=0,
                   help="Timestep label for saved files (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()

    camera_ids = args.ids
    if camera_ids is None:
        devices = sl.Camera.get_device_list()
        camera_ids = sorted(
            d.id for d in devices if d.camera_state == sl.CAMERA_STATE.AVAILABLE
        )
        if not camera_ids:
            print("No ZED cameras detected.")
            sys.exit(1)
        print(f"Auto-detected {len(camera_ids)} camera(s): {camera_ids}")
    else:
        print(f"Using specified camera(s): {camera_ids}")

    detector = Detector(families='tag36h11')
    output_dir = Path(args.output_dir)
    cameras = [ZedCamera(camera_id=cid) for cid in camera_ids]

    try:
        preview_cameras(cameras, detector)

        for i, cam in enumerate(cameras):
            label = f"cam{i}"
            print(f"\n=== Calibrating {label} ===")
            T = calibrate(cam.image, cam.camera_intrinsic, detector, label=label)
            if T is None:
                print(f"Calibration failed for {label}. Make sure AprilTags are visible.")
                sys.exit(1)
            save_camera(cam, T, output_dir / f"camera_{i}", timestep=args.timestep)
    finally:
        for cam in cameras:
            cam.close()

    print(f"\nScene saved -> {output_dir.resolve()}  ({len(cameras)} cameras, t={args.timestep})")


if __name__ == "__main__":
    main()
