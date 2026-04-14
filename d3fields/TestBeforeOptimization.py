"""
TestBeforeOptimization.py

Visualize pipeline outputs before running grasp optimization:
  1. Mask & DINO feature images from GroundingDINO + SAM + DINOv2
  2. Merged 3D point cloud from multi-view depth
  3. Gripper poses overlaid on the point cloud (interactive Plotly) 
  Dimensions: 35mm half-width, 46mm finger length, 67mm TCP offset 

Usage
-----
    # Run detection + show all visualizations
    python TestBeforeOptimization.py -d data/mugs --text "mug"

    # Skip detection, just visualize existing outputs + gripper poses
    python TestBeforeOptimization.py -d data/mugs --skip-detect --pose-idx 0 1 2
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA


# ── Lite6 Gripper Lite geometry (metres) ────────────────────────────────────
# Six key points in the gripper-local frame (Z = approach direction):
#   - center:            TCP origin
#   - left/right tip:    fingertip positions
#   - left/right base:   finger root (where finger meets palm)
#   - end effector:      wrist / mounting point
FINGER_HALF_WIDTH = 0.020   # half of 40 mm max stroke
FINGER_LENGTH     = 0.020   # finger protrusion length
TCP_OFFSET        = 0.047   # TCP offset from wrist

GRIPPER_POINTS_LOCAL = np.array([
    [0,  0,                0],   # 0: center (midpoint at finger base level)
    [0,  FINGER_HALF_WIDTH, FINGER_LENGTH],     # 1: left fingertip
    [0, -FINGER_HALF_WIDTH, FINGER_LENGTH],     # 2: right fingertip
    [0,  FINGER_HALF_WIDTH, 0],  # 3: left finger base
    [0, -FINGER_HALF_WIDTH, 0],  # 4: right finger base
    [0,  0,                -TCP_OFFSET],               # 5: end effector (T's origin)
])

# Edges connecting the six points to form the gripper shape
GRIPPER_EDGES = [
    (1, 3), (2, 4),   # left finger, right finger
    (3, 4),            # palm bar
    (3, 5), (4, 5),   # palm to wrist
    (0, 5),            # center to wrist (stem)
]

# ── Tool-frame → gripper-frame correction ──────────────────────────────────
# The saved `gripper_pose.npy` is the xArm TCP pose (flange +Z = approach,
# offset 47 mm forward via set_tcp_offset). `GRIPPER_POINTS_LOCAL` assumes
# +Z = approach and ±Y = finger-opening. If the Lite6 Gripper Lite is bolted
# onto the flange rotated about Z, the opening axis of the physical gripper
# does not line up with local-Y and the visualised gripper looks rotated
# about the approach axis.
#
# Edit `TOOL_TO_GRIPPER` below to match the actual mounting. It is
# right-multiplied onto the pose:  world_pts = pose @ TOOL_TO_GRIPPER @ local_pts
#
# Common candidates — try them in order until the fingers line up:
#   np.eye(4)                              # no correction
#   Rz(+90 deg)  -> swap  X↔Y (fingers along flange-X)
#   Rz(-90 deg)  -> swap  Y↔X
#   Rz(+180 deg) -> flip  L/R (gripper mounted rotated 180°)
def _Rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
    return T

def _Ry(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return T

def _Rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return T

# Composed tool→gripper correction:
#   _Rx(90) first rotates local +Z (approach) onto tool +Y,
#   _Ry(90) then rotates about the (new) green axis to line up the finger
#   opening direction with the physical gripper.
# Flip either sign (_Rx(-90), _Ry(-90)) if the result points the wrong way.
# TOOL_TO_GRIPPER = _Rx(90) @ _Rz(-45)
TOOL_TO_GRIPPER = np.eye(4)
_ = _Rz  # keep helper available for quick swaps via `_Rz(180)` etc.


# ── Data loading ────────────────────────────────────────────────────────────

def count_cameras(data_dir):
    i = 0
    while (Path(data_dir) / f"camera_{i}").is_dir():
        i += 1
    return i


def load_depth(path):
    """Load depth from .npz (metres) or 16-bit .png (millimetres)."""
    path = str(path)
    if os.path.exists(path.replace('.png', '.npz')):
        return np.load(path.replace('.png', '.npz'))['depth']
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img.astype(np.float32) / 1000.0


def load_scene(data_dir, timestep=0):
    """Load color images, depth maps, intrinsics, and extrinsics."""
    data_dir = Path(data_dir)
    num_cam = count_cameras(data_dir)
    colors, depths, intrinsics, extrinsics = [], [], [], []

    for i in range(num_cam):
        cam = data_dir / f"camera_{i}"
        colors.append(cv2.imread(str(cam / "color" / f"{timestep}.png")))
        depths.append(load_depth(str(cam / "depth" / f"{timestep}.png")))

        params = np.load(str(cam / "camera_params.npy"))
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(K)
        extrinsics.append(np.load(str(cam / "camera_extrinsics.npy")))

    return colors, depths, intrinsics, extrinsics, num_cam


# ── Step 1: Detection + Segmentation + DINO features ───────────────────────

def load_models(device="cuda",
                dino_ckpt="./ckpts/groundingdino_swint_ogc.pth",
                sam_ckpt="./ckpts/sam_vit_b_01ec64.pth"):
    import groundingdino
    from groundingdino.util.inference import Model as GDModel
    from segment_anything import sam_model_registry, SamPredictor

    cfg = os.path.join(groundingdino.__path__[0],
                       'config/GroundingDINO_SwinT_OGC.py')
    gd = GDModel(cfg, dino_ckpt, device=device)

    sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt).to(device)
    sam_pred = SamPredictor(sam)

    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    dinov2.eval().to(device)

    return gd, sam_pred, dinov2


def detect_and_segment(image_bgr, gd_model, sam_pred, text, box_thresh=0.4):
    """Run GroundingDINO + SAM → instance mask (H, W) uint8."""
    dets, phrases = gd_model.predict_with_caption(
        image=image_bgr, caption=text,
        box_threshold=box_thresh, text_threshold=0.3)

    H, W = image_bgr.shape[:2]
    if len(dets) == 0:
        return np.zeros((H, W), np.uint8), []

    sam_pred.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    mask = np.zeros((H, W), np.uint8)
    for i, box in enumerate(dets.xyxy):
        masks_i, scores_i, _ = sam_pred.predict(box=box, multimask_output=True)
        mask[masks_i[np.argmax(scores_i)]] = i + 1

    return mask, phrases


def extract_dino_features(image_bgr, dinov2, device="cuda"):
    """Extract DINOv2 patch features → (patch_h, patch_w, 1024)."""
    H, W = image_bgr.shape[:2]
    ph, pw = H // 14, W // 14
    transform = T.Compose([
        T.Resize((ph * 14, pw * 14)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(img)[:3].unsqueeze(0).to(device)

    with torch.no_grad():
        feats = dinov2.forward_features(tensor)['x_norm_patchtokens']
    return feats[0].cpu().numpy().reshape(ph, pw, -1)


def run_detection_pipeline(data_dir, colors, text, device, box_thresh=0.4):
    """Run GroundingDINO + SAM + DINOv2 on all cameras. Save & return results."""
    data_dir = Path(data_dir)
    print("Loading models...")
    gd, sam_pred, dinov2 = load_models(device=device)

    masks, features_list = [], []
    for i, img in enumerate(colors):
        print(f"\ncamera_{i}: detecting '{text}'...")
        mask, phrases = detect_and_segment(img, gd, sam_pred, text, box_thresh)
        print(f"  {len(phrases)} detection(s): {phrases}")

        # Save mask
        mask_dir = data_dir / f"camera_{i}" / "mask"
        mask_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(mask_dir / "0.png"), mask)
        masks.append(mask)

        # Extract & save DINO features
        print(f"  Extracting DINOv2 features...")
        feats = extract_dino_features(img, dinov2, device)
        feat_dir = data_dir / f"camera_{i}" / "dino_feat"
        feat_dir.mkdir(exist_ok=True)
        np.save(str(feat_dir / "0.npy"), feats)
        features_list.append(feats)

    return masks, features_list


def load_existing_masks_and_features(data_dir, num_cam, timestep=0):
    """Load pre-computed masks and DINO features."""
    data_dir = Path(data_dir)
    masks, features = [], []
    for i in range(num_cam):
        cam = data_dir / f"camera_{i}"
        m = cv2.imread(str(cam / "mask" / f"{timestep}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"No mask at {cam / 'mask' / f'{timestep}.png'}")
        masks.append(m)

        feat_path = cam / "dino_feat" / f"{timestep}.npy"
        if feat_path.exists():
            features.append(np.load(str(feat_path)))
        else:
            features.append(None)
    return masks, features


# ── Visualization 1: Mask & DINO feature images ────────────────────────────

def visualize_masks_and_features(colors, masks, features_list):
    """Show mask overlays and DINO PCA images in a matplotlib grid."""
    n = len(colors)
    has_feats = any(f is not None for f in features_list)
    rows = 2 if has_feats else 1

    fig, axes = plt.subplots(rows, n, figsize=(5 * n, 5 * rows), squeeze=False)
    fig.suptitle("Mask & DINO Feature Visualization", fontsize=14)

    # Row 1: mask overlays
    for i in range(n):
        rgb = cv2.cvtColor(colors[i], cv2.COLOR_BGR2RGB)
        overlay = rgb.copy()
        mask = masks[i]
        for inst in range(1, mask.max() + 1):
            color = np.array([(inst * 80) % 256, (inst * 120 + 60) % 256,
                              (inst * 200 + 120) % 256])
            overlay[mask == inst] = (0.6 * overlay[mask == inst] +
                                     0.4 * color).astype(np.uint8)
        axes[0, i].imshow(overlay)
        axes[0, i].set_title(f"camera_{i} mask")
        axes[0, i].axis("off")

    # Row 2: DINO PCA (fitted jointly on foreground patches)
    if has_feats:
        pca_images = _dino_pca(features_list, masks)
        for i in range(n):
            axes[1, i].imshow(pca_images[i])
            axes[1, i].set_title(f"camera_{i} DINO PCA")
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


def _dino_pca(features_list, masks):
    """PCA-colorize DINO features across views, masked to foreground."""
    all_feats, fg_flags, shapes = [], [], []
    for feat, mask in zip(features_list, masks):
        if feat is None:
            continue
        ph, pw = feat.shape[:2]
        H, W = mask.shape
        mask_r = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)
        fg = mask_r.ravel() > 0
        fg_flags.append(fg)
        shapes.append((ph, pw, H, W))
        all_feats.append(feat.reshape(-1, feat.shape[-1]))

    cat = np.concatenate(all_feats)
    fg_all = np.concatenate(fg_flags)
    if fg_all.sum() < 4:
        return [np.zeros((s[2], s[3], 3), np.uint8) for s in shapes]

    pca = PCA(n_components=3)
    fg_pca = pca.fit_transform(cat[fg_all])
    for c in range(3):
        lo, hi = fg_pca[:, c].min(), fg_pca[:, c].max()
        if hi > lo:
            fg_pca[:, c] = (fg_pca[:, c] - lo) / (hi - lo)

    full = np.zeros((cat.shape[0], 3), np.float32)
    full[fg_all] = fg_pca

    images, offset = [], 0
    for ph, pw, H, W in shapes:
        patch = full[offset:offset + ph * pw].reshape(ph, pw, 3)
        img = cv2.resize(patch, (W, H), interpolation=cv2.INTER_LINEAR)
        images.append((img * 255).clip(0, 255).astype(np.uint8))
        offset += ph * pw
    return images


# ── Visualization 2: Merged 3D point cloud ─────────────────────────────────

def build_point_cloud(colors, depths, intrinsics, extrinsics, masks=None):
    """Back-project masked depth from all views into a merged point cloud."""
    all_pts, all_rgb = [], []

    for i in range(len(colors)):
        depth = depths[i]
        rgb = cv2.cvtColor(colors[i], cv2.COLOR_BGR2RGB) / 255.0
        fx, fy, cx, cy = (intrinsics[i][0, 0], intrinsics[i][1, 1],
                          intrinsics[i][0, 2], intrinsics[i][1, 2])
        T_cam = extrinsics[i]
        T_inv = np.linalg.inv(T_cam)

        fg = depth > 0
        if masks is not None:
            fg &= masks[i] > 0

        H, W = depth.shape
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        u, v, d = uu[fg], vv[fg], depth[fg]

        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)
        pts_world = (T_inv @ pts_cam.T).T[:, :3]

        all_pts.append(pts_world)
        all_rgb.append(rgb[fg])

    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    _, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(idx)

    return np.asarray(pcd.points), np.asarray(pcd.colors)


def visualize_point_cloud(pts, rgb):
    """Show the point cloud in an Open3D viewer."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd], window_name="Merged Point Cloud",
                                      width=1024, height=768)


# ── Visualization 3: Gripper + point cloud in Plotly ────────────────────────

def transform_gripper(pose):
    """Transform local gripper points to world frame using pose (4x4).

    Applies TOOL_TO_GRIPPER correction so local +Z/±Y align with the
    physical approach / opening axes of the mounted gripper.
    """
    ones = np.ones((GRIPPER_POINTS_LOCAL.shape[0], 1))
    local_h = np.hstack([GRIPPER_POINTS_LOCAL, ones])
    world = (pose @ TOOL_TO_GRIPPER @ local_h.T).T[:, :3]
    return world


def make_pose_axes_traces(pose, name="tool", length=0.05):
    """Draw raw pose axes (X=red, Y=green, Z=blue) to diagnose orientation."""
    origin = pose[:3, 3]
    R = pose[:3, :3]
    traces = []
    for vec, col, lbl in [(R[:, 0], "red",   "X"),
                          (R[:, 1], "green", "Y"),
                          (R[:, 2], "blue",  "Z")]:
        tip = origin + length * vec
        traces.append(go.Scatter3d(
            x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
            mode="lines+text", line=dict(color=col, width=6),
            text=["", f"{name}:{lbl}"], textposition="top center",
            showlegend=False,
        ))
    return traces


def make_gripper_traces(pose, name="gripper", color="red"):
    """Create Plotly traces (points + lines) for one gripper pose."""
    pts = transform_gripper(pose)
    labels = ["center", "L-tip", "R-tip", "L-base", "R-base", "wrist"]

    # Scatter for the 6 key points
    scatter = go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers+text", text=labels, textposition="top center",
        marker=dict(size=5, color=color),
        name=name,
    )

    # Lines for the gripper edges
    lx, ly, lz = [], [], []
    for a, b in GRIPPER_EDGES:
        lx += [pts[a, 0], pts[b, 0], None]
        ly += [pts[a, 1], pts[b, 1], None]
        lz += [pts[a, 2], pts[b, 2], None]

    lines = go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode="lines", line=dict(color=color, width=4),
        name=f"{name} edges", showlegend=False,
    )
    return [scatter, lines]


def visualize_gripper_with_pcd(pts, rgb, gripper_poses, subsample=5000):
    """Interactive Plotly visualization of point cloud + gripper poses."""
    # Subsample point cloud for performance
    if len(pts) > subsample:
        idx = np.random.choice(len(pts), subsample, replace=False)
        pts_s, rgb_s = pts[idx], rgb[idx]
    else:
        pts_s, rgb_s = pts, rgb

    colors_str = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                  for r, g, b in rgb_s]

    traces = [go.Scatter3d(
        x=pts_s[:, 0], y=pts_s[:, 1], z=pts_s[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=colors_str, opacity=0.7),
        name="point cloud",
    )]

    palette = ["red", "blue", "green", "orange", "purple",
               "cyan", "magenta", "yellow"]
    for i, pose in enumerate(gripper_poses):
        c = palette[i % len(palette)]
        traces += make_gripper_traces(pose, name=f"gripper_{i}", color=c)
        traces += make_pose_axes_traces(pose, name=f"pose_{i}")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Point Cloud + Gripper Poses (Lite6 Gripper Lite)",
        scene=dict(aspectmode="data"),
        width=1200, height=800,
    )
    fig.show()


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize pipeline outputs before optimization")
    p.add_argument("-d", "--data-dir", required=True,
                   help="Scene directory (from CameraCalibration.py)")
    p.add_argument("--text", type=str, default=None,
                   help="Text prompt for detection (e.g. 'mug')")
    p.add_argument("--skip-detect", action="store_true",
                   help="Skip detection, use existing masks/features")
    p.add_argument("--pose-idx", type=int, nargs="*", default=[0],
                   help="Gripper pose indices to visualize (default: 0)")
    p.add_argument("--box-thresh", type=float, default=0.4)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load scene data
    print("Loading scene data...")
    colors, depths, intrinsics, extrinsics, num_cam = load_scene(data_dir)
    print(f"  {num_cam} camera(s)")

    # Step 1: Detection + segmentation (or load existing)
    if args.skip_detect:
        print("\nLoading existing masks & features...")
        masks, features = load_existing_masks_and_features(data_dir, num_cam)
    else:
        if args.text is None:
            raise ValueError("--text is required when not using --skip-detect")
        masks, features = run_detection_pipeline(
            data_dir, colors, args.text, device, args.box_thresh)

    # Vis 1: Masks & DINO features
    print("\n── Vis 1: Masks & DINO features ──")
    visualize_masks_and_features(colors, masks, features)

    # Vis 2: Merged point cloud
    print("\n── Vis 2: Merged point cloud ──")
    pts, rgb = build_point_cloud(colors, depths, intrinsics, extrinsics, masks)
    print(f"  {len(pts)} points after outlier removal")
    visualize_point_cloud(pts, rgb)

    # Vis 3: Gripper poses + point cloud
    print("\n── Vis 3: Gripper + point cloud (Plotly) ──")
    pose_dir = data_dir / "target_grasp_pose"
    gripper_poses = []
    if pose_dir.is_dir():
        for idx in args.pose_idx:
            pose_subdir = pose_dir / f"pose_{idx:04d}"
            npy = pose_subdir / "gripper_pose.npy"
            txt = pose_subdir / "gripper_pose.txt"
            if npy.exists():
                gripper_poses.append(np.load(str(npy)))
                print(f"  Loaded pose_{idx:04d}/gripper_pose.npy")
            elif txt.exists():
                gripper_poses.append(np.loadtxt(str(txt)))
                print(f"  Loaded pose_{idx:04d}/gripper_pose.txt")
            else:
                print(f"  Warning: no gripper pose found in {pose_subdir}")
    else:
        print(f"  No target_grasp_pose directory at {pose_dir}")

    if gripper_poses:
        visualize_gripper_with_pcd(pts, rgb, gripper_poses)
    else:
        print("  No gripper poses loaded, showing point cloud only...")
        visualize_gripper_with_pcd(pts, rgb, [])


if __name__ == "__main__":
    main()
