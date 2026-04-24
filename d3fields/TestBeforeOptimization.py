"""
TestBeforeOptimization.py

Visualize pipeline outputs before running grasp optimization:
  1. Mask & DINO feature images from GroundingDINO + SAM + DINOv2
  2. Merged 3D point cloud from multi-view depth
  3. EE frame + tool-tip (EE + GRIPPER_TCP_OFFSET along local +Z)
     overlaid on the point cloud (interactive Plotly)

Usage
-----
    # Run detection + show all visualizations
    python TestBeforeOptimization.py -d data/mugs --text "mug"

    # Skip detection, just visualize existing outputs + EE/tool-tip
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


# ── Visualization 3: EE frame + tool tip in Plotly ────────────────────────

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


# ── Gripper geometry, relative to the EE / flange frame ───────────────────
# The recorded ee_pose.npy is the flange pose with NO TCP offset applied,
# so all gripper points below live in the flange-local frame:
#   local +Z : approach direction (out of the flange toward fingertips)
#   local ±Y : finger opening direction (nominal mounting)
GRIPPER_TCP_OFFSET    = 0.047   # 47 mm: flange → TCP along local +Z
GRIPPER_FINGER_LENGTH = 0.023   # 23 mm: finger base → fingertip along local +Z
GRIPPER_FINGER_HALF_W = 0.020   # 20 mm: TCP → finger along local ±Y


def make_tooltip_traces(pose, name, color, tcp_offset=GRIPPER_TCP_OFFSET):
    """Draw the tool-tip marker and the EE→tool-tip connector.

    The tool tip sits at `tcp_offset` along the EE-local +Z axis.
    """
    flange = pose[:3, 3]
    tip = (pose @ np.array([0.0, 0.0, tcp_offset, 1.0]))[:3]

    line = go.Scatter3d(
        x=[flange[0], tip[0]], y=[flange[1], tip[1]], z=[flange[2], tip[2]],
        mode="lines",
        line=dict(color=color, width=6, dash="dot"),
        name=f"{name} offset",
        showlegend=False,
    )
    marker = go.Scatter3d(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        mode="markers+text",
        marker=dict(size=8, color=color, symbol="cross"),
        text=[f"{name} tip ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})"],
        textposition="top center",
        name=f"{name} tip",
    )
    return [line, marker]


def make_finger_traces(pose, name, color,
                       tcp_offset=GRIPPER_TCP_OFFSET,
                       finger_length=GRIPPER_FINGER_LENGTH,
                       finger_half_w=GRIPPER_FINGER_HALF_W):
    """Draw the two fingers plus markers at each finger base and fingertip.

    Each finger is a segment along local +Z, from `tcp_offset` (finger base)
    to `tcp_offset + finger_length` (fingertip), offset by ±finger_half_w on
    local Y.
    """
    z_base = tcp_offset
    z_tip  = tcp_offset + finger_length

    traces = []
    bases_xyz, tips_xyz, base_labels, tip_labels = [], [], [], []
    for sign, side in [(+1, "L"), (-1, "R")]:
        base = (pose @ np.array([0.0, sign * finger_half_w, z_base, 1.0]))[:3]
        tip  = (pose @ np.array([0.0, sign * finger_half_w, z_tip,  1.0]))[:3]
        traces.append(go.Scatter3d(
            x=[base[0], tip[0]], y=[base[1], tip[1]], z=[base[2], tip[2]],
            mode="lines",
            line=dict(color=color, width=5),
            name=f"{name} finger {side}",
            showlegend=False,
        ))
        bases_xyz.append(base)
        tips_xyz.append(tip)
        base_labels.append(f"{name} {side}-base")
        tip_labels.append(f"{name} {side}-tip")

    bases_xyz = np.asarray(bases_xyz)
    tips_xyz = np.asarray(tips_xyz)

    traces.append(go.Scatter3d(
        x=bases_xyz[:, 0], y=bases_xyz[:, 1], z=bases_xyz[:, 2],
        mode="markers+text",
        marker=dict(size=6, color=color, symbol="circle"),
        text=base_labels, textposition="bottom center",
        name=f"{name} finger bases", showlegend=False,
    ))
    traces.append(go.Scatter3d(
        x=tips_xyz[:, 0], y=tips_xyz[:, 1], z=tips_xyz[:, 2],
        mode="markers+text",
        marker=dict(size=7, color=color, symbol="diamond-open"),
        text=tip_labels, textposition="top center",
        name=f"{name} fingertips", showlegend=False,
    ))
    return traces


def make_palm_traces(pose, name, color,
                     tcp_offset=GRIPPER_TCP_OFFSET,
                     finger_half_w=GRIPPER_FINGER_HALF_W):
    """Step 3: close the gripper outline with the palm bar and slanted
    palm-to-flange edges.

    Adds three segments in flange-local coords:
      - palm bar:    [0, +HW, tcp] ↔ [0, -HW, tcp]
      - palm → flange (L/R): [0, ±HW, tcp] ↔ [0, 0, 0]
    The central stem [0,0,tcp] ↔ [0,0,0] is already drawn by Step 1.
    """
    flange    = (pose @ np.array([0.0, 0.0,                0.0,        1.0]))[:3]
    base_L    = (pose @ np.array([0.0,  finger_half_w,     tcp_offset, 1.0]))[:3]
    base_R    = (pose @ np.array([0.0, -finger_half_w,     tcp_offset, 1.0]))[:3]

    segments = [
        ("palm",       base_L, base_R),
        ("palm-flg L", base_L, flange),
        ("palm-flg R", base_R, flange),
    ]
    traces = []
    for label, a, b in segments:
        traces.append(go.Scatter3d(
            x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]],
            mode="lines",
            line=dict(color=color, width=4),
            name=f"{name} {label}",
            showlegend=False,
        ))
    return traces


def make_frame_marker_traces(pose, name, color, axis_length=0.05, marker_size=8):
    """Marker dot + XYZ axes + label for a single 4x4 frame."""
    origin = pose[:3, 3]
    point = go.Scatter3d(
        x=[origin[0]], y=[origin[1]], z=[origin[2]],
        mode="markers+text",
        marker=dict(size=marker_size, color=color, symbol="diamond"),
        text=[f"{name} EE ({origin[0]:.3f}, {origin[1]:.3f}, {origin[2]:.3f})"],
        textposition="bottom center",
        name=f"{name} EE",
    )
    return [point] + make_pose_axes_traces(pose, name=name, length=axis_length)


def visualize_gripper_with_pcd(pts, rgb, gripper_poses, subsample=5000):
    """Interactive Plotly visualization of point cloud + gripper poses.

    Always overlays the base_link origin and (for each loaded pose) the
    end-effector position so the EE pose can be checked against the
    point cloud regardless of whether the gripper geometry is shown.
    """
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

    # base_link origin (poses are saved in base_link frame)
    traces += make_frame_marker_traces(np.eye(4), name="base_link",
                                       color="black", axis_length=0.08,
                                       marker_size=10)

    palette = ["red", "blue", "green", "orange", "purple",
               "cyan", "magenta", "yellow"]
    for i, pose in enumerate(gripper_poses):
        c = palette[i % len(palette)]
        # EE (flange) marker + XYZ axes at EE
        traces += make_frame_marker_traces(pose, name=f"ee_{i}", color=c)
        # Tool-tip marker offset along local +Z
        traces += make_tooltip_traces(pose, name=f"ee_{i}", color=c)
        # Fingers + fingertip / finger-base markers
        traces += make_finger_traces(pose, name=f"ee_{i}", color=c)

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Point Cloud + EE frame & base_link",
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
            ee_npy = pose_subdir / "ee_pose.npy"
            if ee_npy.exists():
                pose = np.load(str(ee_npy))
                gripper_poses.append(pose)
                tip = (pose @ np.array([0.0, 0.0, GRIPPER_TCP_OFFSET, 1.0]))[:3]
                print(f"  Loaded pose_{idx:04d}/ee_pose.npy "
                      f"-> EE  @ ({pose[0,3]:.3f}, {pose[1,3]:.3f}, {pose[2,3]:.3f}) m"
                      f"\n                             "
                      f"-> tip @ ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m")
            else:
                print(f"  Warning: no EE pose found in {pose_subdir}")
    else:
        print(f"  No target_grasp_pose directory at {pose_dir}")

    if gripper_poses:
        visualize_gripper_with_pcd(pts, rgb, gripper_poses)
    else:
        print("  No gripper poses loaded, showing point cloud only...")
        visualize_gripper_with_pcd(pts, rgb, [])


if __name__ == "__main__":
    main()
