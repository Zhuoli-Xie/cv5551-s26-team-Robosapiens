import cv2
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry, SamPredictor
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os


# ============================================================================
# CONFIG
# ============================================================================

PATCH_SIZE   = 14    # DINOv2 ViT-S/14
IMG_SIZE     = 1036   # must be divisible by 14; smaller = faster on CPU
                     # 364 -> 26x26 patch grid, 518 -> 37x37 (slower)
TOP_K_PATCHES = 300  # how many reference patches to use for matching


# ============================================================================
# APRILTAG ROI
# ============================================================================

def detect_apriltag_roi_opencv(image, tag_family="TAG36H11", padding_percent=5):
    """
    Detect AprilTags and create a perspective ROI using the 4 tags as corners.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    family_map = {
        "TAG36H11": cv2.aruco.DICT_APRILTAG_36h11
    }

    dict_type   = family_map.get(tag_family, cv2.aruco.DICT_APRILTAG_36h11)
    aruco_dict  = cv2.aruco.getPredefinedDictionary(dict_type)
    parameters  = cv2.aruco.DetectorParameters()
    detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(corners) < 4:
        print(f"  Warning: Found {len(corners) if corners else 0} AprilTags (need 4)")
        return None, None, None, None, None

    print(f"  Found {len(corners)} AprilTags with IDs: {ids.flatten()}")

    tag_centers      = []
    tag_corners_list = []

    for i, corner in enumerate(corners):
        tag_corners = corner[0]
        tag_corners_list.append(tag_corners)
        center = np.mean(tag_corners, axis=0)
        tag_centers.append(center)
        print(f"    Tag {ids[i][0]}: center at ({center[0]:.1f}, {center[1]:.1f})")

    tag_centers = np.array(tag_centers)
    centroid    = np.mean(tag_centers, axis=0)
    angles      = np.arctan2(tag_centers[:, 1] - centroid[1],
                             tag_centers[:, 0] - centroid[0])
    sorted_indices       = np.argsort(angles)
    ordered_corners_list = [tag_corners_list[i] for i in sorted_indices]

    roi_corners = []
    for tag_corners in ordered_corners_list:
        tag_center  = np.mean(tag_corners, axis=0)
        distances_to_center = [np.linalg.norm(c - centroid) for c in tag_corners]
        inward_corner_idx   = np.argmin(distances_to_center)
        roi_corners.append(tag_corners[inward_corner_idx])

    roi_corners = np.array(roi_corners, dtype=np.int32)

    roi_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_corners], 255)

    print(f"  Perspective ROI corners: {roi_corners.tolist()}")

    detections = list(zip(corners, ids.flatten()))
    return roi_mask, roi_corners, detections, tag_centers, ids.flatten()


# ============================================================================
# FEATURE EXTRACTION  (native patch resolution — no upsampling)
# ============================================================================

def get_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)),
    ])


def extract_features(model, image_rgb: np.ndarray, device) -> np.ndarray:
    """
    Returns patch-level features at NATIVE patch resolution — no upsampling.
    Output shape: (n_patches_h, n_patches_w, feat_dim)
    e.g. with IMG_SIZE=364 and ViT-S/14: (26, 26, 384)
    """
    pil       = Image.fromarray(image_rgb.astype("uint8"))
    tensor    = get_transform()(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out          = model.forward_features(tensor)
        patch_tokens = out["x_norm_patchtokens"]   # (1, n_patches, d)

    n_patches = patch_tokens.shape[1]
    grid_size = int(n_patches ** 0.5)              # 26

    features = patch_tokens[0].reshape(grid_size, grid_size, -1)  # (h, w, d)

    # L2-normalise so cosine sim == dot product later
    norms    = features.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    features = features / norms

    return features.cpu().numpy()


# ============================================================================
# REFERENCE PATCH SELECTION
# ============================================================================

def select_reference_patches(ref_features: np.ndarray, k: int = TOP_K_PATCHES):
    """
    Pick the k most spatially-distinctive patches from the reference image.

    For each patch we compute its mean cosine similarity to every other patch.
    Patches that are LEAST similar to the rest are the most object-specific
    and worth matching — this avoids wasting budget on generic background.
    """
    h, w, d = ref_features.shape
    flat    = ref_features.reshape(-1, d)   # (n, d) — already L2-normed
    n       = flat.shape[0]

    # chunked to stay memory-friendly on CPU
    chunk    = 64
    mean_sim = np.zeros(n, dtype=np.float32)
    for i in range(0, n, chunk):
        sim = flat[i:i+chunk] @ flat.T      # (chunk, n)
        mean_sim[i:i+chunk] = sim.mean(axis=1)

    selected     = np.argsort(mean_sim)[:k]   # lowest mean sim = most distinctive
    patch_ys     = selected // w
    patch_xs     = selected % w
    patch_feats  = flat[selected]             # (k, d)

    return patch_ys, patch_xs, patch_feats


# ============================================================================
# DENSE CORRESPONDENCE
# ============================================================================

def compute_similarity_map(workspace_features: np.ndarray,
                            ref_patch_feats: np.ndarray) -> np.ndarray:
    """
    For every workspace patch find its best-matching reference patch score.

    workspace_features : (h, w, d) — L2-normed
    ref_patch_feats    : (k, d)    — L2-normed

    Returns raw similarity_map: (h, w) — NOT normalized yet.
    Normalization is deferred to find_object_bbox so it can be
    done within the ROI only.
    """
    h, w, d = workspace_features.shape
    flat    = workspace_features.reshape(-1, d)   # (n, d)

    # dot product == cosine sim because both sides are L2-normed
    sim  = flat @ ref_patch_feats.T               # (n, k)
    best = sim.max(axis=1)                        # (n,)

    return best.reshape(h, w)


def find_object_bbox(similarity_map: np.ndarray,
                     workspace_image: np.ndarray,
                     roi_mask: np.ndarray | None = None,
                     percentile: float = 90.0):
    """
    Threshold the similarity map and return the bounding box of the best match.

    Returns:
        bbox_img       — (x, y, w, h) in image pixel coords, or None
        component_mask — binary mask at patch resolution
    """
    h_p, w_p      = similarity_map.shape
    h_img, w_img  = workspace_image.shape[:2]

    sim = similarity_map.copy()

    # apply ROI after similarity computation (not before — avoids boundary corruption)
    if roi_mask is not None:
        roi_p           = cv2.resize(roi_mask, (w_p, h_p), interpolation=cv2.INTER_NEAREST)
        sim[roi_p == 0] = np.nan

    # normalize ONLY within the ROI so outside scores don't skew the range
    valid_mask = ~np.isnan(sim)
    valid_vals = sim[valid_mask]
    if len(valid_vals) == 0:
        return None, None

    lo, hi = valid_vals.min(), valid_vals.max()
    if hi > lo:
        sim[valid_mask] = (valid_vals - lo) / (hi - lo)
    sim[~valid_mask] = 0   # set outside ROI back to 0 after normalizing

    thresh = np.percentile(sim[valid_mask], percentile)
    binary = (sim >= thresh).astype(np.uint8)

    # morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n_labels < 2:
        return None, binary

    areas      = stats[1:, cv2.CC_STAT_AREA]
    best_label = np.argmax(areas) + 1
    component_mask = (labels == best_label).astype(np.uint8)

    x_p     = stats[best_label, cv2.CC_STAT_LEFT]
    y_p     = stats[best_label, cv2.CC_STAT_TOP]
    w_p_box = stats[best_label, cv2.CC_STAT_WIDTH]
    h_p_box = stats[best_label, cv2.CC_STAT_HEIGHT]

    # scale bbox from patch space -> image space with small padding
    scale_x = w_img / w_p
    scale_y = h_img / h_p
    pad     = 2

    x_img     = max(0, int((x_p - pad) * scale_x))
    y_img     = max(0, int((y_p - pad) * scale_y))
    w_img_box = min(w_img - x_img, int((w_p_box + 2*pad) * scale_x))
    h_img_box = min(h_img - y_img, int((h_p_box + 2*pad) * scale_y))

    return (x_img, y_img, w_img_box, h_img_box), component_mask


def find_object_correspondence(reference_image: np.ndarray,
                                workspace_image: np.ndarray,
                                model,
                                device,
                                roi_mask: np.ndarray | None = None,
                                visualize: bool = False):
    """
    Full 2-D correspondence pipeline:
      1. Extract DINOv2 patch features at native resolution
      2. Select most distinctive reference patches
      3. Dense similarity map over workspace
      4. Threshold + bounding box

    Returns:
        bbox_img       — (x, y, w, h) in image pixel coords, or None
        similarity_map — (h_patches, w_patches) float array
    """
    print("  Extracting reference features...")
    ref_features = extract_features(model, reference_image, device)

    print("  Extracting workspace features...")
    ws_features  = extract_features(model, workspace_image, device)

    print(f"  Patch grid: {ref_features.shape[:2]} ref | {ws_features.shape[:2]} workspace")

    print(f"  Selecting top {TOP_K_PATCHES} distinctive reference patches...")
    ref_patch_ys, ref_patch_xs, ref_patch_feats = select_reference_patches(ref_features)

    print("  Computing similarity map...")
    similarity_map = compute_similarity_map(ws_features, ref_patch_feats)

    print(f"  Raw cosine similarity range: [{similarity_map.min():.3f}, {similarity_map.max():.3f}]")
    print(f"  (normalization happens within ROI only, during thresholding)")

    bbox_img, binary_mask = find_object_bbox(
        similarity_map, workspace_image, roi_mask, percentile=90.0
    )

    if visualize:
        visualize_correspondence(
            reference_image, workspace_image,
            ref_features, ref_patch_ys, ref_patch_xs,
            similarity_map, bbox_img, roi_mask
        )

    return bbox_img, similarity_map


# ============================================================================
# SAM SEGMENTATION
# ============================================================================

def segment_with_sam_in_roi(workspace_image, bbox_img, predictor,
                             roi_mask=None):
    """
    Use SAM to segment the object from the bounding box, constrained to ROI.
    bbox_img is already in image pixel coords: (x, y, w, h)
    """
    if bbox_img is None:
        return None

    x, y, w, h = bbox_img

    predictor.set_image(workspace_image)
    masks, scores, _ = predictor.predict(
        box=np.array([x, y, x + w, y + h]),
        multimask_output=False
    )

    sam_mask = masks[0]

    if roi_mask is not None:
        sam_mask = np.logical_and(sam_mask, roi_mask > 0)

    return sam_mask


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models(device):
    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)

    print("Loading DINOv2 ViT-S/14...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2.eval().to(device)

    print("Models loaded!")
    return predictor, dinov2


# ============================================================================
# VISUALISATION
# ============================================================================

def visualize_correspondence(reference_image, workspace_image,
                              ref_features, ref_patch_ys, ref_patch_xs,
                              similarity_map, bbox_img, roi_mask=None):

    h_r, w_r  = reference_image.shape[:2]
    scale_rx  = w_r / ref_features.shape[1]
    scale_ry  = h_r / ref_features.shape[0]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Reference with selected patches
    ref_vis = reference_image.copy()
    for py, px in zip(ref_patch_ys, ref_patch_xs):
        cx = int((px + 0.5) * scale_rx)
        cy = int((py + 0.5) * scale_ry)
        cv2.circle(ref_vis, (cx, cy), 4, (0, 255, 0), -1)
    axes[0].imshow(ref_vis)
    axes[0].set_title(f"Reference — {len(ref_patch_ys)} selected patches")
    axes[0].axis("off")

    # 2. Raw similarity heatmap
    axes[1].imshow(similarity_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Similarity map  max={similarity_map.max():.3f}")
    axes[1].axis("off")
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)

    # 3. Workspace with ROI + bbox
    ws_vis = workspace_image.copy()
    if roi_mask is not None:
        overlay = ws_vis.copy()
        overlay[roi_mask == 0] = (overlay[roi_mask == 0] * 0.4).astype(np.uint8)
        ws_vis = overlay
    if bbox_img is not None:
        x, y, w, h = bbox_img
        cv2.rectangle(ws_vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
    axes[2].imshow(ws_vis)
    axes[2].set_title("Detected bounding box")
    axes[2].axis("off")

    # 4. Heatmap overlaid on workspace (ROI only)
    sim_up  = cv2.resize(similarity_map,
                         (workspace_image.shape[1], workspace_image.shape[0]))
    heatmap = (plt.cm.hot(sim_up)[:, :, :3] * 255).astype(np.uint8)
    blended = workspace_image.copy()
    if roi_mask is not None:
        roi_bool = roi_mask > 0
        blended[roi_bool] = cv2.addWeighted(
            workspace_image, 0.5, heatmap, 0.5, 0
        )[roi_bool]
    else:
        blended = cv2.addWeighted(workspace_image, 0.5, heatmap, 0.5, 0)
    axes[3].imshow(blended)
    axes[3].set_title("Heatmap overlay (ROI only)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_apriltag_roi(image, roi_mask, roi_corners, detections, centers, ids):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    img_with_tags = image.copy()
    if detections:
        for corners, tag_id in detections:
            cv2.polylines(img_with_tags, [corners.astype(np.int32)], True, (0, 255, 0), 2)
            center = np.mean(corners[0], axis=0)
            cv2.circle(img_with_tags, tuple(center.astype(int)), 5, (0, 0, 255), -1)
            cv2.putText(img_with_tags, str(tag_id),
                        (int(center[0]) + 10, int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    axes[0].imshow(img_with_tags)
    axes[0].set_title("Detected AprilTags")
    axes[0].axis("off")

    img_with_roi = image.copy()
    if roi_corners is not None:
        cv2.polylines(img_with_roi, [roi_corners], True, (0, 255, 0), 3)
        overlay = img_with_roi.copy()
        cv2.fillPoly(overlay, [roi_corners], (0, 255, 0))
        img_with_roi = cv2.addWeighted(img_with_roi, 0.7, overlay, 0.3, 0)
        for corner in roi_corners:
            cv2.circle(img_with_roi, tuple(corner), 8, (255, 0, 0), -1)
    axes[1].imshow(img_with_roi)
    axes[1].set_title("Perspective ROI")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_results(reference_image, workspace_image, sam_mask, bbox_img,
                      similarity_map, roi_mask, roi_corners):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(reference_image)
    axes[0, 0].set_title("Reference image")
    axes[0, 0].axis("off")

    ws_with_roi = workspace_image.copy()
    if roi_corners is not None:
        cv2.polylines(ws_with_roi, [roi_corners], True, (0, 255, 0), 3)
    axes[0, 1].imshow(ws_with_roi)
    axes[0, 1].set_title("Workspace with ROI")
    axes[0, 1].axis("off")

    sim_up  = cv2.resize(similarity_map,
                         (workspace_image.shape[1], workspace_image.shape[0]))
    heatmap = (plt.cm.hot(sim_up)[:, :, :3] * 255).astype(np.uint8)
    blended = workspace_image.copy()
    if roi_mask is not None:
        roi_bool = roi_mask > 0
        blended[roi_bool] = cv2.addWeighted(
            workspace_image, 0.5, heatmap, 0.5, 0
        )[roi_bool]
    else:
        blended = cv2.addWeighted(workspace_image, 0.5, heatmap, 0.5, 0)
    axes[0, 2].imshow(blended)
    axes[0, 2].set_title("Similarity heatmap overlay (ROI only)")
    axes[0, 2].axis("off")

    img_with_bbox = workspace_image.copy()
    if bbox_img:
        x, y, w, h = bbox_img
        cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 3)
    if roi_corners is not None:
        cv2.polylines(img_with_bbox, [roi_corners], True, (255, 0, 0), 2)
    axes[1, 0].imshow(img_with_bbox)
    axes[1, 0].set_title("Detected object bbox")
    axes[1, 0].axis("off")

    img_with_mask = workspace_image.copy()
    if sam_mask is not None:
        overlay = img_with_mask.copy()
        overlay[sam_mask] = [0, 255, 0]
        img_with_mask = cv2.addWeighted(img_with_mask, 0.7, overlay, 0.3, 0)
    if roi_corners is not None:
        cv2.polylines(img_with_mask, [roi_corners], True, (255, 0, 0), 2)
    axes[1, 1].imshow(img_with_mask)
    axes[1, 1].set_title("SAM mask (ROI constrained)")
    axes[1, 1].axis("off")

    roi_only = workspace_image.copy()
    if roi_mask is not None:
        roi_only[roi_mask == 0] = 0
    axes[1, 2].imshow(roi_only)
    axes[1, 2].set_title("ROI only")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("D³ FIELDS - ROI-CONSTRAINED OBJECT DETECTION")
    print("=" * 70)

    reference_path = "data/reference_object.jpg"
    workspace_path = "data/cam2.png"

    if not (os.path.exists(reference_path) and os.path.exists(workspace_path)):
        print(f"\nImage files not found!")
        print(f"  Reference : {reference_path}")
        print(f"  Workspace : {workspace_path}")
        exit(1)

    reference_image = cv2.cvtColor(cv2.imread(reference_path), cv2.COLOR_BGR2RGB)
    workspace_image = cv2.cvtColor(cv2.imread(workspace_path),  cv2.COLOR_BGR2RGB)

    # ── Step 1: AprilTag ROI ──────────────────────────────────────────────
    print("\n[1] Detecting AprilTags and creating perspective ROI...")
    print("-" * 50)
    roi_mask, roi_corners, detections, centers, ids = \
        detect_apriltag_roi_opencv(workspace_image)

    if roi_mask is None:
        print("Could not detect 4 AprilTags. Exiting.")
        exit(1)

    print("Created perspective ROI")
    visualize_apriltag_roi(workspace_image, roi_mask, roi_corners,
                            detections, centers, ids)

    # ── Step 2: Load models ───────────────────────────────────────────────
    print("\n[2] Loading models...")
    print("-" * 50)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    predictor, dinov2 = load_models(device)

    # ── Step 3: Dense correspondence ──────────────────────────────────────
    print("\n[3] Finding object via dense correspondence (ROI constrained)...")
    print("-" * 50)
    bbox_img, sim_map = find_object_correspondence(
        reference_image, workspace_image,
        dinov2, device,
        roi_mask=roi_mask,
        visualize=True        # set False to skip debug plots
    )

    if bbox_img is None:
        print("No object found in ROI — check similarity map visualisation")
        exit(1)

    print(f"Found object bbox: {bbox_img}")

    # ── Step 4: SAM segmentation ──────────────────────────────────────────
    print("\n[4] Segmenting with SAM (ROI constrained)...")
    print("-" * 50)
    sam_mask = segment_with_sam_in_roi(
        workspace_image, bbox_img, predictor, roi_mask
    )

    if sam_mask is None or np.sum(sam_mask) == 0:
        print("SAM segmentation failed")
        exit(1)

    print(f"Segmented object — mask area: {np.sum(sam_mask)} pixels")

    # ── Step 5: Final visualisation ───────────────────────────────────────
    print("\n[5] Visualising results...")
    print("-" * 50)
    visualize_results(reference_image, workspace_image, sam_mask, bbox_img,
                      sim_map, roi_mask, roi_corners)
