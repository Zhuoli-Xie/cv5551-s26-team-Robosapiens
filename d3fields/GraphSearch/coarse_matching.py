"""Coarse matching: DINO feature cosine similarity -> rigid alignment (SVD Procrustes)."""

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


def coarse_match_init_pose(fusion_new, object_pcd_new, contact_pts_world,
                           f_star, gripper_pose, device='cuda'):
    """
    Compute an initial SE(3) pose via DINO feature cosine-similarity matching.

    Steps:
        1. Query DINO descriptors on the new scene's object point cloud.
        2. For each reference contact descriptor f_i*, find the point in the
           new scene with maximum cosine similarity -> correspondence point c_i.
        3. Solve the rigid alignment  contact_pts_world -> c  via SVD
           (Procrustes), giving T_coarse.
        4. Recover the gripper pose as  T_init = T_coarse * T_demo.

    Returns:
        init_pose : (4, 4) np array, coarse gripper pose in the new scene
        corr_pts  : (N, 3) np array, matched points in the new scene
        nn_dists  : (N,) np array, cosine distances (1 - cosine similarity) for matches
    """
    # 1. Query DINO features on new object surface
    pts_tensor = torch.from_numpy(object_pcd_new).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = fusion_new.batch_eval(pts_tensor, return_names=['dino_feats'])
    f_new_all = out['dino_feats']  # (N, D)

    # 2. Sanitize NaN/Inf then drop reference contacts whose descriptor is
    # effectively zero. Fusion.eval can emit zeros (no-camera branch) or NaN
    # (DINO patch token underflow); either produces bad cosine similarity.
    # Target descriptors f_new_all also need sanitizing, otherwise NaN rows
    # there turn every cos_sim column NaN.
    f_star_t = torch.from_numpy(f_star).to(device, dtype=torch.float32)
    f_star_t = torch.nan_to_num(f_star_t, nan=0.0, posinf=0.0, neginf=0.0)
    f_new_all = torch.nan_to_num(f_new_all, nan=0.0, posinf=0.0, neginf=0.0)
    feat_norm = torch.linalg.norm(f_star_t, dim=1)
    valid = (feat_norm > 1e-6).cpu().numpy()
    n_drop = int((~valid).sum())
    if n_drop > 0:
        print(f"  Coarse match: dropping {n_drop}/{f_star_t.shape[0]} contacts "
              f"with zero-feature descriptors (outside camera coverage).")
    if valid.sum() < 3:
        raise RuntimeError(
            f"Only {int(valid.sum())} valid contact descriptors "
            "— need >= 3 for SVD Procrustes alignment.")

    # 3. Cosine similarity matching on the valid subset
    f_star_v = F.normalize(f_star_t[torch.from_numpy(valid).to(device)], dim=1)
    f_new_n = F.normalize(f_new_all, dim=1)
    cos_sim_v = f_star_v @ f_new_n.T  # (M_valid, N)
    nn_idx_v = cos_sim_v.argmax(dim=1).cpu().numpy()
    corr_pts_v = object_pcd_new[nn_idx_v]
    best_sim_v = cos_sim_v.max(dim=1).values

    # 4. SVD rigid alignment on valid pairs only
    src = contact_pts_world[valid]
    dst = corr_pts_v
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T
    t_align = dst_mean - R_align @ src_mean

    T_coarse = np.eye(4)
    T_coarse[:3, :3] = R_align
    T_coarse[:3, 3] = t_align

    # 5. Recover gripper pose
    init_pose = T_coarse @ gripper_pose

    print(f"  Coarse match: median cosine similarity = "
          f"{best_sim_v.median().item():.4f} "
          f"(over {int(valid.sum())} valid contacts)")
    print(f"  Coarse alignment residual (mean): "
          f"{np.linalg.norm(dst - (R_align @ src.T).T - t_align, axis=1).mean():.4f} m")

    # 6. Re-expand to full length so downstream viz keeps aligned indices.
    # Invalid rows: corr point = self (no-op line) and nn_dist = 1 (max).
    M = contact_pts_world.shape[0]
    corr_pts = contact_pts_world.copy()
    corr_pts[valid] = corr_pts_v
    nn_dists = np.ones(M, dtype=np.float64)
    nn_dists[valid] = (1.0 - best_sim_v).cpu().numpy()
    return init_pose, corr_pts, nn_dists


# ---------------------------------------------------------------------------
# Side-by-side visualization of coarse matching
# ---------------------------------------------------------------------------

def visualize_coarse_match(object_pcd_ref, contact_pts_world, object_pcd_new,
                           corr_pts, nn_dists, T_coarse, init_pose,
                           gripper_pose, show_lines=True,
                           title="Coarse Match (ref | target)"):
    """
    Side-by-side visualization of the coarse matching result.

        Left  : reference scene + contact points + demo gripper frame
        Right : target scene + matched correspondence points + init_pose frame
        Lines : ref contact -> matched target point, colored by cosine similarity
                (green = high similarity, red = low)

    Parameters
    ----------
    object_pcd_ref   : (Nr, 3) reference scene object point cloud
    contact_pts_world: (M, 3)  reference contact points (source of matching)
    object_pcd_new   : (Nn, 3) target scene object point cloud
    corr_pts         : (M, 3)  target points matched to each reference contact
    nn_dists         : (M,)    cosine distance (1 - cos_sim) per match
    T_coarse         : (4, 4)  rigid alignment contact_pts_world -> corr_pts
    init_pose        : (4, 4)  coarse gripper pose in the target scene
    gripper_pose     : (4, 4)  demo gripper pose in the reference scene
    show_lines       : bool    draw correspondence lines across the two scenes
    """
    cos_sim = 1.0 - nn_dists  # per-match similarity, in [-1, 1]

    # Horizontal offset so the target scene sits to the right of the reference.
    left_xyz = np.concatenate([object_pcd_ref, contact_pts_world], axis=0)
    right_xyz = np.concatenate([object_pcd_new, corr_pts], axis=0)
    gap = 0.25 * max(
        left_xyz[:, 0].max() - left_xyz[:, 0].min(),
        right_xyz[:, 0].max() - right_xyz[:, 0].min(),
        0.1,
    )
    shift_x = (left_xyz[:, 0].max() - right_xyz[:, 0].min()) + gap
    shift = np.array([shift_x, 0.0, 0.0])

    geometries = []

    # ---- Left: reference scene ----
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(object_pcd_ref)
    ref_pcd.paint_uniform_color([0.65, 0.65, 0.72])
    geometries.append(ref_pcd)

    ref_contacts = o3d.geometry.PointCloud()
    ref_contacts.points = o3d.utility.Vector3dVector(contact_pts_world)
    ref_contacts.paint_uniform_color([0.2, 0.4, 1.0])
    geometries.append(ref_contacts)

    demo_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    demo_frame.transform(gripper_pose)
    geometries.append(demo_frame)

    # ---- Right: target scene (shifted along +X) ----
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(object_pcd_new + shift)
    tgt_pcd.paint_uniform_color([0.72, 0.68, 0.55])
    geometries.append(tgt_pcd)

    tgt_corr = o3d.geometry.PointCloud()
    tgt_corr.points = o3d.utility.Vector3dVector(corr_pts + shift)
    tgt_corr.paint_uniform_color([0.2, 1.0, 0.4])
    geometries.append(tgt_corr)

    init_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    T_right = init_pose.copy()
    T_right[:3, 3] = T_right[:3, 3] + shift
    init_frame.transform(T_right)
    geometries.append(init_frame)

    # ---- Correspondence lines, colored by cosine similarity ----
    if show_lines:
        n = contact_pts_world.shape[0]
        line_pts = np.concatenate(
            [contact_pts_world, corr_pts + shift], axis=0)
        line_idx = [[i, i + n] for i in range(n)]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_pts)
        line_set.lines = o3d.utility.Vector2iVector(line_idx)

        s = cos_sim.astype(np.float64)
        s_min, s_max = float(s.min()), float(s.max())
        s_norm = (s - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(s)
        colors = np.zeros((n, 3))
        colors[:, 1] = s_norm          # green  -> high similarity
        colors[:, 0] = 1.0 - s_norm    # red    -> low  similarity
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    # World origin for spatial reference.
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03))

    print(f"  [coarse-viz] cosine similarity: "
          f"min={cos_sim.min():.4f}  mean={cos_sim.mean():.4f}  "
          f"max={cos_sim.max():.4f}")
    print(f"  [coarse-viz] T_coarse =\n{T_coarse}")
    print(f"  [coarse-viz] init_pose =\n{init_pose}")

    o3d.visualization.draw_geometries(
        geometries, window_name=title, width=1400, height=800)


# ---------------------------------------------------------------------------
# Diagnostics: is the DINO feature field actually discriminative?
# ---------------------------------------------------------------------------

def _query_dino(fusion, pts_np, device='cuda'):
    pts = torch.from_numpy(pts_np).to(device, dtype=torch.float32)
    with torch.no_grad():
        out = fusion.batch_eval(pts, return_names=['dino_feats'])
    return out['dino_feats']  # (N, D) torch tensor


def _clean_feats(feats_np):
    """Drop non-finite rows and all-zero rows (invalid points). Returns (X, mask)."""
    X = np.asarray(feats_np, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    nonzero = np.linalg.norm(np.where(finite[:, None], X, 0.0), axis=1) > 1e-8
    keep = finite & nonzero
    return X[keep], keep


def _fit_pca(feats_np, n_components=3, max_samples=20000):
    """Fit PCA on rows of feats_np (N, D). Returns (basis (D, k), mean (D,))."""
    X, _ = _clean_feats(feats_np)
    if X.shape[0] < n_components + 1:
        raise ValueError(
            f"_fit_pca: only {X.shape[0]} finite non-zero rows (need >= {n_components + 1}). "
            "Most feature rows were zero/NaN — check that points project into the cameras.")
    mean = X.mean(axis=0)
    centered = X - mean
    if centered.shape[0] > max_samples:
        idx = np.random.choice(centered.shape[0], max_samples, replace=False)
        centered = centered[idx]
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        # Fall back to covariance eigendecomposition (more tolerant of ill-conditioning).
        cov = centered.T @ centered
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        Vt = V[:, order].T
    return Vt[:n_components].T.astype(np.float64), mean.astype(np.float64)


def _apply_pca(feats_np, basis, mean):
    X = np.asarray(feats_np, dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    proj = (X - mean) @ basis
    lo, hi = proj.min(axis=0), proj.max(axis=0)
    return (proj - lo) / np.maximum(hi - lo, 1e-8)


def _side_by_side_shift(left_xyz, right_xyz, gap_ratio=0.25):
    gap = gap_ratio * max(
        left_xyz[:, 0].max() - left_xyz[:, 0].min(),
        right_xyz[:, 0].max() - right_xyz[:, 0].min(),
        0.1,
    )
    shift_x = (left_xyz[:, 0].max() - right_xyz[:, 0].min()) + gap
    return np.array([shift_x, 0.0, 0.0])


def visualize_feature_pca(fusion_ref, object_pcd_ref,
                          fusion_new, object_pcd_new,
                          contact_pts=None, device='cuda',
                          title="DINO feature PCA (ref | target)"):
    """
    Project the DINO features on both scenes onto 3 shared principal
    components and render the point clouds side-by-side with those colors.

    If semantically-similar parts of the object (neck, body, bottom) appear
    as the SAME color, the feature field itself lacks spatial discrimination
    and no matching strategy downstream can fix it.
    """
    f_ref = _query_dino(fusion_ref, object_pcd_ref, device).cpu().numpy()
    f_new = _query_dino(fusion_new, object_pcd_new, device).cpu().numpy()

    # Shared basis -> colors are comparable across both scenes
    basis, mean = _fit_pca(np.concatenate([f_ref, f_new], axis=0))
    colors_ref = _apply_pca(f_ref, basis, mean)
    colors_new = _apply_pca(f_new, basis, mean)

    shift = _side_by_side_shift(object_pcd_ref, object_pcd_new)

    geos = []
    pcd_l = o3d.geometry.PointCloud()
    pcd_l.points = o3d.utility.Vector3dVector(object_pcd_ref)
    pcd_l.colors = o3d.utility.Vector3dVector(colors_ref)
    geos.append(pcd_l)

    pcd_r = o3d.geometry.PointCloud()
    pcd_r.points = o3d.utility.Vector3dVector(object_pcd_new + shift)
    pcd_r.colors = o3d.utility.Vector3dVector(colors_new)
    geos.append(pcd_r)

    if contact_pts is not None:
        spheres = []
        for p in contact_pts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            s.translate(p)
            s.paint_uniform_color([1.0, 1.0, 1.0])
            spheres.append(s)
        # merge for efficiency
        merged = spheres[0]
        for s in spheres[1:]:
            merged += s
        geos.append(merged)

    geos.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03))
    print(f"  [pca-viz] D={f_ref.shape[1]}  "
          f"explained-var basis shape={basis.shape}")
    o3d.visualization.draw_geometries(
        geos, window_name=title, width=1400, height=800)


def visualize_similarity_heatmap(fusion_new, object_pcd_new,
                                 query_feat, query_xyz,
                                 object_pcd_ref=None, device='cuda',
                                 title="Cosine-similarity heatmap"):
    """
    Color the target pcd by cosine similarity to a single reference
    descriptor `query_feat`. The query's xyz on the reference pcd (if
    provided) is drawn as a red sphere, the argmax on the target as green.

    Use this to check: does the true match light up, or is the similarity
    diffuse / peaked elsewhere?
    """
    f_new = _query_dino(fusion_new, object_pcd_new, device)
    q = torch.from_numpy(np.asarray(query_feat)).to(
        device, dtype=torch.float32).view(1, -1)
    q_n = F.normalize(q, dim=1)
    f_n = F.normalize(f_new, dim=1)
    sim = (q_n @ f_n.T).squeeze(0).cpu().numpy()
    sim = np.where(np.isfinite(sim), sim, 0.0)

    s_min, s_max = float(sim.min()), float(sim.max())
    s_norm = (sim - s_min) / max(s_max - s_min, 1e-8)
    # simple perceptual ramp: dark blue -> cyan -> yellow
    colors = np.zeros((sim.shape[0], 3))
    colors[:, 0] = np.clip(2.0 * s_norm - 1.0, 0.0, 1.0)
    colors[:, 1] = np.clip(2.0 * s_norm,       0.0, 1.0) * 0.9
    colors[:, 2] = np.clip(1.0 - 2.0 * s_norm, 0.0, 1.0) * 0.8

    geos = []
    if object_pcd_ref is not None:
        shift = _side_by_side_shift(object_pcd_ref, object_pcd_new)
        ref = o3d.geometry.PointCloud()
        ref.points = o3d.utility.Vector3dVector(object_pcd_ref)
        ref.paint_uniform_color([0.62, 0.62, 0.70])
        geos.append(ref)

        q_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        q_sphere.translate(np.asarray(query_xyz))
        q_sphere.paint_uniform_color([1.0, 0.15, 0.15])
        geos.append(q_sphere)
    else:
        shift = np.zeros(3)

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(object_pcd_new + shift)
    tgt.colors = o3d.utility.Vector3dVector(colors)
    geos.append(tgt)

    best_idx = int(sim.argmax())
    m_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    m_sphere.translate(object_pcd_new[best_idx] + shift)
    m_sphere.paint_uniform_color([0.15, 1.0, 0.25])
    geos.append(m_sphere)

    geos.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03))
    print(f"  [heatmap] cos sim: min={s_min:.4f} mean={sim.mean():.4f} "
          f"max={s_max:.4f}  argmax at idx {best_idx} "
          f"xyz={object_pcd_new[best_idx]}")
    o3d.visualization.draw_geometries(
        geos, window_name=title, width=1400, height=800)


def visualize_similarity_heatmaps_for_contacts(fusion_new, object_pcd_new,
                                               f_star, contact_pts,
                                               object_pcd_ref=None,
                                               indices=None, device='cuda'):
    """Run the single-query heatmap for a handful of reference contacts.
    Default indices span the contact set: first, middle, last."""
    M = f_star.shape[0]
    if indices is None:
        indices = sorted(set([0, M // 4, M // 2, 3 * M // 4, M - 1]))
    for k, i in enumerate(indices):
        visualize_similarity_heatmap(
            fusion_new, object_pcd_new,
            query_feat=f_star[i], query_xyz=contact_pts[i],
            object_pcd_ref=object_pcd_ref, device=device,
            title=f"Heatmap: contact {i}  ({k + 1}/{len(indices)})  "
                  f"[close window for next]")


# ---------------------------------------------------------------------------
# Step-through diagnostic: open each pcd in its own window, in order.
# ---------------------------------------------------------------------------

def debug_pipeline_pcds(object_pcd_ref, object_pcd_new, contact_pts, f_star,
                        gripper_pose):
    """Open four Open3D windows in sequence to sanity-check inputs before
    coarse matching. Close each window to advance to the next.

        1. Reference pcd + contact points + demo gripper frame
           (are contacts on the surface?)
        2. Target pcd alone + world axes
           (is it in the same frame/scale?)
        3. Reference (gray) + target (orange) overlaid in the same frame
           (do they co-register where expected?)
        4. Contact points colored by ||f_star[i]||
           (zero-norm contacts -> red; these are the ones that produce NaNs)
    """
    f_arr = np.asarray(f_star)
    nan_row = ~np.isfinite(f_arr).all(axis=1)
    f_clean = np.where(np.isfinite(f_arr), f_arr, 0.0)
    f_norm = np.linalg.norm(f_clean, axis=1)
    bad = nan_row | (f_norm < 1e-6)
    print(f"  [debug] f_star: {f_arr.shape[0]} contacts, "
          f"{int(nan_row.sum())} NaN-rows, "
          f"{int(((~nan_row) & (f_norm < 1e-6)).sum())} zero-norm rows, "
          f"{int(bad.sum())} total bad")
    if (~nan_row).any():
        good_norms = f_norm[~nan_row]
        print(f"  [debug] ||f_star|| (finite rows): "
              f"min={good_norms.min():.4e} "
              f"mean={good_norms.mean():.4e} "
              f"max={good_norms.max():.4e}")

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)

    # Window 1: reference pcd + contacts + demo frame
    ref = o3d.geometry.PointCloud()
    ref.points = o3d.utility.Vector3dVector(object_pcd_ref)
    ref.paint_uniform_color([0.62, 0.62, 0.70])

    contact_spheres = []
    for p in contact_pts:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
        s.translate(p)
        s.paint_uniform_color([0.2, 0.4, 1.0])
        contact_spheres.append(s)
    contacts_geom = contact_spheres[0]
    for s in contact_spheres[1:]:
        contacts_geom += s

    demo_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    demo_frame.transform(gripper_pose)

    o3d.visualization.draw_geometries(
        [ref, contacts_geom, demo_frame, axes],
        window_name="[1/4] Reference pcd + contacts + demo gripper "
                    "(close window to continue)",
        width=1200, height=800)

    # Window 2: target pcd alone
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(object_pcd_new)
    tgt.paint_uniform_color([0.88, 0.65, 0.35])
    o3d.visualization.draw_geometries(
        [tgt, axes],
        window_name="[2/4] Target pcd alone  (frame / scale sanity check)",
        width=1200, height=800)

    # Window 3: ref + target overlaid (no shift)
    ref2 = o3d.geometry.PointCloud()
    ref2.points = o3d.utility.Vector3dVector(object_pcd_ref)
    ref2.paint_uniform_color([0.55, 0.55, 0.60])
    o3d.visualization.draw_geometries(
        [ref2, tgt, axes],
        window_name="[3/4] Ref (gray) + Target (orange) overlaid",
        width=1200, height=800)

    # Window 4: contacts colored by descriptor norm (red = zero / NaN)
    good_max = float(f_norm[~nan_row].max()) if (~nan_row).any() else 1.0
    norm_max = max(good_max, 1e-8)
    norm_spheres = []
    for p, n, is_nan in zip(contact_pts, f_norm, nan_row):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        s.translate(p)
        if is_nan or n < 1e-6:
            c = [1.0, 0.1, 0.1]  # red -> zero/NaN descriptor (NaN culprits)
        else:
            g = float(n) / norm_max
            c = [1.0 - g, g, 0.2]
        s.paint_uniform_color(c)
        norm_spheres.append(s)
    norm_geom = norm_spheres[0]
    for s in norm_spheres[1:]:
        norm_geom += s

    ref3 = o3d.geometry.PointCloud()
    ref3.points = o3d.utility.Vector3dVector(object_pcd_ref)
    ref3.paint_uniform_color([0.70, 0.70, 0.75])
    o3d.visualization.draw_geometries(
        [ref3, norm_geom, axes],
        window_name="[4/4] Contact descriptor norm: red = zero (NaN source)",
        width=1200, height=800)
