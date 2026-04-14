"""
Grasp Pose Search via D3Fields Descriptor Matching.

Pipeline (run after CameraCalibration.py + RecordGraspPose.py):

    1. Load reference scene  — camera data + masks + D3Fields fusion
    2. Build contact set     — select points near gripper, query descriptors
    3. Load target scene     — same pipeline on the new scene (or self-test)
    4. Coarse matching       — DINO feature NN + SVD Procrustes alignment
    5. Fine optimization     — multi-start SE(3) optimization on the cost function
    6. Save result           — .npz with optimized pose + metadata
    7. Visualize             — Rerun for optimization trajectory, Open3D for rest

Usage
-----
    python GraspPoseSearch.py -d data/ref_scene --gripper-pose demos/target_grasp_pose/pose_0000/gripper_pose.npy
    python GraspPoseSearch.py -d data/ref_scene --new-scene data/new_scene --gripper-pose gripper.txt
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import numpy as np
import torch

from GraphSearch.data_loading import load_scene, load_gripper_pose, build_object_pcd
from GraphSearch.contact_set import build_contact_set, visualize_contact_set
from GraphSearch.coarse_matching import (
    coarse_match_init_pose,
    visualize_coarse_match,
    visualize_feature_pca,
    visualize_similarity_heatmaps_for_contacts,
    debug_pipeline_pcds,
)
from GraphSearch.optimization import optimize_grasp_pose
from GraphSearch.visualization import (
    visualize_optimization,
    visualize_pose,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Grasp pose search via D3Fields descriptor matching")

    # Scene data
    p.add_argument("-d", "--data-dir", type=str, required=True,
                   help="Reference scene directory")
    p.add_argument("--new-scene", type=str, default=None,
                   help="Target scene directory (default: same as --data-dir for self-test)")
    p.add_argument("-t", "--timestep", type=int, default=0,
                   help="Timestep for reference scene (default: 0)")
    p.add_argument("--new-timestep", type=int, default=None,
                   help="Timestep for target scene (default: same as --timestep)")

    # Gripper
    p.add_argument("--gripper-pose", type=str, required=True,
                   help="Path to demo gripper pose (4x4 .txt or .npy)")

    # Workspace boundaries
    p.add_argument("--bounds", type=float, nargs=6,
                   metavar=("XL", "XU", "YL", "YU", "ZL", "ZU"),
                   default=[-0.5, 0.5, -0.5, 0.5, -0.3, 0.3],
                   help="Workspace bounds: x_lo x_hi y_lo y_hi z_lo z_hi")
    p.add_argument("--z-max", type=float, default=None,
                   help="Discard points above this z (robot frame)")

    # Contact set
    p.add_argument("--n-contact", type=int, default=30,
                   help="Number of contact points (default: 25)")
    p.add_argument("--contact-thresh", type=float, default=0.035,
                   help="Distance threshold for contact region (default: 0.035 m)")
    p.add_argument("--contact-mode", type=str, default="surface",
                   choices=["surface", "shell"],
                   help="Contact-point selection strategy: "
                        "'surface' picks points from the object point cloud only; "
                        "'shell' samples random candidates around the fingertip "
                        "and keeps those within --surface-offset of the surface "
                        "(default: shell).")
    p.add_argument("--surface-offset", type=float, default=0.008,
                   help="(shell mode) Max distance a contact query may sit off "
                        "the object surface (default: 0.008 m).")

    # Optimization
    p.add_argument("--w-f", type=float, default=1.0,
                   help="Feature cost weight (set 0 to disable)")
    p.add_argument("--w-d", type=float, default=0.0,
                   help="Distance cost weight (set 0 to disable), default: 0.1")
    p.add_argument("--w-n", type=float, default=0.0,
                   help="Normal-alignment cost weight (set 0 to disable), default: 0.5")
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--perturb-rot", type=float, default=0.6)
    p.add_argument("--perturb-trans", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--feat-backbone", type=str, default="dinov3",
                   choices=["dinov2", "dinov3"],
                   help="DINO backbone for D3Fields features")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(output_path, best_pose, best_cost, init_pose, T_demo,
                 Q, f_star, all_results):
    """Save inference results to a single .npz file."""
    all_poses = np.stack([p for p, _ in all_results])
    all_costs = np.array([c for _, c in all_results])

    np.savez(
        str(output_path),
        best_pose=best_pose,
        best_cost=np.array(best_cost),
        init_pose=init_pose,
        demo_pose=T_demo,
        query_points_local=Q,
        reference_descriptors=f_star,
        all_restart_poses=all_poses,
        all_restart_costs=all_costs,
    )
    print(f"Saved results -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    boundaries = {
        'x_lower': args.bounds[0], 'x_upper': args.bounds[1],
        'y_lower': args.bounds[2], 'y_upper': args.bounds[3],
        'z_lower': args.bounds[4], 'z_upper': args.bounds[5],
    }

    # ── Step 1-2: Load reference scene + build contact set ────────────────
    print("=== Loading reference scene ===")
    fusion_ref, colors_ref, depths_ref, intrinsics, extrinsics, ref_masks, _ = \
        load_scene(args.data_dir, args.timestep, device,
                   feat_backbone=args.feat_backbone)

    print("\n=== Building contact set ===")
    T_demo = load_gripper_pose(args.gripper_pose)
    print(f"  Gripper pose loaded from {args.gripper_pose}")

    object_pcd = build_object_pcd(
        colors_ref, depths_ref, intrinsics, extrinsics, boundaries,
        masks=ref_masks, z_max=args.z_max)
    print(f"  Object point cloud: {object_pcd.shape[0]} points")

    Q, f_star, contact_pts, Q_left, Q_right = build_contact_set(
        fusion_ref, object_pcd, T_demo,
        n_points=args.n_contact,
        distance_thresh=args.contact_thresh,
        mode=args.contact_mode,
        surface_offset=args.surface_offset,
        device=device)
    visualize_contact_set(object_pcd, contact_pts, T_demo, title="Contact Set Verification")

    # ── Step 3: Load target scene ─────────────────────────────────────────
    new_scene = args.new_scene or args.data_dir
    new_t = args.new_timestep if args.new_timestep is not None else args.timestep
    is_self_test = (new_scene == args.data_dir and new_t == args.timestep)

    if is_self_test:
        print("\nSelf-test mode: using same scene for reference and target.")
        fusion_new = fusion_ref
        colors_new, depths_new = colors_ref, depths_ref
        new_masks = ref_masks
    else:
        print(f"\n=== Loading target scene: {new_scene} t={new_t} ===")
        fusion_new, colors_new, depths_new, _, _, new_masks, _ = \
            load_scene(new_scene, new_t, device,
                       feat_backbone=args.feat_backbone)

    object_pcd_new = build_object_pcd(
        colors_new, depths_new, intrinsics, extrinsics, boundaries,
        masks=new_masks, z_max=args.z_max)
    print(f"  Target object point cloud: {object_pcd_new.shape[0]} points")

    # ── Step 3.4: Step-through pcd debug (close each window to advance) ───
    # print("\n=== Pipeline pcd debug (4 windows) ===")
    # debug_pipeline_pcds(
    #     object_pcd_ref=object_pcd,
    #     object_pcd_new=object_pcd_new,
    #     contact_pts=contact_pts,
    #     f_star=f_star,
    #     gripper_pose=T_demo,
    # )

    # ── Step 3.5: Feature-field diagnostics ───────────────────────────────
    # print("\n=== Feature-field diagnostics ===")
    # (a) PCA color map on both scenes — is the feature field discriminative?
    # visualize_feature_pca(
    #     fusion_ref, object_pcd,
    #     fusion_new, object_pcd_new,
    #     contact_pts=contact_pts, device=device,
    #     title="DINO PCA (ref | target)")
    # (b) Per-query cosine-similarity heatmaps for a spread of contact points.
    # visualize_similarity_heatmaps_for_contacts(
    #     fusion_new, object_pcd_new,
    #     f_star=f_star, contact_pts=contact_pts,
    #     object_pcd_ref=object_pcd, device=device)

    # ── Step 4: Coarse DINO matching -> initial pose ──────────────────────
    print("\n=== Coarse DINO feature matching ===")
    init_pose, corr_pts, nn_dists = coarse_match_init_pose(
        fusion_new, object_pcd_new, contact_pts, f_star,
        T_demo, device=device)
    
    # Open3D: side-by-side coarse match (reference | target) + T_coarse / init_pose
    T_coarse = init_pose @ np.linalg.inv(T_demo)
    visualize_coarse_match(
        object_pcd_ref=object_pcd,
        contact_pts_world=contact_pts,
        object_pcd_new=object_pcd_new,
        corr_pts=corr_pts,
        nn_dists=nn_dists,
        T_coarse=T_coarse,
        init_pose=init_pose,
        gripper_pose=T_demo,
        title="Coarse Match (ref | target)",
    )

    # ── Step 5: Fine optimization ─────────────────────────────────────────
    print("\n=== Fine optimization ===")
    best_pose, best_cost, all_results, best_trajectory = optimize_grasp_pose(
        fusion_new, Q, f_star, init_pose,
        w_f=args.w_f, w_d=args.w_d, w_n=args.w_n,
        n_restarts=args.n_restarts, n_iters=args.n_iters, lr=args.lr,
        perturb_rot=args.perturb_rot, perturb_trans=args.perturb_trans,
        device=device)
    print(f"\nBest cost: {best_cost:.6f}")
    print(f"Optimized pose:\n{best_pose}")

    # ── Step 6: Save results (.npz) ──────────────────────────────────────
    out_path = Path(args.data_dir) / "optimized_grasp.npz"
    save_results(out_path, best_pose, best_cost, init_pose, T_demo,
                 Q, f_star, all_results)

    # ── Step 7: Visualize ─────────────────────────────────────────────────
    print("\n=== Visualization ===")

    # Open3D: final optimized pose
    if best_pose is not None:
        visualize_pose(object_pcd_new, Q, best_pose, contact_pts,
                       title="Optimized Grasp Pose")
    else:
        print("WARNING: optimization did not converge — skipping pose visualization.")

    # Rerun: optimization trajectory only
    visualize_optimization(object_pcd_new, Q, best_trajectory)
    print("Rerun trajectory saved. View with:  rerun data/grasp_opt.rrd")

    return best_pose, best_cost


if __name__ == '__main__':
    main()
