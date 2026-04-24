"""
Visualization functions.

- Open3D: static views (point clouds, poses, DINO matching)
- Rerun:  optimization trajectory only
"""

import numpy as np
import open3d as o3d
import rerun as rr


# ---------------------------------------------------------------------------
# Open3D: generic helpers
# ---------------------------------------------------------------------------

def visualize_o3d(point_clouds, poses=None, title="Debug", frame_size=0.05):
    """Show point clouds and optional SE(3) poses in an Open3D window.

    Args:
        point_clouds: list of (points, color) tuples.
                      points: (N,3) np array, color: (3,) float [0-1].
        poses:        list of (4,4) np arrays (coordinate frames to draw).
        title:        window title.
        frame_size:   axis length for coordinate frames.
    """
    geometries = []
    for pts, color in point_clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    for pose in (poses or []):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame.transform(pose)
        geometries.append(frame)

    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1024, height=768)


# ---------------------------------------------------------------------------
# Open3D: DINO matching visualization
# ---------------------------------------------------------------------------

def visualize_dino_matching(object_pcd_np, contact_pts, corr_pts,
                            nn_dists=None, title="DINO Matching"):
    """
    Visualize DINO feature correspondences: source contacts, matched targets,
    and connecting lines colored by NN distance (green=close, red=far).
    """
    geometries = []

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    obj_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(obj_pcd)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(contact_pts)
    src_pcd.paint_uniform_color([0.2, 0.4, 1.0])
    geometries.append(src_pcd)

    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(corr_pts)
    dst_pcd.paint_uniform_color([0.2, 1.0, 0.4])
    geometries.append(dst_pcd)

    n = contact_pts.shape[0]
    line_points = np.concatenate([contact_pts, corr_pts], axis=0)
    line_indices = [[i, i + n] for i in range(n)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)

    if nn_dists is not None:
        d = nn_dists.copy()
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)
        colors = np.zeros((n, 3))
        colors[:, 0] = d_norm
        colors[:, 1] = 1.0 - d_norm
    else:
        colors = np.full((n, 3), [1.0, 0.8, 0.2])

    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1024, height=768)


# ---------------------------------------------------------------------------
# Open3D: single pose visualization
# ---------------------------------------------------------------------------

def visualize_pose(object_pcd_np, Q, pose, title="Pose", frame_size=0.05):
    """Show a single gripper pose with query points and the robot base frame."""
    geometries = []

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(object_pcd_np)
    obj_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(obj_pcd)

    gripper_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    gripper_frame.transform(pose)
    geometries.append(gripper_frame)

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 2)
    geometries.append(base_frame)

    ones = np.ones((Q.shape[0], 1))
    Q_homo = np.hstack([Q, ones])
    x_world = (pose @ Q_homo.T).T[:, :3]
    q_pcd = o3d.geometry.PointCloud()
    q_pcd.points = o3d.utility.Vector3dVector(x_world)
    q_pcd.paint_uniform_color([1.0, 0.2, 0.2])
    geometries.append(q_pcd)

    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1024, height=768)


# ---------------------------------------------------------------------------
# Rerun: optimization trajectory (the ONLY thing using Rerun)
# ---------------------------------------------------------------------------

_rr_initialized = False


def _ensure_rr(app_id="grasp_pose_search", save_path="data/grasp_opt.rrd"):
    """Initialize Rerun once, saving to an .rrd file (no viewer spawned)."""
    global _rr_initialized
    if not _rr_initialized:
        rr.init(app_id)
        rr.save(save_path)
        _rr_initialized = True


def _log_gripper_axes(entity_path, pose, size=0.05):
    """Log three line segments (RGB = XYZ) representing a coordinate frame."""
    origin = pose[:3, 3]
    axes = pose[:3, :3] * size
    points = np.stack([
        np.stack([origin, origin + axes[:, 0]]),
        np.stack([origin, origin + axes[:, 1]]),
        np.stack([origin, origin + axes[:, 2]]),
    ])
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    rr.log(entity_path, rr.LineStrips3D(points, colors=colors, radii=0.002))


def visualize_optimization(object_pcd_np, Q, trajectory, frame_size=0.05):
    """Log the optimization trajectory to Rerun as a time sequence."""
    _ensure_rr()

    prefix = "optimization"

    rr.log(f"{prefix}/object_pcd", rr.Points3D(
        object_pcd_np,
        colors=np.full((object_pcd_np.shape[0], 3), 180, dtype=np.uint8),
        radii=0.002), static=True)

    ones = np.ones((Q.shape[0], 1))
    Q_homo = np.hstack([Q, ones])

    trail_pts_list = []

    for step_i, (pose_i, cost_i) in enumerate(trajectory):
        rr.set_time("opt_step", sequence=step_i)

        rr.log(f"{prefix}/cost", rr.Scalars(cost_i))
        _log_gripper_axes(f"{prefix}/gripper_frame", pose_i, size=frame_size)

        x_world = (pose_i @ Q_homo.T).T[:, :3]
        red = np.full((x_world.shape[0], 3), [255, 50, 50], dtype=np.uint8)
        rr.log(f"{prefix}/query_pts",
               rr.Points3D(x_world, colors=red, radii=0.004))

        trail_pts_list.append(x_world)
        trail_all = np.concatenate(trail_pts_list, axis=0)
        n_trail = trail_all.shape[0]
        green_vals = np.linspace(80, 230, n_trail).astype(np.uint8)
        trail_colors = np.zeros((n_trail, 3), dtype=np.uint8)
        trail_colors[:, 1] = green_vals
        rr.log(f"{prefix}/trail",
               rr.Points3D(trail_all, colors=trail_colors, radii=0.002))
