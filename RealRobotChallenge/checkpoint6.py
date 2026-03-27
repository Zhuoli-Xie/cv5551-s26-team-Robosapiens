import cv2, numpy, time
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

CUBE_SIZE = 0.025
cube_prompt = 'blue cube'
robot_ip = '192.168.1.183'

# 从 checkpoint3 搬过来的颜色范围
COLOR_RANGES = {
    'blue':   ((100, 80, 50), (130, 255, 255)),
    'red':    ((0, 80, 50), (10, 255, 255)),
    'red2':   ((170, 80, 50), (180, 255, 255)),
    'green':  ((40, 80, 50), (80, 255, 255)),
    'yellow': ((20, 80, 50), (40, 255, 255)),
    'orange': ((10, 80, 50), (20, 255, 255)),
}


def _parse_color(cube_prompt):
    """Extract the color keyword from a cube prompt string."""
    prompt_lower = cube_prompt.lower()
    for color in COLOR_RANGES:
        if color.endswith('2'):
            continue
        if color in prompt_lower:
            return color
    return None


def _get_color_mask(hsv_image, color):
    """Create a binary mask for the given color in HSV space."""
    lower, upper = COLOR_RANGES[color]
    mask = cv2.inRange(hsv_image, numpy.array(lower), numpy.array(upper))
    if color == 'red' and 'red2' in COLOR_RANGES:
        lower2, upper2 = COLOR_RANGES['red2']
        mask2 = cv2.inRange(hsv_image, numpy.array(lower2), numpy.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_transform_cube(observation, camera_intrinsic, camera_pose, cube_prompt='blue cube'):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame,
    as well as relative to the camera frame.

    Uses color segmentation to isolate the target cube's point cloud, then estimates
    pose via oriented bounding box.

    Parameters
    ----------
    observation : list or tuple
        A collection containing [image, point_cloud].
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix (t_cam_robot). All translations in meters.
    cube_prompt : str
        Text prompt to identify the cube color (e.g., 'blue cube').

    Returns
    -------
    tuple or None
        (t_robot_cube, t_cam_cube) as 4x4 matrices, or None if detection fails.
    """
    image, point_cloud = observation

    # --- 颜色分割 ---
    color = _parse_color(cube_prompt)
    if color is None:
        print(f"Could not parse color from prompt: '{cube_prompt}'")
        return None

    if len(image.shape) > 2 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) > 2:
        bgr = image
    else:
        print("Grayscale image cannot be used for color matching.")
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_mask = _get_color_mask(hsv, color)

    # 展平 mask，和 point cloud 像素一一对应
    flat_mask = color_mask.reshape(-1) > 0

    # --- 提取点云，同时用颜色 mask 过滤 ---
    points_cam = point_cloud[:, :, :3].reshape(-1, 3)
    valid_mask = numpy.isfinite(points_cam).all(axis=1) & flat_mask
    points_cam = points_cam[valid_mask]

    print(f"Color-filtered points ({color}): {points_cam.shape[0]}")

    if points_cam.shape[0] < 10:
        print('Insufficient points after color filtering.')
        return None

    # 转到机器人坐标系
    t_robot_cam = numpy.linalg.inv(camera_pose)
    points_robot = (t_robot_cam[:3, :3] @ points_cam.T + t_robot_cam[:3, 3:4]).T

    # 调试信息
    print(f"  X range: {points_robot[:, 0].min():.4f} ~ {points_robot[:, 0].max():.4f}")
    print(f"  Y range: {points_robot[:, 1].min():.4f} ~ {points_robot[:, 1].max():.4f}")
    print(f"  Z range: {points_robot[:, 2].min():.4f} ~ {points_robot[:, 2].max():.4f}")

    # 空间滤波（宽松一些，主要靠颜色已经过滤过了）
    # workspace_mask = (
    #     (points_robot[:, 2] > -0.05) &
    #     (points_robot[:, 2] < 0.15) &
    #     (points_robot[:, 0] > -0.5) &
    #     (points_robot[:, 0] < 0.5) &
    #     (points_robot[:, 1] > -0.5) &
    #     (points_robot[:, 1] < 0.5)
    # )
    cube_points_robot = points_robot

    cv2.destroyAllWindows()
    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(cube_points_robot)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd_vis,coord],window_name="1")

    print(f"  After workspace filter: {cube_points_robot.shape[0]}")

    if cube_points_robot.shape[0] < 10:
        print('Insufficient points for cube detection.')
        return None

    # 创建 Open3D 点云并去除离群点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cube_points_robot)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    distances = pcd.compute_nearest_neighbor_distance()
    distances = numpy.asarray(distances)

    # # DBSCAN 聚类
    epsilon = distances.mean() * 2.5
    labels = numpy.array(pcd.cluster_dbscan(eps=epsilon, min_points=5))
    # unique_labels = numpy.unique(labels)
    # print("123__",unique_labels)
    if len(labels) == 0 or labels.max() < 0:
        print('No clusters found.')
        return None

    # # 选最大 cluster
    if labels.max() < 0:
        cube_pcd = pcd
    else :
        unique_labels, counts = numpy.unique(labels[labels >= 0], return_counts=True)
        largest_label = unique_labels[numpy.argmax(counts)]
        cube_indices = numpy.where(labels == largest_label)[0]
        cube_pcd = pcd.select_by_index(cube_indices)

    # print(f"  Largest cluster points: {len(cube_indices)}")

    # OBB 估计位姿
    obb = cube_pcd.get_oriented_bounding_box()
    center = numpy.array(obb.center)
    rotation = numpy.array(obb.R)

    # 保证右手系
    if numpy.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1

    # 对齐 z 轴朝上
    z_dots = numpy.abs(rotation[2, :])
    z_axis_idx = numpy.argmax(z_dots)

    # 把最接近世界 z 的那一列换到第三列位置
    cols = [0, 1, 2]
    cols.remove(z_axis_idx)
    new_rotation = numpy.column_stack([
        rotation[:, cols[0]],
        rotation[:, cols[1]],
        rotation[:, z_axis_idx]
    ])

    # 保证 z 列朝上
    if new_rotation[2, 2] < 0:
        new_rotation[:, 2] *= -1

    # 保证右手系
    if numpy.linalg.det(new_rotation) < 0:
        new_rotation[:, 1] *= -1

    t_robot_cube = numpy.eye(4)
    t_robot_cube[:3, :3] = new_rotation
    t_robot_cube[:3, 3] = center 

    # 转回相机坐标系用于可视化
    t_cam_cube = camera_pose @ t_robot_cube 

    return t_robot_cube, t_cam_cube


def main():
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud

        # Get camera-to-robot transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return

        # Estimate cube pose from point cloud + color segmentation
        result = get_transform_cube(
            [cv_image, point_cloud], camera_intrinsic, t_cam_robot, cube_prompt
        )
        if result is None:
            return
        t_robot_cube, t_cam_cube = result

        print(t_robot_cube)

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube, size = 0.2)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # Grasp the cube
            grasp_cube(arm, t_robot_cube)

            # Place the cube back down
            place_cube(arm, t_robot_cube)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()


if __name__ == "__main__":
    main()
