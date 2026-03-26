import cv2, numpy, time, torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

CUBE_SIZE = 0.025

robot_ip = ''

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function leverages text prompts to semantically segment a specific 
    cube (e.g., 'red cube') and determines the cube's pose using its 3D point cloud.

    Parameters
    ----------
    observation : list or tuple
        A collection containing [image, point_cloud], where image is the 
        RGB/BGRA array and point_cloud is the registered 3D point cloud.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no matching object is segmented, returns None.
    """
    image, point_cloud = observation

    # Extract XYZ from point cloud (H, W, 4) -> (N, 3), filter NaN/Inf
    points_cam = point_cloud[:, :, :3].reshape(-1, 3)
    valid_mask = numpy.isfinite(points_cam).all(axis=1)
    points_cam = points_cam[valid_mask]

    if points_cam.shape[0] == 0:
        print('No valid points in point cloud.')
        return None

    # Transform points from camera frame to robot frame
    t_robot_cam = numpy.linalg.inv(camera_pose)
    points_robot = (t_robot_cam[:3, :3] @ points_cam.T + t_robot_cam[:3, 3:4]).T

    # Filter points within the workspace (table area)
    # Keep points that are above the table surface and within reasonable bounds
    workspace_mask = (
        (points_robot[:, 2] > 0.005) &   # above table surface
        (points_robot[:, 2] < 0.08) &     # below a reasonable height (cube is ~25mm)
        (points_robot[:, 0] > -0.1) &     # workspace x bounds
        (points_robot[:, 0] < 0.5) &
        (points_robot[:, 1] > -0.5) &     # workspace y bounds
        (points_robot[:, 1] < 0.5)
    )
    cube_points_robot = points_robot[workspace_mask]

    if cube_points_robot.shape[0] < 10:
        print('Insufficient points for cube detection.')
        return None

    # Create Open3D point cloud and remove outliers
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cube_points_robot)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Cluster using DBSCAN to isolate the cube
    labels = numpy.array(pcd.cluster_dbscan(eps=0.01, min_points=10))
    if len(labels) == 0 or labels.max() < 0:
        print('No clusters found.')
        return None

    # Pick the largest cluster as the cube
    unique_labels, counts = numpy.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique_labels[numpy.argmax(counts)]
    cube_indices = numpy.where(labels == largest_label)[0]
    cube_pcd = pcd.select_by_index(cube_indices)

    # Get oriented bounding box for pose estimation
    obb = cube_pcd.get_oriented_bounding_box()

    # Build transformation matrix in robot frame
    center = numpy.array(obb.center)
    rotation = numpy.array(obb.R)

    # Ensure the rotation matrix is right-handed (det = +1)
    if numpy.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1

    # Align z-axis to point upward in robot frame:
    # Find which column of the rotation matrix is most aligned with the world z-axis
    z_axis_idx = numpy.argmax(numpy.abs(rotation[2, :]))
    if rotation[2, z_axis_idx] < 0:
        rotation[:, z_axis_idx] *= -1
        other_idx = [i for i in range(3) if i != z_axis_idx][0]
        rotation[:, other_idx] *= -1

    t_robot_cube = numpy.eye(4)
    t_robot_cube[:3, :3] = rotation
    t_robot_cube[:3, 3] = center

    # Convert back to camera frame for visualization
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

        # Estimate cube pose from point cloud
        result = get_transform_cube([cv_image, point_cloud], camera_intrinsic, t_cam_robot)
        if result is None:
            return
        t_robot_cube, t_cam_cube = result

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
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
