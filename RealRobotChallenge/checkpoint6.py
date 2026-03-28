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
    Estimate the 6D pose of the cube in the robot base frame.
    """
    image, point_cloud = observation

    
    if len(image.shape) > 2 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = numpy.array([0, 100, 50])
    upper_red1 = numpy.array([10, 255, 255])
    lower_red2 = numpy.array([170, 100, 50])
    upper_red2 = numpy.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cube_points = point_cloud[mask == 255]
    
    if cube_points.shape[1] == 4:
        cube_points = cube_points[:, :3]

    valid_mask = ~numpy.isnan(cube_points).any(axis=1) & ~numpy.isinf(cube_points).any(axis=1)
    valid_points = cube_points[valid_mask]

    if len(valid_points) < 50:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    obb = pcd.get_oriented_bounding_box()
    center = obb.center  
    rotation = obb.R     

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = rotation
    t_cam_cube[:3, 3] = center

    T_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = T_robot_cam @ t_cam_cube

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

        t_cam_cube = None
        
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
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

            
            grasp_cube(arm, t_robot_cube)
            time.sleep(1.0)
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