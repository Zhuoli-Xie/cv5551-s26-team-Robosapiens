import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

GRIPPER_LENGTH = 0.067 * 1000
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.025

robot_ip = '192.168.1.183'

def grasp_cube(arm, cube_pose):
    """
    Execute a pick sequence to grasp a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the cube's pose in the robot base frame.
        All translational units in this matrix are in meters.
    """
    # Extract position (meters -> mm)
    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000

    # Extract yaw from cube orientation for gripper alignment
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, yaw = rot.as_euler('xyz', degrees=True)

    PRE_GRASP_HEIGHT = 50  # mm above the cube

    # Open gripper
    arm.open_lite6_gripper()
    time.sleep(0.5)

    # Move to pre-grasp position (safe height above cube)
    arm.set_position(x, y, z + PRE_GRASP_HEIGHT, 180, 0, yaw, wait=True)
    time.sleep(0.3)

    # Descend to grasp position
    arm.set_position(x, y, z, 180, 0, yaw, wait=True)
    time.sleep(0.3)

    # Close gripper to grasp
    arm.close_lite6_gripper()
    time.sleep(0.5)

    # Lift up to safe height
    arm.set_position(x, y, z + PRE_GRASP_HEIGHT, 180, 0, yaw, wait=True)
    time.sleep(0.3)

def place_cube(arm, cube_pose):
    """
    Execute a place sequence to release a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the target placement pose in the robot base frame.
        All translational units in this matrix are in meters.
    """
    # Extract position (meters -> mm)
    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000

    # Extract yaw from cube orientation
    rot = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, yaw = rot.as_euler('xyz', degrees=True)

    PRE_PLACE_HEIGHT = 50  # mm above the placement position

    # Move to pre-place position (safe height)
    arm.set_position(x, y, z + PRE_PLACE_HEIGHT, 180, 0, yaw, wait=True)
    time.sleep(0.3)

    # Descend to place position
    arm.set_position(x, y, z, 180, 0, yaw, wait=True)
    time.sleep(0.3)

    # Open gripper to release
    arm.open_lite6_gripper()
    time.sleep(0.5)

    # Lift up to safe height
    arm.set_position(x, y, z + PRE_PLACE_HEIGHT, 180, 0, yaw, wait=True)
    time.sleep(0.3)

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function uses visual fiducial detection to find the cube's pose in the camera's view, 
    then transforms that pose into the robot's global coordinate system. 

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
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
        If no cube tag is detected, returns None.
    """
    detector = Detector(families=CUBE_TAG_FAMILY)

    # Convert to grayscale if needed
    if len(observation.shape) > 2:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    else:
        gray = observation

    # Detect AprilTags
    tags = detector.detect(gray, estimate_tag_pose=False)

    # Find the cube tag (ID = CUBE_TAG_ID)
    cube_tag = None
    for tag in tags:
        if tag.tag_id == CUBE_TAG_ID:
            cube_tag = tag
            break

    if cube_tag is None:
        print('Cube tag not found.')
        return None

    # Define 3D corner coordinates of the tag in its local frame (z=0)
    # Corner order matches pupil_apriltags: bottom-left, bottom-right, top-right, top-left
    half = CUBE_TAG_SIZE / 2
    object_points = numpy.array([
        [-half,  half, 0],
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
    ], dtype=numpy.float64)

    image_points = cube_tag.corners.astype(numpy.float64)

    # Solve PnP to get tag pose in camera frame
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_intrinsic, None)
    if not success:
        print('PnP failed for cube tag.')
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = rmat
    t_cam_cube[:3, 3] = tvec.flatten()

    # Transform cube pose from camera frame to robot base frame
    t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube

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

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        # Estimate cube pose
        result = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot)
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
