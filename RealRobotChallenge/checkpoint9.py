from checkpoint8 import CubePoseDetector

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT

robot_ip = ''

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Pose Detector
    cube_pose_detector = CubePoseDetector(camera_intrinsic)

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
        cube_pose_detector.set_camera_pose(t_cam_robot)

        # Detect red cube (to pick up)
        result_red = cube_pose_detector.get_transforms([cv_image, point_cloud], 'red cube')
        if result_red is None:
            print('Red cube not found.')
            return
        t_robot_red, t_cam_red = result_red

        # Detect green cube (stack target)
        result_green = cube_pose_detector.get_transforms([cv_image, point_cloud], 'green cube')
        if result_green is None:
            print('Green cube not found.')
            return
        t_robot_green, t_cam_green = result_green

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_red)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_green)
        cv2.namedWindow('Verifying Cube Poses', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Poses', 1280, 720)
        cv2.imshow('Verifying Cube Poses', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # Grasp the red cube
            grasp_cube(arm, t_robot_red)

            # Compute stacking pose: green cube position + stack height offset
            t_stack = t_robot_green.copy()
            t_stack[2, 3] += STACK_HEIGHT

            # Place the red cube on top of the green cube
            place_cube(arm, t_stack)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
