from checkpoint8 import CubePoseDetector

import cv2, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
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

        # TODO
        result_green = cube_pose_detector.get_transforms([cv_image, point_cloud], 'green cube')
        result_red = cube_pose_detector.get_transforms([cv_image, point_cloud], 'red cube')
        t_robot_green, _ = result_green
        t_robot_red, _ = result_red
        grasp_cube(arm, t_robot_red)
        time.sleep(0.5)
        target_place_pose = t_robot_green.copy()
        target_place_pose[2, 3] += STACK_HEIGHT
        place_cube(arm, target_place_pose)
        time.sleep(0.5)
                
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
