from checkpoint8 import CubePoseDetector

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT

stacking_order = ['red cube', 'green cube', 'blue cube']   # From top to bottom
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

        # Detect all three cubes
        cube_poses = {}
        for cube_name in stacking_order:
            result = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_name)
            if result is None:
                print(f'{cube_name} not found.')
                return
            cube_poses[cube_name] = result

        # Visualization - show all cubes
        for cube_name in stacking_order:
            _, t_cam_cube = cube_poses[cube_name]
            draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Poses', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Poses', 1280, 720)
        cv2.imshow('Verifying Cube Poses', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # Stacking order is top to bottom: [top, middle, bottom]
            bottom_cube = stacking_order[2]
            middle_cube = stacking_order[1]
            top_cube = stacking_order[0]

            t_robot_bottom, _ = cube_poses[bottom_cube]

            # Step 1: Pick middle cube and stack on bottom cube
            t_robot_middle, _ = cube_poses[middle_cube]
            grasp_cube(arm, t_robot_middle)

            t_stack1 = t_robot_bottom.copy()
            t_stack1[2, 3] += STACK_HEIGHT
            place_cube(arm, t_stack1)

            # Step 2: Pick top cube and stack on the two-cube tower
            t_robot_top, _ = cube_poses[top_cube]
            grasp_cube(arm, t_robot_top)

            t_stack2 = t_robot_bottom.copy()
            t_stack2[2, 3] += STACK_HEIGHT * 2
            place_cube(arm, t_stack2)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
