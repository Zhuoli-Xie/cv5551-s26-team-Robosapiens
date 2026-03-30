"""
Challenge 1: The Standard Tower
Stack as many standard-sized cubes as possible into a single, stable vertical tower.
Uses pure vision pipeline (no AprilTags).
"""

from checkpoint8 import CubePoseDetector

import cv2, numpy, time
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from checkpoint1 import GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT
from checkpoint6 import CUBE_SIZE

# All cube colors to scan for
CUBE_COLORS = ['red', 'green', 'blue', 'yellow', 'orange']

robot_ip = '192.168.1.158'

MAX_CUBES = 10

# Faster motion parameters
SPEED = 200        # mm/s (default 100)
PRE_HEIGHT = 40    # mm pre-grasp/place height (reduced from 50)


def fast_grasp(arm, cube_pose):
    """Faster grasp sequence with reduced delays."""
    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000

    rot = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, yaw = rot.as_euler('xyz', degrees=True)

    arm.open_lite6_gripper()
    time.sleep(0.3)

    arm.set_position(x, y, z + PRE_HEIGHT, 180, 0, yaw, speed=SPEED, wait=True)
    arm.set_position(x, y, z, 180, 0, yaw, speed=SPEED, wait=True)

    arm.close_lite6_gripper()
    time.sleep(0.3)

    arm.set_position(x, y, z + PRE_HEIGHT, 180, 0, yaw, speed=SPEED, wait=True)


def fast_place(arm, cube_pose):
    """Faster place sequence with reduced delays and slow final descent."""
    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000

    rot = Rotation.from_matrix(cube_pose[:3, :3])
    _, _, yaw = rot.as_euler('xyz', degrees=True)

    arm.set_position(x, y, z + PRE_HEIGHT, 180, 0, yaw, speed=SPEED, wait=True)
    # Slow descent for precise placement
    arm.set_position(x, y, z, 180, 0, yaw, speed=50, wait=True)

    arm.open_lite6_gripper()
    time.sleep(0.3)

    arm.set_position(x, y, z + PRE_HEIGHT, 180, 0, yaw, speed=SPEED, wait=True)


def detect_all_cubes(cube_pose_detector, zed, t_cam_robot):
    """Scan for all visible cubes, passing cached t_cam_robot to avoid segfault."""
    cv_image = zed.image
    point_cloud = zed.point_cloud
    detected = []

    for color in CUBE_COLORS:
        prompt = f'{color} cube'
        result = cube_pose_detector.get_transforms([cv_image, point_cloud], prompt, t_cam_robot)
        if result is not None:
            t_robot_cube, _ = result
            detected.append((prompt, t_robot_cube))

    return detected


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
    arm.move_gohome(speed=SPEED, wait=True)
    time.sleep(0.3)

    try:
        # Step 1: Get camera-to-robot transform once (uses AprilTags on the table)
        cv_image = zed.image
        from checkpoint0 import get_transform_camera_robot
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Failed to get camera-to-robot transform.")
            return

        # Step 2: Initial scan to find all cubes
        detected = detect_all_cubes(cube_pose_detector, zed, t_cam_robot)
        if len(detected) == 0:
            print("No cubes detected. Exiting.")
            return

        print(f"Detected {len(detected)} cube(s): {[d[0] for d in detected]}")

        # Step 3: First cube = tower base (don't move it)
        base_prompt, base_pose = detected[0]
        print(f"Base cube: {base_prompt}")
        cubes_stacked = 1

        # Step 4: Stack remaining cubes on top
        for attempt in range(1, MAX_CUBES):
            # Re-scan (scene changed after each pick-place)
            time.sleep(0.3)
            detected = detect_all_cubes(cube_pose_detector, zed, t_cam_robot)

            if len(detected) == 0:
                print("No more cubes detected.")
                break

            # Find a cube not at the tower position
            target = None
            for prompt, pose in detected:
                dx = abs(pose[0, 3] - base_pose[0, 3])
                dy = abs(pose[1, 3] - base_pose[1, 3])
                if dx < 0.02 and dy < 0.02:
                    continue
                target = (prompt, pose)
                break

            if target is None:
                print("No more cubes available to stack.")
                break

            target_prompt, target_pose = target
            print(f"Picking: {target_prompt} (cube #{cubes_stacked + 1})")

            # Grasp
            fast_grasp(arm, target_pose)

            # Calculate placement: base XY, stacked height
            place_pose = base_pose.copy()
            place_pose[2, 3] += cubes_stacked * STACK_HEIGHT

            # Place
            fast_place(arm, place_pose)

            cubes_stacked += 1
            print(f"Stacked cube #{cubes_stacked}")

            # Quick return to home
            arm.move_gohome(speed=SPEED, wait=True)
            time.sleep(0.3)

        print(f"Tower complete! {cubes_stacked} cubes stacked.")
        print("Waiting 4 seconds for stability check...")
        time.sleep(4)

    finally:
        arm.move_gohome(wait=True)
        time.sleep(0.3)
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()
