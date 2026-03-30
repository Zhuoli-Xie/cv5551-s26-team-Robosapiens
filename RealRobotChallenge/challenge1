"""
Challenge 1: The Standard Tower
Stack as many standard-sized cubes as possible into a single, stable vertical tower.
Uses pure vision pipeline (no AprilTags).
"""

from checkpoint8 import CubePoseDetector

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT

# All cube colors to scan for
CUBE_COLORS = ['red', 'green', 'blue', 'yellow', 'orange']

robot_ip = '192.168.1.183'

# Maximum number of cubes to attempt stacking
MAX_CUBES = 10


def detect_all_cubes(cube_pose_detector, zed):
    """
    Scan for all visible cubes of supported colors.

    Returns
    -------
    list of (str, numpy.ndarray)
        List of (color_name, t_robot_cube) for each detected cube.
    """
    cv_image = zed.image
    point_cloud = zed.point_cloud
    detected = []

    for color in CUBE_COLORS:
        prompt = f'{color} cube'
        result = cube_pose_detector.get_transforms([cv_image, point_cloud], prompt)
        if result is not None:
            t_robot_cube, _ = result
            detected.append((prompt, t_robot_cube))

    return detected


def main():
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Pose Detector (pure vision, no AprilTags)
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
        # --- Step 1: Initial scan to find all cubes ---
        detected = detect_all_cubes(cube_pose_detector, zed)
        if len(detected) == 0:
            print("No cubes detected. Exiting.")
            return

        print(f"Detected {len(detected)} cube(s): {[d[0] for d in detected]}")

        # --- Step 2: Choose the first cube as the tower base (don't move it) ---
        base_prompt, base_pose = detected[0]
        print(f"Base cube: {base_prompt}")
        cubes_stacked = 1  # The base counts as 1

        # --- Step 3: Stack remaining cubes on top of the base ---
        for attempt in range(1, MAX_CUBES):
            # Re-scan to find remaining cubes (scene has changed)
            time.sleep(0.5)
            detected = detect_all_cubes(cube_pose_detector, zed)

            if len(detected) == 0:
                print("No more cubes detected.")
                break

            # Pick the first available cube that is NOT at the base position
            target = None
            for prompt, pose in detected:
                dx = abs(pose[0, 3] - base_pose[0, 3])
                dy = abs(pose[1, 3] - base_pose[1, 3])
                # Skip if too close to the tower base (likely the base itself or already stacked)
                if dx < 0.02 and dy < 0.02:
                    continue
                target = (prompt, pose)
                break

            if target is None:
                print("No more cubes available to stack.")
                break

            target_prompt, target_pose = target
            print(f"Picking up: {target_prompt} (cube #{cubes_stacked + 1})")

            # Grasp the target cube
            grasp_cube(arm, target_pose)
            time.sleep(0.5)

            # Calculate placement position: base XY, stacked height
            place_pose = base_pose.copy()
            place_pose[2, 3] += cubes_stacked * STACK_HEIGHT

            # Place the cube on top of the tower
            place_cube(arm, place_pose)
            time.sleep(0.5)

            cubes_stacked += 1
            print(f"Successfully stacked cube #{cubes_stacked}")

            # Return to home position before next cycle
            arm.move_gohome(wait=True)
            time.sleep(0.5)

        print(f"Tower complete! Total cubes stacked: {cubes_stacked}")
        # Wait for tower to stabilize (must stand 3 seconds unsupported)
        print("Waiting for tower stability check (3 seconds)...")
        time.sleep(4)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()


if __name__ == "__main__":
    main()
