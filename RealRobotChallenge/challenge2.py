"""
Challenge 2: The Irregular Skyscraper
Stack cubes of varied, unknown sizes (15mm–35mm) into the tallest stable tower.
Uses pure vision pipeline (no AprilTags).
Strategy: large cubes on the bottom, small cubes on top for maximum stability.
"""

from checkpoint8 import CubePoseDetector

import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

# All cube colors to scan for
CUBE_COLORS = ['red', 'green', 'blue', 'yellow', 'orange']

robot_ip = '192.168.1.183'

MAX_CUBES = 10


def detect_cube_with_size(cube_pose_detector, image, point_cloud, color):
    """
    Detect a cube and measure its size from the OBB extent.

    Returns
    -------
    tuple or None
        (t_robot_cube, cube_size) where cube_size is the estimated
        side length in meters, or None if not detected.
    """
    prompt = f'{color} cube'

    # Parse color and create mask (reuse logic from CubePoseDetector)
    prompt_lower = prompt.lower()
    target_color = None
    for c in cube_pose_detector.color_ranges:
        if c.replace('2', '') in prompt_lower:
            target_color = c.replace('2', '')
            break
    if not target_color:
        return None

    if len(image.shape) > 2 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image.copy()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower, upper = cube_pose_detector.color_ranges[target_color]
    mask = cv2.inRange(hsv, numpy.array(lower), numpy.array(upper))
    if target_color == 'red':
        lower2, upper2 = cube_pose_detector.color_ranges['red2']
        mask2 = cv2.inRange(hsv, numpy.array(lower2), numpy.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)

    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Extract 3D points
    cube_points = point_cloud[mask == 255]
    if cube_points.shape[1] == 4:
        cube_points = cube_points[:, :3]

    valid_mask = ~numpy.isnan(cube_points).any(axis=1) & ~numpy.isinf(cube_points).any(axis=1)
    valid_points = cube_points[valid_mask]

    if len(valid_points) < 50:
        return None

    # Compute OBB to get pose and size
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    obb = pcd.get_oriented_bounding_box()

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = obb.R
    t_cam_cube[:3, 3] = obb.center

    # Get cube size from OBB extent (take median of 3 extents as side length)
    extent = numpy.array(obb.extent)
    cube_size = float(numpy.median(extent))

    # Transform to robot frame
    result = cube_pose_detector.get_transforms([image, point_cloud], prompt)
    if result is None:
        return None

    t_robot_cube, _ = result

    return t_robot_cube, cube_size


def detect_all_cubes_with_size(cube_pose_detector, zed):
    """
    Scan for all visible cubes and measure their sizes.

    Returns
    -------
    list of (str, numpy.ndarray, float)
        List of (color_name, t_robot_cube, cube_size_meters) sorted by size descending.
    """
    cv_image = zed.image
    point_cloud = zed.point_cloud
    detected = []

    for color in CUBE_COLORS:
        result = detect_cube_with_size(cube_pose_detector, cv_image, point_cloud, color)
        if result is not None:
            t_robot_cube, cube_size = result
            detected.append((f'{color} cube', t_robot_cube, cube_size))

    # Sort by size descending: largest cubes go on the bottom
    detected.sort(key=lambda x: x[2], reverse=True)
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
        # --- Step 1: Initial scan to find all cubes and their sizes ---
        detected = detect_all_cubes_with_size(cube_pose_detector, zed)
        if len(detected) == 0:
            print("No cubes detected. Exiting.")
            return

        print(f"Detected {len(detected)} cube(s):")
        for name, _, size in detected:
            print(f"  {name}: {size * 1000:.1f} mm")

        # --- Step 2: The largest cube becomes the tower base (don't move it) ---
        base_prompt, base_pose, base_size = detected[0]
        print(f"Base cube: {base_prompt} ({base_size * 1000:.1f} mm)")
        cubes_stacked = 1
        # Track cumulative tower height above the base cube's center
        # The base cube center is at base_pose[2,3]; top surface = center + half size
        tower_height = base_size  # height from base center to top of base cube = base_size/2,
                                   # but we place next cube center at base_top + next_size/2

        # Record sizes of stacked cubes in order (bottom to top) for height calculation
        stacked_sizes = [base_size]

        # --- Step 3: Stack remaining cubes, largest first ---
        for attempt in range(1, MAX_CUBES):
            time.sleep(0.5)
            detected = detect_all_cubes_with_size(cube_pose_detector, zed)

            if len(detected) == 0:
                print("No more cubes detected.")
                break

            # Find a cube that is NOT at the tower base position
            target = None
            for prompt, pose, size in detected:
                dx = abs(pose[0, 3] - base_pose[0, 3])
                dy = abs(pose[1, 3] - base_pose[1, 3])
                if dx < 0.02 and dy < 0.02:
                    continue
                target = (prompt, pose, size)
                break

            if target is None:
                print("No more cubes available to stack.")
                break

            target_prompt, target_pose, target_size = target
            print(f"Picking up: {target_prompt} ({target_size * 1000:.1f} mm, cube #{cubes_stacked + 1})")

            # Grasp the target cube
            grasp_cube(arm, target_pose)
            time.sleep(0.5)

            # Calculate placement height:
            # stack_z = base_center_z + sum of all stacked cube sizes above base center
            # For cube i on top of cube i-1:
            #   z = base_z + base_size/2 + sum(sizes[1:i]) + target_size/2
            place_pose = base_pose.copy()
            height_above_base = base_size / 2.0  # from base center to base top
            for s in stacked_sizes[1:]:
                height_above_base += s  # full size of each intermediate cube
            height_above_base += target_size / 2.0  # from top of last cube to center of new cube
            place_pose[2, 3] += height_above_base

            # Place the cube on top of the tower
            place_cube(arm, place_pose)
            time.sleep(0.5)

            stacked_sizes.append(target_size)
            cubes_stacked += 1
            total_height = sum(stacked_sizes) * 1000
            print(f"Stacked cube #{cubes_stacked}, tower height: {total_height:.1f} mm")

            arm.move_gohome(wait=True)
            time.sleep(0.5)

        total_height = sum(stacked_sizes) * 1000
        print(f"Tower complete! {cubes_stacked} cubes, total height: {total_height:.1f} mm")
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
