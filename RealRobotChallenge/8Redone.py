import cv2
import numpy as np
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint6 import CUBE_SIZE

cube_prompt = 'blue cube'
robot_ip = ''


class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.
    This version computes cube yaw from point cloud for accurate stacking.
    """

    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic
        self.color_ranges = {
            'blue': ((100, 80, 50), (130, 255, 255)),
            'red': ((0, 80, 50), (10, 255, 255)),
            'red2': ((170, 80, 50), (180, 255, 255)),
            'green': ((40, 80, 50), (80, 255, 255)),
            'yellow': ((20, 80, 50), (40, 255, 255)),
            'orange': ((10, 80, 50), (20, 255, 255)),
        }

    def get_transforms(self, observation, cube_prompt, t_cam_robot=None):
        """
        Compute cube pose in robot frame with correct yaw orientation.
        Returns: t_robot_cube (4x4), t_cam_cube (4x4) or None if not found.
        """
        image, point_cloud = observation

        # 1. Determine target color
        prompt_lower = cube_prompt.lower()
        target_color = None
        for color in self.color_ranges:
            if color.replace('2', '') in prompt_lower:
                target_color = color.replace('2', '')
                break

        if not target_color:
            print(f"Cannot parse color from prompt '{cube_prompt}'")
            return None

        # 2. HSV Mask
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image.copy()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        lower, upper = self.color_ranges[target_color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if target_color == 'red':
            lower2, upper2 = self.color_ranges['red2']
            mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Keep largest contour only
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No '{cube_prompt}' contours detected.")
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # 3. Mask point cloud
        cube_points = point_cloud[mask == 255]
        if cube_points.shape[1] == 4:
            cube_points = cube_points[:, :3]

        valid_mask = ~np.isnan(cube_points).any(axis=1) & ~np.isinf(cube_points).any(axis=1)
        valid_points = cube_points[valid_mask] / 1000.0  # mm -> meters

        if len(valid_points) < 50:
            print(f"Insufficient point cloud points for '{cube_prompt}'.")
            return None

        # 4. Remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd.points) < 50:
            print(f"Point cloud too sparse after outlier removal for '{cube_prompt}'.")
            return None

        points = np.asarray(pcd.points)

        # 5. Compute center
        center = np.mean(points, axis=0)

        # 6. Oriented Bounding Box for yaw
        obb = pcd.get_oriented_bounding_box()
        R_cube = obb.R
        # Extract yaw (rotation about Z)
        rot = R.from_matrix(R_cube)
        _, _, yaw = rot.as_euler('xyz', degrees=False)
        # Snap yaw to nearest 90 degrees
        yaw = np.round(yaw / (np.pi / 2)) * (np.pi / 2)
        # Build rotation: roll=pi (gripper down), pitch=0, yaw snapped
        R_corrected = R.from_euler('xyz', [np.pi, 0, yaw]).as_matrix()

        # 7. t_cam_cube
        t_cam_cube = np.eye(4)
        t_cam_cube[:3, :3] = R_corrected
        t_cam_cube[:3, 3] = center
        print(f"[{cube_prompt}] center in cam (m): {center}, yaw: {yaw:.2f} rad")

        # 8. Camera -> robot transform
        if t_cam_robot is None:
            t_cam_robot = get_transform_camera_robot(image, self.camera_intrinsic)
            if t_cam_robot is None:
                print("Failed to get camera -> robot transform")
                return None

        T_robot_cam = np.linalg.inv(t_cam_robot)
        t_robot_cube = T_robot_cam @ t_cam_cube

        # 9. Adjust Z to cube center (half cube height)
        t_robot_cube[2, 3] -= CUBE_SIZE / 2.0

        print(f"[{cube_prompt}] center in robot (m): {t_robot_cube[:3, 3]}")
        return t_robot_cube, t_cam_cube


def main():
    # Initialize ZED
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Detector
    cube_pose_detector = CubePoseDetector(camera_intrinsic)

    # Initialize Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # Capture
        cv_image = zed.image
        point_cloud = zed.point_cloud
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Cannot get camera -> robot transform")
            return

        result = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_prompt, t_cam_robot)
        if result is None:
            print("Cannot detect cube pose")
            return
        t_robot_cube, t_cam_cube = result

        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube, size=CUBE_SIZE)
        cv2.imshow('Cube Pose Verification', cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Grasp & Place
        grasp_cube(arm, t_robot_cube)
        time.sleep(1.0)
        place_cube(arm, t_robot_cube)

    finally:
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()
