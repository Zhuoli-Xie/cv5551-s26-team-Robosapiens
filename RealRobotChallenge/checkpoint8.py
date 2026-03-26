import cv2, numpy, time
import open3d as o3d
from scipy.spatial.transform import Rotation
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

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by its 3D point cloud.
    """

    # HSV color ranges for each cube color
    COLOR_RANGES = {
        'red':   [( numpy.array([0, 100, 100]),   numpy.array([10, 255, 255])  ),
                  ( numpy.array([160, 100, 100]), numpy.array([180, 255, 255]) )],
        'blue':  [( numpy.array([100, 100, 100]), numpy.array([130, 255, 255]) )],
        'green': [( numpy.array([40, 80, 80]),    numpy.array([80, 255, 255])  )],
    }

    def __init__(self, camera_intrinsic):
        """
        Initialize the CubePoseDetector with camera parameters.

        Parameters
        ----------
        camera_intrinsic : numpy.ndarray
            The 3x3 intrinsic camera matrix.
        """
        self.camera_intrinsic = camera_intrinsic
        self.t_cam_robot = None

    def set_camera_pose(self, t_cam_robot):
        """Store the camera-to-robot transformation."""
        self.t_cam_robot = t_cam_robot

    def _parse_color(self, cube_prompt):
        """Extract the color name from the text prompt."""
        prompt_lower = cube_prompt.lower()
        for color in self.COLOR_RANGES:
            if color in prompt_lower:
                return color
        return None

    def _get_color_mask(self, image, color):
        """Create a binary mask for the specified color using HSV thresholding."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for lower, upper in self.COLOR_RANGES[color]:
            mask |= cv2.inRange(hsv, lower, upper)
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : list or tuple
            A collection containing [image, point_cloud], where image is the
            RGB/BGRA array and point_cloud is the registered 3D point cloud.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both
            are 4x4 transformation matrices with translations in meters.
            If no matching object is segmented, returns None.
        """
        image, point_cloud = observation

        # Parse target color
        color = self._parse_color(cube_prompt)
        if color is None:
            print(f'Unknown color in prompt: {cube_prompt}')
            return None

        # Prepare BGR image for color detection
        if len(image.shape) == 2:
            print('Color image required.')
            return None
        if image.shape[2] == 4:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr_image = image

        # Create 2D color mask to isolate target cube pixels
        mask = self._get_color_mask(bgr_image, color)

        # Use the 2D mask to extract corresponding 3D points from the point cloud
        # point_cloud is (H, W, 4), mask is (H, W)
        points_cam = point_cloud[:, :, :3]  # (H, W, 3)
        mask_bool = mask > 0

        # Extract masked 3D points
        target_points_cam = points_cam[mask_bool]  # (N, 3)

        # Filter NaN/Inf
        valid_mask = numpy.isfinite(target_points_cam).all(axis=1)
        target_points_cam = target_points_cam[valid_mask]

        if target_points_cam.shape[0] < 10:
            print(f'Insufficient points for {color} cube.')
            return None

        # Transform to robot frame
        t_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        target_points_robot = (t_robot_cam[:3, :3] @ target_points_cam.T + t_robot_cam[:3, 3:4]).T

        # Create Open3D point cloud and remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(target_points_robot)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 10:
            print(f'Insufficient points after filtering for {color} cube.')
            return None

        # Get oriented bounding box for pose estimation
        obb = pcd.get_oriented_bounding_box()
        center = numpy.array(obb.center)
        rotation = numpy.array(obb.R)

        # Ensure right-handed rotation matrix
        if numpy.linalg.det(rotation) < 0:
            rotation[:, 2] *= -1

        # Align z-axis to point upward in robot frame
        z_axis_idx = numpy.argmax(numpy.abs(rotation[2, :]))
        if rotation[2, z_axis_idx] < 0:
            rotation[:, z_axis_idx] *= -1
            other_idx = [i for i in range(3) if i != z_axis_idx][0]
            rotation[:, other_idx] *= -1

        t_robot_cube = numpy.eye(4)
        t_robot_cube[:3, :3] = rotation
        t_robot_cube[:3, 3] = center

        # Convert to camera frame for visualization
        t_cam_cube = self.t_cam_robot @ t_robot_cube

        return t_robot_cube, t_cam_cube

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

        # Detect target cube pose
        result = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_prompt)
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

            # Grasp the target cube
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
