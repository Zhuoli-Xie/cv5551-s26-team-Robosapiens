import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'green cube'
robot_ip = '192.168.1.183'

COLOR_RANGES = {
    'blue':   ((100, 80, 50), (130, 255, 255)),
    'red':    ((0, 80, 50), (10, 255, 255)),     # red wraps around; second range added in code
    'red2':   ((170, 80, 50), (180, 255, 255)),   # upper red hue range
    'green':  ((40, 80, 50), (80, 255, 255)),
    'yellow': ((20, 80, 50), (40, 255, 255)),
    'orange': ((10, 80, 50), (20, 255, 255)),
    }

def _parse_color(cube_prompt):
    """Extract the color keyword from a cube prompt string."""
    prompt_lower = cube_prompt.lower()
    for color in COLOR_RANGES:
        if color.endswith('2'):
            continue
        if color in prompt_lower:
            return color
    return None

def _get_color_mask(hsv_image, color):
    """Create a binary mask for the given color in HSV space."""
    lower, upper = COLOR_RANGES[color]
    mask = cv2.inRange(hsv_image, numpy.array(lower), numpy.array(upper))
    # Red wraps around hue=0/180, so combine both ranges
    if color == 'red' and 'red2' in COLOR_RANGES:
        lower2, upper2 = COLOR_RANGES['red2']
        mask2 = cv2.inRange(hsv_image, numpy.array(lower2), numpy.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)
    # Clean up noise
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by the AprilTags.
    """

    # HSV color ranges for each cube color


    def __init__(self, camera_intrinsic):
        """
        Initialize the CubePoseDetector with camera parameters.

        Parameters
        ----------
        camera_intrinsic : numpy.ndarray
            The 3x3 intrinsic camera matrix.
        """
        self.camera_intrinsic = camera_intrinsic
        self.detector = Detector(families=CUBE_TAG_FAMILY)
        self.t_cam_robot = None

    def set_camera_pose(self, t_cam_robot):
        """Store the camera-to-robot transformation."""
        self.t_cam_robot = t_cam_robot

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : numpy.ndarray
            The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both
            are 4x4 transformation matrices with translations in meters.
            If no matching object or tag is found, returns None.
        """
        # Parse target color from prompt
        if len(observation.shape) > 2:
            gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
        else:
            gray = observation

        # Detect all AprilTags
        tags = self.detector.detect(gray, estimate_tag_pose=False)
        if not tags:
            print("No AprilTags detected.")
            return None

        # Determine which cube corresponds to the cube prompt by HSV color thresholding
        color = _parse_color(cube_prompt)
        if color is None:
            print(f"Could not parse color from prompt: '{cube_prompt}'")
            return None

        # Convert observation to BGR then HSV for color matching
        if len(observation.shape) > 2 and observation.shape[2] == 4:
            bgr = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
        elif len(observation.shape) > 2:
            bgr = observation
        else:
            print("Grayscale image cannot be used for color matching.")
            return None

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        color_mask = _get_color_mask(hsv, color)

        # Find the tag whose center overlaps most with the color mask
        cube_tag = None
        best_score = 0
        region_radius = 30  # pixels around tag center to sample

        for tag in tags:
            cx, cy = int(tag.center[0]), int(tag.center[1])
            # Extract a small region around the tag center
            h, w = color_mask.shape[:2]
            y1 = max(0, cy - region_radius)
            y2 = min(h, cy + region_radius)
            x1 = max(0, cx - region_radius)
            x2 = min(w, cx + region_radius)
            region = color_mask[y1:y2, x1:x2]
            score = numpy.count_nonzero(region)
            if score > best_score:
                best_score = score
                cube_tag = tag

        if cube_tag is None or best_score == 0:
            print(f"No cube matching '{cube_prompt}' found.")
            return None

        # Prepare 3D world coordinates of cube tag corners (in cube frame)
        # pupil_apriltags corner order: TL, TR, BR, BL
        s = CUBE_TAG_SIZE
        half = s / 2

        world_points = numpy.array([
            [-half,  half, 0],   # TL
            [ half,  half, 0],   # TR
            [ half, -half, 0],   # BR
            [-half, -half, 0],   # BL
        ], dtype=float)

        # 2D image points
        image_points = numpy.array(cube_tag.corners, dtype=float)

        # SolvePnP to get cube pose in camera frame
        success, rvec, tvec = cv2.solvePnP(
            world_points,
            image_points,
            self.camera_intrinsic,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            print("PnP failed for cube.")
            return None

        R, _ = cv2.Rodrigues(rvec)

        # Build T_cam_cube
        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = R
        t_cam_cube[:3, 3] = tvec.flatten()

        # Convert to robot frame
        T_robot_cam = numpy.linalg.inv(self.t_cam_robot)
        t_robot_cube = T_robot_cam @ t_cam_cube

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

        # Get camera-to-robot transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        cube_pose_detector.set_camera_pose(t_cam_robot)

        # Detect target cube pose
        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
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
