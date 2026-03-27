import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'blue cube'
robot_ip = ''


def find_color_ranges(image):
    """
    打印图像中红、蓝、绿像素的实际 HSV 范围，并输出建议的 COLOR_RANGES。
    把三个颜色的方块都放在画面里，跑一次即可。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # 用宽松范围捞出大概属于该颜色的像素
    loose = {
        'red_low':  (numpy.array([0,   20, 20]), numpy.array([15,  255, 255])),
        'red_high': (numpy.array([150, 20, 20]), numpy.array([180, 255, 255])),
        'blue':     (numpy.array([75,  20, 20]), numpy.array([145, 255, 255])),
        'green':    (numpy.array([30,  20, 20]), numpy.array([90,  255, 255])),
    }

    print("=" * 50)
    print("COLOR RANGE DETECTION RESULTS")
    print("=" * 50)

    for name, (lo, hi) in loose.items():
        mask = cv2.inRange(hsv, lo, hi)
        count = numpy.count_nonzero(mask)
        if count > 0:
            print(f"[{name}] pixels: {count}")
            print(f"  H: {h[mask > 0].min()} - {h[mask > 0].max()}")
            print(f"  S: {s[mask > 0].min()} - {s[mask > 0].max()}")
            print(f"  V: {v[mask > 0].min()} - {v[mask > 0].max()}")
        else:
            print(f"[{name}] no pixels found")

    print("\n" + "=" * 50)
    print("SUGGESTED COLOR_RANGES (copy into your class):")
    print("=" * 50 + "\n")

    for color, keys in [('red', ['red_low', 'red_high']), ('blue', ['blue']), ('green', ['green'])]:
        ranges = []
        for k in keys:
            lo, hi = loose[k]
            mask = cv2.inRange(hsv, lo, hi)
            if numpy.count_nonzero(mask) > 0:
                h_min, h_max = int(h[mask > 0].min()), int(h[mask > 0].max())
                s_min, s_max = int(s[mask > 0].min()), int(s[mask > 0].max())
                v_min, v_max = int(v[mask > 0].min()), int(v[mask > 0].max())
                # 留点余量
                h_min = max(0, h_min - 5)
                h_max = min(180, h_max + 5)
                s_min = max(0, s_min - 10)
                v_min = max(0, v_min - 10)
                ranges.append(
                    f"(numpy.array([{h_min}, {s_min}, {v_min}]), numpy.array([{h_max}, 255, 255]))"
                )
        if ranges:
            print(f"'{color}': [{', '.join(ranges)}],")
        else:
            print(f"'{color}': [],  # no pixels detected")

    print()


class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g.,
    'blue cube') and determine the cube's pose by the AprilTags.
    """

    # HSV color ranges for each cube color
    # 如果检测不到，先跑 find_color_ranges() 获取实际范围再替换这里
    COLOR_RANGES = {
        'red':   [(numpy.array([0,   30, 30]), numpy.array([15,  255, 255])),
                  (numpy.array([155, 30, 30]), numpy.array([180, 255, 255]))],
        'blue':  [(numpy.array([80,  30, 30]), numpy.array([140, 255, 255]))],
        'green': [(numpy.array([30,  20, 20]), numpy.array([90,  255, 255]))],
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
        self.detector = Detector(families=CUBE_TAG_FAMILY)
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
        # Clean up the mask
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
        color = self._parse_color(cube_prompt)
        if color is None:
            print(f'Unknown color in prompt: {cube_prompt}')
            return None

        # Prepare BGR image for color detection
        if len(observation.shape) == 2:
            print('Color image required for color detection.')
            return None
        if observation.shape[2] == 4:
            bgr_image = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
        else:
            bgr_image = observation

        # Create color mask
        mask = self._get_color_mask(bgr_image, color)
        cv2.imwrite("mask.png", mask)

        # Detect all AprilTags
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY) if len(observation.shape) > 2 else observation
        tags = self.detector.detect(gray, estimate_tag_pose=False)
        print(f'Detected {len(tags)} tags, looking for {color} cube')

        # Find which tag's center falls inside the color mask
        target_tag = None
        for tag in tags:
            cx, cy = int(tag.center[0]), int(tag.center[1])
            if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                if mask[cy, cx] > 0:
                    target_tag = tag
                    break

        # If center didn't match, try checking which tag has most corner overlap with the mask
        if target_tag is None:
            best_score = 0
            for tag in tags:
                score = 0
                for corner in tag.corners:
                    cx, cy = int(corner[0]), int(corner[1])
                    if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
                        if mask[cy, cx] > 0:
                            score += 1
                if score > best_score:
                    best_score = score
                    target_tag = tag
            if best_score == 0:
                print(f'No tag found on {color} cube.')
                return None

        print(f'Matched tag ID {target_tag.tag_id} for {color} cube')

        # Compute pose via PnP
        half = CUBE_TAG_SIZE / 2
        object_points = numpy.array([
            [-half,  half, 0],
            [-half, -half, 0],
            [ half, -half, 0],
            [ half,  half, 0],
        ], dtype=numpy.float64)

        image_points = target_tag.corners.astype(numpy.float64)

        success, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_intrinsic, None)
        if not success:
            print('PnP failed for target cube.')
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = rmat
        t_cam_cube[:3, 3] = tvec.flatten()

        # Transform to robot frame
        t_robot_cube = numpy.linalg.inv(self.t_cam_robot) @ t_cam_cube

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

        # --- 调试：检测颜色范围（确认后可注释掉） ---
        if cv_image.shape[2] == 4:
            bgr_debug = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        else:
            bgr_debug = cv_image
        find_color_ranges(bgr_debug)
        # --- 调试结束 ---

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
