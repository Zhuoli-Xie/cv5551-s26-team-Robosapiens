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

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by the AprilTags.
    """

    # HSV color ranges for each cube color
    COLOR_RANGES = {
        'red':   [( numpy.array([0, 100, 100]),   numpy.array([10, 255, 255])  ),
                  ( numpy.array([160, 100, 100]), numpy.array([180, 255, 255]) )],
        'blue': [( numpy.array([100, 60, 50]), numpy.array([130, 255, 255]) )],
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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # --- 调试：打印整张图里所有像素的 HSV 分布 ---
    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]
    
    # 先用一个宽松范围找到"大概是蓝色"的像素
    loose_mask = cv2.inRange(hsv, numpy.array([80, 30, 30]), numpy.array([140, 255, 255]))
    if loose_mask.sum() > 0:
        print(f"Loose blue pixels found: {loose_mask.sum() // 255}")
        print(f"  H range: {h_channel[loose_mask > 0].min()} - {h_channel[loose_mask > 0].max()}")
        print(f"  S range: {s_channel[loose_mask > 0].min()} - {s_channel[loose_mask > 0].max()}")
        print(f"  V range: {v_channel[loose_mask > 0].min()} - {v_channel[loose_mask > 0].max()}")
    else:
        print("No blue-ish pixels found even with loose range!")
        # 试试打印全图 H 值分布
        unique, counts = numpy.unique(h_channel, return_counts=True)
        top10 = sorted(zip(counts, unique), reverse=True)[:10]
        print("Top 10 H values:", [(int(h), int(c)) for c, h in top10])
    # --- 调试结束 ---
    
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
