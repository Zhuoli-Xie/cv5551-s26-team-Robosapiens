import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'blue cube'
robot_ip = '192.168.1.183'  # 别忘了填入你的 xArm IP

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.
    """

    # HSV color ranges for each cube color
    COLOR_RANGES = {
        'red':   [( numpy.array([0, 100, 100]),   numpy.array([10, 255, 255])  ),
                  ( numpy.array([160, 100, 100]), numpy.array([180, 255, 255]) )],
        'blue':  [( numpy.array([100, 60, 50]),   numpy.array([130, 255, 255]) )], # 修复了中文逗号
        'green': [( numpy.array([40, 80, 80]),    numpy.array([80, 255, 255])  )],
    }

    def __init__(self, camera_intrinsic):
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
        
        # Clean up the mask (形态学开闭运算去噪)
        kernel = numpy.ones((5, 5), numpy.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def get_transforms(self, observation, cube_prompt):
        """Calculate the transformation matrix for a specific prompted cube."""
        color = self._parse_color(cube_prompt)
        if color is None:
            print(f"Could not parse color from prompt: '{cube_prompt}'")
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
        # cv2.imwrite("mask.png", mask) # 取消注释可用于调试掩膜生成情况

        # Detect all AprilTags
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY) if len(observation.shape) > 2 else observation
        tags = self.detector.detect(gray, estimate_tag_pose=False)
        
        if not tags:
            print("No AprilTags detected.")
            return None

        # --- 回滚到版本1的稳健匹配逻辑 (区域积分法) ---
        target_tag = None
        best_score = 0
        region_radius = 30  # pixels around tag center to sample

        for tag in tags:
            cx, cy = int(tag.center[0]), int(tag.center[1])
            h, w = mask.shape[:2]
            y1 = max(0, cy - region_radius)
            y2 = min(h, cy + region_radius)
            x1 = max(0, cx - region_radius)
            x2 = min(w, cx + region_radius)
            
            region = mask[y1:y2, x1:x2]
            score = numpy.count_nonzero(region)
            
            if score > best_score:
                best_score = score
                target_tag = tag

        if target_tag is None or best_score == 0:
            print(f"No cube matching '{cube_prompt}' found.")
            return None

        print(f'Matched tag ID {target_tag.tag_id} for {color} cube')

        # --- 回滚到版本1正确的PnP解算逻辑 ---
        s = CUBE_TAG_SIZE
        half = s / 2
        # 注意：这里的角点顺序必须是 TL, TR, BR, BL 才能正确对应 V1 的解算
        object_points = numpy.array([
            [-half,  half, 0],   # TL
            [ half,  half, 0],   # TR
            [ half, -half, 0],   # BR
            [-half, -half, 0],   # BL
        ], dtype=float)

        image_points = numpy.array(target_tag.corners, dtype=float)

        # 恢复使用 SOLVEPNP_IPPE_SQUARE 算法
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_intrinsic,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

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
            print("Failed to detect the target cube.")
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
            time.sleep(1.0)
            
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
