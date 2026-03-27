import cv2, numpy, time
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

cube_prompt = 'blue cube'
robot_ip = '192.168.1.183'

# Physical size of the cube in meters
CUBE_SIZE = 0.045

COLOR_RANGES = {
    'blue':   ((100, 80, 50),  (130, 255, 255)),
    'red':    ((0,   80, 50),  (10,  255, 255)),
    'red2':   ((170, 80, 50),  (180, 255, 255)),
    'green':  ((40,  80, 50),  (80,  255, 255)),
    'yellow': ((20,  80, 50),  (40,  255, 255)),
    'orange': ((10,  80, 50),  (20,  255, 255)),
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
    if color == 'red' and 'red2' in COLOR_RANGES:
        lower2, upper2 = COLOR_RANGES['red2']
        mask2 = cv2.inRange(hsv_image, numpy.array(lower2), numpy.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_transform_cube(observation, camera_intrinsic, camera_pose, cube_prompt, table_z_robot=0.0):
    """
    Estimate the cube's pose using color segmentation alone (no AprilTag, no depth).

    Strategy
    --------
    1. Segment the cube by color to get a 2D blob in the image.
    2. Fit a rotated bounding rectangle to the blob to get four corner pixels.
    3. Assume the cube sits flat on the table (known Z in robot frame).
    4. Use solvePnP with the known cube face size to recover the full 6-DOF pose
       in the camera frame, then transform to robot frame.

    Parameters
    ----------
    observation : numpy.ndarray
        BGR or BGRA image from the camera.
    camera_intrinsic : numpy.ndarray
        3x3 camera intrinsic matrix.
    camera_pose : numpy.ndarray
        4x4 transform T_cam_robot (camera expressed in robot base frame).
    cube_prompt : str
        Text description of the target cube color, e.g. 'blue cube'.
    table_z_robot : float
        Height of the table surface in the robot base frame (meters).
        The cube's bottom face sits at this Z.  Default 0.0.

    Returns
    -------
    tuple (t_robot_cube, t_cam_cube) or None
        Both are 4x4 numpy arrays with translations in meters.
        t_robot_cube has the cube origin at the top-face center
        (so the gripper descends to the cube top).
        Returns None if detection fails.
    """
    # ------------------------------------------------------------------ #
    # 1. Colour segmentation
    # ------------------------------------------------------------------ #
    color = _parse_color(cube_prompt)
    if color is None:
        print(f"Cannot parse color from prompt: '{cube_prompt}'")
        return None

    if len(observation.shape) > 2 and observation.shape[2] == 4:
        bgr = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
    elif len(observation.shape) == 3:
        bgr = observation.copy()
    else:
        print("Need a colour image for colour-based detection.")
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = _get_color_mask(hsv, color)

    # ------------------------------------------------------------------ #
    # 2. Find the largest contour (the cube face)
    # ------------------------------------------------------------------ #
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No '{color}' blob found in image.")
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:          # sanity check – too small
        print("Detected blob is too small; skipping.")
        return None

    # Rotated rectangle → four corner pixels (image coordinates)
    rect = cv2.minAreaRect(largest)             # (center, (w,h), angle)
    box_pts = cv2.boxPoints(rect)               # shape (4,2), float32

    # Order corners: top-left, top-right, bottom-right, bottom-left
    # (so they correspond to a consistent 3-D model below)
    def order_corners(pts):
        pts = pts[numpy.argsort(pts[:, 1])]     # sort by y (top first)
        top = pts[:2][numpy.argsort(pts[:2, 0])]
        bot = pts[2:][numpy.argsort(pts[2:, 0])]
        return numpy.array([top[0], top[1], bot[1], bot[0]], dtype=numpy.float64)

    image_points = order_corners(box_pts)       # (4,2)

    # ------------------------------------------------------------------ #
    # 3. 3-D model of the visible cube face
    #
    #    We assume the camera sees the TOP face of the cube and that face
    #    is axis-aligned in the cube's local frame:
    #
    #       TL(-h, -h, 0)   TR(+h, -h, 0)
    #       BL(-h, +h, 0)   BR(+h, +h, 0)
    #
    #    (z=0 is the top face; origin at face centre)
    # ------------------------------------------------------------------ #
    h = CUBE_SIZE / 2.0
    object_points = numpy.array([
        [-h, -h, 0],    # top-left
        [ h, -h, 0],    # top-right
        [ h,  h, 0],    # bottom-right
        [-h,  h, 0],    # bottom-left
    ], dtype=numpy.float64)

    # ------------------------------------------------------------------ #
    # 4. PnP → pose of cube top-face in camera frame
    # ------------------------------------------------------------------ #
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_intrinsic,
        None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        print("solvePnP failed.")
        return None

    R, _ = cv2.Rodrigues(rvec)
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3]  = tvec.flatten()

    # ------------------------------------------------------------------ #
    # 5. Transform to robot base frame
    # ------------------------------------------------------------------ #
    T_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = T_robot_cam @ t_cam_cube

    return t_robot_cube, t_cam_cube


def main():

    # ------------------------------------------------------------------ #
    # Initialise hardware
    # ------------------------------------------------------------------ #
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:
        # ---------------------------------------------------------------- #
        # Capture image and calibrate camera → robot transform
        # ---------------------------------------------------------------- #
        cv_image = zed.image

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print("Could not compute camera-robot transform. Aborting.")
            return

        # ---------------------------------------------------------------- #
        # Detect cube pose using colour vision only
        # ---------------------------------------------------------------- #
        result = get_transform_cube(
            cv_image,
            camera_intrinsic,
            t_cam_robot,
            cube_prompt,
            table_z_robot=0.0      # adjust if your table is not at z=0
        )
        if result is None:
            print("Cube not detected. Aborting.")
            return

        t_robot_cube, t_cam_cube = result

        print("Cube pose in robot frame (metres):")
        print(t_robot_cube)

        # ---------------------------------------------------------------- #
        # Visualise
        # ---------------------------------------------------------------- #
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # -------------------------------------------------------------- #
            # Grasp then place the cube back
            # -------------------------------------------------------------- #
            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)

    finally:
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()
