import cv2, numpy, time
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

CUBE_SIZE = 0.025  # 方块边长 25mm
cube_prompt = 'blue cube'
robot_ip = ''

# 颜色范围（从 checkpoint3 搬过来）
COLOR_RANGES = {
    'blue':   ((100, 80, 50), (130, 255, 255)),
    'red':    ((0, 80, 50), (10, 255, 255)),
    'red2':   ((170, 80, 50), (180, 255, 255)),
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
    if color == 'red' and 'red2' in COLOR_RANGES:
        lower2, upper2 = COLOR_RANGES['red2']
        mask2 = cv2.inRange(hsv_image, numpy.array(lower2), numpy.array(upper2))
        mask = cv2.bitwise_or(mask, mask2)
    kernel = numpy.ones((5, 5), numpy.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _order_corners(corners):
    """
    将四个角点排序为：左上、右上、右下、左下。
    这样和 solvePnP 的 3D 点顺序对应。
    """
    # 按 y 排序，前两个是顶部，后两个是底部
    sorted_by_y = corners[numpy.argsort(corners[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    # 顶部按 x 排序：左上、右上
    top = top[numpy.argsort(top[:, 0])]
    # 底部按 x 排序：左下、右下
    bottom = bottom[numpy.argsort(bottom[:, 0])]
    # 顺序：左上、右上、右下、左下
    return numpy.array([top[0], top[1], bottom[1], bottom[0]], dtype=numpy.float64)


def get_transform_cube(observation, camera_intrinsic, camera_pose, cube_prompt='blue cube'):
    """
    用颜色分割找到方块轮廓，用 minAreaRect 拿四个角点，
    再用 solvePnP 估计方块在相机坐标系的位姿。

    Parameters
    ----------
    observation : numpy.ndarray
        相机图像 (BGRA/BGR)。
    camera_intrinsic : numpy.ndarray
        3x3 相机内参矩阵。
    camera_pose : numpy.ndarray
        4x4 相机到机器人的变换矩阵 (t_cam_robot)。
    cube_prompt : str
        颜色提示，如 'blue cube'。

    Returns
    -------
    tuple or None
        (t_robot_cube, t_cam_cube) 均为 4x4 变换矩阵，单位米。
    """
    image = observation

    # --- 颜色分割 ---
    color = _parse_color(cube_prompt)
    if color is None:
        print(f"Could not parse color from prompt: '{cube_prompt}'")
        return None

    if len(image.shape) > 2 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) > 2:
        bgr = image
    else:
        print("Grayscale image cannot be used for color matching.")
        return None

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    color_mask = _get_color_mask(hsv, color)

    # 保存 mask 用于调试
    cv2.imwrite("color_mask_debug.png", color_mask)

    # --- 找轮廓 ---
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found for {color}.")
        return None

    # 过滤太小的轮廓（噪声），选最大的
    min_area = 100  # 像素
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        print(f"No valid contours for {color} (all too small).")
        return None

    # 选面积最大的轮廓
    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"Found {color} contour: area={area:.0f} pixels")

    # --- minAreaRect 获取旋转矩形的四个角点 ---
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)  # 4个角点, shape (4, 2)

    # 排序角点：左上、右上、右下、左下
    image_points = _order_corners(box_points)

    # 调试：在图像上画出检测到的矩形
    debug_img = bgr.copy()
    for i in range(4):
        pt1 = tuple(image_points[i].astype(int))
        pt2 = tuple(image_points[(i + 1) % 4].astype(int))
        cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)
        cv2.circle(debug_img, pt1, 5, (0, 0, 255), -1)
        cv2.putText(debug_img, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite("detected_rect_debug.png", debug_img)

    # --- solvePnP ---
    # 方块顶面的四个角点（3D，方块坐标系，z=0 是顶面）
    half = CUBE_SIZE / 2
    object_points = numpy.array([
        [-half,  half, 0],  # 左上
        [ half,  half, 0],  # 右上
        [ half, -half, 0],  # 右下
        [-half, -half, 0],  # 左下
    ], dtype=numpy.float64)

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

    # 构建 t_cam_cube
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = tvec.flatten()

    print(f"  Cube position in camera frame: {tvec.flatten()}")

    # 转到机器人坐标系
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    print(f"  Cube position in robot frame: {t_robot_cube[:3, 3]}")

    return t_robot_cube, t_cam_cube


def main():
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

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

        # Estimate cube pose
        result = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot, cube_prompt)
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

            # Grasp the cube
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
