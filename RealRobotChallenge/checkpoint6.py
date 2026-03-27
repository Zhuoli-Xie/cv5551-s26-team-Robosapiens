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


def get_transform_cube(observation, camera_intrinsic, camera_pose, cube_prompt='blue cube'):
    """
    用颜色分割找到方块轮廓，把轮廓上所有点映射到方块顶面 3D 坐标，
    再用 solvePnP 估计方块位姿。

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
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"No contours found for {color}.")
        return None

    # 过滤太小的轮廓（噪声），选最大的
    min_area = 100
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        print(f"No valid contours for {color} (all too small).")
        return None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"Found {color} contour: area={area:.0f} pixels, points={len(largest_contour)}")

    # --- 用 minAreaRect 建立局部坐标系 ---
    rect = cv2.minAreaRect(largest_contour)
    center_2d, (w_px, h_px), angle = rect

    if w_px < 1 or h_px < 1:
        print("minAreaRect too small.")
        return None

    # 确保 w_px >= h_px（让宽边一致）
    if w_px < h_px:
        w_px, h_px = h_px, w_px
        angle += 90

    # 调试：画出检测到的矩形
    debug_img = bgr.copy()
    box_pts = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(debug_img, [box_pts], 0, (0, 255, 0), 2)
    cv2.drawContours(debug_img, [largest_contour], 0, (255, 0, 0), 1)
    cv2.circle(debug_img, (int(center_2d[0]), int(center_2d[1])), 5, (0, 0, 255), -1)
    cv2.imwrite("detected_rect_debug.png", debug_img)

    # --- 把轮廓上所有点映射到方块顶面 3D 坐标 ---
    contour_points = largest_contour.reshape(-1, 2).astype(numpy.float64)

    # 旋转矩阵：图像坐标 → 矩形局部坐标
    angle_rad = numpy.deg2rad(angle)
    cos_a = numpy.cos(angle_rad)
    sin_a = numpy.sin(angle_rad)
    rot_mat = numpy.array([[cos_a, sin_a], [-sin_a, cos_a]])

    # 中心化 + 旋转到局部坐标
    centered = contour_points - numpy.array(center_2d)
    local_coords = (rot_mat @ centered.T).T  # (N, 2)

    # 归一化到方块实际尺寸
    half = CUBE_SIZE / 2
    local_coords[:, 0] = local_coords[:, 0] / w_px * CUBE_SIZE
    local_coords[:, 1] = local_coords[:, 1] / h_px * CUBE_SIZE

    # 限制在方块范围内（去掉边缘噪声点）
    within_bounds = (
        (numpy.abs(local_coords[:, 0]) <= half * 1.1) &
        (numpy.abs(local_coords[:, 1]) <= half * 1.1)
    )
    local_coords = local_coords[within_bounds]
    contour_points = contour_points[within_bounds]

    local_coords[:, 0] = numpy.clip(local_coords[:, 0], -half, half)
    local_coords[:, 1] = numpy.clip(local_coords[:, 1], -half, half)

    if len(local_coords) < 4:
        print("Too few points after filtering.")
        return None

    # 3D 点：轮廓对应方块顶面，z=0
    object_points = numpy.zeros((len(local_coords), 3), dtype=numpy.float64)
    object_points[:, 0] = local_coords[:, 0]
    object_points[:, 1] = local_coords[:, 1]
    # z = 0（方块顶面）

    image_points = contour_points.astype(numpy.float64)

    print(f"  Using {len(image_points)} points for solvePnP")

    # --- solvePnP ---
    # 先用 4 角点 IPPE 初始化，再用全部点 ITERATIVE 精化
    box_corners = cv2.boxPoints(rect).astype(numpy.float64)
    # 排序角点
    sorted_by_y = box_corners[numpy.argsort(box_corners[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    top = top[numpy.argsort(top[:, 0])]
    bottom = bottom[numpy.argsort(bottom[:, 0])]
    corners_ordered = numpy.array([top[0], top[1], bottom[1], bottom[0]], dtype=numpy.float64)

    corner_3d = numpy.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=numpy.float64)

    # 初始估计
    success, rvec_init, tvec_init = cv2.solvePnP(
        corner_3d, corners_ordered, camera_intrinsic, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )

    if not success:
        print("Initial solvePnP (IPPE) failed.")
        return None

    # 用全部轮廓点精化
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_intrinsic, None,
        rvec=rvec_init, tvec=tvec_init, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("Refined solvePnP (ITERATIVE) failed, using initial estimate.")
        rvec, tvec = rvec_init, tvec_init

    R, _ = cv2.Rodrigues(rvec)

    # 构建 t_cam_cube
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = tvec.flatten()

    print(f"  Cube in camera frame: {tvec.flatten()}")

    # 转到机器人坐标系
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    print(f"  Cube in robot frame:  {t_robot_cube[:3, 3]}")

    # --- 验证：重投影误差 ---
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_intrinsic, None)
    projected = projected.reshape(-1, 2)
    errors = numpy.linalg.norm(projected - image_points, axis=1)
    print(f"  Reprojection error: mean={errors.mean():.2f}px, max={errors.max():.2f}px")

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
