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


def _order_box_corners(box_points):
    """
    用 sum/diff 方法稳定排序 4 个角点为：左上、右上、右下、左下。
    比按 y 排序更鲁棒，不会因为旋转角度导致顺序错乱。
    """
    s = box_points.sum(axis=1)         # x + y
    d = numpy.diff(box_points, axis=1).flatten()  # y - x

    tl = box_points[numpy.argmin(s)]   # 左上：x+y 最小
    br = box_points[numpy.argmax(s)]   # 右下：x+y 最大
    tr = box_points[numpy.argmin(d)]   # 右上：y-x 最小
    bl = box_points[numpy.argmax(d)]   # 左下：y-x 最大

    return numpy.array([tl, tr, br, bl], dtype=numpy.float64)


def get_transform_cube(observation, camera_intrinsic, camera_pose, cube_prompt='blue cube'):
    """
    用颜色分割找到方块轮廓，用稳定的角点排序 + solvePnP 估计位姿，
    再用全部轮廓点精化。

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
    cv2.imwrite("color_mask_debug.png", color_mask)

    # --- 找轮廓 ---
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print(f"No contours found for {color}.")
        return None

    min_area = 100
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not valid_contours:
        print(f"No valid contours for {color} (all too small).")
        return None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    print(f"Found {color} contour: area={area:.0f}px, points={len(largest_contour)}")

    # --- minAreaRect + 稳定角点排序 ---
    rect = cv2.minAreaRect(largest_contour)
    center_2d, (w_px, h_px), angle = rect
    print(f"  minAreaRect: center={center_2d}, size=({w_px:.1f}, {h_px:.1f}), angle={angle:.1f}")

    if w_px < 1 or h_px < 1:
        print("minAreaRect too small.")
        return None

    box = cv2.boxPoints(rect)
    corners_2d = _order_box_corners(box)  # 左上、右上、右下、左下

    # 调试：画角点和编号
    debug_img = bgr.copy()
    cv2.drawContours(debug_img, [largest_contour], 0, (255, 0, 0), 1)
    cv2.drawContours(debug_img, [box.astype(int)], 0, (0, 255, 0), 2)
    labels = ['TL', 'TR', 'BR', 'BL']
    for i, (pt, label) in enumerate(zip(corners_2d, labels)):
        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
        cv2.putText(debug_img, label, (int(pt[0]) + 8, int(pt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imwrite("detected_corners_debug.png", debug_img)

    # --- 第一步：4 角点 solvePnP (IPPE_SQUARE) ---
    half = CUBE_SIZE / 2
    # 3D 角点对应顺序：左上、右上、右下、左下（方块顶面，z=0）
    corners_3d = numpy.array([
        [-half,  half, 0],   # TL
        [ half,  half, 0],   # TR
        [ half, -half, 0],   # BR
        [-half, -half, 0],   # BL
    ], dtype=numpy.float64)

    success, rvec_init, tvec_init = cv2.solvePnP(
        corners_3d, corners_2d, camera_intrinsic, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not success:
        print("Initial solvePnP (IPPE_SQUARE) failed.")
        return None

    # 验证初始估计：z 应该为正（方块在相机前方）
    if tvec_init[2, 0] < 0:
        print(f"  Warning: initial tvec z={tvec_init[2, 0]:.4f} is negative, flipping.")
        tvec_init = -tvec_init

    # 初始重投影误差
    proj_init, _ = cv2.projectPoints(corners_3d, rvec_init, tvec_init, camera_intrinsic, None)
    err_init = numpy.linalg.norm(proj_init.reshape(-1, 2) - corners_2d, axis=1)
    print(f"  Initial PnP reprojection error: mean={err_init.mean():.2f}px, max={err_init.max():.2f}px")

    # --- 第二步：用全部轮廓点精化 ---
    contour_points = largest_contour.reshape(-1, 2).astype(numpy.float64)

    # 用初始 rvec/tvec 反投影，把 2D 轮廓点映射到方块顶面 z=0 平面
    R_init, _ = cv2.Rodrigues(rvec_init)
    t_init = tvec_init.flatten()
    K_inv = numpy.linalg.inv(camera_intrinsic)

    object_points_list = []
    image_points_list = []

    for pt_2d in contour_points:
        # 像素点 → 归一化相机坐标
        p_homo = numpy.array([pt_2d[0], pt_2d[1], 1.0])
        ray_cam = K_inv @ p_homo  # 相机坐标系下的射线方向

        # 射线与方块顶面 z=0 平面求交
        # 方块坐标系下：P = R^T @ (lambda * ray_cam - t)
        # 令 P_z = 0，解 lambda
        ray_obj = R_init.T @ ray_cam
        t_obj = R_init.T @ t_init

        if abs(ray_obj[2]) < 1e-8:
            continue  # 射线几乎平行于平面，跳过

        lam = t_obj[2] / ray_obj[2]
        if lam < 0:
            continue  # 在相机后方，跳过

        p_obj = lam * ray_obj - t_obj
        p_obj[2] = 0.0  # 强制在 z=0 平面上

        # 过滤超出方块范围的点（留 20% 余量）
        if abs(p_obj[0]) > half * 1.2 or abs(p_obj[1]) > half * 1.2:
            continue

        object_points_list.append(p_obj)
        image_points_list.append(pt_2d)

    if len(object_points_list) < 4:
        print(f"  Too few points for refinement ({len(object_points_list)}), using initial estimate.")
        rvec, tvec = rvec_init, tvec_init
    else:
        obj_pts = numpy.array(object_points_list, dtype=numpy.float64)
        img_pts = numpy.array(image_points_list, dtype=numpy.float64)

        print(f"  Refining with {len(obj_pts)} contour points")

        success_ref, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_intrinsic, None,
            rvec=rvec_init.copy(), tvec=tvec_init.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success_ref:
            print("  Refined solvePnP failed, using initial estimate.")
            rvec, tvec = rvec_init, tvec_init
        else:
            # 精化后重投影误差
            proj_ref, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_intrinsic, None)
            err_ref = numpy.linalg.norm(proj_ref.reshape(-1, 2) - img_pts, axis=1)
            print(f"  Refined reprojection error: mean={err_ref.mean():.2f}px, max={err_ref.max():.2f}px")

    R_final, _ = cv2.Rodrigues(rvec)

    # 构建 t_cam_cube
    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_final
    t_cam_cube[:3, 3] = tvec.flatten()

    print(f"  Cube in camera frame: {tvec.flatten()}")

    # 转到机器人坐标系
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    print(f"  Cube in robot frame:  {t_robot_cube[:3, 3]}")

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
