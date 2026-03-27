import cv2, numpy, time
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
    用颜色 mask 找到方块区域，从 ZED 点云直接读取 3D 坐标，
    用 minAreaRect 获取 2D 旋转角作为抓取朝向。

    流程：
    1. 颜色 HSV 分割 → mask
    2. connectedComponentsWithStats → 找最大连通域，拿质心像素
    3. 从点云直接读该区域的 3D 坐标 → 方块中心位置
    4. minAreaRect → 方块在图像中的旋转角 → 转成抓取旋转矩阵

    Parameters
    ----------
    observation : list or tuple
        [image, point_cloud]，image 是 BGRA/BGR，point_cloud 是 ZED 的 (H, W, 4)。
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
    image, point_cloud = observation

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

    # --- 连通域分析，找最大的方块区域 ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        color_mask, connectivity=8
    )

    if num_labels <= 1:
        print(f"No connected components found for {color}.")
        return None

    # 跳过 label=0（背景），找面积最大的连通域
    # stats 列：[x, y, w, h, area]
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = numpy.argmax(areas) + 1  # +1 因为跳过了背景

    comp_area = stats[largest_idx, cv2.CC_STAT_AREA]
    comp_x = stats[largest_idx, cv2.CC_STAT_LEFT]
    comp_y = stats[largest_idx, cv2.CC_STAT_TOP]
    comp_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
    comp_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
    centroid_x, centroid_y = centroids[largest_idx]

    print(f"Found {color} component: area={comp_area}px, "
          f"bbox=({comp_x},{comp_y},{comp_w},{comp_h}), "
          f"centroid=({centroid_x:.1f},{centroid_y:.1f})")

    if comp_area < 50:
        print("Component too small, likely noise.")
        return None

    # --- 从点云直接读取 3D 坐标 ---
    # 取该连通域内所有像素的点云值
    comp_mask = (labels == largest_idx)
    ys, xs = numpy.where(comp_mask)

    # 从点云读 XYZ（点云 shape: H, W, 4，前 3 通道是 XYZ）
    points_3d = point_cloud[ys, xs, :3]  # (N, 3)

    # 过滤 NaN/Inf
    valid = numpy.isfinite(points_3d).all(axis=1)
    points_3d = points_3d[valid]

    print(f"  Valid 3D points from depth: {points_3d.shape[0]}")

    if points_3d.shape[0] < 5:
        print("  Too few valid depth points.")
        return None

    # 去离群点：用中位数 ± 3倍 MAD
    median_pt = numpy.median(points_3d, axis=0)
    dists = numpy.linalg.norm(points_3d - median_pt, axis=1)
    mad = numpy.median(dists)
    if mad > 0:
        inlier_mask = dists < (3.0 * mad + 1e-6)
        points_3d = points_3d[inlier_mask]

    if points_3d.shape[0] < 5:
        print("  Too few points after outlier removal.")
        return None

    # 方块中心 = 所有点的均值（相机坐标系）
    center_cam = points_3d.mean(axis=0)
    print(f"  Cube center in camera frame: [{center_cam[0]:.4f}, {center_cam[1]:.4f}, {center_cam[2]:.4f}]")

    # --- 用 minAreaRect 获取 2D 旋转角 ---
    comp_contours, _ = cv2.findContours(
        comp_mask.astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if comp_contours:
        rect = cv2.minAreaRect(comp_contours[0])
        _, _, angle_deg = rect
        angle_rad = numpy.deg2rad(angle_deg)
    else:
        angle_rad = 0.0

    # --- 构建变换矩阵 ---
    # 相机坐标系下：z 朝前，但我们不知道方块的精确朝向
    # 用一个简单的旋转：z 轴沿相机 z 方向（朝前），xy 按 minAreaRect 角度旋转
    # 这在转到机器人坐标系后会变成 z 朝上（假设相机大致朝下看桌面）
    cos_a = numpy.cos(angle_rad)
    sin_a = numpy.sin(angle_rad)
    R_cam = numpy.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R_cam
    t_cam_cube[:3, 3] = center_cam

    # 转到机器人坐标系
    t_robot_cam = numpy.linalg.inv(camera_pose)
    t_robot_cube = t_robot_cam @ t_cam_cube

    # 强制 z 轴朝上（机器人坐标系下方块就是放在桌面上的）
    rotation = t_robot_cube[:3, :3]
    # 找旋转矩阵中最接近世界 z 的列，换到第三列
    z_dots = numpy.abs(rotation[2, :])
    z_idx = numpy.argmax(z_dots)
    cols = [0, 1, 2]
    cols.remove(z_idx)
    new_R = numpy.column_stack([rotation[:, cols[0]], rotation[:, cols[1]], rotation[:, z_idx]])
    if new_R[2, 2] < 0:
        new_R[:, 2] *= -1
    if numpy.linalg.det(new_R) < 0:
        new_R[:, 1] *= -1
    t_robot_cube[:3, :3] = new_R

    print(f"  Cube in robot frame: [{t_robot_cube[0,3]:.4f}, {t_robot_cube[1,3]:.4f}, {t_robot_cube[2,3]:.4f}]")

    # 调试：在图像上画检测结果
    debug_img = bgr.copy()
    cv2.circle(debug_img, (int(centroid_x), int(centroid_y)), 8, (0, 0, 255), -1)
    cv2.rectangle(debug_img, (comp_x, comp_y), (comp_x + comp_w, comp_y + comp_h), (0, 255, 0), 2)
    cv2.putText(debug_img, f"{color} cube", (comp_x, comp_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("detected_cube_debug.png", debug_img)

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
        point_cloud = zed.point_cloud

        # Get camera-to-robot transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return

        # Estimate cube pose
        result = get_transform_cube(
            [cv_image, point_cloud], camera_intrinsic, t_cam_robot, cube_prompt
        )
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
