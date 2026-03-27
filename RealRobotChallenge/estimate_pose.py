import cv2
import numpy as np
from utils.zed_camera import ZedCamera

# --- 配置参数 ---
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])
AXIS_LENGTH = 0.1 

# --- 假设这是你的手眼标定结果 (必须替换为你真实的标定矩阵) ---
# T_robot_cam 表示相机相对于机器人基座的 [R | t]
T_robot_cam = np.array([
    [0, -1,  0,  0.5],  # 旋转部分
    [-1, 0,  0,  0.0],
    [0,  0, -1,  0.4],
    [0,  0,  0,  1.0]   # 最后一行固定为 [0,0,0,1]
])

def get_cube_pose(cv_image, point_cloud):
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    points_xyz = point_cloud[:, :, :3]
    valid_mask = np.isfinite(points_xyz).all(axis=2) & (mask > 0)
    cube_points = points_xyz[valid_mask]

    if len(cube_points) < 50:
        return None

    # PCA 求解位姿
    center = np.mean(cube_points, axis=0)
    centered = cube_points - center
    cov = np.cov(centered.T)
    vals, vecs = np.linalg.eig(cov)
    R = vecs[:, np.argsort(vals)[::-1]]
    if np.linalg.det(R) < 0: R[:, 2] *= -1

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    return T

def main():
    zed = ZedCamera()
    K = zed.camera_intrinsic.astype(np.float64)

    while True:
        img = zed.image
        pcd = zed.point_cloud
        if img is None or pcd is None: continue

        vis_img = img.copy()
        
        # 1. 在相机系下识别 (T_cam_cube)
        T_cam_cube = get_cube_pose(img, pcd)

        if T_cam_cube is not None:
            # 2. 转换到机器人系 (T_robot_cube)
            T_robot_cube = np.matmul(T_robot_cam, T_cam_cube)

            # --- 打印结果 ---
            print("\n" + "="*30)
            print("【相机坐标系 T_cam_cube】:")
            print(np.round(T_cam_cube, 4))
            print("\n【机器人坐标系 T_robot_cube】:")
            print(np.round(T_robot_cube, 4))
            
            # --- 渲染 (仅在相机系下绘制轴) ---
            rvec, _ = cv2.Rodrigues(T_cam_cube[:3, :3])
            tvec = T_cam_cube[:3, 3]
            cv2.drawFrameAxes(vis_img, K, None, rvec, tvec, AXIS_LENGTH, 3)
            
            # 在图上显示机器人系下的 Z 坐标 (高度)
            z_robot = T_robot_cube[2, 3]
            cv2.putText(vis_img, f"Robot Z: {z_robot:.3f}m", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Result", vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
