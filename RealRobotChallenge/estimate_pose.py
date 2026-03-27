import cv2
import numpy as np
from utils.zed_camera import ZedCamera

# --- 1. 配置参数 ---
# 蓝色 HSV 范围 (根据你的环境微调)
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])
# 坐标轴显示长度 (米) - 增加到 0.1m 以便看得更清楚
AXIS_LENGTH = 0.1 

def get_cube_pose_full_logic(cv_image, point_cloud):
    """
    执行从颜色分割到 4x4 矩阵生成的全流程，包含调试信息。
    """
    # --- 阶段 1: 颜色检测 ---
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    
    # 去噪处理
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 实时查看 Mask 窗口
    cv2.imshow("Debug_Stage_1_Mask", mask)
    
    if np.sum(mask) == 0:
        return None, "Stage 1 Failed: No blue pixels."

    # --- 阶段 2: 提取并过滤点云 ---
    # point_cloud 是 (H, W, 4), 前三通道是 XYZ
    points_xyz = point_cloud[:, :, :3]
    valid_depth = np.isfinite(points_xyz).all(axis=2)
    combined_mask = (mask > 0) & valid_depth
    cube_points = points_xyz[combined_mask]

    if len(cube_points) < 50:
        return None, f"Stage 2 Failed: Only {len(cube_points)} points."

    # --- 阶段 3: PCA 几何计算 ---
    try:
        # 计算质心 (Translation)
        center = np.mean(cube_points, axis=0)
        
        # 计算协方差矩阵与特征向量 (Rotation)
        centered_points = cube_points - center
        cov = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # 按特征值排序确保主轴方向
        sort_indices = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, sort_indices]

        # 确保右手坐标系
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        # 组装 4x4 矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center
        
        return T, "Success"
    except Exception as e:
        return None, f"Stage 3 Failed: PCA Error {e}"

def main():
    # 初始化相机
    zed = ZedCamera()
    # 确保内参矩阵 K 为 float64 且为 3x3 格式
    K = zed.camera_intrinsic.astype(np.float64)
    
    print("--- 启动检测程序 ---")
    print(f"相机内参 K:\n{K}")
    print("按 'q' 键退出程序")

    while True:
        # 获取观测数据
        img = zed.image
        pcd_data = zed.point_cloud
        
        if img is None or pcd_data is None:
            continue

        vis_img = img.copy()
        
        # 执行位姿估算
        T_matrix, status = get_cube_pose_full_logic(img, pcd_data)

        if T_matrix is not None:
            # --- 打印 4x4 矩阵 ---
            print("\n[OK] 检测成功！4x4 Pose Matrix (T_cam_cube):")
            # 格式化打印，保留 4 位小数
            print(np.array2string(T_matrix, formatter={'float_kind':lambda x: "%.4f" % x}))
            
            # --- 渲染坐标轴 ---
            # 1. 转换旋转矩阵为旋转向量 (rvec)
            rvec, _ = cv2.Rodrigues(T_matrix[:3, :3])
            tvec = T_matrix[:3, 3]
            
            # 2. 使用 OpenCV 原生函数绘制轴
            cv2.drawFrameAxes(vis_img, K, None, rvec, tvec, AXIS_LENGTH, thickness=3)
            
            # --- 调试投影像素 (验证为什么“是一个点”) ---
            # 定义 3D 点：原点和 X 轴端点
            test_pts_3d = np.float32([[0,0,0], [AXIS_LENGTH,0,0]]).reshape(-1, 3)
            test_pts_2d, _ = cv2.projectPoints(test_pts_3d, rvec, tvec, K, None)
            u_v = test_pts_2d.reshape(-1, 2)
            
            pixel_diff = np.linalg.norm(u_v[0] - u_v[1])
            print(f"像素调试 -> 原点: {u_v[0].astype(int)}, 轴长在图像中占用: {pixel_diff:.1f} 像素")
            
            # 在图像上标注
            cv2.putText(vis_img, f"Dist: {np.linalg.norm(tvec):.2f}m", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 失败时显示状态原因
            cv2.putText(vis_img, status, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示最终结果
        cv2.imshow("Final Result (Press Q to quit)", vis_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
