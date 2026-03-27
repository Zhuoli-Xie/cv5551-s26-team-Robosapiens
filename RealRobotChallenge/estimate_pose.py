import cv2
import numpy as np
from utils.zed_camera import ZedCamera

# --- Constants ---
CUBE_SIZE = 0.0205 
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])

def get_cube_pose_with_debug(cv_image, point_cloud):
    # --- STAGE 1: COLOR MASKING ---
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 实时查看阶段 1 结果
    cv2.imshow("Stage_1_Mask", mask)
    
    if np.sum(mask) == 0:
        print("[Stage 1 FAILED]: No blue pixels detected. Check HSV ranges.")
        return None
    print("[Stage 1 OK]: Pixels detected.")

    # --- STAGE 2: POINT CLOUD EXTRACTION ---
    points_xyz = point_cloud[:, :, :3]
    valid_depth = np.isfinite(points_xyz).all(axis=2)
    combined_mask = (mask > 0) & valid_depth
    cube_points = points_xyz[combined_mask]

    if len(cube_points) < 50:
        print(f"[Stage 2 FAILED]: Only {len(cube_points)} points found. Too few for pose estimation.")
        return None
    print(f"[Stage 2 OK]: Found {len(cube_points)} valid points.")

    # --- STAGE 3: PCA POSE CALCULATION ---
    try:
        center = np.mean(cube_points, axis=0)
        centered_points = cube_points - center
        cov = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        sort_indices = np.argsort(eigenvalues)[::-1]
        R = eigenvectors[:, sort_indices]

        if np.linalg.det(R) < 0:
            R[:, 2] *= -1

        # 构建 4x4 矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center
        
        print("[Stage 3 OK]: 4x4 Pose Matrix generated.")
        return T
    except Exception as e:
        print(f"[Stage 3 FAILED]: Math error in PCA: {e}")
        return None

def main():
    zed = ZedCamera()
    K = zed.camera_intrinsic.astype(np.float64)

    while True:
        img = zed.image
        pcd_data = zed.point_cloud
        if img is None or pcd_data is None: continue

        vis_img = img.copy()
        T = get_cube_pose_with_debug(img, pcd_data)

        if T is not None:
            # --- STAGE 4: PRINTING & RENDERING ---
            print("\n--- FINAL POSE MATRIX ---")
            print(T)
            
            rvec, _ = cv2.Rodrigues(T[:3, :3])
            tvec = T[:3, 3]
            
            # 绘制坐标轴
            cv2.drawFrameAxes(vis_img, K, None, rvec, tvec, 0.05, thickness=3)
            print("[Stage 4 OK]: Axes drawn on image.")
        else:
            print("--- Detection Cycle Failed ---")

        cv2.imshow("Final_Result", vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
