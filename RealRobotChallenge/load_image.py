import cv2
import numpy as np
import open3d as o3d
import os
import time
from utils.zed_camera import ZedCamera

def save_data(img, pcd_data, folder="data_captured"):
    """保存当前帧的图像和点云"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 保存彩色图像
    img_path = os.path.join(folder, f"color_{timestamp}.png")
    cv2.imwrite(img_path, img)
    
    # 2. 保存点云 (使用 Open3D 转换为标准 .ply 格式)
    # ZED point_cloud: (H, W, 4) -> [X, Y, Z, Color/None]
    points = pcd_data[:, :, :3].reshape(-1, 3)
    
    # 过滤掉无效点 (NaN 或 Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if len(points) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 如果你想保存带颜色的点云（可选）
        # colors = img.reshape(-1, 3)[valid_mask] / 255.0  # BGR to RGB 归一化
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1]) 
        
        pcd_path = os.path.join(folder, f"cloud_{timestamp}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"[已保存] 图像: {img_path}, 点云: {pcd_path}")
    else:
        print("[错误] 未能捕获到有效点云数据")

def main():
    # 初始化
    zed = ZedCamera()
    print("相机已就绪。")
    print("操作提示: [s] 保存当前帧 | [q] 退出程序")

    while True:
        img = zed.image
        pcd = zed.point_cloud
        
        if img is None or pcd is None:
            continue

        # 显示实时画面
        vis_img = img.copy()
        cv2.putText(vis_img, "Press 'S' to Save", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ZED Recorder", vis_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_data(img, pcd)
        elif key == ord('q'):
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
