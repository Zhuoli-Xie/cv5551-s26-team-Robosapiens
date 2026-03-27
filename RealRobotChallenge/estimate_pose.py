import open3d as o3d
import numpy as np
import cv2

def estimate_cube_pose(rgb_img, depth_img, intrinsic_matrix):
    """
    输入:
        rgb_img: (H, W, 3) np.uint8
        depth_img: (H, W) np.float32 (单位通常为米)
        intrinsic_matrix: 相机内参 3x3
    输出:
        pose: 4x4 变换矩阵 (T_camera_cube)
    """
    # 1. 将 RGB-D 转换为点云
    # 创建 Open3D 图像对象
    color = o3d.geometry.Image(rgb_img)
    depth = o3d.geometry.Image(depth_img)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)
    
    pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    pinhole_intrinsics.set_intrinsics(
        width=rgb_img.shape[1], height=rgb_img.shape[0],
        fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
        cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2]
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_intrinsics)
    
    # 2. 预处理：下采样和去噪
    pcd = pcd.voxel_down_sample(voxel_size=0.002) # 2mm 采样
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 3. 平面分割 (RANSAC) - 寻找并移除桌面
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    # 提取桌面以上的点
    pcd_objects = pcd.select_by_index(inliers, invert=True)

    # 4. 欧式聚类 - 找到立方体
    # eps 为聚类半径，min_points 为最少点数
    labels = np.array(pcd_objects.cluster_dbscan(eps=0.01, min_points=10))
    
    if len(labels) == 0:
        print("未发现物体")
        return None

    # 假设场景中最大的独立物体就是正方体（或根据尺寸 0.0205m 过滤）
    max_label = labels.max()
    cube_pcd = None
    for i in range(max_label + 1):
        cluster = pcd_objects.select_by_index(np.where(labels == i)[0])
        obb = cluster.get_oriented_bounding_box()
        # 验证边长是否接近 0.0205m (允许一定误差)
        extents = obb.extent
        if np.allclose(extents, 0.0205, atol=0.008): 
            cube_pcd = cluster
            break
            
    if cube_pcd is None:
        print("未匹配到符合尺寸的正方体")
        return None

    # 5. 提取位姿 (OBB)
    cube_obb = cube_pcd.get_oriented_bounding_box()
    cube_obb.color = (1, 0, 0) # 红色可视化
    
    # 构造 4x4 变换矩阵
    rotation = cube_obb.R
    translation = cube_obb.center
    
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation

    # 6. 可视化
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    # 物体局部坐标系
    cube_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
    cube_frame.transform(pose)
    
    o3d.visualization.draw_geometries([pcd, cube_obb, cube_frame, origin], 
                                      window_name="Cube Pose Estimation")
    
    return pose

# --- 使用示例 ---
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
# pose = estimate_cube_pose(rgb, depth, K)
# print("Cube Pose Matrix:\n", pose)
