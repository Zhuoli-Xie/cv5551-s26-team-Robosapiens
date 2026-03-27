import cv2
import numpy as np
import open3d as o3d

# --- Constants ---
CUBE_SIZE = 0.0205 
COLOR_RANGES = {
    'blue': ((100, 100, 50), (130, 255, 255)), # Adjusted for better sensitivity
}

def _get_color_mask(rgb_image, target_color='blue'):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower, upper = COLOR_RANGES.get(target_color, COLOR_RANGES['blue'])
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def estimate_cube_pose_camera_frame(cv_image, point_cloud, target_color='blue'):
    """
    Directly estimates the cube pose in the Camera Coordinate System.
    """
    # 1. Get the color mask
    mask = _get_color_mask(cv_image, target_color)
    
    # DEBUG: Show the mask to ensure the cube is actually being "seen"
    # cv2.imshow("Debug Mask", mask)

    # 2. Extract points from the Point Cloud using the mask
    # ZED point_cloud is (H, W, 4) -> [X, Y, Z, Color]
    points_xyz = point_cloud[:, :, :3].reshape(-1, 3)
    flat_mask = mask.reshape(-1) > 0
    
    # Filter points: must be in mask AND must be valid numbers (not NaN)
    valid_indices = np.isfinite(points_xyz).all(axis=1) & flat_mask
    cube_points = points_xyz[valid_indices]

    print(f"Points found for {target_color}: {cube_points.shape[0]}")

    if cube_points.shape[0] < 50:
        return None

    # 3. Use Open3D for geometric fitting
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cube_points)
    
    # Remove statistical noise (stray pixels)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    
    # Cluster the points to ensure we only pick the actual cube, not background noise
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=10))
    if labels.size == 0 or labels.max() < 0:
        return None
    
    # Pick the largest cluster
    largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
    final_cube_pcd = pcd.select_by_index(np.where(labels == largest_cluster_idx)[0])

    # 4. Compute the Oriented Bounding Box (OBB)
    obb = final_cube_pcd.get_oriented_bounding_box()
    
    # Build the 4x4 Transformation Matrix T_cam_cube
    t_cam_cube = np.eye(4)
    t_cam_cube[:3, :3] = obb.R
    t_cam_cube[:3, 3] = obb.center

    # 5. Fix Orientation (Optional but recommended for consistent axes)
    # Ensure the local Z-axis of the OBB points roughly towards the camera
    if t_cam_cube[2, 3] > 0: # Cube center is in front of camera
        # This is a simple right-hand check
        if np.linalg.det(t_cam_cube[:3, :3]) < 0:
            t_cam_cube[:3, 2] *= -1
            
    return t_cam_cube

def main():
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    while True:
        img = zed.image
        pcd_data = zed.point_cloud
        
        if img is None or pcd_data is None:
            continue

        vis_img = img.copy()
        
        # Calculate pose ONLY relative to the camera
        t_cam_cube = estimate_cube_pose_camera_frame(img, pcd_data, 'blue')

        if t_cam_cube is not None:
            # Draw the axes
            # Ensure your draw_pose_axes function uses cv2.projectPoints internally
            draw_pose_axes(vis_img, camera_intrinsic, t_cam_cube)
        else:
            cv2.putText(vis_img, "NOT DETECTED", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Camera Frame Pose", vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()
    cv2.destroyAllWindows()
