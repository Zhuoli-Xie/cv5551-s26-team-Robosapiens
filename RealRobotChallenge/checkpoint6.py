import cv2
import numpy as np
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as Rot

# Attempt to import your specific utility modules
try:
    from utils.zed_camera import ZedCamera
    from utils.vis_utils import draw_pose_axes
except ImportError:
    print("Warning: Could not import utility modules. Ensure 'utils' folder is in your path.")

# --- Constants ---
CUBE_SIZE_TARGET = 0.0205  # Standard cube side length in meters
COLOR_RANGES = {
    'blue':   ((100, 80, 50), (130, 255, 255)),
    'red':    ((0, 80, 50), (10, 255, 255)),
    'green':  ((40, 80, 50), (80, 255, 255)),
}

def _get_color_mask(rgb_image, target_color='blue'):
    """
    Create a binary mask for the target color in HSV space with morphological denoising.
    """
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    lower, upper = COLOR_RANGES.get(target_color, COLOR_RANGES['blue'])
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def estimate_cube_pose(cv_image, point_cloud, t_cam_robot, target_color='blue'):
    """
    Estimate cube pose relative to the camera using color-masked point cloud data.
    
    Parameters:
        cv_image: (H, W, 3) BGR image from ZED
        point_cloud: (H, W, 4) XYZRGBA point cloud from ZED (units: meters)
        t_cam_robot: (4, 4) Transformation matrix: Camera in Robot Base frame
        target_color: Color string key for COLOR_RANGES
        
    Returns:
        t_cam_cube: (4, 4) Pose of the cube in Camera frame, or None if failed
    """
    
    # 1. Generate color mask
    color_mask = _get_color_mask(cv_image, target_color)
    flat_mask = color_mask.reshape(-1) > 0

    # 2. Extract XYZ points and filter by color mask
    # ZED point cloud index [:, :, :3] contains X, Y, Z in meters
    points_cam = point_cloud[:, :, :3].reshape(-1, 3)
    
    # Filter out invalid depth values (NaN/Inf) and apply color mask
    valid_mask = np.isfinite(points_cam).all(axis=1) & flat_mask
    points_masked = points_cam[valid_mask]

    if points_masked.shape[0] < 20:
        print(f"[{target_color}] Insufficient points in color mask: {points_masked.shape[0]}")
        return None

    # 3. Transform points to Robot Base frame for stable workspace filtering
    try:
        t_robot_cam = np.linalg.inv(t_cam_robot)
    except np.linalg.LinAlgError:
        print("Error: Hand-eye calibration matrix is singular.")
        return None
        
    points_robot = (t_robot_cam[:3, :3] @ points_masked.T + t_robot_cam[:3, 3:4]).T

    # --- Workspace Filtering (Ignore floor and objects outside reachable area) ---
    # Define bounds in meters relative to robot base
    workspace_mask = (
        (points_robot[:, 2] > 0.005) &   # Keep points > 5mm above table (removes floor)
        (points_robot[:, 2] < 0.150) &   # Max height 15cm
        (points_robot[:, 0] > 0.100) &   # X-range (Forward)
        (points_robot[:, 0] < 0.700) &
        (points_robot[:, 1] > -0.400) &  # Y-range (Left/Right)
        (points_robot[:, 1] < 0.400)
    )
    cube_points_robot = points_robot[workspace_mask]

    if cube_points_robot.shape[0] < 20:
        print("No points left after workspace filtering.")
        return None

    # 4. Open3D Processing (Clustering and Outlier Removal)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cube_points_robot)
    
    # Statistical Outlier Removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.0)
    
    # DBSCAN Clustering to find the largest object
    labels = np.array(pcd.cluster_dbscan(eps=0.015, min_points=10))
    if labels.size == 0 or labels.max() < 0:
        return None
    
    # Select the largest cluster (the cube)
    largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
    cube_pcd_robot = pcd.select_by_index(np.where(labels == largest_cluster_idx)[0])

    # 5. Pose Estimation via Oriented Bounding Box (OBB)
    obb = cube_pcd_robot.get_oriented_bounding_box()
    center_robot = obb.center
    R_robot = obb.R
    
    # --- Pose Normalization (Ensure Z-axis alignment for grasping) ---
    # 1. Ensure a right-handed coordinate system
    if np.linalg.det(R_robot) < 0:
        R_robot[:, 2] *= -1
    
    # 2. Align the closest OBB axis with the World Z-axis (Upward)
    z_dots = np.abs(R_robot[2, :])
    z_axis_col = np.argmax(z_dots)
    
    # Reorder columns to make z_axis_col the 3rd column (Z-axis)
    cols = [0, 1, 2]
    cols.remove(z_axis_col)
    new_R_robot = np.column_stack([R_robot[:, cols[0]], R_robot[:, cols[1]], R_robot[:, z_axis_col]])
    
    # Ensure Z-axis points Up (+Z)
    if new_R_robot[2, 2] < 0:
        new_R_robot[:, 2] *= -1
    
    # Re-verify right-handedness after swap
    if np.linalg.det(new_R_robot) < 0:
        new_R_robot[:, 1] *= -1

    # Construct 4x4 matrix in Robot Frame
    t_robot_cube = np.eye(4)
    t_robot_cube[:3, :3] = new_R_robot
    t_robot_cube[:3, 3] = center_robot

    # 6. Transform back to Camera Frame for output/visualization
    t_cam_cube = t_cam_robot @ t_robot_cube
    
    return t_cam_cube

def main():
    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic
    
    # Replace with your actual Hand-Eye Calibration matrix (T_cam_robot)
    # This matrix describes the camera's pose in the robot's base frame.
    # Placeholder: Assuming camera is 0.5m forward, 0.4m up, tilted down.
    t_cam_robot = np.eye(4)
    t_cam_robot[:3, :3] = Rot.from_euler('xyz', [-135, 0, 0], degrees=True).as_matrix()
    t_cam_robot[:3, 3] = [0.5, 0.0, 0.4]

    cv2.namedWindow("Cube Detection", cv2.WINDOW_NORMAL)
    print("Starting loop. Press 'q' to exit.")

    try:
        while True:
            # Capture RGB and Point Cloud
            img = zed.image
            pcd_data = zed.point_cloud

            if img is None or pcd_data is None:
                continue

            # Copy image for drawing
            vis_img = img.copy()

            # Estimate Pose
            t_cam_cube = estimate_cube_pose(img, pcd_data, t_cam_robot, target_color='blue')

            if t_cam_cube is not None:
                # Draw 3D Axes on the detected cube
                draw_pose_axes(vis_img, camera_intrinsic, t_cam_cube)
                
                pos = t_cam_cube[:3, 3]
                print(f"Cube detected at (Cam): x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            
            cv2.imshow("Cube Detection", vis_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
