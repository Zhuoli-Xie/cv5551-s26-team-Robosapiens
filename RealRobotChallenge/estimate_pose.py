import cv2
import numpy as np
from utils.zed_camera import ZedCamera

# --- Constants ---
CUBE_SIZE = 0.0205 
# Define Blue range (Adjust these if your cube is not detected)
BLUE_LOWER = np.array([100, 100, 50])
BLUE_UPPER = np.array([130, 255, 255])

def get_cube_pose_opencv(cv_image, point_cloud, intrinsic_matrix):
    """
    Estimate cube pose using only OpenCV and Numpy.
    """
    # 1. Color Segmentation
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    
    # Clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 2. Extract 3D Points from ZED Point Cloud
    # point_cloud shape is (H, W, 4) -> [X, Y, Z, RGBA]
    points_xyz = point_cloud[:, :, :3]
    
    # Get coordinates where mask is white and depth is valid
    valid_depth = np.isfinite(points_xyz).all(axis=2)
    combined_mask = (mask > 0) & valid_depth
    
    cube_points = points_xyz[combined_mask]

    if len(cube_points) < 50:
        return None

    # 3. PCA (Principal Component Analysis) to find Pose
    # Mean of points is the Center (Translation)
    center = np.mean(cube_points, axis=0)
    
    # Center the data
    centered_points = cube_points - center
    
    # Calculate Covariance Matrix
    cov = np.cov(centered_points.T)
    
    # Eigenvectors are the principal axes (Rotation)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort eigenvectors by eigenvalues (descending)
    sort_indices = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, sort_indices]

    # Ensure Right-Handed Coordinate System
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    # 4. Prepare for OpenCV Drawing
    # We need rvec (rotation vector) and tvec (translation vector)
    rvec, _ = cv2.Rodrigues(R)
    tvec = center.reshape(3, 1)

    return rvec, tvec

def main():
    zed = ZedCamera()
    # Ensure intrinsic is float64 for OpenCV functions
    K = zed.camera_intrinsic.astype(np.float64)

    print("Running... Press 'q' to quit.")

    while True:
        img = zed.image
        pcd_data = zed.point_cloud
        
        if img is None or pcd_data is None:
            continue

        vis_img = img.copy()
        
        # Estimate Pose
        result = get_cube_pose_opencv(img, pcd_data, K)

        if result is not None:
            rvec, tvec = result
            
            # --- OpenCV Built-in Drawing ---
            # Arguments: image, camera_matrix, dist_coeffs, rvec, tvec, axis_length
            # axis_length set to 0.03m (3cm)
            cv2.drawFrameAxes(vis_img, K, None, rvec, tvec, 0.03)
            
            # Optional: Print distance
            dist = np.linalg.norm(tvec)
            cv2.putText(vis_img, f"Dist: {dist:.3f}m", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(vis_img, "Cube Not Found", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("OpenCV Only Pose Estimation", vis_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
