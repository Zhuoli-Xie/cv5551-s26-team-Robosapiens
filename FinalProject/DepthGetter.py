import pyzed.sl as sl
import numpy as np

# Create ZED camera object
zed = sl.Camera()

# Set initialization parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # High-quality depth
init_params.coordinate_units = sl.UNIT.METER

# Open the camera
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open Error:", status)
    exit()

# Create containers
runtime_params = sl.RuntimeParameters()
depth = sl.Mat()

# Grab one frame
if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

    # Convert to numpy array
    depth_map = depth.get_data()

    print("Depth shape:", depth_map.shape)
    print("Min/Max depth:", np.nanmin(depth_map), np.nanmax(depth_map))

    # Save to .npy
    np.save("depth.npy", depth_map)

# Close camera
zed.close()
