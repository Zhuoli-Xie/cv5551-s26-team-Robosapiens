import pyzed.sl as sl
import numpy as np

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # more stable
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 0.3
init_params.depth_maximum_distance = 5.0

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("Camera Open Error:", status)
    exit()

runtime_params = sl.RuntimeParameters()
runtime_params.confidence_threshold = 50

depth = sl.Mat()

if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    depth_map = depth.get_data()

    print("Depth shape:", depth_map.shape)
    print("Min/Max depth:", np.nanmin(depth_map), np.nanmax(depth_map))
    print("NaNs:", np.isnan(depth_map).sum())

    np.save("depth.npy", depth_map)

zed.close()


import numpy as np

depth_map = np.load("depth.npy")
print(depth_map.shape, depth_map.dtype)

import matplotlib.pyplot as plt

plt.imshow(depth_map, cmap="plasma")  # or "viridis", "inferno"
plt.colorbar(label="Depth")
plt.title("Depth Map")
plt.show()
