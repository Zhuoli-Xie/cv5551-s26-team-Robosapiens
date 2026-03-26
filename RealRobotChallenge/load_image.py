import pyzed.sl as sl
import numpy as np
import cv2

def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return

    runtime = sl.RuntimeParameters()

    image = sl.Mat()
    depth = sl.Mat()

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        rgb = image.get_data()
        depth_map = depth.get_data()

        print("RGB shape:", rgb.shape)
        print("Depth shape:", depth_map.shape)

        cv2.imshow("RGB", rgb)

        # depth 需要可视化（否则你会觉得“啥都没有”）
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        cv2.imshow("Depth", depth_vis)

        cv2.waitKey(0)

    zed.close()

if __name__ == "__main__":
    main()
