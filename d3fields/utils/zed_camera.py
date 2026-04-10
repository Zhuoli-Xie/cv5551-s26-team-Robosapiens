"""Thin wrapper around the ZED SDK (pyzed) for easy grab-and-retrieve."""

import numpy as np
import pyzed.sl as sl


class ZedCamera:
    """Open a ZED camera by serial number or index and grab BGRA + depth frames."""

    def __init__(self, serial_number=None, camera_id=0, resolution=sl.RESOLUTION.HD720, fps=30):
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.METER

        if serial_number is not None:
            init_params.set_from_serial_number(serial_number)
        else:
            init_params.set_from_camera_id(camera_id)

        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {status}")

        self._sl_image = sl.Mat()
        self._sl_depth = sl.Mat()

        # Expose intrinsics
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.fx = calib.left_cam.fx
        self.fy = calib.left_cam.fy
        self.cx = calib.left_cam.cx
        self.cy = calib.left_cam.cy
        self.camera_intrinsic = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.array(calib.left_cam.disto, dtype=np.float64)

    def grab(self):
        """Grab a new frame. Returns True on success."""
        runtime = sl.RuntimeParameters()
        return self.zed.grab(runtime) == sl.ERROR_CODE.SUCCESS

    @property
    def image(self):
        """Grab and return the left image as a BGR numpy array (H, W, 3)."""
        self.grab()
        self.zed.retrieve_image(self._sl_image, sl.VIEW.LEFT)
        bgra = self._sl_image.get_data()
        return bgra[:, :, :3].copy()

    @property
    def depth(self):
        """Grab and return depth map as a float32 numpy array (H, W) in metres."""
        self.grab()
        self.zed.retrieve_measure(self._sl_depth, sl.MEASURE.DEPTH)
        return self._sl_depth.get_data().copy()

    def close(self):
        if hasattr(self, 'zed'):
            self.zed.close()

    def __del__(self):
        self.close()
