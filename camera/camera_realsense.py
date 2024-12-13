import numpy as np
import cv2

import pyrealsense2 as rs


class ColorFrameError(RuntimeError):
    pass

class DepthFrameError(RuntimeError):
    pass

class DistanceError(RuntimeError):
    pass


class RSCamera:
    def __init__(self, color_size=(640, 480), color_format=rs.format.bgr8, color_fps=15,
                 depth_size=(640, 480), depth_format=rs.format.z16, depth_fps=15,
                 decimate=False, spatial=False, hole_filling=False, align=True,
                 init_frames=30, device=0):
        self.decimate = decimate
        self.spatial = spatial
        self.hole_filling = hole_filling
        self.align = align
        self.init_frames = init_frames

        self.pipe = rs.pipeline()
        connect_device = []
        for d in rs.context().devices:
            print('Found device: ',
                  d.get_info(rs.camera_info.name), ' ',
                  d.get_info(rs.camera_info.serial_number))
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                connect_device.append(d.get_info(rs.camera_info.serial_number))


        self.config = rs.config()
        self.config.enable_stream(stream_type=rs.stream.color, width=color_size[0],
                                  height=color_size[1], format=color_format,
                                  framerate=color_fps)
        self.config.enable_stream(stream_type=rs.stream.depth, width=depth_size[0],
                                  height=depth_size[1], format=depth_format,
                                  framerate=depth_fps)
        self.config.enable_device(connect_device[device])
        self.profile = self.pipe.start(self.config)



        self.color_stream_profile = self.profile.get_stream(rs.stream.color)
        self.depth_stream_profile = self.profile.get_stream(rs.stream.depth)

        self.decimate_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        self.align = rs.align(align_to=rs.stream.color)

        self.colorizer = rs.colorizer()

        # Skip first few frames to give the Auto-Exposure time to adjust
        for _ in range(init_frames):
            self.pipe.wait_for_frames()

        self.frameset = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def color_size(self):
        profile = self.color_stream_profile.as_video_stream_profile()
        return profile.width(), profile.height()

    @property
    def color_format(self):
        return self.color_stream_profile.format()

    @property
    def color_fps(self):
        return self.color_stream_profile.fps()

    @property
    def color_intrinsics(self):
        return self.color_stream_profile.as_video_stream_profile().get_intrinsics()

    @property
    def color_cam_matrix(self):
        i = self.color_intrinsics
        return np.array([[i.fx, 0.0, i.ppx],
                         [0.0, i.fy, i.ppy],
                         [0.0, 0.0, 1.0]])

    @property
    def color_dist_coeffs(self):
        return np.array(self.color_intrinsics.coeffs)

    @property
    def depth_size(self):
        profile = self.depth_stream_profile.as_video_stream_profile()
        return profile.width(), profile.height()

    @property
    def depth_format(self):
        return self.depth_stream_profile.format()

    @property
    def depth_fps(self):
        return self.depth_stream_profile.fps()

    @property
    def depth_intrinsics(self):
        return self.depth_stream_profile.as_video_stream_profile().get_intrinsics()

    @property
    def depth_cam_matrix(self):
        i = self.depth_intrinsics
        return np.array([[i.fx, 0.0, i.ppx],
                         [0.0, i.fy, i.ppy],
                         [0.0, 0.0, 1.0]])

    @property
    def depth_dist_coeffs(self):
        return np.array(self.depth_intrinsics.coeffs)

    @property
    def depth_scale(self):
        depth_sensor = self.profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    @property
    def color_frame(self):
        color_frame = self.frameset.get_color_frame()
        if color_frame is None:
            raise ColorFrameError
        return color_frame

    @property
    def color_image(self):
        return np.asanyarray(self.color_frame.get_data())

    @property
    def depth_frame(self):
        depth_frame = self.frameset.get_depth_frame()
        if depth_frame is None:
            raise DepthFrameError
        return depth_frame

    @property
    def depth_image(self):
        return np.asanyarray(self.depth_frame.get_data())

    @property
    def colorized_depth_image(self):
        return np.asanyarray(self.colorizer.colorize(self.depth_frame).get_data())

    def get_distance(self, pixel):
        distance = self.depth_frame.get_distance(*pixel) * 1000
        if not distance:
            raise DistanceError
        return distance

    def deproject_pixel_to_point(self, pixel):
        point = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, list(pixel),
                                                self.get_distance(pixel))
        return np.array(point)

    def project_point_to_pixel(self, point):
        pixel = rs.rs2_project_point_to_pixel(self.color_intrinsics, list(point))
        return np.array(pixel)

    def get_decimation_filter_props(self):
        return self.decimate_filter.get_option(rs.option.filter_magnitude)

    def set_decimation_filter_props(self, filter_magnitude=2):
        self.decimate_filter.set_option(rs.option.filter_magnitude, filter_magnitude)

    def get_spatial_filter_props(self):
        return (self.spatial_filter.get_option(rs.option.filter_magnitude),
            self.spatial_filter.get_option(rs.option.filter_smooth_alpha),
            self.spatial_filter.get_option(rs.option.filter_smooth_delta),
            self.spatial_filter.get_option(rs.option.holes_fill))

    def set_spatial_filter_props(self, filter_magitude=2, smooth_alpha=0.5, smooth_delta=20, holes_fill=0):
        self.spatial_filter.set_option(rs.option.filter_magnitude, filter_magitude)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, smooth_alpha)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, smooth_delta)
        self.spatial_filter.set_option(rs.option.holes_fill, holes_fill)

    def get_hole_filling_props(self):
        return self.hole_filling_filter.set_option(rs.option.holes_fill)

    def set_hole_filling_props(self, hole_filling=1):
        self.hole_filling_filter.set_option(rs.option.holes_fill, hole_filling)

    def read(self):
        self.frameset = self.pipe.wait_for_frames()

        if self.decimate:
            self.frameset = self.decimate_filter.process(self.frameset).as_frameset()
        if self.spatial:
            self.frameset = self.spatial_filter.process(self.frameset).as_frameset()
        if self.hole_filling:
            self.frameset = self.hole_filling_filter.process(self.frameset).as_frameset()
        if self.align:
            self.frameset = self.align.process(self.frameset)

    def close(self):
        self.pipe.stop()



def main():
    try:
        with RSCamera(color_size=(1920, 1080), color_fps=30, depth_size=(848, 480), depth_fps=30,\
                      decimate=True, spatial=True, hole_filling=True, align=True) as camera:
            with np.printoptions(precision=2):
                print(f"camera.color_size: {camera.color_size}")
                print(f"camera.color_format: {camera.color_format}")
                print(f"camera.color_fps: {camera.color_fps}")
                print(f"camera.color_intrinsics: {camera.color_intrinsics}")
                print(f"camera.color_cam_matrix: {camera.color_cam_matrix}")
                print(f"camera.color_dist_coeffs: {camera.color_dist_coeffs}")

                print(f"camera.depth_size: {camera.depth_size}")
                print(f"camera.depth_format: {camera.depth_format}")
                print(f"camera.depth_fps: {camera.depth_fps}")
                print(f"camera.depth_intrinsics: {camera.depth_intrinsics}")
                print(f"camera.depth_cam_matrix: {camera.depth_cam_matrix}")
                print(f"camera.depth_dist_coeffs: {camera.depth_dist_coeffs}")

            camera.read()
            cv2.imshow("Color image", camera.color_image)
            cv2.imshow("Depth image", camera.colorized_depth_image)
            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
