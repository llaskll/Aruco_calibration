import numpy as np
import cv2
import cv2.aruco as aruco

from camera.camera_realsense import RSCamera, ColorFrameError, DepthFrameError, DistanceError
from camera.detector import ArUcoDetector


class NoMarkersDetected(RuntimeError):
    pass

class MultipleMarkersDetected(RuntimeError):
    pass


class ArUcoTracker:
    def __init__(self, detector, track_attempts=1, display_fn=None):
        self.detector = detector
        self.track_attempts = track_attempts
        self.display_fn = display_fn

        self.detector.both_poses = True

        self.pose_idx = None
        self.curr_rmat = None

    @property
    def corners(self):
        return self.detector.corners

    @property
    def ids(self):
        return self.detector.ids

    @property
    def rejected(self):
        return self.detector.rejected

    @property
    def centroid(self):
        if self.detector.centroids is None:
            return None
        return self.detector.centroids[0]

    @property
    def centroid_position(self):
        if self.detector.centroid_positions is None:
            return None
        return self.detector.centroid_positions[0]

    @property
    def rvec(self):
        if self.detector.rvecs is None:
            return None
        if self.pose_idx is None:
            return self.detector.rvecs[0]
        return self.detector.rvecs[0, self.pose_idx]

    @property
    def tvec(self):
        if self.detector.rvecs is None:
            return None
        if self.pose_idx is None:
            return self.detector.tvecs[0]
        return self.detector.tvecs[0, self.pose_idx]

    @property
    def err(self):
        if self.detector.errs is None:
            return None
        if self.pose_idx is None:
            return None
        return self.detector.errs[0, self.pose_idx]

    @property
    def rmat(self):
        if self.detector.rmats is None:
            return None
        if self.pose_idx is None:
            return self.detector.rmats[0]
        return self.detector.rmats[0, self.pose_idx]

    @property
    def pose(self):
        if self.detector.poses is None:
            return None
        if self.pose_idx is None:
            return self.detector.poses[0]
        return self.detector.poses[0, self.pose_idx]

    def reset(self):
        self.pose_idx = None
        self.curr_rmat = None

    def do_track(self):
        self.detector.detect()

        if self.detector.ids is None:
            raise NoMarkersDetected

        if len(self.detector.ids) > 1:
            raise MultipleMarkersDetected

        if self.pose_idx is None:
            # Initialise tracker with smallest error pose
            self.pose_idx = np.argmin(self.detector.errs[0])
        else:
            # Compute delta between current two candidate rotations and previous selected one
            rmat_deltas = [np.dot(self.curr_rmat, rmat.T) for rmat in self.detector.rmats[0]]
            rvec_deltas = [cv2.Rodrigues(rmat_delta)[0] for rmat_delta in rmat_deltas]
            rvec_delta_norms = np.linalg.norm(rvec_deltas, axis=1)
            # Choose pose with smallest rotational delta
            self.pose_idx = np.argmin(np.abs(rvec_delta_norms))

        # Save current rotation matrix for next tracking iteration
        self.curr_rmat = self.detector.rmats[0, self.pose_idx]

    def track(self):
        for i in range(self.track_attempts):
            try:
                self.do_track()
            except (ColorFrameError, DepthFrameError, DistanceError, \
                    NoMarkersDetected, MultipleMarkersDetected) as e:
                if self.display_fn is not None:
                    self.display_fn(self.draw_markers(self.detector.camera.color_image))
                if i < (self.track_attempts - 1):
                    continue
                raise e
            if self.display_fn is not None:
                self.display_fn(self.draw_markers(self.detector.camera.color_image))
            break

    def draw_markers(self, image, axis_length=20.0, border_color=(0, 0, 255)):
        image = image.copy()
        image = aruco.drawDetectedMarkers(image, self.corners, self.ids, border_color)
        if self.rvec is not None and self.tvec is not None:
            aruco.drawAxis(image, self.detector.camera.color_cam_matrix,
                           self.detector.camera.color_dist_coeffs,
                           self.rvec, self.tvec, axis_length)
        return image


def display_fn(img):
    cv2.imshow("Marker image", img)
    cv2.waitKey(1)


def main():
    try:
        with RSCamera(color_size=(1920, 1080), color_fps=30, depth_size=(848, 480), depth_fps=30) as camera:
            detector = ArUcoDetector(camera, marker_length=25.0, dict_id=cv2.aruco.DICT_7X7_50)
            tracker = ArUcoTracker(detector, track_attempts=30, display_fn=display_fn)
            while True:
                try:
                    tracker.track()
                except (ColorFrameError, DepthFrameError, DistanceError, \
                        NoMarkersDetected, MultipleMarkersDetected) as e:
                    cv2.imshow("Tracking error image", aruco.drawDetectedMarkers(camera.color_image,
                                                                                 detector.rejected,
                                                                                 borderColor=(100, 0, 240)))
                    break

                # cv2.imshow("Marker image", tracker.draw_markers(camera.color_image))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                with np.printoptions(precision=2, suppress=True):
                    print(f"detector.corners: {tracker.corners}, detector.ids: {tracker.ids}")
                    print(f"detector.rvecs: {tracker.rvec}, detector.tvecs: {tracker.tvec}")
                    print(f"detector.poses: {tracker.pose}")
                    print(f"detector.centroids: {tracker.centroid}")
                    print(f"detector.centroid_positions: {tracker.centroid_position}")

            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
