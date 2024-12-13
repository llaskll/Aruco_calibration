import numpy as np
import cv2
import cv2.aruco as aruco

from camera.camera_realsense import RSCamera


def get_single_marker_object_points(marker_length):
    return np.array([[-0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, -0.5, 0.0]],
                    dtype=np.float32) * marker_length

def estimate_pose_single_markers(corners, marker_length, cam_matrix, dist_coeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE):
    if len(corners) < 1:
        return None, None, None

    points = get_single_marker_object_points(marker_length)

    # solve PnP object correspondences
    rvecs, tvecs, errs = [], [], []
    for corners_i in corners:
        _, rvecs_i, tvecs_i, errs_i = cv2.solvePnPGeneric(objectPoints=points,
                                                          imagePoints=corners_i,
                                                          cameraMatrix=cam_matrix,
                                                          distCoeffs=dist_coeffs,
                                                          flags=flags)
        rvecs.append(rvecs_i)
        tvecs.append(tvecs_i)
        errs.append(errs_i)

    rvecs = np.array(rvecs).squeeze(axis=-1)
    tvecs = np.array(tvecs).squeeze(axis=-1)
    errs = np.array(errs).squeeze(axis=-1)

    return rvecs, tvecs, errs


class ArUcoDetector:
    def __init__(self, camera, marker_length, dict_id=aruco.DICT_ARUCO_ORIGINAL, both_poses=False):
        self.camera = camera
        self.marker_length = marker_length
        self.dict_id = dict_id
        self.both_poses = both_poses

        self.aruco_dict = aruco.Dictionary_get(dict_id)
        self.aruco_params = aruco.DetectorParameters_create()

        self.corners = []
        self.ids = None
        self.rejected = []
        self.centroids = None
        self.centroid_positions = None
        self.rvecs = None
        self.tvecs = None
        self.errs = None
        self.rmats = None
        self.poses = None

    @property
    def adaptive_thresh_win_size_min(self):
        return self.aruco_params.adaptiveThreshWinSizeMin

    @adaptive_thresh_win_size_min.setter
    def adaptive_thresh_win_size_min(self, val):
        self.aruco_params.adaptiveThreshWinSizeMin = val

    @property
    def adaptive_thresh_win_size_max(self):
        return self.aruco_params.adaptiveThreshWinSizeMax

    @adaptive_thresh_win_size_max.setter
    def adaptive_thresh_win_size_max(self, val):
        self.aruco_params.adaptiveThreshWinSizeMax = val

    @property
    def adaptive_thresh_win_size_step(self):
        return self.aruco_params.adaptiveThreshWinSizeStep

    @adaptive_thresh_win_size_step.setter
    def adaptive_thresh_win_size_step(self, val):
        self.aruco_params.adaptiveThreshWinSizeStep = val

    @property
    def adaptive_thresh_constant(self):
        return self.aruco_params.adaptiveThreshConstant

    @adaptive_thresh_constant.setter
    def adaptive_thresh_constant(self, val):
        self.aruco_params.adaptiveThreshConstant = val

    @property
    def min_marker_perimeter_rate(self):
        return self.aruco_params.minMarkerPerimeterRate

    @min_marker_perimeter_rate.setter
    def min_marker_perimeter_rate(self, val):
        self.aruco_params.minMarkerPerimeterRate = val

    @property
    def max_marker_perimeter_rate(self):
        return self.aruco_params.maxMarkerPerimeterRate

    @max_marker_perimeter_rate.setter
    def max_marker_perimeter_rate(self, val):
        self.aruco_params.maxMarkerPerimeterRate = val

    @property
    def polygonal_approx_accuracy_rate(self):
        return self.aruco_params.polygonalApproxAccuracyRate

    @polygonal_approx_accuracy_rate.setter
    def polygonal_approx_accuracy_rate(self, val):
        self.aruco_params.polygonalApproxAccuracyRate = val

    @property
    def min_corner_distance_rate(self):
        return self.aruco_params.minCornerDistanceRate

    @min_corner_distance_rate.setter
    def min_corner_distance_rate(self, val):
        self.aruco_params.minCornerDistanceRate = val

    @property
    def min_marker_distance_rate(self):
        return self.aruco_params.minMarkerDistanceRate

    @min_marker_distance_rate.setter
    def min_marker_distance_rate(self, val):
        self.aruco_params.minMarkerDistanceRate = val

    @property
    def min_distance_to_border(self):
        return self.aruco_params.minDistanceToBorder

    @min_distance_to_border.setter
    def min_distance_to_border(self, val):
        self.aruco_params.minDistanceToBorder = val

    @property
    def marker_border_bits(self):
        return self.aruco_params.markerBorderBits

    @marker_border_bits.setter
    def marker_border_bits(self, val):
        self.aruco_params.markerBorderBits = val

    @property
    def min_otsu_std_dev(self):
        return self.aruco_params.minOtsuStdDev

    @min_otsu_std_dev.setter
    def min_otsu_std_dev(self, val):
        self.aruco_params.minOtsuStdDev = val

    @property
    def perspective_remove_pixel_per_cell(self):
        return self.aruco_params.perspectiveRemovePixelPerCell

    @perspective_remove_pixel_per_cell.setter
    def perspective_remove_pixel_per_cell(self, val):
        self.aruco_params.perspectiveRemovePixelPerCell = val

    @property
    def perspective_remove_ignored_margin_per_cell(self):
        return self.aruco_params.perspectiveRemoveIgnoredMarginPerCell

    @perspective_remove_ignored_margin_per_cell.setter
    def perspective_remove_ignored_margin_per_cell(self, val):
        self.aruco_params.perspectiveRemoveIgnoredMarginPerCell = val

    @property
    def max_erroneous_bits_in_border_rate(self):
        return self.aruco_params.maxErroneousBitsInBorderRate

    @max_erroneous_bits_in_border_rate.setter
    def max_erroneous_bits_in_border_rate(self, val):
        self.aruco_params.maxErroneousBitsInBorderRate =val

    @property
    def error_correction_rate(self):
        return self.aruco_params.errorCorrectionRate

    @error_correction_rate.setter
    def error_correction_rate(self, val):
        self.aruco_params.errorCorrectionRate = val

    @property
    def corner_refinement_method(self):
        return self.aruco_params.cornerRefinementMethod

    @corner_refinement_method.setter
    def corner_refinement_method(self, val):
        self.aruco_params.cornerRefinementMethod = val

    @property
    def corner_refinement_win_size(self):
        return self.aruco_params.cornerRefinementWinSize

    @corner_refinement_win_size.setter
    def corner_refinement_win_size(self, val):
        self.aruco_params.cornerRefinementWinSize = val

    @property
    def corner_refinement_max_iterations(self):
        return self.aruco_params.cornerRefinementMaxIterations

    @corner_refinement_max_iterations.setter
    def corner_refinement_max_iterations(self, val):
        self.aruco_params.cornerRefinementMaxIterations = val

    @property
    def corner_refinement_min_accuracy(self):
        return self.aruco_params.cornerRefinementMinAccuracy

    @corner_refinement_min_accuracy.setter
    def corner_refinement_min_accuracy(self, val):
        self.aruco_params.cornerRefinementMinAccuracy = val

    def detect(self):
        # Grab frameset
        self.camera.read()

        # Try to detect markers
        self.corners, self.ids, self.rejected = aruco.detectMarkers(self.camera.color_image,
                                                                    self.aruco_dict,
                                                                    parameters=self.aruco_params)

        # Initialize remaining detection state
        self.centroids = None
        self.centroid_positions = None
        self.rvecs = None
        self.tvecs = None
        self.errs = None
        self.rmats = None
        self.poses = None

        # If no markers found, we're done
        if self.ids is None:
            return

        # Estimate centroids and centroid positions in camera frame
        self.centroids = np.mean(np.array(self.corners), axis=2)
        self.centroid_positions = np.array([self.camera.deproject_pixel_to_point(c.squeeze()) for c in self.centroids])

        # Estimate marker poses
        if self.both_poses:
            # Estimate both marker poses (for ambiguous situations)
            self.rvecs, self.tvecs, self.errs = estimate_pose_single_markers(self.corners,
                                                                             self.marker_length,
                                                                             self.camera.color_cam_matrix,
                                                                             self.camera.color_dist_coeffs,
                                                                             flags=cv2.SOLVEPNP_IPPE)
        else:
            # Estimate most likely marker pose
            self.rvecs, self.tvecs, _ = aruco.estimatePoseSingleMarkers(self.corners,
                                                                        self.marker_length,
                                                                        self.camera.color_cam_matrix,
                                                                        self.camera.color_dist_coeffs)

        # Convert rotation and translation vectors into 3x3 rotation matrices and 4x4 homogeneous matrices
        rmats = []
        for rvec_i in self.rvecs:
            rmat_i = []
            for rvec_ij in rvec_i:
                rmat_ij, _ = cv2.Rodrigues(rvec_ij)
                rmat_i.append(rmat_ij)
            rmats.append(np.array(rmat_i))
        rmats = np.array(rmats)
        hmats = np.concatenate((rmats, np.expand_dims(self.tvecs, axis=-1)), axis=-1)
        bottom_row= np.array((0.0, 0.0, 0.0, 1.0)).reshape(1, -1)
        hmats = np.concatenate((hmats, np.tile(bottom_row, rmats.shape[:2] + (1, 1))), axis=2)
        self.rmats = rmats
        self.poses = hmats

    def draw_markers(self, image, axis_length=20.0, border_color=(0, 0, 255)):
        image = image.copy()
        image = aruco.drawDetectedMarkers(image, self.corners, self.ids, border_color)
        if self.rvecs is not None and self.tvecs is not None:
            if self.errs is not None:
                for rvec, tvec, err in zip(self.rvecs, self.tvecs, self.errs):
                    i = np.argmin(err)
                    aruco.drawAxis(image, self.camera.color_cam_matrix, self.camera.color_dist_coeffs,
                                   rvec[i], tvec[i], axis_length)
            else:
                for rvec, tvec in zip(self.rvecs, self.tvecs):
                    aruco.drawAxis(image, self.camera.color_cam_matrix, self.camera.color_dist_coeffs,
                                   rvec, tvec, axis_length)
        return image

def main():
    try:
        with RSCamera(color_size=(1920, 1080), color_fps=30, depth_size=(848, 480), depth_fps=30, \
                      decimate=True, spatial=True, hole_filling=True, align=True) as camera:
            detector = ArUcoDetector(camera, marker_length=25.0, dict_id=cv2.aruco.DICT_7X7_50, both_poses=True)
            detector.detect()
            with np.printoptions(precision=2, suppress=True):
                print(f"detector.corners: {detector.corners}, detector.ids: {detector.ids}")
                print(f"detector.rvecs: {detector.rvecs}, detector.tvecs: {detector.tvecs}")
                print(f"detector.poses: {detector.poses}")
                print(f"detector.centroids: {detector.centroids}")
                print(f"detector.centroid_positions: {detector.centroid_positions}")

            cv2.imshow("Marker image", detector.draw_markers(camera.color_image))
            while cv2.waitKey(1) & 0xFF != ord('q'):
                pass

    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

