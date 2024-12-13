import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from pathlib import Path
import numpy as np
from pkg_resources import parse_version
import cv2
from cri.transforms import quat2euler, euler2quat, inv_transform, transform, mat2euler, euler2mat
from camera.camera_realsense import RSCamera, ColorFrameError, DepthFrameError, DistanceError
from camera.detector import ArUcoDetector
from camera.tracker import ArUcoTracker, display_fn, NoMarkersDetected, MultipleMarkersDetected
from camera.utils import Namespace, transform_euler, inv_transform_euler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pathlib
# RS_RESOLUTION_COLOR = (1280, 720)
# RS_RESOLUTION = (640, 480)
color_size = (1280, 720)
depth_size = (848, 480)
FPS = 10.0
def make_realsense():
    return RSCamera(color_size=color_size, color_fps=30, depth_size=depth_size, depth_fps=30)

class RS_Cam():
    def __init__(self,):

            self.setup_realsense()


    def setup_realsense(self):


        # setup the realsense camera for capturing qunatitative data
        self.rs_camera = make_realsense()
        self.rs_detector = ArUcoDetector(self.rs_camera, marker_length=25.0, dict_id=cv2.aruco.DICT_7X7_50)
        self.rs_tracker = ArUcoTracker(self.rs_detector, track_attempts=30, num_marker_expected=2,display_fn=None)

        # convert extrinsic camera params to 4x4 homogeneous matrices
        self.rs = Namespace()

        m = Namespace()
        # Data dirs, load extrinsics_dir
        m.calib_dir = r'C:\Users\Lenovo\dev\cri\tactile_gym_sim2real_dev\tactile_gym_sim2real\online_experiments\bi_gather_env\realsense_params\dynamics\calib\MG400_calibration_latest'
        ext = Namespace()
        ext.load(os.path.join(m.calib_dir, "extrinsics.pkl"))
        m.ext_rvec = ext.rvec
        m.ext_tvec = ext.tvec
        m.ext_rmat, _ = cv2.Rodrigues(np.array(m.ext_rvec, dtype=np.float64))
        m.t_cam_base = np.hstack((m.ext_rmat, np.array(m.ext_tvec, dtype=np.float64).reshape((-1, 1))))
        m.t_cam_base = np.vstack((m.t_cam_base, np.array((0.0, 0.0, 0.0, 1.0)).reshape(1, -1)))
        m.t_base_cam = np.linalg.pinv(m.t_cam_base)
        m.image_to_arm = ext.image_to_arm
        m.arm_to_image = ext.arm_to_image
        self.rs.t_cam_base = m.t_cam_base
        self.rs.t_base_cam = m.t_base_cam
        self.rs.image_to_arm = ext.image_to_arm
        self.rs.arm_to_image = ext.arm_to_image
        # create a save dir
        self.rs_save_dir = os.path.join(pathlib.Path(__file__).parent.resolve(),
            self.collected_data_sub_folder,
            'rs_data'
        )
        os.makedirs(self.rs_save_dir, exist_ok=True)
        rs_video_file = os.path.join(self.rs_save_dir, 'rs_video.mp4')

        # Initialise tracking data
        [
            self.rs.work_align,
            self.rs.corners,
            self.rs.ids,
            self.rs.cam_poses,
            self.rs.base_poses,
            self.rs.centroids,
            self.rs.cam_centroids,
            self.rs.base_centroids,
            self.rs.base_centroids_acc,
            self.rs.base_centroids_acc_lin,
            self.rs.tcp_poses,
            self.rs.obj_rotations,
        ] = [], [], [], [], [], [], [], [], [], [], [], [], 





        # setup video writer
        self.rs_vid_out = cv2.VideoWriter(
            rs_video_file,
            cv2.VideoWriter_fourcc(*'mp4v'),
            FPS,
            color_size,
        )


    def get_realsense_data(self):
        try:
            self.rs_tracker.track()
        except (ColorFrameError, DepthFrameError, DistanceError, \
                NoMarkersDetected, MultipleMarkersDetected) as e:
                print(e)
                sys.exit('Issue with Realsense Tracking.')

        # grab data needed for object tracking
        if self.save_rs_data_flag:

            # compute marker centroid position and pose in base frame
            
            cam_centroid = self.rs_tracker.centroid_position # 1x3
            
            cam_pose = self.rs_tracker.pose # 4x4
            base_centroid = None
            if cam_centroid is not None:
                if len(self.rs_tracker.ids) > 1:
                    camera_point_h = np.vstack((cam_centroid.T, np.ones(len(self.rs_tracker.ids))))
                    # John's pnp method is less accurate in one dimension
                    base_centroids = np.dot(self.rs.t_base_cam, camera_point_h).squeeze()[:3].T
                    # more accurate extrinsic mat from the image_to_arm
                    base_centoid_accs = np.dot(self.rs.image_to_arm, camera_point_h).squeeze()[:3].T  # 2x3
                    base_centoid_accs_lin = []
                    for c_pos in cam_pose:
                        base_centoid_accs_lin.append(np.dot(self.rs.image_to_arm,  c_pos))  # 4x4
                    
                    # get the 1x6 euler vector from 1x3 position vec
                    centroid_baseframes = [(*base_centroid, 0,0,0) for base_centroid in base_centroids]
                    centroid_baseframe_accs = [(*base_centoid_acc, 0,0,0) for base_centoid_acc in base_centoid_accs]
                    # # get the 1x6 euler vector from 4x4 h mat
                    # set_trace()
                    centroid_baseframe_accs_lin = [mat2euler(base_centoid_acc_lin) for base_centoid_acc_lin in base_centoid_accs_lin]
                    for centroid_baseframe_acc_lin in centroid_baseframe_accs_lin:
                            centroid_baseframe_acc_lin[3:6] = [0,0,0]
                    # set_trace()
                    # transfer the marker poses to shared workframe, and only get the 1x3 position from quat
                    centroid_workframe = np.array([quat2euler(transform(euler2quat(centroid_baseframe, 'rxyz'), self.shared_work_frame_q), 'sxyz')[[0, 1, 2]] for centroid_baseframe in centroid_baseframes])
                    centroid_workframe_acc = np.array([quat2euler(transform(euler2quat(centroid_baseframe_acc, 'rxyz'), self.shared_work_frame_q), 'sxyz')[[0, 1, 2]] for  centroid_baseframe_acc in centroid_baseframe_accs])
                    self.objects_positions = centroid_workframe_acc
                    centroid_workframe_acc_lin = np.array([quat2euler(transform(euler2quat(centroid_baseframe_acc_lin, 'rxyz'), self.shared_work_frame_q), 'sxyz')[[0, 1, 2]] for  centroid_baseframe_acc_lin in centroid_baseframe_accs_lin])
                else:
                    camera_point_h = np.vstack((np.array(cam_centroid).reshape((-1, 1)), (1,)))
                    base_centroid = np.dot(self.rs.t_base_cam, camera_point_h).squeeze()[:3]
                    base_centoid_acc = np.dot(self.rs.image_to_arm, camera_point_h).squeeze()[:3]
                    centroid_baseframe = (*base_centroid, 0,0,0)
                    centroid_baseframe_acc = (*base_centoid_acc, 0,0,0)
                    centroid_workframe = quat2euler(transform(euler2quat(centroid_baseframe, 'rxyz'), self.shared_work_frame_q), 'sxyz')[[0, 1, 2]]
                    centroid_workframe_acc = quat2euler(transform(euler2quat(centroid_baseframe_acc, 'rxyz'), self.shared_work_frame_q), 'sxyz')[[0, 1, 2]]

            cam_pose = self.rs_tracker.pose
            base_pose = None
            if cam_pose is not None:
                if len(self.rs_tracker.ids) > 1:
                    # use the pose for getting marker orientation. The position of the pose is not accurate as it does not use depth image. But the orientation is good.
                    base_pose = np.array([mat2euler(np.dot(self.rs.t_base_cam, pose)) for pose in cam_pose])
                    work_pose = np.array([quat2euler(transform(euler2quat(pose, 'rxyz'), self.shared_work_frame_q), 'sxyz') for pose in base_pose])
                    work_Rz = work_pose[:, 5] 
                else:
                    base_pose = mat2euler(np.dot(self.rs.t_base_cam, cam_pose))
                    work_pose = quat2euler(transform(euler2quat(base_pose, 'rxyz'), self.shared_work_frame_q), 'sxyz')
                    work_Rz = work_pose[5]

            def check_pose(pose):
                if pose.ndim == 1:
                    pose = np.expand_dims(pose, axis=0)

                for i in range(len(pose)):
                    if abs(pose[i])[3] > 90 or abs(pose[i])[4] > 90:
                        return True
                return False

            # First step checking conditions
            if self.first_step:
                # Check if realsense is recording the first rotations correctly
                if check_pose(base_pose):
                    sys.exit('First realsense value was recorded incorrectly.')

                self.first_step = False
            else:
                pass

            # Capture ArUco tracking data
            self.rs.corners.append(self.rs_tracker.corners)
            self.rs.ids.append(self.rs_tracker.ids)
            self.rs.centroids.append(self.rs_tracker.centroid)
            self.rs.cam_centroids.append(cam_centroid)
            self.rs.base_centroids.append(centroid_workframe)
            self.rs.base_centroids_acc.append(centroid_workframe_acc)
            self.rs.base_centroids_acc_lin.append(centroid_workframe_acc_lin)
            self.rs.cam_poses.append(cam_pose)
            self.rs.base_poses.append(work_pose)
            self.rs.obj_rotations.append(work_Rz)
            # set_trace()
            tcp_position_swf = self._robot_0.get_tcp_pose_in_shared_workframe()
            tcp_position_swf_2 =self._robot_1.get_tcp_pose_in_shared_workframe()
            self.rs.tcp_poses.append(np.array([tcp_position_swf, tcp_position_swf_2]))
            # self.rs.tcp_pose_2.append(tcp_position_swf_2)

            # write video frame
            rs_rgb_frame = self.rs_camera.color_image
            self.rs_vid_out.write(rs_rgb_frame)