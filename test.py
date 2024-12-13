import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
from pathlib import Path
import numpy as np
import pickle
from pkg_resources import parse_version
import cv2
from cri.transforms import quat2euler, euler2quat, inv_transform, transform, mat2euler, euler2mat
from camera.camera_realsense import RSCamera, ColorFrameError, DepthFrameError, DistanceError
from camera.detector import ArUcoDetector
from camera.tracker import ArUcoTracker, display_fn, NoMarkersDetected, MultipleMarkersDetected
from camera.utils import Namespace, transform_euler, inv_transform_euler
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pathlib
import matplotlib.pyplot as plt
# from tactile_gym_sim2real.online_experiments.c2fpush.mg400 import MG400_TacTip

# color_size = (1920, 1080)
depth_size = (1280, 720)
# RS_RESOLUTION = (640, 480)
RS_RESOLUTION = (1280, 720)
def make_realsense(device):
    return RSCamera(color_size=RS_RESOLUTION, color_fps=15, depth_size=depth_size, depth_fps=6, device=device)
rs_camera = make_realsense(0)
# camera = make_realsense(1)
rs_detector = ArUcoDetector(rs_camera, marker_length=32.0, dict_id=cv2.aruco.DICT_7X7_50)
# set_trace()
rs_tracker = ArUcoTracker(rs_detector, track_attempts=30, display_fn=None)




# def localize():
#     # rs_camera = make_realsense(1)
#     # # camera = make_realsense(1)
#     # rs_detector = ArUcoDetector(rs_camera, marker_length=32.0, dict_id=cv2.aruco.DICT_7X7_50)
#     # # set_trace()
#     # rs_tracker = ArUcoTracker(rs_detector, track_attempts=30, display_fn=None)
#
#     m = Namespace
#     m.calib_dir = r'D:\AITools\tactile_gym_sim2real_dev\tactile_gym_sim2real\online_experiments\c2fpush\realsense_Calibration\calib\MG400_calibration_latest'
#     ext = Namespace()
#     ext.load(os.path.join(m.calib_dir, "extrinsics.pkl"))
#     # john
#     m.ext_rvec = ext.rvec
#     m.ext_tvec = ext.tvec
#     m.ext_rmat, _ = cv2.Rodrigues(np.array(m.ext_rvec, dtype=np.float64))
#     m.t_cam_base = np.hstack((m.ext_rmat, np.array(m.ext_tvec, dtype=np.float64).reshape((-1, 1))))
#     m.t_cam_base = np.vstack((m.t_cam_base, np.array((0.0, 0.0, 0.0, 1.0)).reshape(1, -1)))
#     m.t_base_cam = np.linalg.pinv(m.t_cam_base)
#     # max
#     m.image_to_arm = ext.image_to_arm
#     m.arm_to_image = ext.arm_to_image
#
#     meta = Namespace()
#     meta.load(os.path.join(m.calib_dir, "meta.pkl"))
#     centroid_positions = meta.centroid_positions
#     poses = meta.cam_poses
#
#
#
#     # cv2.imshow('traj', rs_camera.color_image)
#     # cv2.waitKey(10)
#     base_poses = meta.base_poses
#     try:
#         rs_tracker.track()
#     except (ColorFrameError, DepthFrameError, DistanceError, \
#             NoMarkersDetected, MultipleMarkersDetected) as e:
#         print(e)
#         sys.exit('Issue with Realsense Tracking.')
#
#     # Display ArUco marker
#     marker_image = rs_detector.draw_markers(rs_camera.color_image)
#     # cv2.imwrite(str(m.image_dir/("image_" + str(n_detected) + ".jpg")), marker_image)
#     # cv2.imshow("Detected marker", marker_image)
#     # cv2.waitKey(1)
#     plt.imshow(marker_image)
#     # print(rs_tracker.detector.centroids)
#     # print(rs_detector.centroids)
#     # print(rs_tracker.detector.centroid_positions)
#     # print(mat2euler(rs_tracker.pose))
#     # print(mat2euler(np.dot(m.t_base_cam, rs_tracker.pose)))
#
#     work_frame = (240.0, -270, -142, -180, 0, -90)  # base frame: x->front, y->left, z->up, rz->anticlockwise
#     work_frame_q = euler2quat(work_frame, 'rxyz')
#     pose_baseframe = mat2euler(np.dot(m.t_base_cam, rs_tracker.pose))
#     pose_baseframe_acc = mat2euler(np.dot(m.image_to_arm, rs_tracker.pose))
#     # print("Base frame tvec is ", pose_baseframe[0:3])
#     pose_baseframe_acc[0] -= 39
#     pose_baseframe_acc[0] *= 1.3
#     pose_baseframe_acc[0] -= 5
#     pose_baseframe_acc[1] = (pose_baseframe_acc[1] + 15) * 1.1 + 10
#
#     print("pose_baseframe_acc: Base frame tvec is ", pose_baseframe_acc)
#
#
#     # print("Base frame rvec is ", pose_baseframe[3:6])
#     # pose_workframe = quat2euler(transform(euler2quat(pose_baseframe, 'rxyz'), work_frame_q), 'sxyz')
#     # print("Work frame tvec is ", pose_workframe[0:3])
#     # print("Work frame rvec is ", pose_workframe[3:6])
#     # Rz_workframe = pose_workframe[5] - sensor_offset
#     # print("Work frame Rz is ", Rz_workframe)    # plt.show()
#
#     return pose_baseframe_acc

def datacollection():


    m = Namespace()

    # Data dirs
    m.calib_dir = r'D:\AITools\tactile_gym_sim2real_dev\tactile_gym_sim2real\online_experiments\c2fpush\realsense_Calibration\calib\MG400_calibration_latest'
    # m.calib_dir = m.root_dir/("calib_" + time.strftime('%m%d%H%M'))
    m.image_dir = os.path.join(m.calib_dir, "images")
    os.makedirs(m.image_dir, exist_ok=True)

    base_frame = (0, 0, 0, 0, 0, 0)
    # work_frame = (0, 0, 0, 0, 0, 0)  # base frame: x->front, y->left, z->up, rz->anticlockwise
    # work_frame = (350.0, 0, -25, 0, 0, 0)  # base frame: x->front, y->left, z->up, rz->anticlockwise, for rotate, gather
    work_frame = (150.0, 0, 0, 0, 0, 0)  # base frame: x->front, y->left, z->up, rz->anticlockwise, for push

    # base_start_pose_in_wf = np.array([250, 0, 0, 0, 0, 0])
    base_start_pose_in_wf = np.array([0.0, 0, -20, 0, 0, 0])
    # base_home_pose_in_wf = (250, 0, 0, 0, 0, 0)
    base_home_pose_in_wf = (0.0, 0, -20, 0, 0, 0)

    # x_rng = [0, 220]
    x_rng = [-60, 60]
    y_rng = [-160, 160]
    # z_rng = [-50, -10]
    z_rng = [0, 30]
    alpha_rng = [0, 0]
    beta_rng = [0, 0]
    gamma_rng = [0, 0]

    n_samples = 20

    sensor_offset = 0

    m.base_poses, m.centroids, m.centroid_positions, m.rvecs, m.tvecs, m.cam_poses = [], [], [], [], [], []

    # cali_points = np.array([
    #     [20, -150, -20, 0, 0, 0],
    #     [20, -150, 50, 0, 0, 0],
    #     [140, -150, 50, 0, 0, 0],
    #     [140, -150, 20, 0, 0, 0],
    #     [20, 150, -20, 0, 0, 0],
    #     [20, 150, 50, 0, 0, 0],
    #     [140, 150, 50, 0, 0, 0],
    #     [140, 150, 20, 0, 0, 0],
    # ]) # tcp points to move to in the base frame

    robot = MG400_TacTip('TCP_velocity_control',
                             [300, 0, 0, 0, 0, 0],
                             45,
                             'right_angle',
                             )
    # Set TCP, linear speed,  angular speed and coordinate frame

    robot.robot.linear_speed = 30
    robot.robot.move_linear(base_home_pose_in_wf)



    n_poses, n_detected = 0, 0
    pose_idx = None
    while n_detected < n_samples:
        base_pose = base_start_pose_in_wf + np.random.uniform(
            low=(x_rng[0],
                 y_rng[0],
                 z_rng[0],
                 alpha_rng[0],
                 beta_rng[0],
                 gamma_rng[0]),
            high=(x_rng[1],
                  y_rng[1],
                  z_rng[1],
                  alpha_rng[1],
                  beta_rng[1],
                  gamma_rng[1]))
        robot_move = base_pose.copy()
        # robot_move[5] += sensor_offset
        robot.robot.move_linear(robot_move)

        # for point in cali_points:
        #     base_pose = base_start_pose_in_wf + point
        #     robot_move = base_pose.copy()
        #     robot_move[5] += sensor_offset
        #     robot.move_linear(robot_move)

        time.sleep(6)
        print("robot_move:", robot_move)
        # set_trace()

        n_poses += 1
        try:
            rs_tracker.track()
        except DistanceError:
            print(f"pose {n_poses}: marker distance error - moving to next pose")
            continue
        if rs_detector.ids is None:
            print(f"pose {n_poses}: marker not detected - moving to next pose")
            continue
        if len(rs_detector.ids) > 1:
            print(f"pose {n_poses}: multiple markers detected - moving to next pose")
            continue
        print(f"base pose: {base_pose}")
        print(f"pose {n_poses}: marker detected - moving to next pose")
        n_detected += 1

        # Display ArUco marker
        marker_image = rs_detector.draw_markers(rs_camera.color_image)
        cv2.imwrite(os.path.join(m.image_dir, ("image_" + str(n_poses) + ".jpg")), marker_image)
        cv2.imshow("Detected marker", marker_image)
        cv2.waitKey(1)

        m.base_poses.append(base_pose)
        m.centroids.append(rs_detector.centroids[0])
        m.centroid_positions.append(rs_detector.centroid_positions[0])
        m.cam_poses.append(rs_tracker.pose)
        m.rvecs.append(rs_detector.rvecs[0])
        m.tvecs.append(rs_detector.tvecs[0])

        print("center is ", m.centroid_positions[-1])
        # print("pose is ", m.cam_poses[-1])


        time.sleep(6)

        print('Next postion:')
    m.base_poses = np.array(m.base_poses)
    m.centroids = np.array(m.centroids)
    m.centroid_positions = np.array(m.centroid_positions)
    m.rvecs = np.array(m.rvecs)
    m.tvecs = np.array(m.tvecs)
    # Clean up
    m.save(os.path.join(m.calib_dir, "meta.pkl"))





    # Solve point correspondences for extrinsic camera params and save to file
    ret, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(m.base_poses[:, 0:3].astype('float32')), np.array(m.centroids), \
                                   rs_camera.color_cam_matrix, rs_camera.color_dist_coeffs)

    # Solve psuedo inverse method
    centroid_positions = np.column_stack(
        (m.centroid_positions, np.ones(m.centroid_positions.shape[0]).T)).T
    base_poses = np.column_stack(
        (m.base_poses[:, 0:3], np.ones(m.base_poses.shape[0]).T)).T
    image_to_arm = np.dot(base_poses, np.linalg.pinv(centroid_positions))
    arm_to_image = np.linalg.pinv(image_to_arm)

    with np.printoptions(precision=2, suppress=True):
        print(f"ret: {ret}, rvec: {rvec}, tvec: {tvec}")

    if ret:
        # Save extrinsic params to file
        ext = Namespace()
        # John
        ext.ret = ret
        ext.rvec = rvec
        ext.tvec = tvec
        # Max
        ext.image_to_arm = image_to_arm
        ext.arm_to_image = arm_to_image
        ext.save(os.path.join(m.calib_dir, "extrinsics.pkl"))
    else:
        print("Failed to solve point correspondence - extrinsic parameters not saved")



    # convert

    ext = Namespace()
    ext.load(os.path.join(m.calib_dir, "extrinsics.pkl"))
    # john
    m.ext_rvec = ext.rvec
    m.ext_tvec = ext.tvec
    m.ext_rmat, _ = cv2.Rodrigues(np.array(m.ext_rvec, dtype=np.float64))
    m.t_cam_base = np.hstack((m.ext_rmat, np.array(m.ext_tvec, dtype=np.float64).reshape((-1, 1))))
    m.t_cam_base = np.vstack((m.t_cam_base, np.array((0.0, 0.0, 0.0, 1.0)).reshape(1, -1)))
    m.t_base_cam = np.linalg.pinv(m.t_cam_base)
    # max
    m.image_to_arm = ext.image_to_arm
    m.arm_to_image = ext.arm_to_image

    meta = Namespace()
    meta.load(os.path.join(m.calib_dir, "meta.pkl"))
    centroid_positions = meta.centroid_positions
    poses = meta.cam_poses
    base_poses = meta.base_poses

    for ind, pt in enumerate(zip(centroid_positions, poses)):

        cam_centroid, pose = pt
        # print('cam centroid', cam_centroid)
        # print('cam pose', mat2euler(pose))

        base_centroid = None
        if cam_centroid is not None:
            camera_point_h = np.vstack((np.array(cam_centroid).reshape((-1, 1)), (1,)))
            base_centroid = np.dot(m.t_base_cam, camera_point_h).squeeze()[:3]
            base_centroid_1 = np.dot(m.image_to_arm, camera_point_h).squeeze()[:3]

        print("Translation: ")
        print("Expected: ", base_poses[ind][0:3])
        print('Result 1: ', base_centroid)
        print('Result 2: ', base_centroid_1)

        cam_pose = pose
        base_pose = None
        if cam_pose is not None:
            base_pose = np.dot(m.t_base_cam, cam_pose)
            base_pose_1 = np.dot(m.image_to_arm, cam_pose)

        print("Orientation: ")
        print("Expected: ", base_poses[ind][3:6])
        print("Result 1", mat2euler(base_pose)[3:6])
        print("Result 2", mat2euler(base_pose_1)[3:6])

        print()




if __name__ == '__main__':

    datacollection()
    while 1:
        localize()
        time.sleep(1)


