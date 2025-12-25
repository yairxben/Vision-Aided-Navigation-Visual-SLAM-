import gtsam
import math
from BundleAdjustment import BundleAdjustment
from PoseGraph import PoseGraph
from LinkLoop import LinkLoop
from BundleWindowInteractive import BundleWindowInteractive
from ex6 import get_relative_pose_and_cov_mat_last_kf, get_relative_pose_and_cov_mat_first_kf, plot_initial_estimate, plot_optimized_values
from tracking_database import TrackingDB
from BundleWindow import BundleWindow
from gtsam.utils import plot
from gtsam import symbol
import numpy as np
import matplotlib.pyplot as plt
from VertexGraph import VertexGraph
import pickle
from algorithms_library import (read_images_from_dataset, extract_keypoints_and_inliers,
                                calculate_right_camera_matrix, cv_triangulate_matched_points,
                                find_consensus_matches_indices, read_cameras_matrices,
                                extract_actual_consensus_pixels, plot_trajectories_and_landmarks_ex7 )
import cv2
from ex3 import q1 as ex3_q1
from ex3 import q2 as ex3_q2
from ex3 import q3 as ex3_q3
from ex3 import q4 as ex3_q4
from ex3 import q5_ex7 as ex3_q5

KEYS = [
    297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308,
    472, 473, 474,
    641, 642, 643, 644, 645, 646, 647, 648, 649,
    651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662,
    663, 664, 665,
    668, 669
]


THRESHOLD_INLIERS_ABS = 120

THRESHOLD_MAHALNUBIS = 6000

CANDIDATES_THRESHOLD = 5
THRESHOLD_INLIERS = 30
DETECTOR = cv2.SIFT_create()
CAMERA_SYM = "c"


def q2(kf, candidates):
    # manipulation to the index
    valid_candidates = []
    # Initial data-structure/s for saving the inputs for the tracks creation.
    # candidates_

    candidates_tuples_kp_inliers = []
    img0_left, img0_right = read_images_from_dataset(kf * 5)
    for cand in candidates:
        img1_left, img1_right = read_images_from_dataset(cand * 5)
        K, Ext0_left, Ext0_right = read_cameras_matrices()  # intrinsic & extrinsic camera Matrices
        R0_left, t0_left = Ext0_left[:, :3], Ext0_left[:, 3:]
        R0_right, t0_right = Ext0_right[:, :3], Ext0_right[:, 3:]

        # QUESTION 1
        q1_output = ex3_q1(K, R0_left, R0_right, img0_left, img0_right, img1_left, img1_right, t0_left, t0_right)
        (descriptors0_left, descriptors1_left, inliers_0_0, inliers_1_1, keypoints0_left, keypoints0_right,
         keypoints1_left,
         keypoints1_right, point_cloud_0) = q1_output
        # QUESTION 2
        tracking_matches = ex3_q2(descriptors0_left, descriptors1_left)
        # QUESTION 3
        q3_output = ex3_q3(Ext0_left, Ext0_right, K, R0_right, inliers_0_0, inliers_1_1, keypoints1_left, point_cloud_0,
                           t0_right, tracking_matches)
        R1_left, R1_right, consensus_match_indices_0_1, t1_left, t1_right = q3_output
        # QUESTION 4
        ex3_q4(K, R0_left, R0_right, R1_left, R1_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0,
               inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0,
               t0_left,
               t0_right, t1_left, t1_right, tracking_matches)
        # QUESTION 5:
        inliers_perc, abs_inliers, actual_pixels_inliers = ex3_q5(K, R0_left, R0_right,
                                                                           consensus_match_indices_0_1, img0_left,
                                                                           img1_left, inliers_0_0, inliers_1_1,
                                                                           keypoints0_left, keypoints0_right,
                                                                           keypoints1_left, keypoints1_right,
                                                                           point_cloud_0, t0_left, t0_right,
                                                                           tracking_matches, kf * 5, cand * 5)
        if inliers_perc > THRESHOLD_INLIERS and abs_inliers > THRESHOLD_INLIERS_ABS:
            valid_candidates.append(cand)
            candidates_tuples_kp_inliers.append((cand, actual_pixels_inliers))

    return valid_candidates, candidates_tuples_kp_inliers


def create_tracks_for_loops(
    key_frame_ind: int,
    candidates_tuples_kp_inliers,
    frame_stride: int = 5):
    """
    candidates_tuples_kp_inliers: iterable of (cand, actual_pixels_inliers)
      where actual_pixels_inliers = [
        [x_kf_left,  y_kf_left],
        [x_kf_right, y_kf_right],
        [x_prev_left,y_prev_left],
        [x_prev_right,y_prev_right],
      ]
      and each entry above is a 1-D array-like of length N (floats).
    """
    tracks = []

    for cand, api in candidates_tuples_kp_inliers:
        # Unpack the four views
        (x_kf_L,  y_kf_L)  = api[0]
        (x_kf_R,  y_kf_R)  = api[1]
        (x_prv_L, y_prv_L) = api[2]
        (x_prv_R, y_prv_R) = api[3]
        n = len(x_kf_L)
        cand_links = []
        for i in range(n):
            x_prev_l, y_prev_l = x_prv_L[i], y_prv_L[i]
            x_kf_l,  y_kf_l    = x_kf_L[i], y_kf_L[i]
            x_prev_r, y_prev_r = x_prv_R[i], y_prv_R[i]
            x_kf_r,  y_kf_r    = x_kf_R[i], y_kf_R[i]


            link = LinkLoop(
                link_loop_id = i,
                prev_frame_id=cand * frame_stride,
                key_frame_id=key_frame_ind * frame_stride,
                x_prev_frame_left=x_prev_l,  y_prev_frame_left=y_prev_l,
                x_key_frame_left=x_kf_l,     y_key_frame_left=y_kf_l,
                x_prev_frame_right=x_prev_r, y_prev_frame_right=y_prev_r,
                x_key_frame_right=x_kf_r,    y_key_frame_right=y_kf_r
            )
            cand_links.append(link)

        tracks.append((cand, cand_links))

    return tracks



def mahalanobis_dist(delta, cov):
    r_squared = delta.T @ np.linalg.inv(cov) @ delta
    return r_squared ** 0.5


def estimate_cov_matrix(path, cov_matrices):
    cov_mat = cov_matrices[path[0]]
    for pose in range(1, len(path)):
        cov_mat = cov_mat + cov_matrices[path[pose]]
    return cov_mat


def main(db, poseGraph_saved=False):
    print("Main-Start")
    symbol_c = "c"
    # use ex6.py methods to get the covariance between cosecutive cameras
    bundle = BundleAdjustment(db)
    key_frames_range = bundle.get_key_frames()
    bundle.load("bundle_data_window_size_20_witohut_bad_matches_ver2/",
                "bundle with window_size_20_witohut_bad_matches_ver2", 5)

    # gtsam_cams_bundle = bundle.get_gtsam_cams()

    relative_poses, cov_matrices = [], []

    # if not saved use that
    # for i, window in enumerate(bundle.get_windows()):
    #     if i % 100 == 0:
    #         print(f"try to get relative pose and cov mat for window {i}")
    #     relative_pose, cov_matrix = get_relative_pose_and_cov_mat_last_kf(window)
    #     relative_poses.append(relative_pose)
    #     cov_matrices.append(cov_matrix)
    # # Pickle both arrays into the specified file
    # with open("ex7_q1_relative_poses_cov_matrices.pkl", "wb") as f:
    #     pickle.dump((relative_poses, cov_matrices), f)
    # print("Pickled to ex7_q1_relative_poses_cov_matrices.pkl")
    # otherwise load
    with open("ex7_q1_relative_poses_cov_matrices.pkl", "rb") as f:
        relative_poses, cov_matrices = pickle.load(f)
    print("Loaded file - ex7_q1_relative_poses_cov_matrices.pkl")
    poseGraph = PoseGraph(db, bundle, relative_poses, cov_matrices, bundle.get_key_frames())
    poseGraph.create_factor_graph()

    kf_map_to_loops = []

    # Todo: once we found loop - optimize the locations according that immediately for improve the later - loops
    plot_initial_estimate(poseGraph, loop=True)
    for n in range(len(relative_poses)):  # n is kf num n -> frame n * 5
        candidates, vertex_graph = q1(n, poseGraph, cov_matrices, relative_poses, symbol_c)
        if len(candidates) > 0:
            # candidates_ind = [candidates[i][0] for i in range(len(candidates))]
            candidates_ind = [candidates[i] for i in range(len(candidates))]
            valid_candidates, candidates_tuples_kp_inliers = q2(n, candidates_ind)
            if len(valid_candidates) == 0:
                print(f"Window number {n} (frame {n // 5}): No valid candidates")

            else:
                kf_map_to_loops.append([n, valid_candidates])
                print(f"KF number {n} (frame {n // 5}): {valid_candidates}")
                # Create links and perform bundle to got measurments for the poseGraph.
                tuple_prevs_tracks = create_tracks_for_loops(n, candidates_tuples_kp_inliers)
                rel_poses, cov_mats= q3(n, tuple_prevs_tracks, poseGraph, vertex_graph)


    # plot_optimized_values(poseGraph, loop=True)
    #
    # plot_trajectories_and_landmarks_ex7(pose_graph=poseGraph, bundle_key_frames = key_frames_range,
    #                                      title="Trajectory_results_of_ex7")
        # adding the result as factor to the poseGraph (and optimize ? or optimize all together later)
        # Todo: Maybe create the poseGraph directly from here, i.e: during the loop-closure for getting more precisely locations for later key-frames
        # q4()
    print(f"key_frames_succe:\n{kf_map_to_loops}")


def q1(n, poseGraph, cov_matrices, relative_poses, symbol_c):
    values = poseGraph.get_initial_estimate()
    vertexGraph = VertexGraph(len(cov_matrices), cov_matrices)
    candidates = find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph, poseGraph)
    # iterate over candidates and check if we got consective-matches by ransac and SIFT etc..
    # todo: check wether the index of keframe should match to the index in the dataset of the images
    return candidates, vertexGraph


def q3(key_frame_ind, tuple_prevs_tracks, pose_graph, vertex_graph):





    rel_poses, cov_mats, bundles = [], [], []

    for prev_ind, tracks in tuple_prevs_tracks:
        # Performe Bundle-Window between the loop to our kf. -
        # TODO: Need to create factor graph, NOT LIKE THE USUAL FACTOR-GRAPH but implement new
        #  method and create new tracks for the loop<->kf, using db-methods, but not actual add to db
        #  than perform factor graph according to the results (in the posegraph).
        try:
            bundle = BundleWindow(db, prev_ind, key_frame_ind, False, tracks)
            # bundle = BundleWindow(db, key_frame_ind, prev_ind, False, tracks)
            bundle.create_factor_graph()
            bundle.optimize()

            rel_pose, cov_mat = get_relative_pose_and_cov_mat_first_kf(bundle)  # may throw
        except RuntimeError as e:
            if "Indeterminant linear system" in str(e):
                print(f"[WARN] Marginals ill-posed for loop ({prev_ind} -> {key_frame_ind}). Skipping.")
                # Optionally record for later re-try:
                # bad_pairs.append((prev_ind, key_frame_ind))
                continue
            else:
                raise  # unrelated error:
        rel_poses.append(rel_pose)
        cov_mats.append(cov_mat)
        bundles.append(bundle)
        # q4() - create factor add add to poseGraph
        prev_frame_sym = symbol(CAMERA_SYM, prev_ind)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(cov_mat)
        factor = gtsam.BetweenFactorPose3(prev_frame_sym, symbol(CAMERA_SYM, key_frame_ind), rel_pose, noise_model)
        pose_graph.add_factor(factor)
        vertex_graph.add_edge(prev_ind, key_frame_ind, cov_mat)
        # pose_graph.add_between_factor(rel_pose, cov_mat)


    pose_graph.optimize_poseGraph(loop=True)
    return rel_poses, cov_mats


def gtsam_cams_delta(first_cam_mat, second_cam_mat):
    gtsam_rel_trans = second_cam_mat.between(first_cam_mat)
    return gtsam_translation_to_vec(gtsam_rel_trans.rotation(), gtsam_rel_trans.translation())


def gtsam_translation_to_vec(R_mat, t_vec):
    np_R_mat = np.hstack((R_mat.column(1).reshape(3, 1), R_mat.column(2).reshape(3, 1), R_mat.column(3).reshape(3, 1)))
    euler_angles = rot_mat_to_euler_angles(np_R_mat)
    return np.hstack((euler_angles, t_vec))


def rot_mat_to_euler_angles(R_mat):
    sy = math.sqrt(R_mat[0, 0] * R_mat[0, 0] + R_mat[1, 0] * R_mat[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R_mat[2, 1], R_mat[2, 2])
        y = math.atan2(-R_mat[2, 0], sy)
        z = math.atan2(R_mat[1, 0], R_mat[0, 0])
    else:
        x = math.atan2(-R_mat[1, 2], R_mat[1, 1])
        y = math.atan2(-R_mat[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def find_candidates(cov_matrices, n, relative_poses, symbol_c, values, vertexGraph, poseGraph):
    candidates = []
    cur_cam_mat = values.atPose3(symbol(CAMERA_SYM, n))  # 'n' camera : cur_cam -> world

    for prev_cam_pose_graph_ind in range(
            max(n - 15, 0)):  # Run on the previous Key-Frames 0 <= i < n - 15 (0 <= frame_id -> (n-15)*5

        prev_cam_mat = values.atPose3(symbol(CAMERA_SYM, prev_cam_pose_graph_ind))  # 'i' camera : prev_cam -> world

        # Find the shortest path and estimate its relative covariance
        shortest_path = vertexGraph.find_shortest_path(prev_cam_pose_graph_ind, n)
        estimated_rel_cov = vertexGraph.estimate_rel_cov(shortest_path)

        # Compute Cams delta and their mahalanobis distance
        cams_delta = gtsam_cams_delta(prev_cam_mat, cur_cam_mat)
        dist = mahalanobis_dist(cams_delta, estimated_rel_cov)

        if dist < THRESHOLD_MAHALNUBIS:
            print(
                f"dist from  kf {prev_cam_pose_graph_ind}  to kf {n} (frame: {prev_cam_pose_graph_ind * 5} to frame {n * 5}) is:  {dist}")
            candidates.append([dist, prev_cam_pose_graph_ind])

    # if there are candidates, choose the best MAX_CAND_NUM numbers
    if len(candidates) > 0:
        # print(cur_cam_pose_graph_ind, candidates)

        sorted_candidates = sorted(candidates, key=lambda elem: elem[0])  # Sort candidates by mahalanobis dist
        # Take only the MAX_CAND_NUM candidate number and the index from the original list (without dist)
        candidates = np.array(sorted_candidates[:3]).astype(int)[:, 1]
        # reshape(min(len(sorted_candidates), 3), len(sorted_candidates[0]))

    return candidates

#
if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    main(db)
    # Trying to do interactive winodw,
    # Todo: check why the number of tracks not going low (or how to done that) According to the vehicle's directions i.e when going right we need reduction of tracks and than we finish the bundle window 
    # interactive_window = BundleWindowInteractive(db, first_frame_id=0, all_frames_between=True, little_bundle_tracks=None)
    # interactive_window.create_factor_graph()

