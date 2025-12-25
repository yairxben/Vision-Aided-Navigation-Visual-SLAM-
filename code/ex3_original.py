import os
import random
import cv2
import numpy as np
from algorithms_library import plot_camera_positions, plot_root_ground_truth_and_estimate, \
    plot_supporters_non_supporters, read_cameras_matrices, extract_keypoints_and_inliers, cv_triangulate_matched_points, \
    find_consensus_matches_indices, calculate_front_camera_matrix, extract_actual_consensus_pixels, \
    find_supporter_indices_for_model, \
    estimate_projection_matrices_with_ransac, trying_estimate_projection_matrices_with_ransac_ex7, get_sucees_estimation_ex7, ex7_plot_supporters_non_supporters_after_ransac,\
    compute_trajectory_and_distance, plot_two_3D_point_clouds, read_images_from_dataset, \
    get_supporters_unsupporters_to_plot
import cProfile
import pstats
import io


DATASET_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')
DETECTOR = cv2.SIFT_create()
# DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
MATCHER = cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5),
                                searchParams=dict(checks=50))
NUM_FRAMES = 20
MAX_DEVIATION = 2
Epsilon = 1e-10


def main():
    print("start session:\n")
    img0_left, img0_right = read_images_from_dataset(0)
    img1_left, img1_right = read_images_from_dataset(1)
    K, Ext0_left, Ext0_right = read_cameras_matrices()  # intrinsic & extrinsic camera Matrices
    R0_left, t0_left = Ext0_left[:, :3], Ext0_left[:, 3:]
    R0_right, t0_right = Ext0_right[:, :3], Ext0_right[:, 3:]

    # QUESTION 1
    q1_output = q1(K, R0_left, R0_right, img0_left, img0_right, img1_left, img1_right, t0_left, t0_right)
    (descriptors0_left, descriptors1_left, inliers_0_0, inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left,
     keypoints1_right, point_cloud_0) = q1_output
    # QUESTION 2
    tracking_matches = q2(descriptors0_left, descriptors1_left)
    # QUESTION 3
    q3_output = q3(Ext0_left, Ext0_right, K, R0_right, inliers_0_0, inliers_1_1, keypoints1_left, point_cloud_0,
                   t0_right, tracking_matches)
    R1_left, R1_right, consensus_match_indices_0_1, t1_left, t1_right = q3_output
    # QUESTION 4
    q4(K, R0_left, R0_right, R1_left, R1_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0,
       inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left,
       t0_right, t1_left, t1_right, tracking_matches)
    # QUESTION 5:
    q5(K, R0_left, R0_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0, inliers_1_1,
       keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left, t0_right,
       tracking_matches)
    # Question 6:
    # NUM_FRAMES = 20  # total number of stereo-images in our KITTI dataset
    # q6(NUM_FRAMES)


def q6(NUM_FRAMES):
    estimated_trajectory, ground_truth_trajectory, distances = compute_trajectory_and_distance(num_frames=NUM_FRAMES,
                                                                                               verbose=True)
    plot_root_ground_truth_and_estimate(estimated_trajectory, ground_truth_trajectory)


def q5(K, R0_left, R0_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0, inliers_1_1,
       keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left, t0_right,
       tracking_matches):
    mR, mt, sup, supporter_indices = estimate_projection_matrices_with_ransac(point_cloud_0, consensus_match_indices_0_1,
                                                           inliers_0_0, inliers_1_1,
                                                           keypoints0_left, keypoints0_right,
                                                           keypoints1_left, keypoints1_right,
                                                           K, R0_left, t0_left, R0_right, t0_right,
                                                           verbose=True)
    plot_two_3D_point_clouds(mR, mt, point_cloud_0)
    # plot_inliers_outliers_ransac(consensus_match_indices_0_1, img0_left, img1_left, keypoints0_left, keypoints1_left,
    #                              sup, tracking_matches)
    non_supporting_pixels_back, non_supporting_pixels_front, supporting_pixels_back, supporting_pixels_front =\
        get_supporters_unsupporters_to_plot(
        consensus_match_indices_0_1, keypoints0_left, keypoints1_left, supporter_indices, tracking_matches)

    plot_supporters_non_supporters(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                   non_supporting_pixels_back, non_supporting_pixels_front,
                                   title="q5 - supporters and unsupporters after ransac - pnp")
    return len(supporter_indices) / len(consensus_match_indices_0_1) * 100


def q5_ex7(K, R0_left, R0_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0, inliers_1_1,
       keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left, t0_right,
       tracking_matches, key_frame, frame):
    ransac_success, cons_3d_points, actual_pixels, prev_supporters_indices, Rs, ts, start_time = trying_estimate_projection_matrices_with_ransac_ex7(point_cloud_0, consensus_match_indices_0_1,
                                                           inliers_0_0, inliers_1_1,
                                                           keypoints0_left, keypoints0_right,
                                                           keypoints1_left, keypoints1_right,
                                                           K, R0_left, t0_left, R0_right, t0_right,
                                                           key_frame, frame,
                                                           verbose=True)
    if ransac_success:

        mR, mt, sup, supporter_indices, actual_pixels = get_sucees_estimation_ex7(point_cloud_0, consensus_match_indices_0_1, inliers_0_0,
                                                           inliers_1_1,
                                                           keypoints0_left, keypoints0_right,
                                                           keypoints1_left, keypoints1_right,
                                                           K, R0_right, t0_right,
                                                           True, cons_3d_points, actual_pixels, prev_supporters_indices, Rs, ts, start_time)

        print("Ransac succes with ", len(supporter_indices) / len(consensus_match_indices_0_1) * 100, "Percentage")
        # plot_two_3D_point_clouds(mR, mt, point_cloud_0)
        # plot_inliers_outliers_ransac(consensus_match_indices_0_1, img0_left, img1_left, keypoints0_left, keypoints1_left,
        #                              sup, tracking_matches)
        non_supporting_pixels_back, non_supporting_pixels_front, supporting_pixels_back, supporting_pixels_front =\
            get_supporters_unsupporters_to_plot(
            consensus_match_indices_0_1, keypoints0_left, keypoints1_left, supporter_indices, tracking_matches)

        ex7_plot_supporters_non_supporters_after_ransac(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                       non_supporting_pixels_back, non_supporting_pixels_front,

                                       title=f"key_frame {key_frame} to frame {frame} - supporters and unsupporters after ransac ")
        return len(supporter_indices) / len(consensus_match_indices_0_1) * 100, len(supporter_indices), actual_pixels
    return 0, 0, 0



def q4(K, R0_left, R0_right, R1_left, R1_right, consensus_match_indices_0_1, img0_left, img1_left, inliers_0_0,
       inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0, t0_left,
       t0_right, t1_left, t1_right, tracking_matches):
    real_pixels = extract_actual_consensus_pixels(consensus_match_indices_0_1, inliers_0_0, inliers_1_1,
                                                  keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right)
    consensus_3d_points = point_cloud_0[[m[0] for m in consensus_match_indices_0_1]]
    Rs = [R0_left, R0_right, R1_left, R1_right]
    ts = [t0_left, t0_right, t1_left, t1_right]
    supporter_indices = find_supporter_indices_for_model(consensus_3d_points, real_pixels, K, Rs, ts)
    non_supporting_pixels_back, non_supporting_pixels_front, supporting_pixels_back, supporting_pixels_front = \
        get_supporters_unsupporters_to_plot(
            consensus_match_indices_0_1, keypoints0_left, keypoints1_left, supporter_indices, tracking_matches)

    # plot_supporters_non_supporters(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
    #                                non_supporting_pixels_back, non_supporting_pixels_front,
    #                                title="q4 - supporters and unsupporters NO_ransac")


def q3(Ext0_left, Ext0_right, K, R0_right, inliers_0_0, inliers_1_1, keypoints1_left, point_cloud_0, t0_right,
       tracking_matches):
    consensus_match_indices_0_1 = find_consensus_matches_indices(inliers_0_0, inliers_1_1, tracking_matches)
    is_success, R1_left, t1_left = calculate_front_camera_matrix(random.sample(consensus_match_indices_0_1, 4),
                                                                 point_cloud_0, inliers_1_1, keypoints1_left, K)

    Ext1_left = np.hstack((R1_left, t1_left))
    R1_right = np.dot(Ext1_left[:, :3], R0_right)
    t1_right = np.dot(Ext1_left[:, :3], t0_right) + t1_left
    Ext1_right = np.hstack((R1_right, t1_right.reshape(-1, 1)))
    # plot_camera_positions([Ext0_left, Ext0_right, Ext1_left, Ext1_right])
    return R1_left, R1_right, consensus_match_indices_0_1, t1_left, t1_right


def q2(descriptors0_left, descriptors1_left):
    # find matches in the first tracking pair (img0, img1) and sort for consensus-match
    tracking_matches = sorted(MATCHER.match(descriptors0_left, descriptors1_left), key=lambda m: m.queryIdx)
    return tracking_matches


def q1(K, R0_left, R0_right, img0_left, img0_right, img1_left, img1_right, t0_left, t0_right):
    # triangulate keypoints from stereo pair 0:
    preprocess_pair_0_0 = extract_keypoints_and_inliers(img0_left, img0_right)
    keypoints0_left, descriptors0_left, keypoints0_right, descriptors0_right, inliers_0_0, _ = preprocess_pair_0_0
    point_cloud_0 = cv_triangulate_matched_points(keypoints0_left, keypoints0_right, inliers_0_0,
                                                  K, R0_left, t0_left, R0_right, t0_right)
    # triangulate keypoints from stereo pair 1:
    preprocess_pair_1_1 = extract_keypoints_and_inliers(img1_left, img1_right)
    keypoints1_left, descriptors1_left, keypoints1_right, descriptors1_right, inliers_1_1, _ = preprocess_pair_1_1
    # triangulate pair_1 inliers based on projection matrices from pair_0
    point_cloud_1_with_camera_0 = cv_triangulate_matched_points(keypoints1_left, keypoints1_right, inliers_1_1,
                                                                K, R0_left, t0_left, R0_right, t0_right)
    return descriptors0_left, descriptors1_left, inliers_0_0, inliers_1_1, keypoints0_left, keypoints0_right, keypoints1_left, keypoints1_right, point_cloud_0


if __name__ == '__main__':
    ANALYZE = False
    if ANALYZE:
        # Profile the code
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()

        # Create a stream to hold the profile data
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
        stats.print_stats()

        # Print the profiling results
        print(stream.getvalue())
    else:
        main()
