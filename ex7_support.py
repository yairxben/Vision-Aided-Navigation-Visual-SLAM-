"""Temporary legacy Exercise 7 compatibility helpers pending Exercise 7 validation."""

import time

from matplotlib import pyplot as plt

from tools import (
    build_model,
    calculate_number_of_iteration_for_ransac,
    extract_actual_consensus_pixels,
    find_supporter_indices_for_model,
    get_supporters_unsupporters_to_plot,
)


def ex7_plot_supporters_non_supporters_after_ransac(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                   non_supporting_pixels_back, non_supporting_pixels_front, title):
    """
    Plots keypoints classified as supporters and non-supporters in two images.

    Args:
    - img0_left (np.array): Left image 0.
    - img1_left (np.array): Left image 1.
    - supporting_pixels_back (list): Supporting keypoints in the back image.
    - supporting_pixels_front (list): Supporting keypoints in the front image.
    - non_supporting_pixels_back (list): Non-supporting keypoints in the back image.
    - non_supporting_pixels_front (list): Non-supporting keypoints in the front image.
    """
    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    ax[0].imshow(img0_left, cmap='gray')
    ax[0].set_title("Left Image 0")
    ax[0].axis('off')
    for pt in supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)
    for pt in non_supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='red', markersize=1)
    ax[1].imshow(img1_left, cmap='gray')
    ax[1].set_title("Left Image 1")
    ax[1].axis('off')
    for pt in supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)
    for pt in non_supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='red', markersize=1)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"results_ex7/plots_supp_nonsupps_after_ransac/{title}")
    plt.close()


def get_sucees_estimation_ex7(points_cloud_3d, cons_match_idxs, back_inliers,
                                             front_inliers, kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right, intrinsic_matrix,
                                             R0_right, t0_right,
                                             verbose, cons_3d_points, actual_pixels, prev_supporters_indices, Rs, ts, start_time

                              ):
    if verbose:
        print("Refining RANSAC results...")
    while True:
        curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
        R, t = build_model(curr_supporters, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, Rs[0], ts[0], R0_right, t0_right, use_random=False)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, R, t)
        if len(supporters_indices) > len(prev_supporters_indices):
            prev_supporters_indices = supporters_indices
        else:
            break

    curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
    actual_pixels = extract_actual_consensus_pixels(curr_supporters, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)

    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")

    return R, t, curr_supporters, prev_supporters_indices, actual_pixels


def trying_estimate_projection_matrices_with_ransac_ex7(points_cloud_3d, cons_match_idxs,
                                             back_inliers, front_inliers,
                                             kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right,
                                             intrinsic_matrix,
                                             back_left_rot, back_left_trans,
                                             R0_right, t0_right,
                                             key_frame, frame,
                                             verbose: bool = True):
    """Legacy Exercise 7 RANSAC variant with its original early-failure rule."""
    start_time = time.time()
    success_prob = 0.99
    outlier_prob = 0.99
    num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)

    prev_supporters_indices = []
    cons_3d_points = points_cloud_3d[[m[0] for m in cons_match_idxs]]
    actual_pixels = extract_actual_consensus_pixels(cons_match_idxs, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    if verbose:
        print(f"Starting RANSAC with {num_iterations} iterations. \n kf {key_frame // 5}  to kf {frame // 5} (frame: {key_frame} to frame {frame})")
    constant_num_iteration = 0
    succes = True
    while num_iterations > 0:
        if constant_num_iteration > 50 and len(prev_supporters_indices) < 20:
            succes = False
            break
        constant_num_iteration += 1
        Rs, ts = build_model(cons_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels,
                                                              intrinsic_matrix, Rs, ts)

        if len(supporters_indices) > len(prev_supporters_indices):
            prev_supporters_indices = supporters_indices
            outlier_prob = 1 - len(prev_supporters_indices) / len(cons_match_idxs)
            num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)
        else:
            num_iterations -= 1

    return succes, cons_3d_points, actual_pixels, prev_supporters_indices, Rs, ts, start_time


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
        non_supporting_pixels_back, non_supporting_pixels_front, supporting_pixels_back, supporting_pixels_front =\
            get_supporters_unsupporters_to_plot(
            consensus_match_indices_0_1, keypoints0_left, keypoints1_left, supporter_indices, tracking_matches)

        ex7_plot_supporters_non_supporters_after_ransac(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                       non_supporting_pixels_back, non_supporting_pixels_front,
                                       title=f"key_frame {key_frame} to frame {frame} - supporters and unsupporters after ransac ")
        return len(supporter_indices) / len(consensus_match_indices_0_1) * 100, len(supporter_indices), actual_pixels
    return 0, 0, 0
