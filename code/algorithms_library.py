import os
import random
import time

import cv2
import gtsam
import numpy as np
from matplotlib import pyplot as plt

from tracking_database import TrackingDB
# from tracking_database import TrackingDB
import gtsam.utils.plot as gtsam_plot

PROBLEM_PROMPT = "problem prompt"

# DATASET_PATH = os.path.join(os.getcwd(), r'dataset\sequences\00')
DATASET_PATH_LINUX = os.path.join(os.getcwd(), r'dataset/sequences/00')
DATASET_PATH = os.path.join(os.getcwd(), 'dataset', 'sequences', '00')
LEFT_CAMS_TRANS_PATH = os.path.join(os.getcwd(), 'dataset', 'poses', '00.txt')
#
# DETECTOR = cv2.SIFT_create()
# # DEFAULT_MATCHER = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
# MATCHER = cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5),
#                                 searchParams=dict(checks=50))


# Detector (slightly richer features)
DETECTOR = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=10)

# FLANN for float descriptors (SIFT) — use KDTree
FLANN_INDEX_KDTREE = 1
index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)   # KD-tree
search_params = dict(checks=128)                              # was 50; 128–256 is safer

MATCHER = cv2.FlannBasedMatcher(index_params, search_params)



# Optional symmetric check (compute matches21 the same way and keep only symmetric pairs)


NUM_FRAMES = 3360
MAX_DEVIATION = 2
Epsilon = 1e-10

# matplotlib.use('TkAgg')

DATA_PATH = '../../VAN_ex/dataset/sequences/00/'






def detect_keypoints(img, method='ORB', num_keypoints=500):
    """
    Detects keypoints in an image using the specified method.

    Args:
    - img (np.array): Input image in which to detect keypoints.
    - method (str): Feature detection method ('ORB', 'AKAZE', 'SIFT').
    - num_keypoints (int): Number of keypoints to detect.

    Returns:
    - keypoints (list): Detected keypoints.
    - descriptors (np.array): Descriptors of the detected keypoints.
    """
    if method == 'ORB':
        detector = cv2.ORB_create(nfeatures=num_keypoints)
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
    elif method == 'SIFT':
        detector = cv2.SIFT_create()
    else:
        raise ValueError(f"Unsupported method: {method}")

    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def get_matches_from_kpts(kp1, kp2):
    pass


def draw_keypoints(img, keypoints):
    """
    Draws keypoints on an image.

    Args:
    - img (np.array): Input image.
    - keypoints (list): Detected keypoints.

    Returns:
    - img_with_keypoints (np.array): Image with keypoints drawn.
    """
    return cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)


def read_images(idx):
    """
    Reads a pair of stereo images from the dataset.

    Args:
    - idx (int): Index of the image pair.

    Returns:
    - img1 (np.array): First image of the stereo pair.
    - img2 (np.array): Second image of the stereo pair.
    """
    img_name = '{:06d}.png'.format(idx)

    img1 = cv2.imread(DATASET_PATH + f'image_0/' + img_name, 0)
    img2 = cv2.imread(DATASET_PATH + f'image_1/' + img_name, 0)
    return img1, img2


def apply_ratio_test(matches, ratio_threshold=0.5):
    """
    Applies the ratio test to reject matches.

    Args:
    - matches (list): List of matches obtained from matching descriptors.
    - ratio_threshold (float): Threshold value for the ratio of distances to reject matches.

    Returns:
    - good_matches (list): List of matches passing the ratio test.
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return good_matches


def match_keypoints(descriptors1, descriptors2, matcher="bf"):
    """
        Matches keypoints between two sets of descriptors using the Brute Force Matcher with Hamming distance.

        Args:
        - descriptors1 (np.array): Descriptors of keypoints from the first image.
        - descriptors2 (np.array): Descriptors of keypoints from the second image.

        Returns:
        - matches (list): List of matches between keypoints in the two images.
    """
    if (matcher == "bf"):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
    else:
        matches = MATCHER.match(descriptors1, descriptors2)
    return matches


def read_cameras(calib_file):
    """
       Reads the camera calibration file and extracts the intrinsic and extrinsic parameters.

       Args:
       - calib_file (str): Path to the camera calibration file.

       Returns:
       - k (np.array): Intrinsic camera matrix (3x3).
       - m1 (np.array): Extrinsic parameters of the first camera (3x4).
       - m2 (np.array): Extrinsic parameters of the second camera (3x4).
    """
    with open(calib_file) as f:
        l1 = f.readline().split()[1:]  # Skip first token
        l2 = f.readline().split()[1:]  # Skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2, plot=True):
    """
        Performs the triangulation process for a set of inlier matches between two images and plots the 3D points.

        Args:
        - P0 (np.array): Projection matrix of the first camera (3x4).
        - P1 (np.array): Projection matrix of the second camera (3x4).
        - inliers (list): List of inlier matches.
        - k (np.array): Intrinsic camera matrix (3x3).
        - keypoints1 (list): List of keypoints in the first image.
        - keypoints2 (list): List of keypoints in the second image.

        Returns:
        - points_3D_custom (np.array): Array of triangulated 3D points.
        - pts1 (np.array): Array of inlier points from the first image.
        - pts2 (np.array): Array of inlier points from the second image.
    """

    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in inliers]).T
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in inliers]).T
    points_3D_custom = triangulation(k @ P0, k @ P1, pts1.T, pts2.T)
    # Example usage
    # points = np.random.rand(100, 3) * 10  # Generate some random 3D points
    # plot_3d_points(points, title="3D Points Example", xlim=(0, 10), ylim=(0, 10), zlim=(0, 10))
    if plot:
        plot_3d_points(points_3D_custom, title="Custom Triangulation", xlim=(-10, 10), ylim=(-10, 10), zlim=(-20, 150))
    return points_3D_custom, pts1, pts2


def linear_least_square_pts(left_cam_matrix, right_cam_matrix, left_kp, right_kp):
    """
        Computes the 3D point using linear least squares from corresponding 2D points in stereo images.
        Args:
        - left_cam_matrix (np.array): Projection matrix of the left camera (3x4).
        - right_cam_matrix (np.array): Projection matrix of the right camera (3x4).
        - left_kp (tuple): 2D point in the left image.
        - right_kp (tuple): 2D point in the right image.

        Returns:
        - np.array: 4D homogeneous coordinates of the triangulated point.
    """
    mat_a = np.array([left_cam_matrix[2] * left_kp[0] - left_cam_matrix[0],
                      left_cam_matrix[2] * left_kp[1] - left_cam_matrix[1],
                      right_cam_matrix[2] * right_kp[0] - right_cam_matrix[0],
                      right_cam_matrix[2] * right_kp[1] - right_cam_matrix[1]])
    _, _, vT = np.linalg.svd(mat_a)
    return vT[-1]


def triangulation(left_cam_matrix, right_cam_matrix, left_kp_list, right_kp_list):
    """
        Triangulates 3D points from corresponding 2D points in stereo images.

        Args:
        - left_cam_matrix (np.array): Projection matrix of the left camera (3x4).
        - right_cam_matrix (np.array): Projection matrix of the right camera (3x4).
        - left_kp_list (list): List of 2D points in the left image.
        - right_kp_list (list): List of 2D points in the right image.

        Returns:
        - np.array: Array of triangulated 3D points.
    """
    num_kp = len(left_kp_list)
    triangulation_pts = []
    for i in range(num_kp):
        p4d = linear_least_square_pts(left_cam_matrix, right_cam_matrix, left_kp_list[i], right_kp_list[i])
        p3d = p4d[:3] / p4d[3]
        triangulation_pts.append(p3d)
    return np.array(triangulation_pts)


def plot_3d_points(points, title="3D Points", xlim=None, ylim=None, zlim=None):
    """
    Plots 3D points using matplotlib with fixed axis limits.

    Args:
    - points (np.array): Array of 3D points.
    - title (str): Title of the plot (default is "3D Points").
    - xlim (tuple): Limits for the x-axis (min, max).
    - ylim (tuple): Limits for the y-axis (min, max).
    - zlim (tuple): Limits for the z-axis (min, max).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    # plt.show()


def get_stereo_matches_with_filtered_keypoints_avish_test(img_left, img_right):
    """
    Performs stereo matching with filtered keypoints between two images.

    Args:
    - img_left (np.array): Left image.
    - img_right (np.array): Right image.
    - feature_detector (str): Feature detector to use ('ORB', 'AKAZE'). Default is 'AKAZE'.
    - max_deviation (int): Maximum vertical deviation threshold for matching keypoints. Default is 2 pixels.

    Returns:
    - filtered_keypoints_left (list): Filtered keypoints from the left image.
    - filtered_keypoints_right (list): Filtered keypoints from the right image.
    - filtered_descriptors_left (np.array): Descriptors corresponding to filtered keypoints in the left image.
    - filtered_descriptors_right (np.array): Descriptors corresponding to filtered keypoints in the right image.
    - good_matches (list): List of good matches passing the deviation threshold.
    - keypoints_left (list): All keypoints detected in the left image.
    - keypoints_right (list): All keypoints detected in the right image.
    """

    # Detect keypoints and compute descriptors
    keypoints_left, descriptors_left = DETECTOR.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = DETECTOR.detectAndCompute(img_right, None)
    # Match descriptors
    # matches = MATCHER.match(descriptors_left, descriptors_right)
    # KNN + Lowe ratio + (optional) symmetric cross-check
    matches12 = MATCHER.knnMatch(descriptors_left, descriptors_right, k=2)
    matches = [m for m, n in matches12 if m.distance < 0.8 * n.distance]  # 0.75–0.8 for SIFT
    # Filter matches based on the deviation threshold
    filtered_keypoints_left = []
    filtered_keypoints_right = []
    filtered_descriptors_left = []
    filtered_descriptors_right = []
    good_matches = []
    i = 0
    for match in matches:
        pt_left = keypoints_left[match.queryIdx]
        pt_right = keypoints_right[match.trainIdx]
        if abs(pt_left.pt[1] - pt_right.pt[1]) <= MAX_DEVIATION:
            filtered_keypoints_left.append(pt_left)
            filtered_keypoints_right.append(pt_right)
            filtered_descriptors_left.append(descriptors_left[match.queryIdx])
            filtered_descriptors_right.append(descriptors_right[match.trainIdx])
            # maybe we can do as follows:
            match.trainIdx = i
            match.queryIdx = i
            good_matches.append(match)
            i += 1
    filtered_descriptors_left = np.array(filtered_descriptors_left)
    filtered_descriptors_right = np.array(filtered_descriptors_right)

    return filtered_keypoints_left, filtered_keypoints_right, filtered_descriptors_left, filtered_descriptors_right, good_matches, keypoints_left, keypoints_right


def estimate_complete_trajectory_avish_test(num_frames: int = NUM_FRAMES, verbose=True, db=None):
    """
    Estimates the complete camera trajectory using consecutive image pairs.

    Parameters:
    - num_frames (int): Number of image pairs to process.
    - verbose (bool): Verbosity flag for printing progress.

    Returns:
    - Rs_left (list): List of rotation matrices for the left camera.
    - ts_left (list): List of translation vectors for the left camera.
    - total_elapsed (float): Total elapsed time for processing.
    """
    supporters_percentage = []
    start_time, minutes_counter = time.time(), 0
    if verbose:
        print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

    # load initiial cameras:
    K, M1, M2 = read_cameras_matrices()
    R0_left, t0_left = M1[:, :3], M1[:, 3:]
    R0_right, t0_right = M2[:, :3], M2[:, 3:]
    Rs_left, ts_left = [R0_left], [t0_left]

    # load first pair:
    img0_l, img0_r = read_images_from_dataset(0)
    back_pair_preprocess = get_stereo_matches_with_filtered_keypoints_avish_test(img0_l, img0_r)
    filtered_back_left_kps, filtered_back_right_kps, filtered_back_left_desc, filtered_back_right_desc, filtered_back_inliers, _, _ = back_pair_preprocess
    filtered_desc_left_back_for_link, links = db.create_links(filtered_back_left_desc, filtered_back_left_kps,
                                                              filtered_back_right_kps, filtered_back_inliers)
    db.add_frame(links, filtered_desc_left_back_for_link)

    for idx in range(1, num_frames):
        back_left_R, back_left_t = Rs_left[-1], ts_left[-1]
        back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, R0_right, t0_right)
        points_cloud_3d = cv_triangulate_matched_points(filtered_back_left_kps, filtered_back_right_kps,
                                                        filtered_back_inliers,
                                                        K, back_left_R, back_left_t, back_right_R, back_right_t)

        # run the estimation on the current pair:
        front_left_img, front_right_img = read_images_from_dataset(idx)
        front_pair_preprocess = get_stereo_matches_with_filtered_keypoints_avish_test(front_left_img, front_right_img)
        filtered_front_left_kps, filtered_front_right_kps, filtered_front_left_desc, filtered_front_right_desc, filtered_front_inliers, keypoints_left, keypoints_right = front_pair_preprocess

        filtered_desc_left_front_for_link, links = db.create_links(filtered_front_left_desc, filtered_front_left_kps,
                                                                   filtered_front_right_kps,
                                                                   filtered_front_inliers)

        track_matches = sorted(MATCHER.match(filtered_back_left_desc, filtered_front_left_desc),
                               key=lambda match: match.queryIdx)
        consensus_indices = find_consensus_matches_indices(filtered_back_inliers, filtered_front_inliers, track_matches)
        curr_Rs, curr_ts, curr_supporters, _ = estimate_projection_matrices_with_ransac(points_cloud_3d,
                                                                                        consensus_indices,
                                                                                        filtered_back_inliers,
                                                                                        filtered_front_inliers,
                                                                                        filtered_back_left_kps,
                                                                                        filtered_back_right_kps,
                                                                                        filtered_front_left_kps,
                                                                                        filtered_front_right_kps, K,
                                                                                        back_left_R, back_left_t,
                                                                                        R0_right, t0_right,
                                                                                        verbose=False)

        inliers_idx = [i[2] for i in curr_supporters]
        # Set the indices in inliers_idx to True
        inliers_bool_indices = [i in inliers_idx for i in range(len(track_matches))]
        db.add_frame(links, filtered_desc_left_front_for_link, track_matches, inliers_bool_indices)

        # print update if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and curr_minute > minutes_counter:
            minutes_counter = curr_minute
            print(f"\tProcessed {idx} tracking-pairs in {minutes_counter} minutes")

        # update variables for the next pair:
        # todo: ask David if we need to bootstrap the kps
        Rs_left.append(curr_Rs[2])
        ts_left.append(curr_ts[2])
        filtered_back_left_kps, filtered_back_left_desc = filtered_front_left_kps, filtered_front_left_desc
        filtered_back_right_kps, filtered_back_right_desc = filtered_front_right_kps, filtered_front_right_desc
        filtered_back_inliers = filtered_front_inliers

        supporters_percentage.append(100.00 * len(curr_supporters) / len(filtered_front_inliers))

    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = total_elapsed / 60
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    db.set_supporters_percentage(supporters_percentage)
    return Rs_left, ts_left, total_elapsed, supporters_percentage


def plot_supporters_non_supporters(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                   non_supporting_pixels_back, non_supporting_pixels_front, title):
    """
    Plots keypoints classified as supporters and non-supporters in two images.

    Args:
    - img0_left (np.array): Left image 0.
    - img1_left (np.array): Left image 1.
    - supporting_pixels_back (list): List of supporting keypoints in the back image.
    - supporting_pixels_front (list): List of supporting keypoints in the front image.
    - non_supporting_pixels_back (list): List of non-supporting keypoints in the back image.
    - non_supporting_pixels_front (list): List of non-supporting keypoints in the front image.
    """
    # Create a figure to hold both subplots
    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    # Plotting image left0
    ax[0].imshow(img0_left, cmap='gray')
    ax[0].set_title("Left Image 0")
    ax[0].axis('off')  # Turn off the axis
    for pt in supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)  # Smaller points
    for pt in non_supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='red', markersize=1)  # Smaller points
    # Plotting image left1
    ax[1].imshow(img1_left, cmap='gray')
    ax[1].set_title("Left Image 1")
    ax[1].axis('off')  # Turn off the axis
    for pt in supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)  # Smaller points
    for pt in non_supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='red', markersize=1)  # Smaller points

    # Finalizing plot settings
    plt.suptitle(title)
    plt.tight_layout()  # Adjust subplots to give some space
    # plt.show()
    plt.close()




def ex7_plot_supporters_non_supporters_after_ransac(img0_left, img1_left, supporting_pixels_back, supporting_pixels_front,
                                   non_supporting_pixels_back, non_supporting_pixels_front, title):
    """
    Plots keypoints classified as supporters and non-supporters in two images.

    Args:
    - img0_left (np.array): Left image 0.
    - img1_left (np.array): Left image 1.
    - supporting_pixels_back (list): List of supporting keypoints in the back image.
    - supporting_pixels_front (list): List of supporting keypoints in the front image.
    - non_supporting_pixels_back (list): List of non-supporting keypoints in the back image.
    - non_supporting_pixels_front (list): List of non-supporting keypoints in the front image.
    """
    # Create a figure to hold both subplots
    fig, ax = plt.subplots(2, 1, figsize=(6, 12))
    # Plotting image left0
    ax[0].imshow(img0_left, cmap='gray')
    ax[0].set_title("Left Image 0")
    ax[0].axis('off')  # Turn off the axis
    for pt in supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)  # Smaller points
    for pt in non_supporting_pixels_back:
        ax[0].plot(pt[0], pt[1], 'o', color='red', markersize=1)  # Smaller points
    # Plotting image left1
    ax[1].imshow(img1_left, cmap='gray')
    ax[1].set_title("Left Image 1")
    ax[1].axis('off')  # Turn off the axis
    for pt in supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='cyan', markersize=1)  # Smaller points
    for pt in non_supporting_pixels_front:
        ax[1].plot(pt[0], pt[1], 'o', color='red', markersize=1)  # Smaller points

    # Finalizing plot settings
    plt.suptitle(title)
    plt.tight_layout()  # Adjust subplots to give some space
    plt.savefig(f"results_ex7/plots_supp_nonsupps_after_ransac/{title}")
    plt.close()




def get_stereo_matches_with_filtered_keypoints(img_left, img_right, feature_detector='AKAZE', max_deviation=2):
    """
    Performs stereo matching with filtered keypoints between two images using pre-defined detector and matcher.

    Args:
    - img_left (np.array): Left image.
    - img_right (np.array): Right image.
    - feature_detector (str): Feature detector to use ('ORB', 'AKAZE'). Default is 'AKAZE'.
    - max_deviation (int): Maximum vertical deviation threshold for matching keypoints. Default is 2 pixels.

    Returns:
    - filtered_keypoints_left (list): Filtered keypoints from the left image.
    - filtered_keypoints_right (list): Filtered keypoints from the right image.
    - filtered_descriptors_left (np.array): Descriptors corresponding to filtered keypoints in the left image.
    - filtered_descriptors_right (np.array): Descriptors corresponding to filtered keypoints in the right image.
    - good_matches (list): List of good matches passing the deviation threshold.
    - keypoints_left (list): All keypoints detected in the left image.
    - keypoints_right (list): All keypoints detected in the right image.
    """

    # Initialize the feature detector
    if feature_detector == 'ORB':
        detector = cv2.ORB_create()
    elif feature_detector == 'AKAZE':
        detector = cv2.AKAZE_create(threshold=0.001, nOctaveLayers=2)
    else:
        raise ValueError("Unsupported feature detector")

    # Detect keypoints and compute descriptors
    keypoints_left, descriptors_left = DETECTOR.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = DETECTOR.detectAndCompute(img_right, None)
    bf = MATCHER
    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Filter matches based on the deviation threshold
    filtered_keypoints_left = []
    filtered_keypoints_right = []
    filtered_descriptors_left = []
    filtered_descriptors_right = []
    good_matches = []

    for match in matches:
        pt_left = keypoints_left[match.queryIdx].pt
        pt_right = keypoints_right[match.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) <= max_deviation:
            filtered_keypoints_left.append(keypoints_left[match.queryIdx])
            filtered_keypoints_right.append(keypoints_right[match.trainIdx])
            filtered_descriptors_left.append(descriptors_left[match.queryIdx])
            filtered_descriptors_right.append(descriptors_right[match.trainIdx])
            # maybe we can do as follows:
            # match.trainIdx = i (when i is the number of iteration)
            # match.queryIdx = i
            good_matches.append(match)

    filtered_descriptors_left = np.array(filtered_descriptors_left)
    filtered_descriptors_right = np.array(filtered_descriptors_right)

    return filtered_keypoints_left, filtered_keypoints_right, filtered_descriptors_left, filtered_descriptors_right, good_matches, keypoints_left, keypoints_right


def plot_root_ground_truth_and_estimate(estimated_locations, ground_truth_locations):
    """
    Plots the camera trajectory based on estimated and ground truth locations.

    Args:
    - estimated_locations (np.array): Estimated trajectory locations.
    - ground_truth_locations (np.array): Ground truth trajectory locations.
    """

    # Plot the trajectories
    plt.figure(figsize=(10, 8))
    plt.plot(ground_truth_locations[:, 0], ground_truth_locations[:, 2], label='Ground Truth', color='r',
             linestyle='--')
    plt.plot(estimated_locations[:, 0], estimated_locations[:, 2], label='Estimated', color='b', marker='o')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Camera Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# def plot_3d_points(points, title="3D Points"):
#     """
#         Plots 3D points using matplotlib.
#
#         Args:
#         - points (np.array): Array of 3D points.
#         - title (str): Title of the plot (default is "3D Points").
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.title(title)
#     plt.show()


def reject_matches(keypoints1, keypoints2, matches):
    """
    Rejects matches based on vertical deviation between corresponding keypoints.

    Args:
    - keypoints1 (list): List of keypoints in the first image.
    - keypoints2 (list): List of keypoints in the second image.
    - matches (list): List of matches between keypoints.

    Returns:
    - deviations (list): List of vertical deviations for each match.
    - inliers (list): List of matches classified as inliers (deviation <= 2 pixels).
    - outliers (list): List of matches classified as outliers (deviation > 2 pixels).
    - indices (dict): Dictionary mapping keypoints from the first image to corresponding keypoints in the second image.
    """
    deviations = []
    inliers = []
    outliers = []
    indices = {}
    # idx = 0
    kp_mathces_im1 = [match.queryIdx for match in matches]
    kp_mathces_im2 = [match.trainIdx for match in matches]
    for i, j in zip(kp_mathces_im1, kp_mathces_im2):
        if abs(keypoints1[i].pt[1] - keypoints2[j].pt[1]) <= 2:
            indices[i] = j

    for i, match in enumerate(matches):
        # for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        if deviation > 2:
            outliers.append(match)
        else:
            inliers.append(match)
    return deviations, inliers, outliers, indices


def reject_matches_and_remove_keypoints(keypoints1, keypoints2, matches):
    """
    Rejects matches based on vertical deviation between corresponding points.
plot
    Args:
    - keypoints1 (list): List of keypoints in the first image.
    - keypoints2 (list): List of keypoints in the second image.
    - matches (list): List of matches between keypoints.

    Returns:
    - deviations (list): List of vertical deviations.
    - inliers (list): List of matches with deviations <= 2 pixels.
    - outliers (list): List of matches with deviations > 2 pixels.
    """
    deviations = []
    inliers = []
    outliers = []

    # Create copies of keypoints lists to avoid modifying the originals
    # Convert keypoints1 and keypoints2 to lists if they are tuples
    if isinstance(keypoints1, tuple):
        keypoints1 = list(keypoints1)
    if isinstance(keypoints2, tuple):
        keypoints2 = list(keypoints2)
    keypoints1_filtered = keypoints1.copy()
    keypoints2_filtered = keypoints2.copy()

    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        deviations.append(deviation)

        if deviation > 2:
            # Remove keypoints from filtered lists
            keypoints1_filtered[match.queryIdx] = None
            keypoints2_filtered[match.trainIdx] = None
            outliers.append(match)
        else:
            inliers.append(match)

    # Remove None entries from filtered keypoints lists
    keypoints1_filtered = [kp for kp in keypoints1_filtered if kp is not None]
    keypoints2_filtered = [kp for kp in keypoints2_filtered if kp is not None]

    return deviations, inliers, outliers, keypoints1_filtered, keypoints2_filtered


def init_matches(idx):
    """
        Initializes and matches keypoints between a pair of stereo images.
        Args:
        - idx (int): Index of the image pair.
        Returns:
        - img1_color (np.array): First image in color.
        - img2_color (np.array): Second image in color.
        - keypoints1 (list): List of keypoints in the first image.
        - keypoints2 (list): List of keypoints in the second image.
        - matches (list): List of matches between keypoints.
    """
    img1, img2 = read_images(idx)
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    keypoints1, descriptors1 = detect_keypoints(img1, method='ORB', num_keypoints=500)
    keypoints2, descriptors2 = detect_keypoints(img2, method='ORB', num_keypoints=500)
    matches = match_keypoints(descriptors1, descriptors2)
    return img1_color, img2_color, keypoints1, keypoints2, matches


def cv_triangulation(P0, P1, pts1, pts2):
    """
        Performs triangulation using OpenCV's triangulatePoints function and plots the 3D points.

        Args:
        - P0 (np.array): Projection matrix of the first camera (3x4).
        - P1 (np.array): Projection matrix of the second camera (3x4).
        - pts1 (np.array): Array of points in the first image.
        - pts2 (np.array): Array of points in the second image.

        Returns:
        - points_3D_cv (np.array): Array of triangulated 3D points.
    """
    points_3D_cv = cv2.triangulatePoints(P0, P1, pts1, pts2)
    points_3D_cv /= points_3D_cv[3]  # Normalize the points to make homogeneous coordinates 1
    points_3D_cv = points_3D_cv[:3].T  # Transpose to get an array of shape (N, 3)
    plot_3d_points(points_3D_cv, title="OpenCV Triangulation")
    return points_3D_cv


def cloud_points_triangulation(idx):
    """
    Performs triangulation of 3D points from stereo matches.

    Args:
    - idx (int): Index for image pair selection.

    Returns:
    - k (np.array): Intrinsic camera matrix.
    - P0 (np.array): Projection matrix for the first camera.
    - P1 (np.array): Projection matrix for the second camera.
    - points_3D_custom (np.array): Triangulated 3D points.
    """

    img1_color, img2_color, keypoints1, keypoints2, matches = init_matches(idx)
    deviations, inliers, _, kp_indices = reject_matches(keypoints1, keypoints2, matches)
    k, P0, P1 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = triangulation_process(P0, P1, inliers, k, keypoints1, keypoints2)
    return k, P0, P1, points_3D_custom


def basic_match_with_significance_test():
    """
    Placeholder function for basic match with significance test.

    Currently not implemented.
    """

    pass


def create_dict_to_pnp_avish_test(matches_01, matches_11, filtered_keypoints_left1, filtered_keypoints_right1,
                                  points_3D_00):
    """
    Creates dictionaries for PnP based on filtered keypoints and matches.

    Args:
    - matches_01 (list): Matches between images 0 and 1.
    - matches_11 (list): Matches within image 1.
    - filtered_keypoints_left1 (list): Filtered keypoints in image 1.
    - filtered_keypoints_right1 (list): Filtered keypoints in image 1.
    - points_3D_00 (np.array): 3D points from image 0.

    Returns:
    - new_filtered_keypoints_left1_pts (np.array): 2D points of filtered keypoints in image 1.
    - new_filtered_keypoints_right1_pts (np.array): 2D points of corresponding keypoints in image 1.
    - new_filtered_3D_keypoints_left0 (np.array): Corresponding 3D points in image 0.
    """

    new_filtered_keypoints_left1 = []
    new_filtered_keypoints_right1 = []
    new_filtered_3D_keypoints_left0 = []
    i = 0
    dict_l1_to_r1 = {}
    for match in matches_11:
        dict_l1_to_r1[filtered_keypoints_left1[match.queryIdx]] = filtered_keypoints_right1[match.trainIdx]
    for match in matches_01:
        kp_left1 = filtered_keypoints_left1[match.trainIdx]
        new_filtered_keypoints_left1.append(kp_left1)
        kp_left0 = points_3D_00[match.queryIdx]
        new_filtered_3D_keypoints_left0.append(kp_left0)
        match.queryIdx = i
        match.trainIdx = i
        kp_right1 = dict_l1_to_r1[kp_left1]
        new_filtered_keypoints_right1.append(kp_right1)

    new_filtered_keypoints_left1_pts = [point.pt for point in new_filtered_keypoints_left1]
    new_filtered_keypoints_right1_pts = [point.pt for point in new_filtered_keypoints_right1]
    return (np.array(new_filtered_keypoints_left1_pts), np.array(new_filtered_keypoints_right1_pts),
            np.array(new_filtered_3D_keypoints_left0))


# add fitered_kp_right1, matches11.
def create_dict_to_pnp(matches_01, inliers_matches_11, filtered_keypoints_left1, keypoints_left0, keypoints_left1,
                       keypoints_right1,
                       points_3D_custom):
    """
    Creates dictionaries for PnP based on filtered keypoints and matches.

    Args:
    - matches_01 (list): Matches between images 0 and 1.
    - inliers_matches_11 (list): Inlier matches within image 1.
    - filtered_keypoints_left1 (list): Filtered keypoints in image 1.
    - keypoints_left0 (list): Keypoints in image 0.
    - keypoints_left1 (list): Keypoints in image 1.
    - keypoints_right1 (list): Keypoints in image 1.
    - points_3D_custom (np.array): 3D points from triangulation.

    Returns:
    - points_3d (np.array): 3D points.
    - points_2D_l0 (np.array): 2D points in image 0.
    - points_2D_l1 (np.array): 2D points in image 1.
    - points_2D_r1 (np.array): 2D points in image 1.
    """
    points_2Dleft1_to_2Dright1 = {}
    points_3d = []
    points_2D_l1 = []
    points_2D_r1 = []
    points_2D_l0 = []
    for match in inliers_matches_11:
        points_2Dleft1_to_2Dright1[keypoints_left1[match.queryIdx]] = keypoints_right1[match.trainIdx]
    for match in matches_01:
        # Get the index of the keypoint in the left1 image
        idx_2d_left1 = match.trainIdx
        kp_match_to_l1 = filtered_keypoints_left1[idx_2d_left1]
        pt_2d_r1 = points_2Dleft1_to_2Dright1[kp_match_to_l1].pt
        # get the index of the keypoint of left0 image
        # Get the index of the 3D point
        idx_3d = match.queryIdx
        pt_2d_l0 = keypoints_left0[idx_3d].pt
        # Get the 2D point from filtered_keypoints_left1
        pt_2d_l1 = filtered_keypoints_left1[idx_2d_left1].pt
        # Get the corresponding 3D point from points_3D_custom
        pt_3d = points_3D_custom[idx_3d]
        # Store the points in arrays
        points_3d.append(pt_3d)
        points_2D_l1.append(pt_2d_l1)
        points_2D_r1.append(pt_2d_r1)
        points_2D_l0.append(pt_2d_l0)
    return np.array(points_3d), np.array(points_2D_l0), np.array(points_2D_l1), np.array(points_2D_r1)


def create_in_out_l1_dict(inliers, points_2D_l1, filtered_keypoints_left1):
    """
    Creates a dictionary to classify keypoints in image 1 as inliers or outliers.

    Args:
    - inliers (list): List of inlier matches.
    - points_2D_l1 (np.array): 2D points in image 1.
    - filtered_keypoints_left1 (list): Filtered keypoints in image 1.

    Returns:
    - in_out_l1_dict (dict): Dictionary mapping keypoints in image 1 to True (inliers) or False (outliers).
    """

    in_out_l1_dict = {}
    for i, kp in enumerate(filtered_keypoints_left1):
        if kp.pt in points_2D_l1[inliers]:
            in_out_l1_dict[kp] = True
        else:
            in_out_l1_dict[kp] = False
    return in_out_l1_dict


def reject_matches_and_remove_keypoints1(keypoints1, keypoints2, matches):
    """
    Rejects matches based on vertical deviation between corresponding points.

    Args:
    - keypoints1 (list): List of keypoints in the first image.
    - keypoints2 (list): List of keypoints in the second image.
    - matches (list): List of matches between keypoints.

    Returns:
    - deviations (list): List of vertical deviations.
    - inliers (list): List of matches with deviations <= 2 pixels.
    - outliers (list): List of matches with deviations > 2 pixels.
    """
    deviations = []
    inliers = []
    outliers = []
    idx_inliers = []
    # Create copies of keypoints lists to avoid modifying the originals
    # Convert keypoints1 and keypoints2 to lists if they are tuples
    if isinstance(keypoints1, tuple):
        keypoints1 = list(keypoints1)
    if isinstance(keypoints2, tuple):
        keypoints2 = list(keypoints2)
    keypoints1_filtered = keypoints1.copy()
    keypoints2_filtered = keypoints2.copy()
    i = 0
    for match in matches:
        pt1 = keypoints1[match.queryIdx].pt
        pt2 = keypoints2[match.trainIdx].pt
        deviation = abs(pt1[1] - pt2[1])  # Vertical deviation
        deviations.append(deviation)
        deviations.append(deviation)

        if deviation > 2:
            # Remove keypoints from filtered lists
            keypoints1_filtered[match.queryIdx] = None
            keypoints2_filtered[match.trainIdx] = None
            outliers.append(match)
        else:
            inliers.append(match)

    # Remove None entries from filtered keypoints lists
    keypoints1_filtered = [kp for kp in keypoints1_filtered if kp is not None]
    keypoints2_filtered = [kp for kp in keypoints2_filtered if kp is not None]

    return deviations, inliers, outliers, keypoints1_filtered, keypoints2_filtered


def stack_R_and_t(R, t):
    """
    Stacks rotation matrix and translation vector horizontally.

    Args:
    - R (np.array): Rotation matrix.
    - t (np.array): Translation vector.

    Returns:
    - Rt (np.array): Stacked matrix [R | t].
    """
    Rt = np.hstack((R, t))
    return Rt


def plot_camera_positions(extrinsic_matrices):
    """
    Plots camera positions based on extrinsic matrices.

    Args:
    - extrinsic_matrices (list): List of extrinsic matrices (rotation and translation).

    Displays a 2D plot showing camera positions.
    """
    # Define colors for each camera
    colors = ['r', 'g', 'b', 'c']
    # Extract camera positions from the extrinsic matrices
    positions = []
    for Rt in extrinsic_matrices:
        # The camera position is the negative inverse of the rotation matrix multiplied by the translation vector
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        position = -np.linalg.inv(R).dot(t)
        positions.append(position)
        # print(position)

    positions = np.array(positions)

    # Plot the camera positions in 2D (x-z plane)
    plt.figure()
    for i, position in enumerate(positions):
        plt.scatter(position[0], position[2], c=colors[i], marker='o', label=f'Camera {i}')
        plt.text(position[0], position[2], f'Camera {i}', color=colors[i])

    # Set axis labels
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Camera Positions')

    # Set axis limits
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.grid(True)
    # plt.axis('equal')
    plt.legend()
    # plt.show()


def read_cameras_matrices():
    """
    Reads camera calibration matrices from a file.
    Returns:
    - k (np.array): Intrinsic camera matrix.
    - m1 (np.array): Extrinsic camera matrix for the first camera.
    - m2 (np.array): Extrinsic camera matrix for the second camera.
    """
    with open(os.path.join(DATASET_PATH, 'calib.txt')) as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def read_cameras_matrices_linux():
    """
    Reads camera calibration matrices from a file.
    Returns:
    - k (np.array): Intrinsic camera matrix.
    - m1 (np.array): Extrinsic camera matrix for the first camera.
    - m2 (np.array): Extrinsic camera matrix for the second camera.
    """
    with open(os.path.join(DATASET_PATH, 'calib.txt')) as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
    l1 = [float(i) for i in l1]
    m1 = np.array(l1).reshape(3, 4)
    l2 = [float(i) for i in l2]
    m2 = np.array(l2).reshape(3, 4)
    k = m1[:, :3]
    m1 = np.linalg.inv(k) @ m1
    m2 = np.linalg.inv(k) @ m2
    return k, m1, m2


def extract_bool_inliers_numpy(kp1, kp2, matches):
    # Convert keypoints to arrays
    pts_left = np.array([kp1[match.queryIdx].pt for match in matches])
    pts_right = np.array([kp2[match.trainIdx].pt for match in matches])

    # Compute the absolute difference in y-coordinates
    y_diff = np.abs(pts_left[:, 1] - pts_right[:, 1])

    # Determine inliers based on the maximum deviation
    inliers = y_diff <= MAX_DEVIATION

    return inliers.tolist()


def extract_bool_inliers(kp1, kp2, matches):
    inliers = []
    for match in matches:
        pt_left = kp1[match.queryIdx]
        pt_right = kp2[match.trainIdx]
        if abs(pt_left.pt[1] - pt_right.pt[1]) <= MAX_DEVIATION:
            inliers.append(True)

        else:
            inliers.append(False)

    return inliers


def extract_keypoints_and_inliers(img_left, img_right):
    """
    Detects keypoints and computes descriptors for two input images.

    Parameters:
    - img_left (numpy.ndarray): Input image from the left camera.
    - img_right (numpy.ndarray): Input image from the right camera.

    Returns:
    - keypoints_left (list): Detected keypoints in the left image.
    - descriptors_left (numpy.ndarray): Descriptors corresponding to the keypoints in the left image.
    - keypoints_right (list): Detected keypoints in the right image.
    - descriptors_right (numpy.ndarray): Descriptors corresponding to the keypoints in the right image.
    - inliers (list): Keypoint matches that satisfy a geometric constraint.
    - outliers (list): Keypoint matches that do not satisfy the geometric constraint.
    """
    # Detect keypoints and compute descriptors
    keypoints_left, descriptors_left = DETECTOR.detectAndCompute(img_left, None)
    keypoints_right, descriptors_right = DETECTOR.detectAndCompute(img_right, None)

    # Match descriptors
    matches = MATCHER.match(descriptors_left, descriptors_right)

    # Filter matches based on the deviation threshold
    inliers = []
    outliers = []

    for match in matches:
        pt_left = keypoints_left[match.queryIdx]
        pt_right = keypoints_right[match.trainIdx]
        if abs(pt_left.pt[1] - pt_right.pt[1]) <= MAX_DEVIATION:
            inliers.append(match)
        else:
            outliers.append(match)
    inliers = sorted(inliers, key=lambda match: match.queryIdx)
    outliers = sorted(outliers, key=lambda match: match.queryIdx)
    return keypoints_left, descriptors_left, keypoints_right, descriptors_right, inliers, outliers


def cv_triangulate_matched_points(kps_left, kps_right, inliers,
                                  K, R_back_left, t_back_left, R_back_right, t_back_right):
    """
    Triangulates with cv 3D points from 2D correspondences using camera matrices and keypoints.
    Parameters:
    - kps_left (list): Keypoints in the left image.
    - kps_right (list): Keypoints in the right image.
    - inliers (list): Keypoint matches that are considered inliers.
    - K (numpy.ndarray): Intrinsic camera matrix.
    - R_back_left (numpy.ndarray): Rotation matrix of the left camera.
    - t_back_left (numpy.ndarray): Translation vector of the left camera.
    - R_back_right (numpy.ndarray): Rotation matrix of the right camera.
    - t_back_right (numpy.ndarray): Translation vector of the right camera.

    Returns:
    - X_3d (numpy.ndarray): Triangulated 3D points.
    """
    num_matches = len(inliers)
    pts1 = np.array([kps_left[inliers[i].queryIdx].pt for i in range(num_matches)])
    pts2 = np.array([kps_right[inliers[i].trainIdx].pt for i in range(num_matches)])
    proj_mat_left = K @ np.hstack((R_back_left, t_back_left))
    proj_mat_right = K @ np.hstack((R_back_right, t_back_right))
    X_4d = cv2.triangulatePoints(proj_mat_left, proj_mat_right, pts1.T, pts2.T)
    X_4d /= (X_4d[3] + 1e-10)
    return X_4d[:-1].T


def find_consensus_matches_indices(back_inliers, front_inliers, tracking_inliers):
    """
    Finds consensus matches indices between back and front inliers.

    Parameters:
    - back_inliers (list): Inlier matches from the back camera.
    - front_inliers (list): Inlier matches from the front camera.
    - tracking_inliers (list): Inlier matches used for tracking.

    Returns:
    - consensus (list): List of tuples containing indices of consensus matches.
    """
    # Sort inliers based on their queryIdx, which is O(n log n) for each list
    back_sorted = sorted(back_inliers, key=lambda m: m.queryIdx)
    front_sorted = sorted(front_inliers, key=lambda m: m.queryIdx)
    # Create dictionaries to map queryIdx to their index in the sorted lists
    back_dict = {m.queryIdx: idx for idx, m in enumerate(back_sorted)}
    front_dict = {m.queryIdx: idx for idx, m in enumerate(front_sorted)}
    consensus = []
    # For each inlier in tracking_inliers, attempt to find the corresponding elements in back and front inliers
    for idx, inlier in enumerate(tracking_inliers):
        back_idx = inlier.queryIdx
        front_idx = inlier.trainIdx
        if back_idx in back_dict and front_idx in front_dict:
            idx_of_back = back_dict[back_idx]
            idx_of_front = front_dict[front_idx]
            consensus.append((idx_of_back, idx_of_front, idx))

    return consensus


def find_consensus_matches_indices_new(back_inliers, front_inliers, tracking_inliers):
    pass


def calculate_front_camera_matrix(cons_matches, back_points_cloud,
                                  front_inliers, front_kps_left, intrinsic_matrix):
    """
    Calculates the extrinsic matrix of the front-left camera using PnP based on consensus matches.

    Parameters:
    - cons_matches (list): Consensus matches between back and front cameras.
    - back_points_cloud (numpy.ndarray): 3D points cloud from the back camera.
    - front_inliers (list): Inlier matches from the front camera.
    - front_kps_left (list): Keypoints in the left image of the front camera.
    - intrinsic_matrix (numpy.ndarray): Intrinsic camera matrix.

    Returns:
    - success (bool): Success flag of the PnP solver.
    - rotation (numpy.ndarray): Rotation matrix of the front-left camera.
    - translation (numpy.ndarray): Translation vector of the front-left camera.
    """
    # Use cv2.solvePnP to compute the front-left camera's extrinsic matrix
    # based on at least 4 consensus matches and their corresponding 2D & 3D positions
    num_samples = len(cons_matches)
    if num_samples < 4:
        raise ValueError(f"Must provide at least 4 sampled consensus-matches, {num_samples} given")
    cloud_shape = back_points_cloud.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $back_points_cloud is not a 3D array"
    if cloud_shape[1] != 3:
        back_points_cloud = back_points_cloud.T  # making sure we have shape Nx3 for solvePnP
    points_3D = np.zeros((num_samples, 3))
    points_2D = np.zeros((num_samples, 2))

    # populate the arrays
    for i in range(num_samples):
        cons_match = cons_matches[i]
        points_3D[i] = back_points_cloud[cons_match[0]]
        front_left_matched_kp_idx = front_inliers[cons_match[1]].queryIdx
        points_2D[i] = front_kps_left[front_left_matched_kp_idx].pt

    success, rotation, translation = cv2.solvePnP(objectPoints=points_3D,
                                                  imagePoints=points_2D,
                                                  cameraMatrix=intrinsic_matrix,
                                                  distCoeffs=None,
                                                  flags=cv2.SOLVEPNP_EPNP)
    return success, cv2.Rodrigues(rotation)[0], translation


def calculate_right_camera_matrix(R_left, t_left, right_R0, right_t0):
    """
    Calculates the rotation and translation matrix of the front-right camera
    based on the transformation from the back-left to the front-right camera.

    Parameters:
    - R_left (numpy.ndarray): Rotation matrix of the left camera.
    - t_left (numpy.ndarray): Translation vector of the left camera.
    - right_R0 (numpy.ndarray): Rotation matrix of the right camera relative to the initial position.
    - right_t0 (numpy.ndarray): Translation vector of the right camera relative to the initial position.

    Returns:
    - front_right_Rot (numpy.ndarray): Rotation matrix of the front-right camera.
    - front_right_trans (numpy.ndarray): Translation vector of the front-right camera.
    """

    assert right_R0.shape == (3, 3) and R_left.shape == (3, 3)
    assert right_t0.shape == (3, 1) or right_t0.shape == (3,)
    assert t_left.shape == (3, 1) or t_left.shape == (3,)
    right_t0 = right_t0.reshape((3, 1))
    t_left = t_left.reshape((3, 1))

    front_right_Rot = right_R0 @ R_left
    front_right_trans = right_R0 @ t_left + right_t0
    assert front_right_Rot.shape == (3, 3) and front_right_trans.shape == (3, 1)
    return front_right_Rot, front_right_trans


def calculate_camera_locations(back_left_R, back_left_t, right_R0, right_t0,
                               cons_matches, back_points_cloud, front_inliers, front_kps_left, intrinsic_matrix):
    """
    Calculates the 3D positions of the cameras based on their rotation and translation matrices.
    Parameters:
    - back_left_R (numpy.ndarray): Rotation matrix of the back-left camera.
    - back_left_t (numpy.ndarray): Translation vector of the back-left camera.
    - right_R0 (numpy.ndarray): Rotation matrix of the right camera relative to the initial position.
    - right_t0 (numpy.ndarray): Translation vector of the right camera relative to the initial position.
    - cons_matches (list): Consensus matches between back and front cameras.
    - back_points_cloud (numpy.ndarray): 3D points cloud from the back camera.
    - front_inliers (list): Inlier matches from the front camera.
    - front_kps_left (list): Keypoints in the left image of the front camera.
    - intrinsic_matrix (numpy.ndarray): Intrinsic camera matrix.
    Returns:
    - camera_positions (numpy.ndarray): 4x3 array representing the 3D positions of the 4 cameras.
    """
    # Returns a 4x3 np array representing the 3D position of the 4 cameras,
    # in coordinates of the back_left camera (hence the first line should be np.zeros(3))
    back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, right_R0, right_t0)
    is_success = False
    while not is_success:
        cons_sample = random.sample(cons_matches, 4)
        is_success, front_left_R, front_left_t = calculate_front_camera_matrix(cons_sample, back_points_cloud,
                                                                               front_inliers, front_kps_left,
                                                                               intrinsic_matrix)
    front_right_R, front_right_t = calculate_right_camera_matrix(front_left_R, front_left_t, back_right_R, back_right_t)

    back_right_coordinates = - back_right_R.T @ back_right_t
    front_left_coordinates = - front_left_R.T @ front_left_t
    front_right_coordinates = - front_right_R.T @ front_right_t
    return np.array([np.zeros((3, 1)), back_right_coordinates,
                     front_left_coordinates, front_right_coordinates]).reshape((4, 3))


def plot_tracks(db: TrackingDB):
    all_tracks = db.all_tracks()
    track_lengths = [(trackId, len(db.frames(trackId))) for trackId in all_tracks if len(db.frames(trackId)) > 1]

    # 1. Longest track
    longest_track = max(track_lengths, key=lambda x: x[1])[0]

    # 2. Track with length of 10
    track_length_10 = next((trackId for trackId, length in track_lengths if length == 10), None)

    # 3. Track with length of 5
    track_length_5 = next((trackId for trackId, length in track_lengths if length == 5), None)

    # 4. Random track
    random_track = random.choice(track_lengths)[0]

    tracks_to_plot = [longest_track, track_length_10, track_length_5, random_track]
    track_names = ["longest_track", "track_length_10", "track_length_5", "random_track"]

    for trackId, track_name in zip(tracks_to_plot, track_names):
        if trackId is None:
            continue

        frames = db.frames(trackId)
        for frameId in frames:
            img_left, img_right = read_images_from_dataset(frameId)
            link = db.link(frameId, trackId)
            left_kp = link.left_keypoint()
            right_kp = link.right_keypoint()

            # Draw keypoints
            img_left = cv2.circle(img_left, (int(left_kp[0]), int(left_kp[1])), 5, (0, 255, 0), -1)  # Green color
            img_right = cv2.circle(img_right, (int(right_kp[0]), int(right_kp[1])), 5, (0, 255, 0), -1)  # Green color

            # Draw track lines
            if frameId > frames[0]:
                prev_frameId = frames[frames.index(frameId) - 1]
                prev_link = db.link(prev_frameId, trackId)
                prev_left_kp = prev_link.left_keypoint()
                prev_right_kp = prev_link.right_keypoint()

                img_left = cv2.line(img_left, (int(prev_left_kp[0]), int(prev_left_kp[1])),
                                    (int(left_kp[0]), int(left_kp[1])), (255, 0, 0), 2)  # Blue color
                img_right = cv2.line(img_right, (int(prev_right_kp[0]), int(prev_right_kp[1])),
                                     (int(right_kp[0]), int(right_kp[1])), (255, 0, 0), 2)  # Blue color

            # Save images
            cv2.imwrite(f"{track_name}frame{frameId}_left.png", img_left)
            cv2.imwrite(f"{track_name}frame{frameId}_right.png", img_right)


def compose_transformations(trans1, trans2):

    r2r1 = trans2[:, :-1] @ (trans1[:, :-1])
    r2t1_t2 = (trans2[:, :-1]) @ (trans1[:, -1]) + trans2[:, -1]
    ext_r1 = np.column_stack((r2r1, r2t1_t2))
    return ext_r1



def gtsam_compose_to_first_kf(trans):

    """
    Compose the transformation from the first keyframe to the i-th keyframe
    """
    relative_trans = []
    last = trans[0]
    i = 0
    for t in trans:
        if i == 0:
            i += 1
            continue
        last = last.compose(t)
        relative_trans.append(last)
    return relative_trans


def compose_transformations_gtsam(trans1, trans2):
    r2r1 = trans2.rotation() * trans1.rotation()
    t2 = trans1.translation()

    t2_gtsam = gtsam.gtsam.Point3(0, 0, 0)
    t2_gtsam.x = t2[0]
    t2_gtsam.y = t2[1]
    t2_gtsam.z = t2[2]
    r2t1_t2 = (trans2.rotation() * t2_gtsam)
    r2t1_t2.compose(trans2.translation())
    ext_r1 = gtsam.Pose3(r2r1, r2t1_t2)
    return ext_r1

def project_point(point_3D_world, T, K):
    # Transform the point to the current frame's coordinate system
    R = T[:3, :3]
    t = T[:3, 3]
    t = t.reshape(3, 1)
    point_3D_frame = R @ point_3D_world + t
    # Project the point
    point_2D = K @ point_3D_frame
    point_2D /= point_2D[2]

    return point_2D[:2]


def init_db():
    db = TrackingDB()
    estimated_trajectory, ground_truth_trajectory, distances, supporters_percentage = compute_trajectory_and_distance_avish_test(
        NUM_FRAMES, True, db)
    plot_root_ground_truth_and_estimate(estimated_trajectory, ground_truth_trajectory)
    # plot_tracks(db)
    db.serialize("db_v1")
    # Load your database if necessary
    # db.load('db')
    return db, supporters_percentage


def print_statistics(stats):
    print(f"Total number of tracks: {stats['total_tracks']}")
    print(f"Number of frames: {stats['number_of_frames']}")
    print(f"Mean track length: {stats['mean_track_length']}")
    print(f"Maximum track length: {stats['max_track_length']}")
    print(f"Minimum track length: {stats['min_track_length']}")
    print(f"Mean number of frame links: {stats['mean_frame_links']}")


def calculate_statistics(db):
    all_tracks = db.all_tracks()
    track_lengths = [len(db.frames(trackId)) for trackId in all_tracks if len(db.frames(trackId)) > 1]

    total_tracks = len(track_lengths)
    number_of_frames = db.frame_num()

    mean_track_length = np.mean(track_lengths) if track_lengths else 0
    max_track_length = np.max(track_lengths) if track_lengths else 0
    min_track_length = np.min(track_lengths) if track_lengths else 0

    frame_links = [len(db.tracks(frameId)) for frameId in db.all_frames()]
    mean_frame_links = np.mean(frame_links) if frame_links else 0

    return {
        "total_tracks": total_tracks,
        "number_of_frames": number_of_frames,
        "mean_track_length": mean_track_length,
        "max_track_length": max_track_length,
        "min_track_length": min_track_length,
        "mean_frame_links": mean_frame_links
    }


def calculate_pixels_for_3d_points(points_cloud_3d, intrinsic_matrix, Rs, ts):
    """
    Takes a collection of 3D points in the world and calculates their projection on the cameras' planes.
    The 3D points should be an array of shape 3xN.
    $Rs and $ts are rotation matrices and translation vectors and should both have length M.

    return: a Mx2xN np array of (p_x, p_y) pixel coordinates for each camera
    """
    assert len(Rs) == len(ts), \
        "Number of rotation matrices and translation vectors must be equal"
    assert points_cloud_3d.shape[0] == 3 or points_cloud_3d.shape[1] == 3, \
        f"Must provide a 3D points matrix, input has shape {points_cloud_3d.shape}"
    if points_cloud_3d.shape[0] != 3:
        points_cloud_3d = points_cloud_3d.T

    num_cameras = len(Rs)
    num_points = points_cloud_3d.shape[1]
    pixels = np.zeros((num_cameras, 2, num_points))
    for i in range(num_cameras):
        R, t = Rs[i], ts[i]
        t = np.reshape(t, (3, 1))
        projections = intrinsic_matrix @ (
                R @ points_cloud_3d + t)  # non normalized homogeneous coordinates of shape 3xN
        hom_coordinates = projections / (projections[2] + Epsilon)  # add epsilon to avoid 0 division
        pixels[i] = hom_coordinates[:2]
    return pixels


def project(p3d_pts, projection_cam_mat):
    hom_projected = p3d_pts @ projection_cam_mat[:, :3].T + projection_cam_mat[:, 3].T
    projected = hom_projected[:2] / hom_projected[2]
    return projected


def get_euclidean_distance(a, b):
    # supporters are determined by norma 2.:
    distances = np.sqrt(np.power(a - b, 2).sum(axis=1))

    return distances


def extract_actual_consensus_pixels(cons_matches, back_inliers, front_inliers,
                                    back_left_kps, back_right_kps, front_left_kps, front_right_kps):
    """
    Extracts the 2D pixel coordinates of consensus-matched keypoints from all cameras.

    Parameters:
    - cons_matches (list): Consensus matches between back and front cameras.
    - back_inliers (list): Inlier matches from the back camera.
    - front_inliers (list): Inlier matches from the front camera.
    - back_left_kps (list): Keypoints in the left image of the back camera.
    - back_right_kps (list): Keypoints in the right image of the back camera.
    - front_left_kps (list): Keypoints in the left image of the front camera.
    - front_right_kps (list): Keypoints in the right image of the front camera.
    Returns:
    - consensus_pixels (numpy.ndarray): 4x2xN array containing the 2D pixel coordinates of consensus-matched keypoints.
    """
    # Returns a 4x2xN array containing the 2D pixels of all consensus-matched keypoints
    back_left_pixels, back_right_pixels = [], []
    front_left_pixels, front_right_pixels = [], []
    for m in cons_matches:
        # cons_matches is a list of tuples of indices: (back_inliers_idx, front_inlier_idx, tracking_match_idx)
        single_back_inlier, single_front_inlier = back_inliers[m[0]], front_inliers[m[1]]

        back_left_point = back_left_kps[single_back_inlier.queryIdx].pt
        back_left_pixels.append(np.array(back_left_point))

        back_right_point = back_right_kps[single_back_inlier.trainIdx].pt
        back_right_pixels.append(np.array(back_right_point))

        front_left_point = front_left_kps[single_front_inlier.queryIdx].pt
        front_left_pixels.append(np.array(front_left_point))

        front_right_point = front_right_kps[single_front_inlier.trainIdx].pt
        front_right_pixels.append(np.array(front_right_point))

    back_left_pixels = np.array(back_left_pixels).T
    back_right_pixels = np.array(back_right_pixels).T
    front_left_pixels = np.array(front_left_pixels).T
    front_right_pixels = np.array(front_right_pixels).T
    return np.array([back_left_pixels, back_right_pixels, front_left_pixels, front_right_pixels])


def find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts, max_distance: int = 2):
    """
    Find supporters for the model ($Rs & $ts) our of all consensus-matches.
    A supporter is a consensus match that has a calculated projection (based on $Rs & $ts) that is "close enough"
    to it's actual keypoints' pixels in all four images. The value of "close enough" is the argument $max_distance

    Returns a list of consensus matches that support the current model.
    """

    # make sure we have a Nx3 cloud:
    cloud_shape = cons_3d_points.shape
    assert cloud_shape[0] == 3 or cloud_shape[1] == 3, "Argument $cons_3d_points is not a 3D-points array"
    if cloud_shape[1] != 3:
        cons_3d_points = cons_3d_points.T

    # calculate pixels for all four cameras and make sure it has correct shape
    calculated_pixels = calculate_pixels_for_3d_points(cons_3d_points.T, intrinsic_matrix, Rs, ts)
    assert actual_pixels.shape == calculated_pixels.shape

    # find indices that are no more than $max_distance apart on all 4 projections
    euclidean_distances = np.linalg.norm(actual_pixels - calculated_pixels, ord=2, axis=1)
    supporting_indices = np.where((euclidean_distances <= max_distance).all(axis=0))[0]
    return supporting_indices


def calculate_number_of_iteration_for_ransac(p: float, e: float, s: int) -> int:
    """
    Calculate how many iterations of RANSAC are required to get good enough results,
    i.e. for a set of size $s, with outlier probability $e and success probability $p
    we need N > log(1-$p) / log(1-(1-$e)^$s)

    :param p: float -> required success probability (0 < $p < 1)
    :param e: float -> probability to be outlier (0 < $e < 1)
    :param s: int -> minimal set size (s > 0)
    :return: N: int -> number of iterations
    """
    assert s > 0, "minimal set size must be a positive integer"
    nom = np.log(1 - p)
    denom = np.log(1 - np.power(1 - e, s))
    return int(nom / denom) + 1


def build_model(consensus_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True):
    """
    Builds a model of camera rotations and translations based on consensus matches and 3D point cloud.

    Parameters:
    - consensus_match_idxs (list): Consensus matches between back and front cameras.
    - points_cloud_3d (numpy.ndarray): 3D points cloud from the back camera.
    - front_inliers (list): Inlier matches from the front camera.
    - kps_front_left (list): Keypoints in the left image of the front camera.
    - intrinsic_matrix (numpy.ndarray): Intrinsic camera matrix.
    - back_left_rot (numpy.ndarray): Rotation matrix of the back-left camera.
    - back_left_trans (numpy.ndarray): Translation vector of the back-left camera.
    - R0_right (numpy.ndarray): Initial rotation matrix of the right camera.
    - t0_right (numpy.ndarray): Initial translation vector of the right camera.
    - use_random (bool): Flag to sample consensus matches randomly.

    Returns:
    - Rs (list of numpy.ndarray): Rotation matrices of the cameras.
    - ts (list of numpy.ndarray): Translation vectors of the cameras.
    """
    # calculate the model (R & t of each camera) based on
    # the back-left camera and the [R|t] transformation to Right camera
    back_right_rot, back_right_trans = calculate_right_camera_matrix(back_left_rot, back_left_trans, R0_right, t0_right)
    is_success = False
    while not is_success:
        sample_consensus_matches = random.sample(consensus_match_idxs, 4) if use_random else consensus_match_idxs
        is_success, front_left_rot, front_left_trans = calculate_front_camera_matrix(sample_consensus_matches,
                                                                                     points_cloud_3d, front_inliers,
                                                                                     kps_front_left, intrinsic_matrix)
    front_right_rot, front_right_trans = calculate_right_camera_matrix(front_left_rot, front_left_trans,
                                                                       R0_right, t0_right)
    Rs = [back_left_rot, back_right_rot, front_left_rot, front_right_rot]
    ts = [back_left_trans, back_right_trans, front_left_trans, front_right_trans]
    return Rs, ts


def estimate_projection_matrices_with_ransac(points_cloud_3d, cons_match_idxs,
                                             back_inliers, front_inliers,
                                             kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right,
                                             intrinsic_matrix,
                                             back_left_rot, back_left_trans,
                                             R0_right, t0_right,
                                             verbose: bool = True):
    """
    Implement RANSAC algorithm to estimate extrinsic matrix of the two front cameras,
    based on the two back cameras, the consensus-matches and the 3D points-cloud of the back pair.

    Returns the best fitting model:
        - Rs - rotation matrices of 4 cameras
        - ts - translation vectors of 4 cameras
        - supporters - subset of consensus-matches that support this model,
            i.e. projected keypoints are no more than 2 pixels away from the actual keypoint
    """
    start_time = time.time()
    success_prob = 0.99
    outlier_prob = 0.99  # this value is updated while running RANSAC
    num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)

    prev_supporters_indices = []
    cons_3d_points = points_cloud_3d[[m[0] for m in cons_match_idxs]]
    actual_pixels = extract_actual_consensus_pixels(cons_match_idxs, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    if verbose:
        print(f"Starting RANSAC with {num_iterations} iterations.")

    while num_iterations > 0:
        Rs, ts = build_model(cons_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels,
                                                              intrinsic_matrix, Rs, ts)

        if len(supporters_indices) > len(prev_supporters_indices):
            prev_supporters_indices = supporters_indices
            outlier_prob = 1 - len(prev_supporters_indices) / len(cons_match_idxs)
            num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)
            # if verbose:
            #     print(f"\tRemaining iterations: {num_iterations}\n\t\t" +
            #           f"Number of Supporters: {len(prev_supporters_indices)}")
        else:
            num_iterations -= 1
            # if verbose and num_iterations % 100 == 0:
            #     print(f"Remaining iterations: {num_iterations}\n\t\t" +
            #           f"Number of Supporters: {len(prev_supporters_indices)}")

    # at this point we have a good model (Rs & ts) and we can refine it based on all supporters
    if verbose:
        print("Refining RANSAC results...")
    while True:
        curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
        Rs, ts = build_model(curr_supporters, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, Rs[0], ts[0], R0_right, t0_right, use_random=False)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts)
        if len(supporters_indices) > len(prev_supporters_indices):
            # we can refine the model even further
            prev_supporters_indices = supporters_indices
        else:
            # no more refinement, exit the loop
            break



    # finished, we can return the model
    curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")

    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")

    return Rs, ts, curr_supporters, prev_supporters_indices


def trying_estimate_projection_matrices_with_ransac_ex7(points_cloud_3d, cons_match_idxs,
                                             back_inliers, front_inliers,
                                             kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right,
                                             intrinsic_matrix,
                                             back_left_rot, back_left_trans,
                                             R0_right, t0_right,
                                             key_frame, frame,
                                             verbose: bool = True):
    """
    Implement RANSAC algorithm to estimate extrinsic matrix of the two front cameras,
    based on the two back cameras, the consensus-matches and the 3D points-cloud of the back pair.

    Returns the best fitting model:
        - Rs - rotation matrices of 4 cameras
        - ts - translation vectors of 4 cameras
        - supporters - subset of consensus-matches that support this model,
            i.e. projected keypoints are no more than 2 pixels away from the actual keypoint
    """
    start_time = time.time()
    success_prob = 0.99
    outlier_prob = 0.99  # this value is updated while running RANSAC
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
            # if verbose:
            #     print(f"\tRemaining iterations: {num_iterations}\n\t\t" +
            #           f"Number of Supporters: {len(prev_supporters_indices)}")
        else:
            num_iterations -= 1
            # if verbose and num_iterations % 100 == 0:
            #     print(f"Remaining iterations: {num_iterations}\n\t\t" +
            #           f"Number of Supporters: {len(prev_supporters_indices)}")

    return succes, cons_3d_points, actual_pixels, prev_supporters_indices, Rs, ts, start_time





def get_sucees_estimation_ex7(points_cloud_3d, cons_match_idxs, back_inliers,
                                             front_inliers, kps_back_left, kps_back_right,
                                             kps_front_left, kps_front_right, intrinsic_matrix,
                                             R0_right, t0_right,
                                             verbose, cons_3d_points, actual_pixels, prev_supporters_indices, Rs, ts, start_time

                              ):
    # at this point we have a good model (Rs & ts) and we can refine it based on all supporters
    if verbose:
        print("Refining RANSAC results...")
    while True:
        curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
        R, t = build_model(curr_supporters, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, Rs[0], ts[0], R0_right, t0_right, use_random=False)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, R, t)
        if len(supporters_indices) > len(prev_supporters_indices):
            # we can refine the model even further
            prev_supporters_indices = supporters_indices
        else:
            # no more refinement, exit the loop
            break

    # finished, we can return the model
    curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
    actual_pixels = extract_actual_consensus_pixels(curr_supporters, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)

    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")

    return R, t, curr_supporters, prev_supporters_indices, actual_pixels


def transform_coordinates(points_3d, R, t):
    """
    Transforms 3D coordinates using a rotation matrix and translation vector.

    Parameters:
    - points_3d (numpy.ndarray): 3D coordinates to transform, shape (3, N).
    - R (numpy.ndarray): Rotation matrix, shape (3, 3).
    - t (numpy.ndarray): Translation vector, shape (3,).

    Returns:
    - transformed (numpy.ndarray): Transformed 3D coordinates, shape (3, N).
    """

    input_shape = points_3d.shape
    assert input_shape[0] == 3 or input_shape[1] == 3, \
        f"can only operate on matrices of shape 3xN or Nx3, provided {input_shape}"
    if input_shape[0] != 3:
        points_3d = points_3d.T  # making sure we are working with a 3xN array

    assert t.shape == (3, 1) or t.shape == (3,), \
        f"translation vector must be of size 3, provided {t.shape}"
    if t.shape != (3, 1):
        t = np.reshape(t, (3, 1))  # making sure we are using a 3x1 vector
    assert R.shape == (3, 3), f"rotation matrix must be of shape 3x3, provided {R.shape}"
    transformed = R @ points_3d + t
    assert transformed.shape == points_3d.shape
    return transformed


def estimate_complete_trajectory(num_frames: int = NUM_FRAMES, verbose=True):
    """
    Estimates the complete camera trajectory using consecutive image pairs.

    Parameters:
    - num_frames (int): Number of image pairs to process.
    - verbose (bool): Verbosity flag for printing progress.

    Returns:
    - Rs_left (list): List of rotation matrices for the left camera.
    - ts_left (list): List of translation vectors for the left camera.
    - total_elapsed (float): Total elapsed time for processing.
    """
    start_time, minutes_counter = time.time(), 0
    if verbose:
        print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

    # load initiial cameras:
    K, M1, M2 = read_cameras_matrices()
    R0_left, t0_left = M1[:, :3], M1[:, 3:]
    R0_right, t0_right = M2[:, :3], M2[:, 3:]
    Rs_left, ts_left = [R0_left], [t0_left]

    # load first pair:
    img0_l, img0_r = read_images_from_dataset(0)
    back_pair_preprocess = extract_keypoints_and_inliers(img0_l, img0_r)
    back_left_kps, back_left_desc, back_right_kps, back_right_desc, back_inliers, _ = back_pair_preprocess

    for idx in range(1, num_frames):
        back_left_R, back_left_t = Rs_left[-1], ts_left[-1]
        back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, R0_right, t0_right)
        points_cloud_3d = cv_triangulate_matched_points(back_left_kps, back_right_kps, back_inliers,
                                                        K, back_left_R, back_left_t, back_right_R, back_right_t)

        # run the estimation on the current pair:
        front_left_img, front_right_img = read_images_from_dataset(idx)
        front_pair_preprocess = extract_keypoints_and_inliers(front_left_img, front_right_img)
        front_left_kps, front_left_desc, front_right_kps, front_right_desc, front_inliers, _ = front_pair_preprocess
        track_matches = sorted(MATCHER.match(back_left_desc, front_left_desc),
                               key=lambda match: match.queryIdx)
        consensus_indices = find_consensus_matches_indices(back_inliers, front_inliers, track_matches)
        curr_Rs, curr_ts, _, _ = estimate_projection_matrices_with_ransac(points_cloud_3d, consensus_indices,
                                                                          back_inliers,
                                                                          front_inliers, back_left_kps, back_right_kps,
                                                                          front_left_kps, front_right_kps, K,
                                                                          back_left_R, back_left_t, R0_right, t0_right,
                                                                          verbose=True)
        # print update if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and curr_minute > minutes_counter:
            minutes_counter = curr_minute
            print(f"\tProcessed {idx} tracking-pairs in {minutes_counter} minutes")

        # update variables for the next pair:
        # todo: ask David if we need to bootstrap the kps
        Rs_left.append(curr_Rs[2])
        ts_left.append(curr_ts[2])
        back_left_kps, back_left_desc = front_left_kps, front_left_desc
        back_right_kps, back_right_desc = front_right_kps, front_right_desc
        back_inliers = front_inliers

    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = total_elapsed / 60
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    return Rs_left, ts_left, total_elapsed


def read_poses_truth(seq=(0, NUM_FRAMES)):
    ground_truth_trans = []
    left_cam_trans_path = os.path.join(os.getcwd(), 'dataset', 'poses', '00.txt')
    with open(left_cam_trans_path) as f:
        lines = f.readlines()
    # for i in range(3450):
    for i in range(seq[0], seq[1]):
        left_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        ground_truth_trans.append(left_mat)
    return ground_truth_trans

def get_truth_transformation(path_cams=LEFT_CAMS_TRANS_PATH, num_frames = NUM_FRAMES):
    """
    Reads camera turth transformations from a file


    Returns:
        array of transformations
    """
    truth_arr = []
    with open(path_cams) as f:
        lines = f.readlines()
    for i in range(num_frames):
        left_cam_mat = np.array(lines[i].split(" "))[:-1].astype(float).reshape((3, 4))
        truth_arr.append(left_cam_mat)
    return truth_arr

def compute_relative_gtsam_cam(t):
    return t.translation()


def compute_relative_cam(t):
    return -1 * t[:, :3].T @ t[:, 3]
def compute_trajectory_left_cams(arr):
    glob_cams_locations = []
    for t in arr:
        glob_cams_locations.append(compute_relative_cam(t))
    return np.array(glob_cams_locations)





def compute_trajectory_gtsam_left_cams(arr):
    glob_cams_locations = []
    for t in arr:
        glob_cams_locations.append(compute_relative_gtsam_cam(t))
    return np.array(glob_cams_locations)


def convert_landmarks_cam_rel_to_global(camera, landmarks):
    glob_landmarks = []
    for landmark in landmarks:
        glob_landmark = camera.transformFrom(landmark)
        glob_landmarks.append(glob_landmark)
    return glob_landmarks


def convert_landmarks_to_global(cams, landmarks):
    glob_landmarks = []
    for camera, cam_landmarks in zip(cams, landmarks):
        glob_landmarks_camera = convert_landmarks_cam_rel_to_global(camera, cam_landmarks)
        glob_landmarks += glob_landmarks_camera

    return np.array(glob_landmarks)


def convert_gtsam_cams_to_global(arr):
    relative_arr = []
    last = arr[0]

    for t in arr:
        last = last.compose(t)
        relative_arr.append(last)

    return relative_arr


def plot_trajectories_and_landmarks_ex7(pose_graph=None, bundle_key_frames = None,
                                         title=""):
    if not pose_graph:
        print("pose_graph is ", pose_graph)
        return
    # gtsam_cams_initial_estimate = [gtsam.Pose3()]
    # gtsam_cams_bundle = [gtsam.Pose3()]
    # gtsam_landmarks_bundle = []
    # for window in bundles:
    #     gtsam_cams_initial_estimate.append(window.get_initial_estimate())
    #     gtsam_cams_bundle.append(window.get_optimized_last_camera())
    #     gtsam_landmarks_bundle.append(window.get_optimized_landmarks_lst())

    gtsam_cams_initial_estimate = pose_graph.get_initial_cameras()
    gtsam_cams_bundle = pose_graph.get_optimized_cameras()

    initial_estimate = convert_gtsam_cams_to_global(gtsam_cams_initial_estimate)
    cams = convert_gtsam_cams_to_global(gtsam_cams_bundle)
    landmarks = None
    #Todo: the  initial estimate in ex5 were took from the db class.  here trying to get from values of the graph


    truth_trans = np.array(get_truth_transformation(num_frames=NUM_FRAMES))[bundle_key_frames]
    cams_truth_3d = compute_trajectory_left_cams(truth_trans)
    cams_3d = compute_trajectory_gtsam_left_cams(cams)
    plot_trajectories_and_landmarks(cameras=cams_3d, landmarks=landmarks, initial_estimate_poses=initial_estimate,
                                    cameras_gt=cams_truth_3d, title=f"{title}_with_ground_truth")
    plot_trajectories_and_landmarks(cameras=cams_3d, landmarks=landmarks, title=title)
def plot_trajectories_and_landmarks(cameras=None, landmarks=None,
                                                            initial_estimate_poses=None, cameras_gt=None,
                                                            title="",
                                                            loops=None, numbers=False,
                                                            mahalanobis_dist=None, inliers_perc=None):
    """
    Compare the left cameras relative 2d positions to the ground truth
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    first_legend = []

    landmarks_title = "and landmarks " if landmarks is not None else ""
    loops_title = ""
    dist_title = "Dist = squared mahalanobis distance " if mahalanobis_dist is not None else ""

    # ax.set_title(f"{title} Left cameras {landmarks_title}2d trajectory of {len(cameras_gt)} bundles.\n{dist_title}"
    #              f"{loops_title}")
    ax.set_title(f"{title} left cameras {landmarks_title} 2d trajectory of {len(cameras)} bundles")
    if landmarks is not None:
        a = ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='orange', label="Landmarks")
        first_legend.append(a)

    if initial_estimate_poses is not None:
        first_legend.append(ax.scatter(initial_estimate_poses[:, 0], initial_estimate_poses[:, 2], s=1, c='lime', label="Initial estimate"))

    if cameras is not None:
        first_legend.append(ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Optimized cameras"))

    if cameras_gt is not None:
        first_legend.append(ax.scatter(cameras_gt[:, 0], cameras_gt[:, 2], s=1, c='cyan', label="Cameras ground truth"))

    # Mark loops
    if loops is not None:
        for cur_cam, prev_cams in loops:
            y_diff = 0 if abs(cameras[:, 0][cur_cam] - cameras[:, 0][cur_cam - 1]) < 2 else 15
            x_diff = 0 if abs(cameras[:, 2][cur_cam] - cameras[:, 2][cur_cam - 1]) < 2 else 20

            if numbers:
                ax.text(cameras[:, 0][cur_cam] - x_diff, cameras[:, 2][cur_cam] - y_diff, cur_cam, size=7, fontweight="bold")
            ax.scatter(cameras[:, 0][cur_cam], cameras[:, 2][cur_cam], s=3, c='black')

            if numbers:
                for prev_cam in prev_cams:
                    ax.text(cameras[:, 0][prev_cam], cameras[:, 2][prev_cam], prev_cam, size=7, fontweight="bold")
            ax.scatter(cameras[:, 0][prev_cams], cameras[:, 2][prev_cams], s=1, c='black')

    if landmarks is not None:
        ax.set_xlim(-250, 350)
        ax.set_ylim(-100, 430)

    if loops is not None:
        plt.subplots_adjust(left=0.25, bottom=0.08, right=0.95, top=0.9)

    landmarks_txt = "and landmarks" if landmarks is not None else ""
    mahalanobis_dist_and_inliers = f"Dist: {mahalanobis_dist}; Inliers: {inliers_perc}%\n"

    len_loops = None
    if loops is not None:
        loops_details = "\n".join([str(i) + ")  " + str(cur_cam) + ": " + ",".join([str(prev_cam) for prev_cam in prev_cams])
                                   for i, (cur_cam, prev_cams) in enumerate(loops)])

        loops_txt = mahalanobis_dist_and_inliers + loops_details
        # y = -65 + 9 * len(loops)
        plt.text(-360, -67, loops_txt, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
        len_loops = len(loops)

    first_legend = plt.legend(handles=first_legend, loc='upper left', prop={'size': 7})
    plt.gca().add_artist(first_legend)

    fig.savefig(f"{title}_Bundle_Adjustment_Trajectory.png")

    # fig.savefig(f"Results/{title} Left {len(cameras_gt)} cameras {landmarks_txt} 2d trajectory "
    #             f"m dist {mahalanobis_dist_and_inliers} loops {len_loops} "
    #             f"Bundle_Adjustment_Trajectory.png")
    plt.close(fig)



def xy_triangulation(in_liers, m1c, m2c):
    """
    triangulation for case where: in_lier is xy point.
    (the others are for inliers as key points).
    """
    ps = cv2.triangulatePoints(m1c, m2c, np.array(in_liers[0]).T, np.array(in_liers[1]).T).T
    return np.squeeze(cv2.convertPointsFromHomogeneous(ps))


def read_poses(num_frames=NUM_FRAMES):
    """
    Reads camera poses from a file.

    Returns:
    - Rs (list): List of rotation matrices.
    - ts (list): List of translation vectors.
    """

    Rs, ts = [], []
    file_path = os.path.join(os.getcwd(), 'dataset', 'poses', '00.txt')
    f = open(file_path, 'r')
    for i, line in enumerate(f.readlines()):
        mat = np.array(line.split(), dtype=float).reshape((3, 4))
        Rs.append(mat[:, :3])
        ts.append(mat[:, 3:])
    return Rs[:num_frames], ts[:num_frames]


def calculate_trajectory(Rs, ts):
    """
    Calculates the trajectory of the camera based on rotation and translation matrices.

    Parameters:
    - Rs (list): List of rotation matrices.
    - ts (list): List of translation vectors.

    Returns:
    - trajectory (numpy.ndarray): 3D trajectory of the camera.
    """

    assert len(Rs) == len(ts), \
        "number of rotation matrices and translation vectors mismatch"
    num_samples = len(Rs)
    trajectory = np.zeros((num_samples, 3))
    for i in range(num_samples):
        R, t = Rs[i], ts[i]
        trajectory[i] -= (R.T @ t).reshape((3,))
    return trajectory


def compute_trajectory_and_distance_avish_test(num_frames: int = NUM_FRAMES, verbose: bool = False, db=None):
    """
    Computes the estimated and ground truth camera trajectories and their distances.

    Parameters:
    - num_frames (int): Number of frames/images to process.
    - verbose (bool): Verbosity flag for printing progress.

    Returns:
    - estimated_trajectory (numpy.ndarray): Estimated camera trajectory.
    - ground_truth_trajectory (numpy.ndarray): Ground truth camera trajectory.
    - distances (numpy.ndarray): Distances between estimated and ground truth trajectories.
    """
    if verbose:
        print(f"\nCALCULATING TRAJECTORY FOR {num_frames} IMAGES\n")
    all_R, all_t, elapsed, supporters_percentage = estimate_complete_trajectory_avish_test(num_frames, verbose=verbose,
                                                                                           db=db)
    estimated_trajectory = calculate_trajectory(all_R, all_t)
    poses_R, poses_t = read_poses()
    ground_truth_trajectory = calculate_trajectory(poses_R[:num_frames], poses_t[:num_frames])
    distances = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, ord=2, axis=1)
    db.set_matrices(all_R, all_t)
    return estimated_trajectory, ground_truth_trajectory, distances, supporters_percentage


def compute_trajectory_and_distance(num_frames: int = NUM_FRAMES, verbose: bool = False):
    """
    Computes the estimated and ground truth camera trajectories and their distances.

    Parameters:
    - num_frames (int): Number of frames/images to process.
    - verbose (bool): Verbosity flag for printing progress.

    Returns:
    - estimated_trajectory (numpy.ndarray): Estimated camera trajectory.
    - ground_truth_trajectory (numpy.ndarray): Ground truth camera trajectory.
    - distances (numpy.ndarray): Distances between estimated and ground truth trajectories.
    """
    if verbose:
        print(f"\nCALCULATING TRAJECTORY FOR {num_frames} IMAGES\n")
    all_R, all_t, elapsed = estimate_complete_trajectory(num_frames, verbose=verbose)
    estimated_trajectory = calculate_trajectory(all_R, all_t)
    poses_R, poses_t = read_poses()
    ground_truth_trajectory = calculate_trajectory(poses_R[:num_frames], poses_t[:num_frames])
    distances = np.linalg.norm(estimated_trajectory - ground_truth_trajectory, ord=2, axis=1)
    return estimated_trajectory, ground_truth_trajectory, distances

#
# def convert_rel_landmarks_to_global(cameras, landmarks):
#     """
#     Convert relative to each bundle landmarks to the global coordinate system
#     :param cameras: list of cameras
#     :param landmarks: list of landmarks lists
#     :return: one list of the whole global landmarks
#     """
#     # global_landmarks = []
#     # for bundle_camera, bundle_landmarks in zip(cameras, landmarks):
#     #     bundle_global_landmarks = convert_rel_landmarks_to_global(bundle_camera, bundle_landmarks)
#     #     global_landmarks += bundle_global_landmarks
#     #
#     # return np.array(global_landmarks)
#     global_landmarks = []
#
#     # Loop through each bundle of cameras and corresponding landmarks
#     for i in` range(len(cameras)):
#         camera_position = cameras[i]
#         bundle_landmarks = landmarks[i]
#
#         # Transform landmarks in the bundle to the global coordinate system using the camera pose
#         transformed_landmarks = []
#         for landmark in bundle_landmarks:
#             # Apply translation (assuming rotation is not involved)
#             transformed_landmark = np.array(landmark) + np.array(camera_position)
#             transformed_landmarks.append(transformed_landmark)
#
#         # Append the transformed landmarks to the global list
#         global_landmarks.extend(transformed_landmarks)
#
#     return np.array(global_landmarks)






def calculate_trajectory_key_frames_gtsam(pose3_cameras, num_key_frames=int(3360 / 20)):
    """
    Calculates the trajectory of the camera based on pose3 cameras.

    Parameters:
    - pose3_cameras (list): List of pose3 the key-frames cameras.
    Returns:
    """
    trajectory = np.zeros((num_key_frames, 3))
    for i, pose in enumerate(pose3_cameras):
        R = pose.rotation()
        print("R:", R)
        t = pose.translation()
        print("t:", t)
        new_location_pose = -(R.transpose() @ t)
        print("-R.transpose *  t", new_location_pose)
        # x, y, z, = new_location_pose.x(), new_location_pose.y(), new_location_pose.z()
        trajectory[i] -= new_location_pose.reshape((3,))
    return trajectory


def plot_inliers_outliers_ransac(consensus_match_indices_0_1, img0_left, img1_left, keypoints0_left, keypoints1_left,
                                 sup, tracking_matches):
    """
    Plots supporting and non-supporting matches between two images.

    Parameters:
    - consensus_match_indices_0_1 (list): Consensus match indices between two images.
    - img0_left (numpy.ndarray): Image from the left camera.
    - img1_left (numpy.ndarray): Image from the right camera.
    - keypoints0_left (list): Keypoints in the left image.
    - keypoints1_left (list): Keypoints in the right image.
    - sup (list): List of supporting match indices.
    - tracking_matches (list): All tracking matches between two images.
    """

    supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices_0_1 if
                                   (_a, _b, idx) in sup]
    non_supporting_tracking_matches = [tracking_matches[idx] for (_a, _b, idx) in consensus_match_indices_0_1 if
                                       (_a, _b, idx) not in sup]

    supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in supporting_tracking_matches]]
    supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in supporting_tracking_matches]]
    non_supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in non_supporting_tracking_matches]]
    non_supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in non_supporting_tracking_matches]]

    # Start plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 20))  # Adjust the figsize as needed

    # Plot for img0_left
    axes[0].imshow(img0_left, cmap='gray')
    axes[0].scatter([x for (x, y) in supporting_pixels_back], [y for (x, y) in supporting_pixels_back], s=1, c='orange',
                    marker='*', label='Supporter')
    axes[0].scatter([x for (x, y) in non_supporting_pixels_back], [y for (x, y) in non_supporting_pixels_back], s=1,
                    c='cyan', marker='o', label='Non-Supporter')
    axes[0].axis('off')
    axes[0].set_title("Back Image (img0_left)")

    # Plot for img1_left
    axes[1].imshow(img1_left, cmap='gray')
    axes[1].scatter([x for (x, y) in supporting_pixels_front], [y for (x, y) in supporting_pixels_front], s=1,
                    c='orange', marker='*')
    axes[1].scatter([x for (x, y) in non_supporting_pixels_front], [y for (x, y) in non_supporting_pixels_front], s=1,
                    c='cyan', marker='o')
    axes[1].axis('off')
    axes[1].set_title("Front Image (img1_left)")

    # Add legend and title to the figure instead of the axes to avoid redundancy
    fig.legend(loc='lower center')
    fig.suptitle("Supporting & Non-Supporting Matches", fontsize=16)
    plt.tight_layout()  # Adjust the layout to make the plot compact
    plt.axis('equal')  # Ensures equal scaling
    # plt.show()


# todo: ex2 mistaken putted point_cloud_0 need to put point cloud_1 and point_cloud_0 after T
def plot_two_3D_point_clouds(mR, mt, point_cloud_0):
    """
    Plots two 3D point clouds transformed using given rotation and translation.

    Parameters:
    - mR (list): List of rotation matrices.
    - mt (list): List of translation vectors.
    - point_cloud_0 (numpy.ndarray): 3D point cloud data.
    """

    # create scatter plot of the two point clouds:
    point_cloud_0_transformed_to_1 = transform_coordinates(point_cloud_0.T, mR[2], mt[2])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(point_cloud_0.T[0], point_cloud_0.T[2],
                 point_cloud_0.T[1], c='b', s=2.5, marker='o', label='left0')
    ax.scatter3D(point_cloud_0_transformed_to_1[0], point_cloud_0_transformed_to_1[2],
                 point_cloud_0_transformed_to_1[1], c='r', s=2.5, marker='o', label='left1')
    ax.set_xlim(-17, 17)
    ax.set_ylim(-17, 17)
    ax.set_zlim(-17, 17)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.legend()
    # plt.axis('equal')  # Ensures equal scaling
    # plt.show()


def read_images_from_dataset(idx: int):
    """
        Reads images from a dataset based on index.
        Parameters:
        - idx (int): Index of the image to read.
        Returns:
        - img0 (numpy.ndarray): Image from the left camera.
        - img1 (numpy.ndarray): Image from the right camera.
        """
    image_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(os.path.join(DATASET_PATH, 'image_0', image_name), 0)
    img1 = cv2.imread(os.path.join(DATASET_PATH, 'image_1', image_name), 0)
    return img0, img1


def get_supporters_unsupporters_to_plot(consensus_match_indices_0_1, keypoints0_left, keypoints1_left,
                                        supporter_indices, tracking_matches):
    supporters = [consensus_match_indices_0_1[idx] for idx in supporter_indices]
    # plot the supporters on img0_left and img1_left:
    supporting_tracking_matches = [tracking_matches[idx] for (_, _, idx) in supporters]
    non_supporting_tracking_matches = [m for m in tracking_matches if m not in supporting_tracking_matches]
    supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in supporting_tracking_matches]]
    supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in supporting_tracking_matches]]
    non_supporting_pixels_back = [keypoints0_left[i].pt for i in [m.queryIdx for m in non_supporting_tracking_matches]]
    non_supporting_pixels_front = [keypoints1_left[i].pt for i in [m.trainIdx for m in non_supporting_tracking_matches]]
    return non_supporting_pixels_back, non_supporting_pixels_front, supporting_pixels_back, supporting_pixels_front


def estimate_projection_matrices_with_ransac_for_db(points_cloud_3d, cons_match_idxs,
                                                    back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right,
                                                    kps_front_left, kps_front_right,
                                                    intrinsic_matrix,
                                                    back_left_rot, back_left_trans,
                                                    R0_right, t0_right,
                                                    verbose: bool = True):
    """
    Implement RANSAC algorithm to estimate extrinsic matrix of the two front cameras,
    based on the two back cameras, the consensus-matches and the 3D points-cloud of the back pair.

    Returns the best fitting model:
        - Rs - rotation matrices of 4 cameras
        - ts - translation vectors of 4 cameras
        - supporters - subset of consensus-matches that support this model,
            i.e. projected keypoints are no more than 2 pixels away from the actual keypoint
    """
    start_time = time.time()
    success_prob = 0.99
    outlier_prob = 0.99  # this value is updated while running RANSAC
    num_iterations = calculate_number_of_iteration_for_ransac(success_prob, outlier_prob, 4)

    prev_supporters_indices = []
    cons_3d_points = points_cloud_3d[[m[0] for m in cons_match_idxs]]
    actual_pixels = extract_actual_consensus_pixels(cons_match_idxs, back_inliers, front_inliers,
                                                    kps_back_left, kps_back_right, kps_front_left, kps_front_right)
    if verbose:
        print(f"Starting RANSAC with {num_iterations} iterations.")
        # todo: maybe change to i < ransac_bound or something else maybe like maor?
        #  see wat  in dept frames also worked
    while num_iterations > 0:
        Rs, ts = build_model(cons_match_idxs, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, back_left_rot, back_left_trans, R0_right, t0_right, use_random=True)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels,
                                                              intrinsic_matrix, Rs, ts)

        if len(supporters_indices) > len(prev_supporters_indices):
            prev_supporters_indices = supporters_indices
            outlier_prob = 1 - len(prev_supporters_indices) / len(cons_match_idxs)
            num_iterations = calculate_number_of_iteration_for_ransac(0.99, outlier_prob, 4)
            # if verbose:
            #     print(f"\tRemaining iterations: {num_iterations}\n\t\t" +
            #           f"Number of Supporters: {len(prev_supporters_indices)}")
        else:
            num_iterations -= 1
            # if verbose and num_iterations % 100 == 0:
            #     print(f"Remaining iterations: {num_iterations}\n\t\t" +
            #           f"Number of Supporters: {len(prev_supporters_indices)}")
    if verbose:
        print("Finished Ransac Iterations")
    # at this point we have a good model (Rs & ts) and we can refine it based on all supporters
    if verbose:
        print("Refining RANSAC results...")
    while True:
        curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
        Rs, ts = build_model(curr_supporters, points_cloud_3d, front_inliers, kps_front_left,
                             intrinsic_matrix, Rs[0], ts[0], R0_right, t0_right, use_random=False)
        supporters_indices = find_supporter_indices_for_model(cons_3d_points, actual_pixels, intrinsic_matrix, Rs, ts)
        if len(supporters_indices) > len(prev_supporters_indices):
            # we can refine the model even further
            prev_supporters_indices = supporters_indices
        else:
            # no more refinement, exit the loop
            break

    # finished, we can return the model
    curr_supporters = [cons_match_idxs[idx] for idx in prev_supporters_indices]
    elapsed = time.time() - start_time
    if verbose:
        print(f"RANSAC finished in {elapsed:.2f} seconds\n\tNumber of Supporters: {len(curr_supporters)}")


    return Rs, ts, curr_supporters, prev_supporters_indices


def estimate_complete_trajectory_db(num_frames: int = NUM_FRAMES, db=None, verbose=False):
    """
    Estimates the complete camera trajectory using consecutive image pairs.

    Parameters:
    - num_frames (int): Number of image pairs to process.
    - verbose (bool): Verbosity flag for printing progress.

    Returns:
    - Rs_left (list): List of rotation matrices for the left camera.
    - ts_left (list): List of translation vectors for the left camera.
    - total_elapsed (float): Total elapsed time for processing.
    """
    start_time, minutes_counter = time.time(), 0
    if verbose:
        print(f"Starting to process trajectory for {num_frames} tracking-pairs...")

    # load initiial cameras:
    K, M1, M2 = read_cameras_matrices()
    R0_left, t0_left = M1[:, :3], M1[:, 3:]
    R0_right, t0_right = M2[:, :3], M2[:, 3:]
    Rs_left, ts_left = [R0_left], [t0_left]

    # load first pair:
    img0_l, img0_r = read_images_from_dataset(0)
    back_pair_preprocess = extract_keypoints_and_inliers(img0_l, img0_r)
    back_left_kps, back_left_desc, back_right_kps, back_right_desc, back_inliers, _ = back_pair_preprocess

    filtered_desc_left_back, links = db.create_links(back_left_desc, back_left_kps, back_right_kps, back_inliers)
    db.add_frame(links, filtered_desc_left_back)

    for idx in range(1, num_frames):
        back_left_R, back_left_t = Rs_left[-1], ts_left[-1]
        back_right_R, back_right_t = calculate_right_camera_matrix(back_left_R, back_left_t, R0_right, t0_right)
        points_cloud_3d = cv_triangulate_matched_points(back_left_kps, back_right_kps, back_inliers,
                                                        K, back_left_R, back_left_t, back_right_R, back_right_t)

        # run the estimation on the current pair:
        front_left_img, front_right_img = read_images_from_dataset(idx)
        front_pair_preprocess = extract_keypoints_and_inliers(front_left_img, front_right_img)
        front_left_kps, front_left_desc, front_right_kps, front_right_desc, front_inliers, _ = front_pair_preprocess

        filtered_desc_left_front, links = db.create_links(front_left_desc, front_left_kps, front_right_kps,
                                                          front_inliers)

        track_matches = sorted(MATCHER.match(filtered_desc_left_back, filtered_desc_left_front),
                               key=lambda match: match.queryIdx)

        consensus_indices = find_consensus_matches_indices(back_inliers, front_inliers, track_matches)
        curr_Rs, curr_ts, curr_supporters, _ = estimate_projection_matrices_with_ransac(points_cloud_3d,
                                                                                        consensus_indices,
                                                                                        back_inliers,
                                                                                        front_inliers, back_left_kps,
                                                                                        back_right_kps,
                                                                                        front_left_kps, front_right_kps,
                                                                                        K,
                                                                                        back_left_R, back_left_t,
                                                                                        R0_right, t0_right,
                                                                                        verbose=True)

        inliers_idx = [i[2] for i in curr_supporters]
        # Set the indices in inliers_idx to True
        inliers_bool_indices = [i in inliers_idx for i in range(len(track_matches))]
        db.add_frame(links, filtered_desc_left_front, track_matches, inliers_bool_indices)

        # print update if needed:
        curr_minute = int((time.time() - start_time) / 60)
        if verbose and curr_minute > minutes_counter:
            minutes_counter = curr_minute
            print(f"\tProcessed {idx} tracking-pairs in {minutes_counter} minutes")

        # update variables for the next pair:
        # todo: ask David if we need to bootstrap the kps
        Rs_left.append(curr_Rs[2])
        ts_left.append(curr_ts[2])
        back_left_kps, back_left_desc = front_left_kps, front_left_desc
        back_right_kps, back_right_desc = front_right_kps, front_right_desc
        back_inliers = front_inliers
        filtered_desc_left_back = filtered_desc_left_front

    total_elapsed = time.time() - start_time
    if verbose:
        total_minutes = total_elapsed / 60
        print(f"Finished running for all tracking-pairs. Total runtime: {total_minutes:.2f} minutes")
    return Rs_left, ts_left, total_elapsed


def create_calib_mat_gtsam():
    K_, P_left, P_right = read_cameras_matrices()
    # calculate gtsam.K as the David said.
    baseline = P_right[0, 3]
    calib_mat_gtsam = gtsam.Cal3_S2Stereo(K_[0, 0], K_[1, 1], K_[0, 1], K_[0, 2], K_[1, 2], -baseline)
    return calib_mat_gtsam


def create_ext_matrix_gtsam(db, frame_id):
    current_left_rotation = db.rotation_matrices[frame_id]
    current_left_translation = db.translation_vectors[frame_id]
    # concat row of zeros to R to get 4x3 matrix
    R = np.concatenate((current_left_rotation, np.zeros((1, 3))), axis=0)
    current_left_translation = current_left_translation.reshape(3, 1)
    # concat last_left_translation_homogenus with zero to get 4x1 vector
    zero_row = np.array([0]).reshape(1, 1)
    current_left_translation_homogenus = np.vstack((current_left_translation, zero_row))
    # concat R and last_left_translation_homogenus to get 4x4 matrix
    current_left_transformation_homogenus = np.concatenate((R, current_left_translation_homogenus.reshape(4, 1)),
                                                           axis=1)
    # convert to inverse by gtsam.Pose3.inverse
    Rt_inverse_gtsam = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
    return Rt_inverse_gtsam


def create_ext_mat_gtsam(db, frame_id):
    current_left_rotation = db.rotation_matrices[frame_id]
    current_left_translation = db.translation_vectors[frame_id]

    #create gtsam.Rot3 for create Pose3
    rot = gtsam.Rot3(current_left_rotation)
    #create Pose3
    return gtsam.Pose3(rot, current_left_translation)


def triangulate_gtsam(Rt_inverse_gtsam, calib_mat_gtsam, link, link_loop = False):
    if not link_loop:
        last_left_img_xy = link.left_keypoint()
        last_right_img_xy = link.right_keypoint()
    else:
        last_left_img_xy = link.left_keypoint_kf()
        last_right_img_xy = link.right_keypoint_kf()
    current_frame_camera_left = gtsam.StereoCamera(Rt_inverse_gtsam, calib_mat_gtsam)
    stereo_pt_gtsam = gtsam.StereoPoint2(last_left_img_xy[0], last_right_img_xy[0], last_left_img_xy[1])
    triangulate_p3d_gtsam = current_frame_camera_left.backproject(stereo_pt_gtsam)
    return triangulate_p3d_gtsam


def triangulate_from_gtsam(mat_gtsam, calib_mat_gtsam, link):
    inversed_pose = mat_gtsam.inverse()
    curr_stereo_cam_gtsam = gtsam.StereoCamera(inversed_pose, calib_mat_gtsam)
    stereo_pt = gtsam.StereoPoint2(link.get_x_left(), link.get_x_right(), link.get_y())
    p3d = curr_stereo_cam_gtsam.backproject(stereo_pt)
    return p3d

def find_projection_factor_with_largest_initial_error(graph, initial_values):
    max_error = -1
    max_error_factor = None
    for i in range(graph.size()):
        factor = graph.at(i)
        error = factor.error(initial_values)
        if error > max_error:
            max_error = error
            max_error_factor = factor
    return max_error_factor, max_error


# todo: check if the keys correctly
def print_projection_details(factor, values, K):
    print(factor.keys())
    key_c = factor.keys()[0]
    key_q = factor.keys()[1]

    pose_c = values.atPose3(key_c)
    point_q = values.atPoint3(key_q)

    # Print initial error
    error = factor.error(values)
    print(f"Initial error for camera key {key_c} and point key {key_q}: {error}")

    # Initialize StereoCamera with initial pose
    stereo_camera = gtsam.StereoCamera(pose_c, K)

    # Project the initial position of q
    projected_point = stereo_camera.project(point_q)

    # Get the measurement
    measurement = factor.measured()
    print(f"Measurement is type: {type(measurement)}")
    print(f"projected_point.uL() is type  {type(projected_point.uL())}")
    # Print the projections and the measurement
    print(f"Left projection: {projected_point.uL()}, {projected_point.v()}")
    print(f"Right projection: {projected_point.uR()}, {projected_point.v()}")
    print(f"Measurement: {measurement}")
    projected_point_numpy_left = np.array([projected_point.uL(), projected_point.v()])
    projected_point_numpy_right = np.array([projected_point.uR(), projected_point.v()])
    measurement_numpy_left = np.array([measurement.uL(), measurement.v()])
    measurement_numpy_right = np.array([measurement.uR(), measurement.v()])
    #load images and show the numpy points on images
    img0, img1 = read_images_from_dataset(9)

    # Plot both the left and right images in one call to plt.show() per figure
    fig_left, ax_left = plt.subplots()
    ax_left.imshow(img0, cmap='gray')
    ax_left.scatter(measurement_numpy_left[0], measurement_numpy_left[1], c='r', label='Measurement Left')
    ax_left.scatter(projected_point_numpy_left[0], projected_point_numpy_left[1], c='b', label='Projected Point Left')
    ax_left.legend()

    fig_right, ax_right = plt.subplots()
    ax_right.imshow(img1, cmap='gray')
    ax_right.scatter(measurement_numpy_right[0], measurement_numpy_right[1], c='r', label='Measurement Right')
    ax_right.scatter(projected_point_numpy_right[0], projected_point_numpy_right[1], c='b',
                     label='Projected Point Right')
    ax_right.legend()

    # Show both figures together
    plt.show()




    # # show the points and the measturments on img0 = left
    # fig, ax = plt.subplots()
    # ax.imshow(img0, cmap='gray')
    # ax.scatter(measurement_numpy_left[0], measurement_numpy_left[1], c='r', label='Measurement')
    # ax.scatter(projected_point_numpy_left[0], projected_point_numpy_left[1], c='b', label='Projected Point')
    # ax.legend()
    # plt.show()
    # # show the points and the measturments on img1 = right
    # fig, ax = plt.subplots()
    # ax.imshow(img1, cmap='gray')
    # ax.scatter(measurement_numpy_right[0], measurement_numpy_right[1], c='r', label='Measurement')
    # ax.scatter(projected_point_numpy_right[0], projected_point_numpy_right[1], c='b', label='Projected Point')
    # ax.legend()
    # plt.show()
    #




    # # show the points and the measturments on img0 = left
    # fig, ax = plt.subplots()
    # ax.imshow(img0, cmap='gray')
    # ax.scatter(measurement[0], measurement[1], c='r', label='Measurement')
    # ax.scatter(projected_point.uL(), projected_point.v(), c='b', label='Projected Point')
    # ax.legend()
    # plt.show()
    # # show the points and the measturments on img1 = right
    # fig, ax = plt.subplots()
    # ax.imshow(img1, cmap='gray')
    # ax.scatter(measurement[2], measurement[3], c='r', label='Measurement')
    # ax.scatter(projected_point.uR(), projected_point.v(), c='b', label='Projected Point')
    # ax.legend()
    # plt.show()


    # Compute distances
    left_distance = np.linalg.norm([projected_point.uL() - measurement.uL(), projected_point.v() - measurement.v()])
    right_distance = np.linalg.norm([projected_point.uR() - measurement.uR(), projected_point.v() - measurement.v()])

    # Print distances
    print(f"Distance from measurement (left): {left_distance}")
    print(f"Distance from measurement (right): {right_distance}")


# todo: do the keys to the plot dibur
def plot_3d_trajectory(values, title="3D Trajectory"):
    poses = gtsam.utilities.allPose3s(values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(poses.size()):
        pose = poses.atPose3(i)
        gtsam_plot.plot_pose3(ax, pose, 1)

    plt.title(title)
    plt.show()


def plot_2d_scene(values, keys, points, title="2D Scene"):
    fig, ax = plt.subplots()

    # Plot camera positions
    for key in keys:
        pose = values.atPose3(key)
        position = pose.translation()
        ax.plot(position.x(), position.y(), 'bo')
        ax.text(position.x(), position.y(), str(key), color='blue')

    # Plot points
    for point_key in points:
        point = values.atPoint3(point_key)
        ax.plot(point.x(), point.y(), 'ro')
        ax.text(point.x(), point.y(), str(point_key), color='red')

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()





def plot_cameras_path(path0, path1):
    # Create a plot
    plt.figure(figsize=(8, 6))
    if path0 is not None:
        x, _, z = zip(*path0)
        plt.plot(x, z, marker='o', linestyle='-', color='r', alpha=0.04, label='landmarks')
    if path1 is not None:
        x, _, z = zip(*path1)
        plt.plot(x, z, marker='o', linestyle='-', color='b', alpha=0.04, label='Computed Positions')

    # Add titles and labels
    plt.xlim([-200,200])
    plt.ylim([-200, 200])
    plt.title('2D Path Plot')
    plt.xlabel('X coordinates')
    plt.ylabel('Z coordinates')
    plt.legend()

    # Show the plot
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f"plot_camera_3d_path")
    plt.show()


# def plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth(cameras=None, landmarks=None,
#                                                                        initial_estimate_poses=None, cameras_gt=None,
#                                                                        title="",
#                                                                        loops=None, numbers=False,
#                                                                        mahalanobis_dist=None, inliers_perc=None):
#     """
#     Compare the left cameras relative 2d positions to the ground truth
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     first_legend = []
#
#     landmarks_title = "and landmarks " if landmarks is not None else ""
#     loops_title = ""
#     dist_title = "Dist = squared mahalanobis distance " if mahalanobis_dist is not None else ""
#
#     ax.set_title(f"{title} Left cameras {landmarks_title}2d trajectory of {PROBLEM_PROMPT} bundles.\n{dist_title}"
#                  f"{loops_title}")
#
#     if landmarks is not None:
#         a = ax.scatter(landmarks[:, 0], landmarks[:, 2], s=1, c='orange', label="Landmarks")
#         first_legend.append(a)
#
#     if initial_estimate_poses is not None:
#         first_legend.append(ax.scatter(initial_estimate_poses[:, 0], initial_estimate_poses[:, 2], s=1, c='pink', label="Initial estimate"))
#
#     if cameras is not None:
#         first_legend.append(ax.scatter(cameras[:, 0], cameras[:, 2], s=1, c='red', label="Optimized cameras"))
#
#     if cameras_gt is not None:
#         first_legend.append(ax.scatter(cameras_gt[:, 0], cameras_gt[:, 2], s=1, c='cyan', label="Cameras ground truth"))
#
#     # Mark loops
#     if loops is not None:
#         for cur_cam, prev_cams in loops:
#             y_diff = 0 if abs(cameras[:, 0][cur_cam] - cameras[:, 0][cur_cam - 1]) < 2 else 15
#             x_diff = 0 if abs(cameras[:, 2][cur_cam] - cameras[:, 2][cur_cam - 1]) < 2 else 20
#
#             if numbers:
#                 ax.text(cameras[:, 0][cur_cam] - x_diff, cameras[:, 2][cur_cam] - y_diff, cur_cam, size=7, fontweight="bold")
#             ax.scatter(cameras[:, 0][cur_cam], cameras[:, 2][cur_cam], s=3, c='black')
#
#             if numbers:
#                 for prev_cam in prev_cams:
#                     ax.text(cameras[:, 0][prev_cam], cameras[:, 2][prev_cam], prev_cam, size=7, fontweight="bold")
#             ax.scatter(cameras[:, 0][prev_cams], cameras[:, 2][prev_cams], s=1, c='black')
#
#     if landmarks is not None:
#         ax.set_xlim(-5, 5)
#         ax.set_ylim(-5, 15)
#
#     if loops is not None:
#         plt.subplots_adjust(left=0.25, bottom=0.08, right=0.95, top=0.9)
#
#     landmarks_txt = "and landmarks" if landmarks is not None else ""
#     mahalanobis_dist_and_inliers = f"Dist: {mahalanobis_dist}; Inliers: {inliers_perc}%\n"
#
#     len_loops = None
#     if loops is not None:
#         loops_details = "\n".join([str(i) + ")  " + str(cur_cam) + ": " + ",".join([str(prev_cam) for prev_cam in prev_cams])
#                                    for i, (cur_cam, prev_cams) in enumerate(loops)])
#
#         loops_txt = mahalanobis_dist_and_inliers + loops_details
#         # y = -65 + 9 * len(loops)
#         plt.text(-360, -67, loops_txt, fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
#         len_loops = len(loops)
#
#     first_legend = plt.legend(handles=first_legend, loc='upper left', prop={'size': 7})
#     plt.gca().add_artist(first_legend)
#
#     fig.savefig("Results.png")
#     plt.close(fig)
def plot_left_cams_and_landmarks_one_bundle(cameras, landmarks):
    """
    Plot the 2D (XZ-plane) trajectory of the cameras and the 3D points for both cameras and landmarks.
    Args:
    - cameras (list or np.array): Nx3 array or list of camera positions.
    - landmarks (list or np.array): Mx3 array or list of landmark positions.
    """

    # Convert list of string inputs to numpy arrays if necessary
    if isinstance(cameras, list):
        cameras = np.array([list(map(float, cam.split(','))) for cam in cameras])
    if isinstance(landmarks, list):
        landmarks = np.array([list(map(float, lm.split(','))) for lm in landmarks])

    # Ensure cameras and landmarks are correctly shaped as Nx3 arrays
    cameras = np.array(cameras).reshape(-1, 3)
    landmarks = np.array(landmarks).reshape(-1, 3)

    # Create 2D plot
    plt.figure(figsize=(10, 6))

    # Plot cameras in 2D (XZ-plane)
    plt.scatter(cameras[:, 0], cameras[:, 2], color='red', label='Cameras')

    # Plot landmarks in 2D (XZ-plane)
    plt.scatter(landmarks[:, 0], landmarks[:, 2], color='blue', label='Landmarks')

    # Set axis limits
    plt.xlim([-20, 20])
    plt.ylim([-20, 20])

    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('2D View of Cameras and Landmarks (XZ-plane)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def plot_left_cam_2d_trajectory_and_3d_points_compared_to_ground_truth_full(cameras, landmarks):
    """
    Plot the 2D (XZ-plane) trajectory of the cameras and the 3D points for both cameras and landmarks.
    Args:
    - cameras (list or np.array): Nx3 array or list of camera positions.
    - landmarks (list or np.array): Mx3 array or list of landmark positions.
    """

    # Convert list of string inputs to numpy arrays if necessary
    if isinstance(cameras, list):
        cameras = np.array([list(map(float, cam.split(','))) for cam in cameras])
    if isinstance(landmarks, list):
        landmarks = np.array([list(map(float, lm.split(','))) for lm in landmarks])

    # Ensure cameras and landmarks are correctly shaped as Nx3 arrays
    cameras = np.array(cameras).reshape(-1, 3)
    landmarks = np.array(landmarks).reshape(-1, 3)

    # Create 2D plot
    plt.figure(figsize=(10, 6))

    # Plot cameras in 2D (XZ-plane)
    plt.scatter(cameras[:, 0], cameras[:, 2], color='red', label='Cameras')

    # Plot landmarks in 2D (XZ-plane)
    plt.scatter(landmarks[:, 0], landmarks[:, 2], color='blue', label='Landmarks')

    # Set axis limits
    plt.xlim([-20, 50])
    plt.ylim([-5, 100])

    # Set plot labels and title
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('2D View of Cameras and Landmarks (XZ-plane)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def projection_factors_error(factors, values):
    """
    create a list of errors for each factor in the list of factors
    Args:
        factors: list of factors - gtsam object
        values: gtsam.Values

    Returns: np.array of errors

    """
    errors = []
    for factor in factors:
        errors.append(factor.error(values))

    return np.array(errors)


def plot_re_projection_error(left_proj_dist, right_proj_dist, selected_track):
    """
    Plots the re-projection error for the left and right cameras
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Reprojection error for track: {selected_track}")
    # Plotting the scatter plot
    ax.scatter(range(len(left_proj_dist)), left_proj_dist, color='orange', label='Left Projections')
    ax.scatter(range(len(right_proj_dist)), right_proj_dist, color='blue', label='Right Projections')
    # Plotting the continuous line
    ax.plot(range(len(left_proj_dist)), left_proj_dist, linestyle='-', color='orange')
    ax.plot(range(len(right_proj_dist)), right_proj_dist, linestyle='-', color='blue')
    ax.set_ylabel('Error')
    ax.set_xlabel('Frames')
    ax.legend()
    fig.savefig("Reprojection_error.png")
    plt.close(fig)


def plot_factor_error_graph(factor_projection_errors, frame_idx_triangulate):
    """
    Plots re projection error
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    frame_title = "Last" if frame_idx_triangulate == -1 else "First"
    ax.set_title(f"Factor error from {frame_title} frame")
    plt.scatter(range(len(factor_projection_errors)), factor_projection_errors, label="Factor")
    plt.legend(loc="upper right")
    plt.ylabel('Error')
    plt.xlabel('Frames')
    fig.savefig(f"Factor error graph for {frame_title} frame.png")
    plt.close(fig)


def plot_factor_error_as_function_of_projection_error(projection_error: np.array, factor_error: np.array,
                                                      title='Factor Error as a function of Re-Projection Error'):
    """
    Plots the factor error as a function of the projection error.

    :param projection_error: NumPy array of projection errors for each frame.
    :param factor_error: NumPy array of factor errors for each frame.
    :param title: Title of the plot.
    """
    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(projection_error, factor_error, color='b')
    # Add labels and title
    plt.xlabel('Projection Error')
    plt.ylabel('Factor Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("Factor error as a function of a Re-projection error")



def calculate_and_plot_error_between_est_cams_and_truth_cams(cams_3d, cams_truth_3d):
    """
    Calculate and plot the error between the estimated cameras and the ground truth cameras.
    """
    errors = []
    for est, gt in zip(cams_3d, cams_truth_3d):
        # est_trans = np.array([est.translation().x(), est.translation().y(), est.translation().z()])
        # gt_trans = gt[:3, 3]
        error = np.linalg.norm(est - gt)
        errors.append(error)
    plt.figure(figsize=(12, 6))
    plt.plot(range(0, len(errors) * 20, 20), errors)
    plt.title('Keyframe Localization Error Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Error (meters)')
    plt.grid(True)
    plt.show()


def print_ancoring_error(all_optimized_values, window, keyframe_id):
    """
    Print the anchoring factor final error
    Args:
        all_optimized_values: values of all the optimized factors in the window
        window: the bundle window
        keyframe_id: the keyframe id for getting the error to.

    ->None
    """
    # Print the anchoring factor final error
    if window.graph.size() > 0:
        anchoring_factor = window.graph.at(0)
        anchoring_error = anchoring_factor.error(window.get_optimized_values())
        print(f"Anchoring factor final error: {anchoring_error}")
    else:
        print("No anchoring factor found in the last window.")
