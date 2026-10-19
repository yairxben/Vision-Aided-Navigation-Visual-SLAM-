import os
import random
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


DATASET_PATH = os.path.join(os.getcwd(), 'dataset', 'sequences', '00')

FLANN_INDEX_KDTREE = 1

CANONICAL_SIFT_PARAMETERS = dict(contrastThreshold=0.02, edgeThreshold=10)
CANONICAL_FLANN_INDEX_PARAMETERS = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
CANONICAL_FLANN_SEARCH_PARAMETERS = dict(checks=128)


def create_detector():
    """Create a fresh detector using the approved Exercise 3 configuration."""
    return cv2.SIFT_create(**CANONICAL_SIFT_PARAMETERS)


def create_matcher():
    """Create a fresh matcher using the approved Exercise 3 configuration."""
    return cv2.FlannBasedMatcher(
        dict(CANONICAL_FLANN_INDEX_PARAMETERS),
        dict(CANONICAL_FLANN_SEARCH_PARAMETERS),
    )


DETECTOR = create_detector()
MATCHER = create_matcher()


NUM_FRAMES = 3360
MAX_DEVIATION = 2
Epsilon = 1e-10


def match_reciprocal(descriptors_back, descriptors_front, matcher=MATCHER):
    """Return forward descriptor matches that agree with reverse matching."""
    forward_matches = matcher.match(descriptors_back, descriptors_front)
    reverse_matches = matcher.match(descriptors_front, descriptors_back)
    reverse_map = {match.queryIdx: match.trainIdx for match in reverse_matches}
    return sorted(
        [
            match
            for match in forward_matches
            if reverse_map.get(match.trainIdx) == match.queryIdx
        ],
        key=lambda match: match.queryIdx,
    )




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
        if (abs(pt_left.pt[1] - pt_right.pt[1]) <= MAX_DEVIATION
                and pt_left.pt[0] - pt_right.pt[0] > 0):
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
        track_matches = match_reciprocal(back_left_desc, front_left_desc)
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
