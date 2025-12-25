import cv2
import numpy as np
from algorithms_library import (
    cloud_points_triangulation,
    read_images,
    detect_keypoints, reject_matches,
    reject_matches_and_remove_keypoints, get_stereo_matches_with_filtered_keypoints, read_cameras,
    triangulation_process, create_dict_to_pnp, create_in_out_l1_dict,
    get_stereo_matches_with_filtered_keypoints_avish_test, create_dict_to_pnp_avish_test
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NUM_FRAMES_KITTI = 3360

NUMBER_PTS_FOR_PNP = 4


def get_matches_without_rejection_from_imgs(first_img, second_img):
    keypoints_left, descriptors_left = detect_keypoints(first_img)
    keypoints_right, descriptors_right = detect_keypoints(second_img)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_left, descriptors_left)
    return keypoints_left, keypoints_right, matches, descriptors_left


def get_matches_without_rejection(descriptors_left0, descriptors_left1):
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_left0, descriptors_left1)
    return matches


def get_matches(first_img, second_img):
    keypoints_left, descriptors_left = detect_keypoints(first_img)
    keypoints_right, descriptors_right = detect_keypoints(second_img)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors_left, descriptors_right)
    _, inliers, _, keypoints_first_filtered, keypoints_second_filtered = reject_matches_and_remove_keypoints(
        keypoints_left, keypoints_right, matches)
    return keypoints_first_filtered, keypoints_second_filtered, inliers, descriptors_left


def find_common_keypoints(matches_01, matches_02, matches_03):
    list01 = []
    list02 = []
    list03 = []

    match_dict_01 = {m.queryIdx: m for m in matches_01}
    match_dict_02 = {m.trainIdx: m for m in matches_02}

    for m in matches_03:
        if m.queryIdx in match_dict_01 and m.trainIdx in match_dict_02:
            list01.append(match_dict_01[m.queryIdx])
            list02.append(match_dict_02[m.trainIdx])
            list03.append(m)

    return list01, list02, list03


def compute_extrinsic_matrix(points3D, points2D, K, flag=cv2.SOLVEPNP_P3P):
    Rt, t_left1 = None, None
    succes, rvec, tvec = cv2.solvePnP(points3D, points2D, K, None, flags=flag)
    if succes:
        R, _ = cv2.Rodrigues(rvec)
        Rt = np.hstack((R, tvec))

        t_left1 = tvec.flatten()
    return Rt, t_left1


def plot_camera_positions(extrinsic_matrices):
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
    plt.xlim(-1, 20)
    plt.ylim(-1, 10)

    plt.grid(True)
    # plt.axis('equal')
    plt.legend()
    plt.show()


def rectify(matches, key_points1, key_points2):
    idx_kp1 = {}
    # todo check of query is frame1
    matches_i_in_img1 = [m.queryIdx for m in matches]
    matches_i_in_img2 = [m.trainIdx for m in matches]
    for i, j in zip(matches_i_in_img1, matches_i_in_img2):
        if abs(key_points1[i].pt[1] - key_points2[j].pt[1]) < 2:
            idx_kp1[i] = j
    return idx_kp1


def q1():
    cloud_points_triangulation(0)
    cloud_points_triangulation(1)




def project_points(points_3D, K, Rt):
    """
    Projects 3D points to 2D using camera matrix K and extrinsic matrix Rt
    Args:
    - points_3D: numpy array of shape (N, 3) containing 3D points (homogeneous)
    - K: intrinsic camera matrix (3x3)
    - Rt: extrinsic matrix (3x4)
    Returns:
    - points_2D: numpy array of shape (N, 2) containing 2D projected points
    """
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    points_2D_hom = (points_3D_hom @ Rt.T @ K.T)
    points_2D = (points_2D_hom[:, :2] / points_2D_hom[:, [-1]])
    #the editing of avishay.
    # points_2D = (points_2D_hom[:, :2].T / points_2D_hom[:, -1])
    # points_2D = points_2D.T
    return points_2D


def project_points_version2(points_3D, K, Rt):
    """
    Projects 3D points to 2D using camera matrix K and extrinsic matrix Rt

    Args:
    - points_3D: numpy array of shape (N, 3) containing 3D points (homogeneous)
    - K: intrinsic camera matrix (3x3)
    - Rt: extrinsic matrix (3x4)

    Returns:
    - points_2D: numpy array of shape (N, 2) containing 2D projected points
    """
    points_3D_hom = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))

    points_2D_hom = (points_3D_hom @ Rt.T @ K.T)
    points_2D = (points_2D_hom[:, :2].T / points_2D_hom[:, -1])
    points_2D = points_2D.T
    return points_2D


def plot_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches, supporters):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(cv2.cvtColor(img_left0, cv2.COLOR_BGR2RGB))
    axes[1].imshow(cv2.cvtColor(img_left1, cv2.COLOR_BGR2RGB))

    for i, match in enumerate(matches):
        pt_left0 = keypoints_left0[match.queryIdx].pt
        pt_left1 = keypoints_left1[match.trainIdx].pt

        color = 'cyan' if i in supporters else 'red'

        axes[0].plot(pt_left0[0], pt_left0[1], 'o', markersize=5, color=color)
        axes[1].plot(pt_left1[0], pt_left1[1], 'o', markersize=5, color=color)

    axes[0].set_title('Left Image 0')
    axes[1].set_title('Left Image 1')

    plt.show()


def plot_matches_with_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches, supporters_indices):
    img_left0_supporters = cv2.drawMatches(img_left0, keypoints_left0, img_left0, keypoints_left0, matches, None,
                                           matchesMask=supporters_indices, matchColor=(255, 0, 255),
                                           singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
    img_left1_supporters = cv2.drawMatches(img_left1, keypoints_left1, img_left1, keypoints_left1, matches, None,
                                           matchesMask=supporters_indices, matchColor=(255, 0, 255),
                                           singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)

    # Convert BGR images to RGB for displaying with matplotlib
    img_left0_supporters_rgb = cv2.cvtColor(img_left0_supporters, cv2.COLOR_BGR2RGB)
    img_left1_supporters_rgb = cv2.cvtColor(img_left1_supporters, cv2.COLOR_BGR2RGB)

    # Plotting
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img_left0_supporters_rgb)
    plt.title('Left Image 0 with Matches and Supporters')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_left1_supporters_rgb)
    plt.title('Left Image 1 with Matches and Supporters')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



def find_supporters(points_3D, keypoints_left1, keypoints_right1, K, Rt_10,
                    Rt_11, threshold=2):
    """ Find supporters of the transformation T within a given distance threshold """
    supporters = []
    # Project points to all four images
    projected_left1 = project_points(points_3D, K, Rt_10)
    projected_right1 = project_points(points_3D, K, Rt_11)
    #
    #
    # keypoints_left1_np = np.array(keypoints_left1)
    # keypoints_left1_np = np.transpose(keypoints_left1_np)
    #
    # keypoints_right1_np = np.array(keypoints_right1)
    # keypoints_right1_np = np.transpose(keypoints_right1_np)
    #
    # #
    # diff_left1 = np.sum(np.abs(projected_left1 - keypoints_left1_np), axis=0)
    # diff_right1 = np.sum(np.abs(projected_right1 - keypoints_right1_np), axis=0)
    # # print(diff_right1)
    # # print(diff_left1)
    # # mask: np.ndarray = ((diff_left1 <= threshold) * (diff_right1 <= threshold))
    # mask = (diff_left1 <= threshold) & (diff_right1 <= threshold)
    # # print(mask)
    # return np.where(mask)[0].tolist()
    # # return np.array([i for i in mask if i]).tolist()


    # Check distances for each point
    for i, (pt_left1, pt_right1) in enumerate(
            zip(keypoints_left1, keypoints_right1)):
        d_left1 = np.linalg.norm(projected_left1[i] - pt_left1)
        d_right1 = np.linalg.norm(projected_right1[i] - pt_right1)

        if d_left1 <= threshold and d_right1 <= threshold:
            supporters.append(i)

    #the editing of avishayin version2
    # supporters_l1 = np.power(keypoints_left1 - projected_left1, 2).sum(axis=1) <= 2 ** 2
    # supporters_l2 = np.power(keypoints_right1 - projected_right1, 2).sum(axis=1) <= 2 ** 2
    # x = np.logical_and(supporters_l1, supporters_l2).nonzero()
    # return x

    return supporters





def find_supporters_version2(points_3D, keypoints_left1, keypoints_right1, K, Rt_10,
                    Rt_11, threshold=2):
    """ Find supporters of the transformation T within a given distance threshold """
    supporters = []

    # Project points to all four images
    projected_left1 = project_points(points_3D, K, Rt_10)
    projected_right1 = project_points(points_3D, K, Rt_11)
    #
    #
    # keypoints_left1_np = np.array(keypoints_left1)
    # keypoints_left1_np = np.transpose(keypoints_left1_np)
    #
    # keypoints_right1_np = np.array(keypoints_right1)
    # keypoints_right1_np = np.transpose(keypoints_right1_np)
    #
    # #
    # diff_left1 = np.sum(np.abs(projected_left1 - keypoints_left1_np), axis=0)
    # diff_right1 = np.sum(np.abs(projected_right1 - keypoints_right1_np), axis=0)
    # # print(diff_right1)
    # # print(diff_left1)
    # # mask: np.ndarray = ((diff_left1 <= threshold) * (diff_right1 <= threshold))
    # mask = (diff_left1 <= threshold) & (diff_right1 <= threshold)
    # # print(mask)
    # return np.where(mask)[0].tolist()
    # # return np.array([i for i in mask if i]).tolist()

    supporters_l1 = np.power(keypoints_left1 - projected_left1, 2).sum(axis=1) <= 2 ** 2
    supporters_l2 = np.power(keypoints_right1 - projected_right1, 2).sum(axis=1) <= 2 ** 2
    x = np.logical_and(supporters_l1, supporters_l2).nonzero()
    return x

    # # Check distances for each point
    # for i, (pt_left1, pt_right1) in enumerate(
    #         zip(keypoints_left1, keypoints_right1)):
    #     d_left1 = np.linalg.norm(projected_left1[i] - pt_left1)
    #     d_right1 = np.linalg.norm(projected_right1[i] - pt_right1)
    #
    #     if d_left1 <= threshold and d_right1 <= threshold:
    #         supporters.append(i)
    #
    # return supporters


def plot_point_clouds(points3D_pair0, points3D_pair1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the transformed points3D_pair0
    ax.scatter(points3D_pair0[:, 0], points3D_pair0[:, 1], points3D_pair0[:, 2], c='r', marker='o', label='Pair 0')

    # Plot the points3D_pair1
    ax.scatter(points3D_pair1[:, 0], points3D_pair1[:, 1], points3D_pair1[:, 2], c='b', marker='^', label='Pair 1')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    ax.set_title('Transformed Point Cloud Pair 0 and Point Cloud Pair 1')

    plt.show()


# Add plotting inliers and outliers
def plot_inliers_outliers(img_left0, img_left1, filtered_keypoints_left0, filtered_keypoints_left1, matches, inliers,
                          in_out_l1_dict):
    inlier_indices = set(inliers)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))

    ax0.imshow(cv2.cvtColor(img_left0, cv2.COLOR_BGR2RGB))
    ax1.imshow(cv2.cvtColor(img_left1, cv2.COLOR_BGR2RGB))

    for match in matches:
        color = 'r'
        if (in_out_l1_dict[filtered_keypoints_left1[match.trainIdx]]):
            color = 'g'
        pt_left0 = filtered_keypoints_left0[match.queryIdx].pt
        pt_left1 = filtered_keypoints_left1[match.trainIdx].pt

        ax0.plot(pt_left0[0], pt_left0[1], 'o', markersize=5, color=color)
        ax1.plot(pt_left1[0], pt_left1[1], 'o', markersize=5, color=color)

    ax0.set_title('Left Image 0 - Inliers and Outliers')
    ax1.set_title('Left Image 1 - Inliers and Outliers')

    plt.show()


def read_ground_truth_poses(file_path):
    """
    Reads ground-truth extrinsic matrices from a file.

    Args:
    - file_path: Path to the ground-truth poses file.

    Returns:
    - ground_truth_poses: List of extrinsic matrices (4x4) as numpy arrays.
    """
    ground_truth_poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            pose = np.vstack((pose, [0, 0, 0, 1]))  # Convert to 4x4 matrix
            ground_truth_poses.append(pose)
    return ground_truth_poses


def extract_camera_locations(transformations):
    """
    Extracts camera locations from extrinsic matrices.

    Args:
    - transformations: List of extrinsic matrices (4x4) as numpy arrays.

    Returns:
    - locations: List of camera locations as numpy arrays.
    """
    locations = []
    for Rt in transformations:
        R = Rt[:3, :3]
        t = Rt[:3, 3]
        location = -np.linalg.inv(R).dot(t)
        locations.append(location)
    return np.array(locations)




def q6_with_avish():
    # Read ground-truth extrinsic matrices
    ground_truth_file = 'C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/poses/00.txt'
    ground_truth_poses = read_ground_truth_poses(ground_truth_file)
    _, T_left, T_right = read_cameras(
        'C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt')

    frame_transformations = [T_left]
    print(T_left)
    for i in range(20):
        T, T_left, T_right = ransac_algorithm_online_with_avish_dict(i, T_left, T_right)
        T = np.vstack((T, np.array([[0, 0, 0, 1]])))
        print(T)
        # print(ground_truth_poses[i+1])
        frame_transformations.append(T)

    # Extract camera locations from frame transformations
    estimated_locations = extract_camera_locations(frame_transformations)

    # Extract camera locations from ground truth poses
    ground_truth_locations = extract_camera_locations(ground_truth_poses[:20])

    plot_root_ground_truth_and_estimate(estimated_locations, ground_truth_locations)




def q6():
    # Read ground-truth extrinsic matrices
    ground_truth_file = 'C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/poses/00.txt'
    ground_truth_poses = read_ground_truth_poses(ground_truth_file)
    _, T_left, T_right = read_cameras(
        'C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt')

    frame_transformations = [T_left]
    for i in range(20):
        T, T_left, T_right = ransac_algorithm_online(i, T_left, T_right)
        T = np.vstack((T, np.array([[0, 0, 0, 1]])))
        print(T)
        # print(ground_truth_poses[i+1])
        frame_transformations.append(T)

    # Extract camera locations from frame transformations
    estimated_locations = extract_camera_locations(frame_transformations)

    # Extract camera locations from ground truth poses
    ground_truth_locations = extract_camera_locations(ground_truth_poses[:20])

    plot_root_ground_truth_and_estimate(estimated_locations, ground_truth_locations)


def plot_root_ground_truth_and_estimate(estimated_locations, ground_truth_locations):
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
    plt.show()


def ransac_algorithm_online(idx, Rt_00, Rt_01):
    print(idx)
    img_left0, img_right0 = read_images(idx)
    img_left1, img_right1 = read_images(idx + 1)
    # Get matches for pair 0
    filtered_keypoints_left0, filtered_keypoints_right0, desc_00, _, matches_00, keypoints_left0, keypoints_right0 = (
        get_stereo_matches_with_filtered_keypoints(img_left0, img_right0))

    # Get matches for pair 1

    filtered_keypoints_left1, filtered_keypoints_right1, desc_10, _, matches_11, keypoints_left1, keypoints_right1 = (
        get_stereo_matches_with_filtered_keypoints(img_left1, img_right1))

    # Get matches between left0 and left1
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_01 = bf.match(desc_00, desc_10)

    # Perform cloud triangulation for pair 0
    # k, Rt_00, Rt_01 = (
    #     read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    k, _, P_right = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = (
        triangulation_process(Rt_00, Rt_01, matches_00, k, keypoints_left0, keypoints_right0, plot=False))
    # Create the dictionary for PnP
    points_3D, points_2D_left0, points_2D_left1, points_2D_right1 = create_dict_to_pnp(matches_01, matches_11,
                                                                                       filtered_keypoints_left1,
                                                                                       keypoints_left0, keypoints_left1,
                                                                                       keypoints_right1,
                                                                                       points_3D_custom)

    max_T, group_idx = ransac_pnp(points_3D, points_2D_left1, k, Rt_01, points_2D_right1)
    if (max_T is None):
        print("This frame is create none matrix")
        return

    # Convert 3x4 matrices to 4x4 homogeneous transformation matrices
    max_T_homogeneous = np.vstack([max_T, [0, 0, 0, 1]])
    P_right_homogeneous = np.vstack([P_right, [0, 0, 0, 1]])

    # Compute the transformation matrix for the right image
    max_T_right_homogeneous = np.dot(P_right_homogeneous, max_T_homogeneous)

    # Extract the 3x4 transformation matrix for the right image
    max_T_right = max_T_right_homogeneous[:3, :]
    plot_camera_positions([Rt_00, Rt_01, max_T, max_T_right])

    return max_T, max_T, max_T_right


def q2():
    img_left0, img_right0 = read_images(0)
    img_left1, img_right1 = read_images(1)
    # Get matches for pair 0
    filtered_keypoints_left0, filtered_keypoints_right0, desc_00, _, matches_00, keypoints_left0, keypoints_right0 = (
        get_stereo_matches_with_filtered_keypoints(img_left0, img_right0))

    # Get matches for pair 1

    filtered_keypoints_left1, filtered_keypoints_right1, desc_10, _, matches_11, keypoints_left1, keypoints_right1 = (
        get_stereo_matches_with_filtered_keypoints(img_left1, img_right1))

    # Get matches between left0 and left1
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_01 = bf.match(desc_00, desc_10)

    # Perform cloud triangulation for pair 0 (assuming this was already done in q1)
    k, Rt_00, Rt_01 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = (
        triangulation_process(Rt_00, Rt_01, matches_00, k, keypoints_left0, keypoints_right0, False))
    # Create the dictionary for PnP
    points_3D, points_2D_left0, points_2D_left1, points_2D_right1 = create_dict_to_pnp(matches_01, matches_11,
                                                                                       filtered_keypoints_left1,
                                                                                       keypoints_left0, keypoints_left1,
                                                                                       keypoints_right1,
                                                                                       points_3D_custom)

    # Use the first 4 points for extrinsic matrix computation
    Rt_10, t_10 = compute_extrinsic_matrix(points_3D[:4], points_2D_left1[:4], k)

    # Compute Rt for right0 (already available as Rt_01)
    R_01 = Rt_01[:, :3]
    t_01 = Rt_01[:, 3]
    R_11 = np.dot(Rt_10[:, :3], R_01)
    t_11 = np.dot(Rt_10[:, :3], t_01) + t_10
    Rt_11 = np.hstack((R_11, t_11.reshape(-1, 1)))

    # Plot camera positions
    plot_camera_positions([Rt_00, Rt_10, Rt_01, Rt_11])

    # Find supporters of the transformation
    supporters = find_supporters(points_3D, points_2D_left1, points_2D_right1,
                                 k, Rt_10, Rt_11)

    plot_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches_01, supporters)
    max_T, group_idx = ransac_pnp(points_3D, points_2D_left1, k, Rt_01, points_2D_right1)
    if (max_T is None):
        print("This frame is create none matrix")
        return
    _, _, _, points_3D_l1r1_triangulation = cloud_points_triangulation(1)
    # Transform the point cloud for pair 0 using max_T
    points_3D_pair0_transformed = (max_T @ np.hstack((points_3D_custom, np.ones((points_3D_custom.shape[0], 1)))).T).T[
                                  :, :3]

    # Plot the two 3D point clouds
    plot_point_clouds(points_3D_pair0_transformed, points_3D_l1r1_triangulation)

    # Plot inliers and outliers on images left0 and left1
    in_out_l1_dict = create_in_out_l1_dict(group_idx, points_2D_left1, filtered_keypoints_left1)
    plot_inliers_outliers(img_left0, img_left1, filtered_keypoints_left0, filtered_keypoints_left1, matches_01,
                          group_idx, in_out_l1_dict)


def q6_in_range(start, end):
    # Read ground-truth extrinsic matrices
    ground_truth_file = 'C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/poses/00.txt'
    ground_truth_poses = read_ground_truth_poses(ground_truth_file)
    _, P_left, P_right = read_cameras(
        'C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt')

    frame_transformations = [P_left]
    T_left, T_right = P_left, P_right
    for i in range(end):
        T, T_left, T_right = ransac_algorithm_online(i, T_left, T_right)
        T = np.vstack((T, np.array([[0, 0, 0, 1]])))
        print(T)
        # print(ground_truth_poses[i+1])
        frame_transformations.append(T)

    # Extract camera locations from frame transformations
    estimated_locations = extract_camera_locations(frame_transformations)

    # Extract camera locations from ground truth poses
    ground_truth_locations = extract_camera_locations(ground_truth_poses)

    plot_root_ground_truth_and_estimate(estimated_locations[start:end], ground_truth_locations[start:end])


# def q2_in_range(previous_T, start, end):
#     img_left0, img_right0 = read_images(idx)
#     img_left1, img_right1 = read_images(idx+1)
#     # Get matches for pair 0
#     filtered_keypoints_left0, filtered_keypoints_right0, desc_00, _, matches_00, keypoints_left0, keypoints_right0 = (
#         get_stereo_matches_with_filtered_keypoints(img_left0, img_right0))
#
#     # Get matches for pair 1
#
#     filtered_keypoints_left1, filtered_keypoints_right1, desc_10, _, matches_11, keypoints_left1, keypoints_right1 = (
#         get_stereo_matches_with_filtered_keypoints(img_left1, img_right1))
#
#     # Get matches between left0 and left1
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches_01 = bf.match(desc_00, desc_10)
#
#     # Perform cloud triangulation for pair 0 (assuming this was already done in q1)
#     k, Rt_00, Rt_01 = (
#         read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
#     points_3D_custom, pts1, pts2 = (
#         triangulation_process(Rt_00, Rt_01, matches_00, k, keypoints_left0, keypoints_right0, False))
#     # Create the dictionary for PnP
#     points_3D, points_2D_left0, points_2D_left1, points_2D_right1 = create_dict_to_pnp(matches_01, matches_11,
#                                                                                        filtered_keypoints_left1,
#                                                                                        keypoints_left0, keypoints_left1,
#                                                                                        keypoints_right1,
#                                                                                        points_3D_custom)
#
#     # Use the first 4 points for extrinsic matrix computation
#     Rt_10, t_10 = compute_extrinsic_matrix(points_3D[:4], points_2D_left1[:4], k)
#
#     # Compute Rt for right0 (already available as Rt_01)
#     R_01 = Rt_01[:, :3]
#     t_01 = Rt_01[:, 3]
#     R_11 = np.dot(Rt_10[:, :3], R_01)
#     t_11 = np.dot(Rt_10[:, :3], t_01) + t_10
#     Rt_11 = np.hstack((R_11, t_11.reshape(-1, 1)))
#
#     # Plot camera positions
#     plot_camera_positions([Rt_00, Rt_10, Rt_01, Rt_11])
#
#     # Find supporters of the transformation
#     supporters = find_supporters(points_3D, points_2D_left1, points_2D_right1,
#                                  k, Rt_10, Rt_11)
#
#     plot_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches_01, supporters)
#     max_T, group_idx = ransac_pnp(points_3D, points_2D_left1, k, Rt_01, points_2D_right1)
#     if (max_T is None):
#         print("This frame is create none matrix")
#         return
#     _, _, _, points_3D_l1r1_triangulation = cloud_points_triangulation(1)
#     # Transform the point cloud for pair 0 using max_T
#     points_3D_pair0_transformed = (max_T @ np.hstack((points_3D_custom, np.ones((points_3D_custom.shape[0], 1)))).T).T[
#                                   :, :3]
#
#     # Plot the two 3D point clouds
#     plot_point_clouds(points_3D_pair0_transformed, points_3D_l1r1_triangulation)
#
#     # Plot inliers and outliers on images left0 and left1
#     in_out_l1_dict = create_in_out_l1_dict(group_idx, points_2D_left1, filtered_keypoints_left1)
#     plot_inliers_outliers(img_left0, img_left1, filtered_keypoints_left0, filtered_keypoints_left1, matches_01,
#                           group_idx, in_out_l1_dict)




def q2_tests(idx):
    img_left0, img_right0 = read_images(idx)
    img_left1, img_right1 = read_images(idx + 1)
    # Get matches for pair 0
    filtered_keypoints_left0, filtered_keypoints_right0, desc_00, _, matches_00, keypoints_left0, keypoints_right0 = (
        get_stereo_matches_with_filtered_keypoints(img_left0, img_right0))

    # Get matches for pair 1

    filtered_keypoints_left1, filtered_keypoints_right1, desc_10, _, matches_11, keypoints_left1, keypoints_right1 = (
        get_stereo_matches_with_filtered_keypoints(img_left1, img_right1))

    # Get matches between left0 and left1
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_01 = bf.match(desc_00, desc_10)

    # Perform cloud triangulation for pair 0 (assuming this was already done in q1)
    k, Rt_00, Rt_01 = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = (
        triangulation_process(Rt_00, Rt_01, matches_00, k, keypoints_left0, keypoints_right0, False))
    # Create the dictionary for PnP
    points_3D, points_2D_left0, points_2D_left1, points_2D_right1 = create_dict_to_pnp(matches_01, matches_11,
                                                                                       filtered_keypoints_left1,
                                                                                       keypoints_left0, keypoints_left1,
                                                                                       keypoints_right1,
                                                                                       points_3D_custom)

    # Use the first 4 points for extrinsic matrix computation
    # Rt_10, t_10 = compute_extrinsic_matrix(points_3D[:4], points_2D_left1[:4], k)
    Rt_10, _ = ransac_pnp(points_3D, points_2D_left1, k, Rt_01, points_2D_right1)
    t_10 = Rt_10[:, 3]
    # Compute Rt for right0 (already available as Rt_01)
    R_01 = Rt_01[:, :3]
    t_01 = Rt_01[:, 3]
    R_11 = np.dot(Rt_10[:, :3], R_01)
    t_11 = np.dot(Rt_10[:, :3], t_01) + t_10
    Rt_11 = np.hstack((R_11, t_11.reshape(-1, 1)))

    # Plot camera positions
    plot_camera_positions([Rt_00, Rt_10, Rt_01, Rt_11])

    # Find supporters of the transformation
    supporters = find_supporters(points_3D, points_2D_left1, points_2D_right1,
                                 k, Rt_10, Rt_11)

    plot_supporters(img_left0, img_left1, keypoints_left0, keypoints_left1, matches_01, supporters)
    max_T, group_idx = ransac_pnp(points_3D, points_2D_left1, k, Rt_01, points_2D_right1)
    if (max_T is None):
        print("This frame is create none matrix")
        return
    _, _, _, points_3D_l1r1_triangulation = cloud_points_triangulation(1)
    # Transform the point cloud for pair 0 using max_T
    points_3D_pair0_transformed = (max_T @ np.hstack((points_3D_custom, np.ones((points_3D_custom.shape[0], 1)))).T).T[
                                  :, :3]

    # Plot the two 3D point clouds
    plot_point_clouds(points_3D_pair0_transformed, points_3D_l1r1_triangulation)

    # Plot inliers and outliers on images left0 and left1
    in_out_l1_dict = create_in_out_l1_dict(group_idx, points_2D_left1, filtered_keypoints_left1)
    plot_inliers_outliers(img_left0, img_left1, filtered_keypoints_left0, filtered_keypoints_left1, matches_01,
                          group_idx, in_out_l1_dict)


def compute_bound_ransac(outlier_percentage, probability):
    return np.log(1 - probability) / np.log(1 - np.power(1 - outlier_percentage, NUMBER_PTS_FOR_PNP))


def ransac_pnp(points_3D, points_2D_left1, k, Rt_01, points_2D_right1):
    max_supporters = 0
    group_index = []
    num_inliers, num_outliers = 0, 0
    i = 0
    outlier_percentage, probability = 0.5, 0.99
    max_T = None


    bound_rnasac = compute_bound_ransac(outlier_percentage, probability)
    while outlier_percentage != 0 and i < min(compute_bound_ransac(outlier_percentage, probability), 10000):
        # np.random.seed(42)
        rand_idx_pts = np.random.choice(len(points_3D), NUMBER_PTS_FOR_PNP, replace=False)

        Rt_10, t_10 = compute_extrinsic_matrix(points_3D[rand_idx_pts], points_2D_left1[rand_idx_pts], k)
        if Rt_10 is None:
            i += 1
            continue
        R_01 = Rt_01[:, :3]
        t_01 = Rt_01[:, 3]
        R_11 = np.dot(Rt_10[:, :3], R_01)
        t_11 = np.dot(Rt_10[:, :3], t_01) + t_10
        Rt_11 = np.hstack((R_11, t_11.reshape(-1, 1)))
        supporters_idx = find_supporters(points_3D, points_2D_left1, points_2D_right1,
                                         k, Rt_10, Rt_11)
        #avishay avish edit puted next line in note
        # supporters_idx = list(supporters_idx)[0]
        current_num_supporters = len(supporters_idx)
        if current_num_supporters > max_supporters:
            max_supporters = current_num_supporters
            group_index = supporters_idx

        i += 1
        num_outliers += len(points_3D) - current_num_supporters
        num_inliers += current_num_supporters
        outlier_percentage = min(num_outliers / (num_outliers + num_inliers), 0.99)

    succes, max_r_vec, max_t_vec = cv2.solvePnP(points_3D[group_index], points_2D_left1[group_index], k,
                                                cv2.SOLVEPNP_ITERATIVE)

    if succes:
        R, _ = cv2.Rodrigues(max_r_vec)
        max_T = np.hstack((R, max_t_vec))
    return max_T, group_index


# Define the new function
def get_matched_points(matches_01, matches_02, matches_03, cloud_points_pair0, keypoints_left1_1):
    points3D_pair0 = []
    points2D_left1 = []

    for m in matches_03:
        if m.queryIdx < len(matches_01) and m.trainIdx < len(matches_02):
            idx_3d = matches_01[m.queryIdx].queryIdx
            idx_2d = matches_02[m.trainIdx].trainIdx

            if idx_3d < len(cloud_points_pair0) and idx_2d < len(keypoints_left1_1):
                points3D_pair0.append(cloud_points_pair0[idx_3d])
                points2D_left1.append(keypoints_left1_1[idx_2d].pt)

    return np.array(points3D_pair0), np.array(points2D_left1)



def ransac_algorithm_online_with_avish_dict(idx, Rt_00, Rt_01):
    print(idx)
    img_left0, img_right0 = read_images(idx)
    img_left1, img_right1 = read_images(idx + 1)
    # Get matches for pair 0
    filtered_keypoints_left0, filtered_keypoints_right0, desc_00, _, inliers_matches_00, keypoints_left0, keypoints_right0 = (
        get_stereo_matches_with_filtered_keypoints_avish_test(img_left0, img_right0))

    # Get matches for pair 1
    filtered_keypoints_left1, filtered_keypoints_right1, desc_10, _, inliers_matches_11, keypoints_left1, keypoints_right1 = (
        get_stereo_matches_with_filtered_keypoints_avish_test(img_left1, img_right1))

    # Get matches between left0 and left1
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_01 = bf.match(desc_00, desc_10)

    # Perform cloud triangulation for pair 0
    # k, Rt_00, Rt_01 = (
    #     read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    k, _, P_right = (
        read_cameras('C:/Users/avishay/PycharmProjects/SLAM_AVISHAY_YAIR/VAN_ex/dataset/sequences/00/calib.txt'))
    points_3D_custom, pts1, pts2 = (
        triangulation_process(Rt_00, Rt_01, inliers_matches_00, k, filtered_keypoints_left0, filtered_keypoints_right0,
                              plot=False))
    # Create the dictionary for PnP
    filtered_keypoints_left1, filtered_keypoints_right1, points_3D_custom = create_dict_to_pnp_avish_test(matches_01,
                                                                                                          inliers_matches_11,
                                                                                                          filtered_keypoints_left1,
                                                                                                          filtered_keypoints_right1,
                                                                                                          points_3D_custom)
    # points_3D, points_2D_left0, points_2D_left1, points_2D_right1 = create_dict_to_pnp(matches_01, inliers_matches_11,
    #                                                                                    filtered_keypoints_left1,
    #                                                                                    keypoints_left0, keypoints_left1,
    #                                                                                    keypoints_right1,
    #                                                                                    points_3D_custom)

    max_T, group_idx = ransac_pnp(points_3D_custom, filtered_keypoints_left1, k, P_right, filtered_keypoints_right1)
    if (max_T is None):
        print("This frame is create none matrix")
        return

    # Convert 3x4 matrices to 4x4 homogeneous transformation matrices
    max_T_homogeneous = np.vstack([max_T, [0, 0, 0, 1]])
    P_right_homogeneous = np.vstack([P_right, [0, 0, 0, 1]])

    # Compute the transformation matrix for the right image
    max_T_right_homogeneous = np.dot(P_right_homogeneous, max_T_homogeneous)

    # Extract the 3x4 transformation matrix for the right image
    max_T_right = max_T_right_homogeneous[:3, :]
    plot_camera_positions([Rt_00, Rt_01, max_T, max_T_right])

    return max_T, max_T, max_T_right


def q4(points3D_pair0, points2D_left1, Rt00):
    pass



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

def main():
    # q1()
    # q2()
    # q2_()
    # ransac_algorithm_online(114)
    # for i in range(3):
    #     q2_tests(i)
    # q6()
    # q6_in_range(3, 6)
    q6_with_avish()

if __name__ == '__main__':
    main()
