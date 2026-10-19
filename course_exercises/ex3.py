"""Validated Exercise 3 stereo visual-odometry stages."""

import random

import numpy as np

from tools import (
    MATCHER,
    calculate_front_camera_matrix,
    compute_trajectory_and_distance,
    cv_triangulate_matched_points,
    estimate_projection_matrices_with_ransac,
    extract_actual_consensus_pixels,
    extract_keypoints_and_inliers,
    find_consensus_matches_indices,
    find_supporter_indices_for_model,
    get_supporters_unsupporters_to_plot,
    match_reciprocal,
    plot_root_ground_truth_and_estimate,
    plot_supporters_non_supporters,
    plot_two_3D_point_clouds,
    read_cameras_matrices,
    read_images_from_dataset,
)


def main():
    """Run Exercise 3 sections 3.1 through 3.5 on frames 0 and 1."""
    print("start session:\n")
    image0_left, image0_right = read_images_from_dataset(0)
    image1_left, image1_right = read_images_from_dataset(1)
    intrinsic, ext0_left, ext0_right = read_cameras_matrices()
    rotation0_left, translation0_left = ext0_left[:, :3], ext0_left[:, 3:]
    rotation0_right, translation0_right = ext0_right[:, :3], ext0_right[:, 3:]

    prepared = prepare_consecutive_stereo_pairs(
        intrinsic,
        rotation0_left,
        rotation0_right,
        image0_left,
        image0_right,
        image1_left,
        image1_right,
        translation0_left,
        translation0_right,
    )
    (
        descriptors0_left,
        descriptors1_left,
        inliers0,
        inliers1,
        keypoints0_left,
        keypoints0_right,
        keypoints1_left,
        keypoints1_right,
        point_cloud0,
    ) = prepared
    tracking_matches = match_left_images_reciprocally(
        descriptors0_left, descriptors1_left
    )
    initial_pose = estimate_initial_four_camera_pose(
        ext0_left,
        ext0_right,
        intrinsic,
        rotation0_right,
        inliers0,
        inliers1,
        keypoints1_left,
        point_cloud0,
        translation0_right,
        tracking_matches,
    )
    rotation1_left, rotation1_right, consensus, translation1_left, translation1_right = initial_pose
    classify_four_view_supporters(
        intrinsic,
        rotation0_left,
        rotation0_right,
        rotation1_left,
        rotation1_right,
        consensus,
        image0_left,
        image1_left,
        inliers0,
        inliers1,
        keypoints0_left,
        keypoints0_right,
        keypoints1_left,
        keypoints1_right,
        point_cloud0,
        translation0_left,
        translation0_right,
        translation1_left,
        translation1_right,
        tracking_matches,
    )
    estimate_pose_with_ransac(
        intrinsic,
        rotation0_left,
        rotation0_right,
        consensus,
        image0_left,
        image1_left,
        inliers0,
        inliers1,
        keypoints0_left,
        keypoints0_right,
        keypoints1_left,
        keypoints1_right,
        point_cloud0,
        translation0_left,
        translation0_right,
        tracking_matches,
    )


# Exercise 3.6 trajectory extension.
def estimate_and_plot_trajectory(num_frames):
    estimated_trajectory, ground_truth_trajectory, _ = compute_trajectory_and_distance(
        num_frames=num_frames, verbose=True
    )
    plot_root_ground_truth_and_estimate(
        estimated_trajectory, ground_truth_trajectory
    )


# Exercise 3.5: refine the four-view pose with custom RANSAC.
def estimate_pose_with_ransac(
    intrinsic,
    rotation0_left,
    rotation0_right,
    consensus,
    image0_left,
    image1_left,
    inliers0,
    inliers1,
    keypoints0_left,
    keypoints0_right,
    keypoints1_left,
    keypoints1_right,
    point_cloud0,
    translation0_left,
    translation0_right,
    tracking_matches,
):
    rotations, translations, _, supporter_indices = estimate_projection_matrices_with_ransac(
        point_cloud0,
        consensus,
        inliers0,
        inliers1,
        keypoints0_left,
        keypoints0_right,
        keypoints1_left,
        keypoints1_right,
        intrinsic,
        rotation0_left,
        translation0_left,
        rotation0_right,
        translation0_right,
        verbose=True,
    )
    plot_two_3D_point_clouds(rotations, translations, point_cloud0)
    (
        non_supporting_pixels_back,
        non_supporting_pixels_front,
        supporting_pixels_back,
        supporting_pixels_front,
    ) = get_supporters_unsupporters_to_plot(
        consensus,
        keypoints0_left,
        keypoints1_left,
        supporter_indices,
        tracking_matches,
    )
    plot_supporters_non_supporters(
        image0_left,
        image1_left,
        supporting_pixels_back,
        supporting_pixels_front,
        non_supporting_pixels_back,
        non_supporting_pixels_front,
        title="q5 - supporters and unsupporters after ransac - pnp",
    )
    return len(supporter_indices) / len(consensus) * 100


# Exercise 3.4: classify supporters by reprojection in all four views.
def classify_four_view_supporters(
    intrinsic,
    rotation0_left,
    rotation0_right,
    rotation1_left,
    rotation1_right,
    consensus,
    image0_left,
    image1_left,
    inliers0,
    inliers1,
    keypoints0_left,
    keypoints0_right,
    keypoints1_left,
    keypoints1_right,
    point_cloud0,
    translation0_left,
    translation0_right,
    translation1_left,
    translation1_right,
    tracking_matches,
):
    actual_pixels = extract_actual_consensus_pixels(
        consensus,
        inliers0,
        inliers1,
        keypoints0_left,
        keypoints0_right,
        keypoints1_left,
        keypoints1_right,
    )
    consensus_points = point_cloud0[[match[0] for match in consensus]]
    rotations = [rotation0_left, rotation0_right, rotation1_left, rotation1_right]
    translations = [
        translation0_left,
        translation0_right,
        translation1_left,
        translation1_right,
    ]
    return find_supporter_indices_for_model(
        consensus_points, actual_pixels, intrinsic, rotations, translations
    )


# Exercise 3.3: estimate left1 with PnP and compose the right1 stereo pose.
def estimate_initial_four_camera_pose(
    ext0_left,
    ext0_right,
    intrinsic,
    rotation0_right,
    inliers0,
    inliers1,
    keypoints1_left,
    point_cloud0,
    translation0_right,
    tracking_matches,
):
    consensus = find_consensus_matches_indices(
        inliers0, inliers1, tracking_matches
    )
    _, rotation1_left, translation1_left = calculate_front_camera_matrix(
        random.sample(consensus, 4),
        point_cloud0,
        inliers1,
        keypoints1_left,
        intrinsic,
    )

    ext1_left = np.hstack((rotation1_left, translation1_left))
    # Apply the temporal transformation first and the fixed stereo transform second.
    rotation1_right = np.dot(rotation0_right, ext1_left[:, :3])
    translation1_right = (
        np.dot(rotation0_right, translation1_left) + translation0_right
    )
    return (
        rotation1_left,
        rotation1_right,
        consensus,
        translation1_left,
        translation1_right,
    )


# Exercise 3.2: retain only mutually consistent left0-to-left1 matches.
def match_left_images_reciprocally(descriptors0_left, descriptors1_left):
    return match_reciprocal(
        descriptors0_left, descriptors1_left, matcher=MATCHER
    )


# Exercise 3.1: extract both stereo pairs and triangulate the back pair.
def prepare_consecutive_stereo_pairs(
    intrinsic,
    rotation0_left,
    rotation0_right,
    image0_left,
    image0_right,
    image1_left,
    image1_right,
    translation0_left,
    translation0_right,
):
    pair0 = extract_keypoints_and_inliers(image0_left, image0_right)
    (
        keypoints0_left,
        descriptors0_left,
        keypoints0_right,
        _,
        inliers0,
        _,
    ) = pair0
    point_cloud0 = cv_triangulate_matched_points(
        keypoints0_left,
        keypoints0_right,
        inliers0,
        intrinsic,
        rotation0_left,
        translation0_left,
        rotation0_right,
        translation0_right,
    )
    pair1 = extract_keypoints_and_inliers(image1_left, image1_right)
    (
        keypoints1_left,
        descriptors1_left,
        keypoints1_right,
        _,
        inliers1,
        _,
    ) = pair1
    return (
        descriptors0_left,
        descriptors1_left,
        inliers0,
        inliers1,
        keypoints0_left,
        keypoints0_right,
        keypoints1_left,
        keypoints1_right,
        point_cloud0,
    )


if __name__ == "__main__":
    main()
