import random
import gtsam
import numpy as np
from gtsam import symbol
import gtsam.utils.plot as gtsam_plot
from BundleWindow import BundleWindow
from BundleAdjustment import BundleAdjustment, convert_gtsam_cams_to_global, convert_landmarks_to_global, save
from algorithms_library import (get_truth_transformation, compute_trajectory_left_cams, get_euclidean_distance,
                                compute_trajectory_gtsam_left_cams, plot_trajectories_and_landmarks,
                                create_calib_mat_gtsam, create_ext_matrix_gtsam, triangulate_gtsam,
                                find_projection_factor_with_largest_initial_error, print_projection_details,
                                plot_left_cams_and_landmarks_one_bundle, projection_factors_error,
                                plot_re_projection_error, plot_factor_error_graph, print_ancoring_error,
                                plot_factor_error_as_function_of_projection_error,
                                calculate_and_plot_error_between_est_cams_and_truth_cams)
import matplotlib.pyplot as plt
from tracking_database import TrackingDB

NUM_FRAMES = 3360


def q1(db: TrackingDB):
    # Create the calibration matrix
    calib_mat_gtsam = create_calib_mat_gtsam()
    # Create the values dictionary
    values = gtsam.Values()
    # Select a random track with at least 10 frames
    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)
    frame_ids = [frame_id for frame_id in db.frames(selected_track)]
    track_length = len(frame_ids)
    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)
    # Read transformations
    Rt_inverse_gtsam = create_ext_matrix_gtsam(db, frame_ids[-1])
    triangulate_p3d_gtsam = triangulate_gtsam(Rt_inverse_gtsam, calib_mat_gtsam, link)
    # Update values dictionary
    p3d_sym = symbol("q", 0)
    values.insert(p3d_sym, triangulate_p3d_gtsam)
    # calculate re-projection
    left_projections = []
    right_projections = []
    selected_track_frames = selected_track_frames[::-1]
    frame_ids = frame_ids[::-1]
    frames_l_xy = [db.link(frame, selected_track).left_keypoint() for frame in selected_track_frames]
    frames_r_xy = [db.link(frame, selected_track).right_keypoint() for frame in selected_track_frames]
    factors = []
    for i in range(track_length):
        # Read transformations
        current_left_rotation = db.rotation_matrices[frame_ids[i]]
        current_left_translation = db.translation_vectors[frame_ids[i]]
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
        Rt_inverse_gtsam_current = (gtsam.Pose3.inverse(gtsam.Pose3(current_left_transformation_homogenus)))
        current_frame_camera_left = gtsam.StereoCamera(Rt_inverse_gtsam_current, calib_mat_gtsam)
        project_stereo_pt_gtsam = current_frame_camera_left.project(triangulate_p3d_gtsam)
        left_projections.append([project_stereo_pt_gtsam.uL(), project_stereo_pt_gtsam.v()])
        right_projections.append([project_stereo_pt_gtsam.uR(), project_stereo_pt_gtsam.v()])
        left_pose_sym = symbol("c", frame_ids[i])
        values.insert(left_pose_sym, Rt_inverse_gtsam_current)
        # Factor creation
        gtsam_measurement_pt2 = gtsam.StereoPoint2(frames_l_xy[i][0], frames_r_xy[i][0], frames_l_xy[i][1])
        projection_uncertainty = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.GenericStereoFactor3D(gtsam_measurement_pt2, projection_uncertainty,
                                             symbol("c", frame_ids[i]), p3d_sym, calib_mat_gtsam)
        factors.append(factor)
    left_proj_dist = get_euclidean_distance(np.array(left_projections), np.array(frames_l_xy))
    right_proj_dist = get_euclidean_distance(np.array(right_projections), np.array(frames_r_xy))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2
    # Factor error
    factor_projection_errors = projection_factors_error(factors, values)
    # Plots re-projection error
    plot_re_projection_error(left_proj_dist, right_proj_dist, selected_track)
    plot_factor_error_graph(factor_projection_errors, -1)
    plot_factor_error_as_function_of_projection_error(total_proj_dist, factor_projection_errors)


def q3(db):
    # Create the first window with 20 frames
    first_window = BundleWindow(0, 20, db=db)
    first_window.create_factor_graph()
    error_before_optim = first_window.calculate_graph_error(False)
    first_window.optimize()
    error_after_optim = first_window.calculate_graph_error()
    print(f"The error of the graph before optimize:{error_before_optim}")
    print(f"The error of the graph after optimize:{error_after_optim}")
    print(f"The number of the factors in the graph: {first_window.graph.size()}")
    print(f"The average factor error before optimization:{error_before_optim / first_window.graph.size()}")
    print(f"The average factor error after optimization:{error_after_optim / first_window.graph.size()}")
    max_error_factor, max_error = find_projection_factor_with_largest_initial_error(first_window.graph,
                                                                                    first_window.get_initial_estimate())
    print_projection_details(max_error_factor, first_window.get_initial_estimate(), create_calib_mat_gtsam())
    print_projection_details(max_error_factor, first_window.get_optimized_values(), create_calib_mat_gtsam())
    gtsam.utils.plot.plot_trajectory(fignum=0, values=first_window.get_optimized_values(), title="aa")
    plt.savefig("optimized_trajectory.png")  # Save the plot to a file
    plt.close()  # Close the plot
    # check with (c, 0) and c(,latrframe)
    cams = first_window.get_optimized_cameras()
    cams_relative = []
    for t in cams:
        cams_relative.append(t.translation())
    cams_relative = np.array(cams_relative)
    plot_left_cams_and_landmarks_one_bundle(cameras=cams_relative,
                                            landmarks=first_window.get_optimized_landmarks())


def q4(db):
    title = "window_size_20_witohut_bad_matches_ver2"
    window_size = 5
    num_frames = NUM_FRAMES
    # Create bundle adjusment accross all num_frames and solve with bundle window size 20.
    bundle_adjusment = BundleAdjustment(db, 0, num_frames - 1)
    # bundle_adjusment.solve_with_window_size_20()
    bundle_adjusment.solve_with_interactive_window_size(window_size)
    # Get the cameras of the initial estimate and the optimized bundle and the ground truth, and the landmarks
    gtsam_cams_bundle = bundle_adjusment.get_gtsam_cams()
    gtsam_landmarks_bundle = bundle_adjusment.get_gtsam_landmarks()
    cams = convert_gtsam_cams_to_global(gtsam_cams_bundle)
    landmarks = convert_landmarks_to_global(cams, gtsam_landmarks_bundle)
    bundle_key_frames = bundle_adjusment.get_key_frames()
    truth_trans = np.array(get_truth_transformation(num_frames=num_frames))[bundle_key_frames]
    cams_truth_3d = compute_trajectory_left_cams(truth_trans)
    cams_3d = compute_trajectory_gtsam_left_cams(cams)
    initial_estimate = db.initial_estimate_cams()
    bundle_adjusment.save("bundle_data_window_size_20_witohut_bad_matches_ver2/", "bundle with " + title)

    plot_trajectories_and_landmarks(cameras=cams_3d, landmarks=landmarks, initial_estimate_poses=initial_estimate,
                                    cameras_gt=cams_truth_3d, title="ground_truth_without_bad_matches")
    plot_trajectories_and_landmarks(cameras=cams_3d, landmarks=landmarks, title=title)
    # Print the position of the first frame in the last bundle
    last_window = bundle_adjusment.get_last_window()
    all_optimized_values = last_window.get_optimized_values()
    last_window_range = last_window.get_frame_range()
    keyframe_id_symbol = symbol('c', 0)
    # Print the position of the first frame in the last bundle
    if all_optimized_values.exists(keyframe_id_symbol):
        last_window_first_frame_pose = all_optimized_values.atPose3(keyframe_id_symbol)
        print(f"Position of the first frame in the last bundle: {last_window_first_frame_pose.translation()}")
    else:
        print("The first frame of the last window is not in the optimized values.")
    print_ancoring_error(all_optimized_values, last_window, keyframe_id_symbol)
    # Calculate and plot keyframe localization error
    calculate_and_plot_error_between_est_cams_and_truth_cams(cams_3d, cams_truth_3d)
    # bundle_adjusment.serialize("bundle_adjusment_with_20_window")

if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    # q1(db)
    # q3(db)
    q4(db)
