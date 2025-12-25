import cv2
import numpy as np
from algorithms_library import (compute_trajectory_and_distance_avish_test, plot_root_ground_truth_and_estimate,
                                read_images_from_dataset, read_cameras_matrices, read_poses, read_poses_truth,
                                xy_triangulation, project, get_euclidean_distance, plot_tracks, calculate_statistics,
                                print_statistics, init_db, compose_transformations, project_point)
import random
import matplotlib.pyplot as plt
from tracking_database import TrackingDB

NUM_FRAMES = 3360  # or any number of frames you want to process


def q2(db):
    stats = calculate_statistics(db)
    print_statistics(stats)


def q3(db):
    all_tracks = db.all_tracks()
    track_lengths = [(trackId, len(db.frames(trackId))) for trackId in all_tracks if len(db.frames(trackId)) >= 6]

    if not track_lengths:
        print("No tracks of length >= 6 found.")
        return

    track_to_display = track_lengths[2][0]
    frames = db.frames(track_to_display)

    fig, axes = plt.subplots(len(frames), 2, figsize=(10, len(frames) * 5))

    for idx, frameId in enumerate(frames):
        img_left, _ = read_images_from_dataset(frameId)
        link = db.link(frameId, track_to_display)
        left_kp = link.left_keypoint()

        x, y = int(left_kp[0]), int(left_kp[1])
        top_left_x = max(x - 10, 0)
        top_left_y = max(y - 10, 0)
        bottom_right_x = min(x + 10, img_left.shape[1])
        bottom_right_y = min(y + 10, img_left.shape[0])

        img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_GRAY2RGB)
        img_left_rgb = cv2.rectangle(img_left_rgb, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y),
                                     (0, 255, 0), 2)
        img_left_rgb = cv2.circle(img_left_rgb, (x, y), 2, (255, 0, 0), -1)  # Mark the feature

        # Crop the 20x20 region around the feature
        cropped_region = img_left_rgb[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        # Enlarge the cropped region
        enlarged_region = cv2.resize(cropped_region, (100, 100), interpolation=cv2.INTER_NEAREST)
        enlarged_region = cv2.circle(enlarged_region, (50, 50), 2, (255, 0, 0), -1)  # Mark the feature in zoomed view

        axes[idx, 0].imshow(img_left_rgb)
        axes[idx, 0].set_title(f"Frame {frameId}")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(enlarged_region)
        axes[idx, 1].set_title(f"Zoomed Feature in Frame {frameId}")
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.savefig('q3_track_feature_v1.png')
    plt.show()


def q4(db):
    all_frames = sorted(db.all_frames())
    connectivity = []

    for i in range(len(all_frames) - 1):
        current_frame = all_frames[i]
        next_frame = all_frames[i + 1]

        current_tracks = set(db.tracks(current_frame))
        next_tracks = set(db.tracks(next_frame))

        outgoing_tracks = current_tracks.intersection(next_tracks)
        connectivity.append(len(outgoing_tracks))

    # Plotting the connectivity graph
    plt.figure(figsize=(16, 8))  # Stretched x-axis by increasing the width
    plt.plot(range(len(connectivity)), connectivity, linestyle='-', color='blue', linewidth=0.5)  # Thinner blue line
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Outgoing Tracks')
    plt.title('Connectivity Graph: Number of Outgoing Tracks per Frame')
    plt.grid(True)
    plt.xticks(
        np.arange(0, len(connectivity), step=max(1, len(connectivity) // 20)))  # Set x-ticks to be more spread out
    plt.tight_layout()
    plt.savefig('q4_connectivity_graph_v1.png')
    plt.show()


def q5(db):
    supporters_percentage = db.supporters_percentage
    # Create x-values starting from 1
    frames = list(range(1, len(supporters_percentage) + 1))

    # Plot the percentages with a thin line
    plt.figure(figsize=(10, 6))
    plt.plot(frames, supporters_percentage, linewidth=0.5)
    plt.xlabel('Frames')
    plt.ylabel('Supporters Percentage')
    plt.title('Supporters Percentage Over Frames')
    plt.grid(True)
    plt.show()


def q6(db):
    all_tracks = db.all_tracks()
    track_lengths = [len(db.frames(track_id)) for track_id in all_tracks if len(db.frames(track_id)) > 1]

    # Plotting the track length histogram
    plt.figure(figsize=(14, 8))
    plt.hist(track_lengths, bins=range(1, max(track_lengths) + 1), color='blue', edgecolor='black', alpha=0.7, log=True)
    plt.xlabel('Track Length')
    plt.ylabel('Frequency (log scale)')
    plt.title('Track Length Histogram')
    plt.grid(axis='y')
    plt.xticks(np.arange(1, max(track_lengths) + 1, step=5))
    plt.tight_layout()
    plt.savefig('q6_track_length_histogram_v1.png')
    plt.show()


def q7(db):
    # # Load TrackingDB
    K, P_left, P_right = read_cameras_matrices()

    # Get all valid tracks
    valid_tracks = [track for track in db.all_tracks() if len(db.frames(track)) >= 10]

    # Select a random track of length >= 10
    selected_track = random.choice(valid_tracks)
    selected_track_frames = db.frames(selected_track)
    frame_ids = [frame_id for frame_id in db.frames(selected_track)]

    # Triangulate the 3D point using the last frame of the track
    link = db.link(selected_track_frames[-1], selected_track)

    # Project to all frames in the track

    # K, P_left, P_right = read_cameras_matrices()
    transformations = read_poses_truth(seq=(frame_ids[0], frame_ids[-1] + 1))
    last_left_img_xy = link.left_keypoint()
    last_right_img_xy = link.right_keypoint()
    last_left_transfromation = transformations[-1]
    last_l_projection_mat = K @ last_left_transfromation
    last_r_projection_mat = K @ compose_transformations(last_left_transfromation, P_right)
    p3d = xy_triangulation([last_left_img_xy, last_right_img_xy], last_l_projection_mat,
                           last_r_projection_mat)

    left_projections = []
    right_projections = []

    for trans in transformations:
        left_proj_cam = K @ trans
        right_proj_cam = K @ compose_transformations(trans, P_right)

        left_proj = project(p3d, left_proj_cam)
        right_proj = project(p3d, right_proj_cam)

        left_projections.append(left_proj)
        right_projections.append(right_proj)

    frames_l_xy = [db.link(frame, selected_track).left_keypoint() for frame in selected_track_frames]
    frames_r_xy = [db.link(frame, selected_track).right_keypoint() for frame in selected_track_frames]
    left_proj_dist = get_euclidean_distance(np.array(left_projections), np.array(frames_l_xy))
    right_proj_dist = get_euclidean_distance(np.array(right_projections), np.array(frames_r_xy))
    total_proj_dist = (left_proj_dist + right_proj_dist) / 2

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title(f"Reprojection error for track: {selected_track}")
    # Plotting the scatter plot
    ax.scatter(range(len(total_proj_dist)), total_proj_dist, color='blue', label='Data Points')
    # Plotting the continuous line
    ax.plot(range(len(total_proj_dist)), total_proj_dist, linestyle='-', color='red', label='')

    ax.set_ylabel('Error')
    ax.set_xlabel('Frames')
    ax.legend()

    fig.savefig("Reprojection_error_v1.png")
    plt.close(fig)


if __name__ == "__main__":
    # db, supporters_percentage = init_db()
    db = TrackingDB()
    db.load('db_v1')
    q2(db)
    q3(db)
    q4(db)
    # # # print(supporters_percentage)
    q5(db)
    q6(db)
    q7(db)
