import pickle

import gtsam
from BundleAdjustment import BundleAdjustment
from PoseGraph import PoseGraph
from tracking_database import TrackingDB
from BundleWindow import BundleWindow
from gtsam.utils import plot
from gtsam import symbol
import numpy as np
import matplotlib.pyplot as plt
from gtsam.utils.plot import plot_pose3_on_axes
from algorithms_library import plot_trajectories_and_landmarks

NUM_FRAMES = 3360


def get_relative_pose_and_cov_mat_first_kf(window):
    """

    Args:
        window:

    Returns:

    """
    # assume window optimized
    first_camera = window.get_optimized_first_camera()
    last_camera = window.get_optimized_last_camera(loop=True)
    marginals = window.marginals()
    keys = gtsam.KeyVector()
    sym_ck = symbol('c', 0)
    sym_c1 = window.get_last_frame_symbol(loop=True)
    keys.append(sym_c1)
    keys.append(sym_ck)
    infoCovMatrix = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    covMatrix = np.linalg.inv(infoCovMatrix)
    # multiply by factor for visible to the plot
    # covMatrix = covMatrix * 10
    relative_pose = last_camera.between(first_camera)
    return relative_pose, covMatrix


def get_relative_pose_and_cov_mat_last_kf(window):
    """
    Get relative pose and covariance matrix for the last key frame in the window
    Args:
        window: BundleWindow

    Returns: relative pose and covariance matrix

    """
    # assume window optimized
    first_camera = window.get_optimized_first_camera()
    last_camera = window.get_optimized_last_camera()
    marginals = window.marginals()
    keys = gtsam.KeyVector()
    sym_c0 = symbol('c', 0)
    sym_ck = window.get_last_frame_symbol()
    keys.append(sym_c0)
    keys.append(sym_ck)
    infoCovMatrix = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    covMatrix = np.linalg.inv(infoCovMatrix)
    # multiply by factor for visible to the plot
    # covMatrix = covMatrix * 10
    relative_pose = first_camera.between(last_camera)
    return relative_pose, covMatrix


def q1(db):
    first_window = BundleWindow(db=db, first_key_frame_id=0, last_key_frame_id=10)
    first_window.create_factor_graph()
    first_window.optimize()
    first_window.save("bundle_windows/first_window4_avish")
    pose_c0 = first_window.get_optimized_first_camera()
    pose_ck = first_window.get_optimized_last_camera()
    marginals = first_window.marginals()
    values = first_window.get_optimized_values()
    relative_pose = pose_c0.between(pose_ck)
    keys = gtsam.KeyVector()
    sym_c0 = symbol('c', 0)
    sym_ck = symbol('c', 10)
    keys.append(sym_c0)
    keys.append(sym_ck)
    infoCovMatrix = marginals.jointMarginalInformation(keys).at(keys[-1], keys[-1])
    covMatrix = np.linalg.inv(infoCovMatrix)
    print(covMatrix)

    print(f"Relative pose between c0 and ck: {relative_pose}")
    gtsam.utils.plot.plot_trajectory(fignum=0, values=values, marginals=marginals, scale=1,
                                     title="Covariance poses first bundle")
    plt.savefig("poses_first_bundle_with_covariance_plot_trajctory.png")
    plt.axis('equal')
    plt.savefig("poses_first_bundle_with_covariance_plot_trajctory_with_equal.png")
    plt.close()

    first_window.save("bundle_windows/first_window4_avish")


def q2(db):
    bundle = BundleAdjustment(db)
    bundle.load("bundle_data_window_size_20_witohut_bad_matches_ver2/",
                "bundle with window_size_20_witohut_bad_matches_ver2", 5)

    relative_poses, cov_matrices = [], []
    for i, window in enumerate(bundle.get_windows()):
        print(f"try to get relative pose and cov mat for window {i}")
        relative_pose, cov_matrix = get_relative_pose_and_cov_mat_last_kf(window)
        relative_poses.append(relative_pose)
        cov_matrices.append(cov_matrix)
    poseGraph = PoseGraph(db, bundle, relative_poses, cov_matrices, bundle.get_key_frames())
    poseGraph.create_factor_graph()

    plot_initial_estimate(poseGraph)

    poseGraph.optimize_poseGraph()
    plot_optimized_values(poseGraph)


def plot_optimized_values(poseGraph, loop=False):
    if loop:
        dir = "results_ex7/optimized_pose_graph_after_loop.png"
    else:
        dir = "optimized_values_for_pose_graph_for_david.png"

    print(f"Optimized Error: {poseGraph.graph().error(poseGraph.get_optimized_values())}")
    gtsam.utils.plot.plot_trajectory(fignum=1, values=poseGraph.get_optimized_values(), scale=1,
                                     title="Optimized values")
    plt.axis('equal')
    plt.savefig("optimized_values_for_pose_graph_for_david.png")
    plt.close()
    marginals = poseGraph.marginals()
    gtsam.utils.plot.plot_trajectory(fignum=2, values=poseGraph.get_optimized_values(), marginals=marginals,
                                     scale=0.1, title="Optimized values with marginals")
    plt.axis('equal')
    plt.savefig("optimized_values_with_marginals_for_pose_graph_for_david.png")
    plt.close()
    gtsam.utils.plot.plot_trajectory(fignum=3, values=poseGraph.get_optimized_values(), marginals=marginals,
                                     scale=10, title="Optimized values with marginals_scale_10")
    plt.axis('equal')
    plt.savefig("optimized_values_with_marginals_for_pose_graph_scale_10_for_david.png")
    plt.close()


def plot_initial_estimate(poseGraph, loop=False):
    if loop:
        dir = "results_ex7/initial_estimate_ex07.png"
    else:
        dir = "initial_estimate_for_pose_graph_for_david.png"
    gtsam.utils.plot.plot_trajectory(fignum=0, values=poseGraph.get_initial_estimate(), scale=1,
                                     title="Initial estimate")
    plt.axis('equal')
    plt.savefig(dir)
    plt.close()
    # print factor Error of the graph
    print(f"Initial Error: {poseGraph.graph().error(poseGraph.get_initial_estimate())}")


if __name__ == '__main__':
    db = TrackingDB()
    db.load('db')
    q1(db)
    q2(db)
