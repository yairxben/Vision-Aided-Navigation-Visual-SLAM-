"""Artifact and figure generation for the Exercise 6 Pose Graph run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import gtsam
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from PoseGraph import (
    CONDITIONAL_COVARIANCE_FIELD,
    EDGE_DIRECTION,
    TANGENT_ORDER,
    CanonicalPoseGraphInputs,
    PoseGraph,
    load_canonical_pose_graph_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
VISUALIZATION_COVARIANCE_MULTIPLIER = 0.01
FIRST_BUNDLE_VISUALIZATION_FILENAME = (
    "first_bundle_camera_covariances_3d_visualization_x0.01.png"
)
POSE_GRAPH_VISUALIZATION_FILENAME = (
    "optimized_pose_graph_with_marginals_visualization_x0.01.png"
)


def _pose_matrices(values: gtsam.Values, keys: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [values.atPose3(key).matrix() for key in keys], dtype="<f8"
    )


def _save_json(path: Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _set_3d_equal(axis, points: np.ndarray) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1e-9)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _covariance_ellipsoid(
    axis,
    center: np.ndarray,
    covariance: np.ndarray,
    *,
    color: str,
    alpha: float,
    resolution: int,
) -> None:
    """Draw a raw 95% location-covariance ellipsoid."""

    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    radii = 2.795483482915 * np.sqrt(np.maximum(eigenvalues, 0.0))
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, max(4, resolution // 2))
    unit = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=0,
    )
    points = eigenvectors @ (radii[:, None] * unit.reshape(3, -1))
    points = points.reshape(unit.shape) + center[:, None, None]
    axis.plot_wireframe(
        points[0],
        points[1],
        points[2],
        color=color,
        linewidth=0.35,
        alpha=alpha,
    )


def _global_location_covariance(
    pose_matrix: np.ndarray, tangent_covariance: np.ndarray
) -> np.ndarray:
    rotation = pose_matrix[:3, :3]
    local_translation = tangent_covariance[3:, 3:]
    return rotation @ local_translation @ rotation.T


def plot_first_bundle_camera_covariances(
    inputs: CanonicalPoseGraphInputs,
    output_path: Path,
    *,
    covariance_multiplier: float = 1.0,
    title: str = "First Bundle window: raw 95% location covariances",
) -> None:
    """Plot all saved first-window cameras and location covariances."""

    poses = inputs.first_bundle_poses
    covariances = inputs.first_bundle_marginal_covariances
    centers = poses[:, :3, 3]
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*centers.T, "o-", color="tab:blue", label="camera locations")
    for pose, covariance in zip(poses, covariances):
        visualization_covariance = covariance * covariance_multiplier
        _covariance_ellipsoid(
            axis,
            pose[:3, 3],
            _global_location_covariance(pose, visualization_covariance),
            color="tab:orange",
            alpha=0.8,
            resolution=14,
        )
    _set_3d_equal(axis, centers)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_pose_graph(
    poses: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    centers = poses[:, :3, 3]
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(centers[:, 0], centers[:, 2], "-", linewidth=1.2)
    axis.scatter(centers[:, 0], centers[:, 2], s=7)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_pose_graph_with_marginals(
    poses: np.ndarray,
    marginal_covariances: np.ndarray,
    output_path: Path,
    *,
    covariance_multiplier: float = 1.0,
    title: str = "Optimized Pose Graph with raw 95% marginal covariances",
) -> None:
    """Plot the optimized chain with every marginal location covariance."""

    centers = poses[:, :3, 3]
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*centers.T, "-", color="tab:blue", linewidth=1.0)
    axis.scatter(*centers.T, color="tab:blue", s=5)
    for pose, covariance in zip(poses, marginal_covariances):
        visualization_covariance = covariance * covariance_multiplier
        _covariance_ellipsoid(
            axis,
            pose[:3, 3],
            _global_location_covariance(pose, visualization_covariance),
            color="tab:orange",
            alpha=0.25,
            resolution=8,
        )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _read_ground_truth(path: Path) -> Tuple[np.ndarray, ...]:
    extrinsics = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        pose = np.asarray(line.split(), dtype=float).reshape(3, 4)
        extrinsics.append(
            np.vstack((pose, np.array([0.0, 0.0, 0.0, 1.0])))
        )
    return tuple(extrinsics)


def _rotation_error_deg(error_matrix: np.ndarray) -> float:
    cosine = np.clip(
        (np.trace(error_matrix[:3, :3]) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)))


def write_report_ready_absolute_errors(
    inputs: CanonicalPoseGraphInputs,
    optimized_poses: np.ndarray,
    output_root: Path,
) -> Dict[str, object]:
    """Create no-loop-closure keyframe overlay and absolute error evidence."""

    raw_truth = _read_ground_truth(inputs.ground_truth_path)
    if len(raw_truth) <= inputs.keyframe_ids[-1]:
        raise ValueError("ground-truth file does not cover frame 3359")
    truth_poses = np.asarray(
        [
            raw_truth[0] @ np.linalg.inv(raw_truth[frame_id])
            for frame_id in inputs.keyframe_ids
        ],
        dtype=float,
    )
    estimated_centers = optimized_poses[:, :3, 3]
    truth_centers = truth_poses[:, :3, 3]
    signed_error = estimated_centers - truth_centers
    absolute_error = np.abs(signed_error)
    location_norm = np.linalg.norm(signed_error, axis=1)
    rotation_error = np.asarray(
        [
            _rotation_error_deg(np.linalg.inv(truth) @ estimate)
            for estimate, truth in zip(optimized_poses, truth_poses)
        ]
    )

    csv_path = output_root / "pose_graph_absolute_errors.csv"
    fields = (
        "keyframe_index",
        "frame_id",
        "absolute_x_error_m",
        "absolute_y_error_m",
        "absolute_z_error_m",
        "location_error_norm_m",
        "absolute_rotation_error_deg",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, frame_id in enumerate(inputs.keyframe_ids):
            writer.writerow(
                {
                    "keyframe_index": index,
                    "frame_id": frame_id,
                    "absolute_x_error_m": absolute_error[index, 0],
                    "absolute_y_error_m": absolute_error[index, 1],
                    "absolute_z_error_m": absolute_error[index, 2],
                    "location_error_norm_m": location_norm[index],
                    "absolute_rotation_error_deg": rotation_error[index],
                }
            )

    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(
        truth_centers[:, 0],
        truth_centers[:, 2],
        label="ground truth",
    )
    axis.plot(
        estimated_centers[:, 0],
        estimated_centers[:, 2],
        label="Pose Graph without Loop Closure",
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Keyframe trajectory")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_root / "pose_graph_ground_truth_trajectory.png", dpi=160
    )
    plt.close(figure)

    frame_ids = np.asarray(inputs.keyframe_ids)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(frame_ids, absolute_error[:, 0], label="|x|")
    axes[0].plot(frame_ids, absolute_error[:, 1], label="|y|")
    axes[0].plot(frame_ids, absolute_error[:, 2], label="|z|")
    axes[0].plot(frame_ids, location_norm, label="location norm")
    axes[0].set_ylabel("absolute error [m]")
    axes[0].legend(ncol=4)
    axes[1].plot(frame_ids, rotation_error)
    axes[1].set_xlabel("FrameId")
    axes[1].set_ylabel("absolute rotation error [deg]")
    figure.suptitle("Pose Graph absolute error without Loop Closure")
    figure.tight_layout()
    figure.savefig(output_root / "pose_graph_absolute_errors.png", dpi=160)
    plt.close(figure)

    return {
        "created": True,
        "ground_truth_path": str(
            inputs.ground_truth_path.relative_to(REPOSITORY_ROOT)
        ),
        "location_error_m": {
            "mean": float(np.mean(location_norm)),
            "rmse": float(np.sqrt(np.mean(location_norm ** 2))),
            "median": float(np.median(location_norm)),
            "maximum": float(np.max(location_norm)),
            "final": float(location_norm[-1]),
        },
        "absolute_rotation_error_deg": {
            "mean": float(np.mean(rotation_error)),
            "rmse": float(np.sqrt(np.mean(rotation_error ** 2))),
            "median": float(np.median(rotation_error)),
            "maximum": float(np.max(rotation_error)),
            "final": float(rotation_error[-1]),
        },
        "artifacts": [
            csv_path.name,
            "pose_graph_ground_truth_trajectory.png",
            "pose_graph_absolute_errors.png",
        ],
    }


def _first_constraint_record(
    inputs: CanonicalPoseGraphInputs,
) -> Dict[str, object]:
    edge = inputs.edges[0]
    matrix = np.asarray(edge.relative_pose.matrix(), dtype=float)
    return {
        "schema": "exercise6-first-consecutive-constraint-v1",
        "start_frame_id": edge.start_frame_id,
        "end_frame_id": edge.end_frame_id,
        "start_camera_key": int(gtsam.symbol("c", edge.start_frame_id)),
        "end_camera_key": int(gtsam.symbol("c", edge.end_frame_id)),
        "direction": EDGE_DIRECTION,
        "relative_pose_matrix": matrix.tolist(),
        "translation_m": matrix[:3, 3].tolist(),
        "rotation_vector_rad": np.asarray(
            gtsam.Rot3.Logmap(edge.relative_pose.rotation()), dtype=float
        ).tolist(),
        "conditional_covariance_field": CONDITIONAL_COVARIANCE_FIELD,
        "conditional_covariance": edge.covariance.tolist(),
        "tangent_order": list(TANGENT_ORDER),
        "covariance_scaled": False,
    }


def create_visualization_only_covariance_figures(
    *, input_root: Path, output_root: Path
) -> Dict[str, object]:
    """Add two x0.01 covariance plots without rebuilding the Pose Graph."""

    inputs = load_canonical_pose_graph_inputs(input_root)
    output_root = Path(output_root)
    summary_path = output_root / "summary.json"
    values_path = output_root / "pose_graph_values.npz"
    marginals_path = output_root / "pose_graph_marginals.npz"
    raw_paths = (
        output_root / "first_bundle_camera_covariances_3d.png",
        output_root / "optimized_pose_graph_with_marginals.png",
    )
    new_paths = (
        output_root / FIRST_BUNDLE_VISUALIZATION_FILENAME,
        output_root / POSE_GRAPH_VISUALIZATION_FILENAME,
    )
    if not summary_path.is_file():
        raise FileNotFoundError("canonical E6 summary.json is missing")
    if not values_path.is_file() or not marginals_path.is_file():
        raise FileNotFoundError("canonical E6 pose/marginal NPZ is missing")
    if not all(path.is_file() for path in raw_paths):
        raise FileNotFoundError("canonical raw covariance figures are missing")
    if any(path.exists() for path in new_paths):
        raise FileExistsError(
            "visualization-only covariance figure already exists"
        )

    with np.load(values_path, allow_pickle=False) as values:
        frame_ids = np.array(values["frame_ids"], copy=True)
        optimized_poses = np.array(
            values["optimized_poses"], dtype=float, copy=True
        )
    with np.load(marginals_path, allow_pickle=False) as marginal_data:
        marginal_frame_ids = np.array(
            marginal_data["frame_ids"], copy=True
        )
        marginal_covariances = np.array(
            marginal_data["marginal_covariances"],
            dtype=float,
            copy=True,
        )
    expected_frame_ids = np.asarray(inputs.keyframe_ids, dtype=np.int64)
    if not (
        np.array_equal(frame_ids, expected_frame_ids)
        and np.array_equal(marginal_frame_ids, expected_frame_ids)
        and optimized_poses.shape == (len(inputs.keyframe_ids), 4, 4)
        and marginal_covariances.shape
        == (len(inputs.keyframe_ids), 6, 6)
        and np.all(np.isfinite(optimized_poses))
        and np.all(np.isfinite(marginal_covariances))
    ):
        raise ValueError("canonical E6 plotting arrays differ")

    plot_first_bundle_camera_covariances(
        inputs,
        new_paths[0],
        covariance_multiplier=VISUALIZATION_COVARIANCE_MULTIPLIER,
        title="First Bundle window: visualization covariance ×0.01",
    )
    plot_pose_graph_with_marginals(
        optimized_poses,
        marginal_covariances,
        new_paths[1],
        covariance_multiplier=VISUALIZATION_COVARIANCE_MULTIPLIER,
        title="Optimized Pose Graph: visualization covariance ×0.01",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    visualization = summary.setdefault("visualization_covariance", {})
    visualization.update(
        {
            "raw_plots_preserved": True,
            "raw_covariance_figures_preserved": True,
            "additional_visualization_only_figures_created": True,
            "visualization_only_scaling_used": True,
            "multiplier": VISUALIZATION_COVARIANCE_MULTIPLIER,
            "visualization_only_covariance_multiplier": (
                VISUALIZATION_COVARIANCE_MULTIPLIER
            ),
            "factors_saved_covariance_arrays_marginals_metrics_and_"
            "scientific_results_remain_unscaled": True,
            "additional_figures": [path.name for path in new_paths],
        }
    )
    artifacts = set(summary.get("artifacts", []))
    artifacts.update(path.name for path in new_paths)
    summary["artifacts"] = sorted(artifacts)
    _save_json(summary_path, summary)
    return {
        "status": "pass",
        "visualization_only_covariance_multiplier": (
            VISUALIZATION_COVARIANCE_MULTIPLIER
        ),
        "created": [path.name for path in new_paths],
        "raw_covariance_figures_preserved": True,
        "scientific_results_remain_unscaled": True,
    }


def write_pose_graph_outputs(
    *,
    inputs: CanonicalPoseGraphInputs,
    pose_graph: PoseGraph,
    optimized_values: gtsam.Values,
    marginals: gtsam.Marginals,
    marginal_covariances: np.ndarray,
    initial_error: float,
    final_error: float,
    max_translation_m: float,
    max_rotation_rad: float,
    output_root: Path,
) -> Dict[str, object]:
    """Write the unchanged E6 arrays, figures, metrics, and summary."""

    output_root = Path(output_root)
    initial_poses = _pose_matrices(
        pose_graph.get_initial_estimate(), pose_graph.camera_keys
    )
    optimized_poses = _pose_matrices(
        optimized_values, pose_graph.camera_keys
    )
    np.savez(
        output_root / "pose_graph_values.npz",
        frame_ids=np.asarray(inputs.keyframe_ids, dtype="<i8"),
        initial_poses=initial_poses,
        optimized_poses=optimized_poses,
    )
    np.savez(
        output_root / "pose_graph_marginals.npz",
        frame_ids=np.asarray(inputs.keyframe_ids, dtype="<i8"),
        marginal_covariances=np.asarray(marginal_covariances, dtype="<f8"),
    )

    first_constraint = _first_constraint_record(inputs)
    _save_json(
        output_root / "first_consecutive_constraint.json",
        first_constraint,
    )
    print(
        "First consecutive relative pose (frames {} -> {}):\n{}".format(
            first_constraint["start_frame_id"],
            first_constraint["end_frame_id"],
            np.asarray(first_constraint["relative_pose_matrix"]),
        )
    )
    print(
        "Conditional end-given-start covariance "
        "[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]:\n{}".format(
            np.asarray(first_constraint["conditional_covariance"])
        )
    )

    plot_first_bundle_camera_covariances(
        inputs, output_root / "first_bundle_camera_covariances_3d.png"
    )
    plot_pose_graph(
        initial_poses,
        output_root / "initial_pose_graph.png",
        title="Initial consecutive-keyframe Pose Graph",
    )
    plot_pose_graph(
        optimized_poses,
        output_root / "optimized_pose_graph.png",
        title="Optimized consecutive-keyframe Pose Graph",
    )
    plot_pose_graph_with_marginals(
        optimized_poses,
        marginal_covariances,
        output_root / "optimized_pose_graph_with_marginals.png",
    )

    reporting = write_report_ready_absolute_errors(
        inputs, optimized_poses, output_root
    )
    tolerance = inputs.validation_tolerances[
        "numerical_reconstruction_atol"
    ]
    input_manifest = (
        inputs.input_root / "manifest.json"
    ).relative_to(REPOSITORY_ROOT)
    prior_construction_source = (
        "{}::scientific_configuration.gauge_fixing_prior; "
        "exercise5_full_run.py::run_streamed uses "
        "gtsam.noiseModel.Diagonal.Sigmas(rotation_sigma_rad + "
        "translation_sigma_m)"
    ).format(input_manifest)
    summary = {
        "schema": "exercise6-pose-graph-summary-v1",
        "milestone": "E6-I01",
        "status": "pass",
        "scope": "consecutive keyframes only; no Loop Closure",
        "canonical_input_root": str(
            inputs.input_root.relative_to(REPOSITORY_ROOT)
        ),
        "canonical_configuration_sha256": inputs.configuration_sha256,
        "keyframe_count": len(inputs.keyframe_ids),
        "edge_count": len(inputs.edges),
        "first_frame_id": inputs.keyframe_ids[0],
        "last_frame_id": inputs.keyframe_ids[-1],
        "node_symbol_policy": "gtsam.symbol('c', actual FrameId)",
        "factor_counts": {
            "PriorFactorPose3": 1,
            "BetweenFactorPose3": len(inputs.edges),
            "total": pose_graph.graph().size(),
        },
        "variable_count": pose_graph.get_initial_estimate().size(),
        "prior": {
            "construction_source": prior_construction_source,
            "target": "identity Pose3 at frame 0",
            "noise_model": "gtsam.noiseModel.Diagonal.Sigmas",
            "sigmas": inputs.prior_sigmas.tolist(),
            "sigma_order": list(TANGENT_ORDER),
        },
        "edge_direction": EDGE_DIRECTION,
        "covariance": {
            "field": CONDITIONAL_COVARIANCE_FIELD,
            "tangent_order": list(TANGENT_ORDER),
            "factor_construction": (
                "gtsam.noiseModel.Gaussian.Covariance(raw stored "
                "conditional covariance)"
            ),
            "adjoint_transport_applied": False,
            "scaling_applied_to_factors_or_saved_arrays": False,
            "regularization_or_repair_applied": False,
        },
        "initialization": "X_0=Identity; X_{i+1}=X_i.compose(stored edge_i)",
        "initial_edge_reconstruction_max_abs": (
            pose_graph.initial_edge_reconstruction_max_abs
        ),
        "initial_graph_error": initial_error,
        "final_graph_error": final_error,
        "maximum_initial_to_optimized_translation_change_m": (
            max_translation_m
        ),
        "maximum_initial_to_optimized_rotation_change_rad": max_rotation_rad,
        "maximum_initial_to_optimized_rotation_change_deg": float(
            np.degrees(max_rotation_rad)
        ),
        "movement_comparison_tolerance_source": (
            "canonical E5 numerical_reconstruction_atol"
        ),
        "movement_comparison_tolerance": tolerance,
        "optimizer": (
            "one gtsam.LevenbergMarquardtOptimizer with GTSAM 4.2 defaults"
        ),
        "marginals": {
            "construction": "gtsam.Marginals(graph, optimized_values)",
            "success": isinstance(marginals, gtsam.Marginals),
            "count": int(marginal_covariances.shape[0]),
            "saved_covariances_are_raw": True,
        },
        "visualization_covariance": {
            "raw_plots_preserved": True,
            "visualization_only_scaling_used": False,
            "multiplier": None,
        },
        "first_bundle_camera_count": int(inputs.first_bundle_frame_ids.size),
        "report_ready_absolute_error": reporting,
        "execution_boundaries": {
            "tracking_database_loaded": False,
            "bundle_graph_loaded": False,
            "bundle_window_solved": False,
            "e5_i03c_artifacts_used": False,
            "loop_closure_started": False,
        },
        "artifacts": sorted(
            path.name for path in output_root.iterdir() if path.is_file()
        ),
    }
    _save_json(output_root / "summary.json", summary)
    return summary


__all__ = [
    "create_visualization_only_covariance_figures",
    "plot_first_bundle_camera_covariances",
    "plot_pose_graph",
    "plot_pose_graph_with_marginals",
    "write_pose_graph_outputs",
    "write_report_ready_absolute_errors",
]
