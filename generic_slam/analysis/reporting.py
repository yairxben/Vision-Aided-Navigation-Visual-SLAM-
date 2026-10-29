"""Artifact-only analysis and report generation for a completed SLAM run."""

from __future__ import annotations

import csv
import os
import resource
import tempfile
import time
from collections import Counter
from uuid import uuid4
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from ..artifacts.io import (
    read_json,
    require_files,
    save_npz_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from ..artifacts.manifest import (
    StageManifest,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)
from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiStereoSequence
from .bundle import bundle_window_rows
from .loop_closure import loop_summary
from .pose_graph import uncertainty_rows
from .tracking import summarize_tracking, tracking_analysis_rows
from .trajectories import (
    absolute_error_rows,
    camera_to_world_from_extrinsics,
    normalize_camera_to_world,
    relative_error_rows,
    subsection_error_rows,
)

if TYPE_CHECKING:
    from ..bundle.stage import BundleOutput
    from ..keyframes.stage import KeyframeOutput
    from ..loop_closure.stage import LoopClosureOutput
    from ..pose_graph.stage import PoseGraphArtifacts
    from ..tracking.stage import TrackingTrajectoryOutput


ANALYSIS_PRIMARY_ARTIFACTS = (
    "summary.json",
    "tracking/frame_links.csv",
    "tracking/track_lengths.csv",
    "tracking/transitions.csv",
    "tracking/transition_reprojection.csv",
    "tracking/representative_track.csv",
    "pnp/trajectory.npz",
    "pnp/absolute_errors.csv",
    "pnp/consecutive_relative_errors.csv",
    "bundle/window_metrics.csv",
    "bundle/rejections.csv",
    "bundle/absolute_errors.csv",
    "pose_graph/uncertainty.csv",
    "pose_graph/absolute_errors_without_loop.csv",
    "loop_closure/candidate_dispositions.csv",
    "loop_closure/accepted_loops.csv",
    "loop_closure/summary.json",
    "loop_closure/absolute_errors_with_loop.csv",
    "cross_stage/trajectories.npz",
    "cross_stage/keyframe_trajectories.csv",
    "cross_stage/absolute_errors.csv",
    "cross_stage/consecutive_relative_errors.csv",
    "cross_stage/keyframe_relative_errors.csv",
    "tracking/track_statistics.json",
)

FRAME_LINK_FIELDS = ("frame_id", "tracked_link_count", "tracks_continuing_to_next_frame")
TRACK_LENGTH_FIELDS = (
    "track_id", "track_length_frames", "first_frame_id", "last_frame_id",
)
REPRESENTATIVE_TRACK_FIELDS = (
    "track_id", "frame_id", "u_left_px", "u_right_px", "v_mean_px",
)
REPROJECTION_FIELDS = (
    "previous_frame_id", "current_frame_id", "consensus_count", "supporter_count",
    "mean_supporter_reprojection_error_px", "median_supporter_reprojection_error_px",
    "maximum_supporter_reprojection_error_px",
)
BUNDLE_FIELDS = (
    "window_ordinal", "start_frame_id", "end_frame_id", "window_frame_count",
    "candidate_track_count", "accepted_track_count", "rejected_track_count",
    "projection_factor_count", "initial_graph_error", "final_graph_error",
    "mean_final_projection_residual_px", "median_final_projection_residual_px",
    "maximum_final_projection_residual_px",
)
BUNDLE_REJECTION_FIELDS = ("window_ordinal", "track_id", "reason")
UNCERTAINTY_FIELDS = (
    "stage", "frame_id", "rotation_uncertainty_deg", "location_uncertainty_m",
    "rot_x_variance", "rot_y_variance", "rot_z_variance", "trans_x_variance",
    "trans_y_variance", "trans_z_variance",
)
TRAJECTORY_FIELDS = (
    "stage", "frame_id", "x_m", "y_m", "z_m", "ground_truth_x_m",
    "ground_truth_y_m", "ground_truth_z_m",
)
ABSOLUTE_FIELDS = (
    "stage", "frame_id", "translation_error_m", "rotation_error_deg",
    "x_error_m", "y_error_m", "z_error_m",
    "x_signed_error_m", "y_signed_error_m", "z_signed_error_m",
)
RELATIVE_FIELDS = (
    "stage", "start_frame_id", "end_frame_id", "translation_error_m",
    "rotation_error_deg",
)
SUBSECTION_FIELDS = (
    "stage", "requested_length_frames", "start_frame_id", "end_frame_id",
    "actual_frame_delta", "truth_endpoint_distance_m", "translation_error_m",
    "rotation_error_deg", "translation_error_percent_of_truth_endpoint_distance",
    "rotation_error_deg_per_m",
)


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _quarantine_unsealed(root: Path) -> None:
    if root.exists() and not (root / "manifest.json").is_file():
        destination = root.with_name(root.name + ".failed-" + uuid4().hex)
        os.replace(str(root), str(destination))


def _attempt_directory(root: Path) -> Path:
    root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".{}-attempt-".format(root.name), dir=str(root.parent)))


def _csv_rows(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _accepted_candidate_rows(accepted_rows, candidate_rows):
    """Join accepted summaries to their unique matcher-bearing candidate rows."""

    accepted_by_id = {}
    for row in candidate_rows:
        if row.get("disposition") != "accepted":
            continue
        acceptance_id = row.get("acceptance_id", "")
        if not acceptance_id:
            raise ContractError("accepted candidate is missing acceptance_id")
        if acceptance_id in accepted_by_id:
            raise ContractError(
                "multiple accepted candidates share acceptance_id {}".format(acceptance_id)
            )
        accepted_by_id[acceptance_id] = row

    ordered = []
    for accepted in accepted_rows:
        acceptance_id = accepted.get("acceptance_id", "")
        if acceptance_id not in accepted_by_id:
            raise ContractError(
                "accepted-loop summary {} has no accepted candidate row".format(
                    acceptance_id or "<blank>"
                )
            )
        ordered.append(accepted_by_id[acceptance_id])
    return ordered


def _finite_summary(rows, translation_key: str, rotation_key: str) -> Dict[str, float]:
    translation = np.asarray([float(row[translation_key]) for row in rows], dtype=float)
    rotation = np.asarray([float(row[rotation_key]) for row in rows], dtype=float)
    translation = translation[np.isfinite(translation)]
    rotation = rotation[np.isfinite(rotation)]
    if not len(translation) or not len(rotation):
        return {}
    return {
        "translation_mean_m": float(np.mean(translation)),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(translation)))),
        "translation_median_m": float(np.median(translation)),
        "translation_maximum_m": float(np.max(translation)),
        "translation_final_m": float(translation[-1]),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotation)))),
        "rotation_median_deg": float(np.median(rotation)),
        "rotation_maximum_deg": float(np.max(rotation)),
        "rotation_final_deg": float(rotation[-1]),
    }


def _trajectory_population(
    tracking: TrackingTrajectoryOutput,
    bundle: BundleOutput,
    pose_graph: PoseGraphArtifacts,
    loop_closure: LoopClosureOutput,
):
    return {
        "pnp": (
            np.asarray(tracking.frame_ids, dtype=np.int64),
            normalize_camera_to_world(tracking.camera_to_world),
        ),
        "bundle": (
            np.asarray(bundle.keyframe_ids, dtype=np.int64),
            normalize_camera_to_world(bundle.camera_to_first_keyframe),
        ),
        "pose_graph_without_loop": (
            np.asarray(pose_graph.keyframe_ids, dtype=np.int64),
            normalize_camera_to_world(pose_graph.optimized_pose_matrices),
        ),
        "pose_graph_with_loop": (
            np.asarray(loop_closure.keyframe_ids, dtype=np.int64),
            normalize_camera_to_world(loop_closure.final_pose_matrices),
        ),
    }


def _validate_analysis_artifacts(root: Path, keyframe_count: int) -> None:
    require_files(root, ANALYSIS_PRIMARY_ARTIFACTS)
    with np.load(root / "cross_stage" / "trajectories.npz", allow_pickle=False) as data:
        required = {
            "pnp_frame_ids", "pnp_pose_matrices", "keyframe_ids",
            "bundle_pose_matrices", "pose_graph_without_loop_pose_matrices",
            "pose_graph_with_loop_pose_matrices",
        }
        if not required.issubset(data.files):
            raise ContractError("cross-stage trajectory artifact is incomplete")
        if data["keyframe_ids"].shape != (keyframe_count,):
            raise ContractError("cross-stage keyframe population differs")
        for name in (
            "bundle_pose_matrices", "pose_graph_without_loop_pose_matrices",
            "pose_graph_with_loop_pose_matrices",
        ):
            if data[name].shape != (keyframe_count, 4, 4) or not np.all(np.isfinite(data[name])):
                raise ContractError("cross-stage trajectory shape differs")


def run_analysis_stage(
    dataset: KittiStereoSequence,
    config: AlgorithmConfig,
    tracking: TrackingTrajectoryOutput,
    keyframes: KeyframeOutput,
    bundle: BundleOutput,
    pose_graph: PoseGraphArtifacts,
    loop_closure: LoopClosureOutput,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
) -> StageManifest:
    """Generate all reusable numerical reporting inputs without estimators."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_stage_manifest(manifest_path)
        validate_stage_manifest(
            manifest, root, expected_stage="analysis", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("analysis"),
            upstream_stage_ids=upstream_stage_ids,
        )
        _validate_analysis_artifacts(root, len(keyframes.selection.frame_ids))
        return manifest
    _quarantine_unsealed(root)
    attempt = _attempt_directory(root)
    started = time.perf_counter()
    try:
        (
            frame_rows, track_rows, transition_rows, reprojection_rows,
            representative_rows,
        ) = tracking_analysis_rows(root.parent / "tracking")
        write_csv_atomic(attempt / "tracking" / "frame_links.csv", frame_rows, FRAME_LINK_FIELDS)
        write_csv_atomic(attempt / "tracking" / "track_lengths.csv", track_rows, TRACK_LENGTH_FIELDS)
        transition_fields = tuple(transition_rows[0].keys()) if transition_rows else (
            "previous_frame_id", "current_frame_id",
        )
        write_csv_atomic(attempt / "tracking" / "transitions.csv", transition_rows, transition_fields)
        write_csv_atomic(
            attempt / "tracking" / "transition_reprojection.csv",
            reprojection_rows,
            REPROJECTION_FIELDS,
        )
        write_csv_atomic(
            attempt / "tracking" / "representative_track.csv",
            representative_rows,
            REPRESENTATIVE_TRACK_FIELDS,
        )
        tracking_summary = summarize_tracking(frame_rows, track_rows, transition_rows)
        write_json_atomic(attempt / "tracking" / "track_statistics.json", tracking_summary)

        window_rows = bundle_window_rows(root.parent / "bundle")
        write_csv_atomic(attempt / "bundle" / "window_metrics.csv", window_rows, BUNDLE_FIELDS)
        rejection_rows = []
        for directory in sorted((root.parent / "bundle" / "windows").glob("window_*")):
            if not (directory / "seal.json").is_file():
                continue
            ordinal = int(read_json(directory / "summary.json")["window_ordinal"])
            for row in _csv_rows(directory / "rejections.csv"):
                rejection_rows.append({
                    "window_ordinal": ordinal,
                    "track_id": int(row["track_id"]),
                    "reason": row["reason"],
                })
        write_csv_atomic(
            attempt / "bundle" / "rejections.csv",
            rejection_rows,
            BUNDLE_REJECTION_FIELDS,
        )
        uncertainty = uncertainty_rows(
            pose_graph.keyframe_ids, pose_graph.marginal_covariances,
            "pose_graph_without_loop",
        ) + uncertainty_rows(
            loop_closure.keyframe_ids, loop_closure.final_marginal_covariances,
            "pose_graph_with_loop",
        )
        write_csv_atomic(attempt / "pose_graph" / "uncertainty.csv", uncertainty, UNCERTAINTY_FIELDS)

        candidate_rows, accepted_rows, disposition_counts = loop_summary(root.parent / "loop_closure")
        candidate_fields = tuple(candidate_rows[0].keys()) if candidate_rows else (
            "previous_frame_id", "current_frame_id", "disposition",
        )
        accepted_fields = tuple(accepted_rows[0].keys()) if accepted_rows else (
            "acceptance_id", "previous_frame_id", "current_frame_id",
            "previous_keyframe_ordinal", "current_keyframe_ordinal",
        )
        write_csv_atomic(
            attempt / "loop_closure" / "candidate_dispositions.csv",
            candidate_rows,
            candidate_fields,
        )
        write_csv_atomic(
            attempt / "loop_closure" / "accepted_loops.csv",
            accepted_rows,
            accepted_fields,
        )
        write_json_atomic(
            attempt / "loop_closure" / "summary.json",
            {
                "ordered_pair_count": len(candidate_rows),
                "accepted_loop_count": len(accepted_rows),
                "disposition_counts": disposition_counts,
            },
        )

        populations = _trajectory_population(tracking, bundle, pose_graph, loop_closure)
        keyframe_ids = np.asarray(keyframes.selection.frame_ids, dtype=np.int64)
        if not all(np.array_equal(ids, keyframe_ids) for name, (ids, _) in populations.items() if name != "pnp"):
            raise ContractError("cross-stage keyframe populations differ")
        pnp_keyframes = populations["pnp"][1][keyframe_ids]
        keyframe_poses = {
            "pnp": pnp_keyframes,
            "bundle": populations["bundle"][1],
            "pose_graph_without_loop": populations["pose_graph_without_loop"][1],
            "pose_graph_with_loop": populations["pose_graph_with_loop"][1],
        }
        trajectory_arrays = {
            "pnp_frame_ids": populations["pnp"][0],
            "pnp_pose_matrices": populations["pnp"][1],
            "keyframe_ids": keyframe_ids,
        }
        for stage, poses in keyframe_poses.items():
            trajectory_arrays[stage + "_pose_matrices"] = poses
        save_npz_atomic(
            attempt / "pnp" / "trajectory.npz",
            frame_ids=populations["pnp"][0],
            pose_matrices=populations["pnp"][1],
            positions=populations["pnp"][1][:, :3, 3],
        )

        gt_camera_to_world: Optional[np.ndarray] = None
        if dataset.ground_truth_world_to_camera is not None:
            gt_camera_to_world = camera_to_world_from_extrinsics(
                dataset.ground_truth_world_to_camera
            )
            trajectory_arrays["ground_truth_frame_ids"] = np.asarray(dataset.frame_ids, dtype=np.int64)
            trajectory_arrays["ground_truth_pose_matrices"] = gt_camera_to_world
        save_npz_atomic(attempt / "cross_stage" / "trajectories.npz", **trajectory_arrays)

        trajectory_rows = []
        for stage, poses in keyframe_poses.items():
            for ordinal, (frame_id, pose) in enumerate(zip(keyframe_ids, poses)):
                truth_position = (
                    gt_camera_to_world[int(frame_id), :3, 3]
                    if gt_camera_to_world is not None else ("", "", "")
                )
                trajectory_rows.append(
                    {
                        "stage": stage,
                        "frame_id": int(frame_id),
                        "x_m": float(pose[0, 3]),
                        "y_m": float(pose[1, 3]),
                        "z_m": float(pose[2, 3]),
                        "ground_truth_x_m": truth_position[0],
                        "ground_truth_y_m": truth_position[1],
                        "ground_truth_z_m": truth_position[2],
                    }
                )
        write_csv_atomic(
            attempt / "cross_stage" / "keyframe_trajectories.csv",
            trajectory_rows,
            TRAJECTORY_FIELDS,
        )

        absolute_rows = []
        relative_rows = []
        keyframe_relative_rows = []
        subsection_by_length = {
            int(length): []
            for length in config.section("analysis")["relative_subsection_lengths_frames"]
        }
        metric_summary: Dict[str, object] = {}
        gt_omissions = []
        if gt_camera_to_world is None:
            gt_omissions = [
                "absolute trajectory errors",
                "relative trajectory errors",
                "Ground Truth overlays and path-distance-normalized subsections",
            ]
        else:
            for stage, (ids, poses) in populations.items():
                truth = normalize_camera_to_world(gt_camera_to_world[ids])
                stage_absolute = absolute_error_rows(ids, poses, truth, stage)
                stage_relative = relative_error_rows(ids, poses, truth, stage)
                absolute_rows.extend(stage_absolute)
                relative_rows.extend(stage_relative)
                metric_summary[stage] = {
                    "absolute": _finite_summary(
                        stage_absolute, "translation_error_m", "rotation_error_deg"
                    ),
                    "consecutive_relative": _finite_summary(
                        stage_relative, "translation_error_m", "rotation_error_deg"
                    ),
                }
                for length, rows in subsection_by_length.items():
                    rows.extend(subsection_error_rows(ids, poses, truth, stage, length))
            # Same keyframe pairs for every stage, so accumulated PnP is
            # directly comparable with Bundle Adjustment.
            keyframe_truth = normalize_camera_to_world(gt_camera_to_world[keyframe_ids])
            for stage, poses in (
                ("pnp_accumulated", keyframe_poses["pnp"]),
                ("bundle", keyframe_poses["bundle"]),
            ):
                stage_rows = relative_error_rows(keyframe_ids, poses, keyframe_truth, stage)
                keyframe_relative_rows.extend(stage_rows)
                metric_summary.setdefault("keyframe_relative", {})[stage] = _finite_summary(
                    stage_rows, "translation_error_m", "rotation_error_deg"
                )
        write_csv_atomic(
            attempt / "cross_stage" / "keyframe_relative_errors.csv",
            keyframe_relative_rows,
            RELATIVE_FIELDS,
        )
        write_csv_atomic(
            attempt / "cross_stage" / "absolute_errors.csv", absolute_rows, ABSOLUTE_FIELDS
        )
        write_csv_atomic(
            attempt / "cross_stage" / "consecutive_relative_errors.csv",
            relative_rows,
            RELATIVE_FIELDS,
        )
        stage_error_destinations = {
            "pnp": (
                "pnp/absolute_errors.csv", "pnp/consecutive_relative_errors.csv"
            ),
            "bundle": ("bundle/absolute_errors.csv", None),
            "pose_graph_without_loop": (
                "pose_graph/absolute_errors_without_loop.csv", None
            ),
            "pose_graph_with_loop": (
                "loop_closure/absolute_errors_with_loop.csv", None
            ),
        }
        for stage, (absolute_path, relative_path) in stage_error_destinations.items():
            write_csv_atomic(
                attempt / absolute_path,
                [row for row in absolute_rows if row["stage"] == stage],
                ABSOLUTE_FIELDS,
            )
            if relative_path is not None:
                write_csv_atomic(
                    attempt / relative_path,
                    [row for row in relative_rows if row["stage"] == stage],
                    RELATIVE_FIELDS,
                )
        subsection_paths = []
        for length, rows in subsection_by_length.items():
            relative = "cross_stage/relative_subsections_{}.csv".format(length)
            write_csv_atomic(attempt / relative, rows, SUBSECTION_FIELDS)
            subsection_paths.append(relative)

        link_counts = [int(row["tracked_link_count"]) for row in frame_rows]
        track_lengths = [int(row["track_length_frames"]) for row in track_rows]

        write_json_atomic(
            attempt / "summary.json",
            {
                "schema": "generic-slam-analysis-summary-v1",
                "ground_truth_available": gt_camera_to_world is not None,
                "ground_truth_omissions": gt_omissions,
                "frame_count": dataset.frame_count,
                "keyframe_count": len(keyframe_ids),
                "track_count": len(track_rows),
                "tracking_statistics": {
                    "mean_links_per_frame": float(np.mean(link_counts)) if link_counts else None,
                    "mean_track_length_frames": float(np.mean(track_lengths)) if track_lengths else None,
                    "maximum_track_length_frames": max(track_lengths) if track_lengths else None,
                    "low_connectivity_frame_count": sum(
                        int(row["tracked_link_count"])
                        < int(config.section("analysis")["low_connectivity_threshold"])
                        for row in frame_rows
                    ),
                    "median_track_length_frames": tracking_summary["track_length_frames"]["median"],
                    "mean_tracks_continuing_to_next_frame": tracking_summary[
                        "mean_tracks_continuing_to_next_frame"
                    ],
                    "tracks_with_length_at_least": tracking_summary["tracks_with_length_at_least"],
                },
                "matching_funnel_means": tracking_summary["per_transition_means"],
                "bundle_window_count": len(window_rows),
                "bundle_rejection_reason_counts": dict(sorted(Counter(
                    row["reason"] for row in rejection_rows
                ).items())),
                "ordered_loop_pair_count": len(candidate_rows),
                "accepted_loop_count": len(accepted_rows),
                "trajectory_error_metrics": metric_summary,
                "relative_subsection_artifacts": subsection_paths,
            },
        )
        primary = ANALYSIS_PRIMARY_ARTIFACTS + tuple(subsection_paths)
        _validate_analysis_artifacts(attempt, len(keyframe_ids))
        manifest = StageManifest.complete(
            stage="analysis", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("analysis"),
            upstream_stage_ids=upstream_stage_ids,
            counts={
                "frame_count": dataset.frame_count,
                "keyframe_count": len(keyframe_ids),
                "track_count": len(track_rows),
                "bundle_window_count": len(window_rows),
                "ordered_loop_pair_count": len(candidate_rows),
                "accepted_loop_count": len(accepted_rows),
                "ground_truth_pose_count": (
                    len(gt_camera_to_world) if gt_camera_to_world is not None else 0
                ),
            },
            primary_artifacts=primary,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
        )
        write_stage_manifest(attempt, manifest)
        os.replace(str(attempt), str(root))
        return manifest
    except Exception:
        if attempt.exists():
            os.replace(str(attempt), str(attempt) + ".failed")
        raise


def _save_figure(path: Path, figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_lines(
    path: Path, x, series, title: str, ylabel: str, xlabel: str = "FrameId / ordinal"
) -> None:
    figure, axis = plt.subplots(figsize=(9, 4.8))
    for label, values in series:
        axis.plot(x[: len(values)], values, label=label, linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if len(series) > 1:
        axis.legend()
    _save_figure(path, figure)


def _plot_trajectories(path: Path, trajectories, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    for label, poses in trajectories:
        positions = poses[:, :3, 3]
        axis.plot(positions[:, 0], positions[:, 2], label=label, linewidth=1.3)
    axis.set_title(title)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Z [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)
    axis.legend()
    _save_figure(path, figure)


def _plot_raw_covariance_trajectory(
    path: Path,
    poses: np.ndarray,
    covariances: np.ndarray,
    title: str,
    stride: int,
    extra_ordinals=(),
) -> None:
    """Plot raw 95% X-Z ellipses after the validated local-to-global rotation."""

    figure, axis = plt.subplots(figsize=(8, 7))
    positions = poses[:, :3, 3]
    axis.plot(positions[:, 0], positions[:, 2], linewidth=1.3)
    ordinals = set(range(0, len(poses), stride)).union(int(value) for value in extra_ordinals)
    ordinals.add(len(poses) - 1)
    chi_square_95 = 5.991464547107979
    for ordinal in sorted(value for value in ordinals if 0 <= value < len(poses)):
        rotation = poses[ordinal, :3, :3]
        global_translation = rotation @ covariances[ordinal, 3:, 3:] @ rotation.T
        covariance_xz = global_translation[np.ix_((0, 2), (0, 2))]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_xz)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        angle = float(np.degrees(np.arctan2(major[1], major[0])))
        width, height = 2.0 * np.sqrt(chi_square_95 * eigenvalues[::-1])
        axis.add_patch(Ellipse(
            (positions[ordinal, 0], positions[ordinal, 2]),
            width=float(width), height=float(height), angle=angle,
            fill=False, linewidth=0.7, alpha=0.65,
        ))
    axis.set_title(title)
    axis.set_xlabel("X [m]")
    axis.set_ylabel("Z [m]")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)
    _save_figure(path, figure)


def _format_number(value, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return "{:,}".format(value)
    return "{:,.{}f}".format(float(value), digits)


def _summary_markdown(
    analysis_root: Path, output_root: Path, config: AlgorithmConfig, generated
) -> str:
    """Human-readable run summary: tracks, matching funnel, errors, timing."""

    summary = read_json(Path(analysis_root) / "summary.json")
    tracks = read_json(Path(analysis_root) / "tracking" / "track_statistics.json")
    frontend = config.section("frontend")
    detector = frontend["detector"]
    detector_setting = (
        "threshold {}".format(frontend["akaze"]["threshold"]) if detector == "akaze"
        else "max features {}".format(frontend["orb"]["max_features"]) if detector == "orb"
        else "contrast threshold {}".format(frontend["sift"]["contrast_threshold"])
    )
    lines = [
        "# SLAM run summary",
        "",
        "- Sequence frames: {} | keyframes: {} | Bundle windows: {}".format(
            summary["frame_count"], summary["keyframe_count"], summary["bundle_window_count"]
        ),
        "- Feature detector: {} ({}) | config mode: {}".format(
            detector, detector_setting, config.mode
        ),
        "- Accepted Loop Closures: {}".format(summary["accepted_loop_count"]),
        "",
        "## Tracking statistics",
        "",
        "| Statistic | Value |",
        "|---|---|",
        "| Tracks | {} |".format(_format_number(tracks["track_count"])),
        "| Mean / median / max track length [frames] | {} / {} / {} |".format(
            _format_number(tracks["track_length_frames"]["mean"]),
            _format_number(tracks["track_length_frames"]["median"], 1),
            _format_number(tracks["track_length_frames"]["maximum"]),
        ),
        "| Mean tracked links per frame | {} |".format(
            _format_number(tracks["mean_tracked_links_per_frame"], 1)
        ),
        "| Mean / min tracks continuing to next frame | {} / {} |".format(
            _format_number(tracks["mean_tracks_continuing_to_next_frame"], 1),
            _format_number(tracks["minimum_tracks_continuing_to_next_frame"]),
        ),
    ]
    # JSON stores the keys as sorted strings; present them in numeric order.
    for minimum, count in sorted(
        tracks["tracks_with_length_at_least"].items(), key=lambda item: int(item[0])
    ):
        lines.append("| Tracks with length >= {} frames | {} |".format(minimum, _format_number(count)))
    funnel = tracks["per_transition_means"]
    lines += [
        "",
        "## Matching funnel (mean per consecutive-frame transition)",
        "",
        "| Stage | Mean matches |",
        "|---|---|",
        "| Stereo matches (geometric row/disparity check) | {} |".format(
            _format_number(funnel["stereo_matches"], 1)
        ),
        "| Four-view consensus | {} |".format(_format_number(funnel["four_view_consensus"], 1)),
        "| PnP/RANSAC supporters | {} |".format(_format_number(funnel["pnp_ransac_supporters"], 1)),
        "| Supporters as % of consensus | {} % |".format(
            _format_number(funnel["supporter_percentage_of_consensus"], 1)
        ),
    ]
    metrics = summary.get("trajectory_error_metrics", {})
    if metrics:
        lines += [
            "",
            "## Absolute trajectory error vs Ground Truth",
            "",
            "| Trajectory | Mean [m] | RMSE [m] | Max [m] | Final [m] | Rotation RMSE [deg] |",
            "|---|---|---|---|---|---|",
        ]
        for stage in ("pnp", "bundle", "pose_graph_without_loop", "pose_graph_with_loop"):
            absolute = metrics.get(stage, {}).get("absolute", {})
            if absolute:
                lines.append("| {} | {} | {} | {} | {} | {} |".format(
                    stage.replace("_", " "),
                    _format_number(absolute["translation_mean_m"]),
                    _format_number(absolute["translation_rmse_m"]),
                    _format_number(absolute["translation_maximum_m"]),
                    _format_number(absolute["translation_final_m"]),
                    _format_number(absolute["rotation_rmse_deg"]),
                ))
        keyframe_relative = metrics.get("keyframe_relative", {})
        if keyframe_relative:
            lines += [
                "",
                "## Relative error between consecutive keyframes",
                "",
                "| Estimate | Mean translation [m] | RMSE translation [m] | Mean rotation [deg] |",
                "|---|---|---|---|",
            ]
            for stage, label in (("pnp_accumulated", "accumulated PnP"), ("bundle", "Bundle Adjustment")):
                values = keyframe_relative.get(stage, {})
                if values:
                    lines.append("| {} | {} | {} | {} |".format(
                        label,
                        _format_number(values["translation_mean_m"], 3),
                        _format_number(values["translation_rmse_m"], 3),
                        _format_number(values["rotation_mean_deg"], 3),
                    ))
    lines += ["", "## Stage runtime", "", "| Stage | Minutes |", "|---|---|"]
    for stage in ("tracking", "keyframes", "bundle", "pose_graph", "loop_closure", "analysis"):
        manifest_path = Path(output_root) / stage / "manifest.json"
        if manifest_path.is_file():
            minutes = load_stage_manifest(manifest_path).runtime_seconds / 60.0
            lines.append("| {} | {} |".format(stage, _format_number(minutes, 1)))
    lines += ["", "## Figures", ""]
    lines += ["- [{0}]({0})".format(relative) for relative in generated]
    return "\n".join(lines) + "\n"


def _read_float(rows, name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _validate_report_artifacts(root: Path) -> None:
    require_files(root, ("summary.json", "figures/cross_stage_trajectory.png"))
    summary = read_json(root / "summary.json")
    if not isinstance(summary, dict) or not isinstance(summary.get("generated_figures"), list):
        raise ContractError("report summary is invalid")
    require_files(root, tuple(summary["generated_figures"]))


def run_report_stage(
    dataset: KittiStereoSequence,
    config: AlgorithmConfig,
    analysis_root: Path,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
) -> StageManifest:
    """Render figures exclusively from durable dataset/scientific artifacts."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_stage_manifest(manifest_path)
        validate_stage_manifest(
            manifest, root, expected_stage="report", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("report"),
            upstream_stage_ids=upstream_stage_ids,
        )
        _validate_report_artifacts(root)
        return manifest
    _quarantine_unsealed(root)
    attempt = _attempt_directory(root)
    started = time.perf_counter()
    generated = []
    skipped = []

    def figure_path(name: str) -> Path:
        relative = "figures/" + name
        generated.append(relative)
        return attempt / relative

    try:
        frame_rows = _csv_rows(Path(analysis_root) / "tracking" / "frame_links.csv")
        track_rows = _csv_rows(Path(analysis_root) / "tracking" / "track_lengths.csv")
        transition_rows = _csv_rows(Path(analysis_root) / "tracking" / "transitions.csv")
        reprojection_rows = _csv_rows(
            Path(analysis_root) / "tracking" / "transition_reprojection.csv"
        )
        _plot_lines(
            figure_path("tracking_connectivity.png"),
            _read_float(frame_rows, "frame_id"),
            [("tracked links", _read_float(frame_rows, "tracked_link_count"))],
            "Tracking connectivity", "Tracked links",
        )
        # The final frame has no following frame.
        _plot_lines(
            figure_path("tracks_continuing_to_next_frame.png"),
            _read_float(frame_rows[:-1], "frame_id"),
            [("tracks", _read_float(frame_rows[:-1], "tracks_continuing_to_next_frame"))],
            "Feature tracks in each frame that remain connected to the following frame",
            "Tracks", xlabel="FrameId",
        )
        lengths = _read_float(track_rows, "track_length_frames")
        figure, axis = plt.subplots(figsize=(8, 4.8))
        axis.hist(lengths, bins=min(60, max(1, len(np.unique(lengths)))))
        axis.set_title("Track-length distribution")
        axis.set_xlabel("Track length [frames]")
        axis.set_ylabel("Tracks")
        axis.grid(True, alpha=0.25)
        _save_figure(figure_path("track_length_distribution.png"), figure)
        representative = _csv_rows(
            Path(analysis_root) / "tracking" / "representative_track.csv"
        )
        if representative:
            _plot_lines(
                figure_path("representative_track.png"),
                _read_float(representative, "frame_id"),
                [
                    ("left u", _read_float(representative, "u_left_px")),
                    ("right u", _read_float(representative, "u_right_px")),
                    ("mean v", _read_float(representative, "v_mean_px")),
                ],
                "Representative longest track", "Pixel coordinate",
            )
        else:
            skipped.append({
                "figure": "figures/representative_track.png",
                "reason": "the current tracking database contains no completed tracks",
            })
        transition_x = np.arange(len(transition_rows))
        _plot_lines(
            figure_path("transition_matching_statistics.png"), transition_x,
            [
                ("consensus", _read_float(transition_rows, "consensus_count")),
                ("supporters", _read_float(transition_rows, "supporter_count")),
            ],
            "Temporal consensus and PnP supporters", "Matches",
        )
        # Each transition is labelled by its current FrameId.
        current_frame_x = _read_float(transition_rows, "current_frame_id")
        _plot_lines(
            figure_path("matching_stages.png"), current_frame_x,
            [
                ("stereo matches (current frame)",
                 _read_float(transition_rows, "current_stereo_row_count")),
                ("four-view consensus", _read_float(transition_rows, "consensus_count")),
                ("PnP/RANSAC supporters", _read_float(transition_rows, "supporter_count")),
            ],
            "Matching stages for each consecutive-frame transition",
            "Matches", xlabel="Current FrameId",
        )
        _plot_lines(
            figure_path("supporter_rate.png"), current_frame_x,
            [("supporter percentage", _read_float(transition_rows, "supporter_percentage"))],
            "Four-view consensus matches accepted as PnP/RANSAC supporters",
            "Supporters [% of consensus]", xlabel="Current FrameId",
        )
        _plot_lines(
            figure_path("tracking_reprojection.png"), transition_x,
            [("mean", _read_float(reprojection_rows, "mean_supporter_reprojection_error_px")),
             ("maximum", _read_float(reprojection_rows, "maximum_supporter_reprojection_error_px"))],
            "PnP supporter reprojection diagnostics", "Pixel error",
        )

        with np.load(Path(analysis_root) / "cross_stage" / "trajectories.npz", allow_pickle=False) as data:
            trajectories = {name: data[name].copy() for name in data.files}
        stages = (
            "pnp", "bundle", "pose_graph_without_loop", "pose_graph_with_loop",
        )
        common = [(stage.replace("_", " "), trajectories[stage + "_pose_matrices"]) for stage in stages]
        if "ground_truth_pose_matrices" in trajectories:
            common.insert(0, ("Ground Truth", trajectories["ground_truth_pose_matrices"]))
        _plot_trajectories(
            figure_path("cross_stage_trajectory.png"), common,
            "GT / PnP / Bundle / Pose Graph / Loop Closure",
        )
        _plot_trajectories(
            figure_path("pnp_trajectory.png"),
            [(label, poses) for label, poses in common if label in ("Ground Truth", "pnp")],
            "PnP trajectory",
        )
        _plot_trajectories(
            figure_path("pose_graph_before_after_loop.png"),
            [(label, poses) for label, poses in common if label in (
                "Ground Truth", "pose graph without loop", "pose graph with loop"
            )],
            "Pose Graph before and after Loop Closure",
        )

        bundle_rows = _csv_rows(Path(analysis_root) / "bundle" / "window_metrics.csv")
        bundle_x = _read_float(bundle_rows, "window_ordinal")
        _plot_lines(
            figure_path("bundle_optimization_error.png"), bundle_x,
            [("initial", _read_float(bundle_rows, "initial_graph_error")),
             ("final", _read_float(bundle_rows, "final_graph_error"))],
            "Bundle graph error by window", "Raw graph error",
        )
        _plot_lines(
            figure_path("bundle_reprojection.png"), bundle_x,
            [("mean", _read_float(bundle_rows, "mean_final_projection_residual_px")),
             ("maximum", _read_float(bundle_rows, "maximum_final_projection_residual_px"))],
            "Bundle final reprojection residual", "Pixel residual",
        )
        _plot_lines(
            figure_path("bundle_track_eligibility.png"), bundle_x,
            [("accepted", _read_float(bundle_rows, "accepted_track_count")),
             ("rejected", _read_float(bundle_rows, "rejected_track_count"))],
            "Bundle landmark eligibility", "Tracks",
        )
        rejections = _csv_rows(Path(analysis_root) / "bundle" / "rejections.csv")
        rejection_counts = Counter(row["reason"] for row in rejections)
        if rejection_counts:
            figure, axis = plt.subplots(figsize=(8, 4.8))
            labels = sorted(rejection_counts)
            axis.bar(np.arange(len(labels)), [rejection_counts[label] for label in labels])
            axis.set_xticks(np.arange(len(labels)))
            axis.set_xticklabels(labels, rotation=35, ha="right")
            axis.set_title("Bundle landmark rejection reasons")
            axis.set_ylabel("Rejected candidates")
            _save_figure(figure_path("bundle_rejections.png"), figure)
        else:
            skipped.append({
                "figure": "figures/bundle_rejections.png",
                "reason": "no Bundle landmark candidates were rejected",
            })
        first_windows = sorted((Path(analysis_root).parent / "bundle" / "windows").glob("window_*"))
        if first_windows:
            with np.load(first_windows[0] / "values.npz", allow_pickle=False) as values:
                cameras = values["camera_final"].copy()
                landmarks = values["landmark_final"].copy()
            figure, axis = plt.subplots(figsize=(8, 6))
            if len(landmarks):
                axis.scatter(landmarks[:, 0], landmarks[:, 2], s=2, alpha=0.25, label="landmarks")
            axis.plot(cameras[:, 0, 3], cameras[:, 2, 3], "o-", label="cameras")
            axis.set_title("First Bundle window scene")
            axis.set_xlabel("X [m]")
            axis.set_ylabel("Z [m]")
            axis.axis("equal")
            axis.legend()
            axis.grid(True, alpha=0.25)
            _save_figure(figure_path("first_bundle_window_scene.png"), figure)
            with np.load(first_windows[0] / "camera_marginals.npz", allow_pickle=False) as covariance:
                frame_ids = covariance["frame_ids"].copy()
                marginal_covariances = covariance["marginal_covariances"].copy()
            _plot_lines(
                figure_path("first_bundle_window_covariance.png"), frame_ids,
                [
                    ("location", np.sqrt(np.maximum(0.0, np.trace(marginal_covariances[:, 3:, 3:], axis1=1, axis2=2)))),
                    ("rotation", np.degrees(np.sqrt(np.maximum(0.0, np.trace(marginal_covariances[:, :3, :3], axis1=1, axis2=2))))),
                ],
                "First Bundle window raw marginal uncertainty",
                "Root-sum variance (m / deg)",
            )

        uncertainty = _csv_rows(Path(analysis_root) / "pose_graph" / "uncertainty.csv")
        for metric, filename, label in (
            ("location_uncertainty_m", "pose_graph_location_uncertainty.png", "Location uncertainty [m]"),
            ("rotation_uncertainty_deg", "pose_graph_rotation_uncertainty.png", "Rotation uncertainty [deg]"),
        ):
            series = []
            for stage in ("pose_graph_without_loop", "pose_graph_with_loop"):
                selected = [row for row in uncertainty if row["stage"] == stage]
                series.append((stage.replace("_", " "), _read_float(selected, metric)))
            _plot_lines(figure_path(filename), np.arange(len(series[0][1])), series, label, label)
        with np.load(
            Path(analysis_root).parent / "pose_graph" / "initial_values.npz",
            allow_pickle=False,
        ) as values:
            pose_graph_initial = values["pose_matrices"].copy()
        with np.load(
            Path(analysis_root).parent / "pose_graph" / "optimized_values.npz",
            allow_pickle=False,
        ) as values:
            pose_graph_optimized = values["pose_matrices"].copy()
        _plot_trajectories(
            figure_path("pose_graph_initial_optimized.png"),
            [("initial", pose_graph_initial), ("optimized", pose_graph_optimized)],
            "Consecutive Pose Graph initialization and optimization",
        )
        with np.load(
            Path(analysis_root).parent / "pose_graph" / "marginals.npz",
            allow_pickle=False,
        ) as values:
            pose_graph_covariances = values["marginal_covariances"].copy()
        with np.load(
            Path(analysis_root).parent / "loop_closure" / "final_marginals.npz",
            allow_pickle=False,
        ) as values:
            loop_covariances = values["marginal_covariances"].copy()
        stride = int(config.section("analysis")["covariance_display_stride"])
        _plot_raw_covariance_trajectory(
            figure_path("pose_graph_raw_covariance.png"),
            trajectories["pose_graph_without_loop_pose_matrices"],
            pose_graph_covariances,
            "Pose Graph without Loop Closure: raw 95% X-Z covariance",
            stride,
        )

        accepted = _csv_rows(Path(analysis_root) / "loop_closure" / "accepted_loops.csv")
        candidate_rows = _csv_rows(Path(analysis_root) / "loop_closure" / "candidate_dispositions.csv")
        loop_endpoint_ordinals = []
        for row in accepted:
            loop_endpoint_ordinals.extend((
                int(row["previous_keyframe_ordinal"]),
                int(row["current_keyframe_ordinal"]),
            ))
        _plot_raw_covariance_trajectory(
            figure_path("loop_closure_raw_covariance.png"),
            trajectories["pose_graph_with_loop_pose_matrices"],
            loop_covariances,
            "Pose Graph with Loop Closure: raw 95% X-Z covariance",
            stride,
            loop_endpoint_ordinals,
        )
        figure, axis = plt.subplots(figsize=(8, 4.8))
        disposition = read_json(Path(analysis_root) / "loop_closure" / "summary.json")["disposition_counts"]
        labels = list(disposition)
        axis.bar(np.arange(len(labels)), [disposition[label] for label in labels])
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.set_title("Loop candidate dispositions")
        axis.set_ylabel("Ordered pairs")
        _save_figure(figure_path("loop_candidate_dispositions.png"), figure)
        if accepted:
            accepted_candidates = _accepted_candidate_rows(accepted, candidate_rows)
            figure, axis = plt.subplots(figsize=(8, 4.8))
            x = np.arange(len(accepted_candidates))
            axis.bar(x, _read_float(accepted_candidates, "supporter_count"), label="supporters")
            axis.plot(x, _read_float(accepted_candidates, "consensus_count"), "o-", label="consensus")
            axis.set_xticks(x)
            axis.set_xticklabels([
                "{}→{}".format(row["previous_frame_id"], row["current_frame_id"])
                for row in accepted_candidates
            ], rotation=35, ha="right")
            axis.set_title("Accepted Loop Closure match evidence")
            axis.legend()
            _save_figure(figure_path("accepted_loop_statistics.png"), figure)
            first = accepted_candidates[0]
            current_directory = (
                Path(analysis_root).parent / "loop_closure" / "currents"
                / "current_{:04d}_{:06d}".format(
                    int(first["current_keyframe_ordinal"]), int(first["current_frame_id"])
                )
            )
            with np.load(
                current_directory / "match_{:02d}.npz".format(int(first["matcher_invocation"])),
                allow_pickle=False,
            ) as evidence:
                pixels = evidence["observed_pixels"].copy()
                supporter_indices = evidence["supporter_indices"].copy()
            figure, axes = plt.subplots(1, 2, figsize=(11, 5))
            titles = ("Previous left", "Current left")
            for index, axis in enumerate(axes):
                view = pixels[index * 2]
                axis.scatter(view[0], view[1], s=5, alpha=0.25, label="consensus")
                if len(supporter_indices):
                    axis.scatter(
                        view[0, supporter_indices], view[1, supporter_indices],
                        s=9, alpha=0.75, label="supporters",
                    )
                axis.invert_yaxis()
                axis.set_title(titles[index])
                axis.set_xlabel("u [px]")
                axis.set_ylabel("v [px]")
                axis.legend()
            figure.suptitle("First accepted match {}→{}".format(
                first["previous_frame_id"], first["current_frame_id"]
            ))
            _save_figure(figure_path("first_accepted_match.png"), figure)

            snapshot_ids = sorted(set((1, (len(accepted) + 1) // 2, len(accepted))))
            baseline = trajectories["pose_graph_without_loop_pose_matrices"]
            snapshot_series = [("baseline", baseline)]
            for acceptance_id in snapshot_ids:
                accepted_row = next(
                    row for row in accepted if int(row["acceptance_id"]) == acceptance_id
                )
                directory = (
                    Path(analysis_root).parent / "loop_closure" / "currents"
                    / "current_{:04d}_{:06d}".format(
                        int(accepted_row["current_keyframe_ordinal"]),
                        int(accepted_row["current_frame_id"]),
                    )
                )
                with np.load(directory / "state_checkpoint.npz", allow_pickle=False) as checkpoint:
                    snapshot = checkpoint["pose_matrices"].copy()
                    snapshot_covariances = checkpoint["marginal_covariances"].copy()
                snapshot_series.append(("acceptance {}".format(acceptance_id), snapshot))
                _plot_raw_covariance_trajectory(
                    figure_path(
                        "loop_state_acceptance_{:04d}_raw_covariance.png".format(
                            acceptance_id
                        )
                    ),
                    snapshot,
                    snapshot_covariances,
                    "Loop Closure state after acceptance {}: raw 95% X-Z covariance".format(
                        acceptance_id
                    ),
                    stride,
                    loop_endpoint_ordinals,
                )
            if "ground_truth_pose_matrices" in trajectories:
                ground_truth_keyframes = trajectories["ground_truth_pose_matrices"][
                    trajectories["keyframe_ids"]
                ]
                snapshot_series.insert(0, ("Ground Truth", ground_truth_keyframes))
            _plot_trajectories(
                figure_path("loop_closure_state_snapshots.png"), snapshot_series,
                "Loop Closure baseline / first / middle / final states",
            )
        else:
            skipped.append({
                "figure": "figures/accepted_loop_statistics.png",
                "reason": "the current run accepted zero Loop Closures",
            })
            skipped.append({
                "figure": "figures/first_accepted_match.png",
                "reason": "the current run accepted zero Loop Closures",
            })
            skipped.append({
                "figure": "figures/loop_closure_state_snapshots.png",
                "reason": "the current run accepted zero Loop Closures",
            })

        absolute = _csv_rows(Path(analysis_root) / "cross_stage" / "absolute_errors.csv")
        relative = _csv_rows(
            Path(analysis_root) / "cross_stage" / "consecutive_relative_errors.csv"
        )
        if absolute:
            for metric, filename, ylabel in (
                ("translation_error_m", "absolute_translation_error.png", "Translation error [m]"),
                ("rotation_error_deg", "absolute_rotation_error.png", "Rotation error [deg]"),
            ):
                figure, axis = plt.subplots(figsize=(9, 4.8))
                for stage in stages:
                    selected = [row for row in absolute if row["stage"] == stage]
                    axis.plot(
                        _read_float(selected, "frame_id"), _read_float(selected, metric),
                        label=stage.replace("_", " "), linewidth=1.1,
                    )
                axis.set_title(ylabel)
                axis.set_xlabel("FrameId")
                axis.set_ylabel(ylabel)
                axis.grid(True, alpha=0.25)
                axis.legend()
                _save_figure(figure_path(filename), figure)
            figure, axis = plt.subplots(figsize=(9, 4.8))
            for component, label in (
                ("x_error_m", "|X|"), ("y_error_m", "|Y|"),
                ("z_error_m", "|Z|"), ("translation_error_m", "norm"),
            ):
                selected = [
                    row for row in absolute if row["stage"] == "pose_graph_with_loop"
                ]
                axis.plot(
                    _read_float(selected, "frame_id"), _read_float(selected, component),
                    label=label,
                )
            axis.set_title("Final trajectory absolute X/Y/Z/norm errors")
            axis.set_xlabel("FrameId")
            axis.set_ylabel("Absolute error [m]")
            axis.grid(True, alpha=0.25)
            axis.legend()
            _save_figure(figure_path("absolute_axis_errors.png"), figure)
            pnp_absolute = [row for row in absolute if row["stage"] == "pnp"]
            figure, axis = plt.subplots(figsize=(9, 4.8))
            for component, label in (
                ("x_signed_error_m", "X"), ("y_signed_error_m", "Y"), ("z_signed_error_m", "Z"),
            ):
                axis.plot(
                    _read_float(pnp_absolute, "frame_id"),
                    _read_float(pnp_absolute, component), label=label, linewidth=1.1,
                )
            axis.plot(
                _read_float(pnp_absolute, "frame_id"),
                _read_float(pnp_absolute, "translation_error_m"),
                color="black", label="norm", linewidth=1.4,
            )
            axis.axhline(0.0, color="grey", linewidth=0.6)
            axis.set_title("Absolute PnP position error relative to Ground Truth")
            axis.set_xlabel("FrameId")
            axis.set_ylabel("Position error [m] (estimate - truth)")
            axis.grid(True, alpha=0.25)
            axis.legend()
            _save_figure(figure_path("pnp_absolute_position_errors.png"), figure)
            ground_truth_positions = trajectories["ground_truth_pose_matrices"][:, :3, 3]
            path_distance = np.concatenate((
                [0.0], np.cumsum(np.linalg.norm(np.diff(ground_truth_positions, axis=0), axis=1))
            ))
            figure, axis = plt.subplots(figsize=(9, 4.8))
            for stage in stages:
                selected = [row for row in absolute if row["stage"] == stage]
                frame_ids = np.asarray(
                    [int(row["frame_id"]) for row in selected], dtype=np.int64
                )
                axis.plot(
                    path_distance[frame_ids],
                    _read_float(selected, "translation_error_m"),
                    label=stage.replace("_", " "),
                )
            axis.set_title("Absolute localization error over Ground Truth path distance")
            axis.set_xlabel("Ground Truth path distance [m]")
            axis.set_ylabel("Translation error [m]")
            axis.grid(True, alpha=0.25)
            axis.legend()
            _save_figure(figure_path("ground_truth_path_distance_errors.png"), figure)
            for metric, filename, ylabel in (
                ("translation_error_m", "relative_translation_error.png", "Relative translation error [m]"),
                ("rotation_error_deg", "relative_rotation_error.png", "Relative rotation error [deg]"),
            ):
                # PnP transitions span one frame and keyframe stages span a
                # whole window, so plot by end FrameId rather than ordinal.
                figure, axis = plt.subplots(figsize=(9, 4.8))
                for stage in stages:
                    selected = [row for row in relative if row["stage"] == stage]
                    axis.plot(
                        _read_float(selected, "end_frame_id"), _read_float(selected, metric),
                        label=stage.replace("_", " "),
                    )
                axis.set_title(ylabel + " per stage transition (PnP: frame, others: keyframe)")
                axis.set_xlabel("End FrameId")
                axis.set_ylabel(ylabel)
                axis.grid(True, alpha=0.25)
                axis.legend()
                _save_figure(figure_path(filename), figure)
            keyframe_relative = _csv_rows(
                Path(analysis_root) / "cross_stage" / "keyframe_relative_errors.csv"
            )
            figure, axis = plt.subplots(figsize=(9, 4.8))
            for stage, label in (("pnp_accumulated", "accumulated PnP"), ("bundle", "Bundle Adjustment")):
                selected = [row for row in keyframe_relative if row["stage"] == stage]
                axis.plot(
                    _read_float(selected, "end_frame_id"),
                    _read_float(selected, "translation_error_m"), label=label, linewidth=1.1,
                )
            axis.set_title("Relative translation error between consecutive keyframes")
            axis.set_xlabel("End keyframe FrameId")
            axis.set_ylabel("Relative translation error [m]")
            axis.grid(True, alpha=0.25)
            axis.legend()
            _save_figure(figure_path("keyframe_relative_translation_error.png"), figure)
            for length in config.section("analysis")["relative_subsection_lengths_frames"]:
                rows = _csv_rows(
                    Path(analysis_root) / "cross_stage" / "relative_subsections_{}.csv".format(length)
                )
                figure, axis = plt.subplots(figsize=(9, 4.8))
                for stage in stages:
                    selected = [row for row in rows if row["stage"] == stage]
                    axis.plot(
                        np.arange(len(selected)), _read_float(selected, "translation_error_m"),
                        label=stage.replace("_", " "),
                    )
                axis.set_title("Relative translation error: {}-frame subsections".format(length))
                axis.set_xlabel("Subsection ordinal")
                axis.set_ylabel("Translation error [m]")
                axis.grid(True, alpha=0.25)
                axis.legend()
                _save_figure(
                    figure_path("relative_subsections_{}.png".format(length)), figure
                )
        else:
            skipped.append({
                "figure_group": "Ground Truth error and relative-subsection figures",
                "reason": "Ground Truth is absent for this sequence",
            })

        (attempt / "summary.md").write_text(
            _summary_markdown(analysis_root, root.parent, config, generated),
            encoding="utf-8",
        )
        write_json_atomic(
            attempt / "summary.json",
            {
                "schema": "generic-slam-report-summary-v1",
                "ground_truth_available": bool(absolute),
                "generated_figures": generated,
                "skipped_figures": skipped,
                "figure_count": len(generated),
            },
        )
        _validate_report_artifacts(attempt)
        manifest = StageManifest.complete(
            stage="report", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("report"),
            upstream_stage_ids=upstream_stage_ids,
            counts={
                "figure_count": len(generated),
                "skipped_figure_count": len(skipped),
            },
            primary_artifacts=("summary.json", "summary.md") + tuple(generated),
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
        )
        write_stage_manifest(attempt, manifest)
        os.replace(str(attempt), str(root))
        return manifest
    except Exception:
        if attempt.exists():
            os.replace(str(attempt), str(attempt) + ".failed")
        raise


__all__ = ["run_analysis_stage", "run_report_stage"]
