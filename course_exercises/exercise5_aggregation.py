"""Build deterministic Exercise 5 tables from already sealed windows.

Aggregation is a post-solve stage: it never reconstructs a Bundle graph or
changes saved poses/covariances.  The output tables are the machine-readable
boundary consumed by reporting and curated submission materialization.
"""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from exercise5_full_run import (
    ArtifactValidationError,
    DEFAULT_CONFIG_PATH,
    load_run_configuration,
    validate_run_root,
)


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _window_directories(source_root: Path) -> Tuple[Path, ...]:
    return tuple(sorted((Path(source_root) / "windows").glob("window_[0-9]*")))


def _homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    return np.vstack((np.asarray(extrinsic, dtype=float), [0.0, 0.0, 0.0, 1.0]))


def _read_pose_rows(path: Path) -> Tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(line.split(), dtype=float).reshape(3, 4)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    )


def _rotation_error_deg(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _relative_error(
    estimate: np.ndarray, truth: np.ndarray
) -> Tuple[float, float]:
    error = np.linalg.inv(truth) @ estimate
    return float(np.linalg.norm(error[:3, 3])), _rotation_error_deg(error)


def aggregate_window_metrics(
    source_root: Path, output_path: Optional[Path] = None
) -> Path:
    """Write one row of graph, factor, runtime, and memory metrics per window."""

    source_root = Path(source_root)
    rows = []
    for directory in _window_directories(source_root):
        summary = _read_json(directory / "summary.json")
        residuals = _read_csv(directory / "projection_residuals.csv")
        initial_distances = np.asarray(
            [float(row["initial_stereo_pixel_distance_px"]) for row in residuals]
        )
        final_distances = np.asarray(
            [float(row["final_stereo_pixel_distance_px"]) for row in residuals]
        )
        bundle = summary["bundle_window"]
        rows.append(
            {
                "window_index": summary["window_index"],
                "start_frame_id": summary["start_frame_id"],
                "end_frame_id": summary["end_frame_id"],
                "camera_count": bundle["counts"]["cameras"],
                "candidate_track_count": bundle["candidate_track_count"],
                "accepted_track_count": bundle["accepted_track_count"],
                "rejected_track_count": bundle["rejected_track_count"],
                "projection_factor_count": bundle["counts"]["projection_factors"],
                "initial_total_graph_error": bundle["initial_errors"][
                    "total_graph_error"
                ],
                "final_total_graph_error": bundle["final_errors"][
                    "total_graph_error"
                ],
                "initial_mean_projection_factor_error": bundle["initial_errors"][
                    "mean_projection_factor_error"
                ],
                "final_mean_projection_factor_error": bundle["final_errors"][
                    "mean_projection_factor_error"
                ],
                "initial_median_stereo_pixel_distance_px": float(
                    np.median(initial_distances)
                ),
                "final_median_stereo_pixel_distance_px": float(
                    np.median(final_distances)
                ),
                "optimizer_iterations": summary["runtime"]["optimizer_iterations"],
                "window_total_seconds": summary["runtime"]["window_total_seconds"],
                "peak_rss_bytes": summary["runtime"]["peak_rss_bytes"],
            }
        )
    if not rows:
        raise ArtifactValidationError("no sealed windows to aggregate")
    output_path = (
        Path(output_path)
        if output_path is not None
        else source_root / "aggregate" / "data" / "window_metrics.csv"
    )
    return _write_csv(output_path, tuple(rows[0]), rows)


def aggregate_window_rejections(
    source_root: Path, output_path: Optional[Path] = None
) -> Path:
    """Flatten each window's landmark-rejection counts into a CSV table."""

    rows = []
    for directory in _window_directories(source_root):
        summary = _read_json(directory / "summary.json")
        for reason, count in sorted(
            summary["bundle_window"]["rejection_reason_counts"].items()
        ):
            rows.append(
                {
                    "window_index": summary["window_index"],
                    "start_frame_id": summary["start_frame_id"],
                    "end_frame_id": summary["end_frame_id"],
                    "reason": reason,
                    "count": count,
                }
            )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root) / "aggregate" / "data" / "window_rejections.csv"
    )
    fields = (
        "window_index",
        "start_frame_id",
        "end_frame_id",
        "reason",
        "count",
    )
    return _write_csv(output_path, fields, rows)


def aggregate_projection_residuals(
    source_root: Path, output_path: Optional[Path] = None
) -> Path:
    """Stream every per-window projection row into one atomic CSV output."""

    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root)
        / "aggregate"
        / "data"
        / "all_projection_residuals.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = output_path.parent / (
        ".{}.attempt.{}.tmp".format(output_path.name, uuid.uuid4().hex)
    )
    row_count = 0
    try:
        with attempt.open("x", encoding="utf-8", newline="") as output_handle:
            writer = None
            for directory in _window_directories(source_root):
                summary = _read_json(directory / "summary.json")
                with (directory / "projection_residuals.csv").open(
                    "r", encoding="utf-8", newline=""
                ) as input_handle:
                    for row in csv.DictReader(input_handle):
                        combined = {
                            "window_index": summary["window_index"],
                            **row,
                        }
                        if writer is None:
                            writer = csv.DictWriter(
                                output_handle,
                                fieldnames=tuple(combined),
                            )
                            writer.writeheader()
                        writer.writerow(combined)
                        row_count += 1
        if row_count == 0:
            raise ArtifactValidationError("no projection residual rows")
        attempt.replace(output_path)
    finally:
        if attempt.exists():
            attempt.unlink()
    return output_path


def _trajectory_evidence(
    source_root: Path,
    configuration: Mapping[str, object],
    repository_root: Path,
) -> Tuple[List[Dict[str, object]], Tuple[np.ndarray, ...]]:
    """Compose Bundle edges and align Bundle, PnP, and ground-truth poses."""

    directories = _window_directories(source_root)
    edges = [
        np.asarray(_read_json(directory / "endpoint_edge.json")["matrix"], dtype=float)
        for directory in directories
    ]
    # endpoint_edge.json stores T_Cstart_from_Cend, so forward composition
    # reconstructs the keyframe trajectory in the first camera's frame.
    bundle_poses = [np.eye(4)]
    for edge in edges:
        bundle_poses.append(bundle_poses[-1] @ edge)
    keyframes = configuration["keyframe_selection"]["ordered_keyframe_ids"][
        : len(bundle_poses)
    ]
    companion = _read_json(
        repository_root
        / configuration["canonical_inputs"]["companion"]["path"]
    )
    raw_pnp = companion["tracking_state"]["world_to_camera_poses"]
    pnp_extrinsics = tuple(_homogeneous(np.asarray(pose)) for pose in raw_pnp)
    truth_extrinsics = tuple(
        _homogeneous(pose)
        for pose in _read_pose_rows(
            repository_root
            / configuration["canonical_inputs"]["ground_truth"]["path"]
        )
    )
    pnp_poses = [
        pnp_extrinsics[0] @ np.linalg.inv(pnp_extrinsics[frame_id])
        for frame_id in keyframes
    ]
    truth_poses = [
        truth_extrinsics[0] @ np.linalg.inv(truth_extrinsics[frame_id])
        for frame_id in keyframes
    ]
    rows = []
    for index, frame_id in enumerate(keyframes):
        bundle_center = bundle_poses[index][:3, 3]
        pnp_center = pnp_poses[index][:3, 3]
        truth_center = truth_poses[index][:3, 3]
        rows.append(
            {
                "keyframe_index": index,
                "frame_id": frame_id,
                "bundle_x_m": bundle_center[0],
                "bundle_y_m": bundle_center[1],
                "bundle_z_m": bundle_center[2],
                "pnp_x_m": pnp_center[0],
                "pnp_y_m": pnp_center[1],
                "pnp_z_m": pnp_center[2],
                "ground_truth_x_m": truth_center[0],
                "ground_truth_y_m": truth_center[1],
                "ground_truth_z_m": truth_center[2],
            }
        )
    return rows, tuple(bundle_poses)


def compose_keyframe_trajectories(
    source_root: Path,
    configuration: Mapping[str, object],
    repository_root: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Write aligned Bundle, PnP, and ground-truth keyframe coordinates."""

    rows, _ = _trajectory_evidence(source_root, configuration, repository_root)
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root)
        / "aggregate"
        / "data"
        / "keyframe_trajectories.csv"
    )
    return _write_csv(output_path, tuple(rows[0]), rows)


def compute_keyframe_localization_errors(
    source_root: Path, output_path: Optional[Path] = None
) -> Path:
    """Compare absolute Bundle and PnP keyframe positions with ground truth."""

    trajectory = _read_csv(
        Path(source_root)
        / "aggregate"
        / "data"
        / "keyframe_trajectories.csv"
    )
    rows = []
    for row in trajectory:
        bundle = np.asarray(
            [row["bundle_x_m"], row["bundle_y_m"], row["bundle_z_m"]], dtype=float
        )
        pnp = np.asarray([row["pnp_x_m"], row["pnp_y_m"], row["pnp_z_m"]], dtype=float)
        truth = np.asarray(
            [
                row["ground_truth_x_m"],
                row["ground_truth_y_m"],
                row["ground_truth_z_m"],
            ],
            dtype=float,
        )
        rows.append(
            {
                "keyframe_index": row["keyframe_index"],
                "frame_id": row["frame_id"],
                "bundle_error_m": np.linalg.norm(bundle - truth),
                "pnp_error_m": np.linalg.norm(pnp - truth),
            }
        )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root)
        / "aggregate"
        / "data"
        / "keyframe_localization_errors.csv"
    )
    return _write_csv(output_path, tuple(rows[0]), rows)


def _canonical_relative(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    return first @ np.linalg.inv(second)


def compute_consecutive_relative_errors(
    source_root: Path,
    configuration: Mapping[str, object],
    repository_root: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Compare every saved consecutive Bundle/PnP transform with ground truth."""

    companion = _read_json(
        repository_root
        / configuration["canonical_inputs"]["companion"]["path"]
    )
    pnp = tuple(
        _homogeneous(np.asarray(pose))
        for pose in companion["tracking_state"]["world_to_camera_poses"]
    )
    truth = tuple(
        _homogeneous(pose)
        for pose in _read_pose_rows(
            repository_root
            / configuration["canonical_inputs"]["ground_truth"]["path"]
        )
    )
    rows = []
    for directory in _window_directories(source_root):
        edge = _read_json(directory / "endpoint_edge.json")
        index = int(_read_json(directory / "summary.json")["window_index"])
        start, end = edge["start_frame_id"], edge["end_frame_id"]
        estimate = np.asarray(edge["matrix"], dtype=float)
        truth_relative = _canonical_relative(truth[start], truth[end])
        pnp_relative = _canonical_relative(pnp[start], pnp[end])
        bundle_translation, bundle_rotation = _relative_error(
            estimate, truth_relative
        )
        pnp_translation, pnp_rotation = _relative_error(
            pnp_relative, truth_relative
        )
        rows.append(
            {
                "window_index": index,
                "start_frame_id": start,
                "end_frame_id": end,
                "bundle_translation_error_m": bundle_translation,
                "bundle_rotation_error_deg": bundle_rotation,
                "pnp_translation_error_m": pnp_translation,
                "pnp_rotation_error_deg": pnp_rotation,
            }
        )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root)
        / "aggregate"
        / "data"
        / "consecutive_relative_errors.csv"
    )
    return _write_csv(output_path, tuple(rows[0]), rows)


def compute_relative_subsections(
    source_root: Path,
    configuration: Mapping[str, object],
    repository_root: Path,
    length: int,
    output_path: Optional[Path] = None,
) -> Path:
    """Evaluate Bundle drift over keyframe subsections near one frame length."""

    trajectory_rows, bundle_poses = _trajectory_evidence(
        source_root, configuration, repository_root
    )
    keyframes = [int(row["frame_id"]) for row in trajectory_rows]
    truth = tuple(
        _homogeneous(pose)
        for pose in _read_pose_rows(
            repository_root
            / configuration["canonical_inputs"]["ground_truth"]["path"]
        )
    )
    truth_centers = np.asarray(
        [-pose[:3, :3].T @ pose[:3, 3] for pose in truth]
    )
    rows = []
    for start_index, start_frame in enumerate(keyframes):
        target = start_frame + length - 1
        candidates = [
            (abs(frame - target), frame, index)
            for index, frame in enumerate(keyframes)
            if index > start_index
        ]
        if not candidates or target > 3359:
            continue
        _, end_frame, end_index = min(candidates)
        estimate = np.linalg.inv(bundle_poses[start_index]) @ bundle_poses[end_index]
        truth_relative = _canonical_relative(truth[start_frame], truth[end_frame])
        translation, rotation = _relative_error(estimate, truth_relative)
        distance = float(
            np.sum(
                np.linalg.norm(
                    np.diff(truth_centers[start_frame : end_frame + 1], axis=0),
                    axis=1,
                )
            )
        )
        if distance <= 0.0:
            raise ArtifactValidationError("relative subsection has zero GT distance")
        rows.append(
            {
                "requested_length_frames": length,
                "start_frame_id": start_frame,
                "target_end_frame_id": target,
                "selected_end_frame_id": end_frame,
                "ground_truth_distance_m": distance,
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "translation_error_m_per_m": translation / distance,
                "translation_error_percent": 100.0 * translation / distance,
                "rotation_error_deg_per_m": rotation / distance,
            }
        )
    fields = (
        "requested_length_frames",
        "start_frame_id",
        "target_end_frame_id",
        "selected_end_frame_id",
        "ground_truth_distance_m",
        "translation_error_m",
        "rotation_error_deg",
        "translation_error_m_per_m",
        "translation_error_percent",
        "rotation_error_deg_per_m",
    )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root)
        / "aggregate"
        / "data"
        / "relative_subsections_{}.csv".format(length)
    )
    return _write_csv(output_path, fields, rows)


def write_low_connectivity_table(
    source_root: Path,
    configuration: Mapping[str, object],
    repository_root: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Retain transition-connectivity context for the frozen flagged windows."""

    transitions = _read_csv(
        repository_root
        / configuration["canonical_inputs"]["transition_table"]["path"]
    )
    by_front = {
        int(row.get("front_frame_id", row.get("frame_id", -1))): row
        for row in transitions
    }
    low_indices = set(
        configuration["keyframe_selection"]["statistics"][
            "low_connectivity_window_indices"
        ]
    )
    rows = []
    for directory in _window_directories(source_root):
        summary = _read_json(directory / "summary.json")
        index = int(summary["window_index"])
        if index not in low_indices:
            continue
        start, end = summary["start_frame_id"], summary["end_frame_id"]
        connectivity = []
        for frame_id in range(start + 1, end + 1):
            row = by_front.get(frame_id, {})
            raw = row.get("final_supporter_count", row.get("connectivity", ""))
            if raw != "":
                connectivity.append(int(float(raw)))
        rows.append(
            {
                "window_index": index,
                "start_frame_id": start,
                "end_frame_id": end,
                "minimum_transition_connectivity": (
                    min(connectivity) if connectivity else ""
                ),
            }
        )
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path(source_root)
        / "aggregate"
        / "data"
        / "low_connectivity_windows.csv"
    )
    return _write_csv(
        output_path,
        (
            "window_index",
            "start_frame_id",
            "end_frame_id",
            "minimum_transition_connectivity",
        ),
        rows,
    )


def aggregate_all(
    source_root: Path,
    *,
    repository_root: Path,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
    allow_partial: bool = False,
) -> Tuple[Path, ...]:
    """Validate a run root and generate the complete aggregate table set."""

    source_root = Path(source_root)
    repository_root = Path(repository_root)
    # Never aggregate an unverified window population.
    validate_run_root(
        source_root,
        configuration_path=configuration_path,
        allow_partial=allow_partial,
    )
    configuration, _ = load_run_configuration(configuration_path)
    outputs = [
        aggregate_window_metrics(source_root),
        aggregate_window_rejections(source_root),
        aggregate_projection_residuals(source_root),
        compose_keyframe_trajectories(
            source_root, configuration, repository_root
        ),
        compute_keyframe_localization_errors(source_root),
        compute_consecutive_relative_errors(
            source_root, configuration, repository_root
        ),
    ]
    for length in configuration["artifact_contract"][
        "relative_subsection_policy"
    ]["requested_lengths_frames"]:
        outputs.append(
            compute_relative_subsections(
                source_root,
                configuration,
                repository_root,
                length,
            )
        )
    outputs.append(
        write_low_connectivity_table(
            source_root, configuration, repository_root
        )
    )
    return tuple(outputs)


__all__ = [
    "aggregate_all",
    "aggregate_projection_residuals",
    "aggregate_window_metrics",
    "aggregate_window_rejections",
    "compose_keyframe_trajectories",
    "compute_consecutive_relative_errors",
    "compute_keyframe_localization_errors",
    "compute_relative_subsections",
    "write_low_connectivity_table",
]
