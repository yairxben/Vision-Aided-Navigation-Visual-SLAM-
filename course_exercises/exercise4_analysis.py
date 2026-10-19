"""Deterministic Exercise 4 TrackingDB analysis and artifact helpers.

This module consumes an already-built lecturer TrackingDB.  It deliberately
contains no feature extraction, matching, motion-estimation PnP, or RANSAC.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from exercise4_database_builder import (
    validate_tracking_database_consistency_linear,
)
from tracking_database import NO_ID, Link, TrackingDB
from tracking_database_adapter import TrackingDBCompanion


ROOT = Path(__file__).resolve().parent
FINAL_COMPANION_SCHEMA = "exercise4-final-companion-v1"
PRODUCTION_COMPANION_SCHEMA = "exercise4-production-companion-v1"
PRODUCTION_COMPANION_VERSION = 1
STREAMING_DATABASE_DIGEST_SCHEMA = "exercise4-trackingdb-stream-v1"
ANALYSIS_SCHEMA = "exercise4-production-analysis-v1"
POSE_CONVENTION = "world-to-camera: x_i = R_i X + t_i"
SELECTION_POLICY_6 = (
    "lower-median eligible length among tracks length>=6; "
    "smallest TrackId at that length"
)
SELECTION_POLICY_10 = (
    "lower-median eligible length among tracks length>=10; "
    "smallest TrackId at that length"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Dict[str, object]:
    with path.open(encoding="utf-8") as file_handle:
        value = json.load(file_handle)
    if not isinstance(value, dict):
        raise ValueError("JSON artifact must contain an object")
    return value


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=list(fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _update_length_prefixed(digest, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _update_links(digest, links: Iterable[Link]) -> None:
    buffer = bytearray()
    count = 0
    for link in links:
        buffer.extend(
            struct.pack(
                "<ddd",
                float(link.x_left),
                float(link.x_right),
                float(link.y),
            )
        )
        count += 1
        if len(buffer) >= 1024 * 1024:
            digest.update(buffer)
            buffer.clear()
    if buffer:
        digest.update(buffer)
    digest.update(struct.pack("<Q", count))


def streaming_database_digest(database: TrackingDB) -> str:
    """Hash complete TrackingDB state without constructing a huge payload."""

    digest = hashlib.sha256()
    _update_length_prefixed(
        digest, STREAMING_DATABASE_DIGEST_SCHEMA.encode("ascii")
    )
    digest.update(
        struct.pack(
            "<qq",
            int(database.last_frameId),
            int(database.last_trackId),
        )
    )
    digest.update(struct.pack("<Q", len(database.trackId_to_frames)))
    for track_id, frames in database.trackId_to_frames.items():
        digest.update(struct.pack("<qQ", int(track_id), len(frames)))
        if frames:
            digest.update(np.asarray(frames, dtype="<i8").tobytes())

    digest.update(struct.pack("<Q", len(database.frameId_to_lfeature)))
    for frame_id, features in database.frameId_to_lfeature.items():
        normalized = np.ascontiguousarray(features)
        digest.update(struct.pack("<q", int(frame_id)))
        _update_length_prefixed(
            digest, normalized.dtype.str.encode("ascii")
        )
        digest.update(struct.pack("<Q", normalized.ndim))
        digest.update(
            np.asarray(normalized.shape, dtype="<i8").tobytes()
        )
        digest.update(memoryview(normalized).cast("B"))

    digest.update(
        struct.pack("<Q", len(database.frameId_to_trackIds_list))
    )
    for frame_id, rows in database.frameId_to_trackIds_list.items():
        digest.update(struct.pack("<qQ", int(frame_id), len(rows)))
        if rows:
            digest.update(np.asarray(rows, dtype="<i8").tobytes())

    digest.update(struct.pack("<Q", len(database.linkId_to_link)))
    link_buffer = bytearray()
    for (frame_id, track_id), link in database.linkId_to_link.items():
        link_buffer.extend(
            struct.pack(
                "<qqddd",
                int(frame_id),
                int(track_id),
                float(link.x_left),
                float(link.x_right),
                float(link.y),
            )
        )
        if len(link_buffer) >= 1024 * 1024:
            digest.update(link_buffer)
            link_buffer.clear()
    if link_buffer:
        digest.update(link_buffer)

    _update_links(digest, database.prev_frame_links)
    digest.update(struct.pack("<Q", len(database.leftover_links)))
    for frame_id, links in database.leftover_links.items():
        digest.update(struct.pack("<q", int(frame_id)))
        _update_links(digest, links)
    return digest.hexdigest()


def runtime_free_transition(transition: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in transition.items()
        if key != "runtime_seconds"
    }


TRANSITION_FIELDS = (
    "back_frame",
    "front_frame",
    "success",
    "failure",
    "back_left_feature_count",
    "front_left_feature_count",
    "back_right_feature_count",
    "front_right_feature_count",
    "back_stereo_raw_match_count",
    "front_stereo_raw_match_count",
    "back_stereo_inlier_count",
    "front_stereo_inlier_count",
    "temporal_forward_match_count",
    "temporal_reverse_match_count",
    "reciprocal_temporal_match_count",
    "consensus_count",
    "supporter_count",
    "supporter_percentage",
    "ransac_random_hypothesis_count",
    "ransac_refinement_model_count",
    "pnp_attempt_count",
    "pnp_failure_count",
    "tracks_created",
    "tracks_extended",
    "tracks_terminated",
    "connectivity",
    "complete_temporal_row_count",
    "true_validity_mask_count",
    "final_supporter_validity_exact",
    "selected_consensus_non_supporter_count",
    "non_supporters_entered",
    "preflight_rejection_count",
    "not_stereo_valid_rejection_count",
    "duplicate_previous_row_rejection_count",
    "duplicate_destination_rejection_count",
    "duplicate_conflict_count",
)


def flatten_transition(transition: Dict[str, object]) -> Dict[str, object]:
    rejections = transition["preflight_rejections"]
    return {
        "back_frame": transition["back_frame"],
        "front_frame": transition["front_frame"],
        "success": transition["success"],
        "failure": transition["failure"],
        "back_left_feature_count": transition[
            "back_left_feature_count"
        ],
        "front_left_feature_count": transition[
            "front_left_feature_count"
        ],
        "back_right_feature_count": transition[
            "back_right_feature_count"
        ],
        "front_right_feature_count": transition[
            "front_right_feature_count"
        ],
        "back_stereo_raw_match_count": transition[
            "stereo_raw_match_count_back"
        ],
        "front_stereo_raw_match_count": transition[
            "stereo_raw_match_count_front"
        ],
        "back_stereo_inlier_count": transition["back_stereo_row_count"],
        "front_stereo_inlier_count": transition[
            "front_stereo_row_count"
        ],
        "temporal_forward_match_count": transition[
            "temporal_forward_match_count"
        ],
        "temporal_reverse_match_count": transition[
            "temporal_reverse_match_count"
        ],
        "reciprocal_temporal_match_count": transition[
            "reciprocal_match_count"
        ],
        "consensus_count": transition["consensus_count"],
        "supporter_count": transition["supporter_count"],
        "supporter_percentage": transition["supporter_percentage"],
        "ransac_random_hypothesis_count": transition[
            "ransac_random_hypothesis_count"
        ],
        "ransac_refinement_model_count": transition[
            "ransac_refinement_model_count"
        ],
        "pnp_attempt_count": transition["pnp_attempt_count"],
        "pnp_failure_count": transition["pnp_failure_count"],
        "tracks_created": transition["tracks_created"],
        "tracks_extended": transition["tracks_extended"],
        "tracks_terminated": transition["tracks_terminated"],
        "connectivity": transition["connectivity"],
        "complete_temporal_row_count": transition[
            "complete_temporal_row_count"
        ],
        "true_validity_mask_count": transition[
            "true_validity_mask_count"
        ],
        "final_supporter_validity_exact": transition[
            "final_supporter_validity_exact"
        ],
        "selected_consensus_non_supporter_count": transition[
            "selected_consensus_non_supporter_count"
        ],
        "non_supporters_entered": transition["non_supporters_entered"],
        "preflight_rejection_count": rejections[
            "total_preflight_candidate_rejections"
        ],
        "not_stereo_valid_rejection_count": rejections[
            "not_stereo_valid_in_both_frames"
        ],
        "duplicate_previous_row_rejection_count": rejections[
            "duplicate_previous_row"
        ],
        "duplicate_destination_rejection_count": rejections[
            "duplicate_current_destination"
        ],
        "duplicate_conflict_count": rejections[
            "duplicate_or_destination_conflict_records"
        ],
    }


def _load_database(database_path: Path) -> TrackingDB:
    if database_path.suffix != ".pkl":
        raise ValueError("database path must end in .pkl")
    database = TrackingDB()
    database.load(str(database_path.with_suffix("")))
    return database


def load_tracking_artifacts(
    database_path: Path, companion_path: Path
) -> Tuple[TrackingDB, TrackingDBCompanion, Dict[str, object]]:
    """Load and verify one supported serialized database/companion pair."""

    database_path = database_path.resolve()
    companion_path = companion_path.resolve()
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    if not companion_path.is_file():
        raise FileNotFoundError(companion_path)

    raw_companion = read_json(companion_path)
    schema = raw_companion.get("schema")
    if schema not in (
        FINAL_COMPANION_SCHEMA,
        PRODUCTION_COMPANION_SCHEMA,
    ):
        raise ValueError("unsupported companion schema: {}".format(schema))
    if schema == PRODUCTION_COMPANION_SCHEMA and raw_companion.get(
        "schema_version"
    ) != PRODUCTION_COMPANION_VERSION:
        raise ValueError("unsupported production companion version")

    tracking_state = raw_companion.get("tracking_state")
    transitions = raw_companion.get("transition_metadata")
    if not isinstance(tracking_state, dict):
        raise ValueError("companion tracking state is missing")
    if not isinstance(transitions, list):
        raise ValueError("companion transition metadata is missing")
    typed_companion = TrackingDBCompanion.from_dict(tracking_state)

    database_metadata = raw_companion.get("database")
    if not isinstance(database_metadata, dict):
        raise ValueError("companion database identity is missing")
    observed_file_size = database_path.stat().st_size
    observed_file_sha256 = file_sha256(database_path)
    if int(database_metadata.get("size_bytes", -1)) != observed_file_size:
        raise AssertionError("database byte size differs from companion")
    if database_metadata.get("file_sha256") != observed_file_sha256:
        raise AssertionError("database file SHA-256 differs from companion")

    database = _load_database(database_path)
    validate_tracking_database_consistency_linear(database)
    observed_streaming_digest = streaming_database_digest(database)
    if database_metadata.get("streaming_digest_schema") != (
        STREAMING_DATABASE_DIGEST_SCHEMA
    ):
        raise AssertionError("database streaming-digest schema differs")
    if database_metadata.get("streaming_digest") != (
        observed_streaming_digest
    ):
        raise AssertionError("database streaming digest differs")

    frame_count = database.frame_num()
    if len(typed_companion.world_to_camera_poses) != frame_count:
        raise AssertionError("database and companion frame counts differ")
    if len(transitions) != max(0, frame_count - 1):
        raise AssertionError("database and transition counts differ")
    if raw_companion.get("frame_count") != frame_count:
        raise AssertionError("companion declared frame count differs")
    if raw_companion.get("transition_count") != len(transitions):
        raise AssertionError("companion declared transition count differs")

    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            raise ValueError("transition metadata row is not an object")
        if (
            transition.get("back_frame") != index
            or transition.get("front_frame") != index + 1
        ):
            raise AssertionError("transition FrameIds are not consecutive")
        if int(transition["supporter_count"]) != (
            typed_companion.supporter_counts[index]
        ):
            raise AssertionError("transition supporter count differs")
        if float(transition["supporter_percentage"]) != (
            typed_companion.supporter_percentages[index]
        ):
            raise AssertionError("transition supporter percentage differs")

    normalized = {
        "schema": schema,
        "raw_companion": raw_companion,
        "tracking_state": tracking_state,
        "transitions": transitions,
        "database": {
            "path": str(database_path),
            "size_bytes": observed_file_size,
            "file_sha256": observed_file_sha256,
            "streaming_digest_schema": STREAMING_DATABASE_DIGEST_SCHEMA,
            "streaming_digest": observed_streaming_digest,
        },
        "companion": {
            "path": str(companion_path),
            "file_sha256": file_sha256(companion_path),
        },
    }
    return database, typed_companion, normalized


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise AssertionError("numeric summary requires finite values")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _histogram_percentile(histogram: Counter, percentile: float) -> float:
    total = sum(histogram.values())
    if total <= 0:
        raise AssertionError("track-length histogram is empty")
    rank = (percentile / 100.0) * (total - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))

    def value_at(index: int) -> int:
        cumulative = 0
        for value in sorted(histogram):
            cumulative += histogram[value]
            if index < cumulative:
                return int(value)
        raise AssertionError("histogram percentile rank is invalid")

    lower = value_at(lower_index)
    upper = value_at(upper_index)
    return float(lower + (rank - lower_index) * (upper - lower))


def _histogram_summary(histogram: Counter) -> Dict[str, object]:
    total = int(sum(histogram.values()))
    return {
        "count": total,
        "minimum": int(min(histogram)),
        "mean": float(
            sum(length * count for length, count in histogram.items())
            / total
        ),
        "median": _histogram_percentile(histogram, 50),
        "maximum": int(max(histogram)),
        "percentiles": {
            "p{}".format(percentile): _histogram_percentile(
                histogram, percentile
            )
            for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "histogram": {
            str(length): int(count)
            for length, count in sorted(histogram.items())
        },
    }


def _lower_median_eligible_length(
    histogram: Counter, minimum_length: int
) -> Tuple[int, int]:
    eligible_count = sum(
        count
        for length, count in histogram.items()
        if length >= minimum_length
    )
    if eligible_count == 0:
        raise AssertionError(
            "no track has length at least {}".format(minimum_length)
        )
    target = (eligible_count - 1) // 2
    cumulative = 0
    for length in sorted(histogram):
        if length < minimum_length:
            continue
        cumulative += histogram[length]
        if target < cumulative:
            return int(length), int(eligible_count)
    raise AssertionError("eligible track selection failed")


def _smallest_track_at_length(
    database: TrackingDB, target_length: int
) -> int:
    for track_id, frames in database.trackId_to_frames.items():
        if len(frames) == target_length:
            return int(track_id)
    raise AssertionError("no track has selected length")


def _quarter_summaries(values: Sequence[float]) -> List[Dict[str, object]]:
    array = np.asarray(values, dtype=float)
    quarter_count = min(4, len(array))
    boundaries = np.linspace(
        0, len(array), quarter_count + 1, dtype=int
    )
    summaries = []
    for quarter in range(quarter_count):
        start = int(boundaries[quarter])
        stop = int(boundaries[quarter + 1])
        summaries.append(
            {
                "quarter": quarter + 1,
                "index_range_inclusive": [start, stop - 1],
                **numeric_summary(array[start:stop]),
            }
        )
    return summaries


def load_ground_truth_poses(count: int) -> List[np.ndarray]:
    path = ROOT / "dataset" / "poses" / "00.txt"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < count:
        raise AssertionError("ground-truth pose file is too short")
    return [
        np.asarray(line.split(), dtype=float).reshape(3, 4)
        for line in lines[:count]
    ]


def load_calibration() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / "dataset" / "sequences" / "00" / "calib.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise AssertionError("calibration file is incomplete")
    projection_left = np.asarray(
        lines[0].split()[1:], dtype=float
    ).reshape(3, 4)
    projection_right = np.asarray(
        lines[1].split()[1:], dtype=float
    ).reshape(3, 4)
    intrinsic = projection_left[:, :3]
    left_pose = np.linalg.inv(intrinsic) @ projection_left
    right_pose = np.linalg.inv(intrinsic) @ projection_right
    return intrinsic, left_pose, right_pose


def _relative_pose(
    back_pose: np.ndarray, front_pose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    back_rotation = back_pose[:, :3]
    back_translation = back_pose[:, 3:]
    front_rotation = front_pose[:, :3]
    front_translation = front_pose[:, 3:]
    rotation = front_rotation @ back_rotation.T
    translation = front_translation - rotation @ back_translation
    return rotation, translation


def _anchored_centers(poses: Sequence[np.ndarray]) -> np.ndarray:
    initial = poses[0]
    centers = []
    for pose in poses:
        rotation, translation = _relative_pose(initial, pose)
        centers.append((-rotation.T @ translation).reshape(3))
    return np.asarray(centers)


def _trajectory_metrics(
    poses: Sequence[np.ndarray], transitions: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    truth = load_ground_truth_poses(len(poses))
    estimated_centers = _anchored_centers(poses)
    truth_centers = _anchored_centers(truth)
    center_errors = np.linalg.norm(estimated_centers - truth_centers, axis=1)
    truth_steps = np.linalg.norm(np.diff(truth_centers, axis=0), axis=1)
    total_truth_distance = float(np.sum(truth_steps))
    determinants = np.asarray(
        [np.linalg.det(pose[:, :3]) for pose in poses]
    )
    orthogonality = np.asarray(
        [
            np.linalg.norm(
                pose[:, :3].T @ pose[:, :3] - np.eye(3), ord="fro"
            )
            for pose in poses
        ]
    )
    return {
        "successful_transitions": int(
            sum(bool(row["success"]) for row in transitions)
        ),
        "failed_transitions": int(
            sum(not bool(row["success"]) for row in transitions)
        ),
        "invalid_pose_count": int(
            sum(not np.all(np.isfinite(pose)) for pose in poses)
        ),
        "malformed_rotation_count": int(
            np.sum(
                (np.abs(determinants - 1.0) > 1e-6)
                | (orthogonality > 1e-6)
            )
        ),
        "maximum_determinant_error": float(
            np.max(np.abs(determinants - 1.0))
        ),
        "maximum_orthogonality_error": float(np.max(orthogonality)),
        "camera_center_error_m": numeric_summary(center_errors),
        "final_frame_position_error_m": float(center_errors[-1]),
        "total_ground_truth_distance_m": total_truth_distance,
        "final_drift_percent": float(
            100.0 * center_errors[-1] / total_truth_distance
        ),
        "pnp_attempt_count": int(
            sum(int(row["pnp_attempt_count"]) for row in transitions)
        ),
        "pnp_failure_count": int(
            sum(int(row["pnp_failure_count"]) for row in transitions)
        ),
    }


TRACK_LENGTH_FIELDS = (
    "track_id",
    "length",
    "first_frame",
    "last_frame",
)
HISTOGRAM_FIELDS = ("track_length", "track_count")
FRAME_LINK_FIELDS = (
    "frame_id",
    "stereo_row_count",
    "tracked_link_count",
    "untracked_link_count",
)
BELOW_THRESHOLD_FIELDS = (
    "back_frame",
    "front_frame",
    "supporter_percentage",
)


def _compute_track_statistics(
    database: TrackingDB,
    write_track_row: Callable[[Dict[str, object]], None],
) -> Dict[str, object]:
    """Compute ordered track statistics and deterministic selections."""

    track_histogram = Counter()
    track_count = 0
    link_count_from_tracks = 0
    for track_id, frames in database.trackId_to_frames.items():
        length = len(frames)
        if length <= 1:
            continue
        track_histogram[length] += 1
        track_count += 1
        link_count_from_tracks += length
        write_track_row(
            {
                "track_id": int(track_id),
                "length": int(length),
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
            }
        )
    if link_count_from_tracks != database.link_num():
        raise AssertionError("track lengths do not sum to Link count")
    histogram_rows = [
        {"track_length": length, "track_count": count}
        for length, count in sorted(track_histogram.items())
    ]

    representative_length, representative_eligible = (
        _lower_median_eligible_length(track_histogram, 6)
    )
    reprojection_length, reprojection_eligible = (
        _lower_median_eligible_length(track_histogram, 10)
    )
    representative_track = _smallest_track_at_length(
        database, representative_length
    )
    reprojection_track = _smallest_track_at_length(
        database, reprojection_length
    )
    return {
        "histogram": track_histogram,
        "track_count": track_count,
        "histogram_rows": histogram_rows,
        "representative_length": representative_length,
        "representative_eligible": representative_eligible,
        "representative_track": representative_track,
        "reprojection_length": reprojection_length,
        "reprojection_eligible": reprojection_eligible,
        "reprojection_track": reprojection_track,
    }


def _compute_frame_and_transition_statistics(
    database: TrackingDB,
    transitions: Sequence[Dict[str, object]],
    supporter_threshold_percent: float,
) -> Dict[str, object]:
    """Compute ordered frame, connectivity, and transition statistics."""

    frame_links = []
    frame_rows = []
    for frame_id in database.all_frames():
        rows = database.frameId_to_trackIds_list[frame_id]
        tracked = sum(track_id != NO_ID for track_id in rows)
        frame_links.append(tracked)
        frame_rows.append(
            {
                "frame_id": int(frame_id),
                "stereo_row_count": len(rows),
                "tracked_link_count": tracked,
                "untracked_link_count": len(rows) - tracked,
            }
        )
    if sum(frame_links) != database.link_num():
        raise AssertionError("frame Links do not sum to Link count")

    connectivity = []
    previous_tracks = {
        track_id
        for track_id in database.frameId_to_trackIds_list[0]
        if track_id != NO_ID
    }
    for index, transition in enumerate(transitions):
        current_tracks = {
            track_id
            for track_id in database.frameId_to_trackIds_list[index + 1]
            if track_id != NO_ID
        }
        observed = len(previous_tracks.intersection(current_tracks))
        expected_values = (
            int(transition["connectivity"]),
            int(transition["supporter_count"]),
            int(transition["true_validity_mask_count"]),
        )
        if any(value != observed for value in expected_values):
            raise AssertionError(
                "connectivity/supporter equality failed at transition {}"
                .format(index)
            )
        connectivity.append(observed)
        previous_tracks = current_tracks

    transition_rows = [flatten_transition(row) for row in transitions]
    below_threshold = [
        {
            "back_frame": int(row["back_frame"]),
            "front_frame": int(row["front_frame"]),
            "supporter_percentage": float(row["supporter_percentage"]),
        }
        for row in transitions
        if float(row["supporter_percentage"])
        < supporter_threshold_percent
    ]
    return {
        "frame_links": frame_links,
        "frame_rows": frame_rows,
        "connectivity": connectivity,
        "transition_rows": transition_rows,
        "below_threshold": below_threshold,
    }


def _write_database_metric_tables(
    database: TrackingDB,
    transitions: Sequence[Dict[str, object]],
    analysis_dir: Path,
    supporter_threshold_percent: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Write the protected Exercise 4 metric tables in their fixed order."""

    track_lengths_path = analysis_dir / "track_lengths.csv"
    with track_lengths_path.open(
        "w", newline="", encoding="utf-8"
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=list(TRACK_LENGTH_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        track_statistics = _compute_track_statistics(
            database,
            writer.writerow,
        )
    write_csv(
        analysis_dir / "track_length_histogram.csv",
        HISTOGRAM_FIELDS,
        track_statistics["histogram_rows"],
    )
    frame_and_transition_statistics = (
        _compute_frame_and_transition_statistics(
            database,
            transitions,
            supporter_threshold_percent,
        )
    )
    write_csv(
        analysis_dir / "frame_links.csv",
        FRAME_LINK_FIELDS,
        frame_and_transition_statistics["frame_rows"],
    )
    write_csv(
        analysis_dir / "transition_metrics.csv",
        TRANSITION_FIELDS,
        frame_and_transition_statistics["transition_rows"],
    )
    write_csv(
        analysis_dir / "below_threshold_transitions.csv",
        BELOW_THRESHOLD_FIELDS,
        frame_and_transition_statistics["below_threshold"],
    )
    return track_statistics, frame_and_transition_statistics


def compute_database_metrics(
    database: TrackingDB,
    typed_companion: TrackingDBCompanion,
    transitions: Sequence[Dict[str, object]],
    analysis_dir: Path,
    supporter_threshold_percent: float,
) -> Tuple[Dict[str, object], int, int, Dict[str, List[object]]]:
    """Compute tracking/trajectory metrics and deterministic track choices.

    Returns the metrics, the representative and reprojection TrackIds, and the
    ordered series consumed by figure generation.
    """

    track_statistics, frame_and_transition_statistics = (
        _write_database_metric_tables(
            database,
            transitions,
            analysis_dir,
            supporter_threshold_percent,
        )
    )

    track_histogram = track_statistics["histogram"]
    track_count = track_statistics["track_count"]
    representative_length = track_statistics["representative_length"]
    representative_eligible = track_statistics["representative_eligible"]
    representative_track = track_statistics["representative_track"]
    reprojection_length = track_statistics["reprojection_length"]
    reprojection_eligible = track_statistics["reprojection_eligible"]
    reprojection_track = track_statistics["reprojection_track"]
    frame_links = frame_and_transition_statistics["frame_links"]
    connectivity = frame_and_transition_statistics["connectivity"]
    below_threshold = frame_and_transition_statistics["below_threshold"]

    supporter_counts = [int(row["supporter_count"]) for row in transitions]
    supporter_percentages = [
        float(row["supporter_percentage"]) for row in transitions
    ]
    final_ten_percent_count = max(1, len(connectivity) // 10)
    track_summary = _histogram_summary(track_histogram)
    track_summary.update(
        {
            "count_length_2": int(track_histogram.get(2, 0)),
            "count_length_3": int(track_histogram.get(3, 0)),
            "count_length_at_least_6": int(
                sum(
                    count
                    for length, count in track_histogram.items()
                    if length >= 6
                )
            ),
            "count_length_at_least_10": int(
                sum(
                    count
                    for length, count in track_histogram.items()
                    if length >= 10
                )
            ),
            "count_length_at_least_20": int(
                sum(
                    count
                    for length, count in track_histogram.items()
                    if length >= 20
                )
            ),
            "count_length_at_least_50": int(
                sum(
                    count
                    for length, count in track_histogram.items()
                    if length >= 50
                )
            ),
            "right_censored_at_frame": database.last_frameId,
        }
    )
    min_frame_links = min(frame_links)
    min_connectivity = min(connectivity)
    min_percentage = min(supporter_percentages)
    metrics = {
        "schema": ANALYSIS_SCHEMA,
        "scope": {
            "frame_range_inclusive": [0, database.last_frameId],
            "frame_count": database.frame_num(),
            "transition_count": len(transitions),
        },
        "database": {
            "nontrivial_track_count": track_count,
            "tracked_link_count": database.link_num(),
            "last_track_id": database.last_trackId,
        },
        "track_lengths": track_summary,
        "tracked_links_per_frame": {
            **numeric_summary(frame_links),
            "percentiles": {
                "p{}".format(percentile): float(
                    np.percentile(frame_links, percentile)
                )
                for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
            "worst_frames": [
                frame_id
                for frame_id, value in enumerate(frame_links)
                if value == min_frame_links
            ],
            "quarters": _quarter_summaries(frame_links),
            "final_100_frames": numeric_summary(frame_links[-100:]),
        },
        "connectivity": {
            **numeric_summary(connectivity),
            "worst_transitions": [
                [index, index + 1]
                for index, value in enumerate(connectivity)
                if value == min_connectivity
            ],
            "quarters": _quarter_summaries(connectivity),
            "final_10_percent_transition_count": final_ten_percent_count,
            "final_10_percent": numeric_summary(
                connectivity[-final_ten_percent_count:]
            ),
            "final_100_transitions": numeric_summary(connectivity[-100:]),
        },
        "supporter_count": numeric_summary(supporter_counts),
        "supporter_percentage": {
            **numeric_summary(supporter_percentages),
            "threshold_percent": float(supporter_threshold_percent),
            "below_threshold_count": len(below_threshold),
            "below_30_percent_count": int(
                sum(value < 30.0 for value in supporter_percentages)
            ),
            "worst_transitions": [
                [index, index + 1]
                for index, value in enumerate(supporter_percentages)
                if value == min_percentage
            ],
            "quarters": _quarter_summaries(supporter_percentages),
            "final_100_transitions": numeric_summary(
                supporter_percentages[-100:]
            ),
        },
        "transition_totals": {
            "reciprocal_match_count": int(
                sum(int(row["reciprocal_match_count"]) for row in transitions)
            ),
            "consensus_count": int(
                sum(int(row["consensus_count"]) for row in transitions)
            ),
            "supporter_count": int(sum(supporter_counts)),
            "pnp_attempt_count": int(
                sum(int(row["pnp_attempt_count"]) for row in transitions)
            ),
            "pnp_failure_count": int(
                sum(int(row["pnp_failure_count"]) for row in transitions)
            ),
        },
        "lifecycle": {
            "tracks_created_total": int(
                sum(int(row["tracks_created"]) for row in transitions)
            ),
            "tracks_extended_total": int(
                sum(int(row["tracks_extended"]) for row in transitions)
            ),
            "tracks_terminated_total": int(
                sum(int(row["tracks_terminated"]) for row in transitions)
            ),
            "final_active_track_count": frame_links[-1],
        },
        "supporter_only_insertion": {
            "all_transition_connectivity_equals_supporters": True,
            "final_supporter_validity_exact": all(
                bool(row["final_supporter_validity_exact"])
                for row in transitions
            ),
            "reciprocal_or_consensus_non_supporters_entered": int(
                sum(int(row["non_supporters_entered"]) for row in transitions)
            ),
            "selected_consensus_non_supporter_total": int(
                sum(
                    int(row["selected_consensus_non_supporter_count"])
                    for row in transitions
                )
            ),
            "preflight_rejection_total": int(
                sum(
                    int(
                        row["preflight_rejections"][
                            "total_preflight_candidate_rejections"
                        ]
                    )
                    for row in transitions
                )
            ),
            "duplicate_conflict_total": int(
                sum(
                    int(
                        row["preflight_rejections"][
                            "duplicate_or_destination_conflict_records"
                        ]
                    )
                    for row in transitions
                )
            ),
        },
        "deterministic_selections": {
            "representative_track": {
                "selection_policy": SELECTION_POLICY_6,
                "eligible_track_count": representative_eligible,
                "selected_length": representative_length,
                "track_id": representative_track,
            },
            "reprojection_track": {
                "selection_policy": SELECTION_POLICY_10,
                "eligible_track_count": reprojection_eligible,
                "selected_length": reprojection_length,
                "track_id": reprojection_track,
            },
        },
        "trajectory": _trajectory_metrics(
            typed_companion.world_to_camera_poses, transitions
        ),
    }

    lifecycle = metrics["lifecycle"]
    if lifecycle["tracks_created_total"] != track_count:
        raise AssertionError("created track total differs")
    if (
        2 * track_count + lifecycle["tracks_extended_total"]
        != database.link_num()
    ):
        raise AssertionError("track lifecycle Link identity differs")
    if (
        lifecycle["tracks_terminated_total"]
        + lifecycle["final_active_track_count"]
        != track_count
    ):
        raise AssertionError("terminated/active track identity differs")
    if metrics["supporter_only_insertion"][
        "reciprocal_or_consensus_non_supporters_entered"
    ]:
        raise AssertionError("a non-supporter entered TrackingDB")

    series = {
        "frame_links": frame_links,
        "connectivity": connectivity,
        "supporter_percentages": supporter_percentages,
        "histogram_lengths": sorted(track_histogram),
        "histogram_counts": [
            track_histogram[length] for length in sorted(track_histogram)
        ],
    }
    return metrics, representative_track, reprojection_track, series


def write_representative_track(
    database: TrackingDB, track_id: int, analysis_dir: Path
) -> Dict[str, object]:
    """Write full-frame and cropped observations for one selected track."""

    frames = database.frames(track_id)
    if len(frames) < 6:
        raise AssertionError("representative track is shorter than six")
    figure, axes = plt.subplots(
        len(frames),
        2,
        figsize=(12, max(4, 3.0 * len(frames))),
        squeeze=False,
    )
    observations = []
    for row_index, frame_id in enumerate(frames):
        image_path = (
            ROOT
            / "dataset"
            / "sequences"
            / "00"
            / "image_0"
            / "{:06d}.png".format(frame_id)
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        link = database.link(frame_id, track_id)
        x = int(round(link.x_left))
        y = int(round(link.y))
        x0 = max(x - 10, 0)
        y0 = max(y - 10, 0)
        x1 = min(x + 10, image.shape[1])
        y1 = min(y + 10, image.shape[0])
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(rgb, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        cv2.circle(rgb, (x, y), 2, (255, 0, 0), -1)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            raise AssertionError("representative crop is empty")
        enlarged = cv2.resize(
            crop, (160, 160), interpolation=cv2.INTER_NEAREST
        )
        axes[row_index, 0].imshow(rgb)
        axes[row_index, 0].set_title("Frame {}".format(frame_id))
        axes[row_index, 1].imshow(enlarged)
        axes[row_index, 1].set_title(
            "20x20 crop ({:.2f}, {:.2f})".format(link.x_left, link.y)
        )
        for axis in axes[row_index]:
            axis.axis("off")
        observations.append(
            {
                "frame_id": int(frame_id),
                "x_left": float(link.x_left),
                "x_right": float(link.x_right),
                "y": float(link.y),
                "crop_bounds_half_open": [x0, y0, x1, y1],
                "crop_width": x1 - x0,
                "crop_height": y1 - y0,
            }
        )
    figure.suptitle(
        "Deterministic representative TrackId {}, length {}".format(
            track_id, len(frames)
        )
    )
    figure.tight_layout()
    figure.savefig(
        analysis_dir / "figures" / "representative_track.png", dpi=180
    )
    plt.close(figure)
    evidence = {
        "selection_policy": SELECTION_POLICY_6,
        "track_id": int(track_id),
        "track_length": len(frames),
        "frames": list(frames),
        "crop_policy": (
            "literal half-open [round(x)-10:round(x)+10) and y; "
            "clipped to image bounds"
        ),
        "observations": observations,
    }
    write_json(analysis_dir / "representative_track.json", evidence)
    return evidence


def _right_camera_pose(
    left_rotation: np.ndarray,
    left_translation: np.ndarray,
    stereo_rotation: np.ndarray,
    stereo_translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        stereo_rotation @ left_rotation,
        stereo_rotation @ left_translation + stereo_translation,
    )


def _project_point(
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    point_world: np.ndarray,
) -> Tuple[np.ndarray, float]:
    point_camera = rotation @ point_world.reshape(3, 1) + translation
    depth = float(point_camera[2, 0])
    if not math.isfinite(depth) or depth <= 0.0:
        raise AssertionError("reprojection violates cheirality")
    homogeneous = intrinsic @ point_camera
    pixel = homogeneous[:2, 0] / homogeneous[2, 0]
    if not np.all(np.isfinite(pixel)):
        raise AssertionError("reprojected pixel is nonfinite")
    return pixel, depth


def _error_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "maximum": float(np.max(array)),
    }


REPROJECTION_FIELDS = (
    "anchor",
    "anchor_frame",
    "frame_id",
    "observed_left_x",
    "observed_left_y",
    "projected_left_x",
    "projected_left_y",
    "left_residual_x",
    "left_residual_y",
    "left_error_px",
    "left_depth",
    "observed_right_x",
    "observed_right_y",
    "projected_right_x",
    "projected_right_y",
    "right_residual_x",
    "right_residual_y",
    "right_error_px",
    "right_depth",
)


def compute_reprojection(
    database: TrackingDB, track_id: int, analysis_dir: Path
) -> Dict[str, object]:
    """Measure stereo reprojection error for one consecutive long track.

    The 3D point is triangulated independently from the first and last stereo
    observations using KITTI ground-truth poses, then projected through every
    frame in the track.
    """

    frames = database.frames(track_id)
    if len(frames) < 10:
        raise AssertionError("reprojection track is shorter than ten")
    if frames != list(range(frames[0], frames[-1] + 1)):
        raise AssertionError("reprojection track is not consecutive")
    poses = load_ground_truth_poses(database.frame_num())
    intrinsic, initial_left, initial_right = load_calibration()
    if not np.allclose(
        initial_left,
        np.hstack((np.eye(3), np.zeros((3, 1)))),
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError("left calibration pose is not identity")
    stereo_rotation = initial_right[:, :3]
    stereo_translation = initial_right[:, 3:]

    def camera_pair(frame_id: int):
        left_pose = poses[frame_id]
        left_rotation = left_pose[:, :3]
        left_translation = left_pose[:, 3:]
        right_rotation, right_translation = _right_camera_pose(
            left_rotation,
            left_translation,
            stereo_rotation,
            stereo_translation,
        )
        return (
            left_rotation,
            left_translation,
            right_rotation,
            right_translation,
        )

    anchors = {}
    csv_rows = []
    for anchor_name, anchor_frame in (
        ("first", frames[0]),
        ("last", frames[-1]),
    ):
        link = database.link(anchor_frame, track_id)
        left_rotation, left_translation, right_rotation, right_translation = (
            camera_pair(anchor_frame)
        )
        homogeneous_world = cv2.triangulatePoints(
            intrinsic @ np.hstack((left_rotation, left_translation)),
            intrinsic @ np.hstack((right_rotation, right_translation)),
            np.asarray([[link.x_left], [link.y]], dtype=float),
            np.asarray([[link.x_right], [link.y]], dtype=float),
        )
        scale = float(homogeneous_world[3, 0])
        if not math.isfinite(scale) or abs(scale) < 1e-15:
            raise AssertionError("triangulated homogeneous scale is invalid")
        point_world = homogeneous_world[:3, 0] / scale
        if not np.all(np.isfinite(point_world)):
            raise AssertionError("triangulated point is nonfinite")

        rows = []
        left_errors = []
        right_errors = []
        for frame_id in frames:
            observed = database.link(frame_id, track_id)
            frame_left_r, frame_left_t, frame_right_r, frame_right_t = (
                camera_pair(frame_id)
            )
            projected_left, left_depth = _project_point(
                intrinsic, frame_left_r, frame_left_t, point_world
            )
            projected_right, right_depth = _project_point(
                intrinsic, frame_right_r, frame_right_t, point_world
            )
            observed_left = np.asarray([observed.x_left, observed.y])
            observed_right = np.asarray([observed.x_right, observed.y])
            residual_left = projected_left - observed_left
            residual_right = projected_right - observed_right
            left_error = float(np.linalg.norm(residual_left))
            right_error = float(np.linalg.norm(residual_right))
            left_errors.append(left_error)
            right_errors.append(right_error)
            row = {
                "anchor": anchor_name,
                "anchor_frame": int(anchor_frame),
                "frame_id": int(frame_id),
                "observed_left_x": float(observed_left[0]),
                "observed_left_y": float(observed_left[1]),
                "projected_left_x": float(projected_left[0]),
                "projected_left_y": float(projected_left[1]),
                "left_residual_x": float(residual_left[0]),
                "left_residual_y": float(residual_left[1]),
                "left_error_px": left_error,
                "left_depth": left_depth,
                "observed_right_x": float(observed_right[0]),
                "observed_right_y": float(observed_right[1]),
                "projected_right_x": float(projected_right[0]),
                "projected_right_y": float(projected_right[1]),
                "right_residual_x": float(residual_right[0]),
                "right_residual_y": float(residual_right[1]),
                "right_error_px": right_error,
                "right_depth": right_depth,
            }
            rows.append(row)
            csv_rows.append(row)
        anchors[anchor_name] = {
            "anchor_frame": int(anchor_frame),
            "world_point": point_world.tolist(),
            "left_error_summary_px": _error_summary(left_errors),
            "right_error_summary_px": _error_summary(right_errors),
            "rows": rows,
        }

    first_link = database.link(frames[0], track_id)
    last_link = database.link(frames[-1], track_id)
    first_point = np.asarray(anchors["first"]["world_point"])
    last_point = np.asarray(anchors["last"]["world_point"])
    evidence = {
        "pose_source": "dataset/poses/00.txt",
        "pose_convention": POSE_CONVENTION,
        "right_pose_formula": (
            "R_right = R_stereo @ R_left; "
            "t_right = R_stereo @ t_left + t_stereo"
        ),
        "selection_policy": SELECTION_POLICY_10,
        "track_id": int(track_id),
        "track_length": len(frames),
        "track_frames": list(frames),
        "anchors": anchors,
        "comparison": {
            "world_point_difference_norm": float(
                np.linalg.norm(first_point - last_point)
            ),
            "first_disparity_pixels": float(
                first_link.x_left - first_link.x_right
            ),
            "last_disparity_pixels": float(
                last_link.x_left - last_link.x_right
            ),
            "left_mean_first_minus_last_px": float(
                anchors["first"]["left_error_summary_px"]["mean"]
                - anchors["last"]["left_error_summary_px"]["mean"]
            ),
            "right_mean_first_minus_last_px": float(
                anchors["first"]["right_error_summary_px"]["mean"]
                - anchors["last"]["right_error_summary_px"]["mean"]
            ),
        },
    }
    write_json(analysis_dir / "reprojection.json", evidence)
    write_csv(
        analysis_dir / "reprojection.csv", REPROJECTION_FIELDS, csv_rows
    )
    return evidence


def write_figures(
    metrics: Dict[str, object],
    reprojection: Dict[str, object],
    series: Dict[str, List[object]],
    analysis_dir: Path,
) -> None:
    """Write the required tracking, connectivity, supporter, and error plots."""

    figures = analysis_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    lengths = np.asarray(series["histogram_lengths"])
    counts = np.asarray(series["histogram_counts"])
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(lengths, counts, width=0.9)
    axis.set_yscale("log")
    axis.set_xlabel("Track length (frames)")
    axis.set_ylabel("Track count (log scale)")
    axis.set_title("Track-length distribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "track_length_histogram.png", dpi=180)
    plt.close(figure)

    frame_ids = np.arange(len(series["frame_links"]))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(frame_ids, series["frame_links"], linewidth=0.7)
    axis.set_xlabel("FrameId")
    axis.set_ylabel("Tracked Links")
    axis.set_title("Tracked Links per frame")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "frame_links.png", dpi=180)
    plt.close(figure)

    front_frames = np.arange(1, len(series["connectivity"]) + 1)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(front_frames, series["connectivity"], linewidth=0.7)
    axis.set_xlabel("Front FrameId")
    axis.set_ylabel("Connecting tracks")
    axis.set_title("Consecutive-frame connectivity")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figures / "connectivity.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        front_frames, series["supporter_percentages"], linewidth=0.7
    )
    axis.axhline(
        metrics["supporter_percentage"]["threshold_percent"],
        color="red",
        linestyle="--",
        linewidth=0.8,
        label="reporting threshold",
    )
    axis.set_xlabel("Front FrameId")
    axis.set_ylabel("Final supporters / consensus (%)")
    axis.set_title("Final RANSAC-supporter percentage")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "supporter_percentage.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for anchor_name, linestyle in (("first", "-"), ("last", "--")):
        rows = reprojection["anchors"][anchor_name]["rows"]
        frames = [row["frame_id"] for row in rows]
        axes[0].plot(
            frames,
            [row["left_error_px"] for row in rows],
            linestyle=linestyle,
            label=anchor_name,
        )
        axes[1].plot(
            frames,
            [row["right_error_px"] for row in rows],
            linestyle=linestyle,
            label=anchor_name,
        )
    axes[0].set_ylabel("Left error (px)")
    axes[1].set_ylabel("Right error (px)")
    axes[1].set_xlabel("FrameId")
    axes[0].set_title(
        "Stereo reprojection: TrackId {}, length {}".format(
            reprojection["track_id"], reprojection["track_length"]
        )
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "reprojection.png", dpi=180)
    plt.close(figure)


def _finalize_analysis_artifacts(
    analysis_dir: Path,
    normalized: Dict[str, object],
    metrics: Dict[str, object],
    representative: Dict[str, object],
    reprojection: Dict[str, object],
    supporter_threshold_percent: float,
) -> Dict[str, object]:
    """Hash and finalize deterministic analysis and provenance artifacts."""

    deterministic_files = [
        "track_lengths.csv",
        "track_length_histogram.csv",
        "frame_links.csv",
        "transition_metrics.csv",
        "below_threshold_transitions.csv",
        "representative_track.json",
        "reprojection.json",
        "reprojection.csv",
    ]
    deterministic_artifact_sha256 = {
        relative: file_sha256(analysis_dir / relative)
        for relative in deterministic_files
    }
    scientific_state_sha256 = canonical_sha256(
        {
            "database_streaming_digest": normalized["database"][
                "streaming_digest"
            ],
            "metrics": metrics,
            "representative_track": representative,
            "reprojection": reprojection,
            "deterministic_artifact_sha256": (
                deterministic_artifact_sha256
            ),
        }
    )
    metrics["scientific_state_sha256"] = scientific_state_sha256
    write_json(analysis_dir / "metrics.json", metrics)

    source_paths = (
        "ex4.py",
        "exercise4_workflow.py",
        "exercise4_analysis.py",
        "exercise4_database_builder.py",
        "tracking_database_adapter.py",
        "tracking_database.py",
        "tools.py",
    )
    provenance = {
        "schema": "exercise4-production-analysis-provenance-v1",
        "input_companion_schema": normalized["schema"],
        "analysis_contract": (
            "serialized TrackingDB analysis only; no feature extraction, "
            "matching, motion PnP, or RANSAC"
        ),
        "pose_convention": POSE_CONVENTION,
        "database": normalized["database"],
        "companion": normalized["companion"],
        "dataset_inputs": {
            "calibration": {
                "path": "dataset/sequences/00/calib.txt",
                "sha256": file_sha256(
                    ROOT / "dataset" / "sequences" / "00" / "calib.txt"
                ),
            },
            "ground_truth_poses": {
                "path": "dataset/poses/00.txt",
                "sha256": file_sha256(
                    ROOT / "dataset" / "poses" / "00.txt"
                ),
            },
            "representative_images": {
                "dataset/sequences/00/image_0/{:06d}.png".format(
                    frame_id
                ): file_sha256(
                    ROOT
                    / "dataset"
                    / "sequences"
                    / "00"
                    / "image_0"
                    / "{:06d}.png".format(frame_id)
                )
                for frame_id in representative["frames"]
            },
        },
        "source_sha256": {
            relative: file_sha256(ROOT / relative)
            for relative in source_paths
        },
        "command_contract": (
            "python ex4.py analyze --database <PATH.pkl> "
            "--companion <PATH.json> --output-dir <DIR>"
        ),
        "supporter_threshold_percent": float(
            supporter_threshold_percent
        ),
        "deterministic_artifact_sha256": {
            **deterministic_artifact_sha256,
            "metrics.json": file_sha256(analysis_dir / "metrics.json"),
        },
        "scientific_state_sha256": scientific_state_sha256,
    }
    write_json(analysis_dir / "provenance.json", provenance)
    return {
        "metrics": metrics,
        "provenance": provenance,
        "representative_track": representative,
        "reprojection": reprojection,
    }


def analyze_tracking_database(
    database_path: Path,
    companion_path: Path,
    analysis_dir: Path,
    supporter_threshold_percent: float = 30.0,
) -> Dict[str, object]:
    """Load, validate, analyze, and write deterministic Exercise 4 output."""

    if not math.isfinite(supporter_threshold_percent) or not (
        0.0 <= supporter_threshold_percent <= 100.0
    ):
        raise ValueError("supporter threshold must be in [0, 100]")
    analysis_dir.mkdir(parents=True, exist_ok=False)
    (analysis_dir / "figures").mkdir()

    # Reload and verify the serialized TrackingDB/companion contract.
    database, typed_companion, normalized = load_tracking_artifacts(
        database_path, companion_path
    )
    # Compute tracking, connectivity, supporter, and trajectory statistics.
    metrics, representative_id, reprojection_id, series = (
        compute_database_metrics(
            database,
            typed_companion,
            normalized["transitions"],
            analysis_dir,
            supporter_threshold_percent,
        )
    )
    # Produce deterministic representative-track and reprojection evidence.
    representative = write_representative_track(
        database, representative_id, analysis_dir
    )
    reprojection = compute_reprojection(
        database, reprojection_id, analysis_dir
    )
    # Render figures only after all underlying metrics are available.
    write_figures(metrics, reprojection, series, analysis_dir)
    return _finalize_analysis_artifacts(
        analysis_dir=analysis_dir,
        normalized=normalized,
        metrics=metrics,
        representative=representative,
        reprojection=reprojection,
        supporter_threshold_percent=supporter_threshold_percent,
    )
