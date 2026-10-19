"""Deterministic persistence primitives for Exercise 5 sealed evidence.

This module owns artifact schemas, byte-normalized serialization, and
fail-closed validation of prepared and sealed Bundle-window directories.
Scientific Bundle construction, optimization, endpoint covariance extraction,
and streamed orchestration remain in their existing Exercise 5 modules.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from exercise5_bundle_window import BundleWindowResult, ProjectionFactorRecord


# Artifact schemas and deterministic serialization


WINDOW_SCHEMA = "exercise5-window-artifacts-v1"
HASH_SCHEMA = "exercise5-window-artifact-hashes-v1"
RUN_STATE_SCHEMA = "exercise5-run-state-v1"
MANIFEST_SCHEMA = "exercise5-run-manifest-v1"
BASE_WINDOW_FILES = (
    "summary.json",
    "ids.npz",
    "values.npz",
    "tracks.csv",
    "projection_residuals.csv",
    "endpoint_edge.json",
    "endpoint_covariances.npz",
)
FIRST_WINDOW_FILES = (
    "camera_marginals.npz",
    "maximum_initial_projection_factor.json",
    "overlay_left_initial.png",
    "overlay_right_initial.png",
    "overlay_left_final.png",
    "overlay_right_final.png",
    "scene_3d.png",
    "scene_birds_eye.png",
)


class ArtifactValidationError(RuntimeError):
    """Raised when an unsealed or sealed artifact fails closed validation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    data = _canonical_json_bytes(value)
    with Path(path).open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _little_float64(value: object) -> np.ndarray:
    array = np.asarray(value, dtype="<f8")
    return np.ascontiguousarray(array)


def _little_int64(value: object) -> np.ndarray:
    array = np.asarray(value, dtype="<i8")
    return np.ascontiguousarray(array)


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    normalized = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind in "iu":
            normalized[name] = _little_int64(array)
        elif array.dtype.kind == "f":
            normalized[name] = _little_float64(array)
        else:
            raise TypeError("NPZ array {} has unsupported dtype".format(name))
    np.savez(path, **normalized)
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping]
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


# Prepared and sealed artifact validation


def _window_name(index: int, start: int, end: int) -> str:
    return "window_{:04d}_{}_{}".format(index, start, end)


def _window_required_files(index: int) -> Tuple[str, ...]:
    return BASE_WINDOW_FILES + (FIRST_WINDOW_FILES if index == 0 else ())


def _validate_png(path: Path) -> None:
    signature = Path(path).read_bytes()[:8]
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ArtifactValidationError("{} is not a readable PNG".format(path))


def _validate_npz_shapes(
    directory: Path, summary: Mapping[str, object], first_window: bool
) -> None:
    counts = summary["bundle_window"]["counts"]
    camera_count = counts["cameras"]
    landmark_count = counts["landmarks"]
    with np.load(directory / "ids.npz", allow_pickle=False) as ids:
        expected_ids = {
            "frame_ids": (camera_count,),
            "candidate_track_ids": (
                summary["bundle_window"]["candidate_track_count"],
            ),
            "accepted_track_ids": (landmark_count,),
            "rejected_track_ids": (
                summary["bundle_window"]["rejected_track_count"],
            ),
        }
        if set(ids.files) != set(expected_ids):
            raise ArtifactValidationError("ids.npz members differ")
        for name, shape in expected_ids.items():
            if ids[name].shape != shape or ids[name].dtype != np.dtype("<i8"):
                raise ArtifactValidationError("ids.npz {} differs".format(name))
    with np.load(directory / "values.npz", allow_pickle=False) as values:
        expected_values = {
            "initial_camera_poses": (camera_count, 4, 4),
            "final_camera_poses": (camera_count, 4, 4),
            "initial_landmarks": (landmark_count, 3),
            "final_landmarks": (landmark_count, 3),
        }
        if set(values.files) != set(expected_values):
            raise ArtifactValidationError("values.npz members differ")
        for name, shape in expected_values.items():
            if (
                values[name].shape != shape
                or values[name].dtype != np.dtype("<f8")
                or not np.all(np.isfinite(values[name]))
            ):
                raise ArtifactValidationError("values.npz {} differs".format(name))
    with np.load(directory / "endpoint_covariances.npz", allow_pickle=False) as cov:
        expected_cov = {
            "joint_covariance": (12, 12),
            "joint_information": (12, 12),
            "conditional_end_given_start_covariance": (6, 6),
        }
        if set(cov.files) != set(expected_cov):
            raise ArtifactValidationError("endpoint covariance members differ")
        for name, shape in expected_cov.items():
            if (
                cov[name].shape != shape
                or cov[name].dtype != np.dtype("<f8")
                or not np.all(np.isfinite(cov[name]))
            ):
                raise ArtifactValidationError(
                    "endpoint covariance {} differs".format(name)
                )
    if first_window:
        with np.load(directory / "camera_marginals.npz", allow_pickle=False) as cov:
            expected = {
                "frame_ids": (camera_count,),
                "poses": (camera_count, 4, 4),
                "marginal_covariances": (camera_count, 6, 6),
            }
            if set(cov.files) != set(expected):
                raise ArtifactValidationError("camera marginal members differ")
            for name, shape in expected.items():
                if cov[name].shape != shape or not np.all(np.isfinite(cov[name])):
                    raise ArtifactValidationError(
                        "camera marginal {} differs".format(name)
                    )


def validate_prepared_window(
    directory: Path,
    *,
    expected_index: int,
    expected_start: int,
    expected_end: int,
    config_sha256: str,
    source_identity_sha256: str,
    allow_hash_file: bool = False,
) -> Dict[str, object]:
    """Validate an unsealed window's identity, schemas, shapes, and rows."""

    directory = Path(directory)
    required = set(_window_required_files(expected_index))
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if allow_hash_file:
        observed.discard("artifact_hashes.json")
    if observed != required:
        raise ArtifactValidationError(
            "prepared window files differ: missing {}, extra {}".format(
                sorted(required - observed), sorted(observed - required)
            )
        )
    summary = _read_json(directory / "summary.json")
    expected_equalities = (
        (summary.get("schema"), WINDOW_SCHEMA, "summary schema"),
        (summary.get("schema_version"), 1, "summary schema version"),
        (summary.get("seal_status"), "prepared-for-seal", "seal status"),
        (summary.get("window_index"), expected_index, "window index"),
        (summary.get("start_frame_id"), expected_start, "start FrameId"),
        (summary.get("end_frame_id"), expected_end, "end FrameId"),
        (
            summary.get("configuration_sha256"),
            config_sha256,
            "configuration identity",
        ),
        (
            summary.get("source_input_identity_sha256"),
            source_identity_sha256,
            "source/input identity",
        ),
    )
    for observed_value, expected_value, name in expected_equalities:
        if observed_value != expected_value:
            raise ArtifactValidationError("{} differs".format(name))
    frame_ids = summary["bundle_window"]["ordered_frame_ids"]
    if frame_ids != list(range(expected_start, expected_end + 1)):
        raise ArtifactValidationError("window FrameIds differ")
    _validate_npz_shapes(directory, summary, expected_index == 0)
    edge = _read_json(directory / "endpoint_edge.json")
    if (
        edge.get("schema") != "exercise5-endpoint-edge-v1"
        or edge.get("start_frame_id") != expected_start
        or edge.get("end_frame_id") != expected_end
    ):
        raise ArtifactValidationError("endpoint edge identity differs")
    for csv_name, expected_rows in (
        ("tracks.csv", summary["bundle_window"]["candidate_track_count"]),
        (
            "projection_residuals.csv",
            summary["bundle_window"]["counts"]["projection_factors"],
        ),
    ):
        with (directory / csv_name).open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_rows:
            raise ArtifactValidationError("{} row count differs".format(csv_name))
    if expected_index == 0:
        for filename in FIRST_WINDOW_FILES:
            if filename.endswith(".png"):
                _validate_png(directory / filename)
        maximum = _read_json(
            directory / "maximum_initial_projection_factor.json"
        )
        if maximum.get("schema") != (
            "exercise5-maximum-initial-projection-factor-v1"
        ):
            raise ArtifactValidationError("maximum factor schema differs")
    return summary


def _hash_entries(
    directory: Path, filenames: Sequence[str]
) -> List[Dict[str, object]]:
    return [
        {
            "path": name,
            "size_bytes": int((directory / name).stat().st_size),
            "sha256": sha256_file(directory / name),
        }
        for name in sorted(filenames)
    ]


def validate_sealed_window(
    directory: Path,
    *,
    expected_index: int,
    expected_start: int,
    expected_end: int,
    config_sha256: str,
    source_identity_sha256: str,
) -> Dict[str, object]:
    """Reload a sealed window and verify its identity and every file hash."""

    directory = Path(directory)
    hash_path = directory / "artifact_hashes.json"
    if not hash_path.is_file():
        raise ArtifactValidationError("sealed window has no hash seal")
    seal = _read_json(hash_path)
    if (
        seal.get("schema") != HASH_SCHEMA
        or seal.get("schema_version") != 1
        or seal.get("status") != "sealed"
        or seal.get("window_index") != expected_index
        or seal.get("start_frame_id") != expected_start
        or seal.get("end_frame_id") != expected_end
        or seal.get("configuration_sha256") != config_sha256
        or seal.get("source_input_identity_sha256") != source_identity_sha256
    ):
        raise ArtifactValidationError("window seal identity differs")
    expected_files = set(_window_required_files(expected_index))
    entries = seal.get("artifacts")
    if not isinstance(entries, list):
        raise ArtifactValidationError("window seal artifact index differs")
    indexed = {entry["path"]: entry for entry in entries}
    if set(indexed) != expected_files:
        raise ArtifactValidationError("window seal artifact set differs")
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_files != expected_files | {"artifact_hashes.json"}:
        raise ArtifactValidationError("sealed window file set differs")
    for name in sorted(expected_files):
        path = directory / name
        if (
            int(path.stat().st_size) != indexed[name]["size_bytes"]
            or sha256_file(path) != indexed[name]["sha256"]
        ):
            raise ArtifactValidationError(
                "sealed artifact hash differs for {}".format(name)
            )
    summary = validate_prepared_window(
        directory,
        expected_index=expected_index,
        expected_start=expected_start,
        expected_end=expected_end,
        config_sha256=config_sha256,
        source_identity_sha256=source_identity_sha256,
        allow_hash_file=True,
    )
    return {"summary": summary, "seal": seal, "directory": str(directory)}


# Quarantine primitive used by the sealing lifecycle below


def quarantine_attempt(
    attempt_directory: Path,
    quarantine_root: Path,
    *,
    exception: BaseException,
) -> Path:
    """Move a failed attempt aside and retain the triggering exception."""

    attempt_directory = Path(attempt_directory)
    quarantine_root = Path(quarantine_root)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / (
        "{}.failed.{}".format(attempt_directory.name, uuid.uuid4().hex)
    )
    if attempt_directory.exists():
        os.replace(str(attempt_directory), str(target))
    else:
        target.mkdir()
    _write_json(
        target / "failure.json",
        {
            "schema": "exercise5-window-failure-v1",
            "failed_utc": utc_now(),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
        },
    )
    _fsync_directory(target)
    _fsync_directory(quarantine_root)
    return target


# Window evidence construction


def _project_record(
    result: BundleWindowResult,
    record: ProjectionFactorRecord,
    calibration: gtsam.Cal3_S2Stereo,
    values: gtsam.Values,
) -> Tuple[np.ndarray, np.ndarray, float]:
    point = values.atPoint3(record.landmark_key)
    pose = values.atPose3(record.camera_key)
    depth = float(np.asarray(pose.transformTo(point), dtype=float)[2])
    if not np.isfinite(depth) or depth <= 0.0:
        raise ArtifactValidationError(
            "projection ({}, {}) has invalid depth".format(
                record.frame_id, record.track_id
            )
        )
    projection = np.asarray(
        gtsam.StereoCamera(pose, calibration).project(point).vector(),
        dtype=float,
    )
    residual = projection - np.asarray(record.measurement, dtype=float)
    factor_error = float(result.graph.at(record.factor_index).error(values))
    if not np.all(np.isfinite(projection)) or not np.isfinite(factor_error):
        raise ArtifactValidationError("projection evidence is nonfinite")
    return projection, residual, factor_error


def _camera_centers_from_extrinsics(
    poses: Sequence[np.ndarray],
) -> np.ndarray:
    return np.asarray(
        [
            -np.asarray(pose, dtype=float)[:, :3].T
            @ np.asarray(pose, dtype=float)[:, 3]
            for pose in poses
        ],
        dtype=float,
    )


def _cumulative_path(centers: np.ndarray) -> np.ndarray:
    if len(centers) == 0:
        return np.empty((0,), dtype=float)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(steps)))



def _valid_measurement_frames(
    database: object, track_id: int, start: int, end: int
) -> Tuple[List[int], int]:
    valid = []
    in_window = 0
    for raw_frame in database.frames(track_id):
        frame_id = int(raw_frame)
        if not start <= frame_id <= end:
            continue
        in_window += 1
        link = database.link(frame_id, track_id)
        measurement = np.asarray(
            [link.x_left, link.x_right, link.y], dtype=float
        )
        if np.all(np.isfinite(measurement)) and measurement[0] > measurement[1]:
            valid.append(frame_id)
    return sorted(valid), in_window


def _build_tracks_rows(
    result: BundleWindowResult, database: object
) -> List[Dict[str, object]]:
    start, end = result.frame_ids[0], result.frame_ids[-1]
    accepted = set(result.accepted_track_ids)
    rows = []
    for track_id in result.candidate_track_ids:
        valid_frames, in_window_count = _valid_measurement_frames(
            database, track_id, start, end
        )
        initialization = result.initialization_frame_by_track_id.get(track_id)
        rejection = result.rejected_track_reason_by_id.get(track_id, "")
        if initialization is not None:
            attempts = [
                frame
                for frame in reversed(valid_frames)
                if frame >= initialization
            ]
        elif rejection and rejection != "insufficient_valid_observations":
            attempts = list(reversed(valid_frames))
        else:
            attempts = []
        if track_id in accepted:
            expected_valid = result.accepted_observation_count_by_track_id[
                track_id
            ]
            if expected_valid != len(valid_frames):
                raise ArtifactValidationError(
                    "track diagnostic valid count differs from solver"
                )
        rows.append(
            {
                "track_id": track_id,
                "accepted": int(track_id in accepted),
                "in_window_observation_count": in_window_count,
                "valid_observation_count": len(valid_frames),
                "invalid_observation_count": in_window_count - len(valid_frames),
                "distinct_valid_frame_count": len(set(valid_frames)),
                "triangulation_frame_id": (
                    "" if initialization is None else initialization
                ),
                "triangulation_attempt_frame_ids_json": json.dumps(attempts),
                "rejection_reason": rejection,
            }
        )
    return rows


def _build_projection_rows(
    result: BundleWindowResult,
    calibration: gtsam.Cal3_S2Stereo,
    poses: Sequence[np.ndarray],
    ground_truth: Sequence[np.ndarray],
) -> List[Dict[str, object]]:
    start = result.frame_ids[0]
    end = result.frame_ids[-1]
    pnp_path = _cumulative_path(
        _camera_centers_from_extrinsics(poses[start : end + 1])
    )
    gt_path = _cumulative_path(
        _camera_centers_from_extrinsics(ground_truth[start : end + 1])
    )
    rows = []
    for record in sorted(
        result.projection_factor_records,
        key=lambda item: (item.frame_id, item.track_id),
    ):
        initial_projection, initial_residual, initial_error = _project_record(
            result, record, calibration, result.initial_values
        )
        final_projection, final_residual, final_error = _project_record(
            result, record, calibration, result.final_values
        )
        measurement = np.asarray(record.measurement, dtype=float)
        row = {
            "frame_id": record.frame_id,
            "track_id": record.track_id,
            "factor_index": record.factor_index,
            "triangulation_frame_id": result.initialization_frame_by_track_id[
                record.track_id
            ],
            "window_start_frame_id": start,
            "frame_offset_from_window_start": record.frame_id - start,
            "triangulation_offset_from_window_start": (
                result.initialization_frame_by_track_id[record.track_id] - start
            ),
            "pnp_path_distance_from_window_start_m": pnp_path[
                record.frame_id - start
            ],
            "ground_truth_path_distance_from_window_start_m": gt_path[
                record.frame_id - start
            ],
            "measurement_u_left_px": measurement[0],
            "measurement_u_right_px": measurement[1],
            "measurement_v_px": measurement[2],
            "initial_projection_u_left_px": initial_projection[0],
            "initial_projection_u_right_px": initial_projection[1],
            "initial_projection_v_px": initial_projection[2],
            "final_projection_u_left_px": final_projection[0],
            "final_projection_u_right_px": final_projection[1],
            "final_projection_v_px": final_projection[2],
            "initial_residual_u_left_px": initial_residual[0],
            "initial_residual_u_right_px": initial_residual[1],
            "initial_residual_v_px": initial_residual[2],
            "final_residual_u_left_px": final_residual[0],
            "final_residual_u_right_px": final_residual[1],
            "final_residual_v_px": final_residual[2],
            "initial_left_pixel_distance_px": np.linalg.norm(
                initial_residual[[0, 2]]
            ),
            "initial_right_pixel_distance_px": np.linalg.norm(
                initial_residual[[1, 2]]
            ),
            "initial_stereo_pixel_distance_px": np.linalg.norm(initial_residual),
            "final_left_pixel_distance_px": np.linalg.norm(
                final_residual[[0, 2]]
            ),
            "final_right_pixel_distance_px": np.linalg.norm(
                final_residual[[1, 2]]
            ),
            "final_stereo_pixel_distance_px": np.linalg.norm(final_residual),
            "initial_factor_error": initial_error,
            "final_factor_error": final_error,
        }
        rows.append(row)
    return rows


def _result_arrays(result: BundleWindowResult) -> Dict[str, np.ndarray]:
    initial_camera_poses = []
    final_camera_poses = []
    for frame_id in result.frame_ids:
        key = result.camera_key_by_frame_id[frame_id]
        initial_camera_poses.append(result.initial_values.atPose3(key).matrix())
        final_camera_poses.append(result.final_values.atPose3(key).matrix())
    initial_landmarks = []
    final_landmarks = []
    for track_id in result.accepted_track_ids:
        key = result.landmark_key_by_track_id[track_id]
        initial_landmarks.append(result.initial_values.atPoint3(key))
        final_landmarks.append(result.final_values.atPoint3(key))
    return {
        "initial_camera_poses": _little_float64(initial_camera_poses),
        "final_camera_poses": _little_float64(final_camera_poses),
        "initial_landmarks": _little_float64(initial_landmarks),
        "final_landmarks": _little_float64(final_landmarks),
    }


def _maximum_diagnostic_json(result: BundleWindowResult) -> Dict[str, object]:
    value = result.serializable_summary()["maximum_initial_projection_error"]
    return {
        "schema": "exercise5-maximum-initial-projection-factor-v1",
        **value,
    }


def _summary(
    result: BundleWindowResult,
    *,
    window_index: int,
    config_sha256: str,
    source_identity_sha256: str,
    covariance_diagnostics: Mapping[str, object],
    runtime_seconds: float,
    peak_rss_bytes: int,
    tolerance: Mapping[str, float],
    tangent_order: Sequence[str],
) -> Dict[str, object]:
    initial_error = result.initial_errors.total_graph_error
    final_error = result.final_errors.total_graph_error
    allowance = max(
        tolerance["graph_error_nonincrease_absolute"],
        tolerance["graph_error_nonincrease_relative"] * abs(initial_error),
    )
    if not np.isfinite(initial_error) or not np.isfinite(final_error):
        raise ArtifactValidationError("graph error is nonfinite")
    if final_error > initial_error + allowance:
        raise ArtifactValidationError("graph error increased beyond tolerance")
    summary = result.serializable_summary()
    return {
        "schema": WINDOW_SCHEMA,
        "schema_version": 1,
        "seal_status": "prepared-for-seal",
        "window_index": window_index,
        "start_frame_id": result.frame_ids[0],
        "end_frame_id": result.frame_ids[-1],
        "configuration_sha256": config_sha256,
        "source_input_identity_sha256": source_identity_sha256,
        "tangent_order": list(tangent_order),
        "bundle_window": summary,
        "covariance_diagnostics": covariance_diagnostics,
        "runtime": {
            "window_total_seconds": runtime_seconds,
            "graph_construction_seconds": (
                result.graph_construction_runtime_seconds
            ),
            "optimization_seconds": result.optimization_runtime_seconds,
            "optimizer_iterations": result.optimizer_iterations,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "graph_error_gate": {
            "initial": initial_error,
            "final": final_error,
            "allowed_increase": allowance,
            "passed": True,
        },
    }




# Sealing and quarantine lifecycle


def _seal_window_artifacts(
    *,
    windows_root: Path,
    quarantine_root: Path,
    result: BundleWindowResult,
    database: object,
    calibration: gtsam.Cal3_S2Stereo,
    world_to_camera_poses: Sequence[np.ndarray],
    ground_truth_poses: Sequence[np.ndarray],
    configuration: Mapping[str, object],
    config_sha256: str,
    source_identity_sha256: str,
    window_index: int,
    endpoint_covariances: Mapping[str, object],
    first_window_camera_marginals: Optional[Mapping[str, np.ndarray]],
    runtime_seconds: float,
    peak_rss_bytes: int,
    endpoint_edge_extractor: Callable[..., Dict[str, object]],
    tangent_order: Sequence[str],
    image_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Write, reload, hash-last, and atomically promote one valid window.

    All files are built in a unique attempt directory.  The directory becomes
    resumable scientific evidence only after prepared validation, hash-seal
    creation, sealed reload, and atomic promotion have all succeeded.
    """

    start, end = result.frame_ids[0], result.frame_ids[-1]
    name = _window_name(window_index, start, end)
    windows_root = Path(windows_root)
    quarantine_root = Path(quarantine_root)
    windows_root.mkdir(parents=True, exist_ok=True)
    final_directory = windows_root / name
    if final_directory.exists():
        return validate_sealed_window(
            final_directory,
            expected_index=window_index,
            expected_start=start,
            expected_end=end,
            config_sha256=config_sha256,
            source_identity_sha256=source_identity_sha256,
        )
    attempt = windows_root / (
        ".{}.attempt.{}.tmp".format(name, uuid.uuid4().hex)
    )
    attempt.mkdir(parents=False)
    try:
        # Prepare normalized machine-readable evidence while the optimized
        # graph result is still available to the streaming caller.
        tolerance = configuration["scientific_configuration"][
            "validation_tolerances"
        ]
        summary = _summary(
            result,
            window_index=window_index,
            config_sha256=config_sha256,
            source_identity_sha256=source_identity_sha256,
            covariance_diagnostics=endpoint_covariances["diagnostics"],
            runtime_seconds=runtime_seconds,
            peak_rss_bytes=peak_rss_bytes,
            tolerance=tolerance,
            tangent_order=tangent_order,
        )
        _write_json(attempt / "summary.json", summary)
        _save_npz(
            attempt / "ids.npz",
            frame_ids=_little_int64(result.frame_ids),
            candidate_track_ids=_little_int64(result.candidate_track_ids),
            accepted_track_ids=_little_int64(result.accepted_track_ids),
            rejected_track_ids=_little_int64(result.rejected_track_ids),
        )
        arrays = _result_arrays(result)
        _save_npz(attempt / "values.npz", **arrays)
        track_rows = _build_tracks_rows(result, database)
        _write_csv(
            attempt / "tracks.csv",
            (
                "track_id",
                "accepted",
                "in_window_observation_count",
                "valid_observation_count",
                "invalid_observation_count",
                "distinct_valid_frame_count",
                "triangulation_frame_id",
                "triangulation_attempt_frame_ids_json",
                "rejection_reason",
            ),
            track_rows,
        )
        projection_rows = _build_projection_rows(
            result,
            calibration,
            world_to_camera_poses,
            ground_truth_poses,
        )
        _write_csv(
            attempt / "projection_residuals.csv",
            tuple(projection_rows[0].keys()),
            projection_rows,
        )
        _write_json(
            attempt / "endpoint_edge.json",
            endpoint_edge_extractor(
                result,
                rotation_atol=tolerance["rotation_orthogonality_max_abs"],
            ),
        )
        _save_npz(
            attempt / "endpoint_covariances.npz",
            joint_covariance=endpoint_covariances["joint_covariance"],
            joint_information=endpoint_covariances["joint_information"],
            conditional_end_given_start_covariance=endpoint_covariances[
                "conditional_end_given_start_covariance"
            ],
        )
        if window_index == 0:
            # Only the first window carries the extra Marginals and visual
            # evidence required by the Exercise 5 report.
            if first_window_camera_marginals is None:
                raise ArtifactValidationError(
                    "first window camera marginals are required"
                )
            _save_npz(
                attempt / "camera_marginals.npz",
                **first_window_camera_marginals
            )
            _write_json(
                attempt / "maximum_initial_projection_factor.json",
                _maximum_diagnostic_json(result),
            )
            from exercise5_reporting import (
                plot_first_window_artifacts,
            )

            plot_first_window_artifacts(
                result=result,
                calibration=calibration,
                output_directory=attempt,
                image_root=image_root,
            )
        # Validate the complete unhashed payload before declaring any file a
        # member of a sealed artifact set.
        validate_prepared_window(
            attempt,
            expected_index=window_index,
            expected_start=start,
            expected_end=end,
            config_sha256=config_sha256,
            source_identity_sha256=source_identity_sha256,
        )
        filenames = _window_required_files(window_index)
        # The hash index is intentionally written last: its presence means the
        # payload was complete and prepared validation already passed.
        seal = {
            "schema": HASH_SCHEMA,
            "schema_version": 1,
            "status": "sealed",
            "sealed_utc": utc_now(),
            "window_index": window_index,
            "start_frame_id": start,
            "end_frame_id": end,
            "configuration_sha256": config_sha256,
            "source_input_identity_sha256": source_identity_sha256,
            "artifacts": _hash_entries(attempt, filenames),
        }
        _write_json(attempt / "artifact_hashes.json", seal)
        _fsync_directory(attempt)
        validate_sealed_window(
            attempt,
            expected_index=window_index,
            expected_start=start,
            expected_end=end,
            config_sha256=config_sha256,
            source_identity_sha256=source_identity_sha256,
        )
        # Atomic directory promotion is the only transition from attempt state
        # to a resumable final window.
        os.replace(str(attempt), str(final_directory))
        _fsync_directory(windows_root)
        return validate_sealed_window(
            final_directory,
            expected_index=window_index,
            expected_start=start,
            expected_end=end,
            config_sha256=config_sha256,
            source_identity_sha256=source_identity_sha256,
        )
    except BaseException as exception:
        quarantine_attempt(
            attempt, quarantine_root, exception=exception
        )
        raise


# Run-level configuration, state, manifest, and checksum support


def snapshot_configuration(
    output_root: Path,
    configuration: Mapping[str, object],
    config_sha256: str,
) -> Path:
    """Persist the exact run configuration and its original file SHA."""

    path = Path(output_root) / "configuration.json"
    _write_json(
        path,
        {
            "schema": "exercise5-configuration-snapshot-v1",
            "source_sha256": config_sha256,
            "configuration": configuration,
        },
    )
    return path


def update_run_state(
    output_root: Path,
    *,
    status: str,
    expected_window_count: int,
    completed_window_indices: Sequence[int],
    history: Sequence[Mapping[str, object]],
    scope: str,
) -> Path:
    """Write resumable operational progress separately from the manifest."""

    path = Path(output_root) / "run_state.json"
    _write_json(
        path,
        {
            "schema": RUN_STATE_SCHEMA,
            "schema_version": 1,
            "status": status,
            "scope": scope,
            "updated_utc": utc_now(),
            "expected_window_count": expected_window_count,
            "completed_window_indices": list(completed_window_indices),
            "completion_bitmap": [
                index in set(completed_window_indices)
                for index in range(expected_window_count)
            ],
            "history": list(history),
        },
    )
    return path


def write_generator_registry(
    output_root: Path, configuration: Mapping[str, object]
) -> Path:
    """Snapshot the frozen artifact-to-producer ownership map."""

    path = Path(output_root) / "provenance" / "generators.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "schema": "exercise5-generator-registry-v1",
            "producers": configuration["artifact_contract"][
                "producer_registry"
            ],
        },
    )
    return path


def write_checksum_index(output_root: Path) -> Path:
    """Hash every non-quarantine run artifact except the index itself."""

    output_root = Path(output_root)
    path = output_root / "checksums.sha256"
    candidates = sorted(
        item
        for item in output_root.rglob("*")
        if item.is_file()
        and item != path
        and "quarantine" not in item.parts
    )
    lines = [
        "{}  {}".format(sha256_file(item), item.relative_to(output_root))
        for item in candidates
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def build_run_manifest(
    *,
    configuration: Mapping[str, object],
    config_sha256: str,
    source_identity: Mapping[str, object],
    completed_window_indices: Sequence[int],
    status: str,
    scope: str,
    created_utc: Optional[str] = None,
) -> Dict[str, object]:
    """Build the run identity and completion record without writing it."""

    expected = configuration["keyframe_selection"]["statistics"][
        "window_count"
    ]
    if status == "complete-and-sealed" and list(completed_window_indices) != list(
        range(expected)
    ):
        raise ArtifactValidationError(
            "complete manifest requires every canonical window"
        )
    identity_hash = hashlib.sha256(
        _canonical_json_bytes(source_identity)
    ).hexdigest()
    return {
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": status,
        "scope": scope,
        "created_utc": created_utc if created_utc is not None else utc_now(),
        "configuration_sha256": config_sha256,
        "source_input_identity": source_identity,
        "source_input_identity_sha256": identity_hash,
        "coordinate_contract": configuration["coordinate_contract"],
        "scientific_configuration": configuration[
            "scientific_configuration"
        ],
        "ordered_keyframe_ids": configuration["keyframe_selection"][
            "ordered_keyframe_ids"
        ],
        "expected_window_count": expected,
        "completed_window_indices": list(completed_window_indices),
        "tangent_transport_status": (
            "conditional endpoint tangent evidence retained; "
            "BetweenFactorPose3 transport pending E5-I04"
        ),
    }


def write_and_seal_manifest(
    output_root: Path,
    *,
    configuration: Mapping[str, object],
    config_sha256: str,
    source_identity: Mapping[str, object],
    completed_window_indices: Sequence[int],
    status: str,
    scope: str,
) -> Path:
    """Serialize the current run manifest deterministically."""

    path = Path(output_root) / "manifest.json"
    _write_json(
        path,
        build_run_manifest(
            configuration=configuration,
            config_sha256=config_sha256,
            source_identity=source_identity,
            completed_window_indices=completed_window_indices,
            status=status,
            scope=scope,
        ),
    )
    return path




__all__ = [
    "ArtifactValidationError",
    "BASE_WINDOW_FILES",
    "FIRST_WINDOW_FILES",
    "HASH_SCHEMA",
    "MANIFEST_SCHEMA",
    "RUN_STATE_SCHEMA",
    "WINDOW_SCHEMA",
    "build_run_manifest",
    "quarantine_attempt",
    "sha256_file",
    "snapshot_configuration",
    "update_run_state",
    "utc_now",
    "validate_prepared_window",
    "validate_sealed_window",
    "write_and_seal_manifest",
    "write_checksum_index",
    "write_generator_registry",
]
