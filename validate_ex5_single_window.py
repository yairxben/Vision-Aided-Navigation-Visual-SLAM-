"""Fresh-process canonical validation for the bounded E5-I02 window.

This driver loads the verified Exercise 4 database/companion pair exactly
once, solves only inclusive frames 0..9 when invoked for this milestone, and
writes compact evidence only beneath a caller-supplied ``/tmp/e5_i02_*``
directory.  All scientific probe parameters are required command-line
arguments and remain explicitly noncanonical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping, Sequence, Tuple

import gtsam
import numpy as np

from exercise4_analysis import load_tracking_artifacts
from exercise5_bundle_window import (
    BundleWindowResult,
    ProbeLandmarkEligibility,
    solve_bundle_window,
)


ROOT = Path(__file__).resolve().parent
CANONICAL_DATABASE_PATH = (
    ROOT / "diagnostics" / "ex4_final" / "tracking_database.pkl"
).resolve()
CANONICAL_COMPANION_PATH = (
    ROOT
    / "diagnostics"
    / "ex4_final"
    / "tracking_database_companion.json"
).resolve()
CANONICAL_CALIBRATION_PATH = (
    ROOT / "dataset" / "sequences" / "00" / "calib.txt"
).resolve()

EXPECTED_DATABASE_SIZE_BYTES = 2_484_226_639
EXPECTED_DATABASE_SHA256 = (
    "e0eba0d75887af544f3fdc0ada50932adb5c1edf52ea983a2a1be2c8244ce187"
)
EXPECTED_DATABASE_STREAMING_DIGEST = (
    "186fe6dbab6371cf1e4474ab833059e960218b8f6968a6c5caa597bc95fa46eb"
)
EXPECTED_COMPANION_SHA256 = (
    "5bbd24404acf9c0b1b54bcd9183a1a711f574e532a327043bdd2477cb7e76360"
)
EXPECTED_CALIBRATION_SHA256 = (
    "7cfc9ab047da6bd829d669e31b2a0d72c278d084fa0c02f2f0320795d51a5d01"
)
EXPECTED_FRAME_COUNT = 3360
REPOSITORY_MINIMUM_MEM_AVAILABLE_BYTES = 4_294_967_296
REPOSITORY_MINIMUM_MEM_PLUS_SWAP_BYTES = 6_567_682_560


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate exactly one canonical E5-I02 Bundle window"
    )
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--companion-path", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--start-frame-id", type=int, required=True)
    parser.add_argument("--end-frame-id", type=int, required=True)
    parser.add_argument(
        "--projection-sigma-px", type=float, required=True
    )
    parser.add_argument(
        "--anchor-rotation-sigma-rad", type=float, required=True
    )
    parser.add_argument(
        "--anchor-translation-sigma-m", type=float, required=True
    )
    parser.add_argument(
        "--minimum-valid-observations", type=int, required=True
    )
    parser.add_argument(
        "--minimum-distinct-frames", type=int, required=True
    )
    parser.add_argument("--rotation-atol", type=float, required=True)
    parser.add_argument(
        "--probe-numerical-atol", type=float, required=True
    )
    parser.add_argument(
        "--minimum-mem-available-bytes", type=int, required=True
    )
    parser.add_argument(
        "--minimum-mem-plus-swap-bytes", type=int, required=True
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> Dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _memory_snapshot() -> Dict[str, int]:
    fields: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(
        encoding="utf-8"
    ).splitlines():
        name, raw_value = line.split(":", 1)
        if name in ("MemAvailable", "SwapFree"):
            fields[name] = int(raw_value.split()[0]) * 1024
    if set(fields) != {"MemAvailable", "SwapFree"}:
        raise RuntimeError("Linux memory capacity fields are unavailable")
    return {
        "mem_available_bytes": fields["MemAvailable"],
        "swap_free_bytes": fields["SwapFree"],
        "mem_available_plus_swap_free_bytes": (
            fields["MemAvailable"] + fields["SwapFree"]
        ),
    }


def _vm_hwm_bytes() -> int:
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux VmHWM is unavailable")


def _output_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != Path("/tmp") or not resolved.name.startswith(
        "e5_i02_"
    ):
        raise ValueError(
            "output directory must be a new /tmp/e5_i02_* directory"
        )
    resolved.mkdir(parents=False, exist_ok=True)
    summary_path = resolved / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    return resolved


def _read_calibration(
    path: Path, *, validation_atol: float
) -> Tuple[gtsam.Cal3_S2Stereo, Dict[str, float]]:
    rows: Dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        label, values = line.split(":", 1)
        rows[label] = np.asarray(values.split(), dtype=float).reshape(3, 4)
    if "P0" not in rows or "P1" not in rows:
        raise ValueError("calibration must contain P0 and P1")
    projection_left = rows["P0"]
    projection_right = rows["P1"]
    if not np.all(np.isfinite(projection_left)) or not np.all(
        np.isfinite(projection_right)
    ):
        raise ValueError("calibration projection matrices are nonfinite")
    intrinsic = projection_left[:, :3]
    left_extrinsic = np.linalg.solve(intrinsic, projection_left)
    right_extrinsic = np.linalg.solve(intrinsic, projection_right)
    if (
        np.max(np.abs(left_extrinsic[:, :3] - np.eye(3)))
        > validation_atol
        or np.max(np.abs(left_extrinsic[:, 3])) > validation_atol
        or np.max(np.abs(right_extrinsic[:, :3] - np.eye(3)))
        > validation_atol
        or np.max(np.abs(right_extrinsic[1:, 3])) > validation_atol
    ):
        raise AssertionError(
            "calibration is not the expected rectified stereo contract"
        )
    baseline = -float(right_extrinsic[0, 3])
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise AssertionError("calibration baseline is not finite and positive")
    calibration = gtsam.Cal3_S2Stereo(
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
        baseline,
    )
    return calibration, {
        "fx": float(calibration.fx()),
        "fy": float(calibration.fy()),
        "skew": float(calibration.skew()),
        "cx": float(calibration.px()),
        "cy": float(calibration.py()),
        "baseline_m": float(calibration.baseline()),
        "right_extrinsic_tx_m": float(right_extrinsic[0, 3]),
    }


def _validate_exact_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    database_path = args.database_path.resolve()
    companion_path = args.companion_path.resolve()
    calibration_path = args.calibration_path.resolve()
    expected_observed = (
        (CANONICAL_DATABASE_PATH, database_path, "database"),
        (CANONICAL_COMPANION_PATH, companion_path, "companion"),
        (CANONICAL_CALIBRATION_PATH, calibration_path, "calibration"),
    )
    for expected, observed, name in expected_observed:
        if observed != expected:
            raise ValueError(
                "{} path is not the canonical E5-I02 input".format(name)
            )
    return database_path, companion_path, calibration_path


def _assert_canonical_loaded_pair(
    *,
    database: object,
    typed_companion: object,
    normalized: Mapping[str, object],
    calibration_sha256: str,
) -> Dict[str, object]:
    raw = normalized["raw_companion"]
    observed_database = normalized["database"]
    observed_companion = normalized["companion"]
    expected_equalities = (
        (normalized["schema"], "exercise4-final-companion-v1", "schema"),
        (raw.get("schema_version"), 1, "schema_version"),
        (
            raw.get("tracking_state_schema"),
            "exercise4-trackingdb-companion-v1",
            "tracking_state_schema",
        ),
        (
            raw.get("frame_range_inclusive"),
            [0, EXPECTED_FRAME_COUNT - 1],
            "frame_range_inclusive",
        ),
        (raw.get("frame_count"), EXPECTED_FRAME_COUNT, "frame_count"),
        (
            raw.get("transition_count"),
            EXPECTED_FRAME_COUNT - 1,
            "transition_count",
        ),
        (
            observed_database["size_bytes"],
            EXPECTED_DATABASE_SIZE_BYTES,
            "database size",
        ),
        (
            observed_database["file_sha256"],
            EXPECTED_DATABASE_SHA256,
            "database SHA-256",
        ),
        (
            observed_database["streaming_digest"],
            EXPECTED_DATABASE_STREAMING_DIGEST,
            "database streaming digest",
        ),
        (
            observed_companion["file_sha256"],
            EXPECTED_COMPANION_SHA256,
            "companion SHA-256",
        ),
        (
            calibration_sha256,
            EXPECTED_CALIBRATION_SHA256,
            "calibration SHA-256",
        ),
        (
            raw["provenance"]["dataset"]["calibration_sha256"],
            calibration_sha256,
            "companion calibration SHA-256",
        ),
        (database.frame_num(), EXPECTED_FRAME_COUNT, "database frame count"),
        (
            len(typed_companion.world_to_camera_poses),
            EXPECTED_FRAME_COUNT,
            "pose count",
        ),
    )
    for observed, expected, name in expected_equalities:
        if observed != expected:
            raise AssertionError(
                "{} differs: observed {!r}, expected {!r}".format(
                    name, observed, expected
                )
            )
    return {
        "schema": normalized["schema"],
        "schema_version": raw["schema_version"],
        "tracking_state_schema": raw["tracking_state_schema"],
        "frame_count": raw["frame_count"],
        "transition_count": raw["transition_count"],
        "frame_range_inclusive": raw["frame_range_inclusive"],
        "database_size_bytes": observed_database["size_bytes"],
        "database_sha256": observed_database["file_sha256"],
        "database_streaming_digest_schema": observed_database[
            "streaming_digest_schema"
        ],
        "database_streaming_digest": observed_database[
            "streaming_digest"
        ],
        "companion_sha256": observed_companion["file_sha256"],
        "calibration_sha256": calibration_sha256,
    }


def _finite_values(result: BundleWindowResult) -> bool:
    for key in result.camera_key_by_frame_id.values():
        if not np.all(
            np.isfinite(result.initial_values.atPose3(key).matrix())
        ) or not np.all(
            np.isfinite(result.final_values.atPose3(key).matrix())
        ):
            return False
    for key in result.landmark_key_by_track_id.values():
        if not np.all(
            np.isfinite(result.initial_values.atPoint3(key))
        ) or not np.all(np.isfinite(result.final_values.atPoint3(key))):
            return False
    return True


def _all_projection_depths_positive(
    result: BundleWindowResult,
) -> bool:
    for values in (result.initial_values, result.final_values):
        for record in result.projection_factor_records:
            camera_point = np.asarray(
                values.atPose3(record.camera_key).transformTo(
                    values.atPoint3(record.landmark_key)
                )
            )
            if (
                not np.all(np.isfinite(camera_point))
                or not float(camera_point[2]) > 0.0
            ):
                return False
    return True


def _maximum_diagnostic_consistency(
    *,
    result: BundleWindowResult,
    calibration: gtsam.Cal3_S2Stereo,
) -> Dict[str, object]:
    diagnostic = result.maximum_initial_projection_error
    matching_records = [
        record
        for record in result.projection_factor_records
        if record.factor_index == diagnostic.factor_index
    ]
    if len(matching_records) != 1:
        raise AssertionError(
            "maximum diagnostic does not select one projection record"
        )
    record = matching_records[0]
    if (
        diagnostic.frame_id,
        diagnostic.track_id,
        diagnostic.camera_key,
        diagnostic.landmark_key,
    ) != (
        record.frame_id,
        record.track_id,
        record.camera_key,
        record.landmark_key,
    ):
        raise AssertionError(
            "maximum diagnostic identity differs from its factor record"
        )
    factor = result.graph.at(record.factor_index)
    factor_measurement = np.asarray(factor.measured().vector(), dtype=float)
    recorded_measurement = np.asarray(record.measurement, dtype=float)
    diagnostic_measurement = np.asarray(
        diagnostic.measurement, dtype=float
    )
    measurement_difference = float(
        max(
            np.max(np.abs(factor_measurement - recorded_measurement)),
            np.max(np.abs(factor_measurement - diagnostic_measurement)),
        )
    )
    state_diagnostics: Dict[str, Dict[str, object]] = {}
    maximum_consistency_difference = measurement_difference
    for state_name, values, recorded_projection, recorded_residual in (
        (
            "initial",
            result.initial_values,
            diagnostic.initial_projection,
            diagnostic.initial_residual,
        ),
        (
            "final",
            result.final_values,
            diagnostic.final_projection,
            diagnostic.final_residual,
        ),
    ):
        projection = np.asarray(
            gtsam.StereoCamera(
                values.atPose3(record.camera_key), calibration
            )
            .project(values.atPoint3(record.landmark_key))
            .vector(),
            dtype=float,
        )
        residual = projection - factor_measurement
        factor_residual = np.asarray(
            factor.unwhitenedError(values), dtype=float
        )
        recorded_projection_array = np.asarray(
            recorded_projection, dtype=float
        )
        recorded_residual_array = np.asarray(recorded_residual, dtype=float)
        differences = {
            "projection_max_abs_difference": float(
                np.max(np.abs(projection - recorded_projection_array))
            ),
            "residual_max_abs_difference": float(
                np.max(np.abs(residual - recorded_residual_array))
            ),
            "factor_residual_max_abs_difference": float(
                np.max(np.abs(residual - factor_residual))
            ),
        }
        maximum_consistency_difference = max(
            maximum_consistency_difference, *differences.values()
        )
        state_diagnostics[state_name] = {
            "direct_projection": projection.tolist(),
            "direct_residual": residual.tolist(),
            **differences,
        }

    all_initial_errors = [
        float(
            result.graph.at(record.factor_index).error(
                result.initial_values
            )
        )
        for record in result.projection_factor_records
    ]
    selected_error_difference = abs(
        diagnostic.initial_factor_error - max(all_initial_errors)
    )
    maximum_consistency_difference = max(
        maximum_consistency_difference, selected_error_difference
    )
    return {
        "record_frame_id": record.frame_id,
        "record_track_id": record.track_id,
        "measurement_max_abs_difference": measurement_difference,
        "selected_error_vs_max_absolute_difference": (
            selected_error_difference
        ),
        "maximum_consistency_absolute_difference": (
            maximum_consistency_difference
        ),
        "states": state_diagnostics,
    }


def _validation_gates(
    *,
    result: BundleWindowResult,
    calibration: gtsam.Cal3_S2Stereo,
    start_frame_id: int,
    end_frame_id: int,
    minimum_valid_observations: int,
    minimum_distinct_frames: int,
    probe_numerical_atol: float,
    input_stats_unchanged: bool,
) -> Tuple[Dict[str, bool], Dict[str, object]]:
    expected_frame_ids = tuple(range(start_frame_id, end_frame_id + 1))
    first_camera_matrix = np.asarray(
        result.initial_values.atPose3(
            result.camera_key_by_frame_id[start_frame_id]
        ).matrix()
    )
    first_camera_identity_error = float(
        np.max(np.abs(first_camera_matrix - np.eye(4)))
    )
    pose_prior_count = sum(
        isinstance(result.graph.at(index), gtsam.PriorFactorPose3)
        for index in range(result.graph.size())
    )
    pose_priors = [
        result.graph.at(index)
        for index in range(result.graph.size())
        if isinstance(result.graph.at(index), gtsam.PriorFactorPose3)
    ]
    prior_targets_first_camera_identity = (
        len(pose_priors) == 1
        and list(pose_priors[0].keys())
        == [result.camera_key_by_frame_id[start_frame_id]]
        and float(
            np.max(
                np.abs(
                    np.asarray(pose_priors[0].prior().matrix())
                    - np.eye(4)
                )
            )
        )
        <= probe_numerical_atol
    )
    stereo_factor_count = sum(
        isinstance(result.graph.at(index), gtsam.GenericStereoFactor3D)
        for index in range(result.graph.size())
    )
    accepted_observation_requirements = True
    for track_id in result.accepted_track_ids:
        frames = {
            frame_id
            for frame_id, observed_track_id in result.observation_pairs
            if observed_track_id == track_id
        }
        accepted_observation_requirements = (
            accepted_observation_requirements
            and result.accepted_observation_count_by_track_id[track_id]
            >= minimum_valid_observations
            and len(frames) >= minimum_distinct_frames
        )

    metric_values = []
    for metrics in (result.initial_errors, result.final_errors):
        metric_values.extend(
            [
                metrics.total_graph_error,
                metrics.mean_all_factor_error,
                metrics.projection_factor_error_sum,
                metrics.mean_projection_factor_error,
                metrics.anchor_factor_error,
                metrics.graph_vs_factor_sum_absolute_difference,
            ]
        )
    diagnostic = result.maximum_initial_projection_error
    metric_values.extend(
        [
            diagnostic.initial_factor_error,
            diagnostic.final_factor_error,
            *diagnostic.measurement,
            *diagnostic.initial_projection,
            *diagnostic.final_projection,
            *diagnostic.initial_residual,
            *diagnostic.final_residual,
        ]
    )
    reported_metrics_finite = bool(
        np.all(np.isfinite(np.asarray(metric_values, dtype=float)))
    )

    scale = max(
        1.0,
        abs(result.initial_errors.total_graph_error),
        abs(result.final_errors.total_graph_error),
    )
    measured_error_roundoff_allowance = max(
        result.initial_errors.graph_vs_factor_sum_absolute_difference,
        result.final_errors.graph_vs_factor_sum_absolute_difference,
        (
            np.finfo(float).eps
            * scale
            * float(result.total_factor_count)
        ),
    )
    maximum_consistency = _maximum_diagnostic_consistency(
        result=result, calibration=calibration
    )
    gates = {
        "ordered_frames_are_exact_requested_window": (
            result.frame_ids == expected_frame_ids
        ),
        "ten_ordered_cameras_cover_frames_0_through_9": (
            result.camera_count == 10
            and result.frame_ids == tuple(range(10))
        ),
        "first_camera_is_identity": (
            first_camera_identity_error <= probe_numerical_atol
        ),
        "exactly_one_pose_prior": (
            pose_prior_count == 1 and result.anchor_factor_count == 1
        ),
        "pose_prior_targets_first_camera_at_identity": (
            prior_targets_first_camera_identity
        ),
        "accepted_landmark_exists": result.accepted_track_count > 0,
        "projection_factor_exists": result.projection_factor_count > 0,
        "all_graph_variables_are_finite": _finite_values(result),
        "all_reported_metrics_are_finite": reported_metrics_finite,
        "raw_graph_error_matches_sum_of_factor_errors": (
            result.initial_errors.graph_vs_factor_sum_absolute_difference
            <= probe_numerical_atol
            and result.final_errors.graph_vs_factor_sum_absolute_difference
            <= probe_numerical_atol
        ),
        "all_projection_depths_are_positive": (
            _all_projection_depths_positive(result)
        ),
        "accepted_landmarks_meet_probe_observation_policy": (
            accepted_observation_requirements
        ),
        "observation_factor_pairs_are_unique": (
            len(result.observation_pairs)
            == len(set(result.observation_pairs))
        ),
        "projection_records_match_observation_pairs": (
            tuple(
                (record.frame_id, record.track_id)
                for record in result.projection_factor_records
            )
            == result.observation_pairs
        ),
        "projection_factor_count_matches_records": (
            stereo_factor_count
            == result.projection_factor_count
            == len(result.projection_factor_records)
        ),
        "total_factor_count_is_prior_plus_projections": (
            result.total_factor_count
            == result.projection_factor_count + 1
            == result.graph.size()
        ),
        "factor_residual_dimensions_cover_variable_dimensions": (
            result.factor_residual_dimension >= result.variable_dimension
        ),
        "camera_landmark_graph_is_anchor_connected": (
            result.connected_to_anchor
            and all(
                degree > 0
                for degree in result.camera_factor_degrees.values()
            )
        ),
        "optimization_does_not_increase_raw_total_error": (
            result.final_errors.total_graph_error
            <= result.initial_errors.total_graph_error
            + measured_error_roundoff_allowance
        ),
        "final_anchor_error_is_probe_numerically_negligible": (
            result.final_errors.anchor_factor_error
            <= probe_numerical_atol
        ),
        "maximum_projection_diagnostic_is_consistent": (
            maximum_consistency[
                "maximum_consistency_absolute_difference"
            ]
            <= probe_numerical_atol
        ),
        "canonical_input_file_stats_are_unchanged": input_stats_unchanged,
    }
    diagnostics = {
        "first_camera_identity_max_abs_error": (
            first_camera_identity_error
        ),
        "observed_pose_prior_count": pose_prior_count,
        "observed_stereo_factor_count": stereo_factor_count,
        "measured_error_roundoff_allowance": (
            measured_error_roundoff_allowance
        ),
        "probe_numerical_atol": probe_numerical_atol,
        "maximum_factor_consistency": maximum_consistency,
    }
    return {name: bool(passed) for name, passed in gates.items()}, diagnostics


def _write_json(path: Path, value: object) -> None:
    def numpy_scalar(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            "Object of type {} is not JSON serializable".format(
                value.__class__.__name__
            )
        )

    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
        default=numpy_scalar,
    )
    path.write_text(payload + "\n", encoding="utf-8")


def main() -> None:
    process_started = perf_counter()
    args = _parse_args()
    output_directory = _output_directory(args.output_directory)
    database_path, companion_path, calibration_path = _validate_exact_paths(
        args
    )
    tolerance_values = np.asarray(
        [
            args.projection_sigma_px,
            args.anchor_rotation_sigma_rad,
            args.anchor_translation_sigma_m,
            args.rotation_atol,
            args.probe_numerical_atol,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(tolerance_values)) or not np.all(
        tolerance_values > 0.0
    ):
        raise ValueError(
            "all probe noise and tolerance values must be finite and positive"
        )
    if (
        args.minimum_mem_available_bytes
        < REPOSITORY_MINIMUM_MEM_AVAILABLE_BYTES
        or args.minimum_mem_plus_swap_bytes
        < REPOSITORY_MINIMUM_MEM_PLUS_SWAP_BYTES
    ):
        raise ValueError(
            "memory thresholds must meet the existing repository safety "
            "precedent"
        )
    if (args.start_frame_id, args.end_frame_id) != (0, 9):
        raise ValueError("E5-I02 canonical validation is restricted to 0..9")
    if (
        args.minimum_valid_observations,
        args.minimum_distinct_frames,
    ) != (2, 2):
        raise ValueError(
            "E5-I02 canonical probe eligibility is explicitly 2 observations "
            "in 2 distinct frames"
        )

    memory_before_load = _memory_snapshot()
    memory_guard = {
        "minimum_mem_available_bytes": (
            args.minimum_mem_available_bytes
        ),
        "minimum_mem_plus_swap_bytes": (
            args.minimum_mem_plus_swap_bytes
        ),
        **memory_before_load,
    }
    if (
        memory_before_load["mem_available_bytes"]
        < args.minimum_mem_available_bytes
        or memory_before_load["mem_available_plus_swap_free_bytes"]
        < args.minimum_mem_plus_swap_bytes
    ):
        raise MemoryError(
            "available memory does not satisfy the explicit E5-I02 guard"
        )
    print(
        "E5-I02 memory preflight passed: {}".format(
            json.dumps(memory_guard, sort_keys=True)
        ),
        flush=True,
    )

    input_stats_before = {
        "database": _file_identity(database_path),
        "companion": _file_identity(companion_path),
        "calibration": _file_identity(calibration_path),
    }
    calibration_sha256 = _sha256(calibration_path)
    calibration, calibration_summary = _read_calibration(
        calibration_path, validation_atol=args.rotation_atol
    )

    load_started = perf_counter()
    database, typed_companion, normalized = load_tracking_artifacts(
        database_path, companion_path
    )
    load_runtime_seconds = perf_counter() - load_started
    input_identity = _assert_canonical_loaded_pair(
        database=database,
        typed_companion=typed_companion,
        normalized=normalized,
        calibration_sha256=calibration_sha256,
    )
    poses = typed_companion.world_to_camera_poses
    del normalized
    del typed_companion
    print(
        "E5-I02 canonical pair loaded and validated once in {:.6f} s".format(
            load_runtime_seconds
        ),
        flush=True,
    )

    projection_noise = gtsam.noiseModel.Isotropic.Sigma(
        3, args.projection_sigma_px
    )
    anchor_sigmas = np.asarray(
        [
            args.anchor_rotation_sigma_rad,
            args.anchor_rotation_sigma_rad,
            args.anchor_rotation_sigma_rad,
            args.anchor_translation_sigma_m,
            args.anchor_translation_sigma_m,
            args.anchor_translation_sigma_m,
        ]
    )
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(anchor_sigmas)
    landmark_eligibility = ProbeLandmarkEligibility(
        minimum_valid_observations=args.minimum_valid_observations,
        minimum_distinct_frames=args.minimum_distinct_frames,
    )

    solve_started = perf_counter()
    result = solve_bundle_window(
        database=database,
        world_to_camera_poses=poses,
        calibration=calibration,
        start_frame_id=args.start_frame_id,
        end_frame_id=args.end_frame_id,
        projection_noise=projection_noise,
        anchor_noise=anchor_noise,
        landmark_eligibility=landmark_eligibility,
        rotation_atol=args.rotation_atol,
    )
    solve_runtime_seconds = perf_counter() - solve_started
    print(
        "E5-I02 canonical 0..9 graph built and optimized in {:.6f} s".format(
            solve_runtime_seconds
        ),
        flush=True,
    )

    input_stats_after = {
        "database": _file_identity(database_path),
        "companion": _file_identity(companion_path),
        "calibration": _file_identity(calibration_path),
    }
    memory_after_solve = _memory_snapshot()
    input_stats_unchanged = input_stats_before == input_stats_after
    gates, gate_diagnostics = _validation_gates(
        result=result,
        calibration=calibration,
        start_frame_id=args.start_frame_id,
        end_frame_id=args.end_frame_id,
        minimum_valid_observations=args.minimum_valid_observations,
        minimum_distinct_frames=args.minimum_distinct_frames,
        probe_numerical_atol=args.probe_numerical_atol,
        input_stats_unchanged=input_stats_unchanged,
    )
    failed_gates = sorted(name for name, passed in gates.items() if not passed)

    runtime_before_evidence_write = perf_counter() - process_started
    peak_rss_bytes = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    report = {
        "schema": "exercise5-i02-single-window-validation-v1",
        "milestone": "E5-I02",
        "scope": "one canonical Bundle window, inclusive frames 0..9",
        "status": "pass" if not failed_gates else "fail",
        "configuration_status": (
            "explicit noncanonical E5-I02 implementation-probe values; "
            "not a final noise, keyframe, triangulation, landmark, or "
            "numerical-tolerance policy"
        ),
        "probe_configuration": {
            "start_frame_id": args.start_frame_id,
            "end_frame_id": args.end_frame_id,
            "projection_noise_sigmas_px": [
                args.projection_sigma_px
            ]
            * 3,
            "anchor_noise_sigmas_rotation_then_translation": (
                anchor_sigmas.tolist()
            ),
            "minimum_valid_observations": (
                args.minimum_valid_observations
            ),
            "minimum_distinct_frames": args.minimum_distinct_frames,
            "rotation_atol": args.rotation_atol,
            "probe_numerical_atol": args.probe_numerical_atol,
        },
        "canonical_input_identity": input_identity,
        "canonical_input_stats_before": input_stats_before,
        "canonical_input_stats_after": input_stats_after,
        "calibration": calibration_summary,
        "memory_preflight": memory_guard,
        "memory_after_single_window_solve": memory_after_solve,
        "runtime_and_peak_memory": {
            "canonical_load_and_full_validation_seconds": (
                load_runtime_seconds
            ),
            "single_window_solve_seconds": solve_runtime_seconds,
            "graph_construction_seconds": (
                result.graph_construction_runtime_seconds
            ),
            "levenberg_marquardt_seconds": (
                result.optimization_runtime_seconds
            ),
            "process_seconds_before_evidence_write": (
                runtime_before_evidence_write
            ),
            "resource_ru_maxrss_bytes": peak_rss_bytes,
            "proc_status_vm_hwm_bytes": _vm_hwm_bytes(),
        },
        "bundle_window": result.serializable_summary(),
        "validation_gates": gates,
        "validation_gate_diagnostics": gate_diagnostics,
        "failed_gates": failed_gates,
    }
    summary_path = output_directory / "summary.json"
    _write_json(summary_path, report)
    if failed_gates:
        raise AssertionError(
            "E5-I02 canonical validation gates failed: {}".format(
                failed_gates
            )
        )

    compact_output = {
        "status": report["status"],
        "summary_path": str(summary_path),
        "counts": report["bundle_window"]["counts"],
        "candidate_track_count": result.candidate_track_count,
        "accepted_track_count": result.accepted_track_count,
        "rejected_track_count": result.rejected_track_count,
        "rejection_reason_counts": dict(result.rejection_reason_counts),
        "invalid_observation_reason_counts": dict(
            result.invalid_observation_reason_counts
        ),
        "initial_errors": report["bundle_window"]["initial_errors"],
        "final_errors": report["bundle_window"]["final_errors"],
        "maximum_initial_projection_error": report["bundle_window"][
            "maximum_initial_projection_error"
        ],
        "runtime_and_peak_memory": report["runtime_and_peak_memory"],
        "failed_gates": failed_gates,
    }
    print(json.dumps(compact_output, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
