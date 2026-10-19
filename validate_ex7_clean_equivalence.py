"""Strict stored-evidence equivalence gate for historical and clean E7 runs.

This validator never performs matching, RANSAC, Bundle adjustment, Pose Graph
optimization, or Marginals construction.  It validates sealed files and
compares the retained scientific evidence directly.  Compact comparison
evidence is written atomically only after every gate passes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from exercise7_sequential_loop_closure import validate_current_seal


EXPECTED_KEYFRAME_COUNT = 356
EXPECTED_ORDERED_PAIR_COUNT = 63_190
EXPECTED_MATCHER_INVOCATION_COUNT = 1_002
EXPECTED_DISPOSITIONS = {
    "accepted": 6,
    "cap_not_evaluated": 23_382,
    "cluster_suppressed": 1_007,
    "mahalanobis_rejected": 32_579,
    "matcher_rejected": 994,
    "measurement_failed": 2,
    "temporal_excluded": 5_220,
}
EXPECTED_ACCEPTED_LOOPS = (
    (1, 63, 1478, 10, 160, 140, 51, 36.42857142857143),
    (2, 327, 2365, 33, 245, 192, 85, 44.270833333333336),
    (3, 2266, 3206, 232, 336, 445, 347, 77.97752808988764),
    (4, 2355, 3295, 243, 346, 176, 62, 35.22727272727273),
    (5, 289, 3307, 29, 349, 143, 43, 30.06993006993007),
    (6, 372, 3359, 40, 355, 147, 46, 31.292517006802722),
)
CLEAN_SOURCE_COMMIT = "7d908bedca47ef44ca42cc1ae641c2a2c3f51491"
SCIENTIFIC_ATOL = 1e-12


class EquivalenceError(RuntimeError):
    """Raised when a required internal or cross-root gate fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EquivalenceError(message)


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), "{} is not a JSON object".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hashes(root: Path, exclude_names: Iterable[str] = ()) -> Dict[str, str]:
    excluded = set(exclude_names)
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _content_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root)).encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _max_abs(left: np.ndarray, right: np.ndarray, label: str) -> float:
    _require(left.shape == right.shape, "{} shape differs".format(label))
    _require(left.dtype == right.dtype, "{} dtype differs".format(label))
    if left.size == 0:
        return 0.0
    difference = np.abs(left.astype(float) - right.astype(float))
    _require(np.all(np.isfinite(difference)), "{} difference is non-finite".format(label))
    return float(np.max(difference))


def _maximum_location(left: np.ndarray, right: np.ndarray) -> object:
    if left.size == 0:
        return None
    difference = np.abs(left.astype(float) - right.astype(float))
    maximum = float(np.max(difference))
    if maximum == 0.0:
        return None
    return [int(value) for value in np.unravel_index(int(np.argmax(difference)), difference.shape)]


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        return fields, [dict(row) for row in reader]


def _validate_final_manifest(root: Path) -> None:
    final_root = root / "final"
    manifest = _read_json(final_root / "manifest.json")
    _require(
        manifest.get("schema") == "exercise7-i04-manifest-v1"
        and manifest.get("status") == "complete-and-sealed"
        and manifest.get("technical_run_status") == "pass"
        and int(manifest.get("current_keyframe_count", -1)) == EXPECTED_KEYFRAME_COUNT
        and int(manifest.get("ordered_pair_count", -1)) == EXPECTED_ORDERED_PAIR_COUNT
        and int(manifest.get("successful_loop_closure_count", -1)) == 6,
        "{} final manifest scalar contract differs".format(root),
    )
    observed_seals = {
        "currents/current_{:04d}/seal.json".format(index): _sha256(
            root / "currents" / "current_{:04d}".format(index) / "seal.json"
        )
        for index in range(EXPECTED_KEYFRAME_COUNT)
    }
    _require(
        manifest.get("current_seal_sha256") == observed_seals,
        "{} final manifest current-seal hashes differ".format(root),
    )
    observed_derived = _directory_hashes(final_root, exclude_names=("manifest.json",))
    _require(
        manifest.get("derived_artifact_sha256") == observed_derived,
        "{} final derived-artifact hashes differ".format(root),
    )


def _validate_root(root: Path) -> Mapping[str, object]:
    root = root.resolve()
    _require(root.is_dir(), "scientific root is missing: {}".format(root))
    expected_directories = {
        "current_{:04d}".format(index) for index in range(EXPECTED_KEYFRAME_COUNT)
    }
    observed_directories = {
        path.name for path in (root / "currents").iterdir() if path.is_dir()
    }
    _require(
        observed_directories == expected_directories,
        "{} sealed-current directory population differs".format(root),
    )

    currents: List[Mapping[str, object]] = []
    accepted_artifacts: List[Path] = []
    frame_ids: List[int] = []
    for index in range(EXPECTED_KEYFRAME_COUNT):
        directory = root / "currents" / "current_{:04d}".format(index)
        sealed = validate_current_seal(directory)
        current = sealed["current"]
        _require(int(current["current_ordinal"]) == index, "current ordinal differs")
        frame_ids.append(int(current["current_frame_id"]))
        currents.append(current)
        for accepted in current["accepted_loops"]:
            accepted_artifacts.append(directory / str(accepted["artifact"]))

    _require(len(set(frame_ids)) == EXPECTED_KEYFRAME_COUNT, "FrameIds are not unique")
    _require(frame_ids[0] == 0 and frame_ids[-1] == 3359, "FrameId endpoints differ")
    _require(len(accepted_artifacts) == 6, "accepted artifact count differs")
    _validate_final_manifest(root)

    run_state = _read_json(root / "run_state.json")
    validation = _read_json(root / "final" / "validation.json")
    summary = _read_json(root / "final" / "summary.json")
    _require(
        run_state.get("status") == "complete-and-sealed"
        and int(run_state.get("completed_current_count", -1)) == EXPECTED_KEYFRAME_COUNT
        and int(run_state.get("last_completed_current_ordinal", -1)) == 355
        and int(run_state.get("state_version", -1)) == 6
        and int(run_state.get("successful_loop_closure_count", -1)) == 6
        and run_state.get("technical_run_status") == "pass",
        "{} completed run-state contract differs".format(root),
    )
    _require(
        validation.get("technical_run_status") == "pass"
        and bool(validation.get("all_current_keyframes_sealed"))
        and int(validation.get("current_keyframe_count", -1)) == EXPECTED_KEYFRAME_COUNT
        and int(validation.get("ordered_pair_count", -1)) == EXPECTED_ORDERED_PAIR_COUNT
        and int(validation.get("matcher_invocation_count", -1))
        == EXPECTED_MATCHER_INVOCATION_COUNT
        and validation.get("disposition_counts") == EXPECTED_DISPOSITIONS
        and int(validation.get("successful_loop_closure_count", -1)) == 6
        and int(validation.get("candidate_graph_edge_count", -1)) == 361
        and int(validation.get("pose_graph_factor_count", -1)) == 362
        and int(validation.get("all_node_marginal_count", -1)) == EXPECTED_KEYFRAME_COUNT
        and int(validation.get("state_version", -1)) == 6,
        "{} sealed technical validation differs".format(root),
    )
    _require(
        summary.get("technical_run_status") == "pass"
        and summary.get("exercise7_deliverable_status") == "pass"
        and summary.get("technical_validation") == validation,
        "{} sealed summary differs from validation".format(root),
    )
    return {
        "root": root,
        "currents": currents,
        "frame_ids": frame_ids,
        "accepted_artifacts": accepted_artifacts,
        "summary": summary,
        "validation": validation,
        "fingerprint": _content_fingerprint(root),
    }


def _normalized_current(current: Mapping[str, object]) -> Mapping[str, object]:
    normalized = copy.deepcopy(current)
    normalized.pop("runtime_seconds", None)
    normalized.pop("peak_rss_bytes", None)
    for matcher in normalized["matcher_invocations"]:
        matcher.pop("runtime_seconds", None)
    return normalized


def _compare_csv_excluding(
    historical_path: Path, clean_path: Path, excluded_fields: Iterable[str], label: str
) -> None:
    historical_fields, historical_rows = _read_csv(historical_path)
    clean_fields, clean_rows = _read_csv(clean_path)
    _require(historical_fields == clean_fields, "{} CSV schema differs".format(label))
    excluded = set(excluded_fields)
    normalized_historical = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in historical_rows
    ]
    normalized_clean = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in clean_rows
    ]
    _require(
        normalized_historical == normalized_clean,
        "{} non-runtime/provenance rows differ".format(label),
    )


def _numeric_csv_maximum(
    historical_path: Path, clean_path: Path, fields: Sequence[str], label: str
) -> float:
    historical_header, historical_rows = _read_csv(historical_path)
    clean_header, clean_rows = _read_csv(clean_path)
    _require(
        historical_header == clean_header and len(historical_rows) == len(clean_rows),
        "{} CSV population differs".format(label),
    )
    maximum = 0.0
    for historical, clean in zip(historical_rows, clean_rows):
        for field in fields:
            maximum = max(maximum, abs(float(historical[field]) - float(clean[field])))
    _require(maximum <= SCIENTIFIC_ATOL, "{} exceeds tolerance".format(label))
    return maximum


def _summary_statistics(values: np.ndarray) -> Mapping[str, float]:
    return {
        "final": float(values[-1]),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def _verify_report_metrics(root: Path, final_marginals: np.ndarray) -> None:
    report_root = root / "final" / "reports"
    report_summary = _read_json(report_root / "summary.json")
    _, error_rows = _read_csv(report_root / "data" / "absolute_pose_errors.csv")
    _, uncertainty_rows = _read_csv(
        report_root / "data" / "marginal_uncertainty_sizes.csv"
    )
    metrics = report_summary["metrics"]
    metric_sources = {
        ("with_loop", "location_error_m"): np.asarray(
            [float(row["with_loop_location_error_norm_m"]) for row in error_rows]
        ),
        ("with_loop", "absolute_rotation_error_deg"): np.asarray(
            [float(row["with_loop_rotation_error_deg"]) for row in error_rows]
        ),
        ("without_loop", "location_error_m"): np.asarray(
            [float(row["without_loop_location_error_norm_m"]) for row in error_rows]
        ),
        ("without_loop", "absolute_rotation_error_deg"): np.asarray(
            [float(row["without_loop_rotation_error_deg"]) for row in error_rows]
        ),
        ("with_loop", "location_uncertainty_m"): np.asarray(
            [float(row["with_loop_location_uncertainty_m"]) for row in uncertainty_rows]
        ),
        ("with_loop", "rotation_uncertainty_deg"): np.asarray(
            [float(row["with_loop_rotation_uncertainty_deg"]) for row in uncertainty_rows]
        ),
        ("without_loop", "location_uncertainty_m"): np.asarray(
            [float(row["without_loop_location_uncertainty_m"]) for row in uncertainty_rows]
        ),
        ("without_loop", "rotation_uncertainty_deg"): np.asarray(
            [float(row["without_loop_rotation_uncertainty_deg"]) for row in uncertainty_rows]
        ),
    }
    for (state, name), values in metric_sources.items():
        observed = _summary_statistics(values)
        expected = metrics[state][name]
        _require(
            all(abs(observed[key] - float(expected[key])) <= SCIENTIFIC_ATOL for key in observed),
            "{} {} summary metric re-derivation differs".format(root, name),
        )

    expected_location = np.sqrt(
        np.trace(final_marginals[:, 3:6, 3:6], axis1=1, axis2=2)
    )
    expected_rotation_deg = np.degrees(
        np.sqrt(np.trace(final_marginals[:, 0:3, 0:3], axis1=1, axis2=2))
    )
    recorded_location = np.asarray(
        [float(row["with_loop_location_uncertainty_m"]) for row in uncertainty_rows]
    )
    recorded_rotation_deg = np.asarray(
        [float(row["with_loop_rotation_uncertainty_deg"]) for row in uncertainty_rows]
    )
    _require(
        float(np.max(np.abs(expected_location - recorded_location))) <= SCIENTIFIC_ATOL
        and float(np.max(np.abs(expected_rotation_deg - recorded_rotation_deg)))
        <= SCIENTIFIC_ATOL,
        "{} raw final Marginals do not reproduce report uncertainty".format(root),
    )


def _compare_roots(historical: Mapping[str, object], clean: Mapping[str, object]) -> Mapping[str, object]:
    historical_root = historical["root"]
    clean_root = clean["root"]
    _require(historical["frame_ids"] == clean["frame_ids"], "FrameId sequence differs")

    candidate_covariance_max = 0.0
    determinant_max = 0.0
    log_determinant_max = 0.0
    mahalanobis_max = 0.0
    factor_error_max = 0.0
    matcher_invocations = 0
    disposition_counter: Counter = Counter()
    accepted_records: List[Tuple[Mapping[str, object], Path, Path]] = []
    candidate_path_byte_identical_count = 0
    current_runtime_difference_count = 0
    current_rss_difference_count = 0
    matcher_runtime_difference_count = 0

    exact_path_arrays = {
        "pair_indices",
        "path_frame_ids",
        "path_frame_offsets",
        "path_edge_indices",
        "path_edge_offsets",
    }
    numeric_path_arrays = {
        "accumulated_covariances",
        "covariance_determinants",
        "covariance_log_determinants",
    }

    for ordinal, (historical_current, clean_current) in enumerate(
        zip(historical["currents"], clean["currents"])
    ):
        _require(
            _normalized_current(historical_current) == _normalized_current(clean_current),
            "current {:04d} non-runtime scientific record differs".format(ordinal),
        )
        current_runtime_difference_count += int(
            historical_current.get("runtime_seconds") != clean_current.get("runtime_seconds")
        )
        current_rss_difference_count += int(
            historical_current.get("peak_rss_bytes") != clean_current.get("peak_rss_bytes")
        )
        for historical_matcher, clean_matcher in zip(
            historical_current["matcher_invocations"], clean_current["matcher_invocations"]
        ):
            matcher_runtime_difference_count += int(
                historical_matcher.get("runtime_seconds")
                != clean_matcher.get("runtime_seconds")
            )

        candidates = historical_current["candidates"]
        _require(len(candidates) == ordinal, "candidate population differs at current")
        for previous_ordinal, (historical_row, clean_row) in enumerate(
            zip(candidates, clean_current["candidates"])
        ):
            _require(
                int(historical_row["previous_keyframe_ordinal"]) == previous_ordinal
                and int(historical_row["pair_index"])
                == ordinal * (ordinal - 1) // 2 + previous_ordinal,
                "ordered pair indexing differs",
            )
            disposition_counter[str(historical_row["terminal_disposition"])] += 1
            if historical_row.get("mahalanobis") is not None:
                mahalanobis_max = max(
                    mahalanobis_max,
                    abs(float(historical_row["mahalanobis"]) - float(clean_row["mahalanobis"])),
                )
                factor_error_max = max(
                    factor_error_max,
                    abs(float(historical_row["factor_error"]) - float(clean_row["factor_error"])),
                )
        matcher_invocations += len(historical_current["matcher_invocations"])

        historical_path = historical_root / "currents" / "current_{:04d}".format(ordinal) / "candidate_paths.npz"
        clean_path = clean_root / "currents" / "current_{:04d}".format(ordinal) / "candidate_paths.npz"
        historical_arrays = _load_npz(historical_path)
        clean_arrays = _load_npz(clean_path)
        _require(set(historical_arrays) == set(clean_arrays), "candidate-path members differ")
        for name in exact_path_arrays:
            _require(
                np.array_equal(historical_arrays[name], clean_arrays[name]),
                "current {:04d} {} differs".format(ordinal, name),
            )
        for name in numeric_path_arrays:
            difference = _max_abs(
                historical_arrays[name], clean_arrays[name],
                "current {:04d} {}".format(ordinal, name),
            )
            _require(difference <= SCIENTIFIC_ATOL, "candidate path numeric difference exceeds tolerance")
            if name == "accumulated_covariances":
                candidate_covariance_max = max(candidate_covariance_max, difference)
            elif name == "covariance_determinants":
                determinant_max = max(determinant_max, difference)
            else:
                log_determinant_max = max(log_determinant_max, difference)
        candidate_path_byte_identical_count += int(_sha256(historical_path) == _sha256(clean_path))

        for accepted in historical_current["accepted_loops"]:
            clean_accepted = clean_current["accepted_loops"][0]
            accepted_records.append(
                (
                    accepted,
                    historical_root / "currents" / "current_{:04d}".format(ordinal) / str(accepted["artifact"]),
                    clean_root / "currents" / "current_{:04d}".format(ordinal) / str(clean_accepted["artifact"]),
                )
            )

    _require(sum(disposition_counter.values()) == EXPECTED_ORDERED_PAIR_COUNT, "ordered pair count differs")
    _require(dict(disposition_counter) == EXPECTED_DISPOSITIONS, "aggregate dispositions differ")
    _require(matcher_invocations == EXPECTED_MATCHER_INVOCATION_COUNT, "matcher count differs")
    _require(len(accepted_records) == 6, "accepted-loop population differs")

    checkpoint_results = []
    accepted_rows = []
    accepted_relative_pose_max = 0.0
    conditional_covariance_max = 0.0
    accepted_measurement_max = 0.0
    retained_ransac_max = 0.0
    accepted_artifact_byte_identical_count = 0
    exact_accepted_arrays = {
        "acceptance_id",
        "previous_frame_id",
        "current_frame_id",
        "previous_keyframe_ordinal",
        "current_keyframe_ordinal",
        "state_version",
        "frame_ids",
        "candidate_consensus_indices",
        "accepted_consensus_indices",
        "supporter_indices",
        "consensus_mapping",
        "consensus_supporter_mask",
        "rejected_consensus_indices",
        "rejected_reason_codes",
    }
    numeric_accepted_arrays = {
        "optimized_poses",
        "marginal_covariances",
        "initial_relative_pose",
        "optimized_relative_pose",
        "conditional_covariance",
        "previous_stereo_measurements",
        "current_stereo_measurements",
        "consensus_four_view_pixels",
        "ransac_best_rotations",
        "ransac_best_translations",
    }
    final_historical_arrays = None
    final_clean_arrays = None
    for index, ((accepted, historical_path, clean_path), expected) in enumerate(
        zip(accepted_records, EXPECTED_ACCEPTED_LOOPS), start=1
    ):
        observed = (
            int(accepted["acceptance_id"]),
            int(accepted["previous_frame_id"]),
            int(accepted["current_frame_id"]),
            int(accepted["previous_keyframe_ordinal"]),
            int(accepted["current_keyframe_ordinal"]),
            int(accepted["consensus_count"]),
            int(accepted["supporter_count"]),
            float(accepted["supporter_percentage"]),
        )
        _require(observed == expected, "accepted-loop identity/order differs")
        _require(
            accepted["ransac_terminal_status"] == "adaptive_completed"
            and accepted["factor_direction"] == "previous_FrameId_to_current_FrameId"
            and accepted["tangent_order"]
            == ["rot_x_rad", "rot_y_rad", "rot_z_rad", "trans_x_m", "trans_y_m", "trans_z_m"]
            and accepted["covariance_modified"] is False
            and int(accepted["state_version_before"]) == index - 1
            and int(accepted["state_version_after"]) == index
            and int(accepted["factor_count_after"]) == 356 + index
            and int(accepted["marginal_count"]) == EXPECTED_KEYFRAME_COUNT,
            "accepted-loop state/factor/covariance contract differs",
        )
        historical_arrays = _load_npz(historical_path)
        clean_arrays = _load_npz(clean_path)
        _require(set(historical_arrays) == set(clean_arrays), "accepted artifact members differ")
        for name in exact_accepted_arrays:
            _require(
                np.array_equal(historical_arrays[name], clean_arrays[name]),
                "acceptance {} {} differs".format(index, name),
            )
        differences: Dict[str, float] = {}
        for name in numeric_accepted_arrays:
            differences[name] = _max_abs(
                historical_arrays[name], clean_arrays[name],
                "acceptance {} {}".format(index, name),
            )
            _require(
                differences[name] <= SCIENTIFIC_ATOL,
                "acceptance {} {} exceeds tolerance".format(index, name),
            )
        pose_difference = differences["optimized_poses"]
        marginal_difference = differences["marginal_covariances"]
        relative_difference = max(
            differences["initial_relative_pose"], differences["optimized_relative_pose"]
        )
        accepted_relative_pose_max = max(accepted_relative_pose_max, relative_difference)
        conditional_covariance_max = max(
            conditional_covariance_max, differences["conditional_covariance"]
        )
        accepted_measurement_max = max(
            accepted_measurement_max,
            differences["previous_stereo_measurements"],
            differences["current_stereo_measurements"],
            differences["consensus_four_view_pixels"],
        )
        retained_ransac_max = max(
            retained_ransac_max,
            differences["ransac_best_rotations"],
            differences["ransac_best_translations"],
        )
        checkpoint_results.append(
            {
                "acceptance_id": index,
                "previous_frame_id": expected[1],
                "current_frame_id": expected[2],
                "pose_max_abs_difference": pose_difference,
                "pose_maximum_location": _maximum_location(
                    historical_arrays["optimized_poses"], clean_arrays["optimized_poses"]
                ),
                "marginal_max_abs_difference": marginal_difference,
                "marginal_maximum_location": _maximum_location(
                    historical_arrays["marginal_covariances"], clean_arrays["marginal_covariances"]
                ),
            }
        )
        accepted_rows.append(
            {
                "acceptance_id": index,
                "previous_frame_id": expected[1],
                "current_frame_id": expected[2],
                "previous_ordinal": expected[3],
                "current_ordinal": expected[4],
                "consensus_historical": expected[5],
                "consensus_clean": int(clean["currents"][expected[4]]["accepted_loops"][0]["consensus_count"]),
                "supporters_historical": expected[6],
                "supporters_clean": int(clean["currents"][expected[4]]["accepted_loops"][0]["supporter_count"]),
                "supporter_percentage_historical": expected[7],
                "supporter_percentage_clean": float(clean["currents"][expected[4]]["accepted_loops"][0]["supporter_percentage"]),
                "relative_pose_max_abs_difference": relative_difference,
                "conditional_covariance_max_abs_difference": differences["conditional_covariance"],
                "checkpoint_pose_max_abs_difference": pose_difference,
                "checkpoint_marginal_max_abs_difference": marginal_difference,
                "status": "pass",
            }
        )
        accepted_artifact_byte_identical_count += int(_sha256(historical_path) == _sha256(clean_path))
        final_historical_arrays = historical_arrays
        final_clean_arrays = clean_arrays

    _require(final_historical_arrays is not None and final_clean_arrays is not None, "final state missing")

    historical_aggregate = historical_root / "final" / "aggregate"
    clean_aggregate = clean_root / "final" / "aggregate"
    _compare_csv_excluding(
        historical_aggregate / "candidate_dispositions.csv",
        clean_aggregate / "candidate_dispositions.csv",
        (),
        "candidate dispositions",
    )
    _compare_csv_excluding(
        historical_aggregate / "matcher_invocations.csv",
        clean_aggregate / "matcher_invocations.csv",
        ("runtime_seconds",),
        "matcher invocations",
    )
    _compare_csv_excluding(
        historical_aggregate / "accepted_loops.csv",
        clean_aggregate / "accepted_loops.csv",
        ("artifact_path", "successful_match_artifact_path"),
        "accepted loops",
    )
    _compare_csv_excluding(
        historical_aggregate / "current_keyframes.csv",
        clean_aggregate / "current_keyframes.csv",
        ("runtime_seconds", "peak_rss_bytes"),
        "current keyframes",
    )

    historical_report = historical_root / "final" / "reports"
    clean_report = clean_root / "final" / "reports"
    report_files = (
        "data/trajectory_comparison.csv",
        "data/absolute_pose_errors.csv",
        "data/marginal_uncertainty_sizes.csv",
    )
    for relative in report_files:
        _require(
            _sha256(historical_report / relative) == _sha256(clean_report / relative),
            "report scientific CSV differs: {}".format(relative),
        )
    trajectory_fields = (
        "ground_truth_x_m", "ground_truth_y_m", "ground_truth_z_m",
        "with_loop_x_m", "with_loop_y_m", "with_loop_z_m",
        "without_loop_x_m", "without_loop_y_m", "without_loop_z_m",
    )
    final_trajectory_max = _numeric_csv_maximum(
        historical_report / "data" / "trajectory_comparison.csv",
        clean_report / "data" / "trajectory_comparison.csv",
        trajectory_fields,
        "trajectory comparison",
    )
    error_fields = (
        "with_loop_absolute_x_error_m", "with_loop_absolute_y_error_m",
        "with_loop_absolute_z_error_m", "with_loop_location_error_norm_m",
        "with_loop_rotation_error_deg", "without_loop_absolute_x_error_m",
        "without_loop_absolute_y_error_m", "without_loop_absolute_z_error_m",
        "without_loop_location_error_norm_m", "without_loop_rotation_error_deg",
    )
    report_error_max = _numeric_csv_maximum(
        historical_report / "data" / "absolute_pose_errors.csv",
        clean_report / "data" / "absolute_pose_errors.csv",
        error_fields,
        "absolute pose errors",
    )
    uncertainty_fields = (
        "with_loop_location_uncertainty_m", "with_loop_rotation_uncertainty_deg",
        "with_loop_rotation_uncertainty_rad", "without_loop_location_uncertainty_m",
        "without_loop_rotation_uncertainty_deg", "without_loop_rotation_uncertainty_rad",
    )
    report_uncertainty_max = _numeric_csv_maximum(
        historical_report / "data" / "marginal_uncertainty_sizes.csv",
        clean_report / "data" / "marginal_uncertainty_sizes.csv",
        uncertainty_fields,
        "marginal uncertainty",
    )
    historical_report_summary = _read_json(historical_report / "summary.json")
    clean_report_summary = _read_json(clean_report / "summary.json")
    _require(
        historical_report_summary["metrics"] == clean_report_summary["metrics"],
        "report summary metrics differ",
    )
    _verify_report_metrics(historical_root, final_historical_arrays["marginal_covariances"])
    _verify_report_metrics(clean_root, final_clean_arrays["marginal_covariances"])

    return {
        "accepted_rows": accepted_rows,
        "checkpoint_results": checkpoint_results,
        "discrete_gates": {
            "status": "pass",
            "keyframe_count": EXPECTED_KEYFRAME_COUNT,
            "frame_id_sequence_exact": True,
            "ordered_pair_count": EXPECTED_ORDERED_PAIR_COUNT,
            "pair_index_and_order_exact": True,
            "matcher_invocation_count": matcher_invocations,
            "matcher_schedule_non_runtime_exact": True,
            "candidate_dispositions_exact": True,
            "accepted_loop_order_exact": True,
            "state_version_progression": [0, 1, 2, 3, 4, 5, 6],
            "factor_progression": [356, 357, 358, 359, 360, 361, 362],
            "final_factor_count": 362,
            "final_marginal_count": EXPECTED_KEYFRAME_COUNT,
        },
        "aggregate_disposition_counts": dict(disposition_counter),
        "maximum_observed_differences": {
            "candidate_accumulated_covariance": candidate_covariance_max,
            "candidate_covariance_determinant": determinant_max,
            "candidate_covariance_log_determinant": log_determinant_max,
            "candidate_mahalanobis": mahalanobis_max,
            "candidate_factor_error": factor_error_max,
            "accepted_relative_poses": accepted_relative_pose_max,
            "accepted_conditional_covariance": conditional_covariance_max,
            "accepted_match_measurements_and_pixels": accepted_measurement_max,
            "retained_ransac_rotations_and_translations": retained_ransac_max,
            "final_trajectory_and_baseline_trajectory": final_trajectory_max,
            "absolute_error_metrics": report_error_max,
            "uncertainty_metrics": report_uncertainty_max,
        },
        "byte_identity_observations": {
            "candidate_path_npz": {
                "identical": candidate_path_byte_identical_count,
                "total": EXPECTED_KEYFRAME_COUNT,
            },
            "accepted_checkpoint_npz": {
                "identical": accepted_artifact_byte_identical_count,
                "total": 6,
            },
            "final_report_scientific_csv": {"identical": 3, "total": 3},
        },
        "runtime_difference_observations": {
            "current_runtime_fields_differing": current_runtime_difference_count,
            "current_peak_rss_fields_differing": current_rss_difference_count,
            "matcher_runtime_fields_differing": matcher_runtime_difference_count,
        },
    }


def _write_outputs(
    output_root: Path,
    historical: Mapping[str, object],
    clean: Mapping[str, object],
    comparison: Mapping[str, object],
) -> Mapping[str, object]:
    output_root = output_root.resolve()
    _require(not output_root.exists(), "comparison output root already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    expected_differences = [
        "historical versus clean E5/E6 root names and protected-input hashes",
        "E7 source hashes changed with the clean readability source layout",
        "current and matcher runtime fields plus current peak RSS fields",
        "accepted artifact_path and successful_match_artifact_path root prefixes",
        "provenance, seals, manifests, and hashes transitively covering those fields",
        "canonical command text and total invocation runtime/RSS bookkeeping",
        "report snapshot source paths/hashes and pending report-freeze traceability stage",
    ]
    summary = {
        "schema": "exercise7-clean-equivalence-summary-v1",
        "status": "pass",
        "historical_root": str(historical["root"]),
        "clean_root": str(clean["root"]),
        "clean_source_commit": CLEAN_SOURCE_COMMIT,
        "scientific_tolerance_absolute": SCIENTIFIC_ATOL,
        "internal_validation": {
            "historical": "pass",
            "clean": "pass",
            "sealed_current_count_each": EXPECTED_KEYFRAME_COUNT,
            "accepted_artifact_count_each": 6,
            "final_manifest_hashes_verified": True,
        },
        "discrete_gates": comparison["discrete_gates"],
        "aggregate_disposition_counts": comparison["aggregate_disposition_counts"],
        "accepted_loop_identity_order": [
            {
                "acceptance_id": row["acceptance_id"],
                "previous_frame_id": row["previous_frame_id"],
                "current_frame_id": row["current_frame_id"],
                "previous_ordinal": row["previous_ordinal"],
                "current_ordinal": row["current_ordinal"],
            }
            for row in comparison["accepted_rows"]
        ],
        "maximum_observed_differences": comparison["maximum_observed_differences"],
        "checkpoint_comparison": comparison["checkpoint_results"],
        "final_report_metric_equivalence": "pass",
        "byte_identity_observations": comparison["byte_identity_observations"],
        "runtime_difference_observations": comparison["runtime_difference_observations"],
        "expected_provenance_and_runtime_only_differences": expected_differences,
        "scientific_roots_modified": False,
        "historical_content_fingerprint_before_after": [
            historical["fingerprint"], historical["fingerprint"]
        ],
        "clean_content_fingerprint_before_after": [clean["fingerprint"], clean["fingerprint"]],
    }

    attempt = Path(
        tempfile.mkdtemp(prefix=".{}-attempt-".format(output_root.name), dir=str(output_root.parent))
    )
    try:
        summary_path = attempt / "summary.json"
        accepted_path = attempt / "accepted_loop_comparison.csv"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        fields = [
            "acceptance_id", "previous_frame_id", "current_frame_id",
            "previous_ordinal", "current_ordinal", "consensus_historical",
            "consensus_clean", "supporters_historical", "supporters_clean",
            "supporter_percentage_historical", "supporter_percentage_clean",
            "relative_pose_max_abs_difference",
            "conditional_covariance_max_abs_difference",
            "checkpoint_pose_max_abs_difference",
            "checkpoint_marginal_max_abs_difference", "status",
        ]
        with accepted_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(comparison["accepted_rows"])
        manifest = {
            "schema": "exercise7-clean-equivalence-manifest-v1",
            "status": "pass",
            "artifact_sha256": {
                "summary.json": _sha256(summary_path),
                "accepted_loop_comparison.csv": _sha256(accepted_path),
            },
        }
        with (attempt / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        _require(
            _content_fingerprint(historical["root"]) == historical["fingerprint"],
            "historical root changed during comparison",
        )
        _require(
            _content_fingerprint(clean["root"]) == clean["fingerprint"],
            "clean root changed during comparison",
        )
        os.replace(str(attempt), str(output_root))
        return manifest
    except BaseException:
        if attempt.exists():
            shutil.rmtree(str(attempt))
        raise


def validate_equivalence(historical_root: Path, clean_root: Path, output_root: Path) -> Mapping[str, object]:
    historical = _validate_root(historical_root)
    clean = _validate_root(clean_root)
    comparison = _compare_roots(historical, clean)
    manifest = _write_outputs(output_root, historical, clean, comparison)
    return {
        "status": "pass",
        "historical_internal_validation": "pass",
        "clean_internal_validation": "pass",
        "output_root": str(output_root.resolve()),
        "manifest": manifest,
        "maximum_observed_differences": comparison["maximum_observed_differences"],
        "checkpoint_comparison": comparison["checkpoint_results"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare stored historical and clean E7 evidence")
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("diagnostics/ex7_clean_equivalence")
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    arguments = _parser().parse_args(argv)
    result = validate_equivalence(
        arguments.historical_root, arguments.clean_root, arguments.output_root
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
