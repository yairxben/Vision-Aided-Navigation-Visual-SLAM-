"""Run the exact bounded E7-I03 Loop Closure integration gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Mapping, Sequence

import gtsam
import numpy as np

from exercise5_full_run import (
    sha256_file,
)
from exercise7_consensus_matching import (
    ADAPTIVE_COMPLETED,
    CANONICAL_SEED,
    DECLARED_CANDIDATE_PAIRS,
    MAX_HYPOTHESIS_ATTEMPTS,
    REPOSITORY_ROOT,
    evaluate_candidate_pair,
    validate_declared_candidate_pairs,
    validate_pair_evidence,
)
from exercise7_pose_graph_update import (
    apply_one_loop_factor,
    initial_loop_relative_pose,
    load_protected_e6_state,
)
from exercise7_small_bundle import (
    CURRENT_FRAME_ID,
    EXPECTED_CONSENSUS_COUNT,
    EXPECTED_SUPPORTER_COUNT,
    PREVIOUS_FRAME_ID,
    calibration_from_configuration,
    observations_from_pair_run,
    solve_small_bundle,
)
from PoseGraph import TANGENT_ORDER, camera_symbol


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex7_i03"
E7_I02_SUMMARY = REPOSITORY_ROOT / "diagnostics" / "ex7_i02" / "summary.json"
E7_I02_RESULTS = (
    REPOSITORY_ROOT / "diagnostics" / "ex7_i02" / "pair_results.csv"
)
E6_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex6_pose_graph"
E6_SUMMARY = E6_ROOT / "summary.json"
E6_VALUES = E6_ROOT / "pose_graph_values.npz"
E6_MARGINALS = E6_ROOT / "pose_graph_marginals.npz"
E5_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex5_final"
E5_MANIFEST = E5_ROOT / "manifest.json"
E5_CONFIGURATION = E5_ROOT / "configuration.json"
HISTORICAL_E5_CONFIGURATION_SHA256 = (
    "15c6430e397d7fe94ffe77d8f27fd8daa707449c6eb4c9227e345ca070b6e1e7"
)
CALIBRATION_PATH = (
    REPOSITORY_ROOT / "dataset" / "sequences" / "00" / "calib.txt"
)

EXPECTED_FIXED_SHA256 = {
    "diagnostics/ex7_i02/summary.json": (
        "42516565852821264eb37ec7a087c51e2ca1b5f477efcb7cc4c831ae3d94bdba"
    ),
    "diagnostics/ex7_i02/pair_results.csv": (
        "b80a6ef07e0f1482a6eb73579e5c0870a2fba2bc43ba3ee1792ae7b8b9933b91"
    ),
    "diagnostics/ex6_pose_graph/summary.json": (
        "12b469fc384f50843e6d5d731f6ab841f0de7b25fb65c07d24899041db97b468"
    ),
    "diagnostics/ex6_pose_graph/pose_graph_values.npz": (
        "be502673d8c73ccfaf757bc02cb6d4378e7f39de6f4e35f77ed40c465c164b54"
    ),
    "diagnostics/ex6_pose_graph/pose_graph_marginals.npz": (
        "1d5877564636d1e28bbb046bb7f64c1b25b3b3e182e27393fb2c04043de59da4"
    ),
    "diagnostics/ex5_final/manifest.json": (
        "030fe49447ed04cafa1dfacd5b7d0b620c799104c69190c959072b588dabc518"
    ),
    "diagnostics/ex5_final/configuration.json": (
        "03ebebdc3b59c7a4c05c6b275dd5060b76f69f2a03946e5597cd6f192a93afab"
    ),
    "dataset/sequences/00/calib.txt": (
        "7cfc9ab047da6bd829d669e31b2a0d72c278d084fa0c02f2f0320795d51a5d01"
    ),
    "PoseGraph.py": (
        "a19df30e3b60117fb53464b25c7dcc8bd902a0371fddadc52ba7a967593ca877"
    ),
    "exercise5_full_run.py": (
        "5f250d3744ae6a061ba6af6cfbccc00b391750bc6a76ce0eb66f10ab31fa50b9"
    ),
    "exercise5_bundle_window.py": (
        "59a19455c6b3d5f5e187fca8aecd4541aebabf3a843714940e21dd07dbd0ed65"
    ),
    "exercise5_geometry.py": (
        "5502e71ff256cd8b25adb58786974df69afa309d68222de5bd749fa1b14fb6c8"
    ),
}

REPROJECTION_FIELDS = (
    "supporter_ordinal",
    "consensus_index",
    "frame_id",
    "factor_index",
    "camera_key",
    "landmark_key",
    "measurement_u_left",
    "measurement_u_right",
    "measurement_v_mean",
    "initial_projection_u_left",
    "initial_projection_u_right",
    "initial_projection_v",
    "initial_residual_u_left",
    "initial_residual_u_right",
    "initial_residual_v",
    "initial_factor_error",
    "final_projection_u_left",
    "final_projection_u_right",
    "final_projection_v",
    "final_residual_u_left",
    "final_residual_u_right",
    "final_residual_v",
    "final_factor_error",
)


class E7I03ValidationError(RuntimeError):
    """Raised when the canonical integration gate cannot pass."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise E7I03ValidationError(message)


def _relative(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPOSITORY_ROOT.resolve()))


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _fixed_hashes() -> Dict[str, str]:
    observed = {}
    for relative, expected in EXPECTED_FIXED_SHA256.items():
        path = REPOSITORY_ROOT / relative
        _require(path.is_file(), "protected input is missing: {}".format(relative))
        digest = sha256_file(path)
        _require(
            digest == expected,
            "protected input identity differs: {}".format(relative),
        )
        observed[relative] = digest

    e7_summary = _read_json(E7_I02_SUMMARY)
    protected = e7_summary.get(
        "protected_input_sha256_before_and_after", {}
    )
    _require(
        isinstance(protected, dict) and protected,
        "E7-I02 protected-source identity evidence is missing",
    )
    for relative, expected in protected.items():
        path = REPOSITORY_ROOT / relative
        _require(
            path.is_file() and sha256_file(path) == expected,
            "protected E7 upstream identity differs: {}".format(relative),
        )
        observed[relative] = expected
    return dict(sorted(observed.items()))


def _validate_exact_pair_artifacts() -> Mapping[str, object]:
    summary = _read_json(E7_I02_SUMMARY)
    _require(
        summary.get("schema") == "exercise7-consensus-matching-probe-v1"
        and summary.get("milestone") == "E7-I02"
        and summary.get("status") == "pass",
        "protected E7-I02 summary contract differs",
    )
    records = [
        record
        for record in summary.get("pair_results", [])
        if record.get("pair_index") == 3
    ]
    _require(len(records) == 1, "protected E7-I02 pair index 3 is missing")
    record = records[0]
    _require(
        record.get("previous_frame_id") == PREVIOUS_FRAME_ID
        and record.get("current_frame_id") == CURRENT_FRAME_ID
        and record.get("four_view_consensus_count")
        == EXPECTED_CONSENSUS_COUNT
        and record.get("final_supporter_count")
        == EXPECTED_SUPPORTER_COUNT
        and record.get("terminal_status") == ADAPTIVE_COMPLETED,
        "protected E7-I02 pair result differs from 356->3359, 515/461",
    )
    with E7_I02_RESULTS.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["pair_index"]) == 3
        ]
    _require(
        len(rows) == 1
        and int(rows[0]["previous_frame_id"]) == PREVIOUS_FRAME_ID
        and int(rows[0]["current_frame_id"]) == CURRENT_FRAME_ID
        and int(rows[0]["four_view_consensus_count"])
        == EXPECTED_CONSENSUS_COUNT
        and int(rows[0]["final_supporter_count"])
        == EXPECTED_SUPPORTER_COUNT
        and rows[0]["terminal_status"] == ADAPTIVE_COMPLETED,
        "protected E7-I02 CSV pair result differs",
    )
    return record


def _pose_record(pose: gtsam.Pose3) -> Mapping[str, object]:
    rotation_vector = np.asarray(
        gtsam.Rot3.Logmap(pose.rotation()), dtype=float
    ).reshape(3)
    translation = np.asarray(pose.translation(), dtype=float).reshape(3)
    return {
        "matrix": np.asarray(pose.matrix(), dtype=float),
        "rotation_vector_rad": rotation_vector,
        "rotation_norm_rad": float(np.linalg.norm(rotation_vector)),
        "translation_m": translation,
        "translation_norm_m": float(np.linalg.norm(translation)),
    }


def _reprojection_summary(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    initial = np.asarray(
        [row["initial_factor_error"] for row in rows], dtype=float
    )
    final = np.asarray(
        [row["final_factor_error"] for row in rows], dtype=float
    )
    _require(
        initial.size > 0
        and np.all(np.isfinite(initial))
        and np.all(np.isfinite(final)),
        "reprojection factor errors are missing or nonfinite",
    )
    return {
        "row_count": int(initial.size),
        "initial_factor_error_sum": float(np.sum(initial)),
        "initial_factor_error_mean": float(np.mean(initial)),
        "initial_factor_error_maximum": float(np.max(initial)),
        "final_factor_error_sum": float(np.sum(final)),
        "final_factor_error_mean": float(np.mean(final)),
        "final_factor_error_maximum": float(np.max(final)),
    }


def _write_reprojection_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPROJECTION_FIELDS)
        writer.writeheader()
        for row in rows:
            measurement = row["measurement"]
            initial_projection = row["initial_projection"]
            initial_residual = row["initial_residual"]
            final_projection = row["final_projection"]
            final_residual = row["final_residual"]
            writer.writerow(
                {
                    "supporter_ordinal": row["supporter_ordinal"],
                    "consensus_index": row["consensus_index"],
                    "frame_id": row["frame_id"],
                    "factor_index": row["factor_index"],
                    "camera_key": row["camera_key"],
                    "landmark_key": row["landmark_key"],
                    "measurement_u_left": measurement[0],
                    "measurement_u_right": measurement[1],
                    "measurement_v_mean": measurement[2],
                    "initial_projection_u_left": initial_projection[0],
                    "initial_projection_u_right": initial_projection[1],
                    "initial_projection_v": initial_projection[2],
                    "initial_residual_u_left": initial_residual[0],
                    "initial_residual_u_right": initial_residual[1],
                    "initial_residual_v": initial_residual[2],
                    "initial_factor_error": row["initial_factor_error"],
                    "final_projection_u_left": final_projection[0],
                    "final_projection_u_right": final_projection[1],
                    "final_projection_v": final_projection[2],
                    "final_residual_u_left": final_residual[0],
                    "final_residual_u_right": final_residual[1],
                    "final_residual_v": final_residual[2],
                    "final_factor_error": row["final_factor_error"],
                }
            )


def _validate_output_root(output_root: Path) -> Path:
    selected = Path(output_root).resolve()
    _require(
        selected == DEFAULT_OUTPUT_ROOT.resolve()
        or REPOSITORY_ROOT.resolve() not in selected.parents,
        "non-default E7-I03 output must be outside the repository",
    )
    if selected.exists():
        raise FileExistsError(
            "E7-I03 output root already exists: {}".format(selected)
        )
    selected.parent.mkdir(parents=True, exist_ok=True)
    return selected


def _reload_artifacts(root: Path) -> Mapping[str, object]:
    expected = {
        "summary.json",
        "small_bundle.npz",
        "small_bundle_reprojection.csv",
        "loop_factor.json",
        "pose_graph_update.npz",
    }
    _require(
        {path.name for path in root.iterdir() if path.is_file()} == expected,
        "E7-I03 artifact population differs",
    )
    summary = _read_json(root / "summary.json")
    loop = _read_json(root / "loop_factor.json")
    with np.load(root / "small_bundle.npz", allow_pickle=False) as data:
        _require(
            data["conditional_covariance"].shape == (6, 6)
            and data["optimized_camera_poses"].shape == (2, 4, 4),
            "small-Bundle artifact reload failed",
        )
    with np.load(root / "pose_graph_update.npz", allow_pickle=False) as data:
        _require(
            data["frame_ids"].shape == (356,)
            and data["optimized_poses"].shape == (356, 4, 4)
            and data["marginal_covariances"].shape == (356, 6, 6),
            "Pose Graph artifact reload failed",
        )
    with (root / "small_bundle_reprojection.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    _require(
        summary.get("status") == "pass"
        and loop.get("factor_count_after_append") == 357
        and row_count
        == 2 * summary["small_bundle"]["accepted_landmark_count"],
        "E7-I03 JSON/CSV artifact reload failed",
    )
    return {
        "success": True,
        "artifact_count": len(expected),
        "reprojection_row_count": row_count,
    }


def run_validation(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> Mapping[str, object]:
    """Run exactly one 356->3359 integration gate and retain compact evidence."""

    started = time.perf_counter()
    selected_output = _validate_output_root(output_root)
    fixed_hashes_before = _fixed_hashes()
    protected_pair_record = _validate_exact_pair_artifacts()
    configuration_snapshot = _read_json(E5_CONFIGURATION)
    _require(
        configuration_snapshot.get("schema")
        == "exercise5-configuration-snapshot-v1"
        and isinstance(configuration_snapshot.get("configuration"), dict),
        "historical E5 configuration snapshot differs",
    )
    configuration = configuration_snapshot["configuration"]
    configuration_sha256 = configuration_snapshot["source_sha256"]
    _require(
        configuration_sha256
        == HISTORICAL_E5_CONFIGURATION_SHA256,
        "loaded E5 configuration identity differs",
    )
    manifest = _read_json(E5_MANIFEST)
    _require(
        manifest.get("status") == "complete-and-sealed"
        and manifest.get("scope") == "canonical"
        and manifest.get("configuration_sha256") == configuration_sha256,
        "canonical E5 manifest contract differs",
    )
    calibration = calibration_from_configuration(
        configuration, CALIBRATION_PATH
    )

    pairs = validate_declared_candidate_pairs()
    _require(
        pairs == DECLARED_CANDIDATE_PAIRS and len(pairs) == 8,
        "declared E7-I02 pair contract differs",
    )
    pair = pairs[3]
    pair_run = evaluate_candidate_pair(
        pair,
        seed=CANONICAL_SEED,
        maximum_attempts=MAX_HYPOTHESIS_ATTEMPTS,
    )
    validate_pair_evidence(pair_run.evidence)
    _require(
        pair_run.evidence.ransac.status == ADAPTIVE_COMPLETED
        and pair_run.evidence.counts.four_view_consensus_count
        == EXPECTED_CONSENSUS_COUNT
        and pair_run.evidence.counts.final_supporter_count
        == EXPECTED_SUPPORTER_COUNT,
        "canonical E7-I02 matching replay did not reproduce 515/461",
    )
    observations = observations_from_pair_run(pair_run)

    e6_state = load_protected_e6_state(
        summary_path=E6_SUMMARY,
        values_path=E6_VALUES,
        marginals_path=E6_MARGINALS,
        e5_root=E5_ROOT,
    )
    bundle_initial_relative = initial_loop_relative_pose(e6_state)
    bundle = solve_small_bundle(
        observations=observations,
        initial_relative_pose=bundle_initial_relative,
        calibration=calibration,
        configuration=configuration,
    )
    pose_update = apply_one_loop_factor(
        state=e6_state,
        relative_pose=bundle.optimized_relative_pose,
        conditional_covariance=bundle.conditional_covariance,
    )
    _require(
        pose_update.graph.size() == 357,
        "canonical augmented Pose Graph factor count is not 357",
    )
    _require(
        pose_update.marginal_covariances.shape == (356, 6, 6),
        "canonical augmented Pose Graph lacks all 356 Marginals",
    )

    temporary = Path(
        tempfile.mkdtemp(
            prefix=".ex7_i03.", dir=str(selected_output.parent)
        )
    )
    try:
        candidate_measurements_previous = np.asarray(
            [item.previous_measurement for item in observations],
            dtype="<f8",
        )
        candidate_measurements_current = np.asarray(
            [item.current_measurement for item in observations],
            dtype="<f8",
        )
        accepted_indices = np.asarray(
            [
                item.consensus_index
                for item in bundle.accepted_observations
            ],
            dtype="<i8",
        )
        rejected_indices = np.asarray(
            sorted(bundle.rejection_reason_by_consensus_index),
            dtype="<i8",
        )
        previous_key = int(camera_symbol(PREVIOUS_FRAME_ID))
        current_key = int(camera_symbol(CURRENT_FRAME_ID))
        np.savez(
            temporary / "small_bundle.npz",
            candidate_consensus_indices=np.asarray(
                [item.consensus_index for item in observations],
                dtype="<i8",
            ),
            accepted_consensus_indices=accepted_indices,
            rejected_consensus_indices=rejected_indices,
            candidate_previous_measurements=candidate_measurements_previous,
            candidate_current_measurements=candidate_measurements_current,
            initial_landmarks=np.asarray(
                bundle.initial_landmarks, dtype="<f8"
            ),
            initial_camera_poses=np.asarray(
                [
                    bundle.initial_values.atPose3(previous_key).matrix(),
                    bundle.initial_values.atPose3(current_key).matrix(),
                ],
                dtype="<f8",
            ),
            optimized_camera_poses=np.asarray(
                [
                    bundle.optimized_values.atPose3(previous_key).matrix(),
                    bundle.optimized_values.atPose3(current_key).matrix(),
                ],
                dtype="<f8",
            ),
            initial_relative_pose=np.asarray(
                bundle.initial_relative_pose.matrix(), dtype="<f8"
            ),
            optimized_relative_pose=np.asarray(
                bundle.optimized_relative_pose.matrix(), dtype="<f8"
            ),
            joint_covariance=np.asarray(
                bundle.joint_covariance, dtype="<f8"
            ),
            joint_information=np.asarray(
                bundle.joint_information, dtype="<f8"
            ),
            conditional_covariance=np.asarray(
                bundle.conditional_covariance, dtype="<f8"
            ),
            conditional_covariance_eigenvalues=np.asarray(
                bundle.covariance_eigenvalues, dtype="<f8"
            ),
        )
        _write_reprojection_csv(
            temporary / "small_bundle_reprojection.csv",
            bundle.reprojection_rows,
        )
        np.savez(
            temporary / "pose_graph_update.npz",
            frame_ids=np.asarray(
                e6_state.inputs.keyframe_ids, dtype="<i8"
            ),
            pre_loop_optimized_poses=np.asarray(
                pose_update.pre_loop_pose_matrices, dtype="<f8"
            ),
            optimized_poses=np.asarray(
                pose_update.optimized_pose_matrices, dtype="<f8"
            ),
            marginal_covariances=np.asarray(
                pose_update.marginal_covariances, dtype="<f8"
            ),
        )
        loop_record = {
            "schema": "exercise7-i03-loop-factor-v1",
            "start_frame_id": PREVIOUS_FRAME_ID,
            "end_frame_id": CURRENT_FRAME_ID,
            "start_camera_key": previous_key,
            "end_camera_key": current_key,
            "direction": "T_C356_from_C3359",
            "relative_pose": _pose_record(
                bundle.optimized_relative_pose
            ),
            "conditional_end_given_start_covariance": (
                bundle.conditional_covariance
            ),
            "tangent_order": list(TANGENT_ORDER),
            "covariance_repair_applied": False,
            "local_bundle_standalone_factor_error": (
                bundle.standalone_factor_error
            ),
            "global_pre_optimization_factor_error": (
                pose_update.loop_factor_pre_optimization_error
            ),
            "global_post_optimization_factor_error": (
                pose_update.loop_factor_post_optimization_error
            ),
            "factor_count_before_append": 356,
            "factor_count_after_append": 357,
        }
        _save_json(temporary / "loop_factor.json", loop_record)

        selected_movements = {
            str(frame_id): pose_update.movement_by_frame_id[frame_id]
            for frame_id in (0, PREVIOUS_FRAME_ID, CURRENT_FRAME_ID)
        }
        summary = {
            "schema": "exercise7-i03-integration-gate-v1",
            "schema_version": 1,
            "milestone": "E7-I03",
            "status": "pass",
            "scope": (
                "one validated 356->3359 loop; bounded integration gate only"
            ),
            "protected_inputs": {
                "sha256": fixed_hashes_before,
                "e7_i02_pair_record": protected_pair_record,
                "e5_configuration_sha256": configuration_sha256,
                "e6_fresh_vs_protected_pose_max_abs": (
                    e6_state.fresh_vs_protected_pose_max_abs
                ),
            },
            "matching_replay": {
                "pair_index": pair.pair_index,
                "previous_frame_id": pair.previous_frame_id,
                "current_frame_id": pair.current_frame_id,
                "seed": CANONICAL_SEED,
                "maximum_hypothesis_attempts": MAX_HYPOTHESIS_ATTEMPTS,
                "terminal_status": pair_run.evidence.ransac.status,
                "hypothesis_attempt_count": (
                    pair_run.evidence.ransac.total_hypothesis_attempt_count
                ),
                "four_view_consensus_count": (
                    pair_run.evidence.counts.four_view_consensus_count
                ),
                "final_supporter_count": (
                    pair_run.evidence.counts.final_supporter_count
                ),
            },
            "small_bundle": {
                "camera_frame_ids": [
                    PREVIOUS_FRAME_ID,
                    CURRENT_FRAME_ID,
                ],
                "camera_keys": [previous_key, current_key],
                "landmark_symbol": "q",
                "candidate_supporter_count": len(
                    bundle.candidate_observations
                ),
                "accepted_landmark_count": len(
                    bundle.accepted_observations
                ),
                "rejected_landmark_count": len(
                    bundle.rejection_reason_by_consensus_index
                ),
                "rejection_reason_counts": (
                    bundle.rejection_reason_counts
                ),
                "final_depth_rebuild_count": (
                    bundle.final_depth_rebuild_count
                ),
                "final_depth_rebuild_history": (
                    bundle.final_depth_rebuild_history
                ),
                "variable_count": int(bundle.initial_values.size()),
                "factor_count": int(bundle.graph.size()),
                "prior_factor_count": 1,
                "generic_stereo_factor_count": (
                    2 * len(bundle.accepted_observations)
                ),
                "between_factor_count": 0,
                "initial_error": bundle.initial_error,
                "first_optimizer_error": bundle.first_optimizer_error,
                "first_optimizer_anchor_identity_max_abs": (
                    bundle.first_optimizer_anchor_identity_max_abs
                ),
                "second_optimizer_error": bundle.second_optimizer_error,
                "second_optimizer_anchor_identity_max_abs": (
                    bundle.second_optimizer_anchor_identity_max_abs
                ),
                "gauge_normalization": {
                    "applied": True,
                    "optimizer_passes": 2,
                    "operation": (
                        "c356->Identity; c3359->c356.between(c3359); "
                        "landmarks->c356.transformTo"
                    ),
                    "factor_or_noise_changed": False,
                    "error_after_normalization": (
                        bundle.gauge_normalized_error
                    ),
                },
                "final_error": bundle.final_error,
                "local_anchor_identity_max_abs": (
                    bundle.local_anchor_identity_max_abs
                ),
                "initial_relative_pose": _pose_record(
                    bundle.initial_relative_pose
                ),
                "optimized_relative_pose": _pose_record(
                    bundle.optimized_relative_pose
                ),
                "direction": {
                    "name": "T_C356_from_C3359",
                    "construction": (
                        "optimized_c356.between(optimized_c3359)"
                    ),
                    "composition_max_abs": (
                        bundle.direction_composition_max_abs
                    ),
                    "point_mapping_max_abs_m": (
                        bundle.direction_point_mapping_max_abs
                    ),
                    "inverse_point_mapping_rmse_m_diagnostic": (
                        bundle.inverse_point_mapping_rmse_m
                    ),
                },
                "reprojection": _reprojection_summary(
                    bundle.reprojection_rows
                ),
            },
            "conditional_covariance": {
                "shape": list(bundle.conditional_covariance.shape),
                "eigenvalues": bundle.covariance_eigenvalues,
                "minimum_eigenvalue": float(
                    np.min(bundle.covariance_eigenvalues)
                ),
                "maximum_eigenvalue": float(
                    np.max(bundle.covariance_eigenvalues)
                ),
                "validation": bundle.covariance_diagnostics,
                "standalone_factor_error": (
                    bundle.standalone_factor_error
                ),
                "standalone_residual_max_abs": (
                    bundle.standalone_residual_max_abs
                ),
                "standalone_mahalanobis_crosscheck_abs": (
                    bundle.standalone_mahalanobis_crosscheck_abs
                ),
                "direct_gaussian_covariance": True,
                "repair_applied": False,
            },
            "pose_graph_update": {
                "node_count": 356,
                "factor_count": int(pose_update.graph.size()),
                "factor_composition": {
                    "prior": 1,
                    "consecutive_between": 355,
                    "loop_between": 1,
                },
                "original_no_loop_graph_error_diagnostic": (
                    pose_update.original_graph_error
                ),
                "augmented_pre_optimization_error": (
                    pose_update.augmented_pre_optimization_error
                ),
                "raw_optimizer_error": (
                    pose_update.raw_optimizer_error
                ),
                "raw_optimizer_anchor_identity_max_abs": (
                    pose_update.raw_optimizer_anchor_identity_max_abs
                ),
                "gauge_normalization": {
                    "applied": True,
                    "operation": (
                        "all poses -> raw_c0.between(raw_pose)"
                    ),
                    "factor_or_noise_changed": False,
                },
                "augmented_post_optimization_error": (
                    pose_update.augmented_post_optimization_error
                ),
                "loop_factor_pre_optimization_error": (
                    pose_update.loop_factor_pre_optimization_error
                ),
                "loop_factor_post_optimization_error": (
                    pose_update.loop_factor_post_optimization_error
                ),
                "global_anchor_identity_max_abs": (
                    pose_update.global_anchor_identity_max_abs
                ),
                "selected_pose_movements": selected_movements,
                "maximum_translation_m": (
                    pose_update.maximum_translation_m
                ),
                "maximum_translation_frame_id": (
                    pose_update.maximum_translation_frame_id
                ),
                "maximum_rotation_rad": pose_update.maximum_rotation_rad,
                "maximum_rotation_frame_id": (
                    pose_update.maximum_rotation_frame_id
                ),
                "marginals": {
                    "construction": (
                        "gtsam.Marginals(augmented_graph, optimized_values)"
                    ),
                    "count": int(
                        pose_update.marginal_covariances.shape[0]
                    ),
                    "all_finite": bool(
                        np.all(
                            np.isfinite(
                                pose_update.marginal_covariances
                            )
                        )
                    ),
                    "symmetry_max_abs": (
                        pose_update.marginal_symmetry_max_abs
                    ),
                    "minimum_eigenvalue_diagnostic": (
                        pose_update.marginal_minimum_eigenvalue
                    ),
                    "raw_covariances_saved": True,
                },
            },
            "acceptance_policy": {
                "trajectory_improvement_required": False,
                "ground_truth_metrics_computed": False,
                "large_finite_non_anchor_movement_is_diagnostic": True,
                "e7_i04_started": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "artifacts": [
                "summary.json",
                "small_bundle.npz",
                "small_bundle_reprojection.csv",
                "loop_factor.json",
                "pose_graph_update.npz",
            ],
        }
        _save_json(temporary / "summary.json", summary)
        reload_validation = _reload_artifacts(temporary)
        summary["artifact_reload_validation"] = reload_validation
        _save_json(temporary / "summary.json", summary)

        fixed_hashes_after = _fixed_hashes()
        _require(
            fixed_hashes_after == fixed_hashes_before,
            "a protected upstream source or artifact changed",
        )
        os.replace(str(temporary), str(selected_output))
        return summary
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded E7-I03 356->3359 integration gate"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="new output directory (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_validation(output_root=arguments.output_root)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
