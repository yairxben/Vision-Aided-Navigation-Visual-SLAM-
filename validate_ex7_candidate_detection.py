"""Generate the bounded E7-I01 candidate-detection evidence.

The probe evaluates every previous keyframe for three deterministic current
keyframes: nearest integer one-third, two-thirds, and final canonical
ordinals.  It applies no temporal exclusion and no Mahalanobis threshold.
"""

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
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from PoseGraph import (
    CONDITIONAL_COVARIANCE_FIELD,
    EDGE_DIRECTION,
    TANGENT_ORDER,
)
from exercise7_candidate_detection import (
    COURSE_HEURISTIC_POLICY,
    CandidateScore,
    CovarianceGraph,
    build_optimized_values,
    deterministic_probe_current_ordinals,
    load_candidate_detection_inputs,
    score_previous_keyframes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex7_i01"
HISTORICAL_E5_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex5_final"
HISTORICAL_E6_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex6_pose_graph"
PROTECTED_ARTIFACT_ROOTS = (
    REPOSITORY_ROOT / "diagnostics" / "ex5_final",
    REPOSITORY_ROOT / "diagnostics" / "ex5_track_survival_082",
    REPOSITORY_ROOT / "diagnostics" / "ex6_pose_graph",
    REPOSITORY_ROOT / "project_submission" / "bundle",
)
CSV_FIELDS = (
    "row_id",
    "current_keyframe_ordinal",
    "current_frame_id",
    "rank_within_current",
    "previous_keyframe_ordinal",
    "previous_frame_id",
    "path_edge_count",
    "path_covariance_determinant",
    "path_covariance_log_determinant",
    "factor_error",
    "mahalanobis_squared",
    "mahalanobis",
    "path_policy",
)


def _save_json(path: Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _relative_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _validate_output_root(output_root: Path) -> Path:
    selected = Path(output_root).resolve()
    for protected in PROTECTED_ARTIFACT_ROOTS:
        protected = protected.resolve()
        if selected == protected or protected in selected.parents:
            raise ValueError(
                "E7-I01 output must not be inside a protected artifact root"
            )
    if selected.exists():
        raise FileExistsError(
            "E7-I01 output root already exists: {}".format(selected)
        )
    selected.parent.mkdir(parents=True, exist_ok=True)
    return selected


def _numeric_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("summary values must be a nonempty finite vector")
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _current_result_summary(
    current_ordinal: int,
    current_frame_id: int,
    ranked_scores: Sequence[CandidateScore],
) -> Dict[str, object]:
    if len(ranked_scores) != current_ordinal:
        raise RuntimeError(
            "all previous keyframes must be scored for each probe current"
        )
    top = []
    for rank, score in enumerate(ranked_scores[:5], start=1):
        top.append(
            {
                "rank": rank,
                "previous_keyframe_ordinal": (
                    score.previous_keyframe_ordinal
                ),
                "previous_frame_id": score.previous_frame_id,
                "path_edge_count": len(score.path.edge_indices),
                "factor_error": score.factor_error,
                "mahalanobis_squared": score.mahalanobis_squared,
                "mahalanobis": score.mahalanobis,
            }
        )
    return {
        "current_keyframe_ordinal": current_ordinal,
        "current_frame_id": current_frame_id,
        "scored_previous_keyframe_count": len(ranked_scores),
        "factor_error": _numeric_summary(
            np.asarray([score.factor_error for score in ranked_scores])
        ),
        "mahalanobis_squared": _numeric_summary(
            np.asarray(
                [score.mahalanobis_squared for score in ranked_scores]
            )
        ),
        "mahalanobis": _numeric_summary(
            np.asarray([score.mahalanobis for score in ranked_scores])
        ),
        "path_covariance_determinant": _numeric_summary(
            np.asarray(
                [
                    score.path.covariance_determinant
                    for score in ranked_scores
                ]
            )
        ),
        "top_five_unfiltered_scores": top,
    }


def _write_candidate_csv(
    path: Path,
    ranked_groups: Sequence[
        Tuple[int, int, Sequence[CandidateScore]]
    ],
) -> List[Tuple[int, int, CandidateScore]]:
    aligned_rows = []
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        row_id = 0
        for current_ordinal, current_frame_id, ranked_scores in ranked_groups:
            for rank, score in enumerate(ranked_scores, start=1):
                if score.current_keyframe_ordinal != current_ordinal:
                    raise RuntimeError("score current ordinal differs")
                if score.current_frame_id != current_frame_id:
                    raise RuntimeError("score current FrameId differs")
                writer.writerow(
                    {
                        "row_id": row_id,
                        "current_keyframe_ordinal": current_ordinal,
                        "current_frame_id": current_frame_id,
                        "rank_within_current": rank,
                        "previous_keyframe_ordinal": (
                            score.previous_keyframe_ordinal
                        ),
                        "previous_frame_id": score.previous_frame_id,
                        "path_edge_count": len(score.path.edge_indices),
                        "path_covariance_determinant": (
                            score.path.covariance_determinant
                        ),
                        "path_covariance_log_determinant": (
                            score.path.covariance_log_determinant
                        ),
                        "factor_error": score.factor_error,
                        "mahalanobis_squared": (
                            score.mahalanobis_squared
                        ),
                        "mahalanobis": score.mahalanobis,
                        "path_policy": score.path.policy,
                    }
                )
                aligned_rows.append((rank, row_id, score))
                row_id += 1
    return aligned_rows


def _write_path_npz(
    path: Path,
    aligned_rows: Sequence[Tuple[int, int, CandidateScore]],
) -> None:
    row_ids = []
    ranks = []
    previous_ordinals = []
    current_ordinals = []
    previous_frame_ids = []
    current_frame_ids = []
    factor_errors = []
    mahalanobis_squared = []
    mahalanobis = []
    determinants = []
    log_determinants = []
    path_covariances = []
    path_frame_id_offsets = [0]
    path_frame_ids_flat = []
    path_edge_index_offsets = [0]
    path_edge_indices_flat = []

    for rank, row_id, score in aligned_rows:
        row_ids.append(row_id)
        ranks.append(rank)
        previous_ordinals.append(score.previous_keyframe_ordinal)
        current_ordinals.append(score.current_keyframe_ordinal)
        previous_frame_ids.append(score.previous_frame_id)
        current_frame_ids.append(score.current_frame_id)
        factor_errors.append(score.factor_error)
        mahalanobis_squared.append(score.mahalanobis_squared)
        mahalanobis.append(score.mahalanobis)
        determinants.append(score.path.covariance_determinant)
        log_determinants.append(
            score.path.covariance_log_determinant
        )
        path_covariances.append(score.path.covariance)
        path_frame_ids_flat.extend(score.path.frame_ids)
        path_frame_id_offsets.append(len(path_frame_ids_flat))
        path_edge_indices_flat.extend(score.path.edge_indices)
        path_edge_index_offsets.append(len(path_edge_indices_flat))

    np.savez(
        path,
        row_ids=np.asarray(row_ids, dtype="<i8"),
        ranks_within_current=np.asarray(ranks, dtype="<i8"),
        previous_keyframe_ordinals=np.asarray(
            previous_ordinals, dtype="<i8"
        ),
        current_keyframe_ordinals=np.asarray(
            current_ordinals, dtype="<i8"
        ),
        previous_frame_ids=np.asarray(previous_frame_ids, dtype="<i8"),
        current_frame_ids=np.asarray(current_frame_ids, dtype="<i8"),
        factor_errors=np.asarray(factor_errors, dtype="<f8"),
        mahalanobis_squared=np.asarray(
            mahalanobis_squared, dtype="<f8"
        ),
        mahalanobis=np.asarray(mahalanobis, dtype="<f8"),
        path_covariance_determinants=np.asarray(
            determinants, dtype="<f8"
        ),
        path_covariance_log_determinants=np.asarray(
            log_determinants, dtype="<f8"
        ),
        path_covariances=np.asarray(path_covariances, dtype="<f8"),
        path_frame_id_offsets=np.asarray(
            path_frame_id_offsets, dtype="<i8"
        ),
        path_frame_ids_flat=np.asarray(path_frame_ids_flat, dtype="<i8"),
        path_edge_index_offsets=np.asarray(
            path_edge_index_offsets, dtype="<i8"
        ),
        path_edge_indices_flat=np.asarray(
            path_edge_indices_flat, dtype="<i8"
        ),
    )


def _reload_artifacts(
    csv_path: Path, npz_path: Path, *, expected_row_count: int
) -> Mapping[str, object]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != expected_row_count:
        raise RuntimeError("candidate score CSV row count differs")
    if [int(row["row_id"]) for row in csv_rows] != list(
        range(expected_row_count)
    ):
        raise RuntimeError("candidate score CSV row ordering differs")
    if any(
        row["path_policy"] != COURSE_HEURISTIC_POLICY for row in csv_rows
    ):
        raise RuntimeError("candidate score CSV policy differs")

    expected_members = {
        "row_ids",
        "ranks_within_current",
        "previous_keyframe_ordinals",
        "current_keyframe_ordinals",
        "previous_frame_ids",
        "current_frame_ids",
        "factor_errors",
        "mahalanobis_squared",
        "mahalanobis",
        "path_covariance_determinants",
        "path_covariance_log_determinants",
        "path_covariances",
        "path_frame_id_offsets",
        "path_frame_ids_flat",
        "path_edge_index_offsets",
        "path_edge_indices_flat",
    }
    with np.load(npz_path, allow_pickle=False) as data:
        if set(data.files) != expected_members:
            raise RuntimeError("candidate path NPZ members differ")
        if not np.array_equal(
            data["row_ids"], np.arange(expected_row_count)
        ):
            raise RuntimeError("candidate path NPZ row ordering differs")
        if data["path_covariances"].shape != (
            expected_row_count,
            6,
            6,
        ):
            raise RuntimeError("candidate path covariance shape differs")
        for member in (
            "factor_errors",
            "mahalanobis_squared",
            "mahalanobis",
            "path_covariance_determinants",
            "path_covariance_log_determinants",
            "path_covariances",
        ):
            if not np.all(np.isfinite(data[member])):
                raise RuntimeError(
                    "{} contains nonfinite values".format(member)
                )
        if data["path_frame_id_offsets"].shape != (
            expected_row_count + 1,
        ):
            raise RuntimeError("path FrameId offsets shape differs")
        if data["path_edge_index_offsets"].shape != (
            expected_row_count + 1,
        ):
            raise RuntimeError("path edge offsets shape differs")
        if (
            int(data["path_frame_id_offsets"][-1])
            != int(data["path_frame_ids_flat"].size)
            or int(data["path_edge_index_offsets"][-1])
            != int(data["path_edge_indices_flat"].size)
        ):
            raise RuntimeError("flattened path offsets are inconsistent")
    return {
        "success": True,
        "candidate_scores_csv_rows": expected_row_count,
        "candidate_paths_npz_rows": expected_row_count,
    }


def run_bounded_probe(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, object]:
    """Run the three-current E7-I01 probe and seal no upstream artifact."""

    started = time.perf_counter()
    selected_output_root = _validate_output_root(output_root)
    inputs = load_candidate_detection_inputs(
        e6_root=HISTORICAL_E6_ROOT,
        e5_root=HISTORICAL_E5_ROOT,
    )
    graph = CovarianceGraph(
        inputs.keyframe_ids,
        inputs.edges,
        covariance_symmetry_atol=inputs.covariance_symmetry_atol,
    )
    optimized_values = build_optimized_values(inputs)

    current_ordinals = deterministic_probe_current_ordinals(
        len(inputs.keyframe_ids)
    )
    current_frame_ids = tuple(
        inputs.keyframe_ids[ordinal] for ordinal in current_ordinals
    )
    if not all(
        frame_id in inputs.frame_id_to_ordinal
        for frame_id in current_frame_ids
    ):
        raise RuntimeError("derived probe FrameId is not canonical")

    ranked_groups = []
    per_current = []
    for current_ordinal, current_frame_id in zip(
        current_ordinals, current_frame_ids
    ):
        ranked_scores = score_previous_keyframes(
            inputs,
            graph,
            optimized_values,
            current_frame_id,
        )
        ranked_groups.append(
            (current_ordinal, current_frame_id, ranked_scores)
        )
        per_current.append(
            _current_result_summary(
                current_ordinal,
                current_frame_id,
                ranked_scores,
            )
        )

    expected_row_count = int(sum(current_ordinals))
    observed_row_count = int(
        sum(len(group[2]) for group in ranked_groups)
    )
    if observed_row_count != expected_row_count:
        raise RuntimeError("bounded probe candidate count differs")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".ex7_i01.",
            dir=str(selected_output_root.parent),
        )
    )
    try:
        csv_path = temporary_root / "candidate_scores.csv"
        npz_path = temporary_root / "candidate_paths_and_covariances.npz"
        aligned_rows = _write_candidate_csv(csv_path, ranked_groups)
        _write_path_npz(npz_path, aligned_rows)
        reload_validation = _reload_artifacts(
            csv_path,
            npz_path,
            expected_row_count=expected_row_count,
        )

        elapsed_seconds = float(time.perf_counter() - started)
        peak_rss_bytes = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        summary = {
            "schema": "exercise7-candidate-detection-probe-v1",
            "schema_version": 1,
            "milestone": "E7-I01",
            "status": "pass",
            "scope": (
                "Exercise 7 sections 7.1a-7.1b only; bounded "
                "candidate scoring without selection"
            ),
            "canonical_inputs": {
                "e6_root": _relative_path(inputs.e6_root),
                "e6_summary": _relative_path(
                    inputs.e6_root / "summary.json"
                ),
                "e6_optimized_values": _relative_path(
                    inputs.e6_root / "pose_graph_values.npz"
                ),
                "e5_root": _relative_path(inputs.e5_root),
                "configuration_sha256": (
                    inputs.configuration_sha256
                ),
                "keyframe_count": len(inputs.keyframe_ids),
                "edge_count": len(inputs.edges),
                "node_symbols": "gtsam.symbol('c', actual FrameId)",
                "e5_validation_reused": (
                    "PoseGraph.load_canonical_pose_graph_inputs"
                ),
                "duplicate_seal_or_hash_framework_created": False,
            },
            "probe": {
                "current_selection": (
                    "nearest integer ordinals at one-third and "
                    "two-thirds of [0, N-1], plus N-1"
                ),
                "current_keyframe_ordinals": list(current_ordinals),
                "current_frame_ids": list(current_frame_ids),
                "all_previous_keyframes_scored": True,
                "candidate_row_count": observed_row_count,
                "temporal_exclusion": None,
                "mahalanobis_threshold": None,
                "candidate_selection_performed": False,
            },
            "course_covariance_path_heuristic": {
                "policy": COURSE_HEURISTIC_POLICY,
                "description": (
                    "Dijkstra-style full 6x6 matrix labels; relax by "
                    "raw covariance addition and prefer the smaller "
                    "determinant of the accumulated covariance"
                ),
                "queue_comparison": (
                    "positive slogdet, equivalent ordering to determinant"
                ),
                "standard_additive_cost_dijkstra_optimality_claimed": False,
                "strict_improvement_only": True,
                "deterministic_tie_order": "actual FrameId",
                "adjacency_direction": (
                    "historical start FrameId -> current end FrameId"
                ),
                "raw_covariance_field": (
                    CONDITIONAL_COVARIANCE_FIELD
                ),
                "edge_direction": EDGE_DIRECTION,
                "tangent_order": list(TANGENT_ORDER),
                "covariance_sum": "direct raw 6x6 matrix addition",
                "adjoint_transport_applied": False,
                "covariance_scaling_applied": False,
                "regularization_applied": False,
                "pseudoinverse_applied": False,
                "reverse_covariance_created": False,
                "maor_scalar_policy_used_canonically": False,
                "maor_bounded_comparison_performed": False,
            },
            "candidate_scoring": {
                "factor": (
                    "gtsam.BetweenFactorPose3(past_key, current_key, "
                    "gtsam.Pose3(), "
                    "gtsam.noiseModel.Gaussian.Covariance(path_covariance))"
                ),
                "factor_error_relation": (
                    "factor_error = 0.5 * mahalanobis_squared"
                ),
                "manual_euler_residual_used": False,
                "manual_covariance_inverse_used": False,
                "threshold": None,
            },
            "results_by_current": per_current,
            "artifact_reload_validation": reload_validation,
            "execution_boundaries": {
                "images_loaded": False,
                "tracking_database_loaded": False,
                "matching_performed": False,
                "pnp_performed": False,
                "bundle_graph_built": False,
                "bundle_optimized": False,
                "loop_factor_added": False,
                "pose_graph_optimized": False,
                "marginals_recomputed": False,
                "e7_i02_started": False,
            },
            "runtime_seconds": elapsed_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "artifacts": [
                "candidate_paths_and_covariances.npz",
                "candidate_scores.csv",
                "summary.json",
            ],
        }
        summary_path = temporary_root / "summary.json"
        _save_json(summary_path, summary)
        reloaded_summary = _read_summary(summary_path)
        if (
            reloaded_summary.get("status") != "pass"
            or reloaded_summary.get("probe", {}).get(
                "candidate_row_count"
            )
            != expected_row_count
        ):
            raise RuntimeError("E7-I01 summary reload validation failed")
        os.replace(str(temporary_root), str(selected_output_root))
        return summary
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def _read_summary(path: Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError("summary must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bounded E7-I01 candidate scoring from canonical E6/E5 inputs"
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="new output directory; it must not already exist",
    )
    return parser


def main(argv: Sequence[str] = None) -> int:
    arguments = _parser().parse_args(argv)
    summary = run_bounded_probe(output_root=arguments.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
