"""Run and retain the exact bounded E7-I02 consensus-matching probe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

import tools
from exercise7_consensus_matching import (
    ADAPTIVE_COMPLETED,
    CANONICAL_E7_I01_SCORES,
    CANONICAL_E7_I01_SCORES_SHA256,
    CANONICAL_SEED,
    DECLARED_CANDIDATE_PAIRS,
    FOUR_VIEW_SUPPORTER_MAX_DISTANCE_PIXELS,
    MAX_HYPOTHESIS_ATTEMPTS,
    POOR_DISTANT_CONTROL,
    PROMISING_DISTANT,
    REPOSITORY_ROOT,
    RANSAC_SAMPLE_SIZE,
    RANSAC_SUCCESS_PROBABILITY,
    CandidatePair,
    PairEvidence,
    PairRun,
    evaluate_candidate_pair,
    file_sha256,
    validate_declared_candidate_pairs,
    validate_pair_evidence,
)


DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex7_i02"
PROTECTED_ARTIFACT_ROOTS = (
    REPOSITORY_ROOT / "diagnostics" / "ex3_final",
    REPOSITORY_ROOT / "diagnostics" / "ex4_final",
    REPOSITORY_ROOT / "diagnostics" / "ex4_i04",
    REPOSITORY_ROOT / "diagnostics" / "ex5_final",
    REPOSITORY_ROOT / "diagnostics" / "ex5_track_survival_082",
    REPOSITORY_ROOT / "diagnostics" / "ex5_i04",
    REPOSITORY_ROOT / "diagnostics" / "ex6_pose_graph",
    REPOSITORY_ROOT / "diagnostics" / "ex7_i01",
    REPOSITORY_ROOT / "project_submission",
)
PROTECTED_INPUT_FILES = (
    REPOSITORY_ROOT / "tools.py",
    REPOSITORY_ROOT / "ex3.py",
    REPOSITORY_ROOT / "exercise7_candidate_detection.py",
    REPOSITORY_ROOT / "test_ex7_candidate_detection.py",
    REPOSITORY_ROOT / "validate_ex7_candidate_detection.py",
    REPOSITORY_ROOT / "diagnostics" / "ex7_i01" / "candidate_scores.csv",
    REPOSITORY_ROOT / "diagnostics" / "ex7_i01" / "summary.json",
    REPOSITORY_ROOT
    / "diagnostics"
    / "ex7_i01"
    / "candidate_paths_and_covariances.npz",
)

CSV_FIELDS = (
    "pair_index",
    "category",
    "current_frame_id",
    "current_keyframe_ordinal",
    "previous_frame_id",
    "previous_keyframe_ordinal",
    "rank_within_current",
    "ordinal_gap",
    "frame_id_gap",
    "mahalanobis_squared",
    "mahalanobis",
    "previous_left_keypoint_count",
    "previous_right_keypoint_count",
    "current_left_keypoint_count",
    "current_right_keypoint_count",
    "previous_raw_stereo_match_count",
    "current_raw_stereo_match_count",
    "previous_stereo_inlier_count",
    "current_stereo_inlier_count",
    "reciprocal_temporal_match_count",
    "four_view_consensus_count",
    "final_supporter_count",
    "final_supporter_percentage",
    "terminal_status",
    "terminal_reason",
    "total_hypothesis_attempt_count",
    "random_hypothesis_attempt_count",
    "refinement_hypothesis_attempt_count",
    "required_hypothesis_count",
    "capped_required_hypothesis_count",
    "maximum_hypothesis_attempt_count",
    "successful_pnp_call_count",
    "failed_pnp_call_count",
    "invalid_geometry_model_count",
    "valid_model_count",
    "refinement_pnp_outcomes_json",
    "image_loading_seconds",
    "stereo_extraction_seconds",
    "temporal_matching_seconds",
    "triangulation_seconds",
    "ransac_seconds",
    "total_runtime_seconds",
)


def _relative_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _save_json(path: Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_output_root(output_root: Path) -> Path:
    selected = Path(output_root).resolve()
    for protected in PROTECTED_ARTIFACT_ROOTS:
        protected = protected.resolve()
        if selected == protected or protected in selected.parents:
            raise ValueError(
                "E7-I02 output must not be inside a protected artifact root"
            )
    if selected.exists():
        raise FileExistsError(
            "E7-I02 output root already exists: {}".format(selected)
        )
    selected.parent.mkdir(parents=True, exist_ok=True)
    return selected


def _protected_hashes() -> Dict[str, str]:
    hashes = {}
    for path in PROTECTED_INPUT_FILES:
        if not path.is_file():
            raise FileNotFoundError(
                "protected E7-I02 input is unavailable: {}".format(path)
            )
        hashes[_relative_path(path)] = file_sha256(path)
    return hashes


def _preflight_images(pairs: Sequence[CandidatePair]) -> None:
    frame_ids = sorted(
        {
            frame_id
            for pair in pairs
            for frame_id in (
                pair.previous_frame_id,
                pair.current_frame_id,
            )
        }
    )
    for frame_id in frame_ids:
        image_name = "{:06d}.png".format(frame_id)
        for directory in ("image_0", "image_1"):
            path = (
                REPOSITORY_ROOT
                / "dataset"
                / "sequences"
                / "00"
                / directory
                / image_name
            )
            if not path.is_file():
                raise FileNotFoundError(
                    "declared E7-I02 image is unavailable: {}".format(path)
                )


def _refinement_records(evidence: PairEvidence) -> List[Dict[str, object]]:
    return [
        asdict(outcome) for outcome in evidence.ransac.refinement_outcomes
    ]


def _evidence_record(evidence: PairEvidence) -> Dict[str, object]:
    pair = evidence.pair
    counts = evidence.counts
    ransac = evidence.ransac
    runtime = evidence.runtime
    return {
        "pair_index": pair.pair_index,
        "category": pair.category,
        "current_frame_id": pair.current_frame_id,
        "current_keyframe_ordinal": pair.current_keyframe_ordinal,
        "previous_frame_id": pair.previous_frame_id,
        "previous_keyframe_ordinal": pair.previous_keyframe_ordinal,
        "rank_within_current": pair.rank_within_current,
        "ordinal_gap": pair.ordinal_gap,
        "frame_id_gap": pair.frame_id_gap,
        "mahalanobis_squared": pair.mahalanobis_squared,
        "mahalanobis": pair.mahalanobis,
        "selection_reason": pair.selection_reason,
        "previous_left_keypoint_count": (
            counts.previous_left_keypoint_count
        ),
        "previous_right_keypoint_count": (
            counts.previous_right_keypoint_count
        ),
        "current_left_keypoint_count": counts.current_left_keypoint_count,
        "current_right_keypoint_count": (
            counts.current_right_keypoint_count
        ),
        "previous_raw_stereo_match_count": (
            counts.previous_raw_stereo_match_count
        ),
        "current_raw_stereo_match_count": (
            counts.current_raw_stereo_match_count
        ),
        "previous_stereo_inlier_count": (
            counts.previous_stereo_inlier_count
        ),
        "current_stereo_inlier_count": counts.current_stereo_inlier_count,
        "reciprocal_temporal_match_count": (
            counts.reciprocal_temporal_match_count
        ),
        "four_view_consensus_count": counts.four_view_consensus_count,
        "final_supporter_count": counts.final_supporter_count,
        "final_supporter_percentage": counts.final_supporter_percentage,
        "terminal_status": ransac.status,
        "terminal_reason": ransac.terminal_reason,
        "total_hypothesis_attempt_count": (
            ransac.total_hypothesis_attempt_count
        ),
        "random_hypothesis_attempt_count": (
            ransac.random_hypothesis_attempt_count
        ),
        "refinement_hypothesis_attempt_count": (
            ransac.refinement_hypothesis_attempt_count
        ),
        "required_hypothesis_count": ransac.required_hypothesis_count,
        "capped_required_hypothesis_count": (
            ransac.capped_required_hypothesis_count
        ),
        "maximum_hypothesis_attempt_count": (
            ransac.maximum_hypothesis_attempt_count
        ),
        "successful_pnp_call_count": ransac.successful_pnp_call_count,
        "failed_pnp_call_count": ransac.failed_pnp_call_count,
        "invalid_geometry_model_count": (
            ransac.invalid_geometry_model_count
        ),
        "valid_model_count": ransac.valid_model_count,
        "refinement_pnp_outcomes": _refinement_records(evidence),
        "image_loading_seconds": runtime.image_loading_seconds,
        "stereo_extraction_seconds": runtime.stereo_extraction_seconds,
        "temporal_matching_seconds": runtime.temporal_matching_seconds,
        "triangulation_seconds": runtime.triangulation_seconds,
        "ransac_seconds": runtime.ransac_seconds,
        "total_runtime_seconds": runtime.total_seconds,
    }


def _csv_record(evidence: PairEvidence) -> Dict[str, object]:
    row = _evidence_record(evidence)
    row.pop("selection_reason")
    refinement = row.pop("refinement_pnp_outcomes")
    row["refinement_pnp_outcomes_json"] = json.dumps(
        refinement, sort_keys=True, separators=(",", ":")
    )
    return row


def _write_pair_csv(path: Path, evidence: Sequence[PairEvidence]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for pair_evidence in evidence:
            writer.writerow(_csv_record(pair_evidence))


def _visualization_order_key(evidence: PairEvidence) -> Tuple[int, int, int, int]:
    return (
        -evidence.counts.final_supporter_count,
        evidence.pair.rank_within_current,
        evidence.pair.current_frame_id,
        evidence.pair.previous_frame_id,
    )


def _eligible_visualization(run: PairRun) -> bool:
    return bool(
        run.evidence.pair.category == PROMISING_DISTANT
        and run.evidence.ransac.status == ADAPTIVE_COMPLETED
        and run.evidence.counts.final_supporter_count > 0
    )


def _save_match_visualization(path: Path, run: PairRun) -> None:
    evidence = run.evidence
    visualization = run.visualization
    previous_image = visualization.previous_image_left
    current_image = visualization.current_image_left
    previous_height, previous_width = previous_image.shape[:2]
    current_height, current_width = current_image.shape[:2]
    canvas_height = max(previous_height, current_height)
    canvas = np.zeros(
        (canvas_height, previous_width + current_width), dtype=np.uint8
    )
    canvas[:previous_height, :previous_width] = previous_image
    canvas[:current_height, previous_width:] = current_image

    supporter_set = set(visualization.supporter_indices)
    figure, axis = plt.subplots(figsize=(16, 6))
    axis.imshow(canvas, cmap="gray")
    for consensus_index, consensus_match in enumerate(
        visualization.consensus
    ):
        temporal_index = int(consensus_match[2])
        temporal_match = visualization.reciprocal_matches[temporal_index]
        previous_point = visualization.previous_keypoints_left[
            temporal_match.queryIdx
        ].pt
        current_point = visualization.current_keypoints_left[
            temporal_match.trainIdx
        ].pt
        is_supporter = consensus_index in supporter_set
        color = "cyan" if is_supporter else "red"
        alpha = 0.85 if is_supporter else 0.22
        linewidth = 0.8 if is_supporter else 0.35
        axis.plot(
            [previous_point[0], current_point[0] + previous_width],
            [previous_point[1], current_point[1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
    axis.axvline(previous_width, color="white", linewidth=0.5)
    axis.set_axis_off()
    axis.set_title(
        (
            "E7-I02 bounded matching evidence — not an accepted Loop Closure\n"
            "previous FrameId {} → current FrameId {}; rank {}; "
            "{} / {} supporters ({:.3f}%)"
        ).format(
            evidence.pair.previous_frame_id,
            evidence.pair.current_frame_id,
            evidence.pair.rank_within_current,
            evidence.counts.final_supporter_count,
            evidence.counts.four_view_consensus_count,
            evidence.counts.final_supporter_percentage,
        )
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _policy_evidence(evidence: Sequence[PairEvidence]) -> Dict[str, object]:
    promising = [
        value
        for value in evidence
        if value.pair.category == PROMISING_DISTANT
    ]
    adjacent = [
        value
        for value in evidence
        if value.pair.category == "adjacent_control"
    ]
    poor = [
        value
        for value in evidence
        if value.pair.category == POOR_DISTANT_CONTROL
    ]
    adaptive_promising = [
        value
        for value in promising
        if value.ransac.status == ADAPTIVE_COMPLETED
    ]
    return {
        "decisions_frozen": False,
        "mahalanobis_threshold": {
            "frozen": False,
            "adaptive_promising_scores": [
                value.pair.mahalanobis for value in adaptive_promising
            ],
            "poor_control_scores": [
                value.pair.mahalanobis for value in poor
            ],
            "decision": None,
        },
        "temporal_exclusion": {
            "frozen": False,
            "adjacent_control_results": [
                {
                    "current_frame_id": value.pair.current_frame_id,
                    "previous_frame_id": value.pair.previous_frame_id,
                    "ordinal_gap": value.pair.ordinal_gap,
                    "frame_id_gap": value.pair.frame_id_gap,
                    "terminal_status": value.ransac.status,
                    "final_supporter_count": (
                        value.counts.final_supporter_count
                    ),
                    "final_supporter_percentage": (
                        value.counts.final_supporter_percentage
                    ),
                }
                for value in adjacent
            ],
            "decision": None,
        },
        "maximum_candidate_count": {
            "frozen": False,
            "adaptive_promising_ranks_by_current": {
                str(current_frame_id): sorted(
                    value.pair.rank_within_current
                    for value in adaptive_promising
                    if value.pair.current_frame_id == current_frame_id
                )
                for current_frame_id in sorted(
                    {
                        value.pair.current_frame_id
                        for value in promising
                    }
                )
            },
            "decision": None,
        },
        "bounded_evidence_requires_review": True,
    }


def _reload_artifacts(
    temporary_root: Path,
    *,
    expected_visualization: Optional[Path],
) -> Dict[str, object]:
    csv_path = temporary_root / "pair_results.csv"
    summary_path = temporary_root / "summary.json"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(DECLARED_CANDIDATE_PAIRS):
        raise RuntimeError("E7-I02 CSV row count differs")
    if [int(row["pair_index"]) for row in rows] != list(range(8)):
        raise RuntimeError("E7-I02 CSV pair ordering differs")
    for row in rows:
        if int(row["total_hypothesis_attempt_count"]) != (
            int(row["successful_pnp_call_count"])
            + int(row["failed_pnp_call_count"])
        ):
            raise RuntimeError("E7-I02 reloaded attempt accounting differs")
        refinement = json.loads(row["refinement_pnp_outcomes_json"])
        if len(refinement) != int(
            row["refinement_hypothesis_attempt_count"]
        ):
            raise RuntimeError("E7-I02 reloaded refinement count differs")
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise RuntimeError("E7-I02 summary must be a JSON object")
    if (
        summary.get("milestone") != "E7-I02"
        or summary.get("pair_count") != 8
        or summary.get("status") != "pass"
    ):
        raise RuntimeError("E7-I02 summary reload validation failed")
    if (
        expected_visualization is not None
        and not expected_visualization.is_file()
    ):
        raise RuntimeError("E7-I02 expected visualization is unavailable")
    return {
        "success": True,
        "pair_results_csv_rows": len(rows),
        "summary_reloaded": True,
        "visualization_reloaded": expected_visualization is not None,
    }


def run_bounded_probe(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Dict[str, object]:
    """Run exactly eight independent candidate pairs and retain evidence."""

    started = time.perf_counter()
    selected_output_root = _validate_output_root(output_root)
    pairs = validate_declared_candidate_pairs()
    if pairs != DECLARED_CANDIDATE_PAIRS:
        raise RuntimeError("declared pair order differs")
    if file_sha256(CANONICAL_E7_I01_SCORES) != (
        CANONICAL_E7_I01_SCORES_SHA256
    ):
        raise RuntimeError("E7-I01 candidate source differs")
    _preflight_images(pairs)
    protected_hashes_before = _protected_hashes()

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=".ex7_i02.",
            dir=str(selected_output_root.parent),
        )
    )
    evidence: List[PairEvidence] = []
    selected_visualization_run: Optional[PairRun] = None
    try:
        for pair in pairs:
            pair_run = evaluate_candidate_pair(
                pair,
                seed=CANONICAL_SEED,
                maximum_attempts=MAX_HYPOTHESIS_ATTEMPTS,
            )
            validate_pair_evidence(pair_run.evidence)
            evidence.append(pair_run.evidence)
            if _eligible_visualization(pair_run) and (
                selected_visualization_run is None
                or _visualization_order_key(pair_run.evidence)
                < _visualization_order_key(
                    selected_visualization_run.evidence
                )
            ):
                selected_visualization_run = pair_run
            print(
                (
                    "pair {}/8 current={} previous={} status={} "
                    "attempts={}/required={} supporters={}/{} ({:.3f}%) "
                    "runtime={:.3f}s"
                ).format(
                    pair.pair_index + 1,
                    pair.current_frame_id,
                    pair.previous_frame_id,
                    pair_run.evidence.ransac.status,
                    pair_run.evidence.ransac.total_hypothesis_attempt_count,
                    pair_run.evidence.ransac.required_hypothesis_count,
                    pair_run.evidence.counts.final_supporter_count,
                    pair_run.evidence.counts.four_view_consensus_count,
                    pair_run.evidence.counts.final_supporter_percentage,
                    pair_run.evidence.runtime.total_seconds,
                ),
                flush=True,
            )

        csv_path = temporary_root / "pair_results.csv"
        _write_pair_csv(csv_path, evidence)

        visualization_relative_path = None
        visualization_path = None
        if selected_visualization_run is not None:
            figures_root = temporary_root / "figures"
            figures_root.mkdir()
            chosen = selected_visualization_run.evidence
            visualization_path = figures_root / (
                "bounded_match_previous_{:04d}_current_{:04d}.png".format(
                    chosen.pair.previous_frame_id,
                    chosen.pair.current_frame_id,
                )
            )
            _save_match_visualization(
                visualization_path, selected_visualization_run
            )
            visualization_relative_path = str(
                visualization_path.relative_to(temporary_root)
            )

        protected_hashes_after = _protected_hashes()
        if protected_hashes_after != protected_hashes_before:
            raise RuntimeError(
                "a protected source or upstream artifact changed"
            )

        elapsed_seconds = float(time.perf_counter() - started)
        peak_rss_bytes = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        summary = {
            "schema": "exercise7-consensus-matching-probe-v1",
            "schema_version": 1,
            "milestone": "E7-I02",
            "status": "pass",
            "scope": (
                "exact eight-pair bounded consensus matching; "
                "candidate policies remain unfrozen"
            ),
            "pair_count": len(evidence),
            "declared_pair_indices": [
                pair.pair_index for pair in pairs
            ],
            "canonical_input": {
                "candidate_scores_csv": _relative_path(
                    CANONICAL_E7_I01_SCORES
                ),
                "candidate_scores_sha256": (
                    CANONICAL_E7_I01_SCORES_SHA256
                ),
                "dataset": "KITTI odometry sequence 00 stereo images",
                "direct_image_loading": True,
                "tracking_database_loaded": False,
            },
            "image_pipeline": {
                "detector": dict(tools.CANONICAL_SIFT_PARAMETERS),
                "matcher_index": dict(
                    tools.CANONICAL_FLANN_INDEX_PARAMETERS
                ),
                "matcher_search": dict(
                    tools.CANONICAL_FLANN_SEARCH_PARAMETERS
                ),
                "stereo_policy": (
                    "vertical agreement <= 2 pixels and positive disparity"
                ),
                "temporal_policy": "reciprocal one-nearest-neighbor",
                "four_view_consensus": True,
                "supporter_max_distance_pixels_all_four_views": (
                    FOUR_VIEW_SUPPORTER_MAX_DISTANCE_PIXELS
                ),
            },
            "bounded_ransac": {
                "implementation": (
                    "E7-specific lower-level solvePnP and supporter driver"
                ),
                "seed_reset_independently_per_pair": CANONICAL_SEED,
                "sample_size": RANSAC_SAMPLE_SIZE,
                "success_probability": RANSAC_SUCCESS_PROBABILITY,
                "maximum_total_hypothesis_attempts_per_pair": (
                    MAX_HYPOTHESIS_ATTEMPTS
                ),
                "every_solvePnP_call_counted": True,
                "failed_solvePnP_calls_counted": True,
                "hidden_retry_loop_used": False,
                "attempt_counter_reset_after_improvement": False,
                "full_exercise3_ransac_called": False,
                "exercise3_build_model_called": False,
                "refinement_requires_at_least_four_supporters": True,
                "refinement_accepts_strict_supporter_increase_only": True,
                "budget_exhausted_best_model_is_accepted_loop": False,
            },
            "pair_results": [
                _evidence_record(value) for value in evidence
            ],
            "terminal_status_counts": {
                status: sum(
                    value.ransac.status == status for value in evidence
                )
                for status in sorted(
                    {value.ransac.status for value in evidence}
                )
            },
            "policy_evidence": _policy_evidence(evidence),
            "visualization": {
                "created": visualization_path is not None,
                "path": visualization_relative_path,
                "selection": (
                    "adaptive-completed promising distant pair with "
                    "largest final supporter count; ties by rank, current "
                    "FrameId, then previous FrameId"
                ),
                "label": (
                    "bounded matching evidence; not an accepted Loop Closure"
                ),
            },
            "protected_input_sha256_before_and_after": (
                protected_hashes_after
            ),
            "execution_boundaries": {
                "declared_pairs_only": True,
                "undeclared_pair_evaluated": False,
                "full_all_candidate_run_performed": False,
                "tracking_database_loaded": False,
                "bundle_built": False,
                "loop_factor_added": False,
                "pose_graph_optimized": False,
                "marginals_recomputed": False,
                "final_report_work_started": False,
                "policies_frozen": False,
            },
            "runtime_seconds": elapsed_seconds,
            "pair_runtime_seconds_sum": float(
                sum(value.runtime.total_seconds for value in evidence)
            ),
            "peak_rss_bytes": peak_rss_bytes,
            "artifacts": [
                "pair_results.csv",
                "summary.json",
            ]
            + (
                [visualization_relative_path]
                if visualization_relative_path is not None
                else []
            ),
        }
        summary_path = temporary_root / "summary.json"
        _save_json(summary_path, summary)
        reload_validation = _reload_artifacts(
            temporary_root,
            expected_visualization=visualization_path,
        )
        summary["artifact_reload_validation"] = reload_validation
        _save_json(summary_path, summary)
        final_reload = _read_json(summary_path)
        if (
            not isinstance(final_reload, dict)
            or final_reload.get("artifact_reload_validation", {}).get(
                "success"
            )
            is not True
        ):
            raise RuntimeError("final E7-I02 summary reload failed")
        os.replace(str(temporary_root), str(selected_output_root))
        return summary
    except BaseException:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact bounded E7-I02 consensus-matching probe"
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
