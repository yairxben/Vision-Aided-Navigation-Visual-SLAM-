"""Sequential full Exercise 7 Loop Closure production runner.

The visible scientific flow is candidate detection -> consensus matching ->
two-pose stereo Bundle -> cumulative Pose Graph update -> seal/replay.  The
validated E7-I04 ordering remains exact: candidate detection is frozen once
per current keyframe, and every successfully integrated loop is immediately
optimized, marginalized, inserted into the covariance graph, and sealed before
it may suppress later candidates.

Here Loop Closure means recognizing a revisit to previously observed scene or
location evidence; it does not mean discovering a new area.  For each current
keyframe, detection is computed once from an immutable snapshot of the latest
optimized Values and covariance graph.  That current's score/rank population
then stays frozen even if an accepted candidate updates the live Pose Graph.
The newest state initializes subsequent same-current small Bundles, while the
next current keyframe performs fresh detection against the newest optimized
Values and accepted covariance edges.

The orchestration distinguishes four kinds of state.  Scientific state is the
cumulative Pose Graph/Values/Marginals plus the directed covariance graph.
Candidate-local temporary state is matching, RANSAC, Bundle, row, and staging
evidence that may fail without a scientific mutation.  Persisted accepted
state consists of a fully validated factor update, covariance edge, and
suppression anchor.  Sealed artifact state is the only durable resume point:
each current is validated and atomically renamed before a later current can
observe it.  A fatal interruption discards its unsealed attempt; replay rebuilds
the in-memory state from the contiguous validated seal prefix.

Input and output roots are explicit.  Clean roots are the defaults, while a
coherent historical E5/E6 pair may still be selected for read-only validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import cv2
import gtsam
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import tools
from PoseGraph import CANONICAL_INPUT_ROOT, TANGENT_ORDER, camera_symbol
from exercise7_candidate_detection import (
    CandidateDetectionError,
    CandidateDetectionInputs,
    CandidateScore,
    CovarianceGraph,
    candidate_detection_inputs_from_values,
    load_candidate_detection_inputs,
    score_previous_keyframes,
)
from exercise7_consensus_matching import (
    CANONICAL_SEED,
    MAX_HYPOTHESIS_ATTEMPTS,
    CandidatePair,
    ConsensusMatchingError,
    PairEvidence,
    PairRun,
    candidate_pair_from_score,
    evaluate_candidate_pair,
    production_match_policy_decision,
    validate_candidate_pair,
    validate_pair_evidence,
)
from exercise7_pose_graph_update import (
    CumulativePoseGraphState,
    PoseGraphUpdateResult,
    PoseGraphUpdateValidationError,
    load_protected_e6_state,
)
from exercise7_small_bundle import (
    SmallBundleResult,
    SmallBundleValidationError,
    calibration_from_configuration,
    observations_from_pair_run,
    solve_small_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_E5_ROOT = CANONICAL_INPUT_ROOT
DEFAULT_E6_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex6_clean_final"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex7_clean_final"
HISTORICAL_E5_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex5_final"
HISTORICAL_E6_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex6_pose_graph"
HISTORICAL_E7_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex7_i04"

# Compatibility aliases for callers that imported the previous module-level
# E6 paths.  Active code derives these files from its selected E6 root.
E6_ROOT = DEFAULT_E6_ROOT
E6_SUMMARY = E6_ROOT / "summary.json"
E6_VALUES = E6_ROOT / "pose_graph_values.npz"
E6_MARGINALS = E6_ROOT / "pose_graph_marginals.npz"
CALIBRATION_PATH = (
    REPOSITORY_ROOT / "dataset" / "sequences" / "00" / "calib.txt"
)
PROTECTED_ARTIFACT_ROOTS = (
    HISTORICAL_E5_ROOT,
    DEFAULT_E5_ROOT,
    REPOSITORY_ROOT / "diagnostics" / "ex5_track_survival_082",
    HISTORICAL_E6_ROOT,
    DEFAULT_E6_ROOT,
    HISTORICAL_E7_ROOT,
    REPOSITORY_ROOT / "project_submission" / "bundle",
    REPOSITORY_ROOT / "project_submission" / "bundle_clean_final",
    REPOSITORY_ROOT / "project_submission" / "loop_closure",
    REPOSITORY_ROOT / "project_submission" / "loop_closure_clean_final",
)

POLICY_SCHEMA = "exercise7-i04-production-policy-v1"
CURRENT_SCHEMA = "exercise7-i04-current-v1"
CURRENT_SEAL_SCHEMA = "exercise7-i04-current-seal-v1"
SUMMARY_SCHEMA = "exercise7-i04-summary-v1"
TRACEABILITY_SCHEMA = "exercise7-i04-traceability-v1"

MAHALANOBIS_SQUARED_THRESHOLD = 36_000_000.0
MAHALANOBIS_THRESHOLD = 6000.0
MINIMUM_TEMPORAL_ORDINAL_GAP = 16
MAXIMUM_MATCHER_INVOCATIONS_PER_CURRENT = 3
MINIMUM_CONSENSUS_COUNT = 30
MINIMUM_SUPPORTER_COUNT = 30
MINIMUM_SUPPORTER_PERCENTAGE = 30.0
CLUSTER_PREVIOUS_ORDINAL_RADIUS = 10
CLUSTER_CURRENT_ORDINAL_RADIUS = 10
EXPECTED_KEYFRAME_COUNT = 356
EXPECTED_ORDERED_PAIR_COUNT = 63_190
BASE_POSE_GRAPH_FACTOR_COUNT = 356
BASE_COVARIANCE_EDGE_COUNT = 355

MATCHER_CSV_FIELDS = (
    "pair_index",
    "previous_frame_id",
    "current_frame_id",
    "previous_keyframe_ordinal",
    "current_keyframe_ordinal",
    "rank_within_current",
    "temporary_region_id",
    "scheduler_pass",
    "matcher_slot",
    "seed",
    "maximum_hypothesis_attempts",
    "outcome",
    "failure_type",
    "failure_message",
    "match_policy_reason",
    "acceptance_id",
    "eligible_landmark_count",
    "measurement_artifact",
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
    "refinement_outcomes",
    "runtime_seconds",
)

ACCEPTED_LOOP_CSV_FIELDS = (
    "acceptance_id",
    "pair_index",
    "previous_frame_id",
    "current_frame_id",
    "previous_keyframe_ordinal",
    "current_keyframe_ordinal",
    "consensus_count",
    "supporter_count",
    "supporter_percentage",
    "ransac_terminal_status",
    "eligible_landmark_count",
    "rejected_landmark_count",
    "factor_direction",
    "tangent_order",
    "covariance_modified",
    "state_version_before",
    "state_version_after",
    "factor_count_after",
    "graph_error_before_candidate_factor",
    "graph_error_after_factor_before_optimization",
    "graph_error_after_optimization",
    "loop_factor_error_before_optimization",
    "loop_factor_error_after_optimization",
    "marginal_count",
    "marginal_symmetry_max_abs",
    "marginal_minimum_eigenvalue",
    "small_bundle_initial_error",
    "small_bundle_final_error",
    "artifact",
    "artifact_path",
    "successful_match_artifact",
    "successful_match_artifact_path",
    "previous_left_image_path",
    "previous_left_image_sha256",
    "current_left_image_path",
    "current_left_image_sha256",
    "consensus_pixel_layout",
)


class SequentialLoopClosureError(RuntimeError):
    """Raised when the E7-I04 production contract cannot be preserved."""


@dataclass(frozen=True)
class Exercise7InputPaths:
    """Files derived from one explicitly selected, coherent E5/E6 pair."""

    e5_root: Path
    e6_root: Path

    @classmethod
    def from_roots(
        cls, *, e5_root: Path, e6_root: Path
    ) -> "Exercise7InputPaths":
        return cls(
            e5_root=Path(e5_root).resolve(),
            e6_root=Path(e6_root).resolve(),
        )

    @property
    def e5_manifest(self) -> Path:
        return self.e5_root / "manifest.json"

    @property
    def e5_configuration(self) -> Path:
        return self.e5_root / "configuration.json"

    @property
    def e6_summary(self) -> Path:
        return self.e6_root / "summary.json"

    @property
    def e6_values(self) -> Path:
        return self.e6_root / "pose_graph_values.npz"

    @property
    def e6_marginals(self) -> Path:
        return self.e6_root / "pose_graph_marginals.npz"


@dataclass(frozen=True)
class ProductionPolicy:
    """Frozen detection, scheduling, matching, and suppression policy.

    Ordinal gaps/radii concern positions in the canonical keyframe sequence;
    candidate endpoints and graph symbols remain actual FrameIds.  The
    per-current cap counts calls to ``evaluate_candidate_pair``, not candidates
    scored, thresholded, suppressed, or merely offered by the scheduler.
    """

    mahalanobis_squared_threshold: float = (
        MAHALANOBIS_SQUARED_THRESHOLD
    )
    mahalanobis_threshold: float = MAHALANOBIS_THRESHOLD
    temporal_ordinal_gap_minimum: int = MINIMUM_TEMPORAL_ORDINAL_GAP
    maximum_matcher_invocations_per_current: int = (
        MAXIMUM_MATCHER_INVOCATIONS_PER_CURRENT
    )
    minimum_consensus_count: int = MINIMUM_CONSENSUS_COUNT
    minimum_supporter_count: int = MINIMUM_SUPPORTER_COUNT
    minimum_supporter_percentage: float = MINIMUM_SUPPORTER_PERCENTAGE
    previous_ordinal_cluster_radius: int = (
        CLUSTER_PREVIOUS_ORDINAL_RADIUS
    )
    current_ordinal_cluster_radius: int = CLUSTER_CURRENT_ORDINAL_RADIUS
    pair_seed: int = CANONICAL_SEED
    maximum_hypothesis_attempts: int = MAX_HYPOTHESIS_ATTEMPTS

    def validate(self) -> None:
        _require(
            self.mahalanobis_squared_threshold
            == self.mahalanobis_threshold ** 2,
            "squared and unsquared Mahalanobis thresholds differ",
        )
        _require(
            self.temporal_ordinal_gap_minimum == 16
            and self.maximum_matcher_invocations_per_current == 3
            and self.minimum_consensus_count == 30
            and self.minimum_supporter_count == 30
            and self.minimum_supporter_percentage == 30.0
            and self.previous_ordinal_cluster_radius == 10
            and self.current_ordinal_cluster_radius == 10
            and self.pair_seed == 0
            and self.maximum_hypothesis_attempts == 3000,
            "E7-I04 production policy differs from the frozen values",
        )

    def record(self) -> Mapping[str, object]:
        value = asdict(self)
        value.update(
            {
                "schema": POLICY_SCHEMA,
                "mahalanobis_definition": (
                    "d_squared = 2 * BetweenFactorPose3.error"
                ),
                "mahalanobis_squared_threshold_units": (
                    "dimensionless squared residual"
                ),
                "mahalanobis_threshold_units": (
                    "dimensionless residual"
                ),
                "mahalanobis_comparison": "strictly_less_than",
                "candidate_ranking": [
                    "mahalanobis_squared",
                    "previous_keyframe_ordinal",
                    "previous_FrameId",
                ],
                "accepted_cluster_relation": (
                    "abs(delta_previous_ordinal)<=10 and "
                    "abs(delta_current_ordinal)<=10"
                ),
                "cluster_expansion": "anchored_nontransitive",
                "accepted_ransac_terminal_statuses": [
                    "adaptive_completed",
                    "budget_exhausted_with_best_model",
                ],
                "match_acceptance": (
                    "complete retained geometry plus consensus>=30, "
                    "supporters>=30, supporter_percentage>=30.0"
                ),
                "exact_duplicate_pair": (
                    "same ordered previous_FrameId,current_FrameId; "
                    "suppressed without consuming matcher capacity"
                ),
                "temporary_region_schedule": (
                    "rank-ordered anchored nontransitive regions; one live "
                    "head per region before pass-2 same-region fallback"
                ),
                "matcher_cap_accounting": (
                    "only actual evaluate_candidate_pair invocations consume "
                    "the per-current cap"
                ),
                "persistent_cluster_suppression": (
                    "created only after Bundle, Pose Graph optimization, "
                    "all Marginals, and covariance-edge insertion succeed"
                ),
                "candidate_detection_frequency": (
                    "once at start of each current keyframe; queue frozen "
                    "within current"
                ),
                "latest_state_use": (
                    "same-current Bundle initialization and every subsequent "
                    "current use latest optimized Values; subsequent current "
                    "detection also uses latest Marginals/covariance graph"
                ),
                "ground_truth_used_for_acceptance": False,
            }
        )
        return value


@dataclass(frozen=True)
class AcceptedLoopAnchor:
    """Persisted nontransitive suppression anchor from an integrated loop."""

    acceptance_id: int
    previous_frame_id: int
    current_frame_id: int
    previous_ordinal: int
    current_ordinal: int
    state_version: int


@dataclass(frozen=True)
class RankedCandidate:
    """Threshold-passing score plus its frozen rank within one current."""

    pair_index: int
    score: CandidateScore
    rank_within_current: int

    @property
    def previous_frame_id(self) -> int:
        return int(self.score.previous_frame_id)

    @property
    def current_frame_id(self) -> int:
        return int(self.score.current_frame_id)

    @property
    def previous_ordinal(self) -> int:
        return int(self.score.previous_keyframe_ordinal)

    @property
    def current_ordinal(self) -> int:
        return int(self.score.current_keyframe_ordinal)


@dataclass
class TemporaryProposalRegion:
    """Candidate-local anchored region used only for fair matcher scheduling."""

    region_id: int
    anchor_previous_ordinal: int
    anchor_current_ordinal: int
    candidates: List[RankedCandidate]
    cursor: int = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SequentialLoopClosureError(message)


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(
            _jsonable(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name), dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_json(temporary, value)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_pair_index(
    previous_ordinal: int, current_ordinal: int
) -> int:
    previous_ordinal = int(previous_ordinal)
    current_ordinal = int(current_ordinal)
    _require(
        0 <= previous_ordinal < current_ordinal,
        "ordered-pair index requires 0 <= previous < current",
    )
    return (
        current_ordinal * (current_ordinal - 1) // 2
        + previous_ordinal
    )


def same_anchored_rectangle(
    first_previous_ordinal: int,
    first_current_ordinal: int,
    second_previous_ordinal: int,
    second_current_ordinal: int,
    *,
    previous_radius: int = CLUSTER_PREVIOUS_ORDINAL_RADIUS,
    current_radius: int = CLUSTER_CURRENT_ORDINAL_RADIUS,
) -> bool:
    return bool(
        abs(int(first_previous_ordinal) - int(second_previous_ordinal))
        <= int(previous_radius)
        and abs(int(first_current_ordinal) - int(second_current_ordinal))
        <= int(current_radius)
    )


def accepted_anchor_for_candidate(
    candidate: RankedCandidate,
    anchors: Sequence[AcceptedLoopAnchor],
    policy: ProductionPolicy,
) -> Optional[AcceptedLoopAnchor]:
    for anchor in sorted(anchors, key=lambda item: item.acceptance_id):
        if same_anchored_rectangle(
            candidate.previous_ordinal,
            candidate.current_ordinal,
            anchor.previous_ordinal,
            anchor.current_ordinal,
            previous_radius=policy.previous_ordinal_cluster_radius,
            current_radius=policy.current_ordinal_cluster_radius,
        ):
            return anchor
    return None


def form_temporary_proposal_regions(
    candidates: Sequence[RankedCandidate],
    policy: ProductionPolicy,
) -> Tuple[TemporaryProposalRegion, ...]:
    """Greedily group a ranked queue around non-transitive proposal anchors."""

    regions: List[TemporaryProposalRegion] = []
    for candidate in candidates:
        selected: Optional[TemporaryProposalRegion] = None
        for region in regions:
            if same_anchored_rectangle(
                candidate.previous_ordinal,
                candidate.current_ordinal,
                region.anchor_previous_ordinal,
                region.anchor_current_ordinal,
                previous_radius=policy.previous_ordinal_cluster_radius,
                current_radius=policy.current_ordinal_cluster_radius,
            ):
                selected = region
                break
        if selected is None:
            selected = TemporaryProposalRegion(
                region_id=len(regions),
                anchor_previous_ordinal=candidate.previous_ordinal,
                anchor_current_ordinal=candidate.current_ordinal,
                candidates=[],
            )
            regions.append(selected)
        selected.candidates.append(candidate)
    return tuple(regions)


def evaluate_region_schedule(
    regions: Sequence[TemporaryProposalRegion],
    *,
    maximum_invocations: int,
    candidate_is_live: Callable[
        [RankedCandidate, TemporaryProposalRegion, int], bool
    ],
    evaluate_live_candidate: Callable[
        [RankedCandidate, TemporaryProposalRegion, int, int], None
    ],
) -> int:
    """Run fair region-first passes while allowing deterministic fallback."""

    _require(maximum_invocations >= 0, "matcher cap must be nonnegative")
    invocation_count = 0
    scheduler_pass = 1
    while invocation_count < maximum_invocations:
        offered_live_candidate = False
        for region in regions:
            candidate: Optional[RankedCandidate] = None
            while region.cursor < len(region.candidates):
                proposed = region.candidates[region.cursor]
                region.cursor += 1
                if candidate_is_live(proposed, region, scheduler_pass):
                    candidate = proposed
                    break
            if candidate is None:
                continue
            offered_live_candidate = True
            invocation_count += 1
            evaluate_live_candidate(
                candidate,
                region,
                scheduler_pass,
                invocation_count,
            )
            if invocation_count >= maximum_invocations:
                break
        if not offered_live_candidate:
            break
        scheduler_pass += 1
    return invocation_count


def _base_candidate_row(
    *,
    keyframe_ids: Sequence[int],
    previous_ordinal: int,
    current_ordinal: int,
    state_version: int,
) -> Dict[str, object]:
    previous_frame_id = int(keyframe_ids[previous_ordinal])
    current_frame_id = int(keyframe_ids[current_ordinal])
    return {
        "pair_index": ordered_pair_index(previous_ordinal, current_ordinal),
        "previous_keyframe_ordinal": int(previous_ordinal),
        "current_keyframe_ordinal": int(current_ordinal),
        "previous_frame_id": previous_frame_id,
        "current_frame_id": current_frame_id,
        "ordinal_gap": int(current_ordinal - previous_ordinal),
        "frame_id_gap": int(current_frame_id - previous_frame_id),
        "detection_state_version": int(state_version),
        "factor_error": None,
        "mahalanobis_squared": None,
        "mahalanobis": None,
        "rank_within_current": None,
        "temporary_region_id": None,
        "temporary_region_anchor_previous_ordinal": None,
        "temporary_region_anchor_current_ordinal": None,
        "scheduler_pass": None,
        "matcher_slot": None,
        "matcher_slot_consumed": False,
        "accepted_anchor_id": None,
        "accepted_anchor_delta_previous_ordinal": None,
        "accepted_anchor_delta_current_ordinal": None,
        "terminal_disposition": None,
        "terminal_reason": None,
    }


def _pair_evidence_record(evidence: PairEvidence) -> Dict[str, object]:
    counts = evidence.counts
    ransac = evidence.ransac
    runtime = evidence.runtime
    return {
        "pair_index": int(evidence.pair.pair_index),
        "previous_frame_id": int(evidence.pair.previous_frame_id),
        "current_frame_id": int(evidence.pair.current_frame_id),
        "previous_keyframe_ordinal": int(
            evidence.pair.previous_keyframe_ordinal
        ),
        "current_keyframe_ordinal": int(
            evidence.pair.current_keyframe_ordinal
        ),
        "rank_within_current": int(
            evidence.pair.rank_within_current
        ),
        "previous_left_keypoint_count": int(
            counts.previous_left_keypoint_count
        ),
        "previous_right_keypoint_count": int(
            counts.previous_right_keypoint_count
        ),
        "current_left_keypoint_count": int(
            counts.current_left_keypoint_count
        ),
        "current_right_keypoint_count": int(
            counts.current_right_keypoint_count
        ),
        "previous_raw_stereo_match_count": int(
            counts.previous_raw_stereo_match_count
        ),
        "current_raw_stereo_match_count": int(
            counts.current_raw_stereo_match_count
        ),
        "four_view_consensus_count": int(
            counts.four_view_consensus_count
        ),
        "final_supporter_count": int(counts.final_supporter_count),
        "final_supporter_percentage": float(
            counts.final_supporter_percentage
        ),
        "reciprocal_temporal_match_count": int(
            counts.reciprocal_temporal_match_count
        ),
        "previous_stereo_inlier_count": int(
            counts.previous_stereo_inlier_count
        ),
        "current_stereo_inlier_count": int(
            counts.current_stereo_inlier_count
        ),
        "terminal_status": ransac.status,
        "terminal_reason": ransac.terminal_reason,
        "total_hypothesis_attempt_count": int(
            ransac.total_hypothesis_attempt_count
        ),
        "random_hypothesis_attempt_count": int(
            ransac.random_hypothesis_attempt_count
        ),
        "refinement_hypothesis_attempt_count": int(
            ransac.refinement_hypothesis_attempt_count
        ),
        "required_hypothesis_count": int(
            ransac.required_hypothesis_count
        ),
        "capped_required_hypothesis_count": int(
            ransac.capped_required_hypothesis_count
        ),
        "maximum_hypothesis_attempt_count": int(
            ransac.maximum_hypothesis_attempt_count
        ),
        "successful_pnp_call_count": int(
            ransac.successful_pnp_call_count
        ),
        "failed_pnp_call_count": int(ransac.failed_pnp_call_count),
        "invalid_geometry_model_count": int(
            ransac.invalid_geometry_model_count
        ),
        "valid_model_count": int(ransac.valid_model_count),
        "refinement_outcomes": [
            asdict(outcome) for outcome in ransac.refinement_outcomes
        ],
        "runtime_seconds": {
            "image_loading": float(runtime.image_loading_seconds),
            "stereo_extraction": float(runtime.stereo_extraction_seconds),
            "temporal_matching": float(runtime.temporal_matching_seconds),
            "triangulation": float(runtime.triangulation_seconds),
            "ransac": float(runtime.ransac_seconds),
            "total": float(runtime.total_seconds),
        },
    }


def _write_candidate_path_evidence(
    path: Path, candidates: Sequence[RankedCandidate]
) -> None:
    pair_indices: List[int] = []
    covariances: List[np.ndarray] = []
    determinants: List[float] = []
    log_determinants: List[float] = []
    frame_ids_flat: List[int] = []
    frame_offsets = [0]
    edge_indices_flat: List[int] = []
    edge_offsets = [0]
    for candidate in candidates:
        candidate_path = candidate.score.path
        pair_indices.append(candidate.pair_index)
        covariances.append(
            np.asarray(candidate_path.covariance, dtype="<f8")
        )
        determinants.append(float(candidate_path.covariance_determinant))
        log_determinants.append(
            float(candidate_path.covariance_log_determinant)
        )
        frame_ids_flat.extend(int(value) for value in candidate_path.frame_ids)
        frame_offsets.append(len(frame_ids_flat))
        edge_indices_flat.extend(
            int(value) for value in candidate_path.edge_indices
        )
        edge_offsets.append(len(edge_indices_flat))
    covariance_array = (
        np.stack(covariances, axis=0)
        if covariances
        else np.empty((0, 6, 6), dtype="<f8")
    )
    np.savez_compressed(
        path,
        pair_indices=np.asarray(pair_indices, dtype="<i8"),
        accumulated_covariances=np.asarray(
            covariance_array, dtype="<f8"
        ),
        covariance_determinants=np.asarray(determinants, dtype="<f8"),
        covariance_log_determinants=np.asarray(
            log_determinants, dtype="<f8"
        ),
        path_frame_ids=np.asarray(frame_ids_flat, dtype="<i8"),
        path_frame_offsets=np.asarray(frame_offsets, dtype="<i8"),
        path_edge_indices=np.asarray(edge_indices_flat, dtype="<i8"),
        path_edge_offsets=np.asarray(edge_offsets, dtype="<i8"),
    )


def _save_successful_match_figure(path: Path, pair_run: PairRun) -> None:
    evidence = pair_run.evidence
    visualization = pair_run.visualization
    previous_image = np.asarray(visualization.previous_image_left)
    current_image = np.asarray(visualization.current_image_left)
    _require(
        previous_image.ndim == 2 and current_image.ndim == 2,
        "successful-match source images must be grayscale",
    )
    previous_height, previous_width = previous_image.shape
    current_height, current_width = current_image.shape
    canvas = np.zeros(
        (
            max(previous_height, current_height),
            previous_width + current_width,
        ),
        dtype=previous_image.dtype,
    )
    canvas[:previous_height, :previous_width] = previous_image
    canvas[:current_height, previous_width:] = current_image
    supporter_set = set(
        int(value) for value in visualization.supporter_indices
    )
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
        supporter = consensus_index in supporter_set
        axis.plot(
            [previous_point[0], current_point[0] + previous_width],
            [previous_point[1], current_point[1]],
            color="cyan" if supporter else "red",
            alpha=0.85 if supporter else 0.22,
            linewidth=0.8 if supporter else 0.35,
        )
    axis.axvline(previous_width, color="white", linewidth=0.5)
    axis.legend(
        handles=[
            Line2D([0], [0], color="cyan", label="supporter"),
            Line2D([0], [0], color="red", label="consensus non-supporter"),
        ],
        loc="lower center",
        ncol=2,
        framealpha=0.8,
    )
    axis.set_axis_off()
    axis.set_title(
        (
            "Accepted Loop Closure: FrameId {} \N{RIGHTWARDS ARROW} {}\n"
            "{} / {} supporters ({:.3f}%)"
        ).format(
            evidence.pair.previous_frame_id,
            evidence.pair.current_frame_id,
            evidence.counts.final_supporter_count,
            evidence.counts.four_view_consensus_count,
            evidence.counts.final_supporter_percentage,
        )
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_accepted_loop_artifact(
    path: Path,
    *,
    acceptance_id: int,
    pair_run: PairRun,
    bundle: SmallBundleResult,
    update: PoseGraphUpdateResult,
    frame_ids: Sequence[int],
) -> None:
    observations = tuple(bundle.candidate_observations)
    accepted = tuple(bundle.accepted_observations)
    visualization = pair_run.visualization
    consensus_four_view_pixels = tools.extract_actual_consensus_pixels(
        visualization.consensus,
        visualization.previous_stereo_inliers,
        visualization.current_stereo_inliers,
        visualization.previous_keypoints_left,
        visualization.previous_keypoints_right,
        visualization.current_keypoints_left,
        visualization.current_keypoints_right,
    )
    supporter_set = {
        int(value) for value in visualization.supporter_indices
    }
    ransac = pair_run.evidence.ransac
    _require(
        ransac.best_rotations is not None
        and ransac.best_translations is not None,
        "accepted loop lacks its retained RANSAC model",
    )
    rejected_items = sorted(
        bundle.rejection_reason_by_consensus_index.items()
    )
    np.savez_compressed(
        path,
        acceptance_id=np.asarray([acceptance_id], dtype="<i8"),
        previous_frame_id=np.asarray(
            [pair_run.evidence.pair.previous_frame_id], dtype="<i8"
        ),
        current_frame_id=np.asarray(
            [pair_run.evidence.pair.current_frame_id], dtype="<i8"
        ),
        previous_keyframe_ordinal=np.asarray(
            [pair_run.evidence.pair.previous_keyframe_ordinal], dtype="<i8"
        ),
        current_keyframe_ordinal=np.asarray(
            [pair_run.evidence.pair.current_keyframe_ordinal], dtype="<i8"
        ),
        state_version=np.asarray([update.state_version], dtype="<i8"),
        frame_ids=np.asarray(frame_ids, dtype="<i8"),
        optimized_poses=np.asarray(
            update.optimized_pose_matrices, dtype="<f8"
        ),
        marginal_covariances=np.asarray(
            update.marginal_covariances, dtype="<f8"
        ),
        initial_relative_pose=np.asarray(
            bundle.initial_relative_pose.matrix(), dtype="<f8"
        ),
        optimized_relative_pose=np.asarray(
            bundle.optimized_relative_pose.matrix(), dtype="<f8"
        ),
        conditional_covariance=np.asarray(
            bundle.conditional_covariance, dtype="<f8"
        ),
        candidate_consensus_indices=np.asarray(
            [item.consensus_index for item in observations], dtype="<i8"
        ),
        accepted_consensus_indices=np.asarray(
            [item.consensus_index for item in accepted], dtype="<i8"
        ),
        previous_stereo_measurements=np.asarray(
            [item.previous_measurement for item in observations], dtype="<f8"
        ),
        current_stereo_measurements=np.asarray(
            [item.current_measurement for item in observations], dtype="<f8"
        ),
        supporter_indices=np.asarray(
            visualization.supporter_indices, dtype="<i8"
        ),
        consensus_mapping=np.asarray(
            visualization.consensus, dtype="<i8"
        ).reshape((-1, 3)),
        consensus_four_view_pixels=np.asarray(
            consensus_four_view_pixels, dtype="<f8"
        ),
        consensus_supporter_mask=np.asarray(
            [
                consensus_index in supporter_set
                for consensus_index in range(
                    len(visualization.consensus)
                )
            ],
            dtype=np.uint8,
        ),
        ransac_best_rotations=np.asarray(
            ransac.best_rotations, dtype="<f8"
        ).reshape((4, 3, 3)),
        ransac_best_translations=np.asarray(
            ransac.best_translations, dtype="<f8"
        ).reshape((4, 3)),
        rejected_consensus_indices=np.asarray(
            [item[0] for item in rejected_items], dtype="<i8"
        ),
        rejected_reason_codes=np.asarray(
            [item[1] for item in rejected_items], dtype="<U128"
        ),
    )


def _hash_directory_files(
    directory: Path, *, exclude_names: Iterable[str] = ()
) -> Mapping[str, str]:
    excluded = set(exclude_names)
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _validate_candidate_path_evidence(
    path: Path,
    current: Mapping[str, object],
    *,
    candidate_graph: Optional[CovarianceGraph] = None,
) -> None:
    required = {
        "pair_indices",
        "accumulated_covariances",
        "covariance_determinants",
        "covariance_log_determinants",
        "path_frame_ids",
        "path_frame_offsets",
        "path_edge_indices",
        "path_edge_offsets",
    }
    with np.load(path, allow_pickle=False) as data:
        _require(
            set(data.files) == required,
            "candidate-path artifact members differ",
        )
        arrays = {
            name: np.asarray(data[name]).copy() for name in data.files
        }
    passing_rows = sorted(
        (
            row
            for row in current.get("candidates", [])
            if row.get("rank_within_current") is not None
        ),
        key=lambda row: int(row["rank_within_current"]),
    )
    expected_pair_indices = np.asarray(
        [int(row["pair_index"]) for row in passing_rows], dtype=np.int64
    )
    pair_indices = np.asarray(arrays["pair_indices"], dtype=np.int64)
    covariances = np.asarray(
        arrays["accumulated_covariances"], dtype=float
    )
    determinants = np.asarray(
        arrays["covariance_determinants"], dtype=float
    )
    log_determinants = np.asarray(
        arrays["covariance_log_determinants"], dtype=float
    )
    frame_ids = np.asarray(arrays["path_frame_ids"], dtype=np.int64)
    frame_offsets = np.asarray(
        arrays["path_frame_offsets"], dtype=np.int64
    )
    edge_indices = np.asarray(
        arrays["path_edge_indices"], dtype=np.int64
    )
    edge_offsets = np.asarray(
        arrays["path_edge_offsets"], dtype=np.int64
    )
    count = len(passing_rows)
    _require(
        np.array_equal(
            np.asarray(
                [int(row["rank_within_current"]) for row in passing_rows],
                dtype=np.int64,
            ),
            np.arange(1, count + 1, dtype=np.int64),
        )
        and np.array_equal(pair_indices, expected_pair_indices)
        and covariances.shape == (count, 6, 6)
        and determinants.shape == (count,)
        and log_determinants.shape == (count,)
        and frame_offsets.shape == (count + 1,)
        and edge_offsets.shape == (count + 1,)
        and frame_offsets[0] == 0
        and edge_offsets[0] == 0
        and frame_offsets[-1] == len(frame_ids)
        and edge_offsets[-1] == len(edge_indices)
        and np.all(np.diff(frame_offsets) >= 0)
        and np.all(np.diff(edge_offsets) >= 0)
        and np.all(np.isfinite(covariances))
        and np.all(np.isfinite(determinants))
        and np.all(np.isfinite(log_determinants)),
        "candidate-path array population or shape differs",
    )
    edge_count_before = int(current["candidate_graph_edge_count_before"])
    if candidate_graph is not None:
        _require(
            len(candidate_graph.edges) == edge_count_before,
            "candidate-path validation graph has the wrong edge count",
        )
    for index, row in enumerate(passing_rows):
        path_frames = frame_ids[
            frame_offsets[index] : frame_offsets[index + 1]
        ]
        path_edges = edge_indices[
            edge_offsets[index] : edge_offsets[index + 1]
        ]
        covariance = covariances[index]
        sign, observed_log_determinant = np.linalg.slogdet(covariance)
        observed_determinant = float(np.linalg.det(covariance))
        _require(
            len(path_frames) == len(path_edges) + 1
            and len(path_edges) >= 1
            and int(path_frames[0]) == int(row["previous_frame_id"])
            and int(path_frames[-1]) == int(row["current_frame_id"])
            and np.all(path_edges >= 0)
            and np.all(path_edges < edge_count_before)
            and np.max(np.abs(covariance - covariance.T)) <= 1e-9
            and sign > 0.0
            and np.isclose(
                determinants[index],
                observed_determinant,
                rtol=1e-12,
                atol=0.0,
            )
            and np.isclose(
                log_determinants[index],
                observed_log_determinant,
                rtol=1e-12,
                atol=1e-12,
            ),
            "one retained candidate path is internally inconsistent",
        )
        np.linalg.cholesky(covariance)
        if candidate_graph is not None:
            accumulated = np.zeros((6, 6), dtype=float)
            for path_position, edge_index in enumerate(path_edges):
                edge = candidate_graph.edge_by_index[int(edge_index)]
                _require(
                    edge.start_frame_id == int(path_frames[path_position])
                    and edge.end_frame_id
                    == int(path_frames[path_position + 1]),
                    "retained candidate path edge endpoints differ",
                )
                accumulated = accumulated + edge.covariance
            _require(
                np.array_equal(accumulated, covariance),
                "retained candidate path covariance is not the raw edge sum",
            )
            expected_path = (
                candidate_graph.find_course_covariance_path(
                    int(row["previous_frame_id"]),
                    int(row["current_frame_id"]),
                )
            )
            _require(
                tuple(path_frames.tolist()) == expected_path.frame_ids
                and tuple(path_edges.tolist())
                == expected_path.edge_indices
                and np.array_equal(
                    covariance, expected_path.covariance
                )
                and determinants[index]
                == expected_path.covariance_determinant
                and log_determinants[index]
                == expected_path.covariance_log_determinant,
                "retained path differs from the deterministic course path",
            )


def _validate_sealed_accepted_artifact(
    artifact_path: Path,
    accepted: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    arrays = _accepted_artifact_data(artifact_path)
    scalar_names = (
        "acceptance_id",
        "previous_frame_id",
        "current_frame_id",
        "previous_keyframe_ordinal",
        "current_keyframe_ordinal",
        "state_version",
    )
    _require(
        all(np.asarray(arrays[name]).shape == (1,) for name in scalar_names),
        "sealed accepted-loop scalar shape differs",
    )
    acceptance_id = int(arrays["acceptance_id"][0])
    previous_ordinal = int(
        arrays["previous_keyframe_ordinal"][0]
    )
    current_ordinal = int(arrays["current_keyframe_ordinal"][0])
    frame_ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
    _require(
        len(frame_ids) == EXPECTED_KEYFRAME_COUNT
        and int(frame_ids[0]) == 0
        and int(frame_ids[-1]) == 3359
        and 0 <= previous_ordinal < current_ordinal < len(frame_ids)
        and acceptance_id == int(accepted["acceptance_id"])
        and int(arrays["state_version"][0]) == acceptance_id
        and int(arrays["previous_frame_id"][0])
        == int(accepted["previous_frame_id"])
        == int(frame_ids[previous_ordinal])
        and int(arrays["current_frame_id"][0])
        == int(accepted["current_frame_id"])
        == int(frame_ids[current_ordinal])
        and previous_ordinal
        == int(accepted["previous_keyframe_ordinal"])
        and current_ordinal
        == int(accepted["current_keyframe_ordinal"])
        == int(current["current_ordinal"])
        and int(accepted["state_version_after"]) == acceptance_id
        and int(accepted["state_version_before"])
        == acceptance_id - 1
        and int(accepted["factor_count_after"])
        == BASE_POSE_GRAPH_FACTOR_COUNT + acceptance_id
        and accepted["factor_direction"]
        == "previous_FrameId_to_current_FrameId"
        and accepted["tangent_order"] == list(TANGENT_ORDER)
        and accepted["covariance_modified"] is False,
        "sealed accepted-loop endpoint/state contract differs",
    )
    _sealed_values_from_accepted_artifact(
        arrays,
        keyframe_ids=frame_ids,
        covariance_symmetry_atol=1e-9,
    )
    _validate_accepted_match_arrays(arrays, accepted)
    covariance = np.asarray(
        arrays["conditional_covariance"], dtype=float
    )
    _require(
        covariance.shape == (6, 6)
        and np.all(np.isfinite(covariance))
        and np.max(np.abs(covariance - covariance.T)) <= 1e-9
        and np.asarray(arrays["initial_relative_pose"]).shape == (4, 4)
        and np.asarray(arrays["optimized_relative_pose"]).shape == (4, 4),
        "sealed accepted-loop pose/covariance contract differs",
    )
    np.linalg.cholesky(covariance)
    gtsam.Pose3(np.asarray(arrays["initial_relative_pose"], dtype=float))
    gtsam.Pose3(
        np.asarray(arrays["optimized_relative_pose"], dtype=float)
    )
    for image_prefix in ("previous", "current"):
        image_path = REPOSITORY_ROOT / str(
            accepted["{}_left_image_path".format(image_prefix)]
        )
        _require(
            image_path.is_file()
            and sha256_file(image_path)
            == accepted["{}_left_image_sha256".format(image_prefix)],
            "sealed accepted-loop image provenance differs",
        )
    candidates = [
        row
        for row in current["candidates"]
        if int(row["pair_index"]) == int(accepted["pair_index"])
    ]
    matchers = [
        record
        for record in current["matcher_invocations"]
        if int(record["pair_index"]) == int(accepted["pair_index"])
        and record["outcome"] == "accepted"
    ]
    _require(
        len(candidates) == 1
        and candidates[0]["terminal_disposition"] == "accepted"
        and int(candidates[0]["accepted_anchor_id"]) == acceptance_id
        and len(matchers) == 1
        and int(matchers[0]["acceptance_id"]) == acceptance_id,
        "sealed accepted-loop candidate/matcher mapping differs",
    )


def validate_current_seal(
    current_directory: Path,
    *,
    candidate_graph: Optional[CovarianceGraph] = None,
) -> Mapping[str, object]:
    current_directory = Path(current_directory)
    seal_path = current_directory / "seal.json"
    _require(seal_path.is_file(), "current-keyframe seal is missing")
    seal = _read_json(seal_path)
    _require(
        seal.get("schema") == CURRENT_SEAL_SCHEMA
        and seal.get("status") == "sealed",
        "current-keyframe seal contract differs",
    )
    expected = seal.get("artifacts")
    observed = _hash_directory_files(
        current_directory, exclude_names=("seal.json",)
    )
    _require(
        isinstance(expected, dict) and expected == observed,
        "current-keyframe artifact hashes differ",
    )
    current = _read_json(current_directory / "current.json")
    _require(
        current.get("schema") == CURRENT_SCHEMA
        and current.get("seal_status") == "prepared-for-seal"
        and int(current.get("current_ordinal", -1))
        == int(seal.get("current_ordinal", -2)),
        "sealed current-keyframe summary contract differs",
    )
    _require(
        int(current.get("current_frame_id", -1))
        == int(seal.get("current_frame_id", -2)),
        "sealed current-keyframe FrameId differs",
    )
    accepted = current.get("accepted_loops")
    _require(
        isinstance(current.get("candidates"), list)
        and isinstance(current.get("matcher_invocations"), list)
        and isinstance(accepted, list),
        "sealed current-keyframe record collections differ",
    )
    expected_artifacts = {"current.json", "candidate_paths.npz"}
    for record in accepted:
        expected_artifacts.add(str(record["artifact"]))
        successful = record.get("successful_match_artifact")
        if successful is not None:
            expected_artifacts.add(str(successful))
    _require(
        set(observed) == expected_artifacts,
        "sealed current-keyframe artifact population differs",
    )
    _validate_candidate_path_evidence(
        current_directory / "candidate_paths.npz",
        current,
        candidate_graph=candidate_graph,
    )
    for record in accepted:
        _validate_sealed_accepted_artifact(
            current_directory / str(record["artifact"]),
            record,
            current,
        )
    return {"seal": seal, "current": current}


def _seal_current_attempt(
    attempt: Path,
    final_directory: Path,
    *,
    current_ordinal: int,
    current_frame_id: int,
) -> None:
    artifacts = _hash_directory_files(attempt)
    _write_json(
        attempt / "seal.json",
        {
            "schema": CURRENT_SEAL_SCHEMA,
            "schema_version": 1,
            "status": "sealed",
            "current_ordinal": int(current_ordinal),
            "current_frame_id": int(current_frame_id),
            "artifacts": artifacts,
        },
    )
    validate_current_seal(attempt)
    _require(
        not final_directory.exists(),
        "sealed current-keyframe directory already exists",
    )
    os.replace(str(attempt), str(final_directory))
    validate_current_seal(final_directory)


def _relative_repository_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _paths_overlap(first: Path, second: Path) -> bool:
    first = Path(first).resolve()
    second = Path(second).resolve()
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _require_safe_output_root(output_root: Path) -> Path:
    """Reject any scientific output target overlapping protected evidence."""

    selected = Path(output_root).resolve()
    _require(
        selected != REPOSITORY_ROOT.resolve(),
        "E7 scientific output root must not be the repository root",
    )
    for protected in PROTECTED_ARTIFACT_ROOTS:
        _require(
            not _paths_overlap(selected, protected),
            "E7 scientific output overlaps protected evidence: {}".format(
                protected
            ),
        )
    return selected


def _load_selected_e5_configuration(
    paths: Exercise7InputPaths, *, expected_sha256: str
) -> Tuple[Mapping[str, object], str]:
    """Load the configuration snapshot sealed inside the selected E5 root."""

    snapshot = _read_json(paths.e5_configuration)
    _require(
        isinstance(snapshot, dict)
        and snapshot.get("schema")
        == "exercise5-configuration-snapshot-v1"
        and snapshot.get("source_sha256") == expected_sha256
        and isinstance(snapshot.get("configuration"), dict)
        and snapshot["configuration"].get("schema")
        == "exercise5-production-config",
        "selected E5 configuration snapshot differs from its manifest",
    )
    return snapshot["configuration"], str(snapshot["source_sha256"])


def _production_identity(
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Mapping[str, object]:
    paths = Exercise7InputPaths.from_roots(
        e5_root=e5_root, e6_root=e6_root
    )
    protected_paths = (
        paths.e5_manifest,
        paths.e5_configuration,
        paths.e6_summary,
        paths.e6_values,
        paths.e6_marginals,
        CALIBRATION_PATH,
    )
    source_paths = (
        REPOSITORY_ROOT / "PoseGraph.py",
        REPOSITORY_ROOT / "tools.py",
        REPOSITORY_ROOT / "exercise7_candidate_detection.py",
        REPOSITORY_ROOT / "exercise7_consensus_matching.py",
        REPOSITORY_ROOT / "exercise7_small_bundle.py",
        REPOSITORY_ROOT / "exercise7_pose_graph_update.py",
        Path(__file__).resolve(),
    )
    for path in protected_paths + source_paths:
        _require(path.is_file(), "production input is missing: {}".format(path))
    return {
        "schema": "exercise7-i04-production-identity-v1",
        "protected_inputs_sha256": {
            _relative_repository_path(path): sha256_file(path)
            for path in protected_paths
        },
        "source_sha256": {
            _relative_repository_path(path): sha256_file(path)
            for path in source_paths
        },
    }


def _inspect_output_root(
    output_root: Path,
    policy: ProductionPolicy,
    identity: Mapping[str, object],
) -> Mapping[str, object]:
    """Inspect the exact run/resume contract without creating or writing."""

    selected = _require_safe_output_root(output_root)
    if not selected.exists():
        return {
            "status": "new-safe-root",
            "path": _relative_repository_path(selected),
            "exists": False,
        }

    policy_path = selected / "policy.json"
    identity_path = selected / "provenance.json"
    _require(
        selected.is_dir(),
        "E7-I04 output root exists but is not a directory",
    )
    _require(
        policy_path.is_file()
        and identity_path.is_file()
        and (selected / "currents").is_dir(),
        "existing E7-I04 root is not a resumable production root",
    )
    _require(
        _read_json(policy_path) == policy.record(),
        "existing E7-I04 production policy differs",
    )
    _require(
        _read_json(identity_path) == identity,
        "protected inputs or production sources changed since run start",
    )
    return {
        "status": "existing-exact-resumable-root",
        "path": _relative_repository_path(selected),
        "exists": True,
        "sealed_current_directory_count": len(
            tuple(
                path
                for path in (selected / "currents").glob(
                    "current_[0-9][0-9][0-9][0-9]"
                )
                if path.is_dir()
            )
        ),
        "final_package_present": (selected / "final").is_dir(),
    }


def _prepare_output_root(
    output_root: Path,
    policy: ProductionPolicy,
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Mapping[str, object]:
    output_root = _require_safe_output_root(output_root)
    policy.validate()
    identity = _production_identity(e5_root=e5_root, e6_root=e6_root)
    output_state = _inspect_output_root(output_root, policy, identity)
    if not output_state["exists"]:
        output_root.mkdir(parents=True)
        (output_root / "currents").mkdir()
        _write_json(output_root / "policy.json", policy.record())
        _write_json(output_root / "provenance.json", identity)
    return identity


def _load_baseline_runtime(
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Tuple[
    CandidateDetectionInputs,
    CovarianceGraph,
    CumulativePoseGraphState,
    Mapping[str, object],
    str,
    gtsam.Cal3_S2Stereo,
]:
    paths = Exercise7InputPaths.from_roots(
        e5_root=e5_root, e6_root=e6_root
    )
    candidate_inputs = load_candidate_detection_inputs(
        e6_root=paths.e6_root,
        e5_root=paths.e5_root,
    )
    _require(
        len(candidate_inputs.keyframe_ids) == EXPECTED_KEYFRAME_COUNT
        and candidate_inputs.keyframe_ids[0] == 0
        and candidate_inputs.keyframe_ids[-1] == 3359
        and len(candidate_inputs.edges) == BASE_COVARIANCE_EDGE_COUNT,
        "canonical E7-I01 input population differs",
    )
    candidate_graph = CovarianceGraph(
        candidate_inputs.keyframe_ids,
        candidate_inputs.edges,
        covariance_symmetry_atol=candidate_inputs.covariance_symmetry_atol,
    )
    protected = load_protected_e6_state(
        summary_path=paths.e6_summary,
        values_path=paths.e6_values,
        marginals_path=paths.e6_marginals,
        e5_root=paths.e5_root,
    )
    cumulative = CumulativePoseGraphState.from_protected_e6(protected)
    _require(
        tuple(protected.inputs.keyframe_ids)
        == tuple(candidate_inputs.keyframe_ids),
        "E6 and E7 canonical keyframe populations differ",
    )
    configuration, configuration_sha256 = _load_selected_e5_configuration(
        paths,
        expected_sha256=candidate_inputs.configuration_sha256,
    )
    _require(
        configuration_sha256
        == candidate_inputs.configuration_sha256,
        "candidate and Bundle configurations differ",
    )
    calibration = calibration_from_configuration(
        configuration, CALIBRATION_PATH
    )
    return (
        candidate_inputs,
        candidate_graph,
        cumulative,
        configuration,
        configuration_sha256,
        calibration,
    )


def preflight_exercise7(
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    policy: ProductionPolicy = ProductionPolicy(),
) -> Mapping[str, object]:
    """Validate a future run's inputs and output lane without writing.

    This builds only the already-sealed consecutive E6 graph in memory.  It
    performs no image matching, RANSAC, Bundle solve, Pose Graph optimization,
    Marginals construction, directory creation, or artifact mutation.
    """

    policy.validate()
    paths = Exercise7InputPaths.from_roots(
        e5_root=e5_root, e6_root=e6_root
    )
    identity = _production_identity(
        e5_root=paths.e5_root, e6_root=paths.e6_root
    )
    output_state = _inspect_output_root(output_root, policy, identity)
    (
        candidate_inputs,
        candidate_graph,
        cumulative_state,
        _,
        configuration_sha256,
        _,
    ) = _load_baseline_runtime(
        e5_root=paths.e5_root,
        e6_root=paths.e6_root,
    )
    _require(
        len(candidate_inputs.keyframe_ids) == EXPECTED_KEYFRAME_COUNT
        and len(candidate_graph.edges) == BASE_COVARIANCE_EDGE_COUNT
        and cumulative_state.graph.size()
        == BASE_POSE_GRAPH_FACTOR_COUNT
        and cumulative_state.marginal_covariances.shape
        == (EXPECTED_KEYFRAME_COUNT, 6, 6),
        "preflight baseline population differs from the frozen contract",
    )
    return {
        "schema": "exercise7-preflight-v1",
        "status": "pass",
        "selected_roots": {
            "e5": _relative_repository_path(paths.e5_root),
            "e6": _relative_repository_path(paths.e6_root),
            "e7_scientific": _relative_repository_path(output_root),
        },
        "configuration_sha256": configuration_sha256,
        "contracts": {
            "keyframe_count": len(candidate_inputs.keyframe_ids),
            "consecutive_covariance_edge_count": len(candidate_graph.edges),
            "baseline_pose_graph_factor_count": int(
                cumulative_state.graph.size()
            ),
            "baseline_marginal_count": int(
                cumulative_state.marginal_covariances.shape[0]
            ),
            "first_frame_id": int(candidate_inputs.keyframe_ids[0]),
            "last_frame_id": int(candidate_inputs.keyframe_ids[-1]),
        },
        "output_root": output_state,
        "production_identity": identity,
        "scientific_computation": {
            "matching": False,
            "ransac": False,
            "bundle_solve": False,
            "pose_graph_optimization": False,
            "marginals_construction": False,
        },
        "artifact_writes": False,
    }


def _accepted_artifact_data(path: Path) -> Mapping[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "acceptance_id",
            "previous_frame_id",
            "current_frame_id",
            "previous_keyframe_ordinal",
            "current_keyframe_ordinal",
            "state_version",
            "frame_ids",
            "optimized_poses",
            "marginal_covariances",
            "initial_relative_pose",
            "optimized_relative_pose",
            "conditional_covariance",
            "candidate_consensus_indices",
            "accepted_consensus_indices",
            "previous_stereo_measurements",
            "current_stereo_measurements",
            "supporter_indices",
            "consensus_mapping",
            "consensus_four_view_pixels",
            "consensus_supporter_mask",
            "ransac_best_rotations",
            "ransac_best_translations",
            "rejected_consensus_indices",
            "rejected_reason_codes",
        }
        _require(
            set(data.files) == required,
            "accepted-loop artifact members differ",
        )
        return {name: np.array(data[name], copy=True) for name in data.files}


def _sealed_values_from_accepted_artifact(
    arrays: Mapping[str, np.ndarray],
    *,
    keyframe_ids: Sequence[int],
    covariance_symmetry_atol: float,
) -> Tuple[gtsam.Values, np.ndarray, np.ndarray]:
    expected_frame_ids = np.asarray(keyframe_ids, dtype=np.int64)
    frame_ids = np.asarray(arrays["frame_ids"], dtype=np.int64)
    poses = np.asarray(arrays["optimized_poses"], dtype=float)
    marginals = np.asarray(
        arrays["marginal_covariances"], dtype=float
    )
    _require(
        np.array_equal(frame_ids, expected_frame_ids)
        and poses.shape == (len(keyframe_ids), 4, 4)
        and marginals.shape == (len(keyframe_ids), 6, 6)
        and np.all(np.isfinite(poses))
        and np.all(np.isfinite(marginals))
        and float(np.max(np.abs(marginals - marginals.transpose(0, 2, 1))))
        <= covariance_symmetry_atol,
        "accepted-loop sealed state arrays differ",
    )
    values = gtsam.Values()
    for frame_id, matrix in zip(keyframe_ids, poses):
        _require(
            np.max(
                np.abs(
                    matrix[3]
                    - np.asarray([0.0, 0.0, 0.0, 1.0])
                )
            )
            <= 1e-9
            and np.max(
                np.abs(
                    matrix[:3, :3].T @ matrix[:3, :3] - np.eye(3)
                )
            )
            <= 1e-9
            and abs(float(np.linalg.det(matrix[:3, :3])) - 1.0)
            <= 1e-9,
            "accepted-loop sealed pose is not rigid",
        )
        values.insert(int(camera_symbol(frame_id)), gtsam.Pose3(matrix))
    reconstructed = np.asarray(
        [
            values.atPose3(int(camera_symbol(frame_id))).matrix()
            for frame_id in keyframe_ids
        ],
        dtype=float,
    )
    _require(
        np.array_equal(reconstructed, poses),
        "accepted-loop sealed poses do not round-trip exactly through Values",
    )
    return values, poses.copy(), marginals.copy()


def _validate_accepted_match_arrays(
    arrays: Mapping[str, np.ndarray],
    accepted: Mapping[str, object],
) -> None:
    consensus_count = int(accepted["consensus_count"])
    supporter_count = int(accepted["supporter_count"])
    eligible_count = int(accepted["eligible_landmark_count"])
    rejected_count = int(accepted["rejected_landmark_count"])
    supporter_indices = np.asarray(
        arrays["supporter_indices"], dtype=np.int64
    )
    supporter_mask = np.asarray(
        arrays["consensus_supporter_mask"], dtype=np.uint8
    )
    candidate_indices = np.asarray(
        arrays["candidate_consensus_indices"], dtype=np.int64
    )
    accepted_indices = np.asarray(
        arrays["accepted_consensus_indices"], dtype=np.int64
    )
    rejected_indices = np.asarray(
        arrays["rejected_consensus_indices"], dtype=np.int64
    )
    rejected_reasons = np.asarray(arrays["rejected_reason_codes"])
    _require(
        np.asarray(arrays["consensus_mapping"]).shape
        == (consensus_count, 3)
        and np.asarray(arrays["consensus_four_view_pixels"]).shape
        == (4, 2, consensus_count)
        and supporter_mask.shape == (consensus_count,)
        and set(np.unique(supporter_mask)).issubset({0, 1})
        and len(supporter_indices) == supporter_count
        and np.isclose(
            float(accepted["supporter_percentage"]),
            (
                100.0 * supporter_count / consensus_count
                if consensus_count
                else 0.0
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        and len(set(supporter_indices.tolist())) == supporter_count
        and np.all(supporter_indices >= 0)
        and np.all(supporter_indices < consensus_count)
        and np.array_equal(
            np.flatnonzero(supporter_mask), supporter_indices
        )
        and np.array_equal(candidate_indices, supporter_indices)
        and len(accepted_indices) == eligible_count
        and len(rejected_indices) == rejected_count
        and len(rejected_reasons) == rejected_count
        and set(accepted_indices.tolist()).isdisjoint(
            set(rejected_indices.tolist())
        )
        and set(accepted_indices.tolist()).union(
            set(rejected_indices.tolist())
        )
        == set(candidate_indices.tolist())
        and np.asarray(arrays["previous_stereo_measurements"]).shape
        == (supporter_count, 3)
        and np.asarray(arrays["current_stereo_measurements"]).shape
        == (supporter_count, 3)
        and np.asarray(arrays["ransac_best_rotations"]).shape
        == (4, 3, 3)
        and np.asarray(arrays["ransac_best_translations"]).shape
        == (4, 3)
        and all(
            np.all(np.isfinite(np.asarray(arrays[name], dtype=float)))
            for name in (
                "consensus_four_view_pixels",
                "previous_stereo_measurements",
                "current_stereo_measurements",
                "ransac_best_rotations",
                "ransac_best_translations",
            )
        ),
        "accepted-loop match/supporter evidence differs",
    )
    for rotation in np.asarray(
        arrays["ransac_best_rotations"], dtype=float
    ):
        _require(
            np.max(np.abs(rotation.T @ rotation - np.eye(3))) <= 1e-6
            and abs(float(np.linalg.det(rotation)) - 1.0) <= 1e-6,
            "accepted-loop retained RANSAC rotation is not rigid",
        )


def _validate_sealed_detection_state(
    current: Mapping[str, object],
    *,
    baseline_inputs: CandidateDetectionInputs,
    candidate_graph: CovarianceGraph,
    state: CumulativePoseGraphState,
) -> None:
    current_ordinal = int(current["current_ordinal"])
    current_frame_id = int(current["current_frame_id"])
    eligible_rows = [
        row
        for row in current["candidates"]
        if int(row["ordinal_gap"]) >= MINIMUM_TEMPORAL_ORDINAL_GAP
    ]
    if eligible_rows:
        live_inputs = candidate_detection_inputs_from_values(
            baseline_inputs, state.optimized_values
        )
        scores = score_previous_keyframes(
            live_inputs,
            candidate_graph,
            state.optimized_values,
            current_frame_id,
            previous_frame_ids=[
                int(row["previous_frame_id"]) for row in eligible_rows
            ],
        )
    else:
        scores = ()
    score_by_pair = {
        (
            int(score.previous_keyframe_ordinal),
            int(score.current_keyframe_ordinal),
        ): score
        for score in scores
    }
    _require(
        len(scores) == len(eligible_rows)
        and len(score_by_pair) == len(scores),
        "sealed detection score population differs on replay",
    )
    expected_passing = sorted(
        (
            score
            for score in scores
            if score.mahalanobis_squared
            < MAHALANOBIS_SQUARED_THRESHOLD
        ),
        key=lambda score: (
            score.mahalanobis_squared,
            score.previous_keyframe_ordinal,
            score.previous_frame_id,
        ),
    )
    expected_rank = {
        (
            int(score.previous_keyframe_ordinal),
            current_ordinal,
        ): rank
        for rank, score in enumerate(expected_passing, start=1)
    }
    for row in eligible_rows:
        key = (
            int(row["previous_keyframe_ordinal"]),
            current_ordinal,
        )
        score = score_by_pair[key]
        rank = expected_rank.get(key)
        _require(
            float(row["factor_error"]) == score.factor_error
            and float(row["mahalanobis_squared"])
            == score.mahalanobis_squared
            and float(row["mahalanobis"]) == score.mahalanobis
            and (
                int(row["rank_within_current"])
                if row["rank_within_current"] is not None
                else None
            )
            == rank,
            "sealed detection score or deterministic rank differs",
        )


def _replay_sealed_currents(
    output_root: Path,
    baseline_inputs: CandidateDetectionInputs,
    candidate_graph: CovarianceGraph,
    state: CumulativePoseGraphState,
) -> Tuple[
    CovarianceGraph,
    CumulativePoseGraphState,
    List[AcceptedLoopAnchor],
    Set[Tuple[int, int]],
    int,
]:
    """Rebuild the exact durable scientific prefix from validated current seals.

    Each accepted artifact is replayed in acceptance order to reconstruct the
    cumulative Pose Graph, raw covariance edges, suppression anchors, and exact
    evaluated-pair set.  Unsealed attempt directories are never resume state.
    """

    anchors: List[AcceptedLoopAnchor] = []
    evaluated_pairs: Set[Tuple[int, int]] = set()
    next_current_ordinal = 0
    currents_root = Path(output_root) / "currents"
    tolerance = baseline_inputs.pose_reconstruction_atol
    for current_ordinal, current_frame_id in enumerate(
        baseline_inputs.keyframe_ids
    ):
        directory = currents_root / "current_{:04d}".format(current_ordinal)
        if not directory.exists():
            break
        sealed = validate_current_seal(
            directory, candidate_graph=candidate_graph
        )
        current = sealed["current"]
        state_version_before = state.state_version
        _require(
            int(current["current_ordinal"]) == current_ordinal
            and int(current["current_frame_id"]) == current_frame_id
            and int(current["state_version_before"])
            == state_version_before
            and int(current["candidate_graph_edge_count_before"])
            == len(candidate_graph.edges)
            and int(current["candidate_count"]) == current_ordinal,
            "sealed current ordering or input state differs",
        )
        candidate_rows = current.get("candidates", [])
        candidates_by_pair = {
            int(row["pair_index"]): row for row in candidate_rows
        }
        _require(
            len(candidates_by_pair) == len(candidate_rows)
            and all(
                int(row["current_keyframe_ordinal"]) == current_ordinal
                and int(row["current_frame_id"]) == current_frame_id
                and 0
                <= int(row["previous_keyframe_ordinal"])
                < current_ordinal
                and int(row["pair_index"])
                == ordered_pair_index(
                    int(row["previous_keyframe_ordinal"]),
                    current_ordinal,
                )
                and int(row["previous_frame_id"])
                == int(
                    baseline_inputs.keyframe_ids[
                        int(row["previous_keyframe_ordinal"])
                    ]
                )
                and int(row["ordinal_gap"])
                == current_ordinal
                - int(row["previous_keyframe_ordinal"])
                and int(row["frame_id_gap"])
                == current_frame_id - int(row["previous_frame_id"])
                and int(row["detection_state_version"])
                == state_version_before
                for row in candidate_rows
            ),
            "sealed candidate-row provenance differs",
        )
        _validate_sealed_detection_state(
            current,
            baseline_inputs=baseline_inputs,
            candidate_graph=candidate_graph,
            state=state,
        )
        matcher_records = current.get("matcher_invocations", [])
        _require(
            int(current["matcher_invocation_count"])
            == len(matcher_records)
            and sorted(
                int(match["matcher_slot"]) for match in matcher_records
            )
            == list(range(1, len(matcher_records) + 1)),
            "sealed matcher-slot accounting differs",
        )
        for match in matcher_records:
            key = (
                int(match["previous_frame_id"]),
                int(match["current_frame_id"]),
            )
            pair_index = int(match["pair_index"])
            row = candidates_by_pair.get(pair_index)
            _require(
                key not in evaluated_pairs
                and row is not None
                and int(match["current_keyframe_ordinal"])
                == current_ordinal
                and key[1] == current_frame_id
                and int(match["previous_keyframe_ordinal"])
                == int(row["previous_keyframe_ordinal"])
                and key[0] == int(row["previous_frame_id"])
                and int(match["rank_within_current"])
                == int(row["rank_within_current"])
                and bool(row["matcher_slot_consumed"])
                and int(row["matcher_slot"])
                == int(match["matcher_slot"])
                and str(row["terminal_disposition"])
                == str(match["outcome"]),
                "a sealed matcher pair appears more than once",
            )
            evaluated_pairs.add(key)
        accepted_records = current.get("accepted_loops", [])
        _require(
            int(current["accepted_loop_count"]) == len(accepted_records),
            "sealed accepted-loop count differs",
        )
        for accepted in accepted_records:
            artifact_path = directory / accepted["artifact"]
            arrays = _accepted_artifact_data(artifact_path)
            for scalar_name in (
                "acceptance_id",
                "previous_frame_id",
                "current_frame_id",
                "previous_keyframe_ordinal",
                "current_keyframe_ordinal",
                "state_version",
            ):
                _require(
                    np.asarray(arrays[scalar_name]).shape == (1,),
                    "accepted-loop scalar array shape differs",
                )
            acceptance_id = int(arrays["acceptance_id"][0])
            previous_frame_id = int(arrays["previous_frame_id"][0])
            accepted_current_frame_id = int(
                arrays["current_frame_id"][0]
            )
            previous_ordinal = int(
                arrays["previous_keyframe_ordinal"][0]
            )
            accepted_current_ordinal = int(
                arrays["current_keyframe_ordinal"][0]
            )
            pair_index = int(accepted["pair_index"])
            row = candidates_by_pair.get(pair_index)
            matching_matcher_records = [
                match
                for match in matcher_records
                if int(match["pair_index"]) == pair_index
                and match["outcome"] == "accepted"
            ]
            _require(
                acceptance_id == state.state_version + 1
                and int(arrays["state_version"][0]) == acceptance_id
                and int(accepted["acceptance_id"]) == acceptance_id
                and int(accepted["previous_frame_id"])
                == previous_frame_id
                and int(accepted["current_frame_id"])
                == accepted_current_frame_id
                and int(accepted["previous_keyframe_ordinal"])
                == previous_ordinal
                and int(accepted["current_keyframe_ordinal"])
                == accepted_current_ordinal
                and accepted_current_ordinal == current_ordinal
                and accepted_current_frame_id == current_frame_id,
                "accepted-loop replay ordering differs",
            )
            _require(
                0 <= previous_ordinal < current_ordinal
                and int(
                    baseline_inputs.keyframe_ids[previous_ordinal]
                )
                == previous_frame_id
                and row is not None
                and row["terminal_disposition"] == "accepted"
                and previous_ordinal
                == int(row["previous_keyframe_ordinal"])
                and previous_frame_id
                == int(row["previous_frame_id"])
                and accepted_current_ordinal
                == int(row["current_keyframe_ordinal"])
                and accepted_current_frame_id
                == int(row["current_frame_id"])
                and int(row["accepted_anchor_id"]) == acceptance_id
                and len(matching_matcher_records) == 1
                and int(
                    matching_matcher_records[0]["acceptance_id"]
                )
                == acceptance_id,
                "accepted-loop endpoint or cross-record mapping differs",
            )
            restored_values, restored_poses, restored_marginals = (
                _sealed_values_from_accepted_artifact(
                    arrays,
                    keyframe_ids=baseline_inputs.keyframe_ids,
                    covariance_symmetry_atol=(
                        baseline_inputs.covariance_symmetry_atol
                    ),
                )
            )
            _validate_accepted_match_arrays(arrays, accepted)
            for image_prefix in ("previous", "current"):
                image_path_value = str(
                    accepted["{}_left_image_path".format(image_prefix)]
                )
                image_path = REPOSITORY_ROOT / image_path_value
                _require(
                    image_path.is_file()
                    and sha256_file(image_path)
                    == accepted[
                        "{}_left_image_sha256".format(image_prefix)
                    ],
                    "accepted-loop source image provenance differs",
                )
            covariance = np.asarray(
                arrays["conditional_covariance"], dtype=float
            )
            _require(
                covariance.shape == (6, 6)
                and np.all(np.isfinite(covariance))
                and np.max(np.abs(covariance - covariance.T))
                <= baseline_inputs.covariance_symmetry_atol,
                "accepted-loop raw covariance differs",
            )
            np.linalg.cholesky(covariance)
            prospective_graph = candidate_graph.with_directed_edge(
                previous_frame_id,
                accepted_current_frame_id,
                covariance,
                source_directory=artifact_path,
            )
            update = state.apply_loop_factor(
                previous_frame_id=previous_frame_id,
                current_frame_id=accepted_current_frame_id,
                relative_pose=gtsam.Pose3(
                    arrays["optimized_relative_pose"]
                ),
                conditional_covariance=covariance,
            )
            _require(
                update.state_version == acceptance_id
                and float(
                    np.max(
                        np.abs(
                            update.optimized_pose_matrices
                            - arrays["optimized_poses"]
                        )
                    )
                )
                <= tolerance
                and float(
                    np.max(
                        np.abs(
                            update.marginal_covariances
                            - arrays["marginal_covariances"]
                        )
                    )
                )
                <= max(tolerance, 1e-9),
                "accepted-loop numerical replay differs from its seal",
            )
            state = CumulativePoseGraphState(
                inputs=state.inputs,
                graph=update.graph,
                optimized_values=restored_values,
                optimized_pose_matrices=restored_poses,
                marginal_covariances=restored_marginals,
                state_version=acceptance_id,
                base_factor_count=state.base_factor_count,
            )
            candidate_graph = prospective_graph
            anchors.append(
                AcceptedLoopAnchor(
                    acceptance_id=acceptance_id,
                    previous_frame_id=previous_frame_id,
                    current_frame_id=accepted_current_frame_id,
                    previous_ordinal=previous_ordinal,
                    current_ordinal=accepted_current_ordinal,
                    state_version=acceptance_id,
                )
            )
        _require(
            int(current["state_version_after"]) == state.state_version
            and state.state_version
            == state_version_before + len(accepted_records)
            and state.graph.size()
            == BASE_POSE_GRAPH_FACTOR_COUNT + state.state_version
            and int(current["candidate_graph_edge_count_after"])
            == len(candidate_graph.edges)
            and len(candidate_graph.edges)
            == BASE_COVARIANCE_EDGE_COUNT + state.state_version
            and len(anchors) == state.state_version,
            "sealed current output state version differs",
        )
        next_current_ordinal = current_ordinal + 1
    for later_ordinal in range(
        next_current_ordinal, len(baseline_inputs.keyframe_ids)
    ):
        _require(
            not (
                currents_root
                / "current_{:04d}".format(later_ordinal)
            ).exists(),
            "sealed currents are not a contiguous prefix",
        )
    _require(
        len(candidate_graph.edges)
        == BASE_COVARIANCE_EDGE_COUNT + state.state_version,
        "replayed covariance graph edge count differs",
    )
    return (
        candidate_graph,
        state,
        anchors,
        evaluated_pairs,
        next_current_ordinal,
    )


def _cluster_suppress_row(
    row: MutableMapping[str, object],
    candidate: RankedCandidate,
    anchor: AcceptedLoopAnchor,
    *,
    reason: str,
) -> None:
    row["accepted_anchor_id"] = anchor.acceptance_id
    row["accepted_anchor_delta_previous_ordinal"] = (
        candidate.previous_ordinal - anchor.previous_ordinal
    )
    row["accepted_anchor_delta_current_ordinal"] = (
        candidate.current_ordinal - anchor.current_ordinal
    )
    row["terminal_disposition"] = "cluster_suppressed"
    row["terminal_reason"] = reason


def _process_current_keyframe(
    *,
    output_root: Path,
    current_ordinal: int,
    baseline_inputs: CandidateDetectionInputs,
    candidate_graph: CovarianceGraph,
    state: CumulativePoseGraphState,
    configuration: Mapping[str, object],
    calibration: gtsam.Cal3_S2Stereo,
    anchors: List[AcceptedLoopAnchor],
    evaluated_pairs: Set[Tuple[int, int]],
    policy: ProductionPolicy,
) -> CovarianceGraph:
    """Process one current from frozen detection through an atomic durable seal.

    Detection scores/ranks are computed once from the current entry snapshot.
    Candidate matching and small-Bundle work is temporary: ordinary matcher or
    measurement failures receive terminal evidence but cannot change the
    cumulative Pose Graph, covariance graph, or suppression anchors.  A passing
    measurement proposes one raw-covariance loop factor; the Pose Graph holder
    publishes it only after immediate LM and all-node Marginals succeed.  The
    accepted covariance edge and suppression anchor are then published, and
    all current evidence is validated and atomically sealed.

    Multiple accepted loops in one current use the latest optimized Values for
    Bundle initialization, but do not cause that current's frozen detection
    population to be rescored.  Later currents see the latest published Pose
    Graph and covariance graph.  The returned graph is therefore the detection
    graph for the next current; durable recovery still depends solely on seals.
    """

    started = time.perf_counter()
    current_frame_id = int(
        baseline_inputs.keyframe_ids[current_ordinal]
    )
    final_directory = (
        Path(output_root)
        / "currents"
        / "current_{:04d}".format(current_ordinal)
    )
    _require(
        not final_directory.exists(),
        "current-keyframe output already exists before processing",
    )
    attempt = Path(
        tempfile.mkdtemp(
            prefix=".current_{:04d}.attempt.".format(current_ordinal),
            dir=str(Path(output_root) / "currents"),
        )
    )
    # Phase 1: establish the current's immutable detection snapshot identity.
    # The held scientific state is the latest sealed/replayed state; scores are
    # computed once from it below and never refreshed within this current.
    state_version_before = int(state.state_version)
    candidate_edge_count_before = len(candidate_graph.edges)
    _require(
        state.graph.size()
        == BASE_POSE_GRAPH_FACTOR_COUNT + state_version_before
        and candidate_edge_count_before
        == BASE_COVARIANCE_EDGE_COUNT + state_version_before
        and len(anchors) == state_version_before,
        "current-keyframe input graph/state counts differ",
    )
    # Phase 2: enumerate the complete ordered previous/current pair population.
    # Every pair receives one terminal disposition before this current is sealed.
    rows: List[Dict[str, object]] = [
        _base_candidate_row(
            keyframe_ids=baseline_inputs.keyframe_ids,
            previous_ordinal=previous_ordinal,
            current_ordinal=current_ordinal,
            state_version=state_version_before,
        )
        for previous_ordinal in range(current_ordinal)
    ]
    rows_by_pair_index = {
        int(row["pair_index"]): row for row in rows
    }
    threshold_candidates: List[RankedCandidate] = []
    matcher_records: List[Dict[str, object]] = []
    accepted_records: List[Dict[str, object]] = []
    try:
        # Phase 3: exclude the previous 15 keyframe ordinals. This uses ordinal
        # distance, while candidate identities and graph nodes remain FrameIds.
        temporally_eligible_ids: List[int] = []
        for previous_ordinal, row in enumerate(rows):
            if (
                current_ordinal - previous_ordinal
                < policy.temporal_ordinal_gap_minimum
            ):
                row["terminal_disposition"] = "temporal_excluded"
                row["terminal_reason"] = (
                    "current_ordinal_minus_previous_ordinal_below_16"
                )
            else:
                temporally_eligible_ids.append(
                    int(baseline_inputs.keyframe_ids[previous_ordinal])
                )

        # Phase 4: snapshot the latest Pose Graph Values and score every eligible
        # pair using the course covariance path and GTSAM Mahalanobis residual.
        # This single batch freezes detection for the remainder of the current.
        if temporally_eligible_ids:
            live_inputs = candidate_detection_inputs_from_values(
                baseline_inputs, state.optimized_values
            )
            scores = score_previous_keyframes(
                live_inputs,
                candidate_graph,
                state.optimized_values,
                current_frame_id,
                previous_frame_ids=temporally_eligible_ids,
            )
            scores = tuple(
                sorted(
                    scores,
                    key=lambda score: (
                        score.mahalanobis_squared,
                        score.previous_keyframe_ordinal,
                        score.previous_frame_id,
                    ),
                )
            )
            # Phase 5: apply the strict d_squared < 36,000,000 gate, then rank
            # deterministically by score, previous ordinal, and previous FrameId.
            passing_scores = [
                score
                for score in scores
                if score.mahalanobis_squared
                < policy.mahalanobis_squared_threshold
            ]
            rank_by_pair = {
                (
                    score.previous_frame_id,
                    score.current_frame_id,
                ): rank
                for rank, score in enumerate(passing_scores, start=1)
            }
            for score in scores:
                pair_index = ordered_pair_index(
                    score.previous_keyframe_ordinal,
                    score.current_keyframe_ordinal,
                )
                row = rows_by_pair_index[pair_index]
                row["factor_error"] = float(score.factor_error)
                row["mahalanobis_squared"] = float(
                    score.mahalanobis_squared
                )
                row["mahalanobis"] = float(score.mahalanobis)
                if (
                    score.mahalanobis_squared
                    >= policy.mahalanobis_squared_threshold
                ):
                    row["terminal_disposition"] = (
                        "mahalanobis_rejected"
                    )
                    row["terminal_reason"] = (
                        "mahalanobis_squared_not_strictly_below_36000000"
                    )
                    continue
                rank = rank_by_pair[
                    (score.previous_frame_id, score.current_frame_id)
                ]
                row["rank_within_current"] = rank
                threshold_candidates.append(
                    RankedCandidate(
                        pair_index=pair_index,
                        score=score,
                        rank_within_current=rank,
                    )
                )

        # Phase 6: suppress exact prior matcher invocations and candidates inside
        # durable +/-10 accepted-anchor rectangles. Only integrated loops create
        # these persistent, anchored, nontransitive suppression regions.
        live_for_regions: List[RankedCandidate] = []
        for candidate in threshold_candidates:
            row = rows_by_pair_index[candidate.pair_index]
            exact_key = (
                candidate.previous_frame_id,
                candidate.current_frame_id,
            )
            if exact_key in evaluated_pairs:
                row["terminal_disposition"] = "duplicate_suppressed"
                row["terminal_reason"] = (
                    "exact_ordered_pair_has_prior_sealed_matcher_invocation"
                )
                continue
            accepted_anchor = accepted_anchor_for_candidate(
                candidate, anchors, policy
            )
            if accepted_anchor is not None:
                _cluster_suppress_row(
                    row,
                    candidate,
                    accepted_anchor,
                    reason="inside_preexisting_accepted_anchor",
                )
                continue
            live_for_regions.append(candidate)

        # Phase 7: form rank-ordered temporary anchored proposal regions. These
        # candidate-local regions organize scheduling and never become durable
        # suppression merely because a candidate was considered or rejected.
        regions = form_temporary_proposal_regions(
            live_for_regions, policy
        )
        for region in regions:
            for local_rank, candidate in enumerate(
                region.candidates, start=1
            ):
                row = rows_by_pair_index[candidate.pair_index]
                row["temporary_region_id"] = region.region_id
                row["temporary_region_anchor_previous_ordinal"] = (
                    region.anchor_previous_ordinal
                )
                row["temporary_region_anchor_current_ordinal"] = (
                    region.anchor_current_ordinal
                )
                row["temporary_region_local_rank"] = local_rank

        # Phase 8: define the fair region-first schedule: one live head from
        # each region precedes same-region fallback, with liveness rechecked
        # after any accepted loop publishes a new suppression anchor.
        def candidate_is_live(
            candidate: RankedCandidate,
            region: TemporaryProposalRegion,
            scheduler_pass: int,
        ) -> bool:
            del region
            row = rows_by_pair_index[candidate.pair_index]
            if row["terminal_disposition"] is not None:
                return False
            exact_key = (
                candidate.previous_frame_id,
                candidate.current_frame_id,
            )
            if exact_key in evaluated_pairs:
                row["scheduler_pass"] = scheduler_pass
                row["terminal_disposition"] = "duplicate_suppressed"
                row["terminal_reason"] = (
                    "exact_ordered_pair_has_prior_matcher_invocation"
                )
                return False
            accepted_anchor = accepted_anchor_for_candidate(
                candidate, anchors, policy
            )
            if accepted_anchor is not None:
                row["scheduler_pass"] = scheduler_pass
                _cluster_suppress_row(
                    row,
                    candidate,
                    accepted_anchor,
                    reason="inside_accepted_anchor_before_matcher_call",
                )
                return False
            return True

        def evaluate_live_candidate(
            candidate: RankedCandidate,
            region: TemporaryProposalRegion,
            scheduler_pass: int,
            matcher_slot: int,
        ) -> None:
            nonlocal candidate_graph
            # Phase 9: consume one matcher slot only for this actual invocation.
            # Score calculation, suppression, and scheduler consideration do
            # not count toward the maximum-three per-current matcher budget.
            row = rows_by_pair_index[candidate.pair_index]
            row["scheduler_pass"] = scheduler_pass
            row["matcher_slot"] = matcher_slot
            row["matcher_slot_consumed"] = True
            exact_key = (
                candidate.previous_frame_id,
                candidate.current_frame_id,
            )
            _require(
                exact_key not in evaluated_pairs,
                "live candidate became an exact duplicate",
            )
            evaluated_pairs.add(exact_key)
            pair = candidate_pair_from_score(
                candidate.score,
                pair_index=candidate.pair_index,
                rank_within_current=candidate.rank_within_current,
                category="production_candidate",
                selection_reason=(
                    "E7-I04 region {} pass {} matcher slot {}"
                ).format(
                    region.region_id, scheduler_pass, matcher_slot
                ),
            )
            validate_candidate_pair(
                pair, canonical_frame_ids=baseline_inputs.keyframe_ids
            )
            matcher_record: Dict[str, object] = {
                "pair_index": candidate.pair_index,
                "previous_frame_id": candidate.previous_frame_id,
                "current_frame_id": candidate.current_frame_id,
                "previous_keyframe_ordinal": candidate.previous_ordinal,
                "current_keyframe_ordinal": candidate.current_ordinal,
                "rank_within_current": candidate.rank_within_current,
                "temporary_region_id": region.region_id,
                "scheduler_pass": scheduler_pass,
                "matcher_slot": matcher_slot,
                "seed": policy.pair_seed,
                "maximum_hypothesis_attempts": (
                    policy.maximum_hypothesis_attempts
                ),
            }
            # Candidate-local matching: direct four-view consensus and bounded
            # RANSAC retain evidence but cannot yet mutate scientific state.
            try:
                pair_run = evaluate_candidate_pair(
                    pair,
                    seed=policy.pair_seed,
                    maximum_attempts=policy.maximum_hypothesis_attempts,
                )
                validate_pair_evidence(pair_run.evidence)
            except (
                FileNotFoundError,
                ConsensusMatchingError,
                cv2.error,
            ) as error:
                row["terminal_disposition"] = "matcher_failed"
                row["terminal_reason"] = "{}:{}".format(
                    type(error).__name__, str(error)
                )
                matcher_record.update(
                    {
                        "outcome": "matcher_failed",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
                matcher_records.append(matcher_record)
                return

            # Phase 10: apply the complete-model plus inclusive 30/30/30
            # consensus/supporter policy. Search status is evidence, not a
            # substitute for these scientific gates.
            matcher_record.update(_pair_evidence_record(pair_run.evidence))
            accepted_by_matcher, match_reason = (
                production_match_policy_decision(pair_run.evidence)
            )
            matcher_record["match_policy_reason"] = match_reason
            if not accepted_by_matcher:
                row["terminal_disposition"] = "matcher_rejected"
                row["terminal_reason"] = match_reason
                matcher_record["outcome"] = "matcher_rejected"
                matcher_records.append(matcher_record)
                return

            # Phase 11: build the candidate-local two-pose stereo Bundle
            # measurement. The latest cumulative relative pose initializes the
            # current camera only; stereo factors determine the measurement.
            initial_relative_pose = state.initial_relative_pose(
                candidate.previous_frame_id,
                candidate.current_frame_id,
            )
            try:
                observations = observations_from_pair_run(pair_run)
                bundle = solve_small_bundle(
                    observations=observations,
                    initial_relative_pose=initial_relative_pose,
                    calibration=calibration,
                    configuration=configuration,
                    expected_supporter_count=len(observations),
                    previous_frame_id=candidate.previous_frame_id,
                    current_frame_id=candidate.current_frame_id,
                )
            except (
                SmallBundleValidationError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as error:
                # Phase 12: retain a candidate-local measurement failure. No
                # Pose Graph factor, covariance edge, or suppression anchor was
                # published, so later candidates see unchanged scientific state.
                row["terminal_disposition"] = "measurement_failed"
                row["terminal_reason"] = "{}:{}".format(
                    type(error).__name__, str(error)
                )
                matcher_record.update(
                    {
                        "outcome": "measurement_failed",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                )
                matcher_records.append(matcher_record)
                return

            # Phase 13: prepare one accepted-loop transaction: one directed
            # previous-to-current factor with the raw conditional covariance,
            # plus an immutable prospective detection-graph edge.
            next_acceptance_id = state.state_version + 1
            covariance = np.asarray(
                bundle.conditional_covariance, dtype=float
            )
            artifact_name = "accepted_{:04d}.npz".format(
                next_acceptance_id
            )
            prospective_candidate_graph = (
                candidate_graph.with_directed_edge(
                    candidate.previous_frame_id,
                    candidate.current_frame_id,
                    covariance,
                    source_directory=(
                        final_directory / artifact_name
                    ),
                )
            )
            _require(
                len(prospective_candidate_graph.edges)
                == BASE_COVARIANCE_EDGE_COUNT + next_acceptance_id,
                "prospective candidate graph did not gain exactly one edge",
            )
            # Phase 14: apply_loop_factor performs immediate LM, frame-0 gauge
            # normalization, and all 356 Marginals on a working graph; its held
            # Pose Graph state changes only after that full proposal validates.
            update = state.apply_loop_factor(
                previous_frame_id=candidate.previous_frame_id,
                current_frame_id=candidate.current_frame_id,
                relative_pose=bundle.optimized_relative_pose,
                conditional_covariance=covariance,
            )
            _require(
                update.state_version == next_acceptance_id
                and update.pre_update_state_version
                == next_acceptance_id - 1,
                "cumulative Pose Graph state version did not advance once",
            )
            _require(
                update.graph.size()
                == BASE_POSE_GRAPH_FACTOR_COUNT + next_acceptance_id,
                "accepted loop did not add exactly one Pose Graph factor",
            )
            # Stage the accepted numerical checkpoint inside this current's
            # temporary directory. It is not durable until the phase-18 seal.
            _save_accepted_loop_artifact(
                attempt / artifact_name,
                acceptance_id=next_acceptance_id,
                pair_run=pair_run,
                bundle=bundle,
                update=update,
                frame_ids=baseline_inputs.keyframe_ids,
            )
            successful_match_artifact = None
            if next_acceptance_id == 1:
                successful_match_artifact = "successful_match.png"
                _save_successful_match_figure(
                    attempt / successful_match_artifact, pair_run
                )
            # Phase 15: publish the byte-preserved raw covariance edge for
            # FUTURE detection. The present current's scores remain frozen.
            candidate_graph = prospective_candidate_graph
            # Phase 16: publish the accepted anchor only after measurement,
            # Pose Graph transaction, Marginals, and staged artifact succeed.
            # This is the sole creation point for persistent cluster suppression.
            anchor = AcceptedLoopAnchor(
                acceptance_id=next_acceptance_id,
                previous_frame_id=candidate.previous_frame_id,
                current_frame_id=candidate.current_frame_id,
                previous_ordinal=candidate.previous_ordinal,
                current_ordinal=candidate.current_ordinal,
                state_version=next_acceptance_id,
            )
            anchors.append(anchor)
            row["terminal_disposition"] = "accepted"
            row["terminal_reason"] = (
                "matching_bundle_pose_graph_marginals_and_edge_passed"
            )
            row["accepted_anchor_id"] = next_acceptance_id
            accepted_record = {
                "acceptance_id": next_acceptance_id,
                "artifact": artifact_name,
                "successful_match_artifact": successful_match_artifact,
                "pair_index": candidate.pair_index,
                "previous_frame_id": candidate.previous_frame_id,
                "current_frame_id": candidate.current_frame_id,
                "previous_keyframe_ordinal": candidate.previous_ordinal,
                "current_keyframe_ordinal": candidate.current_ordinal,
                "consensus_count": (
                    pair_run.evidence.counts.four_view_consensus_count
                ),
                "supporter_count": (
                    pair_run.evidence.counts.final_supporter_count
                ),
                "supporter_percentage": (
                    pair_run.evidence.counts.final_supporter_percentage
                ),
                "ransac_terminal_status": (
                    pair_run.evidence.ransac.status
                ),
                "eligible_landmark_count": len(
                    bundle.accepted_observations
                ),
                "rejected_landmark_count": len(
                    bundle.rejection_reason_by_consensus_index
                ),
                "factor_direction": (
                    "previous_FrameId_to_current_FrameId"
                ),
                "tangent_order": list(TANGENT_ORDER),
                "covariance_modified": False,
                "state_version_before": update.pre_update_state_version,
                "state_version_after": update.state_version,
                "factor_count_after": int(update.graph.size()),
                "graph_error_before_candidate_factor": (
                    update.original_graph_error
                ),
                "graph_error_after_factor_before_optimization": (
                    update.augmented_pre_optimization_error
                ),
                "graph_error_after_optimization": (
                    update.augmented_post_optimization_error
                ),
                "loop_factor_error_before_optimization": (
                    update.loop_factor_pre_optimization_error
                ),
                "loop_factor_error_after_optimization": (
                    update.loop_factor_post_optimization_error
                ),
                "marginal_count": int(
                    update.marginal_covariances.shape[0]
                ),
                "marginal_symmetry_max_abs": (
                    update.marginal_symmetry_max_abs
                ),
                "marginal_minimum_eigenvalue": (
                    update.marginal_minimum_eigenvalue
                ),
                "small_bundle_initial_error": bundle.initial_error,
                "small_bundle_final_error": bundle.final_error,
            }
            previous_left_image_path = (
                REPOSITORY_ROOT
                / "dataset"
                / "sequences"
                / "00"
                / "image_0"
                / "{:06d}.png".format(candidate.previous_frame_id)
            )
            current_left_image_path = (
                REPOSITORY_ROOT
                / "dataset"
                / "sequences"
                / "00"
                / "image_0"
                / "{:06d}.png".format(candidate.current_frame_id)
            )
            _require(
                previous_left_image_path.is_file()
                and current_left_image_path.is_file(),
                "accepted-loop source image is missing",
            )
            accepted_record.update(
                {
                    "previous_left_image_path": (
                        _relative_repository_path(
                            previous_left_image_path
                        )
                    ),
                    "previous_left_image_sha256": sha256_file(
                        previous_left_image_path
                    ),
                    "current_left_image_path": (
                        _relative_repository_path(current_left_image_path)
                    ),
                    "current_left_image_sha256": sha256_file(
                        current_left_image_path
                    ),
                    "consensus_pixel_layout": (
                        "[previous_left,previous_right,current_left,"
                        "current_right] x [u,v] x consensus_index"
                    ),
                }
            )
            accepted_records.append(accepted_record)
            matcher_record.update(
                {
                    "outcome": "accepted",
                    "acceptance_id": next_acceptance_id,
                    "eligible_landmark_count": len(
                        bundle.accepted_observations
                    ),
                    "measurement_artifact": artifact_name,
                }
            )
            matcher_records.append(matcher_record)

        # Phase 8 execution: run the frozen fair schedule under the cap of three
        # actual matcher invocations; callbacks perform phases 9 through 16.
        matcher_invocation_count = evaluate_region_schedule(
            regions,
            maximum_invocations=(
                policy.maximum_matcher_invocations_per_current
            ),
            candidate_is_live=candidate_is_live,
            evaluate_live_candidate=evaluate_live_candidate,
        )
        _require(
            matcher_invocation_count == len(matcher_records)
            and matcher_invocation_count
            <= policy.maximum_matcher_invocations_per_current,
            "matcher invocation accounting differs",
        )

        for candidate in threshold_candidates:
            row = rows_by_pair_index[candidate.pair_index]
            if row["terminal_disposition"] is not None:
                continue
            accepted_anchor = accepted_anchor_for_candidate(
                candidate, anchors, policy
            )
            if accepted_anchor is not None:
                _cluster_suppress_row(
                    row,
                    candidate,
                    accepted_anchor,
                    reason="inside_accepted_anchor_after_scheduler",
                )
            else:
                row["terminal_disposition"] = "cap_not_evaluated"
                row["terminal_reason"] = (
                    "three_matcher_invocation_cap_exhausted"
                    if matcher_invocation_count
                    >= policy.maximum_matcher_invocations_per_current
                    else "no_remaining_live_scheduler_offer"
                )

        _require(
            len(rows) == current_ordinal
            and all(
                row["terminal_disposition"] is not None for row in rows
            ),
            "current-keyframe candidate dispositions are incomplete",
        )
        expected_acceptance_ids = list(
            range(
                state_version_before + 1,
                state_version_before + len(accepted_records) + 1,
            )
        )
        _require(
            [int(record["acceptance_id"]) for record in accepted_records]
            == expected_acceptance_ids
            and state.state_version
            == state_version_before + len(accepted_records)
            and state.graph.size()
            == BASE_POSE_GRAPH_FACTOR_COUNT + state.state_version
            and len(candidate_graph.edges)
            == BASE_COVARIANCE_EDGE_COUNT + state.state_version
            and len(anchors) == state.state_version,
            "current-keyframe output graph/state counts differ",
        )
        # Phase 17: assemble complete candidate dispositions, path evidence,
        # matcher outcomes, accepted checkpoints, provenance, and state counts
        # in the candidate-local attempt directory.
        _write_candidate_path_evidence(
            attempt / "candidate_paths.npz", threshold_candidates
        )
        disposition_counts: Dict[str, int] = {}
        for row in rows:
            disposition = str(row["terminal_disposition"])
            disposition_counts[disposition] = (
                disposition_counts.get(disposition, 0) + 1
            )
        current_record = {
            "schema": CURRENT_SCHEMA,
            "schema_version": 1,
            "seal_status": "prepared-for-seal",
            "current_ordinal": current_ordinal,
            "current_frame_id": current_frame_id,
            "state_version_before": state_version_before,
            "state_version_after": state.state_version,
            "candidate_graph_edge_count_before": (
                candidate_edge_count_before
            ),
            "candidate_graph_edge_count_after": len(
                candidate_graph.edges
            ),
            "candidate_count": len(rows),
            "temporally_eligible_count": sum(
                row["terminal_disposition"] != "temporal_excluded"
                for row in rows
            ),
            "mahalanobis_passing_count": len(threshold_candidates),
            "temporary_region_count": len(regions),
            "matcher_invocation_count": matcher_invocation_count,
            "accepted_loop_count": len(accepted_records),
            "disposition_counts": disposition_counts,
            "candidates": rows,
            "matcher_invocations": matcher_records,
            "accepted_loops": accepted_records,
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
        _write_json(attempt / "current.json", current_record)
        # Phase 18: validate and atomically rename the current attempt. Only this
        # sealed state is durable/resumable or visible to a later current.
        _seal_current_attempt(
            attempt,
            final_directory,
            current_ordinal=current_ordinal,
            current_frame_id=current_frame_id,
        )
        return candidate_graph
    except BaseException:
        if attempt.exists():
            shutil.rmtree(attempt)
        raise


def _collect_complete_records(
    output_root: Path, keyframe_ids: Sequence[int]
) -> Tuple[
    List[Mapping[str, object]],
    List[Mapping[str, object]],
    List[Mapping[str, object]],
    List[Mapping[str, object]],
]:
    currents: List[Mapping[str, object]] = []
    candidates: List[Mapping[str, object]] = []
    matcher_invocations: List[Mapping[str, object]] = []
    accepted_loops: List[Mapping[str, object]] = []
    for current_ordinal, current_frame_id in enumerate(keyframe_ids):
        directory = (
            Path(output_root)
            / "currents"
            / "current_{:04d}".format(current_ordinal)
        )
        sealed = validate_current_seal(directory)
        current = sealed["current"]
        _require(
            int(current["current_ordinal"]) == current_ordinal
            and int(current["current_frame_id"]) == current_frame_id,
            "final current-keyframe ordering differs",
        )
        currents.append(current)
        candidates.extend(current["candidates"])
        matcher_invocations.extend(current["matcher_invocations"])
        for record in current["accepted_loops"]:
            normalized = dict(record)
            normalized["artifact_path"] = _relative_repository_path(
                directory / record["artifact"]
            )
            successful = record.get("successful_match_artifact")
            normalized["successful_match_artifact_path"] = (
                _relative_repository_path(directory / successful)
                if successful
                else None
            )
            accepted_loops.append(normalized)
    return currents, candidates, matcher_invocations, accepted_loops


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return value


def _write_mapping_csv(
    path: Path,
    records: Sequence[Mapping[str, object]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    fields = (
        list(fieldnames)
        if fieldnames is not None
        else sorted(
            {str(key) for record in records for key in record.keys()}
        )
    )
    _require(bool(fields), "CSV schema cannot be empty")
    expected = set(fields)
    _require(
        all(set(record).issubset(expected) for record in records),
        "CSV record contains a field outside its frozen schema",
    )
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: _csv_value(record.get(field))
                    for field in fields
                }
            )


def _load_baseline_arrays(
    *, e6_root: Path = DEFAULT_E6_ROOT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_e6_root = Path(e6_root).resolve()
    values_path = selected_e6_root / "pose_graph_values.npz"
    marginals_path = selected_e6_root / "pose_graph_marginals.npz"
    with np.load(values_path, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        poses = np.asarray(data["optimized_poses"], dtype=float)
    with np.load(marginals_path, allow_pickle=False) as data:
        marginal_frame_ids = np.asarray(
            data["frame_ids"], dtype=np.int64
        )
        marginals = np.asarray(
            data["marginal_covariances"], dtype=float
        )
    _require(
        np.array_equal(frame_ids, marginal_frame_ids)
        and poses.shape == (EXPECTED_KEYFRAME_COUNT, 4, 4)
        and marginals.shape == (EXPECTED_KEYFRAME_COUNT, 6, 6)
        and np.all(np.isfinite(poses))
        and np.all(np.isfinite(marginals)),
        "protected E6 report arrays are invalid",
    )
    return frame_ids, poses, marginals


def _read_ground_truth(path: Path) -> Tuple[np.ndarray, ...]:
    extrinsics = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        pose = np.asarray(line.split(), dtype=float).reshape(3, 4)
        extrinsics.append(
            np.vstack((pose, np.asarray([0.0, 0.0, 0.0, 1.0])))
        )
    return tuple(extrinsics)


def _truth_keyframe_poses(
    ground_truth_path: Path, frame_ids: Sequence[int]
) -> np.ndarray:
    raw = _read_ground_truth(ground_truth_path)
    _require(
        len(raw) > int(frame_ids[-1]),
        "Ground Truth does not cover the final canonical FrameId",
    )
    return np.asarray(
        [raw[0] @ np.linalg.inv(raw[int(frame_id)]) for frame_id in frame_ids],
        dtype=float,
    )


def _rotation_error_degrees(
    estimates: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    errors = []
    for estimate, reference in zip(estimates, truth):
        error_rotation = (
            np.linalg.inv(reference) @ estimate
        )[:3, :3]
        cosine = np.clip(
            (float(np.trace(error_rotation)) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        errors.append(float(np.degrees(np.arccos(cosine))))
    return np.asarray(errors, dtype=float)


def _absolute_error_arrays(
    estimates: np.ndarray, truth: np.ndarray
) -> Mapping[str, np.ndarray]:
    signed = estimates[:, :3, 3] - truth[:, :3, 3]
    return {
        "absolute_xyz_m": np.abs(signed),
        "location_norm_m": np.linalg.norm(signed, axis=1),
        "rotation_error_deg": _rotation_error_degrees(estimates, truth),
    }


def _uncertainty_sizes(
    marginal_covariances: np.ndarray,
) -> Mapping[str, np.ndarray]:
    _require(
        marginal_covariances.ndim == 3
        and marginal_covariances.shape[1:] == (6, 6)
        and np.all(np.isfinite(marginal_covariances)),
        "Marginal covariance array is invalid",
    )
    rotation_trace = np.trace(
        marginal_covariances[:, :3, :3], axis1=1, axis2=2
    )
    location_trace = np.trace(
        marginal_covariances[:, 3:, 3:], axis1=1, axis2=2
    )
    _require(
        np.all(rotation_trace >= 0.0)
        and np.all(location_trace >= 0.0),
        "raw Marginal block trace is negative",
    )
    rotation_rad = np.sqrt(rotation_trace)
    return {
        "location_m": np.sqrt(location_trace),
        "rotation_rad": rotation_rad,
        "rotation_deg": np.degrees(rotation_rad),
    }


def _summary_statistics(values: np.ndarray) -> Mapping[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "final": float(values[-1]),
    }


def _plot_trajectory_comparison(
    output_path: Path,
    truth_poses: np.ndarray,
    baseline_poses: np.ndarray,
    final_poses: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    axis.plot(
        truth_poses[:, 0, 3],
        truth_poses[:, 2, 3],
        label="Ground Truth",
    )
    axis.plot(
        baseline_poses[:, 0, 3],
        baseline_poses[:, 2, 3],
        label="Pose Graph without Loop Closure",
    )
    axis.plot(
        final_poses[:, 0, 3],
        final_poses[:, 2, 3],
        label="Pose Graph with Loop Closure",
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Canonical keyframe trajectories")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_absolute_errors(
    output_path: Path,
    frame_ids: np.ndarray,
    baseline_errors: Mapping[str, np.ndarray],
    final_errors: Mapping[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(
        frame_ids,
        baseline_errors["location_norm_m"],
        label="without Loop Closure",
    )
    axes[0].plot(
        frame_ids,
        final_errors["location_norm_m"],
        label="with Loop Closure",
    )
    axes[0].set_ylabel("absolute location error [m]")
    axes[0].legend()
    axes[1].plot(
        frame_ids,
        baseline_errors["rotation_error_deg"],
        label="without Loop Closure",
    )
    axes[1].plot(
        frame_ids,
        final_errors["rotation_error_deg"],
        label="with Loop Closure",
    )
    axes[1].set_xlabel("FrameId")
    axes[1].set_ylabel("absolute rotation error [deg]")
    axes[1].legend()
    figure.suptitle("Absolute Pose Graph errors")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_uncertainty_comparison(
    output_path: Path,
    frame_ids: np.ndarray,
    baseline_uncertainty: Mapping[str, np.ndarray],
    final_uncertainty: Mapping[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(
        frame_ids,
        baseline_uncertainty["location_m"],
        label="without Loop Closure",
    )
    axes[0].plot(
        frame_ids,
        final_uncertainty["location_m"],
        label="with Loop Closure",
    )
    axes[0].set_ylabel(r"$\sqrt{\mathrm{trace}(\Sigma_t)}$ [m]")
    axes[0].legend()
    axes[1].plot(
        frame_ids,
        baseline_uncertainty["rotation_deg"],
        label="without Loop Closure",
    )
    axes[1].plot(
        frame_ids,
        final_uncertainty["rotation_deg"],
        label="with Loop Closure",
    )
    axes[1].set_xlabel("FrameId")
    axes[1].set_ylabel(r"$\sqrt{\mathrm{trace}(\Sigma_r)}$ [deg]")
    axes[1].legend()
    figure.suptitle("Raw Pose3 Marginal uncertainty size")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _snapshot_slot_ids(accepted_count: int) -> Tuple[int, int, int, int]:
    if accepted_count == 0:
        return (0, 0, 0, 0)
    if accepted_count == 1:
        return (0, 1, 1, 1)
    if accepted_count == 2:
        return (0, 1, 1, 2)
    return (0, 1, int(math.ceil(accepted_count / 2.0)), accepted_count)


def _snapshot_fallback_reason(
    slot_name: str, accepted_count: int
) -> Optional[str]:
    if accepted_count == 0 and slot_name in ("first", "middle"):
        return "zero accepted loops; protected E6 fallback"
    if accepted_count == 0 and slot_name == "final":
        return "zero accepted loops; protected E6 is the final state"
    if accepted_count == 1 and slot_name == "middle":
        return "one accepted loop; first accepted state reused"
    if accepted_count == 2 and slot_name == "middle":
        return "two accepted loops; earlier of the two middle states"
    return None


def _load_accepted_state(
    artifact_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    arrays = _accepted_artifact_data(artifact_path)
    return (
        np.asarray(arrays["optimized_poses"], dtype=float),
        np.asarray(arrays["marginal_covariances"], dtype=float),
    )


def _plot_result_aware_snapshots(
    output_path: Path,
    slots: Sequence[Mapping[str, object]],
    state_by_id: Mapping[int, Tuple[np.ndarray, np.ndarray]],
) -> None:
    location_sizes = [
        _uncertainty_sizes(state_by_id[int(slot["state_id"])][1])[
            "location_m"
        ]
        for slot in slots
    ]
    minimum = min(float(np.min(values)) for values in location_sizes)
    maximum = max(float(np.max(values)) for values in location_sizes)
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter = None
    for axis, slot, location_size in zip(
        axes.ravel(), slots, location_sizes
    ):
        poses = state_by_id[int(slot["state_id"])][0]
        axis.plot(poses[:, 0, 3], poses[:, 2, 3], color="0.65")
        scatter = axis.scatter(
            poses[:, 0, 3],
            poses[:, 2, 3],
            c=location_size,
            s=10,
            cmap="viridis",
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        axis.set_aspect("equal", adjustable="box")
        suffix = (
            "\n{}".format(slot["fallback_reason"])
            if slot.get("fallback_reason")
            else ""
        )
        axis.set_title("{}{}".format(slot["label"], suffix))
    _require(scatter is not None, "snapshot plot contains no states")
    figure.colorbar(
        scatter,
        ax=list(axes.ravel()),
        label=r"raw $\sqrt{\mathrm{trace}(\Sigma_t)}$ [m]",
        shrink=0.85,
    )
    figure.suptitle("Result-aware Pose Graph states with raw Marginals")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _build_report_artifacts(
    report_root: Path,
    *,
    frame_ids: np.ndarray,
    baseline_poses: np.ndarray,
    baseline_marginals: np.ndarray,
    final_poses: np.ndarray,
    final_marginals: np.ndarray,
    ground_truth_path: Path,
    accepted_loops: Sequence[Mapping[str, object]],
    e6_values_path: Path,
    e6_marginals_path: Path,
) -> Mapping[str, object]:
    report_root.mkdir(parents=True)
    figures_root = report_root / "figures"
    data_root = report_root / "data"
    figures_root.mkdir()
    data_root.mkdir()

    truth_poses = _truth_keyframe_poses(ground_truth_path, frame_ids)
    baseline_errors = _absolute_error_arrays(
        baseline_poses, truth_poses
    )
    final_errors = _absolute_error_arrays(final_poses, truth_poses)
    baseline_uncertainty = _uncertainty_sizes(baseline_marginals)
    final_uncertainty = _uncertainty_sizes(final_marginals)

    error_records = []
    uncertainty_records = []
    for index, frame_id in enumerate(frame_ids):
        error_records.append(
            {
                "keyframe_ordinal": index,
                "frame_id": int(frame_id),
                "without_loop_absolute_x_error_m": (
                    baseline_errors["absolute_xyz_m"][index, 0]
                ),
                "without_loop_absolute_y_error_m": (
                    baseline_errors["absolute_xyz_m"][index, 1]
                ),
                "without_loop_absolute_z_error_m": (
                    baseline_errors["absolute_xyz_m"][index, 2]
                ),
                "without_loop_location_error_norm_m": (
                    baseline_errors["location_norm_m"][index]
                ),
                "without_loop_rotation_error_deg": (
                    baseline_errors["rotation_error_deg"][index]
                ),
                "with_loop_absolute_x_error_m": (
                    final_errors["absolute_xyz_m"][index, 0]
                ),
                "with_loop_absolute_y_error_m": (
                    final_errors["absolute_xyz_m"][index, 1]
                ),
                "with_loop_absolute_z_error_m": (
                    final_errors["absolute_xyz_m"][index, 2]
                ),
                "with_loop_location_error_norm_m": (
                    final_errors["location_norm_m"][index]
                ),
                "with_loop_rotation_error_deg": (
                    final_errors["rotation_error_deg"][index]
                ),
            }
        )
        uncertainty_records.append(
            {
                "keyframe_ordinal": index,
                "frame_id": int(frame_id),
                "without_loop_location_uncertainty_m": (
                    baseline_uncertainty["location_m"][index]
                ),
                "with_loop_location_uncertainty_m": (
                    final_uncertainty["location_m"][index]
                ),
                "without_loop_rotation_uncertainty_rad": (
                    baseline_uncertainty["rotation_rad"][index]
                ),
                "with_loop_rotation_uncertainty_rad": (
                    final_uncertainty["rotation_rad"][index]
                ),
                "without_loop_rotation_uncertainty_deg": (
                    baseline_uncertainty["rotation_deg"][index]
                ),
                "with_loop_rotation_uncertainty_deg": (
                    final_uncertainty["rotation_deg"][index]
                ),
            }
        )
    _write_mapping_csv(
        data_root / "absolute_pose_errors.csv", error_records
    )
    _write_mapping_csv(
        data_root / "marginal_uncertainty_sizes.csv",
        uncertainty_records,
    )
    trajectory_records = [
        {
            "keyframe_ordinal": index,
            "frame_id": int(frame_id),
            "ground_truth_x_m": truth_poses[index, 0, 3],
            "ground_truth_y_m": truth_poses[index, 1, 3],
            "ground_truth_z_m": truth_poses[index, 2, 3],
            "without_loop_x_m": baseline_poses[index, 0, 3],
            "without_loop_y_m": baseline_poses[index, 1, 3],
            "without_loop_z_m": baseline_poses[index, 2, 3],
            "with_loop_x_m": final_poses[index, 0, 3],
            "with_loop_y_m": final_poses[index, 1, 3],
            "with_loop_z_m": final_poses[index, 2, 3],
        }
        for index, frame_id in enumerate(frame_ids)
    ]
    _write_mapping_csv(
        data_root / "trajectory_comparison.csv", trajectory_records
    )

    _plot_trajectory_comparison(
        figures_root / "trajectory_comparison.png",
        truth_poses,
        baseline_poses,
        final_poses,
    )
    _plot_absolute_errors(
        figures_root / "absolute_pose_errors.png",
        frame_ids,
        baseline_errors,
        final_errors,
    )
    _plot_uncertainty_comparison(
        figures_root / "marginal_uncertainty_comparison.png",
        frame_ids,
        baseline_uncertainty,
        final_uncertainty,
    )

    accepted_count = len(accepted_loops)
    state_by_id: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        0: (baseline_poses, baseline_marginals)
    }
    source_by_id: Dict[int, Mapping[str, object]] = {
        0: {
            "state_id": 0,
            "source_values": _relative_repository_path(e6_values_path),
            "source_marginals": _relative_repository_path(
                e6_marginals_path
            ),
            "source_sha256": {
                "values": sha256_file(e6_values_path),
                "marginals": sha256_file(e6_marginals_path),
            },
        }
    }
    accepted_by_id = {
        int(record["acceptance_id"]): record for record in accepted_loops
    }
    _require(
        set(accepted_by_id) == set(range(1, accepted_count + 1)),
        "accepted-loop IDs are not contiguous",
    )
    for acceptance_id, record in accepted_by_id.items():
        artifact_path = REPOSITORY_ROOT / str(record["artifact_path"])
        state_by_id[acceptance_id] = _load_accepted_state(artifact_path)
        source_by_id[acceptance_id] = {
            "state_id": acceptance_id,
            "accepted_pair": [
                int(record["previous_frame_id"]),
                int(record["current_frame_id"]),
            ],
            "source_artifact": str(record["artifact_path"]),
            "source_sha256": sha256_file(artifact_path),
        }
    slot_names = ("baseline", "first", "middle", "final")
    slot_labels = (
        "Protected E6 baseline",
        "Immediately after first accepted loop",
        "Deterministic middle accepted-loop state",
        "Final state",
    )
    selected_ids = _snapshot_slot_ids(accepted_count)
    slots = []
    for name, label, state_id in zip(
        slot_names, slot_labels, selected_ids
    ):
        slot = {
            "slot": name,
            "label": label,
            "state_id": state_id,
            "source": source_by_id[state_id],
            "fallback_reason": _snapshot_fallback_reason(
                name, accepted_count
            ),
        }
        slots.append(slot)
    _write_json(
        data_root / "pose_graph_snapshot_selection.json",
        {
            "schema": "exercise7-i04-result-aware-snapshots-v1",
            "accepted_loop_count": accepted_count,
            "acceptance_order": (
                "current_ordinal_then_within_current_matcher_slot"
            ),
            "even_middle_tie_break": "earlier_middle_acceptance",
            "slots": slots,
        },
    )
    _plot_result_aware_snapshots(
        figures_root / "result_aware_pose_graph_snapshots.png",
        slots,
        state_by_id,
    )

    successful_match_path = None
    if accepted_count:
        successful_match_path = accepted_by_id[1].get(
            "successful_match_artifact_path"
        )
        _require(
            successful_match_path is not None
            and (REPOSITORY_ROOT / successful_match_path).is_file(),
            "first accepted loop lacks its successful-match figure",
        )

    metrics = {
        "without_loop": {
            "location_error_m": _summary_statistics(
                baseline_errors["location_norm_m"]
            ),
            "absolute_rotation_error_deg": _summary_statistics(
                baseline_errors["rotation_error_deg"]
            ),
            "location_uncertainty_m": _summary_statistics(
                baseline_uncertainty["location_m"]
            ),
            "rotation_uncertainty_deg": _summary_statistics(
                baseline_uncertainty["rotation_deg"]
            ),
        },
        "with_loop": {
            "location_error_m": _summary_statistics(
                final_errors["location_norm_m"]
            ),
            "absolute_rotation_error_deg": _summary_statistics(
                final_errors["rotation_error_deg"]
            ),
            "location_uncertainty_m": _summary_statistics(
                final_uncertainty["location_m"]
            ),
            "rotation_uncertainty_deg": _summary_statistics(
                final_uncertainty["rotation_deg"]
            ),
        },
    }
    report_summary = {
        "schema": "exercise7-i04-report-artifacts-v1",
        "successful_match_visualization": {
            "status": "created" if successful_match_path else "not_applicable",
            "path": successful_match_path,
            "reason": (
                None
                if successful_match_path
                else "zero accepted Loop Closures"
            ),
        },
        "snapshot_selection": slots,
        "uncertainty_measure": {
            "location": "sqrt(trace(raw Sigma[3:6,3:6]))",
            "location_unit": "m",
            "rotation": "sqrt(trace(raw Sigma[0:3,0:3]))",
            "rotation_native_unit": "rad",
            "rotation_plot_unit": "deg",
            "covariance_modified": False,
        },
        "metrics": metrics,
        "figures": [
            "figures/trajectory_comparison.png",
            "figures/absolute_pose_errors.png",
            "figures/marginal_uncertainty_comparison.png",
            "figures/result_aware_pose_graph_snapshots.png",
        ],
        "data": [
            "data/trajectory_comparison.csv",
            "data/absolute_pose_errors.csv",
            "data/marginal_uncertainty_sizes.csv",
            "data/pose_graph_snapshot_selection.json",
        ],
    }
    _write_json(report_root / "summary.json", report_summary)
    return report_summary


def _technical_validation(
    *,
    currents: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    matcher_invocations: Sequence[Mapping[str, object]],
    accepted_loops: Sequence[Mapping[str, object]],
    candidate_graph: CovarianceGraph,
    state: CumulativePoseGraphState,
    baseline_poses: np.ndarray,
    baseline_marginals: np.ndarray,
) -> Mapping[str, object]:
    accepted_count = len(accepted_loops)
    pair_indices = [int(row["pair_index"]) for row in candidates]
    _require(
        len(currents) == EXPECTED_KEYFRAME_COUNT
        and len(candidates) == EXPECTED_ORDERED_PAIR_COUNT
        and set(pair_indices) == set(range(EXPECTED_ORDERED_PAIR_COUNT))
        and len(pair_indices) == len(set(pair_indices)),
        "complete canonical candidate population does not reconcile",
    )
    _require(
        all(row.get("terminal_disposition") for row in candidates),
        "a canonical pair lacks its terminal disposition",
    )
    accepted_by_id = {
        int(record["acceptance_id"]): record
        for record in accepted_loops
    }
    _require(
        len(accepted_by_id) == accepted_count
        and set(accepted_by_id) == set(range(1, accepted_count + 1)),
        "accepted-loop IDs are not contiguous",
    )
    running_version = 0
    global_matcher_pairs: Set[Tuple[int, int]] = set()
    for current_ordinal, current in enumerate(currents):
        current_frame_id = int(
            state.inputs.keyframe_ids[current_ordinal]
        )
        rows = current["candidates"]
        matches = current["matcher_invocations"]
        accepted = current["accepted_loops"]
        rows_by_pair = {
            int(row["pair_index"]): row for row in rows
        }
        expected_pair_indices = list(
            range(
                current_ordinal * (current_ordinal - 1) // 2,
                current_ordinal * (current_ordinal + 1) // 2,
            )
        )
        observed_dispositions: Dict[str, int] = {}
        for row in rows:
            disposition = str(row["terminal_disposition"])
            observed_dispositions[disposition] = (
                observed_dispositions.get(disposition, 0) + 1
            )
            _require(
                int(row["current_keyframe_ordinal"])
                == current_ordinal
                and int(row["current_frame_id"]) == current_frame_id
                and 0
                <= int(row["previous_keyframe_ordinal"])
                < current_ordinal
                and int(row["pair_index"])
                == ordered_pair_index(
                    int(row["previous_keyframe_ordinal"]),
                    current_ordinal,
                )
                and int(row["previous_frame_id"])
                == int(
                    state.inputs.keyframe_ids[
                        int(row["previous_keyframe_ordinal"])
                    ]
                )
                and int(row["ordinal_gap"])
                == current_ordinal
                - int(row["previous_keyframe_ordinal"])
                and int(row["frame_id_gap"])
                == current_frame_id - int(row["previous_frame_id"])
                and int(row["detection_state_version"])
                == running_version,
                "candidate row current/state provenance differs",
            )
            ordinal_gap = int(row["ordinal_gap"])
            if ordinal_gap < MINIMUM_TEMPORAL_ORDINAL_GAP:
                _require(
                    disposition == "temporal_excluded"
                    and row["mahalanobis_squared"] is None,
                    "temporal exclusion disposition differs",
                )
            else:
                mahalanobis_squared = float(
                    row["mahalanobis_squared"]
                )
                factor_error = float(row["factor_error"])
                mahalanobis = float(row["mahalanobis"])
                _require(
                    np.isfinite(mahalanobis_squared)
                    and mahalanobis_squared == 2.0 * factor_error
                    and np.isclose(
                        mahalanobis,
                        math.sqrt(mahalanobis_squared),
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                    "candidate Mahalanobis scalar contract differs",
                )
                if (
                    mahalanobis_squared
                    >= MAHALANOBIS_SQUARED_THRESHOLD
                ):
                    _require(
                        disposition == "mahalanobis_rejected"
                        and row["rank_within_current"] is None,
                        "Mahalanobis rejection comparison differs",
                    )
                else:
                    _require(
                        row["rank_within_current"] is not None,
                        "threshold-passing candidate lacks a rank",
                    )
            if disposition == "cluster_suppressed":
                anchor_id = int(row["accepted_anchor_id"])
                anchor = accepted_by_id.get(anchor_id)
                _require(
                    anchor is not None
                    and anchor_id <= running_version + len(accepted)
                    and int(
                        row[
                            "accepted_anchor_delta_previous_ordinal"
                        ]
                    )
                    == int(row["previous_keyframe_ordinal"])
                    - int(anchor["previous_keyframe_ordinal"])
                    and int(
                        row[
                            "accepted_anchor_delta_current_ordinal"
                        ]
                    )
                    == current_ordinal
                    - int(anchor["current_keyframe_ordinal"])
                    and abs(
                        int(
                            row[
                                "accepted_anchor_delta_previous_ordinal"
                            ]
                        )
                    )
                    <= CLUSTER_PREVIOUS_ORDINAL_RADIUS
                    and abs(
                        int(
                            row[
                                "accepted_anchor_delta_current_ordinal"
                            ]
                        )
                    )
                    <= CLUSTER_CURRENT_ORDINAL_RADIUS,
                    "cluster-suppression anchor evidence differs",
                )
        ranked_rows = sorted(
            (
                row
                for row in rows
                if row["rank_within_current"] is not None
            ),
            key=lambda row: int(row["rank_within_current"]),
        )
        _require(
            [int(row["rank_within_current"]) for row in ranked_rows]
            == list(range(1, len(ranked_rows) + 1))
            and ranked_rows
            == sorted(
                ranked_rows,
                key=lambda row: (
                    float(row["mahalanobis_squared"]),
                    int(row["previous_keyframe_ordinal"]),
                    int(row["previous_frame_id"]),
                ),
            ),
            "deterministic candidate ranking differs",
        )
        _require(
            len(rows) == current_ordinal
            and sorted(rows_by_pair) == expected_pair_indices
            and int(current["candidate_count"]) == len(rows)
            and int(current["current_ordinal"]) == current_ordinal
            and int(current["current_frame_id"]) == current_frame_id
            and int(current["state_version_before"]) == running_version
            and int(current["candidate_graph_edge_count_before"])
            == BASE_COVARIANCE_EDGE_COUNT + running_version
            and int(current["matcher_invocation_count"]) == len(matches)
            and len(matches)
            <= MAXIMUM_MATCHER_INVOCATIONS_PER_CURRENT
            and int(current["accepted_loop_count"]) == len(accepted)
            and current["disposition_counts"]
            == dict(sorted(observed_dispositions.items())),
            "per-current population or input counts differ",
        )
        _require(
            sorted(int(match["matcher_slot"]) for match in matches)
            == list(range(1, len(matches) + 1)),
            "per-current matcher slots differ",
        )
        for match in matches:
            pair_index = int(match["pair_index"])
            row = rows_by_pair.get(pair_index)
            pair = (
                int(match["previous_frame_id"]),
                int(match["current_frame_id"]),
            )
            _require(
                row is not None
                and pair not in global_matcher_pairs
                and int(match["previous_keyframe_ordinal"])
                == int(row["previous_keyframe_ordinal"])
                and int(match["current_keyframe_ordinal"])
                == int(row["current_keyframe_ordinal"])
                and pair
                == (
                    int(row["previous_frame_id"]),
                    int(row["current_frame_id"]),
                )
                and int(match["rank_within_current"])
                == int(row["rank_within_current"])
                and bool(row["matcher_slot_consumed"])
                and int(row["matcher_slot"])
                == int(match["matcher_slot"])
                and row["terminal_disposition"] == match["outcome"],
                "candidate-to-matcher reconciliation differs",
            )
            global_matcher_pairs.add(pair)
        expected_ids = list(
            range(
                running_version + 1,
                running_version + len(accepted) + 1,
            )
        )
        _require(
            [int(record["acceptance_id"]) for record in accepted]
            == expected_ids,
            "per-current accepted-loop order differs",
        )
        for record in accepted:
            acceptance_id = int(record["acceptance_id"])
            pair_index = int(record["pair_index"])
            accepted_matches = [
                match
                for match in matches
                if int(match["pair_index"]) == pair_index
                and match["outcome"] == "accepted"
            ]
            _require(
                accepted_by_id[acceptance_id]["artifact"]
                == record["artifact"]
                and rows_by_pair[pair_index]["terminal_disposition"]
                == "accepted"
                and int(record["previous_keyframe_ordinal"])
                == int(
                    rows_by_pair[pair_index][
                        "previous_keyframe_ordinal"
                    ]
                )
                and int(record["current_keyframe_ordinal"])
                == int(
                    rows_by_pair[pair_index][
                        "current_keyframe_ordinal"
                    ]
                )
                and int(record["previous_frame_id"])
                == int(rows_by_pair[pair_index]["previous_frame_id"])
                and int(record["current_frame_id"])
                == int(rows_by_pair[pair_index]["current_frame_id"])
                and int(
                    rows_by_pair[pair_index]["accepted_anchor_id"]
                )
                == acceptance_id
                and len(accepted_matches) == 1
                and int(accepted_matches[0]["acceptance_id"])
                == acceptance_id
                and int(record["state_version_before"])
                == acceptance_id - 1
                and int(record["state_version_after"])
                == acceptance_id
                and int(record["factor_count_after"])
                == BASE_POSE_GRAPH_FACTOR_COUNT + acceptance_id
                and int(record["marginal_count"])
                == EXPECTED_KEYFRAME_COUNT
                and record["factor_direction"]
                == "previous_FrameId_to_current_FrameId"
                and record["tangent_order"] == list(TANGENT_ORDER)
                and record["covariance_modified"] is False,
                "accepted-loop record reconciliation differs",
            )
        running_version += len(accepted)
        _require(
            int(current["state_version_after"]) == running_version
            and int(current["candidate_graph_edge_count_after"])
            == BASE_COVARIANCE_EDGE_COUNT + running_version,
            "per-current output graph/state counts differ",
        )
    _require(
        running_version == accepted_count
        and len(global_matcher_pairs) == len(matcher_invocations),
        "global matcher or accepted-state accounting differs",
    )
    for first_id in range(1, accepted_count + 1):
        first = accepted_by_id[first_id]
        for second_id in range(first_id + 1, accepted_count + 1):
            second = accepted_by_id[second_id]
            _require(
                not same_anchored_rectangle(
                    int(first["previous_keyframe_ordinal"]),
                    int(first["current_keyframe_ordinal"]),
                    int(second["previous_keyframe_ordinal"]),
                    int(second["current_keyframe_ordinal"]),
                ),
                "two accepted loops occupy one accepted-anchor rectangle",
            )
    accepted_rows = [
        row
        for row in candidates
        if row["terminal_disposition"] == "accepted"
    ]
    _require(
        len(accepted_rows) == accepted_count
        and state.state_version == accepted_count
        and state.graph.size()
        == BASE_POSE_GRAPH_FACTOR_COUNT + accepted_count
        and len(candidate_graph.edges)
        == BASE_COVARIANCE_EDGE_COUNT + accepted_count,
        "accepted-loop graph/state counts do not reconcile",
    )
    _require(
        state.optimized_pose_matrices.shape
        == (EXPECTED_KEYFRAME_COUNT, 4, 4)
        and state.marginal_covariances.shape
        == (EXPECTED_KEYFRAME_COUNT, 6, 6)
        and np.all(np.isfinite(state.optimized_pose_matrices))
        and np.all(np.isfinite(state.marginal_covariances)),
        "final Pose Graph values or Marginals are invalid",
    )
    if accepted_count == 0:
        _require(
            np.array_equal(state.optimized_pose_matrices, baseline_poses)
            and np.array_equal(
                state.marginal_covariances, baseline_marginals
            ),
            "zero-loop final state differs from protected E6",
        )
    disposition_counts: Dict[str, int] = {}
    for row in candidates:
        disposition = str(row["terminal_disposition"])
        disposition_counts[disposition] = (
            disposition_counts.get(disposition, 0) + 1
        )
    return {
        "schema": "exercise7-i04-technical-validation-v1",
        "technical_run_status": "pass",
        "all_current_keyframes_sealed": True,
        "current_keyframe_count": len(currents),
        "ordered_pair_count": len(candidates),
        "matcher_invocation_count": len(matcher_invocations),
        "successful_loop_closure_count": accepted_count,
        "pose_graph_factor_count": int(state.graph.size()),
        "candidate_graph_edge_count": len(candidate_graph.edges),
        "state_version": int(state.state_version),
        "all_node_marginal_count": int(
            state.marginal_covariances.shape[0]
        ),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "zero_loop_is_technically_valid": True,
    }


def _traceability_record(
    *,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Mapping[str, object]:
    source_sha = _production_identity(
        e5_root=e5_root, e6_root=e6_root
    )["source_sha256"]
    return {
        "schema": TRACEABILITY_SCHEMA,
        "schema_version": 1,
        "candidate_detection": {
            "module": "exercise7_candidate_detection.py",
            "functions": [
                "CovarianceGraph.find_course_covariance_path",
                "score_previous_keyframes",
            ],
            "source_sha256": source_sha[
                "exercise7_candidate_detection.py"
            ],
            "formula": "d_squared = 2 * BetweenFactorPose3.error",
        },
        "loop_factor_construction": {
            "matching_module": "exercise7_consensus_matching.py",
            "matching_function": "evaluate_candidate_pair",
            "bundle_module": "exercise7_small_bundle.py",
            "bundle_function": "solve_small_bundle",
            "pose_graph_module": "exercise7_pose_graph_update.py",
            "pose_graph_function": (
                "CumulativePoseGraphState.apply_loop_factor"
            ),
            "tangent_order": list(TANGENT_ORDER),
            "covariance_modified": False,
        },
        "figures": {
            "successful_match.png": (
                "exercise7_sequential_loop_closure."
                "_save_successful_match_figure"
            ),
            "trajectory_comparison.png": (
                "exercise7_sequential_loop_closure."
                "_plot_trajectory_comparison"
            ),
            "absolute_pose_errors.png": (
                "exercise7_sequential_loop_closure."
                "_plot_absolute_errors"
            ),
            "marginal_uncertainty_comparison.png": (
                "exercise7_sequential_loop_closure."
                "_plot_uncertainty_comparison"
            ),
            "result_aware_pose_graph_snapshots.png": (
                "exercise7_sequential_loop_closure."
                "_plot_result_aware_snapshots"
            ),
        },
        "figure_axes_and_units": {
            "trajectory": ["x [m]", "z [m]"],
            "absolute_errors": [
                "FrameId",
                "location [m]",
                "rotation [deg]",
            ],
            "uncertainty": [
                "FrameId",
                "location [m]",
                "rotation [deg]",
            ],
        },
        "report_freeze_line_refresh": "pending_nonblocking",
        "ast_line_ranges_required_for_canonical_run": False,
    }


def _validate_final_package_directory(
    package_root: Path,
    *,
    output_root: Path,
    require_run_state: bool,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Mapping[str, object]:
    package_root = Path(package_root)
    output_root = Path(output_root)
    _require(
        package_root.is_dir()
        and (package_root / "manifest.json").is_file(),
        "final package or manifest is missing",
    )
    policy = _read_json(output_root / "policy.json")
    provenance = _read_json(output_root / "provenance.json")
    _require(
        policy == ProductionPolicy().record()
        and provenance
        == _production_identity(e5_root=e5_root, e6_root=e6_root),
        "final package policy or provenance differs from current sources",
    )
    manifest = _read_json(package_root / "manifest.json")
    _require(
        manifest.get("schema") == "exercise7-i04-manifest-v1"
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "complete-and-sealed"
        and manifest.get("technical_run_status") == "pass"
        and manifest.get("current_keyframe_count")
        == EXPECTED_KEYFRAME_COUNT
        and manifest.get("ordered_pair_count")
        == EXPECTED_ORDERED_PAIR_COUNT
        and manifest.get("policy_sha256")
        == sha256_file(output_root / "policy.json")
        and manifest.get("provenance_sha256")
        == sha256_file(output_root / "provenance.json"),
        "final manifest scalar contract differs",
    )
    observed_current_seals = {}
    for index in range(EXPECTED_KEYFRAME_COUNT):
        current_directory = (
            output_root / "currents" / "current_{:04d}".format(index)
        )
        validate_current_seal(current_directory)
        relative = "currents/current_{:04d}/seal.json".format(index)
        observed_current_seals[relative] = sha256_file(
            current_directory / "seal.json"
        )
    _require(
        manifest.get("current_seal_sha256")
        == observed_current_seals,
        "final manifest current-seal hashes differ",
    )
    observed_derived = _hash_directory_files(
        package_root, exclude_names=("manifest.json",)
    )
    _require(
        manifest.get("derived_artifact_sha256") == observed_derived,
        "final manifest derived-artifact hashes differ",
    )
    required_files = {
        "summary.json",
        "validation.json",
        "traceability.json",
        "aggregate/candidate_dispositions.csv",
        "aggregate/matcher_invocations.csv",
        "aggregate/accepted_loops.csv",
        "aggregate/current_keyframes.csv",
        "aggregate/retained_path_evidence_index.json",
        "reports/summary.json",
        "reports/data/trajectory_comparison.csv",
        "reports/data/absolute_pose_errors.csv",
        "reports/data/marginal_uncertainty_sizes.csv",
        "reports/data/pose_graph_snapshot_selection.json",
        "reports/figures/trajectory_comparison.png",
        "reports/figures/absolute_pose_errors.png",
        "reports/figures/marginal_uncertainty_comparison.png",
        "reports/figures/result_aware_pose_graph_snapshots.png",
    }
    _require(
        required_files.issubset(set(observed_derived)),
        "final package required artifact population differs",
    )
    summary = _read_json(package_root / "summary.json")
    validation = _read_json(package_root / "validation.json")
    traceability = _read_json(package_root / "traceability.json")
    report_summary = _read_json(package_root / "reports" / "summary.json")
    accepted_count = int(
        manifest.get("successful_loop_closure_count", -1)
    )
    expected_deliverable = (
        "pass" if accepted_count > 0 else "incomplete_no_successful_loop"
    )
    _require(
        summary.get("schema") == SUMMARY_SCHEMA
        and summary.get("technical_run_status") == "pass"
        and summary.get("exercise7_deliverable_status")
        == expected_deliverable
        and int(summary.get("successful_loop_closure_count", -1))
        == accepted_count
        and summary.get("technical_validation") == validation
        and summary.get("report") == report_summary
        and validation.get("technical_run_status") == "pass"
        and int(
            validation.get("successful_loop_closure_count", -1)
        )
        == accepted_count
        and traceability.get("schema") == TRACEABILITY_SCHEMA
        and manifest.get("exercise7_deliverable_status")
        == expected_deliverable,
        "final summary, validation, report, or traceability differs",
    )
    for csv_path, expected_fields in (
        (
            package_root
            / "aggregate"
            / "matcher_invocations.csv",
            MATCHER_CSV_FIELDS,
        ),
        (
            package_root / "aggregate" / "accepted_loops.csv",
            ACCEPTED_LOOP_CSV_FIELDS,
        ),
    ):
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            observed_fields = csv.DictReader(handle).fieldnames
        _require(
            observed_fields == list(expected_fields),
            "final zero-capable CSV schema differs",
        )
    report_paths = (
        list(report_summary.get("figures", []))
        + list(report_summary.get("data", []))
    )
    _require(
        all(
            (package_root / "reports" / str(relative)).is_file()
            for relative in report_paths
        ),
        "report summary references a missing artifact",
    )
    if require_run_state:
        run_state = _read_json(output_root / "run_state.json")
        _require(
            run_state.get("status") == "complete-and-sealed"
            and int(run_state.get("completed_current_count", -1))
            == EXPECTED_KEYFRAME_COUNT
            and int(run_state.get("state_version", -1))
            == accepted_count
            and int(
                run_state.get("successful_loop_closure_count", -1)
            )
            == accepted_count
            and run_state.get("technical_run_status") == "pass"
            and run_state.get("exercise7_deliverable_status")
            == expected_deliverable,
            "completed run-state record differs",
        )
    return {
        "manifest": manifest,
        "summary": summary,
        "validation": validation,
        "traceability": traceability,
        "report_summary": report_summary,
    }


def _complete_run_state_record(
    summary: Mapping[str, object]
) -> Mapping[str, object]:
    accepted_count = int(summary["successful_loop_closure_count"])
    return {
        "schema": "exercise7-i04-run-state-v1",
        "status": "complete-and-sealed",
        "completed_current_count": EXPECTED_KEYFRAME_COUNT,
        "last_completed_current_ordinal": EXPECTED_KEYFRAME_COUNT - 1,
        "state_version": accepted_count,
        "successful_loop_closure_count": accepted_count,
        "technical_run_status": "pass",
        "exercise7_deliverable_status": summary[
            "exercise7_deliverable_status"
        ],
    }


def _finalize_complete_run(
    *,
    output_root: Path,
    baseline_inputs: CandidateDetectionInputs,
    candidate_graph: CovarianceGraph,
    state: CumulativePoseGraphState,
    configuration_sha256: str,
    run_started: float,
    e5_root: Path,
    e6_root: Path,
) -> Mapping[str, object]:
    output_root = Path(output_root)
    final_root = output_root / "final"
    if final_root.exists():
        package = _validate_final_package_directory(
            final_root,
            output_root=output_root,
            require_run_state=False,
            e5_root=e5_root,
            e6_root=e6_root,
        )
        summary = package["summary"]
        _write_json_atomic(
            output_root / "run_state.json",
            _complete_run_state_record(summary),
        )
        return summary
    currents, candidates, matcher_invocations, accepted_loops = (
        _collect_complete_records(
            output_root, baseline_inputs.keyframe_ids
        )
    )
    frame_ids, baseline_poses, baseline_marginals = (
        _load_baseline_arrays(e6_root=e6_root)
    )
    _require(
        np.array_equal(
            frame_ids,
            np.asarray(baseline_inputs.keyframe_ids, dtype=np.int64),
        ),
        "report FrameIds differ from canonical candidate inputs",
    )
    validation = _technical_validation(
        currents=currents,
        candidates=candidates,
        matcher_invocations=matcher_invocations,
        accepted_loops=accepted_loops,
        candidate_graph=candidate_graph,
        state=state,
        baseline_poses=baseline_poses,
        baseline_marginals=baseline_marginals,
    )
    attempt = Path(
        tempfile.mkdtemp(prefix=".finalization.", dir=str(output_root))
    )
    try:
        aggregate_root = attempt / "aggregate"
        aggregate_root.mkdir()
        _write_mapping_csv(
            aggregate_root / "candidate_dispositions.csv", candidates
        )
        _write_mapping_csv(
            aggregate_root / "matcher_invocations.csv",
            matcher_invocations,
            fieldnames=MATCHER_CSV_FIELDS,
        )
        _write_mapping_csv(
            aggregate_root / "accepted_loops.csv",
            accepted_loops,
            fieldnames=ACCEPTED_LOOP_CSV_FIELDS,
        )
        current_summaries = [
            {
                key: value
                for key, value in current.items()
                if key
                not in {
                    "candidates",
                    "matcher_invocations",
                    "accepted_loops",
                }
            }
            for current in currents
        ]
        _write_mapping_csv(
            aggregate_root / "current_keyframes.csv", current_summaries
        )
        _write_json(
            aggregate_root / "retained_path_evidence_index.json",
            {
                "schema": "exercise7-i04-path-evidence-index-v1",
                "retention": (
                    "complete path and raw accumulated covariance for every "
                    "Mahalanobis-threshold-passing candidate"
                ),
                "files": [
                    "currents/current_{:04d}/candidate_paths.npz".format(
                        current_ordinal
                    )
                    for current_ordinal in range(EXPECTED_KEYFRAME_COUNT)
                ],
                "above_threshold_full_paths_retained": False,
                "temporal_excluded_full_paths_retained": False,
            },
        )

        report_summary = _build_report_artifacts(
            attempt / "reports",
            frame_ids=frame_ids,
            baseline_poses=baseline_poses,
            baseline_marginals=baseline_marginals,
            final_poses=state.optimized_pose_matrices,
            final_marginals=state.marginal_covariances,
            ground_truth_path=state.inputs.ground_truth_path,
            accepted_loops=accepted_loops,
            e6_values_path=Path(e6_root) / "pose_graph_values.npz",
            e6_marginals_path=(
                Path(e6_root) / "pose_graph_marginals.npz"
            ),
        )
        accepted_count = len(accepted_loops)
        deliverable_status = (
            "pass"
            if accepted_count > 0
            else "incomplete_no_successful_loop"
        )
        final_graph_error = float(
            state.graph.error(state.optimized_values)
        )
        _require(
            np.isfinite(final_graph_error),
            "final accumulated Pose Graph error is nonfinite",
        )
        candidate_local_failure_count = sum(
            row["terminal_disposition"]
            in {"matcher_failed", "measurement_failed"}
            for row in candidates
        )
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": 1,
            "milestone": "E7-I04",
            "technical_run_status": "pass",
            "exercise7_deliverable_status": deliverable_status,
            "scope": (
                "complete sequential canonical Loop Closure over "
                "356 keyframes and FrameIds 0-3359"
            ),
            "configuration_sha256": configuration_sha256,
            "keyframe_count": EXPECTED_KEYFRAME_COUNT,
            "ordered_pair_count": EXPECTED_ORDERED_PAIR_COUNT,
            "matcher_invocation_count": len(matcher_invocations),
            "successful_loop_closure_count": accepted_count,
            "accepted_loop_pairs": [
                {
                    "acceptance_id": record["acceptance_id"],
                    "previous_frame_id": record["previous_frame_id"],
                    "current_frame_id": record["current_frame_id"],
                    "previous_keyframe_ordinal": (
                        record["previous_keyframe_ordinal"]
                    ),
                    "current_keyframe_ordinal": (
                        record["current_keyframe_ordinal"]
                    ),
                    "consensus_count": record["consensus_count"],
                    "supporter_count": record["supporter_count"],
                    "supporter_percentage": (
                        record["supporter_percentage"]
                    ),
                    "ransac_terminal_status": (
                        record["ransac_terminal_status"]
                    ),
                }
                for record in accepted_loops
            ],
            "final_pose_graph": {
                "factor_count": int(state.graph.size()),
                "state_version": int(state.state_version),
                "final_graph_error": final_graph_error,
                "marginal_count": int(
                    state.marginal_covariances.shape[0]
                ),
            },
            "candidate_detection_graph": {
                "edge_count": len(candidate_graph.edges),
                "consecutive_edge_count": BASE_COVARIANCE_EDGE_COUNT,
                "accepted_loop_edge_count": accepted_count,
            },
            "candidate_local_failure_count": (
                candidate_local_failure_count
            ),
            "fatal_failure_count": 0,
            "disposition_counts": validation["disposition_counts"],
            "report": report_summary,
            "technical_validation": validation,
            "runtime_seconds_this_invocation": float(
                time.perf_counter() - run_started
            ),
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "canonical_run_command": (
                ".venv/bin/python exercise7_sequential_loop_closure.py run"
            ),
            "report_freeze_line_refresh": "pending_nonblocking",
        }
        traceability = _traceability_record(
            e5_root=e5_root, e6_root=e6_root
        )
        _write_json(attempt / "validation.json", validation)
        _write_json(attempt / "traceability.json", traceability)
        _write_json(attempt / "summary.json", summary)

        derived_hashes = _hash_directory_files(
            attempt, exclude_names=("manifest.json",)
        )
        current_seal_hashes = {
            "currents/current_{:04d}/seal.json".format(index): sha256_file(
                output_root
                / "currents"
                / "current_{:04d}".format(index)
                / "seal.json"
            )
            for index in range(EXPECTED_KEYFRAME_COUNT)
        }
        manifest = {
            "schema": "exercise7-i04-manifest-v1",
            "schema_version": 1,
            "status": "complete-and-sealed",
            "technical_run_status": "pass",
            "exercise7_deliverable_status": deliverable_status,
            "current_keyframe_count": EXPECTED_KEYFRAME_COUNT,
            "ordered_pair_count": EXPECTED_ORDERED_PAIR_COUNT,
            "successful_loop_closure_count": accepted_count,
            "policy_sha256": sha256_file(output_root / "policy.json"),
            "provenance_sha256": sha256_file(
                output_root / "provenance.json"
            ),
            "current_seal_sha256": current_seal_hashes,
            "derived_artifact_sha256": derived_hashes,
        }
        _write_json(attempt / "manifest.json", manifest)
        _require(
            _read_json(output_root / "provenance.json")
            == _production_identity(e5_root=e5_root, e6_root=e6_root),
            "production sources changed during finalization",
        )
        _validate_final_package_directory(
            attempt,
            output_root=output_root,
            require_run_state=False,
            e5_root=e5_root,
            e6_root=e6_root,
        )
        _require(
            not final_root.exists(),
            "atomic final package destination already exists",
        )
        os.replace(str(attempt), str(final_root))
        _validate_final_package_directory(
            final_root,
            output_root=output_root,
            require_run_state=False,
            e5_root=e5_root,
            e6_root=e6_root,
        )
        _write_json_atomic(
            output_root / "run_state.json",
            _complete_run_state_record(summary),
        )
        return summary
    except BaseException:
        if attempt.exists():
            shutil.rmtree(attempt)
        raise


def _durable_prefix_counts(output_root: Path) -> Tuple[int, int]:
    completed = 0
    state_version = 0
    currents_root = Path(output_root) / "currents"
    for index in range(EXPECTED_KEYFRAME_COUNT):
        directory = currents_root / "current_{:04d}".format(index)
        if not directory.is_dir():
            break
        try:
            sealed = validate_current_seal(directory)
            current = sealed["current"]
            if not (
                int(current["current_ordinal"]) == index
                and int(current["state_version_before"])
                == state_version
            ):
                break
        except Exception:
            # Fatal reporting must preserve the original exception.  Only the
            # independently valid contiguous prefix is durable.
            break
        state_version = int(current["state_version_after"])
        completed += 1
    return completed, state_version


def run_sequential_loop_closure(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    policy: ProductionPolicy = ProductionPolicy(),
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Mapping[str, object]:
    """Run or resume the frozen sequence from one coherent E5/E6 dependency pair.

    Startup validates the selected roots, reconstructs the baseline, and
    replays only the contiguous prefix of atomically sealed currents to recover
    scientific state, accepted anchors, and evaluated-pair history.  Each new
    current is sealed before the next begins.  A fatal unsealed attempt is
    discarded and recorded, so a later invocation resumes from the last valid
    seal rather than trusting partial in-memory or filesystem state.
    """

    run_started = time.perf_counter()
    output_root = _require_safe_output_root(output_root)
    _prepare_output_root(
        output_root,
        policy,
        e5_root=e5_root,
        e6_root=e6_root,
    )
    try:
        (
            baseline_inputs,
            candidate_graph,
            state,
            configuration,
            configuration_sha256,
            calibration,
        ) = _load_baseline_runtime(e5_root=e5_root, e6_root=e6_root)
        (
            candidate_graph,
            state,
            anchors,
            evaluated_pairs,
            next_current_ordinal,
        ) = _replay_sealed_currents(
            output_root, baseline_inputs, candidate_graph, state
        )
        final_root = output_root / "final"
        if final_root.exists():
            _require(
                next_current_ordinal == EXPECTED_KEYFRAME_COUNT,
                "final package exists before every current is sealed",
            )
            package = _validate_final_package_directory(
                final_root,
                output_root=output_root,
                require_run_state=False,
                e5_root=e5_root,
                e6_root=e6_root,
            )
            summary = package["summary"]
            _write_json_atomic(
                output_root / "run_state.json",
                _complete_run_state_record(summary),
            )
            return summary

        for current_ordinal in range(
            next_current_ordinal, EXPECTED_KEYFRAME_COUNT
        ):
            _write_json_atomic(
                output_root / "run_state.json",
                {
                    "schema": "exercise7-i04-run-state-v1",
                    "status": "running",
                    "completed_current_count": current_ordinal,
                    "next_current_ordinal": current_ordinal,
                    "state_version": int(state.state_version),
                    "successful_loop_closure_count": len(anchors),
                },
            )
            candidate_graph = _process_current_keyframe(
                output_root=output_root,
                current_ordinal=current_ordinal,
                baseline_inputs=baseline_inputs,
                candidate_graph=candidate_graph,
                state=state,
                configuration=configuration,
                calibration=calibration,
                anchors=anchors,
                evaluated_pairs=evaluated_pairs,
                policy=policy,
            )
            sealed = validate_current_seal(
                output_root
                / "currents"
                / "current_{:04d}".format(current_ordinal)
            )
            current = sealed["current"]
            _write_json_atomic(
                output_root / "run_state.json",
                {
                    "schema": "exercise7-i04-run-state-v1",
                    "status": "running",
                    "completed_current_count": current_ordinal + 1,
                    "last_completed_current_ordinal": current_ordinal,
                    "next_current_ordinal": current_ordinal + 1,
                    "state_version": int(state.state_version),
                    "successful_loop_closure_count": len(anchors),
                },
            )
            print(
                (
                    "sealed current {}/355 FrameId={} scored={} "
                    "matched={} accepted={} state_version={} runtime={:.3f}s"
                ).format(
                    current_ordinal,
                    current["current_frame_id"],
                    current["temporally_eligible_count"],
                    current["matcher_invocation_count"],
                    current["accepted_loop_count"],
                    state.state_version,
                    current["runtime_seconds"],
                ),
                flush=True,
            )
        return _finalize_complete_run(
            output_root=output_root,
            baseline_inputs=baseline_inputs,
            candidate_graph=candidate_graph,
            state=state,
            configuration_sha256=configuration_sha256,
            run_started=run_started,
            e5_root=e5_root,
            e6_root=e6_root,
        )
    except BaseException as error:
        completed, durable_state_version = _durable_prefix_counts(
            output_root
        )
        _write_json_atomic(
            output_root / "run_state.json",
            {
                "schema": "exercise7-i04-run-state-v1",
                "status": "fatal",
                "completed_current_count": completed,
                "last_completed_current_ordinal": (
                    completed - 1 if completed else None
                ),
                "state_version": durable_state_version,
                "successful_loop_closure_count": (
                    durable_state_version
                ),
                "fatal_failure_type": type(error).__name__,
                "fatal_failure_message": str(error),
            },
        )
        raise


def validate_complete_run(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
) -> Mapping[str, object]:
    """Reload every seal against its explicitly selected E5/E6 identity."""

    policy = ProductionPolicy()
    output_root = Path(output_root)
    policy.validate()
    _require(
        output_root.is_dir()
        and (output_root / "currents").is_dir()
        and (output_root / "policy.json").is_file()
        and (output_root / "provenance.json").is_file()
        and _read_json(output_root / "policy.json") == policy.record()
        and _read_json(output_root / "provenance.json")
        == _production_identity(e5_root=e5_root, e6_root=e6_root),
        "validation target is not the current E7-I04 production root",
    )
    (
        baseline_inputs,
        candidate_graph,
        state,
        _,
        _,
        _,
    ) = _load_baseline_runtime(e5_root=e5_root, e6_root=e6_root)
    (
        candidate_graph,
        state,
        _,
        _,
        next_current_ordinal,
    ) = _replay_sealed_currents(
        output_root, baseline_inputs, candidate_graph, state
    )
    _require(
        next_current_ordinal == EXPECTED_KEYFRAME_COUNT,
        "canonical run is not complete",
    )
    currents, candidates, matcher_invocations, accepted_loops = (
        _collect_complete_records(
            output_root, baseline_inputs.keyframe_ids
        )
    )
    _, baseline_poses, baseline_marginals = _load_baseline_arrays(
        e6_root=e6_root
    )
    validation = _technical_validation(
        currents=currents,
        candidates=candidates,
        matcher_invocations=matcher_invocations,
        accepted_loops=accepted_loops,
        candidate_graph=candidate_graph,
        state=state,
        baseline_poses=baseline_poses,
        baseline_marginals=baseline_marginals,
    )
    package = _validate_final_package_directory(
        output_root / "final",
        output_root=output_root,
        require_run_state=True,
        e5_root=e5_root,
        e6_root=e6_root,
    )
    _require(
        package["validation"] == validation
        and int(
            package["summary"]["successful_loop_closure_count"]
        )
        == len(accepted_loops),
        "independent validation differs from the sealed final package",
    )
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight, run, or validate sequential Exercise 7"
    )
    parser.add_argument(
        "command",
        choices=("preflight", "run", "validate"),
        nargs="?",
        default="run",
    )
    parser.add_argument("--e5-root", type=Path, default=DEFAULT_E5_ROOT)
    parser.add_argument("--e6-root", type=Path, default=DEFAULT_E6_ROOT)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        result = preflight_exercise7(
            e5_root=arguments.e5_root,
            e6_root=arguments.e6_root,
            output_root=arguments.output_root,
        )
    elif arguments.command == "run":
        result = run_sequential_loop_closure(
            output_root=arguments.output_root,
            e5_root=arguments.e5_root,
            e6_root=arguments.e6_root,
        )
    else:
        result = validate_complete_run(
            output_root=arguments.output_root,
            e5_root=arguments.e5_root,
            e6_root=arguments.e6_root,
        )
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AcceptedLoopAnchor",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_E5_ROOT",
    "DEFAULT_E6_ROOT",
    "Exercise7InputPaths",
    "ProductionPolicy",
    "RankedCandidate",
    "SequentialLoopClosureError",
    "TemporaryProposalRegion",
    "accepted_anchor_for_candidate",
    "evaluate_region_schedule",
    "form_temporary_proposal_regions",
    "main",
    "ordered_pair_index",
    "preflight_exercise7",
    "run_sequential_loop_closure",
    "same_anchored_rectangle",
    "validate_complete_run",
]
