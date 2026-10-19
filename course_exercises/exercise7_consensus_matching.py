"""Four-view consensus matching for E7-I02 fixtures and E7-I04 production.

The protected bounded probe still evaluates exactly eight declared
historical/current keyframe pairs.  E7-I04 additionally constructs arbitrary
canonical pairs and evaluates each one independently with the same protected
Exercise 3 image, matching, PnP, stereo-composition, and four-view supporter
primitives.  For each endpoint it extracts stereo features and separates raw
stereo matches from rectified-stereo inliers.  Reciprocal left-image temporal
matches are then intersected with both endpoint stereo-inlier maps to create
four-view consensus: the same feature relationship is present in previous
left/right and current left/right views.  Previous stereo inliers provide the
triangulated 3D input to a bounded custom PnP/RANSAC search.

A final supporter (inlier) is a four-view consensus entry whose reprojection
is within two pixels in every view under the retained model.  Thus raw stereo
matches, stereo inliers, reciprocal temporal matches, four-view consensus
matches, and final supporters are successive and distinct evidence sets.  The
module resets the Python, NumPy, and OpenCV random generators for every pair,
counts every PnP
hypothesis attempt against a fixed budget, and retains whether termination was
adaptive or exhausted the budget with a best valid model.  It never calls the
unbounded Exercise 3 RANSAC driver or its retrying model builder.

The bounded fixture results remain diagnostic evidence only.  Production
acceptance is exposed separately and requires complete retained geometry plus
the frozen consensus/supporter gates.  Counts, terminal status/reason,
hypothesis accounting, retained model, supporter indices, runtimes, and the
in-memory feature mappings needed by the later small Bundle are kept distinct.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

import tools


REPOSITORY_ROOT = Path(__file__).resolve().parent
CANONICAL_E7_I01_SCORES = (
    REPOSITORY_ROOT / "diagnostics" / "ex7_i01" / "candidate_scores.csv"
)
CANONICAL_E7_I01_SCORES_SHA256 = (
    "88edddb8e5a156ba9a3ca44f57db11cedbadbd127a8b25eff86bb9c42f905541"
)

CANONICAL_SEED = 0
RANSAC_SUCCESS_PROBABILITY = 0.99
RANSAC_SAMPLE_SIZE = 4
RANSAC_INITIAL_OUTLIER_PROBABILITY = 0.99
MAX_HYPOTHESIS_ATTEMPTS = 3000
FOUR_VIEW_SUPPORTER_MAX_DISTANCE_PIXELS = 2.0

PROMISING_DISTANT = "promising_distant"
ADJACENT_CONTROL = "adjacent_control"
POOR_DISTANT_CONTROL = "poor_distant_control"

ADAPTIVE_COMPLETED = "adaptive_completed"
BUDGET_EXHAUSTED_WITH_BEST_MODEL = (
    "budget_exhausted_with_best_model"
)
INSUFFICIENT_CONSENSUS = "insufficient_consensus"
NO_VALID_PNP_MODEL = "no_valid_pnp_model"
INVALID_GEOMETRY = "invalid_geometry"

TERMINAL_STATUSES = (
    ADAPTIVE_COMPLETED,
    BUDGET_EXHAUSTED_WITH_BEST_MODEL,
    INSUFFICIENT_CONSENSUS,
    NO_VALID_PNP_MODEL,
    INVALID_GEOMETRY,
)

PRODUCTION_MINIMUM_CONSENSUS_COUNT = 30
PRODUCTION_MINIMUM_SUPPORTER_COUNT = 30
PRODUCTION_MINIMUM_SUPPORTER_PERCENTAGE = 30.0
PRODUCTION_ELIGIBLE_TERMINAL_STATUSES = (
    ADAPTIVE_COMPLETED,
    BUDGET_EXHAUSTED_WITH_BEST_MODEL,
)


class ConsensusMatchingError(RuntimeError):
    """Raised when an E7-I02 input or executable contract is violated."""


@dataclass(frozen=True)
class CandidatePair:
    """One ranked historical/current canonical-keyframe pair."""

    pair_index: int
    category: str
    current_frame_id: int
    current_keyframe_ordinal: int
    previous_frame_id: int
    previous_keyframe_ordinal: int
    rank_within_current: int
    ordinal_gap: int
    frame_id_gap: int
    mahalanobis_squared: float
    mahalanobis: float
    selection_reason: str


def validate_candidate_pair(
    pair: CandidatePair,
    *,
    canonical_frame_ids: Optional[Sequence[int]] = None,
) -> CandidatePair:
    """Validate one arbitrary pair and optionally bind it to canonical IDs."""

    if not isinstance(pair, CandidatePair):
        raise ConsensusMatchingError("candidate must be a CandidatePair")
    integer_fields = {
        "pair_index": pair.pair_index,
        "current_frame_id": pair.current_frame_id,
        "current_keyframe_ordinal": pair.current_keyframe_ordinal,
        "previous_frame_id": pair.previous_frame_id,
        "previous_keyframe_ordinal": pair.previous_keyframe_ordinal,
        "rank_within_current": pair.rank_within_current,
        "ordinal_gap": pair.ordinal_gap,
        "frame_id_gap": pair.frame_id_gap,
    }
    if any(type(value) is not int for value in integer_fields.values()):
        raise ConsensusMatchingError(
            "candidate integer fields must contain exact integers"
        )
    if pair.pair_index < 0 or pair.rank_within_current < 1:
        raise ConsensusMatchingError(
            "candidate index and rank must be nonnegative/positive"
        )
    if (
        pair.previous_frame_id < 0
        or pair.previous_keyframe_ordinal < 0
        or pair.previous_frame_id >= pair.current_frame_id
        or pair.previous_keyframe_ordinal >= pair.current_keyframe_ordinal
    ):
        raise ConsensusMatchingError(
            "candidate endpoints must be ordered previous before current"
        )
    if (
        pair.current_keyframe_ordinal
        - pair.previous_keyframe_ordinal
        != pair.ordinal_gap
        or pair.current_frame_id - pair.previous_frame_id
        != pair.frame_id_gap
    ):
        raise ConsensusMatchingError("candidate endpoint gaps are inconsistent")
    if (
        not isinstance(pair.category, str)
        or not pair.category
        or not isinstance(pair.selection_reason, str)
        or not pair.selection_reason
    ):
        raise ConsensusMatchingError(
            "candidate category and selection reason must be nonempty"
        )
    mahalanobis_squared = float(pair.mahalanobis_squared)
    mahalanobis = float(pair.mahalanobis)
    if (
        not np.isfinite(mahalanobis_squared)
        or mahalanobis_squared < 0.0
        or not np.isfinite(mahalanobis)
        or mahalanobis < 0.0
        or not math.isclose(
            mahalanobis,
            math.sqrt(mahalanobis_squared),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ConsensusMatchingError(
            "candidate Mahalanobis values are invalid or inconsistent"
        )

    if canonical_frame_ids is not None:
        frame_ids = tuple(int(value) for value in canonical_frame_ids)
        if (
            not frame_ids
            or len(set(frame_ids)) != len(frame_ids)
            or pair.current_keyframe_ordinal >= len(frame_ids)
        ):
            raise ConsensusMatchingError(
                "canonical FrameId ordering is invalid for the candidate"
            )
        if (
            frame_ids[pair.previous_keyframe_ordinal]
            != pair.previous_frame_id
            or frame_ids[pair.current_keyframe_ordinal]
            != pair.current_frame_id
        ):
            raise ConsensusMatchingError(
                "candidate FrameIds do not match their canonical ordinals"
            )
    return pair


def candidate_pair_from_score(
    score: object,
    *,
    pair_index: int,
    rank_within_current: int,
    category: str = "production_candidate",
    selection_reason: str = "E7-I04 ranked candidate",
    canonical_frame_ids: Optional[Sequence[int]] = None,
) -> CandidatePair:
    """Construct a validated production pair from an E7-I01 CandidateScore."""

    try:
        previous_frame_id = int(score.previous_frame_id)
        current_frame_id = int(score.current_frame_id)
        previous_ordinal = int(score.previous_keyframe_ordinal)
        current_ordinal = int(score.current_keyframe_ordinal)
        mahalanobis_squared = float(score.mahalanobis_squared)
        mahalanobis = float(score.mahalanobis)
    except (AttributeError, TypeError, ValueError) as error:
        raise ConsensusMatchingError(
            "candidate score does not expose the E7-I01 score contract"
        ) from error
    pair = CandidatePair(
        pair_index=pair_index,
        category=category,
        current_frame_id=current_frame_id,
        current_keyframe_ordinal=current_ordinal,
        previous_frame_id=previous_frame_id,
        previous_keyframe_ordinal=previous_ordinal,
        rank_within_current=rank_within_current,
        ordinal_gap=current_ordinal - previous_ordinal,
        frame_id_gap=current_frame_id - previous_frame_id,
        mahalanobis_squared=mahalanobis_squared,
        mahalanobis=mahalanobis,
        selection_reason=selection_reason,
    )
    return validate_candidate_pair(
        pair, canonical_frame_ids=canonical_frame_ids
    )


DECLARED_CANDIDATE_PAIRS = (
    CandidatePair(
        pair_index=0,
        category=PROMISING_DISTANT,
        current_frame_id=2331,
        current_keyframe_ordinal=237,
        previous_frame_id=270,
        previous_keyframe_ordinal=28,
        rank_within_current=1,
        ordinal_gap=209,
        frame_id_gap=2061,
        mahalanobis_squared=283665.5652750919,
        mahalanobis=532.6026335600416,
        selection_reason="best distant E7-I01 candidate for FrameId 2331",
    ),
    CandidatePair(
        pair_index=1,
        category=PROMISING_DISTANT,
        current_frame_id=2331,
        current_keyframe_ordinal=237,
        previous_frame_id=356,
        previous_keyframe_ordinal=39,
        rank_within_current=4,
        ordinal_gap=198,
        frame_id_gap=1975,
        mahalanobis_squared=380992.51388679864,
        mahalanobis=617.2459103848308,
        selection_reason=(
            "top alternate cluster for FrameId 2331; tests cap above one"
        ),
    ),
    CandidatePair(
        pair_index=2,
        category=PROMISING_DISTANT,
        current_frame_id=3359,
        current_keyframe_ordinal=355,
        previous_frame_id=372,
        previous_keyframe_ordinal=40,
        rank_within_current=1,
        ordinal_gap=315,
        frame_id_gap=2987,
        mahalanobis_squared=22247.18411722392,
        mahalanobis=149.15489974259617,
        selection_reason="best E7-I01 candidate for final FrameId 3359",
    ),
    CandidatePair(
        pair_index=3,
        category=PROMISING_DISTANT,
        current_frame_id=3359,
        current_keyframe_ordinal=355,
        previous_frame_id=356,
        previous_keyframe_ordinal=39,
        rank_within_current=2,
        ordinal_gap=316,
        frame_id_gap=3003,
        mahalanobis_squared=25712.037678732675,
        mahalanobis=160.34973551188875,
        selection_reason="second-ranked final-frame candidate; tests cap two",
    ),
    CandidatePair(
        pair_index=4,
        category=ADJACENT_CONTROL,
        current_frame_id=1146,
        current_keyframe_ordinal=118,
        previous_frame_id=1142,
        previous_keyframe_ordinal=117,
        rank_within_current=1,
        ordinal_gap=1,
        frame_id_gap=4,
        mahalanobis_squared=3870807.5591374994,
        mahalanobis=1967.4367992739942,
        selection_reason=(
            "rank-one adjacent control demonstrating candidate-slot use"
        ),
    ),
    CandidatePair(
        pair_index=5,
        category=ADJACENT_CONTROL,
        current_frame_id=3359,
        current_keyframe_ordinal=355,
        previous_frame_id=3341,
        previous_keyframe_ordinal=354,
        rank_within_current=198,
        ordinal_gap=1,
        frame_id_gap=18,
        mahalanobis_squared=15450818.97624919,
        mahalanobis=3930.752978278995,
        selection_reason=(
            "adjacent control with poor rank but expected local overlap"
        ),
    ),
    CandidatePair(
        pair_index=6,
        category=POOR_DISTANT_CONTROL,
        current_frame_id=2331,
        current_keyframe_ordinal=237,
        previous_frame_id=1055,
        previous_keyframe_ordinal=109,
        rank_within_current=216,
        ordinal_gap=128,
        frame_id_gap=1276,
        mahalanobis_squared=118754800.92252736,
        mahalanobis=10897.467638058273,
        selection_reason="poor-score distant negative control for FrameId 2331",
    ),
    CandidatePair(
        pair_index=7,
        category=POOR_DISTANT_CONTROL,
        current_frame_id=3359,
        current_keyframe_ordinal=355,
        previous_frame_id=1850,
        previous_keyframe_ordinal=190,
        rank_within_current=350,
        ordinal_gap=165,
        frame_id_gap=1509,
        mahalanobis_squared=116943715.81166022,
        mahalanobis=10814.051775891412,
        selection_reason="poor-score distant negative control for FrameId 3359",
    ),
)


@dataclass(frozen=True)
class RefinementPnPOutcome:
    """One recorded all-supporter refinement solvePnP outcome."""

    refinement_index: int
    total_attempt_number: int
    input_supporter_count: int
    pnp_success: bool
    geometry_valid: bool
    output_supporter_count: Optional[int]
    accepted_strict_increase: bool
    reason: str


@dataclass(frozen=True)
class BoundedRansacResult:
    """Complete terminal evidence from the E7-specific bounded custom RANSAC.

    ``adaptive_completed`` means the current best inlier ratio satisfied the
    99%-confidence attempt requirement and strict supporter refinement ended.
    ``budget_exhausted_with_best_model`` means the same complete best-model
    evidence is retained even though the fixed attempt budget ended first.
    Status alone never grants production acceptance.
    """

    status: str
    terminal_reason: str
    total_hypothesis_attempt_count: int
    random_hypothesis_attempt_count: int
    refinement_hypothesis_attempt_count: int
    required_hypothesis_count: int
    capped_required_hypothesis_count: int
    maximum_hypothesis_attempt_count: int
    successful_pnp_call_count: int
    failed_pnp_call_count: int
    invalid_geometry_model_count: int
    valid_model_count: int
    best_supporter_indices: Tuple[int, ...]
    best_rotations: Optional[Tuple[np.ndarray, ...]]
    best_translations: Optional[Tuple[np.ndarray, ...]]
    refinement_outcomes: Tuple[RefinementPnPOutcome, ...]


@dataclass(frozen=True)
class PairCounts:
    """Counts for each successive stereo, temporal, consensus, and inlier set."""

    previous_left_keypoint_count: int
    previous_right_keypoint_count: int
    current_left_keypoint_count: int
    current_right_keypoint_count: int
    previous_raw_stereo_match_count: int
    current_raw_stereo_match_count: int
    previous_stereo_inlier_count: int
    current_stereo_inlier_count: int
    reciprocal_temporal_match_count: int
    four_view_consensus_count: int
    final_supporter_count: int
    final_supporter_percentage: float


@dataclass(frozen=True)
class PairRuntime:
    """Stage and total wall-clock runtime for one independent pair."""

    image_loading_seconds: float
    stereo_extraction_seconds: float
    temporal_matching_seconds: float
    triangulation_seconds: float
    ransac_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class PairEvidence:
    """Serializable pair identity, count, RANSAC, and runtime evidence."""

    pair: CandidatePair
    counts: PairCounts
    ransac: BoundedRansacResult
    runtime: PairRuntime


@dataclass(frozen=True)
class PairVisualizationData:
    """Non-serialized feature mappings used by figures and the small Bundle.

    In particular, consensus indices and final supporter indices preserve the
    link back to both endpoint stereo observations; this is scientific input
    to measurement construction even though the image arrays are not embedded
    in the compact serialized ``PairEvidence``.
    """

    previous_image_left: np.ndarray
    current_image_left: np.ndarray
    previous_keypoints_left: Sequence[cv2.KeyPoint]
    previous_keypoints_right: Sequence[cv2.KeyPoint]
    current_keypoints_left: Sequence[cv2.KeyPoint]
    current_keypoints_right: Sequence[cv2.KeyPoint]
    previous_stereo_inliers: Sequence[cv2.DMatch]
    current_stereo_inliers: Sequence[cv2.DMatch]
    reciprocal_matches: Sequence[cv2.DMatch]
    consensus: Sequence[Tuple[int, int, int]]
    supporter_indices: Tuple[int, ...]


@dataclass(frozen=True)
class PairRun:
    """Serializable evidence paired with its non-serialized feature mappings."""

    evidence: PairEvidence
    visualization: PairVisualizationData


def file_sha256(path: Path) -> str:
    """Return SHA-256 for a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_pair_seeds(seed: int = CANONICAL_SEED) -> None:
    """Reset every random source used by the protected image/PnP path."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    cv2.setRNGSeed(int(seed))


def validate_declared_candidate_pairs(
    candidate_scores_path: Path = CANONICAL_E7_I01_SCORES,
    *,
    require_canonical_hash: bool = True,
) -> Tuple[CandidatePair, ...]:
    """Cross-check every declared pair against the committed E7-I01 CSV."""

    candidate_scores_path = Path(candidate_scores_path)
    if not candidate_scores_path.is_file():
        raise ConsensusMatchingError(
            "E7-I01 candidate score CSV is unavailable: {}".format(
                candidate_scores_path
            )
        )
    observed_hash = file_sha256(candidate_scores_path)
    if (
        require_canonical_hash
        and observed_hash != CANONICAL_E7_I01_SCORES_SHA256
    ):
        raise ConsensusMatchingError(
            "E7-I01 candidate score CSV hash differs from the baseline"
        )

    selected_keys = {
        (pair.current_frame_id, pair.previous_frame_id)
        for pair in DECLARED_CANDIDATE_PAIRS
    }
    rows: Dict[Tuple[int, int], Dict[str, str]] = {}
    with candidate_scores_path.open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            key = (
                int(row["current_frame_id"]),
                int(row["previous_frame_id"]),
            )
            if key not in selected_keys:
                continue
            if key in rows:
                raise ConsensusMatchingError(
                    "declared candidate appears more than once in E7-I01"
                )
            rows[key] = row

    if set(rows) != selected_keys:
        raise ConsensusMatchingError(
            "not every declared E7-I02 pair exists in E7-I01"
        )

    for pair in DECLARED_CANDIDATE_PAIRS:
        validate_candidate_pair(pair)
        row = rows[(pair.current_frame_id, pair.previous_frame_id)]
        integer_expectations = {
            "current_keyframe_ordinal": pair.current_keyframe_ordinal,
            "previous_keyframe_ordinal": pair.previous_keyframe_ordinal,
            "rank_within_current": pair.rank_within_current,
        }
        for field, expected in integer_expectations.items():
            if int(row[field]) != expected:
                raise ConsensusMatchingError(
                    "declared pair {} field {} differs from E7-I01".format(
                        pair.pair_index, field
                    )
                )
        float_expectations = {
            "mahalanobis_squared": pair.mahalanobis_squared,
            "mahalanobis": pair.mahalanobis,
        }
        for field, expected in float_expectations.items():
            if float(row[field]) != expected:
                raise ConsensusMatchingError(
                    "declared pair {} field {} differs from E7-I01".format(
                        pair.pair_index, field
                    )
                )
        if (
            pair.current_keyframe_ordinal
            - pair.previous_keyframe_ordinal
            != pair.ordinal_gap
            or pair.current_frame_id - pair.previous_frame_id
            != pair.frame_id_gap
        ):
            raise ConsensusMatchingError(
                "declared pair gaps are internally inconsistent"
            )

    if tuple(pair.pair_index for pair in DECLARED_CANDIDATE_PAIRS) != tuple(
        range(8)
    ):
        raise ConsensusMatchingError(
            "declared E7-I02 pair indices must be exactly zero through seven"
        )
    return DECLARED_CANDIDATE_PAIRS


def _required_hypothesis_count(
    consensus_count: int, supporter_count: int
) -> int:
    """Return the canonical 99%-confidence four-sample requirement."""

    if consensus_count < RANSAC_SAMPLE_SIZE:
        return 0
    if supporter_count <= 0:
        outlier_probability = RANSAC_INITIAL_OUTLIER_PROBABILITY
    else:
        inlier_fraction = float(supporter_count) / float(consensus_count)
        if not 0.0 < inlier_fraction <= 1.0:
            raise ConsensusMatchingError("invalid RANSAC inlier fraction")
        if inlier_fraction == 1.0:
            return 1
        outlier_probability = 1.0 - inlier_fraction
    return int(
        tools.calculate_number_of_iteration_for_ransac(
            RANSAC_SUCCESS_PROBABILITY,
            outlier_probability,
            RANSAC_SAMPLE_SIZE,
        )
    )


def _valid_rigid_rotation(rotation: np.ndarray) -> bool:
    rotation = np.asarray(rotation, dtype=float)
    return bool(
        rotation.shape == (3, 3)
        and np.all(np.isfinite(rotation))
        and np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
            rtol=0.0,
        )
        and np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0
        )
    )


def _valid_model(
    rotations: Sequence[np.ndarray], translations: Sequence[np.ndarray]
) -> bool:
    if len(rotations) != 4 or len(translations) != 4:
        return False
    for rotation, translation in zip(rotations, translations):
        if not _valid_rigid_rotation(rotation):
            return False
        normalized_translation = np.asarray(translation, dtype=float)
        if normalized_translation.shape not in ((3,), (3, 1)):
            return False
        if not np.all(np.isfinite(normalized_translation)):
            return False
    return True


def _solve_one_model(
    sampled_consensus: Sequence[Tuple[int, int, int]],
    points_cloud_3d: np.ndarray,
    front_inliers: Sequence[cv2.DMatch],
    front_keypoints_left: Sequence[cv2.KeyPoint],
    intrinsic: np.ndarray,
    back_left_rotation: np.ndarray,
    back_left_translation: np.ndarray,
    fixed_right_rotation: np.ndarray,
    fixed_right_translation: np.ndarray,
) -> Tuple[
    bool,
    bool,
    Optional[Tuple[np.ndarray, ...]],
    Optional[Tuple[np.ndarray, ...]],
    str,
]:
    """Perform exactly one lower-level solvePnP call and compose four cameras."""

    try:
        success, front_left_rotation, front_left_translation = (
            tools.calculate_front_camera_matrix(
                sampled_consensus,
                points_cloud_3d,
                front_inliers,
                front_keypoints_left,
                intrinsic,
            )
        )
    except (AssertionError, TypeError, ValueError, cv2.error) as error:
        return False, False, None, None, "solvePnP_exception:{}".format(
            type(error).__name__
        )
    if not bool(success):
        return False, False, None, None, "solvePnP_returned_false"

    try:
        back_right_rotation, back_right_translation = (
            tools.calculate_right_camera_matrix(
                back_left_rotation,
                back_left_translation,
                fixed_right_rotation,
                fixed_right_translation,
            )
        )
        front_right_rotation, front_right_translation = (
            tools.calculate_right_camera_matrix(
                front_left_rotation,
                front_left_translation,
                fixed_right_rotation,
                fixed_right_translation,
            )
        )
    except (AssertionError, TypeError, ValueError) as error:
        return True, False, None, None, "stereo_composition_exception:{}".format(
            type(error).__name__
        )

    rotations = (
        np.asarray(back_left_rotation, dtype=float),
        np.asarray(back_right_rotation, dtype=float),
        np.asarray(front_left_rotation, dtype=float),
        np.asarray(front_right_rotation, dtype=float),
    )
    translations = (
        np.asarray(back_left_translation, dtype=float),
        np.asarray(back_right_translation, dtype=float),
        np.asarray(front_left_translation, dtype=float),
        np.asarray(front_right_translation, dtype=float),
    )
    if not _valid_model(rotations, translations):
        return True, False, None, None, "non_finite_or_non_rigid_model"
    return True, True, rotations, translations, "valid_model"


def _supporters_for_model(
    consensus_points_3d: np.ndarray,
    actual_pixels: np.ndarray,
    intrinsic: np.ndarray,
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
) -> Optional[Tuple[int, ...]]:
    """Evaluate the canonical four-view two-pixel supporter definition."""

    try:
        indices = tools.find_supporter_indices_for_model(
            consensus_points_3d,
            actual_pixels,
            intrinsic,
            rotations,
            translations,
            max_distance=FOUR_VIEW_SUPPORTER_MAX_DISTANCE_PIXELS,
        )
    except (AssertionError, FloatingPointError, TypeError, ValueError):
        return None
    normalized = np.asarray(indices, dtype=int)
    if normalized.ndim != 1:
        return None
    if normalized.size and (
        int(np.min(normalized)) < 0
        or int(np.max(normalized)) >= consensus_points_3d.shape[0]
    ):
        return None
    return tuple(int(value) for value in normalized.tolist())


def bounded_four_view_ransac(
    points_cloud_3d: np.ndarray,
    consensus: Sequence[Tuple[int, int, int]],
    back_inliers: Sequence[cv2.DMatch],
    front_inliers: Sequence[cv2.DMatch],
    back_keypoints_left: Sequence[cv2.KeyPoint],
    back_keypoints_right: Sequence[cv2.KeyPoint],
    front_keypoints_left: Sequence[cv2.KeyPoint],
    front_keypoints_right: Sequence[cv2.KeyPoint],
    intrinsic: np.ndarray,
    back_left_rotation: np.ndarray,
    back_left_translation: np.ndarray,
    fixed_right_rotation: np.ndarray,
    fixed_right_translation: np.ndarray,
    *,
    maximum_attempts: int = MAX_HYPOTHESIS_ATTEMPTS,
) -> BoundedRansacResult:
    """Run deterministic-budget, four-sample PnP/RANSAC for four-view support.

    The triangulated previous-stereo cloud and four-view consensus mappings
    define the geometry.  Each random four-match hypothesis performs exactly
    one PnP call, composes the two right-camera poses, and evaluates the same
    two-pixel reprojection rule in all four views.  Failed PnP calls and invalid
    geometry consume attempts.  The adaptive 99%-confidence requirement can
    decrease as a better supporter ratio is found but the total attempt count
    is monotonic and never exceeds ``maximum_attempts``.

    Once the random requirement is met, all current supporters are used for
    refinement while they strictly increase the supporter set.  The returned
    result retains the best complete geometry and distinguishes adaptive
    completion from budget exhaustion; it makes no production acceptance
    decision itself.
    """

    # Phase 1: validate the fixed budget and fail closed before PnP when the
    # consensus or triangulated input cannot define a four-sample hypothesis.
    if int(maximum_attempts) != maximum_attempts or maximum_attempts < 1:
        raise ValueError("maximum_attempts must be a positive integer")
    maximum_attempts = int(maximum_attempts)
    consensus = tuple(consensus)
    consensus_count = len(consensus)
    if consensus_count < RANSAC_SAMPLE_SIZE:
        return BoundedRansacResult(
            status=INSUFFICIENT_CONSENSUS,
            terminal_reason="fewer_than_four_four_view_consensus_matches",
            total_hypothesis_attempt_count=0,
            random_hypothesis_attempt_count=0,
            refinement_hypothesis_attempt_count=0,
            required_hypothesis_count=0,
            capped_required_hypothesis_count=0,
            maximum_hypothesis_attempt_count=maximum_attempts,
            successful_pnp_call_count=0,
            failed_pnp_call_count=0,
            invalid_geometry_model_count=0,
            valid_model_count=0,
            best_supporter_indices=(),
            best_rotations=None,
            best_translations=None,
            refinement_outcomes=(),
        )

    points_cloud_3d = np.asarray(points_cloud_3d, dtype=float)
    if (
        points_cloud_3d.ndim != 2
        or points_cloud_3d.shape[1] != 3
        or not np.all(np.isfinite(points_cloud_3d))
    ):
        return BoundedRansacResult(
            status=INVALID_GEOMETRY,
            terminal_reason="triangulated_point_cloud_is_invalid",
            total_hypothesis_attempt_count=0,
            random_hypothesis_attempt_count=0,
            refinement_hypothesis_attempt_count=0,
            required_hypothesis_count=0,
            capped_required_hypothesis_count=0,
            maximum_hypothesis_attempt_count=maximum_attempts,
            successful_pnp_call_count=0,
            failed_pnp_call_count=0,
            invalid_geometry_model_count=0,
            valid_model_count=0,
            best_supporter_indices=(),
            best_rotations=None,
            best_translations=None,
            refinement_outcomes=(),
        )
    # Phase 2: bind every consensus entry to its previous-frame 3D point and
    # to the observed pixel in all four stereo/temporal views.
    try:
        consensus_point_indices = np.asarray(
            [match[0] for match in consensus], dtype=int
        )
        consensus_points_3d = points_cloud_3d[consensus_point_indices]
        actual_pixels = tools.extract_actual_consensus_pixels(
            consensus,
            back_inliers,
            front_inliers,
            back_keypoints_left,
            back_keypoints_right,
            front_keypoints_left,
            front_keypoints_right,
        )
    except (AssertionError, IndexError, TypeError, ValueError) as error:
        return BoundedRansacResult(
            status=INVALID_GEOMETRY,
            terminal_reason="consensus_geometry_exception:{}".format(
                type(error).__name__
            ),
            total_hypothesis_attempt_count=0,
            random_hypothesis_attempt_count=0,
            refinement_hypothesis_attempt_count=0,
            required_hypothesis_count=0,
            capped_required_hypothesis_count=0,
            maximum_hypothesis_attempt_count=maximum_attempts,
            successful_pnp_call_count=0,
            failed_pnp_call_count=0,
            invalid_geometry_model_count=0,
            valid_model_count=0,
            best_supporter_indices=(),
            best_rotations=None,
            best_translations=None,
            refinement_outcomes=(),
        )
    if (
        consensus_points_3d.shape != (consensus_count, 3)
        or not np.all(np.isfinite(consensus_points_3d))
        or actual_pixels.shape != (4, 2, consensus_count)
        or not np.all(np.isfinite(actual_pixels))
    ):
        return BoundedRansacResult(
            status=INVALID_GEOMETRY,
            terminal_reason="consensus_points_or_pixels_are_invalid",
            total_hypothesis_attempt_count=0,
            random_hypothesis_attempt_count=0,
            refinement_hypothesis_attempt_count=0,
            required_hypothesis_count=0,
            capped_required_hypothesis_count=0,
            maximum_hypothesis_attempt_count=maximum_attempts,
            successful_pnp_call_count=0,
            failed_pnp_call_count=0,
            invalid_geometry_model_count=0,
            valid_model_count=0,
            best_supporter_indices=(),
            best_rotations=None,
            best_translations=None,
            refinement_outcomes=(),
        )

    # Phase 3: initialize monotonic hypothesis accounting and the best-model
    # record. The adaptive requirement starts from the frozen outlier prior.
    total_attempts = 0
    random_attempts = 0
    refinement_attempts = 0
    successful_pnp_calls = 0
    failed_pnp_calls = 0
    invalid_geometry_models = 0
    valid_models = 0
    required_attempts = _required_hypothesis_count(consensus_count, 0)
    best_supporters: Tuple[int, ...] = ()
    best_rotations: Optional[Tuple[np.ndarray, ...]] = None
    best_translations: Optional[Tuple[np.ndarray, ...]] = None

    # Phase 4: sample four consensus matches per hypothesis. Every solvePnP
    # call consumes one budget slot, including failed or invalid models.
    while total_attempts < min(required_attempts, maximum_attempts):
        sampled_consensus = random.sample(
            consensus, RANSAC_SAMPLE_SIZE
        )
        total_attempts += 1
        random_attempts += 1
        (
            pnp_success,
            geometry_valid,
            rotations,
            translations,
            _,
        ) = _solve_one_model(
            sampled_consensus,
            points_cloud_3d,
            front_inliers,
            front_keypoints_left,
            intrinsic,
            back_left_rotation,
            back_left_translation,
            fixed_right_rotation,
            fixed_right_translation,
        )
        if not pnp_success:
            failed_pnp_calls += 1
            continue
        successful_pnp_calls += 1
        if not geometry_valid or rotations is None or translations is None:
            invalid_geometry_models += 1
            continue
        # A supporter must satisfy the two-pixel rule in every one of the four
        # views; it is stricter evidence than membership in consensus alone.
        supporters = _supporters_for_model(
            consensus_points_3d,
            actual_pixels,
            intrinsic,
            rotations,
            translations,
        )
        if supporters is None:
            invalid_geometry_models += 1
            continue
        valid_models += 1
        if best_rotations is None or len(supporters) > len(best_supporters):
            best_supporters = supporters
            best_rotations = rotations
            best_translations = translations
            required_attempts = _required_hypothesis_count(
                consensus_count, len(best_supporters)
            )

    # Phase 5: retain an explicit terminal failure when no complete valid PnP
    # geometry survived; no partial model is promoted to later stages.
    if best_rotations is None or best_translations is None:
        if successful_pnp_calls > 0 and invalid_geometry_models > 0:
            status = INVALID_GEOMETRY
            terminal_reason = "no_geometry_valid_model_was_produced"
        else:
            status = NO_VALID_PNP_MODEL
            terminal_reason = "no_solvePnP_call_produced_a_valid_model"
        return BoundedRansacResult(
            status=status,
            terminal_reason=terminal_reason,
            total_hypothesis_attempt_count=total_attempts,
            random_hypothesis_attempt_count=random_attempts,
            refinement_hypothesis_attempt_count=0,
            required_hypothesis_count=required_attempts,
            capped_required_hypothesis_count=min(
                required_attempts, maximum_attempts
            ),
            maximum_hypothesis_attempt_count=maximum_attempts,
            successful_pnp_call_count=successful_pnp_calls,
            failed_pnp_call_count=failed_pnp_calls,
            invalid_geometry_model_count=invalid_geometry_models,
            valid_model_count=valid_models,
            best_supporter_indices=(),
            best_rotations=None,
            best_translations=None,
            refinement_outcomes=(),
        )

    # Phase 6: only an adaptively completed random search enters all-supporter
    # refinement, and only a strict supporter increase replaces the best model.
    random_adaptive_completed = total_attempts >= required_attempts
    refinement_outcomes = []
    refinement_needs_another_iteration = len(best_supporters) >= 4
    while (
        random_adaptive_completed
        and refinement_needs_another_iteration
        and total_attempts < maximum_attempts
    ):
        refinement_consensus = tuple(
            consensus[index] for index in best_supporters
        )
        total_attempts += 1
        refinement_attempts += 1
        (
            pnp_success,
            geometry_valid,
            rotations,
            translations,
            model_reason,
        ) = _solve_one_model(
            refinement_consensus,
            points_cloud_3d,
            front_inliers,
            front_keypoints_left,
            intrinsic,
            back_left_rotation,
            back_left_translation,
            fixed_right_rotation,
            fixed_right_translation,
        )
        output_supporter_count: Optional[int] = None
        accepted = False
        reason = model_reason
        if not pnp_success:
            failed_pnp_calls += 1
            refinement_needs_another_iteration = False
        else:
            successful_pnp_calls += 1
            if not geometry_valid or rotations is None or translations is None:
                invalid_geometry_models += 1
                refinement_needs_another_iteration = False
            else:
                supporters = _supporters_for_model(
                    consensus_points_3d,
                    actual_pixels,
                    intrinsic,
                    rotations,
                    translations,
                )
                if supporters is None:
                    invalid_geometry_models += 1
                    reason = "supporter_evaluation_invalid"
                    refinement_needs_another_iteration = False
                else:
                    valid_models += 1
                    output_supporter_count = len(supporters)
                    if len(supporters) > len(best_supporters):
                        best_supporters = supporters
                        best_rotations = rotations
                        best_translations = translations
                        required_attempts = _required_hypothesis_count(
                            consensus_count, len(best_supporters)
                        )
                        accepted = True
                        reason = "strict_supporter_increase"
                    else:
                        refinement_needs_another_iteration = False
                        reason = "no_strict_supporter_increase"
        refinement_outcomes.append(
            RefinementPnPOutcome(
                refinement_index=refinement_attempts - 1,
                total_attempt_number=total_attempts,
                input_supporter_count=len(refinement_consensus),
                pnp_success=bool(pnp_success),
                geometry_valid=bool(geometry_valid),
                output_supporter_count=output_supporter_count,
                accepted_strict_increase=accepted,
                reason=reason,
            )
        )

    # Phase 7: expose how the bounded search ended while preserving the same
    # best-model evidence for the later, separate production policy decision.
    if not random_adaptive_completed:
        status = BUDGET_EXHAUSTED_WITH_BEST_MODEL
        terminal_reason = (
            "total_attempt_budget_reached_before_adaptive_requirement"
        )
    elif (
        refinement_needs_another_iteration
        and total_attempts >= maximum_attempts
    ):
        status = BUDGET_EXHAUSTED_WITH_BEST_MODEL
        terminal_reason = (
            "total_attempt_budget_reached_during_strict_refinement"
        )
    else:
        status = ADAPTIVE_COMPLETED
        terminal_reason = (
            "adaptive_requirement_met_and_refinement_terminated"
        )

    return BoundedRansacResult(
        status=status,
        terminal_reason=terminal_reason,
        total_hypothesis_attempt_count=total_attempts,
        random_hypothesis_attempt_count=random_attempts,
        refinement_hypothesis_attempt_count=refinement_attempts,
        required_hypothesis_count=required_attempts,
        capped_required_hypothesis_count=min(
            required_attempts, maximum_attempts
        ),
        maximum_hypothesis_attempt_count=maximum_attempts,
        successful_pnp_call_count=successful_pnp_calls,
        failed_pnp_call_count=failed_pnp_calls,
        invalid_geometry_model_count=invalid_geometry_models,
        valid_model_count=valid_models,
        best_supporter_indices=best_supporters,
        best_rotations=best_rotations,
        best_translations=best_translations,
        refinement_outcomes=tuple(refinement_outcomes),
    )


def evaluate_candidate_pair(
    pair: CandidatePair,
    *,
    seed: int = CANONICAL_SEED,
    maximum_attempts: int = MAX_HYPOTHESIS_ATTEMPTS,
) -> PairRun:
    """Evaluate one candidate through stereo, temporal, and bounded PnP stages.

    The seed is reset per pair, so pair evidence is independent of evaluation
    order.  The returned ``PairEvidence`` records all count/status/runtime
    contracts; ``PairVisualizationData`` retains the feature-to-consensus and
    supporter mappings required for a later two-pose Bundle.  This function
    does not decide production acceptance and does not mutate Pose Graph state.
    """

    # Phase 1: make the pair evaluation deterministic for the frozen seed and
    # load the previous/current left-right image quartette.
    total_started = time.perf_counter()
    reset_pair_seeds(seed)

    # Continue phase 1 with separately timed loading of the four source images.
    stage_started = time.perf_counter()
    previous_image_left, previous_image_right = (
        tools.read_images_from_dataset(pair.previous_frame_id)
    )
    current_image_left, current_image_right = (
        tools.read_images_from_dataset(pair.current_frame_id)
    )
    image_loading_seconds = time.perf_counter() - stage_started
    if any(
        image is None
        for image in (
            previous_image_left,
            previous_image_right,
            current_image_left,
            current_image_right,
        )
    ):
        raise FileNotFoundError(
            "stereo images are unavailable for declared pair {}".format(
                pair.pair_index
            )
        )

    # Phase 2: extract and stereo-match each endpoint independently. The tool
    # returns both raw stereo outliers and rectified-geometry stereo inliers.
    stage_started = time.perf_counter()
    previous_stereo = tools.extract_keypoints_and_inliers(
        previous_image_left, previous_image_right
    )
    current_stereo = tools.extract_keypoints_and_inliers(
        current_image_left, current_image_right
    )
    stereo_extraction_seconds = time.perf_counter() - stage_started
    (
        previous_keypoints_left,
        previous_descriptors_left,
        previous_keypoints_right,
        _,
        previous_stereo_inliers,
        previous_stereo_outliers,
    ) = previous_stereo
    (
        current_keypoints_left,
        current_descriptors_left,
        current_keypoints_right,
        _,
        current_stereo_inliers,
        current_stereo_outliers,
    ) = current_stereo
    if (
        previous_descriptors_left is None
        or current_descriptors_left is None
    ):
        raise ConsensusMatchingError(
            "declared pair {} has missing left descriptors".format(
                pair.pair_index
            )
        )

    # Phase 3: reciprocal matching between the two left images supplies the
    # temporal relation; intersecting it with both stereo-inlier maps creates
    # four-view consensus rather than yet declaring geometric supporters.
    stage_started = time.perf_counter()
    reciprocal_matches = tools.match_reciprocal(
        previous_descriptors_left,
        current_descriptors_left,
        matcher=tools.create_matcher(),
    )
    consensus = tools.find_consensus_matches_indices(
        previous_stereo_inliers,
        current_stereo_inliers,
        reciprocal_matches,
    )
    temporal_matching_seconds = time.perf_counter() - stage_started

    # Phase 4: use the fixed calibrated stereo geometry. The previous left
    # camera is the local frame for triangulation and PnP model construction.
    intrinsic, initial_left, initial_right = tools.read_cameras_matrices()
    back_left_rotation = initial_left[:, :3]
    back_left_translation = initial_left[:, 3:]
    fixed_right_rotation = initial_right[:, :3]
    fixed_right_translation = initial_right[:, 3:]

    # Phase 5: triangulate the previous endpoint's stereo inliers; consensus
    # indices select from this cloud inside the bounded PnP/RANSAC stage.
    stage_started = time.perf_counter()
    point_cloud = tools.cv_triangulate_matched_points(
        previous_keypoints_left,
        previous_keypoints_right,
        previous_stereo_inliers,
        intrinsic,
        back_left_rotation,
        back_left_translation,
        fixed_right_rotation,
        fixed_right_translation,
    )
    triangulation_seconds = time.perf_counter() - stage_started

    # Phase 6: estimate and retain the best complete four-camera model under
    # deterministic adaptive/budget accounting and the two-pixel supporter rule.
    stage_started = time.perf_counter()
    ransac = bounded_four_view_ransac(
        point_cloud,
        consensus,
        previous_stereo_inliers,
        current_stereo_inliers,
        previous_keypoints_left,
        previous_keypoints_right,
        current_keypoints_left,
        current_keypoints_right,
        intrinsic,
        back_left_rotation,
        back_left_translation,
        fixed_right_rotation,
        fixed_right_translation,
        maximum_attempts=maximum_attempts,
    )
    ransac_seconds = time.perf_counter() - stage_started

    # Phase 7: retain the successive evidence-set sizes and the exact mappings
    # needed for later measurement construction; acceptance remains separate.
    supporter_count = len(ransac.best_supporter_indices)
    consensus_count = len(consensus)
    supporter_percentage = (
        100.0 * float(supporter_count) / float(consensus_count)
        if consensus_count
        else 0.0
    )
    counts = PairCounts(
        previous_left_keypoint_count=len(previous_keypoints_left),
        previous_right_keypoint_count=len(previous_keypoints_right),
        current_left_keypoint_count=len(current_keypoints_left),
        current_right_keypoint_count=len(current_keypoints_right),
        previous_raw_stereo_match_count=(
            len(previous_stereo_inliers) + len(previous_stereo_outliers)
        ),
        current_raw_stereo_match_count=(
            len(current_stereo_inliers) + len(current_stereo_outliers)
        ),
        previous_stereo_inlier_count=len(previous_stereo_inliers),
        current_stereo_inlier_count=len(current_stereo_inliers),
        reciprocal_temporal_match_count=len(reciprocal_matches),
        four_view_consensus_count=consensus_count,
        final_supporter_count=supporter_count,
        final_supporter_percentage=supporter_percentage,
    )
    runtime = PairRuntime(
        image_loading_seconds=float(image_loading_seconds),
        stereo_extraction_seconds=float(stereo_extraction_seconds),
        temporal_matching_seconds=float(temporal_matching_seconds),
        triangulation_seconds=float(triangulation_seconds),
        ransac_seconds=float(ransac_seconds),
        total_seconds=float(time.perf_counter() - total_started),
    )
    evidence = PairEvidence(
        pair=pair,
        counts=counts,
        ransac=ransac,
        runtime=runtime,
    )
    visualization = PairVisualizationData(
        previous_image_left=previous_image_left,
        current_image_left=current_image_left,
        previous_keypoints_left=previous_keypoints_left,
        previous_keypoints_right=previous_keypoints_right,
        current_keypoints_left=current_keypoints_left,
        current_keypoints_right=current_keypoints_right,
        previous_stereo_inliers=previous_stereo_inliers,
        current_stereo_inliers=current_stereo_inliers,
        reciprocal_matches=reciprocal_matches,
        consensus=consensus,
        supporter_indices=ransac.best_supporter_indices,
    )
    return PairRun(evidence=evidence, visualization=visualization)


def _complete_production_model_evidence(
    evidence: PairEvidence,
) -> Tuple[bool, str]:
    """Check the retained best-model contract shared by eligible statuses."""

    counts = evidence.counts
    ransac = evidence.ransac
    if ransac.valid_model_count <= 0:
        return False, "no_valid_retained_model"
    if (
        ransac.best_rotations is None
        or ransac.best_translations is None
        or not _valid_model(
            ransac.best_rotations, ransac.best_translations
        )
    ):
        return False, "retained_best_model_is_incomplete_or_invalid"
    supporters = ransac.best_supporter_indices
    if (
        len(supporters) != counts.final_supporter_count
        or len(set(supporters)) != len(supporters)
        or any(
            type(index) is not int
            or index < 0
            or index >= counts.four_view_consensus_count
            for index in supporters
        )
    ):
        return False, "retained_supporter_mapping_is_invalid"
    if ransac.status == ADAPTIVE_COMPLETED:
        if (
            ransac.terminal_reason
            != "adaptive_requirement_met_and_refinement_terminated"
        ):
            return False, "adaptive_terminal_reason_is_inconsistent"
    elif ransac.status == BUDGET_EXHAUSTED_WITH_BEST_MODEL:
        allowed_reasons = {
            "total_attempt_budget_reached_before_adaptive_requirement",
            "total_attempt_budget_reached_during_strict_refinement",
        }
        if (
            ransac.maximum_hypothesis_attempt_count
            != MAX_HYPOTHESIS_ATTEMPTS
            or ransac.total_hypothesis_attempt_count
            != MAX_HYPOTHESIS_ATTEMPTS
            or ransac.terminal_reason not in allowed_reasons
        ):
            return False, "production_attempt_budget_evidence_is_inconsistent"
    else:
        return False, "ransac_terminal_status_is_not_production_eligible"
    return True, "complete_valid_retained_model"


def production_match_policy_decision(
    evidence: PairEvidence,
) -> Tuple[bool, str]:
    """Apply complete-model and inclusive 30-consensus/30-supporter/30% gates.

    Both adaptive completion and budget exhaustion with a retained best model
    are eligible to face the same scientific gates.  Terminal status is
    evidence about search completion, not an acceptance shortcut.
    """

    validate_pair_evidence(evidence)
    if evidence.ransac.status not in PRODUCTION_ELIGIBLE_TERMINAL_STATUSES:
        return False, "ransac_terminal_status_is_not_production_eligible"
    complete, reason = _complete_production_model_evidence(evidence)
    if not complete:
        return False, reason
    counts = evidence.counts
    if (
        counts.four_view_consensus_count
        < PRODUCTION_MINIMUM_CONSENSUS_COUNT
    ):
        return False, "consensus_below_30"
    if counts.final_supporter_count < PRODUCTION_MINIMUM_SUPPORTER_COUNT:
        return False, "supporters_below_30"
    if (
        counts.final_supporter_percentage
        < PRODUCTION_MINIMUM_SUPPORTER_PERCENTAGE
    ):
        return False, "supporter_percentage_below_30"
    return True, "accepted_complete_model_and_30_30_30"


def validate_pair_evidence(evidence: PairEvidence) -> None:
    """Validate count, attempt, percentage, and terminal-state invariants."""

    validate_candidate_pair(evidence.pair)
    counts = evidence.counts
    ransac = evidence.ransac
    if ransac.status not in TERMINAL_STATUSES:
        raise ConsensusMatchingError("unknown E7-I02 terminal status")
    if not (
        0
        <= counts.final_supporter_count
        <= counts.four_view_consensus_count
        <= counts.reciprocal_temporal_match_count
    ):
        raise ConsensusMatchingError("match-count ordering is invalid")
    expected_percentage = (
        100.0
        * counts.final_supporter_count
        / counts.four_view_consensus_count
        if counts.four_view_consensus_count
        else 0.0
    )
    if not math.isclose(
        counts.final_supporter_percentage,
        expected_percentage,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ConsensusMatchingError("supporter percentage is inconsistent")
    if (
        counts.previous_stereo_inlier_count
        > counts.previous_raw_stereo_match_count
        or counts.current_stereo_inlier_count
        > counts.current_raw_stereo_match_count
    ):
        raise ConsensusMatchingError("stereo-inlier count is invalid")
    if ransac.total_hypothesis_attempt_count != (
        ransac.random_hypothesis_attempt_count
        + ransac.refinement_hypothesis_attempt_count
    ):
        raise ConsensusMatchingError("hypothesis attempt accounting differs")
    if ransac.total_hypothesis_attempt_count > (
        ransac.maximum_hypothesis_attempt_count
    ):
        raise ConsensusMatchingError("hypothesis attempt budget was exceeded")
    if ransac.total_hypothesis_attempt_count != (
        ransac.successful_pnp_call_count + ransac.failed_pnp_call_count
    ):
        raise ConsensusMatchingError(
            "every hypothesis attempt must be one recorded solvePnP call"
        )
    if ransac.capped_required_hypothesis_count != min(
        ransac.required_hypothesis_count,
        ransac.maximum_hypothesis_attempt_count,
    ):
        raise ConsensusMatchingError("capped RANSAC requirement differs")
    if len(ransac.refinement_outcomes) != (
        ransac.refinement_hypothesis_attempt_count
    ):
        raise ConsensusMatchingError(
            "refinement PnP outcomes are not completely recorded"
        )
    accepted_counts = [
        outcome.output_supporter_count
        for outcome in ransac.refinement_outcomes
        if outcome.accepted_strict_increase
    ]
    if any(value is None for value in accepted_counts):
        raise ConsensusMatchingError(
            "accepted refinement lacks a supporter count"
        )
    if any(
        later <= earlier
        for earlier, later in zip(accepted_counts, accepted_counts[1:])
    ):
        raise ConsensusMatchingError(
            "accepted refinements are not strictly increasing"
        )
    if len(ransac.best_supporter_indices) != counts.final_supporter_count:
        raise ConsensusMatchingError(
            "retained supporter indices differ from the final count"
        )
    if len(set(ransac.best_supporter_indices)) != len(
        ransac.best_supporter_indices
    ):
        raise ConsensusMatchingError(
            "retained supporter indices are not unique"
        )
    if any(
        type(index) is not int
        or index < 0
        or index >= counts.four_view_consensus_count
        for index in ransac.best_supporter_indices
    ):
        raise ConsensusMatchingError(
            "retained supporter index is outside the consensus"
        )
    runtime_values = (
        evidence.runtime.image_loading_seconds,
        evidence.runtime.stereo_extraction_seconds,
        evidence.runtime.temporal_matching_seconds,
        evidence.runtime.triangulation_seconds,
        evidence.runtime.ransac_seconds,
        evidence.runtime.total_seconds,
    )
    if not all(
        np.isfinite(value) and value >= 0.0 for value in runtime_values
    ):
        raise ConsensusMatchingError("pair runtime is invalid")
