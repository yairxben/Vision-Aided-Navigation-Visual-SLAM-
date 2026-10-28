"""Validated EPNP and custom four-view RANSAC policies."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError
from .stereo import compose_right_camera, project_points
from .temporal import FourViewCorrespondence, actual_consensus_pixels


ADAPTIVE_COMPLETED = "adaptive_completed"
BUDGET_EXHAUSTED_WITH_BEST_MODEL = "budget_exhausted_with_best_model"
INSUFFICIENT_CONSENSUS = "insufficient_consensus"
NO_VALID_PNP_MODEL = "no_valid_pnp_model"
INVALID_GEOMETRY = "invalid_geometry"


@dataclass(frozen=True)
class CameraModel:
    rotations: Tuple[np.ndarray, ...]
    translations: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class RefinementPnPOutcome:
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
    best_model: Optional[CameraModel]
    refinement_outcomes: Tuple[RefinementPnPOutcome, ...]


@dataclass(frozen=True)
class VisualOdometryRansacResult:
    model: CameraModel
    supporter_indices: Tuple[int, ...]
    supporters: Tuple[FourViewCorrespondence, ...]
    total_hypothesis_attempt_count: int
    elapsed_seconds: float


def required_hypothesis_count(
    success_probability: float,
    outlier_probability: float,
    sample_size: int,
) -> int:
    if sample_size <= 0:
        raise ContractError("RANSAC sample size must be positive")
    numerator = np.log(1.0 - success_probability)
    denominator = np.log(
        1.0 - np.power(1.0 - outlier_probability, sample_size)
    )
    return int(numerator / denominator) + 1


def solve_front_camera(
    consensus_sample: Sequence[FourViewCorrespondence],
    previous_points_3d: np.ndarray,
    current_stereo_inliers: Sequence[cv2.DMatch],
    current_left_keypoints: Sequence[cv2.KeyPoint],
    intrinsic: np.ndarray,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Solve one EPNP model using the exact consensus-to-feature mappings."""

    if len(consensus_sample) < 4:
        raise ContractError("PnP requires at least four consensus matches")
    points_3d = np.asarray(previous_points_3d, dtype=float)
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ContractError("previous point cloud must have shape Nx3")
    object_points = np.asarray(
        [points_3d[item.previous_stereo_index] for item in consensus_sample],
        dtype=float,
    )
    image_points = np.asarray(
        [
            current_left_keypoints[
                current_stereo_inliers[item.current_stereo_index].queryIdx
            ].pt
            for item in consensus_sample
        ],
        dtype=float,
    )
    success, rotation_vector, translation = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=np.asarray(intrinsic, dtype=float),
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    rotation = cv2.Rodrigues(rotation_vector)[0]
    return bool(success), rotation, translation


def build_four_camera_model(
    consensus_sample: Sequence[FourViewCorrespondence],
    previous_points_3d: np.ndarray,
    current_stereo_inliers: Sequence[cv2.DMatch],
    current_left_keypoints: Sequence[cv2.KeyPoint],
    intrinsic: np.ndarray,
    previous_left_rotation: np.ndarray,
    previous_left_translation: np.ndarray,
    fixed_right_rotation: np.ndarray,
    fixed_right_translation: np.ndarray,
) -> Tuple[bool, Optional[CameraModel]]:
    success, current_left_rotation, current_left_translation = solve_front_camera(
        consensus_sample,
        previous_points_3d,
        current_stereo_inliers,
        current_left_keypoints,
        intrinsic,
    )
    if not success:
        return False, None
    previous_right_rotation, previous_right_translation = compose_right_camera(
        previous_left_rotation,
        previous_left_translation,
        fixed_right_rotation,
        fixed_right_translation,
    )
    current_right_rotation, current_right_translation = compose_right_camera(
        current_left_rotation,
        current_left_translation,
        fixed_right_rotation,
        fixed_right_translation,
    )
    return True, CameraModel(
        rotations=(
            np.asarray(previous_left_rotation, dtype=float),
            previous_right_rotation,
            current_left_rotation,
            current_right_rotation,
        ),
        translations=(
            np.asarray(previous_left_translation, dtype=float).reshape(3, 1),
            previous_right_translation,
            current_left_translation,
            current_right_translation,
        ),
    )


_IDENTITY_3 = np.eye(3)


def model_is_valid(model: CameraModel) -> bool:
    if len(model.rotations) != 4 or len(model.translations) != 4:
        return False
    rotations = [np.asarray(rotation, dtype=float) for rotation in model.rotations]
    translations = [
        np.asarray(translation, dtype=float) for translation in model.translations
    ]
    if any(rotation.shape != (3, 3) for rotation in rotations) or any(
        translation.shape not in ((3,), (3, 1)) for translation in translations
    ):
        return False
    rotations = np.stack(rotations)
    if not np.isfinite(rotations).all() or not np.isfinite(
        np.concatenate([translation.reshape(3) for translation in translations])
    ).all():
        return False
    # All four cameras at once. For finite values allclose/isclose with
    # rtol=0 are exactly |a - b| <= atol, without their per-call overhead.
    orthonormality = np.matmul(rotations.transpose(0, 2, 1), rotations)
    if not (np.abs(orthonormality - _IDENTITY_3) <= 1e-5).all():
        return False
    return bool((np.abs(np.linalg.det(rotations) - 1.0) <= 1e-5).all())


def supporter_indices(
    consensus_points_3d: np.ndarray,
    observed_pixels: np.ndarray,
    intrinsic: np.ndarray,
    model: CameraModel,
    maximum_distance_px: float,
) -> Tuple[int, ...]:
    calculated = project_points(
        consensus_points_3d,
        intrinsic,
        model.rotations,
        model.translations,
    )
    if observed_pixels.shape != calculated.shape:
        raise ContractError("observed and projected pixel shapes differ")
    distances = np.linalg.norm(observed_pixels - calculated, ord=2, axis=1)
    return tuple(
        int(index)
        for index in np.where((distances <= maximum_distance_px).all(axis=0))[0]
    )


def _consensus_geometry(
    previous_points_3d: np.ndarray,
    consensus: Sequence[FourViewCorrespondence],
    previous_stereo_inliers: Sequence[cv2.DMatch],
    current_stereo_inliers: Sequence[cv2.DMatch],
    previous_left_keypoints: Sequence[cv2.KeyPoint],
    previous_right_keypoints: Sequence[cv2.KeyPoint],
    current_left_keypoints: Sequence[cv2.KeyPoint],
    current_right_keypoints: Sequence[cv2.KeyPoint],
) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(previous_points_3d, dtype=float)
    consensus_points = points[
        np.asarray([item.previous_stereo_index for item in consensus], dtype=int)
    ]
    observed = actual_consensus_pixels(
        consensus,
        previous_stereo_inliers,
        current_stereo_inliers,
        previous_left_keypoints,
        previous_right_keypoints,
        current_left_keypoints,
        current_right_keypoints,
    )
    return consensus_points, observed


def run_visual_odometry_ransac(
    previous_points_3d: np.ndarray,
    consensus: Sequence[FourViewCorrespondence],
    previous_stereo_inliers: Sequence[cv2.DMatch],
    current_stereo_inliers: Sequence[cv2.DMatch],
    previous_left_keypoints: Sequence[cv2.KeyPoint],
    previous_right_keypoints: Sequence[cv2.KeyPoint],
    current_left_keypoints: Sequence[cv2.KeyPoint],
    current_right_keypoints: Sequence[cv2.KeyPoint],
    intrinsic: np.ndarray,
    previous_left_rotation: np.ndarray,
    previous_left_translation: np.ndarray,
    fixed_right_rotation: np.ndarray,
    fixed_right_translation: np.ndarray,
    *,
    success_probability: float,
    initial_outlier_probability: float,
    sample_size: int,
    maximum_distance_px: float,
    maximum_attempts: Optional[int] = None,
) -> VisualOdometryRansacResult:
    """Run the protected unbounded adaptive VO search and strict refinement."""

    started = time.perf_counter()
    consensus = tuple(consensus)
    if len(consensus) < sample_size:
        raise ContractError("insufficient four-view consensus for VO RANSAC")
    consensus_points, observed = _consensus_geometry(
        previous_points_3d,
        consensus,
        previous_stereo_inliers,
        current_stereo_inliers,
        previous_left_keypoints,
        previous_right_keypoints,
        current_left_keypoints,
        current_right_keypoints,
    )
    remaining = required_hypothesis_count(
        success_probability, initial_outlier_probability, sample_size
    )
    best_supporters: Tuple[int, ...] = ()
    best_model: Optional[CameraModel] = None
    attempts = 0
    while remaining > 0:
        if maximum_attempts is not None and attempts >= maximum_attempts:
            if best_model is None:
                raise ContractError("VO RANSAC exhausted its experimental attempt cap")
            break
        sample = random.sample(consensus, sample_size)
        # The validated VO driver retries failed solvePnP calls without consuming
        # its adaptive iteration count. An explicit experiment cap still counts
        # all calls so a bad dataset cannot run forever when one was requested.
        while True:
            attempts += 1
            try:
                success, model = build_four_camera_model(
                    sample,
                    previous_points_3d,
                    current_stereo_inliers,
                    current_left_keypoints,
                    intrinsic,
                    previous_left_rotation,
                    previous_left_translation,
                    fixed_right_rotation,
                    fixed_right_translation,
                )
            except (cv2.error, ValueError, ContractError):
                success, model = False, None
            if success and model is not None:
                break
            if maximum_attempts is not None and attempts >= maximum_attempts:
                if best_model is None:
                    raise ContractError("VO RANSAC exhausted its experimental attempt cap")
                return VisualOdometryRansacResult(
                    model=best_model,
                    supporter_indices=best_supporters,
                    supporters=tuple(consensus[index] for index in best_supporters),
                    total_hypothesis_attempt_count=attempts,
                    elapsed_seconds=time.perf_counter() - started,
                )
            sample = random.sample(consensus, sample_size)
        supporters = supporter_indices(
            consensus_points, observed, intrinsic, model, maximum_distance_px
        )
        if len(supporters) > len(best_supporters):
            best_model = model
            best_supporters = supporters
            outlier_probability = 1.0 - len(supporters) / len(consensus)
            if outlier_probability <= 0.0:
                remaining = 1
            else:
                remaining = required_hypothesis_count(
                    success_probability, outlier_probability, sample_size
                )
        else:
            remaining -= 1

    if best_model is None or len(best_supporters) < sample_size:
        raise ContractError("VO RANSAC produced no refinable model")
    while True:
        refinement = tuple(consensus[index] for index in best_supporters)
        attempts += 1
        success, model = build_four_camera_model(
            refinement,
            previous_points_3d,
            current_stereo_inliers,
            current_left_keypoints,
            intrinsic,
            best_model.rotations[0],
            best_model.translations[0],
            fixed_right_rotation,
            fixed_right_translation,
        )
        if not success or model is None:
            continue
        supporters = supporter_indices(
            consensus_points, observed, intrinsic, model, maximum_distance_px
        )
        if len(supporters) > len(best_supporters):
            best_supporters = supporters
            best_model = model
        else:
            break
    return VisualOdometryRansacResult(
        model=best_model,
        supporter_indices=best_supporters,
        supporters=tuple(consensus[index] for index in best_supporters),
        total_hypothesis_attempt_count=attempts,
        elapsed_seconds=time.perf_counter() - started,
    )


def _bounded_requirement(
    consensus_count: int,
    supporter_count: int,
    success_probability: float,
    initial_outlier_probability: float,
    sample_size: int,
) -> int:
    if consensus_count < sample_size:
        return 0
    if supporter_count <= 0:
        outlier_probability = initial_outlier_probability
    else:
        inlier_fraction = float(supporter_count) / float(consensus_count)
        if inlier_fraction == 1.0:
            return 1
        outlier_probability = 1.0 - inlier_fraction
    return required_hypothesis_count(
        success_probability, outlier_probability, sample_size
    )


def run_bounded_four_view_ransac(
    previous_points_3d: np.ndarray,
    consensus: Sequence[FourViewCorrespondence],
    previous_stereo_inliers: Sequence[cv2.DMatch],
    current_stereo_inliers: Sequence[cv2.DMatch],
    previous_left_keypoints: Sequence[cv2.KeyPoint],
    previous_right_keypoints: Sequence[cv2.KeyPoint],
    current_left_keypoints: Sequence[cv2.KeyPoint],
    current_right_keypoints: Sequence[cv2.KeyPoint],
    intrinsic: np.ndarray,
    previous_left_rotation: np.ndarray,
    previous_left_translation: np.ndarray,
    fixed_right_rotation: np.ndarray,
    fixed_right_translation: np.ndarray,
    *,
    success_probability: float,
    initial_outlier_probability: float,
    sample_size: int,
    maximum_distance_px: float,
    maximum_attempts: int,
) -> BoundedRansacResult:
    """Run the validated Loop Closure RANSAC with monotonic attempt accounting."""

    if maximum_attempts < 1:
        raise ContractError("maximum RANSAC attempts must be positive")
    consensus = tuple(consensus)
    if len(consensus) < sample_size:
        return BoundedRansacResult(
            INSUFFICIENT_CONSENSUS,
            "fewer_than_sample_size_four_view_consensus_matches",
            0, 0, 0, 0, 0, maximum_attempts, 0, 0, 0, 0, (), None, (),
        )
    points = np.asarray(previous_points_3d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        return BoundedRansacResult(
            INVALID_GEOMETRY, "triangulated_point_cloud_is_invalid",
            0, 0, 0, 0, 0, maximum_attempts, 0, 0, 0, 0, (), None, (),
        )
    try:
        consensus_points, observed = _consensus_geometry(
            points, consensus, previous_stereo_inliers, current_stereo_inliers,
            previous_left_keypoints, previous_right_keypoints,
            current_left_keypoints, current_right_keypoints,
        )
    except (IndexError, TypeError, ValueError, ContractError):
        return BoundedRansacResult(
            INVALID_GEOMETRY, "consensus_geometry_is_invalid",
            0, 0, 0, 0, 0, maximum_attempts, 0, 0, 0, 0, (), None, (),
        )
    if (
        consensus_points.shape != (len(consensus), 3)
        or observed.shape != (4, 2, len(consensus))
        or not np.all(np.isfinite(consensus_points))
        or not np.all(np.isfinite(observed))
    ):
        return BoundedRansacResult(
            INVALID_GEOMETRY, "consensus_points_or_pixels_are_invalid",
            0, 0, 0, 0, 0, maximum_attempts, 0, 0, 0, 0, (), None, (),
        )

    attempts = random_attempts = refinement_attempts = 0
    successful = failed = invalid = valid = 0
    required = _bounded_requirement(
        len(consensus), 0, success_probability, initial_outlier_probability,
        sample_size,
    )
    best_supporters: Tuple[int, ...] = ()
    best_model: Optional[CameraModel] = None
    while attempts < min(required, maximum_attempts):
        sample = random.sample(consensus, sample_size)
        attempts += 1
        random_attempts += 1
        try:
            pnp_success, model = build_four_camera_model(
                sample, points, current_stereo_inliers, current_left_keypoints,
                intrinsic, previous_left_rotation, previous_left_translation,
                fixed_right_rotation, fixed_right_translation,
            )
        except (cv2.error, ValueError, ContractError):
            pnp_success, model = False, None
        if not pnp_success:
            failed += 1
            continue
        successful += 1
        if model is None or not model_is_valid(model):
            invalid += 1
            continue
        try:
            supporters = supporter_indices(
                consensus_points, observed, intrinsic, model, maximum_distance_px
            )
        except (ContractError, FloatingPointError, ValueError):
            invalid += 1
            continue
        valid += 1
        if best_model is None or len(supporters) > len(best_supporters):
            best_model = model
            best_supporters = supporters
            required = _bounded_requirement(
                len(consensus), len(best_supporters), success_probability,
                initial_outlier_probability, sample_size,
            )

    if best_model is None:
        status = INVALID_GEOMETRY if successful and invalid else NO_VALID_PNP_MODEL
        reason = (
            "no_geometry_valid_model_was_produced"
            if status == INVALID_GEOMETRY
            else "no_solvePnP_call_produced_a_valid_model"
        )
        return BoundedRansacResult(
            status, reason, attempts, random_attempts, 0, required,
            min(required, maximum_attempts), maximum_attempts, successful,
            failed, invalid, valid, (), None, (),
        )

    random_completed = attempts >= required
    outcomes = []
    needs_refinement = len(best_supporters) >= sample_size
    while random_completed and needs_refinement and attempts < maximum_attempts:
        refinement = tuple(consensus[index] for index in best_supporters)
        attempts += 1
        refinement_attempts += 1
        try:
            pnp_success, model = build_four_camera_model(
                refinement, points, current_stereo_inliers, current_left_keypoints,
                intrinsic, previous_left_rotation, previous_left_translation,
                fixed_right_rotation, fixed_right_translation,
            )
        except (cv2.error, ValueError, ContractError) as error:
            pnp_success, model = False, None
            model_reason = "solvePnP_exception:{}".format(type(error).__name__)
        else:
            model_reason = "valid_model" if pnp_success else "solvePnP_returned_false"
        geometry_valid = bool(model is not None and model_is_valid(model))
        output_count = None
        accepted = False
        if not pnp_success:
            failed += 1
            needs_refinement = False
        else:
            successful += 1
            if not geometry_valid:
                invalid += 1
                model_reason = "non_finite_or_non_rigid_model"
                needs_refinement = False
            else:
                supporters = supporter_indices(
                    consensus_points, observed, intrinsic, model, maximum_distance_px
                )
                valid += 1
                output_count = len(supporters)
                if len(supporters) > len(best_supporters):
                    best_supporters = supporters
                    best_model = model
                    required = _bounded_requirement(
                        len(consensus), len(best_supporters), success_probability,
                        initial_outlier_probability, sample_size,
                    )
                    accepted = True
                    model_reason = "strict_supporter_increase"
                else:
                    needs_refinement = False
                    model_reason = "no_strict_supporter_increase"
        outcomes.append(
            RefinementPnPOutcome(
                refinement_attempts - 1, attempts, len(refinement),
                bool(pnp_success), geometry_valid, output_count, accepted,
                model_reason,
            )
        )

    if not random_completed:
        status = BUDGET_EXHAUSTED_WITH_BEST_MODEL
        reason = "total_attempt_budget_reached_before_adaptive_requirement"
    elif needs_refinement and attempts >= maximum_attempts:
        status = BUDGET_EXHAUSTED_WITH_BEST_MODEL
        reason = "total_attempt_budget_reached_during_strict_refinement"
    else:
        status = ADAPTIVE_COMPLETED
        reason = "adaptive_requirement_met_and_refinement_terminated"
    return BoundedRansacResult(
        status, reason, attempts, random_attempts, refinement_attempts, required,
        min(required, maximum_attempts), maximum_attempts, successful, failed,
        invalid, valid, best_supporters, best_model, tuple(outcomes),
    )


def reset_random_generators(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
