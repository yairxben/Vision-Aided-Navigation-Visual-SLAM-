"""Two-pose stereo Bundle measurement for an accepted visual match."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import gtsam
import numpy as np

from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiEstimationContext
from ..frontend.temporal import actual_consensus_pixels
from ..pose_graph.state import camera_symbol
from .matching import LoopMatchResult


@dataclass(frozen=True)
class LoopStereoObservation:
    supporter_ordinal: int
    consensus_index: int
    previous_measurement: Tuple[float, float, float]
    current_measurement: Tuple[float, float, float]


@dataclass
class SmallBundleResult:
    previous_frame_id: int
    current_frame_id: int
    candidate_observations: Tuple[LoopStereoObservation, ...]
    accepted_observations: Tuple[LoopStereoObservation, ...]
    rejection_reason_by_consensus_index: Mapping[int, str]
    graph: gtsam.NonlinearFactorGraph
    initial_values: gtsam.Values
    optimized_values: gtsam.Values
    initial_error: float
    first_optimizer_error: float
    gauge_normalized_error: float
    second_optimizer_error: float
    final_error: float
    optimized_relative_pose: gtsam.Pose3
    joint_covariance: np.ndarray
    joint_information: np.ndarray
    conditional_covariance: np.ndarray
    conditional_eigenvalues: np.ndarray
    measurement_rows: Tuple[Mapping[str, object], ...]
    final_depth_rebuild_count: int


def _stereo_measurement(left: np.ndarray, right: np.ndarray) -> Tuple[float, float, float]:
    return float(left[0]), float(right[0]), float(0.5 * (left[1] + right[1]))


def observations_from_match(result: LoopMatchResult) -> Tuple[LoopStereoObservation, ...]:
    pixels = actual_consensus_pixels(
        result.consensus,
        result.previous_stereo.inliers,
        result.current_stereo.inliers,
        result.previous_stereo.left.keypoints,
        result.previous_stereo.right.keypoints,
        result.current_stereo.left.keypoints,
        result.current_stereo.right.keypoints,
    )
    observations = []
    for ordinal, raw_index in enumerate(result.ransac.best_supporter_indices):
        index = int(raw_index)
        observations.append(
            LoopStereoObservation(
                ordinal,
                index,
                _stereo_measurement(pixels[0, :, index], pixels[1, :, index]),
                _stereo_measurement(pixels[2, :, index], pixels[3, :, index]),
            )
        )
    return tuple(observations)


def _measurement_valid(measurement: Sequence[float]) -> Tuple[bool, str]:
    value = np.asarray(measurement, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        return False, "nonfinite_measurement"
    if value[0] - value[1] <= 0.0:
        return False, "nonpositive_disparity"
    return True, ""


def _normalize_gauge(
    values: gtsam.Values,
    previous_key: int,
    current_key: int,
    landmark_keys: Sequence[int],
) -> gtsam.Values:
    identity = gtsam.Pose3()
    anchor = values.atPose3(previous_key)
    normalized = gtsam.Values()
    normalized.insert(previous_key, identity)
    normalized.insert(current_key, anchor.between(values.atPose3(current_key)))
    for key in landmark_keys:
        normalized.insert(
            key,
            np.asarray(anchor.transformTo(values.atPoint3(key)), dtype=float).reshape(3),
        )
    return normalized


def _conditional_covariance(joint: np.ndarray) -> np.ndarray:
    solved = np.linalg.solve(joint[:6, :6], joint[:6, 6:])
    conditional = joint[6:, 6:] - joint[6:, :6] @ solved
    return 0.5 * (conditional + conditional.T)


def _gauge_error_is_consistent(
    optimizer_error: float,
    normalized_error: float,
    absolute_tolerance: float,
) -> bool:
    """Accept only finite, numerically gauge-equivalent graph errors."""

    return bool(
        np.isfinite(optimizer_error)
        and np.isfinite(normalized_error)
        and normalized_error <= optimizer_error + absolute_tolerance
    )


def _projection_rows(
    records,
    graph: gtsam.NonlinearFactorGraph,
    initial: gtsam.Values,
    final: gtsam.Values,
    calibration: gtsam.Cal3_S2Stereo,
):
    rows = []
    for observation, frame_id, factor_index, landmark_key, measurement in records:
        camera_key = camera_symbol(frame_id)
        states = []
        for values in (initial, final):
            camera = gtsam.StereoCamera(values.atPose3(camera_key), calibration)
            projection = np.asarray(
                camera.project(values.atPoint3(landmark_key)).vector(), dtype=float
            )
            residual = projection - np.asarray(measurement, dtype=float)
            states.append((projection, residual, float(graph.at(factor_index).error(values))))
        rows.append(
            {
                "supporter_ordinal": observation.supporter_ordinal,
                "consensus_index": observation.consensus_index,
                "frame_id": frame_id,
                "factor_index": factor_index,
                "landmark_key": landmark_key,
                "measurement": tuple(measurement),
                "initial_projection": tuple(float(v) for v in states[0][0]),
                "initial_residual": tuple(float(v) for v in states[0][1]),
                "initial_factor_error": states[0][2],
                "final_projection": tuple(float(v) for v in states[1][0]),
                "final_residual": tuple(float(v) for v in states[1][1]),
                "final_factor_error": states[1][2],
            }
        )
    return tuple(rows)


def solve_small_bundle(
    *,
    observations: Sequence[LoopStereoObservation],
    initial_relative_pose: gtsam.Pose3,
    calibration: gtsam.Cal3_S2Stereo,
    bundle_config: Mapping[str, object],
    previous_frame_id: int,
    current_frame_id: int,
    _rebuild_count: int = 0,
) -> SmallBundleResult:
    """Optimize the validated prior+two-stereo-factors-per-landmark graph."""

    observations = tuple(observations)
    if previous_frame_id < 0 or previous_frame_id >= current_frame_id:
        raise ContractError("small-Bundle endpoint order is invalid")
    if len(observations) < 2 or len({item.consensus_index for item in observations}) != len(observations):
        raise ContractError("small Bundle needs unique supporter observations")
    previous_key = camera_symbol(previous_frame_id)
    current_key = camera_symbol(current_frame_id)
    maximum_depth = float(bundle_config["maximum_initialization_depth_m"])
    previous_camera = gtsam.StereoCamera(gtsam.Pose3(), calibration)
    accepted = []
    accepted_points = []
    rejected = {}
    for observation in observations:
        previous_valid, previous_reason = _measurement_valid(observation.previous_measurement)
        current_valid, current_reason = _measurement_valid(observation.current_measurement)
        reason = previous_reason if not previous_valid else current_reason if not current_valid else ""
        if not reason:
            point = np.asarray(
                previous_camera.backproject(gtsam.StereoPoint2(*observation.previous_measurement)),
                dtype=float,
            ).reshape(3)
            # Depth of the current observation in its own camera; both
            # observations must be within the limit, as in the window Bundle.
            current_own_depth = float(np.asarray(
                previous_camera.backproject(gtsam.StereoPoint2(*observation.current_measurement)),
                dtype=float,
            ).reshape(3)[2])
            if not np.all(np.isfinite(point)):
                reason = "no_finite_backprojection"
            elif point[2] <= 0.0:
                reason = "no_all_camera_positive_depth_initialization"
            elif point[2] >= maximum_depth or (
                np.isfinite(current_own_depth) and current_own_depth >= maximum_depth
            ):
                reason = "excessive_initialization_depth"
            else:
                current_point = np.asarray(initial_relative_pose.transformTo(point), dtype=float)
                if not np.all(np.isfinite(current_point)) or current_point[2] <= 0.0:
                    reason = "no_all_camera_positive_depth_initialization"
        if reason:
            rejected[observation.consensus_index] = reason
        else:
            accepted.append(observation)
            accepted_points.append(point)
    accepted_observations = tuple(accepted)
    if len(accepted_observations) < 2:
        raise ContractError("eligible loop supporters do not constrain two poses")

    landmark_keys = tuple(
        int(gtsam.symbol("q", item.consensus_index)) for item in accepted_observations
    )
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    projection_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(bundle_config["stereo_sigma_px"], dtype=float)
    )
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(bundle_config["prior_sigmas"], dtype=float)
    )
    graph.add(gtsam.PriorFactorPose3(previous_key, gtsam.Pose3(), anchor_noise))
    initial.insert(previous_key, gtsam.Pose3())
    initial.insert(current_key, initial_relative_pose)
    records = []
    for observation, point, landmark_key in zip(
        accepted_observations, accepted_points, landmark_keys
    ):
        initial.insert(landmark_key, point)
        for frame_id, measurement in (
            (previous_frame_id, observation.previous_measurement),
            (current_frame_id, observation.current_measurement),
        ):
            graph.add(
                gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(*measurement), projection_noise,
                    camera_symbol(frame_id), landmark_key, calibration,
                )
            )
            records.append(
                (observation, frame_id, int(graph.size() - 1), landmark_key, measurement)
            )
    if graph.size() != 1 + 2 * len(accepted_observations):
        raise ContractError("small-Bundle factor population differs")
    initial_error = float(graph.error(initial))
    first = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    first_error = float(graph.error(first))
    normalized = _normalize_gauge(first, previous_key, current_key, landmark_keys)
    normalized_error = float(graph.error(normalized))
    if not _gauge_error_is_consistent(
        first_error,
        normalized_error,
        float(bundle_config["graph_error_nonincrease_absolute"]),
    ):
        raise ContractError("small-Bundle gauge normalization increased error")
    second = gtsam.LevenbergMarquardtOptimizer(graph, normalized).optimize()
    second_error = float(graph.error(second))
    final = _normalize_gauge(second, previous_key, current_key, landmark_keys)
    final_error = float(graph.error(final))
    allowed = max(
        float(bundle_config["graph_error_nonincrease_absolute"]),
        float(bundle_config["graph_error_nonincrease_relative"]) * abs(initial_error),
    )
    if final_error > initial_error + allowed:
        raise ContractError("small-Bundle optimization increased graph error")

    invalid_final = set()
    for observation, key in zip(accepted_observations, landmark_keys):
        point = final.atPoint3(key)
        for camera_key in (previous_key, current_key):
            depth = float(np.asarray(final.atPose3(camera_key).transformTo(point))[2])
            if not np.isfinite(depth) or depth <= 0.0:
                invalid_final.add(observation.consensus_index)
    if invalid_final:
        filtered = tuple(
            item for item in observations if item.consensus_index not in invalid_final
        )
        rebuilt = solve_small_bundle(
            observations=filtered,
            initial_relative_pose=initial_relative_pose,
            calibration=calibration,
            bundle_config=bundle_config,
            previous_frame_id=previous_frame_id,
            current_frame_id=current_frame_id,
            _rebuild_count=_rebuild_count + 1,
        )
        combined = dict(rebuilt.rejection_reason_by_consensus_index)
        combined.update({index: "nonpositive_final_depth" for index in invalid_final})
        rebuilt.candidate_observations = observations
        rebuilt.rejection_reason_by_consensus_index = dict(sorted(combined.items()))
        return rebuilt

    previous_pose = final.atPose3(previous_key)
    current_pose = final.atPose3(current_key)
    optimized_relative = previous_pose.between(current_pose)
    tolerance = float(bundle_config["rotation_atol"])
    if np.max(np.abs(np.asarray(previous_pose.matrix()) - np.eye(4))) > tolerance:
        raise ContractError("small-Bundle local anchor is not identity")
    if np.max(
        np.abs(
            np.asarray(previous_pose.compose(optimized_relative).matrix())
            - np.asarray(current_pose.matrix())
        )
    ) > tolerance:
        raise ContractError("small-Bundle edge direction is inconsistent")

    marginals = gtsam.Marginals(graph, final)
    ordered_keys = gtsam.KeyVector([previous_key, current_key])
    joint = np.asarray(marginals.jointMarginalCovariance(ordered_keys).fullMatrix(), dtype=float)
    information = np.asarray(marginals.jointMarginalInformation(ordered_keys).fullMatrix(), dtype=float)
    conditional = _conditional_covariance(joint)
    if (
        conditional.shape != (6, 6)
        or not np.all(np.isfinite(conditional))
        or not np.allclose(
            conditional, conditional.T,
            atol=float(bundle_config["covariance_symmetry_atol"]), rtol=0.0,
        )
    ):
        raise ContractError("small-Bundle conditional covariance is invalid")
    try:
        np.linalg.cholesky(conditional)
    except np.linalg.LinAlgError as error:
        raise ContractError("small-Bundle conditional covariance is not positive definite") from error
    noise = gtsam.noiseModel.Gaussian.Covariance(conditional)
    standalone = gtsam.BetweenFactorPose3(
        previous_key, current_key, optimized_relative, noise
    )
    if float(standalone.error(final)) > tolerance:
        raise ContractError("small-Bundle loop factor does not reproduce its poses")
    return SmallBundleResult(
        previous_frame_id,
        current_frame_id,
        observations,
        accepted_observations,
        dict(sorted(rejected.items())),
        graph,
        initial,
        final,
        initial_error,
        first_error,
        normalized_error,
        second_error,
        final_error,
        optimized_relative,
        joint,
        information,
        conditional,
        np.linalg.eigvalsh(conditional),
        _projection_rows(records, graph, initial, final, calibration),
        _rebuild_count,
    )


def measure_loop(
    match: LoopMatchResult,
    initial_relative_pose: gtsam.Pose3,
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
) -> SmallBundleResult:
    calibration = dataset.calibration
    gtsam_calibration = gtsam.Cal3_S2Stereo(
        calibration.fx_px, calibration.fy_px, calibration.skew_px,
        calibration.cx_px, calibration.cy_px, calibration.baseline_m,
    )
    return solve_small_bundle(
        observations=observations_from_match(match),
        initial_relative_pose=initial_relative_pose,
        calibration=gtsam_calibration,
        bundle_config=config.section("bundle"),
        previous_frame_id=match.previous_frame_id,
        current_frame_id=match.current_frame_id,
    )


__all__ = [
    "LoopStereoObservation", "SmallBundleResult", "measure_loop",
    "observations_from_match", "solve_small_bundle",
]
