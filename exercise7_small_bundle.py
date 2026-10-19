"""Reusable two-pose stereo Bundle measurement for E7-I03 and E7-I04.

Every graph contains exactly the previous and current loop-endpoint cameras,
one landmark per eligible four-view supporter, one local gauge prior on the
previous camera, and two stereo reprojection factors per landmark.  No Pose
Graph ``BetweenFactorPose3`` belongs inside this measurement Bundle.  The
latest Pose Graph relative pose supplies only the current camera's initial
value; the stereo observations determine the optimized measurement.

The returned pose is ``previous_pose.between(current_pose)``: in the validated
camera convention it maps current-camera coordinates into previous-camera
coordinates and is stored as the previous-to-current factor measurement.  An
ordered two-pose joint covariance/information pair is used to extract the raw
conditional covariance of the current/end pose given the previous/start pose.
That 6x6 matrix uses GTSAM ``Pose3`` tangent order
``[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`` with radians and metres.
It is validated and passed downstream without scaling, transport, repair, or
visualization modification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import gtsam
import numpy as np

import tools
from exercise5_full_run import validate_covariance_evidence
from exercise7_consensus_matching import PairRun, validate_candidate_pair
from PoseGraph import TANGENT_ORDER, camera_symbol


PREVIOUS_FRAME_ID = 356
CURRENT_FRAME_ID = 3359
EXPECTED_CONSENSUS_COUNT = 515
EXPECTED_SUPPORTER_COUNT = 461


class SmallBundleValidationError(RuntimeError):
    """Raised when the bounded E7-I03 small-Bundle contract is violated."""


@dataclass(frozen=True)
class LoopStereoObservation:
    """One final four-view supporter represented at both stereo endpoints.

    Each measurement is ``(u_left, u_right, v)`` in pixels, with ``v`` the
    mean of its rectified left/right image rows.  ``consensus_index`` keeps the
    measurement tied to the retained matcher evidence and landmark symbol.
    """

    supporter_ordinal: int
    consensus_index: int
    previous_measurement: Tuple[float, float, float]
    current_measurement: Tuple[float, float, float]


@dataclass
class SmallBundleResult:
    """Two-pose measurement, raw covariance evidence, and diagnostics.

    ``optimized_relative_pose`` and ``conditional_covariance`` are the only
    scientific measurement pair consumed by the cumulative Pose Graph.  The
    live graph/Values, joint covariance/information, eligibility bookkeeping,
    direction checks, and reprojection rows make that extraction auditable.
    """

    previous_frame_id: int
    current_frame_id: int
    graph: gtsam.NonlinearFactorGraph
    initial_values: gtsam.Values
    optimized_values: gtsam.Values
    candidate_observations: Tuple[LoopStereoObservation, ...]
    accepted_observations: Tuple[LoopStereoObservation, ...]
    rejection_reason_by_consensus_index: Mapping[int, str]
    rejection_reason_counts: Mapping[str, int]
    initial_landmarks: np.ndarray
    initial_error: float
    first_optimizer_error: float
    first_optimizer_anchor_identity_max_abs: float
    gauge_normalized_error: float
    second_optimizer_error: float
    second_optimizer_anchor_identity_max_abs: float
    final_error: float
    initial_relative_pose: gtsam.Pose3
    optimized_relative_pose: gtsam.Pose3
    joint_covariance: np.ndarray
    joint_information: np.ndarray
    conditional_covariance: np.ndarray
    covariance_diagnostics: Mapping[str, object]
    covariance_eigenvalues: np.ndarray
    standalone_factor_error: float
    standalone_residual_max_abs: float
    standalone_mahalanobis_crosscheck_abs: float
    local_anchor_identity_max_abs: float
    direction_composition_max_abs: float
    direction_point_mapping_max_abs: float
    inverse_point_mapping_rmse_m: float
    reprojection_rows: Tuple[Mapping[str, object], ...]
    final_depth_rebuild_count: int
    final_depth_rebuild_history: Tuple[Mapping[str, object], ...]

    @property
    def previous_camera_key(self) -> int:
        return int(camera_symbol(self.previous_frame_id))

    @property
    def current_camera_key(self) -> int:
        return int(camera_symbol(self.current_frame_id))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmallBundleValidationError(message)


def calibration_from_configuration(
    configuration: Mapping[str, object], calibration_path: Path
) -> gtsam.Cal3_S2Stereo:
    """Construct the frozen E5 stereo calibration and verify KITTI P0/P1."""

    values = configuration["calibration"]
    calibration = gtsam.Cal3_S2Stereo(
        values["fx_px"],
        values["fy_px"],
        values["skew_px"],
        values["cx_px"],
        values["cy_px"],
        values["baseline_m"],
    )
    rows = {}
    for line in Path(calibration_path).read_text(encoding="utf-8").splitlines():
        label, raw = line.split(":", 1)
        rows[label] = np.asarray(raw.split(), dtype=float).reshape(3, 4)
    intrinsic = rows["P0"][:, :3]
    right = np.linalg.solve(intrinsic, rows["P1"])
    observed = np.asarray(
        [
            calibration.fx(),
            calibration.fy(),
            calibration.skew(),
            calibration.px(),
            calibration.py(),
            calibration.baseline(),
        ],
        dtype=float,
    )
    expected = np.asarray(
        [
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
            -right[0, 3],
        ],
        dtype=float,
    )
    tolerance = configuration["scientific_configuration"][
        "validation_tolerances"
    ]["numerical_reconstruction_atol"]
    _require(
        float(np.max(np.abs(observed - expected))) <= tolerance,
        "frozen E5 calibration differs from KITTI P0/P1",
    )
    return calibration


def _canonical_stereo_measurement(
    left: np.ndarray, right: np.ndarray
) -> Tuple[float, float, float]:
    values = np.asarray(
        [left[0], right[0], 0.5 * (left[1] + right[1])],
        dtype=float,
    )
    return tuple(float(value) for value in values)


def observations_from_pair_run(
    pair_run: PairRun,
) -> Tuple[LoopStereoObservation, ...]:
    """Map retained final-supporter indices to two endpoint stereo points.

    This is the explicit bridge from matcher evidence to Bundle observations:
    non-supporting consensus matches never enter the graph, while each final
    supporter keeps its consensus index and all four observed image pixels.
    """

    evidence = pair_run.evidence
    pair = evidence.pair
    counts = evidence.counts
    visualization = pair_run.visualization
    validate_candidate_pair(pair)
    _require(
        len(visualization.consensus)
        == counts.four_view_consensus_count
        and len(visualization.supporter_indices)
        == counts.final_supporter_count
        and tuple(visualization.supporter_indices)
        == tuple(evidence.ransac.best_supporter_indices),
        "in-memory supporter mappings differ from the pair evidence",
    )
    actual_pixels = tools.extract_actual_consensus_pixels(
        visualization.consensus,
        visualization.previous_stereo_inliers,
        visualization.current_stereo_inliers,
        visualization.previous_keypoints_left,
        visualization.previous_keypoints_right,
        visualization.current_keypoints_left,
        visualization.current_keypoints_right,
    )
    _require(
        actual_pixels.shape
        == (4, 2, counts.four_view_consensus_count)
        and np.all(np.isfinite(actual_pixels)),
        "pair four-view supporter pixels are invalid",
    )

    observations = []
    seen = set()
    for supporter_ordinal, raw_index in enumerate(
        visualization.supporter_indices
    ):
        consensus_index = int(raw_index)
        _require(
            0 <= consensus_index < counts.four_view_consensus_count,
            "supporter consensus index is outside the pair",
        )
        _require(
            consensus_index not in seen,
            "supporter consensus indices are not unique",
        )
        seen.add(consensus_index)
        observations.append(
            LoopStereoObservation(
                supporter_ordinal=supporter_ordinal,
                consensus_index=consensus_index,
                previous_measurement=_canonical_stereo_measurement(
                    actual_pixels[0, :, consensus_index],
                    actual_pixels[1, :, consensus_index],
                ),
                current_measurement=_canonical_stereo_measurement(
                    actual_pixels[2, :, consensus_index],
                    actual_pixels[3, :, consensus_index],
                ),
            )
        )
    return tuple(observations)


def observations_from_e7_i03_pair_run(
    pair_run: PairRun,
) -> Tuple[LoopStereoObservation, ...]:
    """Preserve the exact 356->3359, 515/461 E7-I03 regression gate."""

    evidence = pair_run.evidence
    pair = evidence.pair
    counts = evidence.counts
    _require(
        pair.pair_index == 3
        and pair.previous_frame_id == PREVIOUS_FRAME_ID
        and pair.current_frame_id == CURRENT_FRAME_ID,
        "E7-I03 accepts only declared E7-I02 pair index 3",
    )
    _require(
        counts.four_view_consensus_count == EXPECTED_CONSENSUS_COUNT
        and counts.final_supporter_count == EXPECTED_SUPPORTER_COUNT,
        "E7-I02 canonical 515/461 pair counts were not reproduced",
    )
    return observations_from_pair_run(pair_run)


def _measurement_is_valid(measurement: Sequence[float]) -> Tuple[bool, str]:
    array = np.asarray(measurement, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return False, "nonfinite_measurement"
    if not float(array[0] - array[1]) > 0.0:
        return False, "nonpositive_disparity"
    return True, ""


def _projection_row(
    *,
    observation: LoopStereoObservation,
    frame_id: int,
    factor_index: int,
    landmark_key: int,
    measurement: Tuple[float, float, float],
    graph: gtsam.NonlinearFactorGraph,
    initial_values: gtsam.Values,
    optimized_values: gtsam.Values,
    calibration: gtsam.Cal3_S2Stereo,
) -> Mapping[str, object]:
    camera_key = int(camera_symbol(frame_id))

    def state(values: gtsam.Values) -> Tuple[np.ndarray, np.ndarray, float]:
        point = np.asarray(values.atPoint3(landmark_key), dtype=float).reshape(3)
        pose = values.atPose3(camera_key)
        camera_point = np.asarray(pose.transformTo(point), dtype=float).reshape(3)
        _require(
            np.all(np.isfinite(camera_point)) and camera_point[2] > 0.0,
            "projection point has nonpositive or nonfinite depth for "
            "consensus {} frame {}: {}".format(
                observation.consensus_index,
                frame_id,
                camera_point.tolist(),
            ),
        )
        projection = np.asarray(
            gtsam.StereoCamera(pose, calibration).project(point).vector(),
            dtype=float,
        ).reshape(3)
        residual = projection - np.asarray(measurement, dtype=float)
        error = float(graph.at(factor_index).error(values))
        _require(
            np.all(np.isfinite(projection))
            and np.all(np.isfinite(residual))
            and np.isfinite(error),
            "projection evidence is nonfinite",
        )
        return projection, residual, error

    initial_projection, initial_residual, initial_factor_error = state(
        initial_values
    )
    final_projection, final_residual, final_factor_error = state(
        optimized_values
    )
    return {
        "supporter_ordinal": observation.supporter_ordinal,
        "consensus_index": observation.consensus_index,
        "frame_id": frame_id,
        "factor_index": factor_index,
        "camera_key": camera_key,
        "landmark_key": landmark_key,
        "measurement": measurement,
        "initial_projection": tuple(float(x) for x in initial_projection),
        "initial_residual": tuple(float(x) for x in initial_residual),
        "initial_factor_error": initial_factor_error,
        "final_projection": tuple(float(x) for x in final_projection),
        "final_residual": tuple(float(x) for x in final_residual),
        "final_factor_error": final_factor_error,
    }


def solve_small_bundle(
    *,
    observations: Sequence[LoopStereoObservation],
    initial_relative_pose: gtsam.Pose3,
    calibration: gtsam.Cal3_S2Stereo,
    configuration: Mapping[str, object],
    previous_frame_id: int = PREVIOUS_FRAME_ID,
    current_frame_id: int = CURRENT_FRAME_ID,
    expected_supporter_count: int = EXPECTED_SUPPORTER_COUNT,
    _final_depth_rebuild_count: int = 0,
) -> SmallBundleResult:
    """Build, optimize, and condition one directed two-pose loop measurement.

    The caller supplies final-supporter stereo observations, the latest Pose
    Graph previous-to-current pose as initialization only, frozen calibration
    and E5 scientific configuration, and the endpoint FrameIds.  The previous
    camera is the identity gauge anchor.  Eligible landmarks must have finite
    positive disparity/backprojection, respect the initialization-depth gate,
    and remain in front of both cameras; invalid landmarks are retained as
    explicit rejection evidence.

    Two LM passes surround exact gauge normalization.  The output pose is
    ``previous.between(current)`` (``T_Cprevious_from_Ccurrent``).  Ordered
    joint covariance and information for ``[previous, current]`` yield the
    conditional end-given-start covariance.  The raw 6x6 result is in
    ``[rotation xyz (rad), translation xyz (m)]`` order and is neither scaled
    for visualization nor otherwise scientifically modified.  The function
    returns local evidence only and does not mutate the cumulative Pose Graph.
    """

    # Phase 1: validate endpoint/supporter identity and load the frozen stereo
    # noise, anchor prior, eligibility, and numerical-tolerance contracts.
    observations = tuple(observations)
    _require(
        type(previous_frame_id) is int
        and type(current_frame_id) is int
        and 0 <= previous_frame_id < current_frame_id,
        "small-Bundle FrameIds must be ordered exact nonnegative integers",
    )
    _require(
        type(expected_supporter_count) is int and expected_supporter_count >= 2,
        "expected supporter count must be an integer at least two",
    )
    _require(
        len(observations) == expected_supporter_count,
        "small Bundle candidate supporter count differs",
    )
    _require(
        len({item.consensus_index for item in observations})
        == len(observations),
        "candidate consensus indices are not unique",
    )
    scientific = configuration["scientific_configuration"]
    tolerances = scientific["validation_tolerances"]
    eligibility = scientific["landmark_eligibility"]
    maximum_depth = float(eligibility["maximum_initialization_depth_m"])
    projection_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            scientific["measurement_uncertainty"]["stereo_sigma_px"],
            dtype=float,
        )
    )
    prior = scientific["gauge_fixing_prior"]
    prior_sigmas = np.asarray(
        prior["rotation_sigma_rad"] + prior["translation_sigma_m"],
        dtype=float,
    )
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)

    previous_key = int(camera_symbol(previous_frame_id))
    current_key = int(camera_symbol(current_frame_id))
    _require(previous_key != current_key, "camera keys collide")
    identity = gtsam.Pose3()
    initial_matrix = np.asarray(initial_relative_pose.matrix(), dtype=float)
    _require(
        initial_matrix.shape == (4, 4)
        and np.all(np.isfinite(initial_matrix)),
        "Pose Graph relative-pose initialization is invalid",
    )

    # Phase 2: initialize landmarks by backprojecting previous-frame stereo.
    # Reject invalid disparity, nonfinite/excessive depth, or points not in
    # front of both initialized endpoint cameras; preserve every reason.
    previous_camera = gtsam.StereoCamera(identity, calibration)
    accepted = []
    accepted_points = []
    rejected: Dict[int, str] = {}
    rejection_counts: Counter = Counter()
    for observation in observations:
        previous_valid, previous_reason = _measurement_is_valid(
            observation.previous_measurement
        )
        current_valid, current_reason = _measurement_is_valid(
            observation.current_measurement
        )
        if not previous_valid or not current_valid:
            reason = previous_reason if not previous_valid else current_reason
        else:
            point = np.asarray(
                previous_camera.backproject(
                    gtsam.StereoPoint2(*observation.previous_measurement)
                ),
                dtype=float,
            ).reshape(3)
            if not np.all(np.isfinite(point)):
                reason = "no_finite_backprojection"
            elif not float(point[2]) > 0.0:
                reason = "no_all_camera_positive_depth_initialization"
            elif point[2] >= maximum_depth:
                reason = "excessive_initialization_depth"
            else:
                current_point = np.asarray(
                    initial_relative_pose.transformTo(point), dtype=float
                ).reshape(3)
                if (
                    not np.all(np.isfinite(current_point))
                    or not float(current_point[2]) > 0.0
                ):
                    reason = "no_all_camera_positive_depth_initialization"
                else:
                    reason = ""
        if reason:
            rejected[observation.consensus_index] = reason
            rejection_counts[reason] += 1
            continue
        accepted.append(observation)
        accepted_points.append(point)

    accepted_observations = tuple(accepted)
    _require(
        len(accepted_observations) >= 2,
        "eligible loop supporters do not dimensionally constrain two poses",
    )
    _require(
        len(accepted_observations) + len(rejected) == len(observations),
        "landmark eligibility accounting is incomplete",
    )
    landmark_keys = tuple(
        int(gtsam.symbol("q", item.consensus_index))
        for item in accepted_observations
    )
    all_keys = (previous_key, current_key) + landmark_keys
    _require(len(set(all_keys)) == len(all_keys), "GTSAM variable keys collide")

    # Phase 3: create exactly two camera variables, the previous-camera gauge
    # prior, one q landmark per eligible supporter, and two stereo factors per
    # landmark. No Pose Graph relative-pose factor is added to this graph.
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    graph.add(
        gtsam.PriorFactorPose3(
            previous_key, identity, anchor_noise
        )
    )
    initial_values.insert(previous_key, identity)
    initial_values.insert(current_key, initial_relative_pose)

    factor_records = []
    for observation, point, landmark_key in zip(
        accepted_observations, accepted_points, landmark_keys
    ):
        initial_values.insert(landmark_key, point)
        for frame_id, measurement in (
            (previous_frame_id, observation.previous_measurement),
            (current_frame_id, observation.current_measurement),
        ):
            graph.add(
                gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(*measurement),
                    projection_noise,
                    int(camera_symbol(frame_id)),
                    landmark_key,
                    calibration,
                )
            )
            factor_records.append(
                (
                    observation,
                    frame_id,
                    int(graph.size() - 1),
                    landmark_key,
                    measurement,
                )
            )

    expected_factor_count = 1 + 2 * len(accepted_observations)
    _require(
        graph.size() == expected_factor_count,
        "small-Bundle factor count differs from one prior plus two per point",
    )
    _require(
        initial_values.size() == 2 + len(accepted_observations),
        "small-Bundle variable count differs",
    )
    residual_dimension = int(
        sum(graph.at(index).dim() for index in range(graph.size()))
    )
    _require(
        residual_dimension >= int(initial_values.dim()),
        "small Bundle is dimensionally underconstrained",
    )
    for index in range(graph.size()):
        factor = graph.at(index)
        if index == 0:
            _require(
                isinstance(factor, gtsam.PriorFactorPose3),
                "first small-Bundle factor is not the local prior",
            )
        else:
            _require(
                isinstance(factor, gtsam.GenericStereoFactor3D),
                "small Bundle contains a non-stereo measurement factor",
            )

    # Phase 4: optimize once, express every value exactly in the optimized
    # previous-camera gauge, optimize again, and normalize that gauge again.
    # The prior still participates scientifically; normalization fixes only
    # the coordinate representation of the same local two-camera solution.
    initial_error = float(graph.error(initial_values))
    _require(np.isfinite(initial_error), "small-Bundle initial error is nonfinite")
    first_optimized_values = gtsam.LevenbergMarquardtOptimizer(
        graph, initial_values
    ).optimize()
    first_optimizer_error = float(graph.error(first_optimized_values))
    _require(
        np.isfinite(first_optimizer_error),
        "small-Bundle first optimizer error is nonfinite",
    )
    first_anchor = first_optimized_values.atPose3(previous_key)
    first_anchor_error = float(
        np.max(np.abs(np.asarray(first_anchor.matrix()) - np.eye(4)))
    )
    gauge_normalized_values = gtsam.Values()
    gauge_normalized_values.insert(previous_key, identity)
    gauge_normalized_values.insert(
        current_key,
        first_anchor.between(
            first_optimized_values.atPose3(current_key)
        ),
    )
    for landmark_key in landmark_keys:
        gauge_normalized_values.insert(
            landmark_key,
            np.asarray(
                first_anchor.transformTo(
                    first_optimized_values.atPoint3(landmark_key)
                ),
                dtype=float,
            ).reshape(3),
        )
    gauge_normalized_error = float(
        graph.error(gauge_normalized_values)
    )
    _require(
        np.isfinite(gauge_normalized_error)
        and gauge_normalized_error <= first_optimizer_error,
        "exact local-gauge normalization did not preserve or reduce error",
    )
    second_optimized_values = gtsam.LevenbergMarquardtOptimizer(
        graph, gauge_normalized_values
    ).optimize()
    second_optimizer_error = float(graph.error(second_optimized_values))
    _require(
        np.isfinite(second_optimizer_error),
        "small-Bundle second optimizer error is nonfinite",
    )
    second_anchor = second_optimized_values.atPose3(previous_key)
    second_anchor_error = float(
        np.max(np.abs(np.asarray(second_anchor.matrix()) - np.eye(4)))
    )
    optimized_values = gtsam.Values()
    optimized_values.insert(previous_key, identity)
    optimized_values.insert(
        current_key,
        second_anchor.between(
            second_optimized_values.atPose3(current_key)
        ),
    )
    for landmark_key in landmark_keys:
        optimized_values.insert(
            landmark_key,
            np.asarray(
                second_anchor.transformTo(
                    second_optimized_values.atPoint3(landmark_key)
                ),
                dtype=float,
            ).reshape(3),
        )
    final_error = float(graph.error(optimized_values))
    _require(np.isfinite(final_error), "small-Bundle final error is nonfinite")
    allowed_increase = max(
        float(tolerances["graph_error_nonincrease_absolute"]),
        float(tolerances["graph_error_nonincrease_relative"])
        * abs(initial_error),
    )
    _require(
        final_error <= initial_error + allowed_increase,
        "small-Bundle optimization increased its own graph error",
    )

    # Phase 5: extract previous.between(current), then verify by composition
    # and point mapping that it maps current-camera coordinates to the
    # previous-camera frame (T_Cprevious_from_Ccurrent).
    optimized_previous = optimized_values.atPose3(previous_key)
    optimized_current = optimized_values.atPose3(current_key)
    anchor_error = float(
        np.max(np.abs(np.asarray(optimized_previous.matrix()) - np.eye(4)))
    )
    _require(
        anchor_error <= tolerances["identity_pose_max_abs"],
        "local Bundle c{} did not remain identity: maximum absolute "
        "matrix error {}".format(previous_frame_id, anchor_error),
    )
    optimized_relative = optimized_previous.between(optimized_current)
    composition_error = float(
        np.max(
            np.abs(
                np.asarray(
                    optimized_previous.compose(optimized_relative).matrix()
                )
                - np.asarray(optimized_current.matrix())
            )
        )
    )
    _require(
        composition_error <= tolerances["numerical_reconstruction_atol"],
        "optimized relative-pose composition direction is inconsistent",
    )

    point_mapping_errors = []
    inverse_mapping_errors = []
    for landmark_key in landmark_keys:
        point = np.asarray(
            optimized_values.atPoint3(landmark_key), dtype=float
        ).reshape(3)
        _require(np.all(np.isfinite(point)), "optimized landmark is nonfinite")
        previous_point = np.asarray(
            optimized_previous.transformTo(point), dtype=float
        ).reshape(3)
        current_point = np.asarray(
            optimized_current.transformTo(point), dtype=float
        ).reshape(3)
        point_mapping_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(
                        optimized_relative.transformFrom(current_point),
                        dtype=float,
                    ).reshape(3)
                    - previous_point
                )
            )
        )
        inverse_mapping_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(
                        optimized_relative.inverse().transformFrom(
                            current_point
                        ),
                        dtype=float,
                    ).reshape(3)
                    - previous_point
                )
            )
        )
    mapping_max = max(point_mapping_errors)
    _require(
        mapping_max <= tolerances["numerical_reconstruction_atol"],
        "optimized relative pose does not map C{} points into C{}".format(
            current_frame_id, previous_frame_id
        ),
    )

    # Phase 6: a landmark that crosses behind either optimized camera is not
    # silently retained. Rebuild the same two-pose problem without those
    # landmarks and preserve the rejection/rebuild history.
    invalid_final_depths = []
    for observation, landmark_key in zip(
        accepted_observations, landmark_keys
    ):
        point = optimized_values.atPoint3(landmark_key)
        for frame_id, pose in (
            (previous_frame_id, optimized_previous),
            (current_frame_id, optimized_current),
        ):
            depth = float(
                np.asarray(pose.transformTo(point), dtype=float).reshape(3)[2]
            )
            if not np.isfinite(depth) or depth <= 0.0:
                invalid_final_depths.append(
                    (observation.consensus_index, frame_id, depth)
                )
    if invalid_final_depths:
        invalid_consensus_indices = {
            consensus_index
            for consensus_index, _, _ in invalid_final_depths
        }
        filtered_observations = tuple(
            observation
            for observation in observations
            if observation.consensus_index
            not in invalid_consensus_indices
        )
        rebuilt = solve_small_bundle(
            observations=filtered_observations,
            initial_relative_pose=initial_relative_pose,
            calibration=calibration,
            configuration=configuration,
            previous_frame_id=previous_frame_id,
            current_frame_id=current_frame_id,
            expected_supporter_count=len(filtered_observations),
            _final_depth_rebuild_count=_final_depth_rebuild_count + 1,
        )
        rebuilt.candidate_observations = observations
        combined_reasons = dict(
            rebuilt.rejection_reason_by_consensus_index
        )
        for consensus_index in sorted(invalid_consensus_indices):
            combined_reasons[consensus_index] = "nonpositive_final_depth"
        rebuilt.rejection_reason_by_consensus_index = dict(
            sorted(combined_reasons.items())
        )
        combined_counts = Counter(rebuilt.rejection_reason_counts)
        combined_counts["nonpositive_final_depth"] += len(
            invalid_consensus_indices
        )
        rebuilt.rejection_reason_counts = dict(
            sorted(combined_counts.items())
        )
        rebuilt.final_depth_rebuild_count = max(
            rebuilt.final_depth_rebuild_count,
            _final_depth_rebuild_count + 1,
        )
        rebuilt.final_depth_rebuild_history = (
            {
                "pass_index": _final_depth_rebuild_count,
                "accepted_landmark_count_before_rebuild": len(
                    accepted_observations
                ),
                "invalid_factor_depth_count": len(
                    invalid_final_depths
                ),
                "rejected_consensus_indices": sorted(
                    invalid_consensus_indices
                ),
                "initial_error": initial_error,
                "first_optimizer_error": first_optimizer_error,
                "first_optimizer_anchor_identity_max_abs": (
                    first_anchor_error
                ),
                "gauge_normalized_error": gauge_normalized_error,
                "second_optimizer_error": second_optimizer_error,
                "second_optimizer_anchor_identity_max_abs": (
                    second_anchor_error
                ),
                "final_error": final_error,
            },
        ) + rebuilt.final_depth_rebuild_history
        return rebuilt

    # Phase 7: retain per-factor projection evidence for the final accepted
    # landmark set before extracting the pose-only covariance evidence.
    reprojection_rows = tuple(
        _projection_row(
            observation=observation,
            frame_id=frame_id,
            factor_index=factor_index,
            landmark_key=landmark_key,
            measurement=measurement,
            graph=graph,
            initial_values=initial_values,
            optimized_values=optimized_values,
            calibration=calibration,
        )
        for (
            observation,
            frame_id,
            factor_index,
            landmark_key,
            measurement,
        ) in factor_records
    )

    # Phase 8: request the ordered [previous, current] 12x12 joint covariance
    # and information, then condition the end/current pose on the start/previous
    # pose using the validated E5 covariance contract.
    marginals = gtsam.Marginals(graph, optimized_values)
    ordered_keys = gtsam.KeyVector([previous_key, current_key])
    joint_covariance = np.asarray(
        marginals.jointMarginalCovariance(ordered_keys).fullMatrix(),
        dtype=float,
    )
    joint_information = np.asarray(
        marginals.jointMarginalInformation(ordered_keys).fullMatrix(),
        dtype=float,
    )
    conditional, covariance_diagnostics = validate_covariance_evidence(
        joint_covariance,
        joint_information,
        None,
        symmetry_atol=tolerances["covariance_symmetry_atol"],
        minimum_eigenvalue=tolerances["covariance_minimum_eigenvalue"],
        reconstruction_atol=tolerances["numerical_reconstruction_atol"],
    )
    _require(
        conditional.shape == (6, 6)
        and np.all(np.isfinite(conditional)),
        "conditional end-given-start covariance is not finite 6x6",
    )
    symmetry_error = float(
        np.max(np.abs(conditional - conditional.T))
    )
    _require(
        symmetry_error <= tolerances["covariance_symmetry_atol"],
        "conditional covariance is not symmetric within tolerance",
    )
    eigenvalues = np.linalg.eigvalsh(conditional)
    _require(
        np.all(np.isfinite(eigenvalues)),
        "conditional covariance eigenvalues are nonfinite",
    )
    try:
        np.linalg.cholesky(conditional)
    except np.linalg.LinAlgError as error:
        raise SmallBundleValidationError(
            "conditional covariance is not positive definite"
        ) from error
    # The conditional is raw scientific evidence: direct Gaussian covariance,
    # rotation-then-translation tangent order, radians/metres, no scaling,
    # regularization, pseudoinverse, transport, or plotting multiplier.
    noise = gtsam.noiseModel.Gaussian.Covariance(conditional)

    # Phase 9: independently instantiate the exact downstream factor contract
    # and cross-check its residual/error against the raw conditional matrix.
    standalone = gtsam.BetweenFactorPose3(
        previous_key, current_key, optimized_relative, noise
    )
    residual = np.asarray(
        standalone.unwhitenedError(optimized_values), dtype=float
    ).reshape(6)
    standalone_error = float(standalone.error(optimized_values))
    direct_error = float(
        0.5 * residual @ np.linalg.solve(conditional, residual)
    )
    standalone_crosscheck = abs(standalone_error - direct_error)
    _require(
        np.all(np.isfinite(residual))
        and np.isfinite(standalone_error)
        and np.isfinite(standalone_crosscheck),
        "standalone loop-factor evidence is nonfinite",
    )
    _require(
        max(float(np.max(np.abs(residual))), standalone_error)
        <= tolerances["numerical_reconstruction_atol"],
        "standalone loop factor is inconsistent with the local Bundle poses",
    )

    covariance_diagnostics = dict(covariance_diagnostics)
    covariance_diagnostics.update(
        {
            "joint_block_order_source": (
                "explicit ordered KeyVector [c{}, c{}]".format(
                    previous_frame_id, current_frame_id
                )
            ),
            "conditional_symmetry_max_abs": symmetry_error,
            "conditional_cholesky_succeeded": True,
            "gaussian_covariance_constructed_directly": True,
            "covariance_repair_applied": False,
            "tangent_order": list(TANGENT_ORDER),
        }
    )
    return SmallBundleResult(
        previous_frame_id=previous_frame_id,
        current_frame_id=current_frame_id,
        graph=graph,
        initial_values=initial_values,
        optimized_values=optimized_values,
        candidate_observations=observations,
        accepted_observations=accepted_observations,
        rejection_reason_by_consensus_index=dict(sorted(rejected.items())),
        rejection_reason_counts=dict(sorted(rejection_counts.items())),
        initial_landmarks=np.asarray(accepted_points, dtype=float),
        initial_error=initial_error,
        first_optimizer_error=first_optimizer_error,
        first_optimizer_anchor_identity_max_abs=first_anchor_error,
        gauge_normalized_error=gauge_normalized_error,
        second_optimizer_error=second_optimizer_error,
        second_optimizer_anchor_identity_max_abs=second_anchor_error,
        final_error=final_error,
        initial_relative_pose=initial_relative_pose,
        optimized_relative_pose=optimized_relative,
        joint_covariance=np.asarray(joint_covariance, dtype=float),
        joint_information=np.asarray(joint_information, dtype=float),
        conditional_covariance=np.asarray(conditional, dtype=float),
        covariance_diagnostics=covariance_diagnostics,
        covariance_eigenvalues=np.asarray(eigenvalues, dtype=float),
        standalone_factor_error=standalone_error,
        standalone_residual_max_abs=float(np.max(np.abs(residual))),
        standalone_mahalanobis_crosscheck_abs=standalone_crosscheck,
        local_anchor_identity_max_abs=anchor_error,
        direction_composition_max_abs=composition_error,
        direction_point_mapping_max_abs=mapping_max,
        inverse_point_mapping_rmse_m=float(
            np.sqrt(np.mean(np.square(inverse_mapping_errors)))
        ),
        reprojection_rows=reprojection_rows,
        final_depth_rebuild_count=_final_depth_rebuild_count,
        final_depth_rebuild_history=(),
    )


__all__ = [
    "CURRENT_FRAME_ID",
    "EXPECTED_CONSENSUS_COUNT",
    "EXPECTED_SUPPORTER_COUNT",
    "LoopStereoObservation",
    "PREVIOUS_FRAME_ID",
    "SmallBundleResult",
    "SmallBundleValidationError",
    "calibration_from_configuration",
    "observations_from_e7_i03_pair_run",
    "observations_from_pair_run",
    "solve_small_bundle",
]
