"""Deterministic single-window Bundle Adjustment science for Exercise 5.

The implementation was first validated by the bounded E5-I02 probe and is
reused unchanged by the canonical streamed runner.  It deliberately contains
no production noise defaults, keyframe policy, serialization, covariance
extraction, or all-window orchestration: callers supply every scientific
choice explicitly.

Canonical poses are world-to-camera transforms ``E_i = T_{C_i<-W}``.  The
window coordinate system is its first camera, and camera Values therefore use
``T_{C_start<-C_i}``, as produced by :func:`camera_pose_in_window`.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from exercise5_geometry import camera_pose_in_window


CAMERA_SYMBOL = "c"
LANDMARK_SYMBOL = "q"

INSUFFICIENT_VALID_OBSERVATIONS = "insufficient_valid_observations"
NO_FINITE_BACKPROJECTION = "no_finite_backprojection"
NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION = (
    "no_all_camera_positive_depth_initialization"
)
NONFINITE_OBSERVATION = "nonfinite_observation"
NONPOSITIVE_DISPARITY = "nonpositive_disparity"
EXCESSIVE_INITIALIZATION_DEPTH = "excessive_initialization_depth"


class BundleWindowValidationError(RuntimeError):
    """Raised when an explicit single-window contract cannot be satisfied."""


@dataclass(frozen=True)
class ProbeLandmarkEligibility:
    """Caller-supplied landmark eligibility for one window.

    The historical class name comes from its E5-I02 validation origin.  The
    canonical runner constructs the same type from the frozen production
    configuration; the class itself supplies no probe-specific defaults.
    """

    minimum_valid_observations: int
    minimum_distinct_frames: int
    maximum_initialization_depth_m: Optional[float] = None

    def __post_init__(self) -> None:
        for name, value in (
            (
                "minimum_valid_observations",
                self.minimum_valid_observations,
            ),
            ("minimum_distinct_frames", self.minimum_distinct_frames),
        ):
            if type(value) is not int or value < 2:
                raise ValueError("{} must be an integer at least two".format(name))
        if self.maximum_initialization_depth_m is not None:
            maximum_depth = float(self.maximum_initialization_depth_m)
            if not np.isfinite(maximum_depth) or maximum_depth <= 0.0:
                raise ValueError(
                    "maximum_initialization_depth_m must be finite and positive"
                )


@dataclass(frozen=True)
class ProjectionFactorRecord:
    """Canonical identity and measurement for one graph projection factor."""

    factor_index: int
    frame_id: int
    track_id: int
    camera_key: int
    landmark_key: int
    measurement: Tuple[float, float, float]


@dataclass(frozen=True)
class BundleErrorMetrics:
    """Raw graph and factor errors at one Values state."""

    total_graph_error: float
    mean_all_factor_error: float
    projection_factor_error_sum: float
    mean_projection_factor_error: float
    anchor_factor_error: float
    graph_vs_factor_sum_absolute_difference: float


@dataclass(frozen=True)
class MaximumProjectionFactorDiagnostic:
    """Initial/final diagnostics for the largest initial projection factor."""

    factor_index: int
    frame_id: int
    track_id: int
    camera_key: int
    landmark_key: int
    measurement: Tuple[float, float, float]
    initial_projection: Tuple[float, float, float]
    final_projection: Tuple[float, float, float]
    initial_residual: Tuple[float, float, float]
    final_residual: Tuple[float, float, float]
    initial_factor_error: float
    final_factor_error: float


@dataclass
class BundleWindowResult:
    """Structured output of one built and optimized Bundle window."""

    frame_ids: Tuple[int, ...]
    camera_key_by_frame_id: Mapping[int, int]
    track_ids: Tuple[int, ...]
    landmark_key_by_track_id: Mapping[int, int]
    candidate_track_ids: Tuple[int, ...]
    accepted_track_ids: Tuple[int, ...]
    rejected_track_ids: Tuple[int, ...]
    rejected_track_reason_by_id: Mapping[int, str]
    rejection_reason_counts: Mapping[str, int]
    invalid_observation_reason_counts: Mapping[str, int]
    accepted_observation_count_by_track_id: Mapping[int, int]
    initialization_frame_by_track_id: Mapping[int, int]
    initialization_depth_by_track_id: Mapping[int, float]
    excessive_depth_frame_by_track_id: Mapping[int, int]
    excessive_depth_m_by_track_id: Mapping[int, float]
    observation_pairs: Tuple[Tuple[int, int], ...]
    projection_factor_records: Tuple[ProjectionFactorRecord, ...]
    graph: gtsam.NonlinearFactorGraph
    initial_values: gtsam.Values
    final_values: gtsam.Values
    candidate_track_count: int
    accepted_track_count: int
    rejected_track_count: int
    camera_count: int
    landmark_count: int
    projection_factor_count: int
    anchor_factor_count: int
    total_factor_count: int
    variable_dimension: int
    factor_residual_dimension: int
    initial_errors: BundleErrorMetrics
    final_errors: BundleErrorMetrics
    maximum_initial_projection_error: MaximumProjectionFactorDiagnostic
    camera_factor_degrees: Mapping[int, int]
    landmark_factor_degrees: Mapping[int, int]
    connected_to_anchor: bool
    graph_construction_runtime_seconds: float
    optimization_runtime_seconds: float
    optimizer_iterations: int

    def serializable_summary(self) -> Dict[str, object]:
        """Return ordered JSON evidence; execution runtimes are volatile."""

        def ordered_mapping(mapping: Mapping[int, int]) -> List[Dict[str, int]]:
            return [
                {"id": int(identifier), "key": int(mapping[identifier])}
                for identifier in sorted(mapping)
            ]

        def error_dict(metrics: BundleErrorMetrics) -> Dict[str, float]:
            return {
                "total_graph_error": metrics.total_graph_error,
                "mean_all_factor_error": metrics.mean_all_factor_error,
                "projection_factor_error_sum": (
                    metrics.projection_factor_error_sum
                ),
                "mean_projection_factor_error": (
                    metrics.mean_projection_factor_error
                ),
                "anchor_factor_error": metrics.anchor_factor_error,
                "graph_vs_factor_sum_absolute_difference": (
                    metrics.graph_vs_factor_sum_absolute_difference
                ),
            }

        diagnostic = self.maximum_initial_projection_error
        return {
            "ordered_frame_ids": list(self.frame_ids),
            "ordered_camera_mapping": ordered_mapping(
                self.camera_key_by_frame_id
            ),
            "ordered_accepted_track_ids": list(self.track_ids),
            "ordered_landmark_mapping": ordered_mapping(
                self.landmark_key_by_track_id
            ),
            "candidate_track_count": self.candidate_track_count,
            "accepted_track_count": self.accepted_track_count,
            "rejected_track_count": self.rejected_track_count,
            "rejection_reason_counts": dict(self.rejection_reason_counts),
            "invalid_observation_reason_counts": dict(
                self.invalid_observation_reason_counts
            ),
            "rejected_tracks": [
                {
                    "track_id": int(track_id),
                    "reason": self.rejected_track_reason_by_id[track_id],
                }
                for track_id in self.rejected_track_ids
            ],
            "accepted_observation_counts": [
                {
                    "track_id": int(track_id),
                    "count": int(
                        self.accepted_observation_count_by_track_id[track_id]
                    ),
                }
                for track_id in self.track_ids
            ],
            "initialization_frames": [
                {
                    "track_id": int(track_id),
                    "frame_id": int(
                        self.initialization_frame_by_track_id[track_id]
                    ),
                    "initialization_depth_m": float(
                        self.initialization_depth_by_track_id[track_id]
                    ),
                }
                for track_id in self.track_ids
            ],
            "excessive_initialization_depths": [
                {
                    "track_id": int(track_id),
                    "frame_id": int(
                        self.excessive_depth_frame_by_track_id[track_id]
                    ),
                    "initialization_depth_m": float(
                        self.excessive_depth_m_by_track_id[track_id]
                    ),
                }
                for track_id in sorted(self.excessive_depth_m_by_track_id)
            ],
            "counts": {
                "cameras": self.camera_count,
                "landmarks": self.landmark_count,
                "projection_factors": self.projection_factor_count,
                "anchor_factors": self.anchor_factor_count,
                "total_factors": self.total_factor_count,
                "graph_variables": int(self.initial_values.size()),
                "variable_dimension": self.variable_dimension,
                "factor_residual_dimension": self.factor_residual_dimension,
                "unique_observation_pairs": len(set(self.observation_pairs)),
            },
            "initial_errors": error_dict(self.initial_errors),
            "final_errors": error_dict(self.final_errors),
            "maximum_initial_projection_error": {
                "factor_index": diagnostic.factor_index,
                "frame_id": diagnostic.frame_id,
                "track_id": diagnostic.track_id,
                "camera_key": diagnostic.camera_key,
                "landmark_key": diagnostic.landmark_key,
                "measurement": list(diagnostic.measurement),
                "initial_projection": list(diagnostic.initial_projection),
                "final_projection": list(diagnostic.final_projection),
                "initial_residual": list(diagnostic.initial_residual),
                "final_residual": list(diagnostic.final_residual),
                "initial_factor_error": diagnostic.initial_factor_error,
                "final_factor_error": diagnostic.final_factor_error,
            },
            "connectivity": {
                "connected_to_anchor": self.connected_to_anchor,
                "camera_factor_degrees": [
                    {
                        "frame_id": int(frame_id),
                        "degree": int(self.camera_factor_degrees[frame_id]),
                    }
                    for frame_id in self.frame_ids
                ],
                "landmark_factor_degrees": [
                    {
                        "track_id": int(track_id),
                        "degree": int(self.landmark_factor_degrees[track_id]),
                    }
                    for track_id in self.track_ids
                ],
            },
            "optimization": {
                "graph_construction_runtime_seconds": (
                    self.graph_construction_runtime_seconds
                ),
                "optimization_runtime_seconds": (
                    self.optimization_runtime_seconds
                ),
                "optimizer_iterations": self.optimizer_iterations,
                "total_error_change": (
                    self.final_errors.total_graph_error
                    - self.initial_errors.total_graph_error
                ),
            },
        }


@dataclass(frozen=True)
class _Observation:
    """One validated stereo measurement for a frame/track pair."""

    frame_id: int
    track_id: int
    measurement: Tuple[float, float, float]

    def stereo_point(self) -> gtsam.StereoPoint2:
        return gtsam.StereoPoint2(*self.measurement)


def _validated_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be an integer".format(name))
    return int(value)


def _validate_noise_model(
    noise_model: object, expected_dimension: int, name: str
) -> np.ndarray:
    if not hasattr(noise_model, "sigmas"):
        raise TypeError("{} must expose Gaussian sigmas".format(name))
    sigmas = np.asarray(noise_model.sigmas(), dtype=float).reshape(-1)
    if sigmas.shape != (expected_dimension,):
        raise ValueError(
            "{} must have {} sigmas, got {}".format(
                name, expected_dimension, sigmas.shape[0]
            )
        )
    if not np.all(np.isfinite(sigmas)) or not np.all(sigmas > 0.0):
        raise ValueError("{} sigmas must be finite and positive".format(name))
    return sigmas.copy()


def _validate_calibration(calibration: gtsam.Cal3_S2Stereo) -> None:
    fields = np.asarray(
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
    if not np.all(np.isfinite(fields)):
        raise ValueError("stereo calibration must contain finite values")
    if fields[0] <= 0.0 or fields[1] <= 0.0:
        raise ValueError("stereo focal lengths must be positive")
    if fields[2] != 0.0:
        raise ValueError(
            "GTSAM 4.2 ignores nonzero Cal3_S2Stereo skew"
        )
    if fields[5] <= 0.0:
        raise ValueError("stereo baseline must be positive")


def _validate_inputs(
    *,
    database: object,
    world_to_camera_poses: Sequence[np.ndarray],
    calibration: gtsam.Cal3_S2Stereo,
    start_frame_id: int,
    end_frame_id: int,
    projection_noise: object,
    anchor_noise: object,
    landmark_eligibility: ProbeLandmarkEligibility,
    rotation_atol: float,
) -> Tuple[int, int, float]:
    start = _validated_integer(start_frame_id, "start_frame_id")
    end = _validated_integer(end_frame_id, "end_frame_id")
    if start < 0 or end < start:
        raise ValueError("window frame IDs must satisfy 0 <= start <= end")
    if not hasattr(database, "frame_num"):
        raise TypeError("database must expose the lecturer TrackingDB API")
    frame_count = _validated_integer(database.frame_num(), "database frame_num")
    if len(world_to_camera_poses) != frame_count:
        raise ValueError(
            "frame-aligned pose count {} differs from database frame count {}"
            .format(len(world_to_camera_poses), frame_count)
        )
    if end >= frame_count:
        raise ValueError("window end frame is outside the database")
    if not isinstance(landmark_eligibility, ProbeLandmarkEligibility):
        raise TypeError(
            "landmark_eligibility must be ProbeLandmarkEligibility"
        )
    tolerance = float(rotation_atol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rotation_atol must be finite and positive")
    _validate_calibration(calibration)
    _validate_noise_model(projection_noise, 3, "projection_noise")
    _validate_noise_model(anchor_noise, 6, "anchor_noise")
    return start, end, tolerance


def _ordered_frame_tracks(
    database: object, frame_ids: Tuple[int, ...]
) -> Tuple[Dict[int, Tuple[int, ...]], Tuple[int, ...]]:
    """Return deterministic per-frame tracks and their sorted union."""

    by_frame: Dict[int, Tuple[int, ...]] = {}
    candidates = set()
    for frame_id in frame_ids:
        raw_track_ids = database.tracks(frame_id)
        normalized = tuple(
            _validated_integer(track_id, "TrackId")
            for track_id in raw_track_ids
        )
        if any(track_id < 0 for track_id in normalized):
            raise BundleWindowValidationError(
                "database.tracks returned a negative TrackId"
            )
        if len(normalized) != len(set(normalized)):
            raise BundleWindowValidationError(
                "frame {} contains duplicate TrackIds".format(frame_id)
            )
        ordered = tuple(sorted(normalized))
        by_frame[frame_id] = ordered
        candidates.update(ordered)
    return by_frame, tuple(sorted(candidates))


def _collect_valid_observations(
    *,
    database: object,
    frame_ids: Tuple[int, ...],
    track_ids_by_frame: Mapping[int, Tuple[int, ...]],
    candidate_track_id: int,
    invalid_observation_reasons: Counter,
) -> Tuple[_Observation, ...]:
    """Validate TrackingDB membership and retain usable stereo observations."""

    declared_frames = tuple(
        _validated_integer(frame_id, "FrameId")
        for frame_id in database.frames(candidate_track_id)
    )
    if len(declared_frames) != len(set(declared_frames)):
        raise BundleWindowValidationError(
            "track {} contains duplicate FrameIds".format(candidate_track_id)
        )
    ordered_declared_window_frames = tuple(
        sorted(
            frame_id
            for frame_id in declared_frames
            if frame_ids[0] <= frame_id <= frame_ids[-1]
        )
    )
    membership_frames = tuple(
        frame_id
        for frame_id in frame_ids
        if candidate_track_id in track_ids_by_frame[frame_id]
    )
    if ordered_declared_window_frames != membership_frames:
        raise BundleWindowValidationError(
            "track/frame membership differs for TrackId {}".format(
                candidate_track_id
            )
        )

    valid: List[_Observation] = []
    for frame_id in membership_frames:
        link = database.link(frame_id, candidate_track_id)
        if link is None:
            raise BundleWindowValidationError(
                "missing Link for ({}, {})".format(
                    frame_id, candidate_track_id
                )
            )
        measurement = np.asarray(
            [link.x_left, link.x_right, link.y], dtype=float
        )
        if not np.all(np.isfinite(measurement)):
            invalid_observation_reasons[NONFINITE_OBSERVATION] += 1
            continue
        if not float(measurement[0] - measurement[1]) > 0.0:
            invalid_observation_reasons[NONPOSITIVE_DISPARITY] += 1
            continue
        valid.append(
            _Observation(
                frame_id=frame_id,
                track_id=candidate_track_id,
                measurement=tuple(float(value) for value in measurement),
            )
        )
    return tuple(valid)


def _initialize_landmark(
    *,
    observations: Tuple[_Observation, ...],
    camera_pose_by_frame_id: Mapping[int, gtsam.Pose3],
    calibration: gtsam.Cal3_S2Stereo,
    maximum_initialization_depth_m: Optional[float],
) -> Tuple[np.ndarray, int, float, int, float, str]:
    """Backproject a landmark into the window frame from a valid observation.

    Observations are tried latest-first.  A candidate must be finite, respect
    the optional initialization-depth limit, and lie in front of every camera
    that will contribute a projection factor.
    """

    saw_finite_backprojection = False
    excessive_frame_id = -1
    excessive_depth_m = float("nan")
    for initialization_observation in reversed(observations):
        camera = gtsam.StereoCamera(
            camera_pose_by_frame_id[initialization_observation.frame_id],
            calibration,
        )
        point = np.asarray(
            camera.backproject(initialization_observation.stereo_point()),
            dtype=float,
        ).reshape(3)
        if not np.all(np.isfinite(point)):
            continue
        saw_finite_backprojection = True
        initialization_camera_point = np.asarray(
            camera_pose_by_frame_id[
                initialization_observation.frame_id
            ].transformTo(point),
            dtype=float,
        ).reshape(3)
        initialization_depth_m = float(initialization_camera_point[2])
        if (
            maximum_initialization_depth_m is not None
            and np.isfinite(initialization_depth_m)
            and initialization_depth_m > 0.0
            and initialization_depth_m
            >= (
                float(maximum_initialization_depth_m)
                - 4.0
                * np.spacing(float(maximum_initialization_depth_m))
            )
        ):
            if excessive_frame_id < 0:
                excessive_frame_id = initialization_observation.frame_id
                excessive_depth_m = initialization_depth_m
            continue
        positive_in_every_camera = True
        for observation in observations:
            camera_point = np.asarray(
                camera_pose_by_frame_id[observation.frame_id].transformTo(point),
                dtype=float,
            ).reshape(3)
            if (
                not np.all(np.isfinite(camera_point))
                or not float(camera_point[2]) > 0.0
            ):
                positive_in_every_camera = False
                break
        if positive_in_every_camera:
            return (
                point,
                initialization_observation.frame_id,
                initialization_depth_m,
                -1,
                float("nan"),
                "",
            )
    if not saw_finite_backprojection:
        return (
            np.empty(0),
            -1,
            float("nan"),
            -1,
            float("nan"),
            NO_FINITE_BACKPROJECTION,
        )
    if excessive_frame_id >= 0:
        return (
            np.empty(0),
            -1,
            float("nan"),
            excessive_frame_id,
            excessive_depth_m,
            EXCESSIVE_INITIALIZATION_DEPTH,
        )
    return (
        np.empty(0),
        -1,
        float("nan"),
        -1,
        float("nan"),
        NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION,
    )


def _assert_connected(
    *,
    frame_ids: Tuple[int, ...],
    accepted_track_ids: Tuple[int, ...],
    observation_pairs: Tuple[Tuple[int, int], ...],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Require one camera-landmark component connected to the anchor camera."""

    camera_nodes = {("camera", frame_id) for frame_id in frame_ids}
    landmark_nodes = {
        ("landmark", track_id) for track_id in accepted_track_ids
    }
    adjacency = {
        node: set() for node in camera_nodes.union(landmark_nodes)
    }
    camera_degrees = {frame_id: 0 for frame_id in frame_ids}
    landmark_degrees = {track_id: 0 for track_id in accepted_track_ids}

    for frame_id, track_id in observation_pairs:
        camera_node = ("camera", frame_id)
        landmark_node = ("landmark", track_id)
        adjacency[camera_node].add(landmark_node)
        adjacency[landmark_node].add(camera_node)
        camera_degrees[frame_id] += 1
        landmark_degrees[track_id] += 1

    anchor = ("camera", frame_ids[0])
    reached = {anchor}
    queue = deque([anchor])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(adjacency[node]):
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)

    all_nodes = set(adjacency)
    if reached != all_nodes:
        unreachable_frames = sorted(
            identifier
            for kind, identifier in all_nodes - reached
            if kind == "camera"
        )
        unreachable_tracks = sorted(
            identifier
            for kind, identifier in all_nodes - reached
            if kind == "landmark"
        )
        raise BundleWindowValidationError(
            "camera-landmark graph is disconnected from frame {}: "
            "unreachable frames {}, unreachable tracks {}".format(
                frame_ids[0], unreachable_frames, unreachable_tracks
            )
        )
    return camera_degrees, landmark_degrees


def _assert_values_finite(
    *,
    values: gtsam.Values,
    camera_key_by_frame_id: Mapping[int, int],
    landmark_key_by_track_id: Mapping[int, int],
    state_name: str,
) -> None:
    expected_keys = set(camera_key_by_frame_id.values()).union(
        landmark_key_by_track_id.values()
    )
    actual_keys = set(int(key) for key in values.keys())
    if actual_keys != expected_keys:
        raise BundleWindowValidationError(
            "{} Values keys differ from the explicit mappings".format(
                state_name
            )
        )
    for frame_id, key in camera_key_by_frame_id.items():
        matrix = np.asarray(values.atPose3(key).matrix(), dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise BundleWindowValidationError(
                "{} camera {} is nonfinite".format(state_name, frame_id)
            )
    for track_id, key in landmark_key_by_track_id.items():
        point = np.asarray(values.atPoint3(key), dtype=float).reshape(3)
        if not np.all(np.isfinite(point)):
            raise BundleWindowValidationError(
                "{} landmark {} is nonfinite".format(state_name, track_id)
            )


def _direct_projection(
    *,
    record: ProjectionFactorRecord,
    values: gtsam.Values,
    calibration: gtsam.Cal3_S2Stereo,
    state_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    pose = values.atPose3(record.camera_key)
    point = np.asarray(
        values.atPoint3(record.landmark_key), dtype=float
    ).reshape(3)
    camera_point = np.asarray(pose.transformTo(point), dtype=float).reshape(3)
    if (
        not np.all(np.isfinite(camera_point))
        or not float(camera_point[2]) > 0.0
    ):
        raise BundleWindowValidationError(
            "{} projection ({}, {}) has nonpositive or nonfinite depth".format(
                state_name, record.frame_id, record.track_id
            )
        )
    projection = np.asarray(
        gtsam.StereoCamera(pose, calibration).project(point).vector(),
        dtype=float,
    ).reshape(3)
    if not np.all(np.isfinite(projection)):
        raise BundleWindowValidationError(
            "{} projection ({}, {}) is nonfinite".format(
                state_name, record.frame_id, record.track_id
            )
        )
    measurement = np.asarray(record.measurement, dtype=float)
    return projection, projection - measurement


def _error_metrics(
    *,
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    projection_records: Tuple[ProjectionFactorRecord, ...],
    calibration: gtsam.Cal3_S2Stereo,
    state_name: str,
) -> Tuple[BundleErrorMetrics, Tuple[float, ...]]:
    """Evaluate raw graph and stereo-factor errors at one Values state."""

    total_error = float(graph.error(values))
    anchor_error = float(graph.at(0).error(values))
    projection_errors: List[float] = []
    factor_error_sum = anchor_error

    for record in projection_records:
        factor = graph.at(record.factor_index)
        _direct_projection(
            record=record,
            values=values,
            calibration=calibration,
            state_name=state_name,
        )
        unwhitened = np.asarray(
            factor.unwhitenedError(values), dtype=float
        ).reshape(3)
        whitened = np.asarray(
            factor.whitenedError(values), dtype=float
        ).reshape(3)
        factor_error = float(factor.error(values))
        expected_error = float(0.5 * whitened @ whitened)
        if not np.all(np.isfinite(unwhitened)) or not np.all(
            np.isfinite(whitened)
        ):
            raise BundleWindowValidationError(
                "{} factor ({}, {}) has nonfinite residuals".format(
                    state_name, record.frame_id, record.track_id
                )
            )
        if not np.isfinite(factor_error) or not np.isfinite(expected_error):
            raise BundleWindowValidationError(
                "{} factor ({}, {}) has nonfinite error".format(
                    state_name, record.frame_id, record.track_id
                )
            )
        if factor_error != expected_error:
            difference = abs(factor_error - expected_error)
            scale = max(1.0, abs(factor_error), abs(expected_error))
            if difference > np.finfo(float).eps * scale * 8.0:
                raise BundleWindowValidationError(
                    "{} factor error disagrees with its whitened residual"
                    .format(state_name)
                )
        projection_errors.append(factor_error)
        factor_error_sum += factor_error

    projection_error_sum = float(sum(projection_errors))
    graph_factor_difference = abs(total_error - factor_error_sum)
    metrics = BundleErrorMetrics(
        total_graph_error=total_error,
        mean_all_factor_error=total_error / float(graph.size()),
        projection_factor_error_sum=projection_error_sum,
        mean_projection_factor_error=(
            projection_error_sum / float(len(projection_records))
        ),
        anchor_factor_error=anchor_error,
        graph_vs_factor_sum_absolute_difference=graph_factor_difference,
    )
    metric_values = np.asarray(
        [
            metrics.total_graph_error,
            metrics.mean_all_factor_error,
            metrics.projection_factor_error_sum,
            metrics.mean_projection_factor_error,
            metrics.anchor_factor_error,
            metrics.graph_vs_factor_sum_absolute_difference,
        ]
    )
    if not np.all(np.isfinite(metric_values)):
        raise BundleWindowValidationError(
            "{} graph metrics are nonfinite".format(state_name)
        )
    return metrics, tuple(projection_errors)


def _maximum_projection_diagnostic(
    *,
    graph: gtsam.NonlinearFactorGraph,
    initial_values: gtsam.Values,
    final_values: gtsam.Values,
    calibration: gtsam.Cal3_S2Stereo,
    projection_records: Tuple[ProjectionFactorRecord, ...],
    initial_projection_errors: Tuple[float, ...],
) -> MaximumProjectionFactorDiagnostic:
    """Describe the factor with the largest initial stereo-factor error."""

    maximum_position = max(
        range(len(projection_records)),
        key=lambda position: (
            initial_projection_errors[position],
            -position,
        ),
    )
    record = projection_records[maximum_position]
    initial_projection, initial_residual = _direct_projection(
        record=record,
        values=initial_values,
        calibration=calibration,
        state_name="initial",
    )
    final_projection, final_residual = _direct_projection(
        record=record,
        values=final_values,
        calibration=calibration,
        state_name="final",
    )
    factor = graph.at(record.factor_index)
    initial_factor_residual = np.asarray(
        factor.unwhitenedError(initial_values), dtype=float
    ).reshape(3)
    final_factor_residual = np.asarray(
        factor.unwhitenedError(final_values), dtype=float
    ).reshape(3)
    for state_name, direct, factor_residual in (
        ("initial", initial_residual, initial_factor_residual),
        ("final", final_residual, final_factor_residual),
    ):
        difference = float(np.max(np.abs(direct - factor_residual)))
        scale = max(
            1.0,
            float(np.max(np.abs(direct))),
            float(np.max(np.abs(factor_residual))),
        )
        if difference > np.finfo(float).eps * scale * 16.0:
            raise BundleWindowValidationError(
                "{} direct and factor residuals disagree for the maximum "
                "projection factor".format(state_name)
            )
    return MaximumProjectionFactorDiagnostic(
        factor_index=record.factor_index,
        frame_id=record.frame_id,
        track_id=record.track_id,
        camera_key=record.camera_key,
        landmark_key=record.landmark_key,
        measurement=record.measurement,
        initial_projection=tuple(float(x) for x in initial_projection),
        final_projection=tuple(float(x) for x in final_projection),
        initial_residual=tuple(float(x) for x in initial_residual),
        final_residual=tuple(float(x) for x in final_residual),
        initial_factor_error=float(factor.error(initial_values)),
        final_factor_error=float(factor.error(final_values)),
    )


def solve_bundle_window(
    *,
    database: object,
    world_to_camera_poses: Sequence[np.ndarray],
    calibration: gtsam.Cal3_S2Stereo,
    start_frame_id: int,
    end_frame_id: int,
    projection_noise: object,
    anchor_noise: object,
    landmark_eligibility: ProbeLandmarkEligibility,
    rotation_atol: float,
) -> BundleWindowResult:
    """Build, validate, and optimize exactly one inclusive Bundle window.

    Camera poses and landmarks share the first camera's coordinate system.
    Exactly one prior anchors that first camera, while every other factor is a
    stereo reprojection constraint.  Production policy remains caller-owned.

    The function preserves GTSAM failures and refuses to optimize a
    disconnected graph.  Invalid observations and invalid landmark
    initializations are counted explicitly; unexpected TrackingDB
    inconsistencies raise :class:`BundleWindowValidationError`.
    """

    start, end, tolerance = _validate_inputs(
        database=database,
        world_to_camera_poses=world_to_camera_poses,
        calibration=calibration,
        start_frame_id=start_frame_id,
        end_frame_id=end_frame_id,
        projection_noise=projection_noise,
        anchor_noise=anchor_noise,
        landmark_eligibility=landmark_eligibility,
        rotation_atol=rotation_atol,
    )
    construction_started = perf_counter()
    frame_ids = tuple(range(start, end + 1))
    camera_key_by_frame_id = {
        frame_id: int(gtsam.symbol(CAMERA_SYMBOL, frame_id))
        for frame_id in frame_ids
    }
    if len(set(camera_key_by_frame_id.values())) != len(frame_ids):
        raise BundleWindowValidationError("camera GTSAM keys collide")

    # Express every camera as T_Cstart_from_Ci so cameras and landmarks share
    # the first frame's coordinate system.
    camera_pose_by_frame_id: Dict[int, gtsam.Pose3] = {}
    initial_values = gtsam.Values()
    origin_extrinsic = world_to_camera_poses[start]
    for frame_id in frame_ids:
        pose = camera_pose_in_window(
            origin_extrinsic,
            world_to_camera_poses[frame_id],
            rotation_atol=tolerance,
        )
        if frame_id == start:
            identity_error = float(
                np.max(np.abs(np.asarray(pose.matrix()) - np.eye(4)))
            )
            if identity_error > tolerance:
                raise BundleWindowValidationError(
                    "first camera is not identity in window coordinates"
                )
            pose = gtsam.Pose3()
        camera_pose_by_frame_id[frame_id] = pose
        initial_values.insert(camera_key_by_frame_id[frame_id], pose)

    # A single first-camera prior fixes the local gauge.  No additional camera
    # or landmark priors are introduced by this solver.
    graph = gtsam.NonlinearFactorGraph()
    anchor_factor = gtsam.PriorFactorPose3(
        camera_key_by_frame_id[start],
        gtsam.Pose3(),
        anchor_noise,
    )
    graph.add(anchor_factor)

    # Collect stereo observations, then initialize only landmarks that satisfy
    # the caller's complete eligibility contract.
    track_ids_by_frame, candidate_track_ids = _ordered_frame_tracks(
        database, frame_ids
    )
    invalid_observation_reasons: Counter = Counter()
    rejected_reasons: Counter = Counter()
    rejected_reason_by_id: Dict[int, str] = {}
    accepted_observations: Dict[int, Tuple[_Observation, ...]] = {}
    initial_landmarks: Dict[int, np.ndarray] = {}
    initialization_frame_by_track_id: Dict[int, int] = {}
    initialization_depth_by_track_id: Dict[int, float] = {}
    excessive_depth_frame_by_track_id: Dict[int, int] = {}
    excessive_depth_m_by_track_id: Dict[int, float] = {}

    for track_id in candidate_track_ids:
        observations = _collect_valid_observations(
            database=database,
            frame_ids=frame_ids,
            track_ids_by_frame=track_ids_by_frame,
            candidate_track_id=track_id,
            invalid_observation_reasons=invalid_observation_reasons,
        )
        distinct_frame_count = len(
            {observation.frame_id for observation in observations}
        )
        if (
            len(observations)
            < landmark_eligibility.minimum_valid_observations
            or distinct_frame_count
            < landmark_eligibility.minimum_distinct_frames
        ):
            reason = INSUFFICIENT_VALID_OBSERVATIONS
            rejected_reasons[reason] += 1
            rejected_reason_by_id[track_id] = reason
            continue

        (
            point,
            initialization_frame_id,
            initialization_depth_m,
            excessive_frame_id,
            excessive_depth_m,
            reason,
        ) = _initialize_landmark(
            observations=observations,
            camera_pose_by_frame_id=camera_pose_by_frame_id,
            calibration=calibration,
            maximum_initialization_depth_m=(
                landmark_eligibility.maximum_initialization_depth_m
            ),
        )
        if reason:
            rejected_reasons[reason] += 1
            rejected_reason_by_id[track_id] = reason
            if reason == EXCESSIVE_INITIALIZATION_DEPTH:
                excessive_depth_frame_by_track_id[track_id] = (
                    excessive_frame_id
                )
                excessive_depth_m_by_track_id[track_id] = excessive_depth_m
            continue
        accepted_observations[track_id] = observations
        initial_landmarks[track_id] = point
        initialization_frame_by_track_id[track_id] = initialization_frame_id
        initialization_depth_by_track_id[track_id] = initialization_depth_m

    accepted_track_ids = tuple(sorted(accepted_observations))
    rejected_track_ids = tuple(sorted(rejected_reason_by_id))
    if len(accepted_track_ids) + len(rejected_track_ids) != len(
        candidate_track_ids
    ):
        raise BundleWindowValidationError(
            "candidate track accounting is incomplete"
        )
    if sum(rejected_reasons.values()) != len(rejected_track_ids):
        raise BundleWindowValidationError(
            "rejected track reasons are not exclusive"
        )
    if not accepted_track_ids:
        raise BundleWindowValidationError(
            "window contains no accepted landmarks"
        )

    landmark_key_by_track_id = {
        track_id: int(gtsam.symbol(LANDMARK_SYMBOL, track_id))
        for track_id in accepted_track_ids
    }
    all_variable_keys = list(camera_key_by_frame_id.values()) + list(
        landmark_key_by_track_id.values()
    )
    if len(all_variable_keys) != len(set(all_variable_keys)):
        raise BundleWindowValidationError("GTSAM variable keys collide")

    # Insert each accepted q landmark before its deterministic sequence of
    # GenericStereoFactor3D observations.
    projection_records: List[ProjectionFactorRecord] = []
    observation_pairs: List[Tuple[int, int]] = []
    seen_observation_pairs = set()
    for track_id in accepted_track_ids:
        landmark_key = landmark_key_by_track_id[track_id]
        initial_values.insert(landmark_key, initial_landmarks[track_id])
        for observation in accepted_observations[track_id]:
            pair = (observation.frame_id, track_id)
            if pair in seen_observation_pairs:
                raise BundleWindowValidationError(
                    "duplicate projection observation {}".format(pair)
                )
            seen_observation_pairs.add(pair)
            factor = gtsam.GenericStereoFactor3D(
                observation.stereo_point(),
                projection_noise,
                camera_key_by_frame_id[observation.frame_id],
                landmark_key,
                calibration,
            )
            graph.add(factor)
            projection_records.append(
                ProjectionFactorRecord(
                    factor_index=int(graph.size() - 1),
                    frame_id=observation.frame_id,
                    track_id=track_id,
                    camera_key=camera_key_by_frame_id[
                        observation.frame_id
                    ],
                    landmark_key=landmark_key,
                    measurement=observation.measurement,
                )
            )
            observation_pairs.append(pair)

    projection_records_tuple = tuple(projection_records)
    observation_pairs_tuple = tuple(observation_pairs)
    if not projection_records_tuple:
        raise BundleWindowValidationError(
            "window contains no projection factors"
        )
    if len(observation_pairs_tuple) != len(set(observation_pairs_tuple)):
        raise BundleWindowValidationError(
            "projection observation pairs are not unique"
        )
    if graph.size() != len(projection_records_tuple) + 1:
        raise BundleWindowValidationError(
            "factor count differs from one prior plus projections"
        )
    if initial_values.size() != len(frame_ids) + len(accepted_track_ids):
        raise BundleWindowValidationError(
            "Values variable count differs from mappings"
        )

    # Fail before optimization if the graph is disconnected, internally
    # inconsistent, nonfinite, or dimensionally underconstrained.
    camera_degrees, landmark_degrees = _assert_connected(
        frame_ids=frame_ids,
        accepted_track_ids=accepted_track_ids,
        observation_pairs=observation_pairs_tuple,
    )
    for track_id, degree in landmark_degrees.items():
        if (
            degree < landmark_eligibility.minimum_valid_observations
            or degree < landmark_eligibility.minimum_distinct_frames
        ):
            raise BundleWindowValidationError(
                "accepted landmark {} has insufficient factors".format(
                    track_id
                )
            )

    variable_dimension = int(initial_values.dim())
    factor_residual_dimension = int(
        sum(
            graph.at(factor_index).dim()
            for factor_index in range(graph.size())
        )
    )
    if factor_residual_dimension < variable_dimension:
        raise BundleWindowValidationError(
            "connected graph is necessarily underconstrained by dimension: "
            "{} factor residual dimensions for {} variable dimensions".format(
                factor_residual_dimension, variable_dimension
            )
        )

    _assert_values_finite(
        values=initial_values,
        camera_key_by_frame_id=camera_key_by_frame_id,
        landmark_key_by_track_id=landmark_key_by_track_id,
        state_name="initial",
    )
    # Retain the raw initial objective before constructing the optimizer.
    initial_errors, initial_projection_errors = _error_metrics(
        graph=graph,
        values=initial_values,
        projection_records=projection_records_tuple,
        calibration=calibration,
        state_name="initial",
    )
    graph_construction_runtime_seconds = perf_counter() - construction_started

    # The frozen production path uses GTSAM's default Levenberg-Marquardt
    # construction exactly as shown here.
    optimization_started = perf_counter()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    final_values = optimizer.optimize()
    optimization_runtime_seconds = perf_counter() - optimization_started

    _assert_values_finite(
        values=final_values,
        camera_key_by_frame_id=camera_key_by_frame_id,
        landmark_key_by_track_id=landmark_key_by_track_id,
        state_name="final",
    )
    final_errors, _ = _error_metrics(
        graph=graph,
        values=final_values,
        projection_records=projection_records_tuple,
        calibration=calibration,
        state_name="final",
    )
    diagnostic = _maximum_projection_diagnostic(
        graph=graph,
        initial_values=initial_values,
        final_values=final_values,
        calibration=calibration,
        projection_records=projection_records_tuple,
        initial_projection_errors=initial_projection_errors,
    )

    return BundleWindowResult(
        frame_ids=frame_ids,
        camera_key_by_frame_id=camera_key_by_frame_id,
        track_ids=accepted_track_ids,
        landmark_key_by_track_id=landmark_key_by_track_id,
        candidate_track_ids=candidate_track_ids,
        accepted_track_ids=accepted_track_ids,
        rejected_track_ids=rejected_track_ids,
        rejected_track_reason_by_id=dict(rejected_reason_by_id),
        rejection_reason_counts=dict(sorted(rejected_reasons.items())),
        invalid_observation_reason_counts=dict(
            sorted(invalid_observation_reasons.items())
        ),
        accepted_observation_count_by_track_id={
            track_id: len(accepted_observations[track_id])
            for track_id in accepted_track_ids
        },
        initialization_frame_by_track_id=dict(
            initialization_frame_by_track_id
        ),
        initialization_depth_by_track_id=dict(
            initialization_depth_by_track_id
        ),
        excessive_depth_frame_by_track_id=dict(
            excessive_depth_frame_by_track_id
        ),
        excessive_depth_m_by_track_id=dict(excessive_depth_m_by_track_id),
        observation_pairs=observation_pairs_tuple,
        projection_factor_records=projection_records_tuple,
        graph=graph,
        initial_values=initial_values,
        final_values=final_values,
        candidate_track_count=len(candidate_track_ids),
        accepted_track_count=len(accepted_track_ids),
        rejected_track_count=len(rejected_track_ids),
        camera_count=len(frame_ids),
        landmark_count=len(accepted_track_ids),
        projection_factor_count=len(projection_records_tuple),
        anchor_factor_count=1,
        total_factor_count=int(graph.size()),
        variable_dimension=variable_dimension,
        factor_residual_dimension=factor_residual_dimension,
        initial_errors=initial_errors,
        final_errors=final_errors,
        maximum_initial_projection_error=diagnostic,
        camera_factor_degrees=camera_degrees,
        landmark_factor_degrees=landmark_degrees,
        connected_to_anchor=True,
        graph_construction_runtime_seconds=(
            graph_construction_runtime_seconds
        ),
        optimization_runtime_seconds=optimization_runtime_seconds,
        optimizer_iterations=int(optimizer.iterations()),
    )


__all__ = [
    "BundleErrorMetrics",
    "BundleWindowResult",
    "BundleWindowValidationError",
    "MaximumProjectionFactorDiagnostic",
    "ProbeLandmarkEligibility",
    "ProjectionFactorRecord",
    "INSUFFICIENT_VALID_OBSERVATIONS",
    "NO_FINITE_BACKPROJECTION",
    "NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION",
    "NONFINITE_OBSERVATION",
    "NONPOSITIVE_DISPARITY",
    "EXCESSIVE_INITIALIZATION_DEPTH",
    "solve_bundle_window",
]
