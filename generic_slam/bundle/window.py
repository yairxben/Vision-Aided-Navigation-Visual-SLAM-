"""Sequence-independent local stereo Bundle Adjustment kernel."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from ..contracts import ContractError
from ..geometry import validate_rigid_pose


CAMERA_SYMBOL = "c"
LANDMARK_SYMBOL = "q"
INSUFFICIENT_VALID_OBSERVATIONS = "insufficient_valid_observations"
NO_FINITE_BACKPROJECTION = "no_finite_backprojection"
NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION = "no_all_camera_positive_depth_initialization"
NONFINITE_OBSERVATION = "nonfinite_observation"
NONPOSITIVE_DISPARITY = "nonpositive_disparity"
EXCESSIVE_INITIALIZATION_DEPTH = "excessive_initialization_depth"


@dataclass(frozen=True)
class LandmarkEligibility:
    minimum_valid_observations: int
    minimum_distinct_frames: int
    maximum_initialization_depth_m: Optional[float]

    def __post_init__(self) -> None:
        if self.minimum_valid_observations < 2 or self.minimum_distinct_frames < 2:
            raise ContractError("Bundle landmark minima must be at least two")
        if self.maximum_initialization_depth_m is not None and (
            not np.isfinite(self.maximum_initialization_depth_m)
            or self.maximum_initialization_depth_m <= 0.0
        ):
            raise ContractError("Bundle maximum initialization depth is invalid")


@dataclass(frozen=True)
class StereoObservation:
    frame_id: int
    track_id: int
    measurement: Tuple[float, float, float]

    def point(self) -> gtsam.StereoPoint2:
        return gtsam.StereoPoint2(*self.measurement)


@dataclass(frozen=True)
class ProjectionFactorRecord:
    factor_index: int
    frame_id: int
    track_id: int
    camera_key: int
    landmark_key: int
    measurement: Tuple[float, float, float]


@dataclass
class BundleWindowResult:
    frame_ids: Tuple[int, ...]
    camera_key_by_frame_id: Mapping[int, int]
    candidate_track_ids: Tuple[int, ...]
    accepted_track_ids: Tuple[int, ...]
    rejected_reason_by_track_id: Mapping[int, str]
    initialization_frame_by_track_id: Mapping[int, int]
    initialization_depth_by_track_id: Mapping[int, float]
    landmark_key_by_track_id: Mapping[int, int]
    projection_records: Tuple[ProjectionFactorRecord, ...]
    graph: gtsam.NonlinearFactorGraph
    initial_values: gtsam.Values
    final_values: gtsam.Values
    initial_graph_error: float
    final_graph_error: float
    initial_projection_errors: np.ndarray
    final_projection_errors: np.ndarray
    initial_residuals: np.ndarray
    final_residuals: np.ndarray
    initial_cheirality_factor_count: int
    final_cheirality_factor_count: int
    camera_factor_degrees: Mapping[int, int]
    landmark_factor_degrees: Mapping[int, int]
    invalid_observation_reason_counts: Mapping[str, int]
    graph_construction_runtime_seconds: float
    optimization_runtime_seconds: float
    optimizer_iterations: int

    @property
    def start_frame_id(self) -> int:
        return self.frame_ids[0]

    @property
    def end_frame_id(self) -> int:
        return self.frame_ids[-1]


def _camera_pose_from_extrinsic(extrinsic: np.ndarray, atol: float) -> gtsam.Pose3:
    value = validate_rigid_pose(extrinsic, atol=atol)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(rotation.T), -rotation.T @ translation)


def _camera_pose_in_window(
    origin_extrinsic: np.ndarray, camera_extrinsic: np.ndarray, atol: float
) -> gtsam.Pose3:
    return _camera_pose_from_extrinsic(origin_extrinsic, atol).between(
        _camera_pose_from_extrinsic(camera_extrinsic, atol)
    )


def _collect_candidates(database: object, frame_ids: Tuple[int, ...]):
    by_frame = {}
    candidate_ids = set()
    for frame_id in frame_ids:
        tracks = tuple(sorted(int(value) for value in database.tracks(frame_id)))
        if len(tracks) != len(set(tracks)) or any(value < 0 for value in tracks):
            raise ContractError("TrackingDB returned invalid TrackIds")
        by_frame[frame_id] = tracks
        candidate_ids.update(tracks)
    return by_frame, tuple(sorted(candidate_ids))


def _observations(
    database: object,
    frame_ids: Tuple[int, ...],
    by_frame: Mapping[int, Tuple[int, ...]],
    track_id: int,
    invalid_reasons: Counter,
) -> Tuple[StereoObservation, ...]:
    declared = tuple(
        sorted(
            int(value)
            for value in database.frames(track_id)
            if frame_ids[0] <= int(value) <= frame_ids[-1]
        )
    )
    membership = tuple(
        frame_id for frame_id in frame_ids if track_id in by_frame[frame_id]
    )
    if declared != membership:
        raise ContractError("TrackingDB track/frame membership differs")
    result = []
    for frame_id in membership:
        link = database.link(frame_id, track_id)
        if link is None:
            raise ContractError("TrackingDB observation is missing")
        measurement = np.asarray([link.x_left, link.x_right, link.y], dtype=float)
        if not np.all(np.isfinite(measurement)):
            invalid_reasons[NONFINITE_OBSERVATION] += 1
        elif measurement[0] - measurement[1] <= 0.0:
            invalid_reasons[NONPOSITIVE_DISPARITY] += 1
        else:
            result.append(
                StereoObservation(
                    frame_id, track_id, tuple(float(value) for value in measurement)
                )
            )
    return tuple(result)


def _initialize_landmark(
    observations: Tuple[StereoObservation, ...],
    camera_poses: Mapping[int, gtsam.Pose3],
    calibration: gtsam.Cal3_S2Stereo,
    maximum_depth: Optional[float],
):
    backprojected = []
    for observation in observations:
        camera = gtsam.StereoCamera(camera_poses[observation.frame_id], calibration)
        point = np.asarray(camera.backproject(observation.point()), dtype=float).reshape(3)
        depth = float("nan")
        if np.all(np.isfinite(point)):
            depth = float(np.asarray(
                camera_poses[observation.frame_id].transformTo(point), dtype=float
            ).reshape(3)[2])
        backprojected.append((observation, point, depth))
    if maximum_depth is not None:
        # Reject the whole landmark if any observation is too far: near-zero
        # disparity leaves depth unconstrained and makes Marginals singular.
        limit = maximum_depth - 4.0 * np.spacing(maximum_depth)
        for observation, _, depth in reversed(backprojected):
            if np.isfinite(depth) and depth > 0.0 and depth >= limit:
                return None, observation.frame_id, depth, EXCESSIVE_INITIALIZATION_DEPTH
    saw_finite = False
    for observation, point, depth in reversed(backprojected):
        if not np.all(np.isfinite(point)):
            continue
        saw_finite = True
        if all(
            np.all(np.isfinite(camera_poses[item.frame_id].transformTo(point)))
            and float(np.asarray(camera_poses[item.frame_id].transformTo(point))[2]) > 0.0
            for item in observations
        ):
            return point, observation.frame_id, depth, ""
    if not saw_finite:
        return None, -1, float("nan"), NO_FINITE_BACKPROJECTION
    return None, -1, float("nan"), NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION


def _connectivity(
    frame_ids: Tuple[int, ...], track_ids: Tuple[int, ...], pairs: Sequence[Tuple[int, int]]
):
    nodes = {("c", value) for value in frame_ids}.union(
        {("q", value) for value in track_ids}
    )
    adjacency = {node: set() for node in nodes}
    camera_degrees = {value: 0 for value in frame_ids}
    landmark_degrees = {value: 0 for value in track_ids}
    for frame_id, track_id in pairs:
        camera = ("c", frame_id)
        landmark = ("q", track_id)
        adjacency[camera].add(landmark)
        adjacency[landmark].add(camera)
        camera_degrees[frame_id] += 1
        landmark_degrees[track_id] += 1
    reached = {("c", frame_ids[0])}
    queue = deque(reached)
    while queue:
        for neighbor in adjacency[queue.popleft()]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    if reached != nodes:
        raise ContractError("Bundle camera-landmark graph is disconnected")
    return camera_degrees, landmark_degrees


def _projection_evidence(
    records: Tuple[ProjectionFactorRecord, ...],
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    calibration: gtsam.Cal3_S2Stereo,
):
    residuals = []
    errors = []
    cheirality_count = 0
    for record in records:
        camera = gtsam.StereoCamera(values.atPose3(record.camera_key), calibration)
        point = np.asarray(values.atPoint3(record.landmark_key), dtype=float).reshape(3)
        factor_residual = np.asarray(
            graph.at(record.factor_index).unwhitenedError(values), dtype=float
        ).reshape(3)
        try:
            projection = np.asarray(camera.project(point).vector(), dtype=float).reshape(3)
        except RuntimeError as error:
            if "Cheirality" not in str(error):
                raise
            # GenericStereoFactor3D converts cheirality into its finite 2*fx
            # penalty. Retain that exact solver residual as diagnostic evidence
            # instead of allowing reporting to abort a completed optimization.
            cheirality_count += 1
            residual = factor_residual
        else:
            residual = projection - np.asarray(record.measurement, dtype=float)
            if not np.allclose(
                residual, factor_residual, rtol=0.0,
                atol=np.finfo(float).eps * 128,
            ):
                raise ContractError("direct and GTSAM projection residuals differ")
        factor_error = float(graph.at(record.factor_index).error(values))
        if not np.all(np.isfinite(factor_residual)) or not np.isfinite(factor_error):
            raise ContractError("Bundle projection evidence is nonfinite")
        residuals.append(residual)
        errors.append(factor_error)
    return (
        np.asarray(errors, dtype=float),
        np.asarray(residuals, dtype=float),
        cheirality_count,
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
    landmark_eligibility: LandmarkEligibility,
    rotation_atol: float,
    graph_error_nonincrease_absolute: float,
    graph_error_nonincrease_relative: float,
) -> BundleWindowResult:
    """Build and optimize one inclusive window with the validated factor ordering."""

    start = int(start_frame_id)
    end = int(end_frame_id)
    if start < 0 or end <= start or end >= database.frame_num():
        raise ContractError("Bundle window endpoint range is invalid")
    if len(world_to_camera_poses) != database.frame_num():
        raise ContractError("Bundle poses and TrackingDB frames differ")
    construction_started = perf_counter()
    frame_ids = tuple(range(start, end + 1))
    camera_keys = {
        frame_id: int(gtsam.symbol(CAMERA_SYMBOL, frame_id)) for frame_id in frame_ids
    }
    camera_poses = {}
    initial = gtsam.Values()
    for frame_id in frame_ids:
        pose = _camera_pose_in_window(
            np.asarray(world_to_camera_poses[start]),
            np.asarray(world_to_camera_poses[frame_id]),
            rotation_atol,
        )
        if frame_id == start:
            if np.max(np.abs(np.asarray(pose.matrix()) - np.eye(4))) > rotation_atol:
                raise ContractError("Bundle anchor is not local identity")
            pose = gtsam.Pose3()
        camera_poses[frame_id] = pose
        initial.insert(camera_keys[frame_id], pose)
    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.PriorFactorPose3(camera_keys[start], gtsam.Pose3(), anchor_noise))

    by_frame, candidate_tracks = _collect_candidates(database, frame_ids)
    invalid_reasons = Counter()
    accepted = {}
    landmarks = {}
    initialization_frames = {}
    initialization_depths = {}
    rejected = {}
    for track_id in candidate_tracks:
        observations = _observations(
            database, frame_ids, by_frame, track_id, invalid_reasons
        )
        if (
            len(observations) < landmark_eligibility.minimum_valid_observations
            or len({item.frame_id for item in observations})
            < landmark_eligibility.minimum_distinct_frames
        ):
            rejected[track_id] = INSUFFICIENT_VALID_OBSERVATIONS
            continue
        point, frame_id, depth, reason = _initialize_landmark(
            observations,
            camera_poses,
            calibration,
            landmark_eligibility.maximum_initialization_depth_m,
        )
        if reason:
            rejected[track_id] = reason
            continue
        accepted[track_id] = observations
        landmarks[track_id] = point
        initialization_frames[track_id] = frame_id
        initialization_depths[track_id] = depth
    accepted_tracks = tuple(sorted(accepted))
    if not accepted_tracks:
        raise ContractError("Bundle window contains no eligible landmarks")
    landmark_keys = {
        track_id: int(gtsam.symbol(LANDMARK_SYMBOL, track_id))
        for track_id in accepted_tracks
    }
    records = []
    pairs = []
    for track_id in accepted_tracks:
        initial.insert(landmark_keys[track_id], landmarks[track_id])
        for observation in accepted[track_id]:
            graph.add(
                gtsam.GenericStereoFactor3D(
                    observation.point(), projection_noise,
                    camera_keys[observation.frame_id], landmark_keys[track_id],
                    calibration,
                )
            )
            records.append(
                ProjectionFactorRecord(
                    int(graph.size() - 1), observation.frame_id, track_id,
                    camera_keys[observation.frame_id], landmark_keys[track_id],
                    observation.measurement,
                )
            )
            pairs.append((observation.frame_id, track_id))
    records_tuple = tuple(records)
    camera_degrees, landmark_degrees = _connectivity(
        frame_ids, accepted_tracks, pairs
    )
    if sum(graph.at(index).dim() for index in range(graph.size())) < initial.dim():
        raise ContractError("Bundle graph is dimensionally underconstrained")
    initial_error = float(graph.error(initial))
    (
        initial_projection_errors,
        initial_residuals,
        initial_cheirality_factor_count,
    ) = _projection_evidence(
        records_tuple, graph, initial, calibration
    )
    construction_seconds = perf_counter() - construction_started

    optimization_started = perf_counter()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    final = optimizer.optimize()
    optimization_seconds = perf_counter() - optimization_started
    final_error = float(graph.error(final))
    permitted = max(
        float(graph_error_nonincrease_absolute),
        abs(initial_error) * float(graph_error_nonincrease_relative),
    )
    if final_error > initial_error + permitted:
        raise ContractError("Bundle optimization increased graph error")
    (
        final_projection_errors,
        final_residuals,
        final_cheirality_factor_count,
    ) = _projection_evidence(
        records_tuple, graph, final, calibration
    )
    return BundleWindowResult(
        frame_ids=frame_ids,
        camera_key_by_frame_id=camera_keys,
        candidate_track_ids=candidate_tracks,
        accepted_track_ids=accepted_tracks,
        rejected_reason_by_track_id=dict(rejected),
        initialization_frame_by_track_id=initialization_frames,
        initialization_depth_by_track_id=initialization_depths,
        landmark_key_by_track_id=landmark_keys,
        projection_records=records_tuple,
        graph=graph,
        initial_values=initial,
        final_values=final,
        initial_graph_error=initial_error,
        final_graph_error=final_error,
        initial_projection_errors=initial_projection_errors,
        final_projection_errors=final_projection_errors,
        initial_residuals=initial_residuals,
        final_residuals=final_residuals,
        initial_cheirality_factor_count=initial_cheirality_factor_count,
        final_cheirality_factor_count=final_cheirality_factor_count,
        camera_factor_degrees=camera_degrees,
        landmark_factor_degrees=landmark_degrees,
        invalid_observation_reason_counts=dict(sorted(invalid_reasons.items())),
        graph_construction_runtime_seconds=construction_seconds,
        optimization_runtime_seconds=optimization_seconds,
        optimizer_iterations=int(optimizer.iterations()),
    )


__all__ = [
    "BundleWindowResult",
    "LandmarkEligibility",
    "ProjectionFactorRecord",
    "solve_bundle_window",
]
