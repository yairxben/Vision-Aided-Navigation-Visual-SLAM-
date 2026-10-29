"""Sequence-independent local stereo Bundle Adjustment kernel."""

from __future__ import annotations

import math
from collections import Counter
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
# Batched projection evidence must agree with GTSAM's own factor evaluation.
EVIDENCE_AGREEMENT_TOLERANCE = 1e-9
_EVIDENCE_SPOT_CHECKS = 16
_ARITHMETIC_SPOT_CHECKS = 8


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
class _WindowObservations:
    """Valid observations of the candidate tracks, by TrackId then FrameId."""

    track_ids: np.ndarray  # (T,) candidate TrackIds, ascending
    starts: np.ndarray  # (T,) each candidate's slice of the observation rows
    counts: np.ndarray  # (T,)
    frame_ids: np.ndarray  # (M,)
    measurements: np.ndarray  # (M, 3) u_left, u_right, v_mean


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
    # One row per projection factor, in graph order (factor 0 is the anchor).
    projection_factor_indices: np.ndarray
    projection_frame_ids: np.ndarray
    projection_track_ids: np.ndarray
    projection_measurements: np.ndarray
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


def _rotate(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """``matrices[i] @ vectors[i]`` in Eigen's fixed-size 3x3 evaluation order.

    GTSAM's Pose3 transforms sum rows 0-1 as (a + b) + c (one SIMD packet) and
    row 2 as a + (b + c). Matching that order keeps the batched kernel
    bit-identical to the per-object GTSAM calls it replaces.
    """

    a = matrices[:, :, 0] * vectors[:, 0:1]
    b = matrices[:, :, 1] * vectors[:, 1:2]
    c = matrices[:, :, 2] * vectors[:, 2:3]
    result = (a + b) + c
    result[:, 2] = a[:, 2] + (b[:, 2] + c[:, 2])
    return result


def _transform_from(rotations: np.ndarray, translations: np.ndarray, points: np.ndarray):
    """Batched ``Pose3.transformFrom`` (camera to window coordinates)."""

    return _rotate(rotations, points) + translations


def _transform_to(rotations: np.ndarray, translations: np.ndarray, points: np.ndarray):
    """Batched ``Pose3.transformTo`` (window to camera coordinates)."""

    return _rotate(np.transpose(rotations, (0, 2, 1)), points - translations)


def _pose_arrays(poses: Sequence[gtsam.Pose3]) -> Tuple[np.ndarray, np.ndarray]:
    rotations = np.stack([np.asarray(pose.rotation().matrix(), dtype=float) for pose in poses])
    translations = np.stack(
        [np.asarray(pose.translation(), dtype=float).reshape(3) for pose in poses]
    )
    return rotations, translations


def _backproject(measurements: np.ndarray, calibration: gtsam.Cal3_S2Stereo) -> np.ndarray:
    """Batched ``StereoCamera.backproject`` in the camera frame (GTSAM's formula)."""

    fx, fy = calibration.fx(), calibration.fy()
    depth = calibration.baseline() * fx / (measurements[:, 0] - measurements[:, 1])
    x = depth * (measurements[:, 0] - calibration.px()) / fx
    y = depth * (measurements[:, 2] - calibration.py()) / fy
    return np.stack((x, y, depth), axis=1)


def _window_observations(
    database: object,
    frame_ids: Tuple[int, ...],
    invalid_reasons: Counter,
) -> _WindowObservations:
    """Candidate TrackIds (ascending) and each one's valid observations in frame order."""

    window = database.window_observations(frame_ids[0], frame_ids[-1])
    track_ids = window["track_ids"]
    links = window["links"]
    if not len(track_ids):
        raise ContractError("Bundle window contains no eligible landmarks")
    boundaries = np.flatnonzero(np.diff(track_ids)) + 1
    starts = np.concatenate(([0], boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries, [len(track_ids)])).astype(np.int64)
    candidates = track_ids[starts]
    # Every track is a consecutive frame range: its observations in the window
    # must be exactly the overlap of that range with the window.
    declared = (
        np.minimum(window["track_end"][candidates], frame_ids[-1])
        - np.maximum(window["track_start"][candidates], frame_ids[0])
        + 1
    )
    if not np.array_equal(declared, ends - starts):
        raise ContractError("tracking store track/frame membership differs")
    finite = np.isfinite(links).all(axis=1)
    positive = links[:, 0] - links[:, 1] > 0.0
    for reason, count in (
        (NONFINITE_OBSERVATION, int(np.count_nonzero(~finite))),
        (NONPOSITIVE_DISPARITY, int(np.count_nonzero(finite & ~positive))),
    ):
        if count:  # reported reasons are only those that occurred
            invalid_reasons[reason] += count
    valid = finite & positive
    counts = np.add.reduceat(valid.astype(np.int64), starts)
    return _WindowObservations(
        track_ids=candidates.astype(np.int64),
        starts=np.cumsum(counts) - counts,
        counts=counts,
        frame_ids=window["frame_ids"][valid].astype(np.int64),
        measurements=np.asarray(links[valid], dtype=np.float64),
    )


def _segment_rows(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Row indices of the concatenated slices ``starts[i]:starts[i] + counts[i]``."""

    total = int(counts.sum())
    offsets = np.cumsum(counts) - counts
    return np.repeat(starts - offsets, counts) + np.arange(total, dtype=np.int64)


def _distinct_frame_counts(observations: _WindowObservations) -> np.ndarray:
    frames = observations.frame_ids
    new_frame = np.ones(len(frames), dtype=np.int64)
    new_frame[1:] = frames[1:] != frames[:-1]
    nonempty = observations.counts > 0
    new_frame[observations.starts[nonempty]] = 1
    distinct = np.zeros(len(observations.counts), dtype=np.int64)
    if np.any(nonempty):
        # Empty slices have zero width, so each nonempty slice runs to the next.
        distinct[nonempty] = np.add.reduceat(new_frame, observations.starts[nonempty])
    return distinct


def _check_gtsam_arithmetic(
    camera_poses: Sequence[gtsam.Pose3],
    calibration: gtsam.Cal3_S2Stereo,
    frame_offsets: np.ndarray,
    measurements: np.ndarray,
    points: np.ndarray,
) -> None:
    """Fail loudly if this GTSAM build evaluates the transforms differently."""

    for row in np.linspace(0, len(points) - 1, min(len(points), _ARITHMETIC_SPOT_CHECKS)).astype(int):
        pose = camera_poses[int(frame_offsets[row])]
        expected = np.asarray(
            gtsam.StereoCamera(pose, calibration).backproject(
                gtsam.StereoPoint2(*measurements[row].tolist())
            ),
            dtype=float,
        ).reshape(3)
        other = camera_poses[-1 - int(frame_offsets[row])]
        rotation, translation = _pose_arrays((other,))
        if not (
            np.array_equal(expected, points[row])
            and np.array_equal(
                np.asarray(other.transformTo(expected), dtype=float).reshape(3),
                _transform_to(rotation, translation, points[row:row + 1])[0],
            )
        ):
            raise ContractError("GTSAM Pose3 arithmetic differs from the batched kernel")


def _initialize_landmarks(
    observations: _WindowObservations,
    eligible: np.ndarray,
    camera_poses: Sequence[gtsam.Pose3],
    start_frame_id: int,
    calibration: gtsam.Cal3_S2Stereo,
    maximum_depth: Optional[float],
):
    """Per eligible track: rejection reason ("" if accepted), point, frame, depth.

    For each track, the latest observation whose backprojection is finite and
    in front of every camera that observes the track initializes the landmark.
    """

    counts = observations.counts[eligible]
    rows = _segment_rows(observations.starts[eligible], counts)
    track_count, row_count = len(counts), len(rows)
    segment = np.repeat(np.arange(track_count), counts)
    first_row = np.cumsum(counts) - counts
    frame_offsets = observations.frame_ids[rows] - start_frame_id
    measurements = observations.measurements[rows]
    rotations, translations = _pose_arrays(camera_poses)
    rotations, translations = rotations[frame_offsets], translations[frame_offsets]
    points = _transform_from(rotations, translations, _backproject(measurements, calibration))
    _check_gtsam_arithmetic(camera_poses, calibration, frame_offsets, measurements, points)
    finite_point = np.isfinite(points).all(axis=1)
    depths = np.full(row_count, np.nan)
    depths[finite_point] = _transform_to(
        rotations[finite_point], translations[finite_point], points[finite_point]
    )[:, 2]

    reasons = np.full(track_count, "", dtype=object)
    if maximum_depth is not None:
        # Reject the whole landmark if any observation is too far: near-zero
        # disparity leaves depth unconstrained and makes Marginals singular.
        limit = maximum_depth - 4.0 * np.spacing(maximum_depth)
        with np.errstate(invalid="ignore"):
            too_far = np.isfinite(depths) & (depths > 0.0) & (depths >= limit)
        reasons[np.unique(segment[too_far])] = EXCESSIVE_INITIALIZATION_DEPTH

    # Every (candidate row, observing row) pair of the same track.
    pair_counts = counts[segment]
    candidate = np.repeat(np.arange(row_count), pair_counts)
    observer = np.repeat(first_row[segment], pair_counts) + (
        np.arange(len(candidate)) - np.repeat(np.cumsum(pair_counts) - pair_counts, pair_counts)
    )
    with np.errstate(invalid="ignore"):
        in_camera = _transform_to(
            rotations[observer], translations[observer], points[candidate]
        )
        in_front = np.isfinite(in_camera).all(axis=1) & (in_camera[:, 2] > 0.0)
    usable = finite_point & np.logical_and.reduceat(
        in_front, np.cumsum(pair_counts) - pair_counts
    )
    chosen = np.full(track_count, -1, dtype=np.int64)
    usable_rows = np.flatnonzero(usable)
    np.maximum.at(chosen, segment[usable_rows], usable_rows)
    any_finite = np.zeros(track_count, dtype=bool)
    any_finite[segment[finite_point]] = True
    undecided = reasons == ""
    reasons[undecided & (chosen < 0) & any_finite] = NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION
    reasons[undecided & ~any_finite] = NO_FINITE_BACKPROJECTION
    safe = np.maximum(chosen, 0)
    return reasons, points[safe], observations.frame_ids[rows][safe], depths[safe]


def _connectivity(
    frame_ids: Tuple[int, ...],
    track_ids: np.ndarray,
    counts: np.ndarray,
    frame_offsets: np.ndarray,
):
    """Factor degrees; every camera must reach the anchor through landmarks."""

    camera_counts = np.bincount(frame_offsets, minlength=len(frame_ids))
    parent = list(range(len(frame_ids)))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    # Consecutive observations of one landmark link their two cameras.
    same_track = np.repeat(np.arange(len(counts)), counts)
    linked = same_track[1:] == same_track[:-1]
    edges = np.unique(
        np.stack((frame_offsets[:-1][linked], frame_offsets[1:][linked]), axis=1), axis=0
    )
    for first, second in edges.tolist():
        parent[root(first)] = root(second)
    anchor = root(0)
    if any(root(node) != anchor for node in range(len(frame_ids))):
        raise ContractError("Bundle camera-landmark graph is disconnected")
    return (
        dict(zip(frame_ids, camera_counts.tolist())),
        dict(zip(track_ids.tolist(), counts.tolist())),
    )


def _projection_evidence(
    graph: gtsam.NonlinearFactorGraph,
    values: gtsam.Values,
    graph_error: float,
    camera_keys: Sequence[int],
    frame_offsets: np.ndarray,
    landmark_rows: np.ndarray,
    landmark_count: int,
    measurements: np.ndarray,
    calibration: gtsam.Cal3_S2Stereo,
    inverse_sigmas: np.ndarray,
):
    """Per-factor stereo residuals and errors, batched; checked against GTSAM."""

    rotations, translations = _pose_arrays([values.atPose3(key) for key in camera_keys])
    landmarks = np.asarray(gtsam.utilities.extractPoint3(values), dtype=float).reshape(-1, 3)
    if len(landmarks) != landmark_count:
        raise ContractError("Bundle Values landmark population differs")
    in_camera = _transform_to(
        rotations[frame_offsets], translations[frame_offsets], landmarks[landmark_rows]
    )
    # GenericStereoFactor3D turns cheirality into a finite 2*fx penalty.
    cheirality = in_camera[:, 2] <= 0.0
    fx, fy = calibration.fx(), calibration.fy()
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse_depth = 1.0 / in_camera[:, 2]
        u_left = calibration.px() + fx * inverse_depth * in_camera[:, 0]
        u_right = calibration.px() + fx * inverse_depth * (in_camera[:, 0] - calibration.baseline())
        v_mean = calibration.py() + fy * inverse_depth * in_camera[:, 1]
    residuals = np.stack((u_left, u_right, v_mean), axis=1) - measurements
    residuals[cheirality] = 2.0 * fx
    errors = 0.5 * np.sum((residuals * inverse_sigmas) ** 2, axis=1)
    if not (np.isfinite(residuals).all() and np.isfinite(errors).all()):
        raise ContractError("Bundle projection evidence is nonfinite")

    total = float(graph.at(0).error(values)) + math.fsum(errors.tolist())
    tolerance = EVIDENCE_AGREEMENT_TOLERANCE
    if abs(total - graph_error) > tolerance * max(1.0, abs(graph_error)):
        raise ContractError("direct and GTSAM projection errors differ")
    for row in np.linspace(0, len(errors) - 1, min(len(errors), _EVIDENCE_SPOT_CHECKS)).astype(int):
        factor = graph.at(int(row) + 1)
        expected = np.asarray(factor.unwhitenedError(values), dtype=float).reshape(3)
        expected_error = float(factor.error(values))
        if np.max(np.abs(expected - residuals[row])) > tolerance or abs(
            expected_error - errors[row]
        ) > tolerance * max(1.0, abs(expected_error)):
            raise ContractError("direct and GTSAM projection residuals differ")
    return errors, residuals, int(np.count_nonzero(cheirality))


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
        raise ContractError("Bundle poses and tracking store frames differ")
    construction_started = perf_counter()
    frame_ids = tuple(range(start, end + 1))
    camera_keys = {
        frame_id: int(gtsam.symbol(CAMERA_SYMBOL, frame_id)) for frame_id in frame_ids
    }
    camera_poses = []
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
        camera_poses.append(pose)
        initial.insert(camera_keys[frame_id], pose)
    graph = gtsam.NonlinearFactorGraph()
    graph.add(gtsam.PriorFactorPose3(camera_keys[start], gtsam.Pose3(), anchor_noise))

    invalid_reasons = Counter()
    observations = _window_observations(database, frame_ids, invalid_reasons)
    candidate_tracks = observations.track_ids
    reasons = np.full(len(candidate_tracks), INSUFFICIENT_VALID_OBSERVATIONS, dtype=object)
    eligible = (
        observations.counts >= landmark_eligibility.minimum_valid_observations
    ) & (
        _distinct_frame_counts(observations) >= landmark_eligibility.minimum_distinct_frames
    )
    if np.any(eligible):
        eligible_reasons, points, initialization_frames, depths = _initialize_landmarks(
            observations,
            eligible,
            camera_poses,
            start,
            calibration,
            landmark_eligibility.maximum_initialization_depth_m,
        )
        reasons[eligible] = eligible_reasons
        accepted_in_eligible = eligible_reasons == ""
    else:
        accepted_in_eligible = np.zeros(0, dtype=bool)
    accepted = reasons == ""
    if not np.any(accepted):
        raise ContractError("Bundle window contains no eligible landmarks")
    accepted_tracks = candidate_tracks[accepted]
    points = points[accepted_in_eligible]
    rejected = {
        track_id: reason
        for track_id, reason in zip(candidate_tracks.tolist(), reasons.tolist())
        if reason
    }

    accepted_counts = observations.counts[accepted]
    rows = _segment_rows(observations.starts[accepted], accepted_counts)
    factor_frames = observations.frame_ids[rows]
    factor_tracks = np.repeat(accepted_tracks, accepted_counts)
    factor_measurements = observations.measurements[rows]
    frame_offsets = factor_frames - start
    landmark_rows = np.repeat(np.arange(len(accepted_tracks)), accepted_counts)
    landmark_keys = [int(gtsam.symbol(LANDMARK_SYMBOL, track_id)) for track_id in accepted_tracks.tolist()]
    for key, point in zip(landmark_keys, points):
        initial.insert(key, point)
    camera_key_list = [camera_keys[frame_id] for frame_id in frame_ids]
    add = graph.add
    for measurement, frame_offset, landmark_row in zip(
        factor_measurements.tolist(), frame_offsets.tolist(), landmark_rows.tolist()
    ):
        add(
            gtsam.GenericStereoFactor3D(
                gtsam.StereoPoint2(*measurement), projection_noise,
                camera_key_list[frame_offset], landmark_keys[landmark_row], calibration,
            )
        )
    factor_count = len(rows)
    if graph.size() != factor_count + 1:
        raise ContractError("Bundle graph factor population differs")
    camera_degrees, landmark_degrees = _connectivity(
        frame_ids, accepted_tracks, accepted_counts, frame_offsets
    )
    if graph.at(0).dim() + 3 * factor_count < initial.dim():
        raise ContractError("Bundle graph is dimensionally underconstrained")
    inverse_sigmas = np.asarray(projection_noise.invsigmas(), dtype=float).reshape(3)

    def evidence(values: gtsam.Values, graph_error: float):
        return _projection_evidence(
            graph, values, graph_error, camera_key_list, frame_offsets, landmark_rows,
            len(accepted_tracks), factor_measurements, calibration, inverse_sigmas,
        )

    initial_error = float(graph.error(initial))
    (
        initial_projection_errors,
        initial_residuals,
        initial_cheirality_factor_count,
    ) = evidence(initial, initial_error)
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
    ) = evidence(final, final_error)
    accepted_list = accepted_tracks.tolist()
    return BundleWindowResult(
        frame_ids=frame_ids,
        camera_key_by_frame_id=camera_keys,
        candidate_track_ids=tuple(candidate_tracks.tolist()),
        accepted_track_ids=tuple(accepted_list),
        rejected_reason_by_track_id=rejected,
        initialization_frame_by_track_id=dict(
            zip(accepted_list, initialization_frames[accepted_in_eligible].tolist())
        ),
        initialization_depth_by_track_id=dict(
            zip(accepted_list, depths[accepted_in_eligible].tolist())
        ),
        landmark_key_by_track_id=dict(zip(accepted_list, landmark_keys)),
        projection_factor_indices=np.arange(1, factor_count + 1, dtype=np.int64),
        projection_frame_ids=factor_frames,
        projection_track_ids=factor_tracks,
        projection_measurements=factor_measurements,
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
    "solve_bundle_window",
]
