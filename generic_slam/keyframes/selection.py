"""Motion-triggered, bounded-window keyframes derived only from current VO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np

from ..contracts import ContractError
from ..geometry import camera_center, relative_rotation_degrees, validate_rigid_pose


@dataclass(frozen=True)
class KeyframeDecision:
    ordinal: int
    frame_id: int
    previous_frame_id: int
    window_frame_count: int
    path_distance_m: float
    relative_rotation_deg: float
    reason: str


@dataclass(frozen=True)
class KeyframeSelection:
    frame_ids: Tuple[int, ...]
    decisions: Tuple[KeyframeDecision, ...]


def validate_keyframes(keyframes: Tuple[int, ...], final_frame_id: int) -> None:
    if not keyframes or keyframes[0] != 0 or keyframes[-1] != final_frame_id:
        raise ContractError("keyframes must cover first through final FrameId")
    if any(current <= previous for previous, current in zip(keyframes, keyframes[1:])):
        raise ContractError("keyframes must be strictly increasing")
    if any(
        not 5 <= current - previous + 1 <= 20
        for previous, current in zip(keyframes, keyframes[1:])
    ):
        raise ContractError("keyframe windows must contain 5 through 20 frames")


def select_keyframes(
    world_to_camera: np.ndarray, config: Mapping[str, object]
) -> KeyframeSelection:
    """Apply the approved earliest-motion-or-upper-bound policy."""

    poses = np.asarray(world_to_camera, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] not in ((3, 4), (4, 4)):
        raise ContractError("PnP poses must have shape Nx3x4 or Nx4x4")
    if poses.shape[0] < 5:
        raise ContractError("keyframe selection requires at least five frames")
    normalized = np.asarray([validate_rigid_pose(pose) for pose in poses], dtype=float)
    centers = np.asarray([camera_center(pose) for pose in normalized], dtype=float)
    transition_distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    cumulative_path = np.concatenate(([0.0], np.cumsum(transition_distances)))

    minimum_delta = int(config["minimum_endpoint_delta_frames"])
    maximum_delta = int(config["maximum_endpoint_delta_frames"])
    path_threshold = float(config["path_distance_threshold_m"])
    rotation_threshold = float(config["relative_rotation_threshold_deg"])
    final_frame_id = poses.shape[0] - 1
    keyframes = [0]
    decisions = []
    while keyframes[-1] < final_frame_id:
        start = keyframes[-1]
        remaining = final_frame_id - start
        if minimum_delta <= remaining <= maximum_delta:
            endpoint = final_frame_id
            path_distance = float(cumulative_path[endpoint] - cumulative_path[start])
            rotation = relative_rotation_degrees(
                normalized[start, :3, :3], normalized[endpoint, :3, :3]
            )
            reason = "final_tail"
        else:
            upper = min(start + maximum_delta, final_frame_id - minimum_delta)
            endpoint = upper
            reason = "upper_bound"
            path_distance = float(cumulative_path[endpoint] - cumulative_path[start])
            rotation = relative_rotation_degrees(
                normalized[start, :3, :3], normalized[endpoint, :3, :3]
            )
            for candidate in range(start + minimum_delta, upper + 1):
                candidate_path = float(
                    cumulative_path[candidate] - cumulative_path[start]
                )
                candidate_rotation = relative_rotation_degrees(
                    normalized[start, :3, :3], normalized[candidate, :3, :3]
                )
                if candidate_path >= path_threshold or candidate_rotation >= rotation_threshold:
                    endpoint = candidate
                    path_distance = candidate_path
                    rotation = candidate_rotation
                    if candidate_path >= path_threshold and candidate_rotation >= rotation_threshold:
                        reason = "path_and_rotation_threshold"
                    elif candidate_path >= path_threshold:
                        reason = "path_threshold"
                    else:
                        reason = "rotation_threshold"
                    break
        keyframes.append(endpoint)
        decisions.append(
            KeyframeDecision(
                ordinal=len(keyframes) - 1,
                frame_id=endpoint,
                previous_frame_id=start,
                window_frame_count=endpoint - start + 1,
                path_distance_m=path_distance,
                relative_rotation_deg=rotation,
                reason=reason,
            )
        )
    result = tuple(keyframes)
    validate_keyframes(result, final_frame_id)
    return KeyframeSelection(result, tuple(decisions))


__all__ = [
    "KeyframeDecision",
    "KeyframeSelection",
    "select_keyframes",
    "validate_keyframes",
]
