"""Absolute and relative trajectory evaluation with optional Ground Truth."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError


def normalize_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
    poses = np.asarray(camera_to_world, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.all(np.isfinite(poses)):
        raise ContractError("trajectory must contain finite 4x4 poses")
    origin_inverse = np.linalg.inv(poses[0])
    return np.asarray([origin_inverse @ pose for pose in poses], dtype=float)


def camera_to_world_from_extrinsics(world_to_camera: np.ndarray) -> np.ndarray:
    return normalize_camera_to_world(np.linalg.inv(np.asarray(world_to_camera, dtype=float)))


def rotation_magnitude_degrees(rotation: np.ndarray) -> float:
    vector = cv2.Rodrigues(np.asarray(rotation, dtype=float))[0]
    return float(np.degrees(np.linalg.norm(vector)))


def pose_error(estimate: np.ndarray, truth: np.ndarray) -> Tuple[float, float, np.ndarray]:
    estimated = np.asarray(estimate, dtype=float)
    expected = np.asarray(truth, dtype=float)
    error = np.linalg.inv(expected) @ estimated
    coordinate_delta = np.asarray(estimated[:3, 3] - expected[:3, 3], dtype=float)
    return (
        float(np.linalg.norm(coordinate_delta)),
        rotation_magnitude_degrees(error[:3, :3]),
        coordinate_delta,
    )


def absolute_error_rows(
    frame_ids: Sequence[int], estimate: np.ndarray, truth: np.ndarray, stage: str
):
    rows = []
    for frame_id, estimated_pose, truth_pose in zip(frame_ids, estimate, truth):
        translation, rotation, components = pose_error(estimated_pose, truth_pose)
        rows.append(
            {
                "stage": stage,
                "frame_id": int(frame_id),
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "x_error_m": float(abs(components[0])),
                "y_error_m": float(abs(components[1])),
                "z_error_m": float(abs(components[2])),
                # Signed estimate - truth, in the first-camera world frame.
                "x_signed_error_m": float(components[0]),
                "y_signed_error_m": float(components[1]),
                "z_signed_error_m": float(components[2]),
            }
        )
    return rows


def relative_error_rows(
    frame_ids: Sequence[int], estimate: np.ndarray, truth: np.ndarray, stage: str
):
    rows = []
    for ordinal in range(len(frame_ids) - 1):
        estimated_relative = np.linalg.inv(estimate[ordinal]) @ estimate[ordinal + 1]
        truth_relative = np.linalg.inv(truth[ordinal]) @ truth[ordinal + 1]
        translation, rotation, _ = pose_error(estimated_relative, truth_relative)
        rows.append(
            {
                "stage": stage,
                "start_frame_id": int(frame_ids[ordinal]),
                "end_frame_id": int(frame_ids[ordinal + 1]),
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
            }
        )
    return rows


def subsection_error_rows(
    frame_ids: Sequence[int],
    estimate: np.ndarray,
    truth: np.ndarray,
    stage: str,
    requested_length: int,
):
    ids = tuple(int(value) for value in frame_ids)
    rows = []
    for start_ordinal, start in enumerate(ids):
        target = start + requested_length
        if target > ids[-1]:
            continue
        candidates = [
            (abs(frame_id - target), frame_id, ordinal)
            for ordinal, frame_id in enumerate(ids)
            if ordinal > start_ordinal
        ]
        if not candidates:
            continue
        _, end, end_ordinal = min(candidates)
        estimated_relative = np.linalg.inv(estimate[start_ordinal]) @ estimate[end_ordinal]
        truth_relative = np.linalg.inv(truth[start_ordinal]) @ truth[end_ordinal]
        translation, rotation, _ = pose_error(estimated_relative, truth_relative)
        truth_distance = float(np.linalg.norm(truth[end_ordinal, :3, 3] - truth[start_ordinal, :3, 3]))
        rows.append(
            {
                "stage": stage,
                "requested_length_frames": int(requested_length),
                "start_frame_id": start,
                "end_frame_id": end,
                "actual_frame_delta": end - start,
                "truth_endpoint_distance_m": truth_distance,
                "translation_error_m": translation,
                "rotation_error_deg": rotation,
                "translation_error_percent_of_truth_endpoint_distance": (
                    100.0 * translation / truth_distance if truth_distance > 0.0 else float("nan")
                ),
                "rotation_error_deg_per_m": (
                    rotation / truth_distance if truth_distance > 0.0 else float("nan")
                ),
            }
        )
    return rows


__all__ = [
    "absolute_error_rows", "camera_to_world_from_extrinsics",
    "normalize_camera_to_world", "pose_error", "relative_error_rows",
    "rotation_magnitude_degrees", "subsection_error_rows",
]
