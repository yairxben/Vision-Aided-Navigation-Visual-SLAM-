"""Rectified stereo composition, triangulation, and projection primitives."""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError


PROJECTION_EPSILON = 1e-10


def compose_right_camera(
    left_rotation: np.ndarray,
    left_translation: np.ndarray,
    fixed_right_rotation: np.ndarray,
    fixed_right_translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    left_rotation = np.asarray(left_rotation, dtype=float)
    fixed_right_rotation = np.asarray(fixed_right_rotation, dtype=float)
    left_translation = np.asarray(left_translation, dtype=float).reshape(3, 1)
    fixed_right_translation = np.asarray(
        fixed_right_translation, dtype=float
    ).reshape(3, 1)
    if left_rotation.shape != (3, 3) or fixed_right_rotation.shape != (3, 3):
        raise ContractError("camera rotations must be 3x3")
    return (
        fixed_right_rotation @ left_rotation,
        fixed_right_rotation @ left_translation + fixed_right_translation,
    )


def triangulate_stereo_matches(
    left_keypoints: Sequence[cv2.KeyPoint],
    right_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    intrinsic: np.ndarray,
    left_rotation: np.ndarray,
    left_translation: np.ndarray,
    right_rotation: np.ndarray,
    right_translation: np.ndarray,
) -> np.ndarray:
    """Triangulate in the coordinate frame encoded by the two extrinsics."""

    if not matches:
        return np.empty((0, 3), dtype=float)
    points_left = np.asarray(
        [left_keypoints[match.queryIdx].pt for match in matches], dtype=float
    )
    points_right = np.asarray(
        [right_keypoints[match.trainIdx].pt for match in matches], dtype=float
    )
    projection_left = np.asarray(intrinsic, dtype=float) @ np.hstack(
        (
            np.asarray(left_rotation, dtype=float),
            np.asarray(left_translation, dtype=float).reshape(3, 1),
        )
    )
    projection_right = np.asarray(intrinsic, dtype=float) @ np.hstack(
        (
            np.asarray(right_rotation, dtype=float),
            np.asarray(right_translation, dtype=float).reshape(3, 1),
        )
    )
    homogeneous = cv2.triangulatePoints(
        projection_left,
        projection_right,
        points_left.T,
        points_right.T,
    )
    homogeneous /= homogeneous[3] + PROJECTION_EPSILON
    return homogeneous[:3].T


def project_points(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
) -> np.ndarray:
    """Project a common 3-D population into each supplied camera."""

    points = np.asarray(points_3d, dtype=float)
    if points.ndim != 2 or 3 not in points.shape:
        raise ContractError("3-D points must have shape Nx3 or 3xN")
    if points.shape[0] != 3:
        points = points.T
    if len(rotations) != len(translations):
        raise ContractError("rotation and translation populations differ")
    if not rotations:
        return np.zeros((0, 2, points.shape[1]), dtype=float)
    # One stacked product per step instead of a Python loop over cameras.
    stacked_rotations = np.stack([np.asarray(rotation, dtype=float) for rotation in rotations])
    stacked_translations = np.stack(
        [np.asarray(translation, dtype=float).reshape(3, 1) for translation in translations]
    )
    projection = np.matmul(
        np.asarray(intrinsic, dtype=float),
        np.matmul(stacked_rotations, points) + stacked_translations,
    )
    return (projection / (projection[:, 2:3] + PROJECTION_EPSILON))[:, :2]
