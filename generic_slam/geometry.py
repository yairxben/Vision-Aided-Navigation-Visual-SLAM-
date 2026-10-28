"""Validated pose and point conventions shared by generic SLAM stages."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import ContractError


TANGENT_ORDER = (
    "rot_x_rad",
    "rot_y_rad",
    "rot_z_rad",
    "trans_x_m",
    "trans_y_m",
    "trans_z_m",
)


def homogeneous(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape == (4, 4):
        result = value.copy()
    elif value.shape == (3, 4):
        result = np.vstack((value, np.array([0.0, 0.0, 0.0, 1.0])))
    else:
        raise ContractError("pose must have shape (3,4) or (4,4)")
    if not np.all(np.isfinite(result)):
        raise ContractError("pose contains nonfinite values")
    return result


_IDENTITY_3 = np.eye(3)
_HOMOGENEOUS_ROW = np.array([0.0, 0.0, 0.0, 1.0])


def validate_rigid_pose(matrix: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    value = homogeneous(matrix)
    rotation = value[:3, :3]
    # homogeneous() guarantees finite values, for which allclose/isclose with
    # rtol=0 are exactly |a - b| <= atol (this runs ~10^5 times per stage).
    if not (np.abs(rotation.T @ rotation - _IDENTITY_3) <= atol).all():
        raise ContractError("pose rotation is not orthonormal")
    if not abs(np.linalg.det(rotation) - 1.0) <= atol:
        raise ContractError("pose rotation determinant is not one")
    if not (np.abs(value[3] - _HOMOGENEOUS_ROW) <= atol).all():
        raise ContractError("pose homogeneous row is invalid")
    return value


def camera_pose_from_world_to_camera(extrinsic: np.ndarray) -> np.ndarray:
    """Return ``T_W_from_C`` from ``E=T_C_from_W``."""

    return np.linalg.inv(validate_rigid_pose(extrinsic))


def world_to_camera_from_camera_pose(camera_pose: np.ndarray) -> np.ndarray:
    """Return ``E=T_C_from_W`` from ``T_W_from_C``."""

    return np.linalg.inv(validate_rigid_pose(camera_pose))


def camera_pose_in_window(
    window_start_extrinsic: np.ndarray, camera_extrinsic: np.ndarray
) -> np.ndarray:
    """Return ``T_Cstart_from_Camera = E_start @ inverse(E_camera)``."""

    start = validate_rigid_pose(window_start_extrinsic)
    camera = validate_rigid_pose(camera_extrinsic)
    return validate_rigid_pose(start @ np.linalg.inv(camera))


def camera_center(extrinsic: np.ndarray) -> np.ndarray:
    pose = validate_rigid_pose(extrinsic)
    return -(pose[:3, :3].T @ pose[:3, 3])


def transform_point(transform: np.ndarray, point: Sequence[float]) -> np.ndarray:
    pose = validate_rigid_pose(transform)
    value = np.asarray(point, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ContractError("point must be one finite 3-vector")
    return pose[:3, :3] @ value + pose[:3, 3]


def relative_rotation_degrees(first_rotation: np.ndarray, second_rotation: np.ndarray) -> float:
    first = np.asarray(first_rotation, dtype=float)
    second = np.asarray(second_rotation, dtype=float)
    if first.shape != (3, 3) or second.shape != (3, 3):
        raise ContractError("relative rotations require two 3x3 matrices")
    cosine = np.clip((np.trace(second @ first.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


__all__ = [
    "TANGENT_ORDER",
    "camera_center",
    "camera_pose_from_world_to_camera",
    "camera_pose_in_window",
    "homogeneous",
    "relative_rotation_degrees",
    "transform_point",
    "validate_rigid_pose",
    "world_to_camera_from_camera_pose",
]

