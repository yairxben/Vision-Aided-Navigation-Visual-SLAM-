"""Pure coordinate-contract helpers for Exercise 5.

The canonical Exercise 4 poses are world-to-camera extrinsics
``E_i = T_{C_i<-W}``.  GTSAM camera poses use the inverse direction:
``P_i = T_{W<-C_i}``.  A Bundle window beginning at frame ``a`` uses camera
``a`` as its parent frame, so ``P_i^(a) = T_{C_a<-C_i}``.

All helpers are deterministic and side-effect free.  Rotation tolerances are
required from callers deliberately: E5-I01 validates behavior but does not
select a production numerical tolerance.
"""

from __future__ import annotations

from typing import Sequence

import gtsam
import numpy as np


__all__ = [
    "validate_world_to_camera_extrinsic",
    "camera_pose_from_world_to_camera",
    "world_to_camera_from_camera_pose",
    "camera_pose_in_window",
    "world_point_to_camera",
    "camera_point_to_world",
    "camera_point_to_window",
    "window_point_to_camera",
]


def _validated_rotation_atol(rotation_atol: float) -> float:
    """Return a finite, positive caller-supplied rotation tolerance."""

    tolerance = float(rotation_atol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rotation_atol must be finite and positive")
    return tolerance


def _validated_point(point: Sequence[float], name: str) -> np.ndarray:
    """Return a finite three-vector copy."""

    vector = np.asarray(point, dtype=float)
    if vector.shape != (3,):
        raise ValueError("{} must have shape (3,)".format(name))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} must contain only finite values".format(name))
    return vector.copy()


def validate_world_to_camera_extrinsic(
    extrinsic: Sequence[Sequence[float]],
    *,
    rotation_atol: float,
) -> np.ndarray:
    """Validate and copy a 3x4 ``T_camera<-world`` rigid transform.

    The rotation must be finite, orthonormal, and proper (determinant +1)
    within the explicitly supplied absolute tolerance.  Translation must be
    finite.  A float64 copy is returned; invalid inputs raise ``ValueError``.
    """

    tolerance = _validated_rotation_atol(rotation_atol)
    matrix = np.asarray(extrinsic, dtype=float)
    if matrix.shape != (3, 4):
        raise ValueError("world-to-camera extrinsic must have shape (3, 4)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "world-to-camera extrinsic must contain only finite values"
        )

    rotation = matrix[:, :3]
    orthogonality_error = float(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
    )
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if orthogonality_error > tolerance:
        raise ValueError(
            "world-to-camera rotation is not orthonormal: "
            "maximum absolute error {}".format(orthogonality_error)
        )
    if determinant_error > tolerance:
        raise ValueError(
            "world-to-camera rotation is not proper: "
            "absolute determinant error {}".format(determinant_error)
        )
    return matrix.copy()


def camera_pose_from_world_to_camera(
    extrinsic: Sequence[Sequence[float]],
    *,
    rotation_atol: float,
) -> gtsam.Pose3:
    """Convert ``E_i = T_{C_i<-W}`` to ``P_i = T_{W<-C_i}``.

    The result is constructed from an explicit ``Rot3`` and translation after
    validation.  This avoids relying on GTSAM's permissive raw-matrix
    ``Pose3`` constructor.
    """

    world_to_camera = validate_world_to_camera_extrinsic(
        extrinsic, rotation_atol=rotation_atol
    )
    rotation = world_to_camera[:, :3]
    translation = world_to_camera[:, 3]
    camera_to_world_rotation = rotation.T
    camera_center_in_world = -camera_to_world_rotation @ translation
    return gtsam.Pose3(
        gtsam.Rot3(camera_to_world_rotation),
        camera_center_in_world,
    )


def world_to_camera_from_camera_pose(
    camera_pose: gtsam.Pose3,
    *,
    rotation_atol: float,
) -> np.ndarray:
    """Convert ``P_i = T_{W<-C_i}`` to the corresponding 3x4 extrinsic.

    More generally, this returns ``T_{C_i<-parent}`` for a camera-to-parent
    ``Pose3``.  The result is validated because GTSAM 4.2 does not itself
    reject every malformed rotation supplied through its matrix constructor.
    """

    camera_to_parent_rotation = np.asarray(
        camera_pose.rotation().matrix(), dtype=float
    )
    camera_origin_in_parent = np.asarray(
        camera_pose.translation(), dtype=float
    ).reshape(3)
    parent_to_camera_rotation = camera_to_parent_rotation.T
    parent_to_camera_translation = (
        -parent_to_camera_rotation @ camera_origin_in_parent
    )
    extrinsic = np.column_stack(
        (parent_to_camera_rotation, parent_to_camera_translation)
    )
    return validate_world_to_camera_extrinsic(
        extrinsic, rotation_atol=rotation_atol
    )


def camera_pose_in_window(
    window_origin_world_to_camera: Sequence[Sequence[float]],
    camera_world_to_camera: Sequence[Sequence[float]],
    *,
    rotation_atol: float,
) -> gtsam.Pose3:
    """Return camera ``i`` in a window whose parent frame is camera ``a``.

    The returned direction is
    ``P_i^(a) = T_{C_a<-C_i} = E_a E_i^-1``.  It maps camera-``i`` points into
    window-local (camera-``a``) coordinates.  Supplying ``E_a`` for both
    arguments returns the identity pose.
    """

    window_origin_pose = camera_pose_from_world_to_camera(
        window_origin_world_to_camera,
        rotation_atol=rotation_atol,
    )
    camera_pose = camera_pose_from_world_to_camera(
        camera_world_to_camera,
        rotation_atol=rotation_atol,
    )
    return window_origin_pose.between(camera_pose)


def world_point_to_camera(
    extrinsic: Sequence[Sequence[float]],
    world_point: Sequence[float],
    *,
    rotation_atol: float,
) -> np.ndarray:
    """Map a world point with ``x_C = R_CW X_W + t_CW``.

    Passing the window-origin extrinsic ``E_a`` maps a world point directly
    into window-local coordinates because the window frame is ``C_a``.
    """

    world_to_camera = validate_world_to_camera_extrinsic(
        extrinsic, rotation_atol=rotation_atol
    )
    point = _validated_point(world_point, "world_point")
    return world_to_camera[:, :3] @ point + world_to_camera[:, 3]


def camera_point_to_world(
    extrinsic: Sequence[Sequence[float]],
    camera_point: Sequence[float],
    *,
    rotation_atol: float,
) -> np.ndarray:
    """Map a camera point with ``X_W = R_CW.T (x_C - t_CW)``.

    Passing the window-origin extrinsic ``E_a`` maps a window-local point back
    into world coordinates.
    """

    world_to_camera = validate_world_to_camera_extrinsic(
        extrinsic, rotation_atol=rotation_atol
    )
    point = _validated_point(camera_point, "camera_point")
    rotation = world_to_camera[:, :3]
    translation = world_to_camera[:, 3]
    return rotation.T @ (point - translation)


def camera_point_to_window(
    window_camera_pose: gtsam.Pose3,
    camera_point: Sequence[float],
    *,
    rotation_atol: float,
) -> np.ndarray:
    """Map a camera-``i`` point into camera-``a`` window coordinates.

    ``window_camera_pose`` must be ``T_{C_a<-C_i}``, as returned by
    :func:`camera_pose_in_window`.  This is GTSAM ``Pose3.transformFrom``.
    """

    world_to_camera_from_camera_pose(
        window_camera_pose, rotation_atol=rotation_atol
    )
    point = _validated_point(camera_point, "camera_point")
    return np.asarray(
        window_camera_pose.transformFrom(point), dtype=float
    ).reshape(3)


def window_point_to_camera(
    window_camera_pose: gtsam.Pose3,
    window_point: Sequence[float],
    *,
    rotation_atol: float,
) -> np.ndarray:
    """Map a camera-``a`` window point into camera-``i`` coordinates.

    ``window_camera_pose`` must be ``T_{C_a<-C_i}``, as returned by
    :func:`camera_pose_in_window`.  This is GTSAM ``Pose3.transformTo``.
    """

    world_to_camera_from_camera_pose(
        window_camera_pose, rotation_atol=rotation_atol
    )
    point = _validated_point(window_point, "window_point")
    return np.asarray(
        window_camera_pose.transformTo(point), dtype=float
    ).reshape(3)
