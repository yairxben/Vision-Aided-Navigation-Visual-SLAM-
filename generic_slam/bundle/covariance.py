"""Raw ordered endpoint covariance and relative-pose extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gtsam
import numpy as np

from ..contracts import ContractError
from ..geometry import TANGENT_ORDER, validate_rigid_pose
from .window import BundleWindowResult


@dataclass(frozen=True)
class BundleCovarianceEvidence:
    endpoint_edge: np.ndarray
    joint_covariance: np.ndarray
    joint_information: np.ndarray
    conditional_end_given_start_covariance: np.ndarray
    camera_frame_ids: np.ndarray
    camera_poses: np.ndarray
    camera_marginal_covariances: np.ndarray


def _validate_covariance(
    matrix: np.ndarray, shape: Tuple[int, int], symmetry_atol: float,
    minimum_eigenvalue: float, label: str,
) -> np.ndarray:
    value = np.asarray(matrix, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ContractError("{} is invalid".format(label))
    if not np.allclose(value, value.T, atol=symmetry_atol, rtol=0.0):
        raise ContractError("{} is not symmetric".format(label))
    symmetric = 0.5 * (value + value.T)
    if float(np.min(np.linalg.eigvalsh(symmetric))) < minimum_eigenvalue:
        raise ContractError("{} has an invalid eigenvalue".format(label))
    return value


def extract_bundle_covariances(
    result: BundleWindowResult,
    *,
    symmetry_atol: float,
    minimum_eigenvalue: float,
    rotation_atol: float,
) -> BundleCovarianceEvidence:
    marginals = gtsam.Marginals(result.graph, result.final_values)
    start = result.start_frame_id
    end = result.end_frame_id
    start_key = result.camera_key_by_frame_id[start]
    end_key = result.camera_key_by_frame_id[end]
    keys = gtsam.KeyVector([start_key, end_key])
    joint = _validate_covariance(
        np.asarray(marginals.jointMarginalCovariance(keys).fullMatrix(), dtype=float),
        (12, 12), symmetry_atol, minimum_eigenvalue, "joint endpoint covariance",
    )
    information = _validate_covariance(
        np.asarray(marginals.jointMarginalInformation(keys).fullMatrix(), dtype=float),
        (12, 12), symmetry_atol, minimum_eigenvalue, "joint endpoint information",
    )
    solved = np.linalg.solve(joint[:6, :6], joint[:6, 6:])
    conditional = joint[6:, 6:] - joint[6:, :6] @ solved
    conditional = _validate_covariance(
        0.5 * (conditional + conditional.T), (6, 6), symmetry_atol,
        minimum_eigenvalue, "conditional end-given-start covariance",
    )
    # Construction is itself an executable validation of downstream semantics.
    gtsam.noiseModel.Gaussian.Covariance(conditional)
    start_pose = result.final_values.atPose3(start_key)
    end_pose = result.final_values.atPose3(end_key)
    edge = validate_rigid_pose(
        np.asarray(start_pose.between(end_pose).matrix(), dtype=float),
        atol=rotation_atol,
    )
    camera_poses = []
    camera_covariances = []
    for frame_id in result.frame_ids:
        key = result.camera_key_by_frame_id[frame_id]
        camera_poses.append(np.asarray(result.final_values.atPose3(key).matrix(), dtype=float))
        camera_covariances.append(
            _validate_covariance(
                np.asarray(marginals.marginalCovariance(key), dtype=float),
                (6, 6), symmetry_atol, minimum_eigenvalue,
                "camera marginal covariance",
            )
        )
    return BundleCovarianceEvidence(
        endpoint_edge=edge,
        joint_covariance=joint,
        joint_information=information,
        conditional_end_given_start_covariance=conditional,
        camera_frame_ids=np.asarray(result.frame_ids, dtype=np.int64),
        camera_poses=np.asarray(camera_poses, dtype=float),
        camera_marginal_covariances=np.asarray(camera_covariances, dtype=float),
    )


__all__ = ["BundleCovarianceEvidence", "extract_bundle_covariances"]
