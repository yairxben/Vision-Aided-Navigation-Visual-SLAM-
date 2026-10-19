"""Validated consecutive Pose3 chain construction and optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import gtsam
import numpy as np

from ..contracts import ContractError
from .state import PoseGraphState, build_factor_graph


@dataclass(frozen=True)
class ConsecutivePoseGraphResult:
    state: PoseGraphState
    initial_pose_matrices: np.ndarray
    optimized_pose_matrices: np.ndarray
    marginal_covariances: np.ndarray
    initial_graph_error: float
    final_graph_error: float
    initial_edge_reconstruction_max_abs: float
    maximum_translation_movement_m: float
    maximum_rotation_movement_rad: float
    optimizer_iterations: int


def solve_consecutive_pose_graph(
    *,
    keyframe_ids: Sequence[int],
    relative_pose_matrices: np.ndarray,
    covariance_matrices: np.ndarray,
    prior_sigmas: np.ndarray,
    reconstruction_atol: float,
    graph_error_nonincrease_absolute: float,
    graph_error_nonincrease_relative: float,
) -> ConsecutivePoseGraphResult:
    graph, initial, keys = build_factor_graph(
        keyframe_ids, relative_pose_matrices, covariance_matrices, prior_sigmas
    )
    edges = np.asarray(relative_pose_matrices, dtype=float)
    reconstruction = []
    for ordinal, edge in enumerate(edges):
        observed = initial.atPose3(keys[ordinal]).between(
            initial.atPose3(keys[ordinal + 1])
        )
        reconstruction.append(
            float(np.max(np.abs(np.asarray(observed.matrix()) - edge)))
        )
    maximum_reconstruction = max(reconstruction) if reconstruction else 0.0
    if maximum_reconstruction > reconstruction_atol:
        raise ContractError("initial Pose Graph trajectory does not reconstruct edges")
    initial_error = float(graph.error(initial))
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    optimized = optimizer.optimize()
    final_error = float(graph.error(optimized))
    allowed = max(
        graph_error_nonincrease_absolute,
        graph_error_nonincrease_relative * abs(initial_error),
    )
    if final_error > initial_error + allowed:
        raise ContractError("consecutive Pose Graph optimization increased error")

    translation_movement = []
    rotation_movement = []
    for key in keys:
        first = initial.atPose3(key)
        second = optimized.atPose3(key)
        translation_movement.append(
            float(np.linalg.norm(np.asarray(second.translation()) - np.asarray(first.translation())))
        )
        rotation_movement.append(
            float(
                np.linalg.norm(
                    np.asarray(
                        gtsam.Rot3.Logmap(first.rotation().between(second.rotation())),
                        dtype=float,
                    )
                )
            )
        )
    marginals = gtsam.Marginals(graph, optimized)
    state = PoseGraphState(
        tuple(int(value) for value in keyframe_ids), keys, graph, optimized, marginals
    )
    initial_matrices = np.asarray([initial.atPose3(key).matrix() for key in keys], dtype=float)
    optimized_matrices = state.pose_matrices()
    covariance_array = state.marginal_covariances()
    return ConsecutivePoseGraphResult(
        state=state,
        initial_pose_matrices=initial_matrices,
        optimized_pose_matrices=optimized_matrices,
        marginal_covariances=covariance_array,
        initial_graph_error=initial_error,
        final_graph_error=final_error,
        initial_edge_reconstruction_max_abs=maximum_reconstruction,
        maximum_translation_movement_m=max(translation_movement),
        maximum_rotation_movement_rad=max(rotation_movement),
        optimizer_iterations=int(optimizer.iterations()),
    )


__all__ = ["ConsecutivePoseGraphResult", "solve_consecutive_pose_graph"]
