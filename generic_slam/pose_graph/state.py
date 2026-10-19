"""In-memory graph/value state shared with sequential Loop Closure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import gtsam
import numpy as np

from ..contracts import ContractError


def camera_symbol(frame_id: int) -> int:
    return int(gtsam.symbol("c", int(frame_id)))


@dataclass
class PoseGraphState:
    keyframe_ids: Tuple[int, ...]
    camera_keys: Tuple[int, ...]
    graph: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    marginals: gtsam.Marginals

    def pose_matrices(self) -> np.ndarray:
        return np.asarray(
            [self.values.atPose3(key).matrix() for key in self.camera_keys], dtype=float
        )

    def marginal_covariances(self) -> np.ndarray:
        return np.asarray(
            [self.marginals.marginalCovariance(key) for key in self.camera_keys],
            dtype=float,
        )


def build_factor_graph(
    keyframe_ids: Sequence[int],
    relative_pose_matrices: np.ndarray,
    covariance_matrices: np.ndarray,
    prior_sigmas: np.ndarray,
) -> Tuple[gtsam.NonlinearFactorGraph, gtsam.Values, Tuple[int, ...]]:
    frame_ids = tuple(int(value) for value in keyframe_ids)
    edges = np.asarray(relative_pose_matrices, dtype=float)
    covariances = np.asarray(covariance_matrices, dtype=float)
    if (
        not frame_ids
        or any(b <= a for a, b in zip(frame_ids, frame_ids[1:]))
        or edges.shape != (len(frame_ids) - 1, 4, 4)
        or covariances.shape != (len(frame_ids) - 1, 6, 6)
    ):
        raise ContractError("consecutive Pose Graph populations differ")
    sigmas = np.asarray(prior_sigmas, dtype=float)
    if sigmas.shape != (6,) or not np.all(np.isfinite(sigmas)):
        raise ContractError("Pose Graph prior sigmas are invalid")
    keys = tuple(camera_symbol(frame_id) for frame_id in frame_ids)
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()
    identity = gtsam.Pose3()
    graph.add(
        gtsam.PriorFactorPose3(
            keys[0], identity, gtsam.noiseModel.Diagonal.Sigmas(sigmas)
        )
    )
    values.insert(keys[0], identity)
    current = identity
    for ordinal, (matrix, covariance) in enumerate(zip(edges, covariances)):
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(covariance)):
            raise ContractError("Pose Graph edge evidence is nonfinite")
        relative = gtsam.Pose3(matrix)
        noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
        graph.add(gtsam.BetweenFactorPose3(keys[ordinal], keys[ordinal + 1], relative, noise))
        current = current.compose(relative)
        values.insert(keys[ordinal + 1], current)
    return graph, values, keys


def restore_pose_graph_state(
    keyframe_ids: Sequence[int],
    relative_pose_matrices: np.ndarray,
    covariance_matrices: np.ndarray,
    prior_sigmas: np.ndarray,
    value_matrices: np.ndarray,
) -> PoseGraphState:
    graph, _, keys = build_factor_graph(
        keyframe_ids, relative_pose_matrices, covariance_matrices, prior_sigmas
    )
    matrices = np.asarray(value_matrices, dtype=float)
    if matrices.shape != (len(keys), 4, 4):
        raise ContractError("restored Pose Graph Values shape differs")
    values = gtsam.Values()
    for key, matrix in zip(keys, matrices):
        values.insert(key, gtsam.Pose3(matrix))
    marginals = gtsam.Marginals(graph, values)
    return PoseGraphState(tuple(int(v) for v in keyframe_ids), keys, graph, values, marginals)


__all__ = ["PoseGraphState", "build_factor_graph", "camera_symbol", "restore_pose_graph_state"]
