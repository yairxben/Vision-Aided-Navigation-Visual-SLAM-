"""Transactional cumulative Pose Graph state for sequential Loop Closure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import gtsam
import numpy as np

from ..contracts import ContractError
from ..pose_graph.stage import PoseGraphOutput
from ..pose_graph.state import build_factor_graph, camera_symbol


@dataclass(frozen=True)
class PoseGraphUpdateEvidence:
    previous_frame_id: int
    current_frame_id: int
    pre_update_state_version: int
    state_version: int
    pre_update_graph_error: float
    augmented_pre_optimization_error: float
    raw_optimizer_error: float
    normalized_post_optimization_error: float
    loop_factor_pre_optimization_error: float
    loop_factor_post_optimization_error: float
    maximum_translation_movement_m: float
    maximum_rotation_movement_rad: float
    marginal_symmetry_max_abs: float
    marginal_minimum_eigenvalue: float


class CumulativePoseGraphState:
    """Publish a loop factor only after optimization and all Marginals succeed."""

    def __init__(
        self,
        *,
        keyframe_ids: Sequence[int],
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        base_factor_count: int,
        state_version: int,
        bundle_config: Mapping[str, object],
    ) -> None:
        self.keyframe_ids = tuple(int(value) for value in keyframe_ids)
        self.camera_keys = tuple(camera_symbol(value) for value in self.keyframe_ids)
        self.graph = gtsam.NonlinearFactorGraph(graph)
        self.values = gtsam.Values(values)
        self.base_factor_count = int(base_factor_count)
        self.state_version = int(state_version)
        self.bundle_config = bundle_config
        self.marginals = gtsam.Marginals(self.graph, self.values)

    @classmethod
    def from_pose_graph(
        cls, pose_graph: PoseGraphOutput, bundle_config: Mapping[str, object]
    ) -> "CumulativePoseGraphState":
        return cls(
            keyframe_ids=pose_graph.keyframe_ids,
            graph=pose_graph.state.graph,
            values=pose_graph.state.values,
            base_factor_count=int(pose_graph.state.graph.size()),
            state_version=0,
            bundle_config=bundle_config,
        )

    @classmethod
    def restore(
        cls,
        *,
        pose_graph: PoseGraphOutput,
        accepted_previous_frame_ids: np.ndarray,
        accepted_current_frame_ids: np.ndarray,
        accepted_relative_pose_matrices: np.ndarray,
        accepted_covariances: np.ndarray,
        value_matrices: np.ndarray,
        bundle_config: Mapping[str, object],
    ) -> "CumulativePoseGraphState":
        graph, _, keys = build_factor_graph(
            pose_graph.keyframe_ids,
            pose_graph.relative_pose_matrices,
            pose_graph.edge_covariances,
            pose_graph.prior_sigmas,
        )
        for previous, current, matrix, covariance in zip(
            accepted_previous_frame_ids,
            accepted_current_frame_ids,
            accepted_relative_pose_matrices,
            accepted_covariances,
        ):
            graph.add(
                gtsam.BetweenFactorPose3(
                    camera_symbol(int(previous)), camera_symbol(int(current)),
                    gtsam.Pose3(matrix),
                    gtsam.noiseModel.Gaussian.Covariance(covariance),
                )
            )
        values = gtsam.Values()
        matrices = np.asarray(value_matrices, dtype=float)
        if matrices.shape != (len(keys), 4, 4):
            raise ContractError("Loop Closure checkpoint Values shape differs")
        for key, matrix in zip(keys, matrices):
            values.insert(key, gtsam.Pose3(matrix))
        return cls(
            keyframe_ids=pose_graph.keyframe_ids,
            graph=graph,
            values=values,
            base_factor_count=len(pose_graph.keyframe_ids),
            state_version=len(accepted_previous_frame_ids),
            bundle_config=bundle_config,
        )

    def pose_matrices(self) -> np.ndarray:
        return np.asarray(
            [self.values.atPose3(key).matrix() for key in self.camera_keys], dtype=float
        )

    def marginal_covariances(self) -> np.ndarray:
        return np.asarray(
            [self.marginals.marginalCovariance(key) for key in self.camera_keys], dtype=float
        )

    def initial_relative_pose(self, previous_frame_id: int, current_frame_id: int) -> gtsam.Pose3:
        return self.values.atPose3(camera_symbol(previous_frame_id)).between(
            self.values.atPose3(camera_symbol(current_frame_id))
        )

    def apply_loop_factor(
        self,
        *,
        previous_frame_id: int,
        current_frame_id: int,
        relative_pose: gtsam.Pose3,
        conditional_covariance: np.ndarray,
    ) -> PoseGraphUpdateEvidence:
        if previous_frame_id not in self.keyframe_ids or current_frame_id not in self.keyframe_ids:
            raise ContractError("loop factor endpoint is not a keyframe")
        if self.keyframe_ids.index(previous_frame_id) >= self.keyframe_ids.index(current_frame_id):
            raise ContractError("loop factor endpoint order is invalid")
        if self.graph.size() != self.base_factor_count + self.state_version:
            raise ContractError("cumulative Pose Graph version/factor count differs")
        covariance = np.asarray(conditional_covariance, dtype=float)
        symmetry_atol = float(self.bundle_config["covariance_symmetry_atol"])
        if (
            covariance.shape != (6, 6)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, atol=symmetry_atol, rtol=0.0)
        ):
            raise ContractError("loop covariance is not finite symmetric 6x6")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ContractError("loop covariance is not positive definite") from error
        loop_factor = gtsam.BetweenFactorPose3(
            camera_symbol(previous_frame_id), camera_symbol(current_frame_id),
            relative_pose, gtsam.noiseModel.Gaussian.Covariance(covariance),
        )
        working_graph = gtsam.NonlinearFactorGraph(self.graph)
        working_graph.add(loop_factor)
        pre_error = float(self.graph.error(self.values))
        augmented_pre = float(working_graph.error(self.values))
        loop_pre = float(loop_factor.error(self.values))
        raw_values = gtsam.LevenbergMarquardtOptimizer(
            working_graph, self.values
        ).optimize()
        raw_error = float(working_graph.error(raw_values))
        raw_anchor = raw_values.atPose3(self.camera_keys[0])
        normalized = gtsam.Values()
        for key in self.camera_keys:
            normalized.insert(key, raw_anchor.between(raw_values.atPose3(key)))
        normalized_error = float(working_graph.error(normalized))
        allowed_absolute = float(self.bundle_config["graph_error_nonincrease_absolute"])
        if normalized_error > raw_error + allowed_absolute:
            raise ContractError("global gauge normalization increased Pose Graph error")
        allowed = max(
            allowed_absolute,
            float(self.bundle_config["graph_error_nonincrease_relative"]) * abs(augmented_pre),
        )
        if normalized_error > augmented_pre + allowed:
            raise ContractError("loop Pose Graph optimization increased graph error")
        loop_post = float(loop_factor.error(normalized))
        before = self.pose_matrices()
        after = np.asarray(
            [normalized.atPose3(key).matrix() for key in self.camera_keys], dtype=float
        )
        translation_movement = np.linalg.norm(after[:, :3, 3] - before[:, :3, 3], axis=1)
        rotation_movement = []
        for key in self.camera_keys:
            rotation_movement.append(
                float(
                    np.linalg.norm(
                        np.asarray(
                            gtsam.Rot3.Logmap(
                                self.values.atPose3(key).rotation().between(
                                    normalized.atPose3(key).rotation()
                                )
                            )
                        )
                    )
                )
            )
        marginals = gtsam.Marginals(working_graph, normalized)
        marginal_array = np.asarray(
            [marginals.marginalCovariance(key) for key in self.camera_keys], dtype=float
        )
        symmetry = float(
            np.max(np.abs(marginal_array - np.swapaxes(marginal_array, 1, 2)))
        )
        if not np.all(np.isfinite(marginal_array)) or symmetry > symmetry_atol:
            raise ContractError("updated Pose Graph Marginals are invalid")
        minimum_eigenvalue = float(
            min(np.min(np.linalg.eigvalsh(value)) for value in marginal_array)
        )
        evidence = PoseGraphUpdateEvidence(
            previous_frame_id,
            current_frame_id,
            self.state_version,
            self.state_version + 1,
            pre_error,
            augmented_pre,
            raw_error,
            normalized_error,
            loop_pre,
            loop_post,
            float(np.max(translation_movement)),
            float(max(rotation_movement)),
            symmetry,
            minimum_eigenvalue,
        )
        # Transaction boundary: publish only after optimization and every
        # marginal covariance have succeeded.
        self.graph = working_graph
        self.values = normalized
        self.marginals = marginals
        self.state_version += 1
        return evidence


__all__ = ["CumulativePoseGraphState", "PoseGraphUpdateEvidence"]
