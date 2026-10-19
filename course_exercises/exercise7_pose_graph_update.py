"""Validated E6 bootstrap and cumulative Loop Closure Pose Graph updates.

The E6 baseline is rebuilt from an explicit E5 artifact root and compared with
the saved optimized E6 Values/Marginals.  Making E5 explicit prevents a fresh
graph from one dependency lane being combined with saved E6 arrays from
another.  The resulting cumulative state begins with 356 camera variables,
355 consecutive factors, and the frame-0 prior.

One accepted Loop Closure proposes exactly one previous-to-current
``BetweenFactorPose3``.  Its measurement is
``previous_pose.between(current_pose)`` and its raw conditional end-given-start
6x6 covariance is used directly as the Gaussian noise covariance.  A working
graph copy is optimized immediately with LM, expressed back in the frame-0
global gauge, and used to recompute all 356 Marginals.  Only after every
construction, optimization, and validation step succeeds does
``CumulativePoseGraphState.apply_loop_factor`` publish the proposed graph,
Values, pose matrices, Marginals, and incremented state version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import gtsam
import numpy as np

from PoseGraph import (
    CANONICAL_INPUT_ROOT,
    CanonicalPoseGraphInputs,
    PoseGraph,
    camera_symbol,
    load_canonical_pose_graph_inputs,
)
from exercise7_small_bundle import CURRENT_FRAME_ID, PREVIOUS_FRAME_ID


class PoseGraphUpdateValidationError(RuntimeError):
    """Raised when a Loop Closure Pose Graph update is invalid."""


@dataclass
class ProtectedE6State:
    """Rebuilt no-loop E6 graph plus its validated saved numerical baseline."""

    inputs: CanonicalPoseGraphInputs
    pose_graph: PoseGraph
    optimized_values: gtsam.Values
    optimized_pose_matrices: np.ndarray
    marginal_covariances: np.ndarray
    fresh_vs_protected_pose_max_abs: float
    original_graph_error: float


@dataclass
class PoseGraphUpdateResult:
    """Validated proposed state and diagnostics for one accepted loop factor.

    The result is computed on a working copy.  Possessing a result does not by
    itself mutate the cumulative holder; publication occurs at the explicit
    transaction boundary in ``apply_loop_factor``.
    """

    graph: gtsam.NonlinearFactorGraph
    pre_loop_values: gtsam.Values
    optimized_values: gtsam.Values
    pre_loop_pose_matrices: np.ndarray
    optimized_pose_matrices: np.ndarray
    marginal_covariances: np.ndarray
    original_graph_error: float
    augmented_pre_optimization_error: float
    raw_optimizer_error: float
    raw_optimizer_anchor_identity_max_abs: float
    augmented_post_optimization_error: float
    loop_factor_pre_optimization_error: float
    loop_factor_post_optimization_error: float
    global_anchor_identity_max_abs: float
    movement_by_frame_id: Mapping[int, Mapping[str, float]]
    maximum_translation_m: float
    maximum_translation_frame_id: int
    maximum_rotation_rad: float
    maximum_rotation_frame_id: int
    marginal_symmetry_max_abs: float
    marginal_minimum_eigenvalue: float
    previous_frame_id: int
    current_frame_id: int
    pre_update_state_version: int
    state_version: int


@dataclass
class CumulativePoseGraphState:
    """Latest fully published Pose Graph, Values, poses, and all-node Marginals.

    ``state_version`` equals the number of successfully integrated Loop
    Closures, so ``graph.size() == base_factor_count + state_version``.  Failed
    proposals leave every held object and the version unchanged.
    """

    inputs: CanonicalPoseGraphInputs
    graph: gtsam.NonlinearFactorGraph
    optimized_values: gtsam.Values
    optimized_pose_matrices: np.ndarray
    marginal_covariances: np.ndarray
    state_version: int
    base_factor_count: int

    @classmethod
    def from_protected_e6(
        cls, protected_state: ProtectedE6State
    ) -> "CumulativePoseGraphState":
        """Copy the protected E6 state without mutating its graph or Values."""

        keyframe_ids = tuple(
            int(frame_id) for frame_id in protected_state.inputs.keyframe_ids
        )
        camera_keys = tuple(
            int(camera_symbol(frame_id)) for frame_id in keyframe_ids
        )
        graph = protected_state.pose_graph.graph()
        _require(
            graph.size() == len(keyframe_ids),
            "protected E6 graph factor count differs from its anchored chain",
        )
        _pose_matrices(protected_state.optimized_values, camera_keys)
        _require(
            protected_state.optimized_pose_matrices.shape
            == (len(keyframe_ids), 4, 4)
            and protected_state.marginal_covariances.shape
            == (len(keyframe_ids), 6, 6)
            and np.all(
                np.isfinite(protected_state.optimized_pose_matrices)
            )
            and np.all(np.isfinite(protected_state.marginal_covariances)),
            "protected E6 numerical state has invalid shapes or values",
        )
        return cls(
            inputs=protected_state.inputs,
            graph=gtsam.NonlinearFactorGraph(graph),
            optimized_values=gtsam.Values(
                protected_state.optimized_values
            ),
            optimized_pose_matrices=np.asarray(
                protected_state.optimized_pose_matrices, dtype=float
            ).copy(),
            marginal_covariances=np.asarray(
                protected_state.marginal_covariances, dtype=float
            ).copy(),
            state_version=0,
            base_factor_count=int(graph.size()),
        )

    @property
    def camera_keys(self) -> Tuple[int, ...]:
        return tuple(
            int(camera_symbol(frame_id))
            for frame_id in self.inputs.keyframe_ids
        )

    def initial_relative_pose(
        self, previous_frame_id: int, current_frame_id: int
    ) -> gtsam.Pose3:
        """Return latest ``previous.between(current)`` for Bundle initialization.

        This pose initializes the candidate-local two-camera Bundle; it is not
        inserted as a measurement or factor in that Bundle.
        """

        previous_key, current_key = _validated_endpoint_keys(
            self.inputs,
            previous_frame_id=previous_frame_id,
            current_frame_id=current_frame_id,
        )
        return self.optimized_values.atPose3(previous_key).between(
            self.optimized_values.atPose3(current_key)
        )

    def apply_loop_factor(
        self,
        *,
        previous_frame_id: int,
        current_frame_id: int,
        relative_pose: gtsam.Pose3,
        conditional_covariance: np.ndarray,
    ) -> PoseGraphUpdateResult:
        """Publish one proposed update only after optimization and Marginals pass.

        ``relative_pose`` is previous-to-current and
        ``conditional_covariance`` is its raw end-given-start covariance in
        GTSAM rotation-then-translation tangent order.  The computation below
        uses working copies; this object mutates only at the assignment block
        after factor, LM, frame-0 gauge, all-node Marginals, and validation have
        completed successfully.
        """

        result = _compute_loop_factor_update(
            inputs=self.inputs,
            current_graph=self.graph,
            current_values=self.optimized_values,
            camera_keys=self.camera_keys,
            base_factor_count=self.base_factor_count,
            pre_update_state_version=self.state_version,
            previous_frame_id=previous_frame_id,
            current_frame_id=current_frame_id,
            relative_pose=relative_pose,
            conditional_covariance=conditional_covariance,
        )
        _require(
            result.graph.size()
            == self.base_factor_count + result.state_version,
            "successful cumulative update has an invalid factor count",
        )

        # This is the transaction boundary. No held state is changed until
        # factor construction, full optimization, validation, and every
        # requested marginal covariance have all completed successfully.
        self.graph = result.graph
        self.optimized_values = result.optimized_values
        self.optimized_pose_matrices = result.optimized_pose_matrices
        self.marginal_covariances = result.marginal_covariances
        self.state_version = result.state_version
        return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PoseGraphUpdateValidationError(message)


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pose_matrices(values: gtsam.Values, keys: Tuple[int, ...]) -> np.ndarray:
    matrices = np.asarray(
        [values.atPose3(key).matrix() for key in keys], dtype=float
    )
    _require(
        matrices.shape == (len(keys), 4, 4)
        and np.all(np.isfinite(matrices)),
        "Pose Graph values are nonfinite or have the wrong shape",
    )
    return matrices


def _validated_endpoint_keys(
    inputs: CanonicalPoseGraphInputs,
    *,
    previous_frame_id: int,
    current_frame_id: int,
) -> Tuple[int, int]:
    keyframe_ids = tuple(int(value) for value in inputs.keyframe_ids)
    previous_frame_id = int(previous_frame_id)
    current_frame_id = int(current_frame_id)
    try:
        previous_ordinal = keyframe_ids.index(previous_frame_id)
        current_ordinal = keyframe_ids.index(current_frame_id)
    except ValueError as error:
        raise PoseGraphUpdateValidationError(
            "loop endpoints must both be canonical keyframes"
        ) from error
    _require(
        previous_ordinal < current_ordinal,
        "loop endpoints must be ordered previous-to-current keyframes",
    )
    return (
        int(camera_symbol(previous_frame_id)),
        int(camera_symbol(current_frame_id)),
    )


def load_protected_e6_state(
    *,
    summary_path: Path,
    values_path: Path,
    marginals_path: Path,
    e5_root: Path = CANONICAL_INPUT_ROOT,
) -> ProtectedE6State:
    """Rebuild E6 from explicit E5 evidence and verify saved no-loop state.

    The fresh consecutive graph and saved E6 Values/Marginals must identify the
    same E5 root, configuration, actual FrameId ordering, and 356/355
    population.  The returned baseline is suitable for copying into cumulative
    state; neither selected artifact root is modified.
    """

    inputs = load_canonical_pose_graph_inputs(e5_root)
    pose_graph = PoseGraph.from_canonical_inputs(inputs)
    summary = _read_json(summary_path)
    declared_e5_root = summary.get("canonical_input_root")
    _require(
        summary.get("schema") == "exercise6-pose-graph-summary-v1"
        and summary.get("milestone") == "E6-I01"
        and summary.get("status") == "pass"
        and summary.get("scope")
        == "consecutive keyframes only; no Loop Closure"
        and summary.get("keyframe_count") == 356
        and summary.get("edge_count") == 355
        and summary.get("factor_counts", {}).get("total") == 356
        and summary.get("canonical_configuration_sha256")
        == inputs.configuration_sha256,
        "protected E6 summary contract differs",
    )
    _require(
        isinstance(declared_e5_root, str)
        and (Path(__file__).resolve().parent / declared_e5_root).resolve()
        == inputs.input_root,
        "protected E6 summary does not identify the selected E5 root",
    )
    with np.load(values_path, allow_pickle=False) as data:
        _require(
            set(data.files)
            == {"frame_ids", "initial_poses", "optimized_poses"},
            "protected E6 Values members differ",
        )
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        optimized_matrices = np.asarray(
            data["optimized_poses"], dtype=float
        )
    with np.load(marginals_path, allow_pickle=False) as data:
        _require(
            set(data.files) == {"frame_ids", "marginal_covariances"},
            "protected E6 Marginals members differ",
        )
        marginal_frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        marginal_covariances = np.asarray(
            data["marginal_covariances"], dtype=float
        )
    expected_ids = np.asarray(inputs.keyframe_ids, dtype=np.int64)
    _require(
        np.array_equal(frame_ids, expected_ids)
        and np.array_equal(marginal_frame_ids, expected_ids)
        and optimized_matrices.shape == (356, 4, 4)
        and marginal_covariances.shape == (356, 6, 6)
        and np.all(np.isfinite(optimized_matrices))
        and np.all(np.isfinite(marginal_covariances)),
        "protected E6 arrays differ from the canonical keyframe contract",
    )

    values = gtsam.Values()
    for frame_id, matrix in zip(inputs.keyframe_ids, optimized_matrices):
        _require(
            np.max(np.abs(matrix[3] - np.asarray([0.0, 0.0, 0.0, 1.0])))
            <= inputs.validation_tolerances["homogeneous_matrix_max_abs"],
            "protected E6 pose has an invalid homogeneous row",
        )
        rotation = matrix[:3, :3]
        _require(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
            <= inputs.validation_tolerances[
                "rotation_orthogonality_max_abs"
            ]
            and abs(float(np.linalg.det(rotation)) - 1.0)
            <= inputs.validation_tolerances["rotation_determinant_abs"],
            "protected E6 pose has a non-rigid rotation",
        )
        values.insert(int(camera_symbol(frame_id)), gtsam.Pose3(matrix))

    fresh_matrices = _pose_matrices(
        pose_graph.get_initial_estimate(), pose_graph.camera_keys
    )
    difference = float(
        np.max(np.abs(fresh_matrices - optimized_matrices))
    )
    _require(
        difference
        <= inputs.validation_tolerances["numerical_reconstruction_atol"],
        "protected E6 optimized Values differ from the fresh canonical chain",
    )
    original_error = float(pose_graph.graph().error(values))
    _require(
        np.isfinite(original_error),
        "protected E6 optimized graph error is nonfinite",
    )
    return ProtectedE6State(
        inputs=inputs,
        pose_graph=pose_graph,
        optimized_values=values,
        optimized_pose_matrices=optimized_matrices.copy(),
        marginal_covariances=marginal_covariances.copy(),
        fresh_vs_protected_pose_max_abs=difference,
        original_graph_error=original_error,
    )


def initial_loop_relative_pose(state: ProtectedE6State) -> gtsam.Pose3:
    """Return only the required c356.between(c3359) Bundle initialization."""

    return state.optimized_values.atPose3(
        int(camera_symbol(PREVIOUS_FRAME_ID))
    ).between(
        state.optimized_values.atPose3(
            int(camera_symbol(CURRENT_FRAME_ID))
        )
    )


def apply_one_loop_factor(
    *,
    state: ProtectedE6State,
    relative_pose: gtsam.Pose3,
    conditional_covariance: np.ndarray,
) -> PoseGraphUpdateResult:
    """Preserve the exact bounded E7-I03 one-factor update interface."""

    cumulative_state = CumulativePoseGraphState.from_protected_e6(state)
    return cumulative_state.apply_loop_factor(
        previous_frame_id=PREVIOUS_FRAME_ID,
        current_frame_id=CURRENT_FRAME_ID,
        relative_pose=relative_pose,
        conditional_covariance=conditional_covariance,
    )


def _compute_loop_factor_update(
    *,
    inputs: CanonicalPoseGraphInputs,
    current_graph: gtsam.NonlinearFactorGraph,
    current_values: gtsam.Values,
    camera_keys: Tuple[int, ...],
    base_factor_count: int,
    pre_update_state_version: int,
    previous_frame_id: int,
    current_frame_id: int,
    relative_pose: gtsam.Pose3,
    conditional_covariance: np.ndarray,
) -> PoseGraphUpdateResult:
    """Scientifically compute and validate one update without publishing it.

    The raw previous-to-current measurement/covariance creates one factor on a
    copy of the cumulative graph.  LM optimization is followed by exact global
    frame-0 gauge normalization, movement/error checks, and Marginals for every
    canonical camera.  A fully populated result is returned only after all
    checks pass; the caller owns the separate publication transaction.
    """

    keyframe_ids = tuple(int(value) for value in inputs.keyframe_ids)
    _require(
        tuple(camera_keys)
        == tuple(int(camera_symbol(frame_id)) for frame_id in keyframe_ids),
        "cumulative Pose Graph camera keys differ from canonical keyframes",
    )
    _require(
        current_graph.size()
        == int(base_factor_count) + int(pre_update_state_version),
        "cumulative Pose Graph factor count differs from its state version",
    )
    pre_update_error = float(current_graph.error(current_values))
    _require(
        np.isfinite(pre_update_error),
        "current cumulative Pose Graph error is nonfinite",
    )
    # Scientific factor contract: use the raw conditional end-given-start
    # covariance directly in [rotation xyz (rad), translation xyz (m)] order.
    covariance = np.asarray(conditional_covariance, dtype=float)
    _require(
        covariance.shape == (6, 6)
        and np.all(np.isfinite(covariance)),
        "loop covariance must be finite 6x6",
    )
    symmetry_error = float(np.max(np.abs(covariance - covariance.T)))
    _require(
        symmetry_error
        <= inputs.validation_tolerances["covariance_symmetry_atol"],
        "loop covariance is asymmetric",
    )
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise PoseGraphUpdateValidationError(
            "loop covariance is not positive definite"
        ) from error
    noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
    previous_key, current_key = _validated_endpoint_keys(
        inputs,
        previous_frame_id=previous_frame_id,
        current_frame_id=current_frame_id,
    )
    loop_factor = gtsam.BetweenFactorPose3(
        previous_key, current_key, relative_pose, noise
    )
    _require(
        tuple(int(key) for key in loop_factor.keys())
        == (previous_key, current_key),
        "loop factor keys or direction differ",
    )

    # Work on a graph copy so a failed factor, LM solve, gauge check, or
    # Marginals construction cannot mutate the last published state.
    working_graph = gtsam.NonlinearFactorGraph(current_graph)
    working_graph.add(loop_factor)
    _require(
        working_graph.size() == current_graph.size() + 1,
        "one-factor working copy did not preserve the cumulative graph",
    )
    _require(
        isinstance(working_graph.at(0), gtsam.PriorFactorPose3),
        "augmented graph first factor is not the canonical prior",
    )
    for factor_index in range(1, working_graph.size()):
        _require(
            isinstance(
                working_graph.at(factor_index),
                gtsam.BetweenFactorPose3,
            ),
            "augmented graph contains an unexpected factor type",
        )

    augmented_pre_error = float(
        working_graph.error(current_values)
    )
    loop_pre_error = float(loop_factor.error(current_values))
    _require(
        np.isfinite(augmented_pre_error) and np.isfinite(loop_pre_error),
        "augmented pre-optimization graph evidence is nonfinite",
    )
    # Optimize immediately after the one-factor insertion, then express all
    # poses relative to the optimized frame-0 pose to restore the global gauge.
    raw_optimized_values = gtsam.LevenbergMarquardtOptimizer(
        working_graph, current_values
    ).optimize()
    raw_optimizer_error = float(
        working_graph.error(raw_optimized_values)
    )
    raw_anchor = raw_optimized_values.atPose3(camera_keys[0])
    raw_anchor_error = float(
        np.max(np.abs(np.asarray(raw_anchor.matrix()) - np.eye(4)))
    )
    optimized_values = gtsam.Values()
    for frame_id in keyframe_ids:
        key = int(camera_symbol(frame_id))
        optimized_values.insert(
            key, raw_anchor.between(raw_optimized_values.atPose3(key))
        )
    augmented_post_error = float(working_graph.error(optimized_values))
    loop_post_error = float(loop_factor.error(optimized_values))
    _require(
        np.isfinite(raw_optimizer_error)
        and np.isfinite(augmented_post_error)
        and np.isfinite(loop_post_error),
        "augmented post-optimization graph evidence is nonfinite",
    )
    _require(
        augmented_post_error
        <= raw_optimizer_error
        + inputs.validation_tolerances[
            "graph_error_nonincrease_absolute"
        ],
        "global frame-0 gauge normalization increased graph error beyond "
        "tolerance: raw {} normalized {}".format(
            raw_optimizer_error, augmented_post_error
        ),
    )
    allowed_increase = max(
        float(
            inputs.validation_tolerances[
                "graph_error_nonincrease_absolute"
            ]
        ),
        float(
            inputs.validation_tolerances[
                "graph_error_nonincrease_relative"
            ]
        )
        * abs(augmented_pre_error),
    )
    _require(
        augmented_post_error <= augmented_pre_error + allowed_increase,
        "augmented Pose Graph optimization increased its own graph error",
    )

    pre_matrices = _pose_matrices(
        current_values, camera_keys
    )
    optimized_matrices = _pose_matrices(
        optimized_values, camera_keys
    )
    movement_by_frame: Dict[int, Mapping[str, float]] = {}
    for frame_id, key in zip(
        keyframe_ids, camera_keys
    ):
        before = current_values.atPose3(key)
        after = optimized_values.atPose3(key)
        translation = float(
            np.linalg.norm(
                np.asarray(after.translation(), dtype=float)
                - np.asarray(before.translation(), dtype=float)
            )
        )
        rotation = float(
            np.linalg.norm(
                np.asarray(
                    gtsam.Rot3.Logmap(
                        before.rotation().between(after.rotation())
                    ),
                    dtype=float,
                )
            )
        )
        _require(
            np.isfinite(translation) and np.isfinite(rotation),
            "Pose Graph movement is nonfinite",
        )
        movement_by_frame[int(frame_id)] = {
            "translation_m": translation,
            "rotation_rad": rotation,
        }

    maximum_translation_frame = max(
        keyframe_ids,
        key=lambda frame_id: (
            movement_by_frame[frame_id]["translation_m"],
            -frame_id,
        ),
    )
    maximum_rotation_frame = max(
        keyframe_ids,
        key=lambda frame_id: (
            movement_by_frame[frame_id]["rotation_rad"],
            -frame_id,
        ),
    )
    frame_zero = optimized_values.atPose3(camera_keys[0])
    anchor_error = float(
        np.max(np.abs(np.asarray(frame_zero.matrix()) - np.eye(4)))
    )
    _require(
        anchor_error <= inputs.validation_tolerances["identity_pose_max_abs"],
        "global Pose Graph frame 0 did not remain identity",
    )

    # Recompute and validate all camera Marginals for the proposed 356-node
    # state; partial marginal success is not publishable.
    marginals = gtsam.Marginals(working_graph, optimized_values)
    marginal_covariances = np.asarray(
        [
            marginals.marginalCovariance(key)
            for key in camera_keys
        ],
        dtype=float,
    )
    _require(
        marginal_covariances.shape == (len(keyframe_ids), 6, 6)
        and np.all(np.isfinite(marginal_covariances)),
        "not all updated Pose Graph Marginals are finite",
    )
    marginal_symmetry = float(
        np.max(
            np.abs(
                marginal_covariances
                - np.swapaxes(marginal_covariances, 1, 2)
            )
        )
    )
    _require(
        marginal_symmetry
        <= inputs.validation_tolerances["covariance_symmetry_atol"],
        "updated Pose Graph Marginals are asymmetric",
    )
    marginal_minimum_eigenvalue = float(
        min(
            np.min(np.linalg.eigvalsh(covariance_matrix))
            for covariance_matrix in marginal_covariances
        )
    )
    _require(
        np.isfinite(marginal_minimum_eigenvalue),
        "updated Pose Graph marginal eigenvalues are nonfinite",
    )
    return PoseGraphUpdateResult(
        graph=working_graph,
        pre_loop_values=current_values,
        optimized_values=optimized_values,
        pre_loop_pose_matrices=pre_matrices,
        optimized_pose_matrices=optimized_matrices,
        marginal_covariances=marginal_covariances,
        original_graph_error=pre_update_error,
        augmented_pre_optimization_error=augmented_pre_error,
        raw_optimizer_error=raw_optimizer_error,
        raw_optimizer_anchor_identity_max_abs=raw_anchor_error,
        augmented_post_optimization_error=augmented_post_error,
        loop_factor_pre_optimization_error=loop_pre_error,
        loop_factor_post_optimization_error=loop_post_error,
        global_anchor_identity_max_abs=anchor_error,
        movement_by_frame_id=movement_by_frame,
        maximum_translation_m=movement_by_frame[
            maximum_translation_frame
        ]["translation_m"],
        maximum_translation_frame_id=int(maximum_translation_frame),
        maximum_rotation_rad=movement_by_frame[
            maximum_rotation_frame
        ]["rotation_rad"],
        maximum_rotation_frame_id=int(maximum_rotation_frame),
        marginal_symmetry_max_abs=marginal_symmetry,
        marginal_minimum_eigenvalue=marginal_minimum_eigenvalue,
        previous_frame_id=int(previous_frame_id),
        current_frame_id=int(current_frame_id),
        pre_update_state_version=int(pre_update_state_version),
        state_version=int(pre_update_state_version) + 1,
    )


__all__ = [
    "CumulativePoseGraphState",
    "PoseGraphUpdateResult",
    "PoseGraphUpdateValidationError",
    "ProtectedE6State",
    "apply_one_loop_factor",
    "initial_loop_relative_pose",
    "load_protected_e6_state",
]
