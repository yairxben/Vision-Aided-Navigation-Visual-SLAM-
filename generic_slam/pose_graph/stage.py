"""Atomic consecutive Pose Graph artifact stage."""

from __future__ import annotations

import os
import resource
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from uuid import uuid4

import numpy as np

from ..artifacts.io import read_json, require_files, save_npz_atomic, write_json_atomic
from ..artifacts.manifest import (
    StageManifest,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)
from ..bundle.stage import BundleOutput
from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiEstimationContext
from ..geometry import validate_rigid_pose
from .consecutive import ConsecutivePoseGraphResult, solve_consecutive_pose_graph
from .state import PoseGraphState, restore_pose_graph_state


@dataclass(frozen=True)
class PoseGraphArtifacts:
    manifest: StageManifest
    keyframe_ids: np.ndarray
    relative_pose_matrices: np.ndarray
    edge_covariances: np.ndarray
    prior_sigmas: np.ndarray
    initial_pose_matrices: np.ndarray
    optimized_pose_matrices: np.ndarray
    marginal_covariances: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.optimized_pose_matrices[:, :3, 3]


@dataclass(frozen=True)
class PoseGraphOutput(PoseGraphArtifacts):
    state: PoseGraphState


def validate_pose_graph_artifacts(root: Path, expected_keyframes: int) -> None:
    require_files(
        root,
        (
            "constraints.npz", "initial_values.npz", "optimized_values.npz",
            "marginals.npz", "summary.json",
        ),
    )
    with np.load(root / "constraints.npz", allow_pickle=False) as data:
        frame_ids = data["frame_ids"]
        edges = data["relative_pose_matrices"]
        covariances = data["conditional_covariances"]
        sigmas = data["prior_sigmas"]
    with np.load(root / "optimized_values.npz", allow_pickle=False) as data:
        optimized_frame_ids = data["frame_ids"]
        optimized = data["pose_matrices"]
    with np.load(root / "initial_values.npz", allow_pickle=False) as data:
        initial_frame_ids = data["frame_ids"]
        initial = data["pose_matrices"]
    with np.load(root / "marginals.npz", allow_pickle=False) as data:
        marginal_frame_ids = data["frame_ids"]
        marginals = data["marginal_covariances"]
    if (
        frame_ids.shape != (expected_keyframes,)
        or not np.array_equal(initial_frame_ids, frame_ids)
        or not np.array_equal(optimized_frame_ids, frame_ids)
        or not np.array_equal(marginal_frame_ids, frame_ids)
        or np.any(np.diff(frame_ids) <= 0)
        or edges.shape != (expected_keyframes - 1, 4, 4)
        or covariances.shape != (expected_keyframes - 1, 6, 6)
        or sigmas.shape != (6,)
        or initial.shape != (expected_keyframes, 4, 4)
        or optimized.shape != (expected_keyframes, 4, 4)
        or marginals.shape != (expected_keyframes, 6, 6)
    ):
        raise ContractError("Pose Graph artifact shapes differ")
    if not all(np.all(np.isfinite(value)) for value in (
        edges, covariances, sigmas, initial, optimized, marginals
    )) or np.any(sigmas <= 0.0):
        raise ContractError("Pose Graph artifacts contain nonfinite values")
    for pose in np.concatenate((edges, initial, optimized), axis=0):
        validate_rigid_pose(pose, atol=1e-7)
    if (
        not np.allclose(covariances, np.swapaxes(covariances, 1, 2), rtol=0.0, atol=1e-7)
        or not np.allclose(marginals, np.swapaxes(marginals, 1, 2), rtol=0.0, atol=1e-7)
    ):
        raise ContractError("Pose Graph covariances are not symmetric")


def _from_result(
    manifest: StageManifest,
    bundle: BundleOutput,
    prior_sigmas: np.ndarray,
    result: ConsecutivePoseGraphResult,
) -> PoseGraphOutput:
    return PoseGraphOutput(
        manifest, bundle.keyframe_ids.copy(), bundle.endpoint_edges.copy(),
        bundle.conditional_covariances.copy(), prior_sigmas.copy(),
        result.initial_pose_matrices, result.optimized_pose_matrices,
        result.marginal_covariances, result.state,
    )


def run_pose_graph_stage(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    bundle: BundleOutput,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
) -> PoseGraphOutput:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_stage_manifest(manifest_path)
        validate_stage_manifest(
            manifest, root, expected_stage="pose_graph",
            sequence_id=dataset.sequence_id, dataset_id=dataset.dataset_id,
            config_fingerprint=config.fingerprint,
            upstream_stage_ids=upstream_stage_ids,
        )
        validate_pose_graph_artifacts(root, len(bundle.keyframe_ids))
        if (
            int(manifest.counts.get("keyframe_count", -1)) != len(bundle.keyframe_ids)
            or int(manifest.counts.get("consecutive_factor_count", -1))
            != len(bundle.keyframe_ids) - 1
        ):
            raise ContractError("Pose Graph manifest counts differ")
        return load_pose_graph_output(root)
    if root.exists():
        os.replace(
            str(root), str(root.with_name(root.name + ".failed-" + uuid4().hex))
        )
    started = time.perf_counter()
    root.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=".pose-graph-attempt-", dir=str(root.parent)))
    bundle_config = config.section("bundle")
    prior_sigmas = np.asarray(bundle_config["prior_sigmas"], dtype=float)
    try:
        result = solve_consecutive_pose_graph(
            keyframe_ids=bundle.keyframe_ids,
            relative_pose_matrices=bundle.endpoint_edges,
            covariance_matrices=bundle.conditional_covariances,
            prior_sigmas=prior_sigmas,
            reconstruction_atol=float(bundle_config["rotation_atol"]),
            graph_error_nonincrease_absolute=float(bundle_config["graph_error_nonincrease_absolute"]),
            graph_error_nonincrease_relative=float(bundle_config["graph_error_nonincrease_relative"]),
        )
        save_npz_atomic(
            attempt / "constraints.npz",
            frame_ids=bundle.keyframe_ids,
            relative_pose_matrices=bundle.endpoint_edges,
            conditional_covariances=bundle.conditional_covariances,
            prior_sigmas=prior_sigmas,
        )
        save_npz_atomic(
            attempt / "initial_values.npz",
            frame_ids=bundle.keyframe_ids,
            pose_matrices=result.initial_pose_matrices,
        )
        save_npz_atomic(
            attempt / "optimized_values.npz",
            frame_ids=bundle.keyframe_ids,
            pose_matrices=result.optimized_pose_matrices,
            positions=result.optimized_pose_matrices[:, :3, 3],
        )
        save_npz_atomic(
            attempt / "marginals.npz",
            frame_ids=bundle.keyframe_ids,
            marginal_covariances=result.marginal_covariances,
        )
        write_json_atomic(
            attempt / "summary.json",
            {
                "schema": "generic-slam-pose-graph-summary-v1",
                "keyframe_count": len(bundle.keyframe_ids),
                "consecutive_factor_count": len(bundle.endpoint_edges),
                "prior_factor_count": 1,
                "total_factor_count": int(result.state.graph.size()),
                "initial_graph_error": result.initial_graph_error,
                "final_graph_error": result.final_graph_error,
                "initial_edge_reconstruction_max_abs": result.initial_edge_reconstruction_max_abs,
                "maximum_translation_movement_m": result.maximum_translation_movement_m,
                "maximum_rotation_movement_rad": result.maximum_rotation_movement_rad,
                "optimizer_iterations": result.optimizer_iterations,
                "edge_direction": "T_Cstart_from_Cend",
                "covariance_policy": "raw conditional end given start",
            },
        )
        validate_pose_graph_artifacts(attempt, len(bundle.keyframe_ids))
        manifest = StageManifest.complete(
            stage="pose_graph", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config.fingerprint,
            upstream_stage_ids=upstream_stage_ids,
            counts={
                "keyframe_count": len(bundle.keyframe_ids),
                "consecutive_factor_count": len(bundle.endpoint_edges),
                "total_factor_count": int(result.state.graph.size()),
            },
            primary_artifacts=(
                "constraints.npz", "initial_values.npz", "optimized_values.npz",
                "marginals.npz", "summary.json",
            ),
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        )
        write_stage_manifest(attempt, manifest)
        os.replace(str(attempt), str(root))
        return _from_result(manifest, bundle, prior_sigmas, result)
    except Exception:
        if attempt.exists():
            os.replace(str(attempt), str(attempt) + ".failed")
        raise


def load_pose_graph_output(root: Path) -> PoseGraphOutput:
    artifacts = load_pose_graph_artifacts(root)
    state = restore_pose_graph_state(
        artifacts.keyframe_ids, artifacts.relative_pose_matrices,
        artifacts.edge_covariances, artifacts.prior_sigmas,
        artifacts.optimized_pose_matrices,
    )
    return PoseGraphOutput(
        artifacts.manifest, artifacts.keyframe_ids, artifacts.relative_pose_matrices,
        artifacts.edge_covariances, artifacts.prior_sigmas,
        artifacts.initial_pose_matrices, artifacts.optimized_pose_matrices,
        artifacts.marginal_covariances, state,
    )


def load_pose_graph_artifacts(root: Path) -> PoseGraphArtifacts:
    """Load saved Values/Marginals without reconstructing a graph or Marginals."""

    root = Path(root)
    manifest = load_stage_manifest(root / "manifest.json")
    with np.load(root / "constraints.npz", allow_pickle=False) as data:
        frame_ids = data["frame_ids"].copy()
        edges = data["relative_pose_matrices"].copy()
        covariances = data["conditional_covariances"].copy()
        sigmas = data["prior_sigmas"].copy()
    with np.load(root / "initial_values.npz", allow_pickle=False) as data:
        initial = data["pose_matrices"].copy()
    with np.load(root / "optimized_values.npz", allow_pickle=False) as data:
        optimized = data["pose_matrices"].copy()
    with np.load(root / "marginals.npz", allow_pickle=False) as data:
        covs = data["marginal_covariances"].copy()
    return PoseGraphArtifacts(
        manifest, frame_ids, edges, covariances, sigmas, initial, optimized,
        covs,
    )


__all__ = [
    "PoseGraphArtifacts", "PoseGraphOutput", "load_pose_graph_artifacts",
    "load_pose_graph_output", "run_pose_graph_stage", "validate_pose_graph_artifacts",
]
