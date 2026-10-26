"""Per-window atomic Bundle streaming, structural reload, and resume."""

from __future__ import annotations

import csv
import os
import resource
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import gtsam
import numpy as np

from ..artifacts.io import (
    read_json,
    require_files,
    save_npz_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from ..artifacts.manifest import (
    StageManifest,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)
from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiEstimationContext
from ..geometry import TANGENT_ORDER, validate_rigid_pose
from ..keyframes.stage import KeyframeOutput
from ..tracking.stage import TrackingOutput
from .covariance import BundleCovarianceEvidence, extract_bundle_covariances
from .window import BundleWindowResult, LandmarkEligibility, solve_bundle_window


WINDOW_ARTIFACTS = (
    "summary.json",
    "values.npz",
    "endpoint_edge.json",
    "endpoint_covariances.npz",
    "camera_marginals.npz",
    "projection_residuals.csv",
    "rejections.csv",
)
PROJECTION_FIELDS = (
    "factor_index", "frame_id", "track_id", "u_left_px", "u_right_px",
    "v_mean_px", "initial_u_left_px", "initial_u_right_px",
    "initial_v_mean_px", "final_u_left_px", "final_u_right_px",
    "final_v_mean_px", "initial_residual_u_left_px",
    "initial_residual_u_right_px", "initial_residual_v_mean_px",
    "final_residual_u_left_px", "final_residual_u_right_px",
    "final_residual_v_mean_px", "initial_factor_error", "final_factor_error",
)


@dataclass(frozen=True)
class BundleOutput:
    manifest: Optional[StageManifest]
    keyframe_ids: np.ndarray
    endpoint_edges: np.ndarray
    conditional_covariances: np.ndarray
    camera_to_first_keyframe: np.ndarray
    world_to_camera: np.ndarray
    positions: np.ndarray


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _window_name(ordinal: int, start: int, end: int) -> str:
    return "window_{:04d}_{:06d}_{:06d}".format(ordinal, start, end)


def _stage_identity(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    upstream_stage_ids: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "schema": "generic-slam-bundle-state-v1",
        "sequence_id": dataset.sequence_id,
        "dataset_id": dataset.dataset_id,
        "config_fingerprint": config.stage_fingerprint("bundle"),
        "upstream_stage_ids": dict(upstream_stage_ids),
    }


def _gtsam_calibration(dataset: KittiEstimationContext) -> gtsam.Cal3_S2Stereo:
    value = dataset.calibration
    return gtsam.Cal3_S2Stereo(
        value.fx_px, value.fy_px, value.skew_px, value.cx_px, value.cy_px,
        value.baseline_m,
    )


def _window_arrays(result: BundleWindowResult):
    camera_initial = []
    camera_final = []
    for frame_id in result.frame_ids:
        key = result.camera_key_by_frame_id[frame_id]
        camera_initial.append(np.asarray(result.initial_values.atPose3(key).matrix(), dtype=float))
        camera_final.append(np.asarray(result.final_values.atPose3(key).matrix(), dtype=float))
    landmark_initial = []
    landmark_final = []
    for track_id in result.accepted_track_ids:
        key = result.landmark_key_by_track_id[track_id]
        landmark_initial.append(np.asarray(result.initial_values.atPoint3(key), dtype=float))
        landmark_final.append(np.asarray(result.final_values.atPoint3(key), dtype=float))
    return {
        "frame_ids": np.asarray(result.frame_ids, dtype=np.int64),
        "camera_initial": np.asarray(camera_initial, dtype=float),
        "camera_final": np.asarray(camera_final, dtype=float),
        "track_ids": np.asarray(result.accepted_track_ids, dtype=np.int64),
        "landmark_initial": np.asarray(landmark_initial, dtype=float).reshape(-1, 3),
        "landmark_final": np.asarray(landmark_final, dtype=float).reshape(-1, 3),
    }


def _projection_rows(result: BundleWindowResult):
    rows = []
    for index, record in enumerate(result.projection_records):
        measurement = np.asarray(record.measurement, dtype=float)
        initial_residual = result.initial_residuals[index]
        final_residual = result.final_residuals[index]
        initial_projection = measurement + initial_residual
        final_projection = measurement + final_residual
        rows.append(
            {
                "factor_index": record.factor_index,
                "frame_id": record.frame_id,
                "track_id": record.track_id,
                "u_left_px": measurement[0],
                "u_right_px": measurement[1],
                "v_mean_px": measurement[2],
                "initial_u_left_px": initial_projection[0],
                "initial_u_right_px": initial_projection[1],
                "initial_v_mean_px": initial_projection[2],
                "final_u_left_px": final_projection[0],
                "final_u_right_px": final_projection[1],
                "final_v_mean_px": final_projection[2],
                "initial_residual_u_left_px": initial_residual[0],
                "initial_residual_u_right_px": initial_residual[1],
                "initial_residual_v_mean_px": initial_residual[2],
                "final_residual_u_left_px": final_residual[0],
                "final_residual_u_right_px": final_residual[1],
                "final_residual_v_mean_px": final_residual[2],
                "initial_factor_error": result.initial_projection_errors[index],
                "final_factor_error": result.final_projection_errors[index],
            }
        )
    return rows


def _seal_window(
    windows_root: Path,
    ordinal: int,
    result: BundleWindowResult,
    covariance: BundleCovarianceEvidence,
    identity: Mapping[str, object],
) -> Path:
    start, end = result.start_frame_id, result.end_frame_id
    destination = windows_root / _window_name(ordinal, start, end)
    if destination.exists():
        validate_window(destination, ordinal, start, end, identity)
        return destination
    windows_root.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(
            prefix=".{}-attempt-".format(destination.name), dir=str(windows_root)
        )
    )
    try:
        save_npz_atomic(attempt / "values.npz", **_window_arrays(result))
        edge = covariance.endpoint_edge
        rotation_vector = cv2.Rodrigues(edge[:3, :3])[0].reshape(3)
        write_json_atomic(
            attempt / "endpoint_edge.json",
            {
                "schema": "generic-slam-bundle-edge-v1",
                "start_frame_id": start,
                "end_frame_id": end,
                "direction": "T_Cstart_from_Cend",
                "matrix": edge.tolist(),
                "rotation_vector_rad": rotation_vector.tolist(),
                "translation_m": edge[:3, 3].tolist(),
            },
        )
        save_npz_atomic(
            attempt / "endpoint_covariances.npz",
            joint_covariance=covariance.joint_covariance,
            joint_information=covariance.joint_information,
            conditional_end_given_start_covariance=(
                covariance.conditional_end_given_start_covariance
            ),
            tangent_order=np.asarray(TANGENT_ORDER),
        )
        save_npz_atomic(
            attempt / "camera_marginals.npz",
            frame_ids=covariance.camera_frame_ids,
            camera_poses=covariance.camera_poses,
            marginal_covariances=covariance.camera_marginal_covariances,
        )
        write_csv_atomic(
            attempt / "projection_residuals.csv",
            _projection_rows(result),
            PROJECTION_FIELDS,
        )
        write_csv_atomic(
            attempt / "rejections.csv",
            (
                {
                    "track_id": track_id,
                    "reason": result.rejected_reason_by_track_id[track_id],
                }
                for track_id in sorted(result.rejected_reason_by_track_id)
            ),
            ("track_id", "reason"),
        )
        rejection_counts = Counter(result.rejected_reason_by_track_id.values())
        summary = {
            "schema": "generic-slam-bundle-window-summary-v1",
            "window_ordinal": ordinal,
            "start_frame_id": start,
            "end_frame_id": end,
            "window_frame_count": len(result.frame_ids),
            "candidate_track_count": len(result.candidate_track_ids),
            "accepted_track_count": len(result.accepted_track_ids),
            "rejected_track_count": len(result.rejected_reason_by_track_id),
            "projection_factor_count": len(result.projection_records),
            "total_factor_count": int(result.graph.size()),
            "initial_graph_error": result.initial_graph_error,
            "final_graph_error": result.final_graph_error,
            "mean_initial_projection_factor_error": float(np.mean(result.initial_projection_errors)),
            "mean_final_projection_factor_error": float(np.mean(result.final_projection_errors)),
            "initial_cheirality_factor_count": result.initial_cheirality_factor_count,
            "final_cheirality_factor_count": result.final_cheirality_factor_count,
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "invalid_observation_reason_counts": dict(result.invalid_observation_reason_counts),
            "optimizer_iterations": result.optimizer_iterations,
            "graph_construction_runtime_seconds": result.graph_construction_runtime_seconds,
            "optimization_runtime_seconds": result.optimization_runtime_seconds,
        }
        write_json_atomic(attempt / "summary.json", summary)
        seal = dict(identity)
        seal.update(
            {
                "schema": "generic-slam-bundle-window-seal-v1",
                "status": "complete",
                "window_ordinal": ordinal,
                "start_frame_id": start,
                "end_frame_id": end,
                "primary_artifacts": list(WINDOW_ARTIFACTS),
            }
        )
        # The seal is last: structural validation, not a checksum inventory,
        # is the resume gate.
        validate_window_payload(attempt, ordinal, start, end)
        write_json_atomic(attempt / "seal.json", seal)
        validate_window(attempt, ordinal, start, end, identity)
        os.replace(str(attempt), str(destination))
        return destination
    except Exception:
        failed = attempt.with_name(attempt.name + ".failed")
        if attempt.exists():
            os.replace(str(attempt), str(failed))
        raise


def validate_window_payload(directory: Path, ordinal: int, start: int, end: int) -> None:
    require_files(directory, WINDOW_ARTIFACTS)
    summary = read_json(directory / "summary.json")
    if (
        int(summary["window_ordinal"]) != ordinal
        or int(summary["start_frame_id"]) != start
        or int(summary["end_frame_id"]) != end
        or not 5 <= int(summary["window_frame_count"]) <= 20
    ):
        raise ContractError("Bundle window summary identity differs")
    with np.load(directory / "values.npz", allow_pickle=False) as data:
        frame_ids = data["frame_ids"]
        camera_initial = data["camera_initial"]
        camera_final = data["camera_final"]
        track_ids = data["track_ids"]
        landmarks = data["landmark_final"]
    if not np.array_equal(frame_ids, np.arange(start, end + 1, dtype=np.int64)):
        raise ContractError("Bundle window FrameIds differ")
    if (
        camera_initial.shape != (len(frame_ids), 4, 4)
        or camera_final.shape != (len(frame_ids), 4, 4)
        or landmarks.shape != (len(track_ids), 3)
        or len(track_ids) != int(summary["accepted_track_count"])
    ):
        raise ContractError("Bundle Values artifact shapes differ")
    if not all(np.all(np.isfinite(value)) for value in (camera_initial, camera_final, landmarks)):
        raise ContractError("Bundle Values artifacts are nonfinite")
    edge = read_json(directory / "endpoint_edge.json")
    if edge.get("direction") != "T_Cstart_from_Cend":
        raise ContractError("Bundle endpoint edge direction differs")
    edge_matrix = validate_rigid_pose(np.asarray(edge["matrix"], dtype=float))
    expected_edge = np.linalg.inv(camera_final[0]) @ camera_final[-1]
    if not np.allclose(edge_matrix, expected_edge, rtol=0.0, atol=1e-9):
        raise ContractError("Bundle endpoint edge differs from optimized cameras")
    with np.load(directory / "endpoint_covariances.npz", allow_pickle=False) as data:
        joint = data["joint_covariance"]
        information = data["joint_information"]
        conditional = data["conditional_end_given_start_covariance"]
        tangent_order = tuple(str(value) for value in data["tangent_order"])
        if (
            joint.shape != (12, 12)
            or information.shape != (12, 12)
            or conditional.shape != (6, 6)
            or tangent_order != TANGENT_ORDER
            or not all(np.all(np.isfinite(value)) for value in (
                joint, information, conditional
            ))
            or not np.allclose(conditional, conditional.T, rtol=0.0, atol=1e-9)
        ):
            raise ContractError("Bundle endpoint covariance shapes differ")
    with np.load(directory / "camera_marginals.npz", allow_pickle=False) as data:
        marginal_frame_ids = data["frame_ids"]
        marginal_poses = data["camera_poses"]
        marginal_covariances = data["marginal_covariances"]
        if (
            not np.array_equal(marginal_frame_ids, frame_ids)
            or marginal_poses.shape != (len(frame_ids), 4, 4)
            or marginal_covariances.shape != (len(frame_ids), 6, 6)
            or not np.all(np.isfinite(marginal_covariances))
        ):
            raise ContractError("Bundle camera marginal shapes differ")
    with (directory / "projection_residuals.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PROJECTION_FIELDS:
            raise ContractError("Bundle projection residual columns differ")


def validate_window(
    directory: Path,
    ordinal: int,
    start: int,
    end: int,
    identity: Mapping[str, object],
) -> None:
    seal_path = directory / "seal.json"
    if not seal_path.is_file():
        raise ContractError("Bundle window is not sealed")
    seal = read_json(seal_path)
    expected = dict(identity)
    for key, value in expected.items():
        if key == "schema":
            continue
        if seal.get(key) != value:
            raise ContractError("Bundle window identity differs at {}".format(key))
    if (
        seal.get("schema") != "generic-slam-bundle-window-seal-v1"
        or seal.get("status") != "complete"
        or int(seal.get("window_ordinal", -1)) != ordinal
        or int(seal.get("start_frame_id", -1)) != start
        or int(seal.get("end_frame_id", -1)) != end
    ):
        raise ContractError("Bundle window seal differs")
    validate_window_payload(directory, ordinal, start, end)


def _load_edges(windows_root: Path, keyframes: Sequence[int]):
    edges = []
    covariances = []
    for ordinal, (start, end) in enumerate(zip(keyframes, keyframes[1:])):
        directory = windows_root / _window_name(ordinal, int(start), int(end))
        edge = read_json(directory / "endpoint_edge.json")
        edges.append(np.asarray(edge["matrix"], dtype=float))
        with np.load(directory / "endpoint_covariances.npz", allow_pickle=False) as data:
            covariances.append(data["conditional_end_given_start_covariance"].copy())
    return np.asarray(edges, dtype=float), np.asarray(covariances, dtype=float)


def _compose_bundle_trajectory(keyframes: Sequence[int], edges: np.ndarray):
    poses = [np.eye(4)]
    for edge in edges:
        poses.append(poses[-1] @ edge)
    camera_to_first = np.asarray(poses, dtype=float)
    world_to_camera = np.linalg.inv(camera_to_first)
    positions = camera_to_first[:, :3, 3].copy()
    return camera_to_first, world_to_camera, positions


def _validate_bundle_payload(
    root: Path,
    keyframes: Sequence[int],
    *,
    identity: Mapping[str, object],
) -> None:
    """Validate durable Bundle contents before publishing its manifest."""

    root = Path(root)
    selected = tuple(int(value) for value in keyframes)
    state = read_json(root / "state.json")
    expected_ordinals = list(range(len(selected) - 1))
    if (
        state.get("status") != "complete"
        or int(state.get("window_count", -1)) != len(expected_ordinals)
        or state.get("completed_window_ordinals") != expected_ordinals
    ):
        raise ContractError("Bundle completion state differs")
    for ordinal, (start, end) in enumerate(zip(selected, selected[1:])):
        validate_window(
            root / "windows" / _window_name(ordinal, start, end),
            ordinal, start, end, identity,
        )
    with np.load(root / "bundle_trajectory.npz", allow_pickle=False) as data:
        frame_ids = data["frame_ids"]
        edges = data["endpoint_edges"]
        covariances = data["conditional_covariances"]
        poses = data["camera_to_first_keyframe"]
        extrinsics = data["world_to_camera"]
        positions = data["positions"]
    count = len(selected)
    if (
        not np.array_equal(frame_ids, np.asarray(selected, dtype=np.int64))
        or edges.shape != (count - 1, 4, 4)
        or covariances.shape != (count - 1, 6, 6)
        or poses.shape != (count, 4, 4)
        or extrinsics.shape != (count, 4, 4)
        or positions.shape != (count, 3)
        or not all(np.all(np.isfinite(value)) for value in (
            edges, covariances, poses, extrinsics, positions
        ))
    ):
        raise ContractError("Bundle aggregate artifact shapes differ")
    recomposed, inverse, recomposed_positions = _compose_bundle_trajectory(selected, edges)
    if (
        not np.allclose(poses, recomposed, rtol=0.0, atol=1e-9)
        or not np.allclose(extrinsics, inverse, rtol=0.0, atol=1e-9)
        or not np.allclose(positions, recomposed_positions, rtol=0.0, atol=1e-9)
    ):
        raise ContractError("Bundle aggregate trajectory does not reproduce endpoint edges")


def validate_bundle_artifacts(
    root: Path,
    keyframes: Sequence[int],
    *,
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    upstream_stage_ids: Mapping[str, str],
) -> None:
    """Structurally validate every durable window and the aggregate trajectory."""

    root = Path(root)
    manifest = load_stage_manifest(root / "manifest.json")
    validate_stage_manifest(
        manifest, root, expected_stage="bundle", sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("bundle"),
        upstream_stage_ids=upstream_stage_ids,
    )
    selected = tuple(int(value) for value in keyframes)
    if dict(manifest.counts) != {
        "keyframe_count": len(selected), "window_count": len(selected) - 1
    }:
        raise ContractError("Bundle manifest counts differ")
    _validate_bundle_payload(
        root, selected,
        identity=_stage_identity(dataset, config, upstream_stage_ids),
    )


def run_bundle_stage(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    tracking: TrackingOutput,
    keyframes: KeyframeOutput,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
    maximum_new_windows: Optional[int] = None,
) -> BundleOutput:
    """Stream windows with independent seals; an optional cap enables fault tests."""

    root = Path(root)
    identity = _stage_identity(dataset, config, upstream_stage_ids)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_stage_manifest(manifest_path)
        validate_stage_manifest(
            manifest, root, expected_stage="bundle", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("bundle"),
            upstream_stage_ids=upstream_stage_ids,
        )
        validate_bundle_artifacts(
            root, keyframes.selection.frame_ids, dataset=dataset, config=config,
            upstream_stage_ids=upstream_stage_ids,
        )
        return load_bundle_output(root, manifest)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = None
    if state_path.is_file():
        state = read_json(state_path)
        if any(state.get(key) != value for key, value in identity.items()):
            raise ContractError("Bundle resume state identity differs")
    keyframe_ids = keyframes.selection.frame_ids
    total_windows = len(keyframe_ids) - 1
    if state is None:
        state = dict(identity)
        state.update(
            {
                "status": "running",
                "window_count": total_windows,
                "completed_window_ordinals": [],
            }
        )
        write_json_atomic(state_path, state)
    elif int(state.get("window_count", -1)) != total_windows:
        raise ContractError("Bundle resume window population differs")

    bundle_config = config.section("bundle")
    calibration = _gtsam_calibration(dataset)
    projection_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(bundle_config["stereo_sigma_px"], dtype=float)
    )
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(bundle_config["prior_sigmas"], dtype=float)
    )
    eligibility = LandmarkEligibility(
        int(bundle_config["minimum_valid_observations"]),
        int(bundle_config["minimum_distinct_frames"]),
        float(bundle_config["maximum_initialization_depth_m"]),
    )
    windows_root = root / "windows"
    completed = []
    new_count = 0
    started = time.perf_counter()
    for ordinal, (start, end) in enumerate(zip(keyframe_ids, keyframe_ids[1:])):
        directory = windows_root / _window_name(ordinal, start, end)
        if directory.is_dir():
            validate_window(directory, ordinal, start, end, identity)
            completed.append(ordinal)
            continue
        if maximum_new_windows is not None and new_count >= maximum_new_windows:
            break
        result = solve_bundle_window(
            database=tracking.database,
            world_to_camera_poses=tracking.world_to_camera,
            calibration=calibration,
            start_frame_id=start,
            end_frame_id=end,
            projection_noise=projection_noise,
            anchor_noise=anchor_noise,
            landmark_eligibility=eligibility,
            rotation_atol=float(bundle_config["rotation_atol"]),
            graph_error_nonincrease_absolute=float(
                bundle_config["graph_error_nonincrease_absolute"]
            ),
            graph_error_nonincrease_relative=float(
                bundle_config["graph_error_nonincrease_relative"]
            ),
        )
        covariance = extract_bundle_covariances(
            result,
            symmetry_atol=float(bundle_config["covariance_symmetry_atol"]),
            minimum_eigenvalue=float(bundle_config["covariance_minimum_eigenvalue"]),
            rotation_atol=float(bundle_config["rotation_atol"]),
        )
        _seal_window(windows_root, ordinal, result, covariance, identity)
        completed.append(ordinal)
        new_count += 1
        state["completed_window_ordinals"] = completed
        write_json_atomic(state_path, state)
        del covariance
        del result

    state["completed_window_ordinals"] = completed
    if len(completed) != total_windows:
        state["status"] = "running"
        write_json_atomic(state_path, state)
        edges, covariances = _load_edges(windows_root, keyframe_ids[: len(completed) + 1])
        local, extrinsics, positions = _compose_bundle_trajectory(
            keyframe_ids[: len(completed) + 1], edges
        )
        return BundleOutput(
            None, np.asarray(keyframe_ids[: len(completed) + 1], dtype=np.int64),
            edges, covariances, local, extrinsics, positions,
        )

    edges, covariances = _load_edges(windows_root, keyframe_ids)
    local, extrinsics, positions = _compose_bundle_trajectory(keyframe_ids, edges)
    save_npz_atomic(
        root / "bundle_trajectory.npz",
        frame_ids=np.asarray(keyframe_ids, dtype=np.int64),
        endpoint_edges=edges,
        conditional_covariances=covariances,
        camera_to_first_keyframe=local,
        world_to_camera=extrinsics,
        positions=positions,
    )
    state["status"] = "complete"
    write_json_atomic(state_path, state)
    _validate_bundle_payload(root, keyframe_ids, identity=identity)
    manifest = StageManifest.complete(
        stage="bundle", sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("bundle"),
        upstream_stage_ids=upstream_stage_ids,
        counts={
            "keyframe_count": len(keyframe_ids),
            "window_count": total_windows,
        },
        primary_artifacts=("state.json", "bundle_trajectory.npz"),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    write_stage_manifest(root, manifest)
    return BundleOutput(
        manifest, np.asarray(keyframe_ids, dtype=np.int64), edges, covariances,
        local, extrinsics, positions,
    )


def load_bundle_output(
    root: Path, manifest: Optional[StageManifest] = None
) -> BundleOutput:
    root = Path(root)
    if manifest is None:
        manifest = load_stage_manifest(root / "manifest.json")
    with np.load(root / "bundle_trajectory.npz", allow_pickle=False) as data:
        return BundleOutput(
            manifest,
            data["frame_ids"].copy(),
            data["endpoint_edges"].copy(),
            data["conditional_covariances"].copy(),
            data["camera_to_first_keyframe"].copy(),
            data["world_to_camera"].copy(),
            data["positions"].copy(),
        )


__all__ = [
    "BundleOutput",
    "load_bundle_output",
    "run_bundle_stage",
    "validate_bundle_artifacts",
    "validate_window",
]
