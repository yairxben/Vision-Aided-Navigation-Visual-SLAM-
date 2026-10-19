"""Thin end-to-end orchestration and managed run identity."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from .artifacts.io import read_json, require_files, save_npy_atomic, save_npz_atomic, write_json_atomic
from .artifacts.manifest import StageManifest, load_stage_manifest, validate_stage_manifest, write_stage_manifest
from .config import (
    SCIENTIFIC_IMPLEMENTATION_REVISION,
    resolve_algorithm_config,
    write_resolved_config,
)
from .contracts import ContractError, GenericSlamError, PreparedRun, RunRequest
from .dataset.kitti import KittiStereoSequence, discover_kitti_sequence
from .dataset.ground_truth import WORLD_TO_CAMERA, normalize_ground_truth_convention


RUN_SCHEMA = "generic-slam-run-v1"
STAGE_ORDER = (
    "tracking", "keyframes", "bundle", "pose_graph", "loop_closure",
    "analysis", "report",
)


def _expected_run_identity(
    *, sequence_id: str, dataset_root: Path, dataset_id: str,
    config_fingerprint: str, run_mode: str, gt_input_convention: str,
) -> Dict[str, object]:
    return {
        "schema": RUN_SCHEMA,
        "sequence_id": sequence_id,
        "dataset_root": str(dataset_root),
        "dataset_id": dataset_id,
        "config_fingerprint": config_fingerprint,
        "run_mode": run_mode,
        "gt_input_convention": gt_input_convention,
        "scientific_implementation_revision": SCIENTIFIC_IMPLEMENTATION_REVISION,
    }


def _prepare_managed_root(root: Path, identity: Mapping[str, object]) -> Dict[str, object]:
    selected = Path(root).expanduser().resolve()
    if selected.exists() and not selected.is_dir():
        raise ContractError("output root exists and is not a directory")
    run_path = selected / "run.json"
    if selected.exists() and any(selected.iterdir()) and not run_path.is_file():
        raise ContractError("refusing unmanaged nonempty output root: {}".format(selected))
    selected.mkdir(parents=True, exist_ok=True)
    if run_path.is_file():
        raw = read_json(run_path)
        if not isinstance(raw, dict):
            raise ContractError("run.json must be a JSON object")
        observed_identity = {key: raw.get(key) for key in identity}
        if observed_identity != dict(identity):
            raise ContractError("existing run identity differs")
        return raw
    record = dict(identity)
    record.update({"status": "preparing", "stages": {}})
    write_json_atomic(run_path, record)
    return record


def _validate_dataset_stage(
    root: Path, dataset: KittiStereoSequence, config_fingerprint: str,
    manifest: StageManifest,
) -> None:
    validate_stage_manifest(
        manifest, root, expected_stage="dataset", sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id, config_fingerprint=config_fingerprint,
        upstream_stage_ids={},
    )
    require_files(root, ("details.json", "calibration.json", "frame_ids.npy"))
    if read_json(root / "details.json") != dataset.manifest_record():
        raise ContractError("existing dataset details differ")
    if read_json(root / "calibration.json") != dataset.calibration.record():
        raise ContractError("existing calibration artifact differs")
    frame_ids = np.load(root / "frame_ids.npy", allow_pickle=False)
    if not np.array_equal(frame_ids, np.asarray(dataset.frame_ids, dtype=np.int64)):
        raise ContractError("existing dataset FrameIds differ")
    gt_path = root / "ground_truth_world_to_camera.npz"
    if dataset.ground_truth_world_to_camera is None:
        if gt_path.exists():
            raise ContractError("unexpected Ground Truth artifact")
    else:
        require_files(root, ("ground_truth_world_to_camera.npz",))
        with np.load(gt_path, allow_pickle=False) as data:
            if (
                "input_convention" not in data.files
                or "internal_convention" not in data.files
                or not np.array_equal(data["frame_ids"], frame_ids)
                or data["world_to_camera"].shape != (dataset.frame_count, 4, 4)
                or not np.all(np.isfinite(data["world_to_camera"]))
                or not np.array_equal(
                    data["world_to_camera"], dataset.ground_truth_world_to_camera
                )
                or str(data["input_convention"].item())
                != dataset.ground_truth_input_convention
                or str(data["internal_convention"].item()) != WORLD_TO_CAMERA
            ):
                raise ContractError("Ground Truth artifact differs")


def _prepare_dataset_stage(
    output_root: Path, dataset: KittiStereoSequence, config_fingerprint: str
) -> StageManifest:
    root = output_root / "dataset"
    if (root / "manifest.json").is_file():
        manifest = load_stage_manifest(root / "manifest.json")
        _validate_dataset_stage(root, dataset, config_fingerprint, manifest)
        return manifest
    if root.exists():
        quarantine = root.with_name(root.name + ".failed-" + str(int(time.time_ns())))
        os.replace(str(root), str(quarantine))
    attempt = Path(tempfile.mkdtemp(prefix=".dataset-attempt-", dir=str(output_root)))
    started = time.perf_counter()
    try:
        write_json_atomic(attempt / "details.json", dataset.manifest_record())
        write_json_atomic(attempt / "calibration.json", dataset.calibration.record())
        save_npy_atomic(attempt / "frame_ids.npy", np.asarray(dataset.frame_ids, dtype=np.int64))
        primary = ["details.json", "calibration.json", "frame_ids.npy"]
        if dataset.ground_truth_world_to_camera is not None:
            save_npz_atomic(
                attempt / "ground_truth_world_to_camera.npz",
                frame_ids=np.asarray(dataset.frame_ids, dtype=np.int64),
                world_to_camera=np.asarray(dataset.ground_truth_world_to_camera, dtype=float),
                input_convention=np.asarray(dataset.ground_truth_input_convention),
                internal_convention=np.asarray(WORLD_TO_CAMERA),
            )
            primary.append("ground_truth_world_to_camera.npz")
        manifest = StageManifest.complete(
            stage="dataset", sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id, config_fingerprint=config_fingerprint,
            upstream_stage_ids={},
            counts={
                "frame_count": dataset.frame_count,
                "ignored_entry_count": len(dataset.ignored_left_entries) + len(dataset.ignored_right_entries),
                "ground_truth_pose_count": dataset.frame_count if dataset.ground_truth_world_to_camera is not None else 0,
            },
            primary_artifacts=primary,
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=0,
        )
        write_stage_manifest(attempt, manifest)
        _validate_dataset_stage(attempt, dataset, config_fingerprint, manifest)
        os.replace(str(attempt), str(root))
        return manifest
    except Exception:
        if attempt.exists():
            os.replace(str(attempt), str(attempt) + ".failed")
        raise


def prepare_run(request: RunRequest) -> PreparedRun:
    config_path = Path(request.config_path).expanduser().resolve() if request.config_path is not None else None
    config = resolve_algorithm_config(config_path)
    gt_convention = normalize_ground_truth_convention(request.gt_convention)
    dataset = discover_kitti_sequence(
        request.dataset_root,
        request.sequence_id,
        ground_truth_convention=gt_convention,
    )
    output_root = Path(request.output_root).expanduser().resolve()
    identity = _expected_run_identity(
        sequence_id=dataset.sequence_id, dataset_root=dataset.dataset_root,
        dataset_id=dataset.dataset_id, config_fingerprint=config.fingerprint,
        run_mode=config.mode, gt_input_convention=gt_convention,
    )
    run_record = _prepare_managed_root(output_root, identity)
    resolved_path = output_root / "resolved_config.json"
    if resolved_path.is_file():
        if read_json(resolved_path) != config.values:
            raise ContractError("existing resolved configuration differs")
    else:
        write_resolved_config(config, resolved_path)
    dataset_manifest = _prepare_dataset_stage(output_root, dataset, config.fingerprint)
    run_record = dict(run_record)
    stages = dict(run_record.get("stages", {}))
    stages["dataset"] = {"status": "complete", "stage_id": dataset_manifest.stage_id}
    run_record.update({
        "status": "ready", "frame_count": dataset.frame_count,
        "final_frame_id": dataset.final_frame_id,
        "ground_truth_available": dataset.ground_truth_path is not None,
        "stages": stages,
    })
    write_json_atomic(output_root / "run.json", run_record)
    return PreparedRun(
        request=RunRequest(
            dataset.dataset_root,
            dataset.sequence_id,
            output_root,
            config_path,
            gt_convention,
        ),
        normalized_sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id,
        config_fingerprint=config.fingerprint,
        run_mode=config.mode,
    )


def _run_stage_child(prepared: PreparedRun, stage: str) -> None:
    command = [
        sys.executable, "-m", "generic_slam._stage_runner", "--stage", stage,
        "--dataset-root", str(prepared.request.dataset_root),
        "--sequence", prepared.normalized_sequence_id,
        "--output-root", str(prepared.request.output_root),
    ]
    if stage in ("analysis", "report"):
        command.extend(("--gt-convention", prepared.request.gt_convention))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise GenericSlamError("stage {} failed with exit status {}".format(stage, completed.returncode))


def run_pipeline(request: RunRequest) -> PreparedRun:
    """Run or safely resume every scientific and reporting stage in order."""

    prepared = prepare_run(request)
    run_path = prepared.request.output_root / "run.json"
    for stage in STAGE_ORDER:
        record = read_json(run_path)
        stages = dict(record.get("stages", {}))
        previous_id = stages.get(stage, {}).get("stage_id")
        stages[stage] = {"status": "running", "stage_id": previous_id}
        record.update({"status": "running", "active_stage": stage, "stages": stages})
        write_json_atomic(run_path, record)
        try:
            _run_stage_child(prepared, stage)
            manifest = load_stage_manifest(prepared.request.output_root / stage / "manifest.json")
        except Exception as error:
            record = read_json(run_path)
            stages = dict(record.get("stages", {}))
            stages[stage] = {
                "status": "failed", "stage_id": previous_id,
                "error": "{}: {}".format(type(error).__name__, error),
            }
            record.update({"status": "failed", "active_stage": stage, "stages": stages})
            write_json_atomic(run_path, record)
            raise
        record = read_json(run_path)
        stages = dict(record.get("stages", {}))
        stages[stage] = {"status": "complete", "stage_id": manifest.stage_id}
        record.update({"status": "running", "active_stage": None, "stages": stages})
        write_json_atomic(run_path, record)
    record = read_json(run_path)
    record.update({"status": "complete", "active_stage": None})
    write_json_atomic(run_path, record)
    return prepared


__all__ = ["STAGE_ORDER", "prepare_run", "run_pipeline"]
