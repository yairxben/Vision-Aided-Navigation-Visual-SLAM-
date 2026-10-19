"""Artifact boundary for runtime-derived keyframes."""

from __future__ import annotations

import csv
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
from uuid import uuid4

import numpy as np

from ..artifacts.io import write_csv_atomic
from ..artifacts.manifest import (
    StageManifest,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)
from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiEstimationContext
from ..tracking.stage import TrackingTrajectoryOutput
from .selection import KeyframeSelection, select_keyframes, validate_keyframes


KEYFRAME_FIELDS = (
    "keyframe_ordinal", "frame_id", "previous_frame_id", "window_frame_count",
    "pnp_path_distance_m", "pnp_relative_rotation_deg", "selection_reason",
)


@dataclass(frozen=True)
class KeyframeOutput:
    manifest: StageManifest
    selection: KeyframeSelection


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_selection(path: Path) -> KeyframeSelection:
    from .selection import KeyframeDecision

    rows = _load_rows(path)
    if not rows:
        raise ContractError("keyframe CSV is empty")
    frame_ids = tuple(int(row["frame_id"]) for row in rows)
    decisions = tuple(
        KeyframeDecision(
            ordinal=int(row["keyframe_ordinal"]),
            frame_id=int(row["frame_id"]),
            previous_frame_id=int(row["previous_frame_id"]),
            window_frame_count=int(row["window_frame_count"]),
            path_distance_m=float(row["pnp_path_distance_m"]),
            relative_rotation_deg=float(row["pnp_relative_rotation_deg"]),
            reason=row["selection_reason"],
        )
        for row in rows[1:]
    )
    return KeyframeSelection(frame_ids, decisions)


def validate_keyframe_artifacts(root: Path, final_frame_id: int) -> None:
    selection = _load_selection(Path(root) / "keyframes.csv")
    validate_keyframes(selection.frame_ids, final_frame_id)
    if len(selection.decisions) != len(selection.frame_ids) - 1:
        raise ContractError("keyframe decision population differs")


def run_keyframe_stage(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    tracking: TrackingTrajectoryOutput,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
) -> KeyframeOutput:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_stage_manifest(manifest_path)
        validate_stage_manifest(
            manifest, root, expected_stage="keyframes",
            sequence_id=dataset.sequence_id, dataset_id=dataset.dataset_id,
            config_fingerprint=config.fingerprint,
            upstream_stage_ids=upstream_stage_ids,
        )
        validate_keyframe_artifacts(root, dataset.final_frame_id)
        selection = _load_selection(root / "keyframes.csv")
        if dict(manifest.counts) != {
            "keyframe_count": len(selection.frame_ids),
            "window_count": len(selection.frame_ids) - 1,
        }:
            raise ContractError("keyframe manifest counts differ")
        return KeyframeOutput(manifest, selection)
    if root.exists():
        os.replace(
            str(root), str(root.with_name(root.name + ".failed-" + uuid4().hex))
        )
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    selection = select_keyframes(
        tracking.world_to_camera, config.section("keyframes")
    )
    rows = [
        {
            "keyframe_ordinal": 0,
            "frame_id": 0,
            "previous_frame_id": "",
            "window_frame_count": 1,
            "pnp_path_distance_m": 0.0,
            "pnp_relative_rotation_deg": 0.0,
            "selection_reason": "initial",
        }
    ]
    rows.extend(
        {
            "keyframe_ordinal": item.ordinal,
            "frame_id": item.frame_id,
            "previous_frame_id": item.previous_frame_id,
            "window_frame_count": item.window_frame_count,
            "pnp_path_distance_m": item.path_distance_m,
            "pnp_relative_rotation_deg": item.relative_rotation_deg,
            "selection_reason": item.reason,
        }
        for item in selection.decisions
    )
    write_csv_atomic(root / "keyframes.csv", rows, KEYFRAME_FIELDS)
    validate_keyframe_artifacts(root, dataset.final_frame_id)
    manifest = StageManifest.complete(
        stage="keyframes", sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id, config_fingerprint=config.fingerprint,
        upstream_stage_ids=upstream_stage_ids,
        counts={
            "keyframe_count": len(selection.frame_ids),
            "window_count": len(selection.frame_ids) - 1,
        },
        primary_artifacts=("keyframes.csv",),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    )
    write_stage_manifest(root, manifest)
    return KeyframeOutput(manifest, selection)


def load_keyframe_output(root: Path) -> KeyframeOutput:
    root = Path(root)
    return KeyframeOutput(
        load_stage_manifest(root / "manifest.json"),
        _load_selection(root / "keyframes.csv"),
    )
