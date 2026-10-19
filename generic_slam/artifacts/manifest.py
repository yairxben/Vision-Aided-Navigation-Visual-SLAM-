"""Minimal stage identity used for completion and resume decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple
from uuid import uuid4

from ..contracts import ContractError
from .io import read_json, require_files, write_json_atomic


@dataclass(frozen=True)
class StageManifest:
    schema: str
    stage: str
    stage_id: str
    status: str
    sequence_id: str
    dataset_id: str
    config_fingerprint: str
    upstream_stage_ids: Mapping[str, str]
    counts: Mapping[str, int]
    primary_artifacts: Tuple[str, ...]
    runtime_seconds: float
    peak_rss_bytes: int

    @classmethod
    def complete(
        cls,
        *,
        stage: str,
        sequence_id: str,
        dataset_id: str,
        config_fingerprint: str,
        upstream_stage_ids: Mapping[str, str],
        counts: Mapping[str, int],
        primary_artifacts: Sequence[str],
        runtime_seconds: float,
        peak_rss_bytes: int,
    ) -> "StageManifest":
        return cls(
            schema="generic-slam-stage-v1",
            stage=stage,
            stage_id=uuid4().hex,
            status="complete",
            sequence_id=sequence_id,
            dataset_id=dataset_id,
            config_fingerprint=config_fingerprint,
            upstream_stage_ids=dict(upstream_stage_ids),
            counts={str(key): int(value) for key, value in counts.items()},
            primary_artifacts=tuple(str(value) for value in primary_artifacts),
            runtime_seconds=float(runtime_seconds),
            peak_rss_bytes=int(peak_rss_bytes),
        )

    def record(self) -> Dict[str, object]:
        value = asdict(self)
        value["primary_artifacts"] = list(self.primary_artifacts)
        return value


def validate_stage_manifest(
    manifest: StageManifest,
    root: Path,
    *,
    expected_stage: str,
    sequence_id: str,
    dataset_id: str,
    config_fingerprint: str,
    upstream_stage_ids: Mapping[str, str],
) -> None:
    if (
        manifest.schema != "generic-slam-stage-v1"
        or manifest.status != "complete"
        or manifest.stage != expected_stage
        or manifest.sequence_id != sequence_id
        or manifest.dataset_id != dataset_id
        or manifest.config_fingerprint != config_fingerprint
        or dict(manifest.upstream_stage_ids) != dict(upstream_stage_ids)
    ):
        raise ContractError("stage manifest identity differs for {}".format(expected_stage))
    if not manifest.stage_id:
        raise ContractError("stage manifest has no stage ID")
    if manifest.runtime_seconds < 0.0 or manifest.peak_rss_bytes < 0:
        raise ContractError("stage runtime metrics are invalid")
    if any(value < 0 for value in manifest.counts.values()):
        raise ContractError("stage counts must be nonnegative")
    for artifact in manifest.primary_artifacts:
        selected = Path(artifact)
        if selected.is_absolute() or ".." in selected.parts or not selected.parts:
            raise ContractError("stage artifact paths must remain relative to the stage")
    require_files(root, manifest.primary_artifacts)


def write_stage_manifest(root: Path, manifest: StageManifest) -> Path:
    path = Path(root) / "manifest.json"
    write_json_atomic(path, manifest.record())
    return path


def load_stage_manifest(path: Path) -> StageManifest:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ContractError("stage manifest must be a JSON object")
    try:
        return StageManifest(
            schema=str(raw["schema"]),
            stage=str(raw["stage"]),
            stage_id=str(raw["stage_id"]),
            status=str(raw["status"]),
            sequence_id=str(raw["sequence_id"]),
            dataset_id=str(raw["dataset_id"]),
            config_fingerprint=str(raw["config_fingerprint"]),
            upstream_stage_ids=dict(raw["upstream_stage_ids"]),
            counts={str(key): int(value) for key, value in raw["counts"].items()},
            primary_artifacts=tuple(str(item) for item in raw["primary_artifacts"]),
            runtime_seconds=float(raw["runtime_seconds"]),
            peak_rss_bytes=int(raw["peak_rss_bytes"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ContractError("invalid stage manifest: {}".format(error))


__all__ = [
    "StageManifest",
    "load_stage_manifest",
    "validate_stage_manifest",
    "write_stage_manifest",
]
