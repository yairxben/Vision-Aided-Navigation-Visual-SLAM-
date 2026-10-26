"""Reuse completed stages of an earlier run whose relevant configuration matches."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Sequence

from ..config import (
    SCIENTIFIC_IMPLEMENTATION_REVISION,
    AlgorithmConfig,
    load_resolved_config,
)
from ..contracts import ContractError
from .io import read_json
from .manifest import load_stage_manifest


def _link_or_copy(source: str, destination: str) -> None:
    # Hard links share the data on disk; stages never modify sealed files in
    # place (they write a new file and rename), so the source run is safe.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _replace_fingerprint(root: Path, old: str, new: str) -> int:
    """Rewrite JSON identity fields as new files, leaving linked originals intact."""

    rewritten = 0
    for path in root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        if old not in text:
            continue
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text.replace(old, new))
        os.replace(temporary, str(path))
        rewritten += 1
    return rewritten


def reuse_completed_stages(
    source_root: Path,
    output_root: Path,
    config: AlgorithmConfig,
    dataset_id: str,
    stages: Sequence[str],
) -> List[str]:
    """Link the longest valid prefix of completed source stages into output_root.

    A stage is reused only if every earlier stage was reused too (stages record
    their upstream stage IDs) and its stage fingerprint, which covers only the
    configuration it and its upstream stages read, is unchanged.
    """

    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if source_root == output_root:
        raise ContractError("--reuse-from must name a different output root")
    run = read_json(source_root / "run.json")
    if not isinstance(run, dict):
        raise ContractError("reuse source has no run.json: {}".format(source_root))
    if run.get("dataset_id") != dataset_id:
        raise ContractError("reuse source was run on a different dataset or sequence")
    if run.get("scientific_implementation_revision") != SCIENTIFIC_IMPLEMENTATION_REVISION:
        raise ContractError(
            "reuse source was produced by a different implementation revision ({})".format(
                run.get("scientific_implementation_revision")
            )
        )
    source_config = load_resolved_config(
        source_root / "resolved_config.json", str(run.get("run_mode"))
    )

    reused: List[str] = []
    for stage in stages:
        target = output_root / stage
        if (target / "manifest.json").is_file():
            reused.append(stage)  # already present from an earlier invocation
            continue
        if target.exists():
            break  # a partial attempt exists here; do not mix runs
        manifest_path = source_root / stage / "manifest.json"
        if not manifest_path.is_file():
            break
        expected = config.stage_fingerprint(stage)
        if source_config.stage_fingerprint(stage) != expected:
            break
        recorded = load_stage_manifest(manifest_path).config_fingerprint
        # Runs made before per-stage fingerprints recorded the whole-config one.
        if recorded not in (expected, source_config.fingerprint):
            break
        staging = Path(tempfile.mkdtemp(prefix=".reuse-{}-".format(stage), dir=str(output_root)))
        try:
            staging.rmdir()
            shutil.copytree(str(source_root / stage), str(staging), copy_function=_link_or_copy)
            if recorded != expected:
                _replace_fingerprint(staging, recorded, expected)
            os.replace(str(staging), str(target))
        except Exception:
            shutil.rmtree(str(staging), ignore_errors=True)
            raise
        reused.append(stage)
    return reused


__all__ = ["reuse_completed_stages"]
