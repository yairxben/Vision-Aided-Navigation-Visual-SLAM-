"""Lightweight atomic artifacts and structural stage manifests."""

from .manifest import StageManifest, load_stage_manifest, write_stage_manifest

__all__ = ["StageManifest", "load_stage_manifest", "write_stage_manifest"]

