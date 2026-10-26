"""Cross-stage immutable contracts for the generic pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class GenericSlamError(RuntimeError):
    """Base exception for a rejected generic SLAM operation."""


class ContractError(GenericSlamError):
    """Raised when a public or cross-stage contract is invalid."""


@dataclass(frozen=True)
class RunRequest:
    """The complete runtime-owned input surface of a normal SLAM run."""

    dataset_root: Path
    sequence_id: str
    output_root: Path
    config_path: Optional[Path] = None
    gt_convention: str = "world-to-camera"
    # Earlier output root whose still-valid completed stages are linked in.
    reuse_from: Optional[Path] = None
    # Last stage to run; later stages are left for a future invocation.
    stop_after: Optional[str] = None


@dataclass(frozen=True)
class PreparedRun:
    """Validated immutable identity used by every scientific stage."""

    request: RunRequest
    normalized_sequence_id: str
    dataset_id: str
    config_fingerprint: str
    run_mode: str
