"""Small atomic persistence helpers; deliberately not a provenance framework."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    return Path(name)


def write_json_atomic(path: Path, value: object) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv_atomic(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            # Same output as csv.DictWriter (missing keys become empty fields)
            # without its per-row dictionary rebuild.
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(fieldnames)
            writer.writerows([row.get(name) for name in fieldnames] for row in rows)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
        os.replace(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def require_files(root: Path, relative_paths: Sequence[str]) -> None:
    missing = [name for name in relative_paths if not (Path(root) / name).is_file()]
    if missing:
        raise FileNotFoundError("missing required artifacts: {}".format(", ".join(missing)))


__all__ = [
    "read_json",
    "require_files",
    "save_npy_atomic",
    "save_npz_atomic",
    "write_csv_atomic",
    "write_json_atomic",
]

