"""Exercise 4 build/analysis workflow and artifact materialization.

The scientific TrackingDB transition pipeline remains in
``exercise4_database_builder.build_exercise4_database``.  This module owns
the surrounding output-safety, serialization, provenance, and serialized
analysis workflow used by the ``ex4.py`` command line.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Sequence


ROOT = Path(__file__).resolve().parent
FINAL_DIAGNOSTICS = (ROOT / "diagnostics" / "ex4_final").resolve()
MAX_PER_INSERTION_VALIDATION_FRAMES = 150
SEED = 0


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _reject_protected_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == FINAL_DIAGNOSTICS or _is_relative_to(
        resolved, FINAL_DIAGNOSTICS
    ):
        raise ValueError(
            "output must not modify diagnostics/ex4_final"
        )


def _require_new_or_empty_root(path: Path) -> None:
    """Require a safe, nonexistent or empty Exercise 4 output root."""

    _reject_protected_output(path)
    if path.exists():
        if not path.is_dir():
            raise FileExistsError("output root is not a directory")
        if any(path.iterdir()):
            raise FileExistsError(
                "output root must be nonexistent or empty"
            )


def _prepare_build_directory(output_root: Path) -> Path:
    _require_new_or_empty_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    build_dir = output_root / "build"
    build_dir.mkdir()
    return build_dir


def _prepare_analysis_directory(
    output_root: Path, database_path: Path, companion_path: Path
) -> Path:
    _reject_protected_output(output_root)
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError("output root is not a directory")
    if not output_root.exists():
        output_root.mkdir(parents=True)
    entries = sorted(path.name for path in output_root.iterdir())
    if entries:
        expected_build = (output_root / "build").resolve()
        inputs_are_owned_build = all(
            _is_relative_to(path.resolve(), expected_build)
            for path in (database_path, companion_path)
        )
        if entries != ["build"] or not inputs_are_owned_build:
            raise FileExistsError(
                "nonempty analyze root is allowed only for its own build/ "
                "artifacts during all orchestration"
            )
    analysis_dir = output_root / "analysis"
    if analysis_dir.exists():
        raise FileExistsError("analysis output already exists")
    return analysis_dir


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return completed.stdout.strip()


def _camera_centers(poses: Sequence[object]):
    import numpy as np

    centers = []
    for pose in poses:
        normalized = np.asarray(pose, dtype=float)
        rotation = normalized[:, :3]
        translation = normalized[:, 3:]
        centers.append((-rotation.T @ translation).reshape(3).tolist())
    return centers


def build_and_serialize_tracking_database(
    frame_count: int,
    output_root: Path,
    validate_every_insertion: bool,
) -> Dict[str, object]:
    """Run the scientific builder and materialize its serialized interface.

    Scientific frame processing is delegated to
    ``build_exercise4_database``. This workflow owns the safe output root,
    TrackingDB serialization, companion metadata, transition table, and build
    summary.
    """

    from exercise4_analysis import (
        PRODUCTION_COMPANION_SCHEMA,
        PRODUCTION_COMPANION_VERSION,
        STREAMING_DATABASE_DIGEST_SCHEMA,
        canonical_sha256,
        file_sha256,
        flatten_transition,
        numeric_summary,
        runtime_free_transition,
        streaming_database_digest,
        write_csv,
        write_json,
        TRANSITION_FIELDS,
    )
    from exercise4_database_builder import (
        COMPANION_SCHEMA,
        build_exercise4_database,
        database_digest,
    )

    build_dir = _prepare_build_directory(output_root)
    final_consistency_mode = (
        "lecturer" if validate_every_insertion else "streaming"
    )
    # Run the sequential stereo/temporal/PnP TrackingDB construction.
    result = build_exercise4_database(
        frame_count,
        seed=SEED,
        validate_after_every_insertion=validate_every_insertion,
        checkpoint_frame_id=None,
        final_consistency_mode=final_consistency_mode,
        transition_observer=None,
    )
    database_streaming_digest = streaming_database_digest(result.database)
    bounded_database_digest = (
        database_digest(result.database)
        if frame_count <= MAX_PER_INSERTION_VALIDATION_FRAMES
        else None
    )

    # Serialize the lecturer database before packaging companion evidence.
    database_base = build_dir / "tracking_database"
    database_path = database_base.with_suffix(".pkl")
    result.database.serialize(str(database_base))
    database_identity = {
        "path": "tracking_database.pkl",
        "size_bytes": database_path.stat().st_size,
        "file_sha256": file_sha256(database_path),
        "streaming_digest_schema": STREAMING_DATABASE_DIGEST_SCHEMA,
        "streaming_digest": database_streaming_digest,
        "bounded_database_digest": bounded_database_digest,
        "serialization_reload_policy": (
            "reload only in a separate analyze process"
        ),
    }

    runtime_free_transitions = [
        runtime_free_transition(row) for row in result.transitions
    ]
    transition_rows = [
        flatten_transition(row) for row in runtime_free_transitions
    ]
    write_csv(
        build_dir / "transitions.csv",
        TRANSITION_FIELDS,
        transition_rows,
    )
    tracking_state = result.companion.as_dict()
    centers = _camera_centers(result.companion.world_to_camera_poses)
    scientific_state_sha256 = canonical_sha256(
        {
            "database_streaming_digest": database_streaming_digest,
            "tracking_state": tracking_state,
            "camera_centers": centers,
            "frame_metadata": {
                "raw_left_feature_counts": result.raw_left_feature_counts,
                "raw_right_feature_counts": result.raw_right_feature_counts,
                "raw_stereo_match_counts": result.raw_stereo_match_counts,
                "stereo_row_counts": result.stereo_row_counts,
            },
            "transitions": runtime_free_transitions,
        }
    )
    source_names = (
        "ex4.py",
        "exercise4_workflow.py",
        "exercise4_analysis.py",
        "exercise4_database_builder.py",
        "tracking_database_adapter.py",
        "tracking_database.py",
        "tools.py",
    )
    from exercise4_analysis import ROOT as ANALYSIS_ROOT

    companion = {
        "schema": PRODUCTION_COMPANION_SCHEMA,
        "schema_version": PRODUCTION_COMPANION_VERSION,
        "work_package": "E4-I04",
        "artifact_role": "canonical production TrackingDB companion",
        "status": "build-complete-awaiting-independent-analysis",
        "frame_range_inclusive": [0, frame_count - 1],
        "frame_count": frame_count,
        "transition_count": frame_count - 1,
        "pose_convention": (
            "world-to-camera: x_i = R_i X + t_i; "
            "C_i = -R_i.T @ t_i"
        ),
        "configuration": result.provenance,
        "consistency": {
            "validate_after_every_insertion": validate_every_insertion,
            "final_consistency_mode": final_consistency_mode,
            "checked_after_frame_ids": (
                result.consistency_checked_after_frame_ids
            ),
        },
        "provenance": {
            "git_head": _git_head(),
            "source_sha256": {
                name: file_sha256(ANALYSIS_ROOT / name)
                for name in source_names
            },
            "input_sha256": {
                "dataset/sequences/00/calib.txt": file_sha256(
                    ROOT / "dataset" / "sequences" / "00" / "calib.txt"
                ),
                "dataset/poses/00.txt": file_sha256(
                    ROOT / "dataset" / "poses" / "00.txt"
                ),
            },
            "command_contract": (
                "python ex4.py build --frame-count {} --output-dir <DIR>{}"
                .format(
                    frame_count,
                    (
                        " --validate-every-insertion"
                        if validate_every_insertion
                        else ""
                    ),
                )
            ),
        },
        "tracking_state_schema": COMPANION_SCHEMA,
        "tracking_state": tracking_state,
        "camera_centers": centers,
        "frame_metadata": {
            "raw_left_feature_counts": result.raw_left_feature_counts,
            "raw_right_feature_counts": result.raw_right_feature_counts,
            "raw_stereo_match_counts": result.raw_stereo_match_counts,
            "stereo_row_counts": result.stereo_row_counts,
        },
        "transition_metadata": runtime_free_transitions,
        "database": database_identity,
        "scientific_state_sha256": scientific_state_sha256,
        "agents_md_modified": False,
    }
    # Materialize poses, supporters, transition evidence, and artifact identity.
    companion_path = build_dir / "tracking_database_companion.json"
    write_json(companion_path, companion)

    transition_runtimes = [
        float(row["runtime_seconds"]) for row in result.transitions
    ]
    summary = {
        "schema": "exercise4-production-build-summary-v1",
        "work_package": "E4-I04",
        "status": "build-complete",
        "frame_count": frame_count,
        "transition_count": frame_count - 1,
        "seed": SEED,
        "process": {
            "pid": os.getpid(),
            "orchestrated_phase": os.environ.get(
                "EX4_ORCHESTRATED_PHASE"
            ),
        },
        "consistency": companion["consistency"],
        "runtime": {
            "total_seconds": float(result.total_runtime_seconds),
            "initial_frame_seconds": float(
                result.initial_frame_runtime_seconds
            ),
            "transition_seconds": numeric_summary(transition_runtimes),
        },
        "database": database_identity,
        "companion": {
            "path": "tracking_database_companion.json",
            "file_sha256": file_sha256(companion_path),
        },
        "scientific_state_sha256": scientific_state_sha256,
        "serialization_reload_performed_in_build_process": False,
    }
    write_json(build_dir / "summary.json", summary)
    print("E4_BUILD_PROCESS_PID={}".format(os.getpid()))
    print(
        "Exercise 4 build complete: {} frames, {}".format(
            frame_count, build_dir
        )
    )
    return summary


def analyze_serialized_tracking_database(
    database_path: Path,
    companion_path: Path,
    output_root: Path,
    supporter_threshold_percent: float,
) -> Dict[str, object]:
    """Analyze a verified serialized TrackingDB in its analysis process.

    The analysis path receives artifact paths rather than an in-memory build
    result, preserving the explicit serialization/reload boundary.
    """

    from exercise4_analysis import analyze_tracking_database

    database_path = database_path.resolve()
    companion_path = companion_path.resolve()
    analysis_dir = _prepare_analysis_directory(
        output_root, database_path, companion_path
    )
    print("E4_ANALYZE_PROCESS_PID={}".format(os.getpid()))
    # Reload, verify, and analyze only the serialized Exercise 4 interface.
    result = analyze_tracking_database(
        database_path,
        companion_path,
        analysis_dir,
        supporter_threshold_percent=supporter_threshold_percent,
    )
    print("Exercise 4 analysis complete: {}".format(analysis_dir))
    return result
