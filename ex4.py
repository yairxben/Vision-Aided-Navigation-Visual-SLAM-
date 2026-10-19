"""Canonical Exercise 4 TrackingDB build and analysis command line.

Reader-facing Exercise 4 route:

1. ``exercise4_database_builder.build_exercise4_database`` runs the actual
   stereo extraction -> temporal matching -> four-view RANSAC -> final-
   supporter TrackingDB insertion pipeline.
2. ``exercise4_workflow`` safely serializes that result and analyzes the
   independently reloaded TrackingDB.
3. This entrypoint preserves the process-isolated ``build -> analyze`` CLI.

The thin wrappers below make the route explicit while retaining historical
``ex4._run_build`` and ``ex4._run_analyze`` callers.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence


ROOT = Path(__file__).resolve().parent
FINAL_DIAGNOSTICS = (ROOT / "diagnostics" / "ex4_final").resolve()
MAX_FRAME_COUNT = 3360
MAX_PER_INSERTION_VALIDATION_FRAMES = 150
SEED = 0


def _require_new_or_empty_root(path: Path) -> None:
    """Compatibility wrapper for existing Exercise 4 orchestration callers."""

    from exercise4_workflow import _require_new_or_empty_root as require_root

    require_root(path)


def _run_build(
    frame_count: int,
    output_root: Path,
    validate_every_insertion: bool,
) -> Dict[str, object]:
    """Enter the scientific TrackingDB build through the workflow boundary."""

    from exercise4_workflow import build_and_serialize_tracking_database

    return build_and_serialize_tracking_database(
        frame_count,
        output_root,
        validate_every_insertion,
    )


def _run_analyze(
    database_path: Path,
    companion_path: Path,
    output_root: Path,
    supporter_threshold_percent: float,
) -> Dict[str, object]:
    """Enter serialized TrackingDB analysis through the workflow boundary."""

    from exercise4_workflow import analyze_serialized_tracking_database

    return analyze_serialized_tracking_database(
        database_path,
        companion_path,
        output_root,
        supporter_threshold_percent,
    )


def _validate_arguments(args) -> None:
    """Validate command-specific bounds before creating any output."""

    if args.command in ("build", "all"):
        if not 2 <= args.frame_count <= MAX_FRAME_COUNT:
            raise ValueError(
                "frame count must be in [2, {}]".format(MAX_FRAME_COUNT)
            )
        if (
            args.validate_every_insertion
            and args.frame_count > MAX_PER_INSERTION_VALIDATION_FRAMES
        ):
            raise ValueError(
                "--validate-every-insertion is limited to at most {} frames"
                .format(MAX_PER_INSERTION_VALIDATION_FRAMES)
            )
        if args.command == "all" and args.frame_count < 10:
            raise ValueError(
                "all requires at least 10 frames for the Exercise 4 "
                "reprojection artifact"
            )
    if args.command in ("analyze", "all"):
        if not math.isfinite(args.supporter_threshold_percent) or not (
            0.0 <= args.supporter_threshold_percent <= 100.0
        ):
            raise ValueError(
                "supporter threshold must be in [0, 100]"
            )


def _run_all(args) -> None:
    """Run build and serialized analysis in separate child processes."""

    output_root = args.output_dir.resolve()
    _require_new_or_empty_root(output_root)
    build_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "build",
        "--frame-count",
        str(args.frame_count),
        "--output-dir",
        str(output_root),
    ]
    if args.validate_every_insertion:
        build_command.append("--validate-every-insertion")
    build_environment = dict(os.environ)
    build_environment["EX4_ORCHESTRATED_PHASE"] = "build"
    build_process = subprocess.Popen(
        build_command, cwd=str(ROOT), env=build_environment
    )
    build_pid = build_process.pid
    build_returncode = build_process.wait()
    if build_returncode != 0:
        raise subprocess.CalledProcessError(
            build_returncode, build_command
        )

    database_path = output_root / "build" / "tracking_database.pkl"
    companion_path = (
        output_root / "build" / "tracking_database_companion.json"
    )
    if not database_path.is_file() or not companion_path.is_file():
        raise AssertionError("build process did not produce required files")

    analyze_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "analyze",
        "--database",
        str(database_path),
        "--companion",
        str(companion_path),
        "--output-dir",
        str(output_root),
        "--supporter-threshold-percent",
        str(args.supporter_threshold_percent),
    ]
    analyze_environment = dict(os.environ)
    analyze_environment["EX4_ORCHESTRATED_PHASE"] = "analyze"
    analyze_process = subprocess.Popen(
        analyze_command, cwd=str(ROOT), env=analyze_environment
    )
    analyze_pid = analyze_process.pid
    analyze_returncode = analyze_process.wait()
    if analyze_returncode != 0:
        raise subprocess.CalledProcessError(
            analyze_returncode, analyze_command
        )
    if len({os.getpid(), build_pid, analyze_pid}) != 3:
        raise AssertionError("all phases did not use distinct processes")
    print(
        "E4_ALL_PROCESS_ISOLATION=parent:{},build:{},analyze:{}".format(
            os.getpid(), build_pid, analyze_pid
        )
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the Exercise 4 ``build``, ``analyze``, and ``all`` CLI."""

    parser = argparse.ArgumentParser(
        description="Canonical Exercise 4 TrackingDB workflow"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build", help="build and serialize a canonical TrackingDB"
    )
    build_parser.add_argument("--frame-count", type=int, required=True)
    build_parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    build_parser.add_argument(
        "--validate-every-insertion",
        action="store_true",
        help=(
            "use bounded per-insertion validation and lecturer final "
            "consistency (maximum 150 frames)"
        ),
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="analyze an existing serialized TrackingDB"
    )
    analyze_parser.add_argument("--database", type=Path, required=True)
    analyze_parser.add_argument("--companion", type=Path, required=True)
    analyze_parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    analyze_parser.add_argument(
        "--supporter-threshold-percent", type=float, default=30.0
    )

    all_parser = subparsers.add_parser(
        "all",
        help=(
            "orchestrate build and analyze in separate fresh processes"
        ),
        description=(
            "orchestrate build and analyze in separate fresh processes; "
            "the build process must exit before analysis starts"
        ),
    )
    all_parser.add_argument("--frame-count", type=int, required=True)
    all_parser.add_argument("--output-dir", type=Path, required=True)
    all_parser.add_argument(
        "--validate-every-insertion",
        action="store_true",
        help=(
            "use bounded per-insertion validation and lecturer final "
            "consistency (maximum 150 frames)"
        ),
    )
    all_parser.add_argument(
        "--supporter-threshold-percent", type=float, default=30.0
    )
    return parser


def main(argv: Sequence[str] = None) -> None:
    """Validate and dispatch one Exercise 4 command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_arguments(args)
        if args.command == "build":
            _run_build(
                args.frame_count,
                args.output_dir.resolve(),
                args.validate_every_insertion,
            )
        elif args.command == "analyze":
            _run_analyze(
                args.database,
                args.companion,
                args.output_dir.resolve(),
                args.supporter_threshold_percent,
            )
        else:
            _run_all(args)
    except (AssertionError, FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
