"""The single user-facing generic SLAM command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .contracts import GenericSlamError, RunRequest
from .dataset.ground_truth import GROUND_TRUTH_CONVENTIONS, WORLD_TO_CAMERA
from .pipeline import STAGE_ORDER, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m generic_slam",
        description="Run sequence-generic KITTI stereo SLAM.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional partial JSON overrides for algorithm parameters",
    )
    parser.add_argument(
        "--gt-convention",
        choices=GROUND_TRUTH_CONVENTIONS,
        default=WORLD_TO_CAMERA,
        help=(
            "input convention of poses/<sequence>.txt for evaluation only; "
            "default: world-to-camera (course/lecturer convention). Use "
            "camera-to-world for original KITTI odometry pose files"
        ),
    )
    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help=(
            "earlier output root: completed stages whose configuration is "
            "unchanged are hard-linked in instead of recomputed"
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default=None,
        help="run only up to and including this stage; rerun later to continue",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        prepared = run_pipeline(
            RunRequest(
                dataset_root=arguments.dataset_root,
                sequence_id=arguments.sequence,
                output_root=arguments.output_root,
                config_path=arguments.config,
                gt_convention=arguments.gt_convention,
                reuse_from=arguments.reuse_from,
                stop_after=arguments.stop_after,
            )
        )
    except GenericSlamError as error:
        print("generic_slam: {}".format(error))
        return 2
    print(
        "generic_slam {} sequence {} at {}".format(
            "stopped after {} for".format(arguments.stop_after)
            if arguments.stop_after else "completed",
            prepared.normalized_sequence_id, prepared.request.output_root,
        )
    )
    return 0


__all__ = ["build_parser", "main"]
