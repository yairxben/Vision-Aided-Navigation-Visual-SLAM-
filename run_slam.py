"""Run the generic stereo SLAM pipeline.

Example:
    python run_slam.py --dataset-root ../dataset --sequence 00 --output-root ~/slam_outputs/seq00

All intermediate stage data goes to ``--output-root`` (several GB for a full
sequence). After a successful run the readable results (``summary.md``,
figures, statistics and the configuration used) are copied to
``results/<output-root name>_<timestamp>/`` in this project, or to
``--results-dir`` if given. See ``--help`` for all pipeline options.
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from generic_slam.cli import main

PROJECT_ROOT = Path(__file__).resolve().parent

# Relative to the output root; small, human-readable artifacts only.
PUBLISHED_ARTIFACTS = (
    "report",
    "analysis/summary.json",
    "analysis/tracking/track_statistics.json",
    "resolved_config.json",
    "run.json",
)


def publish_results(output_root: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=False)
    for relative in PUBLISHED_ARTIFACTS:
        source = output_root / relative
        if source.is_dir():
            shutil.copytree(str(source), str(results_dir / source.name))
        elif source.is_file():
            shutil.copy2(str(source), str(results_dir / source.name))
    print("results copied to {}".format(results_dir))


def run(argv=None) -> int:
    pipeline_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-dir", type=Path)
    known, _ = parser.parse_known_args(pipeline_argv)
    # --results-dir belongs to this wrapper; do not forward it.
    for index, value in enumerate(pipeline_argv):
        if value == "--results-dir":
            del pipeline_argv[index:index + 2]
            break
        if value.startswith("--results-dir="):
            del pipeline_argv[index]
            break

    status = main(pipeline_argv)
    if status != 0 or known.output_root is None:
        return status
    output_root = known.output_root.expanduser().resolve()
    if not (output_root / "report" / "manifest.json").is_file():
        return status  # stopped early (--stop-after): nothing to publish yet
    results_dir = known.results_dir or (
        PROJECT_ROOT / "results" / "{}_{}".format(
            output_root.name, datetime.now().strftime("%Y-%m-%d_%H%M")
        )
    )
    publish_results(output_root, results_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(run())
