"""Profile one pipeline stage, reusing the earlier stages of an existing run.

The earlier stages are hard-linked from --source into a scratch output root,
then only --stage runs under cProfile. The source run is never modified.

Usage:
    python3.8 tools/profile_stage.py --source ~/slam_outputs/seq00_expA \
        --stage loop_closure --dataset-root ../dataset --sequence 00
"""

import argparse
import cProfile
import pstats
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from generic_slam.artifacts.io import read_json  # noqa: E402
from generic_slam.contracts import RunRequest  # noqa: E402
from generic_slam.pipeline import STAGE_ORDER, prepare_run  # noqa: E402


def _group(stats):
    """Self time per C/C++ call or per generic_slam source file."""

    groups = Counter()
    for (file, _line, name), (_cc, _nc, self_time, _ct, _callers) in stats.stats.items():
        if file == "~":
            key = "C/C++  " + (name.split("'")[1] if "'" in name else name)
        elif "generic_slam" in file:
            key = "python " + file.split("generic_slam/")[-1]
        else:
            key = "python (other) " + Path(file).name
        groups[key] += self_time
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="completed output root")
    parser.add_argument("--stage", required=True, choices=STAGE_ORDER)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    arguments = parser.parse_args()

    source = arguments.source.expanduser().resolve()
    work = (arguments.work_root or Path("/tmp/profile_{}".format(arguments.stage))).expanduser()
    if work.exists():
        shutil.rmtree(str(work))
    run = read_json(source / "run.json")
    gt_convention = run.get("gt_input_convention", "world-to-camera")
    # Link every valid stage from the source, then drop the profiled stage and
    # everything after it (removing links never touches the source files).
    prepare_run(RunRequest(
        dataset_root=arguments.dataset_root, sequence_id=arguments.sequence,
        output_root=work, config_path=source / "resolved_config.json",
        gt_convention=gt_convention, reuse_from=source,
    ))
    for stage in STAGE_ORDER[STAGE_ORDER.index(arguments.stage):]:
        if (work / stage).exists():
            shutil.rmtree(str(work / stage))
    missing = [s for s in STAGE_ORDER[:STAGE_ORDER.index(arguments.stage)] if not (work / s / "manifest.json").is_file()]
    if missing:
        raise SystemExit("cannot profile {}: earlier stages not reusable: {}".format(arguments.stage, missing))

    profile_path = work / "{}.prof".format(arguments.stage)
    command = [
        sys.executable, "-m", "cProfile", "-o", str(profile_path),
        "-m", "generic_slam._stage_runner", "--stage", arguments.stage,
        "--dataset-root", str(arguments.dataset_root.expanduser().resolve()),
        "--sequence", arguments.sequence, "--output-root", str(work),
    ]
    if arguments.stage in ("analysis", "report"):
        command += ["--gt-convention", gt_convention]
    print("profiling {} ...".format(arguments.stage), flush=True)
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise SystemExit("stage failed with exit status {}".format(completed.returncode))

    stats = pstats.Stats(str(profile_path))
    total = stats.total_tt
    print("\n{} total {:.1f} s (profiler adds some overhead)".format(arguments.stage, total))
    print("Self time by C/C++ call and by generic_slam file:")
    for key, value in _group(stats).most_common(arguments.top):
        print("  {:5.1f}%  {:8.1f} s  {}".format(100 * value / total, value, key))
    print("\nTop generic_slam functions by cumulative time:")
    stats.sort_stats("cumulative")
    rows = [
        (cumulative, calls, "{}:{} {}".format(file.split("generic_slam/")[-1], line, name))
        for (file, line, name), (_cc, calls, _tt, cumulative, _callers) in stats.stats.items()
        if "generic_slam" in file
    ]
    for cumulative, calls, label in sorted(rows, reverse=True)[: arguments.top]:
        print("  {:8.1f} s  {:8d} calls  {}".format(cumulative, calls, label))
    print("\nFull profile: {}  (open with: python3.8 -m pstats {})".format(profile_path, profile_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
