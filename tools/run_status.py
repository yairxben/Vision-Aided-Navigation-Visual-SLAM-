"""Show the progress of a generic_slam run (read-only; safe while it runs).

Usage:
    python3.8 tools/run_status.py ~/slam_outputs/seq00_akaze
    python3.8 tools/run_status.py ~/slam_outputs/seq00_akaze --watch 30
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

STAGES = (
    "tracking", "keyframes", "bundle", "pose_graph", "loop_closure", "analysis", "report",
)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _keyframe_count(root):
    path = root / "keyframes" / "keyframes.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle) - 1


def _progress(root, stage, frame_count, keyframe_count):
    """(done, total) units of work for stages that report progress on disk."""

    if stage == "tracking":
        # Tracking works in a hidden attempt directory until it completes.
        attempts = sorted(root.glob(".tracking-attempt-*/transition_evidence"))
        done = len(list(attempts[-1].glob("*.npz"))) if attempts else 0
        return done, (frame_count - 1 if frame_count else None)
    if stage == "bundle":
        done = len(list((root / "bundle" / "windows").glob("*/seal.json")))
        return done, (keyframe_count - 1 if keyframe_count else None)
    if stage == "loop_closure":
        currents = root / "loop_closure" / "currents"
        done = len(list(currents.iterdir())) if currents.is_dir() else 0
        return done, (keyframe_count - 1 if keyframe_count else None)
    return None


def _bar(done, total, width=30):
    if not total:
        return "{} done".format(done)
    filled = int(width * min(done, total) / total)
    return "[{}{}] {}/{} ({:.0f}%)".format(
        "#" * filled, "." * (width - filled), done, total, 100.0 * done / total
    )


def status(root):
    run = _read_json(root / "run.json") or {}
    frame_count = run.get("frame_count")
    keyframe_count = _keyframe_count(root)
    lines = ["{}  {}".format(datetime.now().strftime("%H:%M:%S"), root)]
    total_minutes = 0.0
    for stage in STAGES:
        manifest = _read_json(root / stage / "manifest.json")
        if manifest is not None:
            minutes = float(manifest.get("runtime_seconds", 0.0)) / 60.0
            total_minutes += minutes
            lines.append("  {:<13} done         {:6.1f} min".format(stage, minutes))
            continue
        started = (root / stage).exists() or any(root.glob(".{}-attempt-*".format(stage)))
        if not started:
            lines.append("  {:<13} not started".format(stage))
            continue
        progress = _progress(root, stage, frame_count, keyframe_count)
        detail = _bar(*progress) if progress else ""
        lines.append("  {:<13} RUNNING      {}".format(stage, detail))
    if (root / "report" / "manifest.json").is_file():
        lines.append("  finished: {:.1f} min of stage time".format(total_minutes))
    failed = sorted(path.name for path in root.glob("*.failed"))
    failed += sorted(path.name for path in root.glob("*/*.failed"))
    if failed:
        lines.append("  FAILED attempts: {}".format(", ".join(failed)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--watch", type=float, metavar="SECONDS",
                        help="refresh every SECONDS until the run finishes")
    arguments = parser.parse_args()
    root = arguments.output_root.expanduser()
    while True:
        text = status(root)
        if arguments.watch:
            print("\033[2J\033[H", end="")
        print(text, flush=True)
        if not arguments.watch or (root / "report" / "manifest.json").is_file():
            return 0
        time.sleep(arguments.watch)


if __name__ == "__main__":
    raise SystemExit(main())
