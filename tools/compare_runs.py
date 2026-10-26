"""Check that two generic_slam output roots contain identical scientific results.

Every array, table and JSON value is compared exactly, except fields that are
expected to differ between runs: timings, memory use, random stage IDs and
absolute paths. Figures are compared only by name.

Usage:
    python3.8 tools/compare_runs.py ~/slam_outputs/run_a ~/slam_outputs/run_b
    python3.8 tools/compare_runs.py RUN_A RUN_B --stages tracking loop_closure
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Pickled TrackingDB objects reference generic_slam classes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAGES = (
    "dataset", "tracking", "keyframes", "bundle", "pose_graph", "loop_closure",
    "analysis", "report",
)
# Fields whose value legitimately differs between equivalent runs. Config
# fingerprints are identity metadata (their format changed over time); the
# configurations themselves are compared separately via resolved_config.json.
VOLATILE_WORDS = (
    "runtime", "seconds", "rss", "stage_id", "elapsed", "_root", "_path", "fingerprint",
)


def _volatile(key):
    key = str(key)
    return any(word in key for word in VOLATILE_WORDS)


def _plain(value):
    """Convert pickled objects (e.g. Link, which has no __eq__) to comparable values."""

    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if hasattr(value, "__dict__"):
        return (type(value).__name__, _plain(vars(value)))
    return value


def _json_equal(first, second, where, problems):
    if isinstance(first, dict) and isinstance(second, dict):
        keys = {key for key in set(first) | set(second) if not _volatile(key)}
        for key in sorted(keys, key=str):
            if key not in first or key not in second:
                problems.append("{}: key {!r} only in one run".format(where, key))
            else:
                _json_equal(first[key], second[key], "{}.{}".format(where, key), problems)
    elif isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            problems.append("{}: list length {} vs {}".format(where, len(first), len(second)))
        for index, (a, b) in enumerate(zip(first, second)):
            _json_equal(a, b, "{}[{}]".format(where, index), problems)
    elif first != second:
        problems.append("{}: {!r} vs {!r}".format(where, first, second))


def _compare_file(first, second, problems):
    name = str(first)
    suffix = first.suffix
    if suffix == ".npz":
        with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
            if sorted(a.files) != sorted(b.files):
                problems.append("{}: arrays {} vs {}".format(name, a.files, b.files))
                return
            for key in a.files:
                if not np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind == "f"):
                    problems.append("{}: array {!r} differs".format(name, key))
    elif suffix == ".npy":
        if not np.array_equal(np.load(first), np.load(second)):
            problems.append("{}: array differs".format(name))
    elif suffix == ".csv":
        with first.open(newline="") as a, second.open(newline="") as b:
            rows_a, rows_b = list(csv.DictReader(a)), list(csv.DictReader(b))
        if len(rows_a) != len(rows_b):
            problems.append("{}: {} vs {} rows".format(name, len(rows_a), len(rows_b)))
            return
        for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
            for key in row_a:
                if not _volatile(key) and row_a.get(key) != row_b.get(key):
                    problems.append("{}: row {} column {!r}: {!r} vs {!r}".format(
                        name, index, key, row_a.get(key), row_b.get(key)))
                    return
    elif suffix == ".json":
        _json_equal(
            json.loads(first.read_text(encoding="utf-8")),
            json.loads(second.read_text(encoding="utf-8")),
            name, problems,
        )
    elif suffix == ".pkl":
        if first.read_bytes() != second.read_bytes():
            with first.open("rb") as a, second.open("rb") as b:
                if _plain(pickle.load(a)) != _plain(pickle.load(b)):
                    problems.append("{}: pickle content differs".format(name))
    elif suffix == ".md":
        # Everything except the (timing-only) stage runtime section.
        texts = [
            path.read_text(encoding="utf-8").split("## Stage runtime")[0]
            for path in (first, second)
        ]
        if texts[0] != texts[1]:
            problems.append("{}: text differs".format(name))
    # Figures (.png) and anything else: presence is checked by the caller.


def compare(root_a, root_b, stages):
    total_problems = 0
    configs = [root / "resolved_config.json" for root in (root_a, root_b)]
    if all(path.is_file() for path in configs):
        config_problems = []
        _json_equal(*(json.loads(p.read_text(encoding="utf-8")) for p in configs), "config", config_problems)
        print("{:<13} {}".format("config", "IDENTICAL" if not config_problems else "DIFFERS"))
        for problem in config_problems[:8]:
            print("    " + problem)
        total_problems += len(config_problems)
    for stage in stages:
        a, b = root_a / stage, root_b / stage
        if not a.is_dir() or not b.is_dir():
            print("{:<13} SKIPPED (missing in {})".format(
                stage, "both" if not a.is_dir() and not b.is_dir() else ("A" if not a.is_dir() else "B")))
            continue
        files_a = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
        files_b = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
        problems = ["only in A: {}".format(p) for p in sorted(files_a - files_b)][:5]
        problems += ["only in B: {}".format(p) for p in sorted(files_b - files_a)][:5]
        for relative in sorted(files_a & files_b):
            _compare_file(a / relative, b / relative, problems)
        status = "IDENTICAL" if not problems else "DIFFERS ({} issues)".format(len(problems))
        print("{:<13} {:<22} {} files".format(stage, status, len(files_a & files_b)))
        for problem in problems[:8]:
            print("    " + problem.replace(str(a) + "/", "").replace(str(b) + "/", ""))
        total_problems += len(problems)
    return total_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    arguments = parser.parse_args()
    problems = compare(
        arguments.run_a.expanduser(), arguments.run_b.expanduser(), arguments.stages
    )
    print("RESULT:", "identical" if problems == 0 else "{} differences".format(problems))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
