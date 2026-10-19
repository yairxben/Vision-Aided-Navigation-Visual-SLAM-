"""Loop Closure disposition and acceptance aggregation."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def loop_summary(loop_root: Path):
    with (Path(loop_root) / "aggregate" / "candidate_dispositions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        candidates = list(csv.DictReader(handle))
    with (Path(loop_root) / "aggregate" / "accepted_loops.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        accepted = list(csv.DictReader(handle))
    return candidates, accepted, dict(sorted(Counter(row["disposition"] for row in candidates).items()))


__all__ = ["loop_summary"]
