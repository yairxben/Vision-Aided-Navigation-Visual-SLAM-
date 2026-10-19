"""Bundle window summaries derived without re-solving graphs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..artifacts.io import read_json


def bundle_window_rows(bundle_root: Path):
    rows = []
    for directory in sorted((Path(bundle_root) / "windows").glob("window_*")):
        if not directory.is_dir() or not (directory / "seal.json").is_file():
            continue
        summary = read_json(directory / "summary.json")
        residuals = []
        with (directory / "projection_residuals.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                residuals.append(
                    np.linalg.norm(
                        [
                            float(row["final_residual_u_left_px"]),
                            float(row["final_residual_u_right_px"]),
                            float(row["final_residual_v_mean_px"]),
                        ]
                    )
                )
        rows.append(
            {
                "window_ordinal": int(summary["window_ordinal"]),
                "start_frame_id": int(summary["start_frame_id"]),
                "end_frame_id": int(summary["end_frame_id"]),
                "window_frame_count": int(summary["window_frame_count"]),
                "candidate_track_count": int(summary["candidate_track_count"]),
                "accepted_track_count": int(summary["accepted_track_count"]),
                "rejected_track_count": int(summary["rejected_track_count"]),
                "projection_factor_count": int(summary["projection_factor_count"]),
                "initial_graph_error": float(summary["initial_graph_error"]),
                "final_graph_error": float(summary["final_graph_error"]),
                "mean_final_projection_residual_px": float(np.mean(residuals)),
                "median_final_projection_residual_px": float(np.median(residuals)),
                "maximum_final_projection_residual_px": float(np.max(residuals)),
            }
        )
    return rows


__all__ = ["bundle_window_rows"]
