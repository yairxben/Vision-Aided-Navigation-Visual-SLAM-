"""Bundle window summaries derived without re-solving graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..artifacts.io import read_json


def bundle_window_rows(bundle_root: Path):
    rows = []
    for directory in sorted((Path(bundle_root) / "windows").glob("window_*")):
        if not directory.is_dir() or not (directory / "seal.json").is_file():
            continue
        summary = read_json(directory / "summary.json")
        with np.load(directory / "projection_residuals.npz", allow_pickle=False) as data:
            residuals = np.linalg.norm(data["final_residual_px"], axis=1)
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
