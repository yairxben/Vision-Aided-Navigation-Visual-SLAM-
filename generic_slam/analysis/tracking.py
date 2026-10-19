"""TrackingDB and transition statistics from sealed tracking artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def tracking_analysis_rows(tracking_root: Path):
    root = Path(tracking_root)
    with np.load(root / "pnp_trajectory.npz", allow_pickle=False) as trajectory:
        population = trajectory["frame_ids"].copy()
    with np.load(root / "track_observations.npz", allow_pickle=False) as observations:
        observation_frame_ids = observations["frame_ids"].copy()
        observation_track_ids = observations["track_ids"].copy()
        u_left = observations["u_left_px"].copy()
        u_right = observations["u_right_px"].copy()
        v_mean = observations["v_mean_px"].copy()
    frame_counts = dict(zip(*np.unique(observation_frame_ids, return_counts=True)))
    frame_rows = [
        {
            "frame_id": int(frame_id),
            "tracked_link_count": int(frame_counts.get(frame_id, 0)),
        }
        for frame_id in population
    ]
    track_statistics = {}
    for frame_id, track_id in zip(observation_frame_ids, observation_track_ids):
        key = int(track_id)
        statistics = track_statistics.get(key)
        if statistics is None:
            track_statistics[key] = [1, int(frame_id), int(frame_id)]
        else:
            statistics[0] += 1
            statistics[2] = int(frame_id)
    track_rows = [
        {
            "track_id": track_id,
            "track_length_frames": values[0],
            "first_frame_id": values[1],
            "last_frame_id": values[2],
        }
        for track_id, values in sorted(track_statistics.items())
    ]
    representative_rows = []
    if track_rows:
        representative_id = int(max(
            track_rows,
            key=lambda row: (int(row["track_length_frames"]), -int(row["track_id"])),
        )["track_id"])
        selected = np.where(observation_track_ids == representative_id)[0]
        representative_rows = [
            {
                "track_id": representative_id,
                "frame_id": int(observation_frame_ids[index]),
                "u_left_px": float(u_left[index]),
                "u_right_px": float(u_right[index]),
                "v_mean_px": float(v_mean[index]),
            }
            for index in selected
        ]
    with (Path(tracking_root) / "transitions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        transition_rows = list(csv.DictReader(handle))
    reprojection_rows = []
    evidence_root = Path(tracking_root) / "transition_evidence"
    for row in transition_rows:
        previous = int(row["previous_frame_id"])
        current = int(row["current_frame_id"])
        path = evidence_root / "transition_{:06d}_{:06d}.npz".format(previous, current)
        with np.load(path, allow_pickle=False) as data:
            observed = data["observed_pixels"]
            projected = data["projected_pixels"]
            supporter_indices = data["supporter_indices"]
        distances = np.linalg.norm(observed - projected, axis=1) if observed.size else np.empty((4, 0))
        supporter_distances = distances[:, supporter_indices] if len(supporter_indices) else np.empty((4, 0))
        reprojection_rows.append(
            {
                "previous_frame_id": previous,
                "current_frame_id": current,
                "consensus_count": int(observed.shape[2]),
                "supporter_count": len(supporter_indices),
                "mean_supporter_reprojection_error_px": float(np.mean(supporter_distances)) if supporter_distances.size else float("nan"),
                "median_supporter_reprojection_error_px": float(np.median(supporter_distances)) if supporter_distances.size else float("nan"),
                "maximum_supporter_reprojection_error_px": float(np.max(supporter_distances)) if supporter_distances.size else float("nan"),
            }
        )
    return frame_rows, track_rows, transition_rows, reprojection_rows, representative_rows


__all__ = ["tracking_analysis_rows"]
