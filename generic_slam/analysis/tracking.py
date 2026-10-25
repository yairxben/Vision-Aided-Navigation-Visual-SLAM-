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
    # A track observed in frame f "continues" if the same track is also
    # observed in frame f + 1.
    stride = int(population.max()) + 2 if len(population) else 1
    keys = observation_track_ids.astype(np.int64) * stride + observation_frame_ids
    continues = np.isin(keys + 1, keys)
    continuing_counts = dict(
        zip(*np.unique(observation_frame_ids[continues], return_counts=True))
    )
    frame_rows = [
        {
            "frame_id": int(frame_id),
            "tracked_link_count": int(frame_counts.get(frame_id, 0)),
            "tracks_continuing_to_next_frame": int(continuing_counts.get(frame_id, 0)),
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


def summarize_tracking(frame_rows, track_rows, transition_rows):
    """Summary statistics of the TrackingDB and the per-transition match funnel."""

    lengths = np.asarray([int(row["track_length_frames"]) for row in track_rows], dtype=float)
    links = np.asarray([int(row["tracked_link_count"]) for row in frame_rows], dtype=float)
    # The final frame has no following frame, so it is excluded.
    continuing = np.asarray(
        [int(row["tracks_continuing_to_next_frame"]) for row in frame_rows[:-1]], dtype=float
    )
    stereo = np.asarray([float(row["current_stereo_row_count"]) for row in transition_rows])
    consensus = np.asarray([float(row["consensus_count"]) for row in transition_rows])
    supporters = np.asarray([float(row["supporter_count"]) for row in transition_rows])
    percentage = np.asarray([float(row["supporter_percentage"]) for row in transition_rows])

    def _mean(values):
        return float(np.mean(values)) if len(values) else None

    return {
        "frame_count": len(frame_rows),
        "track_count": len(track_rows),
        "track_length_frames": {
            "mean": _mean(lengths),
            "median": float(np.median(lengths)) if len(lengths) else None,
            "maximum": int(lengths.max()) if len(lengths) else None,
        },
        "tracks_with_length_at_least": {
            str(minimum): int(np.sum(lengths >= minimum)) for minimum in (2, 3, 5, 10, 20)
        },
        "mean_tracked_links_per_frame": _mean(links),
        "mean_tracks_continuing_to_next_frame": _mean(continuing),
        "minimum_tracks_continuing_to_next_frame": (
            int(continuing.min()) if len(continuing) else None
        ),
        "per_transition_means": {
            "stereo_matches": _mean(stereo),
            "four_view_consensus": _mean(consensus),
            "pnp_ransac_supporters": _mean(supporters),
            "supporter_percentage_of_consensus": _mean(percentage),
        },
    }


__all__ = ["summarize_tracking", "tracking_analysis_rows"]
