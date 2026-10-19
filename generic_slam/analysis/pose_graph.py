"""Raw Pose3 marginal uncertainty summaries."""

from __future__ import annotations

import numpy as np


def uncertainty_rows(frame_ids, covariances, stage):
    rows = []
    for frame_id, covariance in zip(frame_ids, covariances):
        diagonal = np.diag(covariance)
        rows.append(
            {
                "stage": stage,
                "frame_id": int(frame_id),
                "rotation_uncertainty_deg": float(np.degrees(np.sqrt(max(0.0, np.sum(diagonal[:3]))))),
                "location_uncertainty_m": float(np.sqrt(max(0.0, np.sum(diagonal[3:])))),
                "rot_x_variance": float(diagonal[0]),
                "rot_y_variance": float(diagonal[1]),
                "rot_z_variance": float(diagonal[2]),
                "trans_x_variance": float(diagonal[3]),
                "trans_y_variance": float(diagonal[4]),
                "trans_z_variance": float(diagonal[5]),
            }
        )
    return rows


__all__ = ["uncertainty_rows"]
