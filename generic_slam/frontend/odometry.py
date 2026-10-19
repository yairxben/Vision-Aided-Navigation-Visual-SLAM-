"""Neutral frontend transition contract used by tracking and Loop Closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import StereoFeatures
from .pnp import VisualOdometryRansacResult


@dataclass(frozen=True)
class OdometryTransition:
    previous_frame_id: int
    current_frame_id: int
    previous_stereo: StereoFeatures
    current_stereo: StereoFeatures
    triangulated_previous_points: np.ndarray
    reciprocal_match_count: int
    consensus_count: int
    ransac: VisualOdometryRansacResult
