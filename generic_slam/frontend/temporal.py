"""Reciprocal temporal matching and four-view consensus mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FourViewCorrespondence:
    """Indices into previous stereo, current stereo, and temporal populations."""

    previous_stereo_index: int
    current_stereo_index: int
    temporal_match_index: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (
            self.previous_stereo_index,
            self.current_stereo_index,
            self.temporal_match_index,
        )


def build_four_view_consensus(
    previous_stereo_inliers: Sequence[cv2.DMatch],
    current_stereo_inliers: Sequence[cv2.DMatch],
    temporal_matches: Sequence[cv2.DMatch],
) -> Tuple[FourViewCorrespondence, ...]:
    """Intersect reciprocal left matches with both sorted stereo populations."""

    previous_sorted = sorted(previous_stereo_inliers, key=lambda item: item.queryIdx)
    current_sorted = sorted(current_stereo_inliers, key=lambda item: item.queryIdx)
    previous_map = {
        match.queryIdx: index for index, match in enumerate(previous_sorted)
    }
    current_map = {
        match.queryIdx: index for index, match in enumerate(current_sorted)
    }
    consensus = []
    for temporal_index, match in enumerate(temporal_matches):
        if match.queryIdx in previous_map and match.trainIdx in current_map:
            consensus.append(
                FourViewCorrespondence(
                    previous_stereo_index=previous_map[match.queryIdx],
                    current_stereo_index=current_map[match.trainIdx],
                    temporal_match_index=temporal_index,
                )
            )
    return tuple(consensus)


def actual_consensus_pixels(
    consensus: Sequence[FourViewCorrespondence],
    previous_stereo_inliers: Sequence[cv2.DMatch],
    current_stereo_inliers: Sequence[cv2.DMatch],
    previous_left_keypoints: Sequence[cv2.KeyPoint],
    previous_right_keypoints: Sequence[cv2.KeyPoint],
    current_left_keypoints: Sequence[cv2.KeyPoint],
    current_right_keypoints: Sequence[cv2.KeyPoint],
) -> np.ndarray:
    """Return the protected 4x2xN observed-pixel representation."""

    populations = ([], [], [], [])
    for item in consensus:
        previous = previous_stereo_inliers[item.previous_stereo_index]
        current = current_stereo_inliers[item.current_stereo_index]
        populations[0].append(previous_left_keypoints[previous.queryIdx].pt)
        populations[1].append(previous_right_keypoints[previous.trainIdx].pt)
        populations[2].append(current_left_keypoints[current.queryIdx].pt)
        populations[3].append(current_right_keypoints[current.trainIdx].pt)
    if not consensus:
        return np.empty((4, 2, 0), dtype=float)
    return np.asarray(
        [np.asarray(points, dtype=float).T for points in populations], dtype=float
    )
