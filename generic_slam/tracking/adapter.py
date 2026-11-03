"""Raw OpenCV features to tracking-store rows, with supporter-only insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError
from .store import NO_ID, TrackStore


@dataclass(frozen=True)
class PreparedStereoFrame:
    """Stereo rows of one frame: one row per left feature with a stereo match."""

    raw_descriptor_count: int
    links: np.ndarray  # (rows, 3): u_left, u_right, v (mean of left/right y)
    raw_to_filtered: np.ndarray  # row of each raw left feature, or NO_ID
    filtered_to_raw: np.ndarray  # raw left feature of each row (ascending)


@dataclass(frozen=True)
class PreparedTemporalTransition:
    previous_rows: np.ndarray  # supporter pairs, ascending previous row
    current_rows: np.ndarray
    reciprocal_candidate_count: int
    stereo_compatible_candidate_count: int
    selected_candidate_count: int
    consensus_count: int
    supporter_count: int
    supporter_percentage: float
    rejected_not_stereo_valid: int
    rejected_conflict: int


@dataclass(frozen=True)
class MatchArrays:
    """Matches as parallel arrays (the native frontend's representation)."""

    query: np.ndarray
    train: np.ndarray
    distance: np.ndarray

    def __len__(self) -> int:
        return len(self.query)


def _match_arrays(matches: Sequence[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(matches, MatchArrays):
        return (
            np.asarray(matches.query, dtype=np.int64),
            np.asarray(matches.train, dtype=np.int64),
            np.asarray(matches.distance, dtype=float),
        )
    query = np.fromiter((match.queryIdx for match in matches), dtype=np.int64, count=len(matches))
    train = np.fromiter((match.trainIdx for match in matches), dtype=np.int64, count=len(matches))
    distance = np.fromiter(
        (match.distance for match in matches), dtype=float, count=len(matches)
    )
    return query, train, distance


def prepare_stereo_frame(
    raw_left_descriptors: np.ndarray,
    left_keypoints: Sequence[cv2.KeyPoint],
    right_keypoints: Sequence[cv2.KeyPoint],
    stereo_inliers: Sequence[cv2.DMatch],
) -> PreparedStereoFrame:
    descriptors = np.asarray(raw_left_descriptors)
    if descriptors.ndim != 2 or descriptors.shape[0] != len(left_keypoints):
        raise ContractError("left descriptors and keypoints are not row-aligned")
    if not stereo_inliers:
        raise ContractError("the tracking store cannot accept an empty stereo population")
    query, train, distance = _match_arrays(stereo_inliers)
    if (
        np.any(query < 0) or np.any(query >= descriptors.shape[0])
        or np.any(train < 0) or np.any(train >= len(right_keypoints))
    ):
        raise ContractError("stereo match index is outside feature space")
    # One row per left feature: its best match by (distance, trainIdx, order).
    order = np.lexsort((np.arange(len(query)), train, distance, query))
    first = np.ones(len(order), dtype=bool)
    first[1:] = query[order][1:] != query[order][:-1]
    kept = order[first]  # ascending queryIdx
    filtered_to_raw = query[kept]
    left_points = np.array([left_keypoints[index].pt for index in filtered_to_raw], dtype=float)
    right_points = np.array([right_keypoints[index].pt for index in train[kept]], dtype=float)
    links = np.column_stack(
        (left_points[:, 0], right_points[:, 0], (left_points[:, 1] + right_points[:, 1]) / 2)
    )
    raw_to_filtered = np.full(descriptors.shape[0], NO_ID, dtype=np.int64)
    raw_to_filtered[filtered_to_raw] = np.arange(len(filtered_to_raw))
    return PreparedStereoFrame(
        raw_descriptor_count=int(descriptors.shape[0]),
        links=links,
        raw_to_filtered=raw_to_filtered,
        filtered_to_raw=filtered_to_raw,
    )


def prepare_temporal_transition(
    previous: PreparedStereoFrame,
    current: PreparedStereoFrame,
    reciprocal_matches: Sequence[cv2.DMatch],
    supporter_temporal_indices: Sequence[int],
    consensus_count: int,
) -> PreparedTemporalTransition:
    """Map RANSAC supporters to (previous row, current row) store pairs."""

    if len(previous.filtered_to_raw) == 0 or len(current.filtered_to_raw) == 0 or consensus_count <= 0:
        raise ContractError("transition requires nonempty rows and consensus")
    supporters = np.asarray(tuple(supporter_temporal_indices), dtype=np.int64)
    if len(np.unique(supporters)) != len(supporters):
        raise ContractError("supporter temporal indices must be unique")
    if np.any(supporters < 0) or np.any(supporters >= len(reciprocal_matches)):
        raise ContractError("supporter temporal index is out of range")
    if len(supporters) > consensus_count:
        raise ContractError("supporter count exceeds consensus count")
    query, train, distance = _match_arrays(reciprocal_matches)
    if (
        np.any(query < 0) or np.any(query >= previous.raw_descriptor_count)
        or np.any(train < 0) or np.any(train >= current.raw_descriptor_count)
    ):
        raise ContractError("temporal match index is outside raw feature space")
    is_supporter = np.zeros(len(query), dtype=bool)
    is_supporter[supporters] = True
    previous_row = previous.raw_to_filtered[query]
    current_row = current.raw_to_filtered[train]
    compatible = (previous_row != NO_ID) & (current_row != NO_ID)
    if np.any(is_supporter & ~compatible):
        raise ContractError("final supporter is not stereo-valid")

    # Greedy one-to-one selection in (distance, previous row, current row,
    # match order) order: a candidate loses if either row is already taken.
    candidates = np.nonzero(compatible)[0]
    candidates = candidates[np.lexsort((
        candidates, current_row[candidates], previous_row[candidates], distance[candidates]
    ))]
    selected_mask = np.ones(len(candidates), dtype=bool)
    if (
        len(np.unique(previous_row[candidates])) != len(candidates)
        or len(np.unique(current_row[candidates])) != len(candidates)
    ):
        # Reciprocal matches are one-to-one, so this loop is a safeguard only.
        used_previous, used_current = set(), set()
        for position, index in enumerate(candidates):
            rows = (int(previous_row[index]), int(current_row[index]))
            if rows[0] in used_previous or rows[1] in used_current:
                selected_mask[position] = False
                continue
            used_previous.add(rows[0])
            used_current.add(rows[1])
    rejected_conflict = int(np.count_nonzero(~selected_mask))
    if np.any(is_supporter[candidates[~selected_mask]]):
        raise ContractError("final supporter lost to deterministic row conflict")
    selected = candidates[selected_mask]
    chosen = selected[is_supporter[selected]]
    chosen = chosen[np.argsort(previous_row[chosen], kind="stable")]
    if len(chosen) != len(supporters):
        raise ContractError("supporter-only validity mask differs")
    return PreparedTemporalTransition(
        previous_rows=previous_row[chosen],
        current_rows=current_row[chosen],
        reciprocal_candidate_count=len(query),
        stereo_compatible_candidate_count=len(candidates),
        selected_candidate_count=len(selected),
        consensus_count=int(consensus_count),
        supporter_count=len(supporters),
        supporter_percentage=100.0 * len(supporters) / consensus_count,
        rejected_not_stereo_valid=int(np.count_nonzero(~compatible)),
        rejected_conflict=rejected_conflict,
    )


class TrackingStoreAdapter:
    """Validated sequential insertion boundary around the tracking store."""

    def __init__(self) -> None:
        self.database = TrackStore()
        self.last_frame = None  # type: Optional[PreparedStereoFrame]

    def add_initial(self, frame: PreparedStereoFrame) -> int:
        if self.last_frame is not None:
            raise ContractError("initial tracking frame already exists")
        frame_id = self.database.add_frame(frame.links)
        if frame_id != 0:
            raise ContractError("tracking store did not issue FrameId zero")
        self.last_frame = frame
        return frame_id

    def add_transition(
        self,
        frame: PreparedStereoFrame,
        reciprocal_matches: Sequence[cv2.DMatch],
        supporter_temporal_indices: Sequence[int],
        consensus_count: int,
    ) -> Tuple[int, PreparedTemporalTransition]:
        if self.last_frame is None:
            raise ContractError("initial tracking frame is missing")
        prepared = prepare_temporal_transition(
            self.last_frame,
            frame,
            reciprocal_matches,
            supporter_temporal_indices,
            consensus_count,
        )
        frame_id = self.database.add_frame(
            frame.links, prepared.previous_rows, prepared.current_rows
        )
        if frame_id != self.database.frame_num() - 1:
            raise ContractError("tracking FrameId issuance is not consecutive")
        self.last_frame = frame
        return frame_id, prepared


__all__ = [
    "MatchArrays", "PreparedStereoFrame", "PreparedTemporalTransition",
    "TrackingStoreAdapter", "prepare_stereo_frame", "prepare_temporal_transition",
]
