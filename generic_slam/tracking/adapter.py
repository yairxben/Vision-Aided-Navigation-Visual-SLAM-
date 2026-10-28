"""Neutral raw-feature to lecturer-row adapter with supporter-only insertion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..contracts import ContractError
from .lecturer_store import Link, NO_ID, TrackingDB


@dataclass(frozen=True)
class PreparedStereoFrame:
    raw_descriptor_count: int
    filtered_descriptors: np.ndarray
    links: Tuple[Link, ...]
    stereo_matches: Tuple[cv2.DMatch, ...]
    raw_to_filtered: Tuple[int, ...]
    filtered_to_raw: Tuple[int, ...]


@dataclass(frozen=True)
class PreparedTemporalTransition:
    matches: Tuple[cv2.DMatch, ...]
    validity_mask: Tuple[bool, ...]
    reciprocal_candidate_count: int
    stereo_compatible_candidate_count: int
    selected_candidate_count: int
    consensus_count: int
    supporter_count: int
    supporter_percentage: float
    rejected_not_stereo_valid: int
    rejected_conflict: int


def _match_key(match: cv2.DMatch, original_index: int) -> Tuple[float, int, int]:
    return float(match.distance), int(match.trainIdx), int(original_index)


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
        raise ContractError("TrackingDB cannot accept an empty stereo population")
    best_by_query = {}
    for original_index, match in enumerate(stereo_inliers):
        query = int(match.queryIdx)
        train = int(match.trainIdx)
        if not 0 <= query < descriptors.shape[0] or not 0 <= train < len(right_keypoints):
            raise ContractError("stereo match index is outside feature space")
        previous = best_by_query.get(query)
        if previous is None or _match_key(match, original_index) < _match_key(
            previous[1], previous[0]
        ):
            best_by_query[query] = (original_index, match)
    ordered = tuple(best_by_query[index][1] for index in sorted(best_by_query))
    filtered_to_raw = tuple(int(match.queryIdx) for match in ordered)
    filtered_descriptors, links = TrackingDB.create_links(
        descriptors, tuple(left_keypoints), tuple(right_keypoints), ordered
    )
    raw_to_filtered = [NO_ID] * descriptors.shape[0]
    for filtered_row, raw_row in enumerate(filtered_to_raw):
        raw_to_filtered[raw_row] = filtered_row
    if not np.array_equal(filtered_descriptors, descriptors[list(filtered_to_raw)]):
        raise ContractError("filtered descriptors lost raw-row ordering")
    return PreparedStereoFrame(
        raw_descriptor_count=int(descriptors.shape[0]),
        filtered_descriptors=np.asarray(filtered_descriptors).copy(),
        links=tuple(links),
        stereo_matches=ordered,
        raw_to_filtered=tuple(raw_to_filtered),
        filtered_to_raw=filtered_to_raw,
    )


def prepare_temporal_transition(
    previous: PreparedStereoFrame,
    current: PreparedStereoFrame,
    reciprocal_matches: Sequence[cv2.DMatch],
    supporter_temporal_indices: Sequence[int],
    consensus_count: int,
) -> PreparedTemporalTransition:
    """Produce one ordered DMatch/mask item per previous TrackingDB row."""

    previous_rows = len(previous.filtered_to_raw)
    current_rows = len(current.filtered_to_raw)
    if previous_rows == 0 or current_rows == 0 or consensus_count <= 0:
        raise ContractError("transition requires nonempty rows and consensus")
    supporter_indices = tuple(int(value) for value in supporter_temporal_indices)
    if len(set(supporter_indices)) != len(supporter_indices):
        raise ContractError("supporter temporal indices must be unique")
    if any(not 0 <= value < len(reciprocal_matches) for value in supporter_indices):
        raise ContractError("supporter temporal index is out of range")
    if len(supporter_indices) > consensus_count:
        raise ContractError("supporter count exceeds consensus count")
    supporter_set = set(supporter_indices)

    compatible = []
    rejected_not_stereo = 0
    for original_index, match in enumerate(reciprocal_matches):
        raw_query = int(match.queryIdx)
        raw_train = int(match.trainIdx)
        if (
            not 0 <= raw_query < previous.raw_descriptor_count
            or not 0 <= raw_train < current.raw_descriptor_count
        ):
            raise ContractError("temporal match index is outside raw feature space")
        previous_row = previous.raw_to_filtered[raw_query]
        current_row = current.raw_to_filtered[raw_train]
        supporter = original_index in supporter_set
        if previous_row == NO_ID or current_row == NO_ID:
            if supporter:
                raise ContractError("final supporter is not stereo-valid")
            rejected_not_stereo += 1
            continue
        compatible.append(
            (
                float(match.distance), int(previous_row), int(current_row),
                int(original_index), supporter,
            )
        )

    used_previous = set()
    used_current = set()
    selected = []
    rejected_conflict = 0
    for _, previous_row, current_row, original_index, supporter in sorted(compatible):
        if previous_row in used_previous or current_row in used_current:
            if supporter:
                raise ContractError("final supporter lost to deterministic row conflict")
            rejected_conflict += 1
            continue
        used_previous.add(previous_row)
        used_current.add(current_row)
        selected.append((previous_row, current_row, original_index, supporter))

    placeholder = float(np.finfo(np.float32).max)
    complete = [
        cv2.DMatch(
            _queryIdx=row, _trainIdx=0, _imgIdx=0, _distance=placeholder
        )
        for row in range(previous_rows)
    ]
    valid = [False] * previous_rows
    for previous_row, current_row, original_index, supporter in selected:
        source = reciprocal_matches[original_index]
        complete[previous_row] = cv2.DMatch(
            _queryIdx=previous_row,
            _trainIdx=current_row,
            _imgIdx=0,
            _distance=float(source.distance),
        )
        valid[previous_row] = bool(supporter)
    destinations = [match.trainIdx for match, accepted in zip(complete, valid) if accepted]
    if len(destinations) != len(set(destinations)) or any(
        not 0 <= value < current_rows for value in destinations
    ):
        raise ContractError("supporter destinations are invalid")
    if sum(valid) != len(supporter_indices):
        raise ContractError("supporter-only validity mask differs")
    return PreparedTemporalTransition(
        matches=tuple(complete),
        validity_mask=tuple(valid),
        reciprocal_candidate_count=len(reciprocal_matches),
        stereo_compatible_candidate_count=len(compatible),
        selected_candidate_count=len(selected),
        consensus_count=int(consensus_count),
        supporter_count=len(supporter_indices),
        supporter_percentage=100.0 * len(supporter_indices) / consensus_count,
        rejected_not_stereo_valid=rejected_not_stereo,
        rejected_conflict=rejected_conflict,
    )


class TrackingStoreAdapter:
    """Validated sequential insertion boundary around the lecturer store."""

    def __init__(self) -> None:
        self.database = TrackingDB()
        self.last_frame = None  # type: Optional[PreparedStereoFrame]

    def add_initial(self, frame: PreparedStereoFrame) -> int:
        if self.last_frame is not None:
            raise ContractError("initial tracking frame already exists")
        frame_id = self.database.add_frame(
            list(frame.links), frame.filtered_descriptors.copy()
        )
        if frame_id != 0:
            raise ContractError("lecturer store did not issue FrameId zero")
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
            list(frame.links),
            frame.filtered_descriptors.copy(),
            list(prepared.matches),
            list(prepared.validity_mask),
        )
        if frame_id != self.database.frame_num() - 1:
            raise ContractError("lecturer FrameId issuance is not consecutive")
        self.last_frame = frame
        return frame_id, prepared


def validate_tracking_store(database: TrackingDB, expected_frame_count: int) -> None:
    if database.frame_num() != expected_frame_count:
        raise ContractError("tracking store frame count differs")
    if tuple(database.all_frames()) != tuple(range(expected_frame_count)):
        raise ContractError("tracking store FrameIds are not dense from zero")
    for frame_id in database.all_frames():
        features = database.features(frame_id)
        rows = database.frameId_to_trackIds_list.get(frame_id)
        if features is None or features.ndim != 2 or rows is None:
            raise ContractError("tracking store lacks frame rows")
        if features.shape[0] != len(rows):
            raise ContractError("tracking descriptor and TrackId rows differ")
    for track_id in database.all_tracks():
        frames = database.frames(track_id)
        if len(frames) < 2 or frames != list(range(frames[0], frames[-1] + 1)):
            raise ContractError("tracking store contains a nonconsecutive track")
        for frame_id in frames:
            link = database.link(frame_id, track_id)
            if link is None:
                raise ContractError("tracking store track lacks an observation")
            values = (float(link.x_left), float(link.x_right), float(link.y))
            if not all(math.isfinite(value) for value in values) or not (
                link.x_left > link.x_right
            ):
                raise ContractError("tracking observation violates stereo policy")
