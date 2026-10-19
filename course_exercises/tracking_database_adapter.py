"""Canonical Exercise 3 to lecturer TrackingDB adapter.

The adapter owns index-space conversion and validates every known lecturer
precondition before mutating TrackingDB. Pose and supporter metadata remain
outside the lecturer database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from tracking_database import Link, NO_ID, TrackingDB


@dataclass
class PreparedStereoFrame:
    """A stereo frame expressed in both raw and TrackingDB row spaces."""

    raw_descriptor_count: int
    filtered_descriptors: np.ndarray
    links: List[Link]
    stereo_matches: List[cv2.DMatch]
    raw_to_filtered: Tuple[int, ...]
    filtered_to_raw: Tuple[int, ...]


@dataclass
class PreparedTemporalTransition:
    """Complete previous-row temporal input and its provenance."""

    matches: List[cv2.DMatch]
    validity_mask: List[bool]
    reciprocal_candidate_count: int
    stereo_compatible_candidate_count: int
    selected_candidate_count: int
    consensus_count: int
    supporter_count: int
    selected_supporter_count: int
    supporter_percentage: float
    selected_candidates: List[Dict[str, object]]
    rejected_candidates: List[Dict[str, object]]


@dataclass
class TrackingDBCompanion:
    """Frame-aligned state intentionally kept outside lecturer TrackingDB."""

    world_to_camera_poses: List[np.ndarray] = field(default_factory=list)
    supporter_counts: List[int] = field(default_factory=list)
    supporter_percentages: List[float] = field(default_factory=list)
    transition_frame_pairs: List[Tuple[int, int]] = field(default_factory=list)

    @staticmethod
    def validate_pose(pose: np.ndarray) -> np.ndarray:
        normalized = np.asarray(pose, dtype=float)
        if normalized.shape != (3, 4):
            raise ValueError(
                "world-to-camera pose must have shape (3, 4), got {}".format(
                    normalized.shape
                )
            )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("world-to-camera pose must contain finite values")
        rotation = normalized[:, :3]
        if not np.allclose(
            rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0
        ):
            raise ValueError("pose rotation is not orthonormal")
        if not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0
        ):
            raise ValueError("pose rotation determinant is not one")
        return normalized.copy()

    def validate_initial(self, pose: np.ndarray) -> np.ndarray:
        if (
            self.world_to_camera_poses
            or self.supporter_counts
            or self.supporter_percentages
        ):
            raise ValueError("initial companion state has already been added")
        if self.transition_frame_pairs:
            raise ValueError("initial companion state has transitions")
        return self.validate_pose(pose)

    def validate_transition(
        self,
        front_frame_id: int,
        pose: np.ndarray,
        percentage: float,
        supporter_count: int,
    ) -> np.ndarray:
        if len(self.world_to_camera_poses) != front_frame_id:
            raise ValueError(
                "pose index {} is not the next FrameId {}".format(
                    len(self.world_to_camera_poses), front_frame_id
                )
            )
        if len(self.supporter_percentages) != front_frame_id - 1:
            raise ValueError("supporter percentage is not transition-aligned")
        if len(self.supporter_counts) != front_frame_id - 1:
            raise ValueError("supporter count is not transition-aligned")
        if not np.isfinite(percentage) or not 0.0 <= percentage <= 100.0:
            raise ValueError("supporter percentage must be in [0, 100]")
        if int(supporter_count) != supporter_count or supporter_count < 0:
            raise ValueError("supporter count must be a nonnegative integer")
        return self.validate_pose(pose)

    def append_initial(self, pose: np.ndarray) -> None:
        self.world_to_camera_poses.append(pose.copy())

    def append_transition(
        self,
        front_frame_id: int,
        pose: np.ndarray,
        percentage: float,
        supporter_count: int,
    ) -> None:
        self.world_to_camera_poses.append(pose.copy())
        self.supporter_counts.append(int(supporter_count))
        self.supporter_percentages.append(float(percentage))
        self.transition_frame_pairs.append(
            (front_frame_id - 1, front_frame_id)
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "pose_convention": "world-to-camera: x_i = R_i X + t_i",
            "pose_frame_ids": list(range(len(self.world_to_camera_poses))),
            "world_to_camera_poses": [
                pose.tolist() for pose in self.world_to_camera_poses
            ],
            "supporter_percentage_transition_pairs": [
                list(pair) for pair in self.transition_frame_pairs
            ],
            "supporter_counts": list(self.supporter_counts),
            "supporter_percentages": list(self.supporter_percentages),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TrackingDBCompanion":
        """Restore and validate serialized companion metadata."""

        if data.get("pose_convention") != (
            "world-to-camera: x_i = R_i X + t_i"
        ):
            raise ValueError("unsupported or missing pose convention")
        poses = data.get("world_to_camera_poses")
        pose_frame_ids = data.get("pose_frame_ids")
        percentages = data.get("supporter_percentages")
        supporter_counts = data.get("supporter_counts")
        transition_pairs = data.get(
            "supporter_percentage_transition_pairs"
        )
        if not isinstance(poses, list) or not poses:
            raise ValueError("companion must contain at least one pose")
        if pose_frame_ids != list(range(len(poses))):
            raise ValueError("companion poses are not FrameId-aligned")
        if not isinstance(percentages, list):
            raise ValueError("companion supporter percentages are missing")
        if not isinstance(supporter_counts, list):
            raise ValueError("companion supporter counts are missing")
        expected_pairs = [
            [frame_id, frame_id + 1]
            for frame_id in range(len(poses) - 1)
        ]
        if transition_pairs != expected_pairs:
            raise ValueError(
                "companion supporter percentages are not transition-aligned"
            )
        if len(percentages) != len(poses) - 1:
            raise ValueError("companion percentage count differs from poses")
        if len(supporter_counts) != len(poses) - 1:
            raise ValueError("companion supporter count differs from poses")

        companion = cls()
        initial_pose = companion.validate_initial(np.asarray(poses[0]))
        companion.append_initial(initial_pose)
        for front_frame_id in range(1, len(poses)):
            percentage = float(percentages[front_frame_id - 1])
            raw_supporter_count = supporter_counts[front_frame_id - 1]
            if (
                not isinstance(raw_supporter_count, int)
                or isinstance(raw_supporter_count, bool)
                or raw_supporter_count < 0
            ):
                raise ValueError(
                    "serialized supporter count must be a "
                    "nonnegative integer"
                )
            supporter_count = raw_supporter_count
            pose = companion.validate_transition(
                front_frame_id,
                np.asarray(poses[front_frame_id]),
                percentage,
                supporter_count,
            )
            companion.append_transition(
                front_frame_id, pose, percentage, supporter_count
            )
        return companion


def _match_key(match: cv2.DMatch, original_index: int) -> Tuple[float, int, int]:
    return (
        float(match.distance),
        int(match.trainIdx),
        int(original_index),
    )


def prepare_stereo_frame(
    raw_left_descriptors: np.ndarray,
    keypoints_left: Sequence[cv2.KeyPoint],
    keypoints_right: Sequence[cv2.KeyPoint],
    stereo_inlier_matches: Sequence[cv2.DMatch],
) -> PreparedStereoFrame:
    """Create row-aligned lecturer inputs and explicit raw/filtered mappings.

    Only stereo-valid left features become TrackingDB rows. The returned maps
    preserve the conversion between raw descriptor indices and those filtered
    rows for the later temporal transition.
    """

    descriptors = np.asarray(raw_left_descriptors)
    if descriptors.ndim != 2:
        raise ValueError("left descriptors must be a two-dimensional array")
    raw_count = int(descriptors.shape[0])
    if raw_count != len(keypoints_left):
        raise ValueError("descriptor rows and left keypoints must align")
    if not stereo_inlier_matches:
        raise ValueError("lecturer create_links does not accept empty matches")

    # Keep one deterministic stereo observation for each raw left feature.
    best_by_query = {}
    for original_index, match in enumerate(stereo_inlier_matches):
        query_idx = int(match.queryIdx)
        train_idx = int(match.trainIdx)
        if not 0 <= query_idx < raw_count:
            raise ValueError("stereo queryIdx is outside left descriptor space")
        if not 0 <= train_idx < len(keypoints_right):
            raise ValueError("stereo trainIdx is outside right keypoint space")
        previous = best_by_query.get(query_idx)
        if previous is None or _match_key(
            match, original_index
        ) < _match_key(previous[1], previous[0]):
            best_by_query[query_idx] = (original_index, match)

    ordered_matches = [
        best_by_query[query_idx][1] for query_idx in sorted(best_by_query)
    ]
    ordered_queries = [int(match.queryIdx) for match in ordered_matches]
    if len(ordered_queries) != len(set(ordered_queries)):
        raise AssertionError("stereo queryIdx values must be unique")
    if ordered_queries != sorted(ordered_queries):
        raise AssertionError("stereo matches must be queryIdx-sorted")

    # Materialize lecturer Link rows in raw-left query-index order.
    filtered_descriptors, links = TrackingDB.create_links(
        descriptors,
        tuple(keypoints_left),
        tuple(keypoints_right),
        tuple(ordered_matches),
    )
    filtered_to_raw = tuple(ordered_queries)
    raw_to_filtered = [NO_ID] * raw_count
    for filtered_row, raw_row in enumerate(filtered_to_raw):
        raw_to_filtered[raw_row] = filtered_row

    expected_descriptors = descriptors[list(filtered_to_raw)]
    if not np.array_equal(filtered_descriptors, expected_descriptors):
        raise AssertionError("filtered descriptors lost raw-row ordering")
    if len(links) != len(filtered_to_raw):
        raise AssertionError("descriptor and Link row counts differ")
    for filtered_row, match in enumerate(ordered_matches):
        left_point = keypoints_left[match.queryIdx].pt
        right_point = keypoints_right[match.trainIdx].pt
        link = links[filtered_row]
        expected_y = (left_point[1] + right_point[1]) / 2.0
        if (
            link.x_left != left_point[0]
            or link.x_right != right_point[0]
            or link.y != expected_y
        ):
            raise AssertionError("Link row is not aligned with its descriptor")

    return PreparedStereoFrame(
        raw_descriptor_count=raw_count,
        filtered_descriptors=np.array(filtered_descriptors, copy=True),
        links=list(links),
        stereo_matches=list(ordered_matches),
        raw_to_filtered=tuple(raw_to_filtered),
        filtered_to_raw=filtered_to_raw,
    )


def _candidate_record(
    original_index: int,
    match: cv2.DMatch,
    previous_row: int,
    current_row: int,
    supporter: bool,
    reason: str = "",
) -> Dict[str, object]:
    record = {
        "temporal_index": int(original_index),
        "raw_query_idx": int(match.queryIdx),
        "raw_train_idx": int(match.trainIdx),
        "previous_filtered_row": int(previous_row),
        "current_filtered_row": int(current_row),
        "distance": float(match.distance),
        "is_final_supporter": bool(supporter),
    }
    if reason:
        record["rejection_reason"] = reason
    return record


def prepare_temporal_transition(
    previous_frame: PreparedStereoFrame,
    current_frame: PreparedStereoFrame,
    reciprocal_matches: Sequence[cv2.DMatch],
    final_supporters: Sequence[Tuple[int, int, int]],
    consensus_count: int,
) -> PreparedTemporalTransition:
    """Map raw temporal matches into the complete lecturer row contract.

    Reciprocal matches arrive in raw descriptor space. This function converts
    them to stereo-filtered rows, resolves row conflicts deterministically,
    and returns one ordered match plus validity entry per previous-frame row.
    Only final RANSAC supporters receive a true validity value.
    """

    previous_rows = len(previous_frame.filtered_to_raw)
    current_rows = len(current_frame.filtered_to_raw)
    if previous_rows == 0 or current_rows == 0:
        raise ValueError("TrackingDB frames must contain stereo-valid rows")
    if consensus_count <= 0:
        raise ValueError("supporter percentage requires nonzero consensus")
    if len(final_supporters) > consensus_count:
        raise ValueError("supporter count exceeds consensus count")

    supporter_temporal_indices = []
    for supporter in final_supporters:
        if len(supporter) != 3:
            raise ValueError("supporter must be a three-index consensus tuple")
        temporal_index = int(supporter[2])
        if not 0 <= temporal_index < len(reciprocal_matches):
            raise ValueError("supporter temporal index is out of range")
        supporter_temporal_indices.append(temporal_index)
    if len(supporter_temporal_indices) != len(
        set(supporter_temporal_indices)
    ):
        raise ValueError("supporter temporal indices must be unique")
    supporter_set = set(supporter_temporal_indices)

    # Convert raw temporal endpoints to stereo-filtered TrackingDB rows.
    compatible = []
    rejected = []
    for original_index, match in enumerate(reciprocal_matches):
        raw_query = int(match.queryIdx)
        raw_train = int(match.trainIdx)
        if not 0 <= raw_query < previous_frame.raw_descriptor_count:
            raise ValueError("temporal queryIdx is outside previous raw space")
        if not 0 <= raw_train < current_frame.raw_descriptor_count:
            raise ValueError("temporal trainIdx is outside current raw space")
        previous_row = previous_frame.raw_to_filtered[raw_query]
        current_row = current_frame.raw_to_filtered[raw_train]
        supporter = original_index in supporter_set
        if previous_row == NO_ID or current_row == NO_ID:
            if supporter:
                raise ValueError(
                    "final supporter lacks a stereo-valid TrackingDB row"
                )
            rejected.append(
                _candidate_record(
                    original_index,
                    match,
                    previous_row,
                    current_row,
                    supporter,
                    "not stereo-valid in both frames",
                )
            )
            continue
        compatible.append(
            (
                float(match.distance),
                int(previous_row),
                int(current_row),
                int(original_index),
                match,
                supporter,
            )
        )

    # Resolve duplicate source rows and destinations before database mutation.
    used_previous_rows = set()
    used_current_rows = set()
    selected = []
    for (
        _,
        previous_row,
        current_row,
        original_index,
        match,
        supporter,
    ) in sorted(compatible, key=lambda item: item[:4]):
        reasons = []
        if previous_row in used_previous_rows:
            reasons.append("duplicate previous row")
        if current_row in used_current_rows:
            reasons.append("duplicate current destination")
        if reasons:
            rejected.append(
                _candidate_record(
                    original_index,
                    match,
                    previous_row,
                    current_row,
                    supporter,
                    " and ".join(reasons),
                )
            )
            continue
        used_previous_rows.add(previous_row)
        used_current_rows.add(current_row)
        selected.append(
            (
                previous_row,
                current_row,
                original_index,
                match,
                supporter,
            )
        )

    # Expand selected candidates to TrackingDB's complete previous-row input.
    placeholder_distance = float(np.finfo(np.float32).max)
    complete_matches = [
        cv2.DMatch(
            _queryIdx=previous_row,
            _trainIdx=0,
            _imgIdx=0,
            _distance=placeholder_distance,
        )
        for previous_row in range(previous_rows)
    ]
    validity_mask = [False] * previous_rows
    selected_records = []
    for (
        previous_row,
        current_row,
        original_index,
        match,
        supporter,
    ) in selected:
        complete_matches[previous_row] = cv2.DMatch(
            _queryIdx=previous_row,
            _trainIdx=current_row,
            _imgIdx=0,
            _distance=float(match.distance),
        )
        validity_mask[previous_row] = bool(supporter)
        selected_records.append(
            _candidate_record(
                original_index,
                match,
                previous_row,
                current_row,
                supporter,
            )
        )

    if len(complete_matches) != previous_rows:
        raise AssertionError("temporal cardinality is not previous-row complete")
    query_rows = [int(match.queryIdx) for match in complete_matches]
    if query_rows != list(range(previous_rows)):
        raise AssertionError("temporal query rows are not complete and ordered")
    if len(validity_mask) != previous_rows:
        raise AssertionError("validity mask cardinality differs")
    valid_destinations = [
        int(match.trainIdx)
        for match, valid in zip(complete_matches, validity_mask)
        if valid
    ]
    if any(not 0 <= destination < current_rows for destination in valid_destinations):
        raise AssertionError("valid temporal destination is out of range")
    if len(valid_destinations) != len(set(valid_destinations)):
        raise AssertionError("valid temporal destinations must be unique")

    supporter_count = len(final_supporters)
    supporter_percentage = 100.0 * supporter_count / consensus_count
    return PreparedTemporalTransition(
        matches=complete_matches,
        validity_mask=validity_mask,
        reciprocal_candidate_count=len(reciprocal_matches),
        stereo_compatible_candidate_count=len(compatible),
        selected_candidate_count=len(selected),
        consensus_count=int(consensus_count),
        supporter_count=supporter_count,
        selected_supporter_count=int(sum(validity_mask)),
        supporter_percentage=float(supporter_percentage),
        selected_candidates=sorted(
            selected_records,
            key=lambda item: item["previous_filtered_row"],
        ),
        rejected_candidates=sorted(
            rejected,
            key=lambda item: (
                item["distance"],
                item["raw_query_idx"],
                item["raw_train_idx"],
                item["temporal_index"],
            ),
        ),
    )


class Exercise3TrackingDBAdapter:
    """Validated sequential insertion boundary around lecturer TrackingDB."""

    def __init__(
        self,
        database: TrackingDB = None,
        retain_history: bool = True,
    ):
        self.database = database if database is not None else TrackingDB()
        if self.database.frame_num() != 0:
            raise ValueError("adapter must begin with an empty TrackingDB")
        self.retain_history = bool(retain_history)
        self.frames: List[PreparedStereoFrame] = []
        self.transitions: List[PreparedTemporalTransition] = []
        self.companion = TrackingDBCompanion()

    @classmethod
    def resume(
        cls,
        database: TrackingDB,
        last_frame: PreparedStereoFrame,
        companion: TrackingDBCompanion,
        retain_history: bool = False,
    ) -> "Exercise3TrackingDBAdapter":
        """Resume a serialized database from validated last-frame context."""

        if database.frame_num() < 1:
            raise ValueError("cannot resume an empty TrackingDB")
        if (
            len(companion.world_to_camera_poses)
            != database.frame_num()
        ):
            raise ValueError("database and companion FrameIds are misaligned")
        if (
            len(companion.supporter_percentages)
            != database.frame_num() - 1
        ):
            raise ValueError(
                "database and companion transitions are misaligned"
            )
        if len(companion.supporter_counts) != database.frame_num() - 1:
            raise ValueError(
                "database and companion supporter counts are misaligned"
            )
        if not np.array_equal(
            database.last_features(), last_frame.filtered_descriptors
        ):
            raise ValueError(
                "serialized last descriptors differ from adapter context"
            )
        stored_links = database.all_last_frame_links()
        if len(stored_links) != len(last_frame.links):
            raise ValueError(
                "serialized last Links differ from adapter context"
            )
        for stored_link, context_link in zip(
            stored_links, last_frame.links
        ):
            if (
                stored_link.x_left != context_link.x_left
                or stored_link.x_right != context_link.x_right
                or stored_link.y != context_link.y
            ):
                raise ValueError(
                    "serialized last Link values differ from adapter context"
                )
        if len(last_frame.raw_to_filtered) != (
            last_frame.raw_descriptor_count
        ):
            raise ValueError("last-frame raw mapping cardinality differs")

        adapter = cls.__new__(cls)
        adapter.database = database
        adapter.retain_history = bool(retain_history)
        adapter.frames = [last_frame]
        adapter.transitions = []
        adapter.companion = companion
        return adapter

    def add_initial_frame(
        self, frame: PreparedStereoFrame, world_to_camera_pose: np.ndarray
    ) -> int:
        """Insert frame 0 and initialize its companion pose."""

        if self.frames or self.database.frame_num() != 0:
            raise ValueError("initial frame has already been inserted")
        pose = self.companion.validate_initial(world_to_camera_pose)
        frame_id = self.database.add_frame(
            list(frame.links), frame.filtered_descriptors.copy()
        )
        if frame_id != 0:
            raise AssertionError("lecturer database did not issue FrameId 0")
        self.frames[:] = [frame]
        self.companion.append_initial(pose)
        return frame_id

    def add_frame(
        self,
        frame: PreparedStereoFrame,
        reciprocal_matches: Sequence[cv2.DMatch],
        final_supporters: Sequence[Tuple[int, int, int]],
        consensus_count: int,
        world_to_camera_pose: np.ndarray,
    ) -> Tuple[int, PreparedTemporalTransition]:
        """Adapt and insert one consecutive frame transition.

        ``TrackingDB.add_frame`` receives filtered stereo rows, a complete
        previous-row match list, and a validity mask that is true only for the
        final PnP/RANSAC supporters. Companion pose/supporter state is appended
        only after the lecturer database accepts the frame.
        """

        if not self.frames:
            raise ValueError("insert the initial frame first")
        expected_frame_id = len(
            self.companion.world_to_camera_poses
        )
        if self.database.frame_num() != expected_frame_id:
            raise ValueError("adapter and TrackingDB FrameIds are misaligned")
        transition = prepare_temporal_transition(
            self.frames[-1],
            frame,
            reciprocal_matches,
            final_supporters,
            consensus_count,
        )
        pose = self.companion.validate_transition(
            expected_frame_id,
            world_to_camera_pose,
            transition.supporter_percentage,
            transition.supporter_count,
        )
        frame_id = self.database.add_frame(
            list(frame.links),
            frame.filtered_descriptors.copy(),
            transition.matches,
            transition.validity_mask,
        )
        if frame_id != expected_frame_id:
            raise AssertionError("lecturer FrameId issuance is not consecutive")
        if self.retain_history:
            self.frames.append(frame)
            self.transitions.append(transition)
        else:
            self.frames[:] = [frame]
            self.transitions[:] = [transition]
        self.companion.append_transition(
            frame_id,
            pose,
            transition.supporter_percentage,
            transition.supporter_count,
        )
        return frame_id, transition
