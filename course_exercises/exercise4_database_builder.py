"""Scientific sequential construction of the Exercise 4 TrackingDB.

Each consecutive-frame transition reuses the canonical Exercise 3 stereo,
temporal matching, triangulation, PnP, and RANSAC primitives. Final supporters
are adapted into lecturer TrackingDB row semantics before track insertion.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import platform
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import tools
from tracking_database import Link, NO_ID, TrackingDB
from tracking_database_adapter import (
    Exercise3TrackingDBAdapter,
    PreparedStereoFrame,
    PreparedTemporalTransition,
    TrackingDBCompanion,
    prepare_stereo_frame,
)


COMPANION_SCHEMA = "exercise4-trackingdb-companion-v1"
POSE_CONVENTION = "world-to-camera: x_i = R_i X + t_i"


@dataclass
class RawStereoFrame:
    """Raw keypoints, descriptors, and stereo inliers for one KITTI frame."""

    frame_id: int
    keypoints_left: Sequence[cv2.KeyPoint]
    descriptors_left: np.ndarray
    keypoints_right: Sequence[cv2.KeyPoint]
    right_descriptor_count: int
    raw_stereo_match_count: int
    stereo_inliers: Sequence[cv2.DMatch]


@dataclass
class Exercise4BuildResult:
    """Finished TrackingDB, companion state, and per-stage build evidence."""

    adapter: Exercise3TrackingDBAdapter
    intrinsic: np.ndarray
    initial_right_pose: np.ndarray
    raw_left_feature_counts: List[int]
    raw_right_feature_counts: List[int]
    raw_stereo_match_counts: List[int]
    stereo_row_counts: List[int]
    transitions: List[Dict[str, object]]
    consistency_checked_after_frame_ids: List[int]
    total_runtime_seconds: float
    initial_frame_runtime_seconds: float
    provenance: Dict[str, object]
    checkpoint_round_trip: Optional[Dict[str, object]]
    last_raw_frame: RawStereoFrame

    @property
    def database(self) -> TrackingDB:
        return self.adapter.database

    @property
    def companion(self) -> TrackingDBCompanion:
        return self.adapter.companion

    def scientific_state(self) -> Dict[str, object]:
        """Return deterministic state, excluding runtime measurements."""

        transition_fields = [
            {
                key: value
                for key, value in transition.items()
                if key != "runtime_seconds"
            }
            for transition in self.transitions
        ]
        return {
            "raw_left_feature_counts": self.raw_left_feature_counts,
            "raw_right_feature_counts": self.raw_right_feature_counts,
            "raw_stereo_match_counts": self.raw_stereo_match_counts,
            "stereo_row_counts": self.stereo_row_counts,
            "transitions": transition_fields,
            "companion": self.companion.as_dict(),
            "database_digest": database_digest(self.database),
            "consistency_checked_after_frame_ids": (
                self.consistency_checked_after_frame_ids
            ),
        }


@dataclass
class _EstimatedConsecutiveTransition:
    """Pose, correspondence, and supporter result for one frame pair."""

    front_raw_frame: RawStereoFrame
    front_prepared_frame: PreparedStereoFrame
    reciprocal_matches: Sequence[cv2.DMatch]
    consensus: Sequence[Tuple[int, int, int]]
    final_supporters: Sequence[Tuple[int, int, int]]
    front_world_to_camera_pose: np.ndarray
    temporal_forward_match_count: int
    temporal_reverse_match_count: int
    ransac_random_hypothesis_count: int
    ransac_refinement_model_count: int
    pnp_attempt_count: int
    pnp_failure_count: int


@dataclass
class _InsertedTransitionEvidence:
    """TrackingDB insertion result and track-lifecycle evidence."""

    issued_frame_id: int
    prepared_transition: PreparedTemporalTransition
    tracks_created: int
    tracks_extended: int
    tracks_terminated: int
    supporter_only_validation: Dict[str, object]
    rejection_counts: Dict[str, int]


def reset_canonical_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def link_record(link: Link) -> List[float]:
    return [float(link.x_left), float(link.x_right), float(link.y)]


def links_record(links: Sequence[Link]) -> List[List[float]]:
    return [link_record(link) for link in links]


def array_record(array: np.ndarray) -> Dict[str, object]:
    normalized = np.ascontiguousarray(array)
    return {
        "dtype": normalized.dtype.str,
        "shape": list(normalized.shape),
        "base64": base64.b64encode(normalized.tobytes()).decode("ascii"),
    }


def array_from_record(record: Dict[str, object]) -> np.ndarray:
    dtype = np.dtype(record["dtype"])
    shape = tuple(int(value) for value in record["shape"])
    raw = base64.b64decode(record["base64"].encode("ascii"))
    array = np.frombuffer(raw, dtype=dtype)
    if array.size != int(np.prod(shape)):
        raise ValueError("serialized array byte count differs from shape")
    return array.reshape(shape).copy()


def keypoint_record(keypoint: cv2.KeyPoint) -> List[object]:
    return [
        float(keypoint.pt[0]),
        float(keypoint.pt[1]),
        float(keypoint.size),
        float(keypoint.angle),
        float(keypoint.response),
        int(keypoint.octave),
        int(keypoint.class_id),
    ]


def keypoint_from_record(record: Sequence[object]) -> cv2.KeyPoint:
    if len(record) != 7:
        raise ValueError("serialized keypoint must contain seven values")
    return cv2.KeyPoint(
        float(record[0]),
        float(record[1]),
        float(record[2]),
        float(record[3]),
        float(record[4]),
        int(record[5]),
        int(record[6]),
    )


def match_record(match: cv2.DMatch) -> List[object]:
    return [
        int(match.queryIdx),
        int(match.trainIdx),
        int(match.imgIdx),
        float(match.distance),
    ]


def match_from_record(record: Sequence[object]) -> cv2.DMatch:
    if len(record) != 4:
        raise ValueError("serialized match must contain four values")
    return cv2.DMatch(
        _queryIdx=int(record[0]),
        _trainIdx=int(record[1]),
        _imgIdx=int(record[2]),
        _distance=float(record[3]),
    )


def raw_frame_record(frame: RawStereoFrame) -> Dict[str, object]:
    return {
        "frame_id": int(frame.frame_id),
        "keypoints_left": [
            keypoint_record(keypoint)
            for keypoint in frame.keypoints_left
        ],
        "descriptors_left": array_record(frame.descriptors_left),
        "keypoints_right": [
            keypoint_record(keypoint)
            for keypoint in frame.keypoints_right
        ],
        "right_descriptor_count": int(frame.right_descriptor_count),
        "raw_stereo_match_count": int(frame.raw_stereo_match_count),
        "stereo_inliers": [
            match_record(match) for match in frame.stereo_inliers
        ],
    }


def raw_frame_from_record(record: Dict[str, object]) -> RawStereoFrame:
    frame = RawStereoFrame(
        frame_id=int(record["frame_id"]),
        keypoints_left=tuple(
            keypoint_from_record(value)
            for value in record["keypoints_left"]
        ),
        descriptors_left=array_from_record(
            record["descriptors_left"]
        ),
        keypoints_right=tuple(
            keypoint_from_record(value)
            for value in record["keypoints_right"]
        ),
        right_descriptor_count=int(record["right_descriptor_count"]),
        raw_stereo_match_count=int(record["raw_stereo_match_count"]),
        stereo_inliers=tuple(
            match_from_record(value)
            for value in record["stereo_inliers"]
        ),
    )
    if frame.descriptors_left.shape[0] != len(frame.keypoints_left):
        raise ValueError(
            "serialized raw descriptors and left keypoints differ"
        )
    if frame.right_descriptor_count != len(frame.keypoints_right):
        raise ValueError(
            "serialized right descriptor and keypoint counts differ"
        )
    return frame


class CountingMatcher:
    """Observe canonical forward/reverse cardinalities without rematching."""

    def __init__(self, matcher):
        self.matcher = matcher
        self.match_counts: List[int] = []

    def match(self, *args, **kwargs):
        matches = self.matcher.match(*args, **kwargs)
        self.match_counts.append(len(matches))
        return matches


def database_digest_payload(database: TrackingDB) -> Dict[str, object]:
    return {
        "last_frame_id": int(database.last_frameId),
        "last_track_id": int(database.last_trackId),
        "track_id_to_frames": {
            str(track_id): [int(frame_id) for frame_id in frames]
            for track_id, frames in sorted(
                database.trackId_to_frames.items()
            )
        },
        "frame_id_to_track_rows": {
            str(frame_id): [int(track_id) for track_id in rows]
            for frame_id, rows in sorted(
                database.frameId_to_trackIds_list.items()
            )
        },
        "frame_descriptor_digests": {
            str(frame_id): array_sha256(features)
            for frame_id, features in sorted(
                database.frameId_to_lfeature.items()
            )
        },
        "tracked_links": {
            "{}:{}".format(frame_id, track_id): link_record(link)
            for (frame_id, track_id), link in sorted(
                database.linkId_to_link.items()
            )
        },
        "previous_frame_links": links_record(
            database.all_last_frame_links()
        ),
        "leftover_links": {
            str(frame_id): links_record(links)
            for frame_id, links in sorted(
                database.leftover_links.items()
            )
        },
    }


def database_digest(database: TrackingDB) -> str:
    return canonical_sha256(database_digest_payload(database))


def check_tracking_database_consistency(database: TrackingDB) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        database._check_consistency()


def validate_tracking_database_consistency_linear(
    database: TrackingDB,
) -> None:
    """Validate lecturer invariants in linear time and bounded memory."""

    def validate_link(link: Link) -> None:
        values = np.asarray(
            [link.x_left, link.x_right, link.y], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise AssertionError("Link contains a nonfinite observation")
        if not link.x_left > link.x_right:
            raise AssertionError("Link violates positive disparity")

    expected_frame_ids = range(database.last_frameId + 1)
    for mapping_name, mapping in (
        ("features", database.frameId_to_lfeature),
        ("TrackId rows", database.frameId_to_trackIds_list),
    ):
        if len(mapping) != database.frame_num():
            raise AssertionError(
                "{} mapping FrameId count differs".format(mapping_name)
            )
        for frame_id in expected_frame_ids:
            if frame_id not in mapping:
                raise AssertionError(
                    "{} mapping lacks FrameId {}".format(
                        mapping_name, frame_id
                    )
                )

    link_count = len(database.linkId_to_link)
    track_link_count = 0
    expected_track_id = 0
    for track_id, frames in database.trackId_to_frames.items():
        if track_id != expected_track_id:
            raise AssertionError("issued TrackIds are not consecutive")
        expected_track_id += 1
        if len(frames) < 2:
            raise AssertionError("track length is smaller than two")
        first_frame = frames[0]
        for offset, frame_id in enumerate(frames):
            if frame_id != first_frame + offset:
                raise AssertionError(
                    "track FrameIds are not consecutive"
                )
            if not 0 <= frame_id <= database.last_frameId:
                raise AssertionError("track FrameId is out of range")
            link = database.linkId_to_link.get((frame_id, track_id))
            if link is None:
                raise AssertionError(
                    "track-to-frame entry lacks a Link"
                )
            validate_link(link)
        track_link_count += len(frames)
    if expected_track_id != database.last_trackId + 1:
        raise AssertionError("last TrackId differs from issued tracks")
    if track_link_count != link_count:
        raise AssertionError("track-derived Link count differs")

    frame_link_count = 0
    for frame_id in expected_frame_ids:
        features = database.frameId_to_lfeature[frame_id]
        rows = database.frameId_to_trackIds_list[frame_id]
        if features.ndim != 2:
            raise AssertionError("descriptor array is not two-dimensional")
        if features.shape[0] != len(rows):
            raise AssertionError("descriptor and TrackId rows differ")
        if not np.all(np.isfinite(features)):
            raise AssertionError("descriptor array contains nonfinite values")

        frame_track_ids = set()
        for track_id in rows:
            if track_id == NO_ID:
                continue
            if not 0 <= track_id <= database.last_trackId:
                raise AssertionError("frame row TrackId is out of range")
            if track_id in frame_track_ids:
                raise AssertionError(
                    "one TrackId occupies multiple frame rows"
                )
            frame_track_ids.add(track_id)
            if (frame_id, track_id) not in database.linkId_to_link:
                raise AssertionError("frame row lacks a tracked Link")
        frame_link_count += len(frame_track_ids)

        if frame_id < database.last_frameId:
            leftovers = database.leftover_links.get(frame_id)
            if leftovers is None:
                raise AssertionError("finalized frame lacks leftover Links")
            if len(frame_track_ids) + len(leftovers) != features.shape[0]:
                raise AssertionError(
                    "tracked and leftover row cardinality differs"
                )
            for link in leftovers:
                validate_link(link)
        elif frame_id in database.leftover_links:
            raise AssertionError(
                "last frame unexpectedly has finalized leftovers"
            )

    if frame_link_count != link_count:
        raise AssertionError("frame-derived Link count differs")
    expected_leftover_frame_count = max(0, database.last_frameId)
    if len(database.leftover_links) != expected_leftover_frame_count:
        raise AssertionError("leftover Link FrameId count differs")
    for frame_id in range(expected_leftover_frame_count):
        if frame_id not in database.leftover_links:
            raise AssertionError("leftover Link FrameIds differ")
    last_links = database.all_last_frame_links()
    if last_links is None or len(last_links) != (
        database.last_features().shape[0]
    ):
        raise AssertionError("last-frame Link cardinality differs")
    for link in last_links:
        validate_link(link)


def extract_canonical_stereo_frame(
    frame_id: int,
) -> Tuple[RawStereoFrame, PreparedStereoFrame]:
    """Extract stereo features and prepare stereo-valid TrackingDB rows."""

    image_left, image_right = tools.read_images_from_dataset(frame_id)
    if image_left is None or image_right is None:
        raise FileNotFoundError(
            "KITTI frame {} stereo images are unavailable".format(frame_id)
        )
    (
        keypoints_left,
        descriptors_left,
        keypoints_right,
        descriptors_right,
        stereo_inliers,
        stereo_outliers,
    ) = tools.extract_keypoints_and_inliers(image_left, image_right)
    prepared = prepare_stereo_frame(
        descriptors_left,
        keypoints_left,
        keypoints_right,
        stereo_inliers,
    )
    raw = RawStereoFrame(
        frame_id=frame_id,
        keypoints_left=keypoints_left,
        descriptors_left=descriptors_left,
        keypoints_right=keypoints_right,
        right_descriptor_count=int(descriptors_right.shape[0]),
        raw_stereo_match_count=(
            len(stereo_inliers) + len(stereo_outliers)
        ),
        stereo_inliers=stereo_inliers,
    )
    return raw, prepared


def _rejection_counts(transition) -> Dict[str, int]:
    not_stereo_valid = 0
    duplicate_previous = 0
    duplicate_destination = 0
    conflict_records = 0
    for record in transition.rejected_candidates:
        reason = record["rejection_reason"]
        if "not stereo-valid" in reason:
            not_stereo_valid += 1
        previous_conflict = "duplicate previous row" in reason
        destination_conflict = "duplicate current destination" in reason
        duplicate_previous += int(previous_conflict)
        duplicate_destination += int(destination_conflict)
        conflict_records += int(
            previous_conflict or destination_conflict
        )
    return {
        "total_preflight_candidate_rejections": len(
            transition.rejected_candidates
        ),
        "not_stereo_valid_in_both_frames": not_stereo_valid,
        "duplicate_previous_row": duplicate_previous,
        "duplicate_current_destination": duplicate_destination,
        "duplicate_or_destination_conflict_records": conflict_records,
        "validation_exception_rejections": 0,
    }


def _verify_supporter_only_insertion(
    adapter: Exercise3TrackingDBAdapter,
    back_frame_id: int,
    front_frame_id: int,
    transition,
) -> Dict[str, object]:
    database = adapter.database
    back_rows = database.frameId_to_trackIds_list[back_frame_id]
    front_rows = database.frameId_to_trackIds_list[front_frame_id]
    selected_by_previous_row = {
        int(record["previous_filtered_row"]): record
        for record in transition.selected_candidates
    }
    accepted_track_ids = set()
    consensus_non_supporters = 0
    consensus_non_supporters_entered = 0

    for previous_row, (match, valid) in enumerate(
        zip(transition.matches, transition.validity_mask)
    ):
        record = selected_by_previous_row.get(previous_row)
        if record is None:
            if valid:
                raise AssertionError("placeholder temporal row is valid")
            continue
        current_row = int(match.trainIdx)
        previous_track_id = back_rows[previous_row]
        current_track_id = front_rows[current_row]
        if bool(record["is_final_supporter"]) != bool(valid):
            raise AssertionError(
                "validity mask differs from final-supporter membership"
            )
        if valid:
            if (
                previous_track_id == NO_ID
                or previous_track_id != current_track_id
            ):
                raise AssertionError(
                    "final supporter did not create or extend one TrackId"
                )
            accepted_track_ids.add(previous_track_id)
        else:
            consensus_non_supporters += 1
            if (
                previous_track_id != NO_ID
                and previous_track_id == current_track_id
            ):
                consensus_non_supporters_entered += 1

    connected_track_ids = set(database.tracks(back_frame_id)).intersection(
        database.tracks(front_frame_id)
    )
    if connected_track_ids != accepted_track_ids:
        raise AssertionError(
            "connectivity differs from final-supporter TrackIds"
        )
    if len(accepted_track_ids) != transition.supporter_count:
        raise AssertionError("final-supporter count differs after insertion")
    if transition.selected_supporter_count != transition.supporter_count:
        raise AssertionError("adapter rejected a final supporter")
    if consensus_non_supporters_entered:
        raise AssertionError("a consensus non-supporter entered TrackingDB")

    return {
        "true_validity_mask_count": int(sum(transition.validity_mask)),
        "connecting_track_count": len(connected_track_ids),
        "final_supporter_validity_exact": True,
        "selected_consensus_non_supporter_count": (
            consensus_non_supporters
        ),
        "selected_consensus_non_supporters_entered": 0,
        "reciprocal_non_consensus_candidates_entered": 0,
    }


def _last_frame_context(
    frame: PreparedStereoFrame,
) -> Dict[str, object]:
    return {
        "raw_descriptor_count": frame.raw_descriptor_count,
        "raw_to_filtered": list(frame.raw_to_filtered),
        "filtered_to_raw": list(frame.filtered_to_raw),
        "filtered_descriptor_sha256": array_sha256(
            frame.filtered_descriptors
        ),
        "links_sha256": canonical_sha256(links_record(frame.links)),
    }


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_last_frame_context(
    database: TrackingDB, context: Dict[str, object]
) -> PreparedStereoFrame:
    descriptors = database.last_features()
    links = database.all_last_frame_links()
    if (
        array_sha256(descriptors)
        != context["filtered_descriptor_sha256"]
    ):
        raise AssertionError("serialized continuation descriptors differ")
    if canonical_sha256(links_record(links)) != context["links_sha256"]:
        raise AssertionError("serialized continuation Links differ")
    return PreparedStereoFrame(
        raw_descriptor_count=int(context["raw_descriptor_count"]),
        filtered_descriptors=descriptors,
        links=list(links),
        stereo_matches=[],
        raw_to_filtered=tuple(context["raw_to_filtered"]),
        filtered_to_raw=tuple(context["filtered_to_raw"]),
    )


def round_trip_adapter(
    adapter: Exercise3TrackingDBAdapter,
    provenance: Dict[str, object],
    raw_frame: RawStereoFrame,
) -> Tuple[
    Exercise3TrackingDBAdapter,
    RawStereoFrame,
    Dict[str, object],
]:
    """Round-trip DB plus companion and return a continuation-ready adapter."""

    if raw_frame.frame_id != adapter.database.last_frameId:
        raise ValueError("raw continuation frame differs from TrackingDB")
    database_before = database_digest(adapter.database)
    context = _last_frame_context(adapter.frames[-1])
    raw_context = raw_frame_record(raw_frame)
    artifact = {
        "schema": COMPANION_SCHEMA,
        "provenance": provenance,
        "companion": adapter.companion.as_dict(),
        "continuation": context,
        "raw_frame_continuation": raw_context,
        "python_random_state": random.getstate(),
    }
    artifact_before = canonical_sha256(artifact)

    with tempfile.TemporaryDirectory(
        prefix="exercise4_builder_checkpoint_"
    ) as temporary_directory:
        base_filename = str(Path(temporary_directory) / "tracking_db")
        artifact_path = Path(temporary_directory) / "companion.json"
        with contextlib.redirect_stdout(io.StringIO()):
            adapter.database.serialize(base_filename)
        artifact_path.write_text(
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

        loaded_database = TrackingDB()
        with contextlib.redirect_stdout(io.StringIO()):
            loaded_database.load(base_filename)
        with artifact_path.open(encoding="utf-8") as file_handle:
            loaded_artifact = json.load(file_handle)

    database_after = database_digest(loaded_database)
    artifact_after = canonical_sha256(loaded_artifact)
    if database_before != database_after:
        raise AssertionError("serialized TrackingDB state changed")
    if artifact_before != artifact_after:
        raise AssertionError("serialized companion artifact changed")
    loaded_companion = TrackingDBCompanion.from_dict(
        loaded_artifact["companion"]
    )
    random.setstate(
        _nested_tuple(loaded_artifact["python_random_state"])
    )
    loaded_last_frame = _restore_last_frame_context(
        loaded_database, loaded_artifact["continuation"]
    )
    resumed = Exercise3TrackingDBAdapter.resume(
        loaded_database,
        loaded_last_frame,
        loaded_companion,
        retain_history=False,
    )
    loaded_raw_frame = raw_frame_from_record(
        loaded_artifact["raw_frame_continuation"]
    )
    if raw_frame_record(loaded_raw_frame) != raw_context:
        raise AssertionError("serialized raw continuation frame changed")
    if loaded_raw_frame.frame_id != loaded_database.last_frameId:
        raise AssertionError(
            "serialized raw frame is not TrackingDB's last FrameId"
        )
    check_tracking_database_consistency(resumed.database)
    return resumed, loaded_raw_frame, {
        "frame_id": loaded_database.last_frameId,
        "database_round_trip_identical": True,
        "companion_round_trip_identical": True,
        "raw_frame_round_trip_identical": True,
        "continuation_adapter_restored": True,
        "python_random_state_restored": True,
        "continuation_verification_scope": (
            "same process; OpenCV exposes no RNG-state getter"
        ),
        "opencv_rng_state_serialized": False,
        "raw_descriptor_sha256": array_sha256(
            loaded_raw_frame.descriptors_left
        ),
        "raw_keypoint_counts": {
            "left": len(loaded_raw_frame.keypoints_left),
            "right": len(loaded_raw_frame.keypoints_right),
        },
        "stereo_inlier_count": len(
            loaded_raw_frame.stereo_inliers
        ),
        "database_digest": database_after,
        "companion_artifact_sha256": artifact_after,
        "temporary_files_retained": False,
    }


def build_exercise4_database(
    frame_count: int,
    seed: int = 0,
    validate_after_every_insertion: bool = True,
    checkpoint_frame_id: Optional[int] = None,
    final_consistency_mode: str = "lecturer",
    transition_observer: Optional[
        Callable[[Dict[str, object], np.ndarray, np.ndarray], None]
    ] = None,
) -> Exercise4BuildResult:
    """Build the lecturer database and companion state for frames 0..N-1.

    Frames are processed strictly in order. Frame 0 initializes the TrackingDB;
    every later frame is connected to its predecessor only through the final
    four-view PnP/RANSAC supporters.
    """

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if checkpoint_frame_id is not None and not (
        0 <= checkpoint_frame_id < frame_count - 1
    ):
        raise ValueError(
            "checkpoint must precede the final requested frame"
        )
    if final_consistency_mode not in ("lecturer", "streaming"):
        raise ValueError(
            "final_consistency_mode must be 'lecturer' or 'streaming'"
        )

    reset_canonical_seeds(seed)
    total_start = time.perf_counter()
    intrinsic, initial_left, initial_right = tools.read_cameras_matrices()
    initial_right_rotation = initial_right[:, :3]
    initial_right_translation = initial_right[:, 3:]
    provenance = {
        "builder": "exercise4_database_builder.build_exercise4_database",
        "dataset": "KITTI odometry sequence 00",
        "frame_range_inclusive": [0, frame_count - 1],
        "frame_count": frame_count,
        "seed": seed,
        "pose_convention": POSE_CONVENTION,
        "detector": dict(tools.CANONICAL_SIFT_PARAMETERS),
        "matcher_index": dict(
            tools.CANONICAL_FLANN_INDEX_PARAMETERS
        ),
        "matcher_search": dict(
            tools.CANONICAL_FLANN_SEARCH_PARAMETERS
        ),
        "stereo_policy": (
            "vertical agreement <= 2 pixels and positive disparity"
        ),
        "temporal_policy": "reciprocal one-nearest-neighbor",
        "track_validity_policy": (
            "final four-view custom PnP/RANSAC supporters only"
        ),
        "canonical_sequential_order": True,
        "validate_after_every_insertion": bool(
            validate_after_every_insertion
        ),
        "final_consistency_mode": final_consistency_mode,
        "transition_observer_enabled": transition_observer is not None,
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "source_sha256": {
            "exercise4_database_builder.py": file_sha256(
                Path(__file__).resolve()
            ),
            "tracking_database.py": file_sha256(
                Path(__file__).resolve().parent / "tracking_database.py"
            ),
            "ex3.py": file_sha256(
                Path(__file__).resolve().parent / "ex3.py"
            ),
            "tools.py": file_sha256(
                Path(__file__).resolve().parent / "tools.py"
            ),
            "tracking_database_adapter.py": file_sha256(
                Path(__file__).resolve().parent
                / "tracking_database_adapter.py"
            ),
        },
    }

    # Extract and insert frame 0 to establish stereo rows and the initial pose.
    initial_start = time.perf_counter()
    raw_back, prepared_back = extract_canonical_stereo_frame(0)
    adapter = Exercise3TrackingDBAdapter(retain_history=False)
    adapter.add_initial_frame(prepared_back, initial_left)
    consistency_frame_ids = []
    if validate_after_every_insertion:
        validate_tracking_database_consistency_linear(adapter.database)
        consistency_frame_ids.append(0)
    initial_runtime = time.perf_counter() - initial_start
    raw_left_counts = [int(raw_back.descriptors_left.shape[0])]
    raw_right_counts = [raw_back.right_descriptor_count]
    raw_stereo_match_counts = [raw_back.raw_stereo_match_count]
    stereo_row_counts = [len(prepared_back.filtered_to_raw)]
    transitions: List[Dict[str, object]] = []
    checkpoint_evidence = None

    for front_frame_id in range(1, frame_count):
        transition_start = time.perf_counter()
        back_frame_id = front_frame_id - 1
        back_pose = adapter.companion.world_to_camera_poses[
            back_frame_id
        ]
        # Estimate the consecutive pose and final four-view supporters.
        estimated = _estimate_consecutive_frame_transition(
            raw_back=raw_back,
            front_frame_id=front_frame_id,
            back_world_to_camera_pose=back_pose,
            intrinsic=intrinsic,
            initial_right_rotation=initial_right_rotation,
            initial_right_translation=initial_right_translation,
        )

        # Adapt supporter indices to row space, then create or extend tracks.
        inserted = _insert_supported_transition(
            adapter=adapter,
            back_frame_id=back_frame_id,
            front_frame_id=front_frame_id,
            estimated=estimated,
            validate_after_insertion=validate_after_every_insertion,
            consistency_frame_ids=consistency_frame_ids,
        )
        raw_left_counts.append(
            int(estimated.front_raw_frame.descriptors_left.shape[0])
        )
        raw_right_counts.append(
            estimated.front_raw_frame.right_descriptor_count
        )
        raw_stereo_match_counts.append(
            estimated.front_raw_frame.raw_stereo_match_count
        )
        stereo_row_counts.append(
            len(estimated.front_prepared_frame.filtered_to_raw)
        )
        transitions.append(
            _build_transition_metadata(
                back_frame_id=back_frame_id,
                front_frame_id=front_frame_id,
                raw_back=raw_back,
                estimated=estimated,
                inserted=inserted,
                back_stereo_row_count=stereo_row_counts[-2],
                front_stereo_row_count=stereo_row_counts[-1],
                stereo_raw_match_count_back=(
                    raw_stereo_match_counts[-2]
                ),
                stereo_raw_match_count_front=(
                    raw_stereo_match_counts[-1]
                ),
                transition_start=transition_start,
            )
        )
        if transition_observer is not None:
            transition_observer(
                dict(transitions[-1]),
                back_pose.copy(),
                estimated.front_world_to_camera_pose.copy(),
            )

        raw_back = estimated.front_raw_frame
        prepared_back = estimated.front_prepared_frame
        if checkpoint_frame_id == front_frame_id:
            checkpoint_provenance = dict(provenance)
            checkpoint_provenance["checkpoint_frame_id"] = (
                front_frame_id
            )
            adapter, raw_back, checkpoint_evidence = round_trip_adapter(
                adapter,
                checkpoint_provenance,
                raw_back,
            )

    expected_consistency_frame_ids = (
        list(range(frame_count))
        if validate_after_every_insertion
        else []
    )
    if consistency_frame_ids != expected_consistency_frame_ids:
        raise AssertionError(
            "TrackingDB consistency evidence differs from validation mode"
        )
    if len(adapter.companion.world_to_camera_poses) != frame_count:
        raise AssertionError("companion poses are not FrameId-aligned")
    if len(adapter.companion.supporter_percentages) != frame_count - 1:
        raise AssertionError(
            "companion percentages are not transition-aligned"
        )
    if len(adapter.companion.supporter_counts) != frame_count - 1:
        raise AssertionError(
            "companion supporter counts are not transition-aligned"
        )
    expected_pairs = [
        (frame_id, frame_id + 1)
        for frame_id in range(frame_count - 1)
    ]
    if adapter.companion.transition_frame_pairs != expected_pairs:
        raise AssertionError("companion transition FrameIds differ")
    validate_tracking_database_consistency_linear(adapter.database)
    if final_consistency_mode == "lecturer":
        check_tracking_database_consistency(adapter.database)

    return Exercise4BuildResult(
        adapter=adapter,
        intrinsic=intrinsic,
        initial_right_pose=initial_right,
        raw_left_feature_counts=raw_left_counts,
        raw_right_feature_counts=raw_right_counts,
        raw_stereo_match_counts=raw_stereo_match_counts,
        stereo_row_counts=stereo_row_counts,
        transitions=transitions,
        consistency_checked_after_frame_ids=consistency_frame_ids,
        total_runtime_seconds=time.perf_counter() - total_start,
        initial_frame_runtime_seconds=initial_runtime,
        provenance=provenance,
        checkpoint_round_trip=checkpoint_evidence,
        last_raw_frame=raw_back,
    )


def _estimate_consecutive_frame_transition(
    raw_back: RawStereoFrame,
    front_frame_id: int,
    back_world_to_camera_pose: np.ndarray,
    intrinsic: np.ndarray,
    initial_right_rotation: np.ndarray,
    initial_right_translation: np.ndarray,
) -> _EstimatedConsecutiveTransition:
    """Estimate one consecutive pose and its final RANSAC supporters.

    The returned pose is the front left camera's world-to-camera transform;
    supporter tuples retain their back-stereo, front-stereo, and temporal
    match indices for the adapter stage.
    """

    back_rotation = back_world_to_camera_pose[:, :3]
    back_translation = back_world_to_camera_pose[:, 3:]
    back_right_rotation, back_right_translation = (
        tools.calculate_right_camera_matrix(
            back_rotation,
            back_translation,
            initial_right_rotation,
            initial_right_translation,
        )
    )
    # Triangulate the back-frame stereo inliers in the shared world frame.
    point_cloud = tools.cv_triangulate_matched_points(
        raw_back.keypoints_left,
        raw_back.keypoints_right,
        raw_back.stereo_inliers,
        intrinsic,
        back_rotation,
        back_translation,
        back_right_rotation,
        back_right_translation,
    )

    # Do not move this extraction ahead of the preceding transition.
    raw_front, prepared_front = extract_canonical_stereo_frame(
        front_frame_id
    )
    # Match consecutive left images in both directions and retain reciprocity.
    counting_matcher = CountingMatcher(tools.MATCHER)
    temporal_matches = tools.match_reciprocal(
        raw_back.descriptors_left,
        raw_front.descriptors_left,
        matcher=counting_matcher,
    )
    if len(counting_matcher.match_counts) != 2:
        raise AssertionError(
            "canonical reciprocal matcher did not run both directions"
        )
    forward_match_count, reverse_match_count = counting_matcher.match_counts
    # Require temporal endpoints to participate in both stereo pairs.
    consensus = tools.find_consensus_matches_indices(
        raw_back.stereo_inliers,
        raw_front.stereo_inliers,
        temporal_matches,
    )
    model_counts = {
        "random": 0,
        "refinement": 0,
        "pnp_attempts": 0,
        "pnp_failures": 0,
    }
    original_build_model = tools.build_model
    original_calculate_front_camera_matrix = (
        tools.calculate_front_camera_matrix
    )

    def counted_build_model(*args, **kwargs):
        if kwargs.get("use_random", True):
            model_counts["random"] += 1
        else:
            model_counts["refinement"] += 1
        return original_build_model(*args, **kwargs)

    def counted_calculate_front_camera_matrix(*args, **kwargs):
        model_counts["pnp_attempts"] += 1
        result = original_calculate_front_camera_matrix(*args, **kwargs)
        if not result[0]:
            model_counts["pnp_failures"] += 1
        return result

    tools.build_model = counted_build_model
    tools.calculate_front_camera_matrix = (
        counted_calculate_front_camera_matrix
    )
    # Estimate and refine the front pose; the final set is the track gate.
    try:
        rotations, translations, final_supporters, _ = (
            tools.estimate_projection_matrices_with_ransac(
                point_cloud,
                consensus,
                raw_back.stereo_inliers,
                raw_front.stereo_inliers,
                raw_back.keypoints_left,
                raw_back.keypoints_right,
                raw_front.keypoints_left,
                raw_front.keypoints_right,
                intrinsic,
                back_rotation,
                back_translation,
                initial_right_rotation,
                initial_right_translation,
                verbose=False,
            )
        )
    finally:
        tools.build_model = original_build_model
        tools.calculate_front_camera_matrix = (
            original_calculate_front_camera_matrix
        )
    front_pose = np.hstack((rotations[2], translations[2]))
    return _EstimatedConsecutiveTransition(
        front_raw_frame=raw_front,
        front_prepared_frame=prepared_front,
        reciprocal_matches=temporal_matches,
        consensus=consensus,
        final_supporters=final_supporters,
        front_world_to_camera_pose=front_pose,
        temporal_forward_match_count=forward_match_count,
        temporal_reverse_match_count=reverse_match_count,
        ransac_random_hypothesis_count=model_counts["random"],
        ransac_refinement_model_count=model_counts["refinement"],
        pnp_attempt_count=model_counts["pnp_attempts"],
        pnp_failure_count=model_counts["pnp_failures"],
    )


def _insert_supported_transition(
    adapter: Exercise3TrackingDBAdapter,
    back_frame_id: int,
    front_frame_id: int,
    estimated: _EstimatedConsecutiveTransition,
    validate_after_insertion: bool,
    consistency_frame_ids: List[int],
) -> _InsertedTransitionEvidence:
    """Insert only final supporters and validate the resulting track state.

    The adapter converts raw Exercise 3 indices to TrackingDB rows and then
    delegates track mutation to ``TrackingDB.add_frame``.
    """

    previous_rows_before = list(
        adapter.database.frameId_to_trackIds_list[back_frame_id]
    )
    active_previous_before = {
        track_id
        for track_id in previous_rows_before
        if track_id != NO_ID
    }
    last_track_id_before = adapter.database.last_trackId
    issued_frame_id, transition = adapter.add_frame(
        estimated.front_prepared_frame,
        estimated.reciprocal_matches,
        estimated.final_supporters,
        len(estimated.consensus),
        estimated.front_world_to_camera_pose,
    )
    if issued_frame_id != front_frame_id:
        raise AssertionError("TrackingDB FrameId issuance differs")
    if validate_after_insertion:
        validate_tracking_database_consistency_linear(adapter.database)
        consistency_frame_ids.append(front_frame_id)

    current_tracks = set(adapter.database.tracks(front_frame_id))
    created_tracks = {
        track_id
        for track_id in current_tracks
        if track_id > last_track_id_before
    }
    expected_created_tracks = set(
        range(
            last_track_id_before + 1,
            adapter.database.last_trackId + 1,
        )
    )
    if created_tracks != expected_created_tracks:
        raise AssertionError(
            "new TrackIds differ from the monotonic issued range"
        )
    extended_tracks = current_tracks.intersection(active_previous_before)
    terminated_tracks = active_previous_before.difference(extended_tracks)
    if len(created_tracks) + len(extended_tracks) != len(
        estimated.final_supporters
    ):
        raise AssertionError("track lifecycle differs from final supporters")
    supporter_only = _verify_supporter_only_insertion(
        adapter,
        back_frame_id,
        front_frame_id,
        transition,
    )
    rejections = _rejection_counts(transition)
    if transition.selected_candidate_count != len(estimated.consensus):
        raise AssertionError(
            "adapter candidates differ from four-view consensus"
        )
    if (
        transition.selected_candidate_count
        + rejections["total_preflight_candidate_rejections"]
        != len(estimated.reciprocal_matches)
    ):
        raise AssertionError("reciprocal candidate accounting differs")

    return _InsertedTransitionEvidence(
        issued_frame_id=issued_frame_id,
        prepared_transition=transition,
        tracks_created=len(created_tracks),
        tracks_extended=len(extended_tracks),
        tracks_terminated=len(terminated_tracks),
        supporter_only_validation=supporter_only,
        rejection_counts=rejections,
    )


def _build_transition_metadata(
    back_frame_id: int,
    front_frame_id: int,
    raw_back: RawStereoFrame,
    estimated: _EstimatedConsecutiveTransition,
    inserted: _InsertedTransitionEvidence,
    back_stereo_row_count: int,
    front_stereo_row_count: int,
    stereo_raw_match_count_back: int,
    stereo_raw_match_count_front: int,
    transition_start: float,
) -> Dict[str, object]:
    """Build one transition row with the protected field order."""

    transition = inserted.prepared_transition
    supporter_only = inserted.supporter_only_validation
    return {
        "back_frame": back_frame_id,
        "front_frame": front_frame_id,
        "success": True,
        "failure": None,
        "back_left_feature_count": int(
            raw_back.descriptors_left.shape[0]
        ),
        "front_left_feature_count": int(
            estimated.front_raw_frame.descriptors_left.shape[0]
        ),
        "back_right_feature_count": raw_back.right_descriptor_count,
        "front_right_feature_count": (
            estimated.front_raw_frame.right_descriptor_count
        ),
        "back_stereo_row_count": back_stereo_row_count,
        "front_stereo_row_count": front_stereo_row_count,
        "stereo_raw_match_count_back": stereo_raw_match_count_back,
        "stereo_raw_match_count_front": stereo_raw_match_count_front,
        "temporal_forward_match_count": (
            estimated.temporal_forward_match_count
        ),
        "temporal_reverse_match_count": (
            estimated.temporal_reverse_match_count
        ),
        "reciprocal_match_count": len(estimated.reciprocal_matches),
        "consensus_count": len(estimated.consensus),
        "supporter_count": len(estimated.final_supporters),
        "supporter_percentage": transition.supporter_percentage,
        "complete_temporal_row_count": len(transition.matches),
        "tracks_created": inserted.tracks_created,
        "tracks_extended": inserted.tracks_extended,
        "tracks_terminated": inserted.tracks_terminated,
        "connectivity": supporter_only["connecting_track_count"],
        "true_validity_mask_count": supporter_only[
            "true_validity_mask_count"
        ],
        "final_supporter_validity_exact": supporter_only[
            "final_supporter_validity_exact"
        ],
        "selected_consensus_non_supporter_count": supporter_only[
            "selected_consensus_non_supporter_count"
        ],
        "non_supporters_entered": 0,
        "ransac_random_hypothesis_count": (
            estimated.ransac_random_hypothesis_count
        ),
        "ransac_refinement_model_count": (
            estimated.ransac_refinement_model_count
        ),
        "pnp_attempt_count": estimated.pnp_attempt_count,
        "pnp_failure_count": estimated.pnp_failure_count,
        "preflight_rejections": inserted.rejection_counts,
        "runtime_seconds": time.perf_counter() - transition_start,
    }
