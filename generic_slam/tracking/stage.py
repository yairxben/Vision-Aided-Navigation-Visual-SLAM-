"""Atomic full-sequence sequential VO and tracking stage."""

from __future__ import annotations

import csv
import os
import resource
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple
from uuid import uuid4

import numpy as np

from ..artifacts.io import require_files, save_npz_atomic, write_csv_atomic
from ..artifacts.manifest import (
    StageManifest,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)
from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiEstimationContext
from ..frontend.features import FeatureFrontend, StereoFeatures, stereo_rows
from ..frontend.pnp import reset_random_generators, run_visual_odometry_ransac
from ..frontend.stereo import compose_right_camera, project_points, triangulate_stereo_matches
from ..frontend.temporal import build_four_view_consensus
from ..geometry import camera_center, homogeneous, validate_rigid_pose
from .adapter import (
    PreparedStereoFrame,
    TrackingStoreAdapter,
    prepare_stereo_frame,
    validate_tracking_store,
)
from .lecturer_store import TrackingDB


class _BackgroundWriter:
    """Writes per-transition evidence files off the tracking critical path."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence-writer")
        self._pending = deque()

    def submit(self, function, *args, **kwargs) -> None:
        self._pending.append(self._executor.submit(function, *args, **kwargs))
        # Surface write errors promptly instead of only at the end.
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()

    def finish(self) -> None:
        """Wait for every write and re-raise the first failure."""
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._executor.shutdown(wait=True)


TRANSITION_FIELDS = (
    "previous_frame_id", "current_frame_id", "previous_left_feature_count",
    "current_left_feature_count", "previous_stereo_row_count",
    "current_stereo_row_count", "reciprocal_match_count", "consensus_count",
    "supporter_count", "supporter_percentage", "tracks_created",
    "tracks_extended", "tracks_terminated", "ransac_hypothesis_attempt_count",
    "runtime_seconds",
)


@dataclass(frozen=True)
class TrackingTrajectoryOutput:
    manifest: StageManifest
    frame_ids: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class TrackingOutput(TrackingTrajectoryOutput):
    database: TrackingDB


def _peak_rss_bytes() -> int:
    # Linux ru_maxrss is KiB; this repository's supported execution platform is Linux.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _stereo_frames(
    dataset: KittiEstimationContext, frontend: FeatureFrontend, frame_count: int
) -> Iterator[StereoFeatures]:
    """Yield frames 0..frame_count-1 in order, one frame ahead when possible.

    With a deterministic frontend, frame k+1 is read and extracted on a helper
    thread while the caller processes frame k (causal: it is the newest frame
    that has arrived). FLANN draws from OpenCV's thread-local RNG, so a
    non-deterministic frontend is extracted on the caller's thread as before.
    """

    def load(frame_id: int) -> StereoFeatures:
        left_image, right_image = dataset.read_stereo(frame_id)
        return frontend.extract_stereo(left_image, right_image)

    if not frontend.is_deterministic:
        for frame_id in range(frame_count):
            yield load(frame_id)
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="frame-prefetch") as pool:
        pending = pool.submit(load, 0)
        for frame_id in range(frame_count):
            stereo = pending.result()
            if frame_id + 1 < frame_count:
                pending = pool.submit(load, frame_id + 1)
            yield stereo


def _prepared(stereo: StereoFeatures) -> PreparedStereoFrame:
    return prepare_stereo_frame(
        stereo.left.descriptors,
        stereo.left.keypoints,
        stereo.right.keypoints,
        stereo.inliers,
    )


def _save_database(database: TrackingDB, directory: Path) -> None:
    database.serialize(str(directory / "tracking_database"))


def _load_database(path: Path) -> TrackingDB:
    database = TrackingDB()
    database.load(str(path.with_suffix("")))
    return database


def _observation_arrays(database: TrackingDB) -> Dict[str, np.ndarray]:
    frame_ids = []
    track_ids = []
    u_left = []
    u_right = []
    v_mean = []
    for (frame_id, track_id), link in sorted(database.linkId_to_link.items()):
        frame_ids.append(frame_id)
        track_ids.append(track_id)
        u_left.append(link.x_left)
        u_right.append(link.x_right)
        v_mean.append(link.y)
    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "track_ids": np.asarray(track_ids, dtype=np.int64),
        "u_left_px": np.asarray(u_left, dtype=float),
        "u_right_px": np.asarray(u_right, dtype=float),
        "v_mean_px": np.asarray(v_mean, dtype=float),
    }


def _read_transition_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_tracking_trajectory_artifact(root: Path, expected_frame_count: int) -> None:
    require_files(
        root,
        (
            "pnp_trajectory.npz", "transitions.csv",
        ),
    )
    with np.load(root / "pnp_trajectory.npz", allow_pickle=False) as data:
        frame_ids = data["frame_ids"]
        poses = data["world_to_camera"]
        inverse = data["camera_to_world"]
        positions = data["positions"]
    if not np.array_equal(frame_ids, np.arange(expected_frame_count, dtype=np.int64)):
        raise ContractError("PnP FrameIds differ from runtime dataset")
    if poses.shape != (expected_frame_count, 4, 4) or inverse.shape != poses.shape:
        raise ContractError("PnP trajectory pose shapes differ")
    if positions.shape != (expected_frame_count, 3):
        raise ContractError("PnP position shape differs")
    if not np.all(np.isfinite(poses)) or not np.allclose(
        np.matmul(poses, inverse), np.eye(4), atol=1e-7, rtol=0.0
    ):
        raise ContractError("PnP poses are invalid or not inverse-aligned")
    for pose in poses:
        validate_rigid_pose(pose, atol=1e-7)
    rows = _read_transition_rows(root / "transitions.csv")
    if len(rows) != expected_frame_count - 1:
        raise ContractError("transition row count differs")


def validate_tracking_artifacts(
    root: Path, expected_frame_count: int, database: Optional[TrackingDB] = None
) -> TrackingDB:
    require_files(root, ("tracking_database.pkl", "track_observations.npz"))
    validate_tracking_trajectory_artifact(root, expected_frame_count)
    rows = _read_transition_rows(root / "transitions.csv")
    evidence_root = root / "transition_evidence"
    if not evidence_root.is_dir():
        raise ContractError("tracking transition evidence directory is missing")
    for ordinal, row in enumerate(rows, 1):
        if (
            int(row["previous_frame_id"]) != ordinal - 1
            or int(row["current_frame_id"]) != ordinal
        ):
            raise ContractError("tracking transition ordering differs")
        path = evidence_root / "transition_{:06d}_{:06d}.npz".format(
            ordinal - 1, ordinal
        )
        require_files(evidence_root, (path.name,))
        with np.load(path, allow_pickle=False) as data:
            consensus = data["consensus"]
            supporters = data["supporter_indices"]
            observed = data["observed_pixels"]
            projected = data["projected_pixels"]
        consensus_count = int(row["consensus_count"])
        supporter_count = int(row["supporter_count"])
        if (
            consensus.shape != (consensus_count, 3)
            or supporters.shape != (supporter_count,)
            or observed.shape != (4, 2, consensus_count)
            or projected.shape != observed.shape
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(projected))
            or np.any(supporters < 0)
            or np.any(supporters >= consensus_count)
        ):
            raise ContractError("tracking transition evidence shapes differ")
    if database is None:
        database = _load_database(root / "tracking_database.pkl")
    validate_tracking_store(database, expected_frame_count)
    with np.load(root / "track_observations.npz", allow_pickle=False) as data:
        lengths = {np.asarray(data[name]).shape for name in data.files}
        if len(lengths) != 1 or next(iter(lengths))[0] != database.link_num():
            raise ContractError("tracking observation population differs")
    return database


def _resume_if_complete(
    root: Path,
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    upstream_stage_ids: Mapping[str, str],
) -> Optional[TrackingOutput]:
    path = root / "manifest.json"
    if not path.is_file():
        return None
    manifest = load_stage_manifest(path)
    validate_stage_manifest(
        manifest, root, expected_stage="tracking", sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("tracking"),
        upstream_stage_ids=upstream_stage_ids,
    )
    database = validate_tracking_artifacts(root, dataset.frame_count)
    output = load_tracking_output(root, manifest, database=database)
    expected_counts = {
        "frame_count": dataset.frame_count,
        "transition_count": dataset.frame_count - 1,
        "track_count": output.database.track_num(),
        "observation_count": output.database.link_num(),
    }
    if dict(manifest.counts) != expected_counts:
        raise ContractError("tracking manifest counts differ")
    return output


def run_tracking_stage(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
    frame_limit: Optional[int] = None,
) -> TrackingOutput:
    """Run sequential VO atomically; ``frame_limit`` is test-only bounded scope."""

    destination = Path(root)
    if frame_limit is None:
        frame_count = dataset.frame_count
        resumed = _resume_if_complete(
            destination, dataset, config, upstream_stage_ids
        )
        if resumed is not None:
            return resumed
        if destination.exists():
            os.replace(
                str(destination),
                str(destination.with_name(destination.name + ".failed-" + uuid4().hex)),
            )
    else:
        frame_count = int(frame_limit)
        if not 2 <= frame_count <= dataset.frame_count:
            raise ContractError("bounded tracking frame limit is invalid")
        if destination.exists():
            raise ContractError("bounded tracking destination must not exist")

    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(prefix=".tracking-attempt-", dir=str(destination.parent))
    )
    writer = _BackgroundWriter()
    try:
        reset_random_generators(int(config.section("visual_odometry")["seed"]))
        frontend = FeatureFrontend(config.section("frontend"))
        calibration = dataset.calibration
        intrinsic = calibration.intrinsic
        initial_left = calibration.left_extrinsic
        fixed_right = calibration.right_from_left
        adapter = TrackingStoreAdapter()
        transitions = []
        world_to_camera = [initial_left.copy()]

        frames = _stereo_frames(dataset, frontend, frame_count)
        previous_stereo = next(frames)
        previous_prepared = _prepared(previous_stereo)
        adapter.add_initial(previous_prepared)
        evidence_root = attempt / "transition_evidence"
        evidence_root.mkdir()
        vo_ransac = config.section("visual_odometry")["ransac"]
        match_stereo_rows_only = (
            config.section("frontend")["temporal_matching"]["population"] == "stereo_rows"
        )

        for current_frame_id in range(1, frame_count):
            transition_started = time.perf_counter()
            previous_pose = world_to_camera[-1]
            previous_right_rotation, previous_right_translation = compose_right_camera(
                previous_pose[:3, :3], previous_pose[:3, 3:],
                fixed_right[:3, :3], fixed_right[:3, 3:],
            )
            point_cloud = triangulate_stereo_matches(
                previous_stereo.left.keypoints,
                previous_stereo.right.keypoints,
                previous_stereo.inliers,
                intrinsic,
                previous_pose[:3, :3],
                previous_pose[:3, 3:],
                previous_right_rotation,
                previous_right_translation,
            )
            current_stereo = next(frames)
            current_prepared = _prepared(current_stereo)
            if match_stereo_rows_only:
                reciprocal = frontend.reciprocal_matches_on_rows(
                    previous_stereo.left.descriptors, stereo_rows(previous_stereo),
                    current_stereo.left.descriptors, stereo_rows(current_stereo),
                )
            else:
                reciprocal = frontend.reciprocal_matches(
                    previous_stereo.left.descriptors,
                    current_stereo.left.descriptors,
                )
            consensus = build_four_view_consensus(
                previous_stereo.inliers, current_stereo.inliers, reciprocal
            )
            ransac = run_visual_odometry_ransac(
                point_cloud,
                consensus,
                previous_stereo.inliers,
                current_stereo.inliers,
                previous_stereo.left.keypoints,
                previous_stereo.right.keypoints,
                current_stereo.left.keypoints,
                current_stereo.right.keypoints,
                intrinsic,
                previous_pose[:3, :3],
                previous_pose[:3, 3:],
                fixed_right[:3, :3],
                fixed_right[:3, 3:],
                success_probability=float(vo_ransac["success_probability"]),
                initial_outlier_probability=float(vo_ransac["initial_outlier_probability"]),
                sample_size=int(vo_ransac["sample_size"]),
                maximum_distance_px=float(vo_ransac["supporter_max_distance_px"]),
                maximum_attempts=vo_ransac["maximum_hypothesis_attempts"],
            )
            pose = homogeneous(
                np.hstack((ransac.model.rotations[2], ransac.model.translations[2]))
            )
            supporter_temporal_indices = tuple(
                item.temporal_match_index for item in ransac.supporters
            )
            tracks_before = set(adapter.database.tracks(current_frame_id - 1))
            last_track_before = adapter.database.last_trackId
            issued, prepared_transition = adapter.add_transition(
                current_prepared,
                reciprocal,
                supporter_temporal_indices,
                len(consensus),
            )
            if issued != current_frame_id:
                raise ContractError("issued tracking FrameId differs")
            tracks_after = set(adapter.database.tracks(current_frame_id))
            created = {value for value in tracks_after if value > last_track_before}
            extended = tracks_before.intersection(tracks_after)
            terminated = tracks_before.difference(tracks_after)
            world_to_camera.append(pose)

            consensus_points = point_cloud[
                np.asarray([item.previous_stereo_index for item in consensus], dtype=int)
            ]
            observed = np.empty((4, 2, 0), dtype=float)
            projected = np.empty((4, 2, 0), dtype=float)
            if consensus:
                from ..frontend.temporal import actual_consensus_pixels
                observed = actual_consensus_pixels(
                    consensus, previous_stereo.inliers, current_stereo.inliers,
                    previous_stereo.left.keypoints, previous_stereo.right.keypoints,
                    current_stereo.left.keypoints, current_stereo.right.keypoints,
                )
                projected = project_points(
                    consensus_points, intrinsic, ransac.model.rotations,
                    ransac.model.translations,
                )
            writer.submit(
                save_npz_atomic,
                evidence_root / "transition_{:06d}_{:06d}.npz".format(
                    current_frame_id - 1, current_frame_id
                ),
                consensus=np.asarray([item.as_tuple() for item in consensus], dtype=np.int64).reshape(-1, 3),
                supporter_indices=np.asarray(ransac.supporter_indices, dtype=np.int64),
                observed_pixels=observed,
                projected_pixels=projected,
            )
            transitions.append(
                {
                    "previous_frame_id": current_frame_id - 1,
                    "current_frame_id": current_frame_id,
                    "previous_left_feature_count": len(previous_stereo.left.keypoints),
                    "current_left_feature_count": len(current_stereo.left.keypoints),
                    "previous_stereo_row_count": len(previous_stereo.inliers),
                    "current_stereo_row_count": len(current_stereo.inliers),
                    "reciprocal_match_count": len(reciprocal),
                    "consensus_count": len(consensus),
                    "supporter_count": len(ransac.supporters),
                    "supporter_percentage": prepared_transition.supporter_percentage,
                    "tracks_created": len(created),
                    "tracks_extended": len(extended),
                    "tracks_terminated": len(terminated),
                    "ransac_hypothesis_attempt_count": ransac.total_hypothesis_attempt_count,
                    "runtime_seconds": time.perf_counter() - transition_started,
                }
            )
            previous_stereo = current_stereo

        writer.finish()
        validate_tracking_store(adapter.database, frame_count)
        poses = np.asarray(world_to_camera, dtype=float)
        inverse = np.linalg.inv(poses)
        positions = np.asarray(
            [camera_center(pose) for pose in poses], dtype=float
        )
        _save_database(adapter.database, attempt)
        save_npz_atomic(attempt / "track_observations.npz", **_observation_arrays(adapter.database))
        save_npz_atomic(
            attempt / "pnp_trajectory.npz",
            frame_ids=np.arange(frame_count, dtype=np.int64),
            world_to_camera=poses,
            camera_to_world=inverse,
            positions=positions,
        )
        write_csv_atomic(attempt / "transitions.csv", transitions, TRANSITION_FIELDS)
        validate_tracking_artifacts(attempt, frame_count, database=adapter.database)
        manifest = StageManifest.complete(
            stage="tracking",
            sequence_id=dataset.sequence_id,
            dataset_id=dataset.dataset_id,
            config_fingerprint=config.stage_fingerprint("tracking"),
            upstream_stage_ids=upstream_stage_ids,
            counts={
                "frame_count": frame_count,
                "transition_count": frame_count - 1,
                "track_count": adapter.database.track_num(),
                "observation_count": adapter.database.link_num(),
            },
            primary_artifacts=(
                "tracking_database.pkl", "track_observations.npz",
                "pnp_trajectory.npz", "transitions.csv",
            ),
            runtime_seconds=time.perf_counter() - started,
            peak_rss_bytes=_peak_rss_bytes(),
        )
        write_stage_manifest(attempt, manifest)
        os.replace(str(attempt), str(destination))
        return TrackingOutput(
            manifest, np.arange(frame_count, dtype=np.int64), poses, inverse,
            positions, adapter.database,
        )
    except Exception:
        try:
            writer.finish()  # stop writing into the attempt before moving it
        except Exception:
            pass  # the original error is the one to report
        quarantine = attempt.with_name(attempt.name + ".failed")
        if attempt.exists():
            os.replace(str(attempt), str(quarantine))
        raise


def load_tracking_output(
    root: Path,
    manifest: Optional[StageManifest] = None,
    *,
    database: Optional[TrackingDB] = None,
) -> TrackingOutput:
    root = Path(root)
    if manifest is None:
        manifest = load_stage_manifest(root / "manifest.json")
    with np.load(root / "pnp_trajectory.npz", allow_pickle=False) as data:
        frame_ids = data["frame_ids"].copy()
        poses = data["world_to_camera"].copy()
        inverse = data["camera_to_world"].copy()
        positions = data["positions"].copy()
    if database is None:
        database = _load_database(root / "tracking_database.pkl")
    return TrackingOutput(manifest, frame_ids, poses, inverse, positions, database)


def load_tracking_trajectory_output(root: Path) -> TrackingTrajectoryOutput:
    root = Path(root)
    manifest = load_stage_manifest(root / "manifest.json")
    with np.load(root / "pnp_trajectory.npz", allow_pickle=False) as data:
        return TrackingTrajectoryOutput(
            manifest,
            data["frame_ids"].copy(),
            data["world_to_camera"].copy(),
            data["camera_to_world"].copy(),
            data["positions"].copy(),
        )


__all__ = [
    "TrackingOutput", "TrackingTrajectoryOutput", "load_tracking_output",
    "load_tracking_trajectory_output", "run_tracking_stage",
    "validate_tracking_artifacts", "validate_tracking_trajectory_artifact",
]
