"""Resumable sequential Loop Closure stage with per-current structural seals."""

from __future__ import annotations

import csv
import json
import os
import resource
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..artifacts.io import (
    read_json,
    require_files,
    save_npz_atomic,
    write_csv_atomic,
    write_json_atomic,
)
from ..artifacts.manifest import (
    StageManifest,
    load_stage_manifest,
    validate_stage_manifest,
    write_stage_manifest,
)
from ..config import AlgorithmConfig
from ..contracts import ContractError
from ..dataset.kitti import KittiEstimationContext
from ..frontend.temporal import actual_consensus_pixels
from ..geometry import validate_rigid_pose
from ..pose_graph.stage import PoseGraphOutput
from .detection import CovarianceGraph, score_previous_keyframes
from ..frontend.features import FeatureFrontend
from .matching import LoopMatchResult, StereoFeatureCache, evaluate_loop_pair
from .measurement import SmallBundleResult, measure_loop
from .policy import (
    AcceptedLoopAnchor,
    LoopClosurePolicy,
    RankedCandidate,
    evaluate_region_schedule,
    form_temporary_regions,
    matcher_result_passes,
    ordered_pair_index,
    suppressing_anchor,
)
from .sequential import CumulativePoseGraphState, PoseGraphUpdateEvidence


CANDIDATE_FIELDS = (
    "pair_index", "previous_frame_id", "current_frame_id",
    "previous_keyframe_ordinal", "current_keyframe_ordinal", "ordinal_gap",
    "rank_within_current", "factor_error", "mahalanobis_squared",
    "mahalanobis", "path_frame_ids", "path_edge_indices",
    "path_covariance_determinant", "disposition", "matcher_invocation",
    "matcher_status", "consensus_count", "supporter_count",
    "supporter_percentage", "acceptance_id",
)
CURRENT_ARTIFACTS = (
    "candidates.csv",
    "candidate_paths.npz",
    "matcher_outcomes.json",
    "accepted_factors.npz",
    "state_checkpoint.npz",
)


@dataclass(frozen=True)
class LoopClosureOutput:
    manifest: Optional[StageManifest]
    keyframe_ids: np.ndarray
    final_pose_matrices: np.ndarray
    final_marginal_covariances: np.ndarray
    accepted_previous_frame_ids: np.ndarray
    accepted_current_frame_ids: np.ndarray
    accepted_relative_pose_matrices: np.ndarray
    accepted_covariances: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.final_pose_matrices[:, :3, 3]


@dataclass
class _AcceptedCurrent:
    anchor: AcceptedLoopAnchor
    measurement: SmallBundleResult
    update: PoseGraphUpdateEvidence


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _identity(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    upstream_stage_ids: Mapping[str, str],
) -> Dict[str, object]:
    return {
        "sequence_id": dataset.sequence_id,
        "dataset_id": dataset.dataset_id,
        "config_fingerprint": config.stage_fingerprint("loop_closure"),
        "upstream_stage_ids": dict(upstream_stage_ids),
    }


def _current_name(ordinal: int, frame_id: int) -> str:
    return "current_{:04d}_{:06d}".format(ordinal, frame_id)


def _empty_factors():
    return {
        "acceptance_ids": np.empty(0, dtype=np.int64),
        "previous_frame_ids": np.empty(0, dtype=np.int64),
        "current_frame_ids": np.empty(0, dtype=np.int64),
        "previous_ordinals": np.empty(0, dtype=np.int64),
        "current_ordinals": np.empty(0, dtype=np.int64),
        "relative_pose_matrices": np.empty((0, 4, 4), dtype=float),
        "conditional_covariances": np.empty((0, 6, 6), dtype=float),
    }


def _accepted_arrays(accepted: Sequence[_AcceptedCurrent]):
    if not accepted:
        return _empty_factors()
    return {
        "acceptance_ids": np.asarray([item.anchor.acceptance_id for item in accepted], dtype=np.int64),
        "previous_frame_ids": np.asarray([item.anchor.previous_frame_id for item in accepted], dtype=np.int64),
        "current_frame_ids": np.asarray([item.anchor.current_frame_id for item in accepted], dtype=np.int64),
        "previous_ordinals": np.asarray([item.anchor.previous_ordinal for item in accepted], dtype=np.int64),
        "current_ordinals": np.asarray([item.anchor.current_ordinal for item in accepted], dtype=np.int64),
        "relative_pose_matrices": np.asarray(
            [item.measurement.optimized_relative_pose.matrix() for item in accepted], dtype=float
        ),
        "conditional_covariances": np.asarray(
            [item.measurement.conditional_covariance for item in accepted], dtype=float
        ),
    }


def _load_factors(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def _validate_current(
    directory: Path,
    ordinal: int,
    frame_id: int,
    identity: Mapping[str, object],
    keyframe_count: int,
) -> Mapping[str, object]:
    seal_path = directory / "seal.json"
    if not seal_path.is_file():
        raise ContractError("Loop current is not sealed")
    seal = read_json(seal_path)
    if any(seal.get(key) != value for key, value in identity.items()):
        raise ContractError("Loop current identity differs")
    if (
        seal.get("schema") != "generic-slam-loop-current-seal-v1"
        or seal.get("status") != "complete"
        or int(seal.get("current_ordinal", -1)) != ordinal
        or int(seal.get("current_frame_id", -1)) != frame_id
    ):
        raise ContractError("Loop current seal differs")
    require_files(directory, CURRENT_ARTIFACTS)
    with (directory / "candidates.csv").open("r", encoding="utf-8", newline="") as handle:
        if len(list(csv.DictReader(handle))) != ordinal:
            raise ContractError("Loop candidate disposition population differs")
    outcomes = read_json(directory / "matcher_outcomes.json")
    if not isinstance(outcomes, list):
        raise ContractError("Loop matcher outcomes must be a list")
    expected_invocations = int(seal.get("matcher_invocation_count", -1))
    if (
        len(outcomes) != expected_invocations
        or sorted(int(item["matcher_invocation"]) for item in outcomes)
        != list(range(1, expected_invocations + 1))
    ):
        raise ContractError("Loop matcher invocation accounting differs")
    for outcome in outcomes:
        invocation = int(outcome["matcher_invocation"])
        artifact = outcome.get("match_artifact")
        if not artifact:
            if "failure" not in outcome:
                raise ContractError("Loop matcher outcome lacks evidence or failure")
            continue
        if artifact != "match_{:02d}.npz".format(invocation):
            raise ContractError("Loop match artifact name differs")
        require_files(directory, (artifact,))
        with np.load(
            directory / artifact, allow_pickle=False
        ) as match:
            if (
                match["consensus"].ndim != 2
                or match["consensus"].shape[1] != 3
                or match["observed_pixels"].shape != (
                    4, 2, len(match["consensus"])
                )
                or np.any(match["supporter_indices"] < 0)
                or np.any(match["supporter_indices"] >= len(match["consensus"]))
            ):
                raise ContractError("Loop match artifact shapes differ")
    with np.load(directory / "state_checkpoint.npz", allow_pickle=False) as data:
        if (
            data["frame_ids"].shape != (keyframe_count,)
            or data["pose_matrices"].shape != (keyframe_count, 4, 4)
            or data["marginal_covariances"].shape != (keyframe_count, 6, 6)
            or not np.all(np.isfinite(data["pose_matrices"]))
            or not np.all(np.isfinite(data["marginal_covariances"]))
        ):
            raise ContractError("Loop current checkpoint shapes differ")
        for pose in data["pose_matrices"]:
            validate_rigid_pose(pose, atol=1e-7)
    with np.load(directory / "candidate_paths.npz", allow_pickle=False) as data:
        path_count = len(data["pair_indices"])
        if (
            data["previous_frame_ids"].shape != (path_count,)
            or data["accumulated_covariances"].shape != (path_count, 6, 6)
            or not np.all(np.isfinite(data["accumulated_covariances"]))
        ):
            raise ContractError("Loop candidate path evidence shapes differ")
    factors = _load_factors(directory / "accepted_factors.npz")
    count = len(factors["acceptance_ids"])
    if (
        factors["relative_pose_matrices"].shape != (count, 4, 4)
        or factors["conditional_covariances"].shape != (count, 6, 6)
    ):
        raise ContractError("Loop current accepted factor shapes differ")
    for acceptance_id in factors["acceptance_ids"]:
        require_files(
            directory,
            (
                "measurement_{:04d}.json".format(int(acceptance_id)),
                "measurement_{:04d}.npz".format(int(acceptance_id)),
            ),
        )
        with np.load(
            directory / "measurement_{:04d}.npz".format(int(acceptance_id)),
            allow_pickle=False,
        ) as measurement:
            if (
                measurement["relative_pose"].shape != (4, 4)
                or measurement["conditional_covariance"].shape != (6, 6)
                or measurement["joint_covariance"].shape != (12, 12)
                or measurement["joint_information"].shape != (12, 12)
                or not all(np.all(np.isfinite(measurement[name])) for name in (
                    "relative_pose", "conditional_covariance", "joint_covariance",
                    "joint_information",
                ))
            ):
                raise ContractError("Loop measurement artifact shapes differ")
    return seal


def _replay_prefix(
    root: Path,
    pose_graph: PoseGraphOutput,
    config: AlgorithmConfig,
    identity: Mapping[str, object],
):
    keyframe_ids = tuple(int(value) for value in pose_graph.keyframe_ids)
    all_factor_parts = []
    anchors = []
    last_checkpoint = pose_graph.optimized_pose_matrices
    completed = []
    expected_version = 0
    currents_root = root / "currents"
    for ordinal in range(1, len(keyframe_ids)):
        directory = currents_root / _current_name(ordinal, keyframe_ids[ordinal])
        if not directory.is_dir():
            later = [
                candidate
                for candidate in currents_root.glob("current_*")
                if candidate.is_dir()
                and candidate.name > _current_name(ordinal, keyframe_ids[ordinal])
            ] if currents_root.is_dir() else []
            if later:
                raise ContractError("Loop current seals are not a contiguous prefix")
            break
        seal = _validate_current(
            directory, ordinal, keyframe_ids[ordinal], identity, len(keyframe_ids)
        )
        factors = _load_factors(directory / "accepted_factors.npz")
        all_factor_parts.append(factors)
        for acceptance_id, previous, current, previous_ordinal, current_ordinal in zip(
            factors["acceptance_ids"], factors["previous_frame_ids"],
            factors["current_frame_ids"], factors["previous_ordinals"],
            factors["current_ordinals"],
        ):
            anchors.append(
                AcceptedLoopAnchor(
                    int(acceptance_id), int(previous), int(current),
                    int(previous_ordinal), int(current_ordinal), int(acceptance_id),
                )
            )
        expected_version += len(factors["acceptance_ids"])
        if int(seal["state_version"]) != expected_version:
            raise ContractError("Loop current state version differs")
        with np.load(directory / "state_checkpoint.npz", allow_pickle=False) as data:
            last_checkpoint = data["pose_matrices"].copy()
        completed.append(ordinal)
    if all_factor_parts:
        combined = {
            name: np.concatenate([part[name] for part in all_factor_parts], axis=0)
            for name in _empty_factors()
        }
    else:
        combined = _empty_factors()
    state = CumulativePoseGraphState.restore(
        pose_graph=pose_graph,
        accepted_previous_frame_ids=combined["previous_frame_ids"],
        accepted_current_frame_ids=combined["current_frame_ids"],
        accepted_relative_pose_matrices=combined["relative_pose_matrices"],
        accepted_covariances=combined["conditional_covariances"],
        value_matrices=last_checkpoint,
        bundle_config=config.section("bundle"),
    )
    detection = CovarianceGraph.consecutive(
        keyframe_ids,
        pose_graph.edge_covariances,
        covariance_symmetry_atol=float(config.section("bundle")["covariance_symmetry_atol"]),
    )
    for previous, current, covariance, acceptance_id in zip(
        combined["previous_frame_ids"], combined["current_frame_ids"],
        combined["conditional_covariances"], combined["acceptance_ids"],
    ):
        detection = detection.with_edge(
            int(previous), int(current), covariance,
            source="loop_acceptance_{}".format(int(acceptance_id)),
        )
    return completed, anchors, combined, state, detection


def _match_record(result: LoopMatchResult, invocation: int) -> Dict[str, object]:
    return {
        "matcher_invocation": invocation,
        "previous_frame_id": result.previous_frame_id,
        "current_frame_id": result.current_frame_id,
        "previous_raw_stereo_match_count": result.previous_stereo.raw_match_count,
        "current_raw_stereo_match_count": result.current_stereo.raw_match_count,
        "previous_stereo_inlier_count": len(result.previous_stereo.inliers),
        "current_stereo_inlier_count": len(result.current_stereo.inliers),
        "reciprocal_temporal_match_count": len(result.reciprocal_matches),
        "four_view_consensus_count": len(result.consensus),
        "supporter_count": len(result.ransac.best_supporter_indices),
        "supporter_percentage": result.supporter_percentage,
        "ransac_status": result.ransac.status,
        "ransac_terminal_reason": result.ransac.terminal_reason,
        "hypothesis_attempt_count": result.ransac.total_hypothesis_attempt_count,
    }


def _seal_current(
    *,
    currents_root: Path,
    ordinal: int,
    frame_id: int,
    identity: Mapping[str, object],
    candidate_rows: Sequence[Mapping[str, object]],
    threshold_candidates: Sequence[RankedCandidate],
    matches: Sequence[Tuple[LoopMatchResult, Dict[str, object]]],
    matcher_failures: Sequence[Mapping[str, object]],
    matcher_invocation_count: int,
    accepted: Sequence[_AcceptedCurrent],
    state: CumulativePoseGraphState,
) -> Path:
    destination = currents_root / _current_name(ordinal, frame_id)
    currents_root.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(prefix=".{}-attempt-".format(destination.name), dir=str(currents_root))
    )
    try:
        write_csv_atomic(attempt / "candidates.csv", candidate_rows, CANDIDATE_FIELDS)
        save_npz_atomic(
            attempt / "candidate_paths.npz",
            pair_indices=np.asarray([item.pair_index for item in threshold_candidates], dtype=np.int64),
            previous_frame_ids=np.asarray(
                [item.score.previous_frame_id for item in threshold_candidates], dtype=np.int64
            ),
            accumulated_covariances=np.asarray(
                [item.score.path.covariance for item in threshold_candidates], dtype=float
            ).reshape(-1, 6, 6),
        )
        outcome_records = [dict(record) for record in matcher_failures]
        for result, record in matches:
            stored_record = dict(record)
            pixels = actual_consensus_pixels(
                result.consensus,
                result.previous_stereo.inliers,
                result.current_stereo.inliers,
                result.previous_stereo.left.keypoints,
                result.previous_stereo.right.keypoints,
                result.current_stereo.left.keypoints,
                result.current_stereo.right.keypoints,
            )
            invocation = int(record["matcher_invocation"])
            stored_record["match_artifact"] = "match_{:02d}.npz".format(invocation)
            outcome_records.append(stored_record)
            save_npz_atomic(
                attempt / "match_{:02d}.npz".format(invocation),
                consensus=np.asarray([item.as_tuple() for item in result.consensus], dtype=np.int64).reshape(-1, 3),
                supporter_indices=np.asarray(result.ransac.best_supporter_indices, dtype=np.int64),
                observed_pixels=pixels,
            )
        write_json_atomic(
            attempt / "matcher_outcomes.json",
            sorted(outcome_records, key=lambda item: int(item["matcher_invocation"])),
        )
        accepted_arrays = _accepted_arrays(accepted)
        save_npz_atomic(attempt / "accepted_factors.npz", **accepted_arrays)
        for item in accepted:
            measurement = item.measurement
            save_npz_atomic(
                attempt / "measurement_{:04d}.npz".format(item.anchor.acceptance_id),
                relative_pose=np.asarray(measurement.optimized_relative_pose.matrix(), dtype=float),
                conditional_covariance=measurement.conditional_covariance,
                joint_covariance=measurement.joint_covariance,
                joint_information=measurement.joint_information,
                consensus_indices=np.asarray(
                    [value.consensus_index for value in measurement.accepted_observations],
                    dtype=np.int64,
                ),
                previous_measurements=np.asarray(
                    [value.previous_measurement for value in measurement.accepted_observations], dtype=float
                ),
                current_measurements=np.asarray(
                    [value.current_measurement for value in measurement.accepted_observations], dtype=float
                ),
            )
            write_json_atomic(
                attempt / "measurement_{:04d}.json".format(item.anchor.acceptance_id),
                {
                    "anchor": asdict(item.anchor),
                    "initial_error": measurement.initial_error,
                    "first_optimizer_error": measurement.first_optimizer_error,
                    "gauge_normalized_error": measurement.gauge_normalized_error,
                    "second_optimizer_error": measurement.second_optimizer_error,
                    "final_error": measurement.final_error,
                    "candidate_observation_count": len(measurement.candidate_observations),
                    "accepted_observation_count": len(measurement.accepted_observations),
                    "rejections": dict(measurement.rejection_reason_by_consensus_index),
                    "final_depth_rebuild_count": measurement.final_depth_rebuild_count,
                    "pose_graph_update": asdict(item.update),
                    "projection_rows": list(measurement.measurement_rows),
                },
            )
        save_npz_atomic(
            attempt / "state_checkpoint.npz",
            frame_ids=np.asarray(state.keyframe_ids, dtype=np.int64),
            pose_matrices=state.pose_matrices(),
            marginal_covariances=state.marginal_covariances(),
        )
        require_files(attempt, CURRENT_ARTIFACTS)
        seal = dict(identity)
        seal.update(
            {
                "schema": "generic-slam-loop-current-seal-v1",
                "status": "complete",
                "current_ordinal": ordinal,
                "current_frame_id": frame_id,
                "ordered_pair_count": ordinal,
                "matcher_invocation_count": int(matcher_invocation_count),
                "accepted_loop_count_this_current": len(accepted),
                "state_version": state.state_version,
                "new_anchors": [asdict(item.anchor) for item in accepted],
                "primary_artifacts": list(CURRENT_ARTIFACTS),
            }
        )
        write_json_atomic(attempt / "seal.json", seal)
        _validate_current(
            attempt, ordinal, frame_id, identity, len(state.keyframe_ids)
        )
        os.replace(str(attempt), str(destination))
        return destination
    except Exception:
        if attempt.exists():
            os.replace(str(attempt), str(attempt) + ".failed")
        raise


def _process_current(
    *,
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    policy: LoopClosurePolicy,
    ordinal: int,
    state: CumulativePoseGraphState,
    detection: CovarianceGraph,
    anchors: List[AcceptedLoopAnchor],
    currents_root: Path,
    identity: Mapping[str, object],
    feature_cache: Optional[StereoFeatureCache] = None,
) -> CovarianceGraph:
    keyframe_ids = state.keyframe_ids
    current_frame_id = keyframe_ids[ordinal]
    scores = score_previous_keyframes(
        values=state.values, keyframe_ids=keyframe_ids, graph=detection,
        current_ordinal=ordinal,
    )
    records_by_previous = {}
    eligible = []
    threshold_candidates = []
    for rank, score in enumerate(scores, 1):
        pair_index = ordered_pair_index(score.previous_keyframe_ordinal, ordinal)
        row = {
            "pair_index": pair_index,
            "previous_frame_id": score.previous_frame_id,
            "current_frame_id": score.current_frame_id,
            "previous_keyframe_ordinal": score.previous_keyframe_ordinal,
            "current_keyframe_ordinal": ordinal,
            "ordinal_gap": ordinal - score.previous_keyframe_ordinal,
            "rank_within_current": rank,
            "factor_error": score.factor_error,
            "mahalanobis_squared": score.mahalanobis_squared,
            "mahalanobis": score.mahalanobis,
            "path_frame_ids": json.dumps(score.path.frame_ids),
            "path_edge_indices": json.dumps(score.path.edge_indices),
            "path_covariance_determinant": score.path.covariance_determinant,
            "disposition": "",
            "matcher_invocation": "",
            "matcher_status": "",
            "consensus_count": "",
            "supporter_count": "",
            "supporter_percentage": "",
            "acceptance_id": "",
        }
        candidate = RankedCandidate(pair_index, score, rank)
        if score.mahalanobis_squared < policy.mahalanobis_squared_threshold:
            threshold_candidates.append(candidate)
        if ordinal - score.previous_keyframe_ordinal < policy.minimum_temporal_ordinal_gap:
            row["disposition"] = "temporal_excluded"
        elif not score.mahalanobis_squared < policy.mahalanobis_squared_threshold:
            row["disposition"] = "mahalanobis_rejected"
        else:
            anchor = suppressing_anchor(candidate, anchors, policy)
            if anchor is not None:
                row["disposition"] = "accepted_cluster_suppressed"
                row["acceptance_id"] = anchor.acceptance_id
            else:
                row["disposition"] = "eligible_not_scheduled"
                eligible.append(candidate)
        records_by_previous[score.previous_keyframe_ordinal] = row
    regions = form_temporary_regions(eligible, policy)
    matches = []
    matcher_failures = []
    accepted_current = []

    def candidate_is_live(candidate, _region, _scheduler_pass):
        anchor = suppressing_anchor(candidate, anchors, policy)
        if anchor is not None:
            row = records_by_previous[candidate.previous_ordinal]
            row["disposition"] = "accepted_cluster_suppressed"
            row["acceptance_id"] = anchor.acceptance_id
            return False
        return True

    def evaluate(candidate, _region, _scheduler_pass, invocation):
        nonlocal detection
        row = records_by_previous[candidate.previous_ordinal]
        row["matcher_invocation"] = invocation
        try:
            match = evaluate_loop_pair(
                dataset, config, candidate.score.previous_frame_id,
                candidate.score.current_frame_id, feature_cache,
            )
        except Exception as error:
            row["disposition"] = "matching_failed"
            row["matcher_status"] = "exception:{}".format(type(error).__name__)
            matcher_failures.append({
                "matcher_invocation": invocation,
                "previous_frame_id": candidate.score.previous_frame_id,
                "current_frame_id": candidate.score.current_frame_id,
                "failure": {"type": type(error).__name__, "message": str(error)},
            })
            return
        match_record = _match_record(match, invocation)
        matches.append((match, match_record))
        row["matcher_status"] = match.ransac.status
        row["consensus_count"] = len(match.consensus)
        row["supporter_count"] = len(match.ransac.best_supporter_indices)
        row["supporter_percentage"] = match.supporter_percentage
        if not matcher_result_passes(
            consensus_count=len(match.consensus),
            supporter_count=len(match.ransac.best_supporter_indices),
            supporter_percentage=match.supporter_percentage,
            terminal_status=match.ransac.status,
            policy=policy,
        ):
            row["disposition"] = "matcher_rejected"
            return
        try:
            initial_relative = state.initial_relative_pose(
                match.previous_frame_id, match.current_frame_id
            )
            measurement = measure_loop(match, initial_relative, dataset, config)
            proposed_detection = detection.with_edge(
                match.previous_frame_id,
                match.current_frame_id,
                measurement.conditional_covariance,
                source="loop_acceptance_{}".format(len(anchors) + 1),
            )
            update = state.apply_loop_factor(
                previous_frame_id=match.previous_frame_id,
                current_frame_id=match.current_frame_id,
                relative_pose=measurement.optimized_relative_pose,
                conditional_covariance=measurement.conditional_covariance,
            )
        except Exception as error:
            row["disposition"] = "measurement_or_graph_update_failed"
            match_record["measurement_failure"] = {
                "type": type(error).__name__, "message": str(error)
            }
            return
        acceptance_id = len(anchors) + 1
        anchor = AcceptedLoopAnchor(
            acceptance_id,
            match.previous_frame_id,
            match.current_frame_id,
            candidate.previous_ordinal,
            candidate.current_ordinal,
            state.state_version,
        )
        anchors.append(anchor)
        accepted_current.append(_AcceptedCurrent(anchor, measurement, update))
        detection = proposed_detection
        row["disposition"] = "accepted"
        row["acceptance_id"] = acceptance_id
        match_record["acceptance_id"] = acceptance_id

    matcher_invocation_count = evaluate_region_schedule(
        regions,
        policy.maximum_matcher_invocations_per_current,
        candidate_is_live=candidate_is_live,
        evaluate_live_candidate=evaluate,
    )
    for candidate in eligible:
        row = records_by_previous[candidate.previous_ordinal]
        if row["disposition"] != "eligible_not_scheduled":
            continue
        anchor = suppressing_anchor(candidate, anchors, policy)
        if anchor is not None:
            row["disposition"] = "accepted_cluster_suppressed"
            row["acceptance_id"] = anchor.acceptance_id
        else:
            row["disposition"] = "matcher_cap_not_evaluated"
    rows = [records_by_previous[index] for index in range(ordinal)]
    _seal_current(
        currents_root=currents_root,
        ordinal=ordinal,
        frame_id=current_frame_id,
        identity=identity,
        candidate_rows=rows,
        threshold_candidates=threshold_candidates,
        matches=matches,
        matcher_failures=matcher_failures,
        matcher_invocation_count=matcher_invocation_count,
        accepted=accepted_current,
        state=state,
    )
    return detection


def _aggregate_currents(
    root: Path, keyframe_ids: Sequence[int], accepted: Mapping[str, np.ndarray]
) -> None:
    rows = []
    accepted_rows = []
    for ordinal in range(1, len(keyframe_ids)):
        directory = root / "currents" / _current_name(ordinal, int(keyframe_ids[ordinal]))
        with (directory / "candidates.csv").open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    write_csv_atomic(root / "aggregate" / "candidate_dispositions.csv", rows, CANDIDATE_FIELDS)
    for acceptance_id, previous, current, previous_ordinal, current_ordinal in zip(
        accepted["acceptance_ids"], accepted["previous_frame_ids"],
        accepted["current_frame_ids"], accepted["previous_ordinals"],
        accepted["current_ordinals"],
    ):
        accepted_rows.append(
            {
                "acceptance_id": int(acceptance_id),
                "previous_frame_id": int(previous),
                "current_frame_id": int(current),
                "previous_keyframe_ordinal": int(previous_ordinal),
                "current_keyframe_ordinal": int(current_ordinal),
            }
        )
    write_csv_atomic(
        root / "aggregate" / "accepted_loops.csv",
        accepted_rows,
        (
            "acceptance_id", "previous_frame_id", "current_frame_id",
            "previous_keyframe_ordinal", "current_keyframe_ordinal",
        ),
    )


def run_loop_closure_stage(
    dataset: KittiEstimationContext,
    config: AlgorithmConfig,
    pose_graph: PoseGraphOutput,
    root: Path,
    *,
    upstream_stage_ids: Mapping[str, str],
    maximum_new_currents: Optional[int] = None,
) -> LoopClosureOutput:
    root = Path(root)
    identity = _identity(dataset, config, upstream_stage_ids)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_stage_manifest(manifest_path)
        validate_stage_manifest(
            manifest, root, expected_stage="loop_closure",
            sequence_id=dataset.sequence_id, dataset_id=dataset.dataset_id,
            config_fingerprint=config.stage_fingerprint("loop_closure"),
            upstream_stage_ids=upstream_stage_ids,
        )
        validate_loop_closure_artifacts(root, pose_graph.keyframe_ids)
        return load_loop_closure_output(root)
    root.mkdir(parents=True, exist_ok=True)
    completed, anchors, accepted, state, detection = _replay_prefix(
        root, pose_graph, config, identity
    )
    policy = LoopClosurePolicy.from_config(config.section("loop_closure"))
    frontend = FeatureFrontend(config.section("frontend"))
    feature_cache = (
        StereoFeatureCache(dataset, frontend) if frontend.is_deterministic else None
    )
    started = time.perf_counter()
    new_currents = 0
    next_ordinal = (completed[-1] + 1) if completed else 1
    for ordinal in range(next_ordinal, len(state.keyframe_ids)):
        if maximum_new_currents is not None and new_currents >= maximum_new_currents:
            break
        detection = _process_current(
            dataset=dataset,
            config=config,
            policy=policy,
            ordinal=ordinal,
            state=state,
            detection=detection,
            anchors=anchors,
            currents_root=root / "currents",
            identity=identity,
            feature_cache=feature_cache,
        )
        completed.append(ordinal)
        new_currents += 1
        write_json_atomic(
            root / "state.json",
            {
                "schema": "generic-slam-loop-state-v1",
                **identity,
                "status": "running",
                "completed_current_ordinals": completed,
                "next_current_ordinal": ordinal + 1,
                "accepted_loop_count": state.state_version,
            },
        )

    # Reload the factor population from the now-durable contiguous prefix.
    completed, anchors, accepted, state, detection = _replay_prefix(
        root, pose_graph, config, identity
    )
    if len(completed) != len(state.keyframe_ids) - 1:
        return LoopClosureOutput(
            None,
            np.asarray(state.keyframe_ids, dtype=np.int64),
            state.pose_matrices(),
            state.marginal_covariances(),
            accepted["previous_frame_ids"],
            accepted["current_frame_ids"],
            accepted["relative_pose_matrices"],
            accepted["conditional_covariances"],
        )

    save_npz_atomic(
        root / "accepted_factors.npz", **accepted
    )
    save_npz_atomic(
        root / "final_values.npz",
        frame_ids=np.asarray(state.keyframe_ids, dtype=np.int64),
        pose_matrices=state.pose_matrices(),
        positions=state.pose_matrices()[:, :3, 3],
    )
    save_npz_atomic(
        root / "final_marginals.npz",
        frame_ids=np.asarray(state.keyframe_ids, dtype=np.int64),
        marginal_covariances=state.marginal_covariances(),
    )
    _aggregate_currents(root, state.keyframe_ids, accepted)
    ordered_pair_count = len(state.keyframe_ids) * (len(state.keyframe_ids) - 1) // 2
    write_json_atomic(
        root / "state.json",
        {
            "schema": "generic-slam-loop-state-v1",
            **identity,
            "status": "complete",
            "completed_current_ordinals": completed,
            "next_current_ordinal": len(state.keyframe_ids),
            "accepted_loop_count": state.state_version,
            "ordered_pair_count": ordered_pair_count,
        },
    )
    _validate_loop_payload(root, state.keyframe_ids, identity)
    manifest = StageManifest.complete(
        stage="loop_closure", sequence_id=dataset.sequence_id,
        dataset_id=dataset.dataset_id, config_fingerprint=config.stage_fingerprint("loop_closure"),
        upstream_stage_ids=upstream_stage_ids,
        counts={
            "keyframe_count": len(state.keyframe_ids),
            "ordered_pair_count": ordered_pair_count,
            "accepted_loop_count": state.state_version,
            "final_factor_count": int(state.graph.size()),
        },
        primary_artifacts=(
            "state.json", "accepted_factors.npz", "final_values.npz",
            "final_marginals.npz", "aggregate/candidate_dispositions.csv",
            "aggregate/accepted_loops.csv",
        ),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
    )
    write_stage_manifest(root, manifest)
    return LoopClosureOutput(
        manifest,
        np.asarray(state.keyframe_ids, dtype=np.int64),
        state.pose_matrices(),
        state.marginal_covariances(),
        accepted["previous_frame_ids"],
        accepted["current_frame_ids"],
        accepted["relative_pose_matrices"],
        accepted["conditional_covariances"],
    )


def _validate_loop_payload(
    root: Path, keyframe_ids: Sequence[int], identity: Mapping[str, object]
) -> None:
    """Validate completion before the manifest is published."""

    root = Path(root)
    require_files(
        root,
        (
            "state.json", "accepted_factors.npz",
            "final_values.npz", "final_marginals.npz",
            "aggregate/candidate_dispositions.csv", "aggregate/accepted_loops.csv",
        ),
    )
    selected = np.asarray(keyframe_ids, dtype=np.int64)
    state = read_json(root / "state.json")
    expected_pairs = len(selected) * (len(selected) - 1) // 2
    if (
        state.get("status") != "complete"
        or int(state.get("next_current_ordinal", -1)) != len(selected)
        or int(state.get("ordered_pair_count", -1)) != expected_pairs
        or state.get("completed_current_ordinals") != list(range(1, len(selected)))
    ):
        raise ContractError("Loop Closure completion state differs")
    accepted_total = 0
    for ordinal in range(1, len(selected)):
        seal = _validate_current(
            root / "currents" / _current_name(ordinal, int(selected[ordinal])),
            ordinal, int(selected[ordinal]), identity, len(selected),
        )
        accepted_total += int(seal["accepted_loop_count_this_current"])
    with np.load(root / "final_values.npz", allow_pickle=False) as values:
        if (
            not np.array_equal(values["frame_ids"], selected)
            or values["pose_matrices"].shape != (len(selected), 4, 4)
            or not np.all(np.isfinite(values["pose_matrices"]))
        ):
            raise ContractError("Loop Closure final Values differ")
        for pose in values["pose_matrices"]:
            validate_rigid_pose(pose, atol=1e-7)
    with np.load(root / "final_marginals.npz", allow_pickle=False) as marginals:
        if (
            not np.array_equal(marginals["frame_ids"], selected)
            or marginals["marginal_covariances"].shape != (len(selected), 6, 6)
            or not np.all(np.isfinite(marginals["marginal_covariances"]))
            or not np.allclose(
                marginals["marginal_covariances"],
                np.swapaxes(marginals["marginal_covariances"], 1, 2),
                rtol=0.0, atol=1e-7,
            )
        ):
            raise ContractError("Loop Closure final Marginals differ")
    accepted = _load_factors(root / "accepted_factors.npz")
    if (
        len(accepted["acceptance_ids"]) != accepted_total
        or not np.array_equal(
            accepted["acceptance_ids"], np.arange(1, accepted_total + 1, dtype=np.int64)
        )
        or accepted["relative_pose_matrices"].shape != (accepted_total, 4, 4)
        or accepted["conditional_covariances"].shape != (accepted_total, 6, 6)
        or not np.all(np.isin(accepted["previous_frame_ids"], selected))
        or not np.all(np.isin(accepted["current_frame_ids"], selected))
        or not np.all(np.isfinite(accepted["conditional_covariances"]))
        or not np.allclose(
            accepted["conditional_covariances"],
            np.swapaxes(accepted["conditional_covariances"], 1, 2),
            rtol=0.0, atol=1e-7,
        )
        or int(state.get("accepted_loop_count", -1)) != accepted_total
    ):
        raise ContractError("Loop Closure accepted factor population differs")
    for pose in accepted["relative_pose_matrices"]:
        validate_rigid_pose(pose, atol=1e-7)
    with (root / "aggregate" / "candidate_dispositions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        if len(list(csv.DictReader(handle))) != expected_pairs:
            raise ContractError("Loop Closure ordered pair population differs")
    with (root / "aggregate" / "accepted_loops.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        if len(list(csv.DictReader(handle))) != accepted_total:
            raise ContractError("Loop Closure accepted summary population differs")


def validate_loop_closure_artifacts(root: Path, keyframe_ids: Sequence[int]) -> None:
    """Validate the complete contiguous Loop Closure replay boundary."""

    root = Path(root)
    manifest = load_stage_manifest(root / "manifest.json")
    identity = {
        "sequence_id": manifest.sequence_id,
        "dataset_id": manifest.dataset_id,
        "config_fingerprint": manifest.config_fingerprint,
        "upstream_stage_ids": dict(manifest.upstream_stage_ids),
    }
    _validate_loop_payload(root, keyframe_ids, identity)
    expected_pairs = len(keyframe_ids) * (len(keyframe_ids) - 1) // 2
    if (
        int(manifest.counts.get("keyframe_count", -1)) != len(keyframe_ids)
        or int(manifest.counts.get("ordered_pair_count", -1)) != expected_pairs
        or int(manifest.counts.get("accepted_loop_count", -1))
        != int(read_json(root / "state.json")["accepted_loop_count"])
    ):
        raise ContractError("Loop Closure manifest counts differ")


def load_loop_closure_output(root: Path) -> LoopClosureOutput:
    root = Path(root)
    manifest = load_stage_manifest(root / "manifest.json")
    with np.load(root / "final_values.npz", allow_pickle=False) as values:
        frame_ids = values["frame_ids"].copy()
        poses = values["pose_matrices"].copy()
    with np.load(root / "final_marginals.npz", allow_pickle=False) as marginals:
        covariance = marginals["marginal_covariances"].copy()
    accepted = _load_factors(root / "accepted_factors.npz")
    return LoopClosureOutput(
        manifest, frame_ids, poses, covariance,
        accepted["previous_frame_ids"], accepted["current_frame_ids"],
        accepted["relative_pose_matrices"], accepted["conditional_covariances"],
    )


__all__ = [
    "LoopClosureOutput", "load_loop_closure_output", "run_loop_closure_stage",
    "validate_loop_closure_artifacts",
]
