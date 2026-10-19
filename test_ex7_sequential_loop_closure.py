"""Lightweight scheduling and state-selection contracts for E7-I04."""

from __future__ import annotations

from pathlib import Path
import csv
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import gtsam
import numpy as np

from exercise7_candidate_detection import (
    CandidateScore,
    CovarianceEdge,
    CovarianceGraph,
    CovariancePath,
)
from exercise7_consensus_matching import ConsensusMatchingError
from exercise7_sequential_loop_closure import (
    AcceptedLoopAnchor,
    ACCEPTED_LOOP_CSV_FIELDS,
    MATCHER_CSV_FIELDS,
    DEFAULT_E5_ROOT,
    DEFAULT_E6_ROOT,
    DEFAULT_OUTPUT_ROOT,
    HISTORICAL_E5_ROOT,
    HISTORICAL_E6_ROOT,
    HISTORICAL_E7_ROOT,
    ProductionPolicy,
    RankedCandidate,
    SequentialLoopClosureError,
    TemporaryProposalRegion,
    _snapshot_slot_ids,
    _write_mapping_csv,
    _process_current_keyframe,
    _base_candidate_row,
    _load_baseline_runtime,
    _production_identity,
    _replay_sealed_currents,
    _validate_candidate_path_evidence,
    _write_candidate_path_evidence,
    accepted_anchor_for_candidate,
    evaluate_region_schedule,
    form_temporary_proposal_regions,
    ordered_pair_index,
    preflight_exercise7,
    run_sequential_loop_closure,
    same_anchored_rectangle,
    validate_current_seal,
)


def _candidate(previous_ordinal, current_ordinal, rank, score):
    return RankedCandidate(
        pair_index=ordered_pair_index(previous_ordinal, current_ordinal),
        rank_within_current=rank,
        score=SimpleNamespace(
            previous_frame_id=previous_ordinal,
            current_frame_id=1000 + current_ordinal,
            previous_keyframe_ordinal=previous_ordinal,
            current_keyframe_ordinal=current_ordinal,
            mahalanobis_squared=float(score),
            mahalanobis=float(score) ** 0.5,
        ),
    )


class CleanRootAndPreflightContractTest(unittest.TestCase):
    def test_clean_defaults_and_write_free_new_root_preflight(self):
        repository_root = Path(__file__).resolve().parent
        self.assertEqual(
            DEFAULT_E5_ROOT.resolve(),
            (repository_root / "diagnostics/ex5_clean_final").resolve(),
        )
        self.assertEqual(
            DEFAULT_E6_ROOT.resolve(),
            (repository_root / "diagnostics/ex6_clean_final").resolve(),
        )
        self.assertEqual(
            DEFAULT_OUTPUT_ROOT.resolve(),
            (repository_root / "diagnostics/ex7_clean_final").resolve(),
        )
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            output_root = parent / "future-clean-e7"
            before = tuple(parent.iterdir())
            result = preflight_exercise7(output_root=output_root)
            after = tuple(parent.iterdir())
            self.assertEqual(result["status"], "pass")
            self.assertEqual(
                result["output_root"]["status"], "new-safe-root"
            )
            self.assertEqual(result["contracts"]["keyframe_count"], 356)
            self.assertEqual(
                result["contracts"]["consecutive_covariance_edge_count"],
                355,
            )
            self.assertFalse(result["artifact_writes"])
            self.assertEqual(before, after)
            self.assertFalse(output_root.exists())

    def test_preflight_accepts_only_exact_existing_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "existing-clean-e7"
            (output_root / "currents").mkdir(parents=True)
            (output_root / "policy.json").write_text(
                json.dumps(ProductionPolicy().record(), sort_keys=True),
                encoding="utf-8",
            )
            (output_root / "provenance.json").write_text(
                json.dumps(_production_identity(), sort_keys=True),
                encoding="utf-8",
            )
            before = {
                path.relative_to(output_root).as_posix(): path.read_bytes()
                for path in output_root.rglob("*")
                if path.is_file()
            }
            result = preflight_exercise7(output_root=output_root)
            after = {
                path.relative_to(output_root).as_posix(): path.read_bytes()
                for path in output_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                result["output_root"]["status"],
                "existing-exact-resumable-root",
            )
            self.assertEqual(before, after)

            (output_root / "provenance.json").write_text(
                "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                SequentialLoopClosureError,
                "protected inputs or production sources changed",
            ):
                preflight_exercise7(output_root=output_root)

    def test_historical_inputs_are_explicit_and_output_is_protected(self):
        if not HISTORICAL_E5_ROOT.exists() or not HISTORICAL_E6_ROOT.exists():
            self.skipTest(
                "optional historical E7 input fixtures are not present: "
                "{} / {}".format(
                    HISTORICAL_E5_ROOT, HISTORICAL_E6_ROOT
                )
            )
        (
            baseline_inputs,
            candidate_graph,
            state,
            _,
            configuration_sha256,
            _,
        ) = _load_baseline_runtime(
            e5_root=HISTORICAL_E5_ROOT,
            e6_root=HISTORICAL_E6_ROOT,
        )
        self.assertEqual(baseline_inputs.e5_root, HISTORICAL_E5_ROOT.resolve())
        self.assertEqual(baseline_inputs.e6_root, HISTORICAL_E6_ROOT.resolve())
        self.assertEqual(len(candidate_graph.edges), 355)
        self.assertEqual(state.graph.size(), 356)
        self.assertEqual(
            configuration_sha256,
            "15c6430e397d7fe94ffe77d8f27fd8daa707449c6eb4c9227e345ca070b6e1e7",
        )
        with self.assertRaisesRegex(
            SequentialLoopClosureError, "overlaps protected evidence"
        ):
            preflight_exercise7(
                e5_root=HISTORICAL_E5_ROOT,
                e6_root=HISTORICAL_E6_ROOT,
                output_root=HISTORICAL_E7_ROOT,
            )

    def test_representative_historical_seals_replay_without_science(self):
        if not HISTORICAL_E7_ROOT.exists():
            self.skipTest(
                "optional historical E7 sealed-run fixture is not present: "
                + str(HISTORICAL_E7_ROOT)
            )
        for ordinal in (0, 160, 355):
            result = validate_current_seal(
                HISTORICAL_E7_ROOT
                / "currents"
                / "current_{:04d}".format(ordinal)
            )
            self.assertEqual(result["seal"]["status"], "sealed")


class AnchoredSuppressionContractTest(unittest.TestCase):
    def test_anchored_rectangles_do_not_expand_transitively(self):
        self.assertTrue(same_anchored_rectangle(0, 30, 10, 40))
        self.assertTrue(same_anchored_rectangle(10, 40, 20, 50))
        self.assertFalse(same_anchored_rectangle(0, 30, 20, 50))

        anchor = AcceptedLoopAnchor(
            acceptance_id=1,
            previous_frame_id=0,
            current_frame_id=1030,
            previous_ordinal=0,
            current_ordinal=30,
            state_version=1,
        )
        policy = ProductionPolicy()
        covered = _candidate(10, 40, 1, 1.0)
        chained_only = _candidate(20, 50, 2, 2.0)

        self.assertIs(
            accepted_anchor_for_candidate(covered, [anchor], policy),
            anchor,
        )
        self.assertIsNone(
            accepted_anchor_for_candidate(chained_only, [anchor], policy)
        )

    def test_temporary_regions_use_ranked_nontransitive_anchors(self):
        policy = ProductionPolicy()
        candidates = (
            _candidate(0, 30, 1, 1.0),
            _candidate(10, 30, 2, 2.0),
            _candidate(20, 30, 3, 3.0),
        )

        regions = form_temporary_proposal_regions(candidates, policy)

        self.assertEqual(len(regions), 2)
        self.assertEqual(
            [value.previous_ordinal for value in regions[0].candidates],
            [0, 10],
        )
        self.assertEqual(
            [value.previous_ordinal for value in regions[1].candidates],
            [20],
        )


class FairSchedulerContractTest(unittest.TestCase):
    def test_distinct_regions_receive_first_attempts_before_fallback(self):
        first = TemporaryProposalRegion(
            region_id=0,
            anchor_previous_ordinal=0,
            anchor_current_ordinal=30,
            candidates=[
                _candidate(0, 30, 1, 1.0),
                _candidate(1, 30, 3, 3.0),
            ],
        )
        second = TemporaryProposalRegion(
            region_id=1,
            anchor_previous_ordinal=15,
            anchor_current_ordinal=30,
            candidates=[_candidate(15, 30, 2, 2.0)],
        )
        observed = []

        count = evaluate_region_schedule(
            (first, second),
            maximum_invocations=3,
            candidate_is_live=lambda candidate, region, scheduler_pass: True,
            evaluate_live_candidate=lambda candidate, region, scheduler_pass, slot: (
                observed.append(
                    (
                        candidate.previous_ordinal,
                        region.region_id,
                        scheduler_pass,
                        slot,
                    )
                )
            ),
        )

        self.assertEqual(count, 3)
        self.assertEqual(
            observed,
            [(0, 0, 1, 1), (15, 1, 1, 2), (1, 0, 2, 3)],
        )

    def test_suppressed_candidate_consumes_no_matcher_slot(self):
        region = TemporaryProposalRegion(
            region_id=0,
            anchor_previous_ordinal=0,
            anchor_current_ordinal=30,
            candidates=[
                _candidate(0, 30, 1, 1.0),
                _candidate(1, 30, 2, 2.0),
            ],
        )
        observed = []

        count = evaluate_region_schedule(
            (region,),
            maximum_invocations=1,
            candidate_is_live=lambda candidate, unused_region, unused_pass: (
                candidate.previous_ordinal != 0
            ),
            evaluate_live_candidate=lambda candidate, unused_region, scheduler_pass, slot: (
                observed.append(
                    (candidate.previous_ordinal, scheduler_pass, slot)
                )
            ),
        )

        self.assertEqual(count, 1)
        self.assertEqual(observed, [(1, 1, 1)])


class ArtifactSelectionContractTest(unittest.TestCase):
    def test_pair_indices_cover_the_canonical_triangular_population(self):
        observed = [
            ordered_pair_index(previous, current)
            for current in range(1, 356)
            for previous in range(current)
        ]
        self.assertEqual(observed, list(range(63190)))

    def test_result_aware_snapshot_fallbacks_are_deterministic(self):
        self.assertEqual(_snapshot_slot_ids(0), (0, 0, 0, 0))
        self.assertEqual(_snapshot_slot_ids(1), (0, 1, 1, 1))
        self.assertEqual(_snapshot_slot_ids(2), (0, 1, 1, 2))
        self.assertEqual(_snapshot_slot_ids(3), (0, 1, 2, 3))
        self.assertEqual(_snapshot_slot_ids(4), (0, 1, 2, 4))

    def test_zero_loop_csv_contracts_retain_explicit_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, fields in (
                ("matcher.csv", MATCHER_CSV_FIELDS),
                ("accepted.csv", ACCEPTED_LOOP_CSV_FIELDS),
            ):
                path = root / filename
                _write_mapping_csv(path, (), fieldnames=fields)
                with path.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, list(fields))
                    self.assertEqual(list(reader), [])

    def test_path_evidence_must_match_deterministic_course_path(self):
        identity = np.eye(6, dtype=float)
        graph = CovarianceGraph(
            (0, 1, 2, 3),
            (
                CovarianceEdge(0, 0, 1, identity, Path(".")),
                CovarianceEdge(1, 1, 3, identity, Path(".")),
                CovarianceEdge(2, 0, 2, 2.0 * identity, Path(".")),
                CovarianceEdge(3, 2, 3, 2.0 * identity, Path(".")),
            ),
        )
        wrong_covariance = 4.0 * identity
        wrong_path = CovariancePath(
            source_frame_id=0,
            target_frame_id=3,
            frame_ids=(0, 2, 3),
            edge_indices=(2, 3),
            covariance=wrong_covariance,
            covariance_determinant=float(np.linalg.det(wrong_covariance)),
            covariance_log_determinant=float(
                np.linalg.slogdet(wrong_covariance)[1]
            ),
            policy="course-accumulated-covariance-determinant",
        )
        wrong_candidate = RankedCandidate(
            pair_index=ordered_pair_index(0, 3),
            rank_within_current=1,
            score=CandidateScore(
                previous_frame_id=0,
                current_frame_id=3,
                previous_keyframe_ordinal=0,
                current_keyframe_ordinal=3,
                path=wrong_path,
                factor_error=0.5,
                mahalanobis_squared=1.0,
                mahalanobis=1.0,
            ),
        )
        current = {
            "candidate_graph_edge_count_before": 4,
            "candidates": [
                {
                    "pair_index": ordered_pair_index(0, 3),
                    "rank_within_current": 1,
                    "previous_frame_id": 0,
                    "current_frame_id": 3,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paths.npz"
            _write_candidate_path_evidence(path, (wrong_candidate,))
            with self.assertRaisesRegex(
                SequentialLoopClosureError,
                "deterministic course path",
            ):
                _validate_candidate_path_evidence(
                    path, current, candidate_graph=graph
                )
            expected_path = graph.find_course_covariance_path(0, 3)
            correct_candidate = RankedCandidate(
                pair_index=wrong_candidate.pair_index,
                rank_within_current=1,
                score=CandidateScore(
                    previous_frame_id=0,
                    current_frame_id=3,
                    previous_keyframe_ordinal=0,
                    current_keyframe_ordinal=3,
                    path=expected_path,
                    factor_error=0.5,
                    mahalanobis_squared=1.0,
                    mahalanobis=1.0,
                ),
            )
            _write_candidate_path_evidence(path, (correct_candidate,))
            _validate_candidate_path_evidence(
                path, current, candidate_graph=graph
            )


class BoundedSequentialOrchestrationSmokeTest(unittest.TestCase):
    @staticmethod
    def _score(previous_ordinal, mahalanobis_squared):
        covariance = np.eye(6, dtype=float) * (
            1.0 + float(previous_ordinal)
        )
        sign, log_determinant = np.linalg.slogdet(covariance)
        assert sign > 0.0
        return CandidateScore(
            previous_frame_id=previous_ordinal,
            current_frame_id=30,
            previous_keyframe_ordinal=previous_ordinal,
            current_keyframe_ordinal=30,
            path=CovariancePath(
                source_frame_id=previous_ordinal,
                target_frame_id=30,
                frame_ids=(previous_ordinal, 30),
                edge_indices=(previous_ordinal,),
                covariance=covariance,
                covariance_determinant=float(np.linalg.det(covariance)),
                covariance_log_determinant=float(log_determinant),
                policy="bounded-smoke",
            ),
            factor_error=float(mahalanobis_squared) / 2.0,
            mahalanobis_squared=float(mahalanobis_squared),
            mahalanobis=float(mahalanobis_squared) ** 0.5,
        )

    def test_reject_failure_and_same_region_fallback_seal_deterministically(
        self,
    ):
        keyframe_ids = tuple(range(31))
        baseline_inputs = SimpleNamespace(keyframe_ids=keyframe_ids)
        scores = [
            self._score(0, 1.0),
            self._score(14, 2.0),
            self._score(1, 3.0),
        ]
        scores.extend(
            self._score(previous, 36_000_000.0 + previous)
            for previous in range(2, 14)
        )
        fake_graph = SimpleNamespace(edges=tuple(range(355)))
        fake_state = SimpleNamespace(
            state_version=0,
            graph=SimpleNamespace(size=lambda: 356),
            optimized_values=object(),
        )
        rejected_run = SimpleNamespace(
            evidence=SimpleNamespace(pair=SimpleNamespace())
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            (output_root / "currents").mkdir()
            with mock.patch(
                "exercise7_sequential_loop_closure."
                "candidate_detection_inputs_from_values",
                return_value=baseline_inputs,
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "score_previous_keyframes",
                return_value=tuple(scores),
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "evaluate_candidate_pair",
                side_effect=(
                    ConsensusMatchingError("bounded matcher failure"),
                    rejected_run,
                    rejected_run,
                ),
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "validate_pair_evidence"
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "_pair_evidence_record",
                return_value={"terminal_status": "adaptive_completed"},
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "production_match_policy_decision",
                return_value=(False, "bounded_policy_rejection"),
            ):
                observed_graph = _process_current_keyframe(
                    output_root=output_root,
                    current_ordinal=30,
                    baseline_inputs=baseline_inputs,
                    candidate_graph=fake_graph,
                    state=fake_state,
                    configuration={},
                    calibration=object(),
                    anchors=[],
                    evaluated_pairs=set(),
                    policy=ProductionPolicy(),
                )

            self.assertIs(observed_graph, fake_graph)
            sealed = validate_current_seal(
                output_root / "currents" / "current_0030"
            )
            current = sealed["current"]
            self.assertEqual(current["matcher_invocation_count"], 3)
            invocations = current["matcher_invocations"]
            self.assertEqual(
                [
                    (
                        record["previous_keyframe_ordinal"],
                        record["scheduler_pass"],
                        record["matcher_slot"],
                        record["outcome"],
                    )
                    for record in invocations
                ],
                [
                    (0, 1, 1, "matcher_failed"),
                    (14, 1, 2, "matcher_rejected"),
                    (1, 2, 3, "matcher_rejected"),
                ],
            )
            self.assertEqual(current["state_version_after"], 0)
            self.assertEqual(current["candidate_graph_edge_count_after"], 355)

    def test_pre_replay_failure_is_recorded_against_durable_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            (output_root / "currents").mkdir()
            (output_root / "currents" / "current_0000").mkdir()
            with mock.patch(
                "exercise7_sequential_loop_closure._prepare_output_root"
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "_load_baseline_runtime",
                side_effect=RuntimeError("bounded replay bootstrap failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "bounded replay bootstrap failure"
                ):
                    run_sequential_loop_closure(output_root=output_root)
            with (output_root / "run_state.json").open(
                "r", encoding="utf-8"
            ) as handle:
                run_state = json.load(handle)
            self.assertEqual(run_state["status"], "fatal")
            self.assertEqual(run_state["completed_current_count"], 0)
            self.assertEqual(run_state["state_version"], 0)
            self.assertEqual(
                run_state["fatal_failure_message"],
                "bounded replay bootstrap failure",
            )

    def test_positive_acceptance_seals_and_replays_exact_checkpoint(self):
        (
            baseline_inputs,
            initial_candidate_graph,
            protected_state,
            _,
            _,
            _,
        ) = _load_baseline_runtime()
        current_ordinal = 30
        current_frame_id = baseline_inputs.keyframe_ids[current_ordinal]
        eligible_ordinals = tuple(range(15))
        scores = []
        for previous_ordinal in eligible_ordinals:
            previous_frame_id = baseline_inputs.keyframe_ids[
                previous_ordinal
            ]
            path = initial_candidate_graph.find_course_covariance_path(
                previous_frame_id, current_frame_id
            )
            mahalanobis_squared = (
                1.0
                if previous_ordinal == 0
                else 36_000_000.0 + previous_ordinal
            )
            scores.append(
                CandidateScore(
                    previous_frame_id=previous_frame_id,
                    current_frame_id=current_frame_id,
                    previous_keyframe_ordinal=previous_ordinal,
                    current_keyframe_ordinal=current_ordinal,
                    path=path,
                    factor_error=mahalanobis_squared / 2.0,
                    mahalanobis_squared=mahalanobis_squared,
                    mahalanobis=mahalanobis_squared ** 0.5,
                )
            )
        observation_count = 30
        keypoints = tuple(
            cv2.KeyPoint(float(index + 2), float(index + 3), 1.0)
            for index in range(observation_count)
        )
        matches = tuple(
            cv2.DMatch(index, index, 0.0)
            for index in range(observation_count)
        )
        consensus = tuple(
            (index, index, index)
            for index in range(observation_count)
        )
        supporters = tuple(range(observation_count))
        observations = tuple(
            SimpleNamespace(
                consensus_index=index,
                previous_measurement=(
                    float(index + 3),
                    float(index + 2),
                    float(index + 4),
                ),
                current_measurement=(
                    float(index + 3),
                    float(index + 2),
                    float(index + 4),
                ),
            )
            for index in range(observation_count)
        )
        covariance = np.eye(6, dtype=float)
        pose_matrices = protected_state.optimized_pose_matrices.copy()
        marginals = protected_state.marginal_covariances.copy()

        class SizeGraph:
            def __init__(self, size):
                self._size = int(size)

            def size(self):
                return self._size

        class FakeState:
            def __init__(self):
                self.inputs = protected_state.inputs
                self.graph = SizeGraph(356)
                self.optimized_values = gtsam.Values(
                    protected_state.optimized_values
                )
                self.optimized_pose_matrices = pose_matrices.copy()
                self.marginal_covariances = marginals.copy()
                self.state_version = 0
                self.base_factor_count = 356

            def initial_relative_pose(
                self, previous_frame_id, observed_current_frame_id
            ):
                del previous_frame_id, observed_current_frame_id
                return gtsam.Pose3()

            def apply_loop_factor(self, **unused):
                del unused
                update = SimpleNamespace(
                    graph=SizeGraph(357),
                    optimized_pose_matrices=pose_matrices.copy(),
                    marginal_covariances=marginals.copy(),
                    state_version=1,
                    pre_update_state_version=0,
                    original_graph_error=10.0,
                    augmented_pre_optimization_error=12.0,
                    augmented_post_optimization_error=8.0,
                    loop_factor_pre_optimization_error=2.0,
                    loop_factor_post_optimization_error=1.0,
                    marginal_symmetry_max_abs=0.0,
                    marginal_minimum_eigenvalue=1.0,
                )
                self.graph = update.graph
                self.optimized_pose_matrices = (
                    update.optimized_pose_matrices
                )
                self.marginal_covariances = (
                    update.marginal_covariances
                )
                self.state_version = 1
                return update

        def accepted_pair_run(pair, **unused):
            del unused
            counts = SimpleNamespace(
                previous_left_keypoint_count=observation_count,
                previous_right_keypoint_count=observation_count,
                current_left_keypoint_count=observation_count,
                current_right_keypoint_count=observation_count,
                previous_raw_stereo_match_count=observation_count,
                current_raw_stereo_match_count=observation_count,
                previous_stereo_inlier_count=observation_count,
                current_stereo_inlier_count=observation_count,
                reciprocal_temporal_match_count=observation_count,
                four_view_consensus_count=observation_count,
                final_supporter_count=observation_count,
                final_supporter_percentage=100.0,
            )
            ransac = SimpleNamespace(
                status="adaptive_completed",
                terminal_reason="bounded_positive_smoke",
                total_hypothesis_attempt_count=1,
                random_hypothesis_attempt_count=1,
                refinement_hypothesis_attempt_count=0,
                required_hypothesis_count=1,
                capped_required_hypothesis_count=1,
                maximum_hypothesis_attempt_count=3000,
                successful_pnp_call_count=1,
                failed_pnp_call_count=0,
                invalid_geometry_model_count=0,
                valid_model_count=1,
                best_supporter_indices=supporters,
                best_rotations=tuple(
                    np.eye(3, dtype=float) for _ in range(4)
                ),
                best_translations=tuple(
                    np.zeros((3, 1), dtype=float) for _ in range(4)
                ),
                refinement_outcomes=(),
            )
            runtime = SimpleNamespace(
                image_loading_seconds=0.0,
                stereo_extraction_seconds=0.0,
                temporal_matching_seconds=0.0,
                triangulation_seconds=0.0,
                ransac_seconds=0.0,
                total_seconds=0.0,
            )
            evidence = SimpleNamespace(
                pair=pair, counts=counts, ransac=ransac, runtime=runtime
            )
            visualization = SimpleNamespace(
                previous_image_left=np.zeros((40, 40), dtype=np.uint8),
                current_image_left=np.zeros((40, 40), dtype=np.uint8),
                previous_keypoints_left=keypoints,
                previous_keypoints_right=keypoints,
                current_keypoints_left=keypoints,
                current_keypoints_right=keypoints,
                previous_stereo_inliers=matches,
                current_stereo_inliers=matches,
                reciprocal_matches=matches,
                consensus=consensus,
                supporter_indices=supporters,
            )
            return SimpleNamespace(
                evidence=evidence, visualization=visualization
            )

        bundle = SimpleNamespace(
            candidate_observations=observations,
            accepted_observations=observations,
            rejection_reason_by_consensus_index={},
            initial_relative_pose=gtsam.Pose3(),
            optimized_relative_pose=gtsam.Pose3(),
            conditional_covariance=covariance,
            initial_error=2.0,
            final_error=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            (output_root / "currents").mkdir()
            processing_state = FakeState()
            with mock.patch(
                "exercise7_sequential_loop_closure."
                "candidate_detection_inputs_from_values",
                return_value=baseline_inputs,
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "score_previous_keyframes",
                return_value=tuple(scores),
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "evaluate_candidate_pair",
                side_effect=accepted_pair_run,
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "validate_pair_evidence"
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "production_match_policy_decision",
                return_value=(True, "bounded_positive_smoke"),
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "observations_from_pair_run",
                return_value=observations,
            ), mock.patch(
                "exercise7_sequential_loop_closure.solve_small_bundle",
                return_value=bundle,
            ):
                updated_candidate_graph = _process_current_keyframe(
                    output_root=output_root,
                    current_ordinal=current_ordinal,
                    baseline_inputs=baseline_inputs,
                    candidate_graph=initial_candidate_graph,
                    state=processing_state,
                    configuration={},
                    calibration=object(),
                    anchors=[],
                    evaluated_pairs=set(),
                    policy=ProductionPolicy(),
                )
            self.assertEqual(len(updated_candidate_graph.edges), 356)
            current_directory = (
                output_root / "currents" / "current_0030"
            )
            sealed = validate_current_seal(current_directory)
            self.assertEqual(
                sealed["current"]["accepted_loop_count"], 1
            )
            self.assertTrue(
                (current_directory / "accepted_0001.npz").is_file()
            )
            self.assertTrue(
                (current_directory / "successful_match.png").is_file()
            )

            prefix_records = {}
            for ordinal in range(current_ordinal):
                (output_root / "currents" / "current_{:04d}".format(
                    ordinal
                )).mkdir()
                rows = []
                for previous_ordinal in range(ordinal):
                    row = _base_candidate_row(
                        keyframe_ids=baseline_inputs.keyframe_ids,
                        previous_ordinal=previous_ordinal,
                        current_ordinal=ordinal,
                        state_version=0,
                    )
                    row["terminal_disposition"] = "temporal_excluded"
                    row["terminal_reason"] = "bounded_prefix"
                    rows.append(row)
                prefix_records[ordinal] = {
                    "current_ordinal": ordinal,
                    "current_frame_id": baseline_inputs.keyframe_ids[
                        ordinal
                    ],
                    "state_version_before": 0,
                    "state_version_after": 0,
                    "candidate_graph_edge_count_before": 355,
                    "candidate_graph_edge_count_after": 355,
                    "candidate_count": ordinal,
                    "matcher_invocation_count": 0,
                    "accepted_loop_count": 0,
                    "candidates": rows,
                    "matcher_invocations": [],
                    "accepted_loops": [],
                }
            original_validate = validate_current_seal

            def replay_validate(path, *, candidate_graph=None):
                ordinal = int(Path(path).name.split("_")[1])
                if ordinal < current_ordinal:
                    return {
                        "seal": {},
                        "current": prefix_records[ordinal],
                    }
                return original_validate(
                    path, candidate_graph=candidate_graph
                )

            replay_state = FakeState()
            with mock.patch(
                "exercise7_sequential_loop_closure."
                "validate_current_seal",
                side_effect=replay_validate,
            ), mock.patch(
                "exercise7_sequential_loop_closure."
                "_validate_sealed_detection_state"
            ):
                (
                    replay_candidate_graph,
                    replayed_state,
                    anchors,
                    evaluated_pairs,
                    next_ordinal,
                ) = _replay_sealed_currents(
                    output_root,
                    baseline_inputs,
                    initial_candidate_graph,
                    replay_state,
                )
            self.assertEqual(next_ordinal, 31)
            self.assertEqual(replayed_state.state_version, 1)
            self.assertEqual(replayed_state.graph.size(), 357)
            self.assertEqual(len(replay_candidate_graph.edges), 356)
            self.assertEqual(len(anchors), 1)
            self.assertEqual(len(evaluated_pairs), 1)
            self.assertTrue(
                np.array_equal(
                    replayed_state.optimized_pose_matrices,
                    pose_matrices,
                )
            )
            self.assertTrue(
                np.array_equal(
                    replayed_state.marginal_covariances, marginals
                )
            )


if __name__ == "__main__":
    unittest.main()
