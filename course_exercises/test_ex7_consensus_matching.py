"""Synthetic and bounded contract tests for E7-I02."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import exercise7_consensus_matching as matching


def _synthetic_ransac_arguments(consensus_count: int):
    point_cloud = np.column_stack(
        (
            np.linspace(-1.0, 1.0, consensus_count),
            np.linspace(-0.5, 0.5, consensus_count),
            np.full(consensus_count, 10.0),
        )
    )
    consensus = tuple(
        (index, index, index) for index in range(consensus_count)
    )
    identity = np.eye(3)
    zero = np.zeros((3, 1))
    return (
        point_cloud,
        consensus,
        (),
        (),
        (),
        (),
        (),
        (),
        np.eye(3),
        identity,
        zero,
        identity,
        zero,
    )


def _successful_pnp(sampled_consensus, *unused):
    marker = float(len(sampled_consensus))
    return True, np.eye(3), np.asarray([[marker], [0.0], [0.0]])


def _supporter_indices(count: int):
    def fake_supporters(*unused, **unused_keywords):
        return np.arange(count, dtype=int)

    return fake_supporters


class DeclaredCandidateContractTest(unittest.TestCase):
    def test_arbitrary_candidate_constructs_from_e7_i01_score(self) -> None:
        frame_ids = (0, 7, 19, 42)
        score = SimpleNamespace(
            previous_frame_id=7,
            current_frame_id=42,
            previous_keyframe_ordinal=1,
            current_keyframe_ordinal=3,
            mahalanobis_squared=81.0,
            mahalanobis=9.0,
        )

        pair = matching.candidate_pair_from_score(
            score,
            pair_index=17,
            rank_within_current=2,
            canonical_frame_ids=frame_ids,
        )

        self.assertEqual(pair.previous_frame_id, 7)
        self.assertEqual(pair.current_frame_id, 42)
        self.assertEqual(pair.ordinal_gap, 2)
        self.assertEqual(pair.frame_id_gap, 35)
        self.assertEqual(pair.mahalanobis, math.sqrt(81.0))
        self.assertIs(
            matching.validate_candidate_pair(
                pair, canonical_frame_ids=frame_ids
            ),
            pair,
        )

    def test_each_pair_evaluation_resets_all_rngs_to_zero_first(self) -> None:
        score = SimpleNamespace(
            previous_frame_id=7,
            current_frame_id=42,
            previous_keyframe_ordinal=1,
            current_keyframe_ordinal=3,
            mahalanobis_squared=81.0,
            mahalanobis=9.0,
        )
        pair = matching.candidate_pair_from_score(
            score, pair_index=17, rank_within_current=2
        )

        with mock.patch.object(
            matching, "reset_pair_seeds"
        ) as reset_mock, mock.patch.object(
            matching.tools,
            "read_images_from_dataset",
            return_value=(None, None),
        ):
            with self.assertRaises(FileNotFoundError):
                matching.evaluate_candidate_pair(pair)

        reset_mock.assert_called_once_with(matching.CANONICAL_SEED)

    def test_exact_eight_pairs_reload_from_committed_e7_i01(self) -> None:
        if not matching.CANONICAL_E7_I01_SCORES.is_file():
            self.skipTest(
                "optional historical E7-I01 candidate-score fixture is "
                "not present: {}".format(
                    matching.CANONICAL_E7_I01_SCORES
                )
            )
        pairs = matching.validate_declared_candidate_pairs()

        self.assertEqual(len(pairs), 8)
        self.assertEqual(
            tuple(pair.pair_index for pair in pairs), tuple(range(8))
        )
        self.assertEqual(
            tuple(
                (pair.current_frame_id, pair.previous_frame_id)
                for pair in pairs
            ),
            (
                (2331, 270),
                (2331, 356),
                (3359, 372),
                (3359, 356),
                (1146, 1142),
                (3359, 3341),
                (2331, 1055),
                (3359, 1850),
            ),
        )
        self.assertEqual(
            matching.file_sha256(matching.CANONICAL_E7_I01_SCORES),
            matching.CANONICAL_E7_I01_SCORES_SHA256,
        )

    def test_declared_gaps_and_categories_are_mixed(self) -> None:
        pairs = matching.DECLARED_CANDIDATE_PAIRS
        self.assertEqual(
            sum(pair.category == matching.PROMISING_DISTANT for pair in pairs),
            4,
        )
        self.assertEqual(
            sum(pair.category == matching.ADJACENT_CONTROL for pair in pairs),
            2,
        )
        self.assertEqual(
            sum(
                pair.category == matching.POOR_DISTANT_CONTROL
                for pair in pairs
            ),
            2,
        )
        for pair in pairs:
            self.assertEqual(
                pair.current_keyframe_ordinal
                - pair.previous_keyframe_ordinal,
                pair.ordinal_gap,
            )
            self.assertEqual(
                pair.current_frame_id - pair.previous_frame_id,
                pair.frame_id_gap,
            )

    def test_core_never_calls_protected_retrying_ransac_functions(self) -> None:
        tree = ast.parse(inspect.getsource(matching))
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn(
            "estimate_projection_matrices_with_ransac", called_attributes
        )
        self.assertNotIn("build_model", called_attributes)


class BoundedRansacContractTest(unittest.TestCase):
    def _run(
        self,
        consensus_count,
        supporter_function,
        *,
        maximum_attempts=3000,
        pnp_function=_successful_pnp,
        point_cloud_override=None,
    ):
        arguments = list(_synthetic_ransac_arguments(consensus_count))
        if point_cloud_override is not None:
            arguments[0] = point_cloud_override
        with mock.patch.object(
            matching.tools,
            "extract_actual_consensus_pixels",
            return_value=np.zeros((4, 2, consensus_count)),
        ), mock.patch.object(
            matching.tools,
            "calculate_front_camera_matrix",
            side_effect=pnp_function,
        ) as pnp_mock, mock.patch.object(
            matching.tools,
            "find_supporter_indices_for_model",
            side_effect=supporter_function,
        ):
            result = matching.bounded_four_view_ransac(
                *arguments, maximum_attempts=maximum_attempts
            )
        return result, pnp_mock

    def test_adaptive_stopping_uses_monotonic_total_attempts(self) -> None:
        result, pnp_mock = self._run(
            100, _supporter_indices(50), maximum_attempts=3000
        )
        required = matching._required_hypothesis_count(100, 50)

        self.assertEqual(result.status, matching.ADAPTIVE_COMPLETED)
        self.assertEqual(result.required_hypothesis_count, required)
        self.assertEqual(result.random_hypothesis_attempt_count, required)
        self.assertEqual(result.refinement_hypothesis_attempt_count, 1)
        self.assertEqual(
            result.total_hypothesis_attempt_count, required + 1
        )
        self.assertEqual(pnp_mock.call_count, required + 1)
        self.assertEqual(
            result.total_hypothesis_attempt_count,
            result.successful_pnp_call_count
            + result.failed_pnp_call_count,
        )
        self.assertEqual(len(result.best_supporter_indices), 50)
        self.assertFalse(
            result.refinement_outcomes[0].accepted_strict_increase
        )

    def test_budget_exhaustion_retains_best_model_without_refinement(
        self,
    ) -> None:
        result, pnp_mock = self._run(
            100, _supporter_indices(2), maximum_attempts=5
        )

        self.assertEqual(
            result.status,
            matching.BUDGET_EXHAUSTED_WITH_BEST_MODEL,
        )
        self.assertEqual(result.total_hypothesis_attempt_count, 5)
        self.assertEqual(result.random_hypothesis_attempt_count, 5)
        self.assertEqual(result.refinement_hypothesis_attempt_count, 0)
        self.assertEqual(pnp_mock.call_count, 5)
        self.assertEqual(len(result.best_supporter_indices), 2)
        self.assertGreater(result.required_hypothesis_count, 5)

    def test_every_failed_solvepnp_call_consumes_one_attempt(self) -> None:
        def failed_pnp(*unused):
            return False, np.eye(3), np.zeros((3, 1))

        result, pnp_mock = self._run(
            20,
            _supporter_indices(0),
            maximum_attempts=7,
            pnp_function=failed_pnp,
        )

        self.assertEqual(result.status, matching.NO_VALID_PNP_MODEL)
        self.assertEqual(result.total_hypothesis_attempt_count, 7)
        self.assertEqual(result.failed_pnp_call_count, 7)
        self.assertEqual(result.successful_pnp_call_count, 0)
        self.assertEqual(pnp_mock.call_count, 7)

    def test_refinement_records_only_strict_supporter_increases(self) -> None:
        def supporters_by_model(*args, **unused_keywords):
            translations = args[4]
            marker = int(round(float(np.asarray(translations[2])[0])))
            if marker == 4:
                count = 50
            elif marker == 50:
                count = 60
            else:
                count = 60
            return np.arange(count, dtype=int)

        result, pnp_mock = self._run(
            100, supporters_by_model, maximum_attempts=3000
        )
        initial_required = matching._required_hypothesis_count(100, 50)

        self.assertEqual(result.status, matching.ADAPTIVE_COMPLETED)
        self.assertEqual(result.random_hypothesis_attempt_count, initial_required)
        self.assertEqual(result.refinement_hypothesis_attempt_count, 2)
        self.assertEqual(
            result.total_hypothesis_attempt_count, initial_required + 2
        )
        self.assertEqual(pnp_mock.call_count, initial_required + 2)
        self.assertEqual(len(result.refinement_outcomes), 2)
        first, second = result.refinement_outcomes
        self.assertEqual(first.input_supporter_count, 50)
        self.assertEqual(first.output_supporter_count, 60)
        self.assertTrue(first.accepted_strict_increase)
        self.assertEqual(second.input_supporter_count, 60)
        self.assertEqual(second.output_supporter_count, 60)
        self.assertFalse(second.accepted_strict_increase)
        self.assertEqual(len(result.best_supporter_indices), 60)

    def test_insufficient_consensus_never_calls_solvepnp(self) -> None:
        result, pnp_mock = self._run(
            3, _supporter_indices(0), maximum_attempts=10
        )

        self.assertEqual(result.status, matching.INSUFFICIENT_CONSENSUS)
        self.assertEqual(result.total_hypothesis_attempt_count, 0)
        self.assertEqual(pnp_mock.call_count, 0)

    def test_nonfinite_point_cloud_fails_as_invalid_geometry(self) -> None:
        invalid_cloud = np.full((10, 3), np.nan)
        result, pnp_mock = self._run(
            10,
            _supporter_indices(0),
            maximum_attempts=10,
            point_cloud_override=invalid_cloud,
        )

        self.assertEqual(result.status, matching.INVALID_GEOMETRY)
        self.assertEqual(result.total_hypothesis_attempt_count, 0)
        self.assertEqual(pnp_mock.call_count, 0)

    def test_four_view_supporter_distance_is_exactly_two_pixels(self) -> None:
        observed = []

        def capture_distance(*unused, **keywords):
            observed.append(keywords["max_distance"])
            return np.arange(5, dtype=int)

        result, _ = self._run(
            10, capture_distance, maximum_attempts=10
        )

        self.assertIn(result.status, matching.TERMINAL_STATUSES)
        self.assertTrue(observed)
        self.assertEqual(
            set(observed),
            {matching.FOUR_VIEW_SUPPORTER_MAX_DISTANCE_PIXELS},
        )
        self.assertEqual(
            matching.FOUR_VIEW_SUPPORTER_MAX_DISTANCE_PIXELS, 2.0
        )


class ProductionMatchPolicyTest(unittest.TestCase):
    @staticmethod
    def _evidence(status):
        consensus_count = 100
        supporter_count = 30
        required = matching._required_hypothesis_count(
            consensus_count, supporter_count
        )
        if status == matching.ADAPTIVE_COMPLETED:
            random_attempts = required
            refinement_attempts = 1
            terminal_reason = (
                "adaptive_requirement_met_and_refinement_terminated"
            )
            refinement = (
                matching.RefinementPnPOutcome(
                    refinement_index=0,
                    total_attempt_number=required + 1,
                    input_supporter_count=supporter_count,
                    pnp_success=True,
                    geometry_valid=True,
                    output_supporter_count=supporter_count,
                    accepted_strict_increase=False,
                    reason="no_strict_supporter_increase",
                ),
            )
        else:
            random_attempts = matching.MAX_HYPOTHESIS_ATTEMPTS - 1
            refinement_attempts = 1
            terminal_reason = (
                "total_attempt_budget_reached_during_strict_refinement"
            )
            refinement = (
                matching.RefinementPnPOutcome(
                    refinement_index=0,
                    total_attempt_number=matching.MAX_HYPOTHESIS_ATTEMPTS,
                    input_supporter_count=supporter_count - 1,
                    pnp_success=True,
                    geometry_valid=True,
                    output_supporter_count=supporter_count,
                    accepted_strict_increase=True,
                    reason="strict_supporter_increase",
                ),
            )
        total_attempts = random_attempts + refinement_attempts
        score = SimpleNamespace(
            previous_frame_id=7,
            current_frame_id=42,
            previous_keyframe_ordinal=1,
            current_keyframe_ordinal=3,
            mahalanobis_squared=81.0,
            mahalanobis=9.0,
        )
        pair = matching.candidate_pair_from_score(
            score, pair_index=17, rank_within_current=2
        )
        rotations = tuple(np.eye(3) for _ in range(4))
        translations = tuple(np.zeros((3, 1)) for _ in range(4))
        ransac = matching.BoundedRansacResult(
            status=status,
            terminal_reason=terminal_reason,
            total_hypothesis_attempt_count=total_attempts,
            random_hypothesis_attempt_count=random_attempts,
            refinement_hypothesis_attempt_count=refinement_attempts,
            required_hypothesis_count=required,
            capped_required_hypothesis_count=required,
            maximum_hypothesis_attempt_count=(
                matching.MAX_HYPOTHESIS_ATTEMPTS
            ),
            successful_pnp_call_count=total_attempts,
            failed_pnp_call_count=0,
            invalid_geometry_model_count=0,
            valid_model_count=total_attempts,
            best_supporter_indices=tuple(range(supporter_count)),
            best_rotations=rotations,
            best_translations=translations,
            refinement_outcomes=refinement,
        )
        counts = matching.PairCounts(
            previous_left_keypoint_count=120,
            previous_right_keypoint_count=120,
            current_left_keypoint_count=120,
            current_right_keypoint_count=120,
            previous_raw_stereo_match_count=110,
            current_raw_stereo_match_count=110,
            previous_stereo_inlier_count=105,
            current_stereo_inlier_count=105,
            reciprocal_temporal_match_count=100,
            four_view_consensus_count=consensus_count,
            final_supporter_count=supporter_count,
            final_supporter_percentage=30.0,
        )
        runtime = matching.PairRuntime(
            image_loading_seconds=0.0,
            stereo_extraction_seconds=0.0,
            temporal_matching_seconds=0.0,
            triangulation_seconds=0.0,
            ransac_seconds=0.0,
            total_seconds=0.0,
        )
        return matching.PairEvidence(
            pair=pair, counts=counts, ransac=ransac, runtime=runtime
        )

    def test_both_complete_terminal_models_pass_identical_30_30_30(
        self,
    ) -> None:
        for status in matching.PRODUCTION_ELIGIBLE_TERMINAL_STATUSES:
            with self.subTest(status=status):
                accepted, reason = (
                    matching.production_match_policy_decision(
                        self._evidence(status)
                    )
                )
                self.assertTrue(accepted)
                self.assertEqual(
                    reason, "accepted_complete_model_and_30_30_30"
                )

    def test_eligible_status_cannot_bypass_complete_model_gate(self) -> None:
        evidence = self._evidence(
            matching.BUDGET_EXHAUSTED_WITH_BEST_MODEL
        )
        invalid_ransac = replace(evidence.ransac, best_rotations=None)
        invalid = replace(evidence, ransac=invalid_ransac)

        accepted, reason = matching.production_match_policy_decision(invalid)

        self.assertFalse(accepted)
        self.assertEqual(
            reason, "retained_best_model_is_incomplete_or_invalid"
        )


if __name__ == "__main__":
    unittest.main()
