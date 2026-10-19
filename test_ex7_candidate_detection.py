"""Synthetic and bounded canonical tests for E7-I01."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import MappingProxyType

import gtsam
import numpy as np

from PoseGraph import EXPECTED_EDGE_COUNT, EXPECTED_KEYFRAME_COUNT, camera_symbol
from exercise7_candidate_detection import (
    COURSE_HEURISTIC_POLICY,
    MAOR_COMPARISON_POLICY,
    CandidateDetectionError,
    CandidateDetectionInputs,
    CovarianceEdge,
    CovarianceGraph,
    build_optimized_values,
    candidate_detection_inputs_from_values,
    deterministic_probe_current_ordinals,
    evaluate_gtsam_mahalanobis,
    load_candidate_detection_inputs,
    score_candidate_pair,
)


def _edge(
    edge_index: int,
    start_frame_id: int,
    end_frame_id: int,
    covariance: np.ndarray,
) -> CovarianceEdge:
    return CovarianceEdge(
        edge_index=edge_index,
        start_frame_id=start_frame_id,
        end_frame_id=end_frame_id,
        covariance=np.asarray(covariance, dtype=float),
        source_directory=Path("."),
    )


def _synthetic_inputs(
    frame_ids, pose_matrices, edges
) -> CandidateDetectionInputs:
    poses = np.asarray(pose_matrices, dtype=float)
    poses.setflags(write=False)
    keyframe_ids = tuple(int(value) for value in frame_ids)
    return CandidateDetectionInputs(
        e6_root=Path("."),
        e5_root=Path("."),
        configuration_sha256="synthetic",
        keyframe_ids=keyframe_ids,
        frame_id_to_ordinal=MappingProxyType(
            {
                frame_id: ordinal
                for ordinal, frame_id in enumerate(keyframe_ids)
            }
        ),
        optimized_pose_matrices=poses,
        edges=tuple(edges),
        covariance_symmetry_atol=1e-12,
        pose_reconstruction_atol=1e-12,
    )


class SyntheticCovarianceGraphTest(unittest.TestCase):
    def test_noncontiguous_chain_accumulates_raw_covariances(self) -> None:
        frame_ids = (3, 17, 42, 99)
        covariance_arrays = [
            np.diag([0.1, 0.2, 0.3, 1.0, 2.0, 3.0]),
            np.diag([0.2, 0.3, 0.4, 2.0, 3.0, 4.0]),
            np.diag([0.3, 0.4, 0.5, 3.0, 4.0, 5.0]),
        ]
        original_arrays = [value.copy() for value in covariance_arrays]
        graph = CovarianceGraph(
            frame_ids,
            [
                _edge(index, frame_ids[index], frame_ids[index + 1], cov)
                for index, cov in enumerate(covariance_arrays)
            ],
            covariance_symmetry_atol=1e-12,
        )

        result = graph.find_course_covariance_path(3, 99)

        self.assertEqual(result.policy, COURSE_HEURISTIC_POLICY)
        self.assertEqual(result.frame_ids, frame_ids)
        self.assertEqual(result.edge_indices, (0, 1, 2))
        np.testing.assert_allclose(
            result.covariance,
            np.sum(np.stack(covariance_arrays), axis=0),
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreater(result.covariance_determinant, 0.0)
        self.assertTrue(math.isfinite(result.covariance_log_determinant))
        self.assertFalse(result.covariance.flags.writeable)
        for observed, expected in zip(covariance_arrays, original_arrays):
            np.testing.assert_array_equal(observed, expected)

    def test_course_and_maor_policies_choose_different_diamond_paths(
        self,
    ) -> None:
        anisotropic_first = np.diag([0.01, 100.0, 1, 1, 1, 1])
        anisotropic_second = np.diag([100.0, 0.01, 1, 1, 1, 1])
        moderate = np.diag([2.0, 2.0, 1, 1, 1, 1])
        graph = CovarianceGraph(
            (0, 10, 20, 30),
            (
                _edge(0, 0, 10, anisotropic_first),
                _edge(1, 10, 30, anisotropic_second),
                _edge(2, 0, 20, moderate),
                _edge(3, 20, 30, moderate),
            ),
            covariance_symmetry_atol=1e-12,
        )

        course = graph.find_course_covariance_path(0, 30)
        maor = graph.find_maor_comparison_path(0, 30)

        self.assertEqual(course.frame_ids, (0, 20, 30))
        self.assertEqual(course.edge_indices, (2, 3))
        self.assertEqual(course.policy, COURSE_HEURISTIC_POLICY)
        self.assertEqual(maor.frame_ids, (0, 10, 30))
        self.assertEqual(maor.edge_indices, (0, 1))
        self.assertEqual(maor.policy, MAOR_COMPARISON_POLICY)
        self.assertLess(
            course.covariance_determinant,
            maor.covariance_determinant,
        )
        self.assertLess(maor.scalar_cost, 4.0)

    def test_missing_reverse_path_fails_closed(self) -> None:
        graph = CovarianceGraph(
            (0, 7),
            (_edge(0, 0, 7, np.eye(6)),),
            covariance_symmetry_atol=1e-12,
        )
        with self.assertRaisesRegex(
            CandidateDetectionError, "no directed covariance path"
        ):
            graph.find_course_covariance_path(7, 0)

    def test_non_positive_definite_covariance_is_rejected(self) -> None:
        invalid = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
        with self.assertRaisesRegex(
            CandidateDetectionError, "positive definite"
        ):
            CovarianceGraph(
                (0, 7),
                (_edge(0, 0, 7, invalid),),
                covariance_symmetry_atol=1e-12,
            )

    def test_directed_edge_insertion_is_immutable_raw_and_deterministic(
        self,
    ) -> None:
        frame_ids = (3, 17, 42)
        graph = CovarianceGraph(
            frame_ids,
            (
                _edge(4, 3, 17, 2.0 * np.eye(6)),
                _edge(9, 17, 42, 2.0 * np.eye(6)),
            ),
            covariance_symmetry_atol=1e-12,
        )
        raw_covariance = np.diag(
            [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
        )
        raw_bytes = raw_covariance.tobytes(order="C")

        updated = graph.with_directed_edge(
            3,
            42,
            raw_covariance,
            source_directory=Path("accepted-loop-000"),
        )

        self.assertEqual(tuple(edge.edge_index for edge in graph.edges), (4, 9))
        self.assertNotIn(10, graph.edge_by_index)
        self.assertEqual(
            tuple(edge.edge_index for edge in updated.edges), (4, 9, 10)
        )
        inserted = updated.edge_by_index[10]
        self.assertEqual(
            (inserted.start_frame_id, inserted.end_frame_id), (3, 42)
        )
        self.assertEqual(inserted.source_directory, Path("accepted-loop-000"))
        self.assertEqual(inserted.covariance.tobytes(order="C"), raw_bytes)
        self.assertEqual(raw_covariance.tobytes(order="C"), raw_bytes)
        self.assertFalse(inserted.covariance.flags.writeable)
        path = updated.find_course_covariance_path(3, 42)
        self.assertEqual(path.edge_indices, (10,))
        np.testing.assert_array_equal(path.covariance, raw_covariance)

        with self.assertRaisesRegex(
            CandidateDetectionError, "directed covariance arc already exists"
        ):
            updated.with_directed_edge(3, 42, raw_covariance)
        self.assertEqual(len(updated.edges), 3)


class SyntheticMahalanobisTest(unittest.TestCase):
    def test_between_factor_score_has_documented_factor_of_two(self) -> None:
        previous_frame_id = 7
        current_frame_id = 42
        tangent = np.asarray([0.1, -0.2, 0.05, 1.0, -0.5, 0.25])
        current_pose = gtsam.Pose3.Expmap(tangent)
        values = gtsam.Values()
        values.insert(camera_symbol(previous_frame_id), gtsam.Pose3())
        values.insert(camera_symbol(current_frame_id), current_pose)

        factor_error, mahalanobis_squared, mahalanobis = (
            evaluate_gtsam_mahalanobis(
                values,
                previous_frame_id,
                current_frame_id,
                np.eye(6),
            )
        )

        expected_squared = float(np.dot(tangent, tangent))
        self.assertAlmostEqual(factor_error, 0.5 * expected_squared, places=12)
        self.assertAlmostEqual(mahalanobis_squared, expected_squared, places=12)
        self.assertAlmostEqual(
            mahalanobis, math.sqrt(expected_squared), places=12
        )

    def test_coincident_poses_score_zero(self) -> None:
        values = gtsam.Values()
        values.insert(camera_symbol(5), gtsam.Pose3())
        values.insert(camera_symbol(11), gtsam.Pose3())
        score = evaluate_gtsam_mahalanobis(
            values, 5, 11, np.eye(6)
        )
        self.assertEqual(score, (0.0, 0.0, 0.0))

    def test_pair_scoring_requires_historical_to_current_order(self) -> None:
        frame_ids = (5, 11)
        poses = np.asarray([np.eye(4), np.eye(4)])
        edge = _edge(0, 5, 11, np.eye(6))
        inputs = _synthetic_inputs(frame_ids, poses, (edge,))
        graph = CovarianceGraph(frame_ids, (edge,))
        values = build_optimized_values(inputs)

        with self.assertRaisesRegex(
            CandidateDetectionError, "historical FrameId"
        ):
            score_candidate_pair(inputs, graph, values, 11, 5)

    def test_live_values_produce_an_immutable_candidate_input_snapshot(
        self,
    ) -> None:
        frame_ids = (5, 11)
        baseline_poses = np.asarray([np.eye(4), np.eye(4)])
        edge = _edge(0, 5, 11, np.eye(6))
        baseline = _synthetic_inputs(frame_ids, baseline_poses, (edge,))
        live_values = gtsam.Values()
        live_values.insert(camera_symbol(5), gtsam.Pose3())
        live_pose = gtsam.Pose3.Expmap(
            np.asarray([0.1, -0.2, 0.05, 1.0, -0.5, 0.25])
        )
        live_values.insert(camera_symbol(11), live_pose)

        live = candidate_detection_inputs_from_values(
            baseline, live_values
        )

        np.testing.assert_array_equal(
            baseline.optimized_pose_matrices, baseline_poses
        )
        np.testing.assert_allclose(
            live.optimized_pose_matrices[1],
            live_pose.matrix(),
            atol=0.0,
            rtol=0.0,
        )
        self.assertFalse(live.optimized_pose_matrices.flags.writeable)
        self.assertIs(live.edges, baseline.edges)
        reconstructed = build_optimized_values(live)
        np.testing.assert_allclose(
            reconstructed.atPose3(camera_symbol(11)).matrix(),
            live_pose.matrix(),
            atol=0.0,
            rtol=0.0,
        )


class CanonicalCandidateDetectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_candidate_detection_inputs()
        cls.graph = CovarianceGraph(
            cls.inputs.keyframe_ids,
            cls.inputs.edges,
            covariance_symmetry_atol=cls.inputs.covariance_symmetry_atol,
        )
        cls.values = build_optimized_values(cls.inputs)

    def test_canonical_counts_ordering_and_probe_ordinals(self) -> None:
        repository_root = Path(__file__).resolve().parent
        self.assertEqual(
            self.inputs.e5_root,
            (repository_root / "diagnostics/ex5_clean_final").resolve(),
        )
        self.assertEqual(
            self.inputs.e6_root,
            (repository_root / "diagnostics/ex6_clean_final").resolve(),
        )
        self.assertEqual(
            len(self.inputs.keyframe_ids), EXPECTED_KEYFRAME_COUNT
        )
        self.assertEqual(len(self.inputs.edges), EXPECTED_EDGE_COUNT)
        self.assertEqual(self.inputs.keyframe_ids[0], 0)
        self.assertEqual(self.inputs.keyframe_ids[-1], 3359)
        self.assertFalse(self.inputs.optimized_pose_matrices.flags.writeable)

        ordinals = deterministic_probe_current_ordinals(
            len(self.inputs.keyframe_ids)
        )
        self.assertEqual(ordinals, (118, 237, 355))
        probe_frame_ids = tuple(
            self.inputs.keyframe_ids[ordinal] for ordinal in ordinals
        )
        self.assertTrue(
            all(
                frame_id in self.inputs.frame_id_to_ordinal
                for frame_id in probe_frame_ids
            )
        )

    def test_explicit_historical_pair_passes_and_mixed_pair_fails(self) -> None:
        repository_root = Path(__file__).resolve().parent
        historical_e5 = repository_root / "diagnostics/ex5_final"
        historical_e6 = repository_root / "diagnostics/ex6_pose_graph"
        if not historical_e5.exists() or not historical_e6.exists():
            self.skipTest(
                "optional historical E7 candidate-detection fixtures are "
                "not present: {} / {}".format(
                    historical_e5, historical_e6
                )
            )
        historical = load_candidate_detection_inputs(
            e5_root=historical_e5,
            e6_root=historical_e6,
        )
        self.assertEqual(historical.e5_root, historical_e5.resolve())
        self.assertEqual(historical.e6_root, historical_e6.resolve())
        self.assertEqual(len(historical.keyframe_ids), EXPECTED_KEYFRAME_COUNT)
        self.assertEqual(len(historical.edges), EXPECTED_EDGE_COUNT)

        with self.assertRaisesRegex(
            CandidateDetectionError,
            "selected E6 provenance does not identify the selected E5 root",
        ):
            load_candidate_detection_inputs(
                e5_root=self.inputs.e5_root,
                e6_root=historical_e6,
            )

    def test_derived_canonical_pairs_have_exact_paths_and_finite_scores(
        self,
    ) -> None:
        current_ordinals = deterministic_probe_current_ordinals(
            len(self.inputs.keyframe_ids)
        )
        pair_ordinals = (
            (current_ordinals[0] // 2, current_ordinals[0]),
            (current_ordinals[1] // 2, current_ordinals[1]),
            (0, current_ordinals[2]),
        )
        for previous_ordinal, current_ordinal in pair_ordinals:
            previous_frame_id = self.inputs.keyframe_ids[previous_ordinal]
            current_frame_id = self.inputs.keyframe_ids[current_ordinal]
            self.assertIn(
                previous_frame_id, self.inputs.frame_id_to_ordinal
            )
            self.assertIn(current_frame_id, self.inputs.frame_id_to_ordinal)

            score = score_candidate_pair(
                self.inputs,
                self.graph,
                self.values,
                previous_frame_id,
                current_frame_id,
            )
            self.assertEqual(
                score.path.edge_indices,
                tuple(range(previous_ordinal, current_ordinal)),
            )
            expected_covariance = np.sum(
                np.stack(
                    [
                        self.inputs.edges[index].covariance
                        for index in range(
                            previous_ordinal, current_ordinal
                        )
                    ],
                    axis=0,
                ),
                axis=0,
            )
            np.testing.assert_allclose(
                score.path.covariance,
                expected_covariance,
                atol=0.0,
                rtol=0.0,
            )
            self.assertTrue(math.isfinite(score.factor_error))
            self.assertTrue(math.isfinite(score.mahalanobis_squared))
            self.assertTrue(math.isfinite(score.mahalanobis))
            self.assertAlmostEqual(
                score.mahalanobis_squared,
                2.0 * score.factor_error,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
