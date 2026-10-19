"""Synthetic contract tests for the bounded E7-I03 integration gate."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import gtsam
import numpy as np

import exercise7_consensus_matching as matching
from PoseGraph import PoseGraph, camera_symbol
from exercise7_pose_graph_update import (
    ProtectedE6State,
    apply_one_loop_factor,
)
from exercise7_small_bundle import (
    CURRENT_FRAME_ID,
    PREVIOUS_FRAME_ID,
    LoopStereoObservation,
    SmallBundleValidationError,
    observations_from_e7_i03_pair_run,
    observations_from_pair_run,
    solve_small_bundle,
)


def _configuration():
    return {
        "scientific_configuration": {
            "measurement_uncertainty": {
                "stereo_sigma_px": [1.0, 1.0, 1.0],
            },
            "gauge_fixing_prior": {
                "rotation_sigma_rad": [1e-3, 1e-3, 1e-3],
                "translation_sigma_m": [0.1, 0.1, 0.1],
            },
            "landmark_eligibility": {
                "maximum_initialization_depth_m": 300.0,
            },
            "validation_tolerances": {
                "identity_pose_max_abs": 1e-9,
                "numerical_reconstruction_atol": 1e-9,
                "covariance_symmetry_atol": 1e-9,
                "covariance_minimum_eigenvalue": -1e-9,
                "graph_error_nonincrease_absolute": 1e-9,
                "graph_error_nonincrease_relative": 1e-12,
            },
        }
    }


class SyntheticSmallBundleTest(unittest.TestCase):
    def test_generic_observation_extraction_is_not_fixture_bounded(self):
        pair = matching.CandidatePair(
            pair_index=9,
            category="production_candidate",
            current_frame_id=97,
            current_keyframe_ordinal=9,
            previous_frame_id=19,
            previous_keyframe_ordinal=2,
            rank_within_current=1,
            ordinal_gap=7,
            frame_id_gap=78,
            mahalanobis_squared=81.0,
            mahalanobis=9.0,
            selection_reason="synthetic arbitrary pair",
        )
        counts = SimpleNamespace(
            four_view_consensus_count=2,
            final_supporter_count=2,
        )
        evidence = SimpleNamespace(
            pair=pair,
            counts=counts,
            ransac=SimpleNamespace(best_supporter_indices=(0, 1)),
        )
        visualization = SimpleNamespace(
            consensus=((0, 0, 0), (1, 1, 1)),
            supporter_indices=(0, 1),
            previous_stereo_inliers=(),
            current_stereo_inliers=(),
            previous_keypoints_left=(),
            previous_keypoints_right=(),
            current_keypoints_left=(),
            current_keypoints_right=(),
        )
        pair_run = SimpleNamespace(
            evidence=evidence, visualization=visualization
        )
        actual_pixels = np.zeros((4, 2, 2), dtype=float)
        actual_pixels[0, 0, :] = 10.0
        actual_pixels[1, 0, :] = 8.0
        actual_pixels[2, 0, :] = 11.0
        actual_pixels[3, 0, :] = 9.0

        with mock.patch(
            "exercise7_small_bundle.tools.extract_actual_consensus_pixels",
            return_value=actual_pixels,
        ):
            observations = observations_from_pair_run(pair_run)

        self.assertEqual(len(observations), 2)
        self.assertEqual(
            observations[0].previous_measurement, (10.0, 8.0, 0.0)
        )
        with self.assertRaises(SmallBundleValidationError):
            observations_from_e7_i03_pair_run(pair_run)

    def test_two_pose_bundle_direction_covariance_and_factor_types(self):
        calibration = gtsam.Cal3_S2Stereo(
            120.0, 118.0, 0.0, 320.0, 240.0, 0.30
        )
        true_relative = gtsam.Pose3(
            gtsam.Rot3.RzRyRx(0.01, -0.02, 0.015),
            np.asarray([0.35, -0.04, 0.12], dtype=float),
        )
        previous_camera = gtsam.StereoCamera(
            gtsam.Pose3(), calibration
        )
        current_camera = gtsam.StereoCamera(
            true_relative, calibration
        )
        points = (
            np.asarray(
                [
                    -1.2 + 0.35 * index,
                    -0.5 + 0.17 * (index % 4),
                    8.0 + 0.8 * index,
                ],
                dtype=float,
            )
            for index in range(8)
        )
        observations = []
        for index, point in enumerate(points):
            previous = tuple(
                float(value)
                for value in previous_camera.project(point).vector()
            )
            current = tuple(
                float(value)
                for value in current_camera.project(point).vector()
            )
            observations.append(
                LoopStereoObservation(
                    supporter_ordinal=index,
                    consensus_index=index + 20,
                    previous_measurement=previous,
                    current_measurement=current,
                )
            )
        perturbation = np.asarray(
            [0.005, -0.004, 0.003, 0.04, -0.02, 0.03],
            dtype=float,
        )
        initial_relative = true_relative.retract(perturbation)
        previous_frame_id = 19
        current_frame_id = 97
        result = solve_small_bundle(
            observations=observations,
            initial_relative_pose=initial_relative,
            calibration=calibration,
            configuration=_configuration(),
            previous_frame_id=previous_frame_id,
            current_frame_id=current_frame_id,
            expected_supporter_count=len(observations),
        )

        self.assertEqual(result.previous_frame_id, previous_frame_id)
        self.assertEqual(result.current_frame_id, current_frame_id)
        self.assertEqual(
            result.previous_camera_key,
            int(camera_symbol(previous_frame_id)),
        )
        self.assertEqual(
            result.current_camera_key,
            int(camera_symbol(current_frame_id)),
        )
        self.assertTrue(
            result.initial_values.exists(result.previous_camera_key)
        )
        self.assertTrue(
            result.initial_values.exists(result.current_camera_key)
        )
        self.assertEqual(len(result.accepted_observations), 8)
        self.assertEqual(result.rejection_reason_counts, {})
        self.assertEqual(result.initial_values.size(), 10)
        self.assertEqual(result.graph.size(), 17)
        self.assertIsInstance(
            result.graph.at(0), gtsam.PriorFactorPose3
        )
        for index in range(1, result.graph.size()):
            self.assertIsInstance(
                result.graph.at(index), gtsam.GenericStereoFactor3D
            )
        self.assertEqual(
            [
                int(result.graph.at(index).keys()[0])
                for index in range(1, result.graph.size())
            ],
            [
                key
                for _ in observations
                for key in (
                    result.previous_camera_key,
                    result.current_camera_key,
                )
            ],
        )
        for observation in observations:
            self.assertTrue(
                result.initial_values.exists(
                    int(gtsam.symbol("q", observation.consensus_index))
                )
            )
        self.assertLessEqual(result.final_error, result.initial_error)
        self.assertLessEqual(result.local_anchor_identity_max_abs, 1e-9)
        self.assertLessEqual(
            result.direction_composition_max_abs, 1e-9
        )
        self.assertLessEqual(
            result.direction_point_mapping_max_abs, 1e-9
        )
        self.assertEqual(result.conditional_covariance.shape, (6, 6))
        self.assertEqual(
            result.covariance_diagnostics["joint_block_order_source"],
            "explicit ordered KeyVector [c19, c97]",
        )
        np.linalg.cholesky(result.conditional_covariance)
        self.assertLessEqual(result.standalone_factor_error, 1e-9)


class SyntheticPoseGraphUpdateTest(unittest.TestCase):
    def test_one_loop_factor_optimizes_and_recomputes_all_marginals(self):
        covariance = np.diag(
            [1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1]
        )
        first_edge = gtsam.Pose3(
            gtsam.Rot3.RzRyRx(0.01, 0.0, 0.0),
            np.asarray([1.0, 0.0, 0.1]),
        )
        second_edge = gtsam.Pose3(
            gtsam.Rot3.RzRyRx(0.0, -0.01, 0.0),
            np.asarray([1.0, 0.1, 0.0]),
        )
        graph = PoseGraph(
            keyframe_ids=[0, PREVIOUS_FRAME_ID, CURRENT_FRAME_ID],
            relative_poses=[first_edge, second_edge],
            covariance_matrices=[covariance, covariance],
            prior_sigmas=[1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1],
        )
        pre_values = graph.optimize()
        pre_matrices = np.asarray(
            [
                pre_values.atPose3(key).matrix()
                for key in graph.camera_keys
            ],
            dtype=float,
        )
        inputs = SimpleNamespace(
            keyframe_ids=(0, PREVIOUS_FRAME_ID, CURRENT_FRAME_ID),
            validation_tolerances={
                "identity_pose_max_abs": 1e-9,
                "covariance_symmetry_atol": 1e-9,
                "graph_error_nonincrease_absolute": 1e-9,
                "graph_error_nonincrease_relative": 1e-12,
            },
        )
        state = ProtectedE6State(
            inputs=inputs,
            pose_graph=graph,
            optimized_values=pre_values,
            optimized_pose_matrices=pre_matrices,
            marginal_covariances=np.empty((3, 6, 6)),
            fresh_vs_protected_pose_max_abs=0.0,
            original_graph_error=float(graph.graph().error(pre_values)),
        )
        measured_loop = gtsam.Pose3(
            second_edge.rotation(),
            np.asarray([0.75, 0.08, 0.02], dtype=float),
        )
        result = apply_one_loop_factor(
            state=state,
            relative_pose=measured_loop,
            conditional_covariance=np.diag(
                [5e-4, 5e-4, 5e-4, 0.03, 0.03, 0.03]
            ),
        )

        self.assertEqual(graph.graph().size(), 3)
        self.assertEqual(result.graph.size(), 4)
        self.assertLessEqual(
            result.augmented_post_optimization_error,
            result.augmented_pre_optimization_error,
        )
        self.assertEqual(result.marginal_covariances.shape, (3, 6, 6))
        self.assertTrue(np.all(np.isfinite(result.marginal_covariances)))
        self.assertLessEqual(result.global_anchor_identity_max_abs, 1e-9)
        self.assertIn(
            PREVIOUS_FRAME_ID, result.movement_by_frame_id
        )
        self.assertIn(CURRENT_FRAME_ID, result.movement_by_frame_id)
        self.assertTrue(
            result.optimized_values.exists(
                int(camera_symbol(CURRENT_FRAME_ID))
            )
        )


if __name__ == "__main__":
    unittest.main()
