"""Focused synthetic and canonical contract tests for E6-I01."""

from __future__ import annotations

import unittest
from pathlib import Path

import gtsam
import numpy as np

from PoseGraph import (
    CANONICAL_INPUT_ROOT,
    CONDITIONAL_COVARIANCE_FIELD,
    EDGE_DIRECTION,
    EXPECTED_EDGE_COUNT,
    EXPECTED_KEYFRAME_COUNT,
    TANGENT_ORDER,
    PoseGraph,
    camera_symbol,
    load_canonical_pose_graph_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
HISTORICAL_INPUT_ROOT = (
    REPOSITORY_ROOT / "diagnostics" / "ex5_final"
)
CLEAN_CONFIGURATION_SHA256 = (
    "34d60fc57b8e6d7fca8b78807c81a2f971608bed4b5cdb91eb76495e2eac1d6c"
)


class SyntheticPoseGraphChainTest(unittest.TestCase):
    def test_three_pose_chain_optimizes_and_marginalizes(self) -> None:
        first_edge = gtsam.Pose3(
            gtsam.Rot3.RzRyRx(0.01, -0.02, 0.03),
            np.array([1.0, 0.2, 0.1]),
        )
        second_edge = gtsam.Pose3(
            gtsam.Rot3.RzRyRx(-0.02, 0.01, -0.01),
            np.array([0.8, -0.1, 0.3]),
        )
        covariance = np.diag(
            [1e-3, 2e-3, 3e-3, 0.1, 0.2, 0.3]
        )
        graph = PoseGraph(
            keyframe_ids=[0, 7, 19],
            relative_poses=[first_edge, second_edge],
            covariance_matrices=[covariance, covariance],
            prior_sigmas=[1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1],
        )

        self.assertEqual(graph.graph().size(), 3)
        self.assertEqual(graph.get_initial_estimate().size(), 3)
        expected_last = first_edge.compose(second_edge)
        np.testing.assert_allclose(
            graph.get_initial_estimate()
            .atPose3(camera_symbol(19))
            .matrix(),
            expected_last.matrix(),
            atol=1e-12,
            rtol=0.0,
        )
        result = graph.optimize()
        self.assertTrue(
            np.all(
                np.isfinite(
                    result.atPose3(camera_symbol(19)).matrix()
                )
            )
        )
        marginals = graph.compute_marginals()
        covariance_array = graph.marginal_covariance_array()
        self.assertIsInstance(marginals, gtsam.Marginals)
        self.assertEqual(covariance_array.shape, (3, 6, 6))
        self.assertTrue(np.all(np.isfinite(covariance_array)))


class CanonicalPoseGraphContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_canonical_pose_graph_inputs()
        cls.pose_graph = PoseGraph.from_canonical_inputs(cls.inputs)

    def test_canonical_input_and_graph_counts(self) -> None:
        self.assertEqual(
            self.inputs.input_root, CANONICAL_INPUT_ROOT.resolve()
        )
        self.assertEqual(
            self.inputs.configuration_sha256,
            CLEAN_CONFIGURATION_SHA256,
        )
        self.assertEqual(
            len(self.inputs.keyframe_ids), EXPECTED_KEYFRAME_COUNT
        )
        self.assertEqual(len(self.inputs.edges), EXPECTED_EDGE_COUNT)
        self.assertEqual(
            self.inputs.keyframe_ids[0], 0
        )
        self.assertEqual(
            self.inputs.keyframe_ids[-1], 3359
        )
        self.assertEqual(
            len(set(self.inputs.keyframe_ids)),
            EXPECTED_KEYFRAME_COUNT,
        )
        self.assertEqual(self.pose_graph.graph().size(), 356)
        self.assertEqual(
            self.pose_graph.get_initial_estimate().size(), 356
        )
        self.assertEqual(
            self.pose_graph.camera_keys,
            tuple(camera_symbol(value) for value in self.inputs.keyframe_ids),
        )
        self.assertIsInstance(
            self.pose_graph.graph().at(0), gtsam.PriorFactorPose3
        )
        for factor_index in range(1, self.pose_graph.graph().size()):
            factor = self.pose_graph.graph().at(factor_index)
            edge = self.inputs.edges[factor_index - 1]
            self.assertIsInstance(
                factor,
                gtsam.BetweenFactorPose3,
            )
            self.assertEqual(
                factor.keys(),
                [
                    camera_symbol(edge.start_frame_id),
                    camera_symbol(edge.end_frame_id),
                ],
            )
            np.testing.assert_array_equal(
                factor.measured().matrix(), edge.relative_pose.matrix()
            )
            np.testing.assert_allclose(
                factor.noiseModel().covariance(),
                edge.covariance,
                atol=self.inputs.validation_tolerances[
                    "numerical_reconstruction_atol"
                ],
                rtol=0.0,
            )

    def test_canonical_edges_covariances_and_initial_composition(self) -> None:
        tolerance = self.inputs.validation_tolerances[
            "numerical_reconstruction_atol"
        ]
        for index, edge in enumerate(self.inputs.edges):
            self.assertEqual(edge.window_index, index)
            self.assertEqual(
                edge.start_frame_id, self.inputs.keyframe_ids[index]
            )
            self.assertEqual(
                edge.end_frame_id,
                self.inputs.keyframe_ids[index + 1],
            )
            self.assertEqual(EDGE_DIRECTION, "T_Cstart_from_Cend")
            self.assertEqual(
                CONDITIONAL_COVARIANCE_FIELD,
                "conditional_end_given_start_covariance",
            )
            self.assertEqual(
                TANGENT_ORDER,
                (
                    "rot_x_rad",
                    "rot_y_rad",
                    "rot_z_rad",
                    "trans_x_m",
                    "trans_y_m",
                    "trans_z_m",
                ),
            )
            self.assertEqual(edge.covariance.shape, (6, 6))
            self.assertTrue(np.all(np.isfinite(edge.covariance)))
            reconstructed = (
                self.pose_graph.get_initial_estimate()
                .atPose3(camera_symbol(edge.start_frame_id))
                .between(
                    self.pose_graph.get_initial_estimate().atPose3(
                        camera_symbol(edge.end_frame_id)
                    )
                )
            )
            np.testing.assert_allclose(
                reconstructed.matrix(),
                edge.relative_pose.matrix(),
                atol=tolerance,
                rtol=0.0,
            )

    def test_clean_and_historical_e5_pose_graph_inputs_are_identical(
        self,
    ) -> None:
        if not HISTORICAL_INPUT_ROOT.exists():
            self.skipTest(
                "optional historical E5 equivalence fixture is not present: "
                + str(HISTORICAL_INPUT_ROOT)
            )

        historical = load_canonical_pose_graph_inputs(
            HISTORICAL_INPUT_ROOT
        )
        self.assertEqual(
            self.inputs.keyframe_ids, historical.keyframe_ids
        )
        np.testing.assert_array_equal(
            self.inputs.prior_sigmas, historical.prior_sigmas
        )
        for clean_edge, historical_edge in zip(
            self.inputs.edges, historical.edges
        ):
            self.assertEqual(
                (
                    clean_edge.window_index,
                    clean_edge.start_frame_id,
                    clean_edge.end_frame_id,
                ),
                (
                    historical_edge.window_index,
                    historical_edge.start_frame_id,
                    historical_edge.end_frame_id,
                ),
            )
            np.testing.assert_array_equal(
                clean_edge.relative_pose.matrix(),
                historical_edge.relative_pose.matrix(),
            )
            np.testing.assert_array_equal(
                clean_edge.covariance, historical_edge.covariance
            )


if __name__ == "__main__":
    unittest.main()
