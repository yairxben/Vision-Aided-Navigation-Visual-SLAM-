"""Focused cumulative Pose Graph update tests for E7-I04."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gtsam
import numpy as np

from PoseGraph import PoseGraph, camera_symbol
from exercise7_pose_graph_update import (
    CumulativePoseGraphState,
    PoseGraphUpdateValidationError,
    ProtectedE6State,
    load_protected_e6_state,
)


KEYFRAME_IDS = (0, 11, 23, 47)


def _synthetic_protected_state() -> ProtectedE6State:
    covariance = np.diag(
        [1e-3, 1e-3, 1e-3, 0.08, 0.08, 0.08]
    )
    relative_poses = [
        gtsam.Pose3(
            gtsam.Rot3.RzRyRx(0.01, 0.0, 0.0),
            np.asarray([1.0, 0.0, 0.05], dtype=float),
        ),
        gtsam.Pose3(
            gtsam.Rot3.RzRyRx(0.0, -0.015, 0.0),
            np.asarray([0.9, 0.12, 0.0], dtype=float),
        ),
        gtsam.Pose3(
            gtsam.Rot3.RzRyRx(-0.005, 0.0, 0.012),
            np.asarray([1.1, -0.04, 0.08], dtype=float),
        ),
    ]
    pose_graph = PoseGraph(
        keyframe_ids=KEYFRAME_IDS,
        relative_poses=relative_poses,
        covariance_matrices=[covariance] * 3,
        prior_sigmas=[1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1],
    )
    optimized_values = pose_graph.optimize()
    optimized_pose_matrices = np.asarray(
        [
            optimized_values.atPose3(key).matrix()
            for key in pose_graph.camera_keys
        ],
        dtype=float,
    )
    marginals = gtsam.Marginals(pose_graph.graph(), optimized_values)
    marginal_covariances = np.asarray(
        [
            marginals.marginalCovariance(key)
            for key in pose_graph.camera_keys
        ],
        dtype=float,
    )
    inputs = SimpleNamespace(
        keyframe_ids=KEYFRAME_IDS,
        validation_tolerances={
            "identity_pose_max_abs": 1e-9,
            "covariance_symmetry_atol": 1e-9,
            "graph_error_nonincrease_absolute": 1e-9,
            "graph_error_nonincrease_relative": 1e-12,
        },
    )
    return ProtectedE6State(
        inputs=inputs,
        pose_graph=pose_graph,
        optimized_values=optimized_values,
        optimized_pose_matrices=optimized_pose_matrices,
        marginal_covariances=marginal_covariances,
        fresh_vs_protected_pose_max_abs=0.0,
        original_graph_error=float(
            pose_graph.graph().error(optimized_values)
        ),
    )


def _pose_matrices(values: gtsam.Values) -> np.ndarray:
    return np.asarray(
        [
            values.atPose3(int(camera_symbol(frame_id))).matrix()
            for frame_id in KEYFRAME_IDS
        ],
        dtype=float,
    )


class CumulativePoseGraphStateTest(unittest.TestCase):
    def test_bootstrap_uses_the_selected_e5_provenance(self):
        repository_root = Path(__file__).resolve().parent
        clean_e5 = repository_root / "diagnostics/ex5_clean_final"
        clean_e6 = repository_root / "diagnostics/ex6_clean_final"
        state = load_protected_e6_state(
            summary_path=clean_e6 / "summary.json",
            values_path=clean_e6 / "pose_graph_values.npz",
            marginals_path=clean_e6 / "pose_graph_marginals.npz",
            e5_root=clean_e5,
        )
        self.assertEqual(state.inputs.input_root, clean_e5.resolve())
        self.assertEqual(len(state.inputs.keyframe_ids), 356)
        self.assertEqual(state.pose_graph.graph().size(), 356)

        historical_e6 = repository_root / "diagnostics/ex6_pose_graph"
        if not historical_e6.exists():
            self.skipTest(
                "optional historical E6 mismatch fixture is not present: "
                + str(historical_e6)
            )
        with self.assertRaisesRegex(
            PoseGraphUpdateValidationError,
            "protected E6 summary contract differs",
        ):
            load_protected_e6_state(
                summary_path=historical_e6 / "summary.json",
                values_path=historical_e6 / "pose_graph_values.npz",
                marginals_path=historical_e6
                / "pose_graph_marginals.npz",
                e5_root=clean_e5,
            )

    def test_two_arbitrary_updates_use_latest_graph_and_values(self):
        protected = _synthetic_protected_state()
        state = CumulativePoseGraphState.from_protected_e6(protected)
        loop_covariance = np.diag(
            [5e-4, 5e-4, 5e-4, 0.025, 0.025, 0.025]
        )

        first_measurement = state.initial_relative_pose(0, 23).retract(
            np.asarray(
                [0.002, -0.001, 0.001, -0.16, 0.03, -0.02],
                dtype=float,
            )
        )
        first = state.apply_loop_factor(
            previous_frame_id=0,
            current_frame_id=23,
            relative_pose=first_measurement,
            conditional_covariance=loop_covariance,
        )

        self.assertEqual(first.pre_update_state_version, 0)
        self.assertEqual(first.state_version, 1)
        self.assertEqual((first.previous_frame_id, first.current_frame_id), (0, 23))
        self.assertEqual(first.graph.size(), 5)
        self.assertEqual(
            tuple(int(key) for key in first.graph.at(4).keys()),
            (int(camera_symbol(0)), int(camera_symbol(23))),
        )
        first_optimized_matrices = first.optimized_pose_matrices.copy()
        protected_second_initialization = (
            protected.optimized_values.atPose3(int(camera_symbol(11))).between(
                protected.optimized_values.atPose3(int(camera_symbol(47)))
            )
        )
        latest_second_initialization = state.initial_relative_pose(11, 47)
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        np.asarray(latest_second_initialization.matrix())
                        - np.asarray(protected_second_initialization.matrix())
                    )
                )
            ),
            1e-8,
        )

        second_measurement = latest_second_initialization.retract(
            np.asarray(
                [-0.001, 0.002, -0.001, 0.08, -0.04, 0.015],
                dtype=float,
            )
        )
        second = state.apply_loop_factor(
            previous_frame_id=11,
            current_frame_id=47,
            relative_pose=second_measurement,
            conditional_covariance=loop_covariance,
        )

        self.assertEqual(second.pre_update_state_version, 1)
        self.assertEqual(second.state_version, 2)
        self.assertEqual(
            (second.previous_frame_id, second.current_frame_id), (11, 47)
        )
        np.testing.assert_array_equal(
            second.pre_loop_pose_matrices, first_optimized_matrices
        )
        np.testing.assert_array_equal(
            _pose_matrices(second.pre_loop_values),
            first_optimized_matrices,
        )
        self.assertEqual(state.state_version, 2)
        self.assertEqual(state.graph.size(), 6)
        self.assertEqual(
            tuple(int(key) for key in state.graph.at(4).keys()),
            (int(camera_symbol(0)), int(camera_symbol(23))),
        )
        self.assertEqual(
            tuple(int(key) for key in state.graph.at(5).keys()),
            (int(camera_symbol(11)), int(camera_symbol(47))),
        )
        self.assertEqual(state.marginal_covariances.shape, (4, 6, 6))
        self.assertTrue(np.all(np.isfinite(state.marginal_covariances)))
        self.assertAlmostEqual(
            second.original_graph_error,
            float(first.graph.error(first.optimized_values)),
        )

    def test_marginals_failure_does_not_mutate_last_valid_state(self):
        state = CumulativePoseGraphState.from_protected_e6(
            _synthetic_protected_state()
        )
        graph_before = state.graph
        values_before = state.optimized_values
        poses_before = state.optimized_pose_matrices.copy()
        marginals_before = state.marginal_covariances.copy()
        measurement = state.initial_relative_pose(0, 47).retract(
            np.asarray(
                [0.001, 0.0, 0.0, -0.1, 0.0, 0.0], dtype=float
            )
        )

        with patch(
            "exercise7_pose_graph_update.gtsam.Marginals",
            side_effect=RuntimeError("forced Marginals failure"),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "forced Marginals failure"
            ):
                state.apply_loop_factor(
                    previous_frame_id=0,
                    current_frame_id=47,
                    relative_pose=measurement,
                    conditional_covariance=np.diag(
                        [
                            5e-4,
                            5e-4,
                            5e-4,
                            0.025,
                            0.025,
                            0.025,
                        ]
                    ),
                )

        self.assertIs(state.graph, graph_before)
        self.assertIs(state.optimized_values, values_before)
        self.assertEqual(state.state_version, 0)
        self.assertEqual(state.graph.size(), 4)
        np.testing.assert_array_equal(
            state.optimized_pose_matrices, poses_before
        )
        np.testing.assert_array_equal(
            state.marginal_covariances, marginals_before
        )
        np.testing.assert_array_equal(
            _pose_matrices(state.optimized_values), poses_before
        )


if __name__ == "__main__":
    unittest.main()
