"""Deterministic synthetic contracts for the bounded E5-I02 Bundle window.

These tests use a tiny in-memory lecturer ``TrackingDB``.  They never open the
canonical pickle and never import the legacy Exercise 5 Bundle modules.
"""

from __future__ import annotations

import json
import unittest
from typing import Dict, Mapping, Sequence, Tuple

import gtsam
import numpy as np

from exercise5_bundle_window import (
    EXCESSIVE_INITIALIZATION_DEPTH,
    INSUFFICIENT_VALID_OBSERVATIONS,
    NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION,
    NO_FINITE_BACKPROJECTION,
    NONFINITE_OBSERVATION,
    NONPOSITIVE_DISPARITY,
    BundleWindowValidationError,
    ProbeLandmarkEligibility,
    solve_bundle_window,
)
from exercise5_geometry import camera_pose_in_window
from tracking_database import Link, TrackingDB


# Every value below is an explicit, noncanonical E5-I02 test choice.
NONCANONICAL_PROJECTION_SIGMA_PX = 1.0
NONCANONICAL_ANCHOR_ROTATION_SIGMA_RAD = 1e-3
NONCANONICAL_ANCHOR_TRANSLATION_SIGMA_M = 1e-3
NONCANONICAL_ROTATION_ATOL = 1e-12
NONCANONICAL_MINIMUM_VALID_OBSERVATIONS = 2
NONCANONICAL_MINIMUM_DISTINCT_FRAMES = 2
SYNTHETIC_MATRIX_ATOL = 1e-10

OBSERVED_FINDINGS: Dict[str, object] = {}


def _calibration() -> gtsam.Cal3_S2Stereo:
    return gtsam.Cal3_S2Stereo(120.0, 118.0, 0.0, 320.0, 240.0, 0.30)


def _world_to_camera_poses(frame_count: int) -> Tuple[np.ndarray, ...]:
    poses = []
    for frame_id in range(frame_count):
        camera_center = np.array(
            [0.25 * frame_id, 0.015 * (frame_id % 2), 0.0]
        )
        poses.append(
            np.column_stack((np.eye(3), -camera_center))
        )
    return tuple(poses)


def _camera_poses(
    world_to_camera_poses: Sequence[np.ndarray],
) -> Tuple[gtsam.Pose3, ...]:
    origin = world_to_camera_poses[0]
    return tuple(
        camera_pose_in_window(
            origin,
            pose,
            rotation_atol=NONCANONICAL_ROTATION_ATOL,
        )
        for pose in world_to_camera_poses
    )


def _measurement(
    camera_pose: gtsam.Pose3,
    calibration: gtsam.Cal3_S2Stereo,
    point: Sequence[float],
) -> np.ndarray:
    return np.asarray(
        gtsam.StereoCamera(camera_pose, calibration)
        .project(np.asarray(point, dtype=float))
        .vector(),
        dtype=float,
    )


def _link(measurement: Sequence[float]) -> Link:
    values = np.asarray(measurement, dtype=float)
    return Link(float(values[0]), float(values[1]), float(values[2]))


def _database_from_links(
    frame_count: int,
    links: Mapping[Tuple[int, int], Link],
) -> TrackingDB:
    database = TrackingDB()
    database.last_frameId = frame_count - 1
    track_ids = sorted({track_id for _, track_id in links})
    database.last_trackId = max(track_ids) if track_ids else -1
    database.trackId_to_frames = {
        track_id: sorted(
            frame_id
            for frame_id, observed_track_id in links
            if observed_track_id == track_id
        )
        for track_id in track_ids
    }
    database.linkId_to_link = dict(links)
    database.frameId_to_trackIds_list = {
        frame_id: sorted(
            track_id
            for observed_frame_id, track_id in links
            if observed_frame_id == frame_id
        )
        for frame_id in range(frame_count)
    }
    return database


def _noise_models() -> Tuple[object, object]:
    projection = gtsam.noiseModel.Isotropic.Sigma(
        3, NONCANONICAL_PROJECTION_SIGMA_PX
    )
    anchor = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            [
                NONCANONICAL_ANCHOR_ROTATION_SIGMA_RAD,
                NONCANONICAL_ANCHOR_ROTATION_SIGMA_RAD,
                NONCANONICAL_ANCHOR_ROTATION_SIGMA_RAD,
                NONCANONICAL_ANCHOR_TRANSLATION_SIGMA_M,
                NONCANONICAL_ANCHOR_TRANSLATION_SIGMA_M,
                NONCANONICAL_ANCHOR_TRANSLATION_SIGMA_M,
            ]
        )
    )
    return projection, anchor


def _eligibility() -> ProbeLandmarkEligibility:
    return ProbeLandmarkEligibility(
        minimum_valid_observations=(
            NONCANONICAL_MINIMUM_VALID_OBSERVATIONS
        ),
        minimum_distinct_frames=(
            NONCANONICAL_MINIMUM_DISTINCT_FRAMES
        ),
    )


def _solve(
    database: TrackingDB,
    world_to_camera_poses: Sequence[np.ndarray],
):
    projection_noise, anchor_noise = _noise_models()
    return solve_bundle_window(
        database=database,
        world_to_camera_poses=world_to_camera_poses,
        calibration=_calibration(),
        start_frame_id=0,
        end_frame_id=len(world_to_camera_poses) - 1,
        projection_noise=projection_noise,
        anchor_noise=anchor_noise,
        landmark_eligibility=_eligibility(),
        rotation_atol=NONCANONICAL_ROTATION_ATOL,
    )


def _main_synthetic_case():
    calibration = _calibration()
    world_to_camera = _world_to_camera_poses(3)
    cameras = _camera_poses(world_to_camera)
    landmarks = {
        7: np.array([-0.9, -0.4, 6.5]),
        11: np.array([0.2, 0.3, 5.5]),
        19: np.array([1.1, -0.2, 7.0]),
        31: np.array([-0.4, 0.8, 8.0]),
        43: np.array([0.7, -0.9, 9.0]),
        59: np.array([1.4, 0.7, 10.0]),
    }
    links: Dict[Tuple[int, int], Link] = {}
    for track_id, point in landmarks.items():
        for frame_id, camera in enumerate(cameras):
            observed = _measurement(camera, calibration, point)
            if (frame_id, track_id) == (0, 11):
                observed = observed + np.array([12.0, -9.0, 6.0])
            links[(frame_id, track_id)] = _link(observed)

    # Track 71 remains accepted from frames 0 and 1; its newest frame is an
    # explicit invalid-disparity observation and must not receive a factor.
    track_71_point = np.array([0.4, 0.45, 7.5])
    links[(0, 71)] = _link(
        _measurement(cameras[0], calibration, track_71_point)
    )
    links[(1, 71)] = _link(
        _measurement(cameras[1], calibration, track_71_point)
    )
    invalid_71 = _measurement(cameras[2], calibration, track_71_point)
    invalid_71[1] = invalid_71[0] + 1.0
    links[(2, 71)] = _link(invalid_71)

    # Tracks 91 and 97 each have only one valid observation after explicit
    # disparity/nonfinite validation and must be rejected for eligibility.
    rejected_point = np.array([-0.2, 0.1, 6.0])
    valid_91 = _measurement(cameras[0], calibration, rejected_point)
    invalid_91 = _measurement(cameras[1], calibration, rejected_point)
    invalid_91[1] = invalid_91[0]
    links[(0, 91)] = _link(valid_91)
    links[(1, 91)] = _link(invalid_91)

    valid_97 = _measurement(cameras[0], calibration, rejected_point)
    invalid_97 = _measurement(cameras[1], calibration, rejected_point)
    invalid_97[2] = np.nan
    links[(0, 97)] = _link(valid_97)
    links[(1, 97)] = _link(invalid_97)

    # A finite, strictly positive but subnormal disparity overflows during
    # backprojection.  It is observation-valid but cannot initialize a finite
    # landmark, which exercises explicit initialization rejection.
    smallest_positive = float(np.nextafter(0.0, 1.0))
    for frame_id in (0, 1):
        links[(frame_id, 101)] = Link(
            smallest_positive, 0.0, 1.0
        )

    database = _database_from_links(3, links)
    result = _solve(database, world_to_camera)
    return database, world_to_camera, result


class SyntheticBundleWindowContractTest(unittest.TestCase):
    """Validate construction, rejection accounting, diagnostics, and LM."""

    @classmethod
    def setUpClass(cls):
        cls.database, cls.world_to_camera, cls.result = (
            _main_synthetic_case()
        )

    def test_identity_prior_key_mappings_and_factor_counts(self):
        result = self.result
        self.assertEqual(result.frame_ids, (0, 1, 2))
        self.assertEqual(
            result.camera_key_by_frame_id,
            {
                frame_id: gtsam.symbol("c", frame_id)
                for frame_id in result.frame_ids
            },
        )
        self.assertEqual(result.track_ids, (7, 11, 19, 31, 43, 59, 71))
        self.assertEqual(
            result.landmark_key_by_track_id,
            {
                track_id: gtsam.symbol("q", track_id)
                for track_id in result.track_ids
            },
        )
        np.testing.assert_array_equal(
            result.initial_values.atPose3(
                result.camera_key_by_frame_id[0]
            ).matrix(),
            np.eye(4),
        )
        for frame_id in result.frame_ids:
            expected_pose = camera_pose_in_window(
                self.world_to_camera[0],
                self.world_to_camera[frame_id],
                rotation_atol=NONCANONICAL_ROTATION_ATOL,
            )
            if frame_id == 0:
                expected_pose = gtsam.Pose3()
            np.testing.assert_allclose(
                result.initial_values.atPose3(
                    result.camera_key_by_frame_id[frame_id]
                ).matrix(),
                expected_pose.matrix(),
                rtol=0.0,
                atol=SYNTHETIC_MATRIX_ATOL,
            )
        self.assertIsInstance(result.graph.at(0), gtsam.PriorFactorPose3)
        self.assertEqual(
            list(result.graph.at(0).keys()),
            [result.camera_key_by_frame_id[0]],
        )
        np.testing.assert_array_equal(
            result.graph.at(0).prior().matrix(), np.eye(4)
        )
        self.assertEqual(result.anchor_factor_count, 1)
        self.assertEqual(result.camera_count, 3)
        self.assertEqual(result.landmark_count, 7)
        self.assertEqual(result.projection_factor_count, 20)
        self.assertEqual(result.total_factor_count, 21)
        self.assertEqual(result.variable_dimension, 39)
        self.assertEqual(result.factor_residual_dimension, 66)
        self.assertEqual(result.graph.size(), 21)
        self.assertEqual(result.initial_values.size(), 10)
        self.assertEqual(result.final_values.size(), 10)

    def test_observations_are_unique_and_invalid_inputs_are_rejected(self):
        result = self.result
        self.assertEqual(result.candidate_track_count, 10)
        self.assertEqual(result.accepted_track_count, 7)
        self.assertEqual(result.rejected_track_count, 3)
        self.assertEqual(result.rejected_track_ids, (91, 97, 101))
        self.assertEqual(
            result.rejection_reason_counts,
            {
                INSUFFICIENT_VALID_OBSERVATIONS: 2,
                NO_FINITE_BACKPROJECTION: 1,
            },
        )
        self.assertEqual(
            result.invalid_observation_reason_counts,
            {
                NONFINITE_OBSERVATION: 1,
                NONPOSITIVE_DISPARITY: 2,
            },
        )
        self.assertEqual(
            result.accepted_observation_count_by_track_id[71], 2
        )
        self.assertEqual(result.initialization_frame_by_track_id[71], 1)
        self.assertEqual(
            len(result.observation_pairs),
            len(set(result.observation_pairs)),
        )
        self.assertNotIn((2, 71), result.observation_pairs)
        for rejected_track_id in result.rejected_track_ids:
            self.assertFalse(
                any(
                    track_id == rejected_track_id
                    for _, track_id in result.observation_pairs
                )
            )
        for record in result.projection_factor_records:
            self.assertEqual(
                list(result.graph.at(record.factor_index).keys()),
                [record.camera_key, record.landmark_key],
            )
            link = self.database.link(record.frame_id, record.track_id)
            expected_measurement = np.array(
                [link.x_left, link.x_right, link.y]
            )
            np.testing.assert_array_equal(
                np.asarray(record.measurement), expected_measurement
            )
            np.testing.assert_array_equal(
                result.graph.at(record.factor_index).measured().vector(),
                expected_measurement,
            )
        expected_pairs = {
            (frame_id, track_id)
            for track_id in (7, 11, 19, 31, 43, 59)
            for frame_id in (0, 1, 2)
        }
        expected_pairs.update({(0, 71), (1, 71)})
        self.assertEqual(set(result.observation_pairs), expected_pairs)

    def test_landmark_initialization_is_finite_positive_and_deterministic(self):
        result = self.result
        record_by_track: Dict[int, list] = {}
        for record in result.projection_factor_records:
            record_by_track.setdefault(record.track_id, []).append(record)
        for track_id in result.track_ids:
            point = np.asarray(
                result.initial_values.atPoint3(
                    result.landmark_key_by_track_id[track_id]
                )
            )
            self.assertTrue(np.all(np.isfinite(point)))
            self.assertEqual(
                result.initialization_frame_by_track_id[track_id],
                max(record.frame_id for record in record_by_track[track_id]),
            )
            initialization_frame_id = (
                result.initialization_frame_by_track_id[track_id]
            )
            link = self.database.link(initialization_frame_id, track_id)
            independently_backprojected = np.asarray(
                gtsam.StereoCamera(
                    result.initial_values.atPose3(
                        result.camera_key_by_frame_id[
                            initialization_frame_id
                        ]
                    ),
                    _calibration(),
                ).backproject(
                    gtsam.StereoPoint2(
                        link.x_left, link.x_right, link.y
                    )
                )
            )
            np.testing.assert_allclose(
                point,
                independently_backprojected,
                rtol=0.0,
                atol=SYNTHETIC_MATRIX_ATOL,
            )
            for record in record_by_track[track_id]:
                pose = result.initial_values.atPose3(record.camera_key)
                camera_point = np.asarray(pose.transformTo(point))
                self.assertGreater(float(camera_point[2]), 0.0)

        repeated = _solve(self.database, self.world_to_camera)
        self.assertEqual(
            repeated.camera_key_by_frame_id,
            result.camera_key_by_frame_id,
        )
        self.assertEqual(
            repeated.landmark_key_by_track_id,
            result.landmark_key_by_track_id,
        )
        self.assertEqual(repeated.observation_pairs, result.observation_pairs)
        self.assertEqual(
            repeated.initialization_frame_by_track_id,
            result.initialization_frame_by_track_id,
        )
        self.assertAlmostEqual(
            repeated.initial_errors.total_graph_error,
            result.initial_errors.total_graph_error,
            delta=SYNTHETIC_MATRIX_ATOL,
        )
        self.assertAlmostEqual(
            repeated.final_errors.total_graph_error,
            result.final_errors.total_graph_error,
            delta=SYNTHETIC_MATRIX_ATOL,
        )
        self.assertEqual(
            repeated.optimizer_iterations, result.optimizer_iterations
        )
        for frame_id in result.frame_ids:
            np.testing.assert_array_equal(
                repeated.final_values.atPose3(
                    repeated.camera_key_by_frame_id[frame_id]
                ).matrix(),
                result.final_values.atPose3(
                    result.camera_key_by_frame_id[frame_id]
                ).matrix(),
            )
        for track_id in result.track_ids:
            np.testing.assert_array_equal(
                repeated.final_values.atPoint3(
                    repeated.landmark_key_by_track_id[track_id]
                ),
                result.final_values.atPoint3(
                    result.landmark_key_by_track_id[track_id]
                ),
            )

    def test_raw_errors_are_finite_and_optimization_does_not_increase(self):
        result = self.result
        observed = []
        for metrics in (result.initial_errors, result.final_errors):
            observed.extend(
                [
                    metrics.total_graph_error,
                    metrics.mean_all_factor_error,
                    metrics.projection_factor_error_sum,
                    metrics.mean_projection_factor_error,
                    metrics.anchor_factor_error,
                    metrics.graph_vs_factor_sum_absolute_difference,
                ]
            )
        self.assertTrue(np.all(np.isfinite(np.asarray(observed))))
        self.assertLessEqual(
            result.final_errors.total_graph_error,
            result.initial_errors.total_graph_error,
        )
        self.assertLess(
            result.final_errors.anchor_factor_error,
            SYNTHETIC_MATRIX_ATOL,
        )
        for metrics in (result.initial_errors, result.final_errors):
            self.assertAlmostEqual(
                metrics.mean_all_factor_error,
                metrics.total_graph_error / result.total_factor_count,
                delta=SYNTHETIC_MATRIX_ATOL,
            )
            self.assertAlmostEqual(
                metrics.mean_projection_factor_error,
                metrics.projection_factor_error_sum
                / result.projection_factor_count,
                delta=SYNTHETIC_MATRIX_ATOL,
            )
            self.assertLess(
                metrics.graph_vs_factor_sum_absolute_difference,
                SYNTHETIC_MATRIX_ATOL,
            )
        self.assertTrue(result.connected_to_anchor)
        self.assertTrue(
            all(degree > 0 for degree in result.camera_factor_degrees.values())
        )
        self.assertTrue(
            all(
                degree >= NONCANONICAL_MINIMUM_VALID_OBSERVATIONS
                for degree in result.landmark_factor_degrees.values()
            )
        )
        json.dumps(
            result.serializable_summary(),
            sort_keys=True,
            allow_nan=False,
        )

    def test_maximum_initial_error_diagnostic_identifies_known_factor(self):
        result = self.result
        diagnostic = result.maximum_initial_projection_error
        self.assertEqual((diagnostic.frame_id, diagnostic.track_id), (0, 11))
        link = self.database.link(0, 11)
        expected_measurement = np.array(
            [link.x_left, link.x_right, link.y]
        )
        np.testing.assert_array_equal(
            np.asarray(diagnostic.measurement), expected_measurement
        )
        np.testing.assert_allclose(
            np.asarray(diagnostic.initial_residual),
            np.asarray(diagnostic.initial_projection)
            - expected_measurement,
            rtol=0.0,
            atol=SYNTHETIC_MATRIX_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(diagnostic.final_residual),
            np.asarray(diagnostic.final_projection)
            - expected_measurement,
            rtol=0.0,
            atol=SYNTHETIC_MATRIX_ATOL,
        )
        factor = result.graph.at(diagnostic.factor_index)
        self.assertAlmostEqual(
            diagnostic.initial_factor_error,
            float(factor.error(result.initial_values)),
            delta=SYNTHETIC_MATRIX_ATOL,
        )
        self.assertAlmostEqual(
            diagnostic.final_factor_error,
            float(factor.error(result.final_values)),
            delta=SYNTHETIC_MATRIX_ATOL,
        )
        initial_projection_errors = [
            float(
                result.graph.at(record.factor_index).error(
                    result.initial_values
                )
            )
            for record in result.projection_factor_records
        ]
        self.assertAlmostEqual(
            diagnostic.initial_factor_error,
            max(initial_projection_errors),
            delta=SYNTHETIC_MATRIX_ATOL,
        )

    def test_disconnected_camera_landmark_components_stop_before_lm(self):
        calibration = _calibration()
        world_to_camera = _world_to_camera_poses(4)
        cameras = _camera_poses(world_to_camera)
        links: Dict[Tuple[int, int], Link] = {}
        components = {
            201: ((0, 1), np.array([0.2, 0.1, 6.0])),
            203: ((2, 3), np.array([-0.4, 0.3, 7.0])),
        }
        for track_id, (frame_ids, point) in components.items():
            for frame_id in frame_ids:
                links[(frame_id, track_id)] = _link(
                    _measurement(cameras[frame_id], calibration, point)
                )
        database = _database_from_links(4, links)
        with self.assertRaisesRegex(
            BundleWindowValidationError, "disconnected"
        ):
            _solve(database, world_to_camera)

    def test_latest_depth_failure_falls_back_and_full_failure_is_rejected(self):
        calibration = _calibration()
        first_extrinsic = np.column_stack((np.eye(3), np.zeros(3)))
        second_rotation = np.diag([-1.0, 1.0, -1.0])
        second_center = np.array([0.0, 0.0, 10.0])
        second_extrinsic = np.column_stack(
            (second_rotation, -second_rotation @ second_center)
        )
        world_to_camera = (first_extrinsic, second_extrinsic)
        cameras = _camera_poses(world_to_camera)
        links: Dict[Tuple[int, int], Link] = {}

        shared_landmarks = {
            301: np.array([-0.8, -0.4, 4.0]),
            303: np.array([0.9, -0.2, 4.8]),
            307: np.array([-0.3, 0.8, 5.5]),
            311: np.array([0.7, 0.6, 6.2]),
            313: np.array([0.1, -0.7, 5.0]),
            317: np.array([-1.0, 0.4, 6.8]),
        }
        for track_id, point in shared_landmarks.items():
            for frame_id, camera in enumerate(cameras):
                links[(frame_id, track_id)] = _link(
                    _measurement(camera, calibration, point)
                )

        fallback_track_id = 331
        qualifying_point = np.array([0.2, 0.1, 5.0])
        links[(0, fallback_track_id)] = _link(
            _measurement(cameras[0], calibration, qualifying_point)
        )
        latest_behind_first_camera = _measurement(
            gtsam.Pose3(), calibration, np.array([0.2, 0.1, 11.0])
        )
        links[(1, fallback_track_id)] = _link(
            latest_behind_first_camera
        )

        rejected_track_id = 337
        depth_past_second_camera = np.array([0.1, 0.1, 11.0])
        links[(0, rejected_track_id)] = _link(
            _measurement(
                gtsam.Pose3(), calibration, depth_past_second_camera
            )
        )
        links[(1, rejected_track_id)] = _link(
            _measurement(
                gtsam.Pose3(), calibration, depth_past_second_camera
            )
        )

        database = _database_from_links(2, links)
        result = _solve(database, world_to_camera)
        self.assertEqual(
            result.initialization_frame_by_track_id[fallback_track_id], 0
        )
        self.assertEqual(
            result.rejected_track_reason_by_id[rejected_track_id],
            NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION,
        )
        self.assertEqual(
            result.rejection_reason_counts[
                NO_ALL_CAMERA_POSITIVE_DEPTH_INITIALIZATION
            ],
            1,
        )

    def test_course_initialization_depth_gate_is_strict_and_recorded(self):
        calibration = gtsam.Cal3_S2Stereo(
            120.0, 118.0, 0.0, 0.0, 0.0, 0.30
        )
        world_to_camera = _world_to_camera_poses(3)
        cameras = _camera_poses(world_to_camera)
        links: Dict[Tuple[int, int], Link] = {}
        shared_landmarks = {
            501: np.array([-0.8, -0.4, 4.0]),
            503: np.array([0.9, -0.2, 4.8]),
            507: np.array([-0.3, 0.8, 5.5]),
            511: np.array([0.7, 0.6, 6.2]),
            513: np.array([0.1, -0.7, 5.0]),
            517: np.array([-1.0, 0.4, 6.8]),
            521: np.array([0.0, 0.0, 299.0]),
            523: np.array([0.0, 0.0, 300.0]),
            527: np.array([0.0, 0.0, 301.0]),
        }
        for track_id, point in shared_landmarks.items():
            for frame_id, camera in enumerate(cameras):
                links[(frame_id, track_id)] = _link(
                    _measurement(camera, calibration, point)
                )
        database = _database_from_links(3, links)
        projection_noise, anchor_noise = _noise_models()
        result = solve_bundle_window(
            database=database,
            world_to_camera_poses=world_to_camera,
            calibration=calibration,
            start_frame_id=0,
            end_frame_id=2,
            projection_noise=projection_noise,
            anchor_noise=anchor_noise,
            landmark_eligibility=ProbeLandmarkEligibility(
                minimum_valid_observations=2,
                minimum_distinct_frames=2,
                maximum_initialization_depth_m=300.0,
            ),
            rotation_atol=NONCANONICAL_ROTATION_ATOL,
        )
        self.assertIn(521, result.accepted_track_ids)
        self.assertEqual(
            result.initialization_depth_by_track_id[521], 299.0
        )
        self.assertEqual(
            result.rejected_track_reason_by_id[523],
            EXCESSIVE_INITIALIZATION_DEPTH,
        )
        self.assertEqual(
            result.rejected_track_reason_by_id[527],
            EXCESSIVE_INITIALIZATION_DEPTH,
        )
        self.assertEqual(
            result.rejection_reason_counts[
                EXCESSIVE_INITIALIZATION_DEPTH
            ],
            2,
        )
        self.assertAlmostEqual(
            result.excessive_depth_m_by_track_id[523], 300.0
        )
        self.assertAlmostEqual(
            result.excessive_depth_m_by_track_id[527], 301.0
        )
        self.assertEqual(
            result.accepted_observation_count_by_track_id[521], 3
        )
        self.assertEqual(result.initialization_frame_by_track_id[521], 2)
        self.assertNotIn(523, result.landmark_key_by_track_id)
        self.assertNotIn(527, result.landmark_key_by_track_id)

    def test_noise_dimensions_are_validated_before_factor_evaluation(self):
        projection_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        _, anchor_noise = _noise_models()
        with self.assertRaisesRegex(ValueError, "projection_noise"):
            solve_bundle_window(
                database=self.database,
                world_to_camera_poses=self.world_to_camera,
                calibration=_calibration(),
                start_frame_id=0,
                end_frame_id=2,
                projection_noise=projection_noise,
                anchor_noise=anchor_noise,
                landmark_eligibility=_eligibility(),
                rotation_atol=NONCANONICAL_ROTATION_ATOL,
            )
        projection_noise, anchor_noise = _noise_models()
        skewed_calibration = gtsam.Cal3_S2Stereo(
            120.0, 118.0, 2.0, 320.0, 240.0, 0.30
        )
        with self.assertRaisesRegex(ValueError, "ignores nonzero"):
            solve_bundle_window(
                database=self.database,
                world_to_camera_poses=self.world_to_camera,
                calibration=skewed_calibration,
                start_frame_id=0,
                end_frame_id=2,
                projection_noise=projection_noise,
                anchor_noise=anchor_noise,
                landmark_eligibility=_eligibility(),
                rotation_atol=NONCANONICAL_ROTATION_ATOL,
            )

    def test_connected_but_dimensionally_underconstrained_graph_stops(self):
        calibration = _calibration()
        world_to_camera = _world_to_camera_poses(2)
        cameras = _camera_poses(world_to_camera)
        point = np.array([0.2, 0.1, 6.0])
        links = {
            (frame_id, 401): _link(
                _measurement(camera, calibration, point)
            )
            for frame_id, camera in enumerate(cameras)
        }
        database = _database_from_links(2, links)
        with self.assertRaisesRegex(
            BundleWindowValidationError, "underconstrained"
        ):
            _solve(database, world_to_camera)

    @classmethod
    def tearDownClass(cls):
        result = cls.result
        diagnostic = result.maximum_initial_projection_error
        OBSERVED_FINDINGS.update(
            {
                "configuration_status": (
                    "noncanonical deterministic E5-I02 synthetic probe"
                ),
                "candidate_tracks": result.candidate_track_count,
                "accepted_tracks": result.accepted_track_count,
                "rejected_tracks": result.rejected_track_count,
                "rejection_reason_counts": dict(
                    result.rejection_reason_counts
                ),
                "invalid_observation_reason_counts": dict(
                    result.invalid_observation_reason_counts
                ),
                "cameras": result.camera_count,
                "landmarks": result.landmark_count,
                "projection_factors": result.projection_factor_count,
                "total_factors": result.total_factor_count,
                "initial_total_graph_error": (
                    result.initial_errors.total_graph_error
                ),
                "final_total_graph_error": (
                    result.final_errors.total_graph_error
                ),
                "initial_mean_all_factor_error": (
                    result.initial_errors.mean_all_factor_error
                ),
                "final_mean_all_factor_error": (
                    result.final_errors.mean_all_factor_error
                ),
                "initial_mean_projection_factor_error": (
                    result.initial_errors.mean_projection_factor_error
                ),
                "final_mean_projection_factor_error": (
                    result.final_errors.mean_projection_factor_error
                ),
                "initial_anchor_error": (
                    result.initial_errors.anchor_factor_error
                ),
                "final_anchor_error": (
                    result.final_errors.anchor_factor_error
                ),
                "maximum_initial_projection_factor": {
                    "frame_id": diagnostic.frame_id,
                    "track_id": diagnostic.track_id,
                    "factor_error": diagnostic.initial_factor_error,
                    "measurement": list(diagnostic.measurement),
                    "initial_projection": list(
                        diagnostic.initial_projection
                    ),
                    "initial_residual": list(diagnostic.initial_residual),
                    "final_projection": list(diagnostic.final_projection),
                    "final_residual": list(diagnostic.final_residual),
                },
                "optimizer_iterations": result.optimizer_iterations,
                "connected_to_anchor": result.connected_to_anchor,
            }
        )


def tearDownModule():
    print("\nE5-I02 synthetic observed findings:")
    print(
        json.dumps(
            OBSERVED_FINDINGS,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    unittest.main()
