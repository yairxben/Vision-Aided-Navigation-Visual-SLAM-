"""Bounded executable contracts for Exercise 5 geometry and canonical input.

These tests intentionally do not import the legacy Bundle modules, construct
an optimization graph, or open the canonical TrackingDB pickle.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path
from typing import Dict, Sequence, Tuple

import gtsam
import numpy as np

from exercise5_geometry import (
    camera_point_to_window,
    camera_point_to_world,
    camera_pose_from_world_to_camera,
    camera_pose_in_window,
    validate_world_to_camera_extrinsic,
    window_point_to_camera,
    world_point_to_camera,
    world_to_camera_from_camera_pose,
)


ROOT = Path(__file__).resolve().parent
CALIBRATION_PATH = ROOT / "dataset" / "sequences" / "00" / "calib.txt"
FINAL_EVIDENCE = ROOT / "diagnostics" / "ex4_final"
COMPANION_PATH = FINAL_EVIDENCE / "tracking_database_companion.json"
MANIFEST_PATH = FINAL_EVIDENCE / "manifest.json"
COMPLETE_PATH = FINAL_EVIDENCE / "COMPLETE.json"
VALIDATION_PATH = FINAL_EVIDENCE / "validation.json"
CHECKSUMS_PATH = FINAL_EVIDENCE / "SHA256SUMS"
I04_VALIDATION_PATH = ROOT / "diagnostics" / "ex4_i04" / "validation.json"

# These thresholds are local test gates, not Exercise 5 production choices.
SYNTHETIC_ROTATION_ATOL = 1e-12
CANONICAL_ROTATION_ATOL = 1e-9
TEST_MATRIX_ATOL = 1e-11
TEST_ONLY_FACTOR_SIGMA = 2.5

OBSERVED_FINDINGS: Dict[str, object] = {}


def _record(name: str, value: object) -> None:
    """Retain a deterministic observed result for the bounded test report."""

    if isinstance(value, np.generic):
        value = value.item()
    OBSERVED_FINDINGS[name] = value


def _maximum_absolute_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return a scalar maximum absolute difference."""

    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _rodrigues(rotation_vector: Sequence[float]) -> np.ndarray:
    """Construct a test rotation independently from GTSAM."""

    vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(vector))
    axis = vector / angle
    x, y, z = axis
    cross_matrix = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * cross_matrix
        + (1.0 - math.cos(angle)) * (cross_matrix @ cross_matrix)
    )


def _extrinsic(
    rotation_vector: Sequence[float], translation: Sequence[float]
) -> np.ndarray:
    """Build a nontrivial synthetic world-to-camera extrinsic."""

    return np.column_stack(
        (_rodrigues(rotation_vector), np.asarray(translation, dtype=float))
    )


def _homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    """Embed a 3x4 transform in a homogeneous matrix for test algebra."""

    matrix = np.eye(4)
    matrix[:3, :] = extrinsic
    return matrix


def _read_json(path: Path) -> Dict[str, object]:
    """Read a bounded JSON evidence file."""

    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    """Hash a small contract input; never called with the TrackingDB path."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_calibration() -> Tuple[np.ndarray, np.ndarray]:
    """Read the canonical KITTI P0/P1 stereo projection matrices."""

    rows = {}
    for line in CALIBRATION_PATH.read_text(encoding="utf-8").splitlines():
        label, values = line.split(":", 1)
        rows[label] = np.asarray(values.split(), dtype=float).reshape(3, 4)
    return rows["P0"], rows["P1"]


def _gtsam_stereo_calibration() -> Tuple[
    gtsam.Cal3_S2Stereo, np.ndarray, np.ndarray, np.ndarray
]:
    """Construct the pinned GTSAM stereo calibration directly from P0/P1."""

    projection_left, projection_right = _read_calibration()
    intrinsic = projection_left[:, :3]
    left_extrinsic = np.linalg.solve(intrinsic, projection_left)
    right_extrinsic = np.linalg.solve(intrinsic, projection_right)
    baseline = -float(right_extrinsic[0, 3])
    calibration = gtsam.Cal3_S2Stereo(
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
        baseline,
    )
    return calibration, intrinsic, left_extrinsic, right_extrinsic


class WorldToCameraValidationTest(unittest.TestCase):
    """Validate the pure 3x4 rigid-transform boundary."""

    def test_valid_extrinsic_is_copied_and_invalid_inputs_are_rejected(self):
        valid = _extrinsic((0.31, -0.22, 0.17), (1.2, -0.7, 2.3))
        validated = validate_world_to_camera_extrinsic(
            valid, rotation_atol=SYNTHETIC_ROTATION_ATOL
        )
        np.testing.assert_array_equal(validated, valid)
        self.assertFalse(np.shares_memory(validated, valid))

        with self.assertRaises(ValueError):
            validate_world_to_camera_extrinsic(
                np.eye(4), rotation_atol=SYNTHETIC_ROTATION_ATOL
            )

        nonfinite = valid.copy()
        nonfinite[0, 3] = np.nan
        with self.assertRaises(ValueError):
            validate_world_to_camera_extrinsic(
                nonfinite, rotation_atol=SYNTHETIC_ROTATION_ATOL
            )

        nonorthonormal = valid.copy()
        nonorthonormal[0, 0] += 1e-3
        with self.assertRaises(ValueError):
            validate_world_to_camera_extrinsic(
                nonorthonormal, rotation_atol=SYNTHETIC_ROTATION_ATOL
            )

        reflection = valid.copy()
        reflection[:, 0] *= -1.0
        with self.assertRaises(ValueError):
            validate_world_to_camera_extrinsic(
                reflection, rotation_atol=SYNTHETIC_ROTATION_ATOL
            )

        for invalid_tolerance in (0.0, -1.0, np.inf, np.nan):
            with self.assertRaises(ValueError):
                validate_world_to_camera_extrinsic(
                    valid, rotation_atol=invalid_tolerance
                )


class GtsamPose3DirectionContractTest(unittest.TestCase):
    """Exercise nontrivial Pose3 directions and matrix equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.extrinsic_a = _extrinsic(
            (0.31, -0.22, 0.17), (1.2, -0.7, 2.3)
        )
        cls.extrinsic_i = _extrinsic(
            (-0.19, 0.27, 0.41), (-0.4, 1.1, 0.6)
        )
        cls.pose_a = camera_pose_from_world_to_camera(
            cls.extrinsic_a, rotation_atol=SYNTHETIC_ROTATION_ATOL
        )
        cls.pose_i = camera_pose_from_world_to_camera(
            cls.extrinsic_i, rotation_atol=SYNTHETIC_ROTATION_ATOL
        )
        cls.world_point = np.array([4.2, -1.3, 8.7])

    def test_transform_from_and_transform_to_directions(self):
        expected_camera = (
            self.extrinsic_i[:, :3] @ self.world_point
            + self.extrinsic_i[:, 3]
        )
        transformed_to = np.asarray(
            self.pose_i.transformTo(self.world_point)
        )
        transformed_from = np.asarray(
            self.pose_i.transformFrom(expected_camera)
        )

        transform_to_error = float(
            np.linalg.norm(transformed_to - expected_camera)
        )
        transform_from_error = float(
            np.linalg.norm(transformed_from - self.world_point)
        )
        _record("pose3_transform_to_l2_error", transform_to_error)
        _record("pose3_transform_from_l2_error", transform_from_error)
        self.assertLess(transform_to_error, TEST_MATRIX_ATOL)
        self.assertLess(transform_from_error, TEST_MATRIX_ATOL)

    def test_inverse_compose_and_between_match_matrix_algebra(self):
        homogeneous_a = _homogeneous(self.extrinsic_a)
        homogeneous_i = _homogeneous(self.extrinsic_i)

        inverse_error = _maximum_absolute_error(
            self.pose_i.inverse().matrix(), homogeneous_i
        )
        compose_error = _maximum_absolute_error(
            self.pose_a.compose(self.pose_i).matrix(),
            np.linalg.inv(homogeneous_a) @ np.linalg.inv(homogeneous_i),
        )
        between_pose = self.pose_a.between(self.pose_i)
        between_expected = homogeneous_a @ np.linalg.inv(homogeneous_i)
        between_error = _maximum_absolute_error(
            between_pose.matrix(), between_expected
        )
        recovery_error = _maximum_absolute_error(
            self.pose_a.compose(between_pose).matrix(),
            self.pose_i.matrix(),
        )
        reverse_between_error = _maximum_absolute_error(
            self.pose_i.between(self.pose_a).matrix(),
            between_pose.inverse().matrix(),
        )

        _record("pose3_inverse_matrix_max_abs_error", inverse_error)
        _record("pose3_compose_matrix_max_abs_error", compose_error)
        _record("pose3_between_matrix_max_abs_error", between_error)
        _record("pose3_compose_recovery_max_abs_error", recovery_error)
        _record(
            "pose3_reverse_between_max_abs_error", reverse_between_error
        )
        for error in (
            inverse_error,
            compose_error,
            between_error,
            recovery_error,
            reverse_between_error,
        ):
            self.assertLess(error, TEST_MATRIX_ATOL)

    def test_pose_conversion_window_formula_and_anchor_identity(self):
        expected_pose_i = np.linalg.inv(_homogeneous(self.extrinsic_i))
        pose_conversion_error = _maximum_absolute_error(
            self.pose_i.matrix(), expected_pose_i
        )
        extrinsic_round_trip = world_to_camera_from_camera_pose(
            self.pose_i, rotation_atol=SYNTHETIC_ROTATION_ATOL
        )
        extrinsic_round_trip_error = _maximum_absolute_error(
            extrinsic_round_trip, self.extrinsic_i
        )

        window_pose = camera_pose_in_window(
            self.extrinsic_a,
            self.extrinsic_i,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        expected_window = (
            _homogeneous(self.extrinsic_a)
            @ np.linalg.inv(_homogeneous(self.extrinsic_i))
        )
        window_formula_error = _maximum_absolute_error(
            window_pose.matrix(), expected_window
        )

        rotation_a = self.extrinsic_a[:, :3]
        rotation_i = self.extrinsic_i[:, :3]
        translation_a = self.extrinsic_a[:, 3]
        translation_i = self.extrinsic_i[:, 3]
        expanded_window = np.eye(4)
        expanded_window[:3, :3] = rotation_a @ rotation_i.T
        expanded_window[:3, 3] = (
            translation_a
            - rotation_a @ rotation_i.T @ translation_i
        )
        expanded_formula_error = _maximum_absolute_error(
            window_pose.matrix(), expanded_window
        )

        anchor_pose = camera_pose_in_window(
            self.extrinsic_a,
            self.extrinsic_a,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        anchor_identity_error = _maximum_absolute_error(
            anchor_pose.matrix(), np.eye(4)
        )

        _record(
            "world_to_camera_pose_conversion_max_abs_error",
            pose_conversion_error,
        )
        _record(
            "world_to_camera_pose_round_trip_max_abs_error",
            extrinsic_round_trip_error,
        )
        _record("window_pose_formula_max_abs_error", window_formula_error)
        _record(
            "window_pose_expanded_formula_max_abs_error",
            expanded_formula_error,
        )
        _record(
            "window_anchor_identity_max_abs_error", anchor_identity_error
        )
        for error in (
            pose_conversion_error,
            extrinsic_round_trip_error,
            window_formula_error,
            expanded_formula_error,
            anchor_identity_error,
        ):
            self.assertLess(error, TEST_MATRIX_ATOL)

    def test_world_camera_and_window_point_paths_are_consistent(self):
        point_i = world_point_to_camera(
            self.extrinsic_i,
            self.world_point,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        expected_window_point = world_point_to_camera(
            self.extrinsic_a,
            self.world_point,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        window_pose = camera_pose_in_window(
            self.extrinsic_a,
            self.extrinsic_i,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        point_i_through_window = camera_point_to_window(
            window_pose,
            point_i,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        recovered_point_i = window_point_to_camera(
            window_pose,
            expected_window_point,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )
        recovered_world = camera_point_to_world(
            self.extrinsic_i,
            point_i,
            rotation_atol=SYNTHETIC_ROTATION_ATOL,
        )

        window_path_error = float(
            np.linalg.norm(point_i_through_window - expected_window_point)
        )
        reverse_window_path_error = float(
            np.linalg.norm(recovered_point_i - point_i)
        )
        world_round_trip_error = float(
            np.linalg.norm(recovered_world - self.world_point)
        )
        _record("world_camera_window_path_l2_error", window_path_error)
        _record(
            "window_camera_reverse_path_l2_error",
            reverse_window_path_error,
        )
        _record("world_camera_point_round_trip_l2_error", world_round_trip_error)
        for error in (
            window_path_error,
            reverse_window_path_error,
            world_round_trip_error,
        ):
            self.assertLess(error, TEST_MATRIX_ATOL)

    def test_gtsam_42_raw_pose_matrix_constructor_is_permissive(self):
        pose_from_three_by_four = gtsam.Pose3(self.extrinsic_i)

        zero_bottom_row = np.zeros((4, 4))
        zero_bottom_row[:3, :] = self.extrinsic_i
        pose_from_zero_bottom_row = gtsam.Pose3(zero_bottom_row)
        np.testing.assert_array_equal(
            pose_from_zero_bottom_row.matrix()[3], np.array([0, 0, 0, 1])
        )

        invalid_rotation_matrix = np.eye(4)
        invalid_rotation_matrix[0, 0] = 1.1
        invalid_pose = gtsam.Pose3(invalid_rotation_matrix)
        retained_invalid_rotation_error = float(
            np.max(
                np.abs(
                    invalid_pose.rotation().matrix().T
                    @ invalid_pose.rotation().matrix()
                    - np.eye(3)
                )
            )
        )
        self.assertGreater(retained_invalid_rotation_error, 0.1)
        with self.assertRaises(ValueError):
            world_to_camera_from_camera_pose(
                invalid_pose, rotation_atol=SYNTHETIC_ROTATION_ATOL
            )

        _record(
            "gtsam_42_pose3_accepts_3x4_matrix",
            pose_from_three_by_four.matrix().shape == (4, 4),
        )
        _record("gtsam_42_pose3_accepts_zero_bottom_row", True)
        _record(
            "gtsam_42_retained_invalid_rotation_orthogonality_error",
            retained_invalid_rotation_error,
        )


class StereoCalibrationContractTest(unittest.TestCase):
    """Validate canonical KITTI P0/P1 calibration against GTSAM stereo."""

    @classmethod
    def setUpClass(cls):
        (
            cls.calibration,
            cls.intrinsic,
            cls.left_extrinsic,
            cls.right_extrinsic,
        ) = _gtsam_stereo_calibration()
        cls.projection_left, cls.projection_right = _read_calibration()
        cls.points = np.array(
            [
                [1.2, -0.3, 8.0],
                [-2.1, 0.8, 15.5],
                [0.25, 1.1, 3.7],
            ]
        )

    def test_intrinsics_baseline_and_stereo_measurement_order(self):
        np.testing.assert_allclose(
            self.left_extrinsic,
            np.column_stack((np.eye(3), np.zeros(3))),
            rtol=0.0,
            atol=TEST_MATRIX_ATOL,
        )
        np.testing.assert_allclose(
            self.right_extrinsic[:, :3],
            np.eye(3),
            rtol=0.0,
            atol=TEST_MATRIX_ATOL,
        )

        expected_fields = np.array(
            [
                self.intrinsic[0, 0],
                self.intrinsic[1, 1],
                self.intrinsic[0, 1],
                self.intrinsic[0, 2],
                self.intrinsic[1, 2],
                -self.right_extrinsic[0, 3],
            ]
        )
        observed_fields = np.array(
            [
                self.calibration.fx(),
                self.calibration.fy(),
                self.calibration.skew(),
                self.calibration.px(),
                self.calibration.py(),
                self.calibration.baseline(),
            ]
        )
        np.testing.assert_array_equal(observed_fields, expected_fields)

        baseline = float(self.calibration.baseline())
        self.assertGreater(baseline, 0.0)
        self.assertAlmostEqual(
            float(self.projection_right[0, 3]),
            -float(self.calibration.fx()) * baseline,
            delta=TEST_MATRIX_ATOL,
        )
        right_camera_center = (
            -self.right_extrinsic[:, :3].T @ self.right_extrinsic[:, 3]
        )
        np.testing.assert_allclose(
            right_camera_center,
            np.array([baseline, 0.0, 0.0]),
            rtol=0.0,
            atol=TEST_MATRIX_ATOL,
        )
        self.assertEqual(
            gtsam.StereoCamera(gtsam.Pose3(), self.calibration).baseline(),
            baseline,
        )

        stereo_point = gtsam.StereoPoint2(101.25, 62.5, 45.75)
        np.testing.assert_array_equal(
            stereo_point.vector(), np.array([101.25, 62.5, 45.75])
        )
        self.assertEqual(stereo_point.uL(), 101.25)
        self.assertEqual(stereo_point.uR(), 62.5)
        self.assertEqual(stereo_point.v(), 45.75)

        negative_calibration = gtsam.Cal3_S2Stereo(
            self.calibration.fx(),
            self.calibration.fy(),
            self.calibration.skew(),
            self.calibration.px(),
            self.calibration.py(),
            -baseline,
        )
        negative_measurement = gtsam.StereoCamera(
            gtsam.Pose3(), negative_calibration
        ).project(self.points[0])
        negative_disparity = float(
            negative_measurement.uL() - negative_measurement.uR()
        )
        self.assertLess(negative_disparity, 0.0)

        _record(
            "calibration",
            {
                "fx": float(self.calibration.fx()),
                "fy": float(self.calibration.fy()),
                "cx": float(self.calibration.px()),
                "cy": float(self.calibration.py()),
                "skew": float(self.calibration.skew()),
                "right_extrinsic_tx_m": float(self.right_extrinsic[0, 3]),
                "gtsam_baseline_m": baseline,
                "negative_baseline_disparity_px": negative_disparity,
            },
        )

    def test_project_backproject_and_explicit_projection_equations(self):
        camera = gtsam.StereoCamera(gtsam.Pose3(), self.calibration)
        projection_errors = []
        projection_matrix_errors = []
        disparity_errors = []
        round_trip_errors = []

        for point in self.points:
            measurement = camera.project(point)
            observed = np.asarray(measurement.vector())
            x, y, z = point
            expected = np.array(
                [
                    self.calibration.fx() * x / z + self.calibration.px(),
                    self.calibration.fx()
                    * (x - self.calibration.baseline())
                    / z
                    + self.calibration.px(),
                    self.calibration.fy() * y / z + self.calibration.py(),
                ]
            )

            homogeneous_point = np.append(point, 1.0)
            left_h = self.projection_left @ homogeneous_point
            right_h = self.projection_right @ homogeneous_point
            from_projection_matrices = np.array(
                [
                    left_h[0] / left_h[2],
                    right_h[0] / right_h[2],
                    left_h[1] / left_h[2],
                ]
            )

            projection_errors.append(
                _maximum_absolute_error(observed, expected)
            )
            projection_matrix_errors.append(
                _maximum_absolute_error(observed, from_projection_matrices)
            )
            expected_disparity = (
                self.calibration.fx() * self.calibration.baseline() / z
            )
            disparity_errors.append(
                abs(float(observed[0] - observed[1]) - expected_disparity)
            )
            self.assertGreater(float(observed[0] - observed[1]), 0.0)
            backprojected = np.asarray(camera.backproject(measurement))
            round_trip_errors.append(
                float(np.linalg.norm(backprojected - point))
            )

        maximum_projection_error = max(projection_errors)
        maximum_matrix_error = max(projection_matrix_errors)
        maximum_disparity_error = max(disparity_errors)
        maximum_round_trip_error = max(round_trip_errors)
        _record(
            "stereo_explicit_projection_max_abs_error_px",
            maximum_projection_error,
        )
        _record(
            "stereo_projection_matrix_max_abs_error_px",
            maximum_matrix_error,
        )
        _record(
            "stereo_disparity_formula_max_abs_error_px",
            maximum_disparity_error,
        )
        _record(
            "stereo_project_backproject_max_l2_error_m",
            maximum_round_trip_error,
        )
        for error in (
            maximum_projection_error,
            maximum_matrix_error,
            maximum_disparity_error,
            maximum_round_trip_error,
        ):
            self.assertLess(error, TEST_MATRIX_ATOL)

    def test_gtsam_42_stereo_projection_ignores_nonzero_skew(self):
        noncanonical_skew = 13.75
        skewed_calibration = gtsam.Cal3_S2Stereo(
            self.calibration.fx(),
            self.calibration.fy(),
            noncanonical_skew,
            self.calibration.px(),
            self.calibration.py(),
            self.calibration.baseline(),
        )
        point = self.points[0]
        observed = np.asarray(
            gtsam.StereoCamera(
                gtsam.Pose3(), skewed_calibration
            ).project(point).vector()
        )
        rectified_u_left = (
            skewed_calibration.fx() * point[0] / point[2]
            + skewed_calibration.px()
        )
        full_intrinsic_u_left = (
            rectified_u_left
            + noncanonical_skew * point[1] / point[2]
        )
        self.assertAlmostEqual(
            float(observed[0]), rectified_u_left, delta=TEST_MATRIX_ATOL
        )
        ignored_skew_difference = abs(
            float(observed[0]) - full_intrinsic_u_left
        )
        self.assertGreater(ignored_skew_difference, 0.0)
        _record(
            "gtsam_42_nonzero_skew_ignored_difference_px",
            ignored_skew_difference,
        )


class StereoFactorErrorContractTest(unittest.TestCase):
    """Evaluate a tiny stereo factor without building or optimizing a graph."""

    @classmethod
    def setUpClass(cls):
        cls.calibration, _, _, _ = _gtsam_stereo_calibration()
        cls.camera_pose = gtsam.Pose3(
            gtsam.Rot3(_rodrigues((0.17, -0.11, 0.23))),
            np.array([1.4, -0.8, 2.2]),
        )
        cls.camera_point = np.array([1.2, -0.3, 8.0])
        cls.parent_landmark = np.asarray(
            cls.camera_pose.transformFrom(cls.camera_point)
        )
        cls.measurement = gtsam.StereoCamera(
            cls.camera_pose, cls.calibration
        ).project(cls.parent_landmark)
        cls.camera_key = gtsam.symbol("c", 7)
        cls.landmark_key = gtsam.symbol("q", 19)
        cls.noise = gtsam.noiseModel.Isotropic.Sigma(
            3, TEST_ONLY_FACTOR_SIGMA
        )
        cls.values = gtsam.Values()
        cls.values.insert(cls.camera_key, cls.camera_pose)
        cls.values.insert(cls.landmark_key, cls.parent_landmark)

    def _factor(
        self, measurement: gtsam.StereoPoint2
    ) -> gtsam.GenericStereoFactor3D:
        return gtsam.GenericStereoFactor3D(
            measurement,
            self.noise,
            self.camera_key,
            self.landmark_key,
            self.calibration,
        )

    def test_factor_uses_camera_to_parent_pose_and_parent_landmark(self):
        factor = self._factor(self.measurement)
        self.assertEqual(
            list(factor.keys()), [self.camera_key, self.landmark_key]
        )
        perfect_error = float(factor.error(self.values))
        perfect_residual = np.asarray(factor.unwhitenedError(self.values))
        np.testing.assert_allclose(
            perfect_residual, np.zeros(3), rtol=0.0, atol=TEST_MATRIX_ATOL
        )
        self.assertLess(perfect_error, TEST_MATRIX_ATOL)

        wrong_landmark_values = gtsam.Values()
        wrong_landmark_values.insert(self.camera_key, self.camera_pose)
        wrong_landmark_values.insert(self.landmark_key, self.camera_point)
        wrong_landmark_error = float(factor.error(wrong_landmark_values))

        wrong_pose_values = gtsam.Values()
        wrong_pose_values.insert(self.camera_key, self.camera_pose.inverse())
        wrong_pose_values.insert(self.landmark_key, self.parent_landmark)
        wrong_pose_error = float(factor.error(wrong_pose_values))
        self.assertGreater(wrong_landmark_error, 1.0)
        self.assertGreater(wrong_pose_error, 1.0)

        _record("stereo_factor_perfect_error", perfect_error)
        _record(
            "stereo_factor_wrong_landmark_frame_error",
            wrong_landmark_error,
        )
        _record(
            "stereo_factor_inverted_camera_pose_error", wrong_pose_error
        )

    def test_perturbed_measurement_matches_gaussian_squared_error(self):
        perturbation = np.array([1.25, -0.75, 2.0])
        perturbed_vector = self.measurement.vector() + perturbation
        factor = self._factor(gtsam.StereoPoint2(*perturbed_vector))
        residual = np.asarray(factor.unwhitenedError(self.values))
        whitened_residual = np.asarray(factor.whitenedError(self.values))
        expected_residual = -perturbation
        expected_whitened = expected_residual / TEST_ONLY_FACTOR_SIGMA
        np.testing.assert_allclose(
            residual,
            expected_residual,
            rtol=0.0,
            atol=TEST_MATRIX_ATOL,
        )
        np.testing.assert_allclose(
            whitened_residual,
            expected_whitened,
            rtol=0.0,
            atol=TEST_MATRIX_ATOL,
        )

        observed_error = float(factor.error(self.values))
        expected_error = float(
            residual @ residual
            / (2.0 * TEST_ONLY_FACTOR_SIGMA * TEST_ONLY_FACTOR_SIGMA)
        )
        relationship_error = abs(observed_error - expected_error)
        self.assertLess(relationship_error, TEST_MATRIX_ATOL)

        _record("stereo_factor_test_only_sigma_px", TEST_ONLY_FACTOR_SIGMA)
        _record(
            "stereo_factor_perturbation_px", perturbation.tolist()
        )
        _record("stereo_factor_unwhitened_residual_px", residual.tolist())
        _record(
            "stereo_factor_whitened_residual",
            whitened_residual.tolist(),
        )
        _record("stereo_factor_perturbed_error", observed_error)
        _record("stereo_factor_expected_gaussian_error", expected_error)
        _record(
            "stereo_factor_error_relationship_abs_error",
            relationship_error,
        )


class CanonicalCompanionInputContractTest(unittest.TestCase):
    """Validate only the canonical companion and retained small evidence."""

    @classmethod
    def setUpClass(cls):
        cls.companion_bytes = COMPANION_PATH.read_bytes()
        cls.companion = json.loads(cls.companion_bytes.decode("utf-8"))
        cls.manifest = _read_json(MANIFEST_PATH)
        cls.complete = _read_json(COMPLETE_PATH)
        cls.validation = _read_json(VALIDATION_PATH)
        cls.i04_validation = _read_json(I04_VALIDATION_PATH)
        cls.tracking_state = cls.companion["tracking_state"]
        cls.poses = np.asarray(
            cls.tracking_state["world_to_camera_poses"], dtype=float
        )
        cls.camera_centers = np.asarray(
            cls.companion["camera_centers"], dtype=float
        )

    def test_schema_frame_ids_pose_counts_and_convention(self):
        self.assertIs(type(self.companion["schema_version"]), int)
        self.assertIs(type(self.companion["frame_count"]), int)
        self.assertIs(type(self.companion["transition_count"]), int)
        self.assertEqual(
            self.companion["schema"], "exercise4-final-companion-v1"
        )
        self.assertEqual(self.companion["schema_version"], 1)
        self.assertEqual(
            self.companion["tracking_state_schema"],
            "exercise4-trackingdb-companion-v1",
        )
        self.assertEqual(self.companion["frame_count"], 3360)
        self.assertEqual(
            self.companion["frame_range_inclusive"], [0, 3359]
        )
        self.assertEqual(self.companion["transition_count"], 3359)
        self.assertEqual(
            self.companion["pose_convention"],
            "world-to-camera: x_i = R_i X + t_i; "
            "C_i = -R_i.T @ t_i",
        )
        self.assertEqual(
            self.tracking_state["pose_convention"],
            "world-to-camera: x_i = R_i X + t_i",
        )
        self.assertTrue(
            all(
                type(frame_id) is int
                for frame_id in self.tracking_state["pose_frame_ids"]
            )
        )
        self.assertEqual(
            self.tracking_state["pose_frame_ids"], list(range(3360))
        )
        raw_poses = self.tracking_state["world_to_camera_poses"]
        self.assertTrue(
            all(
                isinstance(pose, list)
                and len(pose) == 3
                and all(
                    isinstance(row, list)
                    and len(row) == 4
                    and all(
                        type(value) in (int, float) for value in row
                    )
                    for row in pose
                )
                for pose in raw_poses
            )
        )
        self.assertEqual(self.poses.shape, (3360, 3, 4))
        self.assertEqual(self.camera_centers.shape, (3360, 3))
        transition_pairs = self.tracking_state[
            "supporter_percentage_transition_pairs"
        ]
        self.assertTrue(
            all(
                isinstance(pair, list)
                and len(pair) == 2
                and all(type(frame_id) is int for frame_id in pair)
                for pair in transition_pairs
            )
        )
        self.assertEqual(
            transition_pairs,
            [[frame_id, frame_id + 1] for frame_id in range(3359)],
        )
        self.assertEqual(
            len(self.tracking_state["supporter_counts"]), 3359
        )
        self.assertEqual(
            len(self.tracking_state["supporter_percentages"]), 3359
        )
        _record("companion_frame_count", int(self.poses.shape[0]))
        _record("companion_pose_shape", list(self.poses.shape))
        _record(
            "companion_ordered_frame_id_range",
            [
                self.tracking_state["pose_frame_ids"][0],
                self.tracking_state["pose_frame_ids"][-1],
            ],
        )

    def test_all_poses_are_finite_rigid_transforms(self):
        self.assertTrue(np.all(np.isfinite(self.poses)))
        rotations = self.poses[:, :, :3]
        orthogonality_residuals = (
            np.transpose(rotations, (0, 2, 1)) @ rotations - np.eye(3)
        )
        max_abs_by_frame = np.max(
            np.abs(orthogonality_residuals), axis=(1, 2)
        )
        frobenius_by_frame = np.linalg.norm(
            orthogonality_residuals, axis=(1, 2)
        )
        determinant_errors = np.abs(np.linalg.det(rotations) - 1.0)

        maximum_orthogonality_error = float(np.max(max_abs_by_frame))
        maximum_frobenius_error = float(np.max(frobenius_by_frame))
        maximum_determinant_error = float(np.max(determinant_errors))
        maximum_orthogonality_frame = int(np.argmax(max_abs_by_frame))
        maximum_determinant_frame = int(np.argmax(determinant_errors))

        for pose in self.poses:
            validate_world_to_camera_extrinsic(
                pose, rotation_atol=CANONICAL_ROTATION_ATOL
            )

        _record(
            "companion_rotation_orthogonality_max_abs_error",
            maximum_orthogonality_error,
        )
        _record(
            "companion_rotation_orthogonality_max_frobenius_error",
            maximum_frobenius_error,
        )
        _record(
            "companion_rotation_orthogonality_worst_frame",
            maximum_orthogonality_frame,
        )
        _record(
            "companion_rotation_determinant_max_abs_error",
            maximum_determinant_error,
        )
        _record(
            "companion_rotation_determinant_worst_frame",
            maximum_determinant_frame,
        )
        self.assertLess(
            maximum_orthogonality_error, CANONICAL_ROTATION_ATOL
        )
        self.assertLess(
            maximum_determinant_error, CANONICAL_ROTATION_ATOL
        )

    def test_selected_pose_conversions_and_window_point_path(self):
        selected_frame_ids = (0, 1, 149, 1000, 3359)
        round_trip_errors = []
        inverse_matrix_errors = []
        center_errors = []

        for frame_id in selected_frame_ids:
            extrinsic = self.poses[frame_id]
            camera_pose = camera_pose_from_world_to_camera(
                extrinsic, rotation_atol=CANONICAL_ROTATION_ATOL
            )
            expected_pose_matrix = np.linalg.inv(_homogeneous(extrinsic))
            inverse_matrix_errors.append(
                _maximum_absolute_error(
                    camera_pose.matrix(), expected_pose_matrix
                )
            )
            round_trip_errors.append(
                _maximum_absolute_error(
                    world_to_camera_from_camera_pose(
                        camera_pose,
                        rotation_atol=CANONICAL_ROTATION_ATOL,
                    ),
                    extrinsic,
                )
            )
            center_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(camera_pose.translation())
                        - self.camera_centers[frame_id]
                    )
                )
            )

        frame_zero_pose = camera_pose_from_world_to_camera(
            self.poses[0], rotation_atol=CANONICAL_ROTATION_ATOL
        )
        frame_zero_identity_error = _maximum_absolute_error(
            frame_zero_pose.matrix(), np.eye(4)
        )

        anchor_frame_id = 1000
        camera_frame_id = 3359
        window_pose = camera_pose_in_window(
            self.poses[anchor_frame_id],
            self.poses[camera_frame_id],
            rotation_atol=CANONICAL_ROTATION_ATOL,
        )
        expected_window_pose = (
            _homogeneous(self.poses[anchor_frame_id])
            @ np.linalg.inv(_homogeneous(self.poses[camera_frame_id]))
        )
        window_formula_error = _maximum_absolute_error(
            window_pose.matrix(), expected_window_pose
        )
        anchor_identity = camera_pose_in_window(
            self.poses[anchor_frame_id],
            self.poses[anchor_frame_id],
            rotation_atol=CANONICAL_ROTATION_ATOL,
        )
        anchor_identity_error = _maximum_absolute_error(
            anchor_identity.matrix(), np.eye(4)
        )

        world_point = np.array([17.0, -3.5, 42.0])
        point_in_camera = world_point_to_camera(
            self.poses[camera_frame_id],
            world_point,
            rotation_atol=CANONICAL_ROTATION_ATOL,
        )
        direct_window_point = world_point_to_camera(
            self.poses[anchor_frame_id],
            world_point,
            rotation_atol=CANONICAL_ROTATION_ATOL,
        )
        through_camera_window_point = camera_point_to_window(
            window_pose,
            point_in_camera,
            rotation_atol=CANONICAL_ROTATION_ATOL,
        )
        canonical_point_path_error = float(
            np.linalg.norm(
                through_camera_window_point - direct_window_point
            )
        )

        observed_errors = (
            round_trip_errors
            + inverse_matrix_errors
            + center_errors
            + [
                frame_zero_identity_error,
                window_formula_error,
                anchor_identity_error,
                canonical_point_path_error,
            ]
        )
        for error in observed_errors:
            self.assertLess(error, CANONICAL_ROTATION_ATOL)

        _record(
            "companion_selected_frame_ids", list(selected_frame_ids)
        )
        _record(
            "companion_selected_pose_round_trip_max_abs_error",
            max(round_trip_errors),
        )
        _record(
            "companion_selected_pose_inverse_matrix_max_abs_error",
            max(inverse_matrix_errors),
        )
        _record(
            "companion_selected_camera_center_max_l2_error",
            max(center_errors),
        )
        _record(
            "companion_frame_zero_identity_max_abs_error",
            frame_zero_identity_error,
        )
        _record(
            "companion_window_1000_to_3359_max_abs_error",
            window_formula_error,
        )
        _record(
            "companion_window_anchor_identity_max_abs_error",
            anchor_identity_error,
        )
        _record(
            "companion_world_camera_window_path_l2_error",
            canonical_point_path_error,
        )

    def test_database_identity_agrees_with_retained_manifest_evidence(self):
        self.assertEqual(
            self.manifest["schema"], "exercise4-final-manifest-v1"
        )
        self.assertEqual(
            self.complete["schema"], "exercise4-final-complete-v1"
        )

        artifact_paths = [
            artifact["path"] for artifact in self.manifest["artifacts"]
        ]
        self.assertEqual(len(artifact_paths), len(set(artifact_paths)))
        artifact_by_path = {
            artifact["path"]: artifact
            for artifact in self.manifest["artifacts"]
        }
        database_identity = self.companion["database"]
        database_manifest = artifact_by_path["tracking_database.pkl"]
        companion_manifest = artifact_by_path[
            "tracking_database_companion.json"
        ]
        serialization = self.validation["serialization"]
        i04_full_analysis = self.i04_validation["full_analyze_only"]

        self.assertEqual(
            database_identity["path"], database_manifest["path"]
        )
        self.assertEqual(
            database_identity["size_bytes"],
            database_manifest["size_bytes"],
        )
        self.assertEqual(
            database_identity["file_sha256"],
            database_manifest["sha256"],
        )
        self.assertEqual(
            database_identity["file_sha256"],
            serialization["database_file_sha256"],
        )
        self.assertEqual(
            database_identity["streaming_digest"],
            self.manifest["database_streaming_digest"],
        )
        self.assertEqual(
            database_identity["streaming_digest"],
            serialization["live_streaming_digest"],
        )
        self.assertEqual(
            database_identity["streaming_digest"],
            serialization["loaded_streaming_digest"],
        )
        self.assertEqual(
            database_identity["streaming_digest_schema"],
            "exercise4-trackingdb-stream-v1",
        )
        self.assertEqual(
            database_identity["streaming_digest_schema"],
            serialization["streaming_digest_schema"],
        )
        self.assertEqual(
            self.companion["scientific_state_sha256"],
            self.manifest["scientific_state_sha256"],
        )

        actual_companion_sha256 = hashlib.sha256(
            self.companion_bytes
        ).hexdigest()
        self.assertEqual(
            actual_companion_sha256, companion_manifest["sha256"]
        )
        self.assertEqual(
            len(self.companion_bytes), companion_manifest["size_bytes"]
        )

        checksum_entries = {}
        for line in CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
            digest, relative_path = line.split(maxsplit=1)
            checksum_entries[relative_path] = digest
        self.assertEqual(
            checksum_entries["tracking_database.pkl"],
            database_identity["file_sha256"],
        )
        self.assertEqual(
            checksum_entries["tracking_database_companion.json"],
            actual_companion_sha256,
        )

        actual_manifest_sha256 = _sha256(MANIFEST_PATH)
        actual_checksums_sha256 = _sha256(CHECKSUMS_PATH)
        self.assertEqual(
            actual_manifest_sha256, self.complete["manifest_sha256"]
        )
        self.assertEqual(
            actual_checksums_sha256, self.complete["sha256sums_sha256"]
        )

        self.assertEqual(
            i04_full_analysis["database_sha256"],
            database_identity["file_sha256"],
        )
        self.assertEqual(
            i04_full_analysis["database_size_bytes"],
            database_identity["size_bytes"],
        )
        self.assertEqual(
            i04_full_analysis["database_streaming_digest"],
            database_identity["streaming_digest"],
        )
        self.assertEqual(
            i04_full_analysis["companion_sha256"],
            actual_companion_sha256,
        )
        self.assertEqual(
            i04_full_analysis["retained_scientific_state_sha256"],
            self.companion["scientific_state_sha256"],
        )

        calibration_sha256 = _sha256(CALIBRATION_PATH)
        self.assertEqual(
            calibration_sha256,
            self.companion["provenance"]["dataset"][
                "calibration_sha256"
            ],
        )
        _record(
            "canonical_database_identity",
            {
                "path": database_identity["path"],
                "size_bytes": database_identity["size_bytes"],
                "file_sha256": database_identity["file_sha256"],
                "streaming_digest_schema": database_identity[
                    "streaming_digest_schema"
                ],
                "streaming_digest": database_identity[
                    "streaming_digest"
                ],
            },
        )
        _record(
            "canonical_companion_sha256", actual_companion_sha256
        )
        _record("canonical_calibration_sha256", calibration_sha256)


def tearDownModule():
    """Print measured E5-I01 findings after the bounded tests."""

    print("\nE5-I01 observed findings:")
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
