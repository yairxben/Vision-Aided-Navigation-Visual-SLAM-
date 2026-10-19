"""Deterministic synthetic contract tests for the E5-I03A machinery."""

from __future__ import annotations

import ast
import copy
import csv
import inspect
import json
import os
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e5_i03a_matplotlib_tests")

import gtsam
import numpy as np
from matplotlib import image as mpl_image

import exercise5_artifacts as artifact_support
import exercise5_full_run as current_full_run
from exercise5_aggregation import (
    aggregate_projection_residuals,
    aggregate_window_metrics,
    aggregate_window_rejections,
)
from exercise5_full_run import (
    ArtifactValidationError,
    BOUNDED_PROBE_MINIMUM_MEM_AVAILABLE_BYTES,
    BOUNDED_PROBE_RSS_SAFETY_MARGIN_BYTES,
    CANONICAL_CONFIRMATION,
    COURSE_MAXIMUM_INITIALIZATION_DEPTH_M,
    COURSE_STEREO_PROJECTION_SIGMAS_PX,
    CovarianceValidationError,
    Exercise5ContractError,
    FIRST_POSE_PRIOR_CANDIDATES,
    MINIMUM_MEM_AVAILABLE_BYTES,
    MINIMUM_MEM_PLUS_SWAP_BYTES,
    PRODUCTION_POST_LOAD_MINIMUM_MEM_AVAILABLE_BYTES,
    PRODUCTION_POST_LOAD_RSS_SAFETY_MARGIN_BYTES,
    VALIDATED_E5_I03A_PEAK_RSS_BYTES,
    VALIDATED_E5_I02_PEAK_RSS_BYTES,
    bounded_probe_post_load_memory_evidence,
    evaluate_standard_marginals_candidate,
    extract_endpoint_covariances,
    extract_first_window_camera_marginals,
    load_frozen_configuration,
    production_post_load_memory_evidence,
    production_post_load_memory_policy,
    seal_window_artifacts,
    select_first_pose_prior_candidate,
    validate_covariance_evidence,
    validate_frozen_configuration,
    validate_prepared_window,
    validate_sealed_window,
    write_and_seal_manifest,
)
from exercise5_reporting import plot_projection_distance, plot_window_errors
from exercise5_submission import (
    _promote_curated_then_complete_source,
    materialize_bundle_package,
)
from test_ex5_bundle_window import _main_synthetic_case


ROOT = Path(__file__).resolve().parent


class StreamingSyntheticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.database, cls.poses, cls.result = _main_synthetic_case()
        cls.configuration, cls.config_sha = load_frozen_configuration()
        cls.calibration = gtsam.Cal3_S2Stereo(500.0, 500.0, 0.0, 320.0, 240.0, 0.2)
        marginals = gtsam.Marginals(cls.result.graph, cls.result.final_values)
        cls.endpoint = extract_endpoint_covariances(
            cls.result, marginals, cls.configuration
        )
        cls.camera_marginals = extract_first_window_camera_marginals(
            cls.result, marginals, cls.configuration
        )
        del marginals
        cls.identity_sha = "a" * 64
        completed = subprocess.run(
            [
                "git",
                "show",
                (
                    "proven-e5-e7-before-cleanup-2026-08-24:"
                    "exercise5_full_run.py"
                ),
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        cls.fallback_full_run = types.ModuleType(
            "exercise5_full_run_proven_fallback"
        )
        cls.fallback_full_run.__file__ = str(ROOT / "exercise5_full_run.py")
        exec(
            compile(
                completed.stdout,
                cls.fallback_full_run.__file__,
                "exec",
            ),
            cls.fallback_full_run.__dict__,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="e5_i03a_test_", dir="/tmp"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _seal(self, index=1, endpoint=None):
        return seal_window_artifacts(
            windows_root=self.root / "windows",
            quarantine_root=self.root / "quarantine",
            result=self.result,
            database=self.database,
            calibration=self.calibration,
            world_to_camera_poses=self.poses,
            ground_truth_poses=self.poses,
            configuration=self.configuration,
            config_sha256=self.config_sha,
            source_identity_sha256=self.identity_sha,
            window_index=index,
            endpoint_covariances=endpoint or self.endpoint,
            first_window_camera_marginals=(
                self.camera_marginals if index == 0 else None
            ),
            runtime_seconds=1.25,
            peak_rss_bytes=123456,
            image_root=None,
        )

    def test_schema_and_frozen_configuration_validation(self):
        validate_frozen_configuration(self.configuration)
        changed = copy.deepcopy(self.configuration)
        changed["scientific_configuration"]["measurement_uncertainty"][
            "stereo_sigma_px"
        ] = [2.0, 2.0, 2.0]
        with self.assertRaises(Exercise5ContractError):
            validate_frozen_configuration(changed)

    def test_artifact_compatibility_surface_and_exception_identity(self):
        self.assertIs(
            current_full_run.ArtifactValidationError,
            artifact_support.ArtifactValidationError,
        )
        moved_public_names = (
            "utc_now",
            "sha256_file",
            "validate_prepared_window",
            "validate_sealed_window",
            "quarantine_attempt",
            "snapshot_configuration",
            "update_run_state",
            "write_generator_registry",
            "write_checksum_index",
            "build_run_manifest",
            "write_and_seal_manifest",
        )
        for name in moved_public_names:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(current_full_run, name),
                    getattr(artifact_support, name),
                )
        compatibility_names = moved_public_names + (
            "seal_window_artifacts",
        )
        for name in compatibility_names:
            with self.subTest(signature=name):
                self.assertEqual(
                    str(inspect.signature(getattr(current_full_run, name))),
                    str(inspect.signature(getattr(self.fallback_full_run, name))),
                )

    def test_artifact_lifecycle_matches_proven_fallback(self):
        common = {
            "database": self.database,
            "calibration": self.calibration,
            "world_to_camera_poses": self.poses,
            "ground_truth_poses": self.poses,
            "configuration": self.configuration,
            "config_sha256": self.config_sha,
            "source_identity_sha256": self.identity_sha,
            "window_index": 1,
            "endpoint_covariances": self.endpoint,
            "first_window_camera_marginals": None,
            "runtime_seconds": 1.25,
            "peak_rss_bytes": 123456,
            "image_root": None,
        }
        fallback = self.fallback_full_run.seal_window_artifacts(
            windows_root=self.root / "fallback" / "windows",
            quarantine_root=self.root / "fallback" / "quarantine",
            result=self.result,
            **common,
        )
        candidate = current_full_run.seal_window_artifacts(
            windows_root=self.root / "candidate" / "windows",
            quarantine_root=self.root / "candidate" / "quarantine",
            result=self.result,
            **common,
        )
        fallback_directory = Path(fallback["directory"])
        candidate_directory = Path(candidate["directory"])
        fallback_files = {
            path.name for path in fallback_directory.iterdir() if path.is_file()
        }
        candidate_files = {
            path.name for path in candidate_directory.iterdir() if path.is_file()
        }
        self.assertEqual(candidate_files, fallback_files)

        for filename in (
            "summary.json",
            "tracks.csv",
            "projection_residuals.csv",
            "endpoint_edge.json",
        ):
            with self.subTest(stable_file=filename):
                self.assertEqual(
                    (candidate_directory / filename).read_bytes(),
                    (fallback_directory / filename).read_bytes(),
                )

        for filename in (
            "ids.npz",
            "values.npz",
            "endpoint_covariances.npz",
        ):
            with self.subTest(npz=filename):
                with np.load(
                    fallback_directory / filename, allow_pickle=False
                ) as expected, np.load(
                    candidate_directory / filename, allow_pickle=False
                ) as observed:
                    self.assertEqual(observed.files, expected.files)
                    for array_name in expected.files:
                        np.testing.assert_array_equal(
                            observed[array_name], expected[array_name]
                        )

        fallback_seal = json.loads(
            (fallback_directory / "artifact_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_seal = json.loads(
            (candidate_directory / "artifact_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        fallback_seal.pop("sealed_utc")
        candidate_seal.pop("sealed_utc")
        fallback_entries = fallback_seal.pop("artifacts")
        candidate_entries = candidate_seal.pop("artifacts")
        self.assertEqual(candidate_seal, fallback_seal)
        self.assertEqual(
            [(entry["path"], entry["size_bytes"]) for entry in candidate_entries],
            [(entry["path"], entry["size_bytes"]) for entry in fallback_entries],
        )

    def test_first_window_artifact_lifecycle_matches_proven_fallback(self):
        common = {
            "database": self.database,
            "calibration": self.calibration,
            "world_to_camera_poses": self.poses,
            "ground_truth_poses": self.poses,
            "configuration": self.configuration,
            "config_sha256": self.config_sha,
            "source_identity_sha256": self.identity_sha,
            "window_index": 0,
            "endpoint_covariances": self.endpoint,
            "first_window_camera_marginals": self.camera_marginals,
            "runtime_seconds": 1.25,
            "peak_rss_bytes": 123456,
            "image_root": None,
        }
        fallback = self.fallback_full_run.seal_window_artifacts(
            windows_root=self.root / "fallback_first" / "windows",
            quarantine_root=self.root / "fallback_first" / "quarantine",
            result=self.result,
            **common,
        )
        candidate = current_full_run.seal_window_artifacts(
            windows_root=self.root / "candidate_first" / "windows",
            quarantine_root=self.root / "candidate_first" / "quarantine",
            result=self.result,
            **common,
        )
        fallback_directory = Path(fallback["directory"])
        candidate_directory = Path(candidate["directory"])
        fallback_files = {
            path.name for path in fallback_directory.iterdir() if path.is_file()
        }
        candidate_files = {
            path.name for path in candidate_directory.iterdir() if path.is_file()
        }
        expected_files = set(
            artifact_support.BASE_WINDOW_FILES
            + artifact_support.FIRST_WINDOW_FILES
            + ("artifact_hashes.json",)
        )
        self.assertEqual(fallback_files, expected_files)
        self.assertEqual(candidate_files, expected_files)

        for filename in (
            "summary.json",
            "tracks.csv",
            "projection_residuals.csv",
            "endpoint_edge.json",
            "maximum_initial_projection_factor.json",
        ):
            with self.subTest(stable_file=filename):
                self.assertEqual(
                    (candidate_directory / filename).read_bytes(),
                    (fallback_directory / filename).read_bytes(),
                )

        for filename in (
            "ids.npz",
            "values.npz",
            "endpoint_covariances.npz",
            "camera_marginals.npz",
        ):
            with self.subTest(npz=filename):
                with np.load(
                    fallback_directory / filename, allow_pickle=False
                ) as expected, np.load(
                    candidate_directory / filename, allow_pickle=False
                ) as observed:
                    self.assertEqual(observed.files, expected.files)
                    for array_name in expected.files:
                        self.assertEqual(
                            observed[array_name].dtype,
                            expected[array_name].dtype,
                        )
                        self.assertEqual(
                            observed[array_name].shape,
                            expected[array_name].shape,
                        )
                        np.testing.assert_array_equal(
                            observed[array_name], expected[array_name]
                        )

        for filename in artifact_support.FIRST_WINDOW_FILES:
            if not filename.endswith(".png"):
                continue
            with self.subTest(png=filename):
                expected_pixels = mpl_image.imread(
                    fallback_directory / filename
                )
                observed_pixels = mpl_image.imread(
                    candidate_directory / filename
                )
                self.assertEqual(observed_pixels.dtype, expected_pixels.dtype)
                self.assertEqual(observed_pixels.shape, expected_pixels.shape)
                np.testing.assert_array_equal(observed_pixels, expected_pixels)

        validation_arguments = {
            "expected_index": 0,
            "expected_start": self.result.frame_ids[0],
            "expected_end": self.result.frame_ids[-1],
            "config_sha256": self.config_sha,
            "source_identity_sha256": self.identity_sha,
        }
        self.fallback_full_run.validate_prepared_window(
            fallback_directory,
            allow_hash_file=True,
            **validation_arguments,
        )
        current_full_run.validate_prepared_window(
            candidate_directory,
            allow_hash_file=True,
            **validation_arguments,
        )
        self.fallback_full_run.validate_sealed_window(
            fallback_directory,
            **validation_arguments,
        )
        current_full_run.validate_sealed_window(
            candidate_directory,
            **validation_arguments,
        )

        fallback_seal = json.loads(
            (fallback_directory / "artifact_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_seal = json.loads(
            (candidate_directory / "artifact_hashes.json").read_text(
                encoding="utf-8"
            )
        )
        fallback_seal.pop("sealed_utc")
        candidate_seal.pop("sealed_utc")
        fallback_entries = fallback_seal.pop("artifacts")
        candidate_entries = candidate_seal.pop("artifacts")
        self.assertEqual(candidate_seal, fallback_seal)
        self.assertEqual(
            [(entry["path"], entry["size_bytes"]) for entry in candidate_entries],
            [(entry["path"], entry["size_bytes"]) for entry in fallback_entries],
        )

    def test_course_projection_depth_and_two_prior_candidates_are_exact(self):
        one_degree = float(np.radians(1.0))
        self.assertEqual(
            COURSE_STEREO_PROJECTION_SIGMAS_PX, (1.0, 1.0, 1.0)
        )
        self.assertEqual(COURSE_MAXIMUM_INITIALIZATION_DEPTH_M, 300.0)
        self.assertEqual(
            FIRST_POSE_PRIOR_CANDIDATES,
            {
                "A": (
                    one_degree**2,
                    one_degree**2,
                    one_degree**2,
                    0.1,
                    0.01,
                    1.0,
                ),
                "B": (
                    one_degree,
                    one_degree,
                    one_degree,
                    0.3,
                    0.1,
                    1.0,
                ),
            },
        )
        self.assertEqual(
            self.configuration["scientific_configuration"][
                "measurement_uncertainty"
            ]["stereo_sigma_px"],
            list(COURSE_STEREO_PROJECTION_SIGMAS_PX),
        )
        prior = self.configuration["scientific_configuration"][
            "gauge_fixing_prior"
        ]
        self.assertEqual(
            prior["authorized_candidates"],
            {
                name: list(sigmas)
                for name, sigmas in FIRST_POSE_PRIOR_CANDIDATES.items()
            },
        )
        self.assertEqual(prior["selected_candidate"], "A")
        self.assertEqual(
            self.configuration["scientific_configuration"][
                "landmark_eligibility"
            ]["maximum_initialization_depth_m"],
            COURSE_MAXIMUM_INITIALIZATION_DEPTH_M,
        )

    def test_standard_marginals_is_once_and_fail_closed(self):
        successful, endpoint, camera_marginals = (
            evaluate_standard_marginals_candidate(
                candidate="A",
                result=self.result,
                configuration=self.configuration,
            )
        )
        self.assertTrue(successful["eligible"])
        self.assertTrue(successful["marginals"]["succeeded"])
        self.assertTrue(successful["covariance_validation"]["passed"])
        self.assertIsNotNone(endpoint)
        self.assertIsNotNone(camera_marginals)

        construction_calls = []

        def failing_factory(graph, values):
            construction_calls.append((graph, values))
            raise RuntimeError("injected standard Marginals failure")

        failed, endpoint, camera_marginals = (
            evaluate_standard_marginals_candidate(
                candidate="B",
                result=self.result,
                configuration=self.configuration,
                marginals_factory=failing_factory,
            )
        )
        self.assertEqual(len(construction_calls), 1)
        self.assertFalse(failed["eligible"])
        self.assertFalse(failed["marginals"]["succeeded"])
        self.assertIn(
            "injected standard Marginals failure",
            failed["marginals"]["exception_message"],
        )
        self.assertIsNone(endpoint)
        self.assertIsNone(camera_marginals)

        with mock.patch(
            "exercise5_full_run.extract_endpoint_covariances",
            side_effect=CovarianceValidationError(
                "injected invalid covariance output"
            ),
        ):
            invalid, endpoint, camera_marginals = (
                evaluate_standard_marginals_candidate(
                    candidate="A",
                    result=self.result,
                    configuration=self.configuration,
                )
            )
        self.assertTrue(invalid["marginals"]["succeeded"])
        self.assertFalse(invalid["covariance_validation"]["passed"])
        self.assertFalse(invalid["eligible"])
        self.assertIsNone(endpoint)
        self.assertIsNone(camera_marginals)

    def test_candidate_selection_is_exact_and_deterministic(self):
        def outcome(candidate, eligible):
            return {
                "candidate": candidate,
                "prior_sigmas": list(
                    FIRST_POSE_PRIOR_CANDIDATES[candidate]
                ),
                "marginals": {"succeeded": eligible},
                "covariance_validation": {"passed": eligible},
                "eligible": eligible,
            }

        self.assertEqual(
            select_first_pose_prior_candidate(
                {"A": outcome("A", True), "B": outcome("B", False)}
            ),
            "A",
        )
        self.assertEqual(
            select_first_pose_prior_candidate(
                {"A": outcome("A", False), "B": outcome("B", True)}
            ),
            "B",
        )
        self.assertEqual(
            select_first_pose_prior_candidate(
                {"A": outcome("A", True), "B": outcome("B", True)}
            ),
            "A",
        )
        self.assertIsNone(
            select_first_pose_prior_candidate(
                {"A": outcome("A", False), "B": outcome("B", False)}
            )
        )
        with self.assertRaises(ValueError):
            select_first_pose_prior_candidate({"A": outcome("A", True)})
        changed = copy.deepcopy(self.configuration)
        changed["keyframe_selection"]["ordered_keyframe_ids"][1] = 8
        with self.assertRaises(Exercise5ContractError):
            validate_frozen_configuration(changed)

    def test_bounded_probe_post_load_memory_guard_passes_incremental_case(self):
        current_rss = 5_000_000_000
        evidence = bounded_probe_post_load_memory_evidence(
            current_process_rss_bytes=current_rss,
            mem_available_bytes=1_000_000_000,
            swap_free_bytes=1_000_000_000,
        )
        expected_increment = (
            VALIDATED_E5_I02_PEAK_RSS_BYTES - current_rss
        )
        self.assertTrue(evidence["passed"])
        self.assertEqual(
            evidence["policy"]["expected_incremental_rss_bytes"],
            expected_increment,
        )
        self.assertEqual(
            evidence["policy"]["required_headroom_bytes"],
            expected_increment + BOUNDED_PROBE_RSS_SAFETY_MARGIN_BYTES,
        )
        self.assertTrue(evidence["future_e5_i03b_policy_unchanged"])

    def test_bounded_probe_post_load_memory_guard_fails_closed(self):
        immediate_pressure = bounded_probe_post_load_memory_evidence(
            current_process_rss_bytes=VALIDATED_E5_I02_PEAK_RSS_BYTES,
            mem_available_bytes=(
                BOUNDED_PROBE_MINIMUM_MEM_AVAILABLE_BYTES - 1
            ),
            swap_free_bytes=4_000_000_000,
        )
        self.assertFalse(immediate_pressure["passed"])
        self.assertFalse(
            immediate_pressure["gates"][
                "minimum_immediate_mem_available_passed"
            ]
        )
        insufficient_headroom = bounded_probe_post_load_memory_evidence(
            current_process_rss_bytes=4_000_000_000,
            mem_available_bytes=600_000_000,
            swap_free_bytes=100_000_000,
        )
        self.assertFalse(insufficient_headroom["passed"])
        self.assertFalse(
            insufficient_headroom["gates"][
                "available_plus_swap_headroom_passed"
            ]
        )
        with self.assertRaises(ValueError):
            bounded_probe_post_load_memory_evidence(
                current_process_rss_bytes=0,
                mem_available_bytes=1,
                swap_free_bytes=1,
            )

    def test_production_post_load_memory_admission_is_fixed_and_incremental(self):
        baseline_rss = 5_000_000_000
        policy = production_post_load_memory_policy(
            post_load_baseline_rss_bytes=baseline_rss
        )
        expected_headroom = (
            VALIDATED_E5_I03A_PEAK_RSS_BYTES
            - baseline_rss
            + PRODUCTION_POST_LOAD_RSS_SAFETY_MARGIN_BYTES
        )
        self.assertEqual(
            policy["reference"]["validated_peak_rss_bytes"],
            5_348_995_072,
        )
        self.assertEqual(
            policy["required_headroom_bytes"], expected_headroom
        )
        mem_available = 1_000_000_000
        swap_free = 0
        self.assertLess(mem_available, MINIMUM_MEM_AVAILABLE_BYTES)
        self.assertLess(
            mem_available + swap_free, MINIMUM_MEM_PLUS_SWAP_BYTES
        )
        first = production_post_load_memory_evidence(
            fixed_policy=policy,
            current_process_rss_bytes=baseline_rss,
            mem_available_bytes=mem_available,
            swap_free_bytes=swap_free,
        )
        later = production_post_load_memory_evidence(
            fixed_policy=policy,
            current_process_rss_bytes=5_300_000_000,
            mem_available_bytes=mem_available,
            swap_free_bytes=swap_free,
        )
        self.assertTrue(first["passed"])
        self.assertTrue(later["passed"])
        self.assertEqual(
            first["fixed_policy"]["required_headroom_bytes"],
            expected_headroom,
        )
        self.assertEqual(
            later["fixed_policy"]["required_headroom_bytes"],
            expected_headroom,
        )

    def test_production_post_load_memory_admission_fails_closed(self):
        policy = production_post_load_memory_policy(
            post_load_baseline_rss_bytes=5_000_000_000
        )
        immediate_pressure = production_post_load_memory_evidence(
            fixed_policy=policy,
            current_process_rss_bytes=5_100_000_000,
            mem_available_bytes=(
                PRODUCTION_POST_LOAD_MINIMUM_MEM_AVAILABLE_BYTES - 1
            ),
            swap_free_bytes=1_000_000_000,
        )
        self.assertFalse(immediate_pressure["passed"])
        self.assertFalse(
            immediate_pressure["gates"][
                "minimum_immediate_mem_available_passed"
            ]
        )
        insufficient_headroom = production_post_load_memory_evidence(
            fixed_policy=policy,
            current_process_rss_bytes=5_100_000_000,
            mem_available_bytes=(
                PRODUCTION_POST_LOAD_MINIMUM_MEM_AVAILABLE_BYTES
            ),
            swap_free_bytes=0,
        )
        self.assertFalse(insufficient_headroom["passed"])
        self.assertFalse(
            insufficient_headroom["gates"][
                "available_plus_swap_headroom_passed"
            ]
        )
        with self.assertRaises(ValueError):
            production_post_load_memory_policy(
                post_load_baseline_rss_bytes=0
            )
        for invalid in (
            {
                "current_process_rss_bytes": 0,
                "mem_available_bytes": 1,
                "swap_free_bytes": 1,
            },
            {
                "current_process_rss_bytes": 1,
                "mem_available_bytes": -1,
                "swap_free_bytes": 1,
            },
            {
                "current_process_rss_bytes": 1,
                "mem_available_bytes": 1,
                "swap_free_bytes": -1,
            },
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    production_post_load_memory_evidence(
                        fixed_policy=policy, **invalid
                    )

    def test_joint_block_order_and_camera_marginal_order(self):
        joint = self.endpoint["joint_covariance"]
        marginals = gtsam.Marginals(self.result.graph, self.result.final_values)
        start = self.result.frame_ids[0]
        end = self.result.frame_ids[-1]
        np.testing.assert_allclose(
            joint[:6, :6],
            marginals.marginalCovariance(
                self.result.camera_key_by_frame_id[start]
            ),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            joint[6:, 6:],
            marginals.marginalCovariance(
                self.result.camera_key_by_frame_id[end]
            ),
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            self.camera_marginals["frame_ids"], self.result.frame_ids
        )

    def test_conditional_covariance_and_information_crosscheck(self):
        joint = self.endpoint["joint_covariance"]
        expected = joint[6:, 6:] - joint[6:, :6] @ np.linalg.solve(
            joint[:6, :6], joint[:6, 6:]
        )
        np.testing.assert_allclose(
            expected,
            self.endpoint["conditional_end_given_start_covariance"],
            atol=1e-12,
        )
        information_expected = np.linalg.solve(
            self.endpoint["joint_information"][6:, 6:], np.eye(6)
        )
        np.testing.assert_allclose(expected, information_expected, atol=1e-12)
        self.assertIn(
            "E5-I04",
            self.endpoint["diagnostics"]["between_factor_noise_status"],
        )
        joint = np.eye(12)
        slightly_inconsistent_information = np.eye(12)
        slightly_inconsistent_information[0, 0] += 1.5e-9
        _, diagnostics = validate_covariance_evidence(
            joint,
            slightly_inconsistent_information,
            None,
            symmetry_atol=1e-9,
            minimum_eigenvalue=-1e-9,
            reconstruction_atol=1e-9,
        )
        warning = diagnostics["numerical_reconstruction_crosschecks"][
            "joint_covariance_information_inverse"
        ]
        self.assertGreater(warning["measured_error"], 1e-9)
        self.assertEqual(warning["warning_threshold"], 1e-9)
        self.assertFalse(warning["passed_threshold"])
        self.assertTrue(warning["warning"])

    def test_covariance_diagnostics_warn_but_usability_failures_stop(self):
        joint = np.eye(12)
        information = np.eye(12)
        nonsymmetric = joint.copy()
        nonsymmetric[0, 1] = 0.1
        _, diagnostics = validate_covariance_evidence(
            nonsymmetric,
            information,
            None,
            symmetry_atol=1e-9,
            minimum_eigenvalue=-1e-9,
            reconstruction_atol=1e-9,
        )
        self.assertTrue(
            diagnostics["joint_covariance"]["symmetry_warning"]
        )
        indefinite = joint.copy()
        indefinite[0, 0] = -1.0
        _, diagnostics = validate_covariance_evidence(
            indefinite,
            information,
            None,
            symmetry_atol=1e-9,
            minimum_eigenvalue=-1e-9,
            reconstruction_atol=1e-9,
        )
        self.assertTrue(
            diagnostics["joint_covariance"][
                "minimum_eigenvalue_warning"
            ]
        )
        singular_information = information.copy()
        singular_information[6, 6] = 0.0
        _, diagnostics = validate_covariance_evidence(
            joint,
            singular_information,
            None,
            symmetry_atol=1e-9,
            minimum_eigenvalue=-1e-9,
            reconstruction_atol=1e-9,
        )
        self.assertTrue(
            diagnostics["joint_information"]["condition_number_warning"]
        )
        information_crosscheck = diagnostics[
            "numerical_reconstruction_crosschecks"
        ]["conditional_information_block"]
        self.assertTrue(information_crosscheck["warning"])
        self.assertEqual(
            information_crosscheck["diagnostic_error"]["type"],
            "LinAlgError",
        )
        singular = joint.copy()
        singular[0, 0] = 0.0
        with self.assertRaises(CovarianceValidationError):
            validate_covariance_evidence(
                singular,
                information,
                None,
                symmetry_atol=1e-9,
                minimum_eigenvalue=-1e-9,
                reconstruction_atol=1e-9,
            )
        nonfinite = joint.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(CovarianceValidationError):
            validate_covariance_evidence(
                nonfinite,
                information,
                None,
                symmetry_atol=1e-9,
                minimum_eigenvalue=-1e-9,
                reconstruction_atol=1e-9,
            )

    def test_atomic_temporary_write_promotion_and_reload(self):
        sealed = self._seal(index=1)
        directory = Path(sealed["directory"])
        self.assertTrue(directory.is_dir())
        self.assertTrue((directory / "artifact_hashes.json").is_file())
        self.assertFalse(
            any(path.name.startswith(".window_") for path in directory.parent.iterdir())
        )
        reloaded = validate_sealed_window(
            directory,
            expected_index=1,
            expected_start=0,
            expected_end=2,
            config_sha256=self.config_sha,
            source_identity_sha256=self.identity_sha,
        )
        self.assertEqual(reloaded["seal"]["status"], "sealed")

    def test_corrupt_and_incomplete_window_rejection(self):
        sealed = self._seal(index=1)
        directory = Path(sealed["directory"])
        with (directory / "summary.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaises(ArtifactValidationError):
            validate_sealed_window(
                directory,
                expected_index=1,
                expected_start=0,
                expected_end=2,
                config_sha256=self.config_sha,
                source_identity_sha256=self.identity_sha,
            )
        incomplete = self.root / "incomplete"
        incomplete.mkdir()
        (incomplete / "summary.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ArtifactValidationError):
            validate_prepared_window(
                incomplete,
                expected_index=1,
                expected_start=0,
                expected_end=2,
                config_sha256=self.config_sha,
                source_identity_sha256=self.identity_sha,
            )

    def test_valid_sealed_window_resume_skip_and_identity_mismatch_rejection(self):
        first = self._seal(index=1)
        second = self._seal(index=1)
        self.assertEqual(first["directory"], second["directory"])
        with self.assertRaises(ArtifactValidationError):
            validate_sealed_window(
                Path(first["directory"]),
                expected_index=1,
                expected_start=0,
                expected_end=2,
                config_sha256="b" * 64,
                source_identity_sha256=self.identity_sha,
            )
        with self.assertRaises(ArtifactValidationError):
            validate_sealed_window(
                Path(first["directory"]),
                expected_index=1,
                expected_start=0,
                expected_end=2,
                config_sha256=self.config_sha,
                source_identity_sha256="c" * 64,
            )

    def test_failed_attempt_is_quarantined(self):
        broken = dict(self.endpoint)
        broken["joint_covariance"] = np.eye(3)
        with self.assertRaises(ArtifactValidationError):
            self._seal(index=1, endpoint=broken)
        quarantined = list((self.root / "quarantine").glob("*.failed.*"))
        self.assertEqual(len(quarantined), 1)
        failure = json.loads(
            (quarantined[0] / "failure.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failure["schema"], "exercise5-window-failure-v1")

    def test_aggregation_from_multiple_sealed_synthetic_windows(self):
        self._seal(index=1)
        self._seal(index=2)
        metrics = aggregate_window_metrics(self.root)
        rejections = aggregate_window_rejections(self.root)
        residuals = aggregate_projection_residuals(self.root)
        with metrics.open("r", encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        with residuals.open("r", encoding="utf-8", newline="") as handle:
            self.assertEqual(
                len(list(csv.DictReader(handle))),
                2 * self.result.projection_factor_count,
            )
        self.assertTrue(rejections.is_file())

    def test_projection_residual_aggregation_streams_in_window_row_order(self):
        self._seal(index=1)
        self._seal(index=2)
        expected_rows = []
        expected_fields = None
        for directory in sorted((self.root / "windows").glob("window_*")):
            summary = json.loads(
                (directory / "summary.json").read_text(encoding="utf-8")
            )
            with (directory / "projection_residuals.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    combined = {
                        "window_index": str(summary["window_index"]),
                        **row,
                    }
                    if expected_fields is None:
                        expected_fields = list(combined)
                    expected_rows.append(combined)
        with mock.patch(
            "exercise5_aggregation._read_csv",
            side_effect=AssertionError(
                "projection aggregation must stream its input CSVs"
            ),
        ):
            output = aggregate_projection_residuals(self.root)
        with output.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, expected_fields)
            self.assertEqual(list(reader), expected_rows)

    def test_reporting_consumes_machine_readable_window_table(self):
        data = self.root / "aggregate" / "data"
        figures = self.root / "aggregate" / "figures"
        data.mkdir(parents=True)
        figures.mkdir(parents=True)
        fields = (
            "window_index",
            "final_mean_projection_factor_error",
            "final_median_stereo_pixel_distance_px",
        )
        with (data / "window_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "window_index": 0,
                        "final_mean_projection_factor_error": 2.0,
                        "final_median_stereo_pixel_distance_px": 1.5,
                    },
                    {
                        "window_index": 1,
                        "final_mean_projection_factor_error": 1.0,
                        "final_median_stereo_pixel_distance_px": 0.5,
                    },
                ]
            )
        plot_window_errors(data, figures)
        self.assertEqual(len(list(figures.glob("*.png"))), 2)

    def test_projection_distance_plot_streams_every_row_in_order(self):
        data = self.root / "aggregate" / "data"
        figures = self.root / "aggregate" / "figures"
        data.mkdir(parents=True)
        figures.mkdir(parents=True)
        fields = (
            "window_index",
            "pnp_path_distance_from_window_start_m",
            "final_stereo_pixel_distance_px",
        )
        rows = (
            {
                "window_index": 2,
                "pnp_path_distance_from_window_start_m": 4.5,
                "final_stereo_pixel_distance_px": 1.25,
            },
            {
                "window_index": 0,
                "pnp_path_distance_from_window_start_m": 0.0,
                "final_stereo_pixel_distance_px": 3.5,
            },
            {
                "window_index": 1,
                "pnp_path_distance_from_window_start_m": 2.75,
                "final_stereo_pixel_distance_px": 0.5,
            },
        )
        with (data / "all_projection_residuals.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        observed = {}
        import matplotlib.axes

        real_scatter = matplotlib.axes.Axes.scatter

        def capture_scatter(axis, x_values, y_values, *args, **kwargs):
            observed["x"] = np.asarray(x_values).copy()
            observed["y"] = np.asarray(y_values).copy()
            return real_scatter(
                axis, x_values, y_values, *args, **kwargs
            )

        with mock.patch(
            "exercise5_reporting._read_csv",
            side_effect=AssertionError(
                "projection-distance reporting must stream its input CSV"
            ),
        ), mock.patch.object(
            matplotlib.axes.Axes,
            "scatter",
            new=capture_scatter,
        ):
            plot_projection_distance(data, figures)
        np.testing.assert_array_equal(
            observed["x"], np.asarray([4.5, 0.0, 2.75], dtype=np.float64)
        )
        np.testing.assert_array_equal(
            observed["y"], np.asarray([1.25, 3.5, 0.5], dtype=np.float64)
        )
        self.assertEqual(observed["x"].dtype, np.dtype(np.float64))
        self.assertEqual(observed["y"].dtype, np.dtype(np.float64))
        self.assertTrue(
            (figures / "projection_error_vs_bundle_start.png").is_file()
        )
        self.assertFalse(
            any(figures.glob(".*.attempt.*.tmp"))
        )

    def test_submission_materializer_rejects_partial_source(self):
        source = self.root / "source"
        source.mkdir()
        identity = {"fixture": "synthetic"}
        write_and_seal_manifest(
            source,
            configuration=self.configuration,
            config_sha256=self.config_sha,
            source_identity=identity,
            completed_window_indices=[],
            status="running",
            scope="bounded-probe",
        )
        with self.assertRaises(ArtifactValidationError):
            materialize_bundle_package(source, self.root / "bundle")
        self.assertFalse((self.root / "bundle").exists())

    def test_submission_promotion_failure_keeps_source_incomplete(self):
        source = self.root / "source"
        source.mkdir()
        old_manifest = {
            "schema": "exercise5-run-manifest-v1",
            "status": "windows-complete-awaiting-aggregation",
        }
        (source / "manifest.json").write_text(
            json.dumps(old_manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged = self.root / ".bundle.attempt.fixture.tmp"
        staged.mkdir()
        (staged / "ready.txt").write_text("validated\n", encoding="utf-8")
        destination = self.root / "bundle"
        prospective_manifest = source / ".manifest.finalize.fixture.tmp"
        prospective_manifest.write_text(
            json.dumps(
                {
                    "schema": "exercise5-run-manifest-v1",
                    "status": "complete-and-sealed",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        prospective_checksums = source / ".checksums.finalize.fixture.tmp"
        prospective_checksums.write_text(
            "fixture  manifest.json\n", encoding="utf-8"
        )
        real_replace = os.replace

        def injected_replace(source_path, destination_path):
            if Path(source_path) == staged:
                raise OSError("injected curated promotion failure")
            return real_replace(source_path, destination_path)

        with mock.patch(
            "exercise5_submission.os.replace",
            side_effect=injected_replace,
        ):
            with self.assertRaisesRegex(
                OSError, "injected curated promotion failure"
            ):
                _promote_curated_then_complete_source(
                    source_root=source,
                    staged_destination=staged,
                    destination=destination,
                    prospective_manifest_path=prospective_manifest,
                    prospective_checksum_path=prospective_checksums,
                )
        observed = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            observed["status"],
            "windows-complete-awaiting-aggregation",
        )
        self.assertFalse(destination.exists())
        self.assertFalse(prospective_manifest.exists())
        self.assertFalse(prospective_checksums.exists())

    def test_ex5_help_and_canonical_command_safety(self):
        help_result = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"), "ex5.py"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "MPLCONFIGDIR": str(self.root / "mpl")},
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("probe", help_result.stdout)
        self.assertIn("submission", help_result.stdout)
        unsafe = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                "ex5.py",
                "run",
                "--output-root",
                str(self.root / "must_not_start"),
                "--confirm",
                "WRONG",
            ],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "MPLCONFIGDIR": str(self.root / "mpl")},
        )
        self.assertNotEqual(unsafe.returncode, 0)
        self.assertIn(CANONICAL_CONFIRMATION, unsafe.stdout)
        self.assertFalse((self.root / "must_not_start").exists())

    def test_ex5_has_no_legacy_imports_or_scientific_implementation(self):
        source = (ROOT / "ex5.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module)
        forbidden = {
            "BundleWindow",
            "BundleAdjustment",
            "algorithms_library",
            "LinkLoop",
        }
        self.assertTrue(forbidden.isdisjoint(imported))
        self.assertNotIn("gtsam", imported)
        self.assertNotIn("numpy", imported)


if __name__ == "__main__":
    unittest.main()
