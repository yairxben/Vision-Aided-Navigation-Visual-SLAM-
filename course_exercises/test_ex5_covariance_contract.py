"""Focused tests for the bounded E5-I04 covariance contract validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_ex5_covariance_contract import (
    EDGE_DIRECTION,
    TANGENT_ORDER,
    run_validation,
)


class E5I04CovarianceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory(
            prefix="e5_i04_test_"
        )
        cls.evidence_root = (
            Path(cls.temporary_directory.name) / "evidence"
        )
        cls.summary = run_validation(evidence_root=cls.evidence_root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def test_synthetic_gtsam_tangent_and_direct_covariance_contract(self):
        synthetic = self.summary["synthetic_gtsam_contract"]
        self.assertEqual(synthetic["status"], "pass")
        self.assertEqual(synthetic["tangent_order"], list(TANGENT_ORDER))
        self.assertTrue(synthetic["direct_covariance_use_passed"])
        self.assertFalse(synthetic["adjoint_transport_required"])

    def test_canonical_window_zero_between_factor_contract(self):
        window = self.summary["canonical_window_zero"]
        self.assertEqual(window["status"], "pass")
        self.assertEqual(window["edge_direction"], EDGE_DIRECTION)
        self.assertEqual(window["conditional_covariance_shape"], [6, 6])
        self.assertTrue(window["gaussian_covariance_model_constructed"])
        self.assertTrue(window["direct_covariance_use_passed"])

    def test_complete_canonical_population_prerequisites(self):
        population = self.summary["complete_population"]
        self.assertEqual(
            population["canonical_window_count_checked"], 355
        )
        self.assertEqual(population["missing_canonical_windows"], 0)
        self.assertTrue(
            population["consecutive_endpoint_identities_passed"]
        )
        self.assertEqual(
            population["gaussian_covariance_models_constructed"], 355
        )

    def test_evidence_and_execution_boundaries(self):
        reloaded = json.loads(
            (self.evidence_root / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(reloaded["status"], "pass")
        self.assertTrue(reloaded["protected_artifacts"]["unchanged"])
        self.assertFalse(
            any(reloaded["execution_boundaries"].values())
        )
        self.assertTrue((self.evidence_root / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
