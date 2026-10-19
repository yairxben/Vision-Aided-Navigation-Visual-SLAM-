"""Focused E4-R01 workflow-separation compatibility tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ex4
import exercise4_workflow


class Exercise4WorkflowImportTest(unittest.TestCase):
    def test_reader_facing_modules_and_support_apis_import(self):
        self.assertTrue(
            callable(exercise4_workflow.build_and_serialize_tracking_database)
        )
        self.assertTrue(
            callable(exercise4_workflow.analyze_serialized_tracking_database)
        )
        self.assertIn(
            "exercise4_database_builder.build_exercise4_database",
            ex4.__doc__,
        )


class Exercise4CompatibilityWrapperTest(unittest.TestCase):
    def test_run_build_delegates_without_changing_arguments(self):
        expected = {"status": "build-complete"}
        with mock.patch.object(
            exercise4_workflow,
            "build_and_serialize_tracking_database",
            return_value=expected,
        ) as delegated:
            observed = ex4._run_build(10, Path("/tmp/e4-wrapper"), True)

        self.assertIs(observed, expected)
        delegated.assert_called_once_with(
            10, Path("/tmp/e4-wrapper"), True
        )

    def test_run_analyze_delegates_without_changing_arguments(self):
        expected = {"status": "analysis-complete"}
        with mock.patch.object(
            exercise4_workflow,
            "analyze_serialized_tracking_database",
            return_value=expected,
        ) as delegated:
            observed = ex4._run_analyze(
                Path("database.pkl"),
                Path("companion.json"),
                Path("/tmp/e4-wrapper"),
                30.0,
            )

        self.assertIs(observed, expected)
        delegated.assert_called_once_with(
            Path("database.pkl"),
            Path("companion.json"),
            Path("/tmp/e4-wrapper"),
            30.0,
        )


class Exercise4OutputSafetyTest(unittest.TestCase):
    def test_compatibility_guard_accepts_only_new_or_empty_roots(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            ex4._require_new_or_empty_root(root)
            (root / "occupied").touch()
            with self.assertRaises(FileExistsError):
                ex4._require_new_or_empty_root(root)

    def test_compatibility_guard_rejects_protected_outputs(self):
        with self.assertRaisesRegex(
            ValueError, "must not modify diagnostics/ex4_final"
        ):
            ex4._require_new_or_empty_root(
                exercise4_workflow.FINAL_DIAGNOSTICS / "bounded"
            )


class Exercise4CliCompatibilityTest(unittest.TestCase):
    def test_build_dispatch_preserves_cli_arguments(self):
        with mock.patch.object(ex4, "_run_build") as run_build:
            ex4.main(
                [
                    "build",
                    "--frame-count",
                    "10",
                    "--output-dir",
                    "/tmp/e4-cli",
                    "--validate-every-insertion",
                ]
            )

        run_build.assert_called_once_with(
            10, Path("/tmp/e4-cli"), True
        )

    def test_parser_retains_all_commands(self):
        parser = ex4.build_parser()
        for command in ("build", "analyze", "all"):
            with self.subTest(command=command):
                self.assertIn(command, parser.format_help())


if __name__ == "__main__":
    unittest.main()
