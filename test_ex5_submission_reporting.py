"""Isolated report-only tests for Exercise 5 submission staging."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault(
    "MPLCONFIGDIR", "/tmp/e5_submission_reporting_matplotlib_tests"
)

import numpy as np
from matplotlib import image as mpl_image

from exercise5_full_run import ArtifactValidationError
from exercise5_reporting import plot_bundle_vs_pnp_comparisons
from exercise5_submission import (
    _validate_staged_curated_package,
    materialize_bundle_package,
)


COMPARISON_FILENAMES = (
    "bundle_vs_pnp_absolute_localization_error.png",
    "bundle_vs_pnp_relative_translation_error.png",
    "bundle_vs_pnp_relative_rotation_error.png",
)


class SubmissionReportingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path: Path, fieldnames, rows) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_png(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mpl_image.imsave(path, np.zeros((2, 2, 3), dtype=float))

    @staticmethod
    def _configuration():
        return {
            "keyframe_selection": {"statistics": {"window_count": 1}},
            "artifact_contract": {
                "aggregate_data_files": [
                    "window_metrics.csv",
                    "all_projection_residuals.csv",
                    "keyframe_localization_errors.csv",
                    "consecutive_relative_errors.csv",
                ],
                "aggregate_figure_files": [],
                "curated_required": {
                    "manifest.json": "curated manifest",
                    "metrics.json": "curated metrics",
                    "data/track_5_1.csv": "track rows",
                    "data/first_window_metrics.csv": "first window",
                    "data/window_metrics.csv": "window metrics",
                    "data/projection_by_bundle_start.csv": "projection rows",
                    "data/keyframe_localization_errors.csv": "absolute errors",
                    "data/consecutive_relative_errors.csv": "relative errors",
                    "figures/track_reprojection_error.png": "track plot",
                    "figures/track_factor_error.png": "track plot",
                    "figures/track_factor_vs_reprojection.png": "track plot",
                    "figures/first_window_overlays.png": "overlays",
                    "figures/first_window_scene_3d.png": "scene",
                    "figures/first_window_scene_birds_eye.png": "scene",
                },
            },
        }

    def _write_source(self, *, invalid_absolute: bool = False) -> None:
        self.source.mkdir()
        (self.source / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "exercise5-run-manifest-v1",
                    "status": "complete-and-sealed",
                    "scope": "canonical",
                    "expected_window_count": 1,
                    "completed_window_indices": [0],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        data = self.source / "aggregate" / "data"
        self._write_csv(
            data / "window_metrics.csv",
            (
                "window_index",
                "initial_total_graph_error",
                "final_total_graph_error",
            ),
            (
                {
                    "window_index": 0,
                    "initial_total_graph_error": 2.0,
                    "final_total_graph_error": 1.0,
                },
            ),
        )
        self._write_csv(
            data / "all_projection_residuals.csv",
            ("window_index", "final_stereo_pixel_distance_px"),
            ({"window_index": 0, "final_stereo_pixel_distance_px": 0.5},),
        )
        self._write_csv(
            data / "keyframe_localization_errors.csv",
            ("frame_id", "bundle_error_m", "pnp_error_m"),
            (
                {
                    "frame_id": 0,
                    "bundle_error_m": "inf" if invalid_absolute else 0.0,
                    "pnp_error_m": 0.0,
                },
            ),
        )
        self._write_csv(
            data / "consecutive_relative_errors.csv",
            (
                "window_index",
                "bundle_translation_error_m",
                "pnp_translation_error_m",
                "bundle_rotation_error_deg",
                "pnp_rotation_error_deg",
            ),
            (
                {
                    "window_index": 0,
                    "bundle_translation_error_m": 0.1,
                    "pnp_translation_error_m": 0.2,
                    "bundle_rotation_error_deg": 0.3,
                    "pnp_rotation_error_deg": 0.4,
                },
            ),
        )
        first_window = self.source / "windows" / "window_0000_0_1"
        for name in (
            "overlay_left_initial.png",
            "overlay_right_initial.png",
            "overlay_left_final.png",
            "overlay_right_final.png",
            "scene_3d.png",
            "scene_birds_eye.png",
        ):
            self._write_png(first_window / name)
        track = self.source / "track_5_1"
        self._write_csv(
            track / "residuals.csv",
            ("frame_id", "factor_error"),
            ({"frame_id": 0, "factor_error": 0.0},),
        )
        for name in (
            "reprojection_error.png",
            "factor_error.png",
            "factor_vs_reprojection.png",
        ):
            self._write_png(track / name)

    def _materialize(self, destination: Path, configuration):
        with mock.patch(
            "exercise5_submission.load_frozen_configuration",
            return_value=(configuration, "c" * 64),
        ), mock.patch(
            "exercise5_submission.validate_run_root",
            return_value={
                "status": "complete-and-sealed",
                "validated_window_indices": [0],
                "configuration_sha256": "c" * 64,
            },
        ):
            return materialize_bundle_package(self.source, destination)

    def test_materializer_stages_and_indexes_comparison_figures(self):
        self._write_source()
        configuration = self._configuration()
        destination = self.root / "bundle"
        with mock.patch(
            "exercise5_submission.plot_bundle_vs_pnp_comparisons",
            wraps=plot_bundle_vs_pnp_comparisons,
        ) as comparison_helper:
            observed = self._materialize(destination, configuration)

        self.assertEqual(observed, destination.resolve())
        comparison_helper.assert_called_once()
        manifest = json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8")
        )
        indexed = {entry["path"] for entry in manifest["artifacts"]}
        for name in COMPARISON_FILENAMES:
            relative = "figures/{}".format(name)
            self.assertIn(relative, indexed)
            image = mpl_image.imread(destination / relative)
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[1], 0)
        _validate_staged_curated_package(destination, configuration)
        self.assertFalse(any(self.root.glob(".bundle.attempt.*.tmp")))

    def test_materializer_fails_closed_on_invalid_report_csv(self):
        self._write_source(invalid_absolute=True)
        destination = self.root / "bundle"
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self._materialize(destination, self._configuration())
        self.assertFalse(destination.exists())
        self.assertFalse(any(self.root.glob(".bundle.attempt.*.tmp")))

    def test_curated_required_artifacts_remain_mandatory(self):
        attempt = self.root / "attempt"
        attempt.mkdir()
        configuration = {
            "artifact_contract": {
                "curated_required": {"required.csv": "fixture"}
            }
        }
        with self.assertRaisesRegex(
            ArtifactValidationError, "required.csv"
        ):
            _validate_staged_curated_package(attempt, configuration)


if __name__ == "__main__":
    unittest.main()
