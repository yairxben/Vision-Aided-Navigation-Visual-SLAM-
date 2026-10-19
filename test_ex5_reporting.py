"""Isolated report-only tests for Exercise 5 figure generation."""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e5_reporting_matplotlib_tests")

import matplotlib.axes
import numpy as np
from matplotlib import image as mpl_image

from exercise5_reporting import (
    _plot_two_series,
    generate_report_figures,
    plot_bundle_vs_pnp_comparisons,
)


class BundleVsPnpReportingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "aggregate" / "data"
        self.figures = self.root / "aggregate" / "figures"
        self.data.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path: Path, fieldnames, rows) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_valid_inputs(self) -> None:
        self._write_csv(
            self.data / "keyframe_localization_errors.csv",
            ("frame_id", "bundle_error_m", "pnp_error_m"),
            (
                {"frame_id": 0, "bundle_error_m": 0.0, "pnp_error_m": 0.0},
                {"frame_id": 7, "bundle_error_m": 1.0, "pnp_error_m": 2.0},
            ),
        )
        self._write_csv(
            self.data / "consecutive_relative_errors.csv",
            (
                "window_index",
                "bundle_translation_error_m",
                "pnp_translation_error_m",
                "bundle_rotation_error_deg",
                "pnp_rotation_error_deg",
            ),
            tuple(
                {
                    "window_index": index,
                    "bundle_translation_error_m": index + 0.1,
                    "pnp_translation_error_m": index + 0.2,
                    "bundle_rotation_error_deg": index + 0.3,
                    "pnp_rotation_error_deg": index + 0.4,
                }
                for index in range(3)
            ),
        )

    def _write_all_report_inputs(self) -> None:
        self._write_valid_inputs()
        self._write_csv(
            self.data / "window_metrics.csv",
            (
                "window_index",
                "final_mean_projection_factor_error",
                "final_median_stereo_pixel_distance_px",
            ),
            (
                {
                    "window_index": 0,
                    "final_mean_projection_factor_error": 1.0,
                    "final_median_stereo_pixel_distance_px": 0.5,
                },
            ),
        )
        self._write_csv(
            self.data / "all_projection_residuals.csv",
            (
                "pnp_path_distance_from_window_start_m",
                "final_stereo_pixel_distance_px",
            ),
            (
                {
                    "pnp_path_distance_from_window_start_m": 1.0,
                    "final_stereo_pixel_distance_px": 0.5,
                },
            ),
        )
        self._write_csv(
            self.data / "keyframe_trajectories.csv",
            (
                "bundle_x_m",
                "bundle_z_m",
                "pnp_x_m",
                "pnp_z_m",
                "ground_truth_x_m",
                "ground_truth_z_m",
            ),
            (
                {
                    "bundle_x_m": 0.0,
                    "bundle_z_m": 0.0,
                    "pnp_x_m": 0.0,
                    "pnp_z_m": 0.0,
                    "ground_truth_x_m": 0.0,
                    "ground_truth_z_m": 0.0,
                },
                {
                    "bundle_x_m": 1.0,
                    "bundle_z_m": 1.0,
                    "pnp_x_m": 1.0,
                    "pnp_z_m": 1.0,
                    "ground_truth_x_m": 1.0,
                    "ground_truth_z_m": 1.0,
                },
            ),
        )
        for length in (100, 400, 800):
            self._write_csv(
                self.data / "relative_subsections_{}.csv".format(length),
                ("start_frame_id", "translation_error_percent"),
                ({"start_frame_id": 0, "translation_error_percent": 1.0},),
            )
        first_window = self.root / "windows" / "window_0000_0_1"
        first_window.mkdir(parents=True)
        poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
        poses[1, 0, 3] = 1.0
        covariances = np.repeat(np.eye(6)[None, :, :], 2, axis=0)
        np.savez(
            first_window / "camera_marginals.npz",
            poses=poses,
            marginal_covariances=covariances,
        )

    def test_comparison_plots_use_every_input_row(self):
        self._write_valid_inputs()
        observed = []
        real_plot = matplotlib.axes.Axes.plot

        def capture_plot(axis, x_values, y_values, *args, **kwargs):
            observed.append(
                (
                    np.asarray(x_values).copy(),
                    np.asarray(y_values).copy(),
                    kwargs.get("label"),
                )
            )
            return real_plot(axis, x_values, y_values, *args, **kwargs)

        with mock.patch.object(
            matplotlib.axes.Axes, "plot", new=capture_plot
        ):
            outputs = plot_bundle_vs_pnp_comparisons(
                self.data, self.figures
            )

        self.assertEqual(
            tuple(path.name for path in outputs),
            (
                "bundle_vs_pnp_absolute_localization_error.png",
                "bundle_vs_pnp_relative_translation_error.png",
                "bundle_vs_pnp_relative_rotation_error.png",
            ),
        )
        self.assertEqual(len(observed), 6)
        np.testing.assert_array_equal(observed[0][0], [0.0, 7.0])
        for index in range(2, 6):
            np.testing.assert_array_equal(observed[index][0], [0.0, 1.0, 2.0])
        self.assertEqual(
            tuple(series[2] for series in observed),
            ("Bundle", "PnP", "Bundle", "PnP", "Bundle", "PnP"),
        )
        for path in outputs:
            image = mpl_image.imread(path)
            self.assertEqual(image.shape, (630, 1400, 4))
            self.assertTrue(np.all(np.isfinite(image)))

    def test_comparison_csv_validation_rejects_bad_inputs(self):
        with self.assertRaises(FileNotFoundError):
            plot_bundle_vs_pnp_comparisons(self.data, self.figures)

        self._write_valid_inputs()
        absolute_path = self.data / "keyframe_localization_errors.csv"
        self._write_csv(
            absolute_path,
            ("frame_id", "bundle_error_m", "pnp_error_m"),
            (),
        )
        with self.assertRaisesRegex(ValueError, "contains no rows"):
            plot_bundle_vs_pnp_comparisons(self.data, self.figures)

        self._write_csv(
            absolute_path,
            ("frame_id", "bundle_error_m"),
            ({"frame_id": 0, "bundle_error_m": 0.0},),
        )
        with self.assertRaisesRegex(ValueError, "required column"):
            plot_bundle_vs_pnp_comparisons(self.data, self.figures)

        self._write_csv(
            absolute_path,
            ("frame_id", "bundle_error_m", "pnp_error_m"),
            ({"frame_id": 0, "bundle_error_m": "inf", "pnp_error_m": 0.0},),
        )
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            plot_bundle_vs_pnp_comparisons(self.data, self.figures)

    def test_common_plot_rejects_mismatched_series_lengths(self):
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            _plot_two_series(
                [0.0, 1.0],
                [0.0],
                [0.0, 1.0],
                xlabel="x",
                ylabel="y",
                title="title",
                output_path=self.figures / "unused.png",
            )

    def test_generate_report_figures_produces_all_thirteen_figures(self):
        self._write_all_report_inputs()
        with mock.patch(
            "exercise5_reporting.plot_bundle_vs_pnp_comparisons",
            wraps=plot_bundle_vs_pnp_comparisons,
        ) as comparison_helper:
            outputs = generate_report_figures(self.root)

        comparison_helper.assert_called_once_with(self.data, self.figures)
        self.assertEqual(
            tuple(path.name for path in outputs),
            (
                "bundle_pnp_ground_truth_trajectory.png",
                "bundle_vs_pnp_absolute_localization_error.png",
                "bundle_vs_pnp_relative_rotation_error.png",
                "bundle_vs_pnp_relative_translation_error.png",
                "consecutive_relative_error.png",
                "first_window_camera_covariances_3d.png",
                "keyframe_localization_error.png",
                "mean_factor_error_by_window.png",
                "median_projection_error_by_window.png",
                "projection_error_vs_bundle_start.png",
                "relative_subsections_100.png",
                "relative_subsections_400.png",
                "relative_subsections_800.png",
            ),
        )
        for path in outputs:
            image = mpl_image.imread(path)
            self.assertGreater(image.shape[0], 0)
            self.assertGreater(image.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
