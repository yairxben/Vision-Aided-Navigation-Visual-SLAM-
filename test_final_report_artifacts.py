"""Focused reporting-only tests for final Project Submission artifacts."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project_submission import finalize_report_artifacts as final


def _write_rows(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RelativeSubsectionReportingTests(unittest.TestCase):
    def test_combined_two_panel_plot_and_means(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for length in (100, 400, 800):
                path = root / "{}.csv".format(length)
                paths[length] = path
                _write_rows(
                    path,
                    (
                        "start_frame",
                        "location_error_percent",
                        "angle_error_per_meter_deg_per_m",
                    ),
                    (
                        {
                            "start_frame": 0,
                            "location_error_percent": length / 100.0,
                            "angle_error_per_meter_deg_per_m": length / 10000.0,
                        },
                        {
                            "start_frame": 1,
                            "location_error_percent": 2.0 * length / 100.0,
                            "angle_error_per_meter_deg_per_m": 2.0 * length / 10000.0,
                        },
                    ),
                )
            output = root / "combined.png"
            summary = final.plot_relative_subsection_comparison(
                paths, output, stage_label="synthetic"
            )
            self.assertGreater(output.stat().st_size, 1000)
            self.assertAlmostEqual(
                summary["100"]["mean_translation_error_percent"], 1.5
            )
            self.assertAlmostEqual(
                summary["800"]["mean_rotation_error_deg_per_m"], 0.12
            )


class BundleReportingTests(unittest.TestCase):
    def test_window_error_denominator_and_projection_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "window.csv"
            fields = (
                "window_index",
                "start_frame_id",
                "projection_factor_count",
                "initial_total_graph_error",
                "final_total_graph_error",
                "initial_median_stereo_pixel_distance_px",
                "final_median_stereo_pixel_distance_px",
            )
            _write_rows(
                metrics,
                fields,
                (
                    dict(zip(fields, (0, 100, 9, 100, 10, 4, 1))),
                    dict(zip(fields, (1, 400, 19, 200, 20, 6, 2))),
                ),
            )
            with mock.patch.object(
                final.plt, "plot", wraps=final.plt.plot
            ) as plot, mock.patch.object(
                final.plt, "xlabel", wraps=final.plt.xlabel
            ) as xlabel:
                summary = final.plot_bundle_window_report(
                    metrics, root / "factor.png", root / "median.png"
                )
            self.assertEqual(len(plot.call_args_list), 4)
            for call in plot.call_args_list:
                self.assertEqual(call.args[0].tolist(), [100.0, 400.0])
            self.assertEqual(
                [call.args[0] for call in xlabel.call_args_list],
                [
                    "Bundle starting keyframe FrameId [frame]",
                    "Bundle starting keyframe FrameId [frame]",
                ],
            )
            self.assertEqual(summary["window_count"], 2)
            self.assertAlmostEqual(summary["initial_mean_error_per_factor"], 10.0)
            self.assertAlmostEqual(summary["final_mean_error_per_factor"], 1.0)

            residuals = root / "residuals.csv"
            residual_fields = (
                "ground_truth_path_distance_from_window_start_m",
                "initial_stereo_pixel_distance_px",
                "final_stereo_pixel_distance_px",
            )
            _write_rows(
                residuals,
                residual_fields,
                (
                    dict(zip(residual_fields, (0.2, 4.0, 2.0))),
                    dict(zip(residual_fields, (0.8, 2.0, 1.0))),
                    dict(zip(residual_fields, (1.2, 6.0, 3.0))),
                ),
            )
            aggregate = root / "aggregate.csv"
            record = final.aggregate_bundle_projection_by_distance(
                residuals, aggregate
            )
            self.assertEqual(record["input_observation_count"], 3)
            self.assertEqual(record["distance_bin_count"], 2)
            rows = final._read_csv(aggregate)
            self.assertAlmostEqual(
                float(rows[0]["mean_final_stereo_projection_error_px"]), 1.5
            )


class PublicationSafetyTests(unittest.TestCase):
    def test_scientific_and_historical_roots_are_refused(self) -> None:
        for target in (final.E5_ROOT, final.E7_ROOT, final.HISTORICAL_ROOTS[-1]):
            with self.assertRaisesRegex(
                final.FinalReportError, "refused|protected evidence"
            ):
                final._safe_final_report_target(target)

    def test_final_code_locations_are_current_and_hash_bound(self) -> None:
        record = final.build_code_locations()
        self.assertEqual(record["status"], "PASS")
        self.assertGreaterEqual(len(record["scientific_stage_locations"]), 11)
        self.assertGreaterEqual(len(record["figure_generator_locations"]), 9)


if __name__ == "__main__":
    unittest.main()
