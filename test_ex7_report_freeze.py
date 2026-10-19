"""Focused lightweight tests for the E7 report freeze and thin entrypoint."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import ex7
import exercise7_report_freeze as report


def _poses(count: int) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None, :, :], count, axis=0)
    poses[:, 0, 3] = np.arange(count, dtype=float)
    poses[:, 2, 3] = np.arange(count, dtype=float) * 2.0
    return poses


class ThinEntrypointTests(unittest.TestCase):
    def test_report_dispatch_is_thin(self) -> None:
        e5_root = Path("/tmp/example-e5")
        e6_root = Path("/tmp/example-e6")
        scientific_root = Path("/tmp/example-scientific")
        arguments = argparse.Namespace(
            command="report",
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
            report_root=Path("/tmp/example-report"),
            submission_root=Path("/tmp/example-submission"),
            confirm="",
        )
        with mock.patch.object(
            ex7, "generate_report_freeze", return_value={"status": "pass"}
        ) as generate:
            self.assertEqual(
                ex7.dispatch_command(arguments), {"status": "pass"}
            )
        generate.assert_called_once_with(
            output_root=Path("/tmp/example-report"),
            e5_root=e5_root,
            e6_root=e6_root,
            scientific_root=scientific_root,
        )

    def test_preflight_and_run_forward_all_selected_roots(self) -> None:
        roots = {
            "e5_root": Path("/tmp/selected-e5"),
            "e6_root": Path("/tmp/selected-e6"),
            "scientific_root": Path("/tmp/selected-e7"),
        }
        preflight_arguments = argparse.Namespace(
            command="preflight",
            report_root=report.DEFAULT_REPORT_ROOT,
            submission_root=report.DEFAULT_SUBMISSION_ROOT,
            confirm="",
            **roots
        )
        with mock.patch.object(
            ex7, "preflight_exercise7", return_value={"status": "pass"}
        ) as preflight:
            ex7.dispatch_command(preflight_arguments)
        preflight.assert_called_once_with(**roots)

        run_arguments = argparse.Namespace(
            command="run",
            report_root=report.DEFAULT_REPORT_ROOT,
            submission_root=report.DEFAULT_SUBMISSION_ROOT,
            confirm=ex7.CANONICAL_RUN_CONFIRMATION,
            **roots
        )
        with mock.patch(
            "exercise7_sequential_loop_closure.run_sequential_loop_closure",
            return_value={"status": "pass"},
        ) as run:
            ex7.dispatch_command(run_arguments)
        run.assert_called_once_with(
            output_root=roots["scientific_root"],
            e5_root=roots["e5_root"],
            e6_root=roots["e6_root"],
        )

    def test_run_requires_exact_confirmation(self) -> None:
        arguments = argparse.Namespace(
            command="run",
            report_root=report.DEFAULT_REPORT_ROOT,
            submission_root=report.DEFAULT_SUBMISSION_ROOT,
            confirm="wrong",
        )
        with self.assertRaisesRegex(RuntimeError, "run refused"):
            ex7.dispatch_command(arguments)

    def test_clean_publish_defaults_do_not_target_historical_packages(self):
        repository_root = Path(__file__).resolve().parent
        self.assertEqual(
            report.DEFAULT_REPORT_ROOT,
            repository_root / "diagnostics/ex7_clean_final/report_freeze",
        )
        self.assertEqual(
            report.DEFAULT_SUBMISSION_ROOT,
            repository_root
            / "project_submission/loop_closure_clean_final",
        )
        self.assertEqual(
            report._require_safe_publish_target(report.DEFAULT_REPORT_ROOT),
            report.DEFAULT_REPORT_ROOT.resolve(),
        )
        self.assertEqual(
            report._require_safe_publish_target(report.DEFAULT_SUBMISSION_ROOT),
            report.DEFAULT_SUBMISSION_ROOT.resolve(),
        )
        for protected in (
            repository_root / "diagnostics/ex7_i04/report_freeze",
            repository_root / "diagnostics/ex7_clean_equivalence",
            repository_root / "diagnostics/ex7_clean_equivalence/nested",
            repository_root / "project_submission/bundle",
            repository_root / "project_submission/bundle_clean_final",
            repository_root / "project_submission/bundle_clean_final/nested",
            repository_root / "project_submission/loop_closure",
        ):
            with self.subTest(protected=protected), self.assertRaisesRegex(
                report.ReportFreezeError,
                "protected scientific/historical evidence",
            ):
                report._require_safe_publish_target(protected)


class ReportMathAndSelectionTests(unittest.TestCase):
    def test_absolute_components_norm_and_rotation(self) -> None:
        truth = _poses(2)
        estimate = truth.copy()
        estimate[1, :3, 3] += [3.0, -4.0, 12.0]
        angle = np.radians(90.0)
        estimate[1, :3, :3] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        metrics = report.compute_absolute_pose_metrics(estimate, truth)
        np.testing.assert_allclose(metrics["absolute_xyz_m"][1], [3, 4, 12])
        self.assertAlmostEqual(metrics["location_norm_m"][1], 13.0)
        self.assertAlmostEqual(metrics["rotation_error_deg"][1], 90.0)

    def test_snapshot_and_edges(self) -> None:
        rows = [
            {
                "acceptance_id": index,
                "previous_frame_id": index,
                "current_frame_id": index + 10,
            }
            for index in range(1, 7)
        ]
        self.assertEqual(report.select_snapshot_state_ids(6), (0, 1, 3, 6))
        self.assertEqual(
            report.accepted_edges_for_state(rows, 3),
            ((1, 11), (2, 12), (3, 13)),
        )

    def test_every_snapshot_retains_every_accepted_loop_endpoint(self) -> None:
        frame_ids = np.arange(20, dtype=np.int64)
        loops = [
            {
                "acceptance_id": index,
                "previous_frame_id": previous,
                "current_frame_id": current,
            }
            for index, (previous, current) in enumerate(
                ((3, 11), (5, 17), (2, 14), (6, 18), (1, 15), (7, 19)),
                start=1,
            )
        ]
        selected = report._covariance_display_indices(
            frame_ids, loops, subsample_stride=4
        )
        expected_endpoints = {1, 2, 3, 5, 6, 7, 11, 14, 15, 17, 18, 19}
        self.assertTrue(expected_endpoints.issubset(selected))
        self.assertTrue({0, 4, 8, 12, 16, 19}.issubset(selected))
        self.assertEqual(report.accepted_edges_for_state(loops, 0), ())
        self.assertEqual(
            report.accepted_edges_for_state(loops, 1), ((3, 11),)
        )
        self.assertEqual(len(report.accepted_edges_for_state(loops, 3)), 3)
        self.assertEqual(len(report.accepted_edges_for_state(loops, 6)), 6)

        poses = _poses(len(frame_ids))
        marginals = np.repeat(
            np.eye(6, dtype=float)[None, :, :], len(frame_ids), axis=0
        )
        states = {
            state_id: (poses.copy(), marginals.copy())
            for state_id in report.SNAPSHOT_ACCEPTANCE_IDS
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            report, "_add_covariance_ellipse"
        ) as add_ellipse:
            report.plot_pose_graph_snapshot_covariances(
                Path(directory) / "snapshots.png",
                frame_ids=frame_ids,
                states=states,
                accepted_loops=loops,
                subsample_stride=4,
            )
        self.assertEqual(add_ellipse.call_count, 4 * len(selected))
        for state_index in range(4):
            calls = add_ellipse.call_args_list[
                state_index * len(selected) : (state_index + 1) * len(selected)
            ]
            displayed_x = {int(call.args[1][0]) for call in calls}
            self.assertTrue(expected_endpoints.issubset(displayed_x))

    def test_accepted_loop_summary_preserves_sealed_populations(self) -> None:
        row = {
            "acceptance_id": "2",
            "previous_frame_id": "327",
            "current_frame_id": "2365",
            "previous_keyframe_ordinal": "44",
            "current_keyframe_ordinal": "253",
            "reciprocal_temporal_match_count": "566",
            "consensus_count": "192",
            "supporter_count": "85",
            "supporter_percentage": "44.270833333333336",
            "ransac_terminal_status": "adaptive_completed",
            "eligible_landmark_count": "85",
            "state_version_after": "2",
        }
        summary = report._accepted_loop_summary_record(row)
        self.assertEqual(summary["previous_keyframe_ordinal"], 44)
        self.assertEqual(summary["current_keyframe_ordinal"], 253)
        self.assertEqual(summary["reciprocal_temporal_match_count"], 566)
        self.assertEqual(summary["consensus_count"], 192)
        self.assertEqual(summary["supporter_count"], 85)
        self.assertAlmostEqual(
            summary["supporter_percentage"], 100.0 * 85.0 / 192.0
        )
        self.assertEqual(
            summary["ransac_terminal_status"], "adaptive_completed"
        )

    def test_alignment_failure_is_exact(self) -> None:
        with self.assertRaisesRegex(
            report.ReportFreezeError,
            "synthetic FrameIds differ from canonical keyframe order",
        ):
            report.require_aligned_frame_ids([0, 7, 11], [0, 8, 11], "synthetic")

    def test_small_synthetic_report_figures(self) -> None:
        frame_ids = np.asarray([0, 5, 10, 15], dtype=np.int64)
        poses = _poses(4)
        marginals = np.repeat(np.eye(6)[None, :, :] * 0.01, 4, axis=0)
        loops = [
            {
                "acceptance_id": index,
                "previous_frame_id": 0,
                "current_frame_id": 15,
            }
            for index in range(1, 7)
        ]
        states = {
            state_id: (poses.copy(), marginals.copy())
            for state_id in (0, 1, 3, 6)
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshots.png"
            report.plot_pose_graph_snapshot_covariances(
                path,
                frame_ids=frame_ids,
                states=states,
                accepted_loops=loops,
                subsample_stride=2,
            )
            self.assertGreater(path.stat().st_size, 1000)


class ManifestAndSubmissionTests(unittest.TestCase):
    def test_manifest_detects_checksum_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "value.txt"
            artifact.write_text("one", encoding="utf-8")
            report._write_json(
                root / "manifest.json",
                report._manifest_for(root, "synthetic-manifest-v1"),
            )
            self.assertEqual(
                report.validate_checksum_manifest(root)["status"], "pass"
            )
            artifact.write_text("two", encoding="utf-8")
            with self.assertRaisesRegex(
                report.ReportFreezeError, "checksum mismatch"
            ):
                report.validate_checksum_manifest(root)

    def test_curated_submission_omits_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            report_root = temporary / "report"
            output_root = temporary / "submission"
            scientific_root = temporary / "scientific"
            (scientific_root / "final").mkdir(parents=True)
            (scientific_root / "final/manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (report_root / "figures").mkdir(parents=True)
            (report_root / "data").mkdir()
            figure_names = (
                "first_accepted_successful_match.png",
                "first_accepted_successful_match_non_supporters_emphasized.png",
                "pose_graph_snapshots_raw_95_covariance.png",
                "pose_graph_snapshots_dense_raw_95_covariance.png",
                "pose_graph_snapshots_dense_x5_radius.png",
                "pose_graph_snapshots_dense_x10_radius.png",
                "pose_graph_snapshot_uncertainty_heatmap.png",
                "trajectory_ground_truth_e6_e7.png",
                "absolute_location_errors.png",
                "absolute_rotation_errors.png",
                "location_uncertainty.png",
                "angular_uncertainty.png",
                "accepted_loop_statistics.png",
            )
            for name in figure_names:
                (report_root / "figures" / name).write_bytes(b"png")
            for name in (
                "accepted_loops.csv",
                "accepted_loop_summary.csv",
                "trajectory_comparison.csv",
                "absolute_pose_errors.csv",
                "marginal_uncertainty_sizes.csv",
                "snapshot_selection.json",
            ):
                (report_root / "data" / name).write_text("x\n", encoding="utf-8")
            (report_root / "report_notes.md").write_text("notes\n", encoding="utf-8")
            report._write_json(
                report_root / "traceability.json",
                {
                    "status": "complete",
                    "report_freeze_line_refresh": "complete",
                },
            )
            report._write_json(
                report_root / "requirements_mapping.json",
                {"status": "complete"},
            )
            (report_root / "requirements_mapping.md").write_text("map\n", encoding="utf-8")
            report._write_json(
                report_root / "summary.json",
                {
                    "schema": "exercise7-report-freeze-summary-v1",
                    "status": "pass",
                    "traceability_status": "complete",
                    "counts": {"accepted_loops": 6},
                },
            )
            report._write_json(
                report_root / "provenance.json",
                {
                    "schema": "exercise7-report-freeze-provenance-v1",
                    "selected_roots": {
                        "e5_root": str((temporary / "e5").resolve()),
                        "e6_root": str((temporary / "e6").resolve()),
                        "scientific_root": str(scientific_root.resolve()),
                    },
                },
            )
            report._write_json(
                report_root / "manifest.json",
                report._manifest_for(
                    report_root, "exercise7-report-freeze-manifest-v1"
                ),
            )
            evidence = {
                "truth_xyz": _poses(3)[:, :3, 3],
                "pnp_xyz": _poses(3)[:, :3, 3],
                "bundle_xyz": _poses(3)[:, :3, 3],
                "states": {
                    0: (_poses(3), np.zeros((3, 6, 6))),
                    6: (_poses(3), np.zeros((3, 6, 6))),
                },
            }
            with mock.patch.object(
                report, "_load_canonical_evidence", return_value=evidence
            ), mock.patch.object(
                report,
                "validate_report_freeze",
                return_value={"status": "pass"},
            ):
                result = report.materialize_loop_closure_submission(
                    report_root=report_root,
                    output_root=output_root,
                    e5_root=temporary / "e5",
                    e6_root=temporary / "e6",
                    scientific_root=scientific_root,
                )
            self.assertEqual(result["status"], "pass")
            self.assertFalse(any("currents" in str(path) for path in output_root.rglob("*")))
            self.assertTrue(
                (output_root / "data/marginal_uncertainty_sizes.csv").is_file()
            )
            self.assertLess(
                sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file()),
                5_000_000,
            )


if __name__ == "__main__":
    unittest.main()
