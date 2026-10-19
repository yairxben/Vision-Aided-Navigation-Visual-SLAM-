"""Focused E4-R03 analysis-presentation tests."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import exercise4_analysis as analysis


class AnalysisPublicContractTest(unittest.TestCase):
    def test_public_signatures_remain_unchanged(self):
        expected = {
            "load_tracking_artifacts": (
                "(database_path: 'Path', companion_path: 'Path') -> "
                "'Tuple[TrackingDB, TrackingDBCompanion, "
                "Dict[str, object]]'"
            ),
            "compute_database_metrics": (
                "(database: 'TrackingDB', typed_companion: "
                "'TrackingDBCompanion', transitions: "
                "'Sequence[Dict[str, object]]', analysis_dir: 'Path', "
                "supporter_threshold_percent: 'float') -> "
                "'Tuple[Dict[str, object], int, int, "
                "Dict[str, List[object]]]'"
            ),
            "compute_reprojection": (
                "(database: 'TrackingDB', track_id: 'int', "
                "analysis_dir: 'Path') -> 'Dict[str, object]'"
            ),
            "write_representative_track": (
                "(database: 'TrackingDB', track_id: 'int', "
                "analysis_dir: 'Path') -> 'Dict[str, object]'"
            ),
            "write_figures": (
                "(metrics: 'Dict[str, object]', reprojection: "
                "'Dict[str, object]', series: "
                "'Dict[str, List[object]]', analysis_dir: 'Path') -> "
                "'None'"
            ),
            "analyze_tracking_database": (
                "(database_path: 'Path', companion_path: 'Path', "
                "analysis_dir: 'Path', supporter_threshold_percent: "
                "'float' = 30.0) -> 'Dict[str, object]'"
            ),
        }
        for name, signature in expected.items():
            self.assertEqual(
                str(inspect.signature(getattr(analysis, name))), signature
            )


class AnalysisPipelineTest(unittest.TestCase):
    def test_public_analysis_path_exposes_course_stage_order(self):
        order = []
        database = object()
        companion = object()
        normalized = {"transitions": [object()]}
        metrics = {"metrics": True}
        series = {"series": []}
        representative = {"frames": [0]}
        reprojection = {"track_id": 10}
        finalized = {"metrics": metrics}

        def loaded(*args):
            order.append("load")
            return database, companion, normalized

        def computed(*args):
            order.append("statistics-and-tables")
            return metrics, 6, 10, series

        def representative_written(*args):
            order.append("representative-track")
            return representative

        def reprojection_computed(*args):
            order.append("reprojection")
            return reprojection

        def figures_written(*args):
            order.append("figures")

        def finalized_artifacts(**kwargs):
            order.append("finalization")
            return finalized

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "portable-output" / "analysis"
            database_path = root / "inputs" / "tracking.pkl"
            companion_path = root / "inputs" / "companion.json"
            with mock.patch.object(
                analysis, "load_tracking_artifacts", side_effect=loaded
            ) as load, mock.patch.object(
                analysis, "compute_database_metrics", side_effect=computed
            ), mock.patch.object(
                analysis,
                "write_representative_track",
                side_effect=representative_written,
            ), mock.patch.object(
                analysis,
                "compute_reprojection",
                side_effect=reprojection_computed,
            ), mock.patch.object(
                analysis, "write_figures", side_effect=figures_written
            ), mock.patch.object(
                analysis,
                "_finalize_analysis_artifacts",
                side_effect=finalized_artifacts,
            ):
                result = analysis.analyze_tracking_database(
                    database_path,
                    companion_path,
                    output,
                    supporter_threshold_percent=25.0,
                )

        self.assertIs(result, finalized)
        self.assertEqual(
            order,
            [
                "load",
                "statistics-and-tables",
                "representative-track",
                "reprojection",
                "figures",
                "finalization",
            ],
        )
        load.assert_called_once_with(database_path, companion_path)


class MetricTableTest(unittest.TestCase):
    def test_metric_table_filenames_columns_and_write_order_are_fixed(self):
        track_statistics = {
            "histogram_rows": [object()],
        }
        frame_statistics = {
            "frame_rows": [object()],
            "transition_rows": [object()],
            "below_threshold": [object()],
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            analysis,
            "_compute_track_statistics",
            return_value=track_statistics,
        ) as compute_tracks, mock.patch.object(
            analysis,
            "_compute_frame_and_transition_statistics",
            return_value=frame_statistics,
        ), mock.patch.object(analysis, "write_csv") as write_csv:
            output = Path(temporary)
            observed = analysis._write_database_metric_tables(
                database=object(),
                transitions=[object()],
                analysis_dir=output,
                supporter_threshold_percent=30.0,
            )
            track_lengths_text = (
                output / "track_lengths.csv"
            ).read_text(encoding="utf-8")

        self.assertEqual(observed, (track_statistics, frame_statistics))
        self.assertTrue(compute_tracks.call_args.args[1])
        self.assertEqual(
            track_lengths_text,
            "track_id,length,first_frame,last_frame\n",
        )
        self.assertEqual(
            write_csv.call_args_list,
            [
                mock.call(
                    output / "track_length_histogram.csv",
                    analysis.HISTOGRAM_FIELDS,
                    track_statistics["histogram_rows"],
                ),
                mock.call(
                    output / "frame_links.csv",
                    analysis.FRAME_LINK_FIELDS,
                    frame_statistics["frame_rows"],
                ),
                mock.call(
                    output / "transition_metrics.csv",
                    analysis.TRANSITION_FIELDS,
                    frame_statistics["transition_rows"],
                ),
                mock.call(
                    output / "below_threshold_transitions.csv",
                    analysis.BELOW_THRESHOLD_FIELDS,
                    frame_statistics["below_threshold"],
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
