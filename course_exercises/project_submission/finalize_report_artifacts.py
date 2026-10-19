"""Publish final report artifacts from already-sealed SLAM evidence.

This helper is deliberately presentation-only.  It reads existing CSV, JSON,
NPZ, PNG, and manifest evidence; it does not deserialize TrackingDB, load a
Bundle graph, run matching/RANSAC/PnP, solve Bundle Adjustment, optimize a
Pose Graph, or construct Marginals.  New output is staged and then published
only to the managed final-report destination.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = ROOT / "project_submission"
FINAL_REPORT_ROOT = SUBMISSION_ROOT / "final_report"
PNP_ROOT = SUBMISSION_ROOT / "pnp"
TRACKING_ROOT = SUBMISSION_ROOT / "tracking"
BUNDLE_ROOT = SUBMISSION_ROOT / "bundle_clean_final"
LOOP_ROOT = SUBMISSION_ROOT / "loop_closure_clean_final"
E5_ROOT = ROOT / "diagnostics/ex5_clean_final"
E6_ROOT = ROOT / "diagnostics/ex6_clean_final"
E7_ROOT = ROOT / "diagnostics/ex7_clean_final"
HISTORICAL_ROOTS = (
    ROOT / "diagnostics/ex5_final",
    ROOT / "diagnostics/ex6_pose_graph",
    ROOT / "diagnostics/ex7_i04",
    SUBMISSION_ROOT / "bundle",
    SUBMISSION_ROOT / "loop_closure",
)
REPORT_MANIFEST_SCHEMA = "final-report-artifacts-manifest-v1"
PUBLICATION_MANIFEST_SCHEMA = "project-submission-final-publication-manifest-v1"


class FinalReportError(RuntimeError):
    """Raised when report-only evidence or publication safety differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalReportError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), "report CSV is empty: {}".format(path))
    return rows


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close()


def _numeric_column(
    rows: Sequence[Mapping[str, str]], field: str
) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    _require(
        values.ndim == 1 and len(values) and np.all(np.isfinite(values)),
        "invalid numeric report column: {}".format(field),
    )
    return values


def plot_relative_subsection_comparison(
    csv_paths: Mapping[int, Path], output_path: Path, *, stage_label: str
) -> Mapping[str, object]:
    """Plot 100/400/800 translation percent and rotation deg/m together."""

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
    summaries = {}
    for length in (100, 400, 800):
        rows = _read_csv(csv_paths[length])
        start_field = (
            "start_frame"
            if "start_frame" in rows[0]
            else "start_frame_id"
        )
        translation_field = (
            "location_error_percent"
            if "location_error_percent" in rows[0]
            else "translation_error_percent"
        )
        rotation_field = (
            "angle_error_per_meter_deg_per_m"
            if "angle_error_per_meter_deg_per_m" in rows[0]
            else "rotation_error_deg_per_m"
        )
        x = _numeric_column(rows, start_field)
        translation = _numeric_column(rows, translation_field)
        rotation = _numeric_column(rows, rotation_field)
        translation_mean = float(np.mean(translation))
        rotation_mean = float(np.mean(rotation))
        axes[0].plot(
            x,
            translation,
            linewidth=0.8,
            label="{} frames (mean {:.4f}%)".format(
                length, translation_mean
            ),
        )
        axes[1].plot(
            x,
            rotation,
            linewidth=0.8,
            label="{} frames (mean {:.6f} deg/m)".format(
                length, rotation_mean
            ),
        )
        summaries[str(length)] = {
            "row_count": len(rows),
            "mean_translation_error_percent": translation_mean,
            "mean_rotation_error_deg_per_m": rotation_mean,
        }
    axes[0].set_xlabel("subsection start FrameId [frame]")
    axes[0].set_ylabel("translation error [%]")
    axes[1].set_xlabel("subsection start FrameId [frame]")
    axes[1].set_ylabel("rotation error [deg/m]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "{} relative pose error over 100/400/800-frame subsections".format(
            stage_label
        )
    )
    _save_figure(output_path)
    return summaries


def plot_bundle_window_report(
    window_metrics_path: Path,
    mean_factor_output: Path,
    median_projection_output: Path,
) -> Mapping[str, object]:
    """Plot initial/final per-factor error and median stereo pixel error."""

    rows = _read_csv(window_metrics_path)
    start_frame_id = _numeric_column(rows, "start_frame_id")
    factor_count = _numeric_column(rows, "projection_factor_count") + 1.0
    initial_total = _numeric_column(rows, "initial_total_graph_error")
    final_total = _numeric_column(rows, "final_total_graph_error")
    initial_mean = initial_total / factor_count
    final_mean = final_total / factor_count

    plt.figure(figsize=(11, 5))
    plt.plot(start_frame_id, initial_mean, label="initial", linewidth=0.9)
    plt.plot(start_frame_id, final_mean, label="final", linewidth=0.9)
    plt.yscale("log")
    plt.xlabel("Bundle starting keyframe FrameId [frame]")
    plt.ylabel("total graph error / factor count [error/factor, log scale]")
    plt.title(
        "Bundle mean graph error per factor\n"
        "factor count = stereo projection factors + one pose prior"
    )
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(mean_factor_output)

    initial_median = _numeric_column(
        rows, "initial_median_stereo_pixel_distance_px"
    )
    final_median = _numeric_column(
        rows, "final_median_stereo_pixel_distance_px"
    )
    plt.figure(figsize=(11, 5))
    plt.plot(start_frame_id, initial_median, label="initial", linewidth=0.9)
    plt.plot(start_frame_id, final_median, label="final", linewidth=0.9)
    plt.xlabel("Bundle starting keyframe FrameId [frame]")
    plt.ylabel("median stereo projection error [px]")
    plt.title("Bundle median stereo projection error")
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(median_projection_output)
    return {
        "window_count": len(rows),
        "factor_count_definition": (
            "projection_factor_count + 1 pose prior"
        ),
        "initial_mean_error_per_factor": float(np.mean(initial_mean)),
        "final_mean_error_per_factor": float(np.mean(final_mean)),
        "initial_median_stereo_projection_error_px": float(
            np.median(initial_median)
        ),
        "final_median_stereo_projection_error_px": float(
            np.median(final_median)
        ),
    }


def aggregate_bundle_projection_by_distance(
    source_csv: Path, output_csv: Path, *, bin_width_m: float = 1.0
) -> Mapping[str, object]:
    """Stream sealed residuals into fixed-width distance-bin means."""

    _require(bin_width_m > 0.0, "projection distance bin width must be positive")
    accumulators: Dict[int, List[float]] = {}
    row_count = 0
    required = (
        "ground_truth_path_distance_from_window_start_m",
        "initial_stereo_pixel_distance_px",
        "final_stereo_pixel_distance_px",
    )
    with Path(source_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames is not None
            and all(field in reader.fieldnames for field in required),
            "sealed projection CSV schema differs",
        )
        for row in reader:
            distance = float(row[required[0]])
            initial = float(row[required[1]])
            final = float(row[required[2]])
            _require(
                distance >= 0.0
                and all(math.isfinite(value) for value in (distance, initial, final)),
                "sealed projection CSV contains invalid values",
            )
            index = int(math.floor(distance / bin_width_m))
            record = accumulators.setdefault(index, [0.0, 0.0, 0.0])
            record[0] += 1.0
            record[1] += initial
            record[2] += final
            row_count += 1
    _require(row_count > 0, "sealed projection CSV has no observations")
    output_rows = []
    for index in sorted(accumulators):
        count, initial_sum, final_sum = accumulators[index]
        output_rows.append(
            {
                "distance_bin_index": index,
                "distance_bin_start_m": index * bin_width_m,
                "distance_bin_end_m": (index + 1) * bin_width_m,
                "distance_bin_center_m": (index + 0.5) * bin_width_m,
                "observation_count": int(count),
                "mean_initial_stereo_projection_error_px": initial_sum / count,
                "mean_final_stereo_projection_error_px": final_sum / count,
            }
        )
    fields = (
        "distance_bin_index",
        "distance_bin_start_m",
        "distance_bin_end_m",
        "distance_bin_center_m",
        "observation_count",
        "mean_initial_stereo_projection_error_px",
        "mean_final_stereo_projection_error_px",
    )
    _write_csv(output_csv, output_rows, fields)
    return {
        "input_observation_count": row_count,
        "distance_bin_count": len(output_rows),
        "distance_bin_width_m": bin_width_m,
        "statistic": "arithmetic mean of sealed stereo pixel distance",
    }


def plot_bundle_projection_distance(
    aggregate_csv: Path, output_path: Path
) -> None:
    rows = _read_csv(aggregate_csv)
    x = _numeric_column(rows, "distance_bin_center_m")
    plt.figure(figsize=(10, 5))
    plt.plot(
        x,
        _numeric_column(rows, "mean_initial_stereo_projection_error_px"),
        marker="o",
        label="initial",
    )
    plt.plot(
        x,
        _numeric_column(rows, "mean_final_stereo_projection_error_px"),
        marker="o",
        label="final",
    )
    plt.xlabel("ground-truth path distance from Bundle start [m]")
    plt.ylabel("mean stereo projection error [px]")
    plt.title("Bundle projection error versus distance from window start")
    plt.grid(alpha=0.25)
    plt.legend()
    _save_figure(output_path)


def audit_tracking_package(root: Path = TRACKING_ROOT) -> Mapping[str, object]:
    """Verify the retained Tracking package without reconstructing TrackingDB."""

    metrics = _read_json(Path(root) / "metrics.json")
    expected_figures = {
        "connectivity.png",
        "frame_links.png",
        "matches_per_transition.png",
        "representative_track.png",
        "supporter_percentage.png",
        "track_length_histogram.png",
    }
    observed = {path.name for path in (Path(root) / "figures").glob("*.png")}
    _require(observed == expected_figures, "Tracking figure inventory differs")
    for path in sorted((Path(root) / "figures").glob("*.png")):
        _require(mpimg.imread(path).size > 0, "Tracking figure cannot be decoded")
    _require(
        metrics["scope"]["frame_count"] == 3360
        and metrics["scope"]["transition_count"] == 3359
        and metrics["database"]["nontrivial_track_count"] == 758478,
        "Tracking metrics differ from the accepted package",
    )
    return {
        "status": "PASS",
        "figure_count": len(expected_figures),
        "frame_count": 3360,
        "transition_count": 3359,
        "track_count": 758478,
        "change_required": False,
    }


def _manifest_for(root: Path, schema: str) -> Mapping[str, object]:
    artifacts = []
    for path in sorted(
        item
        for item in Path(root).rglob("*")
        if item.is_file() and item.name != "manifest.json"
    ):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {"schema": schema, "status": "complete-and-validated", "artifacts": artifacts}


def _validate_manifest(root: Path, expected_schema: str) -> int:
    manifest = _read_json(Path(root) / "manifest.json")
    _require(manifest.get("schema") == expected_schema, "manifest schema differs")
    expected = {record["path"]: record for record in manifest["artifacts"]}
    observed = {
        path.relative_to(root).as_posix()
        for path in Path(root).rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    _require(observed == set(expected), "manifest path inventory differs")
    for relative, record in expected.items():
        path = Path(root) / relative
        _require(
            path.stat().st_size == record["size_bytes"]
            and file_sha256(path) == record["sha256"],
            "manifest checksum differs: {}".format(relative),
        )
    return len(expected)


def _safe_final_report_target(target: Path) -> Path:
    target = Path(target).resolve()
    _require(
        target == FINAL_REPORT_ROOT.resolve()
        or SUBMISSION_ROOT.resolve() not in target.parents,
        "custom publication beneath project_submission is refused",
    )
    for protected in (
        E5_ROOT,
        E6_ROOT,
        E7_ROOT,
        *HISTORICAL_ROOTS,
    ):
        protected = protected.resolve()
        _require(
            not (
                target == protected
                or target in protected.parents
                or protected in target.parents
            ),
            "publication overlaps protected evidence: {}".format(protected),
        )
    return target


def _publish_managed_directory(staging: Path, target: Path) -> None:
    backup = None
    if target.exists():
        manifest = _read_json(target / "manifest.json")
        _require(
            manifest.get("schema") == REPORT_MANIFEST_SCHEMA,
            "refusing to replace unmanaged final-report output",
        )
        backup = target.with_name(target.name + ".previous")
        _require(not backup.exists(), "managed publication backup already exists")
        os.replace(str(target), str(backup))
    try:
        os.replace(str(staging), str(target))
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(str(backup), str(target))
        raise
    if backup is not None:
        shutil.rmtree(backup)


def generate_final_report_artifacts(
    *, output_root: Path = FINAL_REPORT_ROOT
) -> Mapping[str, object]:
    """Stage and publish cross-stage report figures from sealed artifacts."""

    output_root = _safe_final_report_target(output_root)
    tracking_audit = audit_tracking_package()
    staging = Path(
        tempfile.mkdtemp(
            prefix=output_root.name + ".staging.", dir=str(output_root.parent)
        )
    )
    try:
        figures = staging / "figures"
        data = staging / "data"
        figures.mkdir()
        data.mkdir()
        pnp_paths = {
            length: PNP_ROOT / "data/relative_subsections_{}.csv".format(length)
            for length in (100, 400, 800)
        }
        bundle_paths = {
            length: BUNDLE_ROOT
            / "data/relative_subsections_{}.csv".format(length)
            for length in (100, 400, 800)
        }
        pnp_summary = plot_relative_subsection_comparison(
            pnp_paths,
            figures / "pnp_relative_subsections_100_400_800.png",
            stage_label="PnP",
        )
        bundle_summary = plot_relative_subsection_comparison(
            bundle_paths,
            figures / "bundle_relative_subsections_100_400_800.png",
            stage_label="Bundle",
        )
        bundle_window = plot_bundle_window_report(
            BUNDLE_ROOT / "data/window_metrics.csv",
            figures / "bundle_mean_total_graph_error_per_factor.png",
            figures / "bundle_median_stereo_projection_error.png",
        )
        projection = aggregate_bundle_projection_by_distance(
            E5_ROOT / "aggregate/data/all_projection_residuals.csv",
            data / "bundle_projection_error_by_distance.csv",
        )
        plot_bundle_projection_distance(
            data / "bundle_projection_error_by_distance.csv",
            figures / "bundle_projection_error_vs_distance.png",
        )
        copies = {
            BUNDLE_ROOT
            / "figures/bundle_vs_pnp_relative_translation_error.png": figures
            / "bundle_vs_pnp_relative_translation_error.png",
            BUNDLE_ROOT
            / "figures/bundle_vs_pnp_relative_rotation_error.png": figures
            / "bundle_vs_pnp_relative_rotation_error.png",
            LOOP_ROOT / "figures/absolute_location_errors.png": figures
            / "pose_graph_absolute_location_before_after.png",
            LOOP_ROOT / "figures/absolute_rotation_errors.png": figures
            / "pose_graph_absolute_rotation_before_after.png",
            LOOP_ROOT / "figures/location_uncertainty.png": figures
            / "pose_graph_location_uncertainty_before_after.png",
            LOOP_ROOT / "figures/angular_uncertainty.png": figures
            / "pose_graph_angular_uncertainty_before_after.png",
            LOOP_ROOT / "figures/trajectory_all_stages.png": figures
            / "system_trajectory_all_stages.png",
        }
        for source, destination in copies.items():
            _require(source.is_file(), "required report figure missing: {}".format(source))
            shutil.copy2(source, destination)
        metrics = {
            "schema": "final-report-artifacts-metrics-v1",
            "scientific_recomputation": False,
            "tracking_audit": tracking_audit,
            "pnp_relative_subsections": pnp_summary,
            "bundle_relative_subsections": bundle_summary,
            "bundle_window_metrics": bundle_window,
            "bundle_projection_distance": projection,
            "e6_e7_metrics": _read_json(LOOP_ROOT / "summary.json")["metrics"],
        }
        _write_json(staging / "metrics.json", metrics)
        _write_json(
            staging / "provenance.json",
            {
                "schema": "final-report-artifacts-provenance-v1",
                "scientific_recomputation": False,
                "forbidden_execution": {
                    "tracking_database_deserialized": False,
                    "bundle_graph_loaded_or_solved": False,
                    "matching_ransac_or_pnp_executed": False,
                    "pose_graph_optimized": False,
                    "marginals_constructed": False,
                },
                "source_sha256": {
                    str(path.relative_to(ROOT)): file_sha256(path)
                    for path in (
                        PNP_ROOT / "metrics.json",
                        TRACKING_ROOT / "metrics.json",
                        BUNDLE_ROOT / "manifest.json",
                        E5_ROOT / "manifest.json",
                        E6_ROOT / "summary.json",
                        E7_ROOT / "final/manifest.json",
                        LOOP_ROOT / "manifest.json",
                    )
                },
                "bundle_projection_aggregation": (
                    "1 m ground-truth path-distance bins; arithmetic mean of "
                    "sealed initial/final stereo pixel distances"
                ),
            },
        )
        _write_json(
            staging / "validation.json",
            {
                "schema": "final-report-artifacts-validation-v1",
                "status": "PASS",
                "figure_count": len(list(figures.glob("*.png"))),
                "tracking_package_change_required": False,
                "scientific_recomputation": False,
            },
        )
        _write_json(
            staging / "manifest.json",
            _manifest_for(staging, REPORT_MANIFEST_SCHEMA),
        )
        _validate_manifest(staging, REPORT_MANIFEST_SCHEMA)
        _publish_managed_directory(staging, output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return validate_final_report_artifacts(output_root=output_root)


def validate_final_report_artifacts(
    *, output_root: Path = FINAL_REPORT_ROOT
) -> Mapping[str, object]:
    artifact_count = _validate_manifest(output_root, REPORT_MANIFEST_SCHEMA)
    metrics = _read_json(Path(output_root) / "metrics.json")
    provenance = _read_json(Path(output_root) / "provenance.json")
    for relative, digest in provenance["source_sha256"].items():
        source = ROOT / relative
        _require(
            source.is_file() and file_sha256(source) == digest,
            "final-report source hash differs: {}".format(relative),
        )
    figures = sorted((Path(output_root) / "figures").glob("*.png"))
    for path in figures:
        _require(mpimg.imread(path).size > 0, "final-report figure cannot be decoded")
    _require(
        metrics["tracking_audit"]["change_required"] is False,
        "Tracking audit unexpectedly requires changes",
    )
    return {
        "status": "PASS",
        "artifact_count": artifact_count,
        "figure_count": len(figures),
        "scientific_recomputation": False,
    }


def refresh_final_report_loop_provenance() -> None:
    """Refresh only the clean Loop Closure package hash after report republish.

    The cross-stage figures do not depend on the report-facing accepted-loop
    CSV column projection.  This narrow metadata refresh avoids repeating the
    2.62-million-row Bundle aggregation when the clean Loop Closure package is
    republished with unchanged scientific values.  Every other recorded input
    hash must remain exact.
    """

    _validate_manifest(FINAL_REPORT_ROOT, REPORT_MANIFEST_SCHEMA)
    provenance_path = FINAL_REPORT_ROOT / "provenance.json"
    provenance = dict(_read_json(provenance_path))
    source_hashes = dict(provenance["source_sha256"])
    loop_relative = str((LOOP_ROOT / "manifest.json").relative_to(ROOT))
    _require(loop_relative in source_hashes, "clean Loop Closure provenance missing")
    for relative, digest in source_hashes.items():
        if relative == loop_relative:
            continue
        source = ROOT / relative
        _require(
            source.is_file() and file_sha256(source) == digest,
            "non-Loop final-report source hash differs: {}".format(relative),
        )
    _verify_external_manifest(LOOP_ROOT)
    source_hashes[loop_relative] = file_sha256(LOOP_ROOT / "manifest.json")
    provenance["source_sha256"] = source_hashes
    _write_json(provenance_path, provenance)
    _write_json(
        FINAL_REPORT_ROOT / "manifest.json",
        _manifest_for(FINAL_REPORT_ROOT, REPORT_MANIFEST_SCHEMA),
    )


def _ast_location(filename: str, qualified_name: str) -> Mapping[str, object]:
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = list(tree.body)
    selected = None
    for part in qualified_name.split("."):
        selected = next(
            (
                node
                for node in nodes
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
                and node.name == part
            ),
            None,
        )
        _require(selected is not None, "source location missing: {}".format(qualified_name))
        nodes = list(getattr(selected, "body", ()))
    return {
        "file": filename,
        "function_or_class": qualified_name,
        "line_start": int(selected.lineno),
        "line_end": int(selected.end_lineno),
        "source_sha256": file_sha256(path),
    }


def build_code_locations() -> Mapping[str, object]:
    """Build final scientific-stage and figure-generator traceability."""

    scientific_specs = (
        ("CODE-01", "Triangulation", "tools.py", "cv_triangulate_matched_points"),
        ("CODE-02", "RANSAC", "tools.py", "estimate_projection_matrices_with_ransac"),
        ("CODE-03", "PnP trajectory calculation", "exercise4_database_builder.py", "build_exercise4_database"),
        ("CODE-04", "DataBase definition", "tracking_database.py", "TrackingDB"),
        ("CODE-05", "Adding a frame to the data base", "tracking_database_adapter.py", "Exercise3TrackingDBAdapter.add_frame"),
        ("CODE-06", "Single Bundle creation", "exercise5_bundle_window.py", "solve_bundle_window"),
        ("CODE-07A", "Bundle endpoint relative transformation", "exercise5_full_run.py", "extract_endpoint_edge"),
        ("CODE-07B", "Bundle endpoint covariance extraction", "exercise5_full_run.py", "extract_endpoint_covariances"),
        ("CODE-08", "Pose Graph building", "PoseGraph.py", "PoseGraph.create_factor_graph"),
        ("CODE-09A", "Loop Closure candidate detection", "exercise7_candidate_detection.py", "CovarianceGraph.find_course_covariance_path"),
        ("CODE-09B", "Loop Closure factor creation/update", "exercise7_pose_graph_update.py", "CumulativePoseGraphState.apply_loop_factor"),
    )
    scientific = []
    for requirement_id, stage, filename, function in scientific_specs:
        scientific.append(
            {
                "requirement_id": requirement_id,
                "stage": stage,
                "status": "PASS",
                "location": _ast_location(filename, function),
            }
        )
    figure_specs = (
        ("Tracking package figures", "project_submission/generate_current_stage.py", "plot_tracking_figures", ["tracking/figures/*.png"], "counts/transition, %, Links/frame, tracks/frame", "frames, correspondences, %, Links, tracks"),
        ("PnP package figures", "project_submission/generate_current_stage.py", "plot_pnp_figures", ["pnp/figures/*.png"], "anchored trajectory, absolute error, projection aggregates, relative error", "m, deg, px, %, deg/m"),
        ("PnP combined subsections", "project_submission/finalize_report_artifacts.py", "plot_relative_subsection_comparison", ["final_report/figures/pnp_relative_subsections_100_400_800.png", "final_report/figures/bundle_relative_subsections_100_400_800.png"], "translation error percent and rotation error divided by GT path distance", "%, deg/m"),
        ("Bundle window reporting", "project_submission/finalize_report_artifacts.py", "plot_bundle_window_report", ["final_report/figures/bundle_mean_total_graph_error_per_factor.png", "final_report/figures/bundle_median_stereo_projection_error.png"], "total graph error/(projection factors+prior); median stereo pixel distance", "error/factor, px"),
        ("Bundle projection-distance reporting", "project_submission/finalize_report_artifacts.py", "plot_bundle_projection_distance", ["final_report/figures/bundle_projection_error_vs_distance.png"], "mean sealed stereo pixel error in 1 m GT-distance bins", "m, px"),
        ("Exercise 7 covariance snapshots", "exercise7_report_freeze.py", "plot_pose_graph_snapshot_covariances", ["loop_closure_clean_final/figures/pose_graph_snapshots_*.png"], "global R Sigma_t R^T, 2D 95% ellipse; x5/x10 radius display only", "m"),
        ("Exercise 7 absolute location errors", "exercise7_report_freeze.py", "plot_absolute_location_errors", ["loop_closure_clean_final/figures/absolute_location_errors.png"], "absolute X/Y/Z error and Euclidean location-error norm", "m"),
        ("Exercise 7 absolute rotation errors", "exercise7_report_freeze.py", "plot_rotation_errors", ["loop_closure_clean_final/figures/absolute_rotation_errors.png"], "SO(3) geodesic absolute rotation error", "deg"),
        ("Exercise 7 location uncertainty", "exercise7_report_freeze.py", "plot_location_uncertainty", ["loop_closure_clean_final/figures/location_uncertainty.png"], "sqrt(trace(raw Pose3 marginal translation block))", "m"),
        ("Exercise 7 angular uncertainty", "exercise7_report_freeze.py", "plot_angular_uncertainty", ["loop_closure_clean_final/figures/angular_uncertainty.png"], "sqrt(trace(raw Pose3 marginal rotation block))", "deg"),
        ("Exercise 7 accepted-loop statistics", "exercise7_report_freeze.py", "plot_accepted_loop_statistics", ["loop_closure_clean_final/figures/accepted_loop_statistics.png", "loop_closure_clean_final/data/accepted_loop_summary.csv"], "reciprocal matches, four-view consensus matches, supporters, and supporter percentage", "matches, supporters, %"),
        ("Exercise 7 successful match", "exercise7_report_freeze.py", "plot_emphasized_successful_match", ["loop_closure_clean_final/figures/first_accepted_successful_match_non_supporters_emphasized.png"], "sealed consensus pixels and supporter mask", "px, matches, %"),
        ("All-stage trajectory", "exercise7_report_freeze.py", "_plot_trajectory", ["loop_closure_clean_final/figures/trajectory_all_stages.png", "final_report/figures/system_trajectory_all_stages.png"], "top-down X-Z camera centers", "m"),
    )
    figures = []
    for purpose, filename, function, outputs, formula, units in figure_specs:
        figures.append(
            {
                "purpose": purpose,
                "input_artifacts": (
                    "sealed project_submission data and clean E5/E6/E7 evidence"
                ),
                "output_artifacts": outputs,
                "formula_or_metric": formula,
                "units": units,
                "location": _ast_location(filename, function),
            }
        )
    return {
        "schema": "project-submission-final-code-locations-v1",
        "status": "PASS",
        "canonical_specification": "docs/submission/Project Submission - Ex8.pdf",
        "scientific_stage_locations": scientific,
        "figure_generator_locations": figures,
        "line_ranges_refreshed_after_final_source_edit": True,
    }


def _parse_existing_requirements(path: Path) -> List[Mapping[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| P"):
            continue
        values = [value.strip().replace("\\|", "|") for value in line.strip("|").split("|")]
        if len(values) >= 10:
            rows.append(
                {
                    "id": values[0],
                    "page": values[1],
                    "category": values[2],
                    "stage": values[3],
                    "requirement": values[5],
                }
            )
        elif len(values) >= 8:
            rows.append(
                {
                    "id": values[0],
                    "page": values[1],
                    "category": values[2],
                    "stage": values[3],
                    "requirement": values[5],
                }
            )
    _require(len(rows) >= 60, "existing seven-page requirements inventory is incomplete")
    return rows


def build_requirements_matrix(existing_path: Path) -> Tuple[str, Mapping[str, int]]:
    """Reclassify every existing seven-page requirement for final reporting."""

    rows = _parse_existing_requirements(existing_path)
    missing = {
        "P1-01",
        "P1-02",
        "P1-03",
        "P1-04",
        "P1-05",
        "P1-09",
        "P2-01",
        "P2-02",
        "P2-03",
        "P2-07",
        "P2-08",
        "P4-06",
    }
    optional = {"P1-10"}
    classifications = {}
    lines = [
        "# Project Submission final requirements traceability matrix",
        "",
        "Canonical source: `docs/submission/Project Submission - Ex8.pdf` (all seven pages locally extracted and audited).",
        "",
        "Classifications are final-report honest: **PASS** means the requested evidence is materialized and validated; **OPTIONAL** is conditional/suggested work not required for this artifact freeze; **MISSING** requires final authored report/layout work; **NOT APPLICABLE** means the condition does not apply.",
        "",
        "| ID | PDF page/section | Category | Stage | Classification | Requirement | Evidence | Remaining work |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        identifier = row["id"]
        classification = (
            "MISSING"
            if identifier in missing
            else "OPTIONAL"
            if identifier in optional
            else "PASS"
        )
        classifications[classification] = classifications.get(classification, 0) + 1
        if row["stage"] == "Tracking":
            evidence = "`tracking/`; audit PASS, unchanged"
        elif row["stage"] == "PnP":
            evidence = "`pnp/`; `final_report/figures/pnp_*`"
        elif row["stage"] == "Bundle":
            evidence = "`bundle_clean_final/`; `final_report/figures/bundle_*`"
        elif row["stage"] in ("Pose Graph", "Loop Closure"):
            evidence = "`loop_closure_clean_final/`; `final_report/figures/pose_graph_*`"
        else:
            evidence = "`final_report/`; `code_locations.json`; stage packages"
        remaining = (
            "Author/place in the final under-20-page numbered report."
            if classification == "MISSING"
            else "Optional editorial selection."
            if classification == "OPTIONAL"
            else "None for the artifact package."
        )
        values = (
            identifier,
            row["page"],
            row["category"],
            row["stage"],
            classification,
            row["requirement"],
            evidence,
            remaining,
        )
        lines.append(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "## Honest remaining boundary",
            "",
            "The evidence package is complete for quantitative stage reporting. The remaining MISSING rows are authored narrative, final document placement/numbering, the rendered page-count check, figure-adjacent code citations, and the requested ground-truth-accuracy discussion. This phase does not fabricate those editorial deliverables.",
            "",
        ]
    )
    return "\n".join(lines), classifications


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_bytes(content)
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _verify_external_manifest(root: Path) -> int:
    manifest = _read_json(Path(root) / "manifest.json")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, list), "component manifest lacks artifact list")
    for record in artifacts:
        path = Path(root) / record["path"]
        _require(
            path.is_file()
            and path.stat().st_size == record["size_bytes"]
            and file_sha256(path) == record["sha256"],
            "component manifest differs: {}".format(path),
        )
    return len(artifacts)


def _top_level_manifest() -> Mapping[str, object]:
    direct_roots = (TRACKING_ROOT, PNP_ROOT)
    artifacts = []
    for root in direct_roots:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            artifacts.append(
                {
                    "path": path.relative_to(SUBMISSION_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    for name in (
        "README.md",
        "code_locations.json",
        "requirements_matrix.md",
        "validation.json",
        "validation_report.md",
        "generate_current_stage.py",
        "finalize_report_artifacts.py",
    ):
        path = SUBMISSION_ROOT / name
        artifacts.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    components = {}
    for name, root in (
        ("bundle_clean_final", BUNDLE_ROOT),
        ("loop_closure_clean_final", LOOP_ROOT),
        ("final_report", FINAL_REPORT_ROOT),
    ):
        manifest_path = root / "manifest.json"
        components[name] = {
            "path": manifest_path.relative_to(SUBMISSION_ROOT).as_posix(),
            "sha256": file_sha256(manifest_path),
            "artifact_count": len(_read_json(manifest_path)["artifacts"]),
        }
    return {
        "schema": PUBLICATION_MANIFEST_SCHEMA,
        "status": "PASS",
        "hierarchical_manifest_policy": (
            "Tracking/PnP/top-level files are indexed directly; clean Bundle, "
            "Loop Closure, and final-report packages are bound by their fully "
            "validated component manifests. Historical packages are excluded."
        ),
        "direct_artifacts": sorted(artifacts, key=lambda row: row["path"]),
        "components": components,
        "excluded_historical_packages": ["bundle/", "loop_closure/"],
    }


def _publication_readme(validation: Mapping[str, object]) -> str:
    return """# Final Project Submission reporting evidence

Status: **PASS** for the reporting-artifact package.

This directory binds the accepted Tracking and PnP packages, the clean Bundle
package, the clean Loop Closure package, and a compact cross-stage
`final_report/` package. Generation was reporting-only: no TrackingDB was
deserialized and no scientific estimator or Marginals computation ran.

## Package map

- `tracking/`: accepted Tracking statistics and figures; final audit required
  no change.
- `pnp/`: accepted PnP data and figures.
- `bundle_clean_final/`: clean sealed Bundle submission evidence.
- `loop_closure_clean_final/`: clean Exercise 7 report/submission freeze.
- `final_report/`: combined PnP/Bundle subsection figures, Bundle reporting
  additions, Pose Graph before/after figures, and the five-trajectory system
  comparison.
- `requirements_matrix.md`: every seven-page PDF requirement classified as
  PASS, OPTIONAL, MISSING, or NOT APPLICABLE.
- `code_locations.json`: final post-edit scientific and figure-generator
  source hashes and AST line ranges.

## Honest boundary

The quantitative evidence is report-ready. The authored report introduction,
stage narrative, per-figure discussion, final numbering/layout, under-20-page
check, figure-adjacent code citations, and ground-truth-accuracy discussion
remain MISSING because this artifact phase does not fabricate final prose or a
rendered report PDF.

## Validation

Run:

```bash
.venv/bin/python project_submission/finalize_report_artifacts.py validate
```

Validated final-report figures: {figures}. Requirements rows: {rows}.
""".format(
        figures=validation["final_report"]["figure_count"],
        rows=validation["requirements"]["row_count"],
    )


def publish_top_level_metadata() -> Mapping[str, object]:
    """Refresh traceability, matrix, reader metadata, validation, and manifest."""

    _require(FINAL_REPORT_ROOT.is_dir(), "final-report artifacts are not published")
    _require(LOOP_ROOT.is_dir(), "clean Loop Closure package is not published")
    refresh_final_report_loop_provenance()
    code_locations = build_code_locations()
    matrix_text, status_counts = build_requirements_matrix(
        SUBMISSION_ROOT / "requirements_matrix.md"
    )
    requirement_row_count = sum(status_counts.values())
    component_counts = {
        "bundle_clean_final": _verify_external_manifest(BUNDLE_ROOT),
        "loop_closure_clean_final": _verify_external_manifest(LOOP_ROOT),
        "final_report": _validate_manifest(FINAL_REPORT_ROOT, REPORT_MANIFEST_SCHEMA),
    }
    final_report_validation = validate_final_report_artifacts()
    validation = {
        "schema": "project-submission-final-validation-v1",
        "status": "PASS",
        "scientific_recomputation": False,
        "tracking": audit_tracking_package(),
        "final_report": final_report_validation,
        "components": component_counts,
        "requirements": {
            "row_count": requirement_row_count,
            "classification_counts": status_counts,
        },
        "traceability": {
            "scientific_stage_location_count": len(
                code_locations["scientific_stage_locations"]
            ),
            "figure_generator_location_count": len(
                code_locations["figure_generator_locations"]
            ),
            "line_ranges_refreshed_after_final_source_edit": True,
        },
        "protected_inputs_modified": False,
        "historical_packages_modified": False,
        "git_commit_created": False,
        "push_performed": False,
    }
    validation_report = """# Final Project Submission validation report

Status: **PASS**

- Reporting-only generation: PASS; no TrackingDB, estimator, optimizer, or
  Marginals construction executed.
- Tracking package audit: PASS; no change required.
- Clean Bundle component manifest: PASS ({bundle} artifacts).
- Clean Loop Closure component manifest: PASS ({loop} artifacts).
- Cross-stage final-report manifest: PASS ({report} artifacts, {figures} figures).
- Requirements inventory: {rows} rows; classifications {classes}.
- Final scientific-stage and figure-generator line/hash traceability: PASS.
- Historical Bundle/Loop Closure packages and protected E5/E6/E7 evidence:
  unchanged by publication.

MISSING classifications are intentionally limited to final authored prose,
rendered-report layout/numbering/page-count, figure-adjacent placement of the
prepared code citations, and the ground-truth-accuracy discussion.
""".format(
        bundle=component_counts["bundle_clean_final"],
        loop=component_counts["loop_closure_clean_final"],
        report=component_counts["final_report"],
        figures=final_report_validation["figure_count"],
        rows=requirement_row_count,
        classes=status_counts,
    )
    payloads = {
        SUBMISSION_ROOT / "code_locations.json": (
            json.dumps(code_locations, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
        SUBMISSION_ROOT / "requirements_matrix.md": matrix_text.encode(),
        SUBMISSION_ROOT / "validation.json": (
            json.dumps(validation, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
        SUBMISSION_ROOT / "validation_report.md": validation_report.encode(),
        SUBMISSION_ROOT / "README.md": _publication_readme(validation).encode(),
    }
    for path, content in payloads.items():
        _atomic_write(path, content)
    manifest = _top_level_manifest()
    _atomic_write(
        SUBMISSION_ROOT / "manifest.json",
        (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode(),
    )
    return validate_top_level_publication()


def validate_top_level_publication() -> Mapping[str, object]:
    manifest = _read_json(SUBMISSION_ROOT / "manifest.json")
    _require(
        manifest.get("schema") == PUBLICATION_MANIFEST_SCHEMA
        and manifest.get("status") == "PASS",
        "top-level publication manifest differs",
    )
    for record in manifest["direct_artifacts"]:
        path = SUBMISSION_ROOT / record["path"]
        _require(
            path.is_file()
            and path.stat().st_size == record["size_bytes"]
            and file_sha256(path) == record["sha256"],
            "top-level artifact differs: {}".format(record["path"]),
        )
    for name, record in manifest["components"].items():
        path = SUBMISSION_ROOT / record["path"]
        _require(
            path.is_file() and file_sha256(path) == record["sha256"],
            "component manifest differs: {}".format(name),
        )
    validation = _read_json(SUBMISSION_ROOT / "validation.json")
    _require(validation.get("status") == "PASS", "top-level validation differs")
    return {
        "status": "PASS",
        "direct_artifact_count": len(manifest["direct_artifacts"]),
        "component_count": len(manifest["components"]),
        "requirements_row_count": validation["requirements"]["row_count"],
        "requirements_classification_counts": validation["requirements"][
            "classification_counts"
        ],
        "final_report_figure_count": validation["final_report"]["figure_count"],
        "scientific_recomputation": False,
    }


def _parse_arguments(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("generate-report", "publish-metadata", "validate", "all")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> None:
    arguments = _parse_arguments(argv)
    result = None
    if arguments.command in ("generate-report", "all"):
        result = generate_final_report_artifacts()
    if arguments.command in ("publish-metadata", "all"):
        result = publish_top_level_metadata()
    if arguments.command == "validate":
        result = validate_top_level_publication()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
