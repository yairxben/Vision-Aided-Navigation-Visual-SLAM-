"""Materialize the curated Exercise 5 package from sealed scientific evidence.

The transaction is fail-closed and presentation-only.  It validates the
complete source run, stages a smaller reader-facing selection at a new
destination, indexes every staged artifact, and promotes the package only
after validation succeeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Mapping, Optional

from exercise5_full_run import (
    ArtifactValidationError,
    DEFAULT_CONFIG_PATH,
    build_run_manifest,
    load_frozen_configuration,
    sha256_file,
    utc_now,
    validate_run_root,
)
from exercise5_reporting import plot_bundle_vs_pnp_comparisons


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ArtifactValidationError("required curated source is absent: {}".format(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _combine_overlays(first_window: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, filename in zip(
        axes.flat,
        (
            "overlay_left_initial.png",
            "overlay_right_initial.png",
            "overlay_left_final.png",
            "overlay_right_final.png",
        ),
    ):
        axis.imshow(mpimg.imread(first_window / filename))
        axis.set_title(filename.replace(".png", "").replace("_", " "))
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _validate_staged_curated_package(
    attempt: Path, configuration: Mapping[str, object]
) -> None:
    """Require every curated artifact and verify its manifest hash entry."""

    required = set(
        configuration["artifact_contract"]["curated_required"]
    )
    observed = {
        str(path.relative_to(attempt))
        for path in attempt.rglob("*")
        if path.is_file()
    }
    missing = required.difference(observed)
    if missing:
        raise ArtifactValidationError(
            "staged curated package is missing {}".format(sorted(missing))
        )
    manifest = _read_json(attempt / "manifest.json")
    if (
        manifest.get("schema")
        != "exercise5-curated-bundle-manifest-v1"
        or manifest.get("status") != "complete-and-sealed"
    ):
        raise ArtifactValidationError("staged curated manifest differs")
    indexed = {
        entry["path"]: entry for entry in manifest.get("artifacts", [])
    }
    non_manifest_files = observed.difference({"manifest.json"})
    if set(indexed) != non_manifest_files:
        raise ArtifactValidationError(
            "staged curated artifact index differs"
        )
    for relative in sorted(non_manifest_files):
        path = attempt / relative
        entry = indexed[relative]
        if (
            path.stat().st_size != entry["size_bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ArtifactValidationError(
                "staged curated hash differs for {}".format(relative)
            )
        if relative.endswith(".png"):
            if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ArtifactValidationError(
                    "staged curated PNG is unreadable: {}".format(relative)
                )
        elif relative.endswith(".json"):
            _read_json(path)
        elif relative.endswith(".csv"):
            with path.open("r", encoding="utf-8", newline="") as handle:
                if next(csv.reader(handle), None) is None:
                    raise ArtifactValidationError(
                        "staged curated CSV is empty: {}".format(relative)
                    )


def _write_prospective_checksum_index(
    source_root: Path,
    *,
    prospective_manifest_path: Path,
    output_path: Path,
) -> None:
    """Build the checksum index for a source manifest awaiting promotion."""

    entries = []
    excluded = {
        (source_root / "manifest.json").resolve(),
        (source_root / "checksums.sha256").resolve(),
        prospective_manifest_path.resolve(),
        output_path.resolve(),
    }
    for item in sorted(source_root.rglob("*")):
        if (
            item.is_file()
            and item.resolve() not in excluded
            and "quarantine" not in item.parts
        ):
            entries.append(
                (
                    str(item.relative_to(source_root)),
                    sha256_file(item),
                )
            )
    entries.append(("manifest.json", sha256_file(prospective_manifest_path)))
    entries.sort()
    with output_path.open("w", encoding="utf-8") as handle:
        for relative, digest in entries:
            handle.write("{}  {}\n".format(digest, relative))
        handle.flush()
        os.fsync(handle.fileno())


def _rollback_promoted_destination(
    destination: Path, staged_destination: Path
) -> None:
    if not destination.exists():
        return
    rollback = staged_destination.with_name(
        staged_destination.name + ".source-finalization-failed"
    )
    try:
        os.rename(str(destination), str(rollback))
    except OSError:
        shutil.rmtree(destination)


def _promote_curated_then_complete_source(
    *,
    source_root: Path,
    staged_destination: Path,
    destination: Path,
    prospective_manifest_path: Path,
    prospective_checksum_path: Path,
) -> None:
    """Promote curated evidence before atomically marking source complete."""

    old_checksum_path = source_root / "checksums.sha256"
    old_checksum = (
        old_checksum_path.read_bytes() if old_checksum_path.exists() else None
    )
    try:
        os.replace(str(staged_destination), str(destination))
        _fsync_directory(destination.parent)
    except BaseException:
        prospective_manifest_path.unlink(missing_ok=True)
        prospective_checksum_path.unlink(missing_ok=True)
        raise
    try:
        os.replace(str(prospective_checksum_path), str(old_checksum_path))
        _fsync_directory(source_root)
        os.replace(
            str(prospective_manifest_path),
            str(source_root / "manifest.json"),
        )
        _fsync_directory(source_root)
    except BaseException:
        _rollback_promoted_destination(destination, staged_destination)
        if old_checksum is None:
            old_checksum_path.unlink(missing_ok=True)
        else:
            with old_checksum_path.open("wb") as handle:
                handle.write(old_checksum)
                handle.flush()
                os.fsync(handle.fileno())
        prospective_manifest_path.unlink(missing_ok=True)
        prospective_checksum_path.unlink(missing_ok=True)
        _fsync_directory(source_root)
        raise


def materialize_bundle_package(
    source_root: Path,
    destination: Path,
    *,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Materialize a new curated package from a complete canonical source.

    Existing destinations are rejected.  For a source already marked
    ``complete-and-sealed`` the scientific root is strictly read-only; the
    staging transaction writes only beneath the prospective destination.
    """

    source_root = Path(source_root).resolve()
    destination = Path(destination).resolve()
    configuration, config_sha = load_frozen_configuration(configuration_path)
    manifest = _read_json(source_root / "manifest.json")
    expected = configuration["keyframe_selection"]["statistics"]["window_count"]
    all_windows_sealed = (
        manifest.get("scope") == "canonical"
        and manifest.get("expected_window_count") == expected
        and manifest.get("completed_window_indices") == list(range(expected))
        and manifest.get("status")
        in (
            "windows-complete-awaiting-aggregation",
            "complete-and-sealed",
        )
    )
    if not all_windows_sealed:
        raise ArtifactValidationError(
            "submission source is partial, noncanonical, or unsealed"
        )
    # Curated presentation is permitted only after strict window reload.
    validation = validate_run_root(
        source_root,
        configuration_path=configuration_path,
        allow_partial=manifest["status"] != "complete-and-sealed",
    )
    if (
        validation["validated_window_indices"] != list(range(expected))
    ):
        raise ArtifactValidationError(
            "submission source is partial, noncanonical, or unsealed"
        )
    if destination.exists():
        raise FileExistsError(destination)
    attempt = destination.parent / (
        ".{}.attempt.{}.tmp".format(destination.name, uuid.uuid4().hex)
    )
    attempt.mkdir(parents=True)
    prospective_manifest_path = None
    prospective_checksum_path = None
    try:
        # Copy the frozen curated set into an isolated attempt directory.
        data = attempt / "data"
        figures = attempt / "figures"
        data.mkdir()
        figures.mkdir()
        aggregate_data = source_root / "aggregate" / "data"
        aggregate_figures = source_root / "aggregate" / "figures"
        for name in configuration["artifact_contract"]["aggregate_data_files"]:
            _copy(aggregate_data / name, data / name)
        for name in configuration["artifact_contract"]["aggregate_figure_files"]:
            _copy(aggregate_figures / name, figures / name)
        # These three figures are derived only from finalized aggregate CSVs;
        # no Bundle solve or scientific artifact mutation occurs here.
        plot_bundle_vs_pnp_comparisons(data, figures)
        first_window = sorted(
            (source_root / "windows").glob("window_0000_*")
        )[0]
        _copy(first_window / "scene_3d.png", figures / "first_window_scene_3d.png")
        _copy(
            first_window / "scene_birds_eye.png",
            figures / "first_window_scene_birds_eye.png",
        )
        _combine_overlays(first_window, figures / "first_window_overlays.png")
        _copy(
            source_root / "track_5_1" / "residuals.csv",
            data / "track_5_1.csv",
        )
        _copy(
            source_root / "track_5_1" / "reprojection_error.png",
            figures / "track_reprojection_error.png",
        )
        _copy(
            source_root / "track_5_1" / "factor_error.png",
            figures / "track_factor_error.png",
        )
        _copy(
            source_root / "track_5_1" / "factor_vs_reprojection.png",
            figures / "track_factor_vs_reprojection.png",
        )
        _copy(data / "all_projection_residuals.csv", data / "projection_by_bundle_start.csv")
        metrics_rows = []
        with (data / "window_metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            metrics_rows = list(csv.DictReader(handle))
        with (data / "first_window_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(metrics_rows[0]))
            writer.writeheader()
            writer.writerow(metrics_rows[0])
        metrics = {
            "schema": "exercise5-curated-bundle-metrics-v1",
            "window_count": len(metrics_rows),
            "first_window_initial_total_graph_error": float(
                metrics_rows[0]["initial_total_graph_error"]
            ),
            "first_window_final_total_graph_error": float(
                metrics_rows[0]["final_total_graph_error"]
            ),
            "requirement_ids": ["P1-C07", "Exercise-5.1", "Exercise-5.3"],
        }
        _write_json(attempt / "metrics.json", metrics)
        artifacts = []
        for path in sorted(attempt.rglob("*")):
            if path.is_file():
                artifacts.append(
                    {
                        "path": str(path.relative_to(attempt)),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        finalizing_source = manifest["status"] != "complete-and-sealed"
        if finalizing_source:
            token = uuid.uuid4().hex
            prospective_manifest_path = (
                source_root / ".manifest.finalize.{}.tmp".format(token)
            )
            prospective_checksum_path = (
                source_root / ".checksums.finalize.{}.tmp".format(token)
            )
            complete_manifest = build_run_manifest(
                configuration=configuration,
                config_sha256=config_sha,
                source_identity=manifest["source_input_identity"],
                completed_window_indices=list(range(expected)),
                status="complete-and-sealed",
                scope="canonical",
                created_utc=utc_now(),
            )
            _write_json(prospective_manifest_path, complete_manifest)
            _write_prospective_checksum_index(
                source_root,
                prospective_manifest_path=prospective_manifest_path,
                output_path=prospective_checksum_path,
            )
            source_manifest_sha256 = sha256_file(
                prospective_manifest_path
            )
            final_source_validation = {
                **validation,
                "status": "complete-and-sealed",
            }
        else:
            source_manifest_sha256 = sha256_file(
                source_root / "manifest.json"
            )
            final_source_validation = validation
        # Index the complete staged payload before it can become visible at
        # the requested destination.
        curated_manifest = {
            "schema": "exercise5-curated-bundle-manifest-v1",
            "status": "complete-and-sealed",
            "source_manifest_sha256": source_manifest_sha256,
            "source_configuration_sha256": config_sha,
            "source_validation": final_source_validation,
            "artifacts": artifacts,
        }
        _write_json(attempt / "manifest.json", curated_manifest)
        _validate_staged_curated_package(attempt, configuration)
        if finalizing_source:
            _promote_curated_then_complete_source(
                source_root=source_root,
                staged_destination=attempt,
                destination=destination,
                prospective_manifest_path=prospective_manifest_path,
                prospective_checksum_path=prospective_checksum_path,
            )
            validate_run_root(
                source_root,
                configuration_path=configuration_path,
                allow_partial=False,
            )
        else:
            # The normal final-state path atomically promotes only the curated
            # attempt and leaves the sealed scientific source untouched.
            os.replace(str(attempt), str(destination))
            _fsync_directory(destination.parent)
        return destination
    except BaseException:
        if attempt.exists():
            shutil.rmtree(attempt)
        if (
            prospective_manifest_path is not None
            and prospective_manifest_path.exists()
        ):
            prospective_manifest_path.unlink()
        if (
            prospective_checksum_path is not None
            and prospective_checksum_path.exists()
        ):
            prospective_checksum_path.unlink()
        raise


__all__ = ["materialize_bundle_package"]
