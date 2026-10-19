"""Render Exercise 5 report figures from sealed machine-readable evidence.

Reporting is presentation-only: these helpers read retained window/aggregate
data and never construct a Bundle graph, optimize Values, or modify scientific
pose and covariance evidence.
"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_projection_overlay(
    image: np.ndarray,
    *,
    measurement: Sequence[float],
    projection: Sequence[float],
    side: str,
    title: str,
    output_path: Path,
) -> None:
    """Plot one measured/projected image point in a diagnostic crop."""

    coordinate = 0 if side == "left" else 1
    measured = np.asarray([measurement[coordinate], measurement[2]])
    projected = np.asarray([projection[coordinate], projection[2]])
    height, width = image.shape[:2]
    center = np.clip((measured + projected) / 2.0, [0, 0], [width - 1, height - 1])
    radius = 80
    x0 = max(0, int(center[0]) - radius)
    x1 = min(width, int(center[0]) + radius)
    y0 = max(0, int(center[1]) - radius)
    y1 = min(height, int(center[1]) + radius)
    crop = image[y0:y1, x0:x1]
    plt.figure(figsize=(5, 4))
    plt.imshow(crop, cmap="gray")
    plt.scatter(
        [measured[0] - x0],
        [measured[1] - y0],
        marker="x",
        s=80,
        label="measurement",
    )
    plt.scatter(
        [projected[0] - x0],
        [projected[1] - y0],
        facecolors="none",
        edgecolors="tab:red",
        s=80,
        label="projection",
    )
    plt.xlabel("u [px]")
    plt.ylabel("v [px]")
    plt.title(title)
    plt.legend()
    _save_figure(output_path)


def plot_first_window_scene(
    *,
    initial_camera_poses: np.ndarray,
    final_camera_poses: np.ndarray,
    initial_landmarks: np.ndarray,
    final_landmarks: np.ndarray,
    output_path: Path,
    birds_eye: bool,
) -> None:
    """Compare initial and optimized first-window cameras and landmarks."""

    initial_centers = np.asarray([pose[:3, 3] for pose in initial_camera_poses])
    final_centers = np.asarray([pose[:3, 3] for pose in final_camera_poses])
    if birds_eye:
        plt.figure(figsize=(8, 6))
        plt.scatter(initial_landmarks[:, 0], initial_landmarks[:, 2], s=1, alpha=0.2)
        plt.scatter(final_landmarks[:, 0], final_landmarks[:, 2], s=1, alpha=0.2)
        plt.plot(initial_centers[:, 0], initial_centers[:, 2], "o-", label="initial")
        plt.plot(final_centers[:, 0], final_centers[:, 2], "o-", label="final")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.axis("equal")
    else:
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        sample = slice(None, None, max(1, len(initial_landmarks) // 3000))
        axis.scatter(
            initial_landmarks[sample, 0],
            initial_landmarks[sample, 1],
            initial_landmarks[sample, 2],
            s=1,
            alpha=0.2,
        )
        axis.scatter(
            final_landmarks[sample, 0],
            final_landmarks[sample, 1],
            final_landmarks[sample, 2],
            s=1,
            alpha=0.2,
        )
        axis.plot(*initial_centers.T, "o-", label="initial")
        axis.plot(*final_centers.T, "o-", label="final")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
    plt.title("First Bundle window scene")
    plt.legend()
    _save_figure(output_path)


def plot_first_window_artifacts(
    *,
    result: object,
    calibration: object,
    output_directory: Path,
    image_root: Optional[Path],
) -> None:
    """Write the overlays and scene views retained only for window zero."""

    output_directory = Path(output_directory)
    diagnostic = result.maximum_initial_projection_error
    if image_root is None:
        left = np.zeros((375, 1242), dtype=np.uint8)
        right = left.copy()
    else:
        frame_name = "{:06d}.png".format(diagnostic.frame_id)
        left = mpimg.imread(Path(image_root) / "image_0" / frame_name)
        right = mpimg.imread(Path(image_root) / "image_1" / frame_name)
    for side, image in (("left", left), ("right", right)):
        for state, projection in (
            ("initial", diagnostic.initial_projection),
            ("final", diagnostic.final_projection),
        ):
            plot_projection_overlay(
                image,
                measurement=diagnostic.measurement,
                projection=projection,
                side=side,
                title="{} {} projection, frame {}, track {}".format(
                    side.capitalize(), state, diagnostic.frame_id, diagnostic.track_id
                ),
                output_path=output_directory
                / "overlay_{}_{}.png".format(side, state),
            )
    initial_camera_poses = np.asarray(
        [
            result.initial_values.atPose3(
                result.camera_key_by_frame_id[frame_id]
            ).matrix()
            for frame_id in result.frame_ids
        ]
    )
    final_camera_poses = np.asarray(
        [
            result.final_values.atPose3(
                result.camera_key_by_frame_id[frame_id]
            ).matrix()
            for frame_id in result.frame_ids
        ]
    )
    initial_landmarks = np.asarray(
        [
            result.initial_values.atPoint3(
                result.landmark_key_by_track_id[track_id]
            )
            for track_id in result.accepted_track_ids
        ]
    )
    final_landmarks = np.asarray(
        [
            result.final_values.atPoint3(
                result.landmark_key_by_track_id[track_id]
            )
            for track_id in result.accepted_track_ids
        ]
    )
    plot_first_window_scene(
        initial_camera_poses=initial_camera_poses,
        final_camera_poses=final_camera_poses,
        initial_landmarks=initial_landmarks,
        final_landmarks=final_landmarks,
        output_path=output_directory / "scene_3d.png",
        birds_eye=False,
    )
    plot_first_window_scene(
        initial_camera_poses=initial_camera_poses,
        final_camera_poses=final_camera_poses,
        initial_landmarks=initial_landmarks,
        final_landmarks=final_landmarks,
        output_path=output_directory / "scene_birds_eye.png",
        birds_eye=True,
    )


def _read_csv(path: Path) -> Tuple[Dict[str, str], ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return tuple(csv.DictReader(handle))


def _series(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _read_finite_numeric_columns(
    path: Path, required_columns: Sequence[str]
) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("required report CSV is absent: {}".format(path))
    if path.stat().st_size == 0:
        raise ValueError("required report CSV is empty: {}".format(path))

    values = {name: [] for name in required_columns}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or any(
            fieldnames.count(name) != 1 for name in required_columns
        ):
            raise ValueError(
                "report CSV must contain each required column exactly once: "
                "{}".format(tuple(required_columns))
            )
        for row_number, row in enumerate(reader, start=1):
            for name in required_columns:
                try:
                    value = float(row[name])
                except (KeyError, TypeError, ValueError) as exception:
                    raise ValueError(
                        "report CSV has a nonnumeric {} value at data row {}"
                        .format(name, row_number)
                    ) from exception
                if not np.isfinite(value):
                    raise ValueError(
                        "report CSV has a nonfinite {} value at data row {}"
                        .format(name, row_number)
                    )
                values[name].append(value)
    if not values[required_columns[0]]:
        raise ValueError("required report CSV contains no rows: {}".format(path))
    return {
        name: np.asarray(column_values, dtype=float)
        for name, column_values in values.items()
    }


def _plot_two_series(
    x_values: Sequence[float],
    bundle_values: Sequence[float],
    pnp_values: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    x = np.asarray(x_values, dtype=float)
    bundle = np.asarray(bundle_values, dtype=float)
    pnp = np.asarray(pnp_values, dtype=float)
    if any(values.ndim != 1 for values in (x, bundle, pnp)):
        raise ValueError("comparison plot series must be one-dimensional")
    if len(x) == 0:
        raise ValueError("comparison plot series must be nonempty")
    if len(x) != len(bundle) or len(x) != len(pnp):
        raise ValueError("comparison plot series lengths differ")
    if any(not np.all(np.isfinite(values)) for values in (x, bundle, pnp)):
        raise ValueError("comparison plot series must contain only finite values")

    plt.figure(figsize=(10, 4.5))
    plt.plot(x, bundle, label="Bundle", linewidth=1.0)
    plt.plot(x, pnp, label="PnP", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _save_figure(Path(output_path))


def plot_bundle_vs_pnp_comparisons(
    data_directory: Path, figure_directory: Path
) -> Tuple[Path, Path, Path]:
    """Plot the full finalized Bundle-versus-PnP comparison population."""

    data_directory = Path(data_directory)
    figure_directory = Path(figure_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)
    absolute = _read_finite_numeric_columns(
        data_directory / "keyframe_localization_errors.csv",
        ("frame_id", "bundle_error_m", "pnp_error_m"),
    )
    relative = _read_finite_numeric_columns(
        data_directory / "consecutive_relative_errors.csv",
        (
            "window_index",
            "bundle_translation_error_m",
            "pnp_translation_error_m",
            "bundle_rotation_error_deg",
            "pnp_rotation_error_deg",
        ),
    )
    outputs = (
        figure_directory / "bundle_vs_pnp_absolute_localization_error.png",
        figure_directory / "bundle_vs_pnp_relative_translation_error.png",
        figure_directory / "bundle_vs_pnp_relative_rotation_error.png",
    )
    _plot_two_series(
        absolute["frame_id"],
        absolute["bundle_error_m"],
        absolute["pnp_error_m"],
        xlabel="frame id",
        ylabel="absolute localization error [m]",
        title="Bundle vs PnP absolute keyframe localization error",
        output_path=outputs[0],
    )
    _plot_two_series(
        relative["window_index"],
        relative["bundle_translation_error_m"],
        relative["pnp_translation_error_m"],
        xlabel="window index",
        ylabel="relative translation error [m]",
        title="Bundle vs PnP consecutive-keyframe translation error",
        output_path=outputs[1],
    )
    _plot_two_series(
        relative["window_index"],
        relative["bundle_rotation_error_deg"],
        relative["pnp_rotation_error_deg"],
        xlabel="window index",
        ylabel="relative rotation error [deg]",
        title="Bundle vs PnP consecutive-keyframe rotation error",
        output_path=outputs[2],
    )
    return outputs


def plot_window_errors(data_directory: Path, figure_directory: Path) -> None:
    """Plot finalized factor and stereo-pixel error summaries by window."""

    rows = _read_csv(Path(data_directory) / "window_metrics.csv")
    x = _series(rows, "window_index")
    for column, filename, ylabel in (
        (
            "final_mean_projection_factor_error",
            "mean_factor_error_by_window.png",
            "mean projection factor error",
        ),
        (
            "final_median_stereo_pixel_distance_px",
            "median_projection_error_by_window.png",
            "median stereo projection error [px]",
        ),
    ):
        plt.figure(figsize=(9, 4))
        plt.plot(x, _series(rows, column))
        plt.xlabel("window index")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        _save_figure(Path(figure_directory) / filename)


def plot_projection_distance(data_directory: Path, figure_directory: Path) -> None:
    """Plot every final projection residual against distance from window start.

    The input is counted and then streamed into fixed arrays so the full
    population is plotted without constructing a list of CSV dictionaries.
    """

    input_path = Path(data_directory) / "all_projection_residuals.csv"
    required_columns = (
        "pnp_path_distance_from_window_start_m",
        "final_stereo_pixel_distance_px",
    )
    row_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or any(
            fieldnames.count(name) != 1 for name in required_columns
        ):
            raise ValueError(
                "projection residual CSV must contain each required column "
                "exactly once: {}".format(required_columns)
            )
        for _ in reader:
            row_count += 1
    if row_count == 0:
        raise ValueError("projection residual CSV contains no rows")

    path_distances = np.empty(row_count, dtype=np.float64)
    projection_distances = np.empty(row_count, dtype=np.float64)
    observed_count = 0
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or any(
            fieldnames.count(name) != 1 for name in required_columns
        ):
            raise ValueError(
                "projection residual CSV columns changed between passes"
            )
        for row_number, row in enumerate(reader, start=1):
            if observed_count >= row_count:
                raise ValueError(
                    "projection residual CSV row count changed between passes"
                )
            try:
                path_distances[observed_count] = float(
                    row[required_columns[0]]
                )
                projection_distances[observed_count] = float(
                    row[required_columns[1]]
                )
            except (KeyError, TypeError, ValueError) as exception:
                raise ValueError(
                    "projection residual CSV has a nonnumeric required value "
                    "at data row {}".format(row_number)
                ) from exception
            observed_count += 1
    if observed_count != row_count:
        raise ValueError(
            "projection residual CSV row count changed between passes"
        )

    output_path = (
        Path(figure_directory) / "projection_error_vs_bundle_start.png"
    )
    attempt = output_path.parent / (
        ".{}.attempt.{}.tmp".format(output_path.name, uuid.uuid4().hex)
    )
    figure = None
    try:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.scatter(
            path_distances,
            projection_distances,
            s=2,
            alpha=0.2,
        )
        axis.set_xlabel("PnP distance from Bundle start [m]")
        axis.set_ylabel("final stereo projection distance [px]")
        figure.tight_layout()
        figure.savefig(attempt, dpi=140, format="png")
        attempt.replace(output_path)
    finally:
        if figure is not None:
            plt.close(figure)
        if attempt.exists():
            attempt.unlink()


def plot_trajectories(data_directory: Path, figure_directory: Path) -> None:
    """Plot Bundle, PnP, and ground-truth keyframe trajectories together."""

    rows = _read_csv(Path(data_directory) / "keyframe_trajectories.csv")
    plt.figure(figsize=(8, 6))
    for prefix, label in (
        ("bundle", "Bundle"),
        ("pnp", "PnP"),
        ("ground_truth", "ground truth"),
    ):
        plt.plot(
            _series(rows, prefix + "_x_m"),
            _series(rows, prefix + "_z_m"),
            label=label,
        )
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.legend()
    _save_figure(Path(figure_directory) / "bundle_pnp_ground_truth_trajectory.png")


def plot_localization_error(data_directory: Path, figure_directory: Path) -> None:
    """Plot absolute Bundle keyframe localization error."""

    rows = _read_csv(Path(data_directory) / "keyframe_localization_errors.csv")
    plt.figure(figsize=(9, 4))
    plt.plot(_series(rows, "frame_id"), _series(rows, "bundle_error_m"))
    plt.xlabel("FrameId")
    plt.ylabel("Bundle localization error [m]")
    _save_figure(Path(figure_directory) / "keyframe_localization_error.png")


def plot_consecutive_relative_errors(
    data_directory: Path, figure_directory: Path
) -> None:
    """Plot Bundle translation and rotation error for consecutive windows."""

    rows = _read_csv(Path(data_directory) / "consecutive_relative_errors.csv")
    plt.figure(figsize=(9, 4))
    plt.plot(
        _series(rows, "window_index"),
        _series(rows, "bundle_translation_error_m"),
        label="translation [m]",
    )
    plt.plot(
        _series(rows, "window_index"),
        _series(rows, "bundle_rotation_error_deg"),
        label="rotation [deg]",
    )
    plt.xlabel("window index")
    plt.legend()
    _save_figure(Path(figure_directory) / "consecutive_relative_error.png")


def plot_relative_subsections(
    data_directory: Path, figure_directory: Path, length: int
) -> None:
    """Plot normalized translation error for one requested subsection length."""

    rows = _read_csv(
        Path(data_directory) / "relative_subsections_{}.csv".format(length)
    )
    plt.figure(figsize=(9, 4))
    if rows:
        plt.plot(
            _series(rows, "start_frame_id"),
            _series(rows, "translation_error_percent"),
        )
    plt.xlabel("start FrameId")
    plt.ylabel("translation error [%]")
    _save_figure(
        Path(figure_directory) / "relative_subsections_{}.png".format(length)
    )


def plot_first_window_camera_covariances(
    source_root: Path, figure_directory: Path
) -> None:
    """Visualize raw first-window camera translation covariance scales."""

    first = sorted((Path(source_root) / "windows").glob("window_0000_*"))[0]
    with np.load(first / "camera_marginals.npz", allow_pickle=False) as data:
        poses = data["poses"]
        covariance = data["marginal_covariances"]
    centers = np.asarray([pose[:3, 3] for pose in poses])
    scales = np.sqrt(
        np.maximum(0.0, np.asarray([np.trace(cov[3:, 3:]) for cov in covariance]))
    )
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(*centers.T, c=scales, s=30 + 100 * scales / max(scales.max(), 1e-12))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("First-window camera translation covariance scale")
    _save_figure(
        Path(figure_directory) / "first_window_camera_covariances_3d.png"
    )


def plot_track_diagnostic(
    residuals_path: Path, output_directory: Path
) -> Tuple[Path, Path, Path]:
    """Write the three Exercise 5.1 reprojection/factor-error figures."""

    rows = _read_csv(residuals_path)
    frame_ids = _series(rows, "frame_id")
    output_directory = Path(output_directory)
    reprojection = output_directory / "reprojection_error.png"
    factor = output_directory / "factor_error.png"
    relationship = output_directory / "factor_vs_reprojection.png"

    plt.figure(figsize=(8, 4))
    plt.plot(
        frame_ids,
        _series(rows, "left_pixel_distance_px"),
        "o-",
        label="left",
    )
    plt.plot(
        frame_ids,
        _series(rows, "right_pixel_distance_px"),
        "o-",
        label="right",
    )
    plt.xlabel("FrameId")
    plt.ylabel("reprojection distance [px]")
    plt.legend()
    _save_figure(reprojection)

    plt.figure(figsize=(8, 4))
    plt.plot(frame_ids, _series(rows, "factor_error"), "o-")
    plt.xlabel("FrameId")
    plt.ylabel("stereo factor error")
    _save_figure(factor)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        _series(rows, "mean_left_right_pixel_distance_px"),
        _series(rows, "factor_error"),
    )
    plt.xlabel("mean left/right reprojection distance [px]")
    plt.ylabel("stereo factor error")
    _save_figure(relationship)
    return reprojection, factor, relationship


def generate_report_figures(source_root: Path) -> Tuple[Path, ...]:
    """Generate the complete current 13-figure aggregate report set."""

    source_root = Path(source_root)
    data = source_root / "aggregate" / "data"
    figures = source_root / "aggregate" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # Core course figures consume only finalized aggregate tables.
    plot_window_errors(data, figures)
    plot_projection_distance(data, figures)
    plot_trajectories(data, figures)
    plot_localization_error(data, figures)
    plot_consecutive_relative_errors(data, figures)
    # The three explicit Bundle-versus-PnP comparisons were integrated after
    # the sealed clean run and remain derived presentation artifacts.
    plot_bundle_vs_pnp_comparisons(data, figures)
    for length in (100, 400, 800):
        plot_relative_subsections(data, figures, length)
    plot_first_window_camera_covariances(source_root, figures)
    return tuple(sorted(figures.glob("*.png")))


__all__ = [
    "generate_report_figures",
    "plot_bundle_vs_pnp_comparisons",
    "plot_consecutive_relative_errors",
    "plot_first_window_artifacts",
    "plot_first_window_camera_covariances",
    "plot_first_window_scene",
    "plot_localization_error",
    "plot_projection_distance",
    "plot_projection_overlay",
    "plot_relative_subsections",
    "plot_track_diagnostic",
    "plot_trajectories",
    "plot_window_errors",
]
