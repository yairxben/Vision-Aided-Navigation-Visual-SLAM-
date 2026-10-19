"""Deterministic report-only freeze and submission for sealed Exercise 7.

This module is not a scientific pipeline stage.  It reads selected, coherent
E5/E6 artifacts and sealed E7 arrays/CSVs/checkpoints; it performs no feature
matching, RANSAC, Bundle solve, Pose Graph optimization, or Marginals
construction.  Clean roots are the defaults, while coherent historical roots
remain explicit read-only inputs.

All scientific 6x6 covariances remain exactly as saved.  Scalar location
uncertainty is ``sqrt(trace(Sigma[3:6, 3:6]))`` in metres.  Where sealed angular
uncertainty is reported, its source convention is
``sqrt(trace(Sigma[0:3, 0:3]))`` in radians (converted to degrees only for the
reported unit).  A top-down ellipse first rotates the local translation block
into global coordinates, selects its X-Z block, and applies the two-dimensional
95% chi-square contour.  Thus "raw 95% covariance" means raw, unmodified saved
covariance presented as a 95% confidence ellipse; 95% is not a covariance
multiplier.  Plot subsampling and any future x5/x10 *radius* display variant
would be presentation-only and would not alter scientific covariance.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_E5_ROOT = REPOSITORY_ROOT / "diagnostics/ex5_clean_final"
DEFAULT_E6_ROOT = REPOSITORY_ROOT / "diagnostics/ex6_clean_final"
DEFAULT_SCIENTIFIC_ROOT = REPOSITORY_ROOT / "diagnostics/ex7_clean_final"
HISTORICAL_SCIENTIFIC_ROOT = REPOSITORY_ROOT / "diagnostics/ex7_i04"
HISTORICAL_SUBMISSION_ROOT = (
    REPOSITORY_ROOT / "project_submission/loop_closure"
)
HISTORICAL_E5_ROOT = REPOSITORY_ROOT / "diagnostics/ex5_final"
HISTORICAL_E6_ROOT = REPOSITORY_ROOT / "diagnostics/ex6_pose_graph"
HISTORICAL_BUNDLE_SUBMISSION_ROOT = (
    REPOSITORY_ROOT / "project_submission/bundle"
)
CLEAN_EQUIVALENCE_ROOT = (
    REPOSITORY_ROOT / "diagnostics/ex7_clean_equivalence"
)
CLEAN_BUNDLE_SUBMISSION_ROOT = (
    REPOSITORY_ROOT / "project_submission/bundle_clean_final"
)

# Compatibility aliases now identify the clean default lane.  Active report
# functions derive every source path from ReportEvidencePaths below.
CANONICAL_ROOT = DEFAULT_SCIENTIFIC_ROOT
CANONICAL_FINAL = CANONICAL_ROOT / "final"
E6_VALUES = DEFAULT_E6_ROOT / "pose_graph_values.npz"
E6_MARGINALS = DEFAULT_E6_ROOT / "pose_graph_marginals.npz"
E5_TRAJECTORIES = (
    DEFAULT_E5_ROOT / "aggregate/data/keyframe_trajectories.csv"
)
E7_TRAJECTORIES = CANONICAL_FINAL / "reports/data/trajectory_comparison.csv"
E7_ERRORS = CANONICAL_FINAL / "reports/data/absolute_pose_errors.csv"
ACCEPTED_LOOPS = CANONICAL_FINAL / "aggregate/accepted_loops.csv"
DEFAULT_REPORT_ROOT = CANONICAL_ROOT / "report_freeze"
DEFAULT_SUBMISSION_ROOT = (
    REPOSITORY_ROOT / "project_submission/loop_closure_clean_final"
)
SNAPSHOT_ACCEPTANCE_IDS = (0, 1, 3, 6)
ELLIPSE_SUBSAMPLE_STRIDE = 12
CHI_SQUARE_2D_95 = 5.991464547107979


class ReportFreezeError(RuntimeError):
    """Raised when canonical evidence is missing, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class ReportEvidencePaths:
    """Read-only report inputs derived from a coherent E5/E6/E7 root chain."""

    e5_root: Path
    e6_root: Path
    scientific_root: Path

    @classmethod
    def from_roots(
        cls, *, e5_root: Path, e6_root: Path, scientific_root: Path
    ) -> "ReportEvidencePaths":
        return cls(
            e5_root=Path(e5_root).resolve(),
            e6_root=Path(e6_root).resolve(),
            scientific_root=Path(scientific_root).resolve(),
        )

    @property
    def final_root(self) -> Path:
        return self.scientific_root / "final"

    @property
    def e6_values(self) -> Path:
        return self.e6_root / "pose_graph_values.npz"

    @property
    def e6_marginals(self) -> Path:
        return self.e6_root / "pose_graph_marginals.npz"

    @property
    def e5_trajectories(self) -> Path:
        return self.e5_root / "aggregate/data/keyframe_trajectories.csv"

    @property
    def e7_trajectories(self) -> Path:
        return self.final_root / "reports/data/trajectory_comparison.csv"

    @property
    def e7_errors(self) -> Path:
        return self.final_root / "reports/data/absolute_pose_errors.csv"

    @property
    def marginal_uncertainty_sizes(self) -> Path:
        return self.final_root / "reports/data/marginal_uncertainty_sizes.csv"

    @property
    def accepted_loops(self) -> Path:
        return self.final_root / "aggregate/accepted_loops.csv"

    @property
    def matcher_invocations(self) -> Path:
        return self.final_root / "aggregate/matcher_invocations.csv"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportFreezeError(message)


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _display_path(path: Path) -> str:
    try:
        return Path(path).relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def compute_absolute_pose_metrics(
    estimates: np.ndarray, truth: np.ndarray
) -> Mapping[str, np.ndarray]:
    """Compute absolute X/Y/Z/norm [m] and rotation error [deg]."""

    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    _require(
        estimates.shape == truth.shape
        and estimates.ndim == 3
        and estimates.shape[1:] == (4, 4)
        and np.all(np.isfinite(estimates))
        and np.all(np.isfinite(truth)),
        "pose metric arrays must be matching finite Nx4x4 arrays",
    )
    delta = estimates[:, :3, 3] - truth[:, :3, 3]
    rotation_deg = []
    for estimate, reference in zip(estimates, truth):
        error_rotation = (np.linalg.inv(reference) @ estimate)[:3, :3]
        cosine = np.clip(
            (float(np.trace(error_rotation)) - 1.0) / 2.0, -1.0, 1.0
        )
        rotation_deg.append(float(np.degrees(np.arccos(cosine))))
    return {
        "absolute_xyz_m": np.abs(delta),
        "location_norm_m": np.linalg.norm(delta, axis=1),
        "rotation_error_deg": np.asarray(rotation_deg, dtype=float),
    }


def require_aligned_frame_ids(
    reference: Sequence[int],
    candidate: Sequence[int],
    candidate_name: str,
) -> None:
    """Fail exactly when a trajectory does not use canonical FrameIds/order."""

    reference_array = np.asarray(reference, dtype=np.int64)
    candidate_array = np.asarray(candidate, dtype=np.int64)
    _require(
        np.array_equal(reference_array, candidate_array),
        "{} FrameIds differ from canonical keyframe order".format(
            candidate_name
        ),
    )


def select_snapshot_state_ids(accepted_count: int) -> Tuple[int, int, int, int]:
    """Select baseline, first, acceptance 3, and final acceptance 6."""

    _require(
        int(accepted_count) == 6,
        "report freeze requires exactly six sealed accepted Loop Closures",
    )
    return SNAPSHOT_ACCEPTANCE_IDS


def accepted_edges_for_state(
    accepted_loops: Sequence[Mapping[str, object]], state_id: int
) -> Tuple[Tuple[int, int], ...]:
    """Return accepted previous/current FrameId edges present in one state."""

    state_id = int(state_id)
    _require(0 <= state_id <= len(accepted_loops), "invalid snapshot state")
    ordered = sorted(
        accepted_loops, key=lambda row: int(row["acceptance_id"])
    )
    _require(
        [int(row["acceptance_id"]) for row in ordered]
        == list(range(1, len(ordered) + 1)),
        "accepted Loop Closure IDs are not contiguous",
    )
    return tuple(
        (int(row["previous_frame_id"]), int(row["current_frame_id"]))
        for row in ordered[:state_id]
    )


def _load_npz_state(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        pose_key = (
            "optimized_poses"
            if "optimized_poses" in data.files
            else "initial_poses"
        )
        poses = np.asarray(data[pose_key], dtype=float)
        marginals = (
            np.asarray(data["marginal_covariances"], dtype=float)
            if "marginal_covariances" in data.files
            else None
        )
    _require(
        poses.shape == (len(frame_ids), 4, 4)
        and np.all(np.isfinite(poses)),
        "invalid saved Pose Graph values: {}".format(path),
    )
    return frame_ids, poses, marginals


def _load_canonical_evidence(
    *, paths: Optional[ReportEvidencePaths] = None
) -> Mapping[str, object]:
    """Load and align sealed report evidence without recomputing E7 science.

    E5 trajectories, E6 Values/Marginals, E7 accepted checkpoints, and sealed
    E7 trajectory/error CSVs must share the exact actual-FrameId order and
    provenance chain.  The returned arrays are inputs to plotting and scalar
    summaries only; no factor graph or estimator is constructed.
    """

    if paths is None:
        paths = ReportEvidencePaths.from_roots(
            e5_root=DEFAULT_E5_ROOT,
            e6_root=DEFAULT_E6_ROOT,
            scientific_root=DEFAULT_SCIENTIFIC_ROOT,
        )
    required = (
        paths.e5_root / "manifest.json",
        paths.e6_root / "summary.json",
        paths.e6_values,
        paths.e6_marginals,
        paths.e5_trajectories,
        paths.e7_trajectories,
        paths.e7_errors,
        paths.marginal_uncertainty_sizes,
        paths.accepted_loops,
        paths.matcher_invocations,
        paths.final_root / "manifest.json",
        paths.final_root / "summary.json",
        paths.final_root / "validation.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    _require(not missing, "required canonical source missing: " + ", ".join(missing))

    e5_manifest = _read_json(paths.e5_root / "manifest.json")
    e6_summary = _read_json(paths.e6_root / "summary.json")
    e7_summary = _read_json(paths.final_root / "summary.json")
    declared_e5_root = e6_summary.get("canonical_input_root")
    _require(
        e5_manifest.get("status") == "complete-and-sealed"
        and isinstance(declared_e5_root, str)
        and (REPOSITORY_ROOT / declared_e5_root).resolve()
        == paths.e5_root
        and e6_summary.get("canonical_configuration_sha256")
        == e5_manifest.get("configuration_sha256")
        and e7_summary.get("configuration_sha256")
        == e5_manifest.get("configuration_sha256"),
        "selected report roots do not form a coherent E5/E6/E7 chain",
    )

    frame_ids, baseline_poses, _ = _load_npz_state(paths.e6_values)
    with np.load(paths.e6_marginals, allow_pickle=False) as data:
        marginal_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        baseline_marginals = np.asarray(
            data["marginal_covariances"], dtype=float
        )
    require_aligned_frame_ids(frame_ids, marginal_ids, "E6 Marginals")
    _require(
        baseline_marginals.shape == (len(frame_ids), 6, 6)
        and np.all(np.isfinite(baseline_marginals)),
        "invalid protected E6 Marginals",
    )

    loop_rows = _read_csv(paths.accepted_loops)
    matcher_rows = _read_csv(paths.matcher_invocations)
    matcher_by_pair = {
        int(row["pair_index"]): row for row in matcher_rows
    }
    _require(
        len(matcher_by_pair) == len(matcher_rows),
        "matcher invocation pair indices are not unique",
    )
    for row in loop_rows:
        pair_index = int(row["pair_index"])
        _require(
            pair_index in matcher_by_pair,
            "accepted loop lacks its matcher invocation",
        )
        row["reciprocal_temporal_match_count"] = matcher_by_pair[pair_index][
            "reciprocal_temporal_match_count"
        ]
    select_snapshot_state_ids(len(loop_rows))
    states = {0: (baseline_poses, baseline_marginals)}
    for row in loop_rows:
        acceptance_id = int(row["acceptance_id"])
        artifact = (REPOSITORY_ROOT / row["artifact_path"]).resolve()
        _require(
            paths.scientific_root in artifact.parents,
            "accepted artifact belongs to a different E7 scientific root",
        )
        _require(artifact.is_file(), "accepted artifact missing: {}".format(artifact))
        ids, poses, marginals = _load_npz_state(artifact)
        require_aligned_frame_ids(frame_ids, ids, "E7 acceptance {}".format(acceptance_id))
        _require(
            marginals is not None
            and marginals.shape == baseline_marginals.shape
            and np.all(np.isfinite(marginals)),
            "invalid E7 Marginals at acceptance {}".format(acceptance_id),
        )
        states[acceptance_id] = (poses, marginals)

    e5_rows = _read_csv(paths.e5_trajectories)
    e7_rows = _read_csv(paths.e7_trajectories)
    error_rows = _read_csv(paths.e7_errors)
    e5_ids = np.asarray([int(row["frame_id"]) for row in e5_rows], dtype=np.int64)
    e7_ids = np.asarray([int(row["frame_id"]) for row in e7_rows], dtype=np.int64)
    error_ids = np.asarray([int(row["frame_id"]) for row in error_rows], dtype=np.int64)
    require_aligned_frame_ids(frame_ids, e5_ids, "E5 PnP/Bundle")
    require_aligned_frame_ids(frame_ids, e7_ids, "sealed E7 trajectory report")
    require_aligned_frame_ids(frame_ids, error_ids, "sealed E7 error report")

    xyz = lambda rows, prefix: np.asarray(
        [
            [
                float(row[prefix + "_x_m"]),
                float(row[prefix + "_y_m"]),
                float(row[prefix + "_z_m"]),
            ]
            for row in rows
        ],
        dtype=float,
    )
    truth_xyz = xyz(e5_rows, "ground_truth")
    pnp_xyz = xyz(e5_rows, "pnp")
    bundle_xyz = xyz(e5_rows, "bundle")
    e7_truth_xyz = xyz(e7_rows, "ground_truth")
    e7_baseline_xyz = xyz(e7_rows, "without_loop")
    e7_final_xyz = xyz(e7_rows, "with_loop")
    _require(
        np.allclose(truth_xyz, e7_truth_xyz, rtol=0.0, atol=1e-12)
        and np.allclose(
            baseline_poses[:, :3, 3], e7_baseline_xyz, rtol=0.0, atol=1e-12
        )
        and np.allclose(
            states[6][0][:, :3, 3], e7_final_xyz, rtol=0.0, atol=1e-12
        ),
        "trajectory coordinate convention or canonical source alignment differs",
    )
    return {
        "frame_ids": frame_ids,
        "states": states,
        "accepted_loops": loop_rows,
        "truth_xyz": truth_xyz,
        "pnp_xyz": pnp_xyz,
        "bundle_xyz": bundle_xyz,
        "error_rows": error_rows,
    }


def _global_translation_covariance(
    pose: np.ndarray, marginal: np.ndarray
) -> np.ndarray:
    """Rotate the raw local Pose3 translation block into global coordinates."""

    rotation = pose[:3, :3]
    local = marginal[3:6, 3:6]
    return rotation @ local @ rotation.T


def _add_covariance_ellipse(
    axis,
    center_xz: np.ndarray,
    covariance_xz: np.ndarray,
    *,
    radius_scale: float = 1.0,
) -> None:
    """Draw the 2D 95% chi-square contour of one raw global X-Z block.

    ``CHI_SQUARE_2D_95`` converts covariance eigenvalues to 95% semi-axis
    radii; it does not replace or modify the supplied covariance matrix.
    """

    symmetric = (covariance_xz + covariance_xz.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    _require(
        float(np.min(eigenvalues)) >= -1e-10,
        "raw location covariance is not positive semidefinite for plotting",
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    _require(
        math.isfinite(float(radius_scale)) and float(radius_scale) > 0.0,
        "covariance display radius scale must be positive and finite",
    )
    major, minor = (
        float(radius_scale)
        * np.sqrt(CHI_SQUARE_2D_95 * eigenvalues[order])
    )
    direction = eigenvectors[:, order[0]]
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    axis.add_patch(
        Ellipse(
            xy=center_xz,
            width=2.0 * major,
            height=2.0 * minor,
            angle=angle,
            fill=False,
            edgecolor="tab:orange",
            linewidth=0.55,
            alpha=0.75,
        )
    )


def plot_emphasized_successful_match(
    output_path: Path,
    *,
    accepted_artifact: Path,
    previous_image_path: Path,
    current_image_path: Path,
) -> None:
    """Replot the first sealed consensus with non-supporters emphasized.

    The images, four-view consensus pixels, and supporter mask are copied
    from the accepted-loop seal.  No descriptors, matching, RANSAC, or pose
    estimation is performed.
    """

    previous_image = plt.imread(previous_image_path)
    current_image = plt.imread(current_image_path)
    _require(
        previous_image.ndim == 2 and current_image.ndim == 2,
        "sealed successful-match images must be grayscale",
    )
    with np.load(accepted_artifact, allow_pickle=False) as data:
        pixels = np.asarray(data["consensus_four_view_pixels"], dtype=float)
        supporter_mask = np.asarray(
            data["consensus_supporter_mask"], dtype=bool
        )
    _require(
        pixels.ndim == 3
        and pixels.shape[:2] == (4, 2)
        and pixels.shape[2] == len(supporter_mask)
        and np.all(np.isfinite(pixels)),
        "sealed successful-match pixel evidence is invalid",
    )
    previous_height, previous_width = previous_image.shape
    current_height, current_width = current_image.shape
    canvas = np.zeros(
        (max(previous_height, current_height), previous_width + current_width),
        dtype=previous_image.dtype,
    )
    canvas[:previous_height, :previous_width] = previous_image
    canvas[:current_height, previous_width:] = current_image

    figure, axis = plt.subplots(figsize=(16, 6))
    axis.imshow(canvas, cmap="gray")
    for consensus_index in range(pixels.shape[2]):
        supporter = bool(supporter_mask[consensus_index])
        axis.plot(
            [
                pixels[0, 0, consensus_index],
                pixels[2, 0, consensus_index] + previous_width,
            ],
            [pixels[0, 1, consensus_index], pixels[2, 1, consensus_index]],
            color="cyan" if supporter else "red",
            alpha=0.50 if supporter else 0.90,
            linewidth=0.55 if supporter else 1.10,
            zorder=2 if supporter else 3,
        )
    supporter_count = int(np.count_nonzero(supporter_mask))
    consensus_count = int(len(supporter_mask))
    axis.axvline(previous_width, color="white", linewidth=0.5)
    axis.legend(
        handles=[
            Line2D([0], [0], color="cyan", label="supporter"),
            Line2D(
                [0],
                [0],
                color="red",
                linewidth=1.5,
                label="consensus non-supporter (emphasized)",
            ),
        ],
        loc="lower center",
        ncol=2,
        framealpha=0.8,
    )
    axis.set_axis_off()
    axis.set_title(
        "First accepted Loop Closure from sealed evidence\n"
        "{} / {} supporters ({:.3f}%); non-supporters emphasized".format(
            supporter_count,
            consensus_count,
            100.0 * supporter_count / consensus_count,
        )
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    plt.close(figure)


def _covariance_display_indices(
    frame_ids: np.ndarray,
    accepted_loops: Sequence[Mapping[str, object]],
    *,
    subsample_stride: int,
) -> Tuple[int, ...]:
    """Select stride poses, the final pose, and every accepted-loop endpoint."""

    _require(
        int(subsample_stride) > 0,
        "covariance ellipse subsample stride must be positive",
    )
    index_by_frame = {
        int(frame_id): index for index, frame_id in enumerate(frame_ids)
    }
    endpoint_indices = set()
    for row in accepted_loops:
        for field in ("previous_frame_id", "current_frame_id"):
            frame_id = int(row[field])
            _require(
                frame_id in index_by_frame,
                "accepted-loop endpoint is absent from canonical FrameIds: {}".format(
                    frame_id
                ),
            )
            endpoint_indices.add(index_by_frame[frame_id])
    return tuple(
        sorted(
            set(range(0, len(frame_ids), int(subsample_stride)))
            | endpoint_indices
            | {len(frame_ids) - 1}
        )
    )


def plot_pose_graph_snapshot_covariances(
    output_path: Path,
    *,
    frame_ids: np.ndarray,
    states: Mapping[int, Tuple[np.ndarray, np.ndarray]],
    accepted_loops: Sequence[Mapping[str, object]],
    subsample_stride: int = ELLIPSE_SUBSAMPLE_STRIDE,
    radius_scale: float = 1.0,
) -> None:
    """Plot four sealed states, accepted edges, and raw X-Z 95% ellipses.

    For each displayed pose, the raw local translation covariance is rotated
    into the global frame, reduced to its X-Z block, and shown at the 2D 95%
    chi-square contour.  ``subsample_stride`` controls only which ellipses are
    drawn; every accepted-loop endpoint and the final keyframe remain visible
    in every snapshot state, while red edges remain state-dependent.  No
    covariance array or scientific result is changed.  "Raw" describes the
    unscaled saved covariance; "95%" describes its presentation contour.  A
    ``radius_scale`` is display-only: it multiplies drawn ellipse radii after
    the raw-covariance 95% contour is computed and never changes any matrix.
    """

    state_ids = select_snapshot_state_ids(len(accepted_loops))
    index_by_frame = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    display_indices = _covariance_display_indices(
        frame_ids,
        accepted_loops,
        subsample_stride=subsample_stride,
    )
    labels = (
        "E6 baseline (0 accepted loops)",
        "Immediately after acceptance 1",
        "Deterministic middle: acceptance 3",
        "Final: acceptance 6",
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 11))
    for axis, state_id, label in zip(axes.ravel(), state_ids, labels):
        poses, marginals = states[state_id]
        centers = poses[:, :3, 3]
        axis.plot(centers[:, 0], centers[:, 2], color="tab:blue", linewidth=1.0)
        edges = accepted_edges_for_state(accepted_loops, state_id)
        for previous, current in edges:
            previous_index = index_by_frame[previous]
            current_index = index_by_frame[current]
            axis.plot(
                [centers[previous_index, 0], centers[current_index, 0]],
                [centers[previous_index, 2], centers[current_index, 2]],
                color="tab:red",
                linewidth=1.4,
                alpha=0.8,
            )
        for index in display_indices:
            global_covariance = _global_translation_covariance(
                poses[index], marginals[index]
            )
            _add_covariance_ellipse(
                axis,
                centers[index, [0, 2]],
                global_covariance[np.ix_([0, 2], [0, 2])],
                radius_scale=radius_scale,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        axis.legend(
            handles=[
                Line2D([0], [0], color="tab:blue", label="Pose Graph trajectory"),
                Line2D([0], [0], color="tab:red", label="accepted loop edge"),
                Line2D(
                    [0],
                    [0],
                    color="tab:orange",
                    label=(
                        "raw 95% covariance"
                        if radius_scale == 1.0
                        else "raw 95% covariance, displayed at x{} radius".format(
                            int(radius_scale)
                        )
                    ),
                ),
            ],
            loc="best",
            fontsize=7,
        )
    figure.suptitle(
        "Pose Graph states with accepted edges and raw 95% location covariance\n"
        "Ellipses every {}th keyframe plus every accepted-loop endpoint in every state; "
        "radius x{} for display only; red edges are state-dependent".format(
            subsample_stride, "{:g}".format(radius_scale)
        )
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _plot_trajectory(
    path: Path, series: Sequence[Tuple[str, np.ndarray]]
) -> None:
    figure, axis = plt.subplots(figsize=(9, 7))
    for label, xyz in series:
        axis.plot(xyz[:, 0], xyz[:, 2], label=label, linewidth=1.2)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Canonical KITTI 00 keyframe trajectories")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_absolute_location_errors(
    path: Path, frame_ids: np.ndarray, rows: Sequence[Mapping[str, str]]
) -> None:
    """Plot absolute X/Y/Z/norm error [m] before and after Loop Closure."""

    fields = (
        ("absolute_x_error_m", "absolute X error [m]"),
        ("absolute_y_error_m", "absolute Y error [m]"),
        ("absolute_z_error_m", "absolute Z error [m]"),
        ("location_error_norm_m", "total location-error norm [m]"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, (suffix, label) in zip(axes.ravel(), fields):
        for prefix, legend in (
            ("without_loop_", "without Loop Closure"),
            ("with_loop_", "with Loop Closure"),
        ):
            values = np.asarray([float(row[prefix + suffix]) for row in rows])
            axis.plot(frame_ids, values, label=legend)
        axis.set_xlabel("FrameId [frame]")
        axis.set_ylabel(label)
        axis.legend(fontsize=8)
    figure.suptitle("Absolute Pose Graph location errors")
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_rotation_errors(
    path: Path, frame_ids: np.ndarray, rows: Sequence[Mapping[str, str]]
) -> None:
    """Plot absolute rotation error [deg] before and after Loop Closure."""

    figure, axis = plt.subplots(figsize=(10, 5))
    for prefix, legend in (
        ("without_loop_", "without Loop Closure"),
        ("with_loop_", "with Loop Closure"),
    ):
        axis.plot(
            frame_ids,
            [float(row[prefix + "rotation_error_deg"]) for row in rows],
            label=legend,
        )
    axis.set_xlabel("FrameId [frame]")
    axis.set_ylabel("absolute rotation error [deg]")
    axis.set_title("Absolute Pose Graph rotation error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_location_uncertainty(
    path: Path,
    frame_ids: np.ndarray,
    baseline_marginals: np.ndarray,
    final_marginals: np.ndarray,
) -> None:
    """Plot ``sqrt(trace(Sigma[3:6, 3:6]))`` from raw Marginals [m].

    This scalar is a size summary of the local translation block, not a 95%
    contour and not a modified covariance.  The analogous sealed angular
    convention uses ``sqrt(trace(Sigma[0:3, 0:3]))`` in radians, with degree
    conversion only when a report field is explicitly labelled in degrees.
    """

    figure, axis = plt.subplots(figsize=(10, 5))
    for marginals, label in (
        (baseline_marginals, "without Loop Closure"),
        (final_marginals, "with Loop Closure"),
    ):
        values = np.sqrt(
            np.maximum(
                np.trace(marginals[:, 3:6, 3:6], axis1=1, axis2=2), 0.0
            )
        )
        axis.plot(frame_ids, values, label=label)
    axis.set_xlabel("FrameId [frame]")
    axis.set_ylabel(r"location uncertainty $\sqrt{trace(\Sigma_t)}$ [m]")
    axis.set_title("Raw Pose3 Marginal location uncertainty")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_angular_uncertainty(
    path: Path,
    frame_ids: np.ndarray,
    baseline_marginals: np.ndarray,
    final_marginals: np.ndarray,
) -> None:
    """Plot ``sqrt(trace(Sigma[0:3, 0:3]))`` in degrees.

    Saved Pose3 Marginals use rotation-radian tangent coordinates first.
    Conversion to degrees changes only the scalar report unit.
    """

    figure, axis = plt.subplots(figsize=(10, 5))
    for marginals, label in (
        (baseline_marginals, "without Loop Closure"),
        (final_marginals, "with Loop Closure"),
    ):
        radians = np.sqrt(
            np.maximum(
                np.trace(marginals[:, 0:3, 0:3], axis1=1, axis2=2), 0.0
            )
        )
        axis.plot(frame_ids, np.degrees(radians), label=label)
    axis.set_xlabel("FrameId [frame]")
    axis.set_ylabel(
        r"angular uncertainty $\sqrt{trace(\Sigma_r)}$ [deg]"
    )
    axis.set_title("Raw Pose3 Marginal angular uncertainty")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def plot_accepted_loop_statistics(
    path: Path, accepted_loops: Sequence[Mapping[str, object]]
) -> None:
    """Plot consensus/supporter counts and supporter percentage for six loops."""

    labels = [
        "{}→{}".format(row["previous_frame_id"], row["current_frame_id"])
        for row in accepted_loops
    ]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    specifications = (
        (
            "consensus_count",
            "four-view consensus matches [matches]",
        ),
        ("supporter_count", "supporter count [inliers]"),
        ("supporter_percentage", "supporter percentage [%]"),
    )
    for axis, (field, label) in zip(axes, specifications):
        axis.bar(labels, [float(row[field]) for row in accepted_loops])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("accepted Loop Closure pair [previous→current FrameId]")
    figure.suptitle("Accepted Loop Closure match statistics")
    figure.tight_layout()
    figure.savefig(path, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def _accepted_loop_summary_record(
    row: Mapping[str, object]
) -> Mapping[str, object]:
    """Project one sealed accepted-loop row into report-facing terminology."""

    integer_fields = (
        "acceptance_id",
        "previous_frame_id",
        "current_frame_id",
        "previous_keyframe_ordinal",
        "current_keyframe_ordinal",
        "reciprocal_temporal_match_count",
        "consensus_count",
        "supporter_count",
        "eligible_landmark_count",
        "state_version_after",
    )
    record = {key: int(row[key]) for key in integer_fields}
    record["supporter_percentage"] = float(row["supporter_percentage"])
    record["ransac_terminal_status"] = str(row["ransac_terminal_status"])
    return record


def _summary_record(
    evidence: Mapping[str, object], paths: ReportEvidencePaths
) -> Mapping[str, object]:
    canonical = _read_json(paths.final_root / "summary.json")
    report_metrics = canonical["report"]["metrics"]
    return {
        "schema": "exercise7-report-freeze-summary-v1",
        "status": "pass",
        "sequence": "KITTI 00",
        "candidate_policy": _read_json(
            paths.scientific_root / "policy.json"
        ),
        "counts": {
            "keyframes": 356,
            "ordered_pairs": 63190,
            "matcher_invocations": 1002,
            "accepted_loops": 6,
            "candidate_local_measurement_failures": 2,
            "final_pose_graph_factors": 362,
            "candidate_graph_edges": 361,
            "valid_final_marginals": 356,
        },
        "accepted_loops": [
            _accepted_loop_summary_record(row)
            for row in evidence["accepted_loops"]
        ],
        "metrics": report_metrics,
        "scientific_behavior": {
            "matching_acceptance": (
                "complete geometry and frozen 30 consensus / 30 supporters / "
                "30% supporter gates"
            ),
            "bundle_measurement": (
                "two-pose stereo Bundle; Pose Graph relative pose used only "
                "for initialization; raw conditional covariance retained"
            ),
            "update_order": (
                "insert one factor, immediately optimize cumulative Pose "
                "Graph, recompute all 356 Marginals, then update detection edge"
            ),
            "candidate_local_failures": (
                "two passing matches failed exact local-gauge measurement "
                "construction; neither mutated or suppressed the graph"
            ),
            "covariance_modified": False,
        },
        "limitations": [
            "Evidence is specific to KITTI sequence 00 and this frozen policy.",
            "The large improvement is not a cross-sequence generalization claim.",
            "Accepted Loop Closure measurements can remain correlated with the visual front end.",
        ],
        "traceability_status": "complete",
        "historical_sealed_traceability_pending_field_modified": False,
    }


def _summary_markdown(summary: Mapping[str, object]) -> str:
    metrics = summary["metrics"]
    return """# Exercise 7 Loop Closure report-freeze notes

## Section-ready prose

The frozen sequential policy evaluates the lecturer-style accumulated raw
covariance path score with strict $d^2 < 36,000,000$, excludes the prior
15 keyframe ordinals, and permits at most three actual matcher invocations per
current keyframe. A match must retain complete four-view geometry and pass
30 consensus, 30 supporter, and 30% supporter gates. Its two-pose stereo
Bundle uses the current Pose Graph relative pose only for initialization and
returns the previous-to-current relative pose with its raw conditional
end-given-start covariance.

Six Loop Closures were accepted. Each factor was inserted into the cumulative
Pose Graph and followed immediately by optimization and recomputation of all
356 Marginals before later detection. Two otherwise passing candidates failed
the exact local-gauge small-Bundle measurement gate; they were retained as
candidate-local failures and changed neither the graph nor cluster suppression.

Mean absolute location error improved from {before_loc:.6f} m to
{after_loc:.6f} m, and RMSE improved from {before_rmse:.6f} m to
{after_rmse:.6f} m. Mean absolute rotation error improved from
{before_rot:.6f} deg to {after_rot:.6f} deg. Mean raw location uncertainty
size fell from {before_unc:.6f} m to {after_unc:.6f} m.

## Captions

- `first_accepted_successful_match.png`: first accepted pair, with cyan
  supporters and red consensus non-supporters on the two left images.
- `pose_graph_snapshots_raw_95_covariance.png`: E6 baseline, acceptance 1,
  deterministic middle acceptance 3, and final acceptance 6. Red chords are
  accepted Loop Closure edges present at each state. Orange ellipses are raw
  top-down 95% location covariances, transformed from each local translation
  block into the global frame. Ellipses are displayed every 12th keyframe,
  at the final keyframe, and at every accepted-loop endpoint in every state;
  this is plotting-only subsampling, while red edges remain state-dependent.
- `pose_graph_snapshots_dense_raw_95_covariance.png`: the same four states and
  raw covariance treatment at every 4th keyframe plus all accepted-loop
  endpoints in every state.
- `pose_graph_snapshots_dense_x5_radius.png` and
  `pose_graph_snapshots_dense_x10_radius.png`: the dense selection with only
  the displayed 95% ellipse radii multiplied by 5 or 10. Raw covariance
  matrices are unchanged.
- `trajectory_ground_truth_e6_e7.png`: Ground Truth and Pose Graph trajectories
  without and with Loop Closure on common top-down axes.
- `absolute_location_errors.png` and `absolute_rotation_errors.png`: absolute
  component/norm errors in meters and rotation error in degrees.
- `location_uncertainty.png`: raw translation-block uncertainty size
  `sqrt(trace(Sigma[3:6,3:6]))` in meters.
- `angular_uncertainty.png`: raw rotation-block uncertainty size
  `sqrt(trace(Sigma[0:3,0:3]))`, converted from radians to degrees only for
  presentation. `data/marginal_uncertainty_sizes.csv` retains the already
  sealed per-keyframe scalar evidence in both declared angular units.
- `accepted_loop_statistics.png`: consensus, supporter, and supporter-rate
  evidence for all six accepted loops.

## Findings and limitations

This is strong evidence for the frozen course pipeline on KITTI sequence 00,
not a claim of generalization to other sequences, cameras, or calibrations.
The front end and Loop Closure measurements share imagery, so independence is
not claimed. Covariances were neither scaled nor repaired. No new matching,
RANSAC, Bundle solve, Pose Graph optimization, or scientific result was
performed for this report freeze.
""".format(
        before_loc=metrics["without_loop"]["location_error_m"]["mean"],
        after_loc=metrics["with_loop"]["location_error_m"]["mean"],
        before_rmse=metrics["without_loop"]["location_error_m"]["rmse"],
        after_rmse=metrics["with_loop"]["location_error_m"]["rmse"],
        before_rot=metrics["without_loop"]["absolute_rotation_error_deg"]["mean"],
        after_rot=metrics["with_loop"]["absolute_rotation_error_deg"]["mean"],
        before_unc=metrics["without_loop"]["location_uncertainty_m"]["mean"],
        after_unc=metrics["with_loop"]["location_uncertainty_m"]["mean"],
    )


def _function_line_range(path: Path, qualified_name: str) -> Tuple[int, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = qualified_name.split(".")
    nodes = list(tree.body)
    found = None
    for part in parts:
        found = next(
            (
                node
                for node in nodes
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == part
            ),
            None,
        )
        _require(found is not None, "traceability function not found: {} {}".format(path, qualified_name))
        nodes = list(getattr(found, "body", ()))
    return int(found.lineno), int(found.end_lineno)


def build_source_traceability() -> Mapping[str, object]:
    """Build exact post-edit source hashes and AST function line ranges."""

    specifications = (
        ("exercise7_candidate_detection.py", "CovarianceGraph.find_course_covariance_path", "candidate path detection", "raw 6x6 covariance sum; determinant path priority", "dimensionless determinant priority"),
        ("exercise7_candidate_detection.py", "evaluate_gtsam_mahalanobis", "Mahalanobis candidate scoring", "d_squared = 2 * BetweenFactorPose3.error", "dimensionless squared residual"),
        ("exercise7_consensus_matching.py", "bounded_four_view_ransac", "consensus matching/RANSAC", "four-view 2 px supporter rule", "pixels, matches, percent"),
        ("exercise7_consensus_matching.py", "evaluate_candidate_pair", "candidate match evaluation", "reciprocal temporal matching plus bounded RANSAC", "pixels, matches, percent"),
        ("exercise7_small_bundle.py", "solve_small_bundle", "small-Bundle construction and measurement extraction", "relative pose and conditional end-given-start covariance", "rad and m tangent order"),
        ("exercise7_pose_graph_update.py", "CumulativePoseGraphState.apply_loop_factor", "cumulative factor insertion transaction", "one BetweenFactorPose3 per accepted loop", "raw 6x6 covariance"),
        ("exercise7_pose_graph_update.py", "_compute_loop_factor_update", "immediate optimization and Marginals", "LM then all-node Marginals", "Pose3 rad/m"),
        ("exercise7_sequential_loop_closure.py", "_save_successful_match_figure", "retained successful-match figure", "supporter versus consensus non-supporter", "pixels, matches, percent"),
        ("exercise7_sequential_loop_closure.py", "_plot_result_aware_snapshots", "retained scalar uncertainty heatmap", "sqrt(trace(raw translation block))", "m"),
        ("exercise7_report_freeze.py", "plot_emphasized_successful_match", "emphasized successful-match figure", "sealed four-view pixels and supporter mask; no matching", "pixels, matches, percent"),
        ("exercise7_report_freeze.py", "plot_pose_graph_snapshot_covariances", "raw covariance snapshot figure", "global R Sigma_t R^T; 2D chi-square 95% ellipse", "m"),
        ("exercise7_report_freeze.py", "_plot_trajectory", "trajectory figures", "top-down camera translations", "m"),
        ("exercise7_report_freeze.py", "plot_absolute_location_errors", "absolute location-error figure", "|estimate - truth| and L2 norm", "m"),
        ("exercise7_report_freeze.py", "plot_rotation_errors", "absolute rotation-error figure", "acos((trace(R_truth^-1 R_est)-1)/2)", "deg"),
        ("exercise7_report_freeze.py", "plot_location_uncertainty", "location uncertainty figure", "sqrt(trace(raw Sigma[3:6,3:6]))", "m"),
        ("exercise7_report_freeze.py", "plot_angular_uncertainty", "angular uncertainty figure", "degrees(sqrt(trace(raw Sigma[0:3,0:3])))", "deg"),
        ("exercise7_report_freeze.py", "plot_accepted_loop_statistics", "accepted-loop match statistics", "consensus/supporter counts and ratio", "matches, inliers, percent"),
        ("ex7.py", "dispatch_command", "canonical command dispatch", "safe ordered delegation", "n/a"),
    )
    records = []
    for filename, function, purpose, formula, units in specifications:
        path = REPOSITORY_ROOT / filename
        first, last = _function_line_range(path, function)
        records.append(
            {
                "filename": filename,
                "function_name": function,
                "first_line": first,
                "last_line": last,
                "source_sha256": file_sha256(path),
                "purpose": purpose,
                "input_artifacts": _traceability_inputs(purpose),
                "output_artifacts": _traceability_outputs(purpose),
                "formula_or_metric": formula,
                "units": units,
            }
        )
    return {
        "schema": "exercise7-report-freeze-traceability-v1",
        "status": "complete",
        "report_freeze_line_refresh": "complete",
        "sealed_historical_traceability_immutable": True,
        "records": records,
    }


def _traceability_inputs(purpose: str) -> List[str]:
    if "dispatch" in purpose:
        return ["CLI arguments"]
    if purpose in ("candidate path detection", "Mahalanobis candidate scoring"):
        return ["optimized Pose Graph Values", "raw covariance graph"]
    if "match" in purpose or "RANSAC" in purpose:
        return ["declared candidate pair", "left/right image evidence"]
    if "Bundle" in purpose:
        return ["supporter stereo observations", "current Pose Graph initialization"]
    if "factor" in purpose or "optimization" in purpose:
        return ["current cumulative graph/Values", "raw loop relative pose/covariance"]
    return [
        "diagnostics/ex6_clean_final/*.npz",
        "diagnostics/ex7_clean_final sealed arrays/CSVs",
    ]


def _traceability_outputs(purpose: str) -> List[str]:
    if "dispatch" in purpose:
        return ["JSON command result"]
    if "figure" in purpose or "trajectory" in purpose or "uncertainty" in purpose or "statistics" in purpose:
        return ["diagnostics/ex7_clean_final/report_freeze/figures/"]
    return ["sealed Exercise 7 stage evidence"]


def _requirements_mapping() -> Mapping[str, object]:
    return {
        "schema": "exercise7-report-freeze-requirements-v1",
        "requirements": [
            {"id": "E7-successful-match", "outputs": ["figures/first_accepted_successful_match.png", "figures/first_accepted_successful_match_non_supporters_emphasized.png"]},
            {"id": "E7-four-state-pose-graph", "outputs": ["figures/pose_graph_snapshots_raw_95_covariance.png", "figures/pose_graph_snapshots_dense_raw_95_covariance.png", "figures/pose_graph_snapshots_dense_x5_radius.png", "figures/pose_graph_snapshots_dense_x10_radius.png", "figures/pose_graph_snapshot_uncertainty_heatmap.png", "data/snapshot_selection.json"]},
            {"id": "E7-trajectory-errors-uncertainty", "outputs": ["figures/trajectory_ground_truth_e6_e7.png", "figures/absolute_location_errors.png", "figures/absolute_rotation_errors.png", "figures/location_uncertainty.png", "figures/angular_uncertainty.png", "data/marginal_uncertainty_sizes.csv"]},
            {"id": "E7-accepted-match-statistics", "outputs": ["figures/accepted_loop_statistics.png", "data/accepted_loops.csv", "data/accepted_loop_summary.csv"]},
            {"id": "E7-summary-traceability", "outputs": ["summary.json", "report_notes.md", "traceability.json", "requirements_mapping.json"]},
            {"id": "Project-Submission-five-trajectories", "outputs": ["project_submission/loop_closure_clean_final/figures/trajectory_all_stages.png"]},
            {"id": "Project-Submission-curated-provenance", "outputs": ["project_submission/loop_closure_clean_final/provenance.json", "project_submission/loop_closure_clean_final/manifest.json", "project_submission/loop_closure_clean_final/validation.json"]},
        ],
    }


def _require_safe_publish_target(target: Path) -> Path:
    """Prevent publication over scientific or historical evidence roots."""

    selected = Path(target).resolve()
    if selected in {
        DEFAULT_REPORT_ROOT.resolve(),
        DEFAULT_SUBMISSION_ROOT.resolve(),
    }:
        return selected
    for protected in (
        DEFAULT_E5_ROOT,
        DEFAULT_E6_ROOT,
        DEFAULT_SCIENTIFIC_ROOT,
        HISTORICAL_E5_ROOT,
        HISTORICAL_E6_ROOT,
        HISTORICAL_SCIENTIFIC_ROOT,
        CLEAN_EQUIVALENCE_ROOT,
        HISTORICAL_BUNDLE_SUBMISSION_ROOT,
        CLEAN_BUNDLE_SUBMISSION_ROOT,
        HISTORICAL_SUBMISSION_ROOT,
        DEFAULT_SUBMISSION_ROOT,
    ):
        protected = protected.resolve()
        _require(
            not (
                selected == protected
                or selected in protected.parents
                or protected in selected.parents
            ),
            "refusing to publish over protected scientific/historical evidence: {}".format(
                protected
            ),
        )
    return selected


def _prepare_staging(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=target.name + ".staging.", dir=str(target.parent)))


def _publish_staging(staging: Path, target: Path, schema: str) -> None:
    if target.exists():
        summary = target / "summary.json"
        _require(summary.is_file(), "refusing to replace unmanaged output: {}".format(target))
        existing = _read_json(summary)
        _require(existing.get("schema") == schema, "refusing to replace output with another schema")
        shutil.rmtree(target)
    staging.replace(target)


def _manifest_for(root: Path, schema: str) -> Mapping[str, object]:
    artifacts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"):
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {"schema": schema, "status": "complete-and-validated", "artifacts": artifacts}


def validate_checksum_manifest(root: Path) -> Mapping[str, object]:
    """Validate every file declared by a package checksum manifest."""

    manifest_path = Path(root) / "manifest.json"
    _require(manifest_path.is_file(), "manifest is missing")
    manifest = _read_json(manifest_path)
    for artifact in manifest.get("artifacts", []):
        path = Path(root) / artifact["path"]
        _require(path.is_file(), "manifest artifact missing: {}".format(artifact["path"]))
        _require(path.stat().st_size == artifact["size_bytes"], "manifest size mismatch: {}".format(artifact["path"]))
        _require(file_sha256(path) == artifact["sha256"], "manifest checksum mismatch: {}".format(artifact["path"]))
    return {"status": "pass", "artifact_count": len(manifest.get("artifacts", []))}


def generate_report_freeze(
    *,
    output_root: Path = DEFAULT_REPORT_ROOT,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Generate and validate presentation products from sealed evidence only.

    The function aligns and copies existing scientific evidence, computes
    report metrics, plots it in a staging directory, publishes the report
    package, and validates its checksum manifest.  It never invokes matching,
    RANSAC, small Bundle, Pose Graph optimization, or Marginals construction.
    """

    output_root = _require_safe_publish_target(output_root)
    paths = ReportEvidencePaths.from_roots(
        e5_root=e5_root,
        e6_root=e6_root,
        scientific_root=scientific_root,
    )
    evidence = _load_canonical_evidence(paths=paths)
    staging = _prepare_staging(output_root)
    figures = staging / "figures"
    data = staging / "data"
    figures.mkdir()
    data.mkdir()
    frame_ids = evidence["frame_ids"]
    states = evidence["states"]
    accepted = evidence["accepted_loops"]

    shutil.copyfile(
        paths.scientific_root
        / "currents/current_0160/successful_match.png",
        figures / "first_accepted_successful_match.png",
    )
    first_accepted = accepted[0]
    plot_emphasized_successful_match(
        figures / "first_accepted_successful_match_non_supporters_emphasized.png",
        accepted_artifact=(
            REPOSITORY_ROOT / first_accepted["artifact_path"]
        ).resolve(),
        previous_image_path=(
            REPOSITORY_ROOT / first_accepted["previous_left_image_path"]
        ).resolve(),
        current_image_path=(
            REPOSITORY_ROOT / first_accepted["current_left_image_path"]
        ).resolve(),
    )
    shutil.copyfile(
        paths.final_root
        / "reports/figures/result_aware_pose_graph_snapshots.png",
        figures / "pose_graph_snapshot_uncertainty_heatmap.png",
    )
    plot_pose_graph_snapshot_covariances(
        figures / "pose_graph_snapshots_raw_95_covariance.png",
        frame_ids=frame_ids,
        states=states,
        accepted_loops=accepted,
    )
    for filename, radius_scale in (
        ("pose_graph_snapshots_dense_raw_95_covariance.png", 1.0),
        ("pose_graph_snapshots_dense_x5_radius.png", 5.0),
        ("pose_graph_snapshots_dense_x10_radius.png", 10.0),
    ):
        plot_pose_graph_snapshot_covariances(
            figures / filename,
            frame_ids=frame_ids,
            states=states,
            accepted_loops=accepted,
            subsample_stride=4,
            radius_scale=radius_scale,
        )
    _plot_trajectory(
        figures / "trajectory_ground_truth_e6_e7.png",
        (
            ("Ground Truth", evidence["truth_xyz"]),
            ("E6 without Loop Closure", states[0][0][:, :3, 3]),
            ("E7 with Loop Closure", states[6][0][:, :3, 3]),
        ),
    )
    plot_absolute_location_errors(
        figures / "absolute_location_errors.png", frame_ids, evidence["error_rows"]
    )
    plot_rotation_errors(
        figures / "absolute_rotation_errors.png", frame_ids, evidence["error_rows"]
    )
    plot_location_uncertainty(
        figures / "location_uncertainty.png",
        frame_ids,
        states[0][1],
        states[6][1],
    )
    plot_angular_uncertainty(
        figures / "angular_uncertainty.png",
        frame_ids,
        states[0][1],
        states[6][1],
    )
    plot_accepted_loop_statistics(
        figures / "accepted_loop_statistics.png", accepted
    )

    shutil.copyfile(paths.accepted_loops, data / "accepted_loops.csv")
    accepted_summary_fields = (
        "acceptance_id",
        "previous_frame_id",
        "current_frame_id",
        "previous_keyframe_ordinal",
        "current_keyframe_ordinal",
        "reciprocal_temporal_match_count",
        "consensus_count",
        "supporter_count",
        "supporter_percentage",
        "ransac_terminal_status",
        "eligible_landmark_count",
        "state_version_after",
    )
    _write_csv(
        data / "accepted_loop_summary.csv",
        accepted,
        accepted_summary_fields,
    )
    shutil.copyfile(paths.e7_trajectories, data / "trajectory_comparison.csv")
    shutil.copyfile(paths.e7_errors, data / "absolute_pose_errors.csv")
    shutil.copyfile(
        paths.marginal_uncertainty_sizes,
        data / "marginal_uncertainty_sizes.csv",
    )
    all_endpoint_frame_ids = sorted(
        {
            int(row[field])
            for row in accepted
            for field in ("previous_frame_id", "current_frame_id")
        }
    )
    _write_json(
        data / "snapshot_selection.json",
        {
            "state_ids": list(select_snapshot_state_ids(len(accepted))),
            "ellipse_subsample_stride": ELLIPSE_SUBSAMPLE_STRIDE,
            "dense_ellipse_subsample_stride": 4,
            "display_radius_variants": [1, 5, 10],
            "subsampling_scope": (
                "plotting only; every accepted-loop endpoint retained in "
                "every snapshot state"
            ),
            "ellipse_endpoint_frame_ids": all_endpoint_frame_ids,
            "ellipse_endpoint_policy": (
                "all accepted-loop endpoints are included in states 0, 1, 3, and 6"
            ),
            "red_edge_policy": (
                "state-dependent; an edge is drawn only after its acceptance"
            ),
            "covariance": (
                "raw saved Marginals; x5/x10 multiply displayed 95% ellipse "
                "radii only"
            ),
            "edges_by_state": {
                str(state_id): accepted_edges_for_state(accepted, state_id)
                for state_id in SNAPSHOT_ACCEPTANCE_IDS
            },
        },
    )
    summary = _summary_record(evidence, paths)
    _write_json(staging / "summary.json", summary)
    (staging / "report_notes.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    _write_json(staging / "traceability.json", build_source_traceability())
    requirements = _requirements_mapping()
    _write_json(staging / "requirements_mapping.json", requirements)
    (staging / "requirements_mapping.md").write_text(
        "# Exercise 7 and submission output mapping\n\n"
        + "\n".join(
            "- `{}` → {}".format(
                row["id"], ", ".join("`{}`".format(path) for path in row["outputs"])
            )
            for row in requirements["requirements"]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        staging / "provenance.json",
        {
            "schema": "exercise7-report-freeze-provenance-v1",
            "scientific_recomputation": False,
            "coordinate_convention": "T_C0_from_Ck camera translations; top-down X-Z",
            "frame_id_alignment_verified": True,
            "selected_roots": {
                "e5_root": _display_path(paths.e5_root),
                "e6_root": _display_path(paths.e6_root),
                "scientific_root": _display_path(paths.scientific_root),
            },
            "source_sha256": {
                _display_path(path): file_sha256(path)
                for path in (
                    paths.e5_root / "manifest.json",
                    paths.e6_root / "summary.json",
                    paths.e6_values,
                    paths.e6_marginals,
                    paths.e5_trajectories,
                    paths.marginal_uncertainty_sizes,
                    paths.accepted_loops,
                    paths.matcher_invocations,
                    (
                        REPOSITORY_ROOT
                        / first_accepted["artifact_path"]
                    ).resolve(),
                    paths.final_root / "manifest.json",
                )
            },
        },
    )
    _write_json(staging / "manifest.json", _manifest_for(staging, "exercise7-report-freeze-manifest-v1"))
    validate_checksum_manifest(staging)
    _publish_staging(staging, output_root, "exercise7-report-freeze-summary-v1")
    return validate_report_freeze(
        output_root=output_root,
        e5_root=paths.e5_root,
        e6_root=paths.e6_root,
        scientific_root=paths.scientific_root,
    )


def validate_report_freeze(
    *,
    output_root: Path = DEFAULT_REPORT_ROOT,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Validate report products, provenance, counts, and checksums."""

    output_root = Path(output_root)
    manifest_result = validate_checksum_manifest(output_root)
    summary = _read_json(output_root / "summary.json")
    _require(
        summary.get("status") == "pass"
        and summary.get("traceability_status") == "complete"
        and summary["counts"]["accepted_loops"] == 6,
        "report-freeze summary is invalid",
    )
    traceability = _read_json(output_root / "traceability.json")
    _require(
        traceability.get("report_freeze_line_refresh") == "complete",
        "report-freeze source traceability is not complete",
    )
    _require(
        traceability == build_source_traceability(),
        "report-freeze source traceability differs from current source lines or hashes",
    )
    paths = ReportEvidencePaths.from_roots(
        e5_root=e5_root,
        e6_root=e6_root,
        scientific_root=scientific_root,
    )
    provenance = _read_json(output_root / "provenance.json")
    _require(
        provenance.get("selected_roots")
        == {
            "e5_root": _display_path(paths.e5_root),
            "e6_root": _display_path(paths.e6_root),
            "scientific_root": _display_path(paths.scientific_root),
        },
        "report-freeze selected-root provenance differs",
    )
    for source, expected_hash in provenance.get("source_sha256", {}).items():
        source_path = (REPOSITORY_ROOT / source).resolve()
        _require(
            source_path.is_file() and file_sha256(source_path) == expected_hash,
            "report-freeze source hash differs: {}".format(source),
        )
    return {
        "command": "report",
        "status": "pass",
        "output_root": _display_path(output_root),
        "artifact_count": manifest_result["artifact_count"],
        "accepted_loop_count": 6,
        "scientific_recomputation": False,
    }


def materialize_loop_closure_submission(
    *,
    report_root: Path = DEFAULT_REPORT_ROOT,
    output_root: Path = DEFAULT_SUBMISSION_ROOT,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Materialize a compact package without copying canonical current seals."""

    report_root = Path(report_root)
    output_root = _require_safe_publish_target(output_root)
    paths = ReportEvidencePaths.from_roots(
        e5_root=e5_root,
        e6_root=e6_root,
        scientific_root=scientific_root,
    )
    validate_report_freeze(
        output_root=report_root,
        e5_root=paths.e5_root,
        e6_root=paths.e6_root,
        scientific_root=paths.scientific_root,
    )
    evidence = _load_canonical_evidence(paths=paths)
    staging = _prepare_staging(output_root)
    (staging / "figures").mkdir()
    (staging / "data").mkdir()
    report_figures = (
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
    for name in report_figures:
        shutil.copyfile(report_root / "figures" / name, staging / "figures" / name)
    _plot_trajectory(
        staging / "figures/trajectory_all_stages.png",
        (
            ("Ground Truth", evidence["truth_xyz"]),
            ("PnP", evidence["pnp_xyz"]),
            ("Bundle", evidence["bundle_xyz"]),
            ("E6 Pose Graph without Loop Closure", evidence["states"][0][0][:, :3, 3]),
            ("E7 Pose Graph with Loop Closure", evidence["states"][6][0][:, :3, 3]),
        ),
    )
    for name in (
        "accepted_loops.csv",
        "accepted_loop_summary.csv",
        "trajectory_comparison.csv",
        "absolute_pose_errors.csv",
        "marginal_uncertainty_sizes.csv",
        "snapshot_selection.json",
    ):
        shutil.copyfile(report_root / "data" / name, staging / "data" / name)
    shutil.copyfile(report_root / "report_notes.md", staging / "report_notes.md")
    shutil.copyfile(report_root / "traceability.json", staging / "traceability.json")
    shutil.copyfile(report_root / "requirements_mapping.json", staging / "requirements_mapping.json")
    shutil.copyfile(report_root / "requirements_mapping.md", staging / "requirements_mapping.md")
    summary = dict(_read_json(report_root / "summary.json"))
    summary["schema"] = "exercise7-loop-closure-submission-summary-v1"
    summary["package_scope"] = "curated report-ready evidence; no current seals"
    _write_json(staging / "summary.json", summary)
    provenance = dict(_read_json(report_root / "provenance.json"))
    provenance["schema"] = "exercise7-loop-closure-submission-provenance-v1"
    provenance["report_freeze_manifest_sha256"] = file_sha256(report_root / "manifest.json")
    provenance["canonical_e7_final_manifest_sha256"] = file_sha256(
        paths.final_root / "manifest.json"
    )
    provenance["selected_roots"] = {
        "e5_root": _display_path(paths.e5_root),
        "e6_root": _display_path(paths.e6_root),
        "scientific_root": _display_path(paths.scientific_root),
    }
    _write_json(staging / "provenance.json", provenance)
    (staging / "README.md").write_text(
        "# Curated Loop Closure submission evidence\n\n"
        "Report-ready Exercise 7 evidence for KITTI sequence 00. The package "
        "contains compact CSV/JSON data, figures, exact source traceability, "
        "and provenance hashes. It intentionally excludes the 356 canonical "
        "current directories and performs no scientific recomputation.\n\n"
        "Insert `report_notes.md` prose and captions into the final report; "
        "use `figures/trajectory_all_stages.png` for the common-axis Project "
        "Submission trajectory comparison.\n",
        encoding="utf-8",
    )
    _write_json(
        staging / "validation.json",
        {
            "schema": "exercise7-loop-closure-submission-validation-v1",
            "status": "pass",
            "frame_id_alignment_verified": True,
            "coordinate_convention_verified": True,
            "scientific_recomputation": False,
            "canonical_current_directories_copied": False,
        },
    )
    _write_json(staging / "manifest.json", _manifest_for(staging, "exercise7-loop-closure-submission-manifest-v1"))
    validate_checksum_manifest(staging)
    _publish_staging(staging, output_root, "exercise7-loop-closure-submission-summary-v1")
    return validate_submission_package(
        output_root=output_root,
        e5_root=paths.e5_root,
        e6_root=paths.e6_root,
        scientific_root=paths.scientific_root,
    )


def validate_submission_package(
    *,
    output_root: Path = DEFAULT_SUBMISSION_ROOT,
    e5_root: Path = DEFAULT_E5_ROOT,
    e6_root: Path = DEFAULT_E6_ROOT,
    scientific_root: Path = DEFAULT_SCIENTIFIC_ROOT,
) -> Mapping[str, object]:
    """Independently validate curated package scope and manifest hashes."""

    output_root = Path(output_root)
    result = validate_checksum_manifest(output_root)
    paths = [path.relative_to(output_root).as_posix() for path in output_root.rglob("*") if path.is_file()]
    _require(
        not any(path.startswith("currents/") or "accepted_000" in path for path in paths),
        "curated submission copied canonical current/accepted arrays",
    )
    validation = _read_json(output_root / "validation.json")
    _require(validation.get("status") == "pass", "submission validation record is invalid")
    paths = ReportEvidencePaths.from_roots(
        e5_root=e5_root,
        e6_root=e6_root,
        scientific_root=scientific_root,
    )
    provenance = _read_json(output_root / "provenance.json")
    _require(
        provenance.get("selected_roots")
        == {
            "e5_root": _display_path(paths.e5_root),
            "e6_root": _display_path(paths.e6_root),
            "scientific_root": _display_path(paths.scientific_root),
        }
        and provenance.get("canonical_e7_final_manifest_sha256")
        == file_sha256(paths.final_root / "manifest.json"),
        "curated submission source provenance differs",
    )
    return {
        "command": "submission",
        "status": "pass",
        "output_root": _display_path(output_root),
        "artifact_count": result["artifact_count"],
        "size_bytes": sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file()),
        "canonical_current_directories_copied": False,
    }


__all__ = [
    "DEFAULT_REPORT_ROOT",
    "DEFAULT_SUBMISSION_ROOT",
    "DEFAULT_E5_ROOT",
    "DEFAULT_E6_ROOT",
    "DEFAULT_SCIENTIFIC_ROOT",
    "ReportEvidencePaths",
    "ReportFreezeError",
    "accepted_edges_for_state",
    "compute_absolute_pose_metrics",
    "file_sha256",
    "generate_report_freeze",
    "materialize_loop_closure_submission",
    "plot_angular_uncertainty",
    "plot_emphasized_successful_match",
    "plot_absolute_location_errors",
    "plot_accepted_loop_statistics",
    "plot_location_uncertainty",
    "plot_pose_graph_snapshot_covariances",
    "plot_rotation_errors",
    "require_aligned_frame_ids",
    "select_snapshot_state_ids",
    "validate_checksum_manifest",
    "validate_report_freeze",
    "validate_submission_package",
]
