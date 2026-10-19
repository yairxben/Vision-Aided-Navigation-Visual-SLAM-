#!/usr/bin/env python3
"""Generate the current-stage Project Submission evidence package.

This program is intentionally evidence-only.  It consumes the retained
Exercise 4 database and companion without running feature extraction,
matching, RANSAC, motion PnP, Bundle Adjustment, Pose Graph, or Loop Closure.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.dont_write_bytecode = True
MATPLOTLIB_CACHE = Path("/tmp") / (
    "ps_p01_matplotlib_cache_{}".format(os.getpid())
)
MATPLOTLIB_CACHE.mkdir(parents=False, exist_ok=False)
os.environ["MPLCONFIGDIR"] = str(MATPLOTLIB_CACHE)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "ps-p01-current-stage-v1"
WORK_PACKAGE = "PS-P01"
PDF_RELATIVE = "docs/submission/Project Submission - Ex8.pdf"
FINAL_RELATIVE = "diagnostics/ex4_final"
I04_RELATIVE = "diagnostics/ex4_i04"
DATABASE_SHA256 = (
    "e0eba0d75887af544f3fdc0ada50932adb5c1edf52ea983a2a1be2c8244ce187"
)
COMPANION_SHA256 = (
    "5bbd24404acf9c0b1b54bcd9183a1a711f574e532a327043bdd2477cb7e76360"
)
PDF_SHA256 = (
    "0b7279454d1fee04baeef1099c49a99545184064e371019a8a523e01c10987e5"
)
EXPECTED = {
    "frame_count": 3360,
    "transition_count": 3359,
    "track_count": 758478,
    "tracked_link_count": 2517474,
    "mean_track_length": 3.3191127494798796,
    "mean_frame_links": 749.2482142857143,
    "below_30_percent_count": 0,
    "eligible_projection_tracks": 19263,
    "final_position_error_m": 43.63117083651525,
    "total_ground_truth_distance_m": 2584.404086966866,
    "final_drift_percent": 1.6882487942403037,
    "reciprocal_match_total": 5727279,
    "consensus_match_total": 1948690,
    "final_supporter_total": 1758996,
    "relative_row_counts": {100: 3261, 400: 2961, 800: 2561},
}
ROOT = None
OUTPUT = None
COMMAND = None
RESUMED_BEFORE_DATABASE = False
AUTHORIZED_RECOVERY = False
RECOVERY_HISTORY = None

EXPECTED_TRACKING_SCIENTIFIC_STATE_SHA256 = (
    "8d833c8f7a451f58e6a6d66ae469cb0ce4b1f0dc3858d435d531df8458dd6c9a"
)
EXPECTED_PNP_SCIENTIFIC_STATE_SHA256 = (
    "14e5baf14ad02877d5f7881e3b637121c2d5a1968e1fd32e741103b243cbab87"
)
EXPECTED_PACKAGE_SCIENTIFIC_STATE_SHA256 = (
    "0c1519a19a86d4d02bc36b625ae24a507f33d71f67fceaf0b8e197319d741ace"
)


def attempt_history() -> Dict[str, object]:
    if not AUTHORIZED_RECOVERY:
        return {
            "recovery_mode": False,
            "attempts_before_current_process": [],
            "current_process": {
                "mode": "clean-normal-generation",
                "database_deserializations_planned": 1,
            },
        }
    if not isinstance(RECOVERY_HISTORY, dict):
        raise AssertionError(
            "authorized recovery requires explicit provenance JSON"
        )
    if RECOVERY_HISTORY.get("recovery_mode") is not True:
        raise AssertionError("recovery provenance must enable recovery_mode")
    if not isinstance(
        RECOVERY_HISTORY.get("attempts_before_current_process"), list
    ):
        raise AssertionError("recovery provenance attempts must be a list")
    return json.loads(json.dumps(RECOVERY_HISTORY, allow_nan=False))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("nonfinite JSON constant: {}".format(value))
            ),
        )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git"] + list(arguments),
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return completed.stdout


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def read_ground_truth(count: int) -> List[np.ndarray]:
    path = ROOT / "dataset" / "poses" / "00.txt"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < count:
        raise AssertionError("KITTI ground-truth pose file is too short")
    return [
        np.asarray(line.split(), dtype=float).reshape(3, 4)
        for line in lines[:count]
    ]


def read_calibration() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / "dataset" / "sequences" / "00" / "calib.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    projection_left = np.asarray(
        lines[0].split()[1:], dtype=float
    ).reshape(3, 4)
    projection_right = np.asarray(
        lines[1].split()[1:], dtype=float
    ).reshape(3, 4)
    intrinsic = projection_left[:, :3]
    left_pose = np.linalg.inv(intrinsic) @ projection_left
    right_pose = np.linalg.inv(intrinsic) @ projection_right
    expected_left = np.hstack((np.eye(3), np.zeros((3, 1))))
    if not np.allclose(left_pose, expected_left, atol=1e-12, rtol=0.0):
        raise AssertionError("left calibration pose is not identity")
    return intrinsic, left_pose, right_pose


def relative_pose(back: np.ndarray, front: np.ndarray) -> np.ndarray:
    back_rotation = back[:, :3]
    back_translation = back[:, 3:]
    front_rotation = front[:, :3]
    front_translation = front[:, 3:]
    rotation = front_rotation @ back_rotation.T
    translation = front_translation - rotation @ back_translation
    return np.hstack((rotation, translation))


def anchored_poses(poses: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [relative_pose(poses[0], pose) for pose in poses]


def camera_centers(poses: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            (-pose[:, :3].T @ pose[:, 3:]).reshape(3)
            for pose in poses
        ],
        dtype=float,
    )


def rotation_degrees(rotation: np.ndarray) -> float:
    rvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=float))
    return float(np.linalg.norm(rvec) * 180.0 / np.pi)


def numeric_summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise AssertionError("numeric summary received empty/nonfinite data")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
    }


def source_location(
    path: Path, qualified_name: str, reported_file: str = None
) -> Dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = []

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = "{}.{}".format(prefix, node.name) if prefix else node.name
                if name == qualified_name:
                    matches.append(node)
                visit(node.body, name)

    visit(tree.body)
    if not matches:
        raise AssertionError(
            "source location not found: {} in {}".format(
                qualified_name, path
            )
        )
    node = matches[-1]
    return {
        "file": reported_file if reported_file is not None else relative(path),
        "function_or_class": qualified_name,
        "line_start": int(node.lineno),
        "line_end": int(node.end_lineno),
        "line_range_status": (
            "current; mechanical refresh required at report freeze"
        ),
    }


class ErrorAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.left_sum = 0.0
        self.left_square_sum = 0.0
        self.left_maximum = 0.0
        self.right_sum = 0.0
        self.right_square_sum = 0.0
        self.right_maximum = 0.0

    def add(self, left: float, right: float) -> None:
        if not math.isfinite(left) or not math.isfinite(right):
            raise AssertionError("nonfinite projection error")
        self.count += 1
        self.left_sum += left
        self.left_square_sum += left * left
        self.left_maximum = max(self.left_maximum, left)
        self.right_sum += right
        self.right_square_sum += right * right
        self.right_maximum = max(self.right_maximum, right)

    def result(self) -> Dict[str, object]:
        if self.count == 0:
            raise AssertionError("empty projection accumulator")
        return {
            "observation_count": self.count,
            "left_mean_px": self.left_sum / self.count,
            "left_rmse_px": math.sqrt(self.left_square_sum / self.count),
            "left_maximum_px": self.left_maximum,
            "right_mean_px": self.right_sum / self.count,
            "right_rmse_px": math.sqrt(self.right_square_sum / self.count),
            "right_maximum_px": self.right_maximum,
        }


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def materialize_tracking() -> Tuple[Dict[str, object], Dict[str, object]]:
    final = ROOT / FINAL_RELATIVE
    summary = read_json(final / "summary.json")
    measurements = summary["exercise4_measurements"]
    transition_source = read_csv_rows(final / "transitions.csv")
    frame_source = read_csv_rows(final / "frame_links.csv")
    if len(transition_source) != EXPECTED["transition_count"]:
        raise AssertionError("retained transition count differs")
    if len(frame_source) != EXPECTED["frame_count"]:
        raise AssertionError("retained frame count differs")

    transition_rows = []
    for row in transition_source:
        transition_rows.append(
            {
                "back_frame": int(row["back_frame"]),
                "front_frame": int(row["front_frame"]),
                "reciprocal_match_count": int(
                    row["reciprocal_temporal_match_count"]
                ),
                "four_view_consensus_count": int(row["consensus_count"]),
                "final_ransac_supporter_count": int(row["supporter_count"]),
                "final_ransac_supporter_percentage": float(
                    row["supporter_percentage"]
                ),
                "outgoing_track_connectivity": int(row["connectivity"]),
            }
        )
    transition_fields = (
        "back_frame",
        "front_frame",
        "reciprocal_match_count",
        "four_view_consensus_count",
        "final_ransac_supporter_count",
        "final_ransac_supporter_percentage",
        "outgoing_track_connectivity",
    )
    write_csv(
        OUTPUT / "tracking" / "data" / "matching_per_transition.csv",
        transition_fields,
        transition_rows,
    )

    frame_rows = [
        {
            "frame_id": int(row["frame_id"]),
            "tracked_link_count": int(row["tracked_link_count"]),
        }
        for row in frame_source
    ]
    write_csv(
        OUTPUT / "tracking" / "data" / "frame_links.csv",
        ("frame_id", "tracked_link_count"),
        frame_rows,
    )

    histogram = measurements["track_lengths"]["histogram"]
    histogram_rows = [
        {"track_length_frames": int(length), "track_count": int(count)}
        for length, count in sorted(
            histogram.items(), key=lambda item: int(item[0])
        )
    ]
    write_csv(
        OUTPUT / "tracking" / "data" / "track_length_histogram.csv",
        ("track_length_frames", "track_count"),
        histogram_rows,
    )

    representative_source = final / "figures" / "representative_track.png"
    representative_target = (
        OUTPUT / "tracking" / "figures" / "representative_track.png"
    )
    representative_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(representative_source, representative_target)
    representative = read_json(final / "representative_track.json")

    totals = {
        "reciprocal_match_count": sum(
            row["reciprocal_match_count"] for row in transition_rows
        ),
        "four_view_consensus_count": sum(
            row["four_view_consensus_count"] for row in transition_rows
        ),
        "final_ransac_supporter_count": sum(
            row["final_ransac_supporter_count"] for row in transition_rows
        ),
    }
    percentages = [
        row["final_ransac_supporter_percentage"]
        for row in transition_rows
    ]
    connectivities = [
        row["outgoing_track_connectivity"] for row in transition_rows
    ]
    frame_links = [row["tracked_link_count"] for row in frame_rows]
    core = {
        "schema": "ps-p01-tracking-metrics-v1",
        "scope": {
            "dataset": "KITTI odometry sequence 00",
            "frame_range_inclusive": [0, 3359],
            "frame_count": measurements["scope"]["frame_count"],
            "transition_count": measurements["scope"]["transition_count"],
        },
        "database": {
            "nontrivial_track_count": measurements["database"][
                "nontrivial_track_count"
            ],
            "tracked_link_count": measurements["database"][
                "tracked_link_count"
            ],
        },
        "track_lengths": measurements["track_lengths"],
        "frame_links": measurements["tracked_links_per_frame"],
        "matching_pipeline": {
            "definitions": {
                "reciprocal_match_count": (
                    "mutual left-image temporal descriptor matches"
                ),
                "four_view_consensus_count": (
                    "reciprocal temporal matches valid in both stereo pairs"
                ),
                "final_ransac_supporter_count": (
                    "four-view consensus rows accepted by final custom "
                    "PnP/RANSAC reprojection support"
                ),
            },
            "totals": totals,
            "final_supporter_percentage": numeric_summary(percentages),
            "transitions_below_30_percent": [
                [row["back_frame"], row["front_frame"]]
                for row in transition_rows
                if row["final_ransac_supporter_percentage"] < 30.0
            ],
        },
        "connectivity": {
            **numeric_summary(connectivities),
            "definition": (
                "tracks present in a frame with a Link in the next frame; "
                "not a match-count label"
            ),
        },
        "frame_link_definition": (
            "tracked stereo Links stored for each frame; not a match-count label"
        ),
        "representative_track": {
            "track_id": representative["track_id"],
            "track_length": representative["track_length"],
            "selection_policy": representative["selection_policy"],
            "source_figure_sha256": file_sha256(representative_source),
            "materialization": "byte-exact copy of retained validated figure",
        },
        "supporter_only_insertion": measurements["supporter_only_insertion"],
    }
    core["scientific_state_sha256"] = canonical_sha256(core)
    write_json(OUTPUT / "tracking" / "metrics.json", core)
    figure_specs = plot_tracking_figures(
        transition_rows, frame_rows, histogram_rows
    )
    return core, figure_specs


def plot_tracking_figures(
    transitions: Sequence[Dict[str, object]],
    frames: Sequence[Dict[str, object]],
    histogram: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    specs = {}
    front = np.asarray([row["front_frame"] for row in transitions])

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        front,
        [row["reciprocal_match_count"] for row in transitions],
        label="Reciprocal temporal matches",
        linewidth=0.8,
    )
    axis.plot(
        front,
        [row["four_view_consensus_count"] for row in transitions],
        label="Four-view consensus",
        linewidth=0.8,
    )
    axis.plot(
        front,
        [row["final_ransac_supporter_count"] for row in transitions],
        label="Final RANSAC supporters",
        linewidth=0.8,
    )
    axis.set_xlabel("Front frame index [frame]")
    axis.set_ylabel("Correspondence count [count]")
    axis.set_title("Temporal matching stages per transition")
    axis.grid(alpha=0.25)
    axis.legend()
    name = "matches_per_transition.png"
    save_figure(figure, OUTPUT / "tracking" / "figures" / name)
    specs[name] = {
        "x_label": "Front frame index [frame]",
        "y_label": "Correspondence count [count]",
        "series": [
            "reciprocal temporal matches",
            "four-view consensus",
            "final RANSAC supporters",
        ],
    }

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        front,
        [row["final_ransac_supporter_percentage"] for row in transitions],
        linewidth=0.8,
        color="tab:green",
    )
    axis.axhline(30.0, color="tab:red", linestyle="--", label="30% threshold")
    axis.set_xlabel("Front frame index [frame]")
    axis.set_ylabel("Final supporter percentage [%]")
    axis.set_title("Final four-view RANSAC supporter percentage")
    axis.grid(alpha=0.25)
    axis.legend()
    name = "supporter_percentage.png"
    save_figure(figure, OUTPUT / "tracking" / "figures" / name)
    specs[name] = {
        "x_label": "Front frame index [frame]",
        "y_label": "Final supporter percentage [%]",
    }

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        [row["back_frame"] for row in transitions],
        [row["outgoing_track_connectivity"] for row in transitions],
        linewidth=0.8,
        color="tab:purple",
    )
    axis.set_xlabel("Back frame index [frame]")
    axis.set_ylabel("Outgoing track connectivity [tracks]")
    axis.set_title("Consecutive-frame track connectivity")
    axis.grid(alpha=0.25)
    name = "connectivity.png"
    save_figure(figure, OUTPUT / "tracking" / "figures" / name)
    specs[name] = {
        "x_label": "Back frame index [frame]",
        "y_label": "Outgoing track connectivity [tracks]",
    }

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        [row["frame_id"] for row in frames],
        [row["tracked_link_count"] for row in frames],
        linewidth=0.8,
        color="tab:orange",
    )
    axis.set_xlabel("Frame index [frame]")
    axis.set_ylabel("Tracked stereo Links [Links]")
    axis.set_title("Tracked frame Links")
    axis.grid(alpha=0.25)
    name = "frame_links.png"
    save_figure(figure, OUTPUT / "tracking" / "figures" / name)
    specs[name] = {
        "x_label": "Frame index [frame]",
        "y_label": "Tracked stereo Links [Links]",
    }

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        [row["track_length_frames"] for row in histogram],
        [row["track_count"] for row in histogram],
        width=0.9,
        color="tab:blue",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Track length [frames]")
    axis.set_ylabel("Track count [tracks, log scale]")
    axis.set_title("Track-length distribution")
    axis.grid(axis="y", alpha=0.25)
    name = "track_length_histogram.png"
    save_figure(figure, OUTPUT / "tracking" / "figures" / name)
    specs[name] = {
        "x_label": "Track length [frames]",
        "y_label": "Track count [tracks, log scale]",
        "y_scale": "log",
    }

    specs["representative_track.png"] = {
        "axes": "image crops; pixel-coordinate observations are in provenance",
        "units": "pixels",
    }
    return specs


def write_code_locations(git_head: str) -> Dict[str, object]:
    entries = [
        {
            "requirement_id": "CODE-01",
            "stage": "Triangulation",
            "status": "ready",
            "locations": [
                source_location(
                    ROOT / "tools.py", "cv_triangulate_matched_points"
                )
            ],
            "note": "The final same-named module definition is the active production function.",
        },
        {
            "requirement_id": "CODE-02",
            "stage": "RANSAC",
            "status": "ready",
            "locations": [
                source_location(
                    ROOT / "tools.py",
                    "estimate_projection_matrices_with_ransac",
                )
            ],
            "note": "Custom four-view PnP/RANSAC implementation.",
        },
        {
            "requirement_id": "CODE-03",
            "stage": "PnP trajectory calculation",
            "status": "ready",
            "locations": [
                source_location(
                    ROOT / "exercise4_database_builder.py",
                    "build_exercise4_database",
                )
            ],
            "note": "Canonical sequential pose accumulation used for the retained trajectory.",
        },
        {
            "requirement_id": "CODE-04",
            "stage": "DataBase definition",
            "status": "ready",
            "locations": [
                source_location(ROOT / "tracking_database.py", "TrackingDB")
            ],
            "note": "Protected lecturer TrackingDB definition.",
        },
        {
            "requirement_id": "CODE-05",
            "stage": "Adding a frame to the data base",
            "status": "ready",
            "locations": [
                source_location(
                    ROOT / "tracking_database.py", "TrackingDB.add_frame"
                ),
                source_location(
                    ROOT / "tracking_database_adapter.py",
                    "Exercise3TrackingDBAdapter.add_frame",
                ),
            ],
            "note": "Lecturer insertion plus the canonical supporter-only adapter boundary.",
        },
        {
            "requirement_id": "CODE-06",
            "stage": "Single Bundle creation",
            "status": "deferred",
            "locations": [
                source_location(
                    ROOT / "BundleWindow.py", "BundleWindow.create_factor_graph"
                )
            ],
            "note": "Current historical location only; Exercise 5 review/results remain deferred.",
        },
        {
            "requirement_id": "CODE-07",
            "stage": "Relative transformation and covariance extraction",
            "status": "deferred",
            "locations": [
                source_location(
                    ROOT / "ex6.py", "get_relative_pose_and_cov_mat_last_kf"
                )
            ],
            "note": "Current historical location only; later-stage validation remains deferred.",
        },
        {
            "requirement_id": "CODE-08",
            "stage": "PoseGraph building",
            "status": "deferred",
            "locations": [
                source_location(
                    ROOT / "PoseGraph.py", "PoseGraph.create_factor_graph"
                )
            ],
            "note": "Current historical location only; Pose Graph results are not created here.",
        },
        {
            "requirement_id": "CODE-09",
            "stage": "Loop closure detection and factor creation",
            "status": "deferred",
            "locations": [
                source_location(ROOT / "ex7.py", "find_candidates"),
                source_location(ROOT / "ex7.py", "q3"),
            ],
            "note": "Candidate detection and factor insertion locations; Loop Closure is deferred.",
        },
    ]
    for entry in entries:
        for location in entry["locations"]:
            location["source_sha256"] = file_sha256(ROOT / location["file"])
            location["git_commit"] = git_head
    payload = {
        "schema": "ps-p01-code-locations-v1",
        "canonical_specification": PDF_RELATIVE,
        "git_commit": git_head,
        "entry_count": len(entries),
        "report_freeze_policy": (
            "Every current line range is provisional and must be refreshed "
            "mechanically immediately before the final report is frozen."
        ),
        "entries": entries,
    }
    write_json(OUTPUT / "code_locations.json", payload)
    return payload


def input_inventory() -> Dict[str, str]:
    paths = (
        PDF_RELATIVE,
        FINAL_RELATIVE + "/summary.json",
        FINAL_RELATIVE + "/validation.json",
        FINAL_RELATIVE + "/manifest.json",
        FINAL_RELATIVE + "/tracking_database_companion.json",
        FINAL_RELATIVE + "/transitions.csv",
        FINAL_RELATIVE + "/frame_links.csv",
        FINAL_RELATIVE + "/representative_track.json",
        FINAL_RELATIVE + "/figures/representative_track.png",
        FINAL_RELATIVE + "/tracking_database.pkl",
        I04_RELATIVE + "/validation.json",
        I04_RELATIVE + "/report.md",
        "dataset/poses/00.txt",
        "dataset/sequences/00/calib.txt",
        "tools.py",
        "ex3.py",
        "ex4.py",
        "exercise4_analysis.py",
        "exercise4_database_builder.py",
        "tracking_database_adapter.py",
        "tracking_database.py",
    )
    result = {}
    for relative_path in paths:
        path = ROOT / relative_path
        if not path.is_file():
            raise AssertionError("required input missing: {}".format(relative_path))
        result[relative_path] = file_sha256(path)
    if result[PDF_RELATIVE] != PDF_SHA256:
        raise AssertionError("canonical PDF hash differs")
    if result[FINAL_RELATIVE + "/tracking_database.pkl"] != DATABASE_SHA256:
        raise AssertionError("retained database hash differs")
    if result[FINAL_RELATIVE + "/tracking_database_companion.json"] != COMPANION_SHA256:
        raise AssertionError("retained companion hash differs")
    return result


def generator_locations() -> Dict[str, Dict[str, object]]:
    script = Path(__file__).resolve()
    reported = "project_submission/generate_current_stage.py"
    names = (
        "materialize_tracking",
        "plot_tracking_figures",
        "materialize_pnp",
        "relative_subsections",
        "compute_projection_aggregates",
        "plot_pnp_figures",
        "write_code_locations",
        "write_requirements_matrix",
        "validate_stage",
        "write_manifest",
    )
    return {
        name: source_location(script, name, reported_file=reported)
        for name in names
    }


def write_provenance(
    git_head: str,
    git_status_before: str,
    inputs: Dict[str, str],
    tracking_metrics: Dict[str, object],
    pnp_metrics: Dict[str, object],
    tracking_figures: Dict[str, object],
    pnp_figures: Dict[str, object],
) -> None:
    generators = generator_locations()
    common = {
        "schema": SCHEMA,
        "work_package": WORK_PACKAGE,
        "git_commit": git_head,
        "git_status_short_branch_before_generation": git_status_before,
        "generation_command": COMMAND,
        "canonical_specification": {
            "path": PDF_RELATIVE,
            "sha256": inputs[PDF_RELATIVE],
            "page_count": 7,
        },
        "line_range_policy": (
            "current ranges are recorded now and marked for mechanical "
            "refresh at final-report freeze"
        ),
        "generation_attempt_history": attempt_history(),
    }
    tracking = {
        **common,
        "artifact": "Tracking current-stage submission evidence",
        "scientific_state_sha256": tracking_metrics[
            "scientific_state_sha256"
        ],
        "source_inputs_sha256": {
            key: value
            for key, value in inputs.items()
            if key.startswith(FINAL_RELATIVE)
            or key.startswith(I04_RELATIVE)
            or key
            in (
                "ex4.py",
                "exercise4_analysis.py",
                "exercise4_database_builder.py",
                "tracking_database_adapter.py",
                "tracking_database.py",
            )
        },
        "generator_functions": {
            name: generators[name]
            for name in ("materialize_tracking", "plot_tracking_figures")
        },
        "figure_provenance": tracking_figures,
        "terminology": {
            "matches": (
                "used only for reciprocal temporal correspondences; figure "
                "also explicitly labels consensus and final supporters"
            ),
            "connectivity": "outgoing tracks shared with the next frame",
            "frame_links": "tracked stereo observations stored in a frame",
        },
    }
    pnp = {
        **common,
        "artifact": "PnP current-stage submission evidence",
        "scientific_state_sha256": pnp_metrics["scientific_state_sha256"],
        "source_inputs_sha256": {
            key: value
            for key, value in inputs.items()
            if key
            in (
                FINAL_RELATIVE + "/tracking_database.pkl",
                FINAL_RELATIVE + "/tracking_database_companion.json",
                FINAL_RELATIVE + "/summary.json",
                I04_RELATIVE + "/validation.json",
                "dataset/poses/00.txt",
                "dataset/sequences/00/calib.txt",
                "tracking_database.py",
            )
        },
        "generator_functions": {
            name: generators[name]
            for name in (
                "materialize_pnp",
                "relative_subsections",
                "compute_projection_aggregates",
                "plot_pnp_figures",
            )
        },
        "figure_provenance": pnp_figures,
        "projection_population": {
            "eligible_tracks": pnp_metrics["projection_error"][
                "eligible_track_count"
            ],
            "included_tracks": pnp_metrics["projection_error"][
                "included_track_count"
            ],
            "sampling": "none",
            "per_observation_table_retained": False,
        },
        "projection_limitations": [
            "frame offset 0 has near-zero error by construction",
            (
                "the population contains only tracks of length at least 10 "
                "and therefore has long-track selection bias"
            ),
            "maximum available track length is 65",
            (
                "the result combines tracking noise, stereo triangulation, "
                "and accumulated PnP pose error"
            ),
        ],
        "optional_future_enhancements_not_computed": [
            "median projection error",
            "p95 projection error",
            "ground-truth-pose control calculation",
        ],
    }
    write_json(OUTPUT / "tracking" / "provenance.json", tracking)
    write_json(OUTPUT / "pnp" / "provenance.json", pnp)


def write_requirements_matrix() -> List[Dict[str, str]]:
    """Write full seven-page traceability without claiming a final report."""

    rows = [
        ("P1-01", "1", "mandatory", "overall report", "partial", "Explain the system stages in the authors' own words and summarize the algorithms as understood.", "Current-stage README and provenance; final narrative not written.", "README.md; final report prose", "n/a", "Complete original prose at report freeze; editorial review."),
        ("P1-02", "1 §1", "mandatory", "overall report", "report-freeze", "Provide an introduction, background, and explanation of why visual navigation is important.", "No final-report prose in PS-P01.", "Final report introduction", "n/a", "Write and review at report freeze."),
        ("P1-03", "1 §2", "mandatory", "overall report", "partial", "Explain code organization and the functions implementing the main stages; code should be readable and documented.", "code_locations.json records current locations; prose/readability review remains.", "code_locations.json; final report code section", "source lines", "Refresh lines and write explanation at report freeze."),
        ("P1-C01", "1 §2", "code-location", "Tracking", "ready", "Supply function, file, and line location for triangulation.", "tools.py active triangulation definition.", "code_locations.json / CODE-01; write_code_locations", "source lines", "AST location exists; refresh mechanically at report freeze."),
        ("P1-C02", "1 §2", "code-location", "PnP", "ready", "Supply function, file, and line location for RANSAC.", "Canonical custom four-view PnP/RANSAC source.", "code_locations.json / CODE-02; write_code_locations", "source lines", "AST location exists; refresh mechanically at report freeze."),
        ("P1-C03", "1 §2", "code-location", "PnP", "ready", "Supply function, file, and line location for PnP trajectory calculation.", "Canonical sequential pose accumulation source.", "code_locations.json / CODE-03; write_code_locations", "source lines", "AST location exists; refresh mechanically at report freeze."),
        ("P1-C04", "1 §2", "code-location", "Tracking", "ready", "Supply function/class, file, and line location for DataBase definition.", "Protected lecturer TrackingDB class.", "code_locations.json / CODE-04; write_code_locations", "source lines", "AST location exists; refresh mechanically at report freeze."),
        ("P1-C05", "1 §2", "code-location", "Tracking", "ready", "Supply function, file, and line location for adding a frame to the database.", "Lecturer insertion and canonical adapter locations.", "code_locations.json / CODE-05; write_code_locations", "source lines", "AST locations exist; refresh mechanically at report freeze."),
        ("P1-C06", "1 §2", "code-location", "Bundle", "deferred", "Supply function, file, and line location for Single Bundle creation.", "Historical location recorded; Exercise 5 not reviewed here.", "code_locations.json / CODE-06", "source lines", "Exercise 5 validation and report-freeze line refresh."),
        ("P1-C07", "1 §2", "code-location", "Bundle", "deferred", "Supply location for relative transformation, including covariance, extraction from a bundle result.", "Historical ex6 location recorded.", "code_locations.json / CODE-07", "source lines", "Bundle validation and report-freeze line refresh."),
        ("P1-C08", "1 §2", "code-location", "Pose Graph", "deferred", "Supply function, file, and line location for PoseGraph building.", "Historical location recorded; no Pose Graph results generated.", "code_locations.json / CODE-08", "source lines", "Pose Graph package and report-freeze line refresh."),
        ("P1-C09", "1 §2", "code-location", "Loop Closure", "deferred", "Supply locations for loop-closure detection and factor creation.", "Historical detection/factor locations recorded.", "code_locations.json / CODE-09", "source lines", "Loop Closure package and report-freeze line refresh."),
        ("P1-04", "1 §3", "mandatory", "overall report", "partial", "Provide detailed quantitative and qualitative performance analysis for the system stages.", "Quantitative Tracking/PnP evidence ready; qualitative prose and later stages remain.", "tracking/, pnp/, final report analysis", "artifact-specific", "Write analysis; add later-stage evidence."),
        ("P1-05", "1 §3", "conditional", "overall report", "partial", "Where relevant, describe weak-point improvements, the implemented change, and its impact.", "Exercise 3/4 diagnostics preserve comparisons; report prose not written.", "Final report analysis", "artifact-specific", "Select relevant evidence and explain in prose."),
        ("P1-06", "1 §3", "suggested", "overall report", "ready", "Use logarithmic scale where appropriate, such as track-length or covariance graphs.", "Tracking histogram uses logarithmic count scale.", "tracking/figures/track_length_histogram.png; plot_tracking_figures", "tracks, frames", "Figure metadata and decode validation."),
        ("P1-07", "1 §3", "suggested", "overall report", "partial", "Overlay data where appropriate for easy comparison.", "Matching stages and PnP/GT trajectory are overlaid; later overlays deferred.", "Tracking and PnP overlay figures", "artifact-specific", "Figure-series validation; revisit later stages."),
        ("P1-08", "1 §3", "normative", "overall report", "partial", "Use the same scale for comparable graphs.", "Current-stage series sharing figures share axes; cross-stage comparison awaits final layout.", "Current figures; final report layout", "artifact-specific", "Report-freeze visual review."),
        ("P1-09", "1 §4", "mandatory", "overall report", "report-freeze", "Summarize, criticize weak points and unrealistic assumptions, propose improvements and future research.", "Limitations are preserved in provenance; final conclusions not written.", "Final report discussion/conclusions", "n/a", "Write and review at report freeze."),
        ("P1-10", "1 §4", "conditional", "overall report", "report-freeze", "Describe extra research/implementation, motivation, and results if applicable.", "No final selection of extra work has been made.", "Final report optional-extra discussion", "n/a", "Decide and document at report freeze."),
        ("P2-01", "2", "report-format", "overall report", "report-freeze", "Limit the final report to under 20 pages.", "PS-P01 is an evidence package, not the final report.", "Final report", "pages", "Count final rendered PDF pages."),
        ("P2-02", "2", "report-format", "overall report", "report-freeze", "Number all pages and all figures.", "Artifact filenames exist; final numbering/layout not assigned.", "Final report", "page/figure numbers", "Inspect final rendered PDF."),
        ("P2-03", "2", "report-format", "overall report", "partial", "Specify exact code location that created each figure next to the figure.", "Provenance contains generator function and current line range.", "tracking/provenance.json; pnp/provenance.json", "source lines", "Place next to figures and refresh lines at report freeze."),
        ("P2-04", "2", "normative", "overall report", "ready", "Label every figure axis with units.", "All plotted axes have explicit bracketed units; image-crop pixels documented separately.", "tracking/figures/; pnp/figures/", "m, deg, deg/m, px, %, frames, counts", "Figure-spec validation."),
        ("P2-05", "2", "normative", "overall report", "partial", "Use relevant scale, meaningful zoom/crop, colors, and effective overlays for clear figures.", "Current-stage figures use explicit scales/colors/overlays; final layout review remains.", "tracking/figures/; pnp/figures/", "artifact-specific", "Decode and report-freeze visual review."),
        ("P2-06", "2 Performance Analysis", "mandatory", "overall report", "partial", "Present evidence for each stage and the system as a whole.", "Tracking and PnP are ready; later stages deferred.", "tracking/; pnp/; future stage packages", "artifact-specific", "Coverage audit across all stages."),
        ("P2-07", "2 Performance Analysis", "mandatory", "overall report", "report-freeze", "Analyze every graph, highlighting what it shows and positive or negative behavior.", "Data/provenance ready; final explanatory prose not written.", "Final report figure discussion", "n/a", "Editorial traceability review."),
        ("P2-08", "2 Performance Analysis", "normative", "overall report", "partial", "Combine related data, crop meaningfully, and number each figure.", "Related current-stage series are combined; numbering is report-freeze work.", "Current figures; final report", "artifact-specific", "Visual and numbering review."),
        ("P2-T01", "2 Tracking", "mandatory", "Tracking", "ready", "Report total number of tracks.", "758478 validated nontrivial tracks.", "tracking/metrics.json; materialize_tracking", "tracks", "Exact equality to retained E4 evidence."),
        ("P2-T02", "2 Tracking", "mandatory", "Tracking", "ready", "Report number of frames.", "3360 frames, IDs 0 through 3359.", "tracking/metrics.json", "frames", "Exact equality."),
        ("P2-T03", "2 Tracking", "mandatory", "Tracking", "ready", "Report mean track length.", "3.3191127494798796.", "tracking/metrics.json", "frames/track", "Exact equality."),
        ("P2-T04", "2 Tracking", "mandatory", "Tracking", "ready", "Report mean number of frame Links.", "749.2482142857143.", "tracking/metrics.json", "Links/frame", "Exact equality."),
        ("P2-T05", "2 Tracking", "mandatory", "Tracking", "ready", "Graph number of matches per frame.", "Transition figure explicitly separates reciprocal matches, consensus, and final supporters.", "tracking/figures/matches_per_transition.png; plot_tracking_figures", "correspondences/transition", "3359 ordered rows; terminology audit."),
        ("P2-T06", "2 Tracking", "mandatory", "Tracking", "ready", "Graph percentage of inliers per frame.", "Final four-view RANSAC supporter percentage for every transition.", "tracking/figures/supporter_percentage.png; plot_tracking_figures", "%", "3359 finite values; zero below 30%."),
        ("P2-S01", "2 Suggested figures", "suggested", "overall report", "partial", "Include a substantial subset of suggested figures and any useful additions.", "Tracking and PnP current-stage subset is materialized; later suggestions deferred.", "tracking/figures/; pnp/figures/", "artifact-specific", "Final coverage judgment after later stages."),
        ("P2-S02", "2 Suggested figures", "suggested", "Tracking", "ready", "Plot connectivity: tracks in each frame with Links in the next frame.", "All 3359 outgoing connectivity values.", "tracking/figures/connectivity.png; plot_tracking_figures", "tracks", "Definition and row-count validation."),
        ("P3-01", "3", "suggested", "Tracking", "ready", "Plot a track-length histogram.", "Full 758478-track histogram, log count scale.", "tracking/figures/track_length_histogram.png", "tracks, frames", "Histogram sum and decode validation."),
        ("P3-02", "3", "suggested", "overall report", "partial", "Overlay bird's-eye PnP, Bundle, Pose Graph with loop closure, and ground truth.", "PnP and GT overlay ready; Bundle/Pose Graph/Loop Closure absent by design.", "pnp/figures/trajectory_xz.png; future overlays", "m", "Equal-aspect current plot; later-stage dependency."),
        ("P3-03", "3", "suggested", "Bundle", "deferred", "Plot bundle mean factor error per starting keyframe before and after optimization.", "No Bundle results in PS-P01.", "Future Bundle package", "factor error/keyframe", "Exercise 5 dependency."),
        ("P3-04", "3", "suggested", "Bundle", "deferred", "Plot bundle median projection error before and after optimization.", "No Bundle results in PS-P01.", "Future Bundle package", "px", "Exercise 5 dependency."),
        ("P3-05", "3", "suggested", "PnP", "ready", "Plot meaningful projection-error statistic versus distance from PnP triangulation frame.", "All 19263 length>=10 tracks aggregated by frame offset and GT path distance.", "pnp/figures/projection_vs_*.png; compute_projection_aggregates", "px, frames, m", "Population/count and bounded-memory validation."),
        ("P3-06", "3", "suggested", "Bundle", "deferred", "Plot Bundle projection error versus distance from the first frame.", "No Bundle results in PS-P01.", "Future Bundle package", "px, frames/m", "Exercise 5 dependency."),
        ("P3-07", "3 note", "conditional", "overall report", "ready", "If projection collection uses a representative subset, explain its selection.", "No subset used: complete population of all 19263 eligible PnP tracks.", "pnp/metrics.json; pnp/provenance.json", "tracks", "Eligible equals included; sampling none."),
        ("P4-01", "4", "suggested", "PnP", "ready", "Plot absolute PnP X/Y/Z signed errors and total location-error norm in meters.", "3360 frame-anchored rows and figure.", "pnp/figures/absolute_position_errors.png", "m", "Finite ordered data; final error agreement."),
        ("P4-02", "4", "suggested", "PnP", "ready", "Plot absolute PnP angle error in degrees.", "Rodrigues rotation error for 3360 frames.", "pnp/figures/absolute_rotation_error.png", "deg", "Finite ordered data."),
        ("P4-03", "4", "suggested", "Pose Graph", "deferred", "Plot absolute Pose Graph error without loop closure: X/Y/Z, location norm, and angle.", "No Pose Graph results in PS-P01.", "Future Pose Graph package", "m, deg", "Exercise 6 dependency."),
        ("P4-04", "4", "suggested", "Loop Closure", "deferred", "Plot absolute Pose Graph error with loop closure: X/Y/Z, location norm, and angle.", "No Loop Closure results in PS-P01.", "Future Loop Closure package", "m, deg", "Exercise 7 dependency."),
        ("P4-05", "4 Relative Error", "suggested", "Bundle", "deferred", "Compare consecutive-keyframe Bundle and accumulated PnP relative pose errors.", "Canonical keyframes/Bundle results not established.", "Future Bundle/Pose Graph package", "m, deg", "Exercise 5 keyframe dependency."),
        ("P4-06", "4 Relative Error", "mandatory-discussion", "overall report", "report-freeze", "Discuss whether relative errors suggest inaccuracies in ground-truth poses.", "No final discussion claim made.", "Final report relative-error discussion", "n/a", "Interpret validated later-stage comparison."),
        ("P5-01", "5", "suggested", "PnP", "ready", "Evaluate relative PnP subsections of lengths 100, 400, and 800.", "3261, 2961, and 2561 sliding rows.", "pnp/data/relative_subsections_*.csv; relative_subsections", "frames", "Exact row counts and endpoint policy."),
        ("P5-02", "5", "suggested", "PnP", "ready", "For each PnP length, report translation error per meter/error percent, angle error per meter, and averages.", "Both normalized measures and mean values are retained for all lengths.", "pnp/metrics.json; pnp/figures/relative_subsections_*.png", "m/m, %, deg/m", "Finite values and summary recomputation."),
        ("P5-03", "5", "suggested", "Bundle", "deferred", "Evaluate relative Bundle subsections at 100/400/800, choosing closest keyframe if needed, with normalized errors and averages.", "No Bundle/keyframe results in PS-P01.", "Future Bundle package", "m/m, %, deg/m", "Exercise 5 dependency."),
        ("P5-04", "5", "suggested", "Loop Closure", "deferred", "Plot matches per successful loop-closure frame.", "No Loop Closure results in PS-P01.", "Future Loop Closure package", "matches", "Exercise 7 dependency."),
        ("P5-05", "5", "suggested", "Loop Closure", "deferred", "Plot inlier percentage per successful loop-closure frame.", "No Loop Closure results in PS-P01.", "Future Loop Closure package", "%", "Exercise 7 dependency."),
        ("P5-06", "5", "suggested", "Pose Graph", "deferred", "Plot location and angle uncertainty versus keyframe without loop closure.", "No Pose Graph uncertainty results in PS-P01.", "Future Pose Graph package", "m/defined scale, deg/defined scale", "Exercise 6 dependency and uncertainty definition."),
        ("P6-01", "6", "suggested", "Loop Closure", "deferred", "Plot location and angle uncertainty versus keyframe with loop closure.", "No Loop Closure uncertainty results in PS-P01.", "Future Loop Closure package", "m/defined scale, deg/defined scale", "Exercise 7 dependency."),
        ("P6-02", "6", "mandatory-discussion", "Pose Graph", "deferred", "Explain how uncertainty size was measured and how location/angle components were isolated.", "No validated uncertainty measure yet.", "Future Pose Graph report section", "defined later", "Establish and validate uncertainty convention."),
        ("P6-A01", "6 Appendix", "normative", "PnP", "ready", "Measure rotation magnitude with Rodrigues-vector norm converted to degrees.", "rotation_degrees implements the stated formula.", "generate_current_stage.py / rotation_degrees", "deg", "Formula and finite-value validation."),
        ("P6-A02", "6 Appendix", "normative", "PnP", "ready", "Compare estimated and GT relative poses; use consecutive frames or keyframes as appropriate.", "Sliding PnP subsections compare relative movements.", "relative_subsections", "m, deg", "Formula recorded in metrics."),
        ("P6-A03", "6 Appendix", "normative", "PnP", "ready", "For long subsections, report error per meter traveled as in KITTI evaluation.", "Translation m/m and angle deg/m are computed.", "pnp/data/relative_subsections_*.csv", "m/m, deg/m", "Recompute normalized columns."),
        ("P6-A04", "6 Appendix", "normative", "PnP", "ready", "For a length-100 example such as 51→150, compare estimated and true movement and divide by traveled path length.", "Endpoint policy uses start→start+99 for every row, including 51→150.", "relative_subsections", "m/m, deg/m", "Endpoint and denominator assertions."),
        ("P7-A01", "7 Appendix", "normative", "PnP", "ready", "Location numerator is the norm of the translation part of the relative-pose error.", "Recorded location_error_m follows this definition.", "relative_subsections", "m", "Formula validation."),
        ("P7-A02", "7 Appendix", "normative", "PnP", "ready", "Angle numerator is the rotation-part magnitude of the relative-pose error.", "Rodrigues magnitude recorded as angle_error_deg.", "relative_subsections", "deg", "Formula validation."),
        ("P7-A03", "7 Appendix", "normative", "PnP", "ready", "Denominator is summed inter-frame GT path distance, not endpoint displacement, and is shared by location and angle.", "Cumulative sum of consecutive GT camera-center steps supplies both denominators.", "relative_subsections", "m", "Positive denominator and recomputation checks."),
        ("P7-A04", "7 Appendix", "normative", "PnP", "ready", "Use all sliding windows: for length 100, 0→99, 1→100, …, (n-99)→n.", "All windows emitted; exact requested counts validated.", "pnp/data/relative_subsections_*.csv", "frames", "First/last endpoint and count checks."),
        ("P7-A05", "7 Appendix", "normative", "PnP", "ready", "Respect appendix pose, relative-pose, displacement, error-pose, translation-norm, rotation-norm, and total-distance definitions.", "Definitions are explicit in pnp/metrics.json and provenance.", "pnp/metrics.json", "m, deg", "Definition-field presence and numerical validation."),
    ]
    fields = (
        "id",
        "pdf_page_section",
        "category",
        "stage",
        "status",
        "requirement",
        "current_evidence",
        "artifact_and_generator",
        "units",
        "remaining_dependency_and_validation",
    )
    records = [dict(zip(fields, row)) for row in rows]
    lines = [
        "# Project Submission requirements traceability matrix",
        "",
        "Canonical source: `docs/submission/Project Submission - Ex8.pdf` "
        "(all seven pages inspected directly).",
        "",
        "Status meanings: **ready** means current-stage evidence is generated "
        "and validated; **partial** means evidence exists but final presentation "
        "or later-stage coverage remains; **report-freeze** means the item must "
        "be completed only in the final report/layout pass; **deferred** means "
        "it depends on Bundle, Pose Graph, or Loop Closure work not authorized "
        "in PS-P01.",
        "",
        "| ID | PDF page/section | Category | Stage | Status | Requirement | Current evidence | Artifact / generator | Units | Remaining dependency / validation |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in records:
        values = [str(record[field]).replace("|", "\\|") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Deliberate PS-P01 boundary",
            "",
            "No Bundle, Pose Graph, or Loop Closure result artifact is created "
            "by this package. Their requirements remain explicit traceability "
            "rows rather than being mislabeled as current defects. Final report "
            "prose, layout, page/figure numbering, page limit, and line-number "
            "placement remain report-freeze work.",
            "",
        ]
    )
    (OUTPUT / "requirements_matrix.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return records


def write_readme(
    tracking: Dict[str, object], pnp: Dict[str, object]
) -> None:
    relative = pnp["relative_subsections"]["lengths"]
    text = """# Current-stage Project Submission evidence

This PS-P01 package materializes the reviewed Tracking and motion-PnP
evidence currently supported by the canonical KITTI 00 run. It is an
evidence package for later report writing, not the final report itself.

Canonical specification: `docs/submission/Project Submission - Ex8.pdf`
(all seven pages). Canonical retained evidence remains in
`diagnostics/ex4_final/` and `diagnostics/ex4_i04/`; the 2.48 GB TrackingDB
is referenced by hash and is not copied here.

## Scope and terminology

- Tracking covers 3360 frames, 3359 transitions, 758478 nontrivial tracks,
  and 2517474 tracked stereo Links.
- The temporal-matching figure separately labels reciprocal descriptor
  matches, four-view consensus rows, and final RANSAC supporters.
- Connectivity means tracks continuing to the next frame. Frame Links mean
  stored tracked stereo observations. Neither is relabeled as “matches.”
- Motion PnP uses the retained world-to-camera poses, anchored at frame 0,
  and KITTI ground truth.
- Projection error uses all 19263 tracks of length at least 10. Each track is
  triangulated from its first stereo Link using the retained PnP pose and is
  reprojected with retained PnP poses. No per-observation table is retained.
- Retained Track 305 ground-truth reprojection evidence is not reused or
  relabeled as PnP projection error.
- Frame offset 0 has near-zero projection error by construction.
- The projection population contains only tracks of length at least 10 and
  therefore has long-track selection bias; the maximum available track
  length is 65.
- PnP projection error combines tracking noise, stereo triangulation, and
  accumulated PnP pose error.

## Selected validated values

- Mean track length: {mean_track_length} frames/track.
- Mean frame Links: {mean_frame_links} Links/frame.
- Final supporter transitions below 30%: {below_30}.
- Final PnP position error: {final_error} m over {gt_distance} m of
  ground-truth travel ({drift}% drift).
- Sliding subsection row counts: 100 → {rows100}, 400 → {rows400},
  800 → {rows800}.
- Projection population: {projection_tracks} eligible and included tracks,
  {projection_observations} projected observations.

## Package layout

```text
project_submission/
  README.md
  requirements_matrix.md
  code_locations.json
  generate_current_stage.py
  manifest.json
  validation.json
  validation_report.md
  tracking/
    metrics.json
    provenance.json
    data/
    figures/
  pnp/
    metrics.json
    provenance.json
    data/
    figures/
```

`requirements_matrix.md` traces every page of the PDF. Bundle Adjustment,
Pose Graph, and Loop Closure rows are deferred; no result artifact for those
stages is created. Final prose, layout, page/figure numbering, placement of
code references, the under-20-page check, and final line-number refresh are
explicit report-freeze work.

## Regeneration model

Run the generator only in a new staging directory, validate there, and then
atomically publish the entire directory. The generation pass performs one
fresh-process deserialization of the retained TrackingDB solely for bounded
projection aggregation. It does not run feature extraction, matching,
RANSAC, motion PnP, database building, Bundle Adjustment, Pose Graph, or Loop
Closure.

Canonical clean regeneration (the `mktemp` directory is explicitly fresh):

```bash
STAGING_DIR="$(mktemp -d /tmp/ps_p01_staging.XXXXXX)"
.venv/bin/python project_submission/generate_current_stage.py \
  --repo-root "$PWD" \
  --output-dir "$STAGING_DIR" \
  --allow-existing-empty-output
```

Verify the already-published artifacts without loading the TrackingDB:

```bash
.venv/bin/python project_submission/generate_current_stage.py \
  --verify-artifacts-only
```

Every source/input and artifact is hash-bound in `manifest.json`; scientific
cores are hash-bound in the Tracking and PnP metrics and in validation.
The published validation/provenance records preserve this package's two
failed staging attempts and explicitly authorized recovery deserialization.
Future clean generation records clean-run provenance by default. Recovery
history is recorded only when explicitly authorized and supplied as JSON.
""".format(
        mean_track_length=tracking["track_lengths"]["mean"],
        mean_frame_links=tracking["frame_links"]["mean"],
        below_30=len(
            tracking["matching_pipeline"]["transitions_below_30_percent"]
        ),
        final_error=pnp["absolute_errors"]["final_position_error_m"],
        gt_distance=pnp["absolute_errors"]["total_ground_truth_distance_m"],
        drift=pnp["absolute_errors"]["final_drift_percent"],
        rows100=relative["100"]["row_count"],
        rows400=relative["400"]["row_count"],
        rows800=relative["800"]["row_count"],
        projection_tracks=pnp["projection_error"]["included_track_count"],
        projection_observations=pnp["projection_error"]["observation_count"],
    )
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def require_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(
            "{} differs: observed={!r}, expected={!r}".format(
                label, observed, expected
            )
        )


def validate_json_finite(path: Path) -> None:
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: {}".format(value))

    with path.open("r", encoding="utf-8") as handle:
        json.load(handle, parse_constant=reject_constant)


def validate_csv_numeric_ordered(
    path: Path, primary_fields: Sequence[str], expected_rows: int = None
) -> int:
    rows = read_csv_rows(path)
    if expected_rows is not None and len(rows) != expected_rows:
        raise AssertionError(
            "{} row count differs".format(path.relative_to(OUTPUT))
        )
    previous = None
    for row in rows:
        for value in row.values():
            number = float(value)
            if not math.isfinite(number):
                raise AssertionError("CSV contains a nonfinite value")
        key = tuple(float(row[field]) for field in primary_fields)
        if previous is not None and not previous < key:
            raise AssertionError(
                "CSV is not strictly ordered: {}".format(
                    path.relative_to(OUTPUT)
                )
            )
        previous = key
    return len(rows)


def validate_figure_specs(specs: Dict[str, object], directory: Path) -> None:
    for name, spec in specs.items():
        path = directory / name
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0:
            raise AssertionError("figure is not decodable: {}".format(path))
        labels = []
        if "x_label" in spec:
            labels.append(spec["x_label"])
        if "y_label" in spec:
            labels.append(spec["y_label"])
        labels.extend(spec.get("y_labels", []))
        if labels and any("[" not in label or "]" not in label for label in labels):
            raise AssertionError("figure axis label lacks units: {}".format(name))
        if not labels and spec.get("units") != "pixels":
            raise AssertionError("figure has no axis/unit metadata: {}".format(name))


def validate_stage(
    tracking: Dict[str, object],
    pnp: Dict[str, object],
    requirements: Sequence[Dict[str, str]],
    code_locations: Dict[str, object],
    tracking_figures: Dict[str, object],
    pnp_figures: Dict[str, object],
    inputs: Dict[str, str],
    git_status_before: str,
) -> Dict[str, object]:
    prior_attempt_deserializations = sum(
        1
        for attempt in attempt_history()["attempts_before_current_process"]
        if attempt.get("database_deserialized") is True
    )
    gates = {}
    if tracking["scope"]["frame_count"] != EXPECTED["frame_count"]:
        raise AssertionError("Tracking frame count differs")
    if tracking["scope"]["transition_count"] != EXPECTED["transition_count"]:
        raise AssertionError("Tracking transition count differs")
    if tracking["database"]["nontrivial_track_count"] != EXPECTED["track_count"]:
        raise AssertionError("Tracking track count differs")
    if tracking["database"]["tracked_link_count"] != EXPECTED["tracked_link_count"]:
        raise AssertionError("Tracking Link count differs")
    require_close(
        tracking["track_lengths"]["mean"],
        EXPECTED["mean_track_length"],
        "mean track length",
    )
    require_close(
        tracking["frame_links"]["mean"],
        EXPECTED["mean_frame_links"],
        "mean frame Links",
    )
    pipeline = tracking["matching_pipeline"]
    if len(pipeline["transitions_below_30_percent"]) != 0:
        raise AssertionError("a transition is below 30 percent")
    totals = pipeline["totals"]
    if totals["reciprocal_match_count"] != EXPECTED["reciprocal_match_total"]:
        raise AssertionError("reciprocal-match total differs")
    if totals["four_view_consensus_count"] != EXPECTED["consensus_match_total"]:
        raise AssertionError("consensus total differs")
    if totals["final_ransac_supporter_count"] != EXPECTED["final_supporter_total"]:
        raise AssertionError("supporter total differs")
    gates["tracking_exact_agreement"] = "passed"

    absolute = pnp["absolute_errors"]
    require_close(
        absolute["final_position_error_m"],
        EXPECTED["final_position_error_m"],
        "final position error",
    )
    require_close(
        absolute["total_ground_truth_distance_m"],
        EXPECTED["total_ground_truth_distance_m"],
        "total ground-truth distance",
    )
    require_close(
        absolute["final_drift_percent"],
        EXPECTED["final_drift_percent"],
        "final drift",
    )
    for length, expected_count in EXPECTED["relative_row_counts"].items():
        observed = pnp["relative_subsections"]["lengths"][str(length)][
            "row_count"
        ]
        if observed != expected_count:
            raise AssertionError("relative row count differs")
    projection = pnp["projection_error"]
    if projection["eligible_track_count"] != EXPECTED["eligible_projection_tracks"]:
        raise AssertionError("projection eligibility differs")
    if projection["included_track_count"] != EXPECTED["eligible_projection_tracks"]:
        raise AssertionError("projection population is incomplete")
    if projection["database_deserialization_count"] != 1:
        raise AssertionError("database deserialization count differs")
    gates["pnp_exact_and_projection_population"] = "passed"

    csv_specs = (
        ("tracking/data/matching_per_transition.csv", ("front_frame",), 3359),
        ("tracking/data/frame_links.csv", ("frame_id",), 3360),
        ("tracking/data/track_length_histogram.csv", ("track_length_frames",), None),
        ("pnp/data/trajectory_and_absolute_errors.csv", ("frame_id",), 3360),
        ("pnp/data/relative_subsections_100.csv", ("start_frame",), 3261),
        ("pnp/data/relative_subsections_400.csv", ("start_frame",), 2961),
        ("pnp/data/relative_subsections_800.csv", ("start_frame",), 2561),
        ("pnp/data/projection_by_frame_offset.csv", ("frame_offset_frames",), None),
        ("pnp/data/projection_by_path_distance.csv", ("distance_bin_index",), None),
    )
    csv_counts = {}
    for relative_path, keys, count in csv_specs:
        csv_counts[relative_path] = validate_csv_numeric_ordered(
            OUTPUT / relative_path, keys, count
        )
    for path in sorted(OUTPUT.rglob("*.json")):
        validate_json_finite(path)
    gates["finite_ordered_json_csv"] = "passed"

    validate_figure_specs(
        tracking_figures, OUTPUT / "tracking" / "figures"
    )
    validate_figure_specs(pnp_figures, OUTPUT / "pnp" / "figures")
    if not pnp_figures["trajectory_xz.png"].get("equal_aspect"):
        raise AssertionError("trajectory figure is not equal-aspect")
    gates["figures_decodable_labeled_equal_aspect"] = "passed"

    pages = {
        int(record["pdf_page_section"].split()[0])
        for record in requirements
    }
    if pages != set(range(1, 8)):
        raise AssertionError("requirements matrix does not cover all pages")
    statuses = {record["status"] for record in requirements}
    if statuses != {"ready", "partial", "report-freeze", "deferred"}:
        raise AssertionError("requirements status taxonomy is incomplete")
    if len(code_locations["entries"]) != 9:
        raise AssertionError("nine code-location entries are not represented")
    if {entry["requirement_id"] for entry in code_locations["entries"]} != {
        "CODE-{:02d}".format(index) for index in range(1, 10)
    }:
        raise AssertionError("code-location requirement IDs differ")
    gates["complete_pdf_and_code_location_traceability"] = "passed"

    forbidden_suffixes = {".pkl", ".pickle"}
    if any(path.suffix in forbidden_suffixes for path in OUTPUT.rglob("*")):
        raise AssertionError("a TrackingDB/pickle was copied into the package")
    forbidden_directories = {"bundle", "bundle_adjustment", "pose_graph", "loop_closure"}
    if any(
        path.is_dir() and path.name.lower() in forbidden_directories
        for path in OUTPUT.rglob("*")
    ):
        raise AssertionError("a later-stage result directory was created")
    if git_output("status", "--short", "--branch") != git_status_before:
        raise AssertionError("pre-existing Git worktree changed during staging")
    current_inputs = input_inventory()
    if current_inputs != inputs:
        raise AssertionError("a source/input hash changed during generation")
    protected = read_json(ROOT / I04_RELATIVE / "validation.json")[
        "diagnostics_ex4_final"
    ]["sha256"]
    current_protected = {
        relative(path): file_sha256(path)
        for path in sorted((ROOT / FINAL_RELATIVE).rglob("*"))
        if path.is_file() and relative(path) in protected
    }
    if current_protected != protected:
        raise AssertionError("diagnostics/ex4_final protected inventory changed")
    gates["scope_and_protected_inputs"] = "passed"

    return {
        "schema": "ps-p01-validation-v1",
        "work_package": WORK_PACKAGE,
        "status": "passed",
        "gates": gates,
        "tracking": {
            "frame_count": tracking["scope"]["frame_count"],
            "transition_count": tracking["scope"]["transition_count"],
            "track_count": tracking["database"]["nontrivial_track_count"],
            "tracked_link_count": tracking["database"]["tracked_link_count"],
            "below_30_percent_count": len(
                pipeline["transitions_below_30_percent"]
            ),
        },
        "pnp": {
            "final_position_error_m": absolute["final_position_error_m"],
            "total_ground_truth_distance_m": absolute[
                "total_ground_truth_distance_m"
            ],
            "final_drift_percent": absolute["final_drift_percent"],
            "relative_row_counts": {
                key: value["row_count"]
                for key, value in pnp["relative_subsections"]["lengths"].items()
            },
            "projection_eligible_tracks": projection["eligible_track_count"],
            "projection_included_tracks": projection["included_track_count"],
            "projection_observations": projection["observation_count"],
        },
        "traceability": {
            "pdf_pages": sorted(pages),
            "requirements_row_count": len(requirements),
            "status_counts": {
                status: sum(
                    record["status"] == status for record in requirements
                )
                for status in sorted(statuses)
            },
            "code_location_entry_count": len(code_locations["entries"]),
            "line_ranges_require_report_freeze_refresh": True,
        },
        "artifacts": {
            "csv_row_counts": csv_counts,
            "tracking_figure_count": len(tracking_figures),
            "pnp_figure_count": len(pnp_figures),
            "all_figures_decodable": True,
            "all_plotted_axes_labeled_with_units": True,
            "trajectory_equal_aspect": True,
            "per_observation_projection_table_retained": False,
            "tracking_database_copied": False,
        },
        "process_safety": {
            "fresh_process_database_deserializations": 1,
            "resumed_after_pre_deserialization_import_path_correction": (
                RESUMED_BEFORE_DATABASE
            ),
            "two_databases_coexisted": False,
            "feature_extraction_executed": False,
            "matching_executed": False,
            "ransac_executed": False,
            "motion_pnp_executed": False,
            "exercise4_build_executed": False,
            "bundle_adjustment_executed": False,
            "pose_graph_executed": False,
            "loop_closure_executed": False,
            "prior_failed_attempt_database_deserializations": (
                prior_attempt_deserializations
            ),
            "total_ps_p01_database_deserializations_including_failed_attempts": (
                1 + prior_attempt_deserializations
            ),
            "additional_recovery_deserialization_explicitly_authorized": (
                AUTHORIZED_RECOVERY
            ),
        },
        "scope": {
            "pre_existing_git_status_unchanged_during_staging": True,
            "diagnostics_ex4_final_inventory_unchanged": True,
            "agents_md_modified": False,
            "commit_created": False,
            "push_performed": False,
        },
        "generation_attempt_history": attempt_history(),
    }


def write_validation_report(
    validation: Dict[str, object],
    tracking: Dict[str, object],
    pnp: Dict[str, object],
) -> None:
    lengths = pnp["relative_subsections"]["lengths"]
    text = """# PS-P01 validation report

Status: **PASS**

## Scope and safety

- Generated in a fresh staging directory outside `project_submission/`.
- Recovery generation used exactly one explicitly authorized additional
  fresh-process deserialization of the retained TrackingDB; the failed-attempt
  history is preserved in `validation.json` and both provenance files.
- No feature extraction, matching, RANSAC, motion PnP, Exercise 4 build,
  Bundle Adjustment, Pose Graph, or Loop Closure execution.
- No TrackingDB/pickle copied; no per-observation projection table retained.
- The pre-existing Git status and protected `diagnostics/ex4_final/` inventory
  remained unchanged during staging.

## Tracking gates

- Frames/transitions: {frames}/{transitions}.
- Nontrivial tracks/tracked Links: {tracks}/{links}.
- Mean track length: {mean_track_length} frames/track.
- Mean frame Links: {mean_frame_links} Links/frame.
- Reciprocal/consensus/final-supporter totals: {reciprocal}/{consensus}/{supporters}.
- Final-supporter transitions below 30%: {below_30}.
- Matches, consensus, supporters, connectivity, and frame Links remain
  explicitly distinct definitions.

## PnP gates

- Final absolute position error: {final_error} m.
- Ground-truth path distance: {gt_distance} m.
- Final drift: {drift}%.
- Relative subsection rows: 100={rows100}, 400={rows400}, 800={rows800}.
- Relative subsection mean translation/rotation errors:
  - 100 frames: {loc100}% and {ang100} deg/m.
  - 400 frames: {loc400}% and {ang400} deg/m.
  - 800 frames: {loc800}% and {ang800} deg/m.
- PnP projection aggregation includes all {projection_tracks} eligible tracks
  and {projection_observations} observations; sampling was not used.
- Retained Track 305 ground-truth reprojection was not relabeled or reused as
  PnP projection error.
- Frame offset 0 has near-zero projection error by construction.
- The population contains only tracks of length at least 10 and therefore has
  long-track selection bias; the maximum available track length is 65.
- This projection-error result combines tracking noise, stereo triangulation,
  and accumulated PnP pose error.
- Median, p95, and a ground-truth-pose control calculation remain optional
  future enhancements and were not computed in this package.

## Artifact and traceability gates

- Finite, ordered JSON/CSV: PASS.
- Decodable figures with recorded axis labels/units: PASS.
- Frame-0-anchored X-Z trajectory equal aspect: PASS.
- PDF pages covered: 1–7; matrix rows: {matrix_rows}.
- All nine requested code-location entries: represented.
- Current line ranges are marked for mechanical refresh at report freeze.
- Bundle, Pose Graph, Loop Closure, final prose/layout/numbering, and the
  under-20-page check remain deferred or report-freeze requirements.

Scientific state:
`{scientific_state}`
""".format(
        frames=tracking["scope"]["frame_count"],
        transitions=tracking["scope"]["transition_count"],
        tracks=tracking["database"]["nontrivial_track_count"],
        links=tracking["database"]["tracked_link_count"],
        mean_track_length=tracking["track_lengths"]["mean"],
        mean_frame_links=tracking["frame_links"]["mean"],
        reciprocal=tracking["matching_pipeline"]["totals"][
            "reciprocal_match_count"
        ],
        consensus=tracking["matching_pipeline"]["totals"][
            "four_view_consensus_count"
        ],
        supporters=tracking["matching_pipeline"]["totals"][
            "final_ransac_supporter_count"
        ],
        below_30=len(
            tracking["matching_pipeline"]["transitions_below_30_percent"]
        ),
        final_error=pnp["absolute_errors"]["final_position_error_m"],
        gt_distance=pnp["absolute_errors"]["total_ground_truth_distance_m"],
        drift=pnp["absolute_errors"]["final_drift_percent"],
        rows100=lengths["100"]["row_count"],
        rows400=lengths["400"]["row_count"],
        rows800=lengths["800"]["row_count"],
        loc100=lengths["100"]["mean_location_error_percent"],
        ang100=lengths["100"]["mean_angle_error_per_meter_deg_per_m"],
        loc400=lengths["400"]["mean_location_error_percent"],
        ang400=lengths["400"]["mean_angle_error_per_meter_deg_per_m"],
        loc800=lengths["800"]["mean_location_error_percent"],
        ang800=lengths["800"]["mean_angle_error_per_meter_deg_per_m"],
        projection_tracks=pnp["projection_error"]["included_track_count"],
        projection_observations=pnp["projection_error"]["observation_count"],
        matrix_rows=validation["traceability"]["requirements_row_count"],
        scientific_state=validation["scientific_state_sha256"],
    )
    (OUTPUT / "validation_report.md").write_text(text, encoding="utf-8")


def write_manifest(
    git_head: str,
    inputs: Dict[str, str],
    scientific_state_sha256: str,
) -> Dict[str, object]:
    manifest_path = OUTPUT / "manifest.json"
    artifacts = {}
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        artifacts[str(path.relative_to(OUTPUT))] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    payload = {
        "schema": "ps-p01-manifest-v1",
        "work_package": WORK_PACKAGE,
        "git_commit": git_head,
        "generation_command": COMMAND,
        "canonical_specification": {
            "path": PDF_RELATIVE,
            "sha256": inputs[PDF_RELATIVE],
            "page_count": 7,
        },
        "source_inputs_sha256": inputs,
        "scientific_state_sha256": scientific_state_sha256,
        "generation_attempt_history": attempt_history(),
        "artifact_count_excluding_manifest": len(artifacts),
        "artifacts_excluding_manifest": artifacts,
        "manifest_self_hash_policy": (
            "manifest.json is excluded to avoid a recursive self-hash"
        ),
    }
    write_json(manifest_path, payload)
    return payload


def verify_manifest(manifest: Dict[str, object]) -> None:
    expected = manifest["artifacts_excluding_manifest"]
    actual_paths = {
        str(path.relative_to(OUTPUT))
        for path in OUTPUT.rglob("*")
        if path.is_file() and path != OUTPUT / "manifest.json"
    }
    if actual_paths != set(expected):
        raise AssertionError("manifest artifact path inventory differs")
    for relative_path, record in expected.items():
        path = OUTPUT / relative_path
        if file_sha256(path) != record["sha256"]:
            raise AssertionError("manifest artifact hash differs")
        if path.stat().st_size != record["size_bytes"]:
            raise AssertionError("manifest artifact size differs")


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def _require_finite_json(value: object, context: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _require_finite_json(nested, "{}.{}".format(context, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json(nested, "{}[{}]".format(context, index))
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("nonfinite JSON value: {}".format(context))


def verify_artifacts_only(package: Path) -> Dict[str, object]:
    """Verify a published package without reading the retained TrackingDB."""

    global OUTPUT
    OUTPUT = package.resolve()
    if not OUTPUT.is_dir():
        raise AssertionError("artifact package directory does not exist")
    manifest = read_json(OUTPUT / "manifest.json")
    if not isinstance(manifest, dict):
        raise AssertionError("manifest root is not an object")
    verify_manifest(manifest)

    json_paths = sorted(OUTPUT.rglob("*.json"))
    for path in json_paths:
        _require_finite_json(read_json(path), str(path.relative_to(OUTPUT)))

    csv_paths = sorted(OUTPUT.rglob("*.csv"))
    csv_row_counts = {}
    for path in csv_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(
                reader.fieldnames
            ):
                raise AssertionError("invalid CSV header: {}".format(path))
            count = 0
            for row in reader:
                if set(row) != set(reader.fieldnames):
                    raise AssertionError("invalid CSV row shape: {}".format(path))
                for field, raw in row.items():
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        raise AssertionError(
                            "nonnumeric CSV value: {}:{}".format(path, field)
                        )
                    if not math.isfinite(value):
                        raise AssertionError(
                            "nonfinite CSV value: {}:{}".format(path, field)
                        )
                count += 1
        if count == 0:
            raise AssertionError("CSV has no data rows: {}".format(path))
        csv_row_counts[str(path.relative_to(OUTPUT))] = count

    figure_paths = sorted(OUTPUT.rglob("*.png"))
    for path in figure_paths:
        if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
            raise AssertionError("figure cannot be decoded: {}".format(path))

    tracking = read_json(OUTPUT / "tracking" / "metrics.json")
    pnp = read_json(OUTPUT / "pnp" / "metrics.json")
    validation = read_json(OUTPUT / "validation.json")
    components = {
        "tracking": tracking["scientific_state_sha256"],
        "pnp": pnp["scientific_state_sha256"],
    }
    if components["tracking"] != EXPECTED_TRACKING_SCIENTIFIC_STATE_SHA256:
        raise AssertionError("Tracking scientific-state hash differs")
    if components["pnp"] != EXPECTED_PNP_SCIENTIFIC_STATE_SHA256:
        raise AssertionError("PnP scientific-state hash differs")
    if validation["scientific_state_components"] != components:
        raise AssertionError("validation scientific-state components differ")
    package_state = canonical_sha256(components)
    if package_state != EXPECTED_PACKAGE_SCIENTIFIC_STATE_SHA256:
        raise AssertionError("package scientific-state hash differs")
    if validation["scientific_state_sha256"] != package_state:
        raise AssertionError("validation package scientific-state hash differs")
    if manifest["scientific_state_sha256"] != package_state:
        raise AssertionError("manifest package scientific-state hash differs")
    if any(path.suffix in {".pkl", ".pickle"} for path in OUTPUT.rglob("*")):
        raise AssertionError("published package contains a database pickle")

    return {
        "manifest_artifact_count": manifest[
            "artifact_count_excluding_manifest"
        ],
        "json_file_count": len(json_paths),
        "csv_row_counts": csv_row_counts,
        "figure_count": len(figure_paths),
        "tracking_scientific_state_sha256": components["tracking"],
        "pnp_scientific_state_sha256": components["pnp"],
        "package_scientific_state_sha256": package_state,
        "tracking_database_loaded": False,
    }


def preflight(
    output: Path,
    resume_before_database: bool = False,
    allow_existing_empty_output: bool = False,
) -> Tuple[str, str]:
    global OUTPUT
    OUTPUT = output.resolve()
    if not ROOT.is_dir() or not (ROOT / ".git").is_dir():
        raise AssertionError("repository root is invalid")
    published = (ROOT / "project_submission").resolve()
    if (
        OUTPUT == published
        or _path_is_within(OUTPUT, published)
        or _path_is_within(published, OUTPUT)
    ):
        raise AssertionError(
            "staging output must not equal, contain, or be inside the "
            "published project_submission package"
        )
    if OUTPUT == ROOT.resolve() or _path_is_within(OUTPUT, ROOT.resolve()):
        raise AssertionError("staging output must be external to the repository")
    if resume_before_database and allow_existing_empty_output:
        raise AssertionError("resume and existing-empty modes are mutually exclusive")
    if OUTPUT.exists():
        if not OUTPUT.is_dir():
            raise FileExistsError("staging output exists and is not a directory")
        if allow_existing_empty_output:
            if any(OUTPUT.iterdir()):
                raise FileExistsError(
                    "explicitly fresh staging output is not empty"
                )
        elif not resume_before_database:
            raise FileExistsError(
                "staging output already exists; use a new absent path or "
                "--allow-existing-empty-output for an explicitly fresh "
                "empty directory"
            )
    else:
        OUTPUT.mkdir(parents=True, exist_ok=False)
    target_script = OUTPUT / "generate_current_stage.py"
    source_script = Path(__file__).resolve()
    if source_script != target_script.resolve():
        if target_script.exists():
            raise FileExistsError("staging generator already exists")
        shutil.copyfile(source_script, target_script)
    unexpected = [path for path in OUTPUT.iterdir() if path != target_script]
    if unexpected and not resume_before_database:
        raise FileExistsError("staging directory is not fresh")
    if resume_before_database:
        allowed_partial = {
            "code_locations.json",
            "generate_current_stage.py",
            "pnp",
            "tracking",
        }
        if {path.name for path in OUTPUT.iterdir()}.difference(allowed_partial):
            raise AssertionError("unexpected path in pre-database resume staging")
        if (OUTPUT / "pnp" / "metrics.json").exists() or (
            OUTPUT / "pnp" / "data" / "projection_by_frame_offset.csv"
        ).exists():
            raise AssertionError("resume staging passed the database boundary")
    git_head = git_output("rev-parse", "HEAD").strip()
    git_status = git_output("status", "--short", "--branch")
    if file_sha256(ROOT / PDF_RELATIVE) != PDF_SHA256:
        raise AssertionError("canonical PDF hash differs")
    if (ROOT / FINAL_RELATIVE / "tracking_database.pkl").stat().st_size != 2484226639:
        raise AssertionError("retained database size differs")
    return git_head, git_status


def main(argv: Sequence[str] = None) -> None:
    global ROOT, OUTPUT, COMMAND, RESUMED_BEFORE_DATABASE
    global AUTHORIZED_RECOVERY, RECOVERY_HISTORY
    parser = argparse.ArgumentParser(
        description=(
            "Generate PS-P01 into fresh external staging, or verify the "
            "published artifacts without loading TrackingDB."
        )
    )
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--verify-artifacts-only", action="store_true")
    parser.add_argument(
        "--package-dir",
        type=Path,
        help=(
            "package verified by --verify-artifacts-only; defaults to the "
            "directory containing this published generator"
        ),
    )
    parser.add_argument("--allow-existing-empty-output", action="store_true")
    parser.add_argument("--resume-before-database", action="store_true")
    parser.add_argument(
        "--authorized-recovery-deserialization", action="store_true"
    )
    parser.add_argument(
        "--recovery-provenance-json",
        type=Path,
        help=(
            "explicit attempt-history JSON required only with an authorized "
            "recovery; normal generation records clean-run provenance"
        ),
    )
    args = parser.parse_args(argv)
    if args.repo_root is None:
        ROOT = Path(__file__).resolve().parents[1]
    else:
        ROOT = args.repo_root.resolve()
    if args.verify_artifacts_only:
        if args.output_dir is not None:
            parser.error("--output-dir is not used with artifact verification")
        if args.authorized_recovery_deserialization or args.recovery_provenance_json:
            parser.error("recovery options are not used with artifact verification")
        package = (
            args.package_dir.resolve()
            if args.package_dir is not None
            else Path(__file__).resolve().parent
        )
        summary = verify_artifacts_only(package)
        print("PS-P01 artifact-only verification: PASS")
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return
    if args.package_dir is not None:
        parser.error("--package-dir requires --verify-artifacts-only")
    if args.output_dir is None:
        parser.error("--output-dir is required for generation or preflight")
    if args.resume_before_database and not args.authorized_recovery_deserialization:
        parser.error("--resume-before-database requires an authorized recovery")
    if bool(args.authorized_recovery_deserialization) != bool(
        args.recovery_provenance_json
    ):
        parser.error(
            "authorized recovery requires --recovery-provenance-json, and "
            "recovery provenance is not accepted for a normal clean run"
        )
    OUTPUT = args.output_dir.resolve()
    RESUMED_BEFORE_DATABASE = bool(args.resume_before_database)
    AUTHORIZED_RECOVERY = bool(args.authorized_recovery_deserialization)
    RECOVERY_HISTORY = (
        read_json(args.recovery_provenance_json.resolve())
        if args.recovery_provenance_json is not None
        else None
    )
    attempt_history()
    COMMAND = " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])

    try:
        git_head, git_status_before = preflight(
            OUTPUT,
            resume_before_database=args.resume_before_database,
            allow_existing_empty_output=args.allow_existing_empty_output,
        )
        if args.preflight_only:
            inputs = input_inventory()
            if len(inputs) < 20:
                raise AssertionError("input inventory is incomplete")
            print("PS-P01 preflight: PASS")
            return
        for directory in (
            OUTPUT / "tracking" / "data",
            OUTPUT / "tracking" / "figures",
            OUTPUT / "pnp" / "data",
            OUTPUT / "pnp" / "figures",
        ):
            directory.mkdir(
                parents=True, exist_ok=args.resume_before_database
            )

        inputs = input_inventory()
        code_locations = write_code_locations(git_head)
        tracking, tracking_figures = materialize_tracking()
        pnp, pnp_figures = materialize_pnp()
        requirements = write_requirements_matrix()
        write_provenance(
            git_head,
            git_status_before,
            inputs,
            tracking,
            pnp,
            tracking_figures,
            pnp_figures,
        )
        write_readme(tracking, pnp)

        validation = validate_stage(
            tracking,
            pnp,
            requirements,
            code_locations,
            tracking_figures,
            pnp_figures,
            inputs,
            git_status_before,
        )
        scientific_state = {
            "tracking": tracking["scientific_state_sha256"],
            "pnp": pnp["scientific_state_sha256"],
        }
        validation["git"] = {
            "commit": git_head,
            "status_short_branch_before_generation": git_status_before,
            "status_short_branch_after_staging": git_output(
                "status", "--short", "--branch"
            ),
        }
        validation["generation_command"] = COMMAND
        validation["source_inputs_sha256"] = inputs
        validation["scientific_state_components"] = scientific_state
        validation["scientific_state_sha256"] = canonical_sha256(
            scientific_state
        )
        validation["manifest_policy"] = (
            "all package files except manifest.json are SHA-256 bound; "
            "manifest self-hash is excluded to avoid recursion"
        )
        write_json(OUTPUT / "validation.json", validation)
        write_validation_report(validation, tracking, pnp)
        manifest = write_manifest(
            git_head, inputs, validation["scientific_state_sha256"]
        )
        verify_manifest(manifest)
        if git_output("status", "--short", "--branch") != git_status_before:
            raise AssertionError("pre-existing Git status changed after manifest")
        print("PS-P01 staged validation: PASS")
        print("Scientific state: {}".format(validation["scientific_state_sha256"]))
        print("Artifacts: {}".format(manifest["artifact_count_excluding_manifest"] + 1))
    finally:
        if MATPLOTLIB_CACHE.exists():
            shutil.rmtree(str(MATPLOTLIB_CACHE))


def relative_subsections(
    estimated_poses: Sequence[np.ndarray],
    truth_poses: Sequence[np.ndarray],
    cumulative_truth_distance: np.ndarray,
    length: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows = []
    for start_frame in range(len(estimated_poses) - length + 1):
        end_frame = start_frame + length - 1
        estimated_movement = relative_pose(
            estimated_poses[start_frame], estimated_poses[end_frame]
        )
        truth_movement = relative_pose(
            truth_poses[start_frame], truth_poses[end_frame]
        )
        error_pose = relative_pose(estimated_movement, truth_movement)
        path_distance = float(
            cumulative_truth_distance[end_frame]
            - cumulative_truth_distance[start_frame]
        )
        if not math.isfinite(path_distance) or path_distance <= 0.0:
            raise AssertionError("subsection ground-truth distance is invalid")
        location_error = float(np.linalg.norm(error_pose[:, 3]))
        angle_error = rotation_degrees(error_pose[:, :3])
        rows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sequence_length_frames": length,
                "ground_truth_path_distance_m": path_distance,
                "location_error_m": location_error,
                "location_error_per_meter_m_per_m": (
                    location_error / path_distance
                ),
                "location_error_percent": (
                    100.0 * location_error / path_distance
                ),
                "angle_error_deg": angle_error,
                "angle_error_per_meter_deg_per_m": (
                    angle_error / path_distance
                ),
            }
        )
    expected = EXPECTED["relative_row_counts"][length]
    if len(rows) != expected:
        raise AssertionError(
            "relative subsection row count differs for {}: {}".format(
                length, len(rows)
            )
        )
    summary = {
        "sequence_length_frames": length,
        "row_count": len(rows),
        "endpoint_policy": (
            "start frame a, end frame a+length-1; for length 100, "
            "the first row is 0->99"
        ),
        "denominator_policy": (
            "sum of consecutive ground-truth camera-center distances "
            "from start through end, not endpoint displacement"
        ),
        "mean_location_error_per_meter_m_per_m": float(
            np.mean(
                [row["location_error_per_meter_m_per_m"] for row in rows]
            )
        ),
        "mean_location_error_percent": float(
            np.mean([row["location_error_percent"] for row in rows])
        ),
        "mean_angle_error_per_meter_deg_per_m": float(
            np.mean(
                [row["angle_error_per_meter_deg_per_m"] for row in rows]
            )
        ),
        "location_error_m": numeric_summary(
            [row["location_error_m"] for row in rows]
        ),
        "angle_error_deg": numeric_summary(
            [row["angle_error_deg"] for row in rows]
        ),
    }
    return rows, summary


def project_point(
    intrinsic: np.ndarray, pose: np.ndarray, point_world: np.ndarray
) -> np.ndarray:
    point_camera = (
        pose[:, :3] @ point_world.reshape(3, 1) + pose[:, 3:]
    )
    depth = float(point_camera[2, 0])
    if not math.isfinite(depth) or depth <= 0.0:
        raise AssertionError("projection violates cheirality")
    homogeneous = intrinsic @ point_camera
    pixel = homogeneous[:2, 0] / homogeneous[2, 0]
    if not np.all(np.isfinite(pixel)):
        raise AssertionError("projection is nonfinite")
    return pixel


def right_pose(left_pose: np.ndarray, stereo_pose: np.ndarray) -> np.ndarray:
    stereo_rotation = stereo_pose[:, :3]
    stereo_translation = stereo_pose[:, 3:]
    left_rotation = left_pose[:, :3]
    left_translation = left_pose[:, 3:]
    return np.hstack(
        (
            stereo_rotation @ left_rotation,
            stereo_rotation @ left_translation + stereo_translation,
        )
    )


def compute_projection_aggregates(
    estimated_poses: Sequence[np.ndarray],
    cumulative_truth_distance: np.ndarray,
) -> Dict[str, object]:
    """Deserialize once and aggregate all eligible tracks in bounded memory."""

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tracking_database import TrackingDB

    database_path = ROOT / FINAL_RELATIVE / "tracking_database.pkl"
    if file_sha256(database_path) != DATABASE_SHA256:
        raise AssertionError("retained TrackingDB hash differs")
    intrinsic, _, stereo_pose = read_calibration()
    database = TrackingDB()
    database.load(str(database_path.with_suffix("")))
    if database.frame_num() != EXPECTED["frame_count"]:
        raise AssertionError("deserialized TrackingDB frame count differs")

    offset_accumulators = defaultdict(ErrorAccumulator)
    distance_accumulators = defaultdict(ErrorAccumulator)
    distance_track_counts = defaultdict(int)
    distance_bin_width_m = 5.0
    eligible_track_count = 0
    included_track_count = 0
    total_observation_count = 0

    for track_id in range(database.last_trackId + 1):
        frames = database.trackId_to_frames[track_id]
        if len(frames) < 10:
            continue
        eligible_track_count += 1
        if frames != list(range(frames[0], frames[-1] + 1)):
            raise AssertionError("eligible track is not consecutive")
        anchor_frame = frames[0]
        anchor_link = database.link(anchor_frame, track_id)
        anchor_left_pose = estimated_poses[anchor_frame]
        anchor_right_pose = right_pose(anchor_left_pose, stereo_pose)
        homogeneous = cv2.triangulatePoints(
            intrinsic @ anchor_left_pose,
            intrinsic @ anchor_right_pose,
            np.asarray(
                [[anchor_link.x_left], [anchor_link.y]], dtype=float
            ),
            np.asarray(
                [[anchor_link.x_right], [anchor_link.y]], dtype=float
            ),
        )
        scale = float(homogeneous[3, 0])
        if not math.isfinite(scale) or abs(scale) < 1e-15:
            raise AssertionError("PnP triangulation homogeneous scale invalid")
        point_world = homogeneous[:3, 0] / scale
        if not np.all(np.isfinite(point_world)):
            raise AssertionError("PnP triangulated point is nonfinite")

        track_distance_bins = set()
        for frame_id in frames:
            link = database.link(frame_id, track_id)
            left_pose = estimated_poses[frame_id]
            frame_right_pose = right_pose(left_pose, stereo_pose)
            left_pixel = project_point(intrinsic, left_pose, point_world)
            right_pixel = project_point(
                intrinsic, frame_right_pose, point_world
            )
            observed_left = np.asarray([link.x_left, link.y], dtype=float)
            observed_right = np.asarray([link.x_right, link.y], dtype=float)
            left_error = float(np.linalg.norm(left_pixel - observed_left))
            right_error = float(np.linalg.norm(right_pixel - observed_right))
            offset = frame_id - anchor_frame
            path_distance = float(
                cumulative_truth_distance[frame_id]
                - cumulative_truth_distance[anchor_frame]
            )
            distance_bin = int(
                math.floor(path_distance / distance_bin_width_m)
            )
            offset_accumulators[offset].add(left_error, right_error)
            distance_accumulators[distance_bin].add(left_error, right_error)
            track_distance_bins.add(distance_bin)
            total_observation_count += 1
        for distance_bin in track_distance_bins:
            distance_track_counts[distance_bin] += 1
        included_track_count += 1

    if eligible_track_count != EXPECTED["eligible_projection_tracks"]:
        raise AssertionError("eligible projection-track count differs")
    if included_track_count != eligible_track_count:
        raise AssertionError("not all eligible projection tracks were included")

    offset_rows = []
    for offset in sorted(offset_accumulators):
        offset_rows.append(
            {
                "frame_offset_frames": offset,
                "contributing_track_count": offset_accumulators[offset].count,
                **offset_accumulators[offset].result(),
            }
        )
    distance_rows = []
    for distance_bin in sorted(distance_accumulators):
        start = distance_bin * distance_bin_width_m
        distance_rows.append(
            {
                "distance_bin_index": distance_bin,
                "distance_bin_start_m": start,
                "distance_bin_end_m": start + distance_bin_width_m,
                "distance_bin_center_m": start + distance_bin_width_m / 2.0,
                "contributing_track_count": distance_track_counts[distance_bin],
                **distance_accumulators[distance_bin].result(),
            }
        )

    write_csv(
        OUTPUT / "pnp" / "data" / "projection_by_frame_offset.csv",
        tuple(offset_rows[0].keys()),
        offset_rows,
    )
    write_csv(
        OUTPUT / "pnp" / "data" / "projection_by_path_distance.csv",
        tuple(distance_rows[0].keys()),
        distance_rows,
    )

    result = {
        "definition": (
            "Each eligible track is triangulated once from its first stereo "
            "Link using retained PnP poses, then projected into every Link "
            "of that track with the retained PnP poses. This is distinct "
            "from the retained Track 305 ground-truth reprojection analysis."
        ),
        "eligibility": "all TrackingDB tracks with length at least 10",
        "selection": "complete population; no random or deterministic subset",
        "eligible_track_count": eligible_track_count,
        "included_track_count": included_track_count,
        "observation_count": total_observation_count,
        "distance_bin_width_m": distance_bin_width_m,
        "frame_offset_row_count": len(offset_rows),
        "path_distance_row_count": len(distance_rows),
        "database_deserialization_count": 1,
        "aggregation_memory_policy": (
            "fixed accumulators by frame offset and 5 m ground-truth path "
            "distance bin; no per-observation table retained"
        ),
        "offset_rows": offset_rows,
        "distance_rows": distance_rows,
    }

    del database
    gc.collect()
    return result


def materialize_pnp() -> Tuple[Dict[str, object], Dict[str, object]]:
    companion = read_json(
        ROOT / FINAL_RELATIVE / "tracking_database_companion.json"
    )
    raw_estimated = companion["tracking_state"]["world_to_camera_poses"]
    if len(raw_estimated) != EXPECTED["frame_count"]:
        raise AssertionError("retained PnP pose count differs")
    estimated = [np.asarray(pose, dtype=float) for pose in raw_estimated]
    truth = read_ground_truth(len(estimated))
    estimated_anchored = anchored_poses(estimated)
    truth_anchored = anchored_poses(truth)
    estimated_centers = camera_centers(estimated_anchored)
    truth_centers = camera_centers(truth_anchored)
    signed_errors = estimated_centers - truth_centers
    position_errors = np.linalg.norm(signed_errors, axis=1)
    rotation_errors = np.asarray(
        [
            rotation_degrees(
                truth_pose[:, :3] @ estimated_pose[:, :3].T
            )
            for estimated_pose, truth_pose in zip(
                estimated_anchored, truth_anchored
            )
        ],
        dtype=float,
    )
    truth_steps = np.linalg.norm(np.diff(truth_centers, axis=0), axis=1)
    cumulative_truth_distance = np.concatenate(
        (np.asarray([0.0]), np.cumsum(truth_steps))
    )
    total_truth_distance = float(np.sum(truth_steps))
    final_position_error = float(position_errors[-1])
    final_drift = 100.0 * final_position_error / total_truth_distance

    trajectory_rows = []
    for frame_id in range(len(estimated)):
        trajectory_rows.append(
            {
                "frame_id": frame_id,
                "ground_truth_path_distance_m": cumulative_truth_distance[
                    frame_id
                ],
                "estimated_x_m": estimated_centers[frame_id, 0],
                "estimated_y_m": estimated_centers[frame_id, 1],
                "estimated_z_m": estimated_centers[frame_id, 2],
                "ground_truth_x_m": truth_centers[frame_id, 0],
                "ground_truth_y_m": truth_centers[frame_id, 1],
                "ground_truth_z_m": truth_centers[frame_id, 2],
                "signed_x_error_m": signed_errors[frame_id, 0],
                "signed_y_error_m": signed_errors[frame_id, 1],
                "signed_z_error_m": signed_errors[frame_id, 2],
                "total_position_error_m": position_errors[frame_id],
                "rotation_error_deg": rotation_errors[frame_id],
            }
        )
    write_csv(
        OUTPUT / "pnp" / "data" / "trajectory_and_absolute_errors.csv",
        tuple(trajectory_rows[0].keys()),
        trajectory_rows,
    )

    relative_summaries = {}
    relative_rows_by_length = {}
    for length in (100, 400, 800):
        rows, summary = relative_subsections(
            estimated_anchored,
            truth_anchored,
            cumulative_truth_distance,
            length,
        )
        relative_rows_by_length[length] = rows
        relative_summaries[str(length)] = summary
        write_csv(
            OUTPUT
            / "pnp"
            / "data"
            / "relative_subsections_{}.csv".format(length),
            tuple(rows[0].keys()),
            rows,
        )

    projection = compute_projection_aggregates(
        estimated_anchored, cumulative_truth_distance
    )
    projection_offset_rows = projection.pop("offset_rows")
    projection_distance_rows = projection.pop("distance_rows")

    core = {
        "schema": "ps-p01-pnp-metrics-v1",
        "scope": {
            "dataset": "KITTI odometry sequence 00",
            "frame_range_inclusive": [0, 3359],
            "frame_count": len(estimated),
            "pose_convention": "world-to-camera: x_i = R_i X + t_i",
            "anchoring": "both PnP and ground truth expressed relative to frame 0",
        },
        "absolute_error_definition": {
            "signed_axes": "estimated camera center minus ground-truth camera center",
            "position": "Euclidean norm of signed X/Y/Z errors",
            "rotation": (
                "Rodrigues norm in degrees of ground-truth anchored "
                "rotation times estimated anchored rotation transpose"
            ),
        },
        "absolute_errors": {
            "signed_x_m": numeric_summary(signed_errors[:, 0]),
            "signed_y_m": numeric_summary(signed_errors[:, 1]),
            "signed_z_m": numeric_summary(signed_errors[:, 2]),
            "total_position_m": numeric_summary(position_errors),
            "rotation_deg": numeric_summary(rotation_errors),
            "final_position_error_m": final_position_error,
            "total_ground_truth_distance_m": total_truth_distance,
            "final_drift_percent": final_drift,
        },
        "relative_subsections": {
            "appendix_definition": (
                "error pose is the relative pose from estimated subsection "
                "movement to ground-truth subsection movement; translation "
                "norm and Rodrigues rotation magnitude are divided by summed "
                "ground-truth inter-frame path distance"
            ),
            "lengths": relative_summaries,
        },
        "projection_error": projection,
        "retained_track_305_ground_truth_reprojection_reused_as_pnp": False,
    }
    core["scientific_state_sha256"] = canonical_sha256(core)
    write_json(OUTPUT / "pnp" / "metrics.json", core)
    figure_specs = plot_pnp_figures(
        trajectory_rows,
        relative_rows_by_length,
        projection_offset_rows,
        projection_distance_rows,
    )
    return core, figure_specs


def plot_pnp_figures(
    trajectory_rows: Sequence[Dict[str, object]],
    relative_rows_by_length: Dict[int, Sequence[Dict[str, object]]],
    projection_offset_rows: Sequence[Dict[str, object]],
    projection_distance_rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    specs = {}
    frames = [row["frame_id"] for row in trajectory_rows]

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.plot(
        [row["ground_truth_x_m"] for row in trajectory_rows],
        [row["ground_truth_z_m"] for row in trajectory_rows],
        label="KITTI ground truth",
        linewidth=1.2,
        color="black",
    )
    axis.plot(
        [row["estimated_x_m"] for row in trajectory_rows],
        [row["estimated_z_m"] for row in trajectory_rows],
        label="PnP estimate",
        linewidth=1.0,
        color="tab:blue",
    )
    axis.set_xlabel("X position from frame-0 origin [m]")
    axis.set_ylabel("Z position from frame-0 origin [m]")
    axis.set_title("Frame-0-anchored PnP trajectory")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.25)
    axis.legend()
    name = "trajectory_xz.png"
    save_figure(figure, OUTPUT / "pnp" / "figures" / name)
    specs[name] = {
        "x_label": "X position from frame-0 origin [m]",
        "y_label": "Z position from frame-0 origin [m]",
        "equal_aspect": True,
        "series": ["KITTI ground truth", "PnP estimate"],
    }

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        frames,
        [row["signed_x_error_m"] for row in trajectory_rows],
        label="Signed X error",
        linewidth=0.8,
    )
    axis.plot(
        frames,
        [row["signed_y_error_m"] for row in trajectory_rows],
        label="Signed Y error",
        linewidth=0.8,
    )
    axis.plot(
        frames,
        [row["signed_z_error_m"] for row in trajectory_rows],
        label="Signed Z error",
        linewidth=0.8,
    )
    axis.plot(
        frames,
        [row["total_position_error_m"] for row in trajectory_rows],
        label="Total position-error norm",
        linewidth=1.0,
        color="black",
    )
    axis.set_xlabel("Frame index [frame]")
    axis.set_ylabel("Position error [m]")
    axis.set_title("Absolute PnP position errors")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    name = "absolute_position_errors.png"
    save_figure(figure, OUTPUT / "pnp" / "figures" / name)
    specs[name] = {
        "x_label": "Frame index [frame]",
        "y_label": "Position error [m]",
        "series": ["signed X", "signed Y", "signed Z", "total norm"],
    }

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        frames,
        [row["rotation_error_deg"] for row in trajectory_rows],
        linewidth=0.8,
        color="tab:red",
    )
    axis.set_xlabel("Frame index [frame]")
    axis.set_ylabel("Rotation error [deg]")
    axis.set_title("Absolute PnP rotation error")
    axis.grid(alpha=0.25)
    name = "absolute_rotation_error.png"
    save_figure(figure, OUTPUT / "pnp" / "figures" / name)
    specs[name] = {
        "x_label": "Frame index [frame]",
        "y_label": "Rotation error [deg]",
    }

    for length, rows in sorted(relative_rows_by_length.items()):
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        starts = [row["start_frame"] for row in rows]
        axes[0].plot(
            starts,
            [row["location_error_percent"] for row in rows],
            linewidth=0.8,
            color="tab:blue",
        )
        axes[0].set_ylabel("Translation error [%]")
        axes[0].grid(alpha=0.25)
        axes[0].set_title(
            "Relative PnP subsection error: {} frames".format(length)
        )
        axes[1].plot(
            starts,
            [row["angle_error_per_meter_deg_per_m"] for row in rows],
            linewidth=0.8,
            color="tab:orange",
        )
        axes[1].set_xlabel("Subsection start frame [frame]")
        axes[1].set_ylabel("Rotation error per meter [deg/m]")
        axes[1].grid(alpha=0.25)
        name = "relative_subsections_{}.png".format(length)
        save_figure(figure, OUTPUT / "pnp" / "figures" / name)
        specs[name] = {
            "x_label": "Subsection start frame [frame]",
            "y_labels": [
                "Translation error [%]",
                "Rotation error per meter [deg/m]",
            ],
        }

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        [row["frame_offset_frames"] for row in projection_offset_rows],
        [row["left_mean_px"] for row in projection_offset_rows],
        label="Left mean",
        linewidth=1.0,
    )
    axis.plot(
        [row["frame_offset_frames"] for row in projection_offset_rows],
        [row["right_mean_px"] for row in projection_offset_rows],
        label="Right mean",
        linewidth=1.0,
    )
    axis.set_xlabel("Offset from triangulation frame [frames]")
    axis.set_ylabel("Mean projection error [px]")
    axis.set_title("PnP projection error versus frame offset")
    axis.grid(alpha=0.25)
    axis.legend()
    name = "projection_vs_frame_offset.png"
    save_figure(figure, OUTPUT / "pnp" / "figures" / name)
    specs[name] = {
        "x_label": "Offset from triangulation frame [frames]",
        "y_label": "Mean projection error [px]",
        "population": "all tracks of length at least 10",
    }

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        [row["distance_bin_center_m"] for row in projection_distance_rows],
        [row["left_mean_px"] for row in projection_distance_rows],
        label="Left mean",
        linewidth=1.0,
    )
    axis.plot(
        [row["distance_bin_center_m"] for row in projection_distance_rows],
        [row["right_mean_px"] for row in projection_distance_rows],
        label="Right mean",
        linewidth=1.0,
    )
    axis.set_xlabel("GT path distance from triangulation frame [m]")
    axis.set_ylabel("Mean projection error [px]")
    axis.set_title("PnP projection error versus ground-truth path distance")
    axis.grid(alpha=0.25)
    axis.legend()
    name = "projection_vs_path_distance.png"
    save_figure(figure, OUTPUT / "pnp" / "figures" / name)
    specs[name] = {
        "x_label": "GT path distance from triangulation frame [m]",
        "y_label": "Mean projection error [px]",
        "population": "all tracks of length at least 10",
        "distance_bin_width_m": 5.0,
    }
    return specs


if __name__ == "__main__":
    try:
        main()
    finally:
        if MATPLOTLIB_CACHE.exists():
            shutil.rmtree(str(MATPLOTLIB_CACHE))
