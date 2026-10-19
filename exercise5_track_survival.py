"""Frozen E5-I03C track-survival policy, finalization, and comparison.

The experiment configuration is an overlay on the protected E5-I03B
production configuration.  It contains no scientific settings: the loader
verifies the exact production-configuration SHA-256, validates that canonical
configuration with its unchanged strict validator, and replaces only the
keyframe selection and experimental artifact-root metadata in memory.
"""

from __future__ import annotations

import copy
import csv
import gc
import hashlib
import json
import os
import statistics
import uuid
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from exercise5_full_run import (
    ArtifactValidationError,
    DEFAULT_CONFIG_PATH,
    Exercise5ContractError,
    ROOT,
    _enforce_memory_guard,
    _load_canonical_once,
    _peak_rss_bytes,
    build_run_manifest,
    load_frozen_configuration,
    sha256_file,
    utc_now,
    validate_run_root,
    validate_source_and_input_hashes,
)


EXPERIMENT_SCHEMA = "exercise5-track-survival-experiment-config"
EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_RULE_ID = "track-last-frame-quantile-082-tail-safe-v1"
EXPERIMENT_CONFIRMATION = "RUN-TRACK-SURVIVAL-082"
EXPERIMENT_SCOPE = "track-survival-082-experiment"
EXPERIMENT_QUANTILE = 0.82
FIRST_FRAME_ID = 0
FINAL_FRAME_ID = 3359
MINIMUM_ACTUAL_FRAMES = 5
MAXIMUM_ACTUAL_FRAMES = 20
MINIMUM_ENDPOINT_DELTA = MINIMUM_ACTUAL_FRAMES - 1
MAXIMUM_ENDPOINT_DELTA = MAXIMUM_ACTUAL_FRAMES - 1
EXPERIMENT_CONFIG_PATH = ROOT / "exercise5_track_survival_082_config.json"
EXPERIMENT_OUTPUT_ROOT = ROOT / "diagnostics" / "ex5_track_survival_082"
BASELINE_OUTPUT_ROOT = ROOT / "diagnostics" / "ex5_final"
COMPARISON_JSON_NAME = "e5_i03b_vs_e5_i03c_comparison.json"
COMPARISON_MARKDOWN_NAME = "e5_i03b_vs_e5_i03c_comparison.md"
COMPARISON_FIGURE_NAME = "e5_i03b_vs_e5_i03c_trajectory.png"
EXPERIMENT_SOURCE_FILES = (
    "ex5.py",
    "exercise5_track_survival.py",
    "exercise5_full_run.py",
    "exercise5_aggregation.py",
    "exercise5_reporting.py",
)


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _validated_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise Exercise5ContractError("{} must be an integer".format(name))
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exception:
        raise Exercise5ContractError(
            "{} must be an integer".format(name)
        ) from exception
    if normalized != value:
        raise Exercise5ContractError("{} must be an integer".format(name))
    return normalized


def select_track_survival_endpoint(
    last_frame_ids: Sequence[int],
) -> Tuple[int, int]:
    """Apply the exact ``min(n-1, int(0.82*n))`` selection rule."""

    ordered = sorted(
        _validated_integer(value, "last observed FrameId")
        for value in last_frame_ids
    )
    if not ordered:
        raise Exercise5ContractError(
            "track-survival policy cannot select from zero tracks"
        )
    index = min(len(ordered) - 1, int(EXPERIMENT_QUANTILE * len(ordered)))
    return ordered[index], index


def validate_track_survival_keyframes(
    keyframes: Sequence[int],
    *,
    first_frame_id: int = FIRST_FRAME_ID,
    final_frame_id: int = FINAL_FRAME_ID,
) -> Tuple[int, ...]:
    normalized = tuple(
        _validated_integer(value, "keyframe FrameId") for value in keyframes
    )
    if not normalized:
        raise Exercise5ContractError("experimental keyframe list is empty")
    if normalized[0] != first_frame_id:
        raise Exercise5ContractError(
            "experimental keyframes must begin at {}".format(first_frame_id)
        )
    if normalized[-1] != final_frame_id:
        raise Exercise5ContractError(
            "experimental keyframes must end at {}".format(final_frame_id)
        )
    if len(normalized) != len(set(normalized)):
        raise Exercise5ContractError(
            "experimental keyframes contain duplicate FrameIds"
        )
    if any(second <= first for first, second in zip(normalized, normalized[1:])):
        raise Exercise5ContractError(
            "experimental keyframes are not strictly increasing"
        )
    invalid = [
        (first, second, second - first + 1)
        for first, second in zip(normalized, normalized[1:])
        if not MINIMUM_ACTUAL_FRAMES
        <= second - first + 1
        <= MAXIMUM_ACTUAL_FRAMES
    ]
    if invalid:
        raise Exercise5ContractError(
            "experimental windows violate the 5-20 actual-frame contract: "
            "{}".format(invalid[:5])
        )
    return normalized


def _normalize_tracks(database: object, frame_id: int) -> Tuple[int, ...]:
    tracks = tuple(
        _validated_integer(track_id, "TrackId")
        for track_id in database.tracks(frame_id)
    )
    if not tracks:
        raise Exercise5ContractError(
            "frame {} has no tracks for keyframe selection".format(frame_id)
        )
    if len(tracks) != len(set(tracks)) or any(track_id < 0 for track_id in tracks):
        raise Exercise5ContractError(
            "frame {} returned invalid TrackIds".format(frame_id)
        )
    return tuple(sorted(tracks))


def derive_track_survival_keyframes(
    database: object,
    *,
    first_frame_id: int = FIRST_FRAME_ID,
    final_frame_id: int = FINAL_FRAME_ID,
) -> Dict[str, object]:
    """Derive one deterministic full-coverage keyframe list.

    Quantile selection is followed by the two span clamps.  The tail stage
    selects the final frame directly when it fits, or reserves exactly four
    transitions when the maximum-span endpoint would leave a 1-3 transition
    tail.
    """

    if final_frame_id - first_frame_id < MINIMUM_ENDPOINT_DELTA:
        raise Exercise5ContractError(
            "sequence is too short for one five-frame window"
        )
    keyframes = [first_frame_id]
    decisions: List[Dict[str, object]] = []
    adjustment_counts = {
        "minimum_5_frame": 0,
        "maximum_20_frame": 0,
        "final_tail_repair": 0,
    }
    while keyframes[-1] < final_frame_id:
        start = keyframes[-1]
        remaining = final_frame_id - start
        if remaining < MINIMUM_ENDPOINT_DELTA:
            raise Exercise5ContractError(
                "deterministic tail repair reached an invalid {}-transition "
                "tail at frame {}".format(remaining, start)
            )
        tracks = _normalize_tracks(database, start)
        last_frame_ids = []
        for track_id in tracks:
            last = _validated_integer(
                database.last_frame_of_track(track_id),
                "last observed FrameId",
            )
            if last < start or last > final_frame_id:
                raise Exercise5ContractError(
                    "TrackId {} at frame {} has invalid last FrameId {}".format(
                        track_id, start, last
                    )
                )
            last_frame_ids.append(last)
        raw_endpoint, quantile_index = select_track_survival_endpoint(
            last_frame_ids
        )
        after_minimum = max(raw_endpoint, start + MINIMUM_ENDPOINT_DELTA)
        minimum_changed = after_minimum != raw_endpoint
        after_maximum = min(after_minimum, start + MAXIMUM_ENDPOINT_DELTA)
        maximum_changed = after_maximum != after_minimum
        selected = after_maximum
        tail_changed = False
        if remaining <= MAXIMUM_ENDPOINT_DELTA:
            selected = final_frame_id
            tail_changed = selected != after_maximum
        elif 0 < final_frame_id - selected < MINIMUM_ENDPOINT_DELTA:
            selected = final_frame_id - MINIMUM_ENDPOINT_DELTA
            tail_changed = selected != after_maximum
        if minimum_changed:
            adjustment_counts["minimum_5_frame"] += 1
        if maximum_changed:
            adjustment_counts["maximum_20_frame"] += 1
        if tail_changed:
            adjustment_counts["final_tail_repair"] += 1
        actual_frames = selected - start + 1
        if not MINIMUM_ACTUAL_FRAMES <= actual_frames <= MAXIMUM_ACTUAL_FRAMES:
            raise Exercise5ContractError(
                "deterministic policy produced invalid window {}-{} "
                "({} actual frames)".format(start, selected, actual_frames)
            )
        if selected <= start:
            raise Exercise5ContractError(
                "deterministic policy did not advance from frame {}".format(
                    start
                )
            )
        decisions.append(
            {
                "window_index": len(decisions),
                "start_frame_id": start,
                "track_count": len(tracks),
                "quantile_index": quantile_index,
                "raw_quantile_endpoint_frame_id": raw_endpoint,
                "endpoint_after_minimum": after_minimum,
                "endpoint_after_maximum": after_maximum,
                "selected_endpoint_frame_id": selected,
                "actual_frame_count": actual_frames,
                "minimum_changed": minimum_changed,
                "maximum_changed": maximum_changed,
                "final_tail_changed": tail_changed,
            }
        )
        keyframes.append(selected)
    validated = validate_track_survival_keyframes(
        keyframes,
        first_frame_id=first_frame_id,
        final_frame_id=final_frame_id,
    )
    window_lengths = [
        second - first + 1
        for first, second in zip(validated, validated[1:])
    ]
    decision_sha256 = hashlib.sha256(
        json.dumps(
            decisions,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "ordered_keyframe_ids": list(validated),
        "statistics": {
            "keyframe_count": len(validated),
            "window_count": len(validated) - 1,
            "window_actual_frames": {
                "minimum": min(window_lengths),
                "maximum": max(window_lengths),
                "mean": float(statistics.mean(window_lengths)),
                "median": float(statistics.median(window_lengths)),
            },
            "selection_adjustment_counts": adjustment_counts,
            "decision_trace_sha256": decision_sha256,
        },
        "decisions": decisions,
    }


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _connectivity_statistics(
    keyframes: Sequence[int], transition_path: Path
) -> Dict[str, object]:
    rows = _read_csv(transition_path)
    by_front = {}
    for row in rows:
        raw_front = row.get(
            "front_frame_id", row.get("front_frame", row.get("frame_id"))
        )
        raw_connectivity = row.get(
            "final_supporter_count", row.get("connectivity")
        )
        if raw_front is None or raw_connectivity is None:
            raise Exercise5ContractError(
                "transition connectivity columns are absent"
            )
        by_front[int(raw_front)] = int(float(raw_connectivity))
    low = []
    minimum = None
    for index, (start, end) in enumerate(zip(keyframes, keyframes[1:])):
        values = [by_front[frame_id] for frame_id in range(start + 1, end + 1)]
        if len(values) != end - start:
            raise Exercise5ContractError(
                "transition connectivity is incomplete for window {}-{}".format(
                    start, end
                )
            )
        local_minimum = min(values)
        minimum = (
            local_minimum
            if minimum is None
            else min(minimum, local_minimum)
        )
        if local_minimum < 300:
            low.append(index)
    return {
        "minimum_transition_connectivity": minimum,
        "low_connectivity_definition": (
            "window contains at least one transition with connectivity < 300"
        ),
        "low_connectivity_window_indices": low,
    }


def _selection_contract(derived: Mapping[str, object]) -> Dict[str, object]:
    statistics_value = copy.deepcopy(derived["statistics"])
    return {
        "rule_id": EXPERIMENT_RULE_ID,
        "source": (
            "protected lecturer TrackingDB tracks(FrameId) and "
            "last_frame_of_track(TrackId)"
        ),
        "quantile": EXPERIMENT_QUANTILE,
        "selection_index": "min(n - 1, int(0.82 * n))",
        "first_frame_id": FIRST_FRAME_ID,
        "final_frame_id": FINAL_FRAME_ID,
        "minimum_actual_frames_per_window": MINIMUM_ACTUAL_FRAMES,
        "maximum_actual_frames_per_window": MAXIMUM_ACTUAL_FRAMES,
        "minimum_endpoint_delta_frames": MINIMUM_ENDPOINT_DELTA,
        "maximum_endpoint_delta_frames": MAXIMUM_ENDPOINT_DELTA,
        "tail_policy": (
            "Select final FrameId directly when 4-19 transitions remain; "
            "when a maximum-span endpoint would leave 1-3 transitions, "
            "select final_frame_id-4 before the final window."
        ),
        "ordered_keyframe_ids": list(derived["ordered_keyframe_ids"]),
        "statistics": statistics_value,
    }


def derive_and_persist_experiment_configuration(
    *,
    configuration_path: Path = EXPERIMENT_CONFIG_PATH,
    repository_root: Path = ROOT,
    base_configuration_path: Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, object]:
    """Load the protected TrackingDB exactly once and freeze the overlay."""

    repository_root = Path(repository_root).resolve()
    configuration_path = Path(configuration_path).resolve()
    base_configuration_path = Path(base_configuration_path).resolve()
    if configuration_path.exists():
        raise FileExistsError(
            "experimental configuration is already frozen: {}".format(
                configuration_path
            )
        )
    if base_configuration_path != DEFAULT_CONFIG_PATH.resolve():
        raise Exercise5ContractError(
            "E5-I03C must use the protected production configuration"
        )
    base, base_sha = load_frozen_configuration(base_configuration_path)
    preflight = _enforce_memory_guard(
        "before-track-survival-read-only-database-load"
    )
    source_identity = validate_source_and_input_hashes(
        repository_root, base, hash_database=True
    )
    load_started = perf_counter()
    database, _poses, normalized = _load_canonical_once(repository_root, base)
    database_load_seconds = perf_counter() - load_started
    try:
        derived = derive_track_survival_keyframes(database)
    finally:
        del normalized
        del _poses
        del database
        gc.collect()
    keyframe_selection = _selection_contract(derived)
    keyframe_selection["statistics"].update(
        _connectivity_statistics(
            keyframe_selection["ordered_keyframe_ids"],
            repository_root
            / base["canonical_inputs"]["transition_table"]["path"],
        )
    )
    source_contract = {
        relative: sha256_file(repository_root / relative)
        for relative in EXPERIMENT_SOURCE_FILES
    }
    overlay = {
        "schema": EXPERIMENT_SCHEMA,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "milestone": "E5-I03C",
        "status": "frozen-track-survival-082-experiment",
        "created_utc": utc_now(),
        "base_configuration": {
            "path": str(base_configuration_path.relative_to(repository_root)),
            "sha256": base_sha,
        },
        "experimental_source_contract": source_contract,
        "keyframe_selection": keyframe_selection,
        "artifact_overrides": {
            "canonical_root": str(
                EXPERIMENT_OUTPUT_ROOT.relative_to(repository_root)
            ),
            "curated_root": None,
        },
        "derivation": {
            "database_access": "read-only lecturer API",
            "database_load_count": 1,
            "database_load_seconds": database_load_seconds,
            "peak_rss_bytes": _peak_rss_bytes(),
            "memory_preflight": preflight,
            "source_input_identity": source_identity,
            "decision_trace": derived["decisions"],
        },
    }
    _write_json(configuration_path, overlay, exclusive=True)
    loaded, observed_sha = load_experiment_configuration(
        configuration_path, repository_root=repository_root
    )
    del loaded
    return {
        "configuration_path": str(configuration_path),
        "configuration_sha256": observed_sha,
        "database_load_count": 1,
        "database_load_seconds": database_load_seconds,
        "peak_rss_bytes": _peak_rss_bytes(),
        "keyframe_statistics": keyframe_selection["statistics"],
    }


def _validate_statistics(
    selection: Mapping[str, object], keyframes: Sequence[int]
) -> None:
    statistics_value = selection.get("statistics")
    if not isinstance(statistics_value, Mapping):
        raise Exercise5ContractError(
            "experimental keyframe statistics are absent"
        )
    lengths = [
        second - first + 1
        for first, second in zip(keyframes, keyframes[1:])
    ]
    expected = {
        "keyframe_count": len(keyframes),
        "window_count": len(keyframes) - 1,
        "window_actual_frames": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": float(statistics.mean(lengths)),
            "median": float(statistics.median(lengths)),
        },
    }
    for name, value in expected.items():
        if statistics_value.get(name) != value:
            raise Exercise5ContractError(
                "experimental statistic {} differs".format(name)
            )
    adjustments = statistics_value.get("selection_adjustment_counts")
    if (
        not isinstance(adjustments, Mapping)
        or set(adjustments)
        != {
            "minimum_5_frame",
            "maximum_20_frame",
            "final_tail_repair",
        }
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in adjustments.values()
        )
    ):
        raise Exercise5ContractError(
            "experimental adjustment counts differ"
        )


def load_experiment_configuration(
    path: Path = EXPERIMENT_CONFIG_PATH,
    *,
    repository_root: Path = ROOT,
) -> Tuple[Dict[str, object], str]:
    """Validate the overlay and resolve it against protected E5-I03B science."""

    path = Path(path).resolve()
    repository_root = Path(repository_root).resolve()
    overlay = _read_json(path)
    if not isinstance(overlay, dict):
        raise Exercise5ContractError(
            "experimental configuration must be an object"
        )
    if (
        overlay.get("schema") != EXPERIMENT_SCHEMA
        or overlay.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
    ):
        raise Exercise5ContractError(
            "experimental configuration schema/version differs"
        )
    base_contract = overlay.get("base_configuration")
    if not isinstance(base_contract, Mapping):
        raise Exercise5ContractError(
            "experimental base-configuration contract is absent"
        )
    base_path = (repository_root / base_contract.get("path", "")).resolve()
    if base_path != DEFAULT_CONFIG_PATH.resolve():
        raise Exercise5ContractError(
            "experimental configuration does not pin the protected baseline"
        )
    base, base_sha = load_frozen_configuration(base_path)
    if base_contract.get("sha256") != base_sha:
        raise Exercise5ContractError(
            "protected production-configuration SHA-256 differs"
        )
    source_contract = overlay.get("experimental_source_contract")
    if (
        not isinstance(source_contract, Mapping)
        or set(source_contract) != set(EXPERIMENT_SOURCE_FILES)
    ):
        raise Exercise5ContractError(
            "experimental source contract differs"
        )
    for relative, expected_sha in source_contract.items():
        observed = sha256_file(repository_root / relative)
        if observed != expected_sha:
            raise Exercise5ContractError(
                "experimental source hash mismatch for {}: {}".format(
                    relative, observed
                )
            )
    selection = overlay.get("keyframe_selection")
    if not isinstance(selection, Mapping):
        raise Exercise5ContractError(
            "experimental keyframe selection is absent"
        )
    exact_contract = {
        "rule_id": EXPERIMENT_RULE_ID,
        "quantile": EXPERIMENT_QUANTILE,
        "selection_index": "min(n - 1, int(0.82 * n))",
        "first_frame_id": FIRST_FRAME_ID,
        "final_frame_id": FINAL_FRAME_ID,
        "minimum_actual_frames_per_window": MINIMUM_ACTUAL_FRAMES,
        "maximum_actual_frames_per_window": MAXIMUM_ACTUAL_FRAMES,
        "minimum_endpoint_delta_frames": MINIMUM_ENDPOINT_DELTA,
        "maximum_endpoint_delta_frames": MAXIMUM_ENDPOINT_DELTA,
    }
    for name, expected in exact_contract.items():
        if selection.get(name) != expected:
            raise Exercise5ContractError(
                "experimental keyframe field {} differs".format(name)
            )
    keyframes = validate_track_survival_keyframes(
        selection.get("ordered_keyframe_ids", ())
    )
    _validate_statistics(selection, keyframes)
    overrides = overlay.get("artifact_overrides")
    expected_root = str(EXPERIMENT_OUTPUT_ROOT.relative_to(repository_root))
    if (
        not isinstance(overrides, Mapping)
        or overrides.get("canonical_root") != expected_root
        or overrides.get("curated_root") is not None
    ):
        raise Exercise5ContractError(
            "experimental artifact-root contract differs"
        )
    resolved = copy.deepcopy(base)
    resolved["milestone"] = "E5-I03C"
    resolved["status"] = "frozen-track-survival-082-experiment"
    resolved["keyframe_selection"] = copy.deepcopy(selection)
    resolved["artifact_contract"]["canonical_root"] = expected_root
    resolved["artifact_contract"]["curated_root"] = None
    first_end = keyframes[1]
    registry = resolved["artifact_contract"]["producer_registry"]
    old_prefix = "windows/window_0000_0_7/"
    new_prefix = "windows/window_0000_0_{}/".format(first_end)
    resolved["artifact_contract"]["producer_registry"] = {
        (
            new_prefix + name[len(old_prefix) :]
            if name.startswith(old_prefix)
            else name
        ): producer
        for name, producer in registry.items()
    }
    resolved["artifact_contract"]["seal_policy"] = (
        "Only after every experimental window seal, aggregate/report "
        "regeneration, comparison generation, strict validation, and full "
        "checksum verification may manifest.status become "
        "complete-and-sealed. No curated Bundle package is created."
    )
    resolved["experiment_contract"] = copy.deepcopy(overlay)
    return resolved, sha256_file(path)


def _metric_summary(values: np.ndarray) -> Dict[str, float]:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ArtifactValidationError("comparison metric input is invalid")
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "final": float(values[-1]),
    }


def _policy_metrics(source_root: Path) -> Dict[str, object]:
    data = Path(source_root) / "aggregate" / "data"
    absolute_rows = _read_csv(data / "keyframe_localization_errors.csv")
    relative_rows = _read_csv(data / "consecutive_relative_errors.csv")
    bundle_absolute = np.asarray(
        [float(row["bundle_error_m"]) for row in absolute_rows], dtype=float
    )
    pnp_absolute = np.asarray(
        [float(row["pnp_error_m"]) for row in absolute_rows], dtype=float
    )
    bundle_translation = np.asarray(
        [float(row["bundle_translation_error_m"]) for row in relative_rows],
        dtype=float,
    )
    pnp_translation = np.asarray(
        [float(row["pnp_translation_error_m"]) for row in relative_rows],
        dtype=float,
    )
    bundle_rotation = np.asarray(
        [float(row["bundle_rotation_error_deg"]) for row in relative_rows],
        dtype=float,
    )
    pnp_rotation = np.asarray(
        [float(row["pnp_rotation_error_deg"]) for row in relative_rows],
        dtype=float,
    )

    def comparison(
        bundle: np.ndarray, pnp: np.ndarray
    ) -> Dict[str, object]:
        if bundle.shape != pnp.shape:
            raise ArtifactValidationError(
                "Bundle/PnP comparison shapes differ"
            )
        wins = int(np.count_nonzero(bundle < pnp))
        return {
            "bundle": _metric_summary(bundle),
            "pnp": _metric_summary(pnp),
            "bundle_lower_count": wins,
            "bundle_lower_percentage": 100.0 * wins / len(bundle),
            "sample_count": len(bundle),
        }

    return {
        "keyframe_count": len(absolute_rows),
        "window_count": len(relative_rows),
        "absolute_localization_error_m": comparison(
            bundle_absolute, pnp_absolute
        ),
        "consecutive_relative_translation_error_m": comparison(
            bundle_translation, pnp_translation
        ),
        "consecutive_relative_rotation_error_deg": comparison(
            bundle_rotation, pnp_rotation
        ),
    }


def covariance_warning_summary(source_root: Path) -> Dict[str, object]:
    names = (
        "joint_covariance_information_inverse",
        "supplied_conditional_covariance",
        "conditional_information_block",
    )
    by_kind = {name: 0 for name in names}
    finite_by_kind = {name: 0 for name in names}
    maximum_by_kind = {name: None for name in names}
    for directory in sorted(Path(source_root).glob("windows/window_[0-9]*")):
        summary = _read_json(directory / "summary.json")
        crosschecks = summary["covariance_diagnostics"][
            "numerical_reconstruction_crosschecks"
        ]
        for name in names:
            diagnostic = crosschecks[name]
            if not diagnostic["warning"]:
                continue
            by_kind[name] += 1
            measured = diagnostic["measured_error"]
            if measured is not None and np.isfinite(float(measured)):
                finite_by_kind[name] += 1
                maximum_by_kind[name] = (
                    float(measured)
                    if maximum_by_kind[name] is None
                    else max(maximum_by_kind[name], float(measured))
                )
    return {
        "warning_count": sum(by_kind.values()),
        "finite_warning_count": sum(finite_by_kind.values()),
        "warning_count_by_crosscheck": by_kind,
        "finite_warning_count_by_crosscheck": finite_by_kind,
        "maximum_finite_measured_error_by_crosscheck": maximum_by_kind,
        "policy": "finite numerical reconstruction discrepancies are warning-only",
    }


def _trajectory_rows(path: Path) -> List[Dict[str, str]]:
    rows = _read_csv(path)
    if not rows:
        raise ArtifactValidationError("trajectory comparison input is empty")
    return rows


def _plot_policy_trajectories(
    baseline_root: Path, experiment_root: Path, output_path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = _trajectory_rows(
        baseline_root / "aggregate" / "data" / "keyframe_trajectories.csv"
    )
    experiment = _trajectory_rows(
        experiment_root / "aggregate" / "data" / "keyframe_trajectories.csv"
    )

    def points(
        rows: Iterable[Mapping[str, str]], prefix: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        rows = tuple(rows)
        return (
            np.asarray([float(row[prefix + "_x_m"]) for row in rows]),
            np.asarray([float(row[prefix + "_z_m"]) for row in rows]),
        )

    gt_x, gt_z = points(baseline, "ground_truth")
    pnp_x, pnp_z = points(baseline, "pnp")
    b_x, b_z = points(baseline, "bundle")
    c_x, c_z = points(experiment, "bundle")
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.plot(gt_x, gt_z, color="black", linewidth=2.0, label="Ground Truth")
    axis.plot(pnp_x, pnp_z, color="tab:orange", linewidth=1.3, label="PnP")
    axis.plot(
        b_x,
        b_z,
        color="tab:blue",
        linewidth=1.5,
        label="E5-I03B Bundle",
    )
    axis.plot(
        c_x,
        c_z,
        color="tab:green",
        linewidth=1.5,
        label="E5-I03C Bundle",
    )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_title("E5-I03B and E5-I03C trajectory comparison")
    axis.axis("equal")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _markdown_metrics(comparison: Mapping[str, object]) -> str:
    lines = [
        "# E5-I03B versus E5-I03C",
        "",
        (
            "Each policy is evaluated on its own keyframes and consecutive "
            "windows. Frame-by-frame pairing between policies is not claimed."
        ),
        "",
    ]
    policies = comparison["policies"]
    for metric_name, title in (
        ("absolute_localization_error_m", "Absolute localization error [m]"),
        (
            "consecutive_relative_translation_error_m",
            "Consecutive relative-translation error [m]",
        ),
        (
            "consecutive_relative_rotation_error_deg",
            "Consecutive relative-rotation error [deg]",
        ),
    ):
        lines.extend(
            [
                "## {}".format(title),
                "",
                (
                    "| Policy | Samples | Bundle mean | Bundle RMSE | "
                    "Bundle median | Bundle maximum | Bundle final | "
                    "Bundle lower than PnP |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for policy_name in ("E5-I03B", "E5-I03C"):
            metric = policies[policy_name][metric_name]
            bundle = metric["bundle"]
            lines.append(
                "| {} | {} | {:.12g} | {:.12g} | {:.12g} | {:.12g} | "
                "{:.12g} | {}/{} ({:.12g}%) |".format(
                    policy_name,
                    metric["sample_count"],
                    bundle["mean"],
                    bundle["rmse"],
                    bundle["median"],
                    bundle["maximum"],
                    bundle["final"],
                    metric["bundle_lower_count"],
                    metric["sample_count"],
                    metric["bundle_lower_percentage"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Policy status",
            "",
            (
                "No preferred policy is declared. Selection remains pending "
                "user review of the full-trajectory evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate_policy_comparison(
    *,
    experiment_root: Path = EXPERIMENT_OUTPUT_ROOT,
    baseline_root: Path = BASELINE_OUTPUT_ROOT,
) -> Dict[str, object]:
    experiment_root = Path(experiment_root).resolve()
    baseline_root = Path(baseline_root).resolve()
    if baseline_root != BASELINE_OUTPUT_ROOT.resolve():
        raise Exercise5ContractError("comparison baseline root differs")
    output_data = experiment_root / "aggregate" / "data"
    output_figures = experiment_root / "aggregate" / "figures"
    comparison = {
        "schema": "exercise5-keyframe-policy-comparison-v1",
        "milestone": "E5-I03C",
        "comparison_semantics": (
            "Policies are reported side-by-side on their own keyframes; "
            "no frame-by-frame paired comparison is claimed where FrameIds differ."
        ),
        "preferred_policy": None,
        "policy_selection_status": "pending-user-review",
        "policies": {
            "E5-I03B": _policy_metrics(baseline_root),
            "E5-I03C": _policy_metrics(experiment_root),
        },
        "experimental_covariance_warnings": covariance_warning_summary(
            experiment_root
        ),
        "artifact_roots": {
            "E5-I03B": str(baseline_root),
            "E5-I03C": str(experiment_root),
        },
    }
    json_path = output_data / COMPARISON_JSON_NAME
    markdown_path = output_data / COMPARISON_MARKDOWN_NAME
    figure_path = output_figures / COMPARISON_FIGURE_NAME
    _write_json(json_path, comparison)
    _write_text(markdown_path, _markdown_metrics(comparison))
    _plot_policy_trajectories(baseline_root, experiment_root, figure_path)
    return {
        "comparison": comparison,
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "figure_path": str(figure_path),
    }


def _validate_report_artifacts(
    source_root: Path, configuration: Mapping[str, object]
) -> None:
    for category, names in (
        ("data", configuration["artifact_contract"]["aggregate_data_files"]),
        (
            "figures",
            configuration["artifact_contract"]["aggregate_figure_files"],
        ),
    ):
        for name in names:
            path = source_root / "aggregate" / category / name
            if not path.is_file() or path.stat().st_size <= 0:
                raise ArtifactValidationError(
                    "required aggregate artifact is absent: {}".format(path)
                )
            if path.suffix == ".png" and path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ArtifactValidationError(
                    "aggregate PNG is unreadable: {}".format(path)
                )
    for path in (
        source_root / "aggregate" / "data" / COMPARISON_JSON_NAME,
        source_root / "aggregate" / "data" / COMPARISON_MARKDOWN_NAME,
        source_root / "aggregate" / "figures" / COMPARISON_FIGURE_NAME,
    ):
        if not path.is_file() or path.stat().st_size <= 0:
            raise ArtifactValidationError(
                "comparison artifact is absent: {}".format(path)
            )


def _write_prospective_checksum_index(
    source_root: Path,
    *,
    prospective_manifest_path: Path,
    output_path: Path,
) -> None:
    excluded = {
        (source_root / "manifest.json").resolve(),
        (source_root / "checksums.sha256").resolve(),
        prospective_manifest_path.resolve(),
        output_path.resolve(),
    }
    entries = []
    for item in sorted(source_root.rglob("*")):
        if (
            item.is_file()
            and item.resolve() not in excluded
            and "quarantine" not in item.parts
        ):
            entries.append(
                (str(item.relative_to(source_root)), sha256_file(item))
            )
    entries.append(("manifest.json", sha256_file(prospective_manifest_path)))
    with output_path.open("x", encoding="utf-8") as handle:
        for relative, digest in sorted(entries):
            handle.write("{}  {}\n".format(digest, relative))
        handle.flush()
        os.fsync(handle.fileno())


def verify_checksum_index(source_root: Path) -> Dict[str, object]:
    source_root = Path(source_root).resolve()
    checksum_path = source_root / "checksums.sha256"
    indexed: Dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or relative in indexed
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ArtifactValidationError("checksum index is malformed")
        indexed[relative] = digest
    observed = {
        str(path.relative_to(source_root))
        for path in source_root.rglob("*")
        if path.is_file()
        and path != checksum_path
        and "quarantine" not in path.parts
    }
    if set(indexed) != observed:
        raise ArtifactValidationError(
            "checksum file set differs: missing={}, extra={}".format(
                sorted(observed.difference(indexed)),
                sorted(set(indexed).difference(observed)),
            )
        )
    for relative, expected in indexed.items():
        observed_sha = sha256_file(source_root / relative)
        if observed_sha != expected:
            raise ArtifactValidationError(
                "checksum mismatch for {}".format(relative)
            )
    return {
        "checksum_path": str(checksum_path),
        "verified_file_count": len(indexed),
        "checksum_index_sha256": sha256_file(checksum_path),
    }


def finalize_experimental_run(
    *,
    source_root: Path = EXPERIMENT_OUTPUT_ROOT,
    configuration_path: Path = EXPERIMENT_CONFIG_PATH,
) -> Dict[str, object]:
    """Mark a fully reported experiment complete without a curated package."""

    source_root = Path(source_root).resolve()
    if source_root != EXPERIMENT_OUTPUT_ROOT.resolve():
        raise Exercise5ContractError("experimental source root differs")
    configuration, config_sha = load_experiment_configuration(
        configuration_path
    )
    manifest_path = source_root / "manifest.json"
    manifest = _read_json(manifest_path)
    expected = configuration["keyframe_selection"]["statistics"]["window_count"]
    if (
        manifest.get("scope") != EXPERIMENT_SCOPE
        or manifest.get("expected_window_count") != expected
        or manifest.get("completed_window_indices") != list(range(expected))
        or manifest.get("status")
        not in (
            "windows-complete-awaiting-aggregation",
            "complete-and-sealed",
        )
    ):
        raise ArtifactValidationError(
            "experimental source is partial or has the wrong scope"
        )
    validate_run_root(
        source_root,
        configuration_path=configuration_path,
        allow_partial=manifest["status"] != "complete-and-sealed",
    )
    _validate_report_artifacts(source_root, configuration)
    if manifest["status"] != "complete-and-sealed":
        token = uuid.uuid4().hex
        prospective_manifest = source_root / (
            ".manifest.finalize.{}.tmp".format(token)
        )
        prospective_checksums = source_root / (
            ".checksums.finalize.{}.tmp".format(token)
        )
        complete = build_run_manifest(
            configuration=configuration,
            config_sha256=config_sha,
            source_identity=manifest["source_input_identity"],
            completed_window_indices=list(range(expected)),
            status="complete-and-sealed",
            scope=EXPERIMENT_SCOPE,
            created_utc=utc_now(),
        )
        _write_json(prospective_manifest, complete, exclusive=True)
        _write_prospective_checksum_index(
            source_root,
            prospective_manifest_path=prospective_manifest,
            output_path=prospective_checksums,
        )
        old_manifest = manifest_path.read_bytes()
        old_checksum_path = source_root / "checksums.sha256"
        old_checksum = (
            old_checksum_path.read_bytes()
            if old_checksum_path.exists()
            else None
        )
        try:
            os.replace(str(prospective_checksums), str(old_checksum_path))
            os.replace(str(prospective_manifest), str(manifest_path))
        except BaseException:
            with manifest_path.open("wb") as handle:
                handle.write(old_manifest)
                handle.flush()
                os.fsync(handle.fileno())
            if old_checksum is None:
                old_checksum_path.unlink(missing_ok=True)
            else:
                with old_checksum_path.open("wb") as handle:
                    handle.write(old_checksum)
                    handle.flush()
                    os.fsync(handle.fileno())
            prospective_manifest.unlink(missing_ok=True)
            prospective_checksums.unlink(missing_ok=True)
            raise
    checksum_validation = verify_checksum_index(source_root)
    strict_validation = validate_run_root(
        source_root,
        configuration_path=configuration_path,
        allow_partial=False,
    )
    return {
        "status": "complete-and-sealed",
        "source_root": str(source_root),
        "window_count": expected,
        "checksum_validation": checksum_validation,
        "strict_source_validation": strict_validation,
    }


__all__ = [
    "BASELINE_OUTPUT_ROOT",
    "COMPARISON_FIGURE_NAME",
    "COMPARISON_JSON_NAME",
    "COMPARISON_MARKDOWN_NAME",
    "EXPERIMENT_CONFIG_PATH",
    "EXPERIMENT_CONFIRMATION",
    "EXPERIMENT_OUTPUT_ROOT",
    "EXPERIMENT_QUANTILE",
    "EXPERIMENT_RULE_ID",
    "EXPERIMENT_SCOPE",
    "covariance_warning_summary",
    "derive_and_persist_experiment_configuration",
    "derive_track_survival_keyframes",
    "finalize_experimental_run",
    "generate_policy_comparison",
    "load_experiment_configuration",
    "select_track_survival_endpoint",
    "validate_track_survival_keyframes",
    "verify_checksum_index",
]
