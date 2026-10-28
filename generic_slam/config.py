"""Validated scientific defaults and lightweight experiment overrides."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import ContractError


SCIENTIFIC_IMPLEMENTATION_REVISION = (
    "transactional-vo-refinement-loop-gauge-tolerance-v2-detector-factory"
)


FEATURE_DETECTORS = ("akaze", "sift")
TEMPORAL_MATCHING_POPULATIONS = ("all_features", "stereo_rows")


DEFAULT_ALGORITHM_CONFIG: Dict[str, Any] = {
    "frontend": {
        # "akaze": binary descriptors, exact brute-force Hamming matching.
        # "sift": float descriptors, FLANN KD-tree matching (settings below).
        "detector": "akaze",
        "akaze": {
            # Below OpenCV's 0.001 default: on KITTI 00 this gives more
            # geometrically valid stereo matches than SIFT at ~2.5x the speed.
            "threshold": 0.0002,
        },
        "sift": {
            "contrast_threshold": 0.02,
            "edge_threshold": 10,
        },
        "flann": {
            "algorithm": 1,
            "trees": 5,
            "checks": 128,
        },
        "stereo": {
            "maximum_vertical_deviation_px": 2.0,
            "require_positive_disparity": True,
        },
        "temporal_matching": {
            # Which left features tracking matches between consecutive frames.
            # "stereo_rows": only stereo-inlier features, the only ones the
            # four-view consensus can use; ~3x cheaper as matching is quadratic.
            # On KITTI 00: tracking 21.2 -> 15.5 min, +1.1% supporters, RMSE
            # within the RANSAC seed noise (2 seeds; configs/experiments).
            # "all_features": every detected feature (the original behaviour).
            "population": "stereo_rows",
        },
    },
    "visual_odometry": {
        "seed": 0,
        "ransac": {
            "success_probability": 0.99,
            "initial_outlier_probability": 0.99,
            "sample_size": 4,
            "supporter_max_distance_px": 2.0,
            "maximum_hypothesis_attempts": None,
        },
    },
    "keyframes": {
        "minimum_endpoint_delta_frames": 4,
        "maximum_endpoint_delta_frames": 19,
        "path_distance_threshold_m": 12.0,
        "relative_rotation_threshold_deg": 6.0,
    },
    "bundle": {
        "stereo_sigma_px": [1.0, 1.0, 1.0],
        "prior_sigmas": [
            0.00030461741978670857,
            0.00030461741978670857,
            0.00030461741978670857,
            0.1,
            0.01,
            1.0,
        ],
        "minimum_valid_observations": 2,
        "minimum_distinct_frames": 2,
        "maximum_initialization_depth_m": 300.0,
        "rotation_atol": 1e-9,
        "covariance_symmetry_atol": 1e-9,
        "covariance_minimum_eigenvalue": -1e-9,
        "graph_error_nonincrease_absolute": 1e-9,
        "graph_error_nonincrease_relative": 1e-12,
    },
    "loop_closure": {
        "mahalanobis_squared_threshold": 36_000_000.0,
        "minimum_temporal_ordinal_gap": 16,
        "maximum_matcher_invocations_per_current": 3,
        "minimum_consensus_count": 30,
        "minimum_supporter_count": 30,
        "minimum_supporter_percentage": 30.0,
        "previous_ordinal_cluster_radius": 10,
        "current_ordinal_cluster_radius": 10,
        "pair_seed": 0,
        "ransac": {
            "success_probability": 0.99,
            "initial_outlier_probability": 0.99,
            "sample_size": 4,
            "supporter_max_distance_px": 2.0,
            # A pair passes only with >= 30% supporters; at that inlier ratio
            # log(0.01) / log(1 - 0.3**4) ~ 566 attempts give 99% confidence.
            # Experiment A on KITTI 00: identical loops, loop closure 42% faster.
            "maximum_hypothesis_attempts": 600,
        },
    },
    "analysis": {
        "relative_subsection_lengths_frames": [100, 400, 800],
        "low_connectivity_threshold": 300,
        "covariance_display_stride": 12,
    },
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _merge_known(
    baseline: Mapping[str, Any], override: Mapping[str, Any], prefix: str = ""
) -> Dict[str, Any]:
    result = copy.deepcopy(dict(baseline))
    for key, value in override.items():
        path = "{}.{}".format(prefix, key) if prefix else str(key)
        if key not in baseline:
            raise ContractError("unknown algorithm configuration field: {}".format(path))
        expected = baseline[key]
        if isinstance(expected, Mapping):
            if not isinstance(value, Mapping):
                raise ContractError("configuration object required at {}".format(path))
            result[key] = _merge_known(expected, value, path)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _number(value: Any, path: str, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("finite number required at {}".format(path))
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ContractError("finite number required at {}".format(path))
    if minimum is not None and normalized < minimum:
        raise ContractError("{} must be at least {}".format(path, minimum))
    return normalized


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError("positive integer required at {}".format(path))
    return int(value)


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("nonnegative integer required at {}".format(path))
    return int(value)


def _probability(value: Any, path: str) -> float:
    normalized = _number(value, path)
    if not 0.0 < normalized < 1.0:
        raise ContractError("{} must be strictly between zero and one".format(path))
    return normalized


def _numeric_vector(
    value: Any, path: str, length: int, *, strictly_positive: bool
) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ContractError("{} must be a {}-element array".format(path, length))
    normalized = tuple(_number(item, path) for item in value)
    if strictly_positive and any(item <= 0.0 for item in normalized):
        raise ContractError("{} values must be strictly positive".format(path))
    return normalized


def validate_algorithm_config(config: Mapping[str, Any]) -> None:
    """Validate values and cross-field invariants without adding run properties."""

    frontend = config["frontend"]
    if frontend["detector"] not in FEATURE_DETECTORS:
        raise ContractError(
            "frontend.detector must be one of {}".format(", ".join(FEATURE_DETECTORS))
        )
    _number(frontend["akaze"]["threshold"], "frontend.akaze.threshold", minimum=0.0)
    _number(frontend["sift"]["contrast_threshold"], "frontend.sift.contrast_threshold", minimum=0.0)
    _number(frontend["sift"]["edge_threshold"], "frontend.sift.edge_threshold", minimum=0.0)
    _nonnegative_integer(frontend["flann"]["algorithm"], "frontend.flann.algorithm")
    _positive_integer(frontend["flann"]["trees"], "frontend.flann.trees")
    _positive_integer(frontend["flann"]["checks"], "frontend.flann.checks")
    _number(
        frontend["stereo"]["maximum_vertical_deviation_px"],
        "frontend.stereo.maximum_vertical_deviation_px",
        minimum=0.0,
    )
    if not isinstance(frontend["stereo"]["require_positive_disparity"], bool):
        raise ContractError("frontend.stereo.require_positive_disparity must be boolean")
    if frontend["temporal_matching"]["population"] not in TEMPORAL_MATCHING_POPULATIONS:
        raise ContractError(
            "frontend.temporal_matching.population must be one of {}".format(
                ", ".join(TEMPORAL_MATCHING_POPULATIONS)
            )
        )

    vo = config["visual_odometry"]
    _nonnegative_integer(vo["seed"], "visual_odometry.seed")
    for prefix, ransac in (
        ("visual_odometry.ransac", vo["ransac"]),
        ("loop_closure.ransac", config["loop_closure"]["ransac"]),
    ):
        _probability(ransac["success_probability"], prefix + ".success_probability")
        _probability(
            ransac["initial_outlier_probability"],
            prefix + ".initial_outlier_probability",
        )
        _positive_integer(ransac["sample_size"], prefix + ".sample_size")
        _number(
            ransac["supporter_max_distance_px"],
            prefix + ".supporter_max_distance_px",
            minimum=0.0,
        )
        maximum = ransac["maximum_hypothesis_attempts"]
        if maximum is not None:
            _positive_integer(maximum, prefix + ".maximum_hypothesis_attempts")

    keyframes = config["keyframes"]
    minimum_delta = _positive_integer(
        keyframes["minimum_endpoint_delta_frames"],
        "keyframes.minimum_endpoint_delta_frames",
    )
    maximum_delta = _positive_integer(
        keyframes["maximum_endpoint_delta_frames"],
        "keyframes.maximum_endpoint_delta_frames",
    )
    if minimum_delta > maximum_delta:
        raise ContractError("keyframe minimum endpoint delta exceeds maximum")
    _number(
        keyframes["path_distance_threshold_m"],
        "keyframes.path_distance_threshold_m",
        minimum=0.0,
    )
    _number(
        keyframes["relative_rotation_threshold_deg"],
        "keyframes.relative_rotation_threshold_deg",
        minimum=0.0,
    )

    bundle = config["bundle"]
    _numeric_vector(bundle["stereo_sigma_px"], "bundle.stereo_sigma_px", 3, strictly_positive=True)
    _numeric_vector(bundle["prior_sigmas"], "bundle.prior_sigmas", 6, strictly_positive=True)
    _positive_integer(bundle["minimum_valid_observations"], "bundle.minimum_valid_observations")
    _positive_integer(bundle["minimum_distinct_frames"], "bundle.minimum_distinct_frames")
    _number(
        bundle["maximum_initialization_depth_m"],
        "bundle.maximum_initialization_depth_m",
        minimum=0.0,
    )
    for name in (
        "rotation_atol",
        "covariance_symmetry_atol",
        "graph_error_nonincrease_absolute",
        "graph_error_nonincrease_relative",
    ):
        _number(bundle[name], "bundle." + name, minimum=0.0)
    _number(bundle["covariance_minimum_eigenvalue"], "bundle.covariance_minimum_eigenvalue")

    loop = config["loop_closure"]
    _number(loop["mahalanobis_squared_threshold"], "loop_closure.mahalanobis_squared_threshold", minimum=0.0)
    for name in (
        "minimum_temporal_ordinal_gap",
        "maximum_matcher_invocations_per_current",
        "minimum_consensus_count",
        "minimum_supporter_count",
    ):
        _positive_integer(loop[name], "loop_closure." + name)
    _number(loop["minimum_supporter_percentage"], "loop_closure.minimum_supporter_percentage", minimum=0.0)
    for name in ("previous_ordinal_cluster_radius", "current_ordinal_cluster_radius", "pair_seed"):
        _nonnegative_integer(loop[name], "loop_closure." + name)

    analysis = config["analysis"]
    lengths = analysis["relative_subsection_lengths_frames"]
    if not isinstance(lengths, list) or not lengths:
        raise ContractError("analysis.relative_subsection_lengths_frames must be nonempty")
    normalized_lengths = tuple(
        _positive_integer(value, "analysis.relative_subsection_lengths_frames")
        for value in lengths
    )
    if tuple(sorted(set(normalized_lengths))) != normalized_lengths:
        raise ContractError("analysis relative subsection lengths must be unique and increasing")
    _nonnegative_integer(analysis["low_connectivity_threshold"], "analysis.low_connectivity_threshold")
    _positive_integer(analysis["covariance_display_stride"], "analysis.covariance_display_stride")


# Configuration sections each stage reads, and the stages it consumes. A
# stage's fingerprint covers only these, so changing e.g. loop_closure keeps
# tracking/keyframes/bundle/pose_graph results valid and reusable.
STAGE_CONFIG_SECTIONS: Dict[str, Tuple[str, ...]] = {
    "dataset": (),
    "tracking": ("frontend", "visual_odometry"),
    "keyframes": ("keyframes",),
    "bundle": ("bundle",),
    "pose_graph": ("bundle",),
    "loop_closure": ("loop_closure", "frontend", "bundle"),
    "analysis": ("analysis",),
    "report": ("analysis", "frontend"),
}
STAGE_UPSTREAM: Dict[str, Tuple[str, ...]] = {
    "dataset": (),
    "tracking": ("dataset",),
    "keyframes": ("tracking",),
    "bundle": ("tracking", "keyframes"),
    "pose_graph": ("bundle",),
    "loop_closure": ("pose_graph",),
    "analysis": ("dataset", "tracking", "keyframes", "bundle", "pose_graph", "loop_closure"),
    "report": ("analysis",),
}


@dataclass(frozen=True)
class AlgorithmConfig:
    """A fully resolved and validated algorithm configuration."""

    values: Mapping[str, Any]
    fingerprint: str
    mode: str

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, Mapping):
            raise ContractError("missing configuration section: {}".format(name))
        return value

    def stage_fingerprint(self, stage: str) -> str:
        """Fingerprint of the configuration a stage and its upstream stages read."""

        if stage not in STAGE_CONFIG_SECTIONS:
            raise ContractError("unknown pipeline stage: {}".format(stage))
        payload = {
            "stage": stage,
            "sections": {name: self.values[name] for name in STAGE_CONFIG_SECTIONS[stage]},
            "upstream": {name: self.stage_fingerprint(name) for name in STAGE_UPSTREAM[stage]},
        }
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def resolve_algorithm_config(path: Optional[Path] = None) -> AlgorithmConfig:
    """Resolve an optional partial JSON override over validated defaults."""

    resolved: Dict[str, Any] = copy.deepcopy(DEFAULT_ALGORITHM_CONFIG)
    mode = "validated-default"
    if path is not None:
        selected = Path(path)
        try:
            raw = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ContractError("cannot load algorithm configuration: {}".format(error))
        if not isinstance(raw, Mapping):
            raise ContractError("algorithm configuration root must be a JSON object")
        resolved = _merge_known(resolved, raw)
        mode = "experiment"
    validate_algorithm_config(resolved)
    fingerprint = hashlib.sha256(_canonical_json_bytes(resolved)).hexdigest()
    return AlgorithmConfig(values=resolved, fingerprint=fingerprint, mode=mode)


def load_resolved_config(path: Path, mode: str) -> AlgorithmConfig:
    """Reload the canonical run configuration used by child stage processes."""

    if mode not in ("validated-default", "experiment"):
        raise ContractError("invalid run configuration mode")
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ContractError("cannot load resolved configuration: {}".format(error))
    if not isinstance(raw, Mapping):
        raise ContractError("resolved configuration root must be a JSON object")
    # Re-merge against the schema to reject missing as well as unknown fields.
    resolved = _merge_known(DEFAULT_ALGORITHM_CONFIG, raw)
    if resolved != raw:
        raise ContractError("resolved configuration is incomplete")
    validate_algorithm_config(resolved)
    fingerprint = hashlib.sha256(_canonical_json_bytes(resolved)).hexdigest()
    return AlgorithmConfig(values=resolved, fingerprint=fingerprint, mode=mode)


def write_resolved_config(config: AlgorithmConfig, path: Path) -> None:
    from .artifacts.io import write_json_atomic

    write_json_atomic(Path(path), config.values)


__all__ = [
    "AlgorithmConfig",
    "DEFAULT_ALGORITHM_CONFIG",
    "FEATURE_DETECTORS",
    "TEMPORAL_MATCHING_POPULATIONS",
    "SCIENTIFIC_IMPLEMENTATION_REVISION",
    "STAGE_CONFIG_SECTIONS",
    "STAGE_UPSTREAM",
    "load_resolved_config",
    "resolve_algorithm_config",
    "validate_algorithm_config",
    "write_resolved_config",
]
