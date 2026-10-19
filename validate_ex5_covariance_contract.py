"""Bounded E5-I04 validation of the downstream BetweenFactorPose3 contract.

This module is read-only with respect to the protected E5-I03B Bundle
artifacts.  It does not load TrackingDB, reconstruct a Bundle graph, solve a
Bundle window, or construct a Pose Graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from exercise5_full_run import (
    DEFAULT_CONFIG_PATH,
    HASH_SCHEMA,
    MANIFEST_SCHEMA,
    ROOT,
    TANGENT_ORDER,
    load_frozen_configuration,
    sha256_file,
)


CANONICAL_ARTIFACT_ROOT = ROOT / "diagnostics" / "ex5_final"
CURATED_ARTIFACT_ROOT = ROOT / "project_submission" / "bundle"
DEFAULT_EVIDENCE_ROOT = ROOT / "diagnostics" / "ex5_i04"
EDGE_DIRECTION = "T_Cstart_from_Cend"
ENDPOINT_EDGE_SCHEMA = "exercise5-endpoint-edge-v1"
SUMMARY_SCHEMA = "exercise5-i04-covariance-contract-validation-v1"


class E5I04ValidationError(RuntimeError):
    """Raised when the bounded downstream covariance contract does not hold."""


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_metadata_digest(root: Path) -> Dict[str, object]:
    """Fingerprint names, sizes, and mtimes without rehashing protected data."""

    root = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        record = "{}\0{}\0{}\n".format(
            path.relative_to(root), stat.st_size, stat.st_mtime_ns
        )
        digest.update(record.encode("utf-8"))
        count += 1
        total_size += int(stat.st_size)
    return {
        "root": str(root),
        "file_count": count,
        "total_size_bytes": total_size,
        "metadata_sha256": digest.hexdigest(),
    }


def _values_with_poses(
    start_key: int,
    end_key: int,
    start_pose: gtsam.Pose3,
    end_pose: gtsam.Pose3,
) -> gtsam.Values:
    values = gtsam.Values()
    values.insert(start_key, start_pose)
    values.insert(end_key, end_pose)
    return values


def validate_synthetic_gtsam_contract(
    *, numerical_atol: float
) -> Dict[str, object]:
    """Run one nontrivial six-coordinate GTSAM tangent/factor check."""

    start_pose = gtsam.Pose3(
        gtsam.Rot3.RzRyRx(0.23, -0.17, 0.31),
        np.asarray([1.2, -0.7, 2.4], dtype=float),
    )
    end_pose = gtsam.Pose3(
        gtsam.Rot3.RzRyRx(-0.19, 0.27, -0.11),
        np.asarray([-0.4, 1.8, 4.1], dtype=float),
    )
    measurement = start_pose.between(end_pose)
    covariance_square_root = np.asarray(
        [
            [0.20, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.03, 0.22, 0.00, 0.00, 0.00, 0.00],
            [0.00, -0.02, 0.24, 0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00, 0.80, 0.00, 0.00],
            [0.00, -0.04, 0.00, 0.00, 0.90, 0.00],
            [0.00, 0.00, 0.06, 0.00, 0.00, 1.00],
        ],
        dtype=float,
    )
    covariance = covariance_square_root @ covariance_square_root.T
    noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)
    start_key = int(gtsam.symbol("c", 0))
    end_key = int(gtsam.symbol("c", 1))
    factor = gtsam.BetweenFactorPose3(
        start_key, end_key, measurement, noise_model
    )
    nominal_values = _values_with_poses(
        start_key, end_key, start_pose, end_pose
    )
    nominal_residual = np.asarray(
        factor.unwhitenedError(nominal_values), dtype=float
    ).reshape(6)
    nominal_error = float(factor.error(nominal_values))

    coordinate_step = 1e-4
    expmap_coordinate_response = np.empty((6, 6), dtype=float)
    for coordinate in range(6):
        delta = np.zeros(6, dtype=float)
        delta[coordinate] = coordinate_step
        pose = gtsam.Pose3.Expmap(delta)
        expmap_coordinate_response[:3, coordinate] = (
            np.asarray(gtsam.Rot3.Logmap(pose.rotation()), dtype=float)
            / coordinate_step
        )
        expmap_coordinate_response[3:, coordinate] = (
            np.asarray(pose.translation(), dtype=float).reshape(3)
            / coordinate_step
        )

    finite_difference_step = 1e-5
    end_local_jacobian = np.empty((6, 6), dtype=float)
    for coordinate in range(6):
        delta = np.zeros(6, dtype=float)
        delta[coordinate] = finite_difference_step
        plus_values = _values_with_poses(
            start_key, end_key, start_pose, end_pose.retract(delta)
        )
        minus_values = _values_with_poses(
            start_key, end_key, start_pose, end_pose.retract(-delta)
        )
        plus = np.asarray(
            factor.unwhitenedError(plus_values), dtype=float
        ).reshape(6)
        minus = np.asarray(
            factor.unwhitenedError(minus_values), dtype=float
        ).reshape(6)
        end_local_jacobian[:, coordinate] = (
            plus - minus
        ) / (2.0 * finite_difference_step)

    perturbation = np.asarray(
        [2e-4, -3e-4, 1e-4, 4e-4, -2e-4, 3e-4], dtype=float
    )
    perturbed_values = _values_with_poses(
        start_key, end_key, start_pose, end_pose.retract(perturbation)
    )
    perturbed_residual = np.asarray(
        factor.unwhitenedError(perturbed_values), dtype=float
    ).reshape(6)
    observed_error = float(factor.error(perturbed_values))
    expected_error = float(
        0.5
        * perturbed_residual
        @ np.linalg.solve(covariance, perturbed_residual)
    )

    identity = np.eye(6)
    ordering_error = float(
        np.max(np.abs(expmap_coordinate_response - identity))
    )
    jacobian_error = float(
        np.max(np.abs(end_local_jacobian - identity))
    )
    gaussian_error_difference = abs(observed_error - expected_error)
    nominal_residual_max_abs = float(np.max(np.abs(nominal_residual)))
    if (
        nominal_residual_max_abs > numerical_atol
        or nominal_error > numerical_atol
        or ordering_error > numerical_atol
        or jacobian_error > numerical_atol
        or gaussian_error_difference > numerical_atol
    ):
        raise E5I04ValidationError(
            "synthetic GTSAM tangent/factor contract differs"
        )
    return {
        "status": "pass",
        "pose_pair": "deterministic-nontrivial",
        "measurement": "start_pose.between(end_pose)",
        "tangent_order": list(TANGENT_ORDER),
        "nominal_residual_max_abs": nominal_residual_max_abs,
        "nominal_factor_error": nominal_error,
        "expmap_coordinate_response": expmap_coordinate_response.tolist(),
        "expmap_ordering_max_abs_error": ordering_error,
        "finite_difference_step": finite_difference_step,
        "end_local_residual_jacobian": end_local_jacobian.tolist(),
        "end_local_jacobian_identity_max_abs_error": jacobian_error,
        "perturbed_factor_error": observed_error,
        "direct_covariance_mahalanobis_error": expected_error,
        "factor_vs_direct_covariance_error_abs": (
            gaussian_error_difference
        ),
        "direct_covariance_use_passed": True,
        "adjoint_transport_required": False,
    }


def _validate_small_sealed_file(
    directory: Path,
    indexed_artifacts: Mapping[str, Mapping[str, object]],
    filename: str,
) -> Path:
    if filename not in indexed_artifacts:
        raise E5I04ValidationError(
            "{} is absent from the existing window seal".format(filename)
        )
    path = directory / filename
    entry = indexed_artifacts[filename]
    if (
        not path.is_file()
        or int(path.stat().st_size) != entry.get("size_bytes")
        or sha256_file(path) != entry.get("sha256")
    ):
        raise E5I04ValidationError(
            "{} differs from the existing window seal".format(path)
        )
    return path


def _load_endpoint_artifacts(
    directory: Path,
    *,
    expected_index: int,
    expected_start: int,
    expected_end: int,
    configuration_sha256: str,
    source_identity_sha256: str,
    include_window_zero_values: bool = False,
) -> Dict[str, object]:
    """Read only the sealed endpoint artifacts needed by E5-I04."""

    seal = _read_json(directory / "artifact_hashes.json")
    if (
        seal.get("schema") != HASH_SCHEMA
        or seal.get("schema_version") != 1
        or seal.get("status") != "sealed"
        or seal.get("window_index") != expected_index
        or seal.get("start_frame_id") != expected_start
        or seal.get("end_frame_id") != expected_end
        or seal.get("configuration_sha256") != configuration_sha256
        or seal.get("source_input_identity_sha256")
        != source_identity_sha256
    ):
        raise E5I04ValidationError(
            "window {} seal identity differs".format(expected_index)
        )
    indexed = {
        entry["path"]: entry for entry in seal.get("artifacts", [])
    }
    edge_path = _validate_small_sealed_file(
        directory, indexed, "endpoint_edge.json"
    )
    covariance_path = _validate_small_sealed_file(
        directory, indexed, "endpoint_covariances.npz"
    )
    edge = _read_json(edge_path)
    with np.load(covariance_path, allow_pickle=False) as data:
        expected_shapes = {
            "joint_covariance": (12, 12),
            "joint_information": (12, 12),
            "conditional_end_given_start_covariance": (6, 6),
        }
        if set(data.files) != set(expected_shapes):
            raise E5I04ValidationError(
                "window {} endpoint covariance members differ".format(
                    expected_index
                )
            )
        covariance_arrays = {
            name: np.asarray(data[name], dtype=float).copy()
            for name in expected_shapes
        }
    for name, shape in expected_shapes.items():
        if covariance_arrays[name].shape != shape:
            raise E5I04ValidationError(
                "window {} {} shape differs".format(expected_index, name)
            )

    result = {
        "seal": seal,
        "edge": edge,
        "covariances": covariance_arrays,
    }
    if include_window_zero_values:
        summary_path = _validate_small_sealed_file(
            directory, indexed, "summary.json"
        )
        values_path = _validate_small_sealed_file(
            directory, indexed, "values.npz"
        )
        result["summary"] = _read_json(summary_path)
        with np.load(values_path, allow_pickle=False) as values:
            if "final_camera_poses" not in values.files:
                raise E5I04ValidationError(
                    "window 0 final camera poses are absent"
                )
            result["final_camera_poses"] = np.asarray(
                values["final_camera_poses"], dtype=float
            ).copy()
    return result


def _validate_manifest_and_retained_reload_evidence(
    source_root: Path,
    curated_root: Path,
    *,
    configuration_sha256: str,
    expected_window_count: int,
) -> Tuple[Mapping[str, object], str]:
    manifest_path = source_root / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_indices = list(range(expected_window_count))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete-and-sealed"
        or manifest.get("scope") != "canonical"
        or manifest.get("configuration_sha256") != configuration_sha256
        or manifest.get("expected_window_count") != expected_window_count
        or manifest.get("completed_window_indices") != expected_indices
    ):
        raise E5I04ValidationError(
            "canonical E5-I03B manifest is not complete and sealed"
        )
    manifest_sha256 = sha256_file(manifest_path)
    curated_manifest = _read_json(curated_root / "manifest.json")
    retained = curated_manifest.get("source_validation", {})
    if (
        curated_manifest.get("source_manifest_sha256") != manifest_sha256
        or retained.get("status") != "complete-and-sealed"
        or retained.get("configuration_sha256") != configuration_sha256
        or retained.get("validated_window_indices") != expected_indices
    ):
        raise E5I04ValidationError(
            "retained complete E5-I03B reload validation differs"
        )
    return manifest, manifest_sha256


def validate_canonical_window_zero(
    loaded: Mapping[str, object],
    *,
    numerical_atol: float,
) -> Dict[str, object]:
    edge = loaded["edge"]
    summary = loaded["summary"]
    poses = np.asarray(loaded["final_camera_poses"], dtype=float)
    covariance = np.asarray(
        loaded["covariances"][
            "conditional_end_given_start_covariance"
        ],
        dtype=float,
    )
    if (
        edge.get("schema") != ENDPOINT_EDGE_SCHEMA
        or edge.get("direction") != EDGE_DIRECTION
        or edge.get("start_frame_id") != 0
        or edge.get("end_frame_id") != 7
        or summary.get("tangent_order") != list(TANGENT_ORDER)
        or poses.ndim != 3
        or poses.shape[1:] != (4, 4)
    ):
        raise E5I04ValidationError("canonical window 0 contract differs")
    start_pose = gtsam.Pose3(poses[0])
    end_pose = gtsam.Pose3(poses[-1])
    saved_measurement = gtsam.Pose3(
        np.asarray(edge["matrix"], dtype=float)
    )
    reproduced = start_pose.between(end_pose)
    edge_error = float(
        np.max(np.abs(reproduced.matrix() - saved_measurement.matrix()))
    )
    noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)
    start_key = int(edge["start_camera_key"])
    end_key = int(edge["end_camera_key"])
    factor = gtsam.BetweenFactorPose3(
        start_key, end_key, saved_measurement, noise_model
    )
    values = _values_with_poses(
        start_key, end_key, start_pose, end_pose
    )
    residual = np.asarray(
        factor.unwhitenedError(values), dtype=float
    ).reshape(6)
    residual_max_abs = float(np.max(np.abs(residual)))
    factor_error = float(factor.error(values))
    if (
        edge_error > numerical_atol
        or residual_max_abs > numerical_atol
        or factor_error > numerical_atol
    ):
        raise E5I04ValidationError(
            "canonical window 0 BetweenFactor contract differs"
        )
    return {
        "status": "pass",
        "window_index": 0,
        "start_frame_id": 0,
        "end_frame_id": 7,
        "edge_direction": EDGE_DIRECTION,
        "start_between_end_edge_max_abs_error": edge_error,
        "conditional_covariance_shape": list(covariance.shape),
        "tangent_order": list(TANGENT_ORDER),
        "gaussian_covariance_model_constructed": True,
        "between_factor_residual_max_abs": residual_max_abs,
        "between_factor_error": factor_error,
        "direct_covariance_use_passed": True,
    }


def validate_complete_population(
    source_root: Path,
    configuration: Mapping[str, object],
    *,
    configuration_sha256: str,
    source_identity_sha256: str,
    symmetry_atol: float,
) -> Dict[str, object]:
    keyframes = configuration["keyframe_selection"][
        "ordered_keyframe_ids"
    ]
    expected_count = len(keyframes) - 1
    expected_names = [
        "window_{:04d}_{}_{}".format(
            index, keyframes[index], keyframes[index + 1]
        )
        for index in range(expected_count)
    ]
    observed_names = sorted(
        path.name
        for path in (source_root / "windows").glob("window_[0-9]*")
        if path.is_dir()
    )
    if observed_names != sorted(expected_names):
        raise E5I04ValidationError(
            "canonical window directory population differs"
        )

    maximum_symmetry_error = 0.0
    previous_end: Optional[int] = None
    window_zero_loaded = None
    for index, name in enumerate(expected_names):
        start = int(keyframes[index])
        end = int(keyframes[index + 1])
        loaded = _load_endpoint_artifacts(
            source_root / "windows" / name,
            expected_index=index,
            expected_start=start,
            expected_end=end,
            configuration_sha256=configuration_sha256,
            source_identity_sha256=source_identity_sha256,
            include_window_zero_values=index == 0,
        )
        edge = loaded["edge"]
        edge_matrix = np.asarray(edge.get("matrix"), dtype=float)
        if (
            edge.get("schema") != ENDPOINT_EDGE_SCHEMA
            or edge.get("start_frame_id") != start
            or edge.get("end_frame_id") != end
            or edge.get("direction") != EDGE_DIRECTION
            or edge_matrix.shape != (4, 4)
            or not np.all(np.isfinite(edge_matrix))
            or (previous_end is not None and start != previous_end)
        ):
            raise E5I04ValidationError(
                "window {} endpoint identity/direction differs".format(index)
            )
        previous_end = end
        covariance_arrays = loaded["covariances"]
        for matrix in covariance_arrays.values():
            if not np.all(np.isfinite(matrix)):
                raise E5I04ValidationError(
                    "window {} covariance evidence is nonfinite".format(
                        index
                    )
                )
        conditional = covariance_arrays[
            "conditional_end_given_start_covariance"
        ]
        symmetry_error = float(
            np.max(np.abs(conditional - conditional.T))
        )
        maximum_symmetry_error = max(
            maximum_symmetry_error, symmetry_error
        )
        if symmetry_error > symmetry_atol:
            raise E5I04ValidationError(
                "window {} conditional covariance is asymmetric".format(
                    index
                )
            )
        gtsam.noiseModel.Gaussian.Covariance(conditional)
        if index == 0:
            window_zero_loaded = loaded
    if previous_end != keyframes[-1] or window_zero_loaded is None:
        raise E5I04ValidationError("canonical endpoint coverage differs")
    return {
        "status": "pass",
        "canonical_window_count_checked": expected_count,
        "first_window_index": 0,
        "last_window_index": expected_count - 1,
        "first_frame_id": int(keyframes[0]),
        "last_frame_id": int(keyframes[-1]),
        "consecutive_endpoint_identities_passed": True,
        "edge_direction": EDGE_DIRECTION,
        "expected_matrix_shapes_passed": True,
        "all_conditional_covariances_finite": True,
        "all_conditional_covariances_symmetric": True,
        "maximum_conditional_covariance_symmetry_max_abs": (
            maximum_symmetry_error
        ),
        "gaussian_covariance_models_constructed": expected_count,
        "missing_canonical_windows": 0,
        "_window_zero_loaded": window_zero_loaded,
    }


def run_validation(
    *,
    source_root: Path = CANONICAL_ARTIFACT_ROOT,
    curated_root: Path = CURATED_ARTIFACT_ROOT,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, object]:
    """Validate E5-I04 and write only its compact summary and report."""

    source_root = Path(source_root).resolve()
    curated_root = Path(curated_root).resolve()
    evidence_root = Path(evidence_root).resolve()
    protected_before = {
        "canonical_e5_i03b": _tree_metadata_digest(source_root),
        "curated_bundle": _tree_metadata_digest(curated_root),
    }
    configuration, configuration_sha256 = load_frozen_configuration(
        configuration_path
    )
    expected_count = configuration["keyframe_selection"]["statistics"][
        "window_count"
    ]
    manifest, manifest_sha256 = (
        _validate_manifest_and_retained_reload_evidence(
            source_root,
            curated_root,
            configuration_sha256=configuration_sha256,
            expected_window_count=expected_count,
        )
    )
    tolerance = configuration["scientific_configuration"][
        "validation_tolerances"
    ]
    synthetic = validate_synthetic_gtsam_contract(
        numerical_atol=tolerance["numerical_reconstruction_atol"]
    )
    population = validate_complete_population(
        source_root,
        configuration,
        configuration_sha256=configuration_sha256,
        source_identity_sha256=manifest[
            "source_input_identity_sha256"
        ],
        symmetry_atol=tolerance["covariance_symmetry_atol"],
    )
    window_zero = validate_canonical_window_zero(
        population.pop("_window_zero_loaded"),
        numerical_atol=tolerance["numerical_reconstruction_atol"],
    )
    protected_after = {
        "canonical_e5_i03b": _tree_metadata_digest(source_root),
        "curated_bundle": _tree_metadata_digest(curated_root),
    }
    protected_unchanged = protected_before == protected_after
    if not protected_unchanged:
        raise E5I04ValidationError(
            "protected E5-I03B artifact metadata changed during validation"
        )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "pass",
        "milestone": "E5-I04",
        "canonical_artifact_root": str(source_root),
        "canonical_manifest_sha256": manifest_sha256,
        "configuration_path": str(Path(configuration_path).resolve()),
        "configuration_sha256": configuration_sha256,
        "verified_edge_direction": EDGE_DIRECTION,
        "pose3_tangent_order": list(TANGENT_ORDER),
        "direct_conditional_covariance_use_passed": True,
        "additional_adjoint_transport_required": False,
        "covariance_scaling_applied": False,
        "covariance_regularization_or_replacement_applied": False,
        "canonical_windows_checked": population[
            "canonical_window_count_checked"
        ],
        "synthetic_gtsam_contract": synthetic,
        "canonical_window_zero": window_zero,
        "complete_population": population,
        "execution_boundaries": {
            "tracking_database_loaded": False,
            "bundle_graph_loaded": False,
            "bundle_window_solved": False,
            "pose_graph_constructed": False,
            "e5_i03c_artifacts_used": False,
        },
        "retained_existing_validation_reused": {
            "source": str(curated_root / "manifest.json"),
            "status": "complete-and-sealed",
            "validated_window_count": expected_count,
        },
        "protected_artifacts": {
            "unchanged": protected_unchanged,
            "before": protected_before,
            "after": protected_after,
        },
    }
    evidence_root.mkdir(parents=True, exist_ok=True)
    _write_json(evidence_root / "summary.json", summary)
    report = """# E5-I04 covariance tangent contract

Status: **PASS**

- Canonical source: `{source}`
- Windows checked: `{count}` (indices 0–{last}, frames 0–3359)
- Edge measurement: `start_pose.between(end_pose)` in direction `{direction}`
- Pose3 tangent order: `{ordering}`
- Synthetic end-local residual Jacobian maximum identity error: `{jacobian}`
- Window 0 edge reproduction maximum error: `{edge_error}`
- Window 0 nominal BetweenFactor error: `{factor_error}`
- Maximum all-window conditional-covariance symmetry error: `{symmetry}`
- Direct `Gaussian.Covariance` use: passed for all {count} windows
- Additional Adjoint transport required: no
- Covariance scaling, regularization, or replacement: none
- TrackingDB loads / Bundle solves / Pose Graph construction: 0 / 0 / 0
- E5-I03C artifacts used: no
- Protected E5-I03B and curated Bundle artifact metadata unchanged: yes

The bounded GTSAM check shows that, at the nominal measurement, an end-pose
local perturbation maps to the BetweenFactor residual with the identity
Jacobian in GTSAM's rotation-x/y/z then translation-x/y/z tangent ordering.
Therefore the saved conditional end-given-start covariance is already in the
factor residual basis and is used directly, without transport or scaling.
""".format(
        source=source_root,
        count=population["canonical_window_count_checked"],
        last=population["last_window_index"],
        direction=EDGE_DIRECTION,
        ordering=", ".join(TANGENT_ORDER),
        jacobian=synthetic[
            "end_local_jacobian_identity_max_abs_error"
        ],
        edge_error=window_zero[
            "start_between_end_edge_max_abs_error"
        ],
        factor_error=window_zero["between_factor_error"],
        symmetry=population[
            "maximum_conditional_covariance_symmetry_max_abs"
        ],
    )
    (evidence_root / "report.md").write_text(report, encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded E5-I04 covariance contract validation"
    )
    parser.add_argument(
        "--source-root", type=Path, default=CANONICAL_ARTIFACT_ROOT
    )
    parser.add_argument(
        "--curated-root", type=Path, default=CURATED_ARTIFACT_ROOT
    )
    parser.add_argument(
        "--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_validation(
        source_root=args.source_root,
        curated_root=args.curated_root,
        evidence_root=args.evidence_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "E5I04ValidationError",
    "run_validation",
    "validate_canonical_window_zero",
    "validate_complete_population",
    "validate_synthetic_gtsam_contract",
]
