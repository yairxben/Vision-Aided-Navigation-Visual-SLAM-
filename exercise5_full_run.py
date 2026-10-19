"""Resumable streamed Exercise 5 Bundle runner and endpoint contracts.

The scientific choices in this module are read from the reviewed
``exercise5_production_config.json`` file.  The validated single-window
implementation remains in :mod:`exercise5_bundle_window`; this module only
orchestrates it and extracts compact scientific evidence while the graph is
live.  Deterministic persistence and sealing live in
:mod:`exercise5_artifacts`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from exercise4_analysis import load_tracking_artifacts
from exercise5_artifacts import (
    ArtifactValidationError,
    BASE_WINDOW_FILES,
    FIRST_WINDOW_FILES,
    HASH_SCHEMA,
    MANIFEST_SCHEMA,
    RUN_STATE_SCHEMA,
    WINDOW_SCHEMA,
    _canonical_json_bytes,
    _little_float64,
    _little_int64,
    _read_json,
    _result_arrays,
    _seal_window_artifacts,
    _window_name,
    _write_csv,
    _write_json,
    _maximum_diagnostic_json,
    build_run_manifest,
    quarantine_attempt,
    sha256_file,
    snapshot_configuration,
    update_run_state,
    utc_now,
    validate_prepared_window,
    validate_sealed_window,
    write_and_seal_manifest,
    write_checksum_index,
    write_generator_registry,
)
from exercise5_bundle_window import (
    EXCESSIVE_INITIALIZATION_DEPTH,
    BundleWindowResult,
    ProbeLandmarkEligibility,
    solve_bundle_window,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "exercise5_production_config.json"
READER_SOURCE_PROVENANCE_PATH = (
    ROOT / "exercise5_reader_source_provenance.json"
)
READER_SOURCE_PROVENANCE_SCHEMA = (
    "exercise5-reader-source-provenance-v1"
)
TANGENT_ORDER = (
    "rot_x_rad",
    "rot_y_rad",
    "rot_z_rad",
    "trans_x_m",
    "trans_y_m",
    "trans_z_m",
)
CANONICAL_CONFIRMATION = "RUN-CANONICAL-355-WINDOWS"
MINIMUM_MEM_AVAILABLE_BYTES = 4_294_967_296
MINIMUM_MEM_PLUS_SWAP_BYTES = 6_567_682_560
VALIDATED_E5_I02_PEAK_RSS_BYTES = 5_348_503_552
BOUNDED_PROBE_RSS_SAFETY_MARGIN_BYTES = 536_870_912
BOUNDED_PROBE_MINIMUM_MEM_AVAILABLE_BYTES = 536_870_912
VALIDATED_E5_I03A_PEAK_RSS_BYTES = 5_348_995_072
PRODUCTION_POST_LOAD_RSS_SAFETY_MARGIN_BYTES = 536_870_912
PRODUCTION_POST_LOAD_MINIMUM_MEM_AVAILABLE_BYTES = 536_870_912
COURSE_STEREO_PROJECTION_SIGMAS_PX = (1.0, 1.0, 1.0)
COURSE_MAXIMUM_INITIALIZATION_DEPTH_M = 300.0
_COURSE_ONE_DEGREE_RAD = float(np.radians(1.0))
FIRST_POSE_PRIOR_CANDIDATES = {
    "A": (
        _COURSE_ONE_DEGREE_RAD**2,
        _COURSE_ONE_DEGREE_RAD**2,
        _COURSE_ONE_DEGREE_RAD**2,
        0.1,
        0.01,
        1.0,
    ),
    "B": (
        _COURSE_ONE_DEGREE_RAD,
        _COURSE_ONE_DEGREE_RAD,
        _COURSE_ONE_DEGREE_RAD,
        0.3,
        0.1,
        1.0,
    ),
}


# Frozen configuration and source-input contracts


class Exercise5ContractError(RuntimeError):
    """Raised when frozen configuration, scientific, or identity data differ."""


class CovarianceValidationError(RuntimeError):
    """Raised when endpoint marginal evidence is not numerically usable."""


def load_frozen_configuration(
    path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[Dict[str, object], str]:
    """Load and validate the immutable canonical-run configuration bytes."""

    path = Path(path).resolve()
    configuration = _read_json(path)
    if not isinstance(configuration, dict):
        raise Exercise5ContractError("production configuration must be an object")
    validate_frozen_configuration(configuration)
    return configuration, sha256_file(path)


def load_run_configuration(
    path: Path = DEFAULT_CONFIG_PATH,
) -> Tuple[Dict[str, object], str]:
    """Load either the strict canonical config or the named E5-I03C overlay."""

    path = Path(path).resolve()
    configuration = _read_json(path)
    if not isinstance(configuration, dict):
        raise Exercise5ContractError("run configuration must be an object")
    if configuration.get("schema") == "exercise5-production-config":
        return load_frozen_configuration(path)
    if (
        configuration.get("schema")
        == "exercise5-track-survival-experiment-config"
    ):
        from exercise5_track_survival import load_experiment_configuration

        return load_experiment_configuration(path)
    raise Exercise5ContractError("run configuration schema differs")


def validate_frozen_configuration(configuration: Mapping[str, object]) -> None:
    """Validate normative E5-A02 choices without supplying fallback defaults."""

    required = {
        "schema",
        "schema_version",
        "canonical_inputs",
        "source_contract",
        "coordinate_contract",
        "calibration",
        "keyframe_selection",
        "scientific_configuration",
        "artifact_contract",
        "run_safety",
    }
    if not required.issubset(configuration):
        raise Exercise5ContractError(
            "configuration is missing {}".format(
                sorted(required.difference(configuration))
            )
        )
    if (
        configuration["schema"] != "exercise5-production-config"
        or configuration["schema_version"] != 1
    ):
        raise Exercise5ContractError("configuration schema/version differs")
    keyframes = configuration["keyframe_selection"]["ordered_keyframe_ids"]
    statistics = configuration["keyframe_selection"]["statistics"]
    if (
        len(keyframes) != 356
        or len(set(keyframes)) != 356
        or keyframes[0] != 0
        or keyframes[-1] != 3359
        or any(b <= a for a, b in zip(keyframes, keyframes[1:]))
        or any(not 5 <= b - a + 1 <= 20 for a, b in zip(keyframes, keyframes[1:]))
        or statistics["keyframe_count"] != 356
        or statistics["window_count"] != 355
    ):
        raise Exercise5ContractError("frozen keyframe contract differs")
    science = configuration["scientific_configuration"]
    measurement = science["measurement_uncertainty"]
    prior = science["gauge_fixing_prior"]
    eligibility = science["landmark_eligibility"]
    tolerances = science["validation_tolerances"]
    if measurement["stereo_sigma_px"] != list(
        COURSE_STEREO_PROJECTION_SIGMAS_PX
    ):
        raise Exercise5ContractError("frozen stereo uncertainty differs")
    common_prior_valid = (
        prior["factor_count"] != 1
        or prior["additional_camera_priors"] != 0
        or prior["landmark_priors"] != 0
    )
    if common_prior_valid:
        raise Exercise5ContractError("frozen gauge prior differs")
    legacy_prior = (
        prior["rotation_sigma_rad"] == [1e-6] * 3
        and prior["translation_sigma_m"] == [1e-6] * 3
        and "authorized_candidates" not in prior
        and "selected_candidate" not in prior
    )
    selected_candidate = prior.get("selected_candidate")
    authorized_candidates = prior.get("authorized_candidates")
    amended_prior = (
        selected_candidate in FIRST_POSE_PRIOR_CANDIDATES
        and authorized_candidates
        == {
            name: list(sigmas)
            for name, sigmas in FIRST_POSE_PRIOR_CANDIDATES.items()
        }
        and prior["rotation_sigma_rad"]
        == list(FIRST_POSE_PRIOR_CANDIDATES[selected_candidate][:3])
        and prior["translation_sigma_m"]
        == list(FIRST_POSE_PRIOR_CANDIDATES[selected_candidate][3:])
    )
    if not legacy_prior and not amended_prior:
        raise Exercise5ContractError("frozen gauge prior differs")
    if (
        eligibility["minimum_valid_observations"] != 2
        or eligibility["minimum_distinct_frames"] != 2
        or eligibility["maximum_landmark_distance_m"] is not None
    ):
        raise Exercise5ContractError("frozen landmark eligibility differs")
    maximum_initialization_depth = eligibility.get(
        "maximum_initialization_depth_m"
    )
    if legacy_prior:
        if maximum_initialization_depth is not None:
            raise Exercise5ContractError(
                "pre-amendment landmark eligibility differs"
            )
    elif (
        maximum_initialization_depth
        != COURSE_MAXIMUM_INITIALIZATION_DEPTH_M
    ):
        raise Exercise5ContractError(
            "amended initialization-depth eligibility differs"
        )
    if configuration["coordinate_contract"]["tangent_order"] != list(
        TANGENT_ORDER
    ):
        raise Exercise5ContractError("frozen tangent ordering differs")
    if (
        tolerances["covariance_symmetry_atol"] != 1e-9
        or tolerances["covariance_minimum_eigenvalue"] != -1e-9
        or tolerances["graph_error_nonincrease_absolute"] != 1e-9
        or tolerances["graph_error_nonincrease_relative"] != 1e-12
    ):
        raise Exercise5ContractError("frozen validation tolerances differ")
    artifact = configuration["artifact_contract"]
    if (
        artifact["schema"] != "exercise5-final-artifact-contract"
        or artifact["schema_version"] != 1
        or artifact["canonical_status"] != "complete-and-sealed"
    ):
        raise Exercise5ContractError("artifact contract differs")
    safety = configuration["run_safety"]
    if (
        safety["load_database_count"] != 1
        or safety["maximum_live_bundle_graphs"] != 1
        or not safety["resume_from_sealed_windows"]
    ):
        raise Exercise5ContractError("run safety contract differs")


def _load_reader_source_successors(
    repository_root: Path,
) -> Mapping[str, Mapping[str, object]]:
    """Load the reviewed predecessor-to-reader-source SHA relationships."""

    path = Path(repository_root) / READER_SOURCE_PROVENANCE_PATH.name
    if not path.is_file():
        return {}
    provenance = _read_json(path)
    canonical_configuration = provenance.get("canonical_run_configuration")
    if (
        provenance.get("schema") != READER_SOURCE_PROVENANCE_SCHEMA
        or provenance.get("schema_version") != 1
        or provenance.get("status") != "accepted-reader-source"
        or not isinstance(canonical_configuration, Mapping)
        or canonical_configuration.get("path")
        != DEFAULT_CONFIG_PATH.name
        or canonical_configuration.get("sha256")
        != sha256_file(Path(repository_root) / DEFAULT_CONFIG_PATH.name)
        or not isinstance(provenance.get("files"), Mapping)
    ):
        raise Exercise5ContractError(
            "reader-source provenance schema or configuration identity differs"
        )
    records = provenance["files"]
    for relative, record in records.items():
        relative_path = Path(relative)
        if (
            not isinstance(relative, str)
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix != ".py"
            or not isinstance(record, Mapping)
        ):
            raise Exercise5ContractError(
                "reader-source provenance path differs: {}".format(relative)
            )
        for name in ("predecessor_sha256", "reader_source_sha256"):
            value = record.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise Exercise5ContractError(
                    "reader-source {} differs for {}".format(name, relative)
                )
        equivalence = record.get("executable_equivalence")
        if not isinstance(equivalence, Mapping):
            raise Exercise5ContractError(
                "reader-source executable equivalence is absent for {}".format(
                    relative
                )
            )
        change_class = record.get("change_class")
        if change_class == "documentation-only":
            valid_equivalence = (
                equivalence.get("status") == "proven"
                and equivalence.get("method")
                == "normalized-ast-equality-after-docstring-removal"
            )
        elif change_class == "documentation-plus-provenance-validation":
            valid_equivalence = (
                equivalence.get("status") == "reviewed-separately"
                and equivalence.get("method")
                == "narrow-source-identity-validation-change"
            )
        else:
            valid_equivalence = False
        if not valid_equivalence:
            raise Exercise5ContractError(
                "reader-source change classification differs for {}".format(
                    relative
                )
            )
        observed = sha256_file(Path(repository_root) / relative_path)
        if observed != record["reader_source_sha256"]:
            raise Exercise5ContractError(
                "reader-source hash mismatch for {}: {}".format(
                    relative, observed
                )
            )
    return records


def _validate_source_hashes(
    repository_root: Path,
    configuration: Mapping[str, object],
) -> Dict[str, str]:
    """Accept exact canonical sources or their explicitly reviewed successors."""

    repository_root = Path(repository_root).resolve()
    successors = _load_reader_source_successors(repository_root)
    sources = {}
    for relative, expected in configuration["source_contract"].items():
        observed = sha256_file(repository_root / relative)
        if observed != expected:
            successor = successors.get(relative)
            if (
                not isinstance(successor, Mapping)
                or successor.get("predecessor_sha256") != expected
                or successor.get("reader_source_sha256") != observed
            ):
                raise Exercise5ContractError(
                    "source hash mismatch for {}: {}".format(relative, observed)
                )
        sources[relative] = observed
    for relative, successor in successors.items():
        sources[relative] = successor["reader_source_sha256"]
    return dict(sorted(sources.items()))


def validate_source_and_input_hashes(
    repository_root: Path,
    configuration: Mapping[str, object],
    *,
    hash_database: bool = True,
) -> Dict[str, object]:
    """Validate current source successors and every canonical input identity.

    The configuration retains the exact source SHAs used by the completed
    canonical run.  A current reader-facing successor is accepted only through
    the separate reviewed predecessor/current record loaded above.
    """

    repository_root = Path(repository_root).resolve()
    sources = _validate_source_hashes(repository_root, configuration)
    inputs = {}
    for name, contract in configuration["canonical_inputs"].items():
        path = repository_root / contract["path"]
        if not path.is_file():
            raise Exercise5ContractError("canonical input is absent: {}".format(path))
        size = int(path.stat().st_size)
        if "size_bytes" in contract and size != contract["size_bytes"]:
            raise Exercise5ContractError(
                "{} size mismatch: {}".format(name, size)
            )
        if name == "tracking_database" and not hash_database:
            observed = None
        else:
            observed = sha256_file(path)
            if observed != contract["sha256"]:
                raise Exercise5ContractError(
                    "{} SHA-256 mismatch: {}".format(name, observed)
                )
        inputs[name] = {
            "path": str(path),
            "size_bytes": size,
            "sha256": observed if observed is not None else contract["sha256"],
            "sha256_verified": observed is not None,
        }
    return {
        "source_hashes": sources,
        "canonical_inputs": inputs,
    }


# Endpoint covariance science


def _matrix_diagnostics(
    matrix: np.ndarray,
    *,
    symmetry_atol: float,
    minimum_eigenvalue: float,
    name: str,
) -> Dict[str, object]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise CovarianceValidationError("{} must be square".format(name))
    if not np.all(np.isfinite(matrix)):
        raise CovarianceValidationError("{} is nonfinite".format(name))
    raw_symmetry_error = float(np.max(np.abs(matrix - matrix.T)))
    symmetry_error_finite = bool(np.isfinite(raw_symmetry_error))
    symmetry_error = (
        raw_symmetry_error if symmetry_error_finite else None
    )
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalue_error = None
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if np.all(np.isfinite(eigenvalues)):
            minimum = float(eigenvalues[0])
            maximum = float(eigenvalues[-1])
        else:
            minimum = None
            maximum = None
            eigenvalue_error = {
                "type": "NonfiniteEigenvalues",
                "message": "eigenvalue diagnostic returned nonfinite values",
            }
    except (np.linalg.LinAlgError, ValueError) as exception:
        minimum = None
        maximum = None
        eigenvalue_error = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    condition_error = None
    try:
        raw_condition = float(np.linalg.cond(symmetric))
    except (np.linalg.LinAlgError, ValueError) as exception:
        raw_condition = float("nan")
        condition_error = {
            "type": type(exception).__name__,
            "message": str(exception),
        }
    condition_finite = bool(np.isfinite(raw_condition))
    symmetry_passed = bool(
        symmetry_error_finite and symmetry_error <= symmetry_atol
    )
    eigenvalue_passed = (
        minimum is not None and minimum >= minimum_eigenvalue
    )
    return {
        "symmetry_max_abs": symmetry_error,
        "symmetry_error_finite": symmetry_error_finite,
        "symmetry_warning_threshold": symmetry_atol,
        "symmetry_passed_threshold": symmetry_passed,
        "symmetry_warning": not symmetry_passed,
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "minimum_eigenvalue_warning_threshold": minimum_eigenvalue,
        "minimum_eigenvalue_passed_threshold": eigenvalue_passed,
        "minimum_eigenvalue_warning": not eigenvalue_passed,
        "eigenvalue_diagnostic_error": eigenvalue_error,
        "condition_number": (
            raw_condition if condition_finite else None
        ),
        "condition_number_finite": condition_finite,
        "condition_number_warning": not condition_finite,
        "condition_number_diagnostic_error": condition_error,
    }


def _reconstruction_crosscheck_diagnostics(
    error: Optional[float],
    *,
    warning_threshold: float,
    name: str,
    diagnostic_error: Optional[BaseException] = None,
) -> Dict[str, object]:
    threshold_valid = bool(
        np.isfinite(warning_threshold) and warning_threshold >= 0.0
    )
    error_finite = bool(error is not None and np.isfinite(error))
    measured_error = float(error) if error_finite else None
    passed = bool(
        threshold_valid
        and error_finite
        and measured_error <= warning_threshold
    )
    return {
        "name": name,
        "measured_error": measured_error,
        "measured_error_finite": error_finite,
        "warning_threshold": (
            warning_threshold if threshold_valid else None
        ),
        "warning_threshold_valid": threshold_valid,
        "passed_threshold": passed,
        "warning": not passed,
        "diagnostic_error": (
            None
            if diagnostic_error is None
            else {
                "type": type(diagnostic_error).__name__,
                "message": str(diagnostic_error),
            }
        ),
    }


def validate_covariance_evidence(
    joint_covariance: np.ndarray,
    joint_information: np.ndarray,
    conditional_covariance: Optional[np.ndarray],
    *,
    symmetry_atol: float,
    minimum_eigenvalue: float,
    reconstruction_atol: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Validate ordered ``(start,end)`` endpoint marginal evidence.

    This computes ``P(end|start)`` in the endpoint marginal tangent
    coordinates.  E5-I04 subsequently proved that the retained conditional
    covariance is already in the downstream ``BetweenFactorPose3`` residual
    basis, so no Adjoint transport, scaling, or regularization is applied.
    """

    joint = _little_float64(joint_covariance)
    information = _little_float64(joint_information)
    if joint.shape != (12, 12) or information.shape != (12, 12):
        raise CovarianceValidationError("joint endpoint matrices must be 12x12")
    joint_diag = _matrix_diagnostics(
        joint,
        symmetry_atol=symmetry_atol,
        minimum_eigenvalue=minimum_eigenvalue,
        name="joint covariance",
    )
    information_diag = _matrix_diagnostics(
        information,
        symmetry_atol=symmetry_atol,
        minimum_eigenvalue=minimum_eigenvalue,
        name="joint information",
    )
    identity_error = float(
        np.max(np.abs(joint @ information - np.eye(12)))
    )
    joint_inverse_crosscheck = _reconstruction_crosscheck_diagnostics(
        identity_error,
        warning_threshold=reconstruction_atol,
        name="joint covariance/information reconstruction error",
    )
    sigma00 = joint[:6, :6]
    sigma01 = joint[:6, 6:]
    sigma10 = joint[6:, :6]
    sigma11 = joint[6:, 6:]
    try:
        solved = np.linalg.solve(sigma00, sigma01)
    except (np.linalg.LinAlgError, ValueError) as exception:
        raise CovarianceValidationError(
            "conditional covariance linear solve failed: {}".format(exception)
        ) from exception
    solve_residual = float(np.max(np.abs(sigma00 @ solved - sigma01)))
    solve_residual_finite = bool(np.isfinite(solve_residual))
    computed = 0.5 * (
        sigma11 - sigma10 @ solved
        + (sigma11 - sigma10 @ solved).T
    )
    supplied_diag = None
    if conditional_covariance is not None:
        supplied = np.asarray(conditional_covariance, dtype=float)
        if supplied.shape != (6, 6):
            raise CovarianceValidationError(
                "conditional endpoint covariance must be 6x6"
            )
        supplied_diag = _matrix_diagnostics(
            supplied,
            symmetry_atol=symmetry_atol,
            minimum_eigenvalue=minimum_eigenvalue,
            name="supplied conditional covariance",
        )
        supplied_difference = float(np.max(np.abs(supplied - computed)))
    else:
        supplied_difference = 0.0
    supplied_crosscheck = _reconstruction_crosscheck_diagnostics(
        supplied_difference,
        warning_threshold=reconstruction_atol,
        name="supplied conditional covariance reconstruction error",
    )
    lambda11 = information[6:, 6:]
    information_block_error = None
    try:
        information_conditional = np.linalg.solve(lambda11, np.eye(6))
        crosscheck = float(
            np.max(np.abs(information_conditional - computed))
        )
    except (np.linalg.LinAlgError, ValueError) as exception:
        crosscheck = None
        information_block_error = exception
    information_block_crosscheck = _reconstruction_crosscheck_diagnostics(
        crosscheck,
        warning_threshold=reconstruction_atol,
        name="conditional covariance information-block cross-check error",
        diagnostic_error=information_block_error,
    )
    conditional_diag = _matrix_diagnostics(
        computed,
        symmetry_atol=symmetry_atol,
        minimum_eigenvalue=minimum_eigenvalue,
        name="conditional covariance",
    )
    diagnostics = {
        "ordered_blocks": ["start", "end"],
        "tangent_order_per_block": list(TANGENT_ORDER),
        "joint_covariance": joint_diag,
        "joint_information": information_diag,
        "conditional_covariance": conditional_diag,
        "supplied_conditional_covariance": supplied_diag,
        "joint_inverse_reconstruction_max_abs": (
            joint_inverse_crosscheck["measured_error"]
        ),
        "sigma00_solve_residual_max_abs": (
            solve_residual if solve_residual_finite else None
        ),
        "sigma00_solve_residual_finite": solve_residual_finite,
        "sigma00_solve_residual_warning": not solve_residual_finite,
        "conditional_supplied_reconstruction_max_abs": (
            supplied_crosscheck["measured_error"]
        ),
        "conditional_information_block_crosscheck_max_abs": (
            information_block_crosscheck["measured_error"]
        ),
        "numerical_reconstruction_crosschecks": {
            "joint_covariance_information_inverse": (
                joint_inverse_crosscheck
            ),
            "supplied_conditional_covariance": supplied_crosscheck,
            "conditional_information_block": information_block_crosscheck,
        },
        "between_factor_noise_status": (
            "retained endpoint tangent conditional evidence; E5-I04 tangent "
            "transport validation required before BetweenFactorPose3 use"
        ),
    }
    return _little_float64(computed), diagnostics


def extract_endpoint_covariances(
    result: BundleWindowResult,
    marginals: gtsam.Marginals,
    configuration: Mapping[str, object],
) -> Dict[str, object]:
    """Extract ordered endpoint Marginals and compute ``P(end | start)``.

    The explicit ``[start, end]`` key order fixes the two 6-coordinate blocks
    in both retained 12x12 matrices and in the Schur-complement conditional.
    """

    start, end = result.frame_ids[0], result.frame_ids[-1]
    keys = gtsam.KeyVector(
        [
            result.camera_key_by_frame_id[start],
            result.camera_key_by_frame_id[end],
        ]
    )
    joint = np.asarray(
        marginals.jointMarginalCovariance(keys).fullMatrix(), dtype=float
    )
    information = np.asarray(
        marginals.jointMarginalInformation(keys).fullMatrix(), dtype=float
    )
    tolerance = configuration["scientific_configuration"][
        "validation_tolerances"
    ]
    conditional, diagnostics = validate_covariance_evidence(
        joint,
        information,
        None,
        symmetry_atol=tolerance["covariance_symmetry_atol"],
        minimum_eigenvalue=tolerance["covariance_minimum_eigenvalue"],
        reconstruction_atol=tolerance["numerical_reconstruction_atol"],
    )
    diagnostics["joint_block_order_source"] = (
        "explicit ordered KeyVector [start, end]"
    )
    return {
        "joint_covariance": _little_float64(joint),
        "joint_information": _little_float64(information),
        "conditional_end_given_start_covariance": conditional,
        "diagnostics": diagnostics,
    }


def extract_first_window_camera_marginals(
    result: BundleWindowResult,
    marginals: gtsam.Marginals,
    configuration: Mapping[str, object],
) -> Dict[str, np.ndarray]:
    """Retain every optimized camera pose/covariance for window zero."""

    tolerance = configuration["scientific_configuration"][
        "validation_tolerances"
    ]
    covariances = []
    poses = []
    for frame_id in result.frame_ids:
        covariance = np.asarray(
            marginals.marginalCovariance(
                result.camera_key_by_frame_id[frame_id]
            ),
            dtype=float,
        )
        _matrix_diagnostics(
            covariance,
            symmetry_atol=tolerance["covariance_symmetry_atol"],
            minimum_eigenvalue=tolerance["covariance_minimum_eigenvalue"],
            name="camera {} marginal covariance".format(frame_id),
        )
        covariances.append(covariance)
        poses.append(
            np.asarray(
                result.final_values.atPose3(
                    result.camera_key_by_frame_id[frame_id]
                ).matrix(),
                dtype=float,
            )
        )
    return {
        "frame_ids": _little_int64(result.frame_ids),
        "poses": _little_float64(poses),
        "marginal_covariances": _little_float64(covariances),
    }


def evaluate_standard_marginals_candidate(
    *,
    candidate: str,
    result: BundleWindowResult,
    configuration: Mapping[str, object],
    marginals_factory=None,
) -> Tuple[
    Dict[str, object],
    Optional[Dict[str, object]],
    Optional[Dict[str, np.ndarray]],
]:
    """Construct and validate standard GTSAM Marginals exactly once."""

    if candidate not in FIRST_POSE_PRIOR_CANDIDATES:
        raise ValueError("candidate must be exactly A or B")
    factory = gtsam.Marginals if marginals_factory is None else marginals_factory
    outcome = {
        "candidate": candidate,
        "prior_sigmas": list(FIRST_POSE_PRIOR_CANDIDATES[candidate]),
        "marginals": {
            "api": "gtsam.Marginals(graph, optimized_values)",
            "construction_count": 1,
            "succeeded": False,
            "exception_type": None,
            "exception_message": None,
        },
        "covariance_validation": {
            "passed": False,
            "exception_type": None,
            "exception_message": None,
        },
        "eligible": False,
    }
    try:
        marginals = factory(result.graph, result.final_values)
    except BaseException as exception:
        outcome["marginals"].update(
            {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
            }
        )
        return outcome, None, None
    outcome["marginals"]["succeeded"] = True
    try:
        endpoint = extract_endpoint_covariances(
            result, marginals, configuration
        )
        camera_marginals = extract_first_window_camera_marginals(
            result, marginals, configuration
        )
    except BaseException as exception:
        outcome["covariance_validation"].update(
            {
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
            }
        )
        del marginals
        return outcome, None, None
    del marginals
    outcome["covariance_validation"].update(
        {
            "passed": True,
            "endpoint_diagnostics": endpoint["diagnostics"],
            "camera_marginal_count": int(
                camera_marginals["frame_ids"].size
            ),
        }
    )
    outcome["eligible"] = True
    return outcome, endpoint, camera_marginals


def select_first_pose_prior_candidate(
    outcomes: Mapping[str, Mapping[str, object]],
) -> Optional[str]:
    """Apply the reviewed A/B eligibility and deterministic tie-break rule."""

    if set(outcomes) != set(FIRST_POSE_PRIOR_CANDIDATES):
        raise ValueError("outcomes must contain exactly candidates A and B")
    eligible = {}
    for candidate in ("A", "B"):
        outcome = outcomes[candidate]
        if outcome.get("prior_sigmas") != list(
            FIRST_POSE_PRIOR_CANDIDATES[candidate]
        ):
            raise ValueError(
                "candidate {} prior sigmas differ".format(candidate)
            )
        eligible[candidate] = bool(
            outcome.get("eligible")
            and outcome.get("marginals", {}).get("succeeded")
            and outcome.get("covariance_validation", {}).get("passed")
        )
    if eligible["A"]:
        return "A"
    if eligible["B"]:
        return "B"
    return None


# Endpoint relative-pose extraction


def _rotation_quaternion_xyzw(rotation: np.ndarray) -> List[float]:
    rotation = np.asarray(rotation, dtype=float)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        values = [
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ]
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = np.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            values = [
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            ]
        elif index == 1:
            scale = np.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            values = [
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            ]
        else:
            scale = np.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            values = [
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
    quaternion = np.asarray(values, dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.tolist()


def extract_endpoint_edge(
    result: BundleWindowResult,
    *,
    rotation_atol: float,
) -> Dict[str, object]:
    """Return the optimized ``start.between(end)`` edge and rigidity evidence.

    Both endpoint poses share the window's first-camera parent frame.  The
    stored direction is therefore ``T_Cstart_from_Cend``.
    """

    start, end = result.frame_ids[0], result.frame_ids[-1]
    start_pose = result.final_values.atPose3(
        result.camera_key_by_frame_id[start]
    )
    end_pose = result.final_values.atPose3(result.camera_key_by_frame_id[end])
    delta = start_pose.between(end_pose)
    matrix = np.asarray(delta.matrix(), dtype=float)
    rotation = matrix[:3, :3]
    diagnostics = {
        "finite": bool(np.all(np.isfinite(matrix))),
        "homogeneous_bottom_row_max_abs": float(
            np.max(np.abs(matrix[3] - np.array([0.0, 0.0, 0.0, 1.0])))
        ),
        "rotation_orthogonality_max_abs": float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        ),
        "rotation_determinant_abs_error": abs(
            float(np.linalg.det(rotation)) - 1.0
        ),
    }
    if (
        not diagnostics["finite"]
        or diagnostics["homogeneous_bottom_row_max_abs"] > rotation_atol
        or diagnostics["rotation_orthogonality_max_abs"] > rotation_atol
        or diagnostics["rotation_determinant_abs_error"] > rotation_atol
    ):
        raise ArtifactValidationError("optimized endpoint edge is not rigid")
    return {
        "schema": "exercise5-endpoint-edge-v1",
        "start_frame_id": start,
        "end_frame_id": end,
        "start_camera_key": int(result.camera_key_by_frame_id[start]),
        "end_camera_key": int(result.camera_key_by_frame_id[end]),
        "direction": "T_Cstart_from_Cend",
        "matrix": matrix.tolist(),
        "quaternion_xyzw": _rotation_quaternion_xyzw(rotation),
        "rotation_vector_rad": np.asarray(
            gtsam.Rot3.Logmap(delta.rotation()), dtype=float
        ).tolist(),
        "translation_m": matrix[:3, 3].tolist(),
        "diagnostics": diagnostics,
    }


# Runtime input loading and memory-safety machinery


def _read_ground_truth(path: Path) -> List[np.ndarray]:
    poses = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        values = np.asarray(line.split(), dtype=float)
        if values.size != 12:
            raise Exercise5ContractError("ground-truth row is not 3x4")
        poses.append(values.reshape(3, 4))
    return poses


def seal_window_artifacts(
    *,
    windows_root: Path,
    quarantine_root: Path,
    result: BundleWindowResult,
    database: object,
    calibration: gtsam.Cal3_S2Stereo,
    world_to_camera_poses: Sequence[np.ndarray],
    ground_truth_poses: Sequence[np.ndarray],
    configuration: Mapping[str, object],
    config_sha256: str,
    source_identity_sha256: str,
    window_index: int,
    endpoint_covariances: Mapping[str, object],
    first_window_camera_marginals: Optional[Mapping[str, np.ndarray]],
    runtime_seconds: float,
    peak_rss_bytes: int,
    image_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Preserve the validated E5 artifact API while delegating persistence."""

    return _seal_window_artifacts(
        windows_root=windows_root,
        quarantine_root=quarantine_root,
        result=result,
        database=database,
        calibration=calibration,
        world_to_camera_poses=world_to_camera_poses,
        ground_truth_poses=ground_truth_poses,
        configuration=configuration,
        config_sha256=config_sha256,
        source_identity_sha256=source_identity_sha256,
        window_index=window_index,
        endpoint_covariances=endpoint_covariances,
        first_window_camera_marginals=first_window_camera_marginals,
        runtime_seconds=runtime_seconds,
        peak_rss_bytes=peak_rss_bytes,
        endpoint_edge_extractor=extract_endpoint_edge,
        tangent_order=TANGENT_ORDER,
        image_root=image_root,
    )


def _read_calibration(
    path: Path, configuration: Mapping[str, object]
) -> gtsam.Cal3_S2Stereo:
    values = configuration["calibration"]
    calibration = gtsam.Cal3_S2Stereo(
        values["fx_px"],
        values["fy_px"],
        values["skew_px"],
        values["cx_px"],
        values["cy_px"],
        values["baseline_m"],
    )
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        label, raw = line.split(":", 1)
        rows[label] = np.asarray(raw.split(), dtype=float).reshape(3, 4)
    intrinsic = rows["P0"][:, :3]
    right = np.linalg.solve(intrinsic, rows["P1"])
    observed = np.asarray(
        [
            calibration.fx(),
            calibration.fy(),
            calibration.skew(),
            calibration.px(),
            calibration.py(),
            calibration.baseline(),
        ]
    )
    expected = np.asarray(
        [
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
            -right[0, 3],
        ]
    )
    tolerance = configuration["scientific_configuration"][
        "validation_tolerances"
    ]["numerical_reconstruction_atol"]
    if float(np.max(np.abs(observed - expected))) > tolerance:
        raise Exercise5ContractError("calibration values differ from frozen data")
    return calibration


def _source_identity_hash(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _current_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("VmRSS:"):
            value = int(line.split()[1]) * 1024
            if value <= 0:
                break
            return value
    raise MemoryError("Linux current process VmRSS is unavailable")


def _memory_snapshot() -> Dict[str, int]:
    fields = {}
    for line in Path("/proc/meminfo").read_text(
        encoding="utf-8"
    ).splitlines():
        name, raw = line.split(":", 1)
        if name in ("MemAvailable", "SwapFree"):
            fields[name] = int(raw.split()[0]) * 1024
    if set(fields) != {"MemAvailable", "SwapFree"}:
        raise MemoryError("Linux memory capacity fields are unavailable")
    return {
        "mem_available_bytes": fields["MemAvailable"],
        "swap_free_bytes": fields["SwapFree"],
        "mem_available_plus_swap_free_bytes": (
            fields["MemAvailable"] + fields["SwapFree"]
        ),
    }


def _enforce_memory_guard(stage: str) -> Dict[str, object]:
    snapshot = _memory_snapshot()
    evidence = {
        "stage": stage,
        "minimum_mem_available_bytes": MINIMUM_MEM_AVAILABLE_BYTES,
        "minimum_mem_plus_swap_bytes": MINIMUM_MEM_PLUS_SWAP_BYTES,
        **snapshot,
    }
    if (
        snapshot["mem_available_bytes"] < MINIMUM_MEM_AVAILABLE_BYTES
        or snapshot["mem_available_plus_swap_free_bytes"]
        < MINIMUM_MEM_PLUS_SWAP_BYTES
    ):
        raise MemoryError(
            "{} memory guard failed: {}".format(
                stage, json.dumps(evidence, sort_keys=True)
            )
        )
    return evidence


def bounded_probe_post_load_memory_evidence(
    *,
    current_process_rss_bytes: int,
    mem_available_bytes: int,
    swap_free_bytes: int,
    validated_reference_peak_rss_bytes: int = (
        VALIDATED_E5_I02_PEAK_RSS_BYTES
    ),
    safety_margin_bytes: int = BOUNDED_PROBE_RSS_SAFETY_MARGIN_BYTES,
    minimum_immediate_mem_available_bytes: int = (
        BOUNDED_PROBE_MINIMUM_MEM_AVAILABLE_BYTES
    ),
) -> Dict[str, object]:
    """Evaluate the bounded 0..7 post-load incremental memory admission.

    This policy is deliberately probe-only.  It uses the validated E5-I02
    frames 0..9 peak as a conservative total-process reference and never
    changes the future canonical-run memory gate.
    """

    named_values = (
        ("current_process_rss_bytes", current_process_rss_bytes),
        (
            "validated_reference_peak_rss_bytes",
            validated_reference_peak_rss_bytes,
        ),
        ("mem_available_bytes", mem_available_bytes),
        ("swap_free_bytes", swap_free_bytes),
        ("safety_margin_bytes", safety_margin_bytes),
        (
            "minimum_immediate_mem_available_bytes",
            minimum_immediate_mem_available_bytes,
        ),
    )
    for name, value in named_values:
        if type(value) is not int or value < 0:
            raise ValueError("{} must be a nonnegative integer".format(name))
    if current_process_rss_bytes == 0 or validated_reference_peak_rss_bytes == 0:
        raise ValueError("RSS evidence must be strictly positive")
    expected_incremental_rss = max(
        0, validated_reference_peak_rss_bytes - current_process_rss_bytes
    )
    required_headroom = expected_incremental_rss + safety_margin_bytes
    available_plus_swap = mem_available_bytes + swap_free_bytes
    immediate_gate = (
        mem_available_bytes >= minimum_immediate_mem_available_bytes
    )
    headroom_gate = available_plus_swap >= required_headroom
    return {
        "schema": "exercise5-bounded-probe-post-load-memory-v1",
        "stage": "after-canonical-database-load-before-window-0",
        "policy_scope": "bounded-e5-i03a-probe-only",
        "future_e5_i03b_policy_unchanged": True,
        "reference": {
            "source": "validated E5-I02 canonical frames 0..9 run",
            "validated_peak_rss_bytes": validated_reference_peak_rss_bytes,
            "bounded_probe_window": [0, 7],
            "reference_window": [0, 9],
        },
        "observed": {
            "current_process_rss_bytes": current_process_rss_bytes,
            "mem_available_bytes": mem_available_bytes,
            "swap_free_bytes": swap_free_bytes,
            "mem_available_plus_swap_free_bytes": available_plus_swap,
        },
        "policy": {
            "safety_margin_bytes": safety_margin_bytes,
            "minimum_immediate_mem_available_bytes": (
                minimum_immediate_mem_available_bytes
            ),
            "formula_expected_incremental_rss_bytes": (
                "max(0, validated_reference_peak_rss_bytes - "
                "current_process_rss_bytes)"
            ),
            "formula_required_headroom_bytes": (
                "expected_incremental_rss_bytes + safety_margin_bytes"
            ),
            "expected_incremental_rss_bytes": expected_incremental_rss,
            "required_headroom_bytes": required_headroom,
        },
        "gates": {
            "minimum_immediate_mem_available_passed": immediate_gate,
            "available_plus_swap_headroom_passed": headroom_gate,
        },
        "passed": bool(immediate_gate and headroom_gate),
    }


def production_post_load_memory_policy(
    *,
    post_load_baseline_rss_bytes: int,
) -> Dict[str, object]:
    """Freeze the canonical post-load headroom requirement exactly once."""

    if (
        type(post_load_baseline_rss_bytes) is not int
        or post_load_baseline_rss_bytes < 0
    ):
        raise ValueError(
            "post_load_baseline_rss_bytes must be a nonnegative integer"
        )
    if post_load_baseline_rss_bytes == 0:
        raise ValueError("RSS evidence must be strictly positive")
    expected_incremental_rss = max(
        0,
        VALIDATED_E5_I03A_PEAK_RSS_BYTES
        - post_load_baseline_rss_bytes,
    )
    policy = {
        "schema": "exercise5-production-post-load-memory-policy-v1",
        "policy_scope": "canonical-e5-i03b",
        "reference": {
            "source": "validated E5-I03A canonical frames 0..7 run",
            "validated_peak_rss_bytes": (
                VALIDATED_E5_I03A_PEAK_RSS_BYTES
            ),
        },
        "post_load_baseline_rss_bytes": post_load_baseline_rss_bytes,
        "safety_margin_bytes": (
            PRODUCTION_POST_LOAD_RSS_SAFETY_MARGIN_BYTES
        ),
        "minimum_immediate_mem_available_bytes": (
            PRODUCTION_POST_LOAD_MINIMUM_MEM_AVAILABLE_BYTES
        ),
        "formula_expected_incremental_rss_bytes": (
            "max(0, validated_reference_peak_rss_bytes - "
            "post_load_baseline_rss_bytes)"
        ),
        "formula_required_headroom_bytes": (
            "expected_incremental_rss_bytes + safety_margin_bytes"
        ),
        "expected_incremental_rss_bytes": expected_incremental_rss,
        "required_headroom_bytes": (
            expected_incremental_rss
            + PRODUCTION_POST_LOAD_RSS_SAFETY_MARGIN_BYTES
        ),
    }
    _validate_production_post_load_memory_policy(policy)
    return policy


def _validate_production_post_load_memory_policy(
    fixed_policy: Mapping[str, object],
) -> None:
    if (
        not isinstance(fixed_policy, Mapping)
        or fixed_policy.get("schema")
        != "exercise5-production-post-load-memory-policy-v1"
        or fixed_policy.get("policy_scope") != "canonical-e5-i03b"
        or not isinstance(fixed_policy.get("reference"), Mapping)
    ):
        raise ValueError("fixed production memory policy is malformed")
    validated_peak = fixed_policy["reference"].get(
        "validated_peak_rss_bytes"
    )
    baseline = fixed_policy.get("post_load_baseline_rss_bytes")
    safety_margin = fixed_policy.get("safety_margin_bytes")
    minimum_immediate = fixed_policy.get(
        "minimum_immediate_mem_available_bytes"
    )
    expected_incremental = fixed_policy.get(
        "expected_incremental_rss_bytes"
    )
    required_headroom = fixed_policy.get("required_headroom_bytes")
    named_values = (
        ("validated_peak_rss_bytes", validated_peak),
        ("post_load_baseline_rss_bytes", baseline),
        ("safety_margin_bytes", safety_margin),
        (
            "minimum_immediate_mem_available_bytes",
            minimum_immediate,
        ),
        ("expected_incremental_rss_bytes", expected_incremental),
        ("required_headroom_bytes", required_headroom),
    )
    for name, value in named_values:
        if type(value) is not int or value < 0:
            raise ValueError("{} must be a nonnegative integer".format(name))
    if validated_peak == 0 or baseline == 0:
        raise ValueError("RSS evidence must be strictly positive")
    if (
        validated_peak != VALIDATED_E5_I03A_PEAK_RSS_BYTES
        or safety_margin
        != PRODUCTION_POST_LOAD_RSS_SAFETY_MARGIN_BYTES
        or minimum_immediate
        != PRODUCTION_POST_LOAD_MINIMUM_MEM_AVAILABLE_BYTES
        or expected_incremental
        != max(0, validated_peak - baseline)
        or required_headroom != expected_incremental + safety_margin
    ):
        raise ValueError("fixed production memory policy is inconsistent")


def production_post_load_memory_evidence(
    *,
    fixed_policy: Mapping[str, object],
    current_process_rss_bytes: int,
    mem_available_bytes: int,
    swap_free_bytes: int,
) -> Dict[str, object]:
    """Evaluate one window against the already frozen production policy."""

    _validate_production_post_load_memory_policy(fixed_policy)
    required_headroom = fixed_policy.get("required_headroom_bytes")
    minimum_immediate = fixed_policy.get(
        "minimum_immediate_mem_available_bytes"
    )
    named_values = (
        ("current_process_rss_bytes", current_process_rss_bytes),
        ("mem_available_bytes", mem_available_bytes),
        ("swap_free_bytes", swap_free_bytes),
        ("required_headroom_bytes", required_headroom),
        ("minimum_immediate_mem_available_bytes", minimum_immediate),
    )
    for name, value in named_values:
        if type(value) is not int or value < 0:
            raise ValueError("{} must be a nonnegative integer".format(name))
    if current_process_rss_bytes == 0:
        raise ValueError("RSS evidence must be strictly positive")
    available_plus_swap = mem_available_bytes + swap_free_bytes
    immediate_gate = mem_available_bytes >= minimum_immediate
    headroom_gate = available_plus_swap >= required_headroom
    return {
        "schema": "exercise5-production-post-load-memory-evidence-v1",
        "stage": "after-canonical-database-load-before-window",
        "fixed_policy": dict(fixed_policy),
        "observed": {
            "current_process_rss_bytes": current_process_rss_bytes,
            "mem_available_bytes": mem_available_bytes,
            "swap_free_bytes": swap_free_bytes,
            "mem_available_plus_swap_free_bytes": available_plus_swap,
        },
        "gates": {
            "minimum_immediate_mem_available_passed": immediate_gate,
            "available_plus_swap_headroom_passed": headroom_gate,
        },
        "passed": bool(immediate_gate and headroom_gate),
    }


def _quarantine_stale_attempts(
    windows_root: Path, quarantine_root: Path
) -> List[str]:
    quarantined = []
    for path in sorted(Path(windows_root).glob(".*.attempt.*.tmp")):
        target = quarantine_attempt(
            path,
            quarantine_root,
            exception=ArtifactValidationError(
                "incomplete temporary attempt found during resume"
            ),
        )
        quarantined.append(str(target))
    return quarantined


def _load_canonical_once(
    repository_root: Path,
    configuration: Mapping[str, object],
) -> Tuple[object, Sequence[np.ndarray], Mapping[str, object]]:
    """Load the E4 TrackingDB/companion once and recheck their identities."""

    inputs = configuration["canonical_inputs"]
    database, companion, normalized = load_tracking_artifacts(
        repository_root / inputs["tracking_database"]["path"],
        repository_root / inputs["companion"]["path"],
    )
    poses = tuple(companion.world_to_camera_poses)
    if database.frame_num() != 3360 or len(poses) != 3360:
        raise Exercise5ContractError("loaded canonical frame count differs")
    loaded_database = normalized["database"]
    loaded_companion = normalized["companion"]
    if (
        loaded_database["file_sha256"]
        != inputs["tracking_database"]["sha256"]
        or loaded_database["streaming_digest"]
        != inputs["tracking_database"]["streaming_digest"]
        or loaded_companion["file_sha256"] != inputs["companion"]["sha256"]
    ):
        raise Exercise5ContractError("loaded TrackingDB identity differs")
    return database, poses, normalized


# Historical bounded prior-recovery and probe support (noncanonical)


def _bundle_solution_sha256(result: BundleWindowResult) -> str:
    digest = hashlib.sha256()
    for values in (
        _little_int64(result.frame_ids),
        _little_int64(result.accepted_track_ids),
        _result_arrays(result)["final_camera_poses"],
        _result_arrays(result)["final_landmarks"],
    ):
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _candidate_track_diagnostic(
    *,
    database: object,
    result: BundleWindowResult,
    calibration: gtsam.Cal3_S2Stereo,
    track_id: int,
) -> Dict[str, object]:
    start, end = result.frame_ids[0], result.frame_ids[-1]
    frame_ids = [
        int(frame_id)
        for frame_id in database.frames(track_id)
        if start <= int(frame_id) <= end
    ]
    measurements = []
    for frame_id in frame_ids:
        link = database.link(frame_id, track_id)
        values = [
            float(link.x_left),
            float(link.x_right),
            float(link.y),
        ]
        measurements.append(
            {
                "frame_id": frame_id,
                "stereo_measurement": values,
                "disparity_px": values[0] - values[1],
            }
        )
    retained = track_id in result.landmark_key_by_track_id
    initialization_frame = result.initialization_frame_by_track_id.get(
        track_id
    )
    initialization_depth = result.initialization_depth_by_track_id.get(
        track_id
    )
    if not retained:
        initialization_frame = result.excessive_depth_frame_by_track_id.get(
            track_id
        )
        initialization_depth = result.excessive_depth_m_by_track_id.get(
            track_id
        )
    initial_point = None
    optimized_point = None
    initial_depths = None
    optimized_depths = None
    if retained:
        key = result.landmark_key_by_track_id[track_id]
        initial = np.asarray(result.initial_values.atPoint3(key), dtype=float)
        optimized = np.asarray(result.final_values.atPoint3(key), dtype=float)
        initial_point = initial.tolist()
        optimized_point = optimized.tolist()
        initial_depths = {
            str(frame_id): float(
                np.asarray(
                    result.initial_values.atPose3(
                        result.camera_key_by_frame_id[frame_id]
                    ).transformTo(initial),
                    dtype=float,
                )[2]
            )
            for frame_id in frame_ids
        }
        optimized_depths = {
            str(frame_id): float(
                np.asarray(
                    result.final_values.atPose3(
                        result.camera_key_by_frame_id[frame_id]
                    ).transformTo(optimized),
                    dtype=float,
                )[2]
            )
            for frame_id in frame_ids
        }
    elif initialization_frame is not None:
        link = database.link(initialization_frame, track_id)
        initial = np.asarray(
            gtsam.StereoCamera(
                result.initial_values.atPose3(
                    result.camera_key_by_frame_id[initialization_frame]
                ),
                calibration,
            ).backproject(
                gtsam.StereoPoint2(
                    float(link.x_left),
                    float(link.x_right),
                    float(link.y),
                )
            ),
            dtype=float,
        )
        initial_point = initial.tolist()
        initial_depths = {
            str(frame_id): float(
                np.asarray(
                    result.initial_values.atPose3(
                        result.camera_key_by_frame_id[frame_id]
                    ).transformTo(initial),
                    dtype=float,
                )[2]
            )
            for frame_id in frame_ids
        }
    return {
        "track_id": track_id,
        "gtsam_key": int(gtsam.symbol("q", track_id)),
        "observing_frame_ids": frame_ids,
        "measurements": measurements,
        "retained": retained,
        "rejection_reason": result.rejected_track_reason_by_id.get(
            track_id
        ),
        "selected_initialization_frame_id": initialization_frame,
        "initialization_depth_m": initialization_depth,
        "initial_point": initial_point,
        "optimized_point": optimized_point,
        "initial_depths_by_observing_frame_m": initial_depths,
        "optimized_depths_by_observing_frame_m": optimized_depths,
    }


def _candidate_probe_metrics(
    *,
    candidate: str,
    result: BundleWindowResult,
    q516: Mapping[str, object],
    candidate_seconds: float,
    marginals_seconds: float,
    outcome: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "candidate": candidate,
        "prior_sigmas": list(FIRST_POSE_PRIOR_CANDIDATES[candidate]),
        "accepted_landmark_count": result.accepted_track_count,
        "rejected_landmark_count": result.rejected_track_count,
        "excessive_initialization_depth_rejection_count": (
            result.rejection_reason_counts.get(
                EXCESSIVE_INITIALIZATION_DEPTH, 0
            )
        ),
        "rejection_reason_counts": dict(result.rejection_reason_counts),
        "q516": dict(q516),
        "graph_counts": {
            "cameras": result.camera_count,
            "landmarks": result.landmark_count,
            "projection_factors": result.projection_factor_count,
            "anchor_factors": result.anchor_factor_count,
            "total_factors": result.total_factor_count,
        },
        "errors": {
            "initial_total_graph_error": (
                result.initial_errors.total_graph_error
            ),
            "final_total_graph_error": result.final_errors.total_graph_error,
            "initial_mean_projection_factor_error": (
                result.initial_errors.mean_projection_factor_error
            ),
            "final_mean_projection_factor_error": (
                result.final_errors.mean_projection_factor_error
            ),
            "final_anchor_factor_error": (
                result.final_errors.anchor_factor_error
            ),
        },
        "runtime": {
            "graph_construction_seconds": (
                result.graph_construction_runtime_seconds
            ),
            "optimizer_seconds": result.optimization_runtime_seconds,
            "marginals_and_covariance_seconds": marginals_seconds,
            "candidate_total_seconds": candidate_seconds,
        },
        "optimizer_iterations": result.optimizer_iterations,
        "peak_rss_bytes": _peak_rss_bytes(),
        "marginals": outcome["marginals"],
        "covariance_validation": outcome["covariance_validation"],
        "eligible": outcome["eligible"],
    }


def run_bounded_two_candidate_probe(
    *,
    output_root: Path,
    repository_root: Path = ROOT,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
) -> Dict[str, object]:
    """Run the single-load, frames-0..7 E5-I03A A/B recovery probe."""

    repository_root = Path(repository_root).resolve()
    output_root = Path(output_root).resolve()
    if (
        output_root.parent != Path("/tmp")
        or not output_root.name.startswith("e5_i03a_")
    ):
        raise ValueError("comparison requires a new /tmp/e5_i03a_* root")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir()
    for relative in (
        "windows",
        "quarantine",
        "aggregate/data",
        "aggregate/figures",
        "provenance",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    process_started = perf_counter()
    configuration, _ = load_frozen_configuration(configuration_path)
    preflight = _enforce_memory_guard(
        "before-canonical-input-validation"
    )
    identity_configuration = json.loads(json.dumps(configuration))
    old_bundle_hash = identity_configuration["source_contract"][
        "exercise5_bundle_window.py"
    ]
    current_bundle_hash = sha256_file(
        repository_root / "exercise5_bundle_window.py"
    )
    identity_configuration["source_contract"][
        "exercise5_bundle_window.py"
    ] = current_bundle_hash
    source_identity = validate_source_and_input_hashes(
        repository_root, identity_configuration, hash_database=True
    )
    identity_sha = _source_identity_hash(source_identity)
    calibration = _read_calibration(
        repository_root
        / configuration["canonical_inputs"]["calibration"]["path"],
        configuration,
    )
    ground_truth = _read_ground_truth(
        repository_root
        / configuration["canonical_inputs"]["ground_truth"]["path"]
    )
    pre_load_memory = _enforce_memory_guard(
        "before-canonical-database-load"
    )
    load_started = perf_counter()
    database, poses, normalized = _load_canonical_once(
        repository_root, configuration
    )
    database_load_seconds = perf_counter() - load_started
    del normalized
    post_load_snapshot = _memory_snapshot()
    post_load_guard = bounded_probe_post_load_memory_evidence(
        current_process_rss_bytes=_current_rss_bytes(),
        mem_available_bytes=post_load_snapshot["mem_available_bytes"],
        swap_free_bytes=post_load_snapshot["swap_free_bytes"],
    )
    history = [
        {
            "event": "strict-preflight-memory-gate",
            "evidence": preflight,
            "utc": utc_now(),
        },
        {
            "event": "strict-pre-load-memory-gate",
            "evidence": pre_load_memory,
            "utc": utc_now(),
        },
        {
            "event": "bounded-probe-post-load-memory-guard",
            "evidence": post_load_guard,
            "utc": utc_now(),
        },
    ]
    update_run_state(
        output_root,
        status="running-candidate-comparison",
        expected_window_count=355,
        completed_window_indices=[],
        history=history,
        scope="bounded-probe",
    )
    if not post_load_guard["passed"]:
        del database
        gc.collect()
        raise MemoryError(
            "bounded comparison post-load memory guard failed: {}".format(
                json.dumps(post_load_guard, sort_keys=True)
            )
        )

    projection_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(COURSE_STEREO_PROJECTION_SIGMAS_PX, dtype=float)
    )
    eligibility = ProbeLandmarkEligibility(
        minimum_valid_observations=2,
        minimum_distinct_frames=2,
        maximum_initialization_depth_m=(
            COURSE_MAXIMUM_INITIALIZATION_DEPTH_M
        ),
    )
    rotation_atol = configuration["scientific_configuration"][
        "validation_tolerances"
    ]["rotation_orthogonality_max_abs"]
    keyframes = configuration["keyframe_selection"][
        "ordered_keyframe_ids"
    ]
    start, end = keyframes[0], keyframes[1]
    if (start, end) != (0, 7):
        del database
        gc.collect()
        raise Exercise5ContractError(
            "frozen first production window is not frames 0..7"
        )

    outcomes = {}
    metrics = {}
    covariance_by_candidate = {}
    camera_marginals_by_candidate = {}
    solution_sha_by_candidate = {}
    try:
        for candidate in ("A", "B"):
            candidate_started = perf_counter()
            anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    FIRST_POSE_PRIOR_CANDIDATES[candidate], dtype=float
                )
            )
            result = solve_bundle_window(
                database=database,
                world_to_camera_poses=poses,
                calibration=calibration,
                start_frame_id=start,
                end_frame_id=end,
                projection_noise=projection_noise,
                anchor_noise=anchor_noise,
                landmark_eligibility=eligibility,
                rotation_atol=rotation_atol,
            )
            solution_sha_by_candidate[candidate] = (
                _bundle_solution_sha256(result)
            )
            marginals_started = perf_counter()
            outcome, endpoint, camera_marginals = (
                evaluate_standard_marginals_candidate(
                    candidate=candidate,
                    result=result,
                    configuration=configuration,
                )
            )
            marginals_seconds = perf_counter() - marginals_started
            q516 = _candidate_track_diagnostic(
                database=database,
                result=result,
                calibration=calibration,
                track_id=516,
            )
            outcomes[candidate] = outcome
            metrics[candidate] = _candidate_probe_metrics(
                candidate=candidate,
                result=result,
                q516=q516,
                candidate_seconds=perf_counter() - candidate_started,
                marginals_seconds=marginals_seconds,
                outcome=outcome,
            )
            if outcome["eligible"]:
                covariance_by_candidate[candidate] = endpoint
                camera_marginals_by_candidate[candidate] = camera_marginals
            del camera_marginals
            del endpoint
            del outcome
            del result
            gc.collect()
            _write_json(
                output_root / "candidate_comparison.json",
                {
                    "schema": "exercise5-i03a-prior-comparison-v1",
                    "status": "running",
                    "window": [start, end],
                    "projection_sigmas_px": list(
                        COURSE_STEREO_PROJECTION_SIGMAS_PX
                    ),
                    "maximum_initialization_depth_m": (
                        COURSE_MAXIMUM_INITIALIZATION_DEPTH_M
                    ),
                    "candidates": metrics,
                },
            )

        selected = select_first_pose_prior_candidate(outcomes)
        comparison = {
            "schema": "exercise5-i03a-prior-comparison-v1",
            "status": (
                "candidate-selected"
                if selected is not None
                else "neither-candidate-eligible"
            ),
            "window": [start, end],
            "database_load_count": 1,
            "projection_sigmas_px": list(
                COURSE_STEREO_PROJECTION_SIGMAS_PX
            ),
            "maximum_initialization_depth_m": (
                COURSE_MAXIMUM_INITIALIZATION_DEPTH_M
            ),
            "selection_rule": (
                "eligible iff standard Marginals and all covariance gates "
                "pass; select sole eligible candidate; select A if both pass"
            ),
            "selected_candidate": selected,
            "candidates": metrics,
        }
        _write_json(output_root / "candidate_comparison.json", comparison)
        history.append(
            {
                "event": "candidate-comparison-complete",
                "selected_candidate": selected,
                "candidate_eligibility": {
                    name: value["eligible"]
                    for name, value in outcomes.items()
                },
                "utc": utc_now(),
            }
        )
        if selected is None:
            _write_json(
                output_root / "failure.json",
                {
                    "schema": "exercise5-i03a-bounded-failure-v1",
                    "status": "neither-candidate-eligible",
                    "candidate_prior_sigmas": {
                        name: list(sigmas)
                        for name, sigmas in FIRST_POSE_PRIOR_CANDIDATES.items()
                    },
                    "candidate_marginals": {
                        name: metrics[name]["marginals"]
                        for name in ("A", "B")
                    },
                    "candidate_covariance_validation": {
                        name: metrics[name]["covariance_validation"]
                        for name in ("A", "B")
                    },
                    "q516": {
                        name: metrics[name]["q516"]
                        for name in ("A", "B")
                    },
                    "factor_counts": {
                        name: metrics[name]["graph_counts"]
                        for name in ("A", "B")
                    },
                },
            )
            update_run_state(
                output_root,
                status="failed-neither-candidate-eligible",
                expected_window_count=355,
                completed_window_indices=[],
                history=history,
                scope="bounded-probe",
            )
            return comparison

        update_run_state(
            output_root,
            status="awaiting-selected-configuration-freeze",
            expected_window_count=355,
            completed_window_indices=[],
            history=history,
            scope="bounded-probe",
        )
        print(
            json.dumps(
                {
                    "event": "selected-candidate-awaiting-freeze",
                    "selected_candidate": selected,
                    "comparison_path": str(
                        output_root / "candidate_comparison.json"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        input(
            "Freeze the selected configuration and affected source hash, "
            "then press Enter to build the final sealed artifact: "
        )

        final_configuration, final_config_sha = load_frozen_configuration(
            configuration_path
        )
        final_prior = final_configuration["scientific_configuration"][
            "gauge_fixing_prior"
        ]
        final_eligibility = final_configuration[
            "scientific_configuration"
        ]["landmark_eligibility"]
        if (
            final_prior.get("selected_candidate") != selected
            or final_eligibility.get("maximum_initialization_depth_m")
            != COURSE_MAXIMUM_INITIALIZATION_DEPTH_M
        ):
            raise Exercise5ContractError(
                "frozen selected candidate/depth does not match comparison"
            )
        final_source_hashes = _validate_source_hashes(
            repository_root, final_configuration
        )
        if final_source_hashes != source_identity["source_hashes"]:
            raise Exercise5ContractError(
                "source identity changed during bounded comparison"
            )
        if old_bundle_hash == current_bundle_hash:
            raise Exercise5ContractError(
                "comparison did not observe the authorized Bundle amendment"
            )

        snapshot_configuration(
            output_root, final_configuration, final_config_sha
        )
        write_generator_registry(output_root, final_configuration)
        write_and_seal_manifest(
            output_root,
            configuration=final_configuration,
            config_sha256=final_config_sha,
            source_identity=source_identity,
            completed_window_indices=[],
            status="running",
            scope="bounded-probe",
        )

        selected_started = perf_counter()
        selected_result = solve_bundle_window(
            database=database,
            world_to_camera_poses=poses,
            calibration=calibration,
            start_frame_id=start,
            end_frame_id=end,
            projection_noise=projection_noise,
            anchor_noise=gtsam.noiseModel.Diagonal.Sigmas(
                np.asarray(
                    FIRST_POSE_PRIOR_CANDIDATES[selected], dtype=float
                )
            ),
            landmark_eligibility=eligibility,
            rotation_atol=rotation_atol,
        )
        if (
            _bundle_solution_sha256(selected_result)
            != solution_sha_by_candidate[selected]
        ):
            raise ArtifactValidationError(
                "selected rebuild differs from compared optimized result"
            )
        selected_runtime = perf_counter() - selected_started
        sealed = seal_window_artifacts(
            windows_root=output_root / "windows",
            quarantine_root=output_root / "quarantine",
            result=selected_result,
            database=database,
            calibration=calibration,
            world_to_camera_poses=poses,
            ground_truth_poses=ground_truth,
            configuration=final_configuration,
            config_sha256=final_config_sha,
            source_identity_sha256=identity_sha,
            window_index=0,
            endpoint_covariances=covariance_by_candidate[selected],
            first_window_camera_marginals=(
                camera_marginals_by_candidate[selected]
            ),
            runtime_seconds=selected_runtime,
            peak_rss_bytes=_peak_rss_bytes(),
            image_root=repository_root / "dataset" / "sequences" / "00",
        )
        validate_sealed_window(
            Path(sealed["directory"]),
            expected_index=0,
            expected_start=start,
            expected_end=end,
            config_sha256=final_config_sha,
            source_identity_sha256=identity_sha,
        )
        del selected_result
        gc.collect()
        write_and_seal_manifest(
            output_root,
            configuration=final_configuration,
            config_sha256=final_config_sha,
            source_identity=source_identity,
            completed_window_indices=[0],
            status="bounded-probe-complete",
            scope="bounded-probe",
        )

        from exercise5_aggregation import aggregate_all
        from exercise5_reporting import generate_report_figures
        from exercise5_submission import materialize_bundle_package

        aggregate_paths = aggregate_all(
            output_root,
            repository_root=repository_root,
            configuration_path=configuration_path,
            allow_partial=True,
        )
        report_paths = generate_report_figures(output_root)
        try:
            materialize_bundle_package(
                output_root,
                output_root / "rejected_partial_submission",
                configuration_path=configuration_path,
            )
        except ArtifactValidationError:
            submission_gate = "partial-source-rejected"
        else:
            raise ArtifactValidationError(
                "partial bounded source was accepted for submission"
            )
        validation = validate_run_root(
            output_root,
            configuration_path=configuration_path,
            allow_partial=True,
        )
        history.append(
            {
                "event": "selected-window-sealed-and-validated",
                "selected_candidate": selected,
                "directory": sealed["directory"],
                "utc": utc_now(),
            }
        )
        update_run_state(
            output_root,
            status="bounded-probe-complete",
            expected_window_count=355,
            completed_window_indices=[0],
            history=history,
            scope="bounded-probe",
        )
        result_summary = {
            "status": "bounded-probe-complete",
            "output_root": str(output_root),
            "database_load_count": 1,
            "database_load_seconds": database_load_seconds,
            "selected_candidate": selected,
            "candidate_comparison": comparison,
            "configuration_sha256": final_config_sha,
            "source_input_identity_sha256": identity_sha,
            "selected_rebuild_seconds": selected_runtime,
            "process_runtime_seconds": perf_counter() - process_started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "memory": {
                "preflight": preflight,
                "before_database_load": pre_load_memory,
                "after_database_load": post_load_guard,
            },
            "sealed_window": sealed,
            "aggregate_tables": [str(path) for path in aggregate_paths],
            "report_figures": [str(path) for path in report_paths],
            "submission_materializer_gate": submission_gate,
            "reload_validation": validation,
        }
        _write_json(output_root / "probe_result.json", result_summary)
        checksum_path = write_checksum_index(output_root)
        result_summary["checksum_index"] = str(checksum_path)
        return result_summary
    finally:
        del database
        gc.collect()


# Canonical streamed Exercise 5 execution path


def run_streamed(
    *,
    output_root: Path,
    repository_root: Path = ROOT,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
    maximum_windows: Optional[int] = None,
    canonical_confirmation: Optional[str] = None,
    bounded_probe: bool = False,
    experimental_policy: Optional[str] = None,
) -> Dict[str, object]:
    """Stream consecutive frozen keyframe windows through one process.

    The canonical E4 database is loaded once.  Each window is solved,
    marginalized, reduced to endpoint evidence, sealed, and released before
    the next window, so at most one Bundle graph and Marginals object are live.
    """

    repository_root = Path(repository_root).resolve()
    output_root = Path(output_root).resolve()
    configuration, config_sha = load_run_configuration(configuration_path)
    expected_count = configuration["keyframe_selection"]["statistics"][
        "window_count"
    ]
    is_experiment = experimental_policy is not None
    if is_experiment:
        from exercise5_track_survival import (
            EXPERIMENT_CONFIRMATION,
            EXPERIMENT_OUTPUT_ROOT,
            EXPERIMENT_RULE_ID,
            EXPERIMENT_SCOPE,
        )

        configured_rule = (
            configuration.get("experiment_contract", {})
            .get("keyframe_selection", {})
            .get("rule_id")
        )
        if (
            experimental_policy != EXPERIMENT_RULE_ID
            or configured_rule != EXPERIMENT_RULE_ID
        ):
            raise Exercise5ContractError(
                "only the frozen track-survival-082 experiment is supported"
            )
        if output_root != EXPERIMENT_OUTPUT_ROOT.resolve():
            raise ValueError(
                "experimental output root must be {}".format(
                    EXPERIMENT_OUTPUT_ROOT
                )
            )
        if canonical_confirmation != EXPERIMENT_CONFIRMATION:
            raise PermissionError(
                "experimental run requires --confirm {}".format(
                    EXPERIMENT_CONFIRMATION
                )
            )
        if maximum_windows is not None or bounded_probe:
            raise ValueError(
                "experimental command must run/resume every configured window"
            )
        scope = EXPERIMENT_SCOPE
    else:
        scope = "bounded-probe" if bounded_probe else "canonical"
    if bounded_probe:
        if (
            output_root.parent != Path("/tmp")
            or not output_root.name.startswith("e5_i03a_")
            or maximum_windows != 1
        ):
            raise ValueError(
                "bounded probe requires a new /tmp/e5_i03a_* root and one window"
            )
        if output_root.exists():
            raise FileExistsError(
                "bounded probe output root must be new: {}".format(output_root)
            )
    else:
        if (
            not is_experiment
            and canonical_confirmation != CANONICAL_CONFIRMATION
        ):
            raise PermissionError(
                "canonical run requires --confirm {}".format(
                    CANONICAL_CONFIRMATION
                )
            )
        if not is_experiment and maximum_windows is not None:
            raise ValueError("canonical command cannot reduce the 355 windows")
    if output_root.exists() and not output_root.is_dir():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "windows",
        "quarantine",
        "aggregate/data",
        "aggregate/figures",
        "track_5_1",
        "provenance",
        "logs",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    process_started = perf_counter()
    preflight_memory = _enforce_memory_guard("before-canonical-input-validation")
    source_identity = validate_source_and_input_hashes(
        repository_root, configuration, hash_database=True
    )
    identity_sha = _source_identity_hash(source_identity)
    snapshot_configuration(output_root, configuration, config_sha)
    write_generator_registry(output_root, configuration)
    stale_quarantines = _quarantine_stale_attempts(
        output_root / "windows", output_root / "quarantine"
    )
    history = [
        {
            "event": "strict-preflight-memory-gate",
            "evidence": preflight_memory,
            "utc": utc_now(),
        }
    ]
    history.extend(
        {
            "event": "stale-incomplete-attempt-quarantined",
            "directory": path,
            "utc": utc_now(),
        }
        for path in stale_quarantines
    )
    completed = []
    keyframes = configuration["keyframe_selection"]["ordered_keyframe_ids"]
    limit = expected_count if maximum_windows is None else maximum_windows
    write_and_seal_manifest(
        output_root,
        configuration=configuration,
        config_sha256=config_sha,
        source_identity=source_identity,
        completed_window_indices=completed,
        status="running",
        scope=scope,
    )
    update_run_state(
        output_root,
        status="running",
        expected_window_count=expected_count,
        completed_window_indices=completed,
        history=history,
        scope=scope,
    )
    calibration = _read_calibration(
        repository_root
        / configuration["canonical_inputs"]["calibration"]["path"],
        configuration,
    )
    ground_truth = _read_ground_truth(
        repository_root
        / configuration["canonical_inputs"]["ground_truth"]["path"]
    )
    pre_load_memory = _enforce_memory_guard(
        "before-canonical-database-load"
    )
    history.append(
        {
            "event": "strict-pre-load-memory-gate",
            "evidence": pre_load_memory,
            "utc": utc_now(),
        }
    )
    update_run_state(
        output_root,
        status="running",
        expected_window_count=expected_count,
        completed_window_indices=completed,
        history=history,
        scope=scope,
    )
    load_started = perf_counter()
    # Deserialize the large E4 TrackingDB exactly once for the entire stream.
    database, poses, normalized = _load_canonical_once(
        repository_root, configuration
    )
    load_seconds = perf_counter() - load_started
    del normalized
    production_memory_policy = None
    if not bounded_probe:
        gc.collect()
        try:
            production_memory_policy = production_post_load_memory_policy(
                post_load_baseline_rss_bytes=_current_rss_bytes()
            )
        except BaseException as exception:
            history.append(
                {
                    "event": "production-post-load-memory-policy-failed",
                    "exception_type": type(exception).__name__,
                    "exception_message": str(exception),
                    "utc": utc_now(),
                }
            )
            update_run_state(
                output_root,
                status="failed",
                expected_window_count=expected_count,
                completed_window_indices=completed,
                history=history,
                scope=scope,
            )
            write_and_seal_manifest(
                output_root,
                configuration=configuration,
                config_sha256=config_sha,
                source_identity=source_identity,
                completed_window_indices=completed,
                status="failed",
                scope=scope,
            )
            del database
            gc.collect()
            raise
        history.append(
            {
                "event": "production-post-load-memory-policy-fixed",
                "evidence": production_memory_policy,
                "utc": utc_now(),
            }
        )
        update_run_state(
            output_root,
            status="running",
            expected_window_count=expected_count,
            completed_window_indices=completed,
            history=history,
            scope=scope,
        )
    projection_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            configuration["scientific_configuration"][
                "measurement_uncertainty"
            ]["stereo_sigma_px"],
            dtype=float,
        )
    )
    prior = configuration["scientific_configuration"]["gauge_fixing_prior"]
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.asarray(
            prior["rotation_sigma_rad"] + prior["translation_sigma_m"],
            dtype=float,
        )
    )
    eligibility_config = configuration["scientific_configuration"][
        "landmark_eligibility"
    ]
    eligibility = ProbeLandmarkEligibility(
        minimum_valid_observations=eligibility_config[
            "minimum_valid_observations"
        ],
        minimum_distinct_frames=eligibility_config[
            "minimum_distinct_frames"
        ],
        maximum_initialization_depth_m=eligibility_config.get(
            "maximum_initialization_depth_m"
        ),
    )
    rotation_atol = configuration["scientific_configuration"][
        "validation_tolerances"
    ]["rotation_orthogonality_max_abs"]
    track_diagnostic = None
    if not bounded_probe:
        track_diagnostic = write_track_diagnostic(
            database=database,
            world_to_camera_poses=poses,
            calibration=calibration,
            output_root=output_root,
            configuration=configuration,
        )
    skipped = 0
    try:
        for index in range(limit):
            # Production endpoints come only from adjacent entries in the
            # frozen ordered keyframe population; no runtime selection occurs.
            start, end = keyframes[index], keyframes[index + 1]
            final_directory = output_root / "windows" / _window_name(
                index, start, end
            )
            if final_directory.exists():
                try:
                    validate_sealed_window(
                        final_directory,
                        expected_index=index,
                        expected_start=start,
                        expected_end=end,
                        config_sha256=config_sha,
                        source_identity_sha256=identity_sha,
                    )
                except BaseException as exception:
                    quarantine_attempt(
                        final_directory,
                        output_root / "quarantine",
                        exception=exception,
                    )
                    raise ArtifactValidationError(
                        "invalid existing window was quarantined; review "
                        "before retrying index {}".format(index)
                    ) from exception
                completed.append(index)
                skipped += 1
                history.append(
                    {
                        "event": "resume-skip-valid-sealed-window",
                        "window_index": index,
                        "utc": utc_now(),
                    }
                )
                continue
            if bounded_probe:
                snapshot = _memory_snapshot()
                memory_before_window = (
                    bounded_probe_post_load_memory_evidence(
                        current_process_rss_bytes=_current_rss_bytes(),
                        mem_available_bytes=snapshot["mem_available_bytes"],
                        swap_free_bytes=snapshot["swap_free_bytes"],
                    )
                )
                history.append(
                    {
                        "event": "bounded-probe-post-load-memory-guard",
                        "window_index": index,
                        "evidence": memory_before_window,
                        "utc": utc_now(),
                    }
                )
                update_run_state(
                    output_root,
                    status="running",
                    expected_window_count=expected_count,
                    completed_window_indices=completed,
                    history=history,
                    scope=scope,
                )
                if not memory_before_window["passed"]:
                    raise MemoryError(
                        "bounded probe post-load memory guard failed: {}".format(
                            json.dumps(
                                memory_before_window, sort_keys=True
                            )
                        )
                    )
            else:
                snapshot = _memory_snapshot()
                memory_before_window = production_post_load_memory_evidence(
                    fixed_policy=production_memory_policy,
                    current_process_rss_bytes=_current_rss_bytes(),
                    mem_available_bytes=snapshot["mem_available_bytes"],
                    swap_free_bytes=snapshot["swap_free_bytes"],
                )
                history.append(
                    {
                        "event": "production-post-load-memory-admission",
                        "window_index": index,
                        "evidence": memory_before_window,
                        "utc": utc_now(),
                    }
                )
                update_run_state(
                    output_root,
                    status="running",
                    expected_window_count=expected_count,
                    completed_window_indices=completed,
                    history=history,
                    scope=scope,
                )
                if not memory_before_window["passed"]:
                    raise MemoryError(
                        "production post-load memory admission failed: {}".format(
                            json.dumps(memory_before_window, sort_keys=True)
                        )
                    )
            window_started = perf_counter()
            # Build and optimize one first-frame-local Bundle graph.
            result = solve_bundle_window(
                database=database,
                world_to_camera_poses=poses,
                calibration=calibration,
                start_frame_id=start,
                end_frame_id=end,
                projection_noise=projection_noise,
                anchor_noise=anchor_noise,
                landmark_eligibility=eligibility,
                rotation_atol=rotation_atol,
            )
            # Marginals exist only long enough to retain the ordered endpoint
            # joint/conditional covariance and window-zero camera evidence.
            marginals = gtsam.Marginals(result.graph, result.final_values)
            endpoint_covariances = extract_endpoint_covariances(
                result, marginals, configuration
            )
            camera_marginals = (
                extract_first_window_camera_marginals(
                    result, marginals, configuration
                )
                if index == 0
                else None
            )
            del marginals
            window_seconds = perf_counter() - window_started
            # Persist and independently reload the compact result before any
            # graph-bearing object is released.
            sealed = seal_window_artifacts(
                windows_root=output_root / "windows",
                quarantine_root=output_root / "quarantine",
                result=result,
                database=database,
                calibration=calibration,
                world_to_camera_poses=poses,
                ground_truth_poses=ground_truth,
                configuration=configuration,
                config_sha256=config_sha,
                source_identity_sha256=identity_sha,
                window_index=index,
                endpoint_covariances=endpoint_covariances,
                first_window_camera_marginals=camera_marginals,
                runtime_seconds=window_seconds,
                peak_rss_bytes=_peak_rss_bytes(),
                image_root=(
                    repository_root / "dataset" / "sequences" / "00"
                ),
            )
            completed.append(index)
            history.append(
                {
                    "event": "window-sealed",
                    "window_index": index,
                    "directory": sealed["directory"],
                    "memory_before_window": memory_before_window,
                    "utc": utc_now(),
                }
            )
            # Enforce the one-live-window memory contract before continuing.
            del sealed
            del camera_marginals
            del endpoint_covariances
            del result
            gc.collect()
            update_run_state(
                output_root,
                status="running",
                expected_window_count=expected_count,
                completed_window_indices=completed,
                history=history,
                scope=scope,
            )
    except BaseException as exception:
        history.append(
            {
                "event": "run-failed",
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "utc": utc_now(),
            }
        )
        update_run_state(
            output_root,
            status="failed",
            expected_window_count=expected_count,
            completed_window_indices=completed,
            history=history,
            scope=scope,
        )
        write_and_seal_manifest(
            output_root,
            configuration=configuration,
            config_sha256=config_sha,
            source_identity=source_identity,
            completed_window_indices=completed,
            status="failed",
            scope=scope,
        )
        raise
    finally:
        del database
        gc.collect()
    run_status = (
        "bounded-probe-complete"
        if bounded_probe and completed == [0]
        else "windows-complete-awaiting-aggregation"
        if len(completed) == expected_count
        else "running"
    )
    update_run_state(
        output_root,
        status=run_status,
        expected_window_count=expected_count,
        completed_window_indices=completed,
        history=history,
        scope=scope,
    )
    write_and_seal_manifest(
        output_root,
        configuration=configuration,
        config_sha256=config_sha,
        source_identity=source_identity,
        completed_window_indices=completed,
        status=run_status,
        scope=scope,
    )
    return {
        "status": run_status,
        "output_root": str(output_root),
        "database_load_count": 1,
        "database_load_seconds": load_seconds,
        "memory_preflight": preflight_memory,
        "memory_before_database_load": pre_load_memory,
        "post_load_memory_policy": production_memory_policy,
        "completed_window_indices": completed,
        "resume_skipped_window_count": skipped,
        "process_runtime_seconds": perf_counter() - process_started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "configuration_sha256": config_sha,
        "source_input_identity_sha256": identity_sha,
        "track_diagnostic": track_diagnostic,
    }


# Sealed-run validation


def validate_run_root(
    output_root: Path,
    *,
    configuration_path: Path = DEFAULT_CONFIG_PATH,
    allow_partial: bool = False,
) -> Dict[str, object]:
    """Strictly reload every declared sealed window under its run identity.

    Existing canonical roots remain bound to the unchanged canonical-run
    configuration SHA and manifest source identity; current reader-source SHAs
    are validated separately by :func:`validate_source_and_input_hashes`.
    """

    configuration, config_sha = load_run_configuration(configuration_path)
    manifest = _read_json(Path(output_root) / "manifest.json")
    identity_sha = manifest["source_input_identity_sha256"]
    keyframes = configuration["keyframe_selection"]["ordered_keyframe_ids"]
    completed = manifest["completed_window_indices"]
    if not allow_partial and manifest["status"] != "complete-and-sealed":
        raise ArtifactValidationError("run root is not complete-and-sealed")
    validated = []
    for index in completed:
        start, end = keyframes[index], keyframes[index + 1]
        validate_sealed_window(
            Path(output_root)
            / "windows"
            / _window_name(index, start, end),
            expected_index=index,
            expected_start=start,
            expected_end=end,
            config_sha256=config_sha,
            source_identity_sha256=identity_sha,
        )
        validated.append(index)
    if not allow_partial and validated != list(range(len(keyframes) - 1)):
        raise ArtifactValidationError("complete run omits windows")
    return {
        "status": manifest["status"],
        "validated_window_indices": validated,
        "configuration_sha256": config_sha,
    }


# Exercise 5.1 track diagnostic and compatibility surface


def write_first_window_diagnostic(*args, **kwargs):
    """Producer-registry alias retained as a named, tested implementation."""

    return _maximum_diagnostic_json(*args, **kwargs)


def write_track_diagnostic(
    *,
    database: object,
    world_to_camera_poses: Sequence[np.ndarray],
    calibration: gtsam.Cal3_S2Stereo,
    output_root: Path,
    configuration: Mapping[str, object],
) -> Dict[str, object]:
    """Write the frozen Exercise 5.1 TrackId evidence without a Bundle graph."""

    contract = configuration["exercise5_track_diagnostic"]
    track_id = int(contract["track_id"])
    expected_frames = list(contract["ordered_frame_ids"])
    observed_frames = [int(frame) for frame in database.frames(track_id)]
    if observed_frames != expected_frames:
        raise Exercise5ContractError("frozen Exercise 5.1 track membership differs")
    measurements = []
    for frame_id in observed_frames:
        link = database.link(frame_id, track_id)
        measurement = np.asarray(
            [link.x_left, link.x_right, link.y], dtype=float
        )
        if (
            not np.all(np.isfinite(measurement))
            or measurement[0] <= measurement[1]
        ):
            raise Exercise5ContractError(
                "frozen Exercise 5.1 track has an invalid observation"
            )
        measurements.append(measurement)
    latest_frame = observed_frames[-1]
    latest_pose = gtsam.Pose3(
        np.linalg.inv(_homogeneous_extrinsic(world_to_camera_poses[latest_frame]))
    )
    point = np.asarray(
        gtsam.StereoCamera(latest_pose, calibration).backproject(
            gtsam.StereoPoint2(*measurements[-1])
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(point)):
        raise Exercise5ContractError("track triangulation is nonfinite")
    rows = []
    for frame_id, measurement in zip(observed_frames, measurements):
        camera_pose = gtsam.Pose3(
            np.linalg.inv(_homogeneous_extrinsic(world_to_camera_poses[frame_id]))
        )
        camera_point = np.asarray(camera_pose.transformTo(point), dtype=float)
        if not np.isfinite(camera_point[2]) or camera_point[2] <= 0.0:
            raise Exercise5ContractError(
                "track triangulation has a nonpositive projection depth"
            )
        projection = np.asarray(
            gtsam.StereoCamera(camera_pose, calibration).project(point).vector(),
            dtype=float,
        )
        residual = projection - measurement
        left = float(np.linalg.norm(residual[[0, 2]]))
        right = float(np.linalg.norm(residual[[1, 2]]))
        stereo = float(np.linalg.norm(residual))
        rows.append(
            {
                "frame_id": frame_id,
                "track_id": track_id,
                "measurement_u_left_px": measurement[0],
                "measurement_u_right_px": measurement[1],
                "measurement_v_px": measurement[2],
                "projection_u_left_px": projection[0],
                "projection_u_right_px": projection[1],
                "projection_v_px": projection[2],
                "residual_u_left_px": residual[0],
                "residual_u_right_px": residual[1],
                "residual_v_px": residual[2],
                "left_pixel_distance_px": left,
                "right_pixel_distance_px": right,
                "mean_left_right_pixel_distance_px": 0.5 * (left + right),
                "stereo_pixel_distance_px": stereo,
                "factor_error": 0.5 * stereo * stereo,
                "depth_m": camera_point[2],
            }
        )
    output = Path(output_root) / "track_5_1"
    output.mkdir(parents=True, exist_ok=True)
    residual_path = output / "residuals.csv"
    _write_csv(residual_path, tuple(rows[0]), rows)
    track = {
        "schema": "exercise5-track-5-1-v1",
        "track_id": track_id,
        "track_length": len(observed_frames),
        "ordered_frame_ids": observed_frames,
        "triangulation_frame_id": latest_frame,
        "triangulated_world_point_m": point.tolist(),
        "measurement_order": ["u_left_px", "u_right_px", "v_px"],
        "measurement_covariance_px2": configuration[
            "scientific_configuration"
        ]["measurement_uncertainty"]["stereo_covariance_px2"],
        "factor_error_formula": "0.5 * residual^T * I_3^-1 * residual",
    }
    _write_json(output / "track.json", track)
    from exercise5_reporting import plot_track_diagnostic

    plot_track_diagnostic(residual_path, output)
    return {
        "track_json": str(output / "track.json"),
        "residuals_csv": str(residual_path),
        "row_count": len(rows),
    }


def _homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    return np.vstack(
        (
            np.asarray(extrinsic, dtype=float),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        )
    )


__all__ = [
    "ArtifactValidationError",
    "CANONICAL_CONFIRMATION",
    "COURSE_MAXIMUM_INITIALIZATION_DEPTH_M",
    "COURSE_STEREO_PROJECTION_SIGMAS_PX",
    "CovarianceValidationError",
    "Exercise5ContractError",
    "FIRST_POSE_PRIOR_CANDIDATES",
    "bounded_probe_post_load_memory_evidence",
    "build_run_manifest",
    "evaluate_standard_marginals_candidate",
    "extract_endpoint_covariances",
    "extract_endpoint_edge",
    "extract_first_window_camera_marginals",
    "load_frozen_configuration",
    "load_run_configuration",
    "production_post_load_memory_evidence",
    "production_post_load_memory_policy",
    "quarantine_attempt",
    "run_bounded_two_candidate_probe",
    "run_streamed",
    "seal_window_artifacts",
    "select_first_pose_prior_candidate",
    "sha256_file",
    "snapshot_configuration",
    "update_run_state",
    "validate_covariance_evidence",
    "validate_frozen_configuration",
    "validate_prepared_window",
    "validate_run_root",
    "validate_sealed_window",
    "validate_source_and_input_hashes",
    "write_and_seal_manifest",
    "write_checksum_index",
    "write_first_window_diagnostic",
    "write_generator_registry",
    "write_track_diagnostic",
]
