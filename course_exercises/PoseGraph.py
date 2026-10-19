"""Consecutive-keyframe Pose Graph construction for Exercise 6.

This module consumes only sealed Exercise 5 endpoint artifacts.  It does not
load a TrackingDB, reconstruct a Bundle window, or perform Loop Closure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parent
CANONICAL_INPUT_ROOT = (
    REPOSITORY_ROOT / "diagnostics" / "ex5_clean_final"
)
EXPECTED_KEYFRAME_COUNT = 356
EXPECTED_EDGE_COUNT = 355
EDGE_DIRECTION = "T_Cstart_from_Cend"
TANGENT_ORDER = (
    "rot_x_rad",
    "rot_y_rad",
    "rot_z_rad",
    "trans_x_m",
    "trans_y_m",
    "trans_z_m",
)
CONDITIONAL_COVARIANCE_FIELD = (
    "conditional_end_given_start_covariance"
)


class CanonicalPoseGraphArtifactError(RuntimeError):
    """Raised when selected E5 inputs violate their sealed contract."""


@dataclass(frozen=True)
class ConsecutivePoseGraphEdge:
    """One canonical consecutive-keyframe measurement and raw covariance."""

    window_index: int
    start_frame_id: int
    end_frame_id: int
    relative_pose: gtsam.Pose3
    covariance: np.ndarray
    source_directory: Path


@dataclass(frozen=True)
class CanonicalPoseGraphInputs:
    """Small, read-only sealed Exercise 5 contract needed by Exercise 6."""

    input_root: Path
    keyframe_ids: Tuple[int, ...]
    edges: Tuple[ConsecutivePoseGraphEdge, ...]
    prior_sigmas: np.ndarray
    validation_tolerances: Mapping[str, float]
    first_bundle_frame_ids: np.ndarray
    first_bundle_poses: np.ndarray
    first_bundle_marginal_covariances: np.ndarray
    ground_truth_path: Path
    configuration_sha256: str


def camera_symbol(frame_id: int) -> int:
    """Return the E6 node key for an actual dataset FrameId."""

    return gtsam.symbol("c", int(frame_id))


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanonicalPoseGraphArtifactError(message)


def _validate_rigid_pose_matrix(
    matrix: np.ndarray, tolerances: Mapping[str, float], *, name: str
) -> None:
    _require(matrix.shape == (4, 4), "{} must be 4x4".format(name))
    _require(np.all(np.isfinite(matrix)), "{} must be finite".format(name))
    rotation = matrix[:3, :3]
    _require(
        np.max(np.abs(matrix[3] - np.array([0.0, 0.0, 0.0, 1.0])))
        <= tolerances["homogeneous_matrix_max_abs"],
        "{} has an invalid homogeneous row".format(name),
    )
    _require(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        <= tolerances["rotation_orthogonality_max_abs"],
        "{} has a non-orthogonal rotation".format(name),
    )
    _require(
        abs(float(np.linalg.det(rotation)) - 1.0)
        <= tolerances["rotation_determinant_abs"],
        "{} has an invalid rotation determinant".format(name),
    )


def load_canonical_pose_graph_inputs(
    input_root: Path = CANONICAL_INPUT_ROOT,
) -> CanonicalPoseGraphInputs:
    """Load the sealed Exercise 5 data required by Exercise 6.

    The selected root must identify itself through its own manifest and
    embedded configuration contract. Ordering, seals, edge direction,
    symbols, and array shapes are checked without rehashing the multi-gigabyte
    artifact tree.
    """

    selected_root = Path(input_root).resolve()
    manifest_path = selected_root / "manifest.json"
    configuration_path = selected_root / "configuration.json"
    _require(
        manifest_path.is_file(), "selected E5 manifest.json is missing"
    )
    _require(
        configuration_path.is_file(),
        "selected E5 configuration.json is missing",
    )
    manifest = _read_json(manifest_path)
    snapshot = _read_json(configuration_path)
    _require(
        manifest.get("schema") == "exercise5-run-manifest-v1"
        and manifest.get("schema_version") == 1,
        "selected E5 manifest schema differs",
    )
    _require(
        manifest.get("status") == "complete-and-sealed"
        and manifest.get("scope") == "canonical",
        "selected E5 root is not a complete-and-sealed canonical run",
    )
    _require(
        snapshot.get("schema") == "exercise5-configuration-snapshot-v1",
        "selected E5 configuration snapshot schema differs",
    )
    configuration = snapshot.get("configuration")
    _require(
        isinstance(configuration, dict)
        and configuration.get("schema") == "exercise5-production-config",
        "embedded production configuration is missing",
    )
    artifact_contract = configuration.get("artifact_contract")
    _require(
        isinstance(artifact_contract, dict),
        "embedded E5 artifact contract is missing",
    )
    declared_root = artifact_contract.get("canonical_root")
    _require(
        isinstance(declared_root, str)
        and (REPOSITORY_ROOT / declared_root).resolve() == selected_root
        and artifact_contract.get("canonical_status")
        == "complete-and-sealed",
        "configuration does not identify the selected sealed E5 root",
    )
    _require(
        snapshot.get("source_sha256")
        == manifest.get("configuration_sha256"),
        "configuration identity differs from the canonical manifest",
    )

    keyframe_ids = tuple(
        int(value) for value in manifest["ordered_keyframe_ids"]
    )
    configured_keyframes = tuple(
        int(value)
        for value in configuration["keyframe_selection"][
            "ordered_keyframe_ids"
        ]
    )
    _require(
        len(keyframe_ids) == EXPECTED_KEYFRAME_COUNT,
        "canonical keyframe count must be exactly 356",
    )
    _require(
        len(set(keyframe_ids)) == EXPECTED_KEYFRAME_COUNT
        and all(
            second > first
            for first, second in zip(keyframe_ids, keyframe_ids[1:])
        ),
        "canonical keyframes must be ordered and unique",
    )
    _require(
        keyframe_ids[0] == 0
        and keyframe_ids[-1] == 3359
        and keyframe_ids == configured_keyframes,
        "canonical keyframe endpoints or configuration ordering differ",
    )
    _require(
        manifest.get("expected_window_count") == EXPECTED_EDGE_COUNT
        and manifest.get("completed_window_indices")
        == list(range(EXPECTED_EDGE_COUNT)),
        "canonical completion list must contain windows 0 through 354",
    )

    coordinate_contract = manifest["coordinate_contract"]
    _require(
        coordinate_contract.get("tangent_order") == list(TANGENT_ORDER),
        "Pose3 tangent order differs from the E5-I04 contract",
    )
    _require(
        "T_Ca_from_Cb" in coordinate_contract.get("edge_measurement", ""),
        "canonical endpoint-edge direction is missing",
    )
    prior = manifest["scientific_configuration"]["gauge_fixing_prior"]
    prior_sigmas = np.asarray(
        prior["rotation_sigma_rad"] + prior["translation_sigma_m"],
        dtype=float,
    )
    _require(
        prior.get("factor_count") == 1
        and prior.get("additional_camera_priors") == 0
        and prior.get("landmark_priors") == 0
        and prior.get("target") == "identity Pose3"
        and prior.get("model") == "Diagonal.Sigmas"
        and prior.get("selected_candidate") == "A"
        and prior.get("sigma_order") == list(TANGENT_ORDER),
        "canonical first-pose prior semantics are ambiguous or differ",
    )
    _require(
        prior_sigmas.shape == (6,)
        and np.all(np.isfinite(prior_sigmas))
        and np.array_equal(
            prior_sigmas,
            np.asarray(prior["authorized_candidates"]["A"], dtype=float),
        ),
        "selected Candidate A prior sigmas differ",
    )

    tolerances = manifest["scientific_configuration"][
        "validation_tolerances"
    ]
    observed_directories = tuple(
        sorted(
            path
            for path in (selected_root / "windows").glob("window_[0-9]*")
            if path.is_dir()
        )
    )
    _require(
        len(observed_directories) == EXPECTED_EDGE_COUNT,
        "canonical window directory count must be exactly 355",
    )

    edges = []
    for index, (start_frame, end_frame) in enumerate(
        zip(keyframe_ids, keyframe_ids[1:])
    ):
        directory = (
            selected_root
            / "windows"
            / "window_{:04d}_{}_{}".format(
                index, start_frame, end_frame
            )
        )
        _require(directory.is_dir(), "canonical window directory is missing")
        seal = _read_json(directory / "artifact_hashes.json")
        _require(
            seal.get("schema") == "exercise5-window-artifact-hashes-v1"
            and seal.get("status") == "sealed"
            and seal.get("window_index") == index
            and seal.get("start_frame_id") == start_frame
            and seal.get("end_frame_id") == end_frame
            and seal.get("configuration_sha256")
            == manifest["configuration_sha256"]
            and seal.get("source_input_identity_sha256")
            == manifest["source_input_identity_sha256"],
            "window {} seal identity differs".format(index),
        )
        edge_data = _read_json(directory / "endpoint_edge.json")
        _require(
            edge_data.get("schema") == "exercise5-endpoint-edge-v1"
            and edge_data.get("start_frame_id") == start_frame
            and edge_data.get("end_frame_id") == end_frame
            and edge_data.get("start_camera_key")
            == int(camera_symbol(start_frame))
            and edge_data.get("end_camera_key")
            == int(camera_symbol(end_frame))
            and edge_data.get("direction") == EDGE_DIRECTION,
            "window {} endpoint identity or direction differs".format(index),
        )
        matrix = np.asarray(edge_data.get("matrix"), dtype=float)
        _validate_rigid_pose_matrix(
            matrix,
            tolerances,
            name="window {} endpoint edge".format(index),
        )
        covariance_path = directory / "endpoint_covariances.npz"
        _require(
            covariance_path.is_file(),
            "window {} endpoint covariance is missing".format(index),
        )
        with np.load(covariance_path, allow_pickle=False) as covariance_data:
            _require(
                CONDITIONAL_COVARIANCE_FIELD in covariance_data.files,
                "window {} conditional covariance field is missing".format(
                    index
                ),
            )
            covariance = np.array(
                covariance_data[CONDITIONAL_COVARIANCE_FIELD],
                dtype=float,
                copy=True,
            )
        _require(
            covariance.shape == (6, 6)
            and np.all(np.isfinite(covariance)),
            "window {} conditional covariance must be finite 6x6".format(
                index
            ),
        )
        _require(
            np.max(np.abs(covariance - covariance.T))
            <= tolerances["covariance_symmetry_atol"],
            "window {} conditional covariance is not symmetric".format(
                index
            ),
        )
        covariance.setflags(write=False)
        edges.append(
            ConsecutivePoseGraphEdge(
                window_index=index,
                start_frame_id=start_frame,
                end_frame_id=end_frame,
                relative_pose=gtsam.Pose3(matrix),
                covariance=covariance,
                source_directory=directory,
            )
        )

    first_path = edges[0].source_directory / "camera_marginals.npz"
    _require(
        first_path.is_file(), "first-window camera marginals are missing"
    )
    with np.load(first_path, allow_pickle=False) as first_data:
        _require(
            set(first_data.files)
            == {"frame_ids", "poses", "marginal_covariances"},
            "first-window camera marginal members differ",
        )
        first_frame_ids = np.array(first_data["frame_ids"], copy=True)
        first_poses = np.array(first_data["poses"], dtype=float, copy=True)
        first_covariances = np.array(
            first_data["marginal_covariances"], dtype=float, copy=True
        )
    _require(
        first_frame_ids.ndim == 1
        and first_frame_ids[0] == keyframe_ids[0]
        and first_frame_ids[-1] == keyframe_ids[1]
        and np.array_equal(
            first_frame_ids,
            np.arange(keyframe_ids[0], keyframe_ids[1] + 1),
        )
        and first_poses.shape == (len(first_frame_ids), 4, 4)
        and first_covariances.shape == (len(first_frame_ids), 6, 6)
        and np.all(np.isfinite(first_poses))
        and np.all(np.isfinite(first_covariances)),
        "first-window camera marginal shapes or ordering differ",
    )

    ground_truth_path = (
        REPOSITORY_ROOT
        / configuration["canonical_inputs"]["ground_truth"]["path"]
    ).resolve()
    _require(
        ground_truth_path.is_file(),
        "canonical ground-truth pose file is missing",
    )
    prior_sigmas.setflags(write=False)
    first_frame_ids.setflags(write=False)
    first_poses.setflags(write=False)
    first_covariances.setflags(write=False)
    return CanonicalPoseGraphInputs(
        input_root=selected_root,
        keyframe_ids=keyframe_ids,
        edges=tuple(edges),
        prior_sigmas=prior_sigmas,
        validation_tolerances=tolerances,
        first_bundle_frame_ids=first_frame_ids,
        first_bundle_poses=first_poses,
        first_bundle_marginal_covariances=first_covariances,
        ground_truth_path=ground_truth_path,
        configuration_sha256=manifest["configuration_sha256"],
    )


class PoseGraph:
    """An anchored consecutive-keyframe Pose3 chain."""

    def __init__(
        self,
        *,
        keyframe_ids: Sequence[int],
        relative_poses: Sequence[gtsam.Pose3],
        covariance_matrices: Sequence[np.ndarray],
        prior_sigmas: Sequence[float],
        reconstruction_atol: float = 1e-9,
    ) -> None:
        self.keyframe_ids = tuple(int(value) for value in keyframe_ids)
        self.relative_poses = tuple(relative_poses)
        self.covariance_matrices = tuple(
            np.asarray(value, dtype=float)
            for value in covariance_matrices
        )
        self.prior_sigmas = np.asarray(prior_sigmas, dtype=float)
        self.reconstruction_atol = float(reconstruction_atol)
        if (
            not self.keyframe_ids
            or len(set(self.keyframe_ids)) != len(self.keyframe_ids)
            or any(
                second <= first
                for first, second in zip(
                    self.keyframe_ids, self.keyframe_ids[1:]
                )
            )
        ):
            raise ValueError("keyframe_ids must be ordered and unique")
        if not (
            len(self.relative_poses)
            == len(self.covariance_matrices)
            == len(self.keyframe_ids) - 1
        ):
            raise ValueError(
                "a consecutive chain needs one pose/covariance per edge"
            )
        if self.prior_sigmas.shape != (6,) or not np.all(
            np.isfinite(self.prior_sigmas)
        ):
            raise ValueError("prior_sigmas must be a finite six-vector")

        self._graph = gtsam.NonlinearFactorGraph()
        self._initial_estimate = gtsam.Values()
        self._optimized_values: Optional[gtsam.Values] = None
        self._optimizer: Optional[
            gtsam.LevenbergMarquardtOptimizer
        ] = None
        self._marginals: Optional[gtsam.Marginals] = None
        self.camera_keys = tuple(
            camera_symbol(frame_id) for frame_id in self.keyframe_ids
        )
        self.initial_edge_reconstruction_max_abs = 0.0
        self.create_factor_graph()

    @classmethod
    def from_canonical_inputs(
        cls, inputs: CanonicalPoseGraphInputs
    ) -> "PoseGraph":
        return cls(
            keyframe_ids=inputs.keyframe_ids,
            relative_poses=[
                edge.relative_pose for edge in inputs.edges
            ],
            covariance_matrices=[
                edge.covariance for edge in inputs.edges
            ],
            prior_sigmas=inputs.prior_sigmas,
            reconstruction_atol=inputs.validation_tolerances[
                "numerical_reconstruction_atol"
            ],
        )

    def create_factor_graph(self) -> None:
        """Build one prior and all consecutive BetweenFactorPose3 factors."""

        self._graph = gtsam.NonlinearFactorGraph()
        self._initial_estimate = gtsam.Values()
        self._optimized_values = None
        self._optimizer = None
        self._marginals = None

        identity = gtsam.Pose3()
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            self.prior_sigmas
        )
        self._graph.add(
            gtsam.PriorFactorPose3(
                self.camera_keys[0], identity, prior_noise
            )
        )
        self._initial_estimate.insert(self.camera_keys[0], identity)

        current_pose = identity
        for index, (relative_pose, covariance) in enumerate(
            zip(self.relative_poses, self.covariance_matrices)
        ):
            if covariance.shape != (6, 6) or not np.all(
                np.isfinite(covariance)
            ):
                raise ValueError(
                    "edge {} covariance must be finite 6x6".format(index)
                )
            # E5-I04 established that this raw stored covariance is already
            # in the BetweenFactorPose3 residual tangent basis.
            noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
            self._graph.add(
                gtsam.BetweenFactorPose3(
                    self.camera_keys[index],
                    self.camera_keys[index + 1],
                    relative_pose,
                    noise,
                )
            )
            current_pose = current_pose.compose(relative_pose)
            self._initial_estimate.insert(
                self.camera_keys[index + 1], current_pose
            )

        reconstruction_errors = []
        for index, relative_pose in enumerate(self.relative_poses):
            reconstructed = self._initial_estimate.atPose3(
                self.camera_keys[index]
            ).between(
                self._initial_estimate.atPose3(
                    self.camera_keys[index + 1]
                )
            )
            reconstruction_errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(reconstructed.matrix())
                            - np.asarray(relative_pose.matrix())
                        )
                    )
                )
            )
        self.initial_edge_reconstruction_max_abs = (
            max(reconstruction_errors) if reconstruction_errors else 0.0
        )
        if (
            self.initial_edge_reconstruction_max_abs
            > self.reconstruction_atol
        ):
            raise RuntimeError(
                "composed initial estimates do not reconstruct stored edges"
            )

    def optimize(self) -> gtsam.Values:
        """Run one default GTSAM Levenberg-Marquardt optimization."""

        self._optimizer = gtsam.LevenbergMarquardtOptimizer(
            self._graph, self._initial_estimate
        )
        self._optimized_values = self._optimizer.optimize()
        return self._optimized_values

    def optimize_poseGraph(self, loop: bool = False) -> gtsam.Values:
        """Compatibility spelling retained; E6-I01 rejects Loop Closure use."""

        if loop:
            raise ValueError("Loop Closure is outside E6-I01")
        return self.optimize()

    def compute_marginals(self) -> gtsam.Marginals:
        if self._optimized_values is None:
            raise RuntimeError("optimize the Pose Graph before Marginals")
        self._marginals = gtsam.Marginals(
            self._graph, self._optimized_values
        )
        return self._marginals

    def marginal_covariance_array(self) -> np.ndarray:
        if self._marginals is None:
            raise RuntimeError("construct Marginals before extracting them")
        return np.asarray(
            [
                self._marginals.marginalCovariance(key)
                for key in self.camera_keys
            ],
            dtype=float,
        )

    def optimization_movement(self) -> Tuple[float, float]:
        """Return maximum translation [m] and rotation [rad] movement."""

        if self._optimized_values is None:
            raise RuntimeError("optimize before measuring movement")
        translation = []
        rotation = []
        for key in self.camera_keys:
            initial = self._initial_estimate.atPose3(key)
            optimized = self._optimized_values.atPose3(key)
            translation.append(
                float(
                    np.linalg.norm(
                        np.asarray(optimized.translation())
                        - np.asarray(initial.translation())
                    )
                )
            )
            rotation.append(
                float(
                    np.linalg.norm(
                        np.asarray(
                            gtsam.Rot3.Logmap(
                                initial.rotation().between(
                                    optimized.rotation()
                                )
                            )
                        )
                    )
                )
            )
        return max(translation), max(rotation)

    def graph(self) -> gtsam.NonlinearFactorGraph:
        return self._graph

    def get_initial_estimate(self) -> gtsam.Values:
        return self._initial_estimate

    def get_optimized_values(self) -> gtsam.Values:
        if self._optimized_values is None:
            raise RuntimeError("Pose Graph has not been optimized")
        return self._optimized_values

    def get_initial_cameras(self) -> list:
        return [
            self._initial_estimate.atPose3(key)
            for key in self.camera_keys
        ]

    def get_optimized_cameras(self) -> list:
        values = self.get_optimized_values()
        return [values.atPose3(key) for key in self.camera_keys]

    def get_initial_poses(self) -> list:
        return self.get_initial_cameras()

    def get_optimized_poses(self) -> list:
        return self.get_optimized_cameras()

    def get_cov_matrices(self) -> list:
        return list(self.covariance_matrices)

    def marginals(self, optimized: bool = True) -> gtsam.Marginals:
        if not optimized:
            return gtsam.Marginals(
                self._graph, self._initial_estimate
            )
        if self._marginals is None:
            return self.compute_marginals()
        return self._marginals


__all__ = [
    "CANONICAL_INPUT_ROOT",
    "CONDITIONAL_COVARIANCE_FIELD",
    "CanonicalPoseGraphArtifactError",
    "CanonicalPoseGraphInputs",
    "ConsecutivePoseGraphEdge",
    "EDGE_DIRECTION",
    "EXPECTED_EDGE_COUNT",
    "EXPECTED_KEYFRAME_COUNT",
    "PoseGraph",
    "TANGENT_ORDER",
    "camera_symbol",
    "load_canonical_pose_graph_inputs",
]
