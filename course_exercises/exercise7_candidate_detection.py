"""Exercise 7 Loop Closure candidate detection and proximity scoring.

Candidate detection consumes a selected E6 optimized Pose Graph and the raw
consecutive edge covariances already validated by
``PoseGraph.load_canonical_pose_graph_inputs``.  It implements only Exercise 7
sections 7.1a--7.1b: covariance-path estimation and Mahalanobis scoring.

The default path follows the clean E5 -> clean E6 chain.  Historical E5/E6
pairs remain available through explicit ``e5_root`` and ``e6_root`` arguments;
the loader rejects mixed provenance before candidate scoring begins.

Graph nodes are actual KITTI ``FrameId`` values; keyframe ordinals are only
their positions in the canonical 356-element sequence.  Each directed edge
stores the raw conditional end-given-start covariance associated with a
validated previous-to-current ``BetweenFactorPose3``.  Its GTSAM ``Pose3``
tangent order and units are
``[rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]`` in radians and metres.

The covariance path search intentionally follows the lecturer's matrix-label
Dijkstra-style heuristic.  It directly sums raw 6x6 edge covariances in their
already validated common tangent contract; it applies no scaling, transport,
regularization, or basis change.  The resulting path covariance is the course
approximation of relative-pose uncertainty between the candidate endpoints.
Its queue preference is ``det(Sigma_path)``, evaluated robustly with
``slogdet``.  This is not a graph with additive scalar edge costs, so this
module does not claim that the standard Dijkstra optimality proof applies.

Endpoint proximity is evaluated by an identity-measurement GTSAM
``BetweenFactorPose3`` with the accumulated path covariance.  Because GTSAM
factor error contains the conventional one-half, ``mahalanobis_squared`` is
exactly ``2 * factor_error``.  These functions score and rank candidates only;
temporal exclusion, thresholds, scheduling, matching, and acceptance belong
to the sequential production policy.

E7-I04 reuses the same mathematics through immutable snapshots of live Pose
Graph values and immutable insertion of accepted directed covariance edges.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from PoseGraph import (
    CANONICAL_INPUT_ROOT,
    CONDITIONAL_COVARIANCE_FIELD,
    EDGE_DIRECTION,
    EXPECTED_EDGE_COUNT,
    EXPECTED_KEYFRAME_COUNT,
    TANGENT_ORDER,
    camera_symbol,
    load_canonical_pose_graph_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent
CANONICAL_E6_ROOT = REPOSITORY_ROOT / "diagnostics" / "ex6_clean_final"
COURSE_HEURISTIC_POLICY = "course-accumulated-covariance-determinant"
MAOR_COMPARISON_POLICY = "maor-sum-sqrt-edge-covariance-determinant"


class CandidateDetectionError(RuntimeError):
    """Raised when an E7-I01 input or numerical contract is violated."""


@dataclass(frozen=True)
class CovarianceEdge:
    """One directed previous-to-current edge with raw Pose3 covariance.

    Endpoints are actual FrameIds.  ``covariance`` is a 6x6 conditional
    end-given-start matrix in rotation-then-translation tangent order, with
    radians/metres units and no scientific or display scaling.
    """

    edge_index: int
    start_frame_id: int
    end_frame_id: int
    covariance: np.ndarray
    source_directory: Path


@dataclass(frozen=True)
class CandidateDetectionInputs:
    """Immutable pose snapshot and protected edge inputs for detection.

    ``keyframe_ids`` maps ordinal positions to actual FrameIds, while
    ``frame_id_to_ordinal`` provides the inverse bookkeeping map.  GTSAM keys
    and covariance-graph nodes use the actual FrameIds.
    """

    e6_root: Path
    e5_root: Path
    configuration_sha256: str
    keyframe_ids: Tuple[int, ...]
    frame_id_to_ordinal: Mapping[int, int]
    optimized_pose_matrices: np.ndarray
    edges: Tuple[CovarianceEdge, ...]
    covariance_symmetry_atol: float
    pose_reconstruction_atol: float


@dataclass(frozen=True)
class CovariancePath:
    """A directed path and its directly accumulated raw 6x6 covariance."""

    source_frame_id: int
    target_frame_id: int
    frame_ids: Tuple[int, ...]
    edge_indices: Tuple[int, ...]
    covariance: np.ndarray
    covariance_determinant: float
    covariance_log_determinant: float
    policy: str
    scalar_cost: Optional[float] = None


@dataclass(frozen=True)
class CandidateScore:
    """Unfiltered GTSAM proximity score for one historical/current pair."""

    previous_frame_id: int
    current_frame_id: int
    previous_keyframe_ordinal: int
    current_keyframe_ordinal: int
    path: CovariancePath
    factor_error: float
    mahalanobis_squared: float
    mahalanobis: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateDetectionError(message)


def _read_json(path: Path) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_rigid_pose_matrix(
    matrix: np.ndarray,
    *,
    name: str,
    homogeneous_atol: float,
    rotation_orthogonality_atol: float,
    rotation_determinant_atol: float,
) -> None:
    _require(matrix.shape == (4, 4), "{} must be 4x4".format(name))
    _require(np.all(np.isfinite(matrix)), "{} must be finite".format(name))
    _require(
        np.max(
            np.abs(matrix[3] - np.asarray([0.0, 0.0, 0.0, 1.0]))
        )
        <= homogeneous_atol,
        "{} has an invalid homogeneous row".format(name),
    )
    rotation = matrix[:3, :3]
    _require(
        np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        <= rotation_orthogonality_atol,
        "{} has a non-orthogonal rotation".format(name),
    )
    _require(
        abs(float(np.linalg.det(rotation)) - 1.0)
        <= rotation_determinant_atol,
        "{} has an invalid rotation determinant".format(name),
    )


def load_candidate_detection_inputs(
    e6_root: Path = CANONICAL_E6_ROOT,
    e5_root: Path = CANONICAL_INPUT_ROOT,
) -> CandidateDetectionInputs:
    """Load and cross-check one coherent read-only E5/E6 input pair.

    The clean E5 Bundle root and its clean E6 no-loop Pose Graph are the
    defaults.  The E5 loader remains the authority for manifest, window-seal,
    relative-pose direction, tangent order, and raw covariance validation.
    This function adds only the E7-specific checks that bind those 355 directed
    edges to the 356 saved E6 optimized poses.  Mixed E5/E6 provenance fails
    before scoring; no TrackingDB, Bundle graph, or image data is loaded.
    """

    selected_e6_root = Path(e6_root).resolve()
    selected_e5_root = Path(e5_root).resolve()

    summary_path = selected_e6_root / "summary.json"
    values_path = selected_e6_root / "pose_graph_values.npz"
    _require(summary_path.is_file(), "selected E6 summary.json is missing")
    _require(
        values_path.is_file(), "selected E6 pose_graph_values.npz is missing"
    )
    summary = _read_json(summary_path)
    _require(
        isinstance(summary, dict)
        and summary.get("schema") == "exercise6-pose-graph-summary-v1"
        and summary.get("milestone") == "E6-I01"
        and summary.get("status") == "pass"
        and summary.get("scope")
        == "consecutive keyframes only; no Loop Closure",
        "selected E6 summary is not the protected E6-I01 baseline",
    )
    _require(
        summary.get("keyframe_count") == EXPECTED_KEYFRAME_COUNT
        and summary.get("edge_count") == EXPECTED_EDGE_COUNT
        and summary.get("first_frame_id") == 0
        and summary.get("last_frame_id") == 3359
        and summary.get("node_symbol_policy")
        == "gtsam.symbol('c', actual FrameId)",
        "E6 keyframe, edge, endpoint, or symbol contract differs",
    )
    covariance_summary = summary.get("covariance")
    _require(
        isinstance(covariance_summary, dict)
        and covariance_summary.get("field")
        == CONDITIONAL_COVARIANCE_FIELD
        and covariance_summary.get("tangent_order") == list(TANGENT_ORDER)
        and covariance_summary.get("adjoint_transport_applied") is False
        and covariance_summary.get(
            "scaling_applied_to_factors_or_saved_arrays"
        )
        is False
        and covariance_summary.get("regularization_or_repair_applied")
        is False
        and summary.get("edge_direction") == EDGE_DIRECTION,
        "E6 covariance contract differs from the protected baseline",
    )

    e5_inputs = load_canonical_pose_graph_inputs(selected_e5_root)
    declared_e5_root = summary.get("canonical_input_root")
    _require(
        isinstance(declared_e5_root, str)
        and (REPOSITORY_ROOT / declared_e5_root).resolve()
        == e5_inputs.input_root
        and summary.get("canonical_configuration_sha256")
        == e5_inputs.configuration_sha256,
        "selected E6 provenance does not identify the selected E5 root",
    )

    with np.load(values_path, allow_pickle=False) as values_data:
        _require(
            set(values_data.files)
            == {"frame_ids", "initial_poses", "optimized_poses"},
            "E6 pose_graph_values.npz members differ",
        )
        frame_ids = np.array(values_data["frame_ids"], copy=True)
        initial_poses = np.array(
            values_data["initial_poses"], dtype=float, copy=True
        )
        optimized_poses = np.array(
            values_data["optimized_poses"], dtype=float, copy=True
        )

    expected_frame_ids = np.asarray(e5_inputs.keyframe_ids, dtype=np.int64)
    _require(
        frame_ids.shape == (EXPECTED_KEYFRAME_COUNT,)
        and np.issubdtype(frame_ids.dtype, np.integer)
        and np.array_equal(frame_ids, expected_frame_ids),
        "E6 FrameIds differ from the canonical E5 keyframe ordering",
    )
    expected_pose_shape = (EXPECTED_KEYFRAME_COUNT, 4, 4)
    _require(
        initial_poses.shape == expected_pose_shape
        and optimized_poses.shape == expected_pose_shape,
        "E6 initial or optimized pose array shape differs",
    )

    tolerances = e5_inputs.validation_tolerances
    for ordinal, (initial, optimized) in enumerate(
        zip(initial_poses, optimized_poses)
    ):
        frame_id = int(frame_ids[ordinal])
        _validate_rigid_pose_matrix(
            initial,
            name="E6 initial pose at FrameId {}".format(frame_id),
            homogeneous_atol=tolerances["homogeneous_matrix_max_abs"],
            rotation_orthogonality_atol=tolerances[
                "rotation_orthogonality_max_abs"
            ],
            rotation_determinant_atol=tolerances[
                "rotation_determinant_abs"
            ],
        )
        _validate_rigid_pose_matrix(
            optimized,
            name="E6 optimized pose at FrameId {}".format(frame_id),
            homogeneous_atol=tolerances["homogeneous_matrix_max_abs"],
            rotation_orthogonality_atol=tolerances[
                "rotation_orthogonality_max_abs"
            ],
            rotation_determinant_atol=tolerances[
                "rotation_determinant_abs"
            ],
        )

    edges = []
    reconstruction_atol = float(tolerances["numerical_reconstruction_atol"])
    for edge in e5_inputs.edges:
        start_ordinal = int(edge.window_index)
        end_ordinal = start_ordinal + 1
        reconstructed = gtsam.Pose3(
            optimized_poses[start_ordinal]
        ).between(gtsam.Pose3(optimized_poses[end_ordinal]))
        reconstruction_error = float(
            np.max(
                np.abs(
                    np.asarray(reconstructed.matrix())
                    - np.asarray(edge.relative_pose.matrix())
                )
            )
        )
        _require(
            reconstruction_error <= reconstruction_atol,
            (
                "E6 optimized values do not reconstruct canonical edge "
                "{} ({} -> {})"
            ).format(
                edge.window_index,
                edge.start_frame_id,
                edge.end_frame_id,
            ),
        )
        edges.append(
            CovarianceEdge(
                edge_index=int(edge.window_index),
                start_frame_id=int(edge.start_frame_id),
                end_frame_id=int(edge.end_frame_id),
                covariance=edge.covariance,
                source_directory=edge.source_directory,
            )
        )

    optimized_poses.setflags(write=False)
    frame_id_to_ordinal = MappingProxyType(
        {
            int(frame_id): ordinal
            for ordinal, frame_id in enumerate(frame_ids)
        }
    )
    return CandidateDetectionInputs(
        e6_root=selected_e6_root,
        e5_root=e5_inputs.input_root,
        configuration_sha256=e5_inputs.configuration_sha256,
        keyframe_ids=tuple(int(value) for value in frame_ids),
        frame_id_to_ordinal=frame_id_to_ordinal,
        optimized_pose_matrices=optimized_poses,
        edges=tuple(edges),
        covariance_symmetry_atol=float(
            tolerances["covariance_symmetry_atol"]
        ),
        pose_reconstruction_atol=reconstruction_atol,
    )


def build_optimized_values(inputs: CandidateDetectionInputs) -> gtsam.Values:
    """Reconstruct E6 optimized values at actual-FrameId camera symbols."""

    values = gtsam.Values()
    for frame_id, matrix in zip(
        inputs.keyframe_ids, inputs.optimized_pose_matrices
    ):
        values.insert(camera_symbol(frame_id), gtsam.Pose3(matrix))
    return values


def candidate_detection_inputs_from_values(
    baseline_inputs: CandidateDetectionInputs,
    optimized_values: gtsam.Values,
) -> CandidateDetectionInputs:
    """Return an immutable pose snapshot from live Pose Graph values.

    Accepted covariance edges remain owned by the separately supplied
    ``CovarianceGraph`` so the caller can publish both objects transactionally.
    """

    expected_keys = tuple(
        int(camera_symbol(frame_id))
        for frame_id in baseline_inputs.keyframe_ids
    )
    observed_keys = tuple(int(key) for key in optimized_values.keys())
    _require(
        len(observed_keys) == len(expected_keys)
        and set(observed_keys) == set(expected_keys),
        "live Pose Graph Values must contain exactly the canonical camera keys",
    )
    try:
        pose_matrices = np.asarray(
            [
                optimized_values.atPose3(key).matrix()
                for key in expected_keys
            ],
            dtype=float,
        )
    except RuntimeError as error:
        raise CandidateDetectionError(
            "live Pose Graph Values contain a non-Pose3 camera value"
        ) from error
    _require(
        pose_matrices.shape
        == (len(baseline_inputs.keyframe_ids), 4, 4)
        and np.all(np.isfinite(pose_matrices)),
        "live Pose Graph pose matrices must be finite 4x4 transforms",
    )
    pose_matrices.setflags(write=False)
    return CandidateDetectionInputs(
        e6_root=baseline_inputs.e6_root,
        e5_root=baseline_inputs.e5_root,
        configuration_sha256=baseline_inputs.configuration_sha256,
        keyframe_ids=baseline_inputs.keyframe_ids,
        frame_id_to_ordinal=baseline_inputs.frame_id_to_ordinal,
        optimized_pose_matrices=pose_matrices,
        edges=baseline_inputs.edges,
        covariance_symmetry_atol=baseline_inputs.covariance_symmetry_atol,
        pose_reconstruction_atol=baseline_inputs.pose_reconstruction_atol,
    )


def deterministic_probe_current_ordinals(
    keyframe_count: int,
) -> Tuple[int, int, int]:
    """Return nearest integer one-third, two-thirds, and final ordinals."""

    if keyframe_count < 4:
        raise ValueError("at least four keyframes are required")
    final_ordinal = int(keyframe_count) - 1
    ordinals = (
        (final_ordinal + 1) // 3,
        (2 * final_ordinal + 1) // 3,
        final_ordinal,
    )
    if len(set(ordinals)) != 3:
        raise ValueError("probe ordinals must be distinct")
    return ordinals


class CovarianceGraph:
    """Directed actual-FrameId graph for the course covariance-path heuristic.

    Consecutive E5 edges and successfully accepted loop edges share the same
    raw conditional-covariance and previous-to-current direction contract.
    Insertions return a new graph so publication can remain transactional.
    """

    def __init__(
        self,
        keyframe_ids: Sequence[int],
        edges: Sequence[CovarianceEdge],
        *,
        covariance_symmetry_atol: float = 1e-9,
    ) -> None:
        self.keyframe_ids = tuple(int(value) for value in keyframe_ids)
        _require(
            bool(self.keyframe_ids)
            and len(set(self.keyframe_ids)) == len(self.keyframe_ids),
            "covariance graph FrameIds must be nonempty and unique",
        )
        self.frame_id_to_ordinal = MappingProxyType(
            {
                frame_id: ordinal
                for ordinal, frame_id in enumerate(self.keyframe_ids)
            }
        )
        self.covariance_symmetry_atol = float(covariance_symmetry_atol)

        normalized_edges = []
        observed_indices = set()
        observed_arcs = set()
        for edge in edges:
            edge_index = int(edge.edge_index)
            start_frame_id = int(edge.start_frame_id)
            end_frame_id = int(edge.end_frame_id)
            _require(
                edge_index not in observed_indices,
                "covariance edge indices must be unique",
            )
            _require(
                (start_frame_id, end_frame_id) not in observed_arcs,
                "directed covariance arcs must be unique",
            )
            _require(
                start_frame_id in self.frame_id_to_ordinal
                and end_frame_id in self.frame_id_to_ordinal,
                "covariance edge endpoint is not a graph FrameId",
            )
            covariance = np.array(edge.covariance, dtype=float, copy=True)
            self._validate_positive_definite_covariance(
                covariance,
                name="edge {} covariance".format(edge_index),
            )
            covariance.setflags(write=False)
            normalized_edges.append(
                CovarianceEdge(
                    edge_index=edge_index,
                    start_frame_id=start_frame_id,
                    end_frame_id=end_frame_id,
                    covariance=covariance,
                    source_directory=Path(edge.source_directory),
                )
            )
            observed_indices.add(edge_index)
            observed_arcs.add((start_frame_id, end_frame_id))

        self.edges = tuple(normalized_edges)
        adjacency: Dict[int, list] = {
            frame_id: [] for frame_id in self.keyframe_ids
        }
        edge_by_index = {}
        for edge in self.edges:
            adjacency[edge.start_frame_id].append(edge)
            edge_by_index[edge.edge_index] = edge
        self.adjacency = MappingProxyType(
            {
                frame_id: tuple(
                    sorted(
                        outgoing,
                        key=lambda edge: (
                            edge.end_frame_id,
                            edge.edge_index,
                        ),
                    )
                )
                for frame_id, outgoing in adjacency.items()
            }
        )
        self.edge_by_index = MappingProxyType(edge_by_index)

    def with_directed_edge(
        self,
        start_frame_id: int,
        end_frame_id: int,
        covariance: np.ndarray,
        *,
        source_directory: Path = Path("."),
    ) -> "CovarianceGraph":
        """Return a new graph with one byte-preserved raw directed covariance.

        The original graph is unchanged.  The caller may therefore validate a
        prospective accepted loop and publish the returned graph only after
        the rest of the scientific transaction succeeds.
        """

        start_frame_id = int(start_frame_id)
        end_frame_id = int(end_frame_id)
        self._require_vertices(start_frame_id, end_frame_id)
        _require(
            (start_frame_id, end_frame_id)
            not in {
                (edge.start_frame_id, edge.end_frame_id)
                for edge in self.edges
            },
            "directed covariance arc already exists",
        )
        raw_covariance = np.asarray(covariance)
        _require(
            raw_covariance.dtype == np.dtype(float),
            "accepted edge covariance must retain float64 storage",
        )
        raw_bytes = raw_covariance.tobytes(order="C")
        next_edge_index = (
            max(self.edge_by_index.keys()) + 1
            if self.edge_by_index
            else 0
        )
        updated_graph = CovarianceGraph(
            self.keyframe_ids,
            self.edges
            + (
                CovarianceEdge(
                    edge_index=next_edge_index,
                    start_frame_id=start_frame_id,
                    end_frame_id=end_frame_id,
                    covariance=raw_covariance,
                    source_directory=Path(source_directory),
                ),
            ),
            covariance_symmetry_atol=self.covariance_symmetry_atol,
        )
        inserted = updated_graph.edge_by_index[next_edge_index].covariance
        _require(
            inserted.dtype == raw_covariance.dtype
            and inserted.tobytes(order="C") == raw_bytes,
            "accepted edge covariance changed during graph insertion",
        )
        return updated_graph

    def _validate_positive_definite_covariance(
        self, covariance: np.ndarray, *, name: str
    ) -> None:
        _require(
            covariance.shape == (6, 6)
            and np.all(np.isfinite(covariance)),
            "{} must be finite 6x6".format(name),
        )
        _require(
            np.max(np.abs(covariance - covariance.T))
            <= self.covariance_symmetry_atol,
            "{} must be symmetric".format(name),
        )
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise CandidateDetectionError(
                "{} must be positive definite".format(name)
            ) from error
        sign, log_determinant = np.linalg.slogdet(covariance)
        _require(
            sign > 0.0 and np.isfinite(log_determinant),
            "{} must have a finite positive determinant".format(name),
        )

    def _path_log_determinant(
        self, covariance: np.ndarray, *, path_description: str
    ) -> float:
        self._validate_positive_definite_covariance(
            covariance,
            name="{} accumulated covariance".format(path_description),
        )
        sign, log_determinant = np.linalg.slogdet(covariance)
        _require(
            sign > 0.0 and np.isfinite(log_determinant),
            "{} accumulated determinant is invalid".format(path_description),
        )
        return float(log_determinant)

    def _require_vertices(self, source_frame_id: int, target_frame_id: int) -> None:
        _require(
            source_frame_id in self.frame_id_to_ordinal,
            "source FrameId {} is not a keyframe".format(source_frame_id),
        )
        _require(
            target_frame_id in self.frame_id_to_ordinal,
            "target FrameId {} is not a keyframe".format(target_frame_id),
        )

    def find_course_covariance_path(
        self, source_frame_id: int, target_frame_id: int
    ) -> CovariancePath:
        """Apply the lecturer's accumulated-covariance Dijkstra heuristic.

        Queue labels are ordered by ``det(Sigma[v])`` (implemented with
        ``slogdet``), and relaxation compares ``det(Sigma[u] + w(u, v))``.
        The matrix additions are direct because every edge already has the
        validated common Pose3 tangent direction/order.  ``Sigma[path]`` is
        retained as the course relative-uncertainty approximation; it is not
        transported or rescaled.

        Because the label is a matrix-derived path quantity rather than a sum
        of scalar edge weights, this is documented and tested as the course
        heuristic, not as an application of the standard Dijkstra optimality
        theorem.  The method does not mutate this covariance graph.
        """

        source_frame_id = int(source_frame_id)
        target_frame_id = int(target_frame_id)
        self._require_vertices(source_frame_id, target_frame_id)
        if source_frame_id == target_frame_id:
            covariance = np.zeros((6, 6), dtype=float)
            covariance.setflags(write=False)
            return CovariancePath(
                source_frame_id=source_frame_id,
                target_frame_id=target_frame_id,
                frame_ids=(source_frame_id,),
                edge_indices=(),
                covariance=covariance,
                covariance_determinant=0.0,
                covariance_log_determinant=float("-inf"),
                policy=COURSE_HEURISTIC_POLICY,
            )

        best_covariance: Dict[int, np.ndarray] = {
            source_frame_id: np.zeros((6, 6), dtype=float)
        }
        best_log_determinant: Dict[int, float] = {
            source_frame_id: float("-inf")
        }
        predecessor: Dict[int, Tuple[int, int]] = {}
        settled = set()
        queue = [(float("-inf"), source_frame_id)]

        while queue:
            queued_log_determinant, frame_id = heapq.heappop(queue)
            if frame_id in settled:
                continue
            if queued_log_determinant != best_log_determinant.get(frame_id):
                continue
            settled.add(frame_id)
            if frame_id == target_frame_id:
                break

            for edge in self.adjacency[frame_id]:
                neighbor = edge.end_frame_id
                if neighbor in settled:
                    continue
                # Course rule: accumulate the raw matrices directly in the
                # validated [rotation(rad), translation(m)] tangent contract.
                candidate_covariance = (
                    best_covariance[frame_id] + edge.covariance
                )
                candidate_log_determinant = self._path_log_determinant(
                    candidate_covariance,
                    path_description="{} -> {}".format(
                        source_frame_id, neighbor
                    ),
                )
                if candidate_log_determinant < best_log_determinant.get(
                    neighbor, float("inf")
                ):
                    best_covariance[neighbor] = candidate_covariance
                    best_log_determinant[
                        neighbor
                    ] = candidate_log_determinant
                    predecessor[neighbor] = (frame_id, edge.edge_index)
                    heapq.heappush(
                        queue, (candidate_log_determinant, neighbor)
                    )

        _require(
            target_frame_id in best_covariance
            and target_frame_id in settled,
            "no directed covariance path exists from {} to {}".format(
                source_frame_id, target_frame_id
            ),
        )
        frame_path = [target_frame_id]
        edge_path = []
        cursor = target_frame_id
        while cursor != source_frame_id:
            _require(
                cursor in predecessor,
                "course heuristic predecessor chain is incomplete",
            )
            parent, edge_index = predecessor[cursor]
            edge_path.append(edge_index)
            frame_path.append(parent)
            cursor = parent
        frame_path.reverse()
        edge_path.reverse()

        covariance = np.array(
            best_covariance[target_frame_id], dtype=float, copy=True
        )
        log_determinant = self._path_log_determinant(
            covariance,
            path_description="{} -> {}".format(
                source_frame_id, target_frame_id
            ),
        )
        determinant = float(np.linalg.det(covariance))
        _require(
            determinant > 0.0 and np.isfinite(determinant),
            "final accumulated covariance determinant is invalid",
        )
        covariance.setflags(write=False)
        return CovariancePath(
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            frame_ids=tuple(frame_path),
            edge_indices=tuple(edge_path),
            covariance=covariance,
            covariance_determinant=determinant,
            covariance_log_determinant=log_determinant,
            policy=COURSE_HEURISTIC_POLICY,
        )

    def find_maor_comparison_path(
        self, source_frame_id: int, target_frame_id: int
    ) -> CovariancePath:
        """Return the optional legacy scalar-edge comparison path."""

        source_frame_id = int(source_frame_id)
        target_frame_id = int(target_frame_id)
        self._require_vertices(source_frame_id, target_frame_id)
        if source_frame_id == target_frame_id:
            covariance = np.zeros((6, 6), dtype=float)
            covariance.setflags(write=False)
            return CovariancePath(
                source_frame_id=source_frame_id,
                target_frame_id=target_frame_id,
                frame_ids=(source_frame_id,),
                edge_indices=(),
                covariance=covariance,
                covariance_determinant=0.0,
                covariance_log_determinant=float("-inf"),
                policy=MAOR_COMPARISON_POLICY,
                scalar_cost=0.0,
            )

        best_cost = {source_frame_id: 0.0}
        predecessor: Dict[int, Tuple[int, int]] = {}
        settled = set()
        queue = [(0.0, source_frame_id)]
        while queue:
            cost, frame_id = heapq.heappop(queue)
            if frame_id in settled:
                continue
            if cost != best_cost.get(frame_id):
                continue
            settled.add(frame_id)
            if frame_id == target_frame_id:
                break
            for edge in self.adjacency[frame_id]:
                neighbor = edge.end_frame_id
                if neighbor in settled:
                    continue
                edge_weight = float(
                    np.sqrt(np.linalg.det(edge.covariance))
                )
                candidate_cost = cost + edge_weight
                if candidate_cost < best_cost.get(neighbor, float("inf")):
                    best_cost[neighbor] = candidate_cost
                    predecessor[neighbor] = (frame_id, edge.edge_index)
                    heapq.heappush(queue, (candidate_cost, neighbor))

        _require(
            target_frame_id in best_cost and target_frame_id in settled,
            "no directed Maor comparison path exists from {} to {}".format(
                source_frame_id, target_frame_id
            ),
        )
        frame_path = [target_frame_id]
        edge_path = []
        cursor = target_frame_id
        while cursor != source_frame_id:
            _require(
                cursor in predecessor,
                "Maor comparison predecessor chain is incomplete",
            )
            parent, edge_index = predecessor[cursor]
            edge_path.append(edge_index)
            frame_path.append(parent)
            cursor = parent
        frame_path.reverse()
        edge_path.reverse()
        covariance = np.sum(
            np.stack(
                [
                    self.edge_by_index[edge_index].covariance
                    for edge_index in edge_path
                ],
                axis=0,
            ),
            axis=0,
        )
        log_determinant = self._path_log_determinant(
            covariance,
            path_description="Maor comparison {} -> {}".format(
                source_frame_id, target_frame_id
            ),
        )
        determinant = float(np.linalg.det(covariance))
        _require(
            determinant > 0.0 and np.isfinite(determinant),
            "Maor comparison accumulated determinant is invalid",
        )
        covariance.setflags(write=False)
        return CovariancePath(
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            frame_ids=tuple(frame_path),
            edge_indices=tuple(edge_path),
            covariance=covariance,
            covariance_determinant=determinant,
            covariance_log_determinant=log_determinant,
            policy=MAOR_COMPARISON_POLICY,
            scalar_cost=float(best_cost[target_frame_id]),
        )


def evaluate_gtsam_mahalanobis(
    optimized_values: gtsam.Values,
    previous_frame_id: int,
    current_frame_id: int,
    path_covariance: np.ndarray,
) -> Tuple[float, float, float]:
    """Evaluate an identity relative-pose residual using the raw path covariance.

    The factor direction is previous FrameId to current FrameId.  GTSAM's
    ``BetweenFactorPose3.error`` is one half of the whitened squared residual,
    hence the returned relation ``mahalanobis_squared = 2 * factor_error``.
    The covariance tangent order is rotation xyz then translation xyz.
    """

    covariance = np.asarray(path_covariance, dtype=float)
    _require(
        covariance.shape == (6, 6) and np.all(np.isfinite(covariance)),
        "path covariance must be finite 6x6",
    )
    try:
        noise = gtsam.noiseModel.Gaussian.Covariance(covariance)
    except RuntimeError as error:
        raise CandidateDetectionError(
            "GTSAM rejected the raw path covariance"
        ) from error
    factor = gtsam.BetweenFactorPose3(
        camera_symbol(int(previous_frame_id)),
        camera_symbol(int(current_frame_id)),
        gtsam.Pose3(),
        noise,
    )
    factor_error = float(factor.error(optimized_values))
    _require(
        np.isfinite(factor_error) and factor_error >= 0.0,
        "GTSAM factor error must be finite and nonnegative",
    )
    mahalanobis_squared = 2.0 * factor_error
    mahalanobis = float(np.sqrt(mahalanobis_squared))
    _require(
        np.isfinite(mahalanobis_squared) and np.isfinite(mahalanobis),
        "Mahalanobis score must be finite",
    )
    return factor_error, mahalanobis_squared, mahalanobis


def score_candidate_pair(
    inputs: CandidateDetectionInputs,
    graph: CovarianceGraph,
    optimized_values: gtsam.Values,
    previous_frame_id: int,
    current_frame_id: int,
) -> CandidateScore:
    """Score one actual-FrameId historical/current pair without filtering.

    Ordinals validate temporal order and locate poses; actual FrameIds identify
    graph nodes and GTSAM camera symbols.  Production policy is deliberately
    absent here so a score remains reusable evidence rather than a decision.
    """

    previous_frame_id = int(previous_frame_id)
    current_frame_id = int(current_frame_id)
    _require(
        previous_frame_id in inputs.frame_id_to_ordinal,
        "previous FrameId {} is not canonical".format(previous_frame_id),
    )
    _require(
        current_frame_id in inputs.frame_id_to_ordinal,
        "current FrameId {} is not canonical".format(current_frame_id),
    )
    previous_ordinal = inputs.frame_id_to_ordinal[previous_frame_id]
    current_ordinal = inputs.frame_id_to_ordinal[current_frame_id]
    _require(
        previous_ordinal < current_ordinal,
        "candidate scoring requires a historical FrameId before the current",
    )
    path = graph.find_course_covariance_path(
        previous_frame_id, current_frame_id
    )
    factor_error, mahalanobis_squared, mahalanobis = (
        evaluate_gtsam_mahalanobis(
            optimized_values,
            previous_frame_id,
            current_frame_id,
            path.covariance,
        )
    )
    return CandidateScore(
        previous_frame_id=previous_frame_id,
        current_frame_id=current_frame_id,
        previous_keyframe_ordinal=previous_ordinal,
        current_keyframe_ordinal=current_ordinal,
        path=path,
        factor_error=factor_error,
        mahalanobis_squared=mahalanobis_squared,
        mahalanobis=mahalanobis,
    )


def score_previous_keyframes(
    inputs: CandidateDetectionInputs,
    graph: CovarianceGraph,
    optimized_values: gtsam.Values,
    current_frame_id: int,
    previous_frame_ids: Optional[Sequence[int]] = None,
) -> Tuple[CandidateScore, ...]:
    """Return a deterministic unfiltered score ranking for one current FrameId.

    Thresholding, temporal exclusion, proposal regions, matcher caps, and
    acceptance are later orchestration responsibilities.
    """

    current_frame_id = int(current_frame_id)
    _require(
        current_frame_id in inputs.frame_id_to_ordinal,
        "current FrameId {} is not canonical".format(current_frame_id),
    )
    current_ordinal = inputs.frame_id_to_ordinal[current_frame_id]
    selected_previous = (
        tuple(inputs.keyframe_ids[:current_ordinal])
        if previous_frame_ids is None
        else tuple(int(value) for value in previous_frame_ids)
    )
    _require(
        len(set(selected_previous)) == len(selected_previous),
        "previous FrameIds must be unique",
    )
    scores = [
        score_candidate_pair(
            inputs,
            graph,
            optimized_values,
            previous_frame_id,
            current_frame_id,
        )
        for previous_frame_id in selected_previous
    ]
    return tuple(
        sorted(
            scores,
            key=lambda score: (
                score.mahalanobis_squared,
                score.previous_frame_id,
            ),
        )
    )


__all__ = [
    "CANONICAL_E6_ROOT",
    "COURSE_HEURISTIC_POLICY",
    "MAOR_COMPARISON_POLICY",
    "CandidateDetectionError",
    "CandidateDetectionInputs",
    "CandidateScore",
    "CovarianceEdge",
    "CovarianceGraph",
    "CovariancePath",
    "build_optimized_values",
    "candidate_detection_inputs_from_values",
    "deterministic_probe_current_ordinals",
    "evaluate_gtsam_mahalanobis",
    "load_candidate_detection_inputs",
    "score_candidate_pair",
    "score_previous_keyframes",
]
