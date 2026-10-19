"""Lecturer-style accumulated-covariance candidate detection."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import gtsam
import numpy as np

from ..contracts import ContractError
from ..pose_graph.state import camera_symbol


COURSE_HEURISTIC_POLICY = "minimum determinant of directly accumulated raw covariance"


@dataclass(frozen=True)
class CovarianceEdge:
    edge_index: int
    start_frame_id: int
    end_frame_id: int
    covariance: np.ndarray
    source: str


@dataclass(frozen=True)
class CovariancePath:
    source_frame_id: int
    target_frame_id: int
    frame_ids: Tuple[int, ...]
    edge_indices: Tuple[int, ...]
    covariance: np.ndarray
    covariance_determinant: float
    covariance_log_determinant: float


@dataclass(frozen=True)
class CandidateScore:
    previous_frame_id: int
    current_frame_id: int
    previous_keyframe_ordinal: int
    current_keyframe_ordinal: int
    path: CovariancePath
    factor_error: float
    mahalanobis_squared: float
    mahalanobis: float


class CovarianceGraph:
    """Forward directed graph; loop insertions return a new immutable instance."""

    def __init__(
        self,
        keyframe_ids: Sequence[int],
        edges: Sequence[CovarianceEdge],
        *,
        covariance_symmetry_atol: float,
    ) -> None:
        self.keyframe_ids = tuple(int(value) for value in keyframe_ids)
        if not self.keyframe_ids or len(set(self.keyframe_ids)) != len(self.keyframe_ids):
            raise ContractError("covariance graph FrameIds are invalid")
        self.frame_id_to_ordinal = {
            frame_id: ordinal for ordinal, frame_id in enumerate(self.keyframe_ids)
        }
        self.covariance_symmetry_atol = float(covariance_symmetry_atol)
        normalized = []
        arcs = set()
        indices = set()
        for edge in edges:
            covariance = np.asarray(edge.covariance, dtype=float).copy()
            self._validate_covariance(covariance)
            arc = (int(edge.start_frame_id), int(edge.end_frame_id))
            if (
                edge.edge_index in indices
                or arc in arcs
                or arc[0] not in self.frame_id_to_ordinal
                or arc[1] not in self.frame_id_to_ordinal
            ):
                raise ContractError("covariance edge identity differs")
            covariance.setflags(write=False)
            normalized.append(
                CovarianceEdge(
                    int(edge.edge_index), arc[0], arc[1], covariance, str(edge.source)
                )
            )
            indices.add(edge.edge_index)
            arcs.add(arc)
        self.edges = tuple(normalized)
        self.edge_by_index = {edge.edge_index: edge for edge in self.edges}
        adjacency = {frame_id: [] for frame_id in self.keyframe_ids}
        for edge in self.edges:
            adjacency[edge.start_frame_id].append(edge)
        self.adjacency = {
            frame_id: tuple(sorted(items, key=lambda item: (item.end_frame_id, item.edge_index)))
            for frame_id, items in adjacency.items()
        }

    @classmethod
    def consecutive(
        cls,
        keyframe_ids: Sequence[int],
        covariances: np.ndarray,
        *,
        covariance_symmetry_atol: float,
    ) -> "CovarianceGraph":
        return cls(
            keyframe_ids,
            tuple(
                CovarianceEdge(index, int(start), int(end), covariance, "bundle")
                for index, (start, end, covariance) in enumerate(
                    zip(keyframe_ids, keyframe_ids[1:], covariances)
                )
            ),
            covariance_symmetry_atol=covariance_symmetry_atol,
        )

    def _validate_covariance(self, covariance: np.ndarray) -> None:
        if (
            covariance.shape != (6, 6)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(
                covariance, covariance.T, atol=self.covariance_symmetry_atol, rtol=0.0
            )
        ):
            raise ContractError("covariance graph edge must be finite symmetric 6x6")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise ContractError("covariance graph edge must be positive definite") from error

    def with_edge(
        self, start_frame_id: int, end_frame_id: int, covariance: np.ndarray,
        *, source: str,
    ) -> "CovarianceGraph":
        if (start_frame_id, end_frame_id) in {
            (edge.start_frame_id, edge.end_frame_id) for edge in self.edges
        }:
            raise ContractError("covariance graph arc already exists")
        next_index = max(self.edge_by_index, default=-1) + 1
        return CovarianceGraph(
            self.keyframe_ids,
            self.edges
            + (
                CovarianceEdge(
                    next_index, int(start_frame_id), int(end_frame_id),
                    np.asarray(covariance, dtype=float), str(source),
                ),
            ),
            covariance_symmetry_atol=self.covariance_symmetry_atol,
        )

    def find_course_covariance_path(
        self, source_frame_id: int, target_frame_id: int
    ) -> CovariancePath:
        if source_frame_id not in self.adjacency or target_frame_id not in self.adjacency:
            raise ContractError("covariance path endpoint is not a keyframe")
        if source_frame_id == target_frame_id:
            return CovariancePath(
                source_frame_id, target_frame_id, (source_frame_id,), (),
                np.zeros((6, 6)), 0.0, float("-inf"),
            )
        best_covariance = {source_frame_id: np.zeros((6, 6), dtype=float)}
        best_logdet = {source_frame_id: float("-inf")}
        predecessor = {}
        settled = set()
        queue = [(float("-inf"), source_frame_id)]
        while queue:
            queued_cost, frame_id = heapq.heappop(queue)
            if frame_id in settled or queued_cost != best_logdet.get(frame_id):
                continue
            settled.add(frame_id)
            if frame_id == target_frame_id:
                break
            for edge in self.adjacency[frame_id]:
                if edge.end_frame_id in settled:
                    continue
                candidate = best_covariance[frame_id] + edge.covariance
                sign, cost = np.linalg.slogdet(candidate)
                if sign <= 0.0 or not np.isfinite(cost):
                    raise ContractError("accumulated path covariance is invalid")
                if cost < best_logdet.get(edge.end_frame_id, float("inf")):
                    best_covariance[edge.end_frame_id] = candidate
                    best_logdet[edge.end_frame_id] = float(cost)
                    predecessor[edge.end_frame_id] = (frame_id, edge.edge_index)
                    heapq.heappush(queue, (float(cost), edge.end_frame_id))
        if target_frame_id not in settled:
            raise ContractError("no directed covariance path exists")
        frames = [target_frame_id]
        edge_indices = []
        cursor = target_frame_id
        while cursor != source_frame_id:
            cursor, edge_index = predecessor[cursor]
            frames.append(cursor)
            edge_indices.append(edge_index)
        frames.reverse()
        edge_indices.reverse()
        covariance = np.asarray(best_covariance[target_frame_id], dtype=float)
        determinant = float(np.linalg.det(covariance))
        return CovariancePath(
            source_frame_id, target_frame_id, tuple(frames), tuple(edge_indices),
            covariance, determinant, best_logdet[target_frame_id],
        )


def score_candidate(
    *,
    values: gtsam.Values,
    keyframe_ids: Sequence[int],
    graph: CovarianceGraph,
    previous_ordinal: int,
    current_ordinal: int,
) -> CandidateScore:
    previous_frame_id = int(keyframe_ids[previous_ordinal])
    current_frame_id = int(keyframe_ids[current_ordinal])
    path = graph.find_course_covariance_path(previous_frame_id, current_frame_id)
    factor = gtsam.BetweenFactorPose3(
        camera_symbol(previous_frame_id), camera_symbol(current_frame_id),
        gtsam.Pose3(), gtsam.noiseModel.Gaussian.Covariance(path.covariance),
    )
    factor_error = float(factor.error(values))
    squared = 2.0 * factor_error
    return CandidateScore(
        previous_frame_id, current_frame_id, previous_ordinal, current_ordinal,
        path, factor_error, squared, float(np.sqrt(squared)),
    )


def score_previous_keyframes(
    *,
    values: gtsam.Values,
    keyframe_ids: Sequence[int],
    graph: CovarianceGraph,
    current_ordinal: int,
) -> Tuple[CandidateScore, ...]:
    scores = tuple(
        score_candidate(
            values=values, keyframe_ids=keyframe_ids, graph=graph,
            previous_ordinal=previous, current_ordinal=current_ordinal,
        )
        for previous in range(current_ordinal)
    )
    return tuple(
        sorted(scores, key=lambda item: (item.mahalanobis_squared, item.previous_frame_id))
    )


__all__ = [
    "CandidateScore", "CovarianceEdge", "CovarianceGraph", "CovariancePath",
    "score_candidate", "score_previous_keyframes",
]
