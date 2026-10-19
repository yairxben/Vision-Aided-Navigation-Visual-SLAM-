"""Frozen Loop Closure gates, fair proposal regions, and suppression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

from .detection import CandidateScore


@dataclass(frozen=True)
class LoopClosurePolicy:
    mahalanobis_squared_threshold: float
    minimum_temporal_ordinal_gap: int
    maximum_matcher_invocations_per_current: int
    minimum_consensus_count: int
    minimum_supporter_count: int
    minimum_supporter_percentage: float
    previous_ordinal_cluster_radius: int
    current_ordinal_cluster_radius: int
    pair_seed: int
    maximum_hypothesis_attempts: int

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "LoopClosurePolicy":
        return cls(
            float(config["mahalanobis_squared_threshold"]),
            int(config["minimum_temporal_ordinal_gap"]),
            int(config["maximum_matcher_invocations_per_current"]),
            int(config["minimum_consensus_count"]),
            int(config["minimum_supporter_count"]),
            float(config["minimum_supporter_percentage"]),
            int(config["previous_ordinal_cluster_radius"]),
            int(config["current_ordinal_cluster_radius"]),
            int(config["pair_seed"]),
            int(config["ransac"]["maximum_hypothesis_attempts"]),
        )


@dataclass(frozen=True)
class AcceptedLoopAnchor:
    acceptance_id: int
    previous_frame_id: int
    current_frame_id: int
    previous_ordinal: int
    current_ordinal: int
    state_version: int


@dataclass(frozen=True)
class RankedCandidate:
    pair_index: int
    score: CandidateScore
    rank_within_current: int

    @property
    def previous_ordinal(self) -> int:
        return self.score.previous_keyframe_ordinal

    @property
    def current_ordinal(self) -> int:
        return self.score.current_keyframe_ordinal


@dataclass
class TemporaryProposalRegion:
    region_id: int
    anchor_previous_ordinal: int
    anchor_current_ordinal: int
    candidates: List[RankedCandidate]
    cursor: int = 0


def ordered_pair_index(previous_ordinal: int, current_ordinal: int) -> int:
    if not 0 <= previous_ordinal < current_ordinal:
        raise ValueError("ordered pair requires previous before current")
    return current_ordinal * (current_ordinal - 1) // 2 + previous_ordinal


def same_anchored_rectangle(
    first_previous: int, first_current: int, second_previous: int, second_current: int,
    *, previous_radius: int, current_radius: int,
) -> bool:
    return bool(
        abs(first_previous - second_previous) <= previous_radius
        and abs(first_current - second_current) <= current_radius
    )


def suppressing_anchor(
    candidate: RankedCandidate,
    anchors: Sequence[AcceptedLoopAnchor],
    policy: LoopClosurePolicy,
) -> Optional[AcceptedLoopAnchor]:
    for anchor in sorted(anchors, key=lambda item: item.acceptance_id):
        if same_anchored_rectangle(
            candidate.previous_ordinal, candidate.current_ordinal,
            anchor.previous_ordinal, anchor.current_ordinal,
            previous_radius=policy.previous_ordinal_cluster_radius,
            current_radius=policy.current_ordinal_cluster_radius,
        ):
            return anchor
    return None


def form_temporary_regions(
    candidates: Sequence[RankedCandidate], policy: LoopClosurePolicy
) -> Tuple[TemporaryProposalRegion, ...]:
    regions = []
    for candidate in candidates:
        selected = None
        for region in regions:
            if same_anchored_rectangle(
                candidate.previous_ordinal, candidate.current_ordinal,
                region.anchor_previous_ordinal, region.anchor_current_ordinal,
                previous_radius=policy.previous_ordinal_cluster_radius,
                current_radius=policy.current_ordinal_cluster_radius,
            ):
                selected = region
                break
        if selected is None:
            selected = TemporaryProposalRegion(
                len(regions), candidate.previous_ordinal,
                candidate.current_ordinal, [],
            )
            regions.append(selected)
        selected.candidates.append(candidate)
    return tuple(regions)


def evaluate_region_schedule(
    regions: Sequence[TemporaryProposalRegion],
    maximum_invocations: int,
    *,
    candidate_is_live: Callable[[RankedCandidate, TemporaryProposalRegion, int], bool],
    evaluate_live_candidate: Callable[[RankedCandidate, TemporaryProposalRegion, int, int], None],
) -> int:
    invocations = 0
    scheduler_pass = 1
    while invocations < maximum_invocations:
        offered = False
        for region in regions:
            candidate = None
            while region.cursor < len(region.candidates):
                proposed = region.candidates[region.cursor]
                region.cursor += 1
                if candidate_is_live(proposed, region, scheduler_pass):
                    candidate = proposed
                    break
            if candidate is None:
                continue
            offered = True
            invocations += 1
            evaluate_live_candidate(candidate, region, scheduler_pass, invocations)
            if invocations >= maximum_invocations:
                break
        if not offered:
            break
        scheduler_pass += 1
    return invocations


def matcher_result_passes(
    *, consensus_count: int, supporter_count: int, supporter_percentage: float,
    terminal_status: str, policy: LoopClosurePolicy,
) -> bool:
    return bool(
        terminal_status in ("adaptive_completed", "budget_exhausted_with_best_model")
        and consensus_count >= policy.minimum_consensus_count
        and supporter_count >= policy.minimum_supporter_count
        and supporter_percentage >= policy.minimum_supporter_percentage
    )


__all__ = [
    "AcceptedLoopAnchor", "LoopClosurePolicy", "RankedCandidate",
    "TemporaryProposalRegion", "evaluate_region_schedule",
    "form_temporary_regions", "matcher_result_passes", "ordered_pair_index",
    "same_anchored_rectangle", "suppressing_anchor",
]
