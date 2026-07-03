"""Time-domain weight-reuse phase classification (round-1 keystone, default-off; pure read of the IR)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence

from mimarsinan.mapping.ir.graph import IRGraph
from mimarsinan.mapping.ir.types import NeuralCore, WeightBank


__all__ = [
    "SegmentReusePhases",
    "WeightReusePlan",
    "classify_segment_phases",
    "format_weight_reuse_summary",
    "weight_reuse_plan_from_graph",
]


def _bank_weight_count(bank: WeightBank) -> int:
    """The resident bank's full weight count (loaded ONCE per reprogram)."""
    return int(bank.core_matrix.size)


def _owned_core_weight_count(core: NeuralCore) -> int:
    """An owned (non-shared) core's weight count (its own reprogram)."""
    if core.core_matrix is None:
        return 0
    return int(core.core_matrix.size)


@dataclass(frozen=True)
class SegmentReusePhases:
    """Per-segment phase split: N reprogram + M reuse passes from bank grouping.

    params_reloaded counts each distinct bank / owned core ONCE (not once per reused position).
    """

    reprogram_passes: int
    reuse_passes: int
    params_reloaded: int

    @property
    def total_passes(self) -> int:
        return self.reprogram_passes + self.reuse_passes

    @property
    def reuse_fraction(self) -> float:
        """Fraction of passes that reuse a resident bank (0.0 when no passes)."""
        if self.total_passes == 0:
            return 0.0
        return self.reuse_passes / self.total_passes


def classify_segment_phases(
    cores: Iterable[NeuralCore],
    weight_banks: Mapping[int, WeightBank],
) -> SegmentReusePhases:
    """Classify a segment's NeuralCores into N reprogram + M reuse passes.

    Cores sharing a weight_bank_id collapse to one reprogram + reuse passes; an owned-matrix core is its own reprogram.
    """
    reprogram_passes = 0
    reuse_passes = 0
    params_reloaded = 0
    seen_banks: set[int] = set()

    for core in cores:
        bank_id = core.weight_bank_id
        if bank_id is None:
            reprogram_passes += 1
            params_reloaded += _owned_core_weight_count(core)
            continue
        if bank_id in seen_banks:
            reuse_passes += 1
            continue
        seen_banks.add(bank_id)
        reprogram_passes += 1
        bank = weight_banks.get(bank_id)
        if bank is not None:
            params_reloaded += _bank_weight_count(bank)

    return SegmentReusePhases(
        reprogram_passes=reprogram_passes,
        reuse_passes=reuse_passes,
        params_reloaded=params_reloaded,
    )


@dataclass(frozen=True)
class WeightReusePlan:
    """Schedule-wide phase split: M reuse + N reprogram phases over a graph.

    sync_barrier_count = total_passes - 1 (one activation-gather barrier between passes).
    """

    reprogram_passes: int
    reuse_passes: int
    params_reloaded: int

    @property
    def total_passes(self) -> int:
        return self.reprogram_passes + self.reuse_passes

    @property
    def reuse_fraction(self) -> float:
        """Fraction of passes that reuse a resident bank (0.0 when no passes)."""
        if self.total_passes == 0:
            return 0.0
        return self.reuse_passes / self.total_passes

    @property
    def sync_barrier_count(self) -> int:
        """The number of activation-gather barriers = ``total_passes - 1`` (0 when none)."""
        return max(self.total_passes - 1, 0)

    @classmethod
    def from_segments(
        cls, segments: Sequence[SegmentReusePhases]
    ) -> "WeightReusePlan":
        """Aggregate per-segment phase splits into one schedule-wide plan."""
        return cls(
            reprogram_passes=sum(s.reprogram_passes for s in segments),
            reuse_passes=sum(s.reuse_passes for s in segments),
            params_reloaded=sum(s.params_reloaded for s in segments),
        )


def format_weight_reuse_summary(plan: WeightReusePlan) -> str:
    """One-line SCM-gate summary: ``N reprogram + M reuse phases`` (the schedule split)."""
    return (
        f"{plan.reprogram_passes} reprogram + {plan.reuse_passes} reuse phases "
        f"({plan.total_passes} total, {plan.reuse_fraction:.1%} reused; "
        f"{plan.params_reloaded} params reloaded)"
    )


def weight_reuse_plan_from_graph(graph: IRGraph) -> WeightReusePlan:
    """Build the schedule-wide WeightReusePlan from an IR graph's cores (pure read; runs on the post-pruning IR)."""
    cores = graph.get_neural_cores()
    weight_banks: Dict[int, WeightBank] = dict(getattr(graph, "weight_banks", {}) or {})
    phases = classify_segment_phases(cores, weight_banks)
    return WeightReusePlan(
        reprogram_passes=phases.reprogram_passes,
        reuse_passes=phases.reuse_passes,
        params_reloaded=phases.params_reloaded,
    )
