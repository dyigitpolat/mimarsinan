"""Time-domain weight-reuse phase classification (round-1 keystone, default-off).

A deployment schedule over IR cores currently treats EVERY pass as a REPROGRAM
(load X params onto Y cores). The weight-reuse mode recognises that a conv kernel's
banks are loaded ONCE across cores (fixed mapping, max parallelism) and the spatial
positions are TIME-MULTIPLEXED through the resident banks — so only the FIRST pass
over a distinct weight bank is a (costly) reprogram and every subsequent pass over
the SAME resident bank is a (cheap) reuse pass.

The reuse-vs-reprogram boundary is ALREADY in the IR and currently DISCARDED:
``conv2d_mapper`` registers one :class:`WeightBank` per conv (``max_neurons=None`` ⇒
one bank) then attaches every spatial-position softcore to that one
``weight_bank_id``. ``{core.weight_bank_id for core in nodes}`` recovers the grouping
in O(cores). A NeuralCore with its OWN ``core_matrix`` (no shared bank) cannot be
time-multiplexed, so it counts as its own reprogram.

This module is a PURE READ of the IR: it changes no mapping/sim/build behaviour. It
exists so the cost model can tell a reuse phase apart from a reprogram phase (today
both cost identically — see ``cost_extraction``); the physical replica-fill build and
cross-pass sim residency that turn the visible savings into REAL on-hardware reuse
are deferred to later rounds and gated on this classification existing first.
"""

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
    """Per-segment phase split: N reprogram + M reuse passes, from bank grouping.

    * ``reprogram_passes`` (N) — #distinct resident weights = #distinct shared banks
      + #owned (non-shared) cores. The FIRST pass over each distinct weight is a full
      reprogram (load X params onto Y cores).
    * ``reuse_passes`` (M) — every subsequent pass that resolves to an already-resident
      bank: ``total_passes - reprogram_passes``. Cost = activation data movement at the
      sync point, NOT a parameter reload.
    * ``params_reloaded`` — Σ over the reprogram passes of the resident weight count
      (each distinct bank / owned core counted ONCE, not once per reused position) —
      the quantity the cost term charges per reload.
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

    Each core is one pass. Cores that share a ``weight_bank_id`` collapse to ONE
    reprogram (the resident bank) + the rest as reuse passes; a core with its own
    ``core_matrix`` (no bank) is always its own reprogram. ``params_reloaded`` sums
    each distinct bank's / owned core's weight count ONCE.
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
    """Schedule-wide phase split: M weight-reuse + N reprogram phases over a graph.

    ``N = reprogram_passes`` = #weight-distinct cores (the unavoidable parameter
    loads); ``M = reuse_passes`` = ``total_passes - N`` (the cheap time-multiplexed
    passes). ``sync_barrier_count`` = ``total_passes - 1`` — every pass gathers its
    positions' outputs at the segment-exit sync point, so there are that many barriers
    between passes (the design's ``(M + N - 1)``).
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
    """Build the schedule-wide :class:`WeightReusePlan` from an IR graph's cores.

    Pure read: groups the graph's NeuralCores by ``weight_bank_id`` (the conv-kernel
    grouping ``conv2d_mapper`` already emits) and classifies into reprogram vs reuse
    passes. Runs on the post-pruning IR for free (pruning shrinks the per-core counts
    that feed the same grouping), so the pruning intersection composes without extra
    machinery.
    """
    cores = graph.get_neural_cores()
    weight_banks: Dict[int, WeightBank] = dict(getattr(graph, "weight_banks", {}) or {})
    phases = classify_segment_phases(cores, weight_banks)
    return WeightReusePlan(
        reprogram_passes=phases.reprogram_passes,
        reuse_passes=phases.reuse_passes,
        params_reloaded=phases.params_reloaded,
    )
