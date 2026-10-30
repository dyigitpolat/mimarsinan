"""The flow/executor seam guard: what a per-event deployment may execute.

One keyed pass over the deployable program, run where the executor is built —
so a refusal names the offending stage before a single cycle is folded, and
no mapper needs a target-specific branch. Fold-INVARIANT transforms pass:
output tiling is per-neuron disjoint, and an identity relay (theta=1, w=1)
maps k events to k spikes exactly. Transforms that re-threshold a PARTIAL SUM
do not, and they refuse by name.
"""

from __future__ import annotations

from typing import Any, Optional

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.models.spiking.serial.refusals import (
    CycleAtomicRefusalError,
    MappingTransformRefusalError,
    SerialMembraneInitError,
)

_INTEGRALITY_EPS = 1e-9


def require_serial_deployment_admissible(
    hybrid_mapping: Any, soma_law: Optional[SomaLaw], *, membrane_init: float,
) -> None:
    """Refuse a program the per-event law cannot execute faithfully."""
    if soma_law is None or not soma_law.is_per_event:
        return
    stages = list(getattr(hybrid_mapping, "stages", []) or [])
    for stage in stages:
        if getattr(stage, "retimed_level_stages", None):
            raise CycleAtomicRefusalError(
                f"per-hop retimed level stages are refused under "
                f"firing_granularity='per_event' (stage {stage.name!r}): a "
                f"retimed hop's input is the COUNT re-encode by definition, "
                f"and the uniform re-encode emits at most one spike per "
                f"cycle — so every level boundary would DESTROY the per-cycle "
                f"multiplicity the per-event law just produced. Deploy the "
                f"fused streaming segment (lif_per_hop_retiming off)."
            )
        mapping = getattr(stage, "hard_core_mapping", None)
        if mapping is None:
            continue
        _require_fold_invariant_placements(stage, mapping)
        _require_window_start_membrane(stage, mapping, membrane_init)


def _require_fold_invariant_placements(stage: Any, mapping: Any) -> None:
    placements_per_core = getattr(
        mapping, "soft_core_placements_per_hard_core", []) or []
    for core_index, placements in enumerate(placements_per_core):
        if len(placements) > 1:
            raise MappingTransformRefusalError(
                f"core coalescing is refused under "
                f"firing_granularity='per_event' (stage {stage.name!r}, hard "
                f"core {core_index}: {len(placements)} soft cores share it). "
                f"Coalesced neurons share one axon order, so each soft core's "
                f"events interleave with the other's — and adjacency is "
                f"count-changing (§2.3). Declare allow_coalescing=false."
            )
        for placement in placements:
            if placement.get("split_group_id") is not None:
                raise MappingTransformRefusalError(
                    f"neuron splitting is refused under "
                    f"firing_granularity='per_event' (stage {stage.name!r}, "
                    f"hard core {core_index}, split group "
                    f"{placement['split_group_id']}): a split fragment "
                    f"re-thresholds a PARTIAL sum, so each fragment fires on "
                    f"its own share of the charge and the fragments' counts "
                    f"do not recombine into the whole neuron's. Declare "
                    f"allow_neuron_splitting=false."
                )


def _require_window_start_membrane(
    stage: Any, mapping: Any, membrane_init: float
) -> None:
    """§1.2's window start, both halves: ``V0*theta`` integral AND in the window.

    Integral, because the pre-charge is programmed into the neuron word's
    integer state field and rounding it would move the window start relative to
    the twins. Inside ``[0, theta)``, because the row-pair lemma's hypothesis is
    "the membrane is below theta on entry" — at ``V0*theta >= theta`` the very
    first zero-magnitude row fires and the fold stops being the serial law.
    """
    if not membrane_init:
        return
    for core_index, core in enumerate(getattr(mapping, "cores", []) or []):
        theta = float(getattr(core, "threshold", 0.0) or 0.0)
        charge = float(membrane_init) * theta
        if abs(charge - round(charge)) > _INTEGRALITY_EPS:
            raise SerialMembraneInitError(
                f"lif_membrane_init={membrane_init} gives V0*theta={charge} "
                f"on stage {stage.name!r} core {core_index} (theta={theta}), "
                f"which is NOT an integer: under "
                f"firing_granularity='per_event' the window-start membrane is "
                f"programmed into the neuron word's integer state field, so a "
                f"fractional pre-charge is unrepresentable on the deployment "
                f"and would make the torch twin disagree with the chip. "
                f"Choose V0 with V0*theta integral (0 is always legal)."
            )
        if not 0.0 <= charge < theta:
            raise SerialMembraneInitError(
                f"lif_membrane_init={membrane_init} gives V0*theta={charge} "
                f"on stage {stage.name!r} core {core_index} (theta={theta}), "
                f"outside the window [0, theta): under "
                f"firing_granularity='per_event' the window MUST start below "
                f"threshold — the row-pair lemma's hypothesis is exactly that, "
                f"and at or above theta the zero-magnitude row of every pair "
                f"fires on its own, so the deployed count would depend on "
                f"which rows are masked. Choose V0 in [0, 1) (0 is always "
                f"legal)."
            )
