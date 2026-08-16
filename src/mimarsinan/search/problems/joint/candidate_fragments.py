"""The candidate view's physics and quantity-context fragments (N0).

The axis gate (``resolve_active_specs(physics=...)``) and the view extraction
must answer from ONE source: the resolved platform the candidate decodes to.
A view built without these fragments admits an axis it cannot extract — the
N0 incident (every candidate raised at extraction while the gate said yes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from mimarsinan.deployment_record.objectives import (
    candidate_context_from_platform,
)
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.chip_simulation.spiking_semantics import (
    is_cascaded_ttfs,
    is_synchronized_ttfs,
)
from mimarsinan.chip_simulation.stage_timesteps import program_latency_steps
from mimarsinan.mapping.noc import (
    LayoutNocFragments,
    collect_noc_fragments,
    execution_stage_latencies,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.onchip_fraction import (
    OnchipFractionEstimate,
    estimate_onchip_fraction,
)

#: (params, macs) estimates of the host/on-chip split — one flow walk each.
OnchipCensus = Tuple[OnchipFractionEstimate, OnchipFractionEstimate]


def make_core_types(pcfg: Dict) -> list:
    """The declared chip's hardcore types, as the layout packer consumes them."""
    return [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in pcfg["cores"]
    ]


def candidate_latency_steps(
    softcores, noc: Optional[LayoutNocFragments], semantics: "StageSemantics",
    timesteps: Optional[int],
) -> Optional[int]:
    """[E1] The executed wall of this candidate's program, or None.

    Requires the pass/level structure (the NoC fragments carry it) and the
    run's firing semantics. Absent when either is unknown — a latency-bearing
    axis then refuses BY NAME rather than pricing a short wall.
    """
    if noc is None or timesteps is None:
        return None
    stages = execution_stage_latencies(
        softcores, noc.pass_placements, retimed=semantics.retimed,
    )
    return program_latency_steps(
        stage_max_latencies=stages, timesteps=int(timesteps),
        is_cycle=semantics.is_cycle, is_cascade=semantics.is_cascade,
        latency_group_count=semantics.latency_group_count,
        chip_latency=semantics.chip_latency,
    )


def stage_semantics_of(
    spiking_mode: str, ttfs_cycle_schedule: str, *, retimed: bool,
) -> "StageSemantics":
    """The executed-window branch this run's firing semantics select."""
    return StageSemantics(
        retimed=bool(retimed),
        is_cycle=is_synchronized_ttfs(spiking_mode, ttfs_cycle_schedule),
        is_cascade=is_cascaded_ttfs(spiking_mode, ttfs_cycle_schedule),
    )


@dataclass(frozen=True)
class StageSemantics:
    """The firing semantics the executed-window rule branches on."""

    retimed: bool = False
    is_cycle: bool = False
    is_cascade: bool = False
    latency_group_count: Optional[int] = None
    chip_latency: Optional[int] = None


def candidate_fragments(
    pcfg: Dict,
    census: Optional[OnchipCensus] = None,
    latency_steps: Optional[int] = None,
):
    """This candidate's physics and quantity context, off its OWN platform."""
    payload = pcfg.get("platform_physics_resolved")
    physics = PlatformPhysics.from_dict(payload) if payload else None
    params_est, macs_est = census if census is not None else (None, None)
    context = candidate_context_from_platform(
        pcfg,
        host_macs=None if macs_est is None else int(macs_est.host),
        onchip_macs=None if macs_est is None else int(macs_est.onchip),
        host_params=None if params_est is None else int(params_est.host),
        onchip_params=None if params_est is None else int(params_est.onchip),
        latency_steps=latency_steps,
    )
    return physics, context


def collect_candidate_noc(
    *,
    softcores,
    core_types,
    census,
    pcfg: Dict,
) -> Optional[LayoutNocFragments]:
    """The candidate's NoC fragments, planned under its OWN capability bits.

    The capability declaration is forwarded WHOLE (``layout_kwargs``), the
    same discipline as the packing census — a fragment set planned under
    different scheduling permissions would place traffic on a program the
    chip never runs.
    """
    return collect_noc_fragments(
        softcores=softcores,
        core_types=core_types,
        census=census,
        **ChipCapabilities.from_platform_constraints(pcfg).layout_kwargs(),
    )


def compute_onchip_census(
    model: Any,
    input_shape: Tuple[int, ...],
    num_classes: int,
    placement: str,
) -> OnchipCensus:
    """Host/on-chip param+MAC counts through the deployment's own estimator."""
    return (
        estimate_onchip_fraction(
            model, tuple(input_shape), int(num_classes),
            encoding_placement=placement, metric="params",
        ),
        estimate_onchip_fraction(
            model, tuple(input_shape), int(num_classes),
            encoding_placement=placement, metric="macs",
        ),
    )
