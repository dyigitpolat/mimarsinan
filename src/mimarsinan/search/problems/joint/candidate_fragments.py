"""The candidate view's physics and quantity-context fragments (N0).

The axis gate (``resolve_active_specs(physics=...)``) and the view extraction
must answer from ONE source: the resolved platform the candidate decodes to.
A view built without these fragments admits an axis it cannot extract — the
N0 incident (every candidate raised at extraction while the gate said yes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_softcore_spans,
    carry_census_from_spans,
)
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
    estimate_onchip_fractions,
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


def candidate_programming_census(
    noc: Optional[LayoutNocFragments], *, weight_bits: Any,
) -> "CandidateProgramming":
    """[E2] The programming multiplicands of this candidate's pass structure.

    A resident pass contributes its cores to core INIT and nothing else — the
    weights it runs on are already installed (owner: "no we cannot charge
    every pass as reprogram").

    An undeclared ``weight_bits`` cannot size a payload, so the BYTES stay
    absent (the DMA term then refuses by name) while the core counts, which
    need no width, still count. Deployment fails loud on the same declaration
    because a sealed record must state exact bytes; a search candidate need
    not lose every programming term over one missing width.
    """
    programs = () if noc is None else noc.pass_programs
    reprogrammed = tuple(
        cells
        for program in programs if not program.resident
        for cells in program.core_cells
    )
    return CandidateProgramming(
        segment_cores=sum(program.cores for program in programs),
        reprogrammed_cores=len(reprogrammed),
        reprogrammed_bytes=None if weight_bits is None else sum(
            params_bytes(cells, weight_bits) for cells in reprogrammed
        ),
        reprogram_passes=sum(1 for p in programs if not p.resident),
        pass_cores=tuple(program.cores for program in programs),
        reprogrammed_cells=reprogrammed,
    )


@dataclass(frozen=True)
class CandidateProgramming:
    """The programming census as the quantity surface consumes it."""

    segment_cores: int
    reprogrammed_cores: int
    #: None when the platform declares no weight width — absent, never zero.
    reprogrammed_bytes: Optional[int]
    reprogram_passes: int
    pass_cores: Tuple[int, ...]
    reprogrammed_cells: Tuple[int, ...]


@dataclass(frozen=True)
class ProgramFacts:
    """What this candidate's pass structure implies for cost, or nothing.

    Every member is derived from the same pass/level resolution, so a
    candidate without a layout carries none of them and the terms that
    multiply them refuse by name. ``carry`` additionally needs the wire
    census (adjacency) and the run's transfer discipline; it is None — not
    zero — without either.
    """

    latency_steps: Optional[int] = None
    programming: Optional[CandidateProgramming] = None
    carry: Optional[Dict[str, int]] = None


def candidate_program_facts(
    softcores, noc: Optional[LayoutNocFragments], semantics: "StageSemantics",
    *, timesteps: Optional[int], weight_bits: Any,
    pass_transfer: Optional[str] = None,
) -> ProgramFacts:
    """The pass-structure facts of one candidate: the executed wall (E1), the
    programming census (E2), and the carry census (H2)."""
    if noc is None:
        return ProgramFacts()
    carry = None
    if (noc.census is not None and pass_transfer is not None
            and timesteps is not None):
        # A sealed pass structure with no crossing wire is a KNOWN zero, not
        # an unknown: an optimizer minimizing carry must see single-pass
        # programs at 0, never refuse them. Unknown stays None (no census /
        # no discipline), the E-series law.
        carry = carry_census_from_spans(
            carried_softcore_spans(
                softcores, noc.pass_placements, noc.census.pair_wires,
            ),
            boundary_count=len(noc.pass_placements),
            timesteps=int(timesteps), transfer=str(pass_transfer),
        )
    return ProgramFacts(
        latency_steps=candidate_latency_steps(
            softcores, noc, semantics, timesteps,
        ),
        programming=candidate_programming_census(noc, weight_bits=weight_bits),
        carry=carry,
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
    program: Optional[ProgramFacts] = None,
):
    """This candidate's physics and quantity context, off its OWN platform."""
    payload = pcfg.get("platform_physics_resolved")
    physics = PlatformPhysics.from_dict(payload) if payload else None
    params_est, macs_est = census if census is not None else (None, None)
    program = program if program is not None else ProgramFacts()
    context = candidate_context_from_platform(
        pcfg,
        host_macs=None if macs_est is None else int(macs_est.host),
        onchip_macs=None if macs_est is None else int(macs_est.onchip),
        host_params=None if params_est is None else int(params_est.host),
        onchip_params=None if params_est is None else int(params_est.onchip),
        latency_steps=program.latency_steps,
        programming=program.programming,
        carry=program.carry,
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
    """Host/on-chip param+MAC counts through the deployment's own estimator —
    one flow conversion for both metrics [H4]."""
    params_est, macs_est = estimate_onchip_fractions(
        model, tuple(input_shape), int(num_classes),
        encoding_placement=placement, metrics=("params", "macs"),
    )
    return params_est, macs_est
