"""The candidate view's physics and quantity-context fragments (N0).

The axis gate (``resolve_active_specs(physics=...)``) and the view extraction
must answer from ONE source: the resolved platform the candidate decodes to.
A view built without these fragments admits an axis it cannot extract — the
N0 incident (every candidate raised at extraction while the gate said yes).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from mimarsinan.deployment_record.build.program_census import (
    CandidateProgramming, candidate_carry_census, candidate_programming_census,
)
from mimarsinan.deployment_record.build.program_census import candidate_program_passes
from mimarsinan.deployment_record.objectives import (
    candidate_context_from_platform,
)
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.mapping.lanes.declaration import lane_declaration, lanes_executed
from mimarsinan.mapping.lanes.output_stage import planned_lane_ops
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, DeployedConversion
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.chip_simulation.spiking_semantics import (
    is_cascaded_ttfs,
    is_synchronized_ttfs,
)
from mimarsinan.chip_simulation.stage_timesteps import program_latency_steps as program_latency_steps
from mimarsinan.chip_simulation.stage_timesteps import semantics_program_steps
from mimarsinan.chip_simulation.window_firing_axis import INTEGRATE_THEN_EMIT
from mimarsinan.mapping.noc.stages import execution_stage_depths
from dataclasses import replace as _dc_replace

from mimarsinan.mapping.noc import (
    LayoutNocFragments,
    collect_noc_fragments,
    execution_stage_latencies,
    execution_stage_placements,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.verification.onchip_fraction import (
    OnchipFractionEstimate,
    estimate_onchip_fractions,
)

#: (params, macs, ops) estimates of the host/on-chip split — ONE flow build.
OnchipCensus = Tuple[
    OnchipFractionEstimate, OnchipFractionEstimate, OnchipFractionEstimate,
]


def make_core_types(pcfg: Dict) -> list:
    """The declared chip's hardcore types — the mapping layer's own adapter."""
    return core_types_from_declaration(pcfg["cores"])


def candidate_latency_steps(
    softcores, noc: Optional[LayoutNocFragments], semantics: "StageSemantics",
    timesteps: Optional[int],
) -> Optional[int]:
    """[E1] The executed wall of this candidate's program, or None.

    Requires the pass/level structure (the NoC fragments carry it) and the
    run's firing semantics. Absent when either is unknown — a latency-bearing
    axis then refuses BY NAME rather than pricing a short wall. [U1 W10] A pass
    the builder pins is priced at its pinned span (the fragments' wire census),
    in the segmentation the fragments were planned under.
    """
    if noc is None or timesteps is None:
        return None
    runs = getattr(noc, "runs", None)
    transparent = tuple(bool(getattr(p, "wavefront", False)) for p in noc.pass_programs)
    stages = execution_stage_latencies(softcores, noc.pass_placements, retimed=semantics.retimed, census=noc.census,
                                       runs=runs, transparent=transparent)
    # [U-LIF2 W11] each stage's dependency depth from the wire census (never the tags)
    depths = execution_stage_depths(softcores, noc.pass_placements, retimed=semantics.retimed, census=noc.census,
                                    runs=runs, transparent=transparent)
    return semantics_program_steps(semantics, stages, depths, timesteps)


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
    #: Per (pass, segment): cells, residency marks and stage latencies (the period census's rows).
    passes: Optional[Any] = None
    #: [W08] The lane work by kind the declared lanes run on the candidate's model (the lane plan's estimate,
    #: ``planned_lane_ops``); None on a chip declaring no ``onchip_arithmetic``.
    lane_ops: Optional[Dict[str, int]] = None
    #: [U-LIF2 W12] The run's resolved window firing: the neuron cycles per window of the energy terms.
    window_firing: Optional[str] = None


def with_stage_placements(
    softcores, noc: Optional[LayoutNocFragments], semantics: "StageSemantics",
) -> Optional[LayoutNocFragments]:
    """[R3] Attach the EXECUTED per-stage placements to the fragments.

    The runner maps each execution stage from tile 0 — a re-timed program
    never runs the pass's co-resident placement, and pricing hops on it
    modeled 113.6 where the executed program measured a true 0.
    """
    if noc is None:
        return None
    return _dc_replace(noc, stage_placements=execution_stage_placements(
        softcores, noc.pass_placements, retimed=semantics.retimed, runs=getattr(noc, "runs", None),
    ))


def candidate_program_facts(
    softcores, noc: Optional[LayoutNocFragments], semantics: "StageSemantics",
    *, timesteps: Optional[int], weight_bits: Any,
    pass_transfer: Optional[str] = None,
) -> ProgramFacts:
    """The pass-structure facts of one candidate: the executed wall (E1), the
    programming census (E2), and the carry census (H2)."""
    if noc is None:
        return ProgramFacts(window_firing=semantics.window_firing)
    carry = candidate_carry_census(softcores, noc, timesteps=timesteps, pass_transfer=pass_transfer)
    return ProgramFacts(
        latency_steps=candidate_latency_steps(
            softcores, noc, semantics, timesteps,
        ),
        programming=candidate_programming_census(noc, weight_bits=weight_bits),
        carry=carry,
        passes=candidate_program_passes(softcores, noc, semantics=semantics, timesteps=timesteps),
        window_firing=semantics.window_firing,
    )


def stage_semantics_of(
    spiking_mode: str, ttfs_cycle_schedule: str, *, retimed: bool, window_firing: Optional[str] = None,
) -> "StageSemantics":
    """The executed-window branch this run's firing semantics select; [U-LIF2 W11] ``window_firing`` is the run's
    resolved value (None away from the synchronized LIF variant)."""
    return StageSemantics(
        retimed=bool(retimed),
        is_cycle=is_synchronized_ttfs(spiking_mode, ttfs_cycle_schedule),
        is_cascade=is_cascaded_ttfs(spiking_mode, ttfs_cycle_schedule),
        window_firing=window_firing,
    )


@dataclass(frozen=True)
class StageSemantics:
    """The firing semantics the executed-window rule branches on."""

    retimed: bool = False
    is_cycle: bool = False
    is_cascade: bool = False
    latency_group_count: Optional[int] = None
    chip_latency: Optional[int] = None
    window_firing: Optional[str] = None  # [U-LIF2 W11] the run's resolved lif_window_firing

    @property
    def integrates_then_emits(self) -> bool:
        return self.window_firing == INTEGRATE_THEN_EMIT


def candidate_fragments(
    pcfg: Dict,
    census: Optional[OnchipCensus] = None,
    program: Optional[ProgramFacts] = None,
):
    """This candidate's physics and quantity context, off its OWN platform."""
    payload = pcfg.get("platform_physics_resolved")
    physics = PlatformPhysics.from_dict(payload) if payload else None
    params_est, macs_est, ops_est = (
        census if census is not None else (None, None, None)
    )
    program = program if program is not None else ProgramFacts()
    context = candidate_context_from_platform(
        pcfg,
        host_macs=None if macs_est is None else int(macs_est.host),
        onchip_macs=None if macs_est is None else int(macs_est.onchip),
        host_params=None if params_est is None else int(params_est.host),
        compute_op_count=None if ops_est is None else int(ops_est.host),
        onchip_params=None if params_est is None else int(params_est.onchip),
        latency_steps=program.latency_steps,
        programming=program.programming,
        carry=program.carry,
        passes=program.passes,
        lane_ops=program.lane_ops,
        window_firing=program.window_firing,
    )
    return physics, context


def value_mapper_repr(model: Any, *, input_shape: Tuple[int, ...], num_classes: int, device: Any, target_tq: int,
                      placement: str) -> Any:
    """[W08] ``model``'s mapper graph under the value-domain packaging, the deployed program's ops (a value-domain
    run converts with it, Torch Mapping; the candidate's layout converts with the spiking default). A native
    builder's flow is its own graph. The conversion reads a copy: ``model`` is never touched."""
    if hasattr(model, "get_mapper_repr"):
        return model.get_mapper_repr()
    flow = convert_torch_model(copy.deepcopy(model), input_shape=tuple(input_shape), num_classes=num_classes,
                               device=device, Tq=target_tq, encoding_layer_placement=placement,
                               packaging=MVM_PACKAGING)
    return flow.get_mapper_repr()


def with_lane_work(program: ProgramFacts, pcfg: Dict, mapper_repr: Callable[[], Any]) -> ProgramFacts:
    """[W08] ``program`` with the lane work the chip's declared lanes run on the candidate's value-domain mapper
    graph (``mapper_repr``, built only where the lanes run; unchanged on a chip declaring no ``onchip_arithmetic``)."""
    if lane_declaration(pcfg) is None:
        return program
    work = planned_lane_ops(mapper_repr() if lanes_executed(pcfg) is not None else None, pcfg)
    return program if work is None else _dc_replace(program, lane_ops=work)


def collect_candidate_noc(
    *,
    softcores,
    core_types,
    census,
    pcfg: Dict,
    runs: Optional[Sequence[int]] = None,
) -> Optional[LayoutNocFragments]:
    """The candidate's NoC fragments, planned under its OWN capability bits.

    The capability declaration is forwarded WHOLE (``layout_kwargs``), the
    same discipline as the packing census — a fragment set planned under
    different scheduling permissions would place traffic on a program the
    chip never runs. [U1 W9] ``runs``: the walk's builder segments
    (``CandidateWalk.runs``), which the fragments carry to every consumer.
    [CL-SG] The declared core types sign each per-axon core's rows.
    """
    return collect_noc_fragments(
        softcores=softcores,
        core_types=core_types,
        census=census,
        runs=runs,
        declared_cores=pcfg.get("cores") or (),
        **ChipCapabilities.from_platform_constraints(pcfg).layout_kwargs(),
    )


@dataclass(frozen=True)
class CandidateWalk:
    """One layout walk of a candidate: its specs, host segments, wire census (None unless collected) and the builder
    segment of each spec (``residency_rules.spec_runs``), which every plane of the candidate plans with."""

    softcores: Tuple[Any, ...]
    host_segments: int
    census: Optional[Any]
    runs: Tuple[int, ...]


def compute_onchip_census(
    model: Any,
    input_shape: Tuple[int, ...],
    num_classes: int,
    placement: str,
    *,
    conversion: Optional[DeployedConversion] = None,
) -> OnchipCensus:
    """Host/on-chip param+MAC counts through the deployment's own estimator —
    one flow conversion for both metrics [H4], under the run's ``conversion``."""
    params_est, macs_est, ops_est = estimate_onchip_fractions(
        model, tuple(input_shape), int(num_classes),
        encoding_placement=placement, metrics=("params", "macs", "ops"),
        conversion=conversion,
    )
    return params_est, macs_est, ops_est
