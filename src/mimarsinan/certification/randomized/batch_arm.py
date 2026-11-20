"""[thru:w4] The batch pipeline's randomized arm (PL/thru/design.md (d) 'The randomized test's extension').

Each instance is the X3a network of the arm's stream applied at P positions, its weights shared (``program.mapper_graph``
with ``positions``: every neural layer a bank of P instances, the form position multiplexing plans), under a semantics
where pipelined multiplexing is legal (the value domain, streamed LIF, every TTFS variant). Its declaration is the cores
as the arm ``on_scheduled`` draws them with both base forms on (``position_multiplexing``, ``pipelined_multiplexing``),
``layer_replication`` on or off, ``wavefront_group_cores`` at all, 3/4 or 1/2 of the drawn cores, and B absent, 2, 3 or
5 (the chip stating as many carry slots), at a pass budget that never binds; and it runs K inputs, K from 1, B, B + 1
and 2B + 1 (partial rounds included). Every output bit of every input of the built program is compared with the
identity program of the same IR run on each input alone, and again with the inputs in reverse order. A deployment the
toolchain refuses by name is a refusal; a run that differs is a counterexample (``ablate`` reruns its seed with each of
the arm's keys taken back, the runner's ablation)."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import mimarsinan.certification.randomized.executors as ex
import mimarsinan.certification.randomized.program as program
from mimarsinan.certification.randomized.arms import ARMS, ON_BATCH_PIPELINE
from mimarsinan.certification.randomized.cost_draws import platform_constraints
from mimarsinan.certification.randomized.instance import VALUE, Instance
from mimarsinan.certification.randomized.runner import (
    COUNTEREXAMPLE, EQUAL, REFUSED, REFUSED_UNNAMED, RUN_ERROR, UNNAMED_FAILURES, VACUOUS, instance_for,
    stratum_seed, targeted_declaration)
from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.value_schema import judge_declaration
from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.wavefront_stages import input_slot, program_slots, wavefront_head
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

ARM_ID = ON_BATCH_PIPELINE
SLOT_DRAWS: Tuple[Optional[int], ...] = (None, 2, 3, 5)
LIMITS = ("all", "3/4", "1/2")
_QUARTERS = {"all": 4, "3/4": 3, "1/2": 2}
POSITIONS = (2, 6)
#: The pass budget the arm declares: large enough never to bind, so a refusal is never the budget's.
PASS_BUDGET = 1 << 16
#: The keys the arm draws beyond its stream's declaration (each one its ablation can take back).
ARM_KEYS = ("layer_replication", "wavefront_group_cores", "batch_pipeline_slots")


@dataclass(frozen=True)
class BatchDraw:
    """One instance's draw beyond its network: positions, the four keys, the inputs K and the declaration."""

    seed: int
    positions: int
    replication: bool
    limit: str
    slots: Optional[int]
    inputs: int
    declaration: Dict[str, Any]
    values: Tuple[Tuple[int, ...], ...] = ()


def draw(seed: int) -> Tuple[Instance, BatchDraw]:
    """The instance of ``seed`` in the arm's stratum and its draw of the four keys and the inputs."""
    arm = ARMS[ARM_ID]
    instance = instance_for(seed, arm)
    declaration = targeted_declaration(seed, arm)
    rng = random.Random(f"{ARM_ID}:{stratum_seed(seed, arm)}")
    positions = rng.randint(*POSITIONS)
    replication = rng.random() < 0.5
    limit = rng.choice(LIMITS)
    slots = rng.choice(SLOT_DRAWS)
    b = slots or 1
    k = rng.choice((1, b, b + 1, 2 * b + 1))
    lo, hi = (0, 8) if instance.semantics == VALUE else (0, instance.timesteps)
    width = positions * instance.network.input_width
    values = tuple(tuple(rng.randint(lo, hi) for _ in range(width)) for _ in range(k))
    declaration.pop("residency_plan", None)  # the arm targets the base forms; a plan composes a different program
    declaration.update(position_multiplexing=True, pipelined_multiplexing=True, max_schedule_passes=PASS_BUDGET)
    if replication:
        cores = sum(int(core["count"]) for core in declaration["cores"])
        declaration.update(layer_replication=True, wavefront_group_cores=max(1, cores * _QUARTERS[limit] // 4))
    if slots is not None:
        declaration.update(batch_pipeline_slots=slots, carry_input_slots=slots)
    for key, value in declaration.items():
        judge_declaration(REGISTRY[key], value, run_domain=None)
    return instance, BatchDraw(int(seed), positions, replication, limit, slots, k, declaration, values)


def _deployment(instance: Instance, d: BatchDraw, constraints: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    """The IR at the declared geometry (relabelled), its identity program and the built program."""
    arm = ARMS[ARM_ID]
    geometry = resolve_platform_mapping_params(constraints["cores"],
                                               allow_coalescing=bool(constraints["allow_coalescing"]))
    run = ex.RUNS[instance.semantics]
    ir = program.integer_ir(program.mapper_graph(instance, arm.encoding_placement, positions=d.positions),
                            max_axons=geometry.effective_max_axons, max_neurons=geometry.effective_max_neurons,
                            allow_coalescing=geometry.allow_coalescing, hardware_bias=geometry.hardware_bias,
                            law=run.law)
    program.relabel(ir, d.seed)
    identity = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=run.retimed)
    return ir, identity, build_hybrid_mapping_for_pipeline(ir, dict(constraints), pipeline_config=dict(run.config))


def inputs(instance: Instance, d: BatchDraw) -> torch.Tensor:
    """The K inputs as the executors read them (``executors.input_tensor``'s rule: the midpoint rate of each count
    under a spiking semantics, the integer values under the value one)."""
    x = torch.tensor(d.values, dtype=torch.float64)
    if instance.semantics != VALUE:
        x = (x + 0.5) / float(instance.timesteps)
    return x


def outputs(mapping: Any, instance: Instance, x: torch.Tensor) -> torch.Tensor:
    """The program's outputs on ``x`` under the instance's semantics (the flows, which run a program of input slots
    slot by slot)."""
    run = ex.RUNS[instance.semantics]
    with torch.no_grad():
        if run.decode == "value":
            return ValueHybridCoreFlow(mapping, device=VALUE_ARITHMETIC.device, dtype=torch.float64)(x).double()
        kwargs = dict(run.flow)
        kwargs.setdefault("spike_mode", "Uniform")
        point = resolve_spiking_point(dict(run.config))
        flow = SpikingHybridCoreFlow((int(x.shape[1]),), mapping, simulation_length=instance.timesteps,
                                     thresholding_mode="<=", pass_transfer=run.transfer,
                                     lif_execution_synchronized=point.integrates_then_emits, spiking_point=point,
                                     **kwargs)
        return flow(x.float()).double()


def differing_bits(got: torch.Tensor, want: torch.Tensor) -> int:
    """Outputs whose bits differ (the sign of a zero included)."""
    a = np.ascontiguousarray(got.reshape(want.shape).double().numpy())
    b = np.ascontiguousarray(want.double().numpy())
    return int(np.count_nonzero(a.view(np.int64) != b.view(np.int64)))


@dataclass
class BatchVerdict:
    seed: int
    semantics: str
    positions: int
    replication: bool
    limit: str
    slots: Optional[int]
    inputs: int
    declaration: Dict[str, Any]
    outcome: str = ""
    detail: str = ""
    refusal: str = ""
    stages: int = 0
    passes: int = 0
    groups: int = 0
    copies: bool = False
    shared_passes: int = 0
    program_slots: Optional[int] = None
    output_width: int = 0
    outputs_compared: int = 0
    differ: int = 0
    differ_reverse: int = 0
    active: float = 0.0
    wall_s: float = 0.0
    without: Tuple[str, ...] = field(default_factory=tuple)
    #: Where a refusal came: the build (``deployment``) or an executor (``run``).
    stage: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _structure(mapping: Any, verdict: BatchVerdict) -> None:
    neural = [s for s in mapping.stages if s.kind == "neural"]
    verdict.stages = len(neural)
    by_pass: Dict[Tuple[Any, Any], set] = {}
    for stage in neural:
        slot = input_slot(stage)
        by_pass.setdefault((stage.schedule_segment_index, stage.schedule_pass_index), set()).add(
            None if slot is None else int(slot.slot))
    verdict.passes = len(by_pass)
    verdict.shared_passes = sum(1 for slots in by_pass.values() if len(slots) > 1)
    verdict.groups = sum(1 for s in neural if wavefront_head(s) is not None)
    verdict.copies = any(_copies_run(stage) for stage in neural)
    verdict.program_slots = program_slots(mapping.stages)


def _copies_run(stage: Any) -> bool:
    """Whether a pass runs two instances of one bank at once (one copy runs one instance per pass, so two copies)."""
    instances: Dict[int, set] = {}
    for records in stage.hard_core_mapping.soft_core_placements_per_hard_core:
        for record in records:
            if record.get("weight_bank_id") is not None:
                instances.setdefault(int(record["weight_bank_id"]), set()).add(int(record["ir_node_id"]))
    return any(len(nodes) > 1 for nodes in instances.values())


def run_instance(seed: int, *, without: Tuple[str, ...] = ()) -> BatchVerdict:
    """One instance of the arm; ``without`` takes back the named keys of its draw (the ablation)."""
    started = time.perf_counter()
    instance, d = draw(seed)
    declaration = {k: v for k, v in d.declaration.items() if k not in without}
    if "layer_replication" in without:
        declaration.pop("wavefront_group_cores", None)
    if "batch_pipeline_slots" in without:
        declaration.pop("carry_input_slots", None)
    verdict = BatchVerdict(seed=int(seed), semantics=instance.semantics, positions=d.positions,
                           replication=d.replication, limit=d.limit, slots=d.slots, inputs=d.inputs,
                           declaration=declaration, without=tuple(without))
    constraints = platform_constraints(declaration, hardware_bias=instance.hardware_bias)
    try:
        _ir, identity, deployed = _deployment(instance, d, constraints)
    except ToolchainRefusal as exc:
        verdict.stage = "deployment"
        return _close(verdict, REFUSED, exc, started)
    except UNNAMED_FAILURES as exc:
        verdict.stage = "deployment"
        return _close(verdict, REFUSED_UNNAMED, exc, started)
    _structure(deployed, verdict)
    try:
        x = inputs(instance, d)
        want = torch.cat([outputs(identity, instance, x[k:k + 1]) for k in range(int(x.shape[0]))])
        got = outputs(deployed, instance, x)
        back = outputs(deployed, instance, x.flip(0)).flip(0)
    except ToolchainRefusal as exc:  # an executor refusing the program by name (as it refuses the identity's)
        verdict.stage = "run"
        return _close(verdict, REFUSED, exc, started)
    except UNNAMED_FAILURES as exc:
        return _close(verdict, RUN_ERROR, exc, started)
    verdict.output_width = int(want.shape[1])
    verdict.outputs_compared = 2 * int(want.numel())
    verdict.differ, verdict.differ_reverse = differing_bits(got, want), differing_bits(back, want)
    verdict.active = float((want != 0).double().mean()) if want.numel() else 0.0
    outcome = COUNTEREXAMPLE if verdict.differ or verdict.differ_reverse else (
        EQUAL if verdict.active > 0 else VACUOUS)
    return _close(verdict, outcome, None, started)


def ablate(seed: int) -> Dict[str, str]:
    """The runner's ablation of a counterexample: its seed rerun with each of the arm's keys taken back, alone and
    together, each outcome by the keys it takes back."""
    variants = (("batch_pipeline_slots",), ("layer_replication",), ("batch_pipeline_slots", "layer_replication"))
    return {"+".join(keys): run_instance(seed, without=keys).outcome for keys in variants}


def _close(verdict: BatchVerdict, outcome: str, exc: Optional[BaseException], started: float) -> BatchVerdict:
    verdict.outcome = outcome
    if exc is not None:
        verdict.refusal = type(exc).__name__
        verdict.detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    verdict.wall_s = round(time.perf_counter() - started, 4)
    return verdict
