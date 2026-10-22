"""The X3a code-level runner: one instance, its identity reference and one drawn deployment, compared per neuron."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

import mimarsinan.certification.randomized.executors as ex
import mimarsinan.certification.randomized.network as network
import mimarsinan.certification.randomized.program as program
from mimarsinan.certification.randomized.arms import ARMS, Arm, scaled
from mimarsinan.certification.randomized.cost_draws import (
    DEFAULT_DRAWS,
    DrawSpace,
    draw_declaration,
    drawn_paths,
    platform_constraints,
)
from mimarsinan.certification.randomized.generator import DEFAULT_SPACE, GeneratorSpace, generate
from mimarsinan.certification.randomized.instance import NEURAL, Instance
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

#: Failures the runner records as measured outcomes. A refusal by name is a ToolchainRefusal; any other of
#: these is a refusal or crash without a name, which the screen reports as a finding.
UNNAMED_FAILURES = (ValueError, AssertionError, RuntimeError, KeyError, IndexError, TypeError,
                    ZeroDivisionError, AttributeError, NotImplementedError)

EQUAL, COUNTEREXAMPLE, REFUSED, REFUSED_UNNAMED = "equal", "counterexample", "refused", "refused_unnamed"
REFERENCE_ERROR, RUN_ERROR, VACUOUS = "reference_error", "run_error", "vacuous"
OUTCOMES = (EQUAL, COUNTEREXAMPLE, REFUSED, REFUSED_UNNAMED, REFERENCE_ERROR, RUN_ERROR, VACUOUS)


@dataclass
class Verdict:
    seed: int
    arm: str
    semantics: str
    motifs: Tuple[str, ...]
    declaration: Dict[str, Any]
    drawn: Tuple[str, ...]
    outcome: str = ""
    detail: str = ""
    stages: int = 0
    multipass: bool = False
    carried: bool = False
    coresident: bool = False
    coalesced: bool = False
    split: bool = False
    windows: int = 0
    mismatches: int = 0
    first: Optional[Tuple[int, int, int]] = None
    active: float = 0.0
    wall_s: float = 0.0
    classes: Tuple[str, ...] = field(default_factory=tuple)
    #: Proposition 2's relation: the identity program against the integer network N (``network.claim``).
    network_claim: str = ""
    network_windows: int = 0
    network_mismatches: int = 0
    network_first: Optional[Tuple[int, int, int]] = None
    #: Fraction of the identity program's neurons whose output differs across the input batch.
    input_dependent: float = 0.0
    #: Per neural layer in document order, the largest weight magnitude (0: an all-zero layer).
    weight_ranges: Tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stratum_seed(seed: int, arm: Arm) -> int:
    """The generator seed of ``seed`` in ``arm``'s stratum: the seed itself, or (for an arm with its own
    ``stream``) a 48-bit draw keyed on the stream, so strata of one seed range draw disjoint networks."""
    if not arm.stream:
        return int(seed)
    return int.from_bytes(hashlib.sha256(f"{arm.stream}:{int(seed)}".encode()).digest()[:6], "big")


def semantics_for(seed: int, arm: Arm) -> str:
    return random.Random(f"semantics:{stratum_seed(seed, arm)}").choice(arm.semantics)


def targeted_declaration(seed: int, arm: Arm, space: DrawSpace = DEFAULT_DRAWS) -> Dict[str, Any]:
    """The field-by-field draw, inside the ranges an arm targets (``axons``, ``neurons``, ``count``) and with
    the flags it pins; a targeting narrows the drawn ranges, it never sets a field the registry forbids."""
    narrowed = {k: v for k, v in arm.declaration.items() if k in ("axons", "neurons", "count")}
    declaration = draw_declaration(stratum_seed(seed, arm), space.__class__(**{**asdict(space), **narrowed}))
    declaration.update({k: v for k, v in arm.declaration.items() if k not in narrowed})
    return declaration


def instance_for(seed: int, arm: Arm, space: GeneratorSpace = DEFAULT_SPACE) -> Instance:
    """The instance of ``seed`` in ``arm``'s stratum (its window range, if it targets one, replacing ``space``'s)."""
    if arm.timesteps is not None:
        space = replace(space, timesteps=arm.timesteps)
    return scaled(generate(stratum_seed(seed, arm), space, semantics=semantics_for(seed, arm), motifs=arm.motifs),
                  arm)


@dataclass
class Reference:
    """The identity-mapped single-pass program at unbounded geometry (the unpartitioned program), run."""

    ir: Any
    mapping: Any
    nodes: Dict[int, torch.Tensor]
    perceptron_of: Dict[str, int]


def reference(instance: Instance, arm: Arm) -> Reference:
    mapped = program.mapper_graph(instance)
    ir = program.integer_ir(mapped, max_axons=None, max_neurons=None, allow_coalescing=False,
                            hardware_bias=instance.hardware_bias)
    mapping = build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=ex.RUNS[instance.semantics].retimed)
    return Reference(ir, mapping, _run(mapping, instance, arm, identity=True), dict(mapped.perceptron_of))


def reference_nodes(instance: Instance, arm: Arm):
    """``(ir, mapping, nodes)`` of :func:`reference`."""
    ref = reference(instance, arm)
    return ref.ir, ref.mapping, ref.nodes


def network_relation(instance: Instance, ref: Reference) -> Tuple[str, int, int, Optional[Tuple[int, int, int]]]:
    """``(claim, windows, mismatches, first (perceptron, sample, neuron))``: the identity program against N.

    Measured where Proposition 2 speaks: judged where it claims N, reported but never judged where its
    claim is conditional on the data-dependent H2-H4. Outside it (cascaded TTFS's greedy schedule, analytical
    TTFS's real-valued activations) nothing is compared."""
    status = network.claim(instance.semantics)
    if status == network.NOT_CLAIMED:
        return status, 0, 0, None
    want = network.evaluate(instance)
    got = ex.per_neuron(ref.ir, ref.nodes)
    windows, mismatches, first = 0, 0, None
    for layer in instance.network.neural_layers():
        index = ref.perceptron_of[layer.name]
        a = torch.from_numpy(network.observable(instance, want[layer.name]))
        b = got[index]
        if a.shape != b.shape:
            raise ex.CoverageError(f"layer {layer.name}: N shape {tuple(a.shape)} against {tuple(b.shape)}")
        differs = a != b
        windows += int(a.numel())
        mismatches += int(differs.sum())
        if first is None and bool(differs.any()):
            sample, neuron = (int(v) for v in torch.nonzero(differs)[0])
            first = (index, sample, neuron)
    return status, windows, mismatches, first


def deployment(instance: Instance, constraints: Mapping[str, Any], seed: int, arm: Arm, *, relabel: bool = True,
               identity: bool = False):
    """The IR at the declared geometry (relabelled), built through the pipeline's own entry point."""
    geometry = resolve_platform_mapping_params(constraints["cores"],
                                               allow_coalescing=bool(constraints["allow_coalescing"]))
    ir = program.integer_ir(program.mapper_graph(instance), max_axons=geometry.effective_max_axons,
                            max_neurons=geometry.effective_max_neurons, allow_coalescing=geometry.allow_coalescing,
                            hardware_bias=geometry.hardware_bias)
    if relabel:
        program.relabel(ir, seed)
    run = ex.RUNS[instance.semantics]
    if identity:
        return ir, build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=run.retimed)
    return ir, build_hybrid_mapping_for_pipeline(ir, dict(constraints), pipeline_config=dict(run.config))


def _run(mapping, instance: Instance, arm: Arm, *, identity: bool = False) -> Dict[int, torch.Tensor]:
    torch.manual_seed(instance.seed)
    dtype = torch.float32 if arm.value_dtype == "float32" else torch.float64
    transfer = None if identity else arm.transfer
    return ex.run_nodes(mapping, instance, transfer=transfer, value_dtype=dtype, spike_mode=arm.encoder,
                        phase_dither=arm.phase_dither)


def compare(ref_ir, ref_nodes, dep_ir, dep_nodes) -> Tuple[int, int, Optional[Tuple[int, int, int]]]:
    """``(windows, mismatches, first divergent (perceptron, sample, neuron))`` at atol 0, every neuron."""
    want, got = ex.per_neuron(ref_ir, ref_nodes), ex.per_neuron(dep_ir, dep_nodes)
    if set(want) != set(got):
        raise ex.CoverageError(f"perceptrons differ: reference {sorted(want)}, deployment {sorted(got)}")
    windows, mismatches, first = 0, 0, None
    for index in sorted(want):
        a, b = want[index], got[index]
        if a.shape != b.shape:
            raise ex.CoverageError(f"perceptron {index}: shape {tuple(a.shape)} against {tuple(b.shape)}")
        differs = ~((a == b) | (torch.isnan(a) & torch.isnan(b)))
        windows += int(a.numel())
        mismatches += int(differs.sum())
        if first is None and bool(differs.any()):
            sample, neuron = (int(v) for v in torch.nonzero(differs)[0])
            first = (index, sample, neuron)
    return windows, mismatches, first


def structure(mapping) -> Dict[str, bool]:
    """What the deployment exercised: passes, carried wires, co-resident soft cores, fused or split cores."""
    passes: Dict[int, int] = {}
    coresident = coalesced = split = False
    seen_slices: Dict[Tuple[int, int], int] = {}
    for stage in mapping.stages:
        if stage.kind != "neural":
            continue
        if stage.schedule_segment_index is not None:
            passes[stage.schedule_segment_index] = passes.get(stage.schedule_segment_index, 0) + 1
        hcm = stage.hard_core_mapping
        for core in hcm.cores:
            coalesced |= bool(getattr(core, "fused_component_axons", None))
        for placements in getattr(hcm, "soft_core_placements_per_hard_core", ()) or ():
            coresident |= len(placements) > 1
            for p in placements:
                key = (int(p.get("ir_node_id", -1)), id(stage))
                seen_slices[key] = seen_slices.get(key, 0) + 1
    split = any(n > 1 for n in seen_slices.values())
    return {"multipass": any(n > 1 for n in passes.values()), "carried": bool(carried_outputs_by_stage(mapping.stages)),
            "coresident": coresident, "coalesced": coalesced, "split": split}


def run_instance(seed: int, arm_id: str = "on", *, control: bool = False, space: GeneratorSpace = DEFAULT_SPACE,
                 draws: DrawSpace = DEFAULT_DRAWS) -> Verdict:
    """One instance of ``arm_id``; ``control`` runs the paired control (the arm's targeting, its field on)."""
    arm = ARMS[arm_id].control() if control else ARMS[arm_id]
    started = time.perf_counter()
    instance = instance_for(seed, arm, space)
    declaration = targeted_declaration(seed, arm, draws)
    constraints = platform_constraints(declaration, hardware_bias=instance.hardware_bias)
    verdict = Verdict(seed=seed, arm=arm_id, semantics=instance.semantics, motifs=instance.motifs,
                      declaration=declaration, drawn=tuple(sorted(drawn_paths(declaration))))
    verdict.weight_ranges = tuple(max((abs(w) for row in layer.weights for w in row), default=0)
                                  for layer in instance.network.layers if layer.kind == NEURAL)
    try:
        ref = reference(instance, arm)
        (verdict.network_claim, verdict.network_windows, verdict.network_mismatches,
         verdict.network_first) = network_relation(instance, ref)
    except UNNAMED_FAILURES + (ToolchainRefusal,) as exc:
        return _close(verdict, REFERENCE_ERROR, exc, started)
    ref_ir, ref_nodes = ref.ir, ref.nodes
    with arm.engaged():
        try:
            dep_ir, dep_map = deployment(instance, constraints, seed, arm)
        except ToolchainRefusal as exc:
            return _close(verdict, REFUSED, exc, started)
        except UNNAMED_FAILURES as exc:
            return _close(verdict, REFUSED_UNNAMED, exc, started)
        verdict.stages = sum(1 for s in dep_map.stages if s.kind == "neural")
        for name, flag in structure(dep_map).items():
            setattr(verdict, name, flag)
        try:
            dep_nodes = _run(dep_map, instance, arm)
            verdict.windows, verdict.mismatches, verdict.first = compare(ref_ir, ref_nodes, dep_ir, dep_nodes)
        except UNNAMED_FAILURES + (ToolchainRefusal,) as exc:
            return _close(verdict, RUN_ERROR, exc, started)
    placed = ex.per_neuron(ref_ir, ref_nodes)
    flat = torch.cat([v.flatten() for v in placed.values()])
    verdict.active = float((flat != 0).double().mean()) if flat.numel() else 0.0
    varying = torch.cat([(v != v[:1]).any(dim=0) for v in placed.values()])
    verdict.input_dependent = float(varying.double().mean()) if varying.numel() else 0.0
    outcome = COUNTEREXAMPLE if verdict.mismatches else (EQUAL if verdict.active > 0 else VACUOUS)
    return _close(verdict, outcome, None, started)


def _close(verdict: Verdict, outcome: str, exc: Optional[BaseException], started: float) -> Verdict:
    verdict.outcome = outcome
    if exc is not None:
        verdict.detail = f"{type(exc).__name__}: {str(exc)[:300]}"
    verdict.wall_s = round(time.perf_counter() - started, 4)
    return verdict
