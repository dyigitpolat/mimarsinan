"""Built programs for the static check's tests: G5's executed-residency vehicles on an integer IR (the pipeline's
order: IR mapping, the integer lattice, the IR latency pass), integer value programs with solved laws, a residency
chain, and the declaration each is checked under."""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

from mimarsinan.certification.program_check import CheckDeclaration, check_deployment
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM

from unit.mapping.residency.executed import vehicles as V

T = V.T


def integer_ir(rep, *, coalescing: bool = False, bits: int = 8):
    """The vehicle's IR on the integer lattice: every untrained perceptron takes the mapping-time grid scale
    ``floor(q_max / max |W|)`` over all its cores (``chip_quantize._matrix_scale``'s fallback, one grid per perceptron
    since E1), then the IR latency pass."""
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS,
                   allow_coalescing=coalescing).map(rep)
    for node in ir.get_neural_cores():
        node.parameter_scale = torch.tensor(0.0)
    for bank in ir.weight_banks.values():
        bank.parameter_scale = torch.tensor(0.0)
    quantize_ir_graph(ir, bits, weight_quantization=True)
    IRLatency(ir).calculate()
    return ir


def decl(semantics: str, count: int = 4, **kwargs) -> CheckDeclaration:
    """The declaration a vehicle program is checked under: its semantics' transfer, the vehicle's window and its
    declared cores (``count`` of G5's 25 x 8 type, the budget the program was built at)."""
    transfer = VERBATIM if semantics == "streamed_lif" else COLLAPSE
    base = dict(semantics=semantics, window=T, pass_transfer=transfer, core_types=tuple(V.cores(count)))
    base.update(kwargs)
    return CheckDeclaration(**base)


def run(program, ir, declaration: CheckDeclaration, **kwargs):
    return check_deployment(program, ir_graph=ir, declaration=declaration, **kwargs)


def streamed_multipass(seed: int = 0, count: int = 3):
    """A U1 program with pins and delivery delays (seed 0 at C 3: pass 0 pinned, pass 1 delayed slices)."""
    ir = integer_ir(V.random_rep(seed))
    return V.build(ir, count, None), ir


def windowed_multipass(seed: int = 0, count: int = 3):
    ir = integer_ir(V.random_rep(seed))
    return V.build(ir, count, None, retimed_level_stages=True), ir


def coalesced(seed: int = 3, count: int = 4):
    """Wide fan-ins on 25-axon cores: fused hard cores (``fused_component_axons``)."""
    ir = integer_ir(V.wide_rep(seed), coalescing=True)
    return V.build(ir, count, None, allow_coalescing=True, allow_neuron_splitting=True), ir


def residency_planned(seed: int = 0, count: int = 4, step: str = "floor"):
    """G5's chain at C 4 under the floor step: five passes, the first two pinning a core each."""
    ir = integer_ir(V.chain_repr(seed))
    return V.build(ir, count, {"step": step}), ir


def resident_chain():
    """Two integer banks x three instances on a two-core pool: three passes of one weight-resident chain."""
    rng = np.random.default_rng(19)
    banks = {b: WeightBank(id=b, core_matrix=np.round(rng.normal(size=(5, 4)) * 9 + 10 * (b + 1)).astype(np.float64))
             for b in range(2)}
    nodes = []
    for b in range(2):
        for inst in range(3):
            srcs = np.array([IRSource(-2, (b * 3 + inst) * 4 + i) for i in range(4)] + [IRSource(-3, 0)], dtype=object)
            nodes.append(NeuralCore(id=len(nodes), name=f"b{b}_i{inst}", input_sources=srcs, core_matrix=None,
                                    weight_bank_id=b, weight_row_slice=(0, 4), perceptron_index=b,
                                    perceptron_output_column=inst, latency=0, threshold=7.0,
                                    hardware_bias=np.full(4, float(b + 1) * 3)))
    ir = IRGraph(nodes=nodes, output_sources=np.array([IRSource(n.id, j) for n in nodes for j in range(4)],
                                                      dtype=object), weight_banks=banks)
    program = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 8, "max_neurons": 8, "count": 2}],
        strategy=MappingStrategy.resolve(ChipCapabilities(allow_scheduling=True)))
    return program, ir


def integer_value(prune=None):
    """G2's integer MLP with solved laws (the ``IntegerProgram``: its IR, and ``packed_mapping`` to build from it)."""
    from unit.chip_simulation.value_run.integer_programs import LAW, build_integer_program, mlp

    return build_integer_program(mlp(), (8,), num_classes=4, readout_law=LAW, prune=prune)


def value_program(passes: Optional[int] = 4, prune=None):
    """An integer MLP on value cores with solved laws (G2's builder), one 32 x 32 core: one pass per layer."""
    program = integer_value(prune)
    return program.packed_mapping(axons=32, neurons=32, count=1, passes=passes), program.ir


def drive_value_program(passes: Optional[int] = 4, *, law_inputs: bool = False):
    """[U-LMP, U-BRG review m-7] G2's integer MLP on the drive route: the Weight Quantization step's own projection
    under the readout law on cores without a bias register, so every always-on row carries its reader's drive and the
    laws read it (``build_integer_program(..., weight_quantization_step=True)``); one 32 x 32 core. ``law_inputs``
    also returns the calibrated ranges and the spec the static check's ``laws`` re-solve reads."""
    from mimarsinan.mapping.export.law_rules import ReadoutLawSpec
    from mimarsinan.mapping.export.law_stamp import readout_quantizers_by_perceptron
    from unit.chip_simulation.value_run.integer_programs import LAW, build_integer_program, mlp

    program = build_integer_program(mlp(), (8,), num_classes=4, readout_law=LAW, weight_quantization_step=True)
    packed = program.packed_mapping(axons=32, neurons=32, count=1, passes=passes)
    if not law_inputs:
        return packed, program.ir
    quantizers = readout_quantizers_by_perceptron(program.model.get_mapper_repr())
    ranges = {int(index): float(entry[2].range) for index, entry in quantizers.items()}
    return packed, program.ir, {"ranges": ranges,
                                "spec": ReadoutLawSpec.from_declaration(LAW, activation_bits=8, weight_bits=8)}


def identity_of(ir, *, retimed: bool = False):
    return build_identity_hybrid_mapping(ir_graph=ir, retimed_level_stages=retimed)


def neural(program) -> list:
    return [s for s in program.stages if s.kind == "neural"]


def clone(program):
    """A deep copy whose cores own dense grids, so a mutant can write one core's weights."""
    out = copy.deepcopy(program)
    for stage in neural(out):
        for unit in [stage] + list(getattr(stage, "retimed_level_stages", None) or ()):
            for core in unit.hard_core_mapping.cores:
                core.core_matrix = np.array(core.get_core_matrix(), dtype=np.float64)
    return out
