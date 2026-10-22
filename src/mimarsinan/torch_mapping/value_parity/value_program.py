"""Value-probe stage 2: a value-domain conversion's identity value program against the native model."""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import gather_final_output_torch
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.support.shape_probe import HostValueBatchLayoutError
from mimarsinan.torch_mapping.converter_handlers.symbolic.pass_result import OpaqueBatchValueError
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import ImpureShapeProgramError
from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.value_parity.contract import FunctionContract
from mimarsinan.torch_mapping.value_parity.localization import (
    DivergingNode,
    first_disagreement,
    mapper_label,
    native_node_values,
    node_operator,
    trace_native,
)
from mimarsinan.torch_mapping.value_parity.probe import Locator, ProbeWitness, judge
from mimarsinan.torch_mapping.value_parity.rounding import (
    declared_rounding_bound,
    lowering_value_bound,
    reassociation_bound,
)
from mimarsinan.torch_mapping.value_parity.verdict import STAGE_VALUE_PROGRAM, ValueParityResult, describe_failure


class LoweredProgram(NamedTuple):
    """An fp64 identity value program and, per labelled mapper, the IR sources of its output."""

    program: ValueHybridCoreFlow
    mapper_sources: List[Tuple[str, np.ndarray]]


def fp64_identity_program(ir_graph) -> ValueHybridCoreFlow:
    """The fp64 identity value program of an IR graph (one hard core per neural core, float weights)."""
    return ValueHybridCoreFlow(
        build_identity_hybrid_mapping(ir_graph=ir_graph), device="cpu", dtype=torch.float64,
    )


def _labelled_ir_sources(mapper_repr) -> List[Tuple[str, np.ndarray]]:
    order, _, _ = mapper_repr._ensure_exec_graph()
    labelled = []
    for mapper in order:
        label, sources = mapper_label(mapper), getattr(mapper, "_ir_sources", None)
        if label is not None and isinstance(sources, np.ndarray) and sources.dtype == object:
            labelled.append((label, sources.copy()))
    return labelled


def lower_identity_program(flow) -> LoweredProgram:
    """Map ``flow`` to an unbounded-core, uncoalesced IR and build its fp64 identity value program.

    The value program of an impure program (``flow.shape_impurity``, ``symbolic/purity.py``) is refused by name: no
    per-sample shape of it is proved, and the program fixes every one; so is that of a program with an opaque batch
    value (``flow.shape_opaque``, review round 14), naming the first one. A host op whose value does not carry the batch
    on axis 0 alone (``batch_layout_violation``, set by the converter) is refused by name too: the program's buffers
    hold one row per sample.
    """
    impurity = getattr(flow, "shape_impurity", None)
    if impurity is not None:
        raise ImpureShapeProgramError(impurity)
    opaque = getattr(flow, "shape_opaque", None)
    if opaque is not None:
        raise OpaqueBatchValueError(*opaque)
    mapper_repr = flow.get_mapper_repr()
    order, _, _ = mapper_repr._ensure_exec_graph()
    for mapper in order:
        violation = getattr(mapper, "batch_layout_violation", None)
        if violation is not None:
            raise HostValueBatchLayoutError(
                getattr(mapper, "name", None), violation, getattr(mapper, "host_operator", None),
            )
    try:
        ir = IRMapping(
            q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
            allow_coalescing=False, hardware_bias=False,
        ).map(mapper_repr)
        sources = _labelled_ir_sources(mapper_repr)
    finally:
        mapper_repr.clear_ir_caches()
    return LoweredProgram(fp64_identity_program(ir), sources)


def identity_value_program(flow) -> ValueHybridCoreFlow:
    """The fp64 identity value program of a converted flow: unbounded cores, float weights, no coalescing."""
    return lower_identity_program(flow).program


def _locate_in_program(lowered: LoweredProgram) -> Locator:
    """First labelled mapper whose IR output disagrees with its native FX node (the witness has no pruning)."""

    def locate(witness: ProbeWitness, _candidate: nn.Module, bound: float) -> Optional[DivergingNode]:
        with torch.no_grad():
            gm = trace_native(witness.traced, witness.input_shape)
            native = native_node_values(gm, witness.inputs)
            _, state, x_flat = lowered.program.run_retaining_state(witness.inputs)
        batch = int(witness.inputs.shape[0])
        labelled = []
        for label, sources in lowered.mapper_sources:
            try:
                values = gather_final_output_torch(
                    sources, state, x_flat, batch, torch.device("cpu"), torch.float64,
                )
            except KeyError:
                continue
            labelled.append((label, values))
        label = first_disagreement(labelled, native, witness.conversion.absorption_plan, bound)
        return None if label is None else DivergingNode(label, node_operator(gm, label))

    return locate


def _lowering_line(contract: FunctionContract):
    """Stage 2's line on a comparison above tau(P): tau(P) + beta(P) at its inputs, capped at the acceptance; beta(P)
    is visited until one output column covers the error's excess over tau(P) (the same verdict)."""

    def derive(witness: ProbeWitness, inputs: torch.Tensor, error: float) -> Tuple[float, Optional[float]]:
        depth = witness.weight_layer_depth or 0
        beta = reassociation_bound(witness.traced, inputs, contract.select_output,
                                   enough=error - declared_rounding_bound(depth))
        return lowering_value_bound(depth, beta), beta

    return derive


def identity_value_program_stage(
    witness: ProbeWitness, *, contract: FunctionContract, bound: Optional[float], context: str,
) -> ValueParityResult:
    """Stage 2: lower the stage-1 witness conversion to IR and judge its identity value program.

    Value-domain packaging only: an event-domain identity program is a spiking simulation, judged by the
    NF-SCM parity gate instead. ``bound=None``: tau(P), widened by beta(P) where a comparison exceeds it.
    """
    line = witness.bound if bound is None else bound
    lowered, exc = guarded(lambda: lower_identity_program(witness.conversion.flow))
    if exc is not None:
        return ValueParityResult(
            ok=False, context=context, stage=STAGE_VALUE_PROGRAM, contract=contract.name, bound=line,
            samples=int(witness.inputs.shape[0]), weight_layer_depth=witness.weight_layer_depth,
            randomized_parameters=witness.randomized_parameters, error=exc,
            reason=f"IR lowering of the witness failed: {describe_failure(exc)}",
        )
    assert lowered is not None
    return judge(
        witness, lowered.program, stage=STAGE_VALUE_PROGRAM, contract=contract, bound=line,
        context=context, locate=_locate_in_program(lowered), flat_output=True,
        derive_line=_lowering_line(contract) if bound is None else None,
    )
