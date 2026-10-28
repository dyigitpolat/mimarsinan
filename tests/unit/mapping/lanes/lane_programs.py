"""Lane programs (WI-5e) built in the pipeline's step order through the steps' own hooks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.ir import ComputeOp, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_absorbed, measure_lane_ranges
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.mapping.support.core_types import bind_core_types, check_core_codes, install_layer_weight_bits
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.mapping.support.quantizer_state import QUANTIZER_STATE_FORMAT
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.declared_state import (
    declare_entry_state,
    declare_readout_state,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.transformations.quantization_bounds import layer_weight_bits, quantization_bounds
from unit.chip_simulation.value_run.integer_programs import IntegerProgram, _zero_channels

LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 24}
ALL = ("L", "E", "P")


def lane_platform(classes: Optional[Sequence[str]] = ALL, *, activation_bits: int = 4, readout_law=None,
                  **extra) -> dict:
    platform = {"activation_bits": activation_bits, "readout_law": dict(readout_law or LAW), **extra}
    if classes is not None:
        platform["onchip_arithmetic"] = {"classes": list(classes)}
    return platform


def write_state(path, perceptrons, lanes=None) -> dict:
    document = {"format": QUANTIZER_STATE_FORMAT, "perceptrons": perceptrons}
    if lanes:
        document["lanes"] = lanes
    path.write_text(json.dumps(document, sort_keys=True))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_lane_program(model: nn.Module, input_shape: tuple, *, num_classes: int, platform: dict,
                       weight_bits: int = 4, max_axons=None, max_neurons=None, prune: Optional[dict] = None,
                       hardware_bias: bool = False, seed: int = 0, calibration_size: int = 32,
                       state_hook=None, allow_coalescing: bool = False) -> IntegerProgram:
    """Boundary Quantization (lane-aware entries), Readout Law (lanes installed, ranges at lane ends), the
    integer weight grid, IR mapping, quantization, pruning and the law hook (lanes fused), as the pipeline orders
    them. ``state_hook(flow, entries, grids, perceptrons, quantizers, lanes)`` may write a declared map into
    ``platform`` before the declarations are read."""
    torch.manual_seed(seed)
    flow = convert_torch_model(model.eval(), input_shape, num_classes, device="cpu", packaging=MVM_PACKAGING).eval()
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    perceptrons = list(flow.get_perceptrons())
    for position, channels in (prune or {}).items():
        _zero_channels(perceptrons[position], channels)
    batches = [torch.randn(calibration_size, *input_shape, generator=torch.Generator().manual_seed(seed + 1))]
    bits = int(platform["activation_bits"])
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr(), lane_absorbed(flow, platform)))
    grids = install_boundary_quantizers(entries, bits)
    calibrate_boundary_scales(flow, entries, grids, batches)
    quantizers = install_readout_quantizers(perceptrons, bits)
    lanes = install_planned_lanes(flow, platform, bits) or []
    every = quantizers + [q for _, q in lanes]
    for quantizer, value in zip(every, measure_lane_ranges(flow, every, batches, 0.999)):
        quantizer.calibrate(max(value, 1e-3))
    if state_hook is not None:
        state_hook(flow, entries, grids, perceptrons, quantizers, lanes)
    declare_entry_state(flow, entries, grids, platform)
    declare_readout_state(flow, perceptrons, quantizers, platform, lanes)
    install_layer_weight_bits(flow, {"weight_bits": weight_bits, **platform})  # WI-5g: a no-op when nothing binds
    q_max = quantization_bounds(weight_bits)[1]
    grids = ParameterGrids.from_config(platform)
    if grids.declared:  # the Weight Quantization projection at rate 1 under the declared scale and bias rules
        napq = NormalizationAwarePerceptronQuantization(weight_bits, "cpu", rate=1.0,
                                                        fractional_scale=grids.fractional_scale,
                                                        float_bias=grids.law_bias)
        for perceptron in perceptrons:
            napq.transform(perceptron)
    else:
        for perceptron in perceptrons:
            peak = float(perceptron.layer.weight.detach().abs().max())
            top = quantization_bounds(layer_weight_bits(perceptron, weight_bits))[1]
            perceptron.set_parameter_scale(max(1.0, math.floor(top / peak)) if peak > 0 else 1.0)
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=q_max, firing_mode="Default", max_axons=max_axons, max_neurons=max_neurons,
                   hardware_bias=hardware_bias, allow_coalescing=allow_coalescing).map(repr_)
    install_boundary_grids(ir, flow, quantizer_state=platform.get("quantizer_state"))
    typed = {"weight_bits": weight_bits, **platform}
    bind_core_types(ir, flow, typed)
    quantize_ir_graph(ir, weight_bits, weight_quantization=True, grids=grids)
    if prune:
        ir = prune_ir_graph(ir, spiking_mode=INERT_SPIKING_MODE)
    install_readout_laws(ir, flow, platform, weight_quantization=True, weight_bits=weight_bits)
    if bind_core_types(ir, flow, typed):  # the Soft Core Mapping order: lane cores bind, codes checked
        check_core_codes(ir)
    return IntegerProgram(flow, ir, tuple(input_shape))


def host_ops(program: IntegerProgram) -> list:
    return [n.op_type for n in program.ir.nodes if isinstance(n, ComputeOp)]


def cores(program: IntegerProgram) -> list:
    return [n for n in program.ir.nodes if isinstance(n, NeuralCore)]


def real_inputs(program: IntegerProgram, batch: int = 32, seed: int = 2) -> torch.Tensor:
    return 1.5 * torch.randn(batch, *program.input_shape, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(seed))
