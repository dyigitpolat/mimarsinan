"""Integer value programs under the Wave 2 grid declarations (WI-5a to WI-5d), built in the pipeline's step order."""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn as nn

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.mapping.support.grid_declarations import ParameterGrids
from mimarsinan.mapping.support.quantizer_state import QUANTIZER_STATE_FORMAT, perceptron_fx_names
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
    measure_seam_quantiles,
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
from mimarsinan.transformations.quantization_bounds import quantization_bounds
from unit.chip_simulation.value_run.integer_programs import IntegerProgram

#: The network input's declared encoder: the harness's offset-floor lane (codes 0..L on [-3, 3]).
INPUT_SCALE, INPUT_OFFSET = 6.0, 3.0


def write_state(path, perceptrons) -> dict:
    path.write_text(json.dumps({"format": QUANTIZER_STATE_FORMAT, "perceptrons": perceptrons}, sort_keys=True))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def declared_platform(declaration, *, activation_bits=5, mult_bits=24, adc_bits=40, output_law="identity",
                      fractional=True, law_bias=True, readout_law=True) -> dict:
    platform: dict = {"activation_bits": activation_bits}
    if readout_law:
        platform["readout_law"] = {"adc_bits": adc_bits, "range": "calibrated", "mult_bits": mult_bits}
    if readout_law and output_law is not None:
        platform["readout_law"]["output_law"] = output_law
    if declaration is not None:
        platform["quantizer_state"] = declaration
    if fractional:
        platform["weight_scale_rule"] = "fractional"
    if law_bias:
        platform["bias_resolution"] = "law"
    return platform


def build_declared_program(model: nn.Module, input_shape: tuple, *, num_classes: int, state_path,
                           activation_bits: int = 5, weight_bits: int = 4, max_axons=None, max_neurons=None,
                           offset_input: bool = True, declare_ranges: bool = True, readout_law: bool = True,
                           **platform_kwargs):
    """``(program, platform, declaration)``: Boundary Quantization, Readout Law, Weight Quantization (rate 1),
    Normalization Fusion and Soft Core Mapping on ``model``, with the declared map written to ``state_path``;
    ``readout_law=False`` builds the float value path (no Readout Law step, no laws)."""
    torch.manual_seed(0)
    flow = convert_torch_model(model.eval(), input_shape, num_classes, device="cpu", packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    batches = [torch.randn(32, *input_shape, generator=torch.Generator().manual_seed(1))]
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    grids = install_boundary_quantizers(entries, activation_bits)
    calibrate_boundary_scales(flow, entries, grids, batches)
    quantizers = install_readout_quantizers(perceptrons, activation_bits) if readout_law else []
    measured = measure_seam_quantiles(flow, quantizers, batches) if readout_law else []
    for quantizer, value in zip(quantizers, measured):
        quantizer.calibrate(max(value, 1e-3))
    fx_of = {id(p): name for name, p in perceptron_fx_names(flow.get_mapper_repr()).items()}
    document: dict = {}
    for perceptron, grid in zip(entries, grids):
        entry = {"scale": float(grid.scale) * 1.25}
        if offset_input and perceptron is perceptrons[0]:
            entry = {"scale": INPUT_SCALE, "offset": INPUT_OFFSET, "rounding": "floor",
                     "codes": [0, grid_levels(activation_bits)]}
        document.setdefault(fx_of[id(perceptron)], {})["entry"] = entry
    if declare_ranges:
        for perceptron, value in zip(perceptrons[:-1], measured):
            document.setdefault(fx_of[id(perceptron)], {})["readout_range"] = max(float(value), 1e-3) * 0.9
    declaration = write_state(state_path, document)
    platform = declared_platform(declaration, activation_bits=activation_bits, readout_law=readout_law,
                                 **platform_kwargs)
    declare_entry_state(flow, entries, grids, platform)
    if readout_law:
        declare_readout_state(flow, perceptrons, quantizers, platform)
    parameter_grids = ParameterGrids.from_config(platform)
    napq = NormalizationAwarePerceptronQuantization(weight_bits, "cpu", rate=1.0,
                                                    fractional_scale=parameter_grids.fractional_scale,
                                                    float_bias=parameter_grids.law_bias)
    for perceptron in perceptrons:
        napq.transform(perceptron)
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=quantization_bounds(weight_bits)[1], firing_mode="Default", max_axons=max_axons,
                   max_neurons=max_neurons, hardware_bias=True).map(repr_)
    install_boundary_grids(ir, flow, quantizer_state=platform.get("quantizer_state"))
    quantize_ir_graph(ir, weight_bits, weight_quantization=True, grids=parameter_grids)
    if readout_law:
        install_readout_laws(ir, flow, platform, weight_quantization=True, weight_bits=weight_bits)
    return IntegerProgram(flow, ir, tuple(input_shape)), platform, declaration


def real_inputs(program: IntegerProgram, batch: int = 32, seed: int = 2) -> torch.Tensor:
    return 1.2 * torch.randn(batch, *program.input_shape, dtype=torch.float64,
                             generator=torch.Generator().manual_seed(seed))
