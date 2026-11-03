"""Attention programs built the pipeline's way with an operand core type declared (N2): convert, lower the
contractions, fuse, entry grids, readout ranges (per consumer group), IR, quantize, laws."""

from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.ir.dynamic_bank import validate_bindings
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.operand import operand_entry_perceptrons, operand_perceptrons
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.mapping.support.operand_lowering import lower_contractions
from mimarsinan.models.families.vit_lite import ViTLiteCCT
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
    calibrate_readout_quantizers,
    install_readout_quantizers,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.transformations.quantization_bounds import quantization_bounds

LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 16}


def operand_platform(*, payload_bits: int = 8, activation_bits: int = 8, grouped: bool = True,
                     cores: Optional[Sequence[dict]] = None, write_source: str = "host") -> dict:
    """A value-domain platform with a static core type and an operand core type."""
    declared = list(cores) if cores is not None else [
        {"max_axons": 256, "max_neurons": 256, "count": 64},
        {"max_axons": 256, "max_neurons": 256, "count": 16,
         "operand": {"payload_bits": payload_bits, "write_source": write_source}},
    ]
    platform = {"cores": declared, "activation_bits": activation_bits, "readout_law": dict(LAW)}
    if grouped:
        platform["readout_ranges"] = "per_consumer_group"
    return platform


@dataclass
class OperandProgram:
    model: nn.Module
    ir: object
    input_shape: tuple
    platform: dict
    report: object = None
    batches: list = field(default_factory=list)

    def identity_mapping(self):
        return build_identity_hybrid_mapping(ir_graph=self.ir)

    def identity(self) -> ValueHybridCoreFlow:
        return ValueHybridCoreFlow(self.identity_mapping(), dtype=torch.float64, device="cpu")

    def packed_mapping(self, *, axons: int = 256, neurons: int = 256, count: int = 999,
                       passes: Optional[int] = None, split: bool = True):
        strategy = MappingStrategy.resolve(ChipCapabilities(
            allow_neuron_splitting=split, allow_coalescing=False,
            allow_scheduling=passes is not None, max_schedule_passes=passes or 8,
        ))
        return build_hybrid_hard_core_mapping(
            ir_graph=self.ir, cores_config=[{"max_axons": axons, "max_neurons": neurons, "count": count}],
            strategy=strategy, max_schedule_passes=passes or 8,
        )

    def packed(self, **kwargs) -> ValueHybridCoreFlow:
        return ValueHybridCoreFlow(self.packed_mapping(**kwargs), dtype=torch.float64, device="cpu")

    def nf(self) -> nn.Module:
        return self.model.double()


def build_operand_program(
    model: nn.Module, input_shape: tuple, *, num_classes: int, platform: Optional[dict] = None,
    weight_bits: int = 8, seed: int = 0, calibration_size: int = 32, readout_law: Optional[dict] = None,
    price: Optional[Callable] = None,
) -> OperandProgram:
    """The integer program of ``model`` with its products lowered onto the platform's operand core type
    (``price``: the stationary choice's price, the declared physics' by default)."""
    platform = dict(platform or operand_platform())
    if readout_law is not None:
        platform["readout_law"] = dict(readout_law)
    bits = int(platform["activation_bits"])
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(model.eval(), input_shape, num_classes, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
    repr_ = flow.get_mapper_repr()
    report = lower_contractions(repr_, platform, price=price)
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    perceptrons = list(flow.get_perceptrons())
    batches = [torch.randn(calibration_size, *input_shape)]
    entries = list(segment_entry_perceptrons(repr_)) + list(operand_entry_perceptrons(repr_))
    grids = install_boundary_quantizers(entries, bits)
    calibrate_boundary_scales(flow, entries, grids, batches)
    readout = perceptrons + list(operand_perceptrons(repr_))
    quantizers = install_readout_quantizers(readout, bits)
    calibrate_readout_quantizers(flow, readout, quantizers, batches, platform=platform)
    q_max = quantization_bounds(weight_bits)[1]
    for perceptron in perceptrons:
        peak = float(perceptron.layer.weight.detach().abs().max())
        perceptron.set_parameter_scale(max(1.0, math.floor(q_max / peak)) if peak > 0 else 1.0)
    repr_.assign_perceptron_indices()
    with contextlib.redirect_stdout(io.StringIO()):
        ir = IRMapping(q_max=q_max, firing_mode="Default").map(repr_)
    install_boundary_grids(ir, flow)
    quantize_ir_graph(ir, weight_bits, weight_quantization=True)
    install_readout_laws(ir, flow, platform, weight_quantization=True, weight_bits=weight_bits)
    validate_bindings(ir, packaging=MVM_PACKAGING)
    return OperandProgram(flow, ir, tuple(input_shape), platform, report, batches)


def attention_block(*, attention: str = "softmax", depth: int = 2, heads: int = 2, dim: int = 8,
                    img: int = 8, patch: int = 4, classes: int = 5, seed: int = 3) -> nn.Module:
    """A two-head ViT-Lite (CCT block) small enough for the unit suite: L = (img / patch)^2 + 1 tokens."""
    torch.manual_seed(seed)
    model = ViTLiteCCT(num_classes=classes, img_size=img, patch=patch, dim=dim, depth=depth, heads=heads,
                       mlp_ratio=2.0, attention=attention).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():  # every all-zero parameter drawn, as G1's certificates do
            if not bool(parameter.abs().sum()):
                parameter.copy_(0.1 * torch.randn(parameter.shape, generator=generator))
    return model


def attention_program(attention: str = "softmax", *, block: Optional[dict] = None, **kwargs) -> OperandProgram:
    return build_operand_program(attention_block(attention=attention, **(block or {})), (3, 8, 8), num_classes=5,
                                 **kwargs)


def image_inputs(batch: int = 16, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch, 3, 8, 8), generator=generator, dtype=torch.float64)
