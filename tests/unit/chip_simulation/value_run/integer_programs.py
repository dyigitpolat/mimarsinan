"""Integer value programs built the pipeline's way (convert, fuse, grids, readout, IR, quantize, prune, laws)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.models.lenet5 import LeNet5
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
    measure_seam_quantiles,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.transformations.quantization_bounds import quantization_bounds

LAW = {"adc_bits": 40, "range": "calibrated", "mult_bits": 16}


@dataclass
class IntegerProgram:
    model: nn.Module
    ir: object
    input_shape: tuple

    def identity(self) -> ValueHybridCoreFlow:
        return ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=self.ir), dtype=torch.float64)

    def packed_mapping(self, *, axons: int, neurons: int, count: int = 999, split: bool = True,
                       coalescing: bool = False, passes: Optional[int] = None,
                       widths: Optional[Sequence[int]] = None):
        """``widths``: one core type of ``count`` cores per weight width (WI-5g core types), else one untyped type."""
        strategy = MappingStrategy.resolve(ChipCapabilities(
            allow_neuron_splitting=split, allow_coalescing=coalescing,
            allow_scheduling=passes is not None, max_schedule_passes=passes or 8,
        ))
        core = {"max_axons": axons, "max_neurons": neurons, "count": count}
        cores = [core] if widths is None else [dict(core, weight_bits=int(w)) for w in widths]
        return build_hybrid_hard_core_mapping(
            ir_graph=self.ir, cores_config=cores, strategy=strategy, max_schedule_passes=passes or 8,
        )

    def packed(self, **kwargs) -> ValueHybridCoreFlow:
        return ValueHybridCoreFlow(self.packed_mapping(**kwargs), dtype=torch.float64)

    def nf(self) -> nn.Module:
        return self.model.double()


def _zero_channels(perceptron, channels: Sequence[int]) -> None:
    with torch.no_grad():
        perceptron.layer.weight.data[list(channels)] = 0.0
        if perceptron.layer.bias is not None:
            perceptron.layer.bias.data[list(channels)] = 0.0


def build_integer_program(
    model: nn.Module, input_shape: tuple, *, num_classes: int, readout_law=None,
    activation_bits: int = 8, weight_bits: int = 8, max_axons=None, max_neurons=None,
    prune: Optional[dict] = None, seed: int = 0, calibration_size: int = 32, allow_coalescing: bool = False,
    boundary_scales: Optional[Sequence[float]] = None, readout_ranges: Optional[Sequence[float]] = None,
    parameter_scales: Optional[Sequence[float]] = None, hardware_bias: bool = False,
) -> IntegerProgram:
    """``prune`` maps perceptron positions to output channels zeroed before mapping (IR pruning then compacts);
    the three scale lists, when given, replace the calibrated boundary scales, readout ranges and weight grids.
    ``hardware_bias`` [fix_b]: the bias on the core (the target's ``has_bias``, as the pipeline's IR mapping reads it),
    else on one axon; a caller building for a declared target passes the target's."""
    torch.manual_seed(seed)
    flow = convert_torch_model(model.eval(), input_shape, num_classes, device="cpu", packaging=MVM_PACKAGING).eval()
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    perceptrons = list(flow.get_perceptrons())
    for position, channels in (prune or {}).items():
        _zero_channels(perceptrons[position], channels)
    batches = [torch.randn(calibration_size, *input_shape)]
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    grids = install_boundary_quantizers(entries, activation_bits)
    calibrate_boundary_scales(flow, entries, grids, batches)
    for quantizer, value in zip(grids, boundary_scales or ()):
        quantizer.calibrate(value)
    quantizers = install_readout_quantizers(perceptrons, activation_bits)
    measured = measure_seam_quantiles(flow, quantizers, batches)
    for quantizer, value in zip(quantizers, readout_ranges or [max(v, 1e-3) for v in measured]):
        quantizer.calibrate(value)
    q_max = quantization_bounds(weight_bits)[1]
    for index, perceptron in enumerate(perceptrons):
        peak = float(perceptron.layer.weight.detach().abs().max())
        default = max(1.0, math.floor(q_max / peak)) if peak > 0 else 1.0
        perceptron.set_parameter_scale(parameter_scales[index] if parameter_scales else default)
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(q_max=q_max, firing_mode="Default", max_axons=max_axons, max_neurons=max_neurons,
                   allow_coalescing=allow_coalescing, hardware_bias=hardware_bias).map(repr_)
    install_boundary_grids(ir, flow)
    quantize_ir_graph(ir, weight_bits, weight_quantization=True)
    if prune:
        ir = prune_ir_graph(ir, spiking_mode=INERT_SPIKING_MODE)
    install_readout_laws(
        ir, flow, {"readout_law": dict(readout_law or LAW), "activation_bits": activation_bits},
        weight_quantization=True, weight_bits=weight_bits,
    )
    return IntegerProgram(flow, ir, tuple(input_shape))


def integer_inputs(program: IntegerProgram, batch: int = 64, seed: int = 1, spread: int = 3) -> torch.Tensor:
    """Integer-valued network inputs (the golden's ``x_int``)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-spread, spread + 1, (batch, *program.input_shape), generator=generator)


# Instances -------------------------------------------------------------------------------------------------


def mlp() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 4),
    ).eval()


def lenet5(shape=(1, 28, 28), classes=10) -> nn.Module:
    torch.manual_seed(5)
    return LeNet5(shape, classes).eval()


def conv1d_net() -> nn.Module:
    torch.manual_seed(9)
    return nn.Sequential(
        nn.Conv1d(2, 6, 3, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(6 * 16, 5),
    ).eval()


class _ResidualBlock(nn.Module):
    """A CIFAR basic block with a projection shortcut: the residual add rides as a host op."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x)))))
        return self.relu2(out + self.shortcut(x))


class _ResNet(nn.Module):
    """CIFAR ResNet: 3x3 stem, no max-pool, basic blocks, average pool, linear head."""

    def __init__(self, widths: Sequence[int], blocks: int, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(widths[0])
        self.relu = nn.ReLU()
        layers, cin = [], widths[0]
        for stage, cout in enumerate(widths):
            for b in range(blocks):
                layers.append(_ResidualBlock(cin, cout, 2 if stage > 0 and b == 0 else 1))
                cin = cout
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, classes)

    def forward(self, x):
        x = self.blocks(self.relu(self.bn(self.stem(x))))
        return self.fc(torch.flatten(self.pool(x), 1))


def _randomize_bn(model: nn.Module) -> nn.Module:
    generator = torch.Generator().manual_seed(3)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.running_mean.copy_(torch.randn(module.num_features, generator=generator) * 0.1)
            module.running_var.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
    return model.eval()


def residual_block_net() -> nn.Module:
    torch.manual_seed(11)
    return _randomize_bn(_ResNet((8, 16), blocks=1))


def resnet18_quarter() -> nn.Module:
    """CIFAR ResNet-18 at width 1/4 (16, 32, 64, 128), BasicBlock [2, 2, 2, 2]."""
    torch.manual_seed(18)
    return _randomize_bn(_ResNet((16, 32, 64, 128), blocks=2))


PRUNE = {0: [1, 3], 2: [5, 6, 7], 3: [2, 9]}
#: name -> (program builder, packed geometry): the acceptance instances below the ResNet-18 scale.
INSTANCES = {
    "mlp": (lambda: build_integer_program(mlp(), (8,), num_classes=4), dict(axons=256, neurons=5)),
    "lenet5": (lambda: build_integer_program(lenet5(), (1, 28, 28), num_classes=10),
               dict(axons=1024, neurons=512)),
    "pruned_bank_conv": (lambda: build_integer_program(lenet5(), (1, 28, 28), num_classes=10, prune=PRUNE),
                         dict(axons=1024, neurons=512)),
    "residual_block": (lambda: build_integer_program(residual_block_net(), (3, 16, 16), num_classes=10),
                       dict(axons=256, neurons=32)),
    "conv1d": (lambda: build_integer_program(conv1d_net(), (2, 16), num_classes=5), dict(axons=128, neurons=16)),
    "pruned_multi_pass": (lambda: build_integer_program(lenet5(), (1, 28, 28), num_classes=10, prune=PRUNE),
                          dict(axons=1024, neurons=256, count=3, passes=128)),
}


def hcm_law_digests(hybrid) -> dict:
    """{IR core id: {neuron: digest}} read back from the packed hard cores (re-assembled in IR column order)."""
    out: dict = {}
    for stage in hybrid.stages:
        if stage.kind != "neural":
            continue
        hcm = stage.hard_core_mapping
        per_core = {}
        for (core_id, neuron), (hc_index, hc_neuron) in hcm.neuron_mapping.items():
            if hc_index not in per_core:
                per_core[hc_index] = hcm.cores[hc_index].readout_law.column_digests()
            out.setdefault(int(core_id), {})[int(neuron)] = per_core[hc_index][hc_neuron]
    return out


def law_digests_by_core(ir) -> dict:
    """{core id: per-column law digests} of every law-carrying IR core."""
    return {int(n.id): n.readout_law.column_digests() for n in ir.nodes if hasattr(n, "readout_law")}


def assert_exact(a: torch.Tensor, b: torch.Tensor) -> None:
    a, b = a.reshape(b.shape).double(), b.double()
    delta = float((a - b).abs().max()) if b.numel() else 0.0
    assert delta == 0.0, f"max|delta| = {delta:.3e} (atol 0)"
    assert b.numel() > 0 and np.count_nonzero(b.numpy()) > 0, "a vacuous comparison: every output is 0"
