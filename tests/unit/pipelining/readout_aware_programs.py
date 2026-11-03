"""[U-ADC] The readout-aware training fixture: a small value-domain (mvm) network through the pipeline's own steps.

Three networks: ``mlp``, a float MLP (``FEATURES`` -> ``HIDDEN`` with batch norm, ``LAYERS - 1`` more
``HIDDEN``-wide layers, -> ``CLASSES``; ReLUs between, ``LAYERS + 1`` cores); ``core``, one fixture core (``FEATURES``
-> ``CLASSES`` with batch norm); ``resnet``, a residual convolution net on ``IMAGE`` inputs (a stem, one basic block
with an identity shortcut, an average pool, a linear head). Each is trained on a tiny teacher-labelled set (``SAMPLES``
samples, the labels a fixed linear teacher's argmax; its train, validation and test splits are one set: the fixture's
validation inputs), converted for the value domain, and taken through Boundary Quantization, Readout Law, Weight
Quantization, Normalization Fusion and Soft Core Mapping on a mock pipeline, at 4-bit weights and activations and the
declared ``readout_law.adc_bits``; ``lanes`` declares ``onchip_arithmetic`` classes (the ReLUs, adds and pools then run
on chip as lanes), else they are host ops. ``deployed`` reads the deployed program (the identity value program of Soft
Core Mapping's IR, fp64) on the validation inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from conftest import MockDataProviderFactory, MockPipeline

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import BoundaryQuantizationStep
from mimarsinan.pipelining.pipeline_steps.quantization.normalization_fusion_step import NormalizationFusionStep
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import ReadoutLawStep
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.tuning.orchestration.adaptation_manager_factory import create_adaptation_manager_for_model

FEATURES, HIDDEN, LAYERS, CLASSES, SAMPLES = 64, 64, 4, 4, 256
IMAGE, CHANNELS = (3, 8, 8), 8
#: Float pretraining of the fixture network on the tiny set (full batch, Adam).
PRETRAIN_STEPS, PRETRAIN_LR = 400, 0.02
#: The fixture's learning rate (Weight Quantization's tuner reads it).
LR = 0.003


def input_shape(net: str) -> tuple:
    return tuple(IMAGE) if net == "resnet" else (FEATURES,)


def document(adc_bits: int, *, lanes: Optional[Sequence[str]] = None, **deployment) -> Dict[str, Any]:
    """The fixture's deployment document: 4-bit weights and activations, a calibrated readout law at ``adc_bits``."""
    platform = {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 128, "has_bias": True}],
        "max_axons": 256, "max_neurons": 256, "has_bias": True, "weight_bits": 4, "activation_bits": 4,
        "readout_law": {"adc_bits": int(adc_bits), "range": "calibrated", "mult_bits": 16},
        **({"onchip_arithmetic": {"classes": list(lanes)}} if lanes else {}),
    }
    params = {
        "core_semantics": "mvm", "model_type": "lenet5", "model_config": {"variant": "lenet5"},
        "model_config_mode": "user", "hw_config_mode": "fixed", "weight_quantization": True, "lr": LR,
        "tuning_budget_scale": 1.0, "num_workers": 0, "wq_endpoint_recovery_steps": 0, **deployment,
    }
    return {"deployment_parameters": params, "platform_constraints": platform}


def pipeline_config(adc_bits: int, *, net: str = "mlp", lanes: Optional[Sequence[str]] = None,
                    **deployment) -> Dict[str, Any]:
    doc = document(adc_bits, lanes=lanes, **deployment)
    config = merge_pipeline_config(doc["deployment_parameters"], doc["platform_constraints"])
    shape = input_shape(net)
    size = 1
    for d in shape:
        size *= int(d)
    config.update(device="cpu", input_shape=shape, input_size=size, num_classes=CLASSES, num_workers=0,
                  onchip_majority_gate=False, scm_torch_sim_parity_check=False, value_parity_samples=SAMPLES)
    return config


class _Block(nn.Module):
    """A basic residual block with an identity shortcut: the add after the second batch norm, then a ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels)
        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()
        self.conv2, self.bn2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))) + x)


class _ResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem, self.bn, self.relu = nn.Conv2d(IMAGE[0], CHANNELS, 3, 1, 1, bias=False), nn.BatchNorm2d(CHANNELS), \
            nn.ReLU()
        self.block = _Block(CHANNELS)
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(CHANNELS * (IMAGE[1] // 2) * (IMAGE[2] // 2), CLASSES)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.block(self.relu(self.bn(self.stem(x))))), 1))


def float_network(seed: int = 0, net: str = "mlp") -> nn.Module:
    torch.manual_seed(seed)
    if net == "core":
        return nn.Sequential(nn.Linear(FEATURES, CLASSES), nn.BatchNorm1d(CLASSES))
    if net == "resnet":
        return _ResNet()
    layers = [nn.Linear(FEATURES, HIDDEN), nn.BatchNorm1d(HIDDEN), nn.ReLU()]
    for _ in range(LAYERS - 1):
        layers += [nn.Linear(HIDDEN, HIDDEN), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(HIDDEN, CLASSES))


def teacher_labels(x: torch.Tensor, seed: int) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    generator = torch.Generator().manual_seed(1000 + seed)
    weight = torch.randn(flat.shape[1], CLASSES, generator=generator)
    return (flat @ weight).argmax(dim=1)


def _pretrain(net: nn.Module, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
    optimizer = torch.optim.Adam(net.parameters(), lr=PRETRAIN_LR)
    net.train()
    for _ in range(PRETRAIN_STEPS):
        optimizer.zero_grad()
        F.cross_entropy(net(x), y).backward()
        optimizer.step()
    return net.eval()


@dataclass
class Fixture:
    pipeline: MockPipeline
    flow: nn.Module
    x: torch.Tensor
    y: torch.Tensor


def _run(pipeline: MockPipeline, step) -> None:
    pipeline.prepare_step(step)
    step.run()


def prepared(tmp_path, adc_bits: int = 4, seed: int = 0, net: str = "mlp", lanes: Optional[Sequence[str]] = None,
             **deployment) -> Fixture:
    """The fixture up to Weight Quantization's entry: trained, converted, Boundary Quantization and Readout Law run."""
    torch.manual_seed(seed)
    shape = input_shape(net)
    factory = MockDataProviderFactory(input_shape=shape, num_classes=CLASSES, size=SAMPLES)
    dataset = factory.create()._ds
    dataset.targets = teacher_labels(dataset.data, seed)
    x, y = dataset.data.clone(), dataset.targets.clone()
    trained = _pretrain(float_network(seed, net), x, y)
    flow = convert_torch_model(trained, shape, CLASSES, device="cpu", packaging=MVM_PACKAGING).eval()
    config = pipeline_config(adc_bits, net=net, lanes=lanes, **deployment)
    pipeline = MockPipeline(config=config, working_directory=str(tmp_path), data_provider_factory=factory)
    pipeline.seed("model", flow, step_name="TorchMappingStep")
    pipeline.seed("adaptation_manager", create_adaptation_manager_for_model(config, flow), step_name="TorchMappingStep")
    pipeline.seed("platform_constraints_resolved", json.loads(json.dumps(build_platform_constraints_resolved(config))),
                  step_name="ModelConfigurationStep")
    _run(pipeline, BoundaryQuantizationStep(pipeline))
    _run(pipeline, ReadoutLawStep(pipeline))
    return Fixture(pipeline, pipeline.cache["ReadoutLawStep.model"], x, y)


def model_of(fixture: Fixture) -> nn.Module:
    """The latest model entry of the fixture's pipeline."""
    for key in ("NormalizationFusionStep.model", "WeightQuantizationStep.model", "ReadoutLawStep.model"):
        if key in fixture.pipeline.cache:
            return fixture.pipeline.cache[key]
    raise KeyError("no model entry")


def weight_quantization(fixture: Fixture) -> Fixture:
    _run(fixture.pipeline, WeightQuantizationStep(fixture.pipeline))
    return fixture


def weight_quantization_process(fixture: Fixture) -> None:
    """Weight Quantization's ``process`` alone (a test that replaces its tuner commits no entry)."""
    step = WeightQuantizationStep(fixture.pipeline)
    fixture.pipeline.prepare_step(step)
    step.process()


def soft_core_mapping(fixture: Fixture):
    """Normalization Fusion then Soft Core Mapping: the deployed IR (every core's readout law installed)."""
    _run(fixture.pipeline, NormalizationFusionStep(fixture.pipeline))
    _run(fixture.pipeline, SoftCoreMappingStep(fixture.pipeline))
    return fixture.pipeline.cache["SoftCoreMappingStep.ir_graph"]


def deployed(ir_graph, x: torch.Tensor) -> torch.Tensor:
    """The deployed program's outputs on ``x``: the identity value program of the IR, fp64."""
    flow = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=ir_graph), dtype=torch.float64)
    with torch.no_grad():
        return flow(x.to(torch.float64))


def count(outputs: torch.Tensor, y: torch.Tensor) -> int:
    return int((outputs.argmax(dim=-1) == y).sum())


def deployed_loss(outputs: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(outputs.to(torch.float64), y))


def law_digests(ir_graph) -> Dict[str, Any]:
    """{core name: per-column law digests} of every law-carrying core of an IR."""
    return {str(n.name): n.readout_law.column_digests() for n in ir_graph.get_neural_cores()
            if getattr(n, "readout_law", None) is not None}
