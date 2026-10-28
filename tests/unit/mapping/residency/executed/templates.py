"""Tier-0 template vehicles at random weights: the template's resolved config, model and conversion, mapped at the
template's platform geometry with the IR latency pass, without data or training (the Wave 0 byte-identity resolution
route)."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
from mimarsinan.pipelining.session import _resolve_platform_constraints
from mimarsinan.torch_mapping.converter import convert_torch_model

TEMPLATES = Path(__file__).resolve().parents[5] / "templates" / "tier_0"
PROVIDER_FACTS = {"MNIST_DataProvider": ((1, 28, 28), 10), "CIFAR10_DataProvider": ((3, 32, 32), 10)}


@dataclass
class TemplateVehicle:
    name: str
    config: Dict[str, Any]
    ir: Any
    specs: List[Any]
    census: Any
    model: Any = None
    #: The walk's program graph (``ProgramGraph.of_walk``), which the shape-only plane's refusal rules read.
    program: Any = None

    @property
    def input_shape(self):
        return tuple(self.config["input_shape"])

    def cores(self, scale: float) -> List[dict]:
        """The template's core types, each count scaled (at least 1)."""
        return [{**ct, "count": max(1, int(ct["count"] * scale))} for ct in self.config["cores"]]

    def pipeline(self):
        return SimpleNamespace(config=self.config)


def resolve(name: str) -> Dict[str, Any]:
    path = next(TEMPLATES.glob(f"{name}*.json"))
    cfg = json.loads(path.read_text())
    dp = dict(cfg["deployment_parameters"])
    dp.setdefault("seed", int(cfg.get("seed", 0)))
    DeploymentPipeline.apply_preset(cfg.get("pipeline_mode", "phased"), dp)
    pc = _resolve_platform_constraints(cfg, dp)
    config = merge_pipeline_config(dp, pc if isinstance(pc, dict) else {})
    shape, classes = PROVIDER_FACTS[cfg["data_provider_name"]]
    config.update(input_shape=shape, num_classes=classes, input_size=int(np.prod(shape)), device="cpu")
    return config


def template_vehicle(name: str, seed: int = 0) -> TemplateVehicle:
    with contextlib.redirect_stdout(io.StringIO()):
        config = resolve(name)
        plan = DeploymentPlan.resolve(config)
        packaging = packaging_contract_for(plan)
        placement = str(config.get("encoding_layer_placement", "subsume"))
        shape, classes = tuple(config["input_shape"]), int(config["num_classes"])
        builder = BUILDERS_REGISTRY[config["model_type"]]("cpu", shape, classes, config)
        torch.manual_seed(seed)
        model = build_model(builder, config.get("model_config", {}), encoding_placement=placement,
                            packaging=packaging).eval()
        with torch.no_grad():
            model(torch.zeros(1, *shape))                       # materializes lazy layers, as the pipeline does
        rep = _mapper_repr(model, config)
        bits = int(config.get("weight_bits", 8))
        # [E-3] at the cores' mapping parameters, as Soft Core Mapping maps (the bias row of cores without on-core
        # bias, the effective axons), never the flat config's bias flag
        pmap = resolve_platform_mapping_params(config["cores"],
                                               allow_coalescing=bool(config.get("allow_coalescing", False)))
        mapping = IRMapping(q_max=float(2 ** (bits - 1) - 1), firing_mode=str(config.get("firing_mode", "Default")),
                            max_axons=pmap.effective_max_axons, max_neurons=pmap.effective_max_neurons,
                            allow_coalescing=pmap.allow_coalescing, hardware_bias=pmap.hardware_bias)
        mapping.collect_wire_census = True
        ir = mapping.map(rep)
        IRLatency(ir).calculate()                   # as SoftCoreMappingStep does: the executors read core latencies
    return TemplateVehicle(name=name, config=config, ir=ir, specs=list(ir.layout_softcores),
                           census=census_of_walk(mapping), model=model, program=ProgramGraph.of_walk(mapping))


def _mapper_repr(model, config: Dict[str, Any]):
    """The model's mapper representation (converted when the model is a torch module), perceptron indices assigned."""
    placement = str(config.get("encoding_layer_placement", "subsume"))
    flow = model if hasattr(model, "get_mapper_repr") else convert_torch_model(
        model, tuple(config["input_shape"]), int(config["num_classes"]),
        packaging=packaging_contract_for(DeploymentPlan.resolve(config)), encoding_layer_placement=placement)
    rep = flow.get_mapper_repr()
    rep.assign_perceptron_indices()
    return rep


def layout_walk(vehicle: TemplateVehicle):
    """The template walked as the search plane walks it (``layout_hook._collect_softcores``): the shape-only
    ``LayoutIRMapping`` at the platform's effective geometry, wire census on; its specs, census, program graph and the
    walk itself."""
    config = vehicle.config
    pmap = resolve_platform_mapping_params(config["cores"],
                                           allow_coalescing=bool(config.get("allow_coalescing", False)))
    walk = LayoutIRMapping(max_axons=pmap.effective_max_axons, max_neurons=pmap.effective_max_neurons,
                           allow_coalescing=pmap.allow_coalescing, hardware_bias=pmap.hardware_bias,
                           collect_wire_census=True)
    with contextlib.redirect_stdout(io.StringIO()):
        specs = walk.collect_layout_softcores(_mapper_repr(vehicle.model, config))
    return specs, census_of_walk(walk), ProgramGraph.of_walk(walk), walk


def activate(ir, gain: float, bits: int = 0) -> None:
    """Make the seeded random vehicle fire: every core's weights (owned or bank) scaled so its mean absolute column
    sum is ``gain`` thresholds; ``bits`` > 0 then puts it on the chip grid as the pipeline does (``quantize_ir_graph``).

    [fix_c] A random vehicle carries no trained grid: the converted model's placeholder scale (1) read as a trained
    NAPQ scale rounded the scaled weights (|w| below 0.5) to 0, so every layer of the Loihi and ODIN vehicles (integer
    chips) was silent or input-independent (W09, fix_b). On the grid its scales are cleared, so quantization takes the
    mapping-time rule (the largest integer scale keeping max |w| on q_max, which the threshold then carries) and the
    weights keep their ratio to the threshold."""
    def scaled(matrix):
        m = np.asarray(matrix, dtype=np.float64)
        return m * (float(gain) / max(float(np.abs(m).sum(axis=0).mean()), 1e-12))

    for bank in ir.weight_banks.values():
        bank.core_matrix = scaled(bank.core_matrix)
    for core in ir.get_neural_cores():
        if core.core_matrix is not None:
            core.core_matrix = scaled(core.core_matrix)
    if bits > 0:
        for owner in list(ir.weight_banks.values()) + list(ir.get_neural_cores()):
            owner.parameter_scale = torch.tensor(0.0)
        quantize_ir_graph(ir, int(bits), weight_quantization=True)


def residency_platform(vehicle: TemplateVehicle, scale: float, residency_plan: Optional[dict]) -> Dict[str, Any]:
    """A resolved-platform dict of the template at scaled core counts, with the declared key (S0 seam stand-in)."""
    platform = {
        "cores": vehicle.cores(scale), "allow_scheduling": True,
        "allow_coalescing": bool(vehicle.config.get("allow_coalescing", False)),
        "allow_neuron_splitting": bool(vehicle.config.get("allow_neuron_splitting", False)),
        "max_schedule_passes": int(vehicle.config.get("max_schedule_passes", 8) or 8),
    }
    if residency_plan is not None:
        platform["residency_plan"] = residency_plan
    return platform
