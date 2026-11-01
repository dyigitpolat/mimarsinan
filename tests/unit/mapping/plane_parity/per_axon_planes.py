"""[U-AX] Both planes of one program on a per-axon pool at one scheduled point, as phase C's plane facts ask them.

The pair is the plane-facts harness's (RA ``explore/cl5/scripts/plane_facts.py``, ``pair``; phase C's SD-11): the
exploration plane is the scaling ladder (``shape_census``, ``schedule_program`` and ``price_scheduled``, ``ladder_row``,
read by ``candidate_facts``; the sign rows of its pass census), the executed plane is the builder's (``IRMapping`` at the
pool's geometry, ``IRLatency``, a copy of the IR on the declared weight grid, then ``build_hybrid_mapping_for_pipeline``;
``program_writes`` priced at K = 1 and read by ``executed_facts``; the sign rows of ``mapping_program_passes``). The
platform is the ODIN bundle through the ladder script's ``resolved_platform``, at a fraction of the packed tiles F.

The executed copy's grid is the quantizer's (``quantize_ir_graph`` at the bundle's weight bits): for the seeded networks
here the rule for weights with no trained scale (one grid per perceptron at floor(q_max / peak)), as the harness
declares for its seeded networks; with ``trained`` the perceptrons' own trained scales.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import importlib.util
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.deployment_record.build.program_census import mapping_program_passes
from mimarsinan.deployment_record.design_point.endurance import priced_at
from mimarsinan.deployment_record.design_point.pricing import candidate_facts, executed_facts
from mimarsinan.deployment_record.design_point.rows import plane_agreement
from mimarsinan.deployment_record.design_point.timeline import program_writes, writes_summary
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.platform.packaging_contract import SPIKING_PACKAGING
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.torch_mapping.converter import convert_torch_model

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "scaling_ladder.py"
TARGET = "odin"
SAMPLED_FRACTION = 0.125


@functools.lru_cache(maxsize=1)
def script() -> Any:
    """The ladder runner (``scripts/scaling_ladder.py``): its platforms, program facts and reading constants."""
    spec = importlib.util.spec_from_file_location("scaling_ladder_script_uax", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def network(model: torch.nn.Module, shape: tuple, classes: int, name: str) -> ladder.LadderNetwork:
    """``model`` converted under the spiking contract, as the ladder runner's ``build_network`` converts a member."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model.eval(), shape, classes, packaging=SPIKING_PACKAGING, strict=False).eval()
        params, macs, ops = estimate_onchip_fractions(flow, shape, classes, metrics=("params", "macs", "ops"))
    return ladder.LadderNetwork(
        name=name, mapper_repr=flow.get_mapper_repr(), resolution=shape,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip},
        conversion=ladder.certificate_record(getattr(flow, "conversion_certificate", None)),
    )


def ir_of(net: ladder.LadderNetwork, chip: Mapping[str, Any]) -> Any:
    """The harness's ``ir_of``: ``IRMapping`` at the pool's tiling limits, then ``IRLatency``."""
    pmap = resolve_platform_mapping_params(list(chip["cores"]), allow_coalescing=bool(chip.get("allow_coalescing")))
    bits = int(chip.get("weight_bits") or 8)
    if hasattr(net.mapper_repr, "assign_perceptron_indices"):
        net.mapper_repr.assign_perceptron_indices()
    with contextlib.redirect_stdout(io.StringIO()):
        ir = IRMapping(q_max=float(2 ** (bits - 1) - 1), firing_mode="Default", max_axons=pmap.effective_max_axons,
                       max_neurons=pmap.effective_max_neurons, allow_coalescing=pmap.allow_coalescing,
                       hardware_bias=pmap.hardware_bias).map(net.mapper_repr)
        IRLatency(ir).calculate()
    net.mapper_repr.clear_ir_caches()
    return ir


def executed_ir(ir: Any, chip: Mapping[str, Any], *, trained: bool = False) -> Any:
    """A copy of ``ir`` on the declared weight grid (``quantize_ir_graph`` at the chip's weight bits): the rule for
    weights with no trained scale (the harness's ``without_trained_scale``: a zero scale, the bias on the weight
    grid), or, with ``trained``, the perceptrons' own trained scales."""
    built = copy.deepcopy(ir)
    if not trained:
        for bank in built.weight_banks.values():
            bank.parameter_scale = torch.tensor(0.0)
        for node in built.nodes:
            if isinstance(node, NeuralCore):
                node.parameter_scale = torch.tensor(0.0)
                if getattr(node, "bias_scale", None) is not None:
                    node.bias_scale = None
    quantize_ir_graph(built, int(chip.get("weight_bits") or 8), weight_quantization=True)
    return built


@dataclass
class Pair:
    census: Any
    point: Dict[str, Any]
    candidate: Dict[str, Any]
    executed: Dict[str, Any]
    planes: Dict[str, Any]
    candidate_sign_rows: List[int]
    executed_sign_rows: List[int]
    candidate_cells: List[int]
    executed_cells: List[int]


def _per_pass(rows: Any, attribute: str) -> List[int]:
    return [int(sum(getattr(r, attribute) or ())) for r in rows]


def pair(net: ladder.LadderNetwork, *, declared: Optional[Mapping[str, Any]] = None,
         fraction: float = SAMPLED_FRACTION, trained: bool = False) -> Pair:
    """The two planes of ``net`` on ODIN at ``fraction`` of F (``declared``: keys the bundle's declaration takes, as
    the harness's departure), compared field by field as the harness compares them (``plane_agreement``)."""
    run = script()
    declared = dict(declared or {})
    chip = run.resolved_platform(TARGET, "declared", None, declared, {})
    census = ladder.shape_census(net, chip)
    ir = ir_of(net, chip)
    tiles = ladder.packed_tiles(census, chip)
    platform = {**run.resolved_platform(TARGET, fraction, tiles, declared, {}), "inference_batch_K": 1,
                "inference_rate_hz": run.RATE_HZ, "min_lifetime_hours": run.LIFETIME_HOURS}
    scheduled = ladder.schedule_program(net, census, platform, run.program_facts, tiles=tiles)
    assert scheduled.refusal is None, scheduled.refusal
    priced = ladder.price_scheduled(net, census, scheduled, platform)
    point = ladder.ladder_row(net, census, priced, platform, run.READING)
    with contextlib.redirect_stdout(io.StringIO()):
        hybrid = build_hybrid_mapping_for_pipeline(executed_ir(ir, chip, trained=trained), platform)
    stages = [s for s in hybrid.stages if getattr(s, "kind", None) == "neural"]
    writes = program_writes(hybrid, platform, weight_bits=int(platform["weight_bits"]))
    summary = writes_summary(writes)
    pricing = priced_at(writes, PlatformPhysics.from_dict(platform["platform_physics_resolved"]), platform, 1)
    band = next((t.band for t in pricing.terms if t.name == "energy_write_mj" and t.band is not None), None)
    summary["energy_write_mj"] = None if band is None else [band.low, band.nominal, band.high]
    executed = executed_facts({"program": {"neural_passes": len(stages)}, "writes": summary}, point["cores"])
    candidate = candidate_facts(point)
    exe = mapping_program_passes(hybrid).rows
    cand = scheduled.facts.passes.rows
    return Pair(census=census, point=point, candidate=candidate, executed=executed,
                planes=plane_agreement(executed, candidate),
                candidate_sign_rows=_per_pass(cand, "core_sign_rows"),
                executed_sign_rows=_per_pass(exe, "core_sign_rows"),
                candidate_cells=_per_pass(cand, "core_cells"), executed_cells=_per_pass(exe, "core_cells"))
