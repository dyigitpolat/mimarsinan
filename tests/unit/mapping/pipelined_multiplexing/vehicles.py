"""Vehicles of pipelined position multiplexing: lane-fused residual integer programs (lanes L and E on chip, the
pool on the host, as the headline members declare them) and the position-multiplexing LIF conv chains; one helper
builds the per-pass (W02) and the pipelined program of either on the scheduled builder."""

from __future__ import annotations

from typing import List

import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.wavefront_stages import SCHEDULE_CORE_ORDINALS, wavefront_head

from unit.chip_simulation.value_run.integer_programs import residual_block_net, resnet18_quarter
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform

PASSES = 1 << 20
#: The position-multiplexing LIF vehicles' cores (``unit.mapping.position_multiplexing.test_programs``).
LIF_CORES = [{"max_axons": 64, "max_neurons": 16, "count": 64, "has_bias": True}]
#: The rows' readout law (32-bit ADC, 24-bit multiply, identity output, calibrated).
LAW = {"adc_bits": 32, "mult_bits": 24, "output_law": "identity", "range": "calibrated"}


def lane_program(model="block", shape=(3, 8, 8)):
    """A residual network with ReLU and the residual add fused on chip (lanes L and E), 4-bit, on 256 x 64 cores."""
    net = residual_block_net() if model == "block" else resnet18_quarter()
    return build_lane_program(net, shape, num_classes=10, platform=lane_platform(["L", "E"], activation_bits=4,
                                                                                 readout_law=LAW),
                              weight_bits=4, max_axons=256, max_neurons=64, hardware_bias=True)


def cores(count: int, axons: int = 256, neurons: int = 64) -> List[dict]:
    return [{"max_axons": axons, "max_neurons": neurons, "count": count, "has_bias": True}]


def capabilities(pipelined, **kwargs) -> ChipCapabilities:
    base = dict(allow_scheduling=True, max_schedule_passes=PASSES, position_multiplexing=True,
                allow_coalescing=True, allow_neuron_splitting=True)
    base.update(kwargs)
    return ChipCapabilities(pipelined_multiplexing=pipelined, **base)


def build(ir, cores_config, pipelined, *, retimed_level_stages=False, per_hop=False, **kwargs):
    caps = capabilities(pipelined, **kwargs)
    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=cores_config, strategy=MappingStrategy.resolve(caps),
                                          max_schedule_passes=caps.max_schedule_passes,
                                          retimed_level_stages=retimed_level_stages, per_hop_neural_segments=per_hop)


def neural(hybrid) -> list:
    return [s for s in hybrid.stages if s.kind == "neural"]


def value(hybrid, x) -> torch.Tensor:
    with torch.no_grad():
        return ValueHybridCoreFlow(hybrid, dtype=torch.float64, device="cpu")(x.double())


def heads(hybrid) -> list:
    return [s for s in neural(hybrid) if wavefront_head(s) is not None]


def levels_per_pass(hybrid, ir) -> List[set]:
    tag = {int(n.id): int(ir.layout_softcores[int(n.layout_softcore_index)].latency_tag or 0)
           for n in ir.get_neural_cores()}
    return [{tag[int(p["ir_node_id"])] for ps in s.hard_core_mapping.soft_core_placements_per_hard_core for p in ps}
            for s in neural(hybrid)]


def resident_pairs(hybrid):
    """Per wholly resident core: (the core, its head's core at its ordinal), walking each residency chain."""
    head = None
    for stage in neural(hybrid):
        cores_ = stage.hard_core_mapping.cores
        if not stage.schedule_weights_resident:
            head = cores_
            continue
        ordinals = getattr(stage, SCHEDULE_CORE_ORDINALS, None) or range(len(cores_))
        for core, o in zip(cores_, ordinals):
            yield stage, core, head[o]
