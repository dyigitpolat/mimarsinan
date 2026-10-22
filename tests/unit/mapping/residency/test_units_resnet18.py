"""Acceptance 4: on CIFAR ResNet-18 w1 built here, the toolchain's residency units equal an independent count from
the module shapes and the packed HCM core count, and the live sets equal the widths recomputed from FX shapes; under
the MVM conversion every inter-layer edge crosses a host op, so those live sets are all zero, and the FX witness is
repeated on the LIF deep CNN of t0_03, whose cuts inside a segment carry non-zero widths.

Every conv output position is its own bank-backed softcore (``conv_helpers.map_shared_weight_conv``), so a layer's
unit is ``positions x ceil(fan_in / axons) x ceil(fan_out / neurons)`` tiles; the geometry (4609 axons, 64 neurons)
makes packing one-to-one (every softcore fills a core's neurons), where the HCM core count is the same number.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.schedule.residency_units import cut_nodes_from_census, residency_units

from . import resnet18 as R

AXONS, NEURONS = 4609, 64


def _unit_name(spec_name: str) -> str:
    """The layer of a softcore name: conv positions end ``_pos<i>_<j>_g<k>``, FC columns ``_col<i>``."""
    return re.sub(r"_(pos|col)\d.*$", "", spec_name)


def _fx_live(model: nn.Module, unit_modules: List[str], input_shape=R.INPUT_SHAPE,
             through: tuple = (nn.BatchNorm2d,)) -> List[int]:
    """Per cut between consecutive units (toolchain order): output elements of units at or before the cut read
    directly (through the modules the conversion folds into a unit, ``through``) by a unit after it; every other op
    runs on the host."""
    graph = fx.symbolic_trace(model)
    ShapeProp(graph).propagate(torch.zeros(1, *input_shape))
    modules = dict(graph.named_modules())
    position = {name: i for i, name in enumerate(unit_modules)}
    nodes = {n.target: n for n in graph.graph.nodes if n.op == "call_module"}

    def direct_consumers(node) -> List[str]:
        out, stack = [], list(node.users)
        while stack:
            user = stack.pop()
            if user.op == "call_module" and isinstance(modules[user.target], through):
                stack.extend(user.users)
            elif user.op == "call_module" and user.target in position:
                out.append(user.target)
        return out

    live = [0] * len(unit_modules)
    for name in unit_modules:
        node = nodes[name]
        readers = [position[c] for c in direct_consumers(node)]
        if readers:
            for cut in range(position[name], max(readers)):
                live[cut] += math.prod(node.meta["tensor_meta"].shape[1:])
    return live


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_units_and_live_sets_of_resnet18_w1():
    model, _flow, rep = R.mvm_mapper_repr(1.0)
    mapping = IRMapping(q_max=1.0, firing_mode="Default", max_axons=AXONS, max_neurons=NEURONS)
    mapping.collect_wire_census = True
    ir = mapping.map(rep)
    specs = list(ir.layout_softcores)
    units = residency_units(specs, cut_nodes_from_census(specs, census_of_walk(mapping)),
                            [LayoutHardCoreType(AXONS, NEURONS, 1)], segments=[s.segment_id or 0 for s in specs],
                            allow_splitting=True)
    names = [_unit_name(specs[m[0]].name) for m in units.members]
    layers = {name.replace(".", "_"): (module, hw) for name, module, hw in R.conv_layers(model)}
    assert sorted(names) == sorted(layers)
    independent = []
    for name in names:
        module, (h, w) = layers[name]
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1] + 1   # folded-BN bias
            fan_out = module.out_channels
        else:
            fan_in, fan_out = module.in_features + 1, module.out_features
        independent.append(h * w * math.ceil(fan_in / AXONS) * math.ceil(fan_out / NEURONS))
    assert list(units.tiles) == independent

    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": AXONS, "max_neurons": NEURONS, "count": sum(units.tiles)}],
        strategy=MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True)))
    layer_of = {int(n.id): n.perceptron_index for n in ir.get_neural_cores()}
    cores_per_layer: Dict[int, int] = {}
    for stage in hybrid.stages:
        if stage.kind != "neural":
            continue
        for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            assert len(placements) == 1                                  # one-to-one packing
            layer = layer_of[int(placements[0]["ir_node_id"])]
            cores_per_layer[layer] = cores_per_layer.get(layer, 0) + 1
    assert [cores_per_layer[layer] for layer in units.layers] == list(units.tiles)

    module_names = [next(n for n, m in model.named_modules() if m is layers[name][0]) for name in names]
    assert list(units.live) == _fx_live(model, module_names)


def test_live_sets_of_the_lif_deep_cnn_of_t0_03_equal_the_fx_widths():
    from .executed import templates as TV

    vehicle = TV.template_vehicle("t0_03")
    specs = vehicle.specs
    units = residency_units(specs, cut_nodes_from_census(specs, vehicle.census),
                            [LayoutHardCoreType(int(vehicle.config["max_axons"]), int(vehicle.config["max_neurons"]), 1)],
                            segments=[s.segment_id or 0 for s in specs])
    by_name = {name.replace(".", "_"): name for name, _ in vehicle.model.named_modules()}
    module_names = [by_name[_unit_name(specs[m[0]].name)] for m in units.members]
    # LIF conversion: BatchNorm and the activation fold into the unit; pooling runs on the host
    want = _fx_live(vehicle.model, module_names, input_shape=vehicle.input_shape, through=(nn.BatchNorm2d, nn.ReLU))
    assert list(units.live) == want
    assert sum(1 for w in want if w > 0) >= 3                            # the witness is not all zeros
