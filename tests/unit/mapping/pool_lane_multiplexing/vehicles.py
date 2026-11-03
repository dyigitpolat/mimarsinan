"""[U-L] Vehicles of pool lanes under position multiplexing: lane programs that fuse the pool on chip (lanes L, E, P)
beside their host-pool twins (L, E: the pool a host op, as the route (b) one-copy rows declare it), built the
pipeline's way at a chip's tiler limits, and the one-copy (W02) and pipelined (U-F4) programs of either through the
pipeline's own builder entry. Inputs are synthetic (the integer generator and seeded normals): no data split is read.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.lanes.ir.pool_cores import LANE_OUTPUTS
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from unit.mapping.lanes.lane_planes import ir_geometry, plane_platform
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform
from unit.mapping.lanes.test_lane_shape_walk import FIXTURES

POOL = ("L", "E", "P")
HOST_POOL = ("L", "E")
PASSES = 1 << 16
#: Chips the small vehicles run on: every level's copies together (8 cores) and in several groups (3 cores).
GEOMETRIES = {
    "together": dict(axons=256, neurons=64, count=8, passes=PASSES, coalescing=True, hardware_bias=True),
    "groups": dict(axons=256, neurons=64, count=3, passes=PASSES, coalescing=True, hardware_bias=True),
}
#: The route (b) one-copy rows' tile (256 x 64, on-core bias, coalescing) at FO*'s core count and a smaller one.
ROUTE_B = dict(axons=256, neurons=64, passes=PASSES, coalescing=True, hardware_bias=True)


def _shared_net():
    from unit.mapping.lanes.test_lane_call_sites import _net

    return _net(True)


#: name -> (model builder, input shape, classes): U-F4's lane-fused residual block, U-K2's network whose ReLU module
#: runs at two call sites, the transition network (two producer-side reductions and a pool lane) and the
#: quarter-width ResNet-18 of the route (b) rows.
NETWORKS = {
    "residual": FIXTURES["residual"],
    "shared": (_shared_net, (3, 16, 16), 10),
    "transition": FIXTURES["transition"],
    "resnet18_quarter": FIXTURES["resnet18_quarter"],
}


def chip(classes: Sequence[str], *, pipelined: Optional[bool] = None, multiplexing: bool = True,
         **geometry) -> Dict[str, Any]:
    """A lane chip both planes read (``lane_planes.plane_platform``), declaring the one-copy arm and, when given,
    its pipelined form."""
    platform = plane_platform(classes, multiplexing=multiplexing, **geometry)
    if pipelined is not None:
        platform["pipelined_multiplexing"] = pipelined
    return platform


@functools.lru_cache(maxsize=None)
def _program(network: str, classes: Tuple[str, ...], axons: int, neurons: int, hardware_bias: bool,
             coalescing: bool, declared: tuple):
    model, shape, num_classes = NETWORKS[network]
    return build_lane_program(model(), shape, num_classes=num_classes,
                              platform=lane_platform(classes, cores=[dict(core) for core in declared]),
                              max_axons=axons, max_neurons=neurons, hardware_bias=hardware_bias,
                              allow_coalescing=coalescing)


def program(network: str, classes: Sequence[str], platform: Dict[str, Any]):
    """``network``'s lane program mapped at ``platform``'s tiler limits and fused on its declared cores, as Soft Core
    Mapping maps and fuses it (``test_lane_shape_walk._fixture``'s rule)."""
    ir = ir_geometry(platform)
    declared = tuple(tuple(sorted(core.items())) for core in platform["cores"])
    return _program(network, tuple(classes), ir["max_axons"], ir["max_neurons"], ir["hardware_bias"],
                    ir["allow_coalescing"], declared)


def build(prog, platform: Dict[str, Any]):
    return build_hybrid_mapping_for_pipeline(prog.ir, platform)


def neural(hybrid) -> list:
    return [s for s in hybrid.stages if s.kind == "neural"]


def value(hybrid, x) -> torch.Tensor:
    with torch.no_grad():
        return ValueHybridCoreFlow(hybrid, dtype=torch.float64, device="cpu")(x.double())


def lane_cores(prog) -> List[NeuralCore]:
    """The cores the lane fusion inserted (a pool lane's reduction cores, a reduction's reduced cores)."""
    return [n for n in prog.ir.nodes if isinstance(n, NeuralCore) and getattr(n, LANE_OUTPUTS, None) is not None]


def passes_of(hybrid, node_ids) -> List[int]:
    """The passes (stage ordinals among the neural stages) that place a soft core of the given IR nodes."""
    wanted = {int(n) for n in node_ids}
    return [k for k, s in enumerate(neural(hybrid))
            if any(int(p["ir_node_id"]) in wanted for ps in s.hard_core_mapping.soft_core_placements_per_hard_core
                   for p in ps)]
