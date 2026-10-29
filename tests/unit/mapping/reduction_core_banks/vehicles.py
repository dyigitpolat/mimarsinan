"""[U-L2] Vehicles of banked reduction cores: U-L's pool-lane vehicles (``pool_lane_multiplexing.vehicles``: lane
programs that fuse the pool on chip, lanes L, E, P, beside their host-pool twins) with the decision declared on the
platform Soft Core Mapping fuses the lanes on, so each pool lane's window groups of one matrix over one producer grid
are instances of one shared bank; the one-copy arm (W02) and its pipelined form (U-F4) are built through the pipeline's
own builder entry. Inputs are synthetic (the integer generator and seeded normals): no data split is read."""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.ir import NeuralCore

from unit.mapping.lanes.lane_planes import ir_geometry
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform
from unit.mapping.pool_lane_multiplexing import vehicles as UL

KEY = "reduction_core_banks"
POOL, HOST_POOL, PASSES, GEOMETRIES, ROUTE_B, NETWORKS = (UL.POOL, UL.HOST_POOL, UL.PASSES, UL.GEOMETRIES, UL.ROUTE_B,
                                                         UL.NETWORKS)
#: The declaration as Soft Core Mapping reads it: the one-copy arm and, under it, the banked reduction cores.
BANKED = {"allow_scheduling": True, "position_multiplexing": True, KEY: True}
build, neural, value, lane_cores, passes_of = UL.build, UL.neural, UL.value, UL.lane_cores, UL.passes_of


def chip(classes: Sequence[str], *, banks: Optional[bool] = True, pipelined: Optional[bool] = None,
         multiplexing: bool = True, **geometry) -> Dict[str, Any]:
    """U-L's lane chip (both planes read it), declaring the decision when ``banks`` is given."""
    platform = UL.chip(classes, pipelined=pipelined, multiplexing=multiplexing, **geometry)
    if banks is not None:
        platform[KEY] = banks
    return platform


@functools.lru_cache(maxsize=None)
def _program(network: str, classes: Tuple[str, ...], axons: int, neurons: int, hardware_bias: bool,
             coalescing: bool, declared: tuple, banks: Optional[bool]):
    model, shape, num_classes = NETWORKS[network]
    extra = {} if banks is None else {**BANKED, KEY: banks}
    return build_lane_program(model(), shape, num_classes=num_classes,
                              platform=lane_platform(classes, cores=[dict(core) for core in declared], **extra),
                              max_axons=axons, max_neurons=neurons, hardware_bias=hardware_bias,
                              allow_coalescing=coalescing)


def program(network: str, classes: Sequence[str], platform: Dict[str, Any], *, banks: Optional[bool] = True):
    """``network``'s lane program at ``platform``'s tiler limits and declared cores (U-L's rule), its lanes fused with
    the decision declared ``banks`` (None: undeclared, U-L's program)."""
    ir = ir_geometry(platform)
    declared = tuple(tuple(sorted(core.items())) for core in platform["cores"])
    return _program(network, tuple(classes), ir["max_axons"], ir["max_neurons"], ir["hardware_bias"],
                    ir["allow_coalescing"], declared, banks)


def unbanked(network: str, classes: Sequence[str], platform: Dict[str, Any]):
    """U-L's unbanked P-lane program of the same network on the same chip (the decision undeclared)."""
    return program(network, classes, platform, banks=None)


def pool_cores(prog) -> List[NeuralCore]:
    """A pool lane's 0/1 reduction cores (the fusion's ``lane_owner`` cores; a reduction's reduced cores carry none)."""
    from mimarsinan.mapping.lanes.ir.fusion import LANE_OWNER

    return [n for n in lane_cores(prog) if getattr(n, LANE_OWNER, None) is not None]


def lane_banks(prog) -> Dict[int, List[NeuralCore]]:
    """{bank id: its instances, in IR order} over the pool lanes' cores."""
    banks: Dict[int, List[NeuralCore]] = {}
    for core in pool_cores(prog):
        if core.weight_bank_id is not None:
            banks.setdefault(int(core.weight_bank_id), []).append(core)
    return banks
