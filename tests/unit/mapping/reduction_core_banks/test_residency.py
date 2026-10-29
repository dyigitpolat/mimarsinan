"""[U-L2, B-2] Residency of the fit-only design with its pool on the chip: CIFAR ResNet-18 at width 1/2 (X10-TA's
architecture: 3x3 stem, BasicBlocks [2, 2, 2, 2] at 32, 64, 128, 256 channels, 1x1 projections, global average pool,
100 classes) on the route (b) one-copy rows' tile (256 x 64, on-core bias, coalescing, 8,192 passes), on the candidate
plane the exploration and W48 price.

U-L's finding (L-3): with the pool on the host the one-copy program fits resident on one copy per bank at FO*'s 204
cores (no pass rewritten per period); with the pool on the P lane its 16 owned 0/1 reduction cores (256 channels at
4 x 4, 16 windows per 256-axon group) cost that residency, and 19 passes are rewritten per period. Banked, the 16
window groups are 16 instances of one bank, which one copy serves, so the program is resident at 204 cores again; the
resident minimum is the host-pool program's plus at most the bank's copy."""

from __future__ import annotations

import contextlib
import functools
import io
import warnings

import pytest
import torch

from mimarsinan.deployment_record.introspection import scaling_ladder as LD

from unit.chip_simulation.value_run.integer_programs import _ResNet
from unit.mapping.lanes.lane_planes import ladder, period_rows, plane_platform

from . import vehicles as V

PASSES = 8192
TILE = dict(axons=256, neurons=64, passes=PASSES, coalescing=True, hardware_bias=True)


@functools.lru_cache(maxsize=1)
def _network():
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
    from mimarsinan.torch_mapping.converter import convert_torch_model

    shape, classes = (3, 32, 32), 100
    torch.manual_seed(0)
    model = _ResNet((32, 64, 128, 256), blocks=2, classes=classes).eval()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model, shape, classes, packaging=MVM_PACKAGING, strict=False).eval()
        params, macs, ops = estimate_onchip_fractions(flow, shape, classes, metrics=("params", "macs", "ops"))
    return LD.LadderNetwork(
        name="R18 w0.5", mapper_repr=flow.get_mapper_repr(), resolution=shape,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip})


@functools.lru_cache(maxsize=None)
def _written(classes: tuple, banks, count: int) -> tuple:
    """(passes, passes rewritten per period, soft cores) of the one-copy program at ``count`` cores."""
    platform = plane_platform(classes, count=count, multiplexing=True, **TILE)
    if banks is not None:
        platform[V.KEY] = banks
    network = _network()
    census = LD.shape_census(network, platform)
    tiles = LD.packed_tiles(census, platform)
    scheduled = LD.schedule_program(network, census, platform, ladder().program_facts, tiles=tiles)
    assert scheduled.refusal is None, scheduled.refusal
    rows = period_rows(scheduled.facts.passes)
    return rows["passes"], sum(1 for w in rows["written"] if w), len(census.softcores)


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_the_fit_only_design_is_resident_at_its_design_count_again():
    host = _written(V.HOST_POOL, None, 204)
    unbanked = _written(V.POOL, None, 204)
    banked = _written(V.POOL, True, 204)
    assert host[1] == 0, host                    # U-L: resident with the pool on the host
    assert unbanked[1] > 0, unbanked             # U-L's finding: the 16 owned pool cores cost the residency
    assert banked[1] == 0, banked                # banked: resident again
    assert banked[2] == unbanked[2] == host[2] + 16  # the same soft cores: 16 instances, one bank
    assert banked[0] == unbanked[0] + 15, (banked, unbanked)  # the pool level: 16 passes on one copy, not 1
