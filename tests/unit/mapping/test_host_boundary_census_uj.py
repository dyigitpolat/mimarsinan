"""[U-J J-1] The host-chip boundary of a candidate program, counted on its shape-only walk.

Every value that crosses between the host and the chip per inference, by the carry census's own rule (a producer's
whole published slice, once per direction, whatever reads it): the network input read on chip, each host operator's
on-chip inputs read back and its outputs read on chip, and the readout of on-chip outputs; one transfer per
direction at each such boundary; and the bytes a MAC-free host operator reads and writes (the memory-bound class, at
the host executor's float32). The census reaches the candidate plane's quantities only where the resolved physics
prices the host-chip link (an edge host block), so every other candidate keeps its quantities byte for byte.

Vehicle: Conv(3->8, 3x3, pad 1) on 3x8x8 -> ReLU -> AvgPool(2) -> Flatten -> Linear(128->10) under the MVM packaging:
the walk keeps the ReLU and the pool on the host, so 64 conv cores emit 512 values the ReLU reads back, the pool
writes 128 values the Linear's core reads, the 192 input values enter the conv cores and the 10 outputs read out."""

from __future__ import annotations

import contextlib
import functools
import io
import warnings

import pytest
import torch
import torch.nn as nn

from mimarsinan.deployment_record.introspection import scaling_ladder as LD
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.support.host_boundary import HOST_VALUE_BYTES, HostBoundaryCensus, host_boundary_census
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

SHAPE, CLASSES = (3, 8, 8), 10
INPUT, CONV_OUT, POOL_OUT = 3 * 8 * 8, 8 * 8 * 8, 8 * 4 * 4


@functools.lru_cache(maxsize=1)
def _flow():
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2), nn.Flatten(),
                          nn.Linear(POOL_OUT, CLASSES)).eval()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return convert_torch_model(model, SHAPE, CLASSES, packaging=MVM_PACKAGING, strict=False).eval()


def _walk(wire_census=True):
    layout = LayoutIRMapping(max_axons=256, max_neurons=256, collect_wire_census=wire_census)
    layout.collect_layout_softcores(_flow().get_mapper_repr())
    return layout


class TestTheWalkCensus:
    def test_every_crossing_value_once_per_direction(self):
        census = host_boundary_census(_walk(), input_elements=INPUT)
        assert census.to_chip_elements == INPUT + POOL_OUT
        assert census.from_chip_elements == CONV_OUT + CLASSES

    def test_one_transfer_per_direction_at_each_boundary(self):
        # the input in, the ReLU's inputs out, the pool's outputs in, the readout out
        assert host_boundary_census(_walk(), input_elements=INPUT).transfers == 4

    def test_the_mac_free_host_operators_read_and_write_their_bytes(self):
        census = host_boundary_census(_walk(), input_elements=INPUT)
        assert census.memory_bound_elements == (CONV_OUT + CONV_OUT) + (CONV_OUT + POOL_OUT)
        assert HOST_VALUE_BYTES == 4

    def test_a_walk_without_its_wire_census_states_none(self):
        assert host_boundary_census(_walk(wire_census=False), input_elements=INPUT) is None


def _network():
    from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions

    flow = _flow()
    params, macs, ops = estimate_onchip_fractions(flow, SHAPE, CLASSES, metrics=("params", "macs", "ops"))
    return LD.LadderNetwork(
        name="cnn", mapper_repr=flow.get_mapper_repr(), resolution=SHAPE,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip})


def _platform(edge: bool, **extra):
    pc = {"target": "study_compass", "activation_bits": 4, "allow_scheduling": False,
          "readout_law": {"adc_bits": 32, "mult_bits": 24, "output_law": "identity", "range": "calibrated"},
          "target_overrides": {"cores": {"value": [{"max_axons": 256, "max_neurons": 64, "count": 128,
                                                     "has_bias": True}], "note": "a test chip"},
                               "onchip_arithmetic": {"value": {"classes": ["L", "E"], "lanes_per_core": 1,
                                                               "lane_block": "ng45_os_lane"},
                                                     "note": "no pool lane: the bundle's P lane would absorb the pool, "
                                                             "leaving no host operator"}}}
    if edge:
        pc["platform_physics_overrides"] = merge_override_blocks(
            {"host": host_description_block("jetson_orin_nano_8gb")})
    pc.update(extra)
    return build_platform_constraints_resolved(merge_pipeline_config({}, pc))


@functools.lru_cache(maxsize=None)
def _priced(edge: bool):
    import scaling_ladder  # noqa: F401  (the CL5 script: the candidate plane's program facts)

    network, platform = _network(), _platform(edge)
    census = LD.shape_census(network, platform)
    scheduled = LD.schedule_program(network, census, platform, scaling_ladder.program_facts,
                                    tiles=LD.packed_tiles(census, platform))
    assert scheduled.refusal is None, scheduled.refusal
    return census, LD.price_scheduled(network, census, scheduled, platform)


@pytest.fixture(autouse=True)
def _scripts_on_path(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))


class TestTheCandidatePlane:
    def test_the_shape_census_carries_the_boundary_of_the_lane_walk(self):
        """The clamp lane (L) absorbs the ReLU, so the pool is the one host operator: it reads the conv cores' 512
        values back and its 128 outputs enter the Linear's core; the input enters and the logits read out."""
        census, _ = _priced(True)
        assert census.host_boundary == HostBoundaryCensus(
            to_chip_elements=INPUT + POOL_OUT, from_chip_elements=CONV_OUT + CLASSES, transfers=4,
            memory_bound_elements=CONV_OUT + POOL_OUT)

    def test_the_edge_host_prices_the_segment_io_at_the_value_grid(self):
        _, priced = _priced(True)
        terms = priced.terms["program"]
        assert "latency_host_link_s" in terms and "energy_host_link_mj" in terms and "energy_host_idle_mj" in terms
        # 4-bit values: (192 + 128) x 4 / 8 = 160 B in, (512 + 10) x 4 / 8 = 261 B out, 4 transfers, no pass carry
        link = terms["latency_host_link_s"]
        physics_t = [_end(priced, "t_link_per_byte", c) for c in range(3)]
        mem_t = [_end(priced, "t_host_mem_per_byte", c) for c in range(3)]
        xfer = [_end(priced, "t_link_transfer", c) for c in range(3)]
        for c, value in enumerate((link.low, link.nominal, link.high)):
            assert value == pytest.approx((160 + 261) * max(physics_t[c], mem_t[c]) + 4 * xfer[c], rel=1e-12)

    def test_without_the_edge_rows_the_candidate_prices_as_before(self):
        _, priced = _priced(False)
        assert not {"latency_host_link_s", "energy_host_link_mj", "energy_host_idle_mj"} & set(priced.terms["program"])


def _end(priced, key, corner):
    from mimarsinan.deployment_record.platform_physics import PlatformPhysics

    band = PlatformPhysics.from_dict(priced.platform["platform_physics_resolved"]).band(key)
    return (band.low, band.nominal, band.high)[corner]


def test_the_write_law_columns_carry_the_host_decomposition_at_k():
    """At a declared batch K the ladder's columns carry the host's time and link, and its idle and link energy, so
    a reading can break the host out of every headline it prices at K."""
    import scaling_ladder

    network, base = _network(), _platform(True)
    census = LD.shape_census(network, base)
    scheduled = LD.schedule_program(network, census, base, scaling_ladder.program_facts,
                                    tiles=LD.packed_tiles(census, base))
    point = _platform(True, inference_batch_K=16, inference_rate_hz=30.0, min_lifetime_hours=1.0)
    columns = LD.price_scheduled(network, census, scheduled, point).write_law["terms"]
    for name in ("latency_host_s", "latency_host_link_s", "energy_host_idle_mj", "energy_host_link_mj",
                 "energy_host_mj"):
        assert columns.get(name) is not None, name
