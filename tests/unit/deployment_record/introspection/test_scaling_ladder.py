"""The shape-only scaling ladder harness: law columns against hand computations.

The point: a two-layer conv network (3x3 conv 1->10, 1x1 conv 10->10, GAP, Linear
10->4) under the value-domain contract on 15-axon x 16-neuron cores without an
on-core bias, so the bias rides one axon and every softcore fills a core alone
(two never fit side by side). Each conv output position is one bank-backed
softcore: 10 x 10 cells (9 taps + bias) and 11 x 10 (10 channels + bias); the head
is 11 x 4. The value-domain contract computes every ReLU and the GAP on the host,
so the host ops split three neural segments (conv1 | conv2 | head), each at one
latency level. A test-only physics declares eight constants, so every Pi has a
closed hand form; the regime thresholds are this test's reading constants.

Two more networks witness one law each: a single 3x3 conv 1->6 (one neural segment
under the value-domain contract; its encoding conv on the host under the spiking
contract, so no on-chip softcore), and a conv pair whose second 3x3 patch is 37
axons wide (fused cores on 16-axon cores when coalescing is undeclared).
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import io
import math
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
import mimarsinan.deployment_record.introspection.scaling_ladder_law as law
import mimarsinan.deployment_record.introspection.scaling_ladder_types as ladder_types
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
from mimarsinan.search.problems.joint.candidate_fragments import (
    StageSemantics,
    candidate_program_facts,
    with_stage_placements,
)
from mimarsinan.torch_mapping.converter import convert_torch_model

MODULES = [Path(ladder.__file__), Path(law.__file__), Path(ladder_types.__file__)]
READING = law.RegimeReading(capacity_free_max=0.099, resident_dominant_min=62.9, source="test reading")
T, ACTIVITY, BITS = 4, 0.5, 4
CONSTANTS = {  # display units: pJ and ns
    "e_mac": 1.0, "t_cycle": 10.0, "e_dma_per_byte": 2.0, "e_program_per_byte": 3.0,
    "e_core_program": 5.0, "t_program_per_byte": 7.0, "e_core_init": 11.0, "t_core_init": 13.0,
}
PHYSICS = profile_from_dict({
    "format_version": 1, "name": "ladder_test", "display_name": "Ladder test", "description_file": "t.md",
    "validity": {"measurement_kind": "projection"},
    "constants": {k: {"nominal": v, "evidence_kind": "estimated", "note": "test"} for k, v in CONSTANTS.items()},
})
CONV1, CONV2, HEAD = 10 * 10, 11 * 10, 11 * 4
SEGMENTS = 3  # conv1 | ReLU (host) | conv2 | ReLU, GAP (host) | head
F_HAND = 2 * 4 * 4 + 1  # packed tiles at side 4: one core per softcore


class _TwoConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.b1, self.r1 = nn.Conv2d(1, 10, 3, padding=1, bias=False), nn.BatchNorm2d(10), nn.ReLU()
        self.c2, self.b2, self.r2 = nn.Conv2d(10, 10, 1, bias=False), nn.BatchNorm2d(10), nn.ReLU()
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(10, 4)

    def forward(self, x):
        x = self.r2(self.b2(self.c2(self.r1(self.b1(self.c1(x))))))
        return self.fc(self.flatten(self.pool(x)))


class _OneConv(nn.Module):
    """One 3x3 conv 1->6 with its bias: every output position a 10 x 6 softcore on one shared bank."""

    def __init__(self):
        super().__init__()
        self.c, self.r = nn.Conv2d(1, 6, 3, padding=1), nn.ReLU()

    def forward(self, x):
        return self.r(self.c(x))


class _WidePatch(nn.Module):
    """conv1 1->4 3x3 (10-axon patches), conv2 4->4 3x3 (37-axon patches), GAP, Linear 4->2 (5 x 2)."""

    def __init__(self):
        super().__init__()
        self.c1, self.b1, self.r1 = nn.Conv2d(1, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU()
        self.c2, self.b2, self.r2 = nn.Conv2d(4, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU()
        self.pool, self.flatten, self.fc = nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)

    def forward(self, x):
        x = self.r2(self.b2(self.c2(self.r1(self.b1(self.c1(x))))))
        return self.fc(self.flatten(self.pool(x)))


def _network(side: int, model: type = _TwoConv, name: str = "two_conv",
             packaging=MVM_PACKAGING) -> ladder.LadderNetwork:
    torch.manual_seed(0)
    shape = (1, side, side)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model().eval(), shape, 4, packaging=packaging, strict=False).eval()
        params, macs, ops = estimate_onchip_fractions(flow, shape, 4, metrics=("params", "macs", "ops"))
    return ladder.LadderNetwork(
        name=name, mapper_repr=flow.get_mapper_repr(), resolution=shape,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip},
        conversion=ladder.certificate_record(getattr(flow, "conversion_certificate", None)),
    )


def _platform(cores: int) -> dict:
    return {
        "cores": [{"max_axons": 15, "max_neurons": 16, "count": cores, "has_bias": False}],
        "allow_scheduling": True, "max_schedule_passes": 64, "allow_neuron_splitting": False,
        "allow_coalescing": True, "weight_bits": BITS, "simulation_steps": T, "activity_factor": ACTIVITY,
        "platform_physics_resolved": PHYSICS.to_dict(),
    }


def _facts(softcores, noc, platform):
    semantics = StageSemantics()
    noc = with_stage_placements(softcores, noc, semantics)
    return noc, candidate_program_facts(softcores, noc, semantics, timesteps=platform["simulation_steps"],
                                        weight_bits=platform["weight_bits"],
                                        pass_transfer=transfer_for(streamed=False))


def _row(side: int = 4, cores: int = 64, k: int = 16, platform: dict | None = None, **declared):
    network = _network(side)
    platform = _platform(cores) if platform is None else platform
    census = ladder.shape_census(network, platform)
    priced = ladder.price_program(network, census, platform, _facts)
    return ladder.ladder_row(network, census, priced, {**platform, "inference_batch_K": k, **declared}, READING)


def _positions(side: int) -> int:
    return side * side


class TestHandComputation:
    def test_cells_tiles_and_x_on_one_point(self):
        row = _row(side=4, cores=64)
        logical = CONV1 + CONV2 + HEAD
        as_mapped = _positions(4) * (CONV1 + CONV2) + HEAD
        tiles = 2 * _positions(4) + 1
        assert (row["logical_cells"], row["as_mapped_cells"], row["packed_tiles"]) == (logical, as_mapped, tiles)
        assert row["x_fluid"] == pytest.approx(logical / (64 * 15 * 16), rel=1e-12)
        assert row["x_packed"] == pytest.approx(tiles / 64, rel=1e-12)
        assert row["replication"] == pytest.approx(as_mapped / logical, rel=1e-12)

    def test_pi_and_pi_over_k_on_one_point(self):
        row = _row(side=4, cores=64, k=16)
        tiles = 2 * _positions(4) + 1
        cells = _positions(4) * (CONV1 + CONV2) + HEAD
        # Every segment fits the 64 cores, so the program runs one pass per segment; each pass is one
        # execution stage at one latency level, which runs T + 0 + 1 steps under the default LIF window
        # (chip_simulation/stage_timesteps.py, the executed-window rule).
        steps = SEGMENTS * (T + 1)
        assert (row["passes"], row["latency_steps"]) == (SEGMENTS, steps)
        once_bytes = _positions(4) * (math.ceil(CONV1 * BITS / 8) + math.ceil(CONV2 * BITS / 8)) \
            + math.ceil(HEAD * BITS / 8)
        assert row["programmed_bytes_once"] == once_bytes and row["programmed_cores_once"] == tiles
        c = CONSTANTS
        pi_e = ((c["e_dma_per_byte"] + c["e_program_per_byte"]) * once_bytes + c["e_core_program"] * tiles) / (
            c["e_mac"] * cells * T * ACTIVITY + c["e_core_init"] * tiles)
        pi_t = c["t_program_per_byte"] * once_bytes / (c["t_cycle"] * steps + c["t_core_init"] * tiles)
        for axis, expected in (("energy", pi_e), ("time", pi_t)):
            assert row[axis]["pi"] == pytest.approx([expected] * 3, rel=1e-12)
            assert row[axis]["pi_over_k"] == pytest.approx([expected / 16] * 3, rel=1e-12)
            print(f"[ladder] hand point: Pi_{axis} = {expected:.6g}, Pi/K = {expected / 16:.6g}")


def _snapshots(census, axons: int, neurons: int, bound: int, *, splitting: bool, coalescing: bool):
    pack = pack_layout(softcores=census.softcores, core_types=[LayoutHardCoreType(axons, neurons, bound)],
                       allow_neuron_splitting=splitting, allow_coalescing=coalescing)
    assert pack.feasible
    return tuple(pack.used_core_snapshots or ())


class TestPackedTiles:
    @pytest.mark.parametrize("axons, neurons, splitting", [(8, 16, False), (8, 8, True)])
    def test_a_split_softcore_counts_one_tile_per_fragment(self, axons, neurons, splitting):
        """With coalescing declared, the packer pre-splits every softcore wider than one 8-axon core
        (10 or 11 axons) into axon-coalescing fragments of one core each, so no used core is fused;
        with 8 neurons and neuron splitting the 10-neuron softcores also split across cores."""
        network = _network(4)
        platform = {**_platform(64), "allow_neuron_splitting": splitting,
                    "cores": [{"max_axons": axons, "max_neurons": neurons, "count": 64, "has_bias": False}]}
        assert platform["allow_coalescing"] is True
        census = ladder.shape_census(network, platform)
        shapes = [(10, 10)] * _positions(4) + [(11, 10)] * _positions(4) + [(11, 4)]
        assert sorted((int(s.input_count), int(s.output_count)) for s in census.softcores) == sorted(shapes)
        hand = sum(math.ceil(a / axons) * math.ceil(n / neurons) for a, n in shapes)
        snapshots = _snapshots(census, axons, neurons, hand, splitting=splitting, coalescing=True)
        assert len(snapshots) == hand and all(int(s.axons_per_core) <= axons for s in snapshots)
        print(f"[ladder] split point {axons}x{neurons}: packed tiles {hand} by hand, {len(snapshots)} used cores")
        assert ladder.packed_tiles(census, platform) == hand

    def test_a_fused_core_counts_its_member_cores(self):
        """With coalescing undeclared, the packer places each 37-axon conv2 patch (4 channels x 9 taps
        + bias) on one fused instance of three 16-axon cores and reports it as one used core, so the
        packed tile count (the hand sum, 65) exceeds the used-core count (33)."""
        network = _network(4, _WidePatch, "wide_patch")
        platform = {**_platform(64), "allow_coalescing": False, "allow_neuron_splitting": False,
                    "cores": [{"max_axons": 16, "max_neurons": 16, "count": 64, "has_bias": False}]}
        census = ladder.shape_census(network, platform)
        shapes = [(10, 4)] * _positions(4) + [(37, 4)] * _positions(4) + [(5, 2)]
        assert sorted((int(s.input_count), int(s.output_count)) for s in census.softcores) == sorted(shapes)
        hand = sum(math.ceil(a / 16) * math.ceil(n / 16) for a, n in shapes)
        assert hand == _positions(4) * (1 + 3) + 1 == 65
        snapshots = _snapshots(census, 16, 16, hand, splitting=False, coalescing=False)
        fused = [s for s in snapshots if int(s.axons_per_core) > 16]
        assert len(snapshots) == len(shapes) == 33 and len(fused) == _positions(4)
        print(f"[ladder] fused point 16x16: packed tiles {hand} by hand, {len(snapshots)} used cores, "
              f"{len(fused)} fused")
        assert ladder.packed_tiles(census, platform) == hand


class TestReplicationAndResolution:
    def test_replication_equals_a_hand_count_on_a_two_layer_conv_point(self):
        row = _row(side=8, cores=256)
        assert row["replication"] == pytest.approx(
            (_positions(8) * (CONV1 + CONV2) + HEAD) / (CONV1 + CONV2 + HEAD), rel=1e-12)

    def test_x_fluid_is_resolution_free_while_x_packed_and_replication_are_not(self):
        low, high = _row(side=4, cores=16), _row(side=8, cores=16)
        assert low["x_fluid"] == high["x_fluid"]
        assert low["x_packed"] != high["x_packed"]
        assert low["replication"] != high["replication"]


class TestDeterminism:
    def test_two_runs_give_identical_rows_except_wall_and_rss(self):
        volatile = {"layout_wall_s", "point_wall_s", "peak_rss_bytes"}
        first, second = _row(cores=8), _row(cores=8)
        assert {k: v for k, v in first.items() if k not in volatile} == {
            k: v for k, v in second.items() if k not in volatile}


class TestRegimeAndOverhead:
    def test_labels_follow_the_thresholds_and_a_crossing_band_straddles(self):
        crossing = law.regime(Band(0.05, 0.1, 0.2, "b"), READING, multiplexed=True)
        assert crossing == {"low": "capacity-free", "nominal": "informative", "high": "informative",
                            "reading": law.STRADDLES}
        assert law.regime(Band(70.0, 80.0, 90.0, "b"), READING, multiplexed=True)["reading"] == "resident-dominant"
        assert law.regime(Band(0.01, 0.02, 0.03, "b"), READING, multiplexed=False)["reading"] == law.RESIDENT

    def test_row_labels_match_the_printed_thresholds(self):
        row = _row(cores=8, k=1)
        for axis in ("energy", "time"):
            labels = row[axis]["regime"]
            for corner, value in zip(law.CORNERS, row[axis]["pi_over_k"]):
                assert labels[corner] == READING.label(value)
        assert row["regime_reading"]["capacity_free_max"] == READING.capacity_free_max

    def test_multi_pass_point_prints_the_priced_and_closed_form_overheads(self):
        row = _row(cores=8, k=16)
        assert row["passes"] >= 2 and row["x_packed"] > 1
        for axis in ("energy", "time"):
            m = row[axis]["overhead"]
            assert m is not None and min(m["toolchain_priced"]) >= 1.0
            print(f"[ladder] multi-pass point, {axis}: M toolchain {m['toolchain_priced']}, "
                  f"floor {m['closed_form_floor']} ({m['rel_diff_floor']}), "
                  f"stateless {m['closed_form_stateless']} ({m['rel_diff_stateless']})")

    def test_kappa_and_m_are_priced_per_corner_without_re_sorting(self):
        compute = Band(1.0, 2.0, 4.0, "compute")
        boundary = law.sub_corners(Band(2.0, 3.0, 4.5, "no host"), compute, "boundary")
        assert boundary == (1.0, 1.0, 0.5)  # corner-aligned: no re-sorting after the subtraction
        m = law.overhead(compute=compute, boundary=boundary, period_writes=Band(4.0, 4.0, 4.0, "period"),
                         once=Band(8.0, 8.0, 8.0, "once"), k=2, x=2.0, passes=4)
        assert m is not None
        kappa = [1.0 / 1.0, 1.0 / 2.0, 0.5 / 4.0]
        assert m["kappa"] == pytest.approx(kappa, rel=1e-15)
        assert m["toolchain_priced"] == pytest.approx(
            [1.0 + ka + 2.0 / c for ka, c in zip(kappa, (1.0, 2.0, 4.0))], rel=1e-15)
        assert m["closed_form_stateless"] == pytest.approx(
            [1.0 + 4.0 / c + ka for ka, c in zip(kappa, (1.0, 2.0, 4.0))], rel=1e-15)
        phi = 4 * (2.0 - 1.0) / (3 * 2.0)
        assert m["phi_floor"] == pytest.approx(phi, rel=1e-15)
        assert m["closed_form_floor"] == pytest.approx(
            [1.0 + phi * 4.0 / c + ka for ka, c in zip(kappa, (1.0, 2.0, 4.0))], rel=1e-15)


class TestScheduleReading:
    """The regime reading, M and the write census follow the priced program, not capacity.

    Under U1 every segment's first pass is non-resident, so a program of several
    segments rewrites its cores on every inference even when C >= F.
    """

    def test_a_segment_multiplexed_program_at_c_equal_f_reads_and_prices_like_c_below_f(self):
        at_f, below = _row(cores=F_HAND, k=1), _row(cores=F_HAND - 1, k=1)
        assert at_f["x_packed"] == 1.0 and below["x_packed"] > 1.0
        assert at_f["capacity_reading"] == law.FITS_BY_CAPACITY
        assert below["capacity_reading"] != law.FITS_BY_CAPACITY
        for row in (at_f, below):
            assert (row["passes"], row["reprogram_passes"], row["segments"]) == (SEGMENTS, SEGMENTS, SEGMENTS)
            assert row["schedule_reading"] == law.schedule_reading(SEGMENTS, SEGMENTS)
            assert row["schedule_reading"].startswith("U1: multiplexed by 3")
            assert row["k_star"]["max_writes_per_cell_per_inference"] == float(SEGMENTS)
        for axis in ("energy", "time"):
            assert at_f[axis]["regime"]["reading"] not in (law.RESIDENT, "fits (no multiplexing)")
            m_f, m_b = at_f[axis]["overhead"], below[axis]["overhead"]
            assert m_f is not None and m_b is not None
            assert m_f["toolchain_priced"] == pytest.approx(m_b["toolchain_priced"], rel=1e-12)
            # At x <= 1 the residency floor writes nothing (phi = 0): the floor form is 1 + kappa.
            assert m_f["phi_floor"] == 0.0
            assert m_f["closed_form_floor"] == pytest.approx([1.0 + ka for ka in m_f["kappa"]], rel=1e-15)
            print(f"[ladder] C = F = {F_HAND}, {axis}: M toolchain {m_f['toolchain_priced']}, "
                  f"floor {m_f['closed_form_floor']}, stateless {m_f['closed_form_stateless']}")

    def test_a_resident_program_writes_no_cell_per_inference(self):
        network = _network(4)
        platform = {**_platform(64), "allow_scheduling": False}
        census = ladder.shape_census(network, platform)
        priced = ladder.price_program(network, census, platform, _facts)
        point = {**platform, "inference_batch_K": 1, "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}
        row = ladder.ladder_row(network, census, priced, point, READING)
        assert (row["passes"], row["reprogram_passes"]) == (1, 1)
        assert row["schedule_reading"].startswith("resident")
        assert row["k_star"]["max_writes_per_cell_per_inference"] == 0.0
        for axis in ("energy", "time"):
            assert row[axis]["regime"]["reading"] == law.RESIDENT and row[axis]["overhead"] is None
        declared = ladder.ladder_row(network, census, dataclasses.replace(priced, endurance=1e6), point, READING)
        assert declared["k_star"]["k_star_census"] == 0.0

    def test_a_multi_pass_resident_program_prints_m_without_period_writes(self):
        """One conv is one neural segment: 16 positions on one shared bank, one 10 x 6 softcore each,
        each alone on a 15-axon core, so F = 16. At C = F/4 the program runs 4 passes and only the
        first writes the bank (the other three are residency passes on it): resident, 0 writes per
        inference, yet multi-pass. M is priced with no period writes, M = 1 + kappa per corner, beside
        the floor form at phi = Q(x - 1)/((Q - 1)x) = 1 (x = Q = 4), which equals the stateless form."""
        network = _network(4, _OneConv, "one_conv")
        census = ladder.shape_census(network, _platform(64))
        f = ladder.packed_tiles(census, _platform(64))
        assert (len(census.softcores), census.segments, f) == (_positions(4), 1, _positions(4))
        platform = _platform(f // 4)
        priced = ladder.price_program(network, census, platform, _facts, tiles=f)
        row = ladder.ladder_row(network, census, priced, {**platform, "inference_batch_K": 16}, READING)
        assert (row["x_packed"], row["passes"], row["reprogram_passes"]) == (4.0, 4, 1)
        assert row["schedule_reading"] == law.schedule_reading(1, 1) and row["schedule_reading"].startswith("resident")
        assert row["k_star"]["max_writes_per_cell_per_inference"] == 0.0
        for axis in ("energy", "time"):
            col = row[axis]
            assert col["regime"]["reading"] == law.RESIDENT
            m = col["overhead"]
            assert m is not None, f"{axis}: no M on a multi-pass point"
            kappa = [b / c for b, c in zip(col["terms"]["boundary"], col["terms"]["compute_only"])]
            assert m["kappa"] == pytest.approx(kappa, rel=1e-15)
            assert m["toolchain_priced"] == pytest.approx([1.0 + ka for ka in kappa], rel=1e-15)
            assert m["phi_floor"] == 1.0
            assert m["closed_form_floor"] == pytest.approx(m["closed_form_stateless"], rel=1e-15)
            assert m["rel_diff_floor"] == pytest.approx(m["rel_diff_stateless"], rel=1e-15)
            assert all(d < 0.0 for d in m["rel_diff_floor"])
            print(f"[ladder] single segment at C = F/4, {axis}: M toolchain {m['toolchain_priced']} "
                  f"(1 + kappa), floor at phi 1 {m['closed_form_floor']} ({m['rel_diff_floor']})")


class TestKStar:
    def test_the_two_factors_multiply_to_the_printed_ratio(self):
        row = _row(cores=1)  # 254 logical cells on one 240-cell core: x_fluid > 1
        ks = row["k_star"]
        assert row["x_fluid"] > 1 and row["x_packed"] > 1 and ks["ratio"] > 0
        assert abs(ks["factor_packing_replication"] * ks["factor_rotation"] - ks["ratio"]) <= 1e-12 * ks["ratio"]

    def test_k_star_needs_rate_lifetime_and_endurance(self):
        ks = _row(cores=8, inference_rate_hz=30.0, min_lifetime_hours=43_800.0)["k_star"]
        assert ks["k_star_absent"] == "undeclared: endurance_cycles"
        full = law.k_star(x_fluid=2.0, x_packed=3.0, max_writes=4.0, max_writes_source="hand",
                          rate_hz=30.0, lifetime_hours=10.0, endurance=1e6)
        assert full["k_star_closed"] == pytest.approx(1.0 * 30.0 * 36_000.0 / 1e6, rel=1e-15)
        assert full["k_star_census"] == pytest.approx(4.0 * 30.0 * 36_000.0 / 1e6, rel=1e-15)


class TestRefusalsAndPurity:
    def test_a_row_may_change_only_the_per_row_declarations(self):
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        priced = ladder.price_program(network, census, platform, _facts)
        with pytest.raises(ladder.LadderDeclarationError) as refused:
            ladder.ladder_row(network, census, priced, {**platform, "weight_bits": 8}, READING)
        assert isinstance(refused.value, ToolchainRefusal) and isinstance(refused.value, ValueError)

    def test_two_declared_core_shapes_are_refused_by_name(self):
        network = _network(4)
        platform = {**_platform(8), "cores": [{"max_axons": 15, "max_neurons": 16, "count": 4, "has_bias": False},
                                              {"max_axons": 16, "max_neurons": 16, "count": 4, "has_bias": False}]}
        census = ladder.shape_census(network, platform)
        with pytest.raises(ladder.LadderGeometryError) as refused:
            ladder.packed_tiles(census, platform)
        assert isinstance(refused.value, ToolchainRefusal) and isinstance(refused.value, ValueError)

    def test_a_program_with_no_on_chip_softcore_is_refused_by_name(self):
        """Under the spiking contract the one conv is the encoding layer and runs on the host: the
        program holds no on-chip softcore, so x_fluid and the replication have no denominator."""
        network = _network(4, _OneConv, "host_only_conv", SPIKING_PACKAGING)
        with pytest.raises(ladder.LadderGeometryError, match="no on-chip softcore") as refused:
            ladder.shape_census(network, _platform(64))
        assert isinstance(refused.value, ToolchainRefusal) and isinstance(refused.value, ValueError)

    def test_an_unschedulable_budget_is_a_row_refused_by_name(self):
        network = _network(4)
        platform = {**_platform(16), "allow_scheduling": False}  # 33 tiles, one pass, 16 cores
        census = ladder.shape_census(network, platform)
        priced = ladder.price_program(network, census, platform, _facts, tiles=F_HAND)
        row = ladder.ladder_row(network, census, priced, {**platform, "inference_batch_K": 1}, READING)
        assert row["refusal"]["name"] == ladder.LadderScheduleRefusal.__name__ and row["refusal"]["reason"]
        assert issubclass(ladder.LadderScheduleRefusal, ToolchainRefusal)
        assert issubclass(ladder.LadderScheduleRefusal, ValueError)

    def test_every_row_prints_the_conversion_certificate(self):
        row = _row(cores=8)
        assert row["conversion"]["certified"] is True and "certified" in row["conversion"]["summary"]
        absent = ladder.certificate_record(None)
        assert absent["certified"] is False and absent["summary"]

    def test_price_program_names_its_stages_in_order(self):
        network = _network(4)
        platform = _platform(8)
        census = ladder.shape_census(network, platform)
        stages: list = []
        ladder.price_program(network, census, platform, _facts, on_stage=stages.append)
        assert stages == ["pack", "schedule", "price"]

    def test_the_harness_imports_no_pipelining_torch_mapping_or_search(self):
        banned = ("mimarsinan.pipelining", "mimarsinan.torch_mapping", "mimarsinan.search")
        for path in MODULES:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                assert not [n for n in names if n.startswith(banned)], (path.name, names)
