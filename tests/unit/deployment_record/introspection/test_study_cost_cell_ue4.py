"""[U-E4 A-1] A study cost cell reproduced: one resident design priced on the study's chip by the toolchain and by
the study's estimator rule.

The study prices a resident design's compute as n_act tile activations at A_TILE_J x tile bits / 65,536 each
(x9_0_v3.census: a layer takes ceil(cin k^2 / R) x ceil(cout / 64) tiles per position, n_act sums them over the
positions; cd_preview.designs), and its area as C tiles x the tile area x s_AU (cd43_common, s_AU at the declared
read 1.130). The toolchain prices the same design on the study chip description (``study_*`` profiles and bundles)
from its candidate plane. Equal within the declared band, and every term that differs named:

  compute    the toolchain's e_mac x MAC events equals the study's n_act x A_tile exactly on a design whose layers
             fill their tiles; on a partly filled tile the study charges the whole activation, so the toolchain's
             term is the study's times the occupancy (MACs / (n_act x MACs per tile));
  lanes, NoC the toolchain adds the lane block's e_lane_op x lane ops and e_noc_value_per_byte_hop x value byte-hops,
             which the study does not price (lane energy about 0.4% of MAC energy, codesign.md sec. 2.2);
  area       the tile part (cells and the columns' logic) is the probe's instance area, which the study rounds to
             four digits; the lane at the block's high end is the study's s_AU = 1.130 of a G3 tile (13.0%, the
             block's 15,000.1 um2 against the study's 15,015 um2); routers, NoC-tile overhead and the fixed global
             area are added by the toolchain and not priced by the study.

Each band is solved for its parts from the three corners the pricer states (every part but the study's own is a
banded constant read from the resolved physics), so the test reads the pricer's output, not its internals.
Synthetic MLPs with seed-0 weights: the price depends on shapes and declarations only.
"""

from __future__ import annotations

import contextlib
import functools
import io
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.deployment_record.introspection import scaling_ladder as LD
from mimarsinan.deployment_record.platform_physics import PlatformPhysics
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
A_TILE_J = 2.6006349206349206e-09    # cd_preview.A_TILE_J["SRAM-DDR"]
TILE_MM2 = {256: 0.1155, 128: 0.07677}           # cd_preview.TILE_MM2 (COMPASS / VR-like G3; ISAAC-like G6)
PROBE_MM2 = {256: 0.115456, 128: 0.076766}       # pd_probe/pd_table.json instance_mm2 (g3, g6)
S_AU = 1.13                                      # price_measured.S_AU_CONVENTION (codesign.md sec. 2.2, D136)
CHIP = {"study_compass": 256, "study_vr_like": 256, "study_isaac_like": 128}
LAW = {"adc_bits": 32, "mult_bits": 24, "output_law": "identity", "range": "calibrated"}


@functools.lru_cache(maxsize=1)
def _ladder():
    sys.path.insert(0, str(SCRIPTS))
    try:
        import scaling_ladder
    finally:
        sys.path.remove(str(SCRIPTS))
    return scaling_ladder


def _network(widths):
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fractions
    from mimarsinan.torch_mapping.converter import convert_torch_model

    layers = [nn.Flatten()]
    for k, (cin, cout) in enumerate(zip(widths[:-1], widths[1:])):
        layers += ([nn.ReLU()] if k else []) + [nn.Linear(cin, cout)]
    torch.manual_seed(0)
    model = nn.Sequential(*layers).eval()
    shape, classes = (widths[0],), widths[-1]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model, shape, classes, packaging=MVM_PACKAGING, strict=False).eval()
        params, macs, ops = estimate_onchip_fractions(flow, shape, classes, metrics=("params", "macs", "ops"))
    return LD.LadderNetwork(
        name="mlp", mapper_repr=flow.get_mapper_repr(), resolution=shape,
        total_params=float(sum(p.numel() for p in flow.parameters())),
        host_census={"host_macs": macs.host, "onchip_macs": macs.onchip, "compute_op_count": ops.host,
                     "host_params": params.host, "onchip_params": params.onchip})


def _platform(target, rows, count, **extra):
    pc = {"target": target, "readout_law": dict(LAW), "activation_bits": 4, "allow_scheduling": False,
          "target_overrides": {"cores": {"value": [{"max_axons": rows, "max_neurons": 64, "count": count,
                                                     "has_bias": True}], "note": "the design's tile count"}}}
    pc.update(extra)
    return build_platform_constraints_resolved(merge_pipeline_config({}, pc))


@functools.lru_cache(maxsize=None)
def _priced(target, widths, count):
    rows = CHIP[target]
    network = _network(widths)
    base = _platform(target, rows, count)
    census = LD.shape_census(network, base)
    scheduled = LD.schedule_program(network, census, base, _ladder().program_facts,
                                    tiles=LD.packed_tiles(census, base))
    assert scheduled.refusal is None, scheduled.refusal
    point = _platform(target, rows, count, inference_batch_K=1, inference_rate_hz=_ladder().RATE_HZ,
                      min_lifetime_hours=_ladder().LIFETIME_HOURS)
    priced = LD.price_scheduled(network, census, scheduled, point)
    return priced, PlatformPhysics.from_dict(point["platform_physics_resolved"]), point


def _study_compute_j(widths, rows):
    """The study's rule (x9_0_v3.census, cd_preview.designs): n_act tiles of R rows by 64 four-bit weights."""
    n_act = sum(math.ceil(cin / rows) * math.ceil(cout / 64) for cin, cout in zip(widths[:-1], widths[1:]))
    return n_act * A_TILE_J * rows * 64 * 4 / 65536, n_act


def _ends(band):
    return np.array([band.low, band.nominal, band.high])


def _parts(total, columns):
    """Least-squares solve of total = known(point) + sum of banded parts: the corners give one equation each."""
    matrix = np.column_stack([np.ones(3)] + [np.asarray(c, dtype=float) for c in columns])
    solution, residual, *_ = np.linalg.lstsq(matrix, total, rcond=None)
    assert np.allclose(matrix @ solution, total, rtol=1e-9, atol=0), "the band is not this sum of parts"
    return solution


#: (chip, widths, cores): designs whose every layer fills its tiles (MACs = n_act x MACs per tile).
EXACT = [("study_compass", (256, 256, 64), 5), ("study_vr_like", (256, 256, 64), 5),
         ("study_isaac_like", (128, 128, 64), 3)]


@pytest.mark.parametrize("target,widths,count", EXACT)
class TestAResidentDesignOnItsStudyChip:
    def test_the_compute_energy_is_the_study_tile_activations_and_lanes_and_noc_are_added(self, target, widths,
                                                                                            count):
        priced, physics, _point = _priced(target, widths, count)
        assert (priced.written_passes, priced.passes) == (0, 1), "a resident design: one pass, nothing rewritten"
        dynamic = _ends(priced.terms["compute_only"]["energy_dynamic_mj"]) * 1e-3
        basis = priced.terms["compute_only"]["energy_dynamic_mj"].basis
        assert "compute: e_mac" in basis and "lane: e_lane_op" in basis and "noc_value" in basis
        mac, lanes, noc = _parts(dynamic, [_ends(physics.band("e_lane_op")) / physics.band("e_lane_op").nominal,
                                           _ends(physics.band("e_noc_value_per_byte_hop")) /
                                           physics.band("e_noc_value_per_byte_hop").nominal])
        study, _n_act = _study_compute_j(widths, CHIP[target])
        assert math.isclose(mac, study, rel_tol=1e-9), "e_mac x MAC events = n_act x A_tile on full tiles"
        assert lanes > 0 and noc >= -1e-9 * mac, "the study prices neither"

    def test_the_area_is_the_study_tile_area_and_the_named_additions(self, target, widths, count):
        priced, physics, point = _priced(target, widths, count)
        area = _ends(priced.terms["program"]["chip_area_mm2"])
        # one lane per core slot of the floorplan (lanes_per_tile x tiles), at least one per core
        slots = point["cores_per_tile_resolved"] * point["tile_grid_rows_resolved"] * point["tile_grid_cols_resolved"]
        assert slots >= count
        lane = _ends(physics.band("area_per_lane")) * 1e6 * slots       # m2 -> mm2
        fixed = _ends(physics.band("area_global_fixed")) * 1e6
        noc_tile = (_ends(physics.band("area_per_router")) + _ends(physics.band("area_per_tile_fixed"))) * 1e6
        logic = _ends(physics.band("area_per_neuron_logic")) * 1e6 * count * 64   # one logic block per column
        cells, noc_tiles = _parts(area - lane - fixed - logic, [noc_tile])
        tile_part = cells + logic[1]
        rows = CHIP[target]
        assert math.isclose(tile_part, count * PROBE_MM2[rows], rel_tol=1e-5), "the probe's instance area"
        assert abs(tile_part / (count * TILE_MM2[rows]) - 1) < 5e-4, "the study's four-digit rounding"
        assert round(noc_tiles) >= 1 and abs(noc_tiles - round(noc_tiles)) < 1e-6, "whole NoC tiles"
        if rows == 256:  # the study's s_AU is 13.0% of a G3 tile, the lane block's high end
            assert abs(physics.band("area_per_lane").high * 1e6 / ((S_AU - 1) * TILE_MM2[rows]) - 1) < 2e-3


class TestAPartlyFilledTile:
    def test_the_study_charges_the_whole_activation_so_the_toolchain_term_is_its_occupancy(self):
        widths = (256, 160, 64)  # 160 outputs take 3 column groups of 64, 160 inputs 160 of 256 rows
        priced, physics, _point = _priced("study_compass", widths, 4)
        dynamic = _ends(priced.terms["compute_only"]["energy_dynamic_mj"]) * 1e-3
        mac, _lanes, _noc = _parts(dynamic, [_ends(physics.band("e_lane_op")) / physics.band("e_lane_op").nominal,
                                             _ends(physics.band("e_noc_value_per_byte_hop")) /
                                             physics.band("e_noc_value_per_byte_hop").nominal])
        study, n_act = _study_compute_j(widths, 256)
        macs = sum(a * b for a, b in zip(widths[:-1], widths[1:]))
        assert n_act == 3 + 1 and macs < n_act * 256 * 64
        assert math.isclose(mac / study, macs / (n_act * 256 * 64), rel_tol=1e-9)
        assert mac < study
