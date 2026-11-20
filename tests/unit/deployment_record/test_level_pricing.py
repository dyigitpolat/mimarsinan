"""The level plane (CL4): a capability level's requirement placement on a declared chip, priced as the record and
candidate planes are (``quantities/from_level.py``, ``cost/absolute/level_pricing.py``).

The period census is G3's one rule reached through ``plane_writes`` (the executed rotation, finding 4 (a)); where the
program fits its budget the residency floor keeps every tile (W03). Every constant is a test constant."""

from __future__ import annotations

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.level_pricing import price_level
from mimarsinan.deployment_record.platform_physics.host_calibration import calibration_diagnostic
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.from_level import (
    NO_LEVEL_MESH,
    SPIKING_EVENTS_WITHHELD,
    LevelViewError,
    from_level,
    lane_area_quantities,
    level_chip,
    level_program_passes,
    pass_rows,
)
from mimarsinan.deployment_record.quantities.value_domain import NO_VALUE_GRID
from mimarsinan.mapping.capability.census import level_census
from mimarsinan.mapping.capability.ladder import HOST_PAYLOAD_WRITE, ON_CHIP_WRITE_BACK, capability_level
from mimarsinan.mapping.capability.level_schedule import CoreGeometry, level_schedule
from mimarsinan.mapping.capability.mac_split import MacCounters

from unit.deployment_record.g3_fixtures import constant
from unit.deployment_record.registered_hosts import declared_rows, registered_fixture
from unit.mapping.capability.ir_fixtures import contraction_graph, lane_graph

_GEOMETRY = CoreGeometry(max_axons=256, max_neurons=256, allow_coalescing=True, allow_splitting=True)
_SPLIT4 = CoreGeometry(max_axons=4, max_neurons=4, allow_coalescing=True, allow_splitting=True)
#: The lane graph's MAC counters: no host op carries MACs; 56 committed cells, all fed by a value.
_LANE_COUNTERS = MacCounters(ir_host_macs=0, estimator_host_macs=0, estimator_total_macs=56, has_multi_input_op=False)
_P_HOST = {"low": 5.0, "nominal": 37.0, "high": 280.0, "unit": "W", "evidence_kind": "published",
           "citation": "edge host (low end); server TDP (high end) (test)", "note": "test band"}


def _platform(*, axons=256, neurons=256, semantics="mvm", **extra):
    declared = {"cores": [{"max_axons": axons, "max_neurons": neurons, "count": 1000}], "weight_bits": 8,
                "core_semantics": semantics, **extra}
    if semantics == "mvm":
        declared.setdefault("activation_bits", 8)
    return declared


#: [U-HOST2] The calibration's numbers as a test host's own description, registered around each test (a calibration
#: itself is a diagnostic, refused as a host block).
_HOST_NAME = "test_level_pricing_host"
_HOST_ROWS = {"host_macs_per_s": {"nominal": 2.0, "unit": "G/s", "evidence_kind": "estimated", "note": "test rate"},
              "t_host_op_overhead": {"low": 10.0, "nominal": 20.0, "high": 40.0, "unit": "us",
                                     "evidence_kind": "estimated", "note": "test dispatch"},
              "p_host": _P_HOST}
_host_description = registered_fixture(_HOST_NAME, _HOST_ROWS)


def _physics(profile="generic_estimated_22nm", **blocks):
    host = declared_rows(_HOST_NAME, _HOST_ROWS)
    return resolve_platform_physics(profile, merge_override_blocks({"host": host, **blocks}))


def test_the_calibration_stays_a_diagnostic():
    rows = calibration_diagnostic(macs_per_s=2.0e9, op_overhead_s=(1e-5, 2e-5, 4e-5), p_host_w=None, identity="h",
                                  declared={"p_host": _P_HOST})
    with pytest.raises(ToolchainRefusal):
        resolve_platform_physics("generic_estimated_22nm", merge_override_blocks({"host": rows}))


def _view(ir, level, budget, *, geometry=_GEOMETRY, payload_write=ON_CHIP_WRITE_BACK, platform=None,
          lanes_per_tile=256, timesteps=None, counters=None):
    placed = capability_level(level)
    row = level_census(ir, placed, value_bits=8, payload_write=payload_write)
    schedule = level_schedule(ir, placed, budget=budget, geometry=geometry, payload_write=payload_write)
    return from_level(row, schedule, platform if platform is not None else _platform(inference_batch_K=16),
                      counters=counters, lanes_per_tile=lanes_per_tile, timesteps=timesteps)


def _value(view, key):
    return view.quantities.get(key).value


def _terms(view, physics=None):
    pricing = price_level(view, physics if physics is not None else _physics())
    return {t.name: t for t in pricing.terms}, pricing


class TestThePeriodRule:
    def test_a_program_that_fits_its_budget_is_programmed_once_and_writes_nothing_per_period(self):
        view = _view(lane_graph(), "C0", 4)
        assert all(p.resident and p.written_cells_backing_store == 0 for p in pass_rows(view))
        assert _value(view, "period_rewrite_bytes") == 0 and _value(view, "runtime_written_cells") == 0
        assert _value(view, "max_writes_per_cell_per_inference") == 0

    def test_a_program_above_its_budget_rewrites_every_pass_by_the_executed_rotation(self):
        """F = 4 tiles on 1: four writing passes, each loading the type's first core (R3), which takes 4 writes per
        period; the census is G3's rule on the level's pass rows."""
        view = _view(lane_graph(), "C0", 1)
        assert [(p.written_cells_backing_store, p.payload_bytes, p.resident) for p in pass_rows(view)] == [
            (16, 16, False), (16, 16, False), (16, 16, False), (8, 8, False)]
        assert _value(view, "period_rewrite_bytes") == 56 and _value(view, "runtime_written_cells") == 56 / 16
        assert _value(view, "max_writes_per_cell_per_inference") == 4 / 16
        placed = capability_level("C0")
        schedule = level_schedule(lane_graph(), placed, budget=1, geometry=_GEOMETRY)
        chip = level_chip(_platform(inference_batch_K=16), 1)
        expected = period_census(level_program_passes(schedule, chip), weight_bits=8, capacity={(256, 256): 1})
        assert pass_rows(view) == expected.census.passes

    def test_the_rotation_reads_the_chips_weight_width(self):
        view = _view(lane_graph(), "C0", 1, platform={**_platform(inference_batch_K=1), "weight_bits": 4})
        assert [p.payload_bytes for p in pass_rows(view)] == [params_bytes(c, 4) for c in (16, 16, 16, 8)]

    def test_the_dynamic_operand_is_written_by_its_producer_every_inference(self):
        view = _view(contraction_graph(), "C3", 16, geometry=_SPLIT4, platform=_platform(
            axons=4, neurons=4, inference_batch_K=16))
        assert [p.written_cells_producer for p in pass_rows(view)] == [12]
        assert _value(view, "runtime_written_cells") == 12, "pinned static tiles, 12 producer cells per inference"
        assert _value(view, "max_writes_per_cell_per_inference") == 1, "one producer write per inference"
        host = _view(contraction_graph(), "C3", 16, geometry=_SPLIT4, payload_write=HOST_PAYLOAD_WRITE,
                     platform=_platform(axons=4, neurons=4, inference_batch_K=16))
        assert [p.written_cells_producer for p in pass_rows(host)] == [0, 12]

    def test_without_a_declared_batch_the_level_states_no_write_quantity(self):
        view = _view(lane_graph(), "C0", 1, platform=_platform())
        assert view.writes is None and not view.quantities.has("period_rewrite_bytes")


class TestTheValueRule:
    def test_the_level_states_events_window_and_carry_at_the_value_grid(self):
        """C5 on 2 tiles: [A D] [F I], each pass one stage of span 1 (window 2 + 2); 4 elements cross the cut."""
        view = _view(lane_graph(), "C5", 2)
        assert _value(view, "latency_steps") == 4 and _value(view, "timesteps") == 1
        assert _value(view, "synaptic_events") == 56, "every committed cell once, no on-chip operator MACs"
        assert _value(view, "carry_out_bytes") == 4 and _value(view, "carry_in_bytes") == 4
        assert _value(view, "sync_count") == 1 and _value(view, "pass_count") == 2
        assert view.quantities.get("latency_steps").provenance == "modeled", "above C0: a requirement placement"
        assert _view(lane_graph(), "C0", 4).quantities.get("cells_used").provenance == "static"

    def test_the_host_op_boundaries_add_their_carry(self):
        view = _view(lane_graph(), "C0", 4)
        assert _value(view, "carry_out_bytes") == 12 and _value(view, "carry_in_bytes") == 8
        assert _value(view, "sync_count") == 5, "five host segments, no pass cut"

    def test_traffic_between_the_levels_cores_is_named_unpriced_never_dropped(self):
        view = _view(lane_graph(), "C5", 2)
        assert view.quantities.withheld_reason("noc_value_bytes") == NO_LEVEL_MESH
        terms, _ = _terms(view)
        assert "e_noc_value_per_byte_hop" in terms["energy_dynamic_mj"].source.split("; unpriced: ")[1]

    def test_a_value_carry_without_the_grid_is_withheld_by_name(self):
        platform = {k: v for k, v in _platform().items() if k != "activation_bits"}
        view = _view(lane_graph(), "C0", 4, platform=platform)
        assert view.quantities.withheld_reason("carry_out_bytes") == NO_VALUE_GRID
        assert _value(_view(lane_graph(), "C5", 4, platform=platform), "carry_out_bytes") == 0

    def test_the_programming_load_is_every_static_tile_once(self):
        view = _view(lane_graph(), "C0", 1, platform=_platform(weight_bits=4))
        assert _value(view, "reprogrammed_bytes") == sum(params_bytes(c, 4) for c in (16, 16, 16, 8))
        assert _value(view, "reprogrammed_cores") == 4 and _value(view, "reprogram_passes") == 4
        assert _value(view, "segment_cores") == 4


class TestASpikingTarget:
    def test_it_is_priced_on_the_time_axis_only_at_its_declared_window(self):
        """C5 on 2 tiles at T = 4: each LIF stage runs T + span + 1 = 6 steps; carry is the dense raster."""
        platform = _platform(semantics="spiking")
        view = _view(lane_graph(), "C5", 2, platform=platform, timesteps=4, counters=_LANE_COUNTERS)
        assert _value(view, "latency_steps") == 2 * (4 + 1 + 1) and _value(view, "timesteps") == 4
        assert view.quantities.withheld_reason("synaptic_events") == SPIKING_EVENTS_WITHHELD
        assert _value(view, "carry_out_bytes") == 4, "4 elements x ceil(4 / 8) B"
        terms, pricing = _terms(view, _physics("truenorth"))
        assert "e2e_latency_s" in terms and "energy_per_inference_mj" not in terms
        assert any(r.name == "energy_per_inference_mj" for r in pricing.refusals)

    def test_it_needs_a_declared_window(self):
        with pytest.raises(LevelViewError, match="declared window"):
            _view(lane_graph(), "C5", 2, platform=_platform(semantics="spiking"))


class TestPriceLevel:
    def test_without_the_mac_counters_the_host_share_is_unknown_and_named(self):
        _, pricing = _terms(_view(lane_graph(), "C0", 1, platform=_platform(inference_batch_K=1)))
        assert any(r.name == "e2e_latency_s" and "host share is unknown" in r.reason for r in pricing.refusals)

    def test_the_write_terms_and_the_host_energy_join_the_headline(self):
        view = _view(lane_graph(), "C0", 1, platform=_platform(inference_batch_K=1), counters=_LANE_COUNTERS)
        terms, pricing = _terms(view)
        assert terms["energy_write_mj"].value > 0 and terms["energy_host_mj"].value >= 0
        assert "energy_per_inference_mj" in terms and "e2e_latency_s" in terms
        assert not any("host" in r.reason for r in pricing.refusals), "no headline refused for a host reason"

    def test_a_fitting_program_pays_no_write_energy(self):
        terms, _ = _terms(_view(lane_graph(), "C0", 4, platform=_platform(inference_batch_K=1),
                                counters=_LANE_COUNTERS))
        assert terms["energy_write_mj"].value == 0 and "energy_per_inference_mj" in terms

    def test_a_schedule_of_another_level_or_a_chip_of_two_core_types_refuses_by_name(self):
        placed = capability_level("C1")
        row = level_census(lane_graph(), placed, value_bits=8)
        other = level_schedule(lane_graph(), capability_level("C0"), budget=4, geometry=_GEOMETRY)
        with pytest.raises(LevelViewError) as caught:
            from_level(row, other, _platform(), counters=None, lanes_per_tile=256)
        assert isinstance(caught.value, ToolchainRefusal)
        mine = level_schedule(lane_graph(), placed, budget=4, geometry=_GEOMETRY)
        two = {**_platform(), "cores": [{"max_axons": 256, "max_neurons": 256, "count": 2}] * 2}
        with pytest.raises(LevelViewError, match="one declared core type"):
            from_level(row, mine, two, counters=None, lanes_per_tile=256)


class TestLanes:
    _LANES = {"area_per_lane": constant(4399.3, unit="um^2", low=439.9, note="L3 lane band (test)"),
              "e_lane_op": constant(1.738, unit="pJ", low=0.1738, note="lane op band (test)"),
              "t_lane_op": constant(1.0, unit="ns", note="one element per lane per cycle (test)")}

    def test_a_level_with_lanes_prices_their_area_and_energy(self):
        physics = _physics(lane=self._LANES)
        c1 = _view(lane_graph(), "C1", 4)
        assert _value(c1, "lanes_per_tile") == 256 and _value(c1, "lane_ops") == 12
        assert _value(_view(lane_graph(), "C0", 4), "lanes_per_tile") == 0
        terms, _ = _terms(c1, physics)
        assert "lanes: area_per_lane" in terms["chip_area_mm2"].source
        assert "lane: e_lane_op" in terms["energy_dynamic_mj"].source

    def test_the_lane_area_of_a_chip_is_its_lanes_on_every_tile(self):
        physics = _physics(lane=self._LANES)
        chip = level_chip(_platform(), 8)
        area = {t.name: t for t in price_absolute(lane_area_quantities(chip, 256), physics).terms}["chip_area_mm2"]
        bare = {t.name: t for t in price_absolute(lane_area_quantities(chip, 0), physics).terms}["chip_area_mm2"]
        assert area.value - bare.value == pytest.approx(4399.3e-6 * 256 * 8, rel=1e-9)
