"""The value-domain quantity rule on both planes: MAC events, one evaluation per stage, value carry."""

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.value_domain import (
    ValueDomainActivityError,
    ValueDomainCensusError,
    crossing_elements,
    names_value_domain,
    value_bytes,
    value_carry,
    value_window,
)

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses

from unit.deployment_record.g3_fixtures import (
    carry_record,
    compute,
    physics,
    record_with,
    schedule,
    segment,
)
from unit.deployment_record.record_fixtures import make_full_record

_PROGRAMMING = type("P", (), dict(segment_cores=5, committed_cells=1234, reprogrammed_cores=5,
                                   reprogrammed_bytes=1234, reprogram_passes=2))()
#: A dense crossing census at T = 32: widths 10 and 6, each 4 B per element per direction.
_CARRY = {"carried_wires": 2, "carried_bytes": 64, "peak_live_bytes": 64,
          "boundary_out_bytes": 64, "boundary_in_bytes": 64}


def _pcfg(**extra):
    base = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 8}], "simulation_steps": 32,
            "activity_factor": 0.0, "weight_bits": 8}
    base.update(extra)
    return base


def _passes(*stage_latencies):
    """One (pass, segment) row per stage-latency tuple: what the candidate plane states."""
    return ProgramPasses(rows=tuple(
        PassRow(segment_index=0, resident=False, core_cells=(1,), marks=frozenset(), core_extents=None,
                compute_steps=None, stage_latencies=tuple(latencies))
        for latencies in stage_latencies))


def _context(pcfg, **kwargs):
    kwargs.setdefault("programming", _PROGRAMMING)
    kwargs.setdefault("carry", _CARRY)
    kwargs.setdefault("latency_steps", 99)
    kwargs.setdefault("passes", _passes((0,), (2, 1)))
    return candidate_context_from_platform(pcfg, host_macs=0, onchip_macs=10, compute_op_count=0,
                                           **kwargs)


def _carry_physics():
    """e_mac and both carry channels, no host and no static power: the headline prices on its own."""
    return profile_from_dict({
        "format_version": 1, "name": "c", "display_name": "C", "description_file": "c.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {key: {"nominal": 1.0, "unit": "pJ", "evidence_kind": "estimated", "note": "test"}
                      for key in ("e_mac", "e_dma_per_byte", "e_readout_per_byte")},
    })


def _candidate_quantities(context):
    return from_candidate(layout=None, chip_param_capacity=None, total_params=None,
                          host_side_segment_count=None, context=context)


class TestTheRule:
    def test_the_surface_names_the_value_domain(self):
        assert names_value_domain({"core_semantics": "mvm"})
        assert not names_value_domain({"core_semantics": "spiking"})
        assert not names_value_domain({})

    def test_each_stage_evaluates_once_after_its_latency_and_input_delivery(self):
        assert value_window([0, 2, 1]) == 1 + 3 + 2
        assert value_window([]) == 0

    def test_the_dense_crossing_census_inverts_exactly(self):
        assert crossing_elements(64, 32) == 16
        assert crossing_elements(3 * 2, 9) == 3        # ceil(9 / 8) = 2 B per element

    def test_a_byte_count_the_census_cannot_produce_refuses_by_name(self):
        with pytest.raises(ValueDomainCensusError) as info:
            crossing_elements(65, 32)
        assert isinstance(info.value, ToolchainRefusal)

    def test_quantities_that_differ_only_in_what_they_withhold_differ(self):
        held = Quantities({}, withheld={"carry_in_bytes": "activation_bits undeclared"})
        assert held != Quantities({})
        assert held == Quantities({}, withheld={"carry_in_bytes": "activation_bits undeclared"})

    def test_values_are_carried_at_the_declared_grid(self):
        assert value_bytes(16, 8) == 16
        assert value_bytes(3, 4) == 2                  # ceil(12 / 8)

    def test_a_nonzero_carry_without_its_census_window_is_withheld_by_name(self):
        """Targeted pass: the grid is declared but the carry census states no window, so the crossing
        elements cannot be recovered; the carry is withheld by name, never sized, never a crash."""
        sized, withheld = value_carry({"boundary_out_bytes": 32, "boundary_in_bytes": 0}, None, 8)
        assert sized == {"carry_in_bytes": 0}
        assert withheld == {"carry_out_bytes": "the carry census states no window"}


class TestCandidatePlane:
    def test_value_domain_context_reads_the_value_rule(self):
        context = _context(_pcfg(core_semantics="mvm", activation_bits=8))
        assert context.timesteps == 1
        assert context.latency_steps == 6
        assert context.carried_raster_bytes is None
        assert (context.carry_out_bytes, context.carry_in_bytes) == (16, 16)
        assert context.activity_factor is None
        values = _candidate_quantities(context)
        assert values.get("synaptic_events").value == 1234.0
        assert values.get("synaptic_events").provenance == "static"
        assert not values.has("carried_raster_bytes")

    def test_without_the_value_grid_the_carry_stays_absent(self):
        context = _context(_pcfg(core_semantics="mvm"))
        assert (context.carry_out_bytes, context.carry_in_bytes) == (None, None)
        values = _candidate_quantities(context)
        assert not values.has("carry_out_bytes") and not values.has("carry_in_bytes")
        for key in ("carry_out_bytes", "carry_in_bytes"):
            assert "activation_bits" in values.withheld_reason(key)

    def test_a_known_zero_carry_stays_zero_without_the_grid(self):
        zero = {"carried_wires": 0, "carried_bytes": 0, "peak_live_bytes": 0,
                "boundary_out_bytes": 0, "boundary_in_bytes": 0}
        context = _context(_pcfg(core_semantics="mvm"), carry=zero)
        assert (context.carry_out_bytes, context.carry_in_bytes) == (0, 0)
        values = _candidate_quantities(context)
        assert values.get("carry_out_bytes").value == values.get("carry_in_bytes").value == 0.0
        assert values.withheld_reason("carry_in_bytes") is None

    def test_a_carry_without_the_grid_is_named_unpriced_on_the_headline(self):
        values = _candidate_quantities(_context(_pcfg(core_semantics="mvm")))
        source = {t.name: t for t in price_absolute(values, _carry_physics()).terms}["energy_per_inference_mj"].source
        assert "unpriced: " in source
        assert "e_dma_per_byte(carry_inject; activation_bits undeclared" in source
        assert "e_readout_per_byte(carry_readout; activation_bits undeclared" in source

    def test_the_dynamic_term_names_a_withheld_carry_as_the_headline_does(self):
        values = _candidate_quantities(_context(_pcfg(core_semantics="mvm")))
        dynamic = {t.name: t for t in price_absolute(values, _carry_physics()).terms}["energy_dynamic_mj"].source
        assert "; unpriced: " in dynamic
        assert "e_dma_per_byte(carry_inject; activation_bits undeclared" in dynamic
        assert "e_readout_per_byte(carry_readout; activation_bits undeclared" in dynamic

    def test_a_declared_activity_factor_refuses_by_name(self):
        with pytest.raises(ValueDomainActivityError) as info:
            _context(_pcfg(core_semantics="mvm", activity_factor=0.1))
        assert isinstance(info.value, ToolchainRefusal)

    def test_a_spiking_context_is_unchanged(self):
        context = _context(_pcfg(activity_factor=0.25))
        assert context.timesteps == 32
        assert context.latency_steps == 99
        assert context.carried_raster_bytes == 64
        assert (context.carry_out_bytes, context.carry_in_bytes) == (64, 64)
        values = _candidate_quantities(context)
        assert values.get("synaptic_events").value == pytest.approx(1234 * 32 * 0.25)
        assert values.get("synaptic_events").provenance == "modeled"

    def test_the_value_headline_is_priced_as_mac_events(self):
        context = _context(_pcfg(core_semantics="mvm", activation_bits=8))
        target = physics("generic_estimated_22nm")
        pricing = price_absolute(_candidate_quantities(context), target)
        terms = {t.name: t for t in pricing.terms}
        assert "energy_per_inference_mj" in terms
        e_mac = target.band("e_mac").nominal
        assert terms["energy_dynamic_mj"].value >= 1234 * e_mac * 1e3
        assert "e_mac x synaptic_events" in terms["energy_dynamic_mj"].source or \
            "compute: e_mac" in terms["energy_dynamic_mj"].source


def _value_record(**platform):
    stages = (
        segment(1, 0, 0, [100, 50], latencies=[0, 0]),
        segment(2, 0, 1, [30], latencies=[1]),
        compute(3),
        segment(4, 1, 0, [20, 10], latencies=[2, 3]),
    )
    platform.setdefault("core_semantics", "mvm")
    return record_with(schedule(stages, carry=carry_record([10, 6], extra_in_trains=(4,))), **platform)


class TestRecordPlane:
    def test_a_value_record_reads_the_value_rule(self):
        values = from_record(_value_record(activation_bits=8))
        assert values.get("timesteps").value == 1.0
        assert values.get("latency_steps").value == 1 + 1 + 2
        assert values.get("synaptic_events").value == 210.0
        assert values.get("synaptic_events").provenance == "static"
        assert values.get("carry_out_bytes").value == 16.0
        # Boundary-in is read from the census, so the extra re-injected train counts.
        assert values.get("carry_in_bytes").value == 20.0
        assert not values.has("carried_raster_bytes")

    def test_without_the_value_grid_the_record_carry_stays_absent(self):
        values = from_record(_value_record())
        assert not values.has("carry_out_bytes") and not values.has("carry_in_bytes")
        # The fixture seals no partition census; with no host work the headline prices on its own.
        hosted = Quantities({**{key: values.get(key) for key in values.keys()},
                             "host_macs": QuantityValue(0.0, "static")}, withheld=values.withheld)
        source = {t.name: t for t in price_absolute(hosted, _carry_physics()).terms}["energy_per_inference_mj"].source
        assert "e_dma_per_byte(carry_inject; activation_bits undeclared" in source
        assert values.withheld_reason("noc_value_bytes") == "the sealed record keeps no wire census (KA-G3-1)"

    def test_the_record_dynamic_term_names_its_withheld_components(self):
        """The carry without the grid and the record's value NoC (KA-G3-1), on a target that declares the
        carry rows but not the value NoC row: every withheld factor is named on the dynamic term too."""
        values = from_record(_value_record())
        hosted = Quantities({**{key: values.get(key) for key in values.keys()},
                             "host_macs": QuantityValue(0.0, "static")}, withheld=values.withheld)
        terms = {t.name: t for t in price_absolute(hosted, _carry_physics()).terms}
        for name in ("energy_dynamic_mj", "energy_per_inference_mj"):
            assert "e_dma_per_byte(carry_inject; activation_bits undeclared" in terms[name].source, name
            assert ("e_noc_value_per_byte_hop(noc_value; the sealed record keeps no wire census"
                    in terms[name].source), name

    def test_a_record_that_carries_nothing_states_a_zero_carry_without_the_grid(self):
        stages = (segment(1, 0, 0, [5], latencies=[0]), compute(2), segment(3, 1, 0, [5], latencies=[0]))
        values = from_record(record_with(schedule(stages), core_semantics="mvm"))
        assert values.get("carry_out_bytes").value == values.get("carry_in_bytes").value == 0.0

    def test_a_record_with_an_unstated_core_latency_states_no_window(self):
        stages = (segment(1, 0, 0, [5, 5], latencies=[0, None]),)
        values = from_record(record_with(schedule(stages), core_semantics="mvm"))
        assert not values.has("latency_steps")

    def test_a_declared_activity_factor_refuses_by_name_on_the_record(self):
        with pytest.raises(ValueDomainActivityError):
            from_record(_value_record(activity_factor=0.2))

    def test_a_spiking_record_is_unchanged(self):
        record = make_full_record()
        values = from_record(record)
        assert values.get("timesteps").value == float(record.timing.s_global)
        assert values.get("latency_steps").value == float(record.timing.latency.compute_steps)
        assert values.has("carried_raster_bytes")

    def test_candidate_and_record_agree_on_the_rule(self):
        """The same program on both planes: one stage per (segment, pass) and the same crossing census."""
        record_values = from_record(_value_record(activation_bits=8))
        carry = {"boundary_out_bytes": 64, "boundary_in_bytes": 80, "carried_bytes": 0,
                 "peak_live_bytes": 0, "carried_wires": 2}
        programming = type("P", (), dict(segment_cores=5, committed_cells=210, reprogrammed_cores=5,
                                          reprogrammed_bytes=210, reprogram_passes=3))()
        context = _context(_pcfg(core_semantics="mvm", activation_bits=8), programming=programming,
                           carry=carry, passes=_passes((0,), (0,), (1,)))
        candidate_values = _candidate_quantities(context)
        for key in ("timesteps", "latency_steps", "synaptic_events", "carry_out_bytes", "carry_in_bytes"):
            assert candidate_values.get(key).value == record_values.get(key).value, key
