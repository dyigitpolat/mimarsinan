"""[U-C2 G-2, survey gap B6.3] The compute time of an event-driven chip is priced on its event census.

The vocabulary's ``t_cycle`` is the wall of one TIMESTEP, and every latency consumer multiplies it by timestep counts
(``latency_steps``). ODIN has no fixed timestep: one input event sweeps the core's 256 neurons at two clock cycles per
synaptic operation (512 cycles per event), so a ``t_cycle`` of one synaptic operation priced a timestep at least 256
times too fast wherever a step carried an event. The time twin of ``e_mac`` (``t_synaptic_op`` x ``synaptic_events``)
prices it where the chip declares it, with ``t_cycle`` left undeclared: the latency, the throughput, the static energy
over that latency, Pi's denominator and the regime labels follow the census, and refuse by name until a census or a
declared activity exists, as the energy already does. The census total states one core's events exactly. On a chip
of several concurrent cores the candidate plane prices each (pass, segment) row at its busiest physical core (its
as-mapped cells x timesteps x activity_factor, the energy's own model), so the latency, the overlapped write exposure
and the layer-pipelined bound follow the per-core census; a view that states the census as one total (a sealed
record) is refused by name [targeted M2]. Loihi's and TrueNorth's ``t_cycle`` are timestep walls (a measured
sustained per-timestep wall, a fixed 1 ms tick) and keep the step law.

The event census is the existing ``synaptic_events`` field (sealed by the event-census unit U-S, ``EnergyRecord.
synaptic_events``; a test constant stands in for it here, by name). Every other number is a test constant or read off
the shipped profiles.
"""

from __future__ import annotations

import pytest

import mimarsinan.deployment_record.cost.compute_time as compute_time
import mimarsinan.deployment_record.introspection.scaling_ladder_law as law
import mimarsinan.deployment_record.introspection.scaling_ladder_reading as reading
from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.correlation import correlate, get_case
from mimarsinan.deployment_record.cost.absolute import price_absolute, price_write_terms
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs
from mimarsinan.deployment_record.cost.compute_time import EVENT_LAW, STEP_LAW, compute_law
from mimarsinan.deployment_record.objectives import OBJECTIVES, run_capability_probe
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS, spec_for
from mimarsinan.deployment_record.platform_physics.registry import profile_description_path
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.gui.wizard import physics_panel
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import constant
from unit.deployment_record.introspection.test_ladder_tile_writes import CELLS, TILES, _physics, _row
from unit.deployment_record.introspection.test_scaling_ladder import ACTIVITY, CONSTANTS, CONV1, CONV2, HEAD, T

#: ODIN at its declared corner: two clock cycles per synaptic operation at 75 MHz; one input event sweeps 256 neurons.
SOP_S, NEURONS, F_CLK_HZ = 2 / 75e6, 256, 75e6
#: A toolchain census of one inference: its executed timesteps, and the event census U-S seals (a test constant
#: standing in for ``EnergyRecord.synaptic_events``).
STEPS, EVENTS = 1584, 2_000_000
#: [targeted M2] The candidate plane's per-core event census: the busiest physical core's events of each pass, summed.
BUSIEST_EVENTS = "synaptic_events_busiest_core"


def _odin():
    return get_platform_physics("odin")


def _q(**values: float) -> Quantities:
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _priced(physics, quantities):
    pricing = price_absolute(quantities, physics)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


class TestTheVocabulary:
    def test_the_time_twin_of_e_mac(self):
        spec, twin = spec_for("t_synaptic_op"), PHYSICS_CONSTANTS["e_mac"]
        assert (spec.group, spec.dimension, spec.multiplicand) == (twin.group, "time", twin.multiplicand)

    def test_t_cycle_stays_one_timestep_and_names_the_event_driven_alternative(self):
        doc = spec_for("t_cycle").doc
        assert "timestep" in doc and "t_synaptic_op" in doc
        assert spec_for("t_cycle").multiplicand == "latency_steps"

    def test_the_per_core_census_is_a_catalog_quantity(self):
        from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS
        assert getattr(compute_time, "BUSIEST_EVENTS", None) == BUSIEST_EVENTS
        doc = QUANTITY_SPECS[BUSIEST_EVENTS].doc
        assert "busiest" in doc and "activity_factor" in doc and "one total" in doc

    def test_the_laws(self):
        assert compute_law(_odin()) == EVENT_LAW == "t_synaptic_op"
        for name in ("loihi", "truenorth", "neurram", "generic_estimated_22nm"):
            assert compute_law(get_platform_physics(name)) == STEP_LAW == "t_cycle", name


class TestTheOdinDeclaration:
    def test_odin_declares_the_synaptic_operation_and_no_timestep(self):
        physics = _odin()
        assert not physics.has("t_cycle")
        band = physics.band("t_synaptic_op")
        assert (band.low, band.nominal, band.high) == pytest.approx((SOP_S,) * 3, rel=1e-5)

    def test_the_source_is_the_paper_and_the_rtl(self):
        value = _odin().constants["t_synaptic_op"]
        text = f"{value.citation} {value.derivation} {value.note}"
        for token in ("frenkel2019odin", "two clock cycles", "512", "256", "controller.v", "1781931"):
            assert token in text, token

    def test_the_description_says_so(self):
        text = profile_description_path("odin").read_text(encoding="utf-8")
        assert "t_synaptic_op" in text
        assert "`t_cycle` is one synaptic operation" not in text


class TestPricing:
    def test_a_timestep_census_is_never_priced_at_one_operation_per_step(self):
        """The defect: 1,584 executed timesteps priced at 26.67 ns each (42.24 us). Without an event census the
        latency is refused by name, and with it the throughput and the static energy over it."""
        terms, refusals = _priced(_odin(), _q(latency_steps=STEPS, cores_physical=1, host_macs=0))
        assert "e2e_latency_s" not in terms and "throughput_inferences_s" not in terms
        assert "t_synaptic_op" in refusals["e2e_latency_s"] and "synaptic_events" in refusals["e2e_latency_s"]
        assert "e2e_latency_s" in refusals["throughput_inferences_s"]

    def test_the_census_prices_the_latency_the_throughput_and_the_static_energy(self):
        terms, refusals = _priced(_odin(), _q(latency_steps=STEPS, synaptic_events=EVENTS, cores_physical=1,
                                              host_macs=0))
        e2e = terms["e2e_latency_s"]
        assert e2e.value == pytest.approx(SOP_S * EVENTS, rel=1e-5), refusals
        assert e2e.source.startswith("compute: t_synaptic_op [")
        assert terms["throughput_inferences_s"].value == pytest.approx(1 / (SOP_S * EVENTS), rel=1e-5)
        physics = _odin()
        static_w = physics.band("p_static_per_core").nominal + physics.band("p_static_global").nominal
        assert terms["energy_static_mj"].value == pytest.approx(static_w * e2e.value * 1e3, rel=1e-9)

    def test_an_input_event_is_one_sweep_of_the_core(self):
        """512 cycles per input event: the census counts one operation per neuron the core sweeps."""
        terms, _ = _priced(_odin(), _q(synaptic_events=NEURONS, cores_physical=1, host_macs=0))
        assert terms["e2e_latency_s"].value * F_CLK_HZ == pytest.approx(512, rel=1e-5)

    def test_a_census_total_alone_on_several_cores_refuses_by_name(self):
        """A view that states the events as one total (a sealed record's census) cannot split them per core."""
        terms, refusals = _priced(_odin(), _q(synaptic_events=EVENTS, cores_physical=64, host_macs=0))
        assert "e2e_latency_s" not in terms
        reason = refusals["e2e_latency_s"]
        assert "64" in reason and "per-core" in reason and "t_synaptic_op" in reason
        assert "one total" in reason and BUSIEST_EVENTS in reason

    def test_several_cores_price_the_busiest_core_of_each_pass(self):
        """[targeted M2] The cores of a pass run concurrently, each its own events one at a time: the pass's wall is its
        busiest core's, and the passes run in turn."""
        busiest = EVENTS // 40
        terms, refusals = _priced(_odin(), _q(synaptic_events=EVENTS, cores_physical=64, host_macs=0,
                                              **{BUSIEST_EVENTS: busiest}))
        e2e = terms["e2e_latency_s"]
        assert e2e.value == pytest.approx(SOP_S * busiest, rel=1e-5), refusals
        assert e2e.source.startswith(f"compute: t_synaptic_op [") and BUSIEST_EVENTS in e2e.source
        assert terms["throughput_inferences_s"].value == pytest.approx(1 / (SOP_S * busiest), rel=1e-5)
        assert "energy_static_mj" in terms

    def test_one_core_keeps_the_census_total(self):
        terms, _ = _priced(_odin(), _q(synaptic_events=EVENTS, cores_physical=1, host_macs=0,
                                       **{BUSIEST_EVENTS: EVENTS}))
        assert terms["e2e_latency_s"].source.startswith("compute: t_synaptic_op [")
        assert BUSIEST_EVENTS not in terms["e2e_latency_s"].source

    def test_a_timestep_wall_takes_precedence(self):
        """A chip declaring both: its timestep wall already holds the step's events (the capability probe declares
        every constant, so it keeps the step law)."""
        physics = profile_from_dict({
            "format_version": 1, "name": "both", "display_name": "both", "description_file": "b.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {"t_cycle": constant(1.0, unit="us"), "t_synaptic_op": constant(1.0, unit="ns")}})
        terms, _ = _priced(physics, _q(latency_steps=10, synaptic_events=1000, cores_physical=4, host_macs=0))
        assert terms["e2e_latency_s"].value == pytest.approx(10e-6, rel=1e-12)
        assert compute_law(physics) == STEP_LAW

    @pytest.mark.parametrize("name", ["loihi", "truenorth"])
    def test_the_other_spiking_chips_keep_their_timestep(self, name):
        physics = get_platform_physics(name)
        assert physics.has("t_cycle") and not physics.has("t_synaptic_op")
        terms, _ = _priced(physics, _q(latency_steps=STEPS, synaptic_events=EVENTS, cores_physical=4, host_macs=0))
        assert terms["e2e_latency_s"].source.startswith("compute: t_cycle [")
        assert terms["e2e_latency_s"].value == pytest.approx(physics.band("t_cycle").nominal * STEPS, rel=1e-12)


class TestTheCandidatePlane:
    def test_odins_latency_axis_follows_the_declared_activity(self):
        spec = OBJECTIVES.get("e2e_latency_s")
        assert spec.available(run_capability_probe("joint", _odin(), activity_factor=0.5))
        assert not spec.available(run_capability_probe("joint", _odin(), activity_factor=0.0))
        truenorth = get_platform_physics("truenorth")
        assert spec.available(run_capability_probe("joint", truenorth, activity_factor=0.0))


class TestTheOverlappedWrite:
    K = 4

    def _census(self):
        return PassPeriodCensus(passes=tuple(
            PassCensus(position=i, segment_index=0, resident=False, cores=1, resident_cores=0,
                       written_cells_backing_store=100, written_cells_producer=0, payload_bytes=50, compute_steps=9)
            for i in range(2)))

    def _policy(self):
        return WritePolicy(batch_k=self.K, overlap_mode="overlapped", shadow_capacity_fraction=0.5,
                           core_capacity=(((256, 256), 4),))

    def test_a_plane_without_a_per_pass_event_census_refuses_by_name(self):
        pricing = price_write_terms(self._census(), _odin(), self._policy())
        reason = {r.name: r.reason for r in pricing.refusals}["latency_write_exposed_s"]
        assert "t_synaptic_op" in reason and "per-pass" in reason and "one total" in reason

    def test_each_pass_hides_writes_behind_its_busiest_cores_operations(self):
        """[targeted M2] The overlapped exposure reads the same per-pass compute as the latency: t_synaptic_op x the
        busiest core's events of the pass the write hides behind."""
        odin = _odin()
        wall = odin.band("t_core_program").nominal  # each pass rewrites its one core: the whole-image wall
        events = (100_000, 1_000_000)  # pass 0 hides 2.67 ms of the next write; pass 1 hides all of it
        pricing = price_write_terms(self._census(), odin, self._policy(), pass_sops=events)
        term = {t.name: t for t in pricing.terms}.get("latency_write_exposed_s")
        assert term is not None, {r.name: r.reason for r in pricing.refusals}
        want = max(0.0, wall / self.K - SOP_S * events[0]) + max(0.0, wall / self.K - SOP_S * events[1])
        assert term.value == pytest.approx(want, rel=1e-5)

    def test_the_plane_carries_its_pass_events_to_the_pricer(self):
        census, policy = self._census(), self._policy()
        inputs = WriteInputs(census=census, policy=policy, pass_sops=(1.0, 2.0))
        assert inputs.pass_sops == (1.0, 2.0)


class TestTheLadder:
    """Pi's denominator, the regime and the layer-pipelined bound on the hand point of ``test_scaling_ladder``."""

    C = {**{k: v for k, v in CONSTANTS.items() if k != "t_cycle"}, "t_synaptic_op": 2.0}

    def test_one_core_prices_the_compute_from_the_activity_census(self):
        row = _row(_physics("ladder_event", self.C), cores=1, k=16)
        compute = row["time"]["terms"]["compute_only"]
        want = (self.C["t_synaptic_op"] * CELLS * T * ACTIVITY + self.C["t_core_init"] * TILES) * 1e-9
        assert compute == pytest.approx([want] * 3, rel=1e-12)
        assert row["time"]["pi"] is not None and row["time"]["regime"] is not None

    def test_several_cores_price_each_pass_at_its_busiest_core(self):
        """[targeted M2] At 64 cores the hand point is one resident pass of three segment rows (conv1 | conv2 | head),
        every softcore alone on its core: each row's busiest core is its largest softcore."""
        row = _row(_physics("ladder_event", self.C), cores=64, k=16)
        time = row["time"]
        busiest = (CONV1 + CONV2 + HEAD) * T * ACTIVITY
        want = (self.C["t_synaptic_op"] * busiest + self.C["t_core_init"] * TILES) * 1e-9
        assert time["terms"]["compute_only"] == pytest.approx([want] * 3, rel=1e-12), time["absent"]
        assert time["pi"] is not None and time["regime"] is not None and "time" in row["axes_priced"]
        assert row["energy"]["pi"] is not None

    def test_a_multiplexed_program_sums_its_passes_busiest_cores(self):
        """At 8 cores the passes rotate: every row's busiest core lies between the segment's largest softcore and the
        whole row, so the event wall lies between the per-segment sum and the serial sum of every core."""
        row = _row(_physics("ladder_event", self.C), cores=8, k=16)
        compute = row["time"]["terms"]["compute_only"]
        assert compute is not None, row["time"]["absent"]
        events = (compute[1] * 1e9 - self.C["t_core_init"] * TILES) / self.C["t_synaptic_op"]
        assert (CONV1 + CONV2 + HEAD) * T * ACTIVITY <= events + 1e-6
        assert events <= CELLS * T * ACTIVITY + 1e-6

    def test_the_layer_pipelined_bound_on_several_cores_is_the_busiest_core_of_any_pass(self):
        row = _row(_physics("ladder_event", self.C), cores=64, k=16)
        semantics = row["latency_semantics"]
        want = self.C["t_synaptic_op"] * max(CONV1, CONV2, HEAD) * T * ACTIVITY * 1e-9
        assert semantics["layer_pipelined_bound_s"] == pytest.approx([want] * 3, rel=1e-12), semantics
        assert "busiest core" in semantics["layer_pipelined_label"]

    def test_the_layer_pipelined_bound_reads_the_event_stage_where_given(self):
        compute, stage = (1e-3, 2e-3, 4e-3), (1e-4, 2e-4, 4e-4)
        event = law.layer_pipelined(compute, 50, multiplexed=False, passes=1, compute_law=EVENT_LAW,
                                    event_stage=stage)
        assert event["layer_pipelined_bound_s"] == pytest.approx(list(stage))
        assert "busiest core" in event["layer_pipelined_label"]
        absent = law.layer_pipelined(compute, 50, multiplexed=True, passes=3, compute_law=EVENT_LAW,
                                     event_stage=stage)
        assert absent["layer_pipelined_bound_s"] is None and "multiplexed" in absent["layer_pipelined_absent"]

    def test_the_layer_pipelined_bound_on_one_event_pipeline_is_the_whole_compute(self):
        compute = (1e-3, 2e-3, 4e-3)
        step = law.layer_pipelined(compute, 50, multiplexed=False, passes=1, compute_law=STEP_LAW)
        event = law.layer_pipelined(compute, 50, multiplexed=False, passes=1, compute_law=EVENT_LAW)
        assert step["layer_pipelined_bound_s"] == pytest.approx([v / 50 for v in compute])
        assert event["layer_pipelined_bound_s"] == pytest.approx(list(compute))
        assert "one core" in event["layer_pipelined_label"]
        semantics = reading.latency_semantics(time_per_inference=compute, compute_only=compute, k=1, rate_hz=None,
                                              carry={}, passes=1, latency_steps=50, multiplexed=False, source="t",
                                              compute_law=EVENT_LAW)
        assert semantics["layer_pipelined_bound_s"] == pytest.approx(list(compute))


class TestTheCandidateCensus:
    """[targeted M2] The candidate plane's per-core event census: per (pass, segment) row, its busiest physical core's
    as-mapped cells (static and operand cells of each tile) x timesteps x activity_factor, formed only where the
    resolved physics prices compute per synaptic operation."""

    ROWS = ProgramPasses(rows=(
        PassRow(segment_index=0, resident=False, core_cells=(30, 10), marks=frozenset(), core_extents=None,
                compute_steps=4),
        PassRow(segment_index=1, resident=False, core_cells=(20, 5), marks=frozenset(), core_extents=None,
                compute_steps=4, producer_cells=25, producer_cores=frozenset({1}), producer_core_cells=(0, 25)),
    ))
    PROGRAMMING = type("Programming", (), {"segment_cores": 4, "committed_cells": 90, "reprogrammed_cores": 4,
                                           "reprogrammed_bytes": None, "reprogram_passes": 2})()

    def _context(self, physics, **declared):
        pcfg = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 4}], "simulation_steps": 4,
                "activity_factor": 0.5, "weight_bits": 4, "platform_physics_resolved": physics.to_dict(), **declared}
        return candidate_context_from_platform(pcfg, passes=self.ROWS, programming=self.PROGRAMMING)

    def _quantities(self, context):
        return from_candidate(layout=None, chip_param_capacity=None, total_params=None, host_side_segment_count=None,
                              context=context)

    def test_each_row_reads_its_busiest_tile_with_its_operand_cells(self):
        context = self._context(_odin())
        assert context.pass_core_events == pytest.approx((30 * 4 * 0.5, 30 * 4 * 0.5))
        quantities = self._quantities(context)
        assert quantities.get("synaptic_events").value == pytest.approx(90 * 4 * 0.5)
        assert quantities.get(BUSIEST_EVENTS).value == pytest.approx(2 * 30 * 4 * 0.5)
        assert quantities.get(BUSIEST_EVENTS).provenance == "modeled"

    def test_the_write_law_receives_the_same_per_pass_events(self):
        context = self._context(_odin(), inference_batch_K=4)
        assert context.writes is not None and context.writes.pass_sops == context.pass_core_events
        assert WriteInputs.of(context.writes).pass_sops == context.pass_core_events

    @pytest.mark.parametrize("name", ["truenorth", "loihi", "generic_estimated_22nm"])
    def test_a_chip_of_the_step_law_states_no_per_core_census(self, name):
        context = self._context(get_platform_physics(name), inference_batch_K=4)
        assert context.pass_core_events is None
        assert context.writes is None or context.writes.pass_sops is None
        assert not self._quantities(context).has(BUSIEST_EVENTS)

    def test_without_a_declared_activity_there_is_no_per_core_census(self):
        context = self._context(_odin(), activity_factor=0.0)
        assert context.pass_core_events is None
        assert not self._quantities(context).has(BUSIEST_EVENTS)


class TestTheCorrelationCases:
    @pytest.mark.parametrize("name", ["odin_accelerated_time", "odin_biological_time"])
    def test_no_operation_count_stands_where_timesteps_are_meant(self, name):
        case = get_case(name)
        assert "latency_steps" not in case.census and "timesteps" not in case.census
        assert case.census["synaptic_events"] == case.operating_conditions["r_sop"] * case.operating_conditions[
            "window_s"]

    def test_the_biological_point_restates_the_operation_wall(self):
        case = get_case("odin_biological_time")
        assert "t_cycle" not in case.overrides
        clock_hz = case.operating_conditions["f_clk_mhz"] * 1e6
        assert case.overrides["t_synaptic_op"] == pytest.approx(2 / clock_hz * 1e9, rel=1e-6)  # ns, two cycles

    @pytest.mark.parametrize("name", ["odin_accelerated_time", "odin_biological_time"])
    def test_the_window_is_one_second_of_operations(self, name):
        result = correlate(get_case(name))
        assert result.passed


class TestTheWizardReadout:
    def test_the_latency_axes_read_either_compute_law(self):
        for axis in ("e2e_latency_s", "throughput_inferences_s"):
            assert physics_panel._AXIS_HEAD_CONSTANT[axis] == ("t_cycle", "t_synaptic_op")

    def test_a_declared_alternative_is_never_named_missing(self):
        assert physics_panel._first_missing(_odin(), "e2e_latency_s") == "a quantity this run does not produce"
        assert physics_panel._first_missing(None, "e2e_latency_s") == "t_cycle"
