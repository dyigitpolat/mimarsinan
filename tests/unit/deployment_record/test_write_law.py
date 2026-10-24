"""CH-A1: one write law — write energy per inference at K, units and ports, lifetime and its floor."""

from types import SimpleNamespace

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.program_census import candidate_program_passes
from mimarsinan.deployment_record.cost.absolute import (
    absolute_pricing_for_record,
    price_absolute,
    price_write_terms,
)
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_inputs
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.objectives.views import CandidateStaticView
from mimarsinan.deployment_record.platform_physics import (
    PlatformPhysics,
    get_platform_physics,
    profile_from_dict,
)
from mimarsinan.deployment_record.quantities.write_law import (
    WriteLawError,
    WritePolicy,
    write_policy_of,
    write_quantities,
)
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

from unit.deployment_record.g3_fixtures import (
    HandPassProgram,
    constant,
    quantities,
    record_with,
    schedule,
    segment,
)


def _pass(position, *, cells=0, producer=0, cores=1, kept=0, payload=None, steps=0, segment=0):
    return PassCensus(position=position, segment_index=segment, resident=False, cores=cores, resident_cores=kept,
                      written_cells_backing_store=cells, written_cells_producer=producer,
                      payload_bytes=cells if payload is None else payload, compute_steps=steps)


def _physics(**constants):
    return profile_from_dict({
        "format_version": 1, "name": "w", "display_name": "W", "description_file": "w.md",
        "validity": {"measurement_kind": "projection", "technology_node_nm": 22.0},
        "constants": constants,
    })


_WRITE = dict(e_write_per_cell=constant(2.0, unit="pJ"),
              write_verify_iterations=constant(3.0, unit="iterations"))


def _terms(pricing):
    return {t.name: t for t in pricing.terms}


def _refusals(pricing):
    return {r.name: r.reason for r in pricing.refusals}


class TestWriteEnergy:
    @pytest.mark.parametrize("k", [1, 16, 64])
    def test_backing_store_writes_are_cells_times_verify_over_k(self, k):
        census = PassPeriodCensus(passes=(_pass(0, cells=1000), _pass(1, cells=500)))
        pricing = price_write_terms(census, _physics(**_WRITE), WritePolicy(batch_k=k))
        expected_mj = 2e-12 * 1500 * 3 / k * 1e3
        assert _terms(pricing)["energy_write_mj"].value == pytest.approx(expected_mj, rel=1e-12)

    @pytest.mark.parametrize("k", [1, 64])
    def test_producer_writes_are_priced_per_inference_whatever_k(self, k):
        census = PassPeriodCensus(passes=(_pass(0, producer=400, payload=0),))
        pricing = price_write_terms(census, _physics(**_WRITE), WritePolicy(batch_k=k))
        assert _terms(pricing)["energy_write_mj"].value == pytest.approx(2e-12 * 400 * 3 * 1e3, rel=1e-12)

    def test_the_backing_store_byte_channel_moves_the_rewritten_payload(self):
        physics = _physics(**_WRITE, e_dma_per_byte=constant(10.0, unit="pJ"))
        census = PassPeriodCensus(passes=(_pass(0, cells=100, payload=100),))
        pricing = price_write_terms(census, physics, WritePolicy(batch_k=4))
        assert _terms(pricing)["energy_write_mj"].value == pytest.approx(
            (10e-12 * 100 / 4 + 2e-12 * 100 * 3 / 4) * 1e3, rel=1e-12)

    def test_the_verify_read_joins_every_write_pulse(self):
        physics = _physics(**_WRITE, e_verify_read_per_cell=constant(0.5, unit="pJ"))
        census = PassPeriodCensus(passes=(_pass(0, cells=100, payload=0),))
        pricing = price_write_terms(census, physics, WritePolicy(batch_k=4))
        assert _terms(pricing)["energy_write_mj"].value == pytest.approx(
            (2e-12 + 0.5e-12) * 3 * 100 / 4 * 1e3, rel=1e-12)

    def test_without_a_commit_energy_the_write_refuses_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        pricing = price_write_terms(census, _physics(), WritePolicy(batch_k=4))
        assert "e_write_per_cell" in _refusals(pricing)["energy_write_mj"]

    def test_a_cell_write_without_verify_iterations_refuses_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        pricing = price_write_terms(census, _physics(e_write_per_cell=constant(2.0, unit="pJ")),
                                    WritePolicy(batch_k=4))
        assert "write_verify_iterations" in _refusals(pricing)["energy_write_mj"]


class TestUnitsAndPorts:
    def test_row_granularity_divides_write_steps_by_units_and_ports(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=600, cores=2), _pass(1, cells=200, cores=1)))
        extents = (((10, 40), (20, 10)), ((5, 40),))       # (rows, columns) of each rewritten core
        policy = WritePolicy(batch_k=2, write_granularity="row", write_units_in_parallel=4, write_ports=2)
        values = write_quantities(census, policy, written_extents=extents)
        assert values["written_units"].value == pytest.approx((30 + 5) / 2)
        # Per pass ceil(rows / (units x ports)): ceil(30 / 8) + ceil(5 / 8) = 4 + 1, per period over K.
        assert values["write_events"].value == pytest.approx(5 / 2)

    @pytest.mark.parametrize("granularity, units", [("cell", 800), ("core", 3), ("column", 90)])
    def test_every_granularity_counts_its_units(self, granularity, units):
        census = PassPeriodCensus(passes=(_pass(0, cells=600, cores=2), _pass(1, cells=200, cores=1)))
        extents = (((10, 40), (20, 10)), ((5, 40),))
        values = write_quantities(census, WritePolicy(batch_k=1, write_granularity=granularity),
                                  written_extents=extents)
        assert values["written_units"].value == units

    def test_producer_writes_add_their_own_write_steps_per_inference(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=40, cores=1), _pass(1, producer=9, payload=0)))
        policy = WritePolicy(batch_k=4, write_granularity="cell", write_units_in_parallel=2, write_ports=2)
        values = write_quantities(census, policy, most_written=1)
        # Backing store ceil(40 / 4) = 10 steps per period over K = 4; producer ceil(9 / 4) = 3 per inference.
        assert values["write_events"].value == pytest.approx(10 / 4 + 3)
        assert values["written_units"].value == pytest.approx(40 / 4 + 9)

    def test_producer_wear_on_the_most_written_core_adds_its_writes_per_inference(self):
        """[N2 fix round, a named change] The wear quantity is the slot rule's: a producer region placed on the
        most-written core (2 static writes per period, 1 producer write per inference) wears it 2 / K + 1; with
        the producer cells on cores the census does not place the wear has no upper end, so it is not stated
        (it was the band's high end, M / K + 1, which assumed one producer write per inference)."""
        census = PassPeriodCensus(passes=(_pass(0, cells=40, cores=1), _pass(1, producer=9, payload=0)))
        placed = write_quantities(census, WritePolicy(batch_k=1), most_written=2, producer_slots=((2, 1),))
        assert placed["max_writes_per_cell_per_inference"].value == 3.0
        unplaced = write_quantities(census, WritePolicy(batch_k=1), most_written=2)
        assert "max_writes_per_cell_per_inference" not in unplaced

    def test_an_undeclared_granularity_leaves_the_units_absent(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=600),))
        values = write_quantities(census, WritePolicy(batch_k=1))
        assert "written_units" not in values and "write_events" not in values
        assert values["runtime_written_cells"].value == 600

    def test_core_types_that_disagree_on_granularity_are_named(self):
        policy = write_policy_of({"inference_batch_K": 4, "cores": [
            {"write_granularity": "row"}, {"write_granularity": "cell"}]})
        assert policy is not None and "write_granularity" in policy.mixed_core_writes

    def test_the_policy_is_absent_without_a_batch(self):
        assert write_policy_of({"write_ports": 2}) is None

    def test_an_unknown_overlap_mode_refuses_by_name(self):
        with pytest.raises(WriteLawError) as info:
            write_policy_of({"inference_batch_K": 1, "write_overlap": {"mode": "diagonal"}})
        assert isinstance(info.value, ToolchainRefusal)


def _lifetime_physics(**extra):
    return _physics(**_WRITE, t_cycle=constant(1.0, unit="us"), e_mac=constant(1.0, unit="pJ"),
                    endurance_cycles={"low": 1e6, "nominal": 1e7, "high": 1e9, "unit": "cycles",
                                      "evidence_kind": "published", "citation": "test p.1",
                                      "note": "secondary test band"}, **extra)


class TestLifetime:
    def test_lifetime_is_endurance_over_the_most_written_cell(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        pricing = price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=10), most_written=1)
        life = _terms(pricing)["lifetime_inferences"]
        assert (life.band.low, life.value, life.band.high) == pytest.approx((1e7, 1e8, 1e10))

    @pytest.mark.parametrize("k", [1, 16])
    def test_a_core_every_pass_rewrites_wears_once_per_pass(self, k):
        """t0_53-class: three passes on three cores write the first core three times per period."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=3), _pass(1, cells=30, cores=3),
                                          _pass(2, cells=10, cores=1)))
        pricing = price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=k), most_written=3)
        assert _terms(pricing)["lifetime_inferences"].value == pytest.approx(1e7 * k / 3)
        assert "3 writes per period" in _terms(pricing)["lifetime_inferences"].source

    def test_producer_wear_does_not_amortize_over_k(self):
        census = PassPeriodCensus(passes=(_pass(0, producer=50, payload=0),))
        lives = [_terms(price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=k),
                                          most_written=0))["lifetime_inferences"].value for k in (1, 64)]
        assert lives == pytest.approx([1e7, 1e7])

    def test_a_census_without_a_core_assignment_prices_an_upper_bound_only(self):
        """Seven core-writes on three cores: the most-written core takes at least ceil(7 / 3) = 3."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=3), _pass(1, cells=30, cores=3),
                                          _pass(2, cells=10, cores=1)))
        policy = WritePolicy(batch_k=1, physical_cores=3, inference_rate_hz=1.0)
        life = _terms(price_write_terms(census, _lifetime_physics(), policy))["lifetime_inferences"]
        assert life.value == pytest.approx(1e7 / 3)
        assert "upper bound" in life.source
        low_floor = WritePolicy(batch_k=1, physical_cores=3, inference_rate_hz=1.0, min_lifetime_hours=1.0)
        judged = price_write_terms(census, _lifetime_physics(), low_floor)
        assert not judged.verdicts and "upper bound" in _refusals(judged)["lifetime_admissible"]
        high_floor = WritePolicy(batch_k=1, physical_cores=3, inference_rate_hz=1.0, min_lifetime_hours=1e9)
        assert [v.verdict for v in price_write_terms(census, _lifetime_physics(), high_floor).verdicts] == \
            ["inadmissible"]

    def test_a_bounded_band_across_the_floor_is_refused_never_judged(self):
        """The pigeonhole bound proves only inadmissibility: a lifetime band straddling the floor refuses."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=3),))
        # Wear >= ceil(3 / 3) = 1 per inference at 1/s: the bound band is 278 h to 2.8e5 h.
        policy = WritePolicy(batch_k=1, physical_cores=3, inference_rate_hz=1.0, min_lifetime_hours=1000.0)
        judged = price_write_terms(census, _lifetime_physics(), policy)
        assert not judged.verdicts
        assert "upper bound" in _refusals(judged)["lifetime_admissible"]

    def test_the_bound_counts_producer_wear_per_inference(self):
        """Three core-writes on three cores at K = 16 bound the wear at 1/16 per inference; a producer
        writes its cells once per inference, so the most-written cell takes at least one write."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=3), _pass(1, producer=5, payload=0)))
        policy = WritePolicy(batch_k=16, physical_cores=3)
        life = _terms(price_write_terms(census, _lifetime_physics(), policy))["lifetime_inferences"]
        assert life.value == pytest.approx(1e7)
        assert "upper bound" in life.source

    def test_producer_cells_on_an_unstated_core_bound_the_wear_from_below(self):
        """[N2 fix round, a named change] The most-written core takes 2 writes per period at K = 4 (0.5 per
        inference); the producer's cells sit on cores the census does not place, so the most-written cell takes
        at least max(0.5, 1) writes per inference and the lifetime is an upper bound. The pre-N2 band capped the
        wear at 0.5 + 1, one producer write per inference, which the slot rule refutes (operand regions of
        several passes on one core take one write per inference each)."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=2), _pass(1, producer=5, payload=0)))
        life = _terms(price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=4),
                                        most_written=2))["lifetime_inferences"]
        assert (life.band.low, life.value, life.band.high) == pytest.approx((1e6, 1e7, 1e9))
        assert "producer" in life.source and "at least" in life.source

    @pytest.mark.parametrize("slots, wear", [(((2, 1),), 1.5), (((0, 1),), 1.0), (((0, 3),), 3.0)])
    def test_producer_cells_placed_by_the_slot_rule_make_the_wear_exact(self, slots, wear):
        """The region on the most-written core (2 / 4 + 1), on a core of its own (1), or three operand passes
        rotated onto one core (3): the wear is exact, max over cores of static / K + producer."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=2), _pass(1, producer=5, payload=0)))
        life = _terms(price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=4), most_written=2,
                                        producer_slots=slots))["lifetime_inferences"]
        assert (life.band.low, life.value, life.band.high) == pytest.approx((1e6 / wear, 1e7 / wear, 1e9 / wear))

    @pytest.mark.parametrize("floor, verdict", [(1e9 / 0.75 / 3600, "inadmissible"),
                                                (1e9 / 1.25 / 3600, None),
                                                (1e6 / 1.75 / 3600, None)])
    def test_an_unplaced_producer_wear_is_judged_at_its_sound_end(self, floor, verdict):
        """Wear at least 1: inadmissible when even the longest lifetime (endurance / 1) fails the floor; any
        other floor is refused by name, since an upper-bound lifetime never admits (it straddled or admitted
        on the pre-N2 band)."""
        census = PassPeriodCensus(passes=(_pass(0, cells=30, cores=2), _pass(1, producer=5, payload=0)))
        policy = WritePolicy(batch_k=4, inference_rate_hz=1.0, min_lifetime_hours=floor)
        judged = price_write_terms(census, _lifetime_physics(), policy, most_written=2)
        if verdict is None:
            assert not judged.verdicts and "upper bound" in _refusals(judged)["lifetime_admissible"]
        else:
            assert [v.verdict for v in judged.verdicts] == [verdict]

    def test_without_a_core_assignment_or_a_core_count_the_lifetime_refuses_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        refused = _refusals(price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=1)))
        assert "physical core" in refused["lifetime_inferences"]

    def test_a_band_across_the_floor_straddles(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        # One write per inference at 1/s: 1e6 / 3600 = 278 h low, 2778 h nominal, 2.8e5 h high.
        policy = WritePolicy(batch_k=1, inference_rate_hz=1.0, min_lifetime_hours=1000.0)
        verdict, = price_write_terms(census, _lifetime_physics(), policy, most_written=1).verdicts
        assert verdict.verdict == "straddles"

    def test_an_undeclared_endurance_refuses_the_lifetime_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        pricing = price_write_terms(census, _physics(**_WRITE), WritePolicy(batch_k=10))
        assert "endurance_cycles" in _refusals(pricing)["lifetime_inferences"]

    def test_nothing_written_wears_nothing(self):
        census = PassPeriodCensus(passes=(_pass(0),))
        pricing = price_write_terms(census, _lifetime_physics(),
                                    WritePolicy(batch_k=1, inference_rate_hz=30.0, min_lifetime_hours=1.0),
                                    most_written=0)
        assert "lifetime_inferences" not in _terms(pricing)
        assert pricing.verdicts[0].verdict == "admitted"

    def test_a_floor_without_a_rate_refuses_by_name(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100),))
        pricing = price_write_terms(census, _lifetime_physics(), WritePolicy(batch_k=1, min_lifetime_hours=10.0),
                                    most_written=1)
        assert "inference_rate_hz" in _refusals(pricing)["lifetime_admissible"]

    def test_below_the_floor_the_program_is_inadmissible_and_keeps_every_priced_term(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100, steps=3), _pass(1, cells=100, steps=3)))
        values = quantities(latency_steps=6, synaptic_events=1000, host_macs=0, cells_physical=1)
        floor = WritePolicy(batch_k=1, inference_rate_hz=30.0, min_lifetime_hours=1e9)
        free = WritePolicy(batch_k=1, inference_rate_hz=30.0)
        physics = _lifetime_physics(t_program_per_byte=constant(1.0, unit="ns"))
        judged = price_absolute(values, physics, writes=WriteInputs(census, floor, most_written=2))
        priced = price_absolute(values, physics, writes=WriteInputs(census, free, most_written=2))
        assert [v.verdict for v in judged.verdicts] == ["inadmissible"]
        assert "endurance_cycles" in judged.verdicts[0].evidence
        assert [t.to_dict() for t in judged.terms] == [t.to_dict() for t in priced.terms]
        assert {t.name for t in judged.terms} >= {"e2e_latency_s", "energy_per_inference_mj", "lifetime_hours"}

    def test_the_headlines_carry_exactly_the_write_terms(self):
        census = PassPeriodCensus(passes=(_pass(0, cells=100, steps=3), _pass(1, cells=200, steps=3)))
        values = quantities(latency_steps=6, synaptic_events=1000, host_macs=0, cells_physical=1)
        physics = _lifetime_physics(t_program_per_byte=constant(1.0, unit="ns"))
        with_k = {t.name: t for t in price_absolute(
            values, physics, writes=WriteInputs(census, WritePolicy(batch_k=4), most_written=2)).terms}
        without = {t.name: t for t in price_absolute(values, physics).terms}
        for headline, write in (("e2e_latency_s", "latency_write_exposed_s"),
                                ("energy_per_inference_mj", "energy_write_mj")):
            assert with_k[write].value > 0
            for corner in ("low", "nominal", "high"):
                assert getattr(with_k[headline].band, corner) == pytest.approx(
                    getattr(without[headline].band, corner) + getattr(with_k[write].band, corner), rel=1e-12)


def _multi_pass_census():
    """A hand two-pass program on a two-core chip, rewritten every inference (K = 1): each core twice."""
    return PassPeriodCensus(passes=(_pass(0, cells=4096, cores=2), _pass(1, cells=4096, cores=2)))


class TestTechnologyVerdicts:
    """At 30 inferences/s and a 5-year floor, RRAM wears out across its band and SRAM does not."""

    _POLICY = WritePolicy(batch_k=1, inference_rate_hz=30.0, min_lifetime_hours=43_800.0)

    def test_the_isaac_like_row_is_inadmissible_across_its_endurance_band(self):
        pricing = price_write_terms(_multi_pass_census(), get_platform_physics("isaac_like"), self._POLICY,
                                    most_written=2)
        verdict, = pricing.verdicts
        assert verdict.verdict == "inadmissible"
        assert verdict.evidence.startswith("endurance_cycles [published:")
        assert verdict.evidence.endswith("; a secondary source]")
        assert get_platform_physics("isaac_like").constants["endurance_cycles"].evidence_kind == "published"

    def test_the_generic_22nm_row_is_admitted(self):
        pricing = price_write_terms(_multi_pass_census(), get_platform_physics("generic_estimated_22nm"),
                                    self._POLICY, most_written=2)
        verdict, = pricing.verdicts
        assert verdict.verdict == "admitted"
        assert "ielmini2018inmemory" in verdict.evidence and "secondary" not in verdict.evidence


def _resolved(**declared):
    return build_platform_constraints_resolved({
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 2, "write_granularity": "cell",
                   "write_units_in_parallel": 2}],
        "weight_bits": 8, "platform_physics_profile": "generic_estimated_22nm", **declared})


class TestPolicyKeysOnBothPlanes:
    _DECLARED = dict(inference_batch_K=16, write_overlap={"mode": "serial", "shadow_capacity_fraction": 0.0},
                     write_ports=2, inference_rate_hz=30.0, min_lifetime_hours=43_800.0)

    def test_the_resolved_surface_carries_the_policy_to_both_planes(self):
        pcfg = _resolved(**self._DECLARED)
        policy = write_policy_of(pcfg)
        assert (policy.batch_k, policy.write_ports, policy.write_granularity, policy.write_units_in_parallel,
                policy.inference_rate_hz, policy.min_lifetime_hours) == (16, 2, "cell", 2, 30.0, 43_800.0)
        assert (policy.core_capacity, policy.physical_cores) == ((((256, 256), 2),), 2)
        assert write_policy_of(_resolved()) is None

    def test_candidate_equals_record_on_the_write_terms_of_a_capacity_split(self):
        pcfg = _resolved(**self._DECLARED)
        softcores = [SimpleNamespace(segment_id=0, latency_tag=i, output_count=4) for i in range(2)]
        noc = SimpleNamespace(pass_placements=(((0, 0), (0, 1)), ((1, 0),)),
                              pass_programs=(HandPassProgram(core_cells=(100, 50)), HandPassProgram(core_cells=(30,))),
                              census=None)
        semantics = SimpleNamespace(retimed=False, is_cycle=False, is_cascade=False, latency_group_count=None,
                                    chip_latency=None)
        context = candidate_context_from_platform(
            pcfg, host_macs=0, passes=candidate_program_passes(softcores, noc, semantics=semantics, timesteps=32))
        view = CandidateStaticView(layout=None, chip_param_capacity=None, total_params=None,
                                   host_side_segment_count=None,
                                   physics=PlatformPhysics.from_dict(pcfg["platform_physics_resolved"]),
                                   quantity_context=context)
        report = view.cost_report()
        candidate = {t.name: t for t in report.energy + report.latency + report.lifetime}
        record = record_with(schedule((segment(1, 0, 0, [100, 50]), segment(2, 0, 1, [30]))), **pcfg)
        sealed = {t.name: t for t in absolute_pricing_for_record(record).terms}
        for name in ("energy_write_mj", "latency_write_exposed_s", "lifetime_inferences", "lifetime_hours"):
            assert candidate[name].to_dict() == sealed[name].to_dict(), name
        # Core 0 is written by both passes: 2 writes per period at K = 16.
        assert sealed["lifetime_inferences"].value == pytest.approx(
            get_platform_physics("generic_estimated_22nm").band("endurance_cycles").nominal * 16 / 2)


class TestWearAndRefusalCoverage:
    """Targeted pass: asserts that kill the reviewer's surviving write-law mutants."""

    def test_producer_cells_with_no_core_rewrite_wear_at_least_once_per_inference(self):
        """No core is rewritten per period and no most-written core is stated, but a producer writes its
        cells every inference: the wear is at least 1, never 'no cell is written'. [N2 fix round, a named
        change] Unplaced, it has no upper end (it was read as exactly 1), so the floor is refused by name;
        placed on its own core by the slot rule it is exactly 1 and admitted."""
        census = PassPeriodCensus(passes=(_pass(0, producer=50, payload=0),))
        policy = WritePolicy(batch_k=8, inference_rate_hz=1.0, min_lifetime_hours=1.0)
        pricing = price_write_terms(census, _lifetime_physics(), policy)
        assert _terms(pricing)["lifetime_inferences"].value == pytest.approx(1e7)
        assert not pricing.verdicts and "upper bound" in _refusals(pricing)["lifetime_admissible"]
        assert "no cell is written" not in _refusals(pricing)["lifetime_admissible"]
        placed = price_write_terms(census, _lifetime_physics(), policy, most_written=0, producer_slots=((0, 1),))
        assert _terms(placed)["lifetime_inferences"].value == pytest.approx(1e7)
        verdict, = placed.verdicts
        assert verdict.verdict == "admitted" and verdict.evidence != "no wear"
        assert "no cell is written" not in verdict.reason

    @pytest.mark.parametrize("floor", [None, 100.0])
    def test_a_plane_without_a_census_refuses_the_verdict_only_under_a_floor(self, floor):
        policy = WritePolicy(batch_k=1, inference_rate_hz=1.0, min_lifetime_hours=floor)
        pricing = price_write_inputs(WriteInputs(census=None, policy=policy, unavailable="no census (test)"),
                                     _lifetime_physics())
        names = {r.name for r in pricing.refusals}
        expected = {"energy_write_mj", "latency_write_exposed_s", "lifetime_inferences"}
        assert names == (expected if floor is None else expected | {"lifetime_admissible"})
        assert {r.reason for r in pricing.refusals} == {"no census (test)"}
