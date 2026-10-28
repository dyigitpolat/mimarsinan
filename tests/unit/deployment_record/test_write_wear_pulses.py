"""[U-C C-1] The write law counts wear in programming pulses: a write technology declares ``pulses_per_write``, the
programming pulses one cell write applies to its cell, and the lifetime divides the endurance (counted in set and
reset cycles) by the most-written cell's writes per inference times its pulses per write. A technology that declares
no pulses per write is priced as one pulse per write, byte for byte as before. Every reader of the endurance band
(the write law's lifetime and floor verdict, the ladder's closed forms, the design point's wear check) reads the
same rule."""

import pytest

from mimarsinan.config_schema.registry.decision_classes import (
    ADMISSIBILITY_BEARING,
    COST_ONLY,
    DECISION_CLASSES,
)
from mimarsinan.deployment_record.cost.absolute import price_write_terms
from mimarsinan.deployment_record.cost.write_lifetime import endurance_in_writes
from mimarsinan.deployment_record.design_point.endurance import endurance_of
from mimarsinan.deployment_record.design_point.wear import hours_of, wear_problems
from mimarsinan.deployment_record.introspection import scaling_ladder_law as law
from mimarsinan.deployment_record.introspection.scaling_ladder_reading import min_batch
from mimarsinan.deployment_record.platform_physics import profile_from_dict, vocabulary
from mimarsinan.deployment_record.platform_physics.constants import spec_for
from mimarsinan.deployment_record.platform_physics.resolve import WRITE_TECHNOLOGY_KEYS, with_write_technology
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import constant

ENDURANCE = {"low": 1e6, "nominal": 1e7, "high": 1e9, "unit": "cycles", "evidence_kind": "published",
             "citation": "test p.1"}
PULSES = {"low": 2.0, "nominal": 4.0, "high": 8.0, "unit": "pulses", "evidence_kind": "derived",
          "derivation": "test: a write-verify of 2 to 8 pulses"}


def _physics(**constants):
    return profile_from_dict({
        "format_version": 1, "name": "w", "display_name": "W", "description_file": "w.md",
        "validity": {"measurement_kind": "projection", "technology_node_nm": 22.0},
        "constants": {"e_write_per_cell": constant(2.0, unit="pJ"),
                      "write_verify_iterations": constant(3.0, unit="iterations"),
                      "t_write_per_unit": constant(1.0, unit="us"), "t_cycle": constant(1.0, unit="us"),
                      "endurance_cycles": ENDURANCE, **constants},
    })


def _census():
    return PassPeriodCensus(passes=tuple(
        PassCensus(position=i, segment_index=0, resident=False, cores=1, resident_cores=0,
                   written_cells_backing_store=100, written_cells_producer=0, payload_bytes=100, compute_steps=1)
        for i in (0, 1)))


def _priced(physics, **policy):
    return price_write_terms(_census(), physics, WritePolicy(batch_k=4, write_granularity="cell", **policy),
                             most_written=2)


def _terms(pricing):
    return {t.name: t for t in pricing.terms}


class TestVocabulary:
    def test_pulses_per_write_is_a_declarable_dimensionless_programming_row(self):
        spec = spec_for("pulses_per_write")
        assert (spec.group, spec.dimension, spec.display_unit, spec.multiplicand) == (
            vocabulary.PROGRAMMING, "dimensionless", "pulses", "declared, not multiplied")
        assert "endurance_cycles" in spec.doc and "pulse" in spec.doc

    def test_it_is_a_write_technology_row(self):
        """A core type's own write physics declares its pulses per write beside its endurance."""
        assert "pulses_per_write" in WRITE_TECHNOLOGY_KEYS
        chip = _physics(pulses_per_write={**PULSES, "low": 1.0, "nominal": 1.0, "high": 1.0})
        typed = with_write_technology(chip, _physics(pulses_per_write=PULSES))
        assert typed.constants["pulses_per_write"].nominal == 4.0


class TestDecisionClass:
    """The constant is registered with its decision class: the lifetime judged against ``min_lifetime_hours`` is its
    only reader, so a declaration of it moves only which programs are admitted (the class rule's "a capacity the
    built program is judged against without being shaped by it"); every priced constant is cost-only."""

    def test_the_lifetime_constants_are_admissibility_bearing(self):
        for key in ("pulses_per_write", "endurance_cycles"):
            cls, reason = vocabulary.decision_class_of(key)
            assert cls == ADMISSIBILITY_BEARING and "min_lifetime_hours" in reason

    def test_every_other_row_is_cost_only_and_every_class_is_a_registry_class(self):
        for key, *_ in vocabulary.ROWS:
            cls, reason = vocabulary.decision_class_of(key)
            assert cls in DECISION_CLASSES and reason
            if key not in ("pulses_per_write", "endurance_cycles"):
                assert cls == COST_ONLY, key

    def test_an_unknown_constant_has_no_class(self):
        with pytest.raises(KeyError):
            vocabulary.decision_class_of("pulses")

    def test_pulses_move_no_priced_energy_or_time(self):
        """The evidence for the class: the write energy and the exposed write time are equal with and without the
        declaration; only the lifetime moves."""
        plain, pulsed = _priced(_physics()), _priced(_physics(pulses_per_write=PULSES))
        for name in ("energy_write_mj", "latency_write_exposed_s"):
            assert _terms(plain)[name].band == _terms(pulsed)[name].band
        assert _terms(plain)["lifetime_inferences"].band != _terms(pulsed)["lifetime_inferences"].band


class TestLifetimeInPulses:
    def test_lifetime_is_endurance_over_writes_times_pulses_corner_by_corner(self):
        """Wear 2 writes per period over K = 4 is 0.5 writes per inference; the shortest life pairs the low endurance
        with the most pulses."""
        life = _terms(_priced(_physics(pulses_per_write=PULSES)))["lifetime_inferences"]
        assert (life.band.low, life.band.nominal, life.band.high) == pytest.approx(
            (1e6 / (0.5 * 8), 1e7 / (0.5 * 4), 1e9 / (0.5 * 2)))
        assert "pulses_per_write" in life.source and "endurance_cycles" in life.source

    def test_without_pulses_the_lifetime_and_its_basis_are_unchanged(self):
        life = _terms(_priced(_physics()))["lifetime_inferences"]
        assert (life.band.low, life.band.nominal, life.band.high) == pytest.approx((2e6, 2e7, 2e9))
        assert life.source.startswith("endurance_cycles / max_writes_per_cell_per_inference (")
        assert "pulses" not in life.source

    def test_one_pulse_per_write_prices_the_same_lifetime(self):
        one = {**PULSES, "low": 1.0, "nominal": 1.0, "high": 1.0}
        pulsed = _terms(_priced(_physics(pulses_per_write=one)))["lifetime_inferences"].band
        plain = _terms(_priced(_physics()))["lifetime_inferences"].band
        assert (pulsed.low, pulsed.nominal, pulsed.high) == (plain.low, plain.nominal, plain.high)

    def test_the_floor_verdict_reads_the_pulse_counted_lifetime(self):
        """At 1 inference/s the plain lifetime is 556-556k h; a floor of 400 h is admitted on writes and straddles on
        pulses (the low corner falls to 69 h), naming the pulses in its evidence."""
        floor = dict(inference_rate_hz=1.0, min_lifetime_hours=400.0)
        plain = _priced(_physics(), **floor).verdicts[0]
        pulsed = _priced(_physics(pulses_per_write=PULSES), **floor).verdicts[0]
        assert (plain.verdict, pulsed.verdict) == ("admitted", "straddles")
        assert "pulses_per_write" in pulsed.evidence and "pulses_per_write" not in plain.evidence

    def test_endurance_in_writes_is_the_one_reading(self):
        band = endurance_in_writes(_physics(pulses_per_write=PULSES))
        assert (band.low, band.nominal, band.high) == pytest.approx((1e6 / 8, 1e7 / 4, 1e9 / 2))
        plain = endurance_in_writes(_physics())
        assert (plain.low, plain.nominal, plain.high) == (1e6, 1e7, 1e9)
        assert endurance_in_writes(profile_from_dict({
            "format_version": 1, "name": "e", "display_name": "E", "description_file": "e.md",
            "validity": {"measurement_kind": "projection"}, "constants": {}})) is None


class TestClosedForms:
    """The ladder's K* and two-ceiling minimum batch divide the endurance by the pulses per write too."""

    def test_k_star_reads_writes_times_pulses(self):
        kwargs = dict(x_fluid=3.0, x_packed=4.0, max_writes=2.0, max_writes_source="t", rate_hz=30.0,
                      lifetime_hours=10.0)
        plain = law.k_star(endurance=1e9, **kwargs)
        pulsed = law.k_star(endurance=1e9, pulses_per_write=8.0, **kwargs)
        assert pulsed["k_star_census"] == pytest.approx(8.0 * plain["k_star_census"])
        assert pulsed["k_star_closed"] == pytest.approx(8.0 * plain["k_star_closed"])
        assert pulsed["pulses_per_write"] == 8.0 and "pulses_per_write" not in plain

    def test_min_batch_pairs_the_low_endurance_with_the_most_pulses(self):
        kwargs = dict(x_fluid=3.0, x_packed=4.0, max_writes=2.0, rate_hz=30.0, lifetime_hours=10.0,
                      endurance=(1e6, 1e7, 1e9), write_time_per_period=(1e-3, 2e-3, 4e-3), write_time_source="t",
                      k=16)
        plain, pulsed = min_batch(**kwargs), min_batch(pulses_per_write=(2.0, 4.0, 8.0), **kwargs)
        assert pulsed["k_endurance_census"] == pytest.approx(
            [plain["k_endurance_census"][0] * 8, plain["k_endurance_census"][1] * 4, plain["k_endurance_census"][2] * 2])
        assert "pulses_per_write" not in plain and pulsed["pulses_per_write"] == [2.0, 4.0, 8.0]

    def test_the_crossover_pairs_each_write_time_corner_with_the_same_pulses_corner(self):
        """[targeted M1] The crossover endurance is in cycles: the writes crossover times the pulses. A write-verify
        technology's write time and its pulses per write come from one iteration count (a write of i iterations takes
        i verify rounds and applies a pulse per iteration), so write-time corner i reads pulses corner i, the law's
        corner convention for write-path constants."""
        kwargs = dict(x_fluid=3.0, x_packed=4.0, max_writes=2.0, rate_hz=30.0, lifetime_hours=10.0,
                      endurance=(1e6, 1e7, 1e9), write_time_per_period=(1e-3, 2e-3, 4e-3), write_time_source="t",
                      k=16)
        plain, pulsed = min_batch(**kwargs), min_batch(pulses_per_write=(2.0, 4.0, 8.0), **kwargs)
        for key in ("e_cross_census", "e_cross_closed_packed"):
            assert pulsed[key] == pytest.approx([plain[key][0] * 2, plain[key][1] * 4, plain[key][2] * 8]), key

    def test_the_binding_ceiling_agrees_with_the_crossover_at_every_corner(self):
        """Endurance binds at a paired corner exactly when that corner's endurance lies at or below its crossover:
        favourable reads the high endurance against the crossover at the low write time, nominal the nominal, adverse
        the low endurance against the high write time. Bands shaped like a write-verify chip's (pulses = 2 +
        iterations, write time growing with the iterations): the favourable crossover with the most pulses would lie
        above the high endurance although the write bandwidth binds there."""
        endurance, pulses, write_time = (1e8, 1e10, 1e12), (3.0, 7.5, 32.0), (2e-3, 5e-3, 33e-3)
        out = min_batch(x_fluid=3.0, x_packed=4.0, max_writes=1.0, rate_hz=30.0, lifetime_hours=43_800.0,
                        endurance=endurance, write_time_per_period=write_time, write_time_source="t", k=1,
                        pulses_per_write=pulses)
        for corner, e_index, t_index in (("favourable", 2, 0), ("nominal", 1, 1), ("adverse", 0, 2)):
            crossover = out["e_cross_census"][t_index]
            assert (out["binding"][corner] == "endurance") == (endurance[e_index] <= crossover), corner
        assert out["binding"]["favourable"] == "write bandwidth"
        assert out["e_cross_census"][0] == pytest.approx(43_800.0 * 3600.0 * 1.0 * 3.0 / 2e-3)


class TestLadderWiring:
    """[targeted M4] The ladder's priced program carries the technology's pulses per write beside its endurance band
    (``price_scheduled`` through ``endurance_fields``), and its row reads them: K* at the nominal pulses, the minimum
    batch's ceilings and crossovers at the band."""

    @staticmethod
    def _platform(**constants):
        from unit.deployment_record.introspection import test_scaling_ladder as ladder_test

        physics = profile_from_dict({
            "format_version": 1, "name": "ladder_wear", "display_name": "Ladder wear", "description_file": "t.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {**{k: {"nominal": v, "evidence_kind": "estimated", "note": "test"}
                             for k, v in ladder_test.CONSTANTS.items()}, "endurance_cycles": ENDURANCE, **constants},
        })
        return {**ladder_test._platform(1), "platform_physics_resolved": physics.to_dict()}

    @staticmethod
    def _row(platform):
        from unit.deployment_record.introspection import test_scaling_ladder as ladder_test

        return ladder_test._row(cores=1, platform=platform, inference_rate_hz=30.0, min_lifetime_hours=43_800.0)

    def test_the_priced_program_and_its_row_read_the_pulses(self):
        plain, pulsed = self._row(self._platform()), self._row(self._platform(pulses_per_write=PULSES))
        assert plain["k_star"]["k_star_census"] > 0
        assert pulsed["k_star"]["pulses_per_write"] == 4.0 and "pulses_per_write" not in plain["k_star"]
        assert pulsed["k_star"]["k_star_census"] == pytest.approx(4.0 * plain["k_star"]["k_star_census"])
        assert pulsed["min_batch"]["pulses_per_write"] == [2.0, 4.0, 8.0]
        assert pulsed["min_batch"]["k_endurance_census"] == pytest.approx(
            [a * f for a, f in zip(plain["min_batch"]["k_endurance_census"], (8.0, 4.0, 2.0))])

    def test_the_endurance_fields_carry_no_pulses_without_an_endurance(self):
        from mimarsinan.deployment_record.introspection.scaling_ladder_reading import endurance_fields

        assert endurance_fields(None) == {"endurance": None, "endurance_band": None, "pulses_band": None}
        assert endurance_fields(_physics(pulses_per_write=PULSES))["pulses_band"] == (2.0, 4.0, 8.0)


class TestDesignPointWear:
    def test_the_endurance_block_states_the_pulses(self):
        block = endurance_of(_physics(pulses_per_write=PULSES))
        assert block["cycles"] == [1e6, 1e7, 1e9] and block["pulses_per_write"] == [2.0, 4.0, 8.0]
        assert "pulses_per_write" not in endurance_of(_physics())

    def test_hours_divide_by_the_pulses_corner_by_corner(self):
        assert hours_of([3.6e6, 3.6e7, 3.6e9], 1.0, 1.0) == pytest.approx([1e3, 1e4, 1e6])
        assert hours_of([3.6e6, 3.6e7, 3.6e9], 1.0, 1.0, pulses=[2.0, 4.0, 8.0]) == pytest.approx(
            [1e3 / 8, 1e4 / 4, 1e6 / 2])

    def test_the_reader_recomputes_a_pulse_counted_entry(self):
        endurance = {"cycles": [3.6e6, 3.6e7, 3.6e9], "evidence_kind": "published", "source": "t",
                     "pulses_per_write": [2.0, 4.0, 8.0]}
        entry = {"role": "static", "write_technology": "w", "max_writes_per_cell_per_inference": 1.0,
                 "endurance": endurance, "hours": [125.0, 2500.0, 500000.0], "verdict": "straddles"}
        row = {"chip": {"core_types": [{"role": "static", "write_technology": "w"}]},
               "lifetime": {"per_core_type": [entry], "hours": entry["hours"], "verdict": "straddles",
                            "binding": "static (w)"}}
        declarations = {"inference_rate_hz": 1.0, "min_lifetime_hours": 1000.0}
        assert wear_problems(row, declarations) == []
        wrong = {**entry, "hours": [1e3, 1e4, 1e6], "verdict": "admitted"}
        row_wrong = {**row, "lifetime": {**row["lifetime"], "per_core_type": [wrong], "hours": wrong["hours"],
                                         "verdict": "admitted"}}
        assert any("not endurance / wear / rate" in p for p in wear_problems(row_wrong, declarations))
