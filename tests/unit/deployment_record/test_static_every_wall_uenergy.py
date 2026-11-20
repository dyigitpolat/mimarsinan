"""[U-ENERGY class 1] Static power integrates over every priced chip time, corner by corner, exactly once,
from ONE static sum resolved once per pricing: the per-inference latency (unchanged), the loads (the one load law),
both sides of Pi and M's period writes on the ladder, the measured model's whole latency, and the step price's exposed
write time. Every constant here is a test constant."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.step_choice import cheapest_step, price_step
from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC, StaticSum
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_terms
from mimarsinan.deployment_record.platform_physics import profile_from_dict
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

J_TO_MJ = 1e3
CORNERS = ("low", "nominal", "high")
BASE = {"e_mac": 0.5, "t_cycle": 100.0, "e_dma_per_byte": 2.0, "e_program_per_byte": 3.0,
        "e_core_program": 5.0, "t_program_per_byte": 7.0}
STATIC = {"p_static_per_core": 0.25, "p_static_global": 4.0}       # mW
BANDED_STATIC = {"p_static_per_core": {"low": 0.1, "nominal": 0.25, "high": 1.0},
                 "p_static_global": {"low": 1.0, "nominal": 4.0, "high": 9.0}}
CORES = 16
PLATFORM = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 8}], "weight_bits": 4}


def _physics(name: str, constants: dict, **extra):
    return profile_from_dict({
        "format_version": 1, "name": name, "display_name": name, "description_file": "t.md",
        "validity": {"measurement_kind": "projection"},
        "constants": {k: ({**v, "evidence_kind": "estimated", "note": "test"} if isinstance(v, dict) else
                          {"nominal": v, "evidence_kind": "estimated", "note": "test"})
                      for k, v in constants.items()},
        **extra,
    })


def _q(**extra) -> Quantities:
    values = {"synaptic_events": 1e6, "cores_physical": CORES, "latency_steps": 40.0, "host_macs": 0.0,
              "compute_op_count": 0.0, **extra}
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _loads():
    rows = tuple(PassRow(segment_index=i, resident=False, core_cells=cells, marks=frozenset(),
                         core_extents=tuple((c // 4, 4) for c in cells), compute_steps=3)
                 for i, cells in enumerate(((40, 20), (32,), (16, 24))))
    from mimarsinan.deployment_record.quantities.planes.loads import load_inputs  # a new name, imported where used
    return load_inputs(ProgramPasses(rows=rows), PLATFORM, weight_bits=4, label="test")


def _terms(physics, quantities, **kw):
    pricing = price_absolute(quantities, physics, **kw)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _watts(physics, cores=CORES):
    return [getattr(physics.band("p_static_per_core"), c) * cores + getattr(physics.band("p_static_global"), c)
            for c in CORNERS]


def _ends(band):
    return [band.low, band.nominal, band.high]


class TestTheHeadline:
    def test_the_per_inference_static_is_unchanged(self):
        physics = _physics("banded", {**BASE, **BANDED_STATIC})
        terms, _ = _terms(physics, _q())
        e2e = _ends(terms["e2e_latency_s"].band)
        assert _ends(terms["energy_static_mj"].band) == pytest.approx(
            [w * t * J_TO_MJ for w, t in zip(_watts(physics), e2e)], rel=1e-12)


class TestOneStaticSumResolvedOnce:
    """m-2: the static sum is computed once inside price_absolute from the resolved physics; a carry buffer whose
    static power its description states per capacity row leaks over the headline, the loads and the measured wall
    alike (P4's description: 11.2768 mW, never 8.0 mW)."""

    def _tabled(self):
        return _physics("p4", {"e_mac": 0.5, "t_cycle": 100.0, **BASE, **STATIC, "e_carry_buffer_per_byte": 1.0,
                               "area_carry_buffer_per_byte": 1.0},
                        carry_buffer={"rows": [{"bytes": 65536, "p_carry_buffer_static_per_byte":
                                                {"nominal": 50.0, "unit": "nW", "evidence_kind": "estimated",
                                                 "note": "test"}}]})

    def test_the_static_power_returned_is_the_headlines_and_holds_the_tabled_buffer(self):
        pricing = price_absolute(_q(carry_buffer_capacity_bytes=65536.0), self._tabled(), loads=_loads())
        terms = {t.name: t for t in pricing.terms}
        headline_w = terms["energy_static_mj"].value / J_TO_MJ / terms["e2e_latency_s"].value
        assert pricing.static_power.watts.nominal == pytest.approx(headline_w, rel=1e-12)
        assert pricing.static_power.watts.nominal == pytest.approx(0.25e-3 * CORES + 4e-3 + 50e-9 * 65536, rel=1e-12)

    def test_the_loads_read_the_same_resolved_sum(self):
        pricing = price_absolute(_q(carry_buffer_capacity_bytes=65536.0), self._tabled(), loads=_loads())
        bare = price_absolute(_q(), _physics("p4_bare", {"e_mac": 0.5, "t_cycle": 100.0, **BASE}), loads=_loads())
        a = {t.name: t for t in bare.terms}
        b = {t.name: t for t in pricing.terms}
        leg = b["energy_load_mj"].value - a["energy_load_mj"].value
        assert leg / J_TO_MJ / b["latency_load_s"].value == pytest.approx(pricing.static_power.watts.nominal,
                                                                          rel=1e-9)


class TestTheLadder:
    """Pi in energy: both of its costs carry their own static leg; M's period writes carry theirs, corner by corner
    (the hand point of test_scaling_ladder on few cores, so that passes rewrite)."""

    def _rows(self, cores):
        from unit.deployment_record.introspection.test_ladder_tile_writes import _row
        from unit.deployment_record.introspection.test_scaling_ladder import CONSTANTS
        bare = _physics("pi_bare", {**CONSTANTS, "t_program_per_byte": {"low": 5.0, "nominal": 7.0, "high": 11.0}})
        static = _physics("pi_static", {**CONSTANTS, **BANDED_STATIC,
                                        "t_program_per_byte": {"low": 5.0, "nominal": 7.0, "high": 11.0}})
        return _row(bare, cores=cores, k=16), _row(static, cores=cores, k=16), static

    def test_both_sides_of_pi_carry_static(self):
        bare, static, physics = self._rows(64)
        watts = _watts(physics, 64)
        for side in ("once", "compute_only"):
            wall = static["time"]["terms"][side]
            leg = [w * t * J_TO_MJ for w, t in zip(watts, wall)]
            got = [a - b for a, b in zip(static["energy"]["terms"][side], bare["energy"]["terms"][side])]
            assert got == pytest.approx(leg, rel=1e-9), side
        once, comp = static["energy"]["terms"]["once"], static["energy"]["terms"]["compute_only"]
        # a ratio's corners flip on its divisor (formulas.div_band): low over high, high over low
        assert static["energy"]["pi"] == pytest.approx([once[0] / comp[2], once[1] / comp[1], once[2] / comp[0]],
                                                       rel=1e-9)

    def test_the_period_column_carries_static_over_its_wall_at_each_corner(self):
        bare, static, physics = self._rows(8)
        wall = static["time"]["terms"]["period"]
        assert wall is not None and wall[1] > 0, "the hand point on 8 cores rewrites passes"
        leg = [w * t * J_TO_MJ for w, t in zip(_watts(physics, 8), wall)]
        got = [a - b for a, b in zip(static["energy"]["terms"]["period"], bare["energy"]["terms"]["period"])]
        assert got == pytest.approx(leg, rel=1e-9)
        assert leg[0] < leg[1] < leg[2]


def _sum(watts):
    """[m-3] A chip's static sum of ``watts`` (W): the step price takes the sum, never a bare band."""
    return StaticSum(watts=watts, declared=True, parts=(watts.basis,))


def _pass(q, *, cores, payload):
    return PassCensus(position=q, segment_index=0, resident=False, cores=cores, resident_cores=0,
                      written_cells_backing_store=payload, written_cells_producer=0, payload_bytes=payload,
                      compute_steps=0)


class TestTheStepPrice:
    """The step price (``step_choice.price_step``; no pricer of record calls it at R5) carries static power over the
    step's exposed write time at each corner."""

    PHYS = _physics("steps", {"e_program_per_byte": 3.0, "t_core_program": {"low": 20.0, "nominal": 50.0,
                                                                            "high": 90.0}})

    def test_the_step_leg_is_priced_at_each_corner(self):
        census = PassPeriodCensus(passes=(_pass(0, cores=2, payload=100), _pass(1, cores=1, payload=60)),
                                  boundaries=())
        policy = WritePolicy(batch_k=8)
        watts = Band(0.2, 0.5, 1.5, "test static")
        bare = price_step(census, self.PHYS, policy, static=NO_STATIC)
        with_static = price_step(census, self.PHYS, policy, static=_sum(watts))
        exposed = price_write_terms(census, self.PHYS, policy).exposed
        for corner in CORNERS:
            assert getattr(with_static.write, corner) - getattr(bare.write, corner) == pytest.approx(
                getattr(watts, corner) * getattr(exposed, corner) * J_TO_MJ, rel=1e-9), corner

    def test_the_step_price_orders_steps_with_static_power(self):
        few_cores = PassPeriodCensus(passes=(_pass(0, cores=1, payload=1000),), boundaries=())
        many_cores = PassPeriodCensus(passes=(_pass(0, cores=8, payload=600),), boundaries=())
        policy = WritePolicy(batch_k=1)
        assert cheapest_step([few_cores, many_cores], self.PHYS, policy, static=NO_STATIC) == 1
        assert cheapest_step([few_cores, many_cores], self.PHYS, policy, static=_sum(Band(1.0, 1.0, 1.0, "w"))) == 0


class TestTheRounds:
    def test_at_a_declared_batch_the_period_load_is_k_times_the_writes_per_inference(self):
        """m-3: on a write-law plane a round of K inferences already holds its reprogramming in the headline (the
        writes per inference are the period load over K, serial), so the rounds are read from the period terms."""
        physics = _physics("rounds", {**BASE, **STATIC})
        loads = _loads()
        period = loads.period.writes
        writes = WriteInputs.of(replace(period, policy=replace(period.policy, batch_k=4)))
        terms, _ = _terms(physics, _q(), writes=writes, loads=loads)
        assert 4 * terms["energy_write_mj"].value == pytest.approx(
            terms["energy_programming_mj"].value - _watts(physics)[1] * terms["latency_programming_s"].value
            * J_TO_MJ, rel=1e-12)
        assert 4 * terms["latency_write_exposed_s"].value == pytest.approx(terms["latency_programming_s"].value,
                                                                           rel=1e-12)


class TestEveryPricedTimeIsCoveredOnce:
    """The ratchet on one pricing: every priced term of unit s is a summand of e2e_latency_s (whose static leg is
    energy_static_mj) or the wall of exactly one static leg of its own energy term. The summands are read
    from the pricer's own table (``metrics.E2E_SUMMANDS``, the labels ``price_latency`` writes), never a hand list,
    and the pricing prices every one of them (an edge host on the shipped generic projection): each summand's band is
    in e2e's source under its label exactly once, the summands sum inside e2e's band at every corner, and none takes a
    static leg of its own. The full ratchet keyed on the time dimension over every package is W2's (T-E1a)."""

    OWN_LEG = {"latency_programming_s": "energy_programming_mj", "latency_load_s": "energy_load_mj"}
    VALUES = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100,
                  segment_cores=1, host_macs=1e5, compute_op_count=2, host_memory_bound_bytes=0, carry_in_bytes=0,
                  carry_out_bytes=0, segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4)

    def _pricing(self):
        from mimarsinan.deployment_record.platform_physics.hosts import store as ST
        from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
        from mimarsinan.deployment_record.platform_physics.resolve import (
            merge_override_blocks, resolve_platform_physics)
        physics = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks(
            {"host": host_description_block("jetson_orin_nano_8gb"), "store": ST.store_block(ST.CHIP, "test")}))
        loads = _loads()
        writes = WriteInputs.of(replace(loads.period.writes, policy=replace(loads.period.writes.policy,
                                                                            batch_k=4)))
        quantities = Quantities({k: QuantityValue(float(v), "static") for k, v in self.VALUES.items()})
        return _terms(physics, quantities, writes=writes, loads=loads)[0]

    def test_each_time_term_has_one_static_leg(self):
        from mimarsinan.deployment_record.cost.absolute.metrics import E2E_SUMMANDS  # a new name, imported here
        terms = self._pricing()
        label = dict(E2E_SUMMANDS)
        times = {t.name: t for t in terms.values() if t.unit == "s"}
        assert set(times) >= {"e2e_latency_s", "latency_programming_s", "latency_load_s"} | set(label), sorted(times)
        e2e = times["e2e_latency_s"]
        inside = [0.0, 0.0, 0.0]
        for name, t in times.items():
            if name == "e2e_latency_s":
                assert terms["energy_static_mj"].source.count("e2e_latency_s") >= 1
            elif name in label:  # a summand of e2e, under its label, exactly once, with no leg of its own
                assert e2e.source.count(label[name] + t.band.basis) == 1, name
                inside = [a + b for a, b in zip(inside, _ends(t.band))]
                assert not any(f"static over {name}" in x.source for x in terms.values()), name
            else:
                assert name in self.OWN_LEG, f"{name}: a priced time with no static rule"
                assert terms[self.OWN_LEG[name]].source.count(f"static over {name}") == 1
        assert all(i <= e * (1 + 1e-12) for i, e in zip(inside, _ends(e2e.band))), (inside, _ends(e2e.band))


class TestTheMeasuredModel:
    def test_the_total_carries_static_over_the_whole_latency(self):
        from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
        from unit.deployment_record.record_fixtures import make_full_record, with_partition
        record = with_partition(make_full_record())  # [composition: U-HOST] its host work priced, so total_s is
        physics = _physics("measured", {**BASE, **BANDED_STATIC})
        declared = replace(record, identity=replace(record.identity, platform={
            **record.identity.platform, "platform_physics_resolved": physics.to_dict()}))
        model = DeploymentCostModel()
        total_s = find_term(model.latency(declared), "total_s").band
        cores = sum(int(c.get("count", 1)) for c in record.identity.platform["cores"])
        leg = [w * t * J_TO_MJ for w, t in zip(_watts(physics, cores), _ends(total_s))]
        a = find_term(model.energy(record), "total_mj").band
        b = find_term(model.energy(declared), "total_mj").band
        assert [y - x for x, y in zip(_ends(a), _ends(b))] == pytest.approx(leg, rel=1e-9)
        assert "no static energy" in find_term(model.energy(record), "total_mj").source

    def test_the_measured_parts_say_they_hold_no_static_power(self):
        from mimarsinan.deployment_record.cost import DeploymentCostModel, find_term
        from unit.deployment_record.record_fixtures import make_full_record
        energy = DeploymentCostModel().energy(make_full_record())
        for name in ("modeled_programming_mj", "modeled_core_init_mj", "modeled_sync_mj"):
            assert "dynamic only, no static power" in find_term(energy, name).source, name
