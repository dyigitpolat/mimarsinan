"""[consts, owner decision of 2026-10-26] The write law charges the per-core programming energy of every rewritten core.

A core the period rewrites costs its payload's fetch (``e_dma_per_byte``) and commit (``e_write_per_cell`` x
iterations, or ``e_program_per_byte``) and, on a chip that declares it, the fixed energy of programming one core
(``e_core_program``): once per rewritten core per period, over K (``write_law.written_cores``, W), so once per tile in
the deployment load. Its basis carries the constant's evidence class. A chip that declares no ``e_core_program`` prices
as before; a kept core, a producer's operand writes and a refused commit charge none of it. Every price that reads
the write law reads the charge: the shares of several write technologies and the residency step choice.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from mimarsinan.deployment_record.build.write_census import PassRow, ProgramPasses
from mimarsinan.deployment_record.cost.absolute.step_choice import cheapest_step, price_step
from mimarsinan.deployment_record.cost.absolute.writes import (
    ENERGY_WRITE, WriteInputs, price_write_inputs, price_write_terms,
)
from mimarsinan.deployment_record.platform_physics import (
    apply_overrides, available_profiles, get_platform_physics, profile_from_dict,
)
from mimarsinan.deployment_record.quantities.write_law import WritePolicy, plane_writes, written_cores
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus
from unit.deployment_record.g3_fixtures import constant

CORE = constant(5.0, unit="pJ", low=2.0, high=11.0)
FETCH_COMMIT = {"e_dma_per_byte": constant(2.0, unit="pJ"), "e_program_per_byte": constant(3.0, unit="pJ")}
J_TO_MJ = 1e3


def _pass(position, *, cells=0, producer=0, cores=1, kept=0, payload=None, segment=0):
    return PassCensus(position=position, segment_index=segment, resident=False, cores=cores, resident_cores=kept,
                      written_cells_backing_store=cells, written_cells_producer=producer,
                      payload_bytes=cells if payload is None else payload, compute_steps=1)


def _physics(**constants):
    return profile_from_dict({
        "format_version": 1, "name": "core_program_test", "display_name": "Core program test",
        "description_file": "t.md", "validity": {"measurement_kind": "projection"}, "constants": constants,
    })


def _without_core_program(physics):
    return replace(physics, constants={k: v for k, v in physics.constants.items() if k != "e_core_program"})


def _energy(pricing):
    return next((t for t in pricing.terms if t.name == ENERGY_WRITE), None)


def _corners(band):
    return [band.low, band.nominal, band.high]


#: Two rotating passes: three cores with one kept and two cores, so the period rewrites W = 2 + 2 cores.
ROTATING = PassPeriodCensus(passes=(_pass(0, cells=120, cores=3, kept=1, payload=60),
                                    _pass(1, cells=80, cores=2, payload=40)))
W, BYTES = 4, 100


class TestEveryRewrittenCore:
    @pytest.mark.parametrize("k", [1, 4, 64])
    def test_each_rewritten_core_is_charged_once_per_period_over_k(self, k):
        assert written_cores(ROTATING) == W
        priced = _energy(price_write_terms(ROTATING, _physics(**FETCH_COMMIT, e_core_program=CORE),
                                           WritePolicy(batch_k=k)))
        assert priced is not None
        want = [((2.0 + 3.0) * BYTES + core * W) * 1e-12 / k * J_TO_MJ for core in (2.0, 5.0, 11.0)]
        assert _corners(priced.band) == pytest.approx(want, rel=1e-12)

    def test_the_charge_carries_the_constants_evidence_class(self):
        chip = _physics(**FETCH_COMMIT, e_core_program=CORE)
        source = _energy(price_write_terms(ROTATING, chip, WritePolicy(batch_k=4))).source
        declared = chip.constants["e_core_program"]
        assert f"e_core_program [{declared.basis}] x rewritten_cores / K" in source
        assert declared.basis.startswith("estimated: ")
        overridden = apply_overrides(chip, {"e_core_program": {**CORE, "evidence_kind": "derived",
                                                               "derivation": "a test derivation"}})
        source = _energy(price_write_terms(ROTATING, overridden, WritePolicy(batch_k=4))).source
        assert "e_core_program [derived (operator override): a test derivation]" in source

    def test_a_chip_that_declares_no_core_programming_energy_prices_as_before(self):
        priced = _energy(price_write_terms(ROTATING, _physics(**FETCH_COMMIT), WritePolicy(batch_k=4)))
        assert _corners(priced.band) == pytest.approx([(2.0 + 3.0) * BYTES * 1e-12 / 4 * J_TO_MJ] * 3, rel=1e-12)
        assert "e_core_program" not in priced.source

    def test_kept_cores_and_a_census_that_rewrites_nothing_charge_none(self):
        chip = _physics(**FETCH_COMMIT, e_core_program=CORE)
        kept = PassPeriodCensus(passes=(_pass(0, cores=3, kept=3), _pass(1, cores=2, kept=2)))
        assert written_cores(kept) == 0
        assert _corners(_energy(price_write_terms(kept, chip, WritePolicy(batch_k=1))).band) == [0.0, 0.0, 0.0]

    def test_a_producers_operand_writes_charge_no_core_programming(self):
        """Operand regions are written per inference into cores the period does not rewrite: payload, not a core."""
        chip = _physics(e_write_per_cell=constant(2.0, unit="pJ"), e_core_program=CORE,
                        write_verify_iterations=constant(3.0, unit="iterations"))
        operand = PassPeriodCensus(passes=(_pass(0, producer=400, cores=2, payload=0),))
        priced = _energy(price_write_terms(operand, chip, WritePolicy(batch_k=8)))
        assert _corners(priced.band) == pytest.approx([2e-12 * 3 * 400 * J_TO_MJ] * 3, rel=1e-12)
        assert "e_core_program" not in priced.source

    def test_a_refused_commit_stays_refused_by_its_own_reason(self):
        chip = _physics(e_dma_per_byte=constant(2.0, unit="pJ"), e_core_program=CORE)
        pricing = price_write_terms(ROTATING, chip, WritePolicy(batch_k=4))
        assert _energy(pricing) is None
        reason = next(r.reason for r in pricing.refusals if r.name == ENERGY_WRITE)
        assert reason.startswith("cells are written per inference but the target declares no commit energy")


@pytest.mark.parametrize("name", available_profiles())
def test_every_shipped_profile_that_declares_it_is_charged_and_no_other_moves(name):
    """Generic over the registry: the write energy at K moves by e_core_program x W / K exactly where the chip declares
    it and the energy prices; a chip without it, or a write it refuses, is unchanged."""
    chip = get_platform_physics(name)
    policy = WritePolicy(batch_k=4)
    with_it = price_write_terms(ROTATING, chip, policy)
    without = price_write_terms(ROTATING, _without_core_program(chip), policy)
    if _energy(without) is None or not chip.has("e_core_program"):
        assert with_it.terms == without.terms and with_it.refusals == without.refusals
        return
    core = chip.band("e_core_program")
    delta = [a - b for a, b in zip(_corners(_energy(with_it).band), _corners(_energy(without).band))]
    assert delta == pytest.approx([c * W / 4 * J_TO_MJ for c in _corners(core)], rel=1e-9)
    assert f"e_core_program [{chip.constants['e_core_program'].basis}]" in _energy(with_it).source
    assert [t for t in with_it.terms if t.name != ENERGY_WRITE] == [t for t in without.terms if t.name != ENERGY_WRITE]


S, T = (64, 64), (16, 16)


def _row(cells, types):
    return PassRow(segment_index=0, resident=False, core_cells=tuple(cells), marks=frozenset(),
                   core_extents=tuple((c, 1) for c in cells), compute_steps=1, stage_latencies=(0,),
                   core_types=tuple(types))


def test_each_write_technologys_share_is_charged_on_its_own_rewritten_cores():
    """N2b: e_core_program is the chip's, not a write row, so each technology's share charges it on the cores it
    rewrites, and the shares add to the chip's W."""
    passes = ProgramPasses(rows=(_row([100, 50], [S, S]), _row([30, 20], [S, T]), _row([40], [T])))
    platform = {"inference_batch_K": 4, "cores": [
        {"max_axons": 64, "max_neurons": 64, "count": 2, "write_granularity": "cell"},
        {"max_axons": 16, "max_neurons": 16, "count": 2, "write_granularity": "cell",
         "write_physics": "generic_estimated_22nm"}]}
    writes = plane_writes(passes, platform, weight_bits=8, timesteps=None, label="hand")
    assert writes is not None and writes.by_technology is not None and writes.census is not None
    shares = [written_cores(part.census) for part in writes.by_technology.parts]
    assert sorted(shares) == [2, 3] and written_cores(writes.census.census) == 5
    chip = _physics(**FETCH_COMMIT, e_core_program=CORE)
    with_it = _energy(price_write_inputs(WriteInputs.of(writes), chip))
    without = _energy(price_write_inputs(WriteInputs.of(writes), _without_core_program(chip)))
    delta = [a - b for a, b in zip(_corners(with_it.band), _corners(without.band))]
    assert delta == pytest.approx([c * 1e-12 * 5 / 4 * J_TO_MJ for c in (2.0, 5.0, 11.0)], rel=1e-9)
    assert with_it.source.count("e_core_program [") == 2


def test_the_residency_step_choice_reads_the_charge():
    """Two steps writing the same bytes: four rewritten cores and no carry, or two cores and a carry. Without a
    per-core energy the first is cheaper; a chip that declares one pays it per rewritten core, so the second wins."""
    four = PassPeriodCensus(passes=(_pass(0, cells=50, cores=2), _pass(1, cells=50, cores=2)))
    two = PassPeriodCensus(passes=(_pass(0, cells=50), _pass(1, cells=50)),
                           boundaries=(PassBoundaryCensus(after_pass=0, carry_out_bytes=10, carry_in_bytes=10),))
    plain = _physics(e_dma_per_byte=constant(0.0, unit="pJ"), e_program_per_byte=constant(1.0, unit="pJ"),
                     e_readout_per_byte=constant(1.0, unit="pJ"))
    chip = apply_overrides(plain, {"e_core_program": constant(20.0, unit="pJ")})
    policy = WritePolicy(batch_k=1)
    assert cheapest_step([four, two], plain, policy) == 0
    assert price_step(four, chip, policy).nominal == pytest.approx((100 + 4 * 20) * 1e-12 * J_TO_MJ, rel=1e-12)
    assert price_step(two, chip, policy).nominal == pytest.approx((100 + 2 * 20 + 10) * 1e-12 * J_TO_MJ, rel=1e-12)
    assert cheapest_step([four, two], chip, policy) == 1
