"""[U-J4 K-2] The declared store rule's scope: the no-store refusal applies only where the physics prices the host link.

Conductor decision of 2026-10-30 (vetoable), U-J3's departure D2 accepted: a chip that states no backing store is
refused its store traffic by name only beside the host-chip link (the edge host's link and host-memory rows), where
two places could price the same byte. A chip with no link rows has one store place, its own channel rows
(``e_dma_per_byte`` in, ``e_readout_per_byte`` out, the commit and the write port), so it prices as on RC1
(d1c05c51): the pass carry both ways, the write transfer, a host-sourced operand payload and a program load on those
rows, no refusal by the store's name and no link term. The tests price by hand from the profile's own rows and
import nothing RC1 lacks, so the same file runs on RC1's tree (the unit's record runs it there). [U-ENERGY] The
program-load test prices the one load law over a census (its helper is that unit's, imported where it is used)."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.step_choice import carry_energy, price_step
from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_terms
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus

CORNERS = ("low", "nominal", "high")
#: The edge host's link and host-memory rows (``hosts.store.LINK_ROWS`` on the unit's tree), spelled out for RC1.
LINK_ROWS = ("e_link_per_byte", "t_link_per_byte", "t_link_transfer", "e_host_mem_per_byte", "t_host_mem_per_byte")
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}
PHYSICS = resolve_platform_physics("generic_estimated_22nm", merge_override_blocks({"write": WRITE}))
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
            host_macs=0, compute_op_count=0)
CARRY = dict(carry_in_bytes=300, carry_out_bytes=200)


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _policy(source=None):
    return WritePolicy(batch_k=4, producer_write_source=source, payload_bits=None if source is None else 8,
                       write_granularity="cell")


def _census(payload=400, producer=0, carry=(0, 0)):
    passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=100, written_cells_producer=producer, payload_bytes=payload,
                         compute_steps=3),
              PassCensus(position=1, segment_index=0, resident=True, cores=1, resident_cores=1,
                         written_cells_backing_store=0, written_cells_producer=0, payload_bytes=0, compute_steps=3))
    return PassPeriodCensus(passes=passes,
                            boundaries=(PassBoundaryCensus(after_pass=0, carry_out_bytes=carry[0],
                                                           carry_in_bytes=carry[1]),))


def _priced(values, writes=None):
    pricing = price_absolute(_q(**values), PHYSICS, writes=writes)
    return {t.name: t.band for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _k(key, corner):
    return getattr(PHYSICS.band(key), corner)


def test_the_chip_states_no_link_rows_and_no_store():
    assert not any(PHYSICS.has(key) for key in LINK_ROWS)
    assert getattr(PHYSICS, "backing_store", None) is None


def test_the_carry_prices_on_the_chips_channel_rows_both_ways():
    with_carry, refused = _priced(dict(BASE, **CARRY))
    without, _ = _priced(dict(BASE, carry_in_bytes=0, carry_out_bytes=0))
    assert not {"energy_per_inference_mj", "e2e_latency_s"} & set(refused)
    for corner in CORNERS:
        carry = 300 * _k("e_dma_per_byte", corner) + 200 * _k("e_readout_per_byte", corner)
        assert getattr(with_carry["energy_dynamic_mj"], corner) - getattr(without["energy_dynamic_mj"], corner) == \
            pytest.approx(carry * 1e3, rel=1e-9), corner


def test_the_write_transfer_prices_on_e_dma_per_byte():
    full = price_write_terms(_census(payload=400), PHYSICS, _policy()).energy
    bare = price_write_terms(_census(payload=0), PHYSICS, _policy()).energy
    for corner in CORNERS:  # 400 B per period over K = 4
        assert getattr(full, corner) - getattr(bare, corner) == pytest.approx(
            100 * _k("e_dma_per_byte", corner) * 1e3, rel=1e-9), corner


def test_a_host_sourced_payload_prices_on_e_dma_per_byte():
    host = price_write_terms(_census(producer=10), PHYSICS, _policy("host")).energy
    own = price_write_terms(_census(producer=10), PHYSICS, _policy("on_chip")).energy
    for corner in CORNERS:  # 10 producer cells x 8 payload bits per inference
        assert getattr(host, corner) - getattr(own, corner) == pytest.approx(
            10 * _k("e_dma_per_byte", corner) * 1e3, rel=1e-9), corner


def test_a_program_load_prices_on_e_dma_per_byte_and_the_commit():
    """[U-ENERGY] A program load is the one load law over the program's census (two written passes of 500 B, so
    1000 B rewritten per round): its transfer on ``e_dma_per_byte``, its commit by the write law (by cell where the
    description declares a cell commit), its static leg over its own wall, no store refusal and no link term."""
    from dataclasses import replace

    from mimarsinan.deployment_record.cost.absolute.writes import price_write_inputs
    from unit.deployment_record.load_census_fixtures import written_program

    loads = written_program((500,), (500,), granularity="cell")
    pricing = price_absolute(_q(**dict(BASE, carry_in_bytes=0, carry_out_bytes=0)), PHYSICS, loads=loads)
    terms = {t.name: t.band for t in pricing.terms}
    no_dma = replace(PHYSICS, constants={k: v for k, v in PHYSICS.constants.items() if k != "e_dma_per_byte"})
    commit = price_write_inputs(WriteInputs.of(loads.period.writes), no_dma, lifetime=False).energy
    for corner in CORNERS:
        dynamic = getattr(terms["energy_programming_mj"], corner) - getattr(terms["energy_static_programming_mj"],
                                                                            corner)
        assert dynamic - getattr(commit, corner) == pytest.approx(1000 * _k("e_dma_per_byte", corner) * 1e3,
                                                                  rel=1e-9), corner
    assert not any(k in terms for k in ("energy_host_link_mj", "latency_host_link_s"))
    assert not any("backing_store" in r.reason for r in pricing.refusals)


def test_store_traffic_is_never_refused_by_the_stores_name_and_nothing_prices_a_link():
    writes = WriteInputs(_census(producer=10, carry=(200, 300)), _policy("host"))
    terms, refused = _priced(dict(BASE, **CARRY), writes=writes)
    assert "energy_per_inference_mj" in terms and "e2e_latency_s" in terms
    assert not any("backing_store" in reason for reason in refused.values())
    assert not {"energy_host_link_mj", "energy_host_mem_mj", "latency_host_link_s"} & set(terms)


def test_the_step_choice_prices_on_the_chips_rows():
    census = _census(producer=10, carry=(200, 300))
    for corner in CORNERS:
        carry = 300 * _k("e_dma_per_byte", corner) + 200 * _k("e_readout_per_byte", corner)
        assert getattr(carry_energy(census, PHYSICS), corner) == pytest.approx(carry * 1e3, rel=1e-9), corner
    step = price_step(census, PHYSICS, _policy("host"), static=NO_STATIC)
    assert step.write == price_write_terms(census, PHYSICS, _policy("host")).energy
