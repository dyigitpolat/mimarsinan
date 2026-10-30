"""[U-J5 L-5, U-J4 finding F1] A host-sourced payload's transfer over the host-chip link is timed, priced by hand,
corner by corner.

The rule (conductor decision, vetoable; the owner's host rule asks for the host-chip boundary I/O to be modeled), with
U-J3's ``compute_time.store_walls`` as its precedent: where the physics prices the host-chip link and an operand core
type's payloads come from the host, a pass's payload bytes (its producer cells x ``payload_bits`` / 8) stream over the
link at the link's per-byte stage (the slower of ``t_link_per_byte`` and ``t_host_mem_per_byte``), so the pass's
producer wall is the slower of its producer write steps and that link wall (the pipelined bound), plus one
``t_link_transfer`` per pass that carries such a payload. The producer writes stay their own per-inference term of the
write law (serial in both overlap modes), so the backing-store payload of a host-owned store (once per period, in the
period's write phase, U-J3's ``store_walls``) and the host-sourced payload (per inference, in the producer's write
phase) take the one link in turn; one rule on either store. Without link rows nothing changes.

Vehicle: U-J4's two-pass period census (400 B of backing-store payload, K = 4) with 10 producer cells written per
inference at 8 payload bits (10 B over the link); ``on_chip`` writes the same cells by the producer's write-back and
moves nothing, so the two differ by the payload's link time alone. ``SLOW`` writes a cell in 10 ns (the write steps
bind: 100 ns against 10 B x 0.30 ns), ``FAST`` in 0.001 ns (the link binds)."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_terms
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus

CORNERS = ("low", "nominal", "high")
HOST = "jetson_orin_nano_8gb"
GENERIC = "generic_estimated_22nm"
BITS = 8


def _write(ns):
    return {"t_write_per_unit": {"nominal": ns, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}


SLOW, FAST = _write(10.0), _write(0.001)
STAGES = ("t_link_per_byte", "t_host_mem_per_byte")
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
            host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, carry_in_bytes=0, carry_out_bytes=0,
            segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4)
#: The text the base printed for an untimed payload; retired where the payload's link time is priced.
UNTIMED = "timed by no term"


def _physics(write=SLOW, store=None, host=True, drop=()):
    blocks = {"write": write}
    if host:
        blocks["host"] = {key: row for key, row in host_description_block(HOST).items() if key not in drop}
    if store is not None:
        blocks["store"] = ST.store_block(store, "test")
    return resolve_platform_physics(GENERIC, merge_override_blocks(blocks))


def _policy(source, mode="serial"):
    return WritePolicy(batch_k=4, producer_write_source=source, payload_bits=BITS, write_granularity="cell",
                       overlap_mode=mode, core_capacity=(((256, 256), 64),))


def _census(producer=(10, 0), payload=400, cells=100):
    passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=cells, written_cells_producer=producer[0], payload_bytes=payload,
                         compute_steps=3),
              PassCensus(position=1, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=0, written_cells_producer=producer[1], payload_bytes=0,
                         compute_steps=3))
    boundary = (PassBoundaryCensus(after_pass=0, carry_out_bytes=0, carry_in_bytes=0),)
    return PassPeriodCensus(passes=passes, boundaries=boundary)


def _k(physics, key, corner):
    return getattr(physics.band(key), corner)


def _step(physics, corner):
    """One producer write step: t_write_per_unit x write_verify_iterations (one cell per step, one port)."""
    return _k(physics, "t_write_per_unit", corner) * _k(physics, "write_verify_iterations", corner)


def _stage(physics, corner):
    return max(_k(physics, key, corner) for key in STAGES if physics.has(key))


def _producer(physics, census, corner, linked, transfer=True):
    """The producer writes per inference in seconds, by hand (every band is canonical SI): per pass its cells' write
    steps, or beside the link the slower of those and its payload at the link's stage, plus one t_link_transfer per
    pass carrying a payload."""
    total = 0.0
    for census_pass in census.passes:
        cells = census_pass.written_cells_producer
        if not cells:
            continue
        own = _step(physics, corner) * cells
        if linked:
            own = max(own, _stage(physics, corner) * cells * BITS / 8.0)
        total += own
    if linked and transfer:
        total += _k(physics, "t_link_transfer", corner) * sum(1 for p in census.passes if p.written_cells_producer)
    return total


def _exposed(physics, source, census=None, mode="serial"):
    pricing = price_write_terms(census or _census(), physics, _policy(source, mode))
    assert pricing.exposed is not None, [r.reason for r in pricing.refusals]
    return pricing.exposed


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _e2e(physics, source, census=None):
    pricing = price_absolute(_q(**BASE), physics, writes=WriteInputs(census or _census(), _policy(source)))
    terms = {t.name: t for t in pricing.terms}
    assert "e2e_latency_s" in terms, [r.reason for r in pricing.refusals]
    return terms["e2e_latency_s"]


def test_the_vehicles_two_write_speeds_bind_on_either_side_of_the_link():
    """SLOW's write steps outlast the payload's link wall at every corner, FAST's do not."""
    for write, binds in ((SLOW, "steps"), (FAST, "link")):
        physics = _physics(write, ST.CHIP)
        for corner in CORNERS:
            steps, link = _step(physics, corner) * 10, _stage(physics, corner) * 10 * BITS / 8.0
            assert (steps > link) == (binds == "steps"), (binds, corner, steps, link)


@pytest.mark.parametrize("write", [SLOW, FAST], ids=["write_bound", "link_bound"])
@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
class TestAHostSourcedPayloadIsTimedOnTheLink:
    """On either store, one rule: the producer wall per pass is the slower of its write steps and its payload at the
    link's stage, plus one t_link_transfer; the backing-store payload's walls (``store_walls``) are unchanged."""

    @pytest.mark.parametrize("mode", ["serial", "overlapped"])
    def test_the_exposed_write_time_adds_the_payloads_link_wall(self, store, write, mode):
        physics = _physics(write, store)
        host, own = _exposed(physics, "host", mode=mode), _exposed(physics, "on_chip", mode=mode)
        census = _census()
        for corner in CORNERS:
            expected = _producer(physics, census, corner, True) - _producer(physics, census, corner, False)
            assert getattr(host, corner) - getattr(own, corner) == pytest.approx(expected, rel=1e-9, abs=1e-21), \
                (mode, corner)

    def test_the_e2e_latency_rises_by_it_and_names_no_untimed_payload(self, store, write):
        physics = _physics(write, store)
        host, own = _e2e(physics, "host"), _e2e(physics, "on_chip")
        census = _census()
        for corner in CORNERS:
            expected = _producer(physics, census, corner, True) - _producer(physics, census, corner, False)
            assert getattr(host.band, corner) - getattr(own.band, corner) == pytest.approx(expected, rel=1e-9), corner
        assert UNTIMED not in host.source and "host-sourced payload: its transfer" not in host.source

    def test_the_basis_states_the_payloads_link_wall(self, store, write):
        basis = _exposed(_physics(write, store), "host").basis
        assert "host-sourced payload" in basis and "t_link_transfer" in basis, basis
        assert "the slower of" in basis and "t_link_per_byte" in basis, basis
        shared = "the backing-store payload takes the same link in the period's write phase"
        assert (shared in basis) == (store == ST.HOST), basis


def test_one_link_transfer_per_pass_that_carries_a_payload():
    """Producer cells in both passes: two transfers, each pass its own slower-of."""
    physics, census = _physics(FAST, ST.CHIP), _census(producer=(10, 6))
    host, own = _exposed(physics, "host", census), _exposed(physics, "on_chip", census)
    for corner in CORNERS:
        expected = _producer(physics, census, corner, True) - _producer(physics, census, corner, False)
        assert getattr(host, corner) - getattr(own, corner) == pytest.approx(expected, rel=1e-9), corner
        assert expected > 2 * _k(physics, "t_link_transfer", corner) * (1 - 1e-12), corner


@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
def test_the_backing_store_payloads_walls_are_unchanged(store):
    """The period's walls (U-J3's ``store_walls`` on a host-owned store) do not read the host-sourced payload: with no
    producer cells the host and on-chip sources price alike."""
    physics, census = _physics(FAST, store), _census(producer=(0, 0))
    host, own = _exposed(physics, "host", census), _exposed(physics, "on_chip", census)
    assert (host.low, host.nominal, host.high, host.basis) == (own.low, own.nominal, own.high, own.basis)


class TestWithoutTheLink:
    """Without link rows the producer writes are their write steps alone, as before, whatever their source."""

    @pytest.mark.parametrize("write", [SLOW, FAST], ids=["write_bound", "link_bound"])
    def test_without_link_rows_nothing_changes(self, write):
        physics, census = _physics(write, host=False), _census()
        host, own = _exposed(physics, "host"), _exposed(physics, "on_chip")
        assert (host.low, host.nominal, host.high, host.basis) == (own.low, own.nominal, own.high, own.basis)
        assert "host-sourced payload" not in host.basis
        for corner in CORNERS:
            period = 100 * _step(physics, corner) / 4  # the backing-store cells' steps over K, serial
            assert getattr(host, corner) == pytest.approx(period + _producer(physics, census, corner, False),
                                                          rel=1e-12), corner

    def test_an_on_chip_write_back_beside_the_link_moves_nothing(self):
        beside, alone = _exposed(_physics(FAST, ST.CHIP), "on_chip"), _exposed(_physics(FAST, host=False), "on_chip")
        assert (beside.low, beside.nominal, beside.high) == (alone.low, alone.nominal, alone.high)


class TestAnUndeclaredLinkTimeRow:
    """Beside the link a row the payload's time needs and the physics does not declare (a hand block; a host
    description declares every row, ``REQUIRED_HOST_ROWS``) is named unpriced on the latency, never dropped."""

    def test_neither_per_byte_stage_leaves_the_payloads_link_time_named(self):
        physics = _physics(FAST, ST.CHIP, drop=STAGES)
        host, own = _exposed(physics, "host"), _exposed(physics, "on_chip")
        assert (host.low, host.nominal, host.high) == (own.low, own.nominal, own.high)
        source = _e2e(physics, "host").source
        assert "t_link_per_byte, t_host_mem_per_byte(host-sourced payload: the physics prices the link and declares " \
               "neither per-byte stage" in source, source

    def test_an_undeclared_transfer_latency_is_named_and_the_stage_still_timed(self):
        physics, census = _physics(FAST, ST.CHIP, drop=("t_link_transfer",)), _census()
        host, own = _exposed(physics, "host"), _exposed(physics, "on_chip")
        for corner in CORNERS:
            expected = (_producer(physics, census, corner, True, transfer=False)
                        - _producer(physics, census, corner, False))
            assert getattr(host, corner) - getattr(own, corner) == pytest.approx(expected, rel=1e-9), corner
        source = _e2e(physics, "host").source
        assert "t_link_transfer(host-sourced payload: one transfer per pass carrying it" in source, source

    def test_a_declared_row_is_never_named(self):
        source = _e2e(_physics(FAST, ST.CHIP), "host").source
        assert "(host-sourced payload:" not in source, source


class TestBothPlanes:
    """N2's attention program packed on 64 x 64 cores (U-J4's plane vehicle): the mapping plane and the sealed record
    state the same period census, so each times the host-sourced payload over the link alike, on either store."""

    @pytest.fixture(scope="class")
    def planes(self):
        from unit.chip_simulation.value_run.operand_programs import attention_program
        from unit.deployment_record.test_link_crossing_planes_uj4 import GEOMETRY, _planes

        mapping = attention_program("softmax").packed_mapping(**GEOMETRY)
        return {source: _planes(mapping, source) for source in ("host", "on_chip")}

    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_each_plane_adds_the_payloads_link_walls(self, planes, store):
        from mimarsinan.deployment_record.cost.absolute.writes import price_write_inputs
        from unit.deployment_record.test_link_crossing_planes_uj4 import PHYSICS as PLANE_PHYSICS

        physics, deltas = PLANE_PHYSICS[store], {}
        for plane in ("mapping", "record"):
            host, own = (next(t.band for t in price_write_inputs(planes[source][plane], physics).terms
                              if t.name == "latency_write_exposed_s") for source in ("host", "on_chip"))
            census = planes["host"][plane].census
            deltas[plane] = tuple(getattr(host, c) - getattr(own, c) for c in CORNERS)
            for corner, delta in zip(CORNERS, deltas[plane]):
                expected = _producer(physics, census, corner, True) - _producer(physics, census, corner, False)
                assert expected > 0 and delta == pytest.approx(expected, rel=1e-9), (plane, corner)
        assert deltas["mapping"] == pytest.approx(deltas["record"], rel=1e-12)
