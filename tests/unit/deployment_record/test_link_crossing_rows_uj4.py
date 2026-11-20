"""[U-J4 K-1] A byte that crosses the host-chip link pays the link and host-memory rows, never the chip's
backing-store rows (``e_dma_per_byte`` in, ``e_readout_per_byte`` out), priced by hand, corner by corner.

The rule (conductor decision of 2026-10-30, from U-J3's review minor m2): wherever the physics prices the host-chip
link (an edge host block), every byte that reaches the chip over the link or leaves it that way is priced on the link
(``e_link_per_byte``) and host memory (``e_host_mem_per_byte``) once, and on none of the chip's store rows. The bytes:
the program's own crossing (the segment I/O, a host-sourced operand payload) on either store, and the store traffic
(the pass carry, the write law's backing-store payload, a program load) where the store is host-owned. The chip's own
end of the link is priced by no row of the vocabulary, so the link's energy names it unpriced (a boundary).

Vehicle: U-J3's two-pass period census (400 B of backing-store payload, K = 4) with 10 producer cells written per
inference at 8 payload bits: a host-sourced payload moves 10 B per inference over the link; the same census with an
on-chip write-back (``on_chip``) writes the same cells and moves nothing, so the two differ by the transfer alone."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost import host_terms as HT
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.step_choice import price_step
from mimarsinan.deployment_record.cost.absolute.walls.static_power import NO_STATIC
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_terms
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.operand_form import AT_STATIONARY, B_STATIONARY, Lowering, OperandForm
from mimarsinan.mapping.support.operand_price import lowering_price, operand_region_census
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus

CORNERS = ("low", "nominal", "high")
HOST = "jetson_orin_nano_8gb"
GENERIC = "generic_estimated_22nm"
#: The producer cells' write time (the write law times producer cells at t_write_per_unit only).
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}
#: The chip's store rows at a value far from the shipped ones: what no link-crossing byte may read.
LOUD_ROWS = {key: {"nominal": 1.0e6, "unit": "pJ", "evidence_kind": "estimated", "note": "test constant"}
             for key in ("e_dma_per_byte", "e_readout_per_byte")}
PAYLOAD_BYTES = 10.0  # 10 producer cells x 8 payload bits per inference


def _physics(store=None, host=True, loud=False):
    blocks = {"write": WRITE}
    if host:
        blocks["host"] = host_description_block(HOST)
    if store is not None:
        blocks["store"] = ST.store_block(store, "test")
    if loud:
        blocks["rows"] = LOUD_ROWS
    return resolve_platform_physics(GENERIC, merge_override_blocks(blocks))


PHYSICS = {store: _physics(store) for store in (ST.CHIP, ST.HOST)}
LOUD = {store: _physics(store, loud=True) for store in (ST.CHIP, ST.HOST)}
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
            host_macs=0, compute_op_count=0, host_memory_bound_bytes=0)
SEGMENT = dict(segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4)
#: The program's own crossing only: no pass carry.
VALUES = dict(BASE, carry_in_bytes=0, carry_out_bytes=0, **SEGMENT)


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _policy(source):
    return WritePolicy(batch_k=4, producer_write_source=source, payload_bits=8, write_granularity="cell")


def _census(payload=400, cells=100, carry=(0, 0), producer=10):
    passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=cells, written_cells_producer=producer, payload_bytes=payload,
                         compute_steps=3),
              PassCensus(position=1, segment_index=0, resident=True, cores=1, resident_cores=1,
                         written_cells_backing_store=0, written_cells_producer=0, payload_bytes=0, compute_steps=3))
    boundary = (PassBoundaryCensus(after_pass=0, carry_out_bytes=carry[0], carry_in_bytes=carry[1]),)
    return PassPeriodCensus(passes=passes, boundaries=boundary)


def _priced(physics, source="host", census=None, loads=None, **values):
    writes = WriteInputs(census or _census(), _policy(source))
    pricing = price_absolute(_q(**(values or VALUES)), physics, writes=writes, loads=loads)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _c(term, corner):
    return getattr(term.band, corner)


def _k(physics, key, corner):
    return getattr(physics.band(key), corner)


def _link_and_memory(physics, corner):
    return _k(physics, "e_link_per_byte", corner) + _k(physics, "e_host_mem_per_byte", corner)


@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
class TestAHostSourcedPayload:
    """The payload crosses the link on either store: priced there once, never on the chip's ``e_dma_per_byte``."""

    def test_the_write_term_prices_its_cells_and_no_transfer(self, store):
        host, refused = _priced(PHYSICS[store], "host")
        own, refused_own = _priced(PHYSICS[store], "on_chip")
        assert refused == refused_own  # the lifetime's own refusal on both (no endurance declared)
        for corner in CORNERS:
            assert _c(host["energy_write_mj"], corner) == pytest.approx(_c(own["energy_write_mj"], corner),
                                                                        rel=1e-12), corner
        assert "e_dma_per_byte x host-sourced" not in host["energy_write_mj"].source
        assert "host-sourced payload crosses the host-chip link" in host["energy_write_mj"].source

    def test_the_link_and_host_memory_price_the_payload_once(self, store):
        host, _ = _priced(PHYSICS[store], "host")
        own, _ = _priced(PHYSICS[store], "on_chip")
        for key, name in (("e_link_per_byte", "energy_host_link_mj"), ("e_host_mem_per_byte", "energy_host_mem_mj")):
            for corner in CORNERS:
                assert _c(host[name], corner) - _c(own[name], corner) == pytest.approx(
                    PAYLOAD_BYTES * _k(PHYSICS[store], key, corner) * 1e3, rel=1e-9), (name, corner)

    def test_the_headline_pays_the_payload_on_the_link_and_host_memory_only(self, store):
        """[U-J5 L-5] Beside the payload's energy the headline carries the static energy over its now-priced link
        time (``test_payload_link_time_uj5.py``): that term's difference is left out here, and nothing else moves."""
        host, _ = _priced(PHYSICS[store], "host")
        own, _ = _priced(PHYSICS[store], "on_chip")
        for corner in CORNERS:
            static = _c(host["energy_static_mj"], corner) - _c(own["energy_static_mj"], corner)
            assert _c(host["energy_per_inference_mj"], corner) - _c(own["energy_per_inference_mj"], corner) - static \
                == pytest.approx(PAYLOAD_BYTES * _link_and_memory(PHYSICS[store], corner) * 1e3, rel=1e-9), corner

    def test_the_write_law_alone_charges_the_payload_no_transfer(self, store):
        """The write law's own pricer (the step choice and the record plane read it), not only the headline."""
        host = price_write_terms(_census(), PHYSICS[store], _policy("host")).energy
        own = price_write_terms(_census(), PHYSICS[store], _policy("on_chip")).energy
        for corner in CORNERS:
            assert getattr(host, corner) == pytest.approx(getattr(own, corner), rel=1e-12), corner


class TestTheChipsStoreRowsNeverPriceALinkCrossingByte:
    """Every energy term is the same whatever the chip's store rows read, wherever only link-crossing bytes move."""

    def _terms(self, physics, values, census):
        pricing = price_absolute(_q(**values), physics, writes=WriteInputs(census, _policy("host")))
        return {t.name: t.band for t in pricing.terms if t.name.startswith("energy")}

    def _same(self, values, census, store):
        shipped, loud = self._terms(PHYSICS[store], values, census), self._terms(LOUD[store], values, census)
        assert shipped.keys() == loud.keys() and "energy_per_inference_mj" in shipped
        for name, band in shipped.items():
            for corner in CORNERS:
                assert getattr(loud[name], corner) == pytest.approx(getattr(band, corner), rel=1e-12), (name, corner)

    def test_on_a_chip_owned_store_the_programs_own_crossing(self):
        """The segment I/O and a host-sourced payload; no carry, no backing-store payload (the chip's own DRAM)."""
        self._same(VALUES, _census(payload=0), ST.CHIP)

    def test_on_a_host_owned_store_every_crossing(self):
        """The carry both ways, the backing-store payload, a host-sourced payload, the segment I/O, a program load."""
        values = dict(VALUES, carry_in_bytes=300, carry_out_bytes=200, reprogrammed_bytes=1000)
        self._same(values, _census(carry=(200, 300)), ST.HOST)

    def test_the_chips_store_traffic_on_its_own_store_still_reads_them(self):
        """Contrast: the carry to the chip's own DRAM is the chip's store traffic, priced on its rows."""
        values = dict(VALUES, carry_in_bytes=300, carry_out_bytes=200)
        shipped, loud = self._terms(PHYSICS[ST.CHIP], values, _census()), self._terms(LOUD[ST.CHIP], values, _census())
        assert loud["energy_dynamic_mj"].nominal > shipped["energy_dynamic_mj"].nominal


class TestTheChipsOwnEndOfTheLink:
    """No vocabulary row prices the chip's end of the link: the link's energy names it unpriced, a boundary."""

    END = "the chip's own end of the host-chip link"

    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_the_links_energy_names_it(self, store):
        terms, _ = _priced(PHYSICS[store], "host")
        assert self.END in terms["energy_host_link_mj"].source
        assert self.END in terms["energy_per_inference_mj"].source

    def test_a_program_load_over_the_link_names_it(self):
        """[U-ENERGY] The program load is the one load law over the program's census (1000 B per round)."""
        from unit.deployment_record.load_census_fixtures import written_program

        loads = written_program((500,), (500,), granularity="cell")
        terms, _ = _priced(PHYSICS[ST.HOST], "host", loads=loads, **VALUES)
        assert self.END in terms["energy_programming_mj"].source

    def test_without_the_link_nothing_names_it(self):
        terms, _ = _priced(_physics(host=False), "host", **dict(BASE, carry_in_bytes=0, carry_out_bytes=0))
        assert not any(self.END in t.source for t in terms.values())


@pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
def test_the_step_choice_prices_a_host_sourced_payload_on_the_link(store):
    """The residency step choice's write energy: the payload's transfer at the link and host memory, not e_dma."""
    physics = PHYSICS[store]
    host = price_step(_census(), physics, _policy("host"), static=NO_STATIC)
    own = price_step(_census(), physics, _policy("on_chip"), static=NO_STATIC)
    for corner in CORNERS:
        assert getattr(host.write, corner) - getattr(own.write, corner) == pytest.approx(
            PAYLOAD_BYTES * _link_and_memory(physics, corner) * 1e3, rel=1e-9), corner


class TestTheLoweringPrice:
    """N2's lowering price reads the same rule: its operand regions' host payload on the link beside it, and its
    streamed vectors at the carry's re-injection on the declared store (the chip's ``e_dma_per_byte`` on its own
    store, the link and host memory on a host-owned one)."""

    FORM = OperandForm(batch=(2,), m=3, k=4, n=5, a_perm=(0, 1, 2), b_perm=(0, 1, 2), out_perm=(0, 1, 2))

    def _platform(self, physics, source="host"):
        return {"platform_physics_resolved": physics.to_dict(), "inference_batch_K": 1, "activation_bits": 8,
                "cores": [{"max_axons": 64, "max_neurons": 64, "count": 64, "write_granularity": "cell",
                           "operand": {"payload_bits": 8, "write_source": source}}]}

    @pytest.mark.parametrize("stationary", [B_STATIONARY, AT_STATIONARY])
    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_the_price_is_the_cells_the_payload_on_the_link_and_the_streamed_vectors(self, store, stationary):
        physics, lowering = PHYSICS[store], Lowering(self.FORM, stationary)
        own = WritePolicy(batch_k=1, producer_write_source="on_chip", payload_bits=8, write_granularity="cell")
        cells = price_write_terms(operand_region_census(lowering), physics, own).energy.nominal * 1e-3
        payload = sum(p.written_cells_producer for p in operand_region_census(lowering).passes) * 8 / 8.0
        per_byte = physics.band("e_link_per_byte").nominal + physics.band("e_host_mem_per_byte").nominal
        inject = per_byte if store == ST.HOST else physics.band("e_dma_per_byte").nominal
        expected = cells + payload * per_byte + lowering.streamed_elements * 1.0 * inject
        assert lowering_price(self._platform(physics))(lowering) == pytest.approx(expected, rel=1e-12)

    def test_the_cheaper_lowering_is_still_the_one_streaming_less(self):
        """On a chip declaring ``e_dma_per_byte`` the order stands: both lowerings place the same cells, so the streamed
        vectors order them. [U-J4 targeted, review M1] Where a store's injection energy changes between zero and
        nonzero the choice moves (``tests/unit/mapping/operand/test_lowering_choice_link_uj4.py``)."""
        price = lowering_price(self._platform(PHYSICS[ST.HOST]))
        b, at = Lowering(self.FORM, B_STATIONARY), Lowering(self.FORM, AT_STATIONARY)
        assert (price(b) < price(at)) == (b.streamed_elements < at.streamed_elements)


def test_the_write_crossing_states_the_payload_on_either_store():
    """U-J3's byte count, kept: the payload is the program's crossing (10 B), the store's 400 B / K = 100 B."""
    crossing = HT.write_link_bytes(WriteInputs(_census(), _policy("host")))
    assert (crossing.store, crossing.program) == (100.0, PAYLOAD_BYTES)
    assert HT.write_link_bytes(WriteInputs(_census(), _policy("on_chip"))).program == 0.0


class TestTheLinkTimeOfAHostSourcedPayload:
    """[U-J5 L-5] The payload's transfer time is priced now, in the write law's producer walls (the slower of its cell
    writes and its bytes at the link's stage, plus one transfer per pass; ``test_payload_link_time_uj5.py`` prices it
    by hand). U-J4 named it unpriced here, values equal; that name is retired where the physics declares the rows."""

    NAME = "host-sourced payload"

    @pytest.mark.parametrize("store", [ST.CHIP, ST.HOST])
    def test_the_latency_prices_it(self, store):
        host, _ = _priced(PHYSICS[store], "host")
        own, _ = _priced(PHYSICS[store], "on_chip")
        assert "timed by no term" not in host["e2e_latency_s"].source and self.NAME in host["e2e_latency_s"].source
        for corner in CORNERS:
            assert _c(host["e2e_latency_s"], corner) > _c(own["e2e_latency_s"], corner), corner

    def test_without_the_link_nothing_is_named(self):
        terms, _ = _priced(_physics(host=False), "host", **dict(BASE, carry_in_bytes=0, carry_out_bytes=0))
        assert self.NAME not in terms["e2e_latency_s"].source


class TestAnUndeclaredLinkRowIsNamed:
    """Beside the link a crossing byte whose link or host-memory energy row the physics does not declare (a hand block
    stating part of the link; a host description declares every row, ``REQUIRED_HOST_ROWS``) is named unpriced on the
    headline, never priced on the chip's store rows nor dropped."""

    def _physics(self, drop):
        block = {key: row for key, row in host_description_block(HOST).items() if key not in drop}
        return resolve_platform_physics(GENERIC, merge_override_blocks(
            {"write": WRITE, "host": block, "store": ST.store_block(ST.CHIP, "test")}))

    @pytest.mark.parametrize("drop", [("e_link_per_byte",), ("e_host_mem_per_byte",),
                                      ("e_link_per_byte", "e_host_mem_per_byte")])
    def test_the_headline_names_the_undeclared_row(self, drop):
        physics = self._physics(drop)
        terms, _ = _priced(physics, "host")
        own, _ = _priced(physics, "on_chip")
        source = terms["energy_per_inference_mj"].source
        for key in drop:
            assert f"{key}(the host-chip crossing bytes" in source.rsplit("; unpriced: ", 1)[-1], key
        for corner in CORNERS:  # nothing of the payload lands on the chip's rows
            assert _c(terms["energy_write_mj"], corner) == pytest.approx(_c(own["energy_write_mj"], corner), rel=1e-12)

    def test_a_declared_row_is_never_named(self):
        terms, _ = _priced(PHYSICS[ST.CHIP], "host")
        assert "(the host-chip crossing bytes" not in terms["energy_per_inference_mj"].source
