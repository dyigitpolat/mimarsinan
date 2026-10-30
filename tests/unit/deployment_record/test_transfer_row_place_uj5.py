"""[U-J5 L-1, U-J4 review m1] The chip description declares where its per-byte transfer row prices, priced by hand.

A chip's per-byte transfer rows (``e_dma_per_byte`` in, ``e_readout_per_byte`` out) price either its backing store (the
vocabulary's meaning, and the default for a row whose place is not declared) or a chip-internal buffer that every
arriving byte passes (``transfer_rows``: ``{"<row>": {"place": "chip_buffer", <source>}}``). U-J4's link-crossing rule
drops only a backing-store row: a byte that crosses the host-chip link pays the link and host memory and none of the
chip's backing-store rows, while a chip-internal buffer row is paid by every byte that enters that buffer, crossing or
not (U-J3's host-owned store drops only backing-store rows the same way: the carry's, the write transfer's, a program
load's). isaac_like declares its ``e_dma_per_byte`` a chip buffer from its own text (a byte read through the tile's
eDRAM buffer); every other shipped profile, the three study chips among them, declares none and prices as before.
Without the link a chip-buffer row prices exactly as the chip's one channel row did."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.step_choice import carry_energy, price_step
from mimarsinan.deployment_record.cost.absolute.writes import WriteInputs, price_write_terms
from mimarsinan.deployment_record.platform_physics import available_profiles
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import host_description_block
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.operand_form import AT_STATIONARY, B_STATIONARY, Lowering, OperandForm
from mimarsinan.mapping.support.operand_price import lowering_price
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus

import mimarsinan.deployment_record.platform_physics as PP

CORNERS = ("low", "nominal", "high")
HOST = "jetson_orin_nano_8gb"
GENERIC, ISAAC = "generic_estimated_22nm", "isaac_like"
STUDY = ("study_compass", "study_isaac_like", "study_vr_like")
FIELD, BUFFER = "transfer_rows", "chip_buffer"
DMA, READOUT = "e_dma_per_byte", "e_readout_per_byte"
PROFILES = Path(PP.__file__).resolve().parent / "profiles"
#: A commit energy for the producer cells (isaac_like publishes none: an inference architecture).
COMMIT = {"e_write_per_cell": {"nominal": 1.0, "unit": "pJ", "evidence_kind": "estimated", "note": "test constant"}}
WRITE = {"t_write_per_unit": {"nominal": 10.0, "unit": "ns", "evidence_kind": "estimated", "note": "test constant"}}
PAYLOAD_BYTES = 10.0  # 10 producer cells x 8 payload bits per inference
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
            host_macs=0, compute_op_count=0, host_memory_bound_bytes=0, segment_in_bytes=50, segment_out_bytes=10,
            host_link_transfers=4)


def _resolved(profile, host=True, store=None, extra=None):
    blocks = dict(extra or {})
    if host:
        blocks["host"] = host_description_block(HOST)
    if store is not None:
        blocks["store"] = ST.store_block(store, "test")
    return resolve_platform_physics(profile, merge_override_blocks(blocks) if blocks else None)


def _with_places(physics, places):
    """``physics`` with a ``transfer_rows`` declaration (as a record's sealed physics carries it), or the refusal."""
    try:
        return PlatformPhysics.from_dict({**physics.to_dict(), FIELD: places}), None
    except Exception as exc:  # noqa: BLE001 - the refusal (or the base's unknown field) is the test's subject
        return None, exc


def _buffered(physics, rows=(DMA, READOUT)):
    places = {row: {"place": BUFFER, "note": "test: a chip-internal buffer every arriving byte passes"} for row in rows}
    built, error = _with_places(physics, places)
    assert built is not None, f"a {FIELD} declaration does not load: {error}"
    return built


def _without_places(physics):
    data = {key: value for key, value in physics.to_dict().items() if key != FIELD}
    return PlatformPhysics.from_dict(data)


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _policy(source="host"):
    return WritePolicy(batch_k=4, producer_write_source=source, payload_bits=8, write_granularity="cell")


def _census(payload=0, cells=0, producer=10, carry=(0, 0)):
    passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=cells, written_cells_producer=producer, payload_bytes=payload,
                         compute_steps=3),
              PassCensus(position=1, segment_index=0, resident=True, cores=1, resident_cores=1,
                         written_cells_backing_store=0, written_cells_producer=0, payload_bytes=0, compute_steps=3))
    boundary = (PassBoundaryCensus(after_pass=0, carry_out_bytes=carry[0], carry_in_bytes=carry[1]),)
    return PassPeriodCensus(passes=passes, boundaries=boundary)


def _terms(physics, census, source="host", **values):
    pricing = price_absolute(_q(**{**BASE, "carry_in_bytes": 0, "carry_out_bytes": 0, **values}), physics,
                             writes=WriteInputs(census, _policy(source)))
    return {t.name: t for t in pricing.terms}


def _k(physics, key, corner):
    return getattr(physics.band(key), corner)


class TestTheDeclaration:
    """isaac_like declares its transfer row's place from its own text; no other shipped profile declares one."""

    def test_isaac_like_declares_its_dma_row_a_chip_buffer_from_its_own_text(self):
        data = json.loads((PROFILES / f"{ISAAC}.json").read_text())
        declared = (data.get(FIELD) or {}).get(DMA) or {}
        assert declared.get("place") == BUFFER, data.get(FIELD)
        assert declared.get("citation") == data["constants"][DMA]["citation"], declared
        assert "eDRAM buffer" in declared.get("note", ""), declared
        assert "eDRAM buffer" in (PROFILES / f"{ISAAC}.md").read_text().split(FIELD, 1)[-1], "the md states it"

    def test_the_resolved_physics_carries_it_so_a_record_reprices_alike(self):
        physics = _resolved(ISAAC, host=False)
        sealed = physics.to_dict()
        assert (sealed.get(FIELD) or {}).get(DMA, {}).get("place") == BUFFER, sealed.get(FIELD)
        assert PlatformPhysics.from_dict(sealed).to_dict() == sealed

    def test_every_other_shipped_profile_declares_none(self):
        for profile in available_profiles():
            if profile == ISAAC:
                continue
            assert FIELD not in json.loads((PROFILES / f"{profile}.json").read_text()), profile
            assert FIELD not in _resolved(profile, host=False).to_dict(), profile


class TestABufferRowBesideTheEdgeHost:
    """isaac_like beside the edge host: a host-sourced payload pays the link and host memory (U-J4) and the tile's
    buffer row too, since every arriving byte passes the buffer."""

    PHYSICS = _resolved(ISAAC, extra={"commit": COMMIT})
    ALONE = _resolved(ISAAC, host=False, extra={"commit": COMMIT})

    def test_the_write_term_pays_the_buffer_row_on_the_payload(self):
        host = price_write_terms(_census(), self.PHYSICS, _policy("host")).energy
        own = price_write_terms(_census(), self.PHYSICS, _policy("on_chip")).energy
        for corner in CORNERS:
            assert getattr(host, corner) - getattr(own, corner) == pytest.approx(
                PAYLOAD_BYTES * _k(self.PHYSICS, DMA, corner) * 1e3, rel=1e-9), corner
        assert "chip-internal buffer" in host.basis and "crosses the host-chip link" in host.basis, host.basis

    def test_beside_the_link_it_is_the_same_charge_as_without_it(self):
        beside = price_write_terms(_census(), self.PHYSICS, _policy("host")).energy
        alone = price_write_terms(_census(), self.ALONE, _policy("host")).energy
        for corner in CORNERS:
            assert getattr(beside, corner) == pytest.approx(getattr(alone, corner), rel=1e-12), corner

    def test_the_link_and_host_memory_still_price_the_payload_once(self):
        host, own = _terms(self.PHYSICS, _census(), "host"), _terms(self.PHYSICS, _census(), "on_chip")
        for key, name in (("e_link_per_byte", "energy_host_link_mj"), ("e_host_mem_per_byte", "energy_host_mem_mj")):
            for corner in CORNERS:
                assert getattr(host[name].band, corner) - getattr(own[name].band, corner) == pytest.approx(
                    PAYLOAD_BYTES * _k(self.PHYSICS, key, corner) * 1e3, rel=1e-9), (name, corner)

    def test_the_step_choice_pays_it_too(self):
        host = price_step(_census(), self.PHYSICS, _policy("host")).write
        own = price_step(_census(), self.PHYSICS, _policy("on_chip")).write
        for corner in CORNERS:
            link = _k(self.PHYSICS, "e_link_per_byte", corner) + _k(self.PHYSICS, "e_host_mem_per_byte", corner)
            assert getattr(host, corner) - getattr(own, corner) == pytest.approx(
                PAYLOAD_BYTES * (link + _k(self.PHYSICS, DMA, corner)) * 1e3, rel=1e-9), corner

    def test_without_the_link_the_declaration_changes_nothing(self):
        """Every term of isaac_like without the link, values and sources, equals the same physics undeclared."""
        census = _census(payload=400, cells=100)
        declared, plain = self.ALONE, _without_places(self.ALONE)
        values = dict(carry_in_bytes=300, carry_out_bytes=200, reprogrammed_bytes=1000)
        a = _terms(declared, census, "host", **values)
        b = _terms(plain, census, "host", **values)
        assert a.keys() == b.keys()
        for name, term in a.items():
            assert (term.band.low, term.band.nominal, term.band.high, term.source) == (
                b[name].band.low, b[name].band.nominal, b[name].band.high, b[name].source), name


@pytest.mark.parametrize("profile", STUDY + (GENERIC,))
def test_a_backing_store_row_beside_the_link_still_prices_no_crossing_byte(profile):
    """The study chips and the generic profile declare no place: their row is the backing store's, as before."""
    physics = _resolved(profile, store=ST.CHIP if profile == GENERIC else None, extra={"write": WRITE})
    host = price_write_terms(_census(), physics, _policy("host")).energy
    own = price_write_terms(_census(), physics, _policy("on_chip")).energy
    for corner in CORNERS:
        assert getattr(host, corner) == pytest.approx(getattr(own, corner), rel=1e-12), (profile, corner)
    assert "chip-internal buffer" not in host.basis


class TestABufferRowOnAHostOwnedStore:
    """U-J3's host-owned store drops the chip's store rows of its traffic; a chip-buffer row stays, paid beside the
    link and host memory: the carry both ways, the write transfer, a program load, the step choice's carry and the
    lowering's injected vectors (generic_estimated_22nm with both transfer rows declared chip buffers)."""

    PLAIN = _resolved(GENERIC, store=ST.HOST, extra={"write": WRITE})
    VALUES = dict(carry_in_bytes=300, carry_out_bytes=200)

    def _pair(self, census, **values):
        return _terms(_buffered(self.PLAIN), census, "on_chip", **values), _terms(self.PLAIN, census, "on_chip",
                                                                                   **values)

    def test_the_carry_pays_the_buffer_rows_and_the_link(self):
        buffered, plain = self._pair(_census(producer=0, carry=(200, 300)), **self.VALUES)
        for corner in CORNERS:
            rows = 300 * _k(self.PLAIN, DMA, corner) + 200 * _k(self.PLAIN, READOUT, corner)
            assert getattr(buffered["energy_dynamic_mj"].band, corner) - getattr(
                plain["energy_dynamic_mj"].band, corner) == pytest.approx(rows * 1e3, rel=1e-9), corner
            for name in ("energy_host_link_mj", "energy_host_mem_mj"):
                assert getattr(buffered[name].band, corner) == getattr(plain[name].band, corner), (name, corner)

    def test_the_write_transfer_pays_the_buffer_row(self):
        census = _census(payload=400, cells=100, producer=0)
        buffered = price_write_terms(census, _buffered(self.PLAIN), _policy("on_chip")).energy
        plain = price_write_terms(census, self.PLAIN, _policy("on_chip")).energy
        for corner in CORNERS:
            assert getattr(buffered, corner) - getattr(plain, corner) == pytest.approx(
                400 / 4 * _k(self.PLAIN, DMA, corner) * 1e3, rel=1e-9), corner

    def test_a_program_load_pays_the_buffer_row(self):
        buffered, plain = self._pair(_census(producer=0), reprogrammed_bytes=1000)
        for corner in CORNERS:
            assert getattr(buffered["energy_programming_mj"].band, corner) - getattr(
                plain["energy_programming_mj"].band, corner) == pytest.approx(
                    1000 * _k(self.PLAIN, DMA, corner) * 1e3, rel=1e-9), corner

    def test_the_step_choices_carry_pays_the_buffer_rows(self):
        census = _census(producer=0, carry=(200, 300))
        buffered, plain = carry_energy(census, _buffered(self.PLAIN)), carry_energy(census, self.PLAIN)
        for corner in CORNERS:
            rows = 300 * _k(self.PLAIN, DMA, corner) + 200 * _k(self.PLAIN, READOUT, corner)
            assert getattr(buffered, corner) - getattr(plain, corner) == pytest.approx(rows * 1e3, rel=1e-9), corner

    def test_the_lowerings_injected_vectors_pay_the_buffer_row(self):
        form = OperandForm(batch=(2,), m=3, k=4, n=5, a_perm=(0, 1, 2), b_perm=(0, 1, 2), out_perm=(0, 1, 2))

        def platform(physics):
            return {"platform_physics_resolved": physics.to_dict(), "inference_batch_K": 1, "activation_bits": 8,
                    "cores": [{"max_axons": 64, "max_neurons": 64, "count": 64, "write_granularity": "cell",
                               "operand": {"payload_bits": 8, "write_source": "on_chip"}}]}
        buffered, plain = lowering_price(platform(_buffered(self.PLAIN))), lowering_price(platform(self.PLAIN))
        for stationary in (B_STATIONARY, AT_STATIONARY):
            lowering = Lowering(form, stationary)
            assert buffered(lowering) - plain(lowering) == pytest.approx(
                lowering.streamed_elements * 1.0 * self.PLAIN.band(DMA).nominal, rel=1e-9), stationary


class TestARefusedDeclaration:
    """A declaration outside its closed form refuses by name."""

    @pytest.mark.parametrize("places, why", [
        ({DMA: {"place": "somewhere", "note": "x"}}, "backing_store"),
        ({"e_mac": {"place": BUFFER, "note": "x"}}, "e_readout_per_byte"),
        ({DMA: {"place": BUFFER, "note": "x", "colour": "red"}}, "colour"),
        ({DMA: {"place": BUFFER}}, "source"),
        ({DMA: "chip_buffer"}, "place"),
    ])
    def test_an_ill_formed_declaration_refuses(self, places, why):
        built, error = _with_places(_resolved(GENERIC, host=False), places)
        assert built is None and FIELD in str(error) and why in str(error), error
        assert "unknown fields" not in str(error), error  # refused for what it states, not as an unknown field

    def test_a_row_the_chip_does_not_declare_refuses(self):
        loihi = _resolved("loihi", host=False)
        assert not loihi.has(DMA)
        built, error = _with_places(loihi, {DMA: {"place": BUFFER, "note": "x"}})
        assert built is None and FIELD in str(error) and "declares no" in str(error), error
        assert "unknown fields" not in str(error), error
