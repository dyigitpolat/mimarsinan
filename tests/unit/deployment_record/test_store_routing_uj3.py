"""[U-J3 R-1] Crossing bytes follow the declared backing store, priced by hand, corner by corner.

Store traffic (the pass carry, ``carry_in_bytes`` / ``carry_out_bytes``, and the write law's backing-store payload, a
period's ``payload_bytes`` over K) goes to the store the chip declares and is priced by that store's rows only:

- chip-owned: the chip's rows (the carry's re-injection ``e_dma_per_byte`` and read-back ``e_readout_per_byte``, the
  write transfer ``e_dma_per_byte``, the chip's write port), nothing on the host-chip link or host memory;
- host-owned: the link and host memory per byte (``e_link_per_byte`` + ``e_host_mem_per_byte``), the slower of their
  per-byte times and one transfer latency per carried pass boundary (the quantity the candidate plane forms) and per
  written pass, and none of the chip's store rows.

What the program itself sends across the boundary (the segment I/O, host-sourced operand payloads) crosses the link on
either store. Where the physics prices the host-chip link and the chip states no store, store traffic refuses by name;
a host-owned store without the link refuses its store traffic by name; a physics without the link and without a
declaration prices as it always has. The per-load programming terms follow the store too, and the residency step
choice prices its write and carry energy on the declared store."""

from __future__ import annotations

import pytest

from mimarsinan.deployment_record.cost import host_terms as HT
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.context import PricingContext
from mimarsinan.deployment_record.cost.absolute.step_choice import StepChoiceError, carry_energy, price_step
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


def _physics(store=None, host=True, profile=GENERIC):
    blocks = {"host": host_description_block(HOST)} if host else {}
    if store is not None:
        blocks["store"] = ST.store_block(store, "test")
    return resolve_platform_physics(profile, merge_override_blocks(blocks) if blocks else None)


CHIP_STORE, HOST_STORE, NO_STORE = _physics(ST.CHIP), _physics(ST.HOST), _physics()
CHIP_ONLY = _physics(host=False)
BASE = dict(synaptic_events=1e6, cores_physical=8, latency_steps=32, timesteps=4, neurons_used=100, segment_cores=1,
            host_macs=0, compute_op_count=0, host_memory_bound_bytes=0)
CARRY = dict(carry_in_bytes=300, carry_out_bytes=200)
SEGMENT = dict(segment_in_bytes=50, segment_out_bytes=10, host_link_transfers=4)
NO_SEGMENT = dict(segment_in_bytes=0, segment_out_bytes=0, host_link_transfers=0)


def _q(**values):
    return Quantities({key: QuantityValue(float(value), "static") for key, value in values.items()})


def _priced(physics, writes=None, **values):
    pricing = price_absolute(_q(**values), physics, writes=writes)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


def _c(term, corner):
    return getattr(term.band, corner)


def _k(physics, key, corner):
    return getattr(physics.band(key), corner)


def _stage(physics, corner):
    return max(_k(physics, "t_link_per_byte", corner), _k(physics, "t_host_mem_per_byte", corner))


def _census(payload=400, cells=100, carry=(0, 0), producer=0):
    passes = (PassCensus(position=0, segment_index=0, resident=False, cores=1, resident_cores=0,
                         written_cells_backing_store=cells, written_cells_producer=producer, payload_bytes=payload,
                         compute_steps=3),
              PassCensus(position=1, segment_index=0, resident=True, cores=1, resident_cores=1,
                         written_cells_backing_store=0, written_cells_producer=0, payload_bytes=0, compute_steps=3))
    boundary = (PassBoundaryCensus(after_pass=0, carry_out_bytes=carry[0], carry_in_bytes=carry[1]),)
    return PassPeriodCensus(passes=passes, boundaries=boundary)


class TestAChipOwnedStore:
    VALUES = dict(BASE, **CARRY, **SEGMENT)

    def test_the_link_and_host_memory_carry_the_programs_own_bytes_only(self):
        terms, refused = _priced(CHIP_STORE, **self.VALUES)
        assert not refused
        for corner in CORNERS:
            assert _c(terms["energy_host_link_mj"], corner) == pytest.approx(
                60 * _k(CHIP_STORE, "e_link_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["energy_host_mem_mj"], corner) == pytest.approx(
                60 * _k(CHIP_STORE, "e_host_mem_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["latency_host_link_s"], corner) == pytest.approx(
                60 * _stage(CHIP_STORE, corner) + 4 * _k(CHIP_STORE, "t_link_transfer", corner), rel=1e-12)

    def test_the_carry_is_priced_on_the_chips_own_store_rows(self):
        terms, _ = _priced(CHIP_STORE, **self.VALUES)
        chip, _ = _priced(CHIP_ONLY, **self.VALUES)
        bare, _ = _priced(CHIP_ONLY, **dict(self.VALUES, carry_in_bytes=0, carry_out_bytes=0))
        for corner in CORNERS:
            assert _c(terms["energy_dynamic_mj"], corner) == pytest.approx(_c(chip["energy_dynamic_mj"], corner),
                                                                          rel=1e-12)
            carry = (300 * _k(CHIP_ONLY, "e_dma_per_byte", corner) + 200 * _k(CHIP_ONLY, "e_readout_per_byte", corner))
            assert _c(chip["energy_dynamic_mj"], corner) - _c(bare["energy_dynamic_mj"], corner) == pytest.approx(
                carry * 1e3, rel=1e-9)

    def test_the_latency_adds_the_programs_crossing_only(self):
        terms, _ = _priced(CHIP_STORE, **self.VALUES)
        chip, _ = _priced(CHIP_ONLY, **self.VALUES)
        for corner in CORNERS:
            assert _c(terms["e2e_latency_s"], corner) == pytest.approx(
                _c(chip["e2e_latency_s"], corner) + _c(terms["latency_host_link_s"], corner), rel=1e-12)


class TestAHostOwnedStore:
    VALUES = dict(BASE, **CARRY, **SEGMENT)

    def test_the_link_and_host_memory_carry_the_store_traffic_too(self):
        terms, refused = _priced(HOST_STORE, **self.VALUES)
        assert not refused
        for corner in CORNERS:
            assert _c(terms["energy_host_link_mj"], corner) == pytest.approx(
                560 * _k(HOST_STORE, "e_link_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["energy_host_mem_mj"], corner) == pytest.approx(
                560 * _k(HOST_STORE, "e_host_mem_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["latency_host_link_s"], corner) == pytest.approx(
                560 * _stage(HOST_STORE, corner) + 4 * _k(HOST_STORE, "t_link_transfer", corner), rel=1e-12)

    def test_the_chips_store_rows_price_none_of_the_carry(self):
        terms, _ = _priced(HOST_STORE, **self.VALUES)
        bare, _ = _priced(CHIP_ONLY, **dict(self.VALUES, carry_in_bytes=0, carry_out_bytes=0))
        for corner in CORNERS:
            assert _c(terms["energy_dynamic_mj"], corner) == pytest.approx(_c(bare["energy_dynamic_mj"], corner),
                                                                          rel=1e-12)
        assert "carry_inject: e_dma_per_byte" not in terms["energy_dynamic_mj"].source
        assert "host-owned" in terms["energy_host_link_mj"].source

    def test_one_crossing_is_priced_once_on_either_store(self):
        """The defect U-J3 fixes: the carry priced on the chip's store rows and on the link."""
        for physics in (CHIP_STORE, HOST_STORE):
            terms, _ = _priced(physics, **self.VALUES)
            link = HT.link_bytes(PricingContext.create(_q(**self.VALUES), physics))[0]
            chip_carry = "carry_readout: e_readout_per_byte" in terms["energy_dynamic_mj"].source
            assert (link == 560) != chip_carry and link in (60, 560)


class TestNoStoreDeclared:
    def test_store_traffic_beside_the_host_link_refuses_by_name(self):
        terms, refused = _priced(NO_STORE, **dict(BASE, **CARRY, **SEGMENT))
        for name in ("e2e_latency_s", "energy_per_inference_mj", "throughput_inferences_s"):
            assert name not in terms and name in refused, name
        for name in ("e2e_latency_s", "energy_per_inference_mj"):  # throughput names its root, the latency
            assert "backing_store" in refused[name] and "pass carry" in refused[name], name

    def test_without_store_traffic_it_prices_as_before(self):
        terms, refused = _priced(NO_STORE, **dict(BASE, carry_in_bytes=0, carry_out_bytes=0, **SEGMENT))
        assert not refused
        for corner in CORNERS:
            assert _c(terms["energy_host_link_mj"], corner) == pytest.approx(
                60 * _k(NO_STORE, "e_link_per_byte", corner) * 1e3, rel=1e-12)

    def test_without_the_host_link_nothing_changes(self):
        terms, refused = _priced(CHIP_ONLY, **dict(BASE, **CARRY))
        assert not refused and "carry_inject: e_dma_per_byte" in terms["energy_dynamic_mj"].source
        assert not {"energy_host_link_mj", "latency_host_link_s"} & set(terms)


def test_a_host_owned_store_without_the_host_link_refuses_its_store_traffic_by_name():
    physics = _physics(ST.HOST, host=False)
    terms, refused = _priced(physics, **dict(BASE, **CARRY))
    assert "e2e_latency_s" not in terms and "host-chip link" in refused["e2e_latency_s"]
    assert "energy_per_inference_mj" not in terms
    terms, refused = _priced(physics, **dict(BASE, carry_in_bytes=0, carry_out_bytes=0))
    assert not refused and "e2e_latency_s" in terms


class TestTheWriteLaw:
    """The period's payload (400 B, K = 4: 100 B per inference) on one written pass of a two-pass period."""

    POLICY = WritePolicy(batch_k=4)
    VALUES = dict(BASE, carry_in_bytes=0, carry_out_bytes=0, **NO_SEGMENT)

    def _writes(self):
        return WriteInputs(_census(), self.POLICY)

    def test_a_chip_owned_store_prices_the_write_on_the_chip_alone(self):
        terms, refused = _priced(CHIP_STORE, writes=self._writes(), **self.VALUES)
        chip, refused_chip = _priced(CHIP_ONLY, writes=self._writes(), **self.VALUES)
        assert refused == refused_chip and "energy_host_link_mj" not in terms  # the lifetime's own refusal on both
        for corner in CORNERS:
            for name in ("energy_write_mj", "latency_write_exposed_s"):
                assert _c(terms[name], corner) == pytest.approx(_c(chip[name], corner), rel=1e-12), name

    def test_a_host_owned_store_prices_the_transfer_on_the_link_and_host_memory(self):
        terms, refused = _priced(HOST_STORE, writes=self._writes(), **self.VALUES)
        chip, refused_chip = _priced(CHIP_ONLY, writes=self._writes(), **self.VALUES)
        assert refused == refused_chip
        for corner in CORNERS:
            assert _c(terms["energy_write_mj"], corner) == pytest.approx(
                _c(chip["energy_write_mj"], corner) - 100 * _k(CHIP_ONLY, "e_dma_per_byte", corner) * 1e3, rel=1e-9)
            assert _c(terms["energy_host_link_mj"], corner) == pytest.approx(
                100 * _k(HOST_STORE, "e_link_per_byte", corner) * 1e3, rel=1e-12)
            assert _c(terms["energy_host_mem_mj"], corner) == pytest.approx(
                100 * _k(HOST_STORE, "e_host_mem_per_byte", corner) * 1e3, rel=1e-12)
        assert "host-owned" in terms["energy_write_mj"].source

    def test_its_written_pass_streams_at_the_slower_stage_and_pays_one_transfer(self):
        terms, _ = _priced(HOST_STORE, writes=self._writes(), **self.VALUES)
        for corner in CORNERS:  # serial exposure: the period's walls over K
            wall = 400 * max(_k(HOST_STORE, "t_program_per_byte", corner), _stage(HOST_STORE, corner))
            assert _c(terms["latency_write_exposed_s"], corner) == pytest.approx(
                (wall + _k(HOST_STORE, "t_link_transfer", corner)) / 4, rel=1e-12)

    def test_no_store_beside_the_link_refuses_the_write_traffic_by_name(self):
        terms, refused = _priced(NO_STORE, writes=self._writes(), **self.VALUES)
        assert "energy_per_inference_mj" not in terms and "e2e_latency_s" not in terms
        assert "backing_store" in refused["e2e_latency_s"] and "payload" in refused["e2e_latency_s"]

    def test_a_host_sourced_operand_payload_crosses_the_link_on_either_store(self):
        policy = WritePolicy(batch_k=4, producer_write_source="host", payload_bits=8)
        writes = WriteInputs(_census(producer=10), policy)
        crossing = HT.write_link_bytes(writes)
        assert (crossing.store, crossing.program) == (100.0, 10.0)
        for physics, total in ((CHIP_STORE, 10.0), (HOST_STORE, 110.0)):
            ctx = PricingContext.create(_q(**self.VALUES), physics)
            ctx.write_link_bytes = crossing
            assert HT.link_bytes(ctx)[0] == total


class TestTheProgrammingTerms:
    VALUES = dict(BASE, carry_in_bytes=0, carry_out_bytes=0, reprogrammed_bytes=1000, **NO_SEGMENT)

    def test_a_chip_owned_store_loads_the_program_over_the_chips_channel(self):
        terms, _ = _priced(CHIP_STORE, **self.VALUES)
        chip, _ = _priced(CHIP_ONLY, **self.VALUES)
        for name in ("energy_programming_mj", "latency_programming_s"):
            for corner in CORNERS:
                assert _c(terms[name], corner) == pytest.approx(_c(chip[name], corner), rel=1e-12), name

    def test_a_host_owned_store_loads_it_over_the_link(self):
        terms, _ = _priced(HOST_STORE, **self.VALUES)
        for corner in CORNERS:
            energy = 1000 * (_k(HOST_STORE, "e_link_per_byte", corner) + _k(HOST_STORE, "e_host_mem_per_byte", corner)
                             + _k(HOST_STORE, "e_program_per_byte", corner)) * 1e3
            assert _c(terms["energy_programming_mj"], corner) == pytest.approx(energy, rel=1e-12)
            time = 1000 * max(_k(HOST_STORE, "t_program_per_byte", corner), _stage(HOST_STORE, corner))
            assert _c(terms["latency_programming_s"], corner) == pytest.approx(time, rel=1e-12)
        assert "e_dma_per_byte" not in terms["energy_programming_mj"].source


class TestTheStepChoice:
    def test_the_carry_is_priced_on_the_declared_store(self):
        census = _census(carry=(300, 200))
        for corner in CORNERS:
            assert getattr(carry_energy(census, CHIP_STORE), corner) == pytest.approx(
                getattr(carry_energy(census, CHIP_ONLY), corner), rel=1e-12)
            per_byte = _k(HOST_STORE, "e_link_per_byte", corner) + _k(HOST_STORE, "e_host_mem_per_byte", corner)
            assert getattr(carry_energy(census, HOST_STORE), corner) == pytest.approx(500 * per_byte * 1e3, rel=1e-12)

    def test_no_store_beside_the_link_refuses_a_carrying_step_by_name(self):
        with pytest.raises(StepChoiceError, match="backing_store"):
            carry_energy(_census(carry=(300, 200)), NO_STORE)

    def test_the_write_transfer_of_a_host_owned_store_is_the_link_and_host_memory(self):
        policy = WritePolicy(batch_k=4)
        step = price_step(_census(), HOST_STORE, policy)
        law = price_write_terms(_census(), HOST_STORE, policy).energy
        for corner in CORNERS:
            per_byte = _k(HOST_STORE, "e_link_per_byte", corner) + _k(HOST_STORE, "e_host_mem_per_byte", corner)
            assert getattr(step.write, corner) == pytest.approx(getattr(law, corner) + 100 * per_byte * 1e3,
                                                                rel=1e-12)
        chip = price_step(_census(), CHIP_STORE, policy)
        assert chip.write == price_write_terms(_census(), CHIP_STORE, policy).energy


def test_the_study_chips_store_is_their_own_dram_on_the_edge_host():
    """The study's own design: the main reading prices carry and writes on the chip's DDR channels only."""
    physics = _physics(profile="study_compass")
    assert physics.backing_store.store == ST.CHIP
    terms, refused = _priced(physics, **dict(BASE, **CARRY, **SEGMENT))
    assert not {"e2e_latency_s", "energy_per_inference_mj"} & set(refused)  # (no cell census: the area's refusal)
    for corner in CORNERS:
        assert _c(terms["energy_host_link_mj"], corner) == pytest.approx(
            60 * _k(physics, "e_link_per_byte", corner) * 1e3, rel=1e-12)
