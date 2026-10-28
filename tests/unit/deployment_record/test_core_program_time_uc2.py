"""[U-C2 G-1, survey gap B6.2] The programming time of a chip that writes a whole core image per reprogrammed core.

ODIN's SPI writes every word of a core's program image on each load, whatever the program occupies (8,192 synapse
words and 256 neuron words, 36,864 byte transactions), so its programming wall is a per-core constant: the time twin
of ``e_core_program``, ``t_core_program`` x reprogrammed cores. A per-byte wall on the occupied cells (the donor-scaled
5.09 to 1,272.73 ns per byte, whose every corner lies below the SPI frame's own floor) underprices a partly used core
and is gone from the profile. The constant is derived from the published frame (40 SCK per byte, SCK at most CLK / 4:
at least 160 clock cycles per byte) and the image size; the card's measured programming run is printed beside it as
a check, never as the declaration. The time twin is read wherever the programming wall is: the per-load term, the
write law's per-pass walls, and through them Pi's numerator, M's period writes and the write-bandwidth batch.
Every number below is a test constant or read off the shipped profile.
"""

from __future__ import annotations

import json

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.deployment_record.cost.absolute import price_absolute, price_write_terms
from mimarsinan.deployment_record.platform_physics import get_platform_physics, profile_from_dict
from mimarsinan.deployment_record.platform_physics.constants import PHYSICS_CONSTANTS, spec_for
from mimarsinan.deployment_record.platform_physics.registry import profile_description_path
from mimarsinan.deployment_record.platform_physics.vocabulary import decision_class_of
from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import constant
from unit.deployment_record.introspection.test_ladder_tile_writes import TILES, _physics, _row
from unit.deployment_record.introspection.test_scaling_ladder import CONSTANTS, READING, _facts, _network, _platform

#: The stock ODIN core image (8,192 x 32-bit synapse words + 256 x 128-bit neuron words), its SPI frame floor
#: (40 SCK per transaction, SCK at most CLK / 4) and the profile's declared clock.
IMAGE_BYTES, FLOOR_CYCLES_PER_BYTE, F_CLK_HZ = 8192 * 4 + 256 * 16, 40 * 4, 75e6
#: The card's programming run of one stock image (MEASURED, quoted in the profile as its check).
CARD_CYCLES, CARD_IMAGE_BYTES = 6_345_255, 36_864
ODIN_WALL_S = IMAGE_BYTES * FLOOR_CYCLES_PER_BYTE / F_CLK_HZ


def _odin():
    return get_platform_physics("odin")


def _q(**values: float) -> Quantities:
    return Quantities({k: QuantityValue(float(v), "static") for k, v in values.items()})


def _terms(physics, quantities):
    pricing = price_absolute(quantities, physics)
    return {t.name: t for t in pricing.terms}, {r.name: r.reason for r in pricing.refusals}


class TestTheVocabularyRow:
    def test_the_time_twin_of_the_per_core_programming_energy(self):
        spec = spec_for("t_core_program")
        twin = PHYSICS_CONSTANTS["e_core_program"]
        assert (spec.group, spec.dimension, spec.multiplicand) == ("programming", "time", twin.multiplicand)
        assert spec.multiplicand == "reprogrammed_cores"

    def test_the_row_states_the_whole_image_rule_and_its_readers(self):
        doc = spec_for("t_core_program").doc
        assert "whole" in doc and "t_program_per_byte" in doc and "write law" in doc

    def test_it_is_a_priced_constant(self):
        assert decision_class_of("t_core_program")[0] == "cost_only"


class TestTheOdinDeclaration:
    def test_odin_declares_a_per_core_wall_and_no_per_byte_wall(self):
        physics = _odin()
        assert physics.has("t_core_program")
        assert not physics.has("t_program_per_byte")

    def test_the_wall_is_the_spi_frame_floor_over_the_whole_image_at_the_declared_clock(self):
        band = _odin().band("t_core_program")
        assert IMAGE_BYTES == 36_864
        for corner in (band.low, band.nominal, band.high):
            assert corner == pytest.approx(ODIN_WALL_S, rel=1e-9)

    def test_the_derivation_names_the_frame_the_clock_bound_and_the_image(self):
        value = _odin().constants["t_core_program"]
        assert value.evidence_kind == "derived"
        text = f"{value.citation} {value.derivation}"
        for token in ("ChFrenkel/ODIN", "1781931", "spi_slave.v", "40 SCK", "CLK/4", "160", "36,864", "75 MHz"):
            assert token in text, token

    def test_the_wall_is_not_donor_scaled(self):
        value = _odin().constants["t_core_program"]
        for donor in ("truenorth", "generic_estimated_22nm", "technology_node_nm", "donor"):
            assert donor not in value.derivation, donor

    def test_the_card_measurement_is_printed_beside_it_as_a_check(self):
        value = _odin().constants["t_core_program"]
        assert "MEASURED" in value.note and "6,345,255" in value.note and "100 MHz" in value.note
        cycles_per_byte = CARD_CYCLES / CARD_IMAGE_BYTES
        # The card's sequencer adds register writes to the frames: at or above the floor, within a tenth of it.
        assert FLOOR_CYCLES_PER_BYTE <= cycles_per_byte <= 1.1 * FLOOR_CYCLES_PER_BYTE

    def test_the_stale_single_reader_note_is_gone(self):
        raw = json.loads(profile_description_path("odin").with_suffix(".json").read_text(encoding="utf-8"))
        text = json.dumps(raw)
        assert "Only the amortized latency_programming_s term reads it" not in text
        note = raw["constants"]["t_core_program"]["note"]
        for reader in ("latency_programming_s", "write law", "Pi", "bandwidth"):
            assert reader in note, reader

    def test_the_description_states_the_new_wall(self):
        text = profile_description_path("odin").read_text(encoding="utf-8")
        assert "t_core_program" in text and "36,864" in text
        assert "5.09 / 285.4 / 1272.7" not in text


class TestThePerLoadTerm:
    def test_an_odin_load_is_priced_whole_whatever_the_occupancy(self):
        sparse = _terms(_odin(), _q(reprogrammed_bytes=100, reprogrammed_cores=3))[0]["latency_programming_s"]
        dense = _terms(_odin(), _q(reprogrammed_bytes=3 * 32_768, reprogrammed_cores=3))[0]["latency_programming_s"]
        assert sparse.value == dense.value == pytest.approx(3 * ODIN_WALL_S, rel=1e-9)
        assert "t_core_program" in sparse.source and "reprogrammed_cores" in sparse.source

    def test_a_chip_declaring_both_walls_pays_both(self):
        physics = profile_from_dict({
            "format_version": 1, "name": "both", "display_name": "both", "description_file": "b.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {"t_program_per_byte": constant(2.0, unit="ns"), "t_core_program": constant(3.0, unit="us")}})
        term = _terms(physics, _q(reprogrammed_bytes=1000, reprogrammed_cores=4))[0]["latency_programming_s"]
        assert term.value == pytest.approx(1000 * 2e-9 + 4 * 3e-6, rel=1e-12)
        assert "t_program_per_byte" in term.source and "t_core_program" in term.source

    def test_a_withheld_core_count_is_named_and_never_drops_the_priced_bytes(self):
        """A record whose reprogrammed cores are withheld by name (U-C C-6's unmarked operand chip): the per-byte part
        still prices and the per-core part is named unpriced with the reason, as the per-load energy term does."""
        physics = profile_from_dict({
            "format_version": 1, "name": "both", "display_name": "both", "description_file": "b.md",
            "validity": {"measurement_kind": "projection"},
            "constants": {"t_program_per_byte": constant(2.0, unit="ns"), "t_core_program": constant(3.0, unit="us")}})
        reason = "the cores holding a static program are not determined"
        quantities = Quantities({"reprogrammed_bytes": QuantityValue(1000.0, "measured")},
                                withheld={"reprogrammed_cores": reason})
        term = _terms(physics, quantities)[0]["latency_programming_s"]
        assert term.value == pytest.approx(1000 * 2e-9, rel=1e-12)
        assert term.source.endswith(f"; unpriced: t_core_program(core_program; {reason})")

    def test_a_withheld_core_count_refuses_the_per_core_wall_by_name_where_it_is_the_only_part(self):
        """[targeted M1] ODIN declares t_core_program and no per-byte wall: with its reprogrammed cores withheld by name
        (a sealed core of no declared type, an unmarked operand chip) nothing of the per-load wall can be priced, so the
        term is refused by name with the withheld reason, as its energy twin is, never dropped without a word."""
        reason = "the cores holding a static program are not determined"
        quantities = Quantities({"reprogrammed_bytes": QuantityValue(1000.0, "measured")},
                                withheld={"reprogrammed_cores": reason})
        terms, refusals = _terms(_odin(), quantities)
        assert "latency_programming_s" not in terms
        assert refusals["latency_programming_s"] == (
            f"t_core_program is declared but its multiplicand reprogrammed_cores is withheld: {reason}")
        assert "energy_programming_mj" in terms  # the per-byte commit energy prices; e_core_program is named unpriced
        assert f"e_core_program(core_program; {reason})" in terms["energy_programming_mj"].source

    def test_a_per_byte_chip_keeps_its_term_as_it_was(self):
        loihi = get_platform_physics("loihi")
        term = _terms(loihi, _q(reprogrammed_bytes=1000, reprogrammed_cores=4))[0]["latency_programming_s"]
        assert term.value == pytest.approx(loihi.band("t_program_per_byte").nominal * 1000, rel=1e-12)
        assert term.source.startswith("t_program_per_byte [") and "t_core_program" not in term.source


def _pass(position, *, cores, cells, operand_only=0, steps=10):
    return PassCensus(position=position, segment_index=0, resident=False, cores=cores, resident_cores=0,
                      written_cells_backing_store=cells, written_cells_producer=0, payload_bytes=cells // 2,
                      compute_steps=steps, operand_only_cores=operand_only)


def _exposed(census, physics, k=1):
    policy = WritePolicy(batch_k=k, core_capacity=(((256, 256), 64),))
    pricing = price_write_terms(census, physics, policy)
    return ({t.name: t for t in pricing.terms}.get("latency_write_exposed_s"),
            {r.name: r.reason for r in pricing.refusals})


class TestTheWriteLaw:
    @pytest.mark.parametrize("k", [1, 16, 64])
    def test_each_rewritten_core_takes_the_whole_image_wall(self, k):
        census = PassPeriodCensus(passes=(_pass(0, cores=3, cells=900), _pass(1, cores=2, cells=50)))
        term, refusals = _exposed(census, _odin(), k)
        assert term is not None, refusals
        assert term.value == pytest.approx(5 * ODIN_WALL_S / k, rel=1e-9)

    def test_the_wall_does_not_follow_the_occupied_cells(self):
        full = PassPeriodCensus(passes=(_pass(0, cores=4, cells=4 * 65_536),))
        sparse = PassPeriodCensus(passes=(_pass(0, cores=4, cells=40),))
        assert _exposed(full, _odin())[0].value == _exposed(sparse, _odin())[0].value

    def test_an_operand_only_core_holds_no_static_program_and_takes_no_wall(self):
        census = PassPeriodCensus(passes=(_pass(0, cores=4, cells=400, operand_only=1),))
        assert _exposed(census, _odin())[0].value == pytest.approx(3 * ODIN_WALL_S, rel=1e-9)

    def test_a_per_byte_chip_keeps_its_walls(self):
        truenorth = get_platform_physics("truenorth")
        census = PassPeriodCensus(passes=(_pass(0, cores=3, cells=900),))
        term, _ = _exposed(census, truenorth)
        assert term.value == pytest.approx(truenorth.band("t_program_per_byte").nominal * 450, rel=1e-12)


class TestTheLadderReadsIt:
    """Pi's numerator, M's period writes and the write-bandwidth batch read the per-core wall (the hand point of
    ``test_scaling_ladder``: every softcore alone on a core)."""

    C = {**{k: v for k, v in CONSTANTS.items() if k != "t_program_per_byte"}, "t_core_program": 7.0}

    def _physics(self):
        """The hand point's test physics with the per-core wall in place of the per-byte one (t_core_program in its
        display unit, us)."""
        assert spec_for("t_core_program").display_unit == "us"
        return _physics("ladder_core_wall", self.C)

    def test_pi_once_is_the_wall_of_every_tile(self):
        row = _row(self._physics(), cores=64, k=16)
        assert row["time"]["terms"]["once"] == pytest.approx([self.C["t_core_program"] * 1e-6 * TILES] * 3, rel=1e-12)
        assert row["time"]["absent"] is None and row["time"]["pi"] is not None

    def test_m_and_the_bandwidth_batch_read_the_period_wall(self):
        network = _network(4)
        point = {**_platform(8), "platform_physics_resolved": self._physics().to_dict(), "inference_batch_K": 16,
                 "inference_rate_hz": 30.0, "min_lifetime_hours": 43_800.0}
        census = ladder.shape_census(network, point)
        row = ladder.ladder_row(network, census, ladder.price_program(network, census, point, _facts), point, READING)
        assert row["schedule_reading"].startswith("U1: multiplexed")
        period = row["time"]["terms"]["period"]
        assert period is not None and period[1] > 0
        cores_rewritten = period[1] / (self.C["t_core_program"] * 1e-6)
        assert cores_rewritten == pytest.approx(round(cores_rewritten), rel=1e-9)
        exposed = row["write_law"]["terms"]["latency_write_exposed_s"]
        assert [p / 16 for p in period] == pytest.approx(exposed, rel=1e-12)
        assert row["min_batch"]["k_bandwidth"] == pytest.approx([30.0 * e * 16 for e in exposed], rel=1e-12)
