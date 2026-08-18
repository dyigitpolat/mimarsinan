"""TS6 — the programming constants every shipped profile now derives.

`programming` was the vocabulary's least-populated group: no target's paper
measures a weight load, so the per-byte commit energy, the per-byte wall and
the per-core setup were absent everywhere and every programming term refused
by name. They are AUTHORED here under the C6 discipline — a derived value is
written down with the arithmetic that produced it, or it stays absent — and
pinned three ways: the vocabulary contract, the written evidence, and the
silicon-correlation cases, which never reprogram and must therefore predict
byte-identically before and after.
"""

import json
import math
import os
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.correlation import (
    available_cases,
    correlate_all,
    correlation_payload_json,
    get_case,
)
from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.platform_physics import get_platform_physics
from mimarsinan.deployment_record.platform_physics.constants import (
    PHYSICS_CONSTANTS,
    spec_for,
)
from mimarsinan.deployment_record.platform_physics.registry import (
    profile_description_path,
)
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.quantities.spec import QUANTITY_SPECS
from mimarsinan.deployment_record.schema.provenance import Band
from mimarsinan.deployment_record.units import unit_for

from unit.deployment_record.record_fixtures import make_full_record

PROFILES = ("loihi", "truenorth", "odin", "isaac_like", "generic_estimated_22nm")
PROGRAMMING_CONSTANTS = ("e_program_per_byte", "t_program_per_byte", "e_core_program")

#: The evidence kind each profile's derived constants must carry: an anchor that is
#: itself an estimate cannot produce anything stronger than an estimate.
MANDATED_EVIDENCE = {
    "loihi": "derived",
    "truenorth": "derived",
    "odin": "derived",
    "isaac_like": "derived",
    "generic_estimated_22nm": "estimated",
}

#: Synapse accesses per BYTE of weight store — what turns a per-EVENT access anchor
#: into the per-byte one the write ratio multiplies (1-bit TrueNorth synapse: 8 per
#: byte; 4-bit ODIN synapse: 2; 9-bit Loihi entry: 1).
ACCESSES_PER_BYTE = {"loihi": 1.0, "truenorth": 8.0, "odin": 2.0}
WRITE_RATIO = (0.5, 3.0)
NVM_RATIO = (10.0, 100.0)
CORE_SETUP_RATIO = (0.5, 2.0)
NEURONS_PER_CORE = 256.0
ISAAC_TILE_CYCLE_BYTES = 1024.0
DONORS = ("truenorth", "generic_estimated_22nm")

GOLDEN = os.path.join(os.path.dirname(__file__), os.pardir, "correlation",
                      "reference_case_predictions_golden.json")


def _physics(name):
    return get_platform_physics(name)


def _value(profile, key):
    return _physics(profile).constants[key]


def _scaled(band: Band, factor: float) -> Band:
    return Band(band.low * factor, band.nominal * factor, band.high * factor,
                basis=band.basis)


def _log_centre(band: Band) -> float:
    """A ratio-spread band states its nominal at the band's log-space centre."""
    return math.sqrt(band.low * band.high)


def _write_anchor(profile) -> Band:
    """The SRAM-class access anchor, per byte of weight store."""
    return _scaled(_physics(profile).band("e_mac"), ACCESSES_PER_BYTE[profile])


def _sram_class_reference() -> float:
    """The SRAM class's per-byte commit energy: the three targets' geometric mean."""
    nominals = [_physics(name).band("e_program_per_byte").nominal
                for name in ACCESSES_PER_BYTE]
    return math.exp(sum(math.log(value) for value in nominals) / len(nominals))


def _core_setup_anchor(profile) -> Band:
    """What a target's per-core program setup is anchored on, and why (see the .md)."""
    physics = _physics(profile)
    if physics.has("e_core_init"):
        return physics.band("e_core_init")
    if profile == "isaac_like":
        return _scaled(physics.band("e_dma_per_byte"), ISAAC_TILE_CYCLE_BYTES)
    per_neuron = "e_neuron_update" if physics.has("e_neuron_update") else "e_mac"
    return _scaled(physics.band(per_neuron), NEURONS_PER_CORE)


def _donor_scaled_walls(profile):
    """Each donor's per-byte wall, scaled to this target's technology node."""
    node = _physics(profile).validity.technology_node_nm
    assert node is not None, f"{profile}: donor scaling needs a declared node"
    scaled = []
    for donor in DONORS:
        donor_physics = _physics(donor)
        donor_node = donor_physics.validity.technology_node_nm
        assert donor_node is not None, donor
        scaled.append(_scaled(donor_physics.band("t_program_per_byte"),
                              node / donor_node))
    return scaled


class TestVocabularyConformance:
    """Every new value declares a vocabulary constant, in that constant's terms."""

    def test_the_three_constants_are_the_vocabularys_programming_terms(self):
        for key in PROGRAMMING_CONSTANTS:
            assert spec_for(key).group == "programming"
        assert spec_for("e_program_per_byte").dimension == "energy"
        assert spec_for("t_program_per_byte").dimension == "time"
        assert spec_for("e_core_program").dimension == "energy"

    def test_the_multiplicands_are_the_program_load_census(self):
        """The commit is charged per PROGRAMMED byte only: a carried activation
        crosses the DMA channel (e_dma_per_byte) without being written into the
        weight store, so sharing one constant would over-charge every inference."""
        assert PHYSICS_CONSTANTS["e_program_per_byte"].multiplicand == "reprogrammed_bytes"
        assert PHYSICS_CONSTANTS["t_program_per_byte"].multiplicand == "reprogrammed_bytes"
        assert PHYSICS_CONSTANTS["e_core_program"].multiplicand == "reprogrammed_cores"
        for key in PROGRAMMING_CONSTANTS:
            assert PHYSICS_CONSTANTS[key].multiplicand in QUANTITY_SPECS

    @pytest.mark.parametrize("profile", PROFILES)
    def test_every_profile_declares_every_programming_constant(self, profile):
        """All five, or a DOM/programming headline refuses by name on the target
        the campaign happens to need."""
        assert _physics(profile).missing(PROGRAMMING_CONSTANTS) == ()

    @pytest.mark.parametrize("profile", PROFILES)
    def test_each_declared_value_validates_in_its_own_unit(self, profile):
        for key in PROGRAMMING_CONSTANTS:
            value = _value(profile, key)
            assert unit_for(value.unit).dimension == spec_for(key).dimension
            assert 0.0 < value.low <= value.nominal <= value.high
            band = _physics(profile).band(key)
            assert band.low <= band.nominal <= band.high


class TestTheWrittenEvidence:
    """A derived constant that does not show its arithmetic is a guess with units."""

    @pytest.mark.parametrize("profile", PROFILES)
    def test_every_new_constant_shows_its_derivation(self, profile):
        for key in PROGRAMMING_CONSTANTS:
            assert _value(profile, key).derivation.strip(), f"{profile}/{key}"

    @pytest.mark.parametrize("profile", PROFILES)
    def test_the_mandated_evidence_kind(self, profile):
        for key in PROGRAMMING_CONSTANTS:
            assert _value(profile, key).evidence_kind == MANDATED_EVIDENCE[profile]

    @pytest.mark.parametrize("profile", ACCESSES_PER_BYTE)
    def test_the_sram_class_derivation_names_its_anchor(self, profile):
        assert "e_mac" in _value(profile, "e_program_per_byte").derivation

    def test_the_core_setup_derivation_names_its_anchor(self):
        anchors = {"loihi": "e_core_init", "truenorth": "e_mac", "odin": "e_mac",
                   "isaac_like": "e_dma_per_byte",
                   "generic_estimated_22nm": "e_neuron_update"}
        for profile, anchor in anchors.items():
            assert anchor in _value(profile, "e_core_program").derivation, profile

    @pytest.mark.parametrize("profile", ("loihi", "odin", "isaac_like"))
    def test_the_donor_scaled_wall_names_its_donors_and_the_scaling_axis(self, profile):
        derivation = _value(profile, "t_program_per_byte").derivation
        for token in DONORS + ("technology_node_nm",):
            assert token in derivation, f"{profile}: {token}"

    def test_the_nvm_profile_states_that_endurance_is_unmodeled(self):
        """A per-inference-reprogrammed NVM array wears out; the model prices the
        write and says, in the constant itself, that it does not price the wear."""
        value = _value("isaac_like", "e_program_per_byte")
        text = f"{value.note} {value.derivation}"
        assert "endurance" in text.lower() and "UNMODELED" in text

    @pytest.mark.parametrize("profile", PROFILES)
    def test_the_description_file_carries_the_derivations_in_prose(self, profile):
        text = profile_description_path(profile).read_text(encoding="utf-8")
        for key in PROGRAMMING_CONSTANTS:
            assert key in text, f"{profile}.md never mentions {key}"


class TestTheDerivationRules:
    """The spec table, recomputed from the constants it anchors on."""

    @pytest.mark.parametrize("profile", ACCESSES_PER_BYTE)
    def test_the_sram_class_band_is_the_write_ratio_spread(self, profile):
        anchor = _write_anchor(profile)
        band = _physics(profile).band("e_program_per_byte")
        assert band.low == pytest.approx(WRITE_RATIO[0] * anchor.low, rel=1e-4)
        assert band.high == pytest.approx(WRITE_RATIO[1] * anchor.high, rel=1e-4)
        assert band.nominal == pytest.approx(_log_centre(band), rel=1e-4)

    def test_the_nvm_band_is_the_literature_multiple_of_the_sram_class(self):
        reference = _sram_class_reference()
        band = _physics("isaac_like").band("e_program_per_byte")
        assert band.low == pytest.approx(NVM_RATIO[0] * reference, rel=1e-3)
        assert band.high == pytest.approx(NVM_RATIO[1] * reference, rel=1e-3)
        assert band.nominal == pytest.approx(_log_centre(band), rel=1e-4)

    def test_the_estimated_exemplar_is_not_anchored_on_a_datapath_multiply(self):
        """generic's e_mac is an 8-bit MULTIPLY, not a memory access, so the
        exemplar stands on its own SRAM-access reasoning — and says so."""
        band = _physics("generic_estimated_22nm").band("e_program_per_byte")
        assert band.nominal == pytest.approx(_log_centre(band), rel=1e-4)
        assert band.low > _physics("generic_estimated_22nm").band("e_mac").low

    @pytest.mark.parametrize("profile", PROFILES)
    def test_core_setup_is_the_declared_ratio_of_its_anchor(self, profile):
        anchor = _core_setup_anchor(profile)
        band = _physics(profile).band("e_core_program")
        assert band.low == pytest.approx(CORE_SETUP_RATIO[0] * anchor.low, rel=1e-4)
        assert band.high == pytest.approx(CORE_SETUP_RATIO[1] * anchor.high, rel=1e-4)
        assert band.nominal == pytest.approx(_log_centre(band), rel=1e-4)

    @pytest.mark.parametrize("profile", ("loihi", "odin", "isaac_like"))
    def test_the_wall_band_spans_both_scaled_donors(self, profile):
        scaled = _donor_scaled_walls(profile)
        band = _physics(profile).band("t_program_per_byte")
        assert band.low == pytest.approx(min(s.low for s in scaled), rel=1e-4)
        assert band.high == pytest.approx(max(s.high for s in scaled), rel=1e-4)
        product = scaled[0].nominal * scaled[1].nominal
        assert band.nominal == pytest.approx(math.sqrt(product), rel=1e-4)

    def test_the_scaled_wall_follows_the_technology_node(self):
        """Donor scaling is the only thing separating these three: a finer node
        must not come out slower, or the scaling axis is not the one declared."""
        walls = [_physics(name).band("t_program_per_byte").nominal
                 for name in ("loihi", "odin", "isaac_like")]
        assert walls == sorted(walls)


class TestNonPerturbation:
    """The reference cases never reprogram, so their predictions may not move."""

    def test_no_reference_case_carries_a_program_load(self):
        program_census = {"reprogrammed_bytes", "reprogrammed_cores",
                          "connectivity_entries"}
        for name in available_cases():
            census = set(get_case(name).census)
            assert not census & program_census, name

    def test_the_shipped_predictions_are_byte_identical_to_the_golden(self):
        with open(GOLDEN, encoding="utf-8") as handle:
            expected = handle.read()
        assert correlation_payload_json(correlate_all()) == expected, (
            "the derived programming constants leaked into a priced axis; "
            "regenerate only when a prediction is MEANT to move")


class TestTheProgrammingTermsAppear:
    """The term-presence pin: a scheduled vehicle's program load now prices."""

    def _priced(self, physics):
        return price_absolute(from_record(make_full_record()), physics)

    def _terms(self, physics):
        return {term.name: term for term in self._priced(physics).terms}

    def _without_programming(self, name):
        """The same profile as before TS6 authored its programming group."""
        physics = _physics(name)
        return replace(physics, constants={
            key: value for key, value in physics.constants.items()
            if key not in PROGRAMMING_CONSTANTS
        })

    def test_loihi_now_prices_the_program_load_it_used_to_refuse(self):
        terms = self._terms(_physics("loihi"))
        assert "energy_programming_mj" in terms
        assert "latency_programming_s" in terms
        assert "e_program_per_byte" in terms["energy_programming_mj"].source
        assert "e_core_program" in terms["energy_programming_mj"].source
        assert "t_program_per_byte" in terms["latency_programming_s"].source

    def test_the_terms_are_the_declared_constants_times_the_sealed_census(self):
        quantities = from_record(make_full_record())
        payload_bytes = quantities.get("reprogrammed_bytes").value
        cores = quantities.get("reprogrammed_cores").value
        physics = _physics("loihi")
        terms = self._terms(physics)
        expected = (physics.band("e_program_per_byte").nominal * payload_bytes
                    + physics.band("e_core_program").nominal * cores) * 1e3
        assert terms["energy_programming_mj"].value == pytest.approx(expected, rel=1e-9)
        assert terms["latency_programming_s"].value == pytest.approx(
            physics.band("t_program_per_byte").nominal * payload_bytes, rel=1e-9)

    def test_without_them_the_terms_were_absent_by_name(self):
        terms = self._terms(self._without_programming("loihi"))
        assert "energy_programming_mj" not in terms
        assert "latency_programming_s" not in terms

    def test_nothing_but_the_program_load_moved(self):
        """The new constants price a per-LOAD overhead. Every per-inference term
        must be identical to what the profile priced before they existed."""
        before = self._terms(self._without_programming("loihi"))
        after = self._terms(_physics("loihi"))
        assert set(after) - set(before) == {"energy_programming_mj",
                                            "latency_programming_s"}
        for name, term in before.items():
            assert after[name].value == term.value, name
            assert after[name].source == term.source, name


def test_the_golden_is_the_payload_the_harness_writes():
    """Regenerated by `python scripts/silicon_correlation.py --json <golden>`, so
    the file and the harness can never drift into two shapes."""
    with open(GOLDEN, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert [case["name"] for case in payload["cases"]] == sorted(available_cases())
