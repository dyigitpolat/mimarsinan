"""[CL2] A profile that describes part of a priced system names the component it leaves out (an SRAM-CIM macro's
backing store); no absolute number is read until a run declares that component's constants beside the profile."""

import pytest

from mimarsinan.deployment_record.cost.absolute import price_absolute
from mimarsinan.deployment_record.cost.absolute.writes import (
    ENERGY_WRITE, LATENCY_WRITE, LIFETIME_HOURS, LIFETIME_INFERENCES, LIFETIME_VERDICT, WriteInputs, price_write_terms,
)
from mimarsinan.deployment_record.platform_physics import (
    PlatformPhysics,
    apply_overrides,
    available_profiles,
    get_platform_physics,
    profile_from_dict,
)
from mimarsinan.deployment_record.platform_physics.profile import ExternalComponent
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus

from unit.deployment_record.g3_fixtures import constant, quantities

_STORE = {"name": "backing store", "constants": ["e_dma_per_byte", "t_program_per_byte"],
          "note": "the weights the part does not hold come from a store it does not contain"}


def _profile(external=None, **constants):
    body = {"format_version": 1, "name": "part", "display_name": "Part", "description_file": "part.md",
            "validity": {"measurement_kind": "silicon"},
            "constants": constants or {"e_mac": constant(1.0, unit="pJ")}}
    if external is not None:
        body["external_components"] = external
    return profile_from_dict(body)


def _store_rows():
    return {"e_dma_per_byte": {"nominal": 100.0, "unit": "pJ", "evidence_kind": "published", "citation": "t p.1"},
            "t_program_per_byte": {"nominal": 0.1, "unit": "ns", "evidence_kind": "published", "citation": "t p.1"}}


class TestDeclaration:
    def test_the_loader_reads_the_component(self):
        physics = _profile([_STORE])
        assert physics.external_components == (
            ExternalComponent("backing store", ("e_dma_per_byte", "t_program_per_byte"), _STORE["note"]),)
        assert physics.undeclared_external() == (physics.external_components[0],)

    @pytest.mark.parametrize("component,match", [
        ({**_STORE, "constants": ["no_such_constant"]}, "no_such_constant"),
        ({**_STORE, "constants": []}, "names no constant"),
        ({**_STORE, "constants": ["e_mac"]}, "e_mac.*the profile declares"),
        ({**_STORE, "name": ""}, "a name"),
        ({**_STORE, "unit": "pJ"}, "unit"),
    ], ids=["unknown", "empty", "declared-by-the-profile", "unnamed", "unknown-field"])
    def test_a_malformed_component_refuses_by_name(self, component, match):
        with pytest.raises((KeyError, ValueError), match=match):
            _profile([component])

    def test_a_profile_without_components_serializes_as_before(self):
        """Every shipped profile but the macro states none, so their resolved physics is unchanged byte for byte."""
        for name in available_profiles():
            data = get_platform_physics(name).to_dict()
            assert ("external_components" in data) == (name == "sram_cim_macro"), name
        assert "external_components" not in _profile().to_dict()

    def test_the_component_round_trips_and_survives_overrides(self):
        physics = _profile([_STORE])
        assert PlatformPhysics.from_dict(physics.to_dict()) == physics
        declared = apply_overrides(physics, _store_rows())
        assert declared.external_components == physics.external_components
        assert declared.undeclared_external() == ()
        partial = apply_overrides(physics, {"e_dma_per_byte": _store_rows()["e_dma_per_byte"]})
        assert partial.undeclared_external() == physics.external_components

    def test_the_macro_names_its_backing_store(self):
        macro = get_platform_physics("sram_cim_macro")
        store, = macro.external_components
        assert store.name == "backing store"
        assert set(store.constants) == {"e_dma_per_byte", "t_program_per_byte"}
        assert "constants.md sec. 5" in store.note


class TestPricing:
    def test_an_undeclared_component_refuses_every_absolute_term_by_name(self):
        pricing = price_absolute(quantities(synaptic_events=10, host_macs=0), _profile([_STORE]))
        assert pricing.terms == () and pricing.verdicts == ()
        refusal, = pricing.refusals
        assert refusal.name == "absolute_pricing"
        assert refusal.reason.startswith("part is read only with its backing store declared: e_dma_per_byte, "
                                         "t_program_per_byte undeclared")
        assert "merge_override_blocks" in refusal.reason

    def test_a_declared_component_prices_as_a_profile_without_one(self):
        plain = price_absolute(quantities(synaptic_events=10, host_macs=0),
                               apply_overrides(_profile(), _store_rows()))
        held = price_absolute(quantities(synaptic_events=10, host_macs=0),
                              apply_overrides(_profile([_STORE]), _store_rows()))
        assert held == plain


def _rotating_census():
    """Two passes on one core, each rewriting 4,096 cells (4,096 B at 8 bits) every period."""
    passes = tuple(PassCensus(position=q, segment_index=0, resident=False, cores=1, resident_cores=0,
                              written_cells_backing_store=4096, written_cells_producer=0, payload_bytes=4096,
                              compute_steps=4) for q in range(2))
    return PassPeriodCensus(passes=passes, label="test: two rotating passes on one core")


_POLICY = WritePolicy(batch_k=1, inference_rate_hz=30.0, min_lifetime_hours=43800.0, physical_cores=1)
_WRITE_NAMES = {ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES, LIFETIME_VERDICT}


def _macro(store: bool) -> PlatformPhysics:
    macro = get_platform_physics("sram_cim_macro")
    return apply_overrides(macro, _store_rows()) if store else macro


def _executed(physics):
    """The executed plane's write law (``design_point.endurance.priced_at`` calls ``price_write_terms`` on the
    built program's period census)."""
    return price_write_terms(_rotating_census(), physics, _POLICY, most_written=2)


def _candidate(physics):
    """The exploration plane's write law (the ladder's ``price_absolute`` with the period census)."""
    return price_absolute(quantities(synaptic_events=10, host_macs=0), physics,
                          writes=WriteInputs(census=_rotating_census(), policy=_POLICY, most_written=2))


def _write_term(pricing, name):
    return next((t.band for t in pricing.terms if t.name == name and t.band is not None), None)


class TestOneWriteGateOnBothPlanes:
    """[fix_c] The write law reads the profile through the same external-component gate as every absolute term:
    the executed plane's write pricer (``price_write_terms``) priced the macro's write energy from its array write
    energy without a backing store (the store's transfer left as an unpriced note) while the exploration plane
    (``price_absolute``) refused it, so the two planes disagreed (W09: 4.74e-05 against None mJ on LeNet-5)."""

    def test_without_its_store_the_macro_prices_no_write_term_on_either_plane(self):
        executed, candidate = _executed(_macro(False)), _candidate(_macro(False))
        assert executed.terms == [] and executed.verdicts == []
        assert _write_term(candidate, ENERGY_WRITE) is None
        refused = {r.name: r.reason for r in executed.refusals}
        assert set(refused) == _WRITE_NAMES
        for name in (ENERGY_WRITE, LATENCY_WRITE, LIFETIME_INFERENCES):
            assert refused[name].startswith("sram_cim_macro is read only with its backing store declared: "
                                            "e_dma_per_byte, t_program_per_byte undeclared")
        assert {r.name: r.reason for r in candidate.refusals if r.name in _WRITE_NAMES} == refused

    def test_with_its_store_both_planes_price_the_same_write_energy(self):
        executed, candidate = _executed(_macro(True)), _candidate(_macro(True))
        energy = _write_term(executed, ENERGY_WRITE)
        assert energy is not None and energy == _write_term(candidate, ENERGY_WRITE)
        assert "e_dma_per_byte" in energy.basis and "unpriced" not in energy.basis
        assert _write_term(executed, LIFETIME_HOURS) == _write_term(candidate, LIFETIME_HOURS)

    def test_a_profile_without_components_prices_writes_as_before(self):
        plain = profile_from_dict({
            "format_version": 1, "name": "plain", "display_name": "Plain", "description_file": "plain.md",
            "validity": {"measurement_kind": "silicon"},
            "constants": {"e_write_per_cell": constant(1.0, unit="pJ"),
                          "write_verify_iterations": constant(1.0, unit="iterations")}})
        assert _write_term(_executed(plain), ENERGY_WRITE) is not None
