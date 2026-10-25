"""[CL2] Page-sourced targets: HERMES-64, the digital SRAM-CIM macro and NeuRRAM as physics bundles, the PUMA node as a
geometry-only bundle, every physics bundle judged against its own profile's validity, and the NVM endurance band."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

import mimarsinan
from mimarsinan.deployment_record.cost.absolute import (
    absolute_pricing_for_record,
    candidate_cost_report,
    emit_physics_report,
    price_absolute,
    price_write_terms,
    validity_refusals,
)
from mimarsinan.deployment_record.cost.absolute.validity import declared_geometry
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.objectives.views import CandidateStaticView
from mimarsinan.deployment_record.platform_physics import PlatformPhysics, get_platform_physics
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks
from mimarsinan.deployment_record.quantities.write_law import WritePolicy
from mimarsinan.mapping.platform.imc_platforms import get_imc_platform
from mimarsinan.mapping.platform.target_bundles import target_bundles
from mimarsinan.mapping.support.period_census import PassCensus, PassPeriodCensus
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import merge_pipeline_config
from mimarsinan.pipelining.core.platform_constraints_resolver import build_platform_constraints_resolved

from unit.deployment_record.g3_fixtures import quantities
from unit.deployment_record.record_fixtures import make_full_record, make_identity

TARGETS_DIR = Path(mimarsinan.__file__).parent / "mapping" / "platform" / "targets"
#: bundle -> (geometry, profile, claim class, faithfulness label)
CL2_PHYSICS = {
    "hermes_64": ("hermes_64", "hermes_64", "headline-eligible", "approximate: analog MVM error not modeled"),
    "neurram": ("neurram_like", "neurram", "headline-eligible",
                "approximate: analog MVM error not modeled; MVM energy read in a secondary source"),
    "sram_cim_macro": ("dimc_macro_128x1024", "sram_cim_macro", "quarantined",
                       "approximate: array write energy derived from the read energy"),
}
#: bundle -> (allow_coalescing, allow_neuron_splitting, the page each permission rests on)
PERMISSIONS = {
    "hermes_64": (True, True, ("gallo2022a", "p.3", "p.9")),
    "neurram": (True, True, ("wan2021edge", "p.13", "p.14")),
    "sram_cim_macro": (True, False, ("jiang2023a", "p.2", "p.3")),
}
NVM_BUNDLES = ("hermes_64", "neurram")
#: A hand backing store (test values, not a chip's): the two rows a backing-store block declares.
BACKING_STORE = {
    "e_dma_per_byte": {"nominal": 162.5, "unit": "pJ", "evidence_kind": "published", "citation": "test p.1"},
    "t_program_per_byte": {"nominal": 0.075, "unit": "ns", "evidence_kind": "published", "citation": "test p.1"},
}


def _pcfg(name, **platform):
    return build_platform_constraints_resolved(merge_pipeline_config({}, {"target": name, **platform}))


def _physics_of(pcfg):
    return PlatformPhysics.from_dict(pcfg["platform_physics_resolved"])


def _bundle_file(name):
    return json.loads((TARGETS_DIR / f"{name}.json").read_text())


def _record_on(pcfg):
    return replace(make_full_record(), identity=replace(make_identity(), platform=pcfg))


def _refusals(pricing):
    return {r.name: r.reason for r in pricing.refusals}


class TestPhysicsBundles:
    @pytest.mark.parametrize("name", sorted(CL2_PHYSICS))
    def test_resolves_with_its_geometry_profile_classes_and_value_domain(self, name):
        geometry, profile, claim, faithfulness = CL2_PHYSICS[name]
        bundle, pcfg = target_bundles()[name], _pcfg(name)
        assert bundle.geometry == geometry and bundle.claim_class == claim
        assert pcfg["cores"] == bundle.values["cores"] and pcfg["weight_bits"] == bundle.values["weight_bits"]
        assert pcfg["platform_physics_resolved"]["name"] == profile
        assert pcfg["core_semantics"] == "mvm"
        assert (pcfg["target_claim_class"], pcfg["target_faithfulness_class"]) == (claim, faithfulness)

    @pytest.mark.parametrize("name", sorted(PERMISSIONS))
    def test_the_permissions_are_the_ones_the_source_states(self, name):
        coalescing, splitting, (key, *pages) = PERMISSIONS[name]
        values = target_bundles()[name].values
        assert (values["allow_coalescing"], values["allow_neuron_splitting"]) == (coalescing, splitting)
        notes = _bundle_file(name)["notes"]
        assert key in notes and all(page in notes for page in pages), name

    def test_hermes_64_names_the_rows_its_profile_lacks(self):
        """The bundle resolves with its class and its refused rows named: the write energy (no pulse energy is
        stated, gallo2022a p.7) refuses by name while the lifetime prices from the endurance band."""
        physics = _physics_of(_pcfg("hermes_64"))
        pricing = price_write_terms(_two_pass_census(), physics, WritePolicy(batch_k=1), most_written=2)
        assert "e_write_per_cell" in _refusals(pricing)["energy_write_mj"]
        assert "lifetime_inferences" in {t.name for t in pricing.terms}
        assert not {"e_write_per_cell", "p_static_global", "e_noc_value_per_byte_hop"} & set(physics.constants)

    def test_hermes_64_takes_the_registered_64_core_geometry(self):
        platform = get_imc_platform("hermes_64")
        assert target_bundles()["hermes_64"].values["cores"] == [dict(core) for core in platform.cores]
        assert platform.total_cores == 64 and platform.weight_bits == 4

    def test_neurram_is_a_physics_bundle_over_the_registered_geometry_once(self):
        bundles = target_bundles()
        assert "neurram_like" not in bundles
        assert [b.name for b in bundles.values() if b.geometry == "neurram_like"] == ["neurram"]

    def test_the_macro_declares_its_geometry_in_bundle_data_with_the_derivation(self):
        bundle, data = target_bundles()["sram_cim_macro"], _bundle_file("sram_cim_macro")["geometry"]
        assert bundle.values["cores"] == [{"max_axons": 128, "max_neurons": 1024, "count": 1, "has_bias": False}]
        assert bundle.values["weight_bits"] == 8
        for fact in ("32*4 SRAM blocks", "64 rows containing 128 bits", "16 SR groups", "128 (4b) IA",
                     "16 32b outputs", "p.2", "p.3"):
            assert fact in data["provenance"], fact


class TestGeometryOnlyPumaNode:
    def test_resolves_its_geometry_with_the_class_and_refuses_physics(self):
        pcfg, platform = _pcfg("puma_node_like"), get_imc_platform("puma_node_like")
        assert pcfg["cores"] == [dict(core) for core in platform.cores] and pcfg["weight_bits"] == 16
        assert pcfg["platform_physics_resolved"] is None
        assert pcfg["target_claim_class"] == "curve-only"
        assert pcfg["target_faithfulness_class"] == "refused: no physics profile"

    def test_the_record_plane_refuses_absolute_pricing_by_name(self, tmp_path):
        pricing = absolute_pricing_for_record(_record_on(_pcfg("puma_node_like")))
        assert pricing is not None and pricing.terms == ()
        reason = _refusals(pricing)["absolute_pricing"]
        assert "target='puma_node_like' declares no physics profile" in reason
        report = json.loads(Path(emit_physics_report(_record_on(_pcfg("puma_node_like")), str(tmp_path)))
                            .read_text())
        assert any(note.startswith("refused absolute_pricing: target='puma_node_like'") for note in report["notes"])

    def test_the_candidate_plane_refuses_absolute_pricing_by_name(self):
        pcfg = _pcfg("puma_node_like")
        view = CandidateStaticView(layout=None, chip_param_capacity=None, total_params=None,
                                   host_side_segment_count=None, physics=None,
                                   quantity_context=candidate_context_from_platform(pcfg))
        report = view.cost_report()
        assert report is not None and report.energy == () and report.area == () and report.latency == ()
        assert any(n.startswith("refused absolute_pricing: target='puma_node_like'") for n in report.notes)

    def test_without_a_target_and_without_physics_nothing_is_priced_or_refused(self, tmp_path):
        pcfg = build_platform_constraints_resolved(merge_pipeline_config({}, {}))
        assert absolute_pricing_for_record(_record_on(pcfg)) is None
        assert emit_physics_report(_record_on(pcfg), str(tmp_path)) is None
        view = CandidateStaticView(layout=None, chip_param_capacity=None, total_params=None,
                                   host_side_segment_count=None, physics=None,
                                   quantity_context=candidate_context_from_platform(pcfg))
        assert view.cost_report() is None

    def test_the_candidate_report_takes_the_refusal_in_place_of_physics(self):
        report = candidate_cost_report(quantities(cores_physical=4), None, refusal=("absolute_pricing", "why"))
        assert report.area == () and "refused absolute_pricing: why" in report.notes
        with pytest.raises(ValueError, match="physics or its pricing refusal"):
            candidate_cost_report(quantities(cores_physical=4), None)


#: bundle -> the aggregate constants its own geometry refuses at its profile's validity (E190)
OWN_VALIDITY_REFUSALS = {
    "odin": ("area_per_core_total",), "truenorth": (), "loihi": (), "isaac_like": (),
    "generic_estimated_22nm": (), "hermes_64": (), "neurram": (), "sram_cim_macro": (),
}


class TestValidityPerBundle:
    """A bundle outside its profile's validity refuses the aggregate constants by name; inside, it prices them."""

    def test_every_physics_bundle_is_judged(self):
        physics = {n for n, b in target_bundles().items() if "platform_physics_profile" in b.values}
        assert physics == set(OWN_VALIDITY_REFUSALS)

    @pytest.mark.parametrize("name", sorted(OWN_VALIDITY_REFUSALS))
    def test_the_bundle_s_own_geometry_against_its_profile(self, name):
        values = target_bundles()[name].values
        physics = get_platform_physics(values["platform_physics_profile"])
        refused = validity_refusals(physics, values["cores"], weight_bits=values["weight_bits"])
        assert tuple(r.name for r in refused) == OWN_VALIDITY_REFUSALS[name]
        for refusal in refused:
            assert refusal.reason.startswith(f"{physics.name}'s aggregates hold at its ")

    def test_odin_s_bundle_lies_outside_odin_s_cells(self):
        values = target_bundles()["odin"].values
        refusal, = validity_refusals(get_platform_physics("odin"), values["cores"],
                                     weight_bits=values["weight_bits"])
        assert "of 3-bit cells; this program's weights take 4 bits" in refusal.reason

    def test_hermes_64_prices_its_published_chip_area_at_its_own_geometry(self):
        physics = _physics_of(_pcfg("hermes_64"))
        pricing = price_absolute(quantities(cores_physical=64, host_macs=0), physics,
                                 geometry=declared_geometry(target_bundles()["hermes_64"].values["cores"]))
        area = {t.name: t for t in pricing.terms}["chip_area_mm2"]
        assert area.value == pytest.approx(64 * 1.392 + 54.912) == pytest.approx(144.0)

    def test_a_core_count_override_outside_the_validity_refuses_the_area_by_name(self):
        cores = [{"max_axons": 256, "max_neurons": 256, "count": 65, "has_bias": False}]
        pcfg = _pcfg("hermes_64", target_overrides={"cores": {"value": cores, "note": "one core more"}})
        assert pcfg["target_faithfulness_class"].endswith("; cores declared")
        pricing = price_absolute(quantities(cores_physical=65, host_macs=0), _physics_of(pcfg),
                                 geometry=declared_geometry(pcfg["cores"]))
        assert "chip_area_mm2" not in {t.name for t in pricing.terms}
        assert "on chips of 1-64 cores; this chip declares 65" in _refusals(pricing)["chip_area_mm2"]


def _two_pass_census():
    """A hand two-pass program on a two-core chip, rewritten every period: each core written twice."""
    return PassPeriodCensus(passes=tuple(
        PassCensus(position=i, segment_index=0, resident=False, cores=2, resident_cores=0,
                   written_cells_backing_store=4096, written_cells_producer=0, payload_bytes=4096, compute_steps=0)
        for i in (0, 1)))


class TestNvmEnduranceBand:
    """NeuRRAM and HERMES-64 carry the NVM band 1e6-1e9 (sridharan2023xformer p.1, SECONDARY) until the owner's fetch
    of lee2011a and gallo2020an lands (L7), so lifetime and K* are priced on every page-sourced NVM point."""

    @pytest.mark.parametrize("name", NVM_BUNDLES)
    def test_the_band_is_the_secondary_one(self, name):
        row = _physics_of(_pcfg(name)).constants["endurance_cycles"]
        assert (row.low, row.high, row.evidence_kind) == (1e6, 1e9, "published")
        assert row.citation == "sridharan2023xformer p.1"
        assert "SECONDARY" in row.note and "lee2011a" in row.note and "gallo2020an" in row.note

    @pytest.mark.parametrize("name", NVM_BUNDLES)
    @pytest.mark.parametrize("k", [1, 16, 64])
    def test_lifetime_is_priced_at_every_batch(self, name, k):
        physics = _physics_of(_pcfg(name))
        policy = WritePolicy(batch_k=k, inference_rate_hz=30.0, min_lifetime_hours=43_800.0)
        pricing = price_write_terms(_two_pass_census(), physics, policy, most_written=2)
        life = {t.name: t for t in pricing.terms}["lifetime_inferences"]
        assert (life.band.low, life.band.high) == pytest.approx((1e6 * k / 2, 1e9 * k / 2))
        verdict, = pricing.verdicts
        assert verdict.evidence.endswith("; a secondary source]")


class TestMacroBackingStore:
    """The macro is read only with a declared backing store (constants.md sec. 5): without it, nothing absolute is
    priced on either plane, by name; with a named backing_store block, the macro prices."""

    def test_without_a_backing_store_both_planes_refuse_by_name(self):
        pcfg = _pcfg("sram_cim_macro")
        pricing = price_absolute(quantities(cores_physical=1, host_macs=0), _physics_of(pcfg))
        assert pricing.terms == ()
        reason = _refusals(pricing)["absolute_pricing"]
        assert "read only with its backing store declared" in reason
        assert "e_dma_per_byte" in reason and "t_program_per_byte" in reason
        record = absolute_pricing_for_record(_record_on(pcfg))
        assert record is not None and record.terms == () and "absolute_pricing" in _refusals(record)
        report = candidate_cost_report(quantities(cores_physical=1, host_macs=0), _physics_of(pcfg))
        assert any(n.startswith("refused absolute_pricing: sram_cim_macro is read only") for n in report.notes)

    def test_a_hand_declared_profile_is_held_to_the_same_rule(self):
        pcfg = build_platform_constraints_resolved({"cores": [{"max_axons": 128, "max_neurons": 1024, "count": 1}],
                                                    "weight_bits": 8, "platform_physics_profile": "sram_cim_macro"})
        pricing = price_absolute(quantities(cores_physical=1, host_macs=0), _physics_of(pcfg))
        assert pricing.terms == () and "absolute_pricing" in _refusals(pricing)

    def test_a_named_backing_store_block_lets_the_macro_price(self):
        overrides = merge_override_blocks({"backing_store": BACKING_STORE})
        pcfg = _pcfg("sram_cim_macro", platform_physics_overrides=overrides)
        physics = _physics_of(pcfg)
        assert physics.constants["e_dma_per_byte"].note == "[backing_store]"
        pricing = price_absolute(quantities(cores_physical=1, host_macs=0, synaptic_events=1000), physics)
        assert "absolute_pricing" not in _refusals(pricing)
        dynamic = {t.name: t for t in pricing.terms}["energy_dynamic_mj"]
        assert dynamic.value == pytest.approx(1000 * 158.7302e-15 * 1e3)


@pytest.mark.parametrize("module", ["mimarsinan.deployment_record.cost.absolute",
                                    "mimarsinan.deployment_record.objectives.views",
                                    "mimarsinan.deployment_record.objectives.context",
                                    "mimarsinan.deployment_record"])
def test_the_pricers_import_first_in_a_fresh_interpreter(module):
    """The pricer and the candidate context read the target's refusal through ``target_bundles``, whose registry
    package imports the config entry points: importing a pricer first must not cycle."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(Path(mimarsinan.__file__).resolve().parents[1])}
    code = (f"import {module}; import mimarsinan.deployment_record.cost.absolute.report as r; "
            "import mimarsinan.deployment_record.objectives.context as c; "
            "assert r.target_bundles.absolute_pricing_refusal({}) is None; "
            "assert c.target_bundles.target_bundles()")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=50)
    assert proc.returncode == 0, proc.stderr[-2000:]
