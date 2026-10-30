"""[U-H H-2, survey gap B6.1] A fabric's identity carries its neuron count.

The claims that name a fabric (``ChipConfig.bundle_claims`` / ``claims_of_bundle``, ``chip_selection``) were the weight
width, the sign granularity, the effective fan-in and the membrane width. Two fabrics that differ only in neurons per
core made the same claims, so the first one listed answered for both: the segment gates, the pass builds (whose
synapse depth and addresses follow the neuron count), the bundle's named fabric, packaging and the envelope audit.
The suite's distinct-claims guard made that a latent blocker (no neuron-only shape could be added), not a silent
wrong fabric; the card's fabric check compared sign and axon slots, never neurons.

The fix, generic: ``max_neurons`` joins the claims, a bundle declares its fabric's neuron count in ``chip_config``
(the declared core type's, from the same envelope that gives the effective fan-in), the freeze refuses a mapped core
wider than that, and the executor refuses a bundle that needs more neurons than the session's fabric declares.
"""

from __future__ import annotations

import json
import shutil

import pytest

import mimarsinan.chip_simulation.odin_fpga.chip_selection as selection
from mimarsinan.chip_simulation.odin_fpga.chip_configs import (
    CORE_KIND_GENERATED,
    STOCK_CHIP,
    WIDE_CHIP,
    ChipConfig,
    ChipConfigError,
    chip_config_named,
)
import mimarsinan.chip_simulation.odin_deployment_bundle as bundle_schema
from mimarsinan.mapping.export.odin_gen.variants import per_event_law, spec_for

from unit.chip_simulation.test_odin_deployment_executor import (
    BUNDLE_NAME,
    PACKAGE,
    REPLAY_NAME,
    reseal,
    run_executor,
)
from integration.odin_hacc_harness import stage_package_host


def _narrow_wide() -> ChipConfig:
    """The widened fabric's every claim but its neuron count: 1024 axons x 128 neurons, 8-bit cells, 16-bit membrane."""
    return ChipConfig(name="odin_identity_test_1024x128_mb16", label="identity test", core_kind=CORE_KIND_GENERATED,
                      core_spec=spec_for(per_event_law(16), axons=1024, neurons=128, weight_bits=8), variant=None)


@pytest.fixture
def three(monkeypatch):
    table = (chip_config_named(STOCK_CHIP), chip_config_named(WIDE_CHIP), _narrow_wide())
    monkeypatch.setattr(selection, "chip_configs", lambda: table)
    return table


def _block(config: ChipConfig) -> dict:
    claims = dict(config.bundle_claims())
    return {**{k: v for k, v in claims.items() if k != "membrane_bits"},
            "soma_law": {"membrane_bits": claims["membrane_bits"]}}


class TestTheNeuronCountIsPartOfTheIdentity:
    def test_the_claims_carry_the_fabric_s_neurons(self):
        for name in (STOCK_CHIP, WIDE_CHIP):
            config = chip_config_named(name)
            assert config.bundle_claims()["max_neurons"] == config.max_neurons

    def test_fabrics_differing_only_in_neurons_make_distinct_claims(self, three):
        wide, narrow = three[1].bundle_claims(), three[2].bundle_claims()
        assert {k: v for k, v in wide.items() if k != "max_neurons"} == \
               {k: v for k, v in narrow.items() if k != "max_neurons"}
        seen = [tuple(sorted(c.bundle_claims().items())) for c in three]
        assert len(set(seen)) == len(seen)

    def test_each_fabric_resolves_from_its_own_claims(self, three):
        for config in three:
            assert selection.chip_config_of_claims(config.bundle_claims()) is config

    def test_a_bundle_block_resolves_by_its_neuron_count(self, three):
        assert selection.chip_config_of_bundle({"chip_config": _block(three[2])}) is three[2]
        assert selection.chip_config_of_bundle({"chip_config": _block(three[1])}) is three[1]

    def test_a_declared_envelope_resolves_by_its_neuron_count(self, three):
        common = dict(weight_bits=8, weight_sign_granularity="per_synapse", effective_max_axons=1023,
                      membrane_bits=16)
        assert selection.chip_config_for(**common, max_neurons=128) is three[2]
        assert selection.chip_config_for(**common, max_neurons=256) is three[1]
        with pytest.raises(ChipConfigError):
            selection.chip_config_for(**common, max_neurons=64)

    def test_a_block_without_the_neuron_count_is_refused_by_name(self):
        """[U-H targeted pass, review M4] Every bundle sealed before B6.1 lacks the claim: it is refused by name, with
        the remedy, never a bare ``KeyError`` (plan section 3.4: a refusal without a name is a defect)."""
        block = _block(chip_config_named(WIDE_CHIP))
        del block["max_neurons"]
        with pytest.raises(bundle_schema.OdinBundleStale, match=r"max_neurons.*[Rr]egenerate"):
            ChipConfig.claims_of_bundle(block)
        with pytest.raises(bundle_schema.OdinBundleStale, match="max_neurons"):
            selection.chip_config_of_bundle({"chip_config": block})

    def test_the_stale_check_names_exactly_the_claims_a_fabric_is_read_by(self):
        """One list of claims for the reader's check and the fabric's own claims (``bundle_claims``), so a claim added
        later cannot reach a reader unchecked."""
        claims = set(chip_config_named(STOCK_CHIP).bundle_claims())
        assert set(bundle_schema.FABRIC_CLAIMS) == (claims - {"membrane_bits"}) | {"soma_law"}


class TestTheShippedBundleDeclaresItsFabric:
    def test_the_committed_bundle_declares_the_stock_fabric_s_neurons(self):
        document = json.loads((PACKAGE / "deployment" / BUNDLE_NAME).read_text())
        assert document["chip_config"]["max_neurons"] == chip_config_named(STOCK_CHIP).max_neurons
        assert selection.chip_config_of_bundle(document).name == STOCK_CHIP

    def test_every_mapped_core_fits_the_declared_neurons(self):
        document = json.loads((PACKAGE / "deployment" / BUNDLE_NAME).read_text())
        assert all(int(core["neurons"]) <= document["chip_config"]["max_neurons"] for core in document["cores"])


class TestTheFreezeRefusesACoreWiderThanItsFabric:
    def test_a_mapped_core_wider_than_the_declared_neurons_is_refused_by_name(self):
        from mimarsinan.chip_simulation.odin_hacc.freeze import BundleRefusal, require_cores_fit_neurons

        cores = [type("Core", (), {"neurons_per_core": 3, "available_neurons": 0})(),
                 type("Core", (), {"neurons_per_core": 300, "available_neurons": 0})()]
        require_cores_fit_neurons("n", cores[:1], max_neurons=256)
        with pytest.raises(BundleRefusal, match=r"core 1.*300.*256"):
            require_cores_fit_neurons("n", cores, max_neurons=256)


class TestTheExecutorRefusesAFabricWithFewerNeurons:
    def test_a_bundle_needing_more_neurons_than_the_session_s_fabric_is_refused(self, tmp_path):
        root = tmp_path / "pkg"
        root.mkdir()
        stage_package_host(root)
        (root / "deployment").mkdir()
        for name in (BUNDLE_NAME, REPLAY_NAME):
            shutil.copyfile(PACKAGE / "deployment" / name, root / "deployment" / name)

        def wider(document):
            document["chip_config"]["max_neurons"] = 2 * chip_config_named(STOCK_CHIP).max_neurons

        reseal(root / "deployment" / BUNDLE_NAME, wider)
        completed = run_executor(root)
        assert completed.returncode != 0
        assert "neuron" in (completed.stdout + completed.stderr)

    def test_a_bundle_sealed_before_the_neuron_count_is_refused_by_name(self, tmp_path):
        """[U-H targeted pass, review M4] The pins and the Stage 1 package's bundles have no ``max_neurons``: the card
        session refuses them in the bundle family, by name and with the remedy, before anything is programmed."""
        root = tmp_path / "pkg"
        root.mkdir()
        stage_package_host(root)
        (root / "deployment").mkdir()
        for name in (BUNDLE_NAME, REPLAY_NAME):
            shutil.copyfile(PACKAGE / "deployment" / name, root / "deployment" / name)
        reseal(root / "deployment" / BUNDLE_NAME, lambda document: document["chip_config"].pop("max_neurons"))
        sealed = json.loads((root / "deployment" / BUNDLE_NAME).read_text())
        reseal(root / "deployment" / REPLAY_NAME,
               lambda replay: replay.__setitem__("bundle_self_hash", sealed["self_hash"]))
        completed = run_executor(root)
        assert completed.returncode == 2, completed.stdout + completed.stderr
        assert "REFUSING: OdinBundleStale" in completed.stderr
        assert "max_neurons" in completed.stderr and "egenerate" in completed.stderr
        assert "Traceback" not in completed.stderr
