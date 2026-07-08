"""Exhaustive provenance + the CONCRETE prospective derived value (round-6 item 2).

A DERIVED key without an SSOT source is a BUILD-TIME error. Every key whose
effective value is produced by an SSOT (not by a plain schema default) names
that source AND can produce its concrete value for the current config state —
the wizard renders that value in green, never prose about where it comes from.
"""

import pytest

from mimarsinan.config_schema.registry import REGISTRY, effective_value
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema,
    FieldType,
    PROVENANCE_SOURCES,
)
from mimarsinan.config_schema.resolve import derived_values_view, resolve_draft
from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.gui.wizard.starter import starter_draft

_MODE_CELLS = (
    ("lif", None),
    ("ttfs", None),
    ("ttfs_quantized", None),
    ("ttfs_cycle_based", "cascaded"),
    ("ttfs_cycle_based", "synchronized"),
)


_CORES = [{"max_axons": 256, "max_neurons": 256, "count": 64}]


def _resolved(spiking_mode="lif", schedule=None, **overrides):
    dp = {"spiking_mode": spiking_mode, **overrides}
    if schedule:
        dp["ttfs_cycle_schedule"] = schedule
    return build_flat_pipeline_config(dp, {"cores": _CORES}, pipeline_mode="phased")


class TestBuildTimeInvariants:
    def test_a_derived_key_without_a_provenance_source_is_rejected(self):
        with pytest.raises(ValueError, match="provenance"):
            ConfigKeySchema(
                flat_key="bogus", group="run", owner="x", type=FieldType.BOOL,
                category=Category.DERIVED, derivation="derived",
                label="Bogus", doc="A derived key with no source.",
                derived_from=("spiking_mode",),
            )

    def test_a_derived_default_without_a_provenance_source_is_rejected(self):
        with pytest.raises(ValueError, match="provenance"):
            ConfigKeySchema(
                flat_key="bogus", group="run", owner="x", type=FieldType.INT,
                category=Category.ADVANCED, label="Bogus",
                doc="A frozen fallback with no named source.",
                derived_default=lambda cfg: 4,
            )

    def test_a_provenance_key_must_be_able_to_produce_a_concrete_value(self):
        """Naming a source without a way to compute the value would leave the
        wizard rendering prose again. Checked at registry BUILD (after the
        defaults SSOT is injected, so a defaulted key is legal)."""
        from mimarsinan.config_schema.registry.build import validate_registry

        sourceless = ConfigKeySchema(
            flat_key="bogus", group="run", owner="x", type=FieldType.INT,
            category=Category.ADVANCED, label="Bogus",
            doc="Sourced but uncomputable.",
            provenance="consumer frozen default",
        )
        with pytest.raises(ValueError, match="concrete"):
            validate_registry((sourceless,))

    def test_legal_values_require_an_enum(self):
        with pytest.raises(ValueError, match="enum"):
            ConfigKeySchema(
                flat_key="bogus", group="run", owner="x", type=FieldType.INT,
                category=Category.ADVANCED, label="Bogus", doc="Not an enum.",
                legal_values=lambda cfg: (1,),
            )

    def test_every_registry_entry_satisfies_the_invariants(self):
        for key, entry in REGISTRY.items():
            if entry.category is Category.DERIVED:
                assert entry.provenance, f"{key}: derived without a source"
            if entry.derived_default is not None:
                assert entry.provenance, f"{key}: frozen fallback without a source"
            if entry.provenance is not None:
                assert entry.provenance in PROVENANCE_SOURCES, key
                assert (
                    entry.has_default()
                    or entry.derived_default is not None
                    or entry.category is Category.DERIVED
                ), f"{key}: provenance without a computable value"


class TestProvenanceCoverage:
    """Round-6: the whole Deployment-target panel is derived — every one of its
    knobs names its SSOT source (the user's enumeration, exhaustively)."""

    DEPLOYMENT_TARGET_SOURCES = {
        "simulation_batch_count": "consumer frozen default",
        "scm_degradation_tolerance": "derivation rule",
        "nf_scm_parity_samples": "derivation rule",
        "nf_scm_parity_atol": "consumer frozen default",
        "nf_scm_parity_max_mismatch_fraction": "consumer frozen default",
        "nf_scm_parity_min_agreement": "consumer frozen default",
        "scm_torch_sim_parity_check": "consumer frozen default",
        "scm_torch_sim_parity_samples": "consumer frozen default",
        "scm_torch_sim_parity_min_agreement": "consumer frozen default",
        "onchip_majority_gate": "consumer frozen default",
        "onchip_majority_min_fraction": "consumer frozen default",
        "onchip_majority_fraction": "consumer frozen default",
        "onchip_min_fraction": "consumer frozen default",
        "capacity_gate": "consumer frozen default",
        "deployment_metric_full_eval": "consumer frozen default",
        "max_simulation_samples": "consumer frozen default",
        "loihi_parity_sample_index": "consumer frozen default",
    }

    def test_the_users_enumeration_now_names_its_source(self):
        for key, source in self.DEPLOYMENT_TARGET_SOURCES.items():
            assert REGISTRY[key].provenance == source, key

    def test_no_deployment_target_knob_is_left_sourceless(self):
        """Every deployment_target key without a schema default is derived."""
        sourceless = [
            key for key, entry in REGISTRY.items()
            if entry.group == "deployment_target"
            and not entry.has_default()
            and entry.provenance is None
        ]
        assert sourceless == []

    def test_a_named_source_never_hides_a_plain_schema_default(self):
        """provenance means 'an SSOT produces this', not 'it has a default'."""
        for key, entry in REGISTRY.items():
            if entry.provenance is None or not entry.has_default():
                continue
            if entry.category is Category.DERIVED or entry.derived_default is not None:
                continue
            values = {
                _resolved(mode, schedule).get(key) for mode, schedule in _MODE_CELLS
            }
            assert values != {entry.default}, (
                f"{key}: claims {entry.provenance!r} but always resolves to its "
                "schema default — drop the provenance"
            )


class TestTheConcreteProspectiveValue:
    """The green in-field text is a VALUE, produced by the SSOT deriver for the
    CURRENT config state — never the ``empty_means`` prose."""

    @pytest.mark.parametrize("mode,schedule", _MODE_CELLS)
    def test_every_sourced_key_yields_a_value_in_every_mode(self, mode, schedule):
        resolved = _resolved(mode, schedule)
        view = derived_values_view(resolved)
        sourced = {k for k, e in REGISTRY.items() if e.provenance is not None}
        assert set(view) == sourced
        uncomputable = {k for k, v in view.items() if v is None}
        # Only genuine "absence is the value" keys may resolve to null.
        assert uncomputable <= {
            # "absence IS the value" (no cap / no registration the registry sees)
            "simulation_batch_count", "weight_source", "pretrained_weight_set",
            "tuning_step_cap_epochs", "calibration_set_policy",
            "proven_recovery_depth", "activation_analysis_batch_size",
            "eval_subsample_target",
        }, sorted(uncomputable)

    def test_the_derived_value_is_never_the_empty_means_prose(self):
        view = derived_values_view(_resolved())
        for key, value in view.items():
            prose = REGISTRY[key].empty_means
            if prose is not None:
                assert value != prose, key

    def test_recipe_overridden_defaults_render_their_mode_value_not_the_default(self):
        """kd_ce_alpha's schema default is 0.3 but the lif recipe folds 0.5 —
        the blue default would have been a lie."""
        lif = derived_values_view(_resolved("lif"))
        ttfs = derived_values_view(_resolved("ttfs"))
        assert lif["kd_ce_alpha"] == 0.5 and REGISTRY["kd_ce_alpha"].default == 0.3
        assert lif["kd_temperature"] == 4.0
        assert ttfs["kd_ce_alpha"] == 0.3
        assert derived_values_view(_resolved("ttfs_quantized"))[
            "activation_scale_quantile"
        ] == 1.0

    def test_the_nf_scm_sample_count_is_mode_aware(self):
        cascaded = derived_values_view(_resolved("ttfs_cycle_based", "cascaded"))
        analytic = derived_values_view(_resolved("ttfs"))
        assert cascaded["nf_scm_parity_samples"] == 64
        assert analytic["nf_scm_parity_samples"] == 2

    def test_the_scm_tolerance_names_the_tolerance_that_actually_governs(self):
        view = derived_values_view(_resolved(degradation_tolerance=0.15))
        assert view["scm_degradation_tolerance"] == 0.15

    def test_the_endpoint_floor_tells_the_truth_outside_the_bit_parity_family(self):
        assert derived_values_view(_resolved("ttfs"))["endpoint_target_floor"] == 0.98
        assert derived_values_view(_resolved("lif"))["endpoint_target_floor"] == 0.0

    def test_an_explicit_declaration_takes_the_value_back(self):
        view = derived_values_view(_resolved(nf_scm_parity_atol=1e-3))
        assert view["nf_scm_parity_atol"] == 1e-3

    def test_a_blocked_derivation_yields_no_hypothetical_values(self):
        resolution = resolve_draft({
            "data_provider_name": "MNIST_DataProvider", "experiment_name": "x",
            "generated_files_path": "./g", "start_step": None,
            "platform_constraints": {},
            "deployment_parameters": {
                "model_type": "lenet5", "model_config": {"variant": "lenet5"},
                "spiking_mode": "lif", "firing_mode": "TTFS",
            },
        })
        assert not resolution.ok
        assert derived_values_view(resolution.resolved) == {}


class TestEffectiveValueIsTheOneAccessor:
    """Consumers read frozen fallbacks through the registry, so the number the
    wizard renders and the number the run uses cannot drift."""

    def test_explicit_wins(self):
        assert effective_value({"nf_scm_parity_samples": 7}, "nf_scm_parity_samples") == 7

    def test_the_disable_sentinel_survives(self):
        """0 is falsy but declared — the gate's documented opt-out."""
        assert effective_value({"nf_scm_parity_samples": 0}, "nf_scm_parity_samples") == 0
        assert effective_value({"capacity_gate": False}, "capacity_gate") is False

    def test_absent_falls_back_to_the_registry_derived_default(self):
        assert effective_value({"spiking_mode": "ttfs"}, "nf_scm_parity_samples") == 2
        assert effective_value(
            {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": "cascaded"},
            "nf_scm_parity_samples",
        ) == 64
        assert effective_value({}, "nf_scm_parity_atol") == 1e-6
        assert effective_value({}, "num_workers") == 4

    def test_absent_falls_back_to_the_schema_default(self):
        assert effective_value({}, "sanafe_sample_count") == 1

    def test_an_unregistered_key_is_loud(self):
        with pytest.raises(KeyError):
            effective_value({}, "not_a_key")


class TestDerivedDefaultsAgreeWithTheConsumers:
    """The registry's frozen fallback IS the number the consumer resolves.
    A drift between the two would render a green lie."""

    def test_deployment_plan_literals_match_the_registry(self):
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        plan = DeploymentPlan.resolve({"model_type": "lenet5", "spiking_mode": "lif"})
        empty = {"spiking_mode": "lif"}
        assert plan.deployment_metric_full_eval == effective_value(
            empty, "deployment_metric_full_eval")
        assert plan.max_simulation_samples == effective_value(
            empty, "max_simulation_samples")
        assert plan.simulation_batch_count == effective_value(
            empty, "simulation_batch_count")
        assert plan.pruning == effective_value(empty, "pruning")
        assert plan.pruning_fraction == effective_value(empty, "pruning_fraction")
        assert plan.prune_sparsity == effective_value(empty, "prune_sparsity")

    def test_the_starter_resolves_every_sourced_key(self):
        resolution = resolve_draft(starter_draft())
        assert resolution.errors == []
        view = derived_values_view(resolution.resolved)
        assert view["nf_scm_parity_samples"] == 2
        assert view["scm_degradation_tolerance"] == 0.15
        assert view["onchip_majority_min_fraction"] == 0.2
