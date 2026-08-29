"""[P3] Streamed-lif derivation: raw-cascade exact-QAT, no re-timing, capability set."""

from mimarsinan.config_schema.runtime import build_flat_pipeline_config
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

_CORES = [{"max_axons": 256, "max_neurons": 256, "count": 64}]


def _resolved(variant):
    dp = {"spiking_family": "lif", "spiking_variant": variant}
    return build_flat_pipeline_config(dp, {"cores": _CORES}, pipeline_mode="phased")


class TestStreamedRecipe:
    def test_streamed_trains_the_raw_cascade_not_the_staircase(self):
        # exact-QAT is inherently WINDOWED (the trained staircase is the
        # per-hop twin); streamed trains through the cycle-accurate NF.
        cfg = _resolved("streamed")
        assert cfg["lif_exact_qat"] is False
        assert cfg["lif_per_hop_retiming"] is False
        assert cfg["cycle_accurate_lif_forward"] is True
        assert cfg["lif_depth_balancing_relays"] is True

    def test_synchronized_keeps_the_retimed_pair(self):
        cfg = _resolved("synchronized")
        assert cfg["lif_exact_qat"] is True
        assert cfg["lif_per_hop_retiming"] is True

    def test_streamed_enables_all_backends(self):
        """[N5 2026-08-09] streamed Loihi enabled: the wave runner's per-core
        replay is free-running-equivalent on gap-1 graphs and the FATAL
        spike-parity gate certifies every run."""
        cfg = _resolved("streamed")
        assert cfg["enable_loihi_simulation"] is True
        assert cfg["enable_nevresim_simulation"] is True
        assert cfg["enable_sanafe_simulation"] is True
        assert _resolved("synchronized")["enable_loihi_simulation"] is True

    def test_policy_row_is_marked_streamed(self):
        recipe = ConversionPolicy.derive("lif", spiking_variant="streamed")
        assert recipe.special_case == "streamed_raw_cascade"
        assert "streaming" in recipe.rationale
        windowed = ConversionPolicy.derive("lif")
        assert windowed.special_case == "bn_freeze"
        assert recipe.knobs == {
            **windowed.knobs,
            "lif_exact_qat": False,
            # The streamed ladder is the one that ramps THROUGH the deployed
            # composition, so proxy_gap is the measurement that keeps the
            # claim honest rung by rung.
            "tuning_full_transform_probe": True,
        }
        assert "tuning_full_transform_probe" not in windowed.knobs

    def test_streamed_locks_scheduling_off(self):
        cfg = _resolved("streamed")
        assert cfg["allow_scheduling"] is False


class TestMvmNeverStreams:
    """[t0_44 catch] value-domain configs have DORMANT spiking axes: their
    derived lif/streamed defaults must not leak legality (the P4 default
    flip silently locked allow_scheduling on the mvm scheduling flagship)."""

    def test_is_streamed_lif_false_for_mvm(self):
        from mimarsinan.chip_simulation.activation_semantics import is_streamed_lif

        assert is_streamed_lif({"core_semantics": "mvm"}) is False
        assert is_streamed_lif({
            "core_semantics": "mvm",
            "spiking_family": "lif",
            "spiking_variant": "streamed",
        }) is False

    def test_mvm_scheduled_config_resolves(self):
        import json
        from pathlib import Path

        from mimarsinan.config_schema.defaults import (
            get_default_deployment_parameters,
            get_default_platform_constraints,
        )
        from mimarsinan.config_schema.deployment_derivation import (
            derive_pipeline_runtime_parameters,
        )

        cfg = json.loads(Path(
            "templates/tier_0/t0_44_mvm_lenet5_wq_sched_pruned.json"
        ).read_text())
        merged = get_default_deployment_parameters()
        merged.update(cfg["deployment_parameters"])
        merged.update(get_default_platform_constraints())
        merged.update(cfg["platform_constraints"])
        derive_pipeline_runtime_parameters(merged)
        assert merged["allow_scheduling"] is True
