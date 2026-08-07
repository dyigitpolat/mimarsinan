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

    def test_streamed_defers_loihi_with_reason(self):
        cfg = _resolved("streamed")
        assert cfg["enable_loihi_simulation"] is False
        assert cfg["enable_nevresim_simulation"] is True
        assert cfg["enable_sanafe_simulation"] is True
        assert _resolved("synchronized")["enable_loihi_simulation"] is True

    def test_policy_row_is_marked_streamed(self):
        recipe = ConversionPolicy.derive("lif", spiking_variant="streamed")
        assert recipe.special_case == "streamed_raw_cascade"
        assert "streaming" in recipe.rationale
        windowed = ConversionPolicy.derive("lif")
        assert windowed.special_case == "bn_freeze"
        assert recipe.knobs == {**windowed.knobs, "lif_exact_qat": False}

    def test_streamed_locks_scheduling_off(self):
        cfg = _resolved("streamed")
        assert cfg["allow_scheduling"] is False
