"""The LIF exactness-promotion knobs (lif_deployment_exactness.md §7).

Registry-validated keys, default OFF; the LIF recipe turns the exact/statistical
corrections ON: membrane readout C2, affine fold C4, depth-balancing relays C5.
Per-hop re-timing C3 stays recipe-OFF: its deployed uniform re-encode has no
NF-twin counterpart, so arming it breaks the torch<->deployed-sim parity gate
(t0_01 0.8438 vs 1.0000 per-neuron-exact with it off); config-explicit arming
remains available for research.
"""

from __future__ import annotations

from mimarsinan.config_schema.defaults import CONFIG_KEYS_SET
from mimarsinan.config_schema.deployment_derivation import derive_deployment_parameters
from mimarsinan.config_schema.registry import REGISTRY, FieldType
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

KNOBS = (
    "lif_membrane_readout",
    "lif_affine_fold",
    "lif_per_hop_retiming",
    "lif_depth_balancing_relays",
)

RECIPE_ON = (
    "lif_membrane_readout",
    "lif_depth_balancing_relays",
)

# retiming: twinless (parity break, t0_01 gate 0.8438); affine fold: A/B
# 2026-07-13 measured harmful in interaction on the trained composition
# (no_fold 0.9061 vs no_both 0.9307 vs pre-arming 0.9326).
RECIPE_OFF = ("lif_per_hop_retiming", "lif_affine_fold")


class TestRegistry:
    def test_knobs_are_registry_validated_bools(self):
        for key in KNOBS:
            entry = REGISTRY[key]
            assert entry.type is FieldType.BOOL, key
            assert entry.doc, key

    def test_knobs_are_config_keys(self):
        for key in KNOBS:
            assert key in CONFIG_KEYS_SET, key


class TestRecipeFold:
    def test_lif_recipe_arms_the_exact_corrections(self):
        knobs = ConversionPolicy.derive("lif").knobs
        for key in RECIPE_ON:
            assert knobs.get(key) is True, key

    def test_lif_recipe_keeps_the_twinless_retiming_off(self):
        # retiming: no NF-twin counterpart (gate 0.8438); affine fold:
        # A/B-measured harmful post-QAT — both fail toward measured.
        knobs = ConversionPolicy.derive("lif").knobs
        for key in RECIPE_OFF:
            assert knobs.get(key) is False, key

    def test_non_lif_recipes_never_arm_them(self):
        for mode in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            knobs = ConversionPolicy.derive(mode).knobs
            for key in KNOBS:
                assert key not in knobs, (mode, key)

    def test_derivation_folds_recipe_on_for_lif(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "activation_quantization": True}
        derive_deployment_parameters(dp)
        for key in RECIPE_ON:
            assert dp[key] is True, key
        for key in RECIPE_OFF:
            assert dp[key] is False, key

    def test_explicit_off_wins_over_the_recipe(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "activation_quantization": True,
              "lif_membrane_readout": False,
              "lif_affine_fold": False,
              "lif_per_hop_retiming": False,
              "lif_depth_balancing_relays": False}
        derive_deployment_parameters(dp)
        for key in KNOBS:
            assert dp[key] is False, key

    def test_explicit_retiming_arm_wins_over_the_recipe_off(self):
        dp = {"spiking_mode": "lif", "weight_quantization": True,
              "activation_quantization": True,
              "lif_per_hop_retiming": True}
        derive_deployment_parameters(dp)
        assert dp["lif_per_hop_retiming"] is True

    def test_ttfs_derivation_leaves_knobs_unset(self):
        dp = {"spiking_mode": "ttfs_quantized", "weight_quantization": True}
        derive_deployment_parameters(dp)
        for key in KNOBS:
            assert key not in dp, key
