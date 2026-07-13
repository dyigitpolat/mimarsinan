"""The LIF exactness-promotion knobs (lif_deployment_exactness.md §7).

Registry-validated keys, default OFF; the LIF recipe turns the exact/statistical
corrections ON: membrane readout C2, depth-balancing relays C5. The affine fold
C4 stays recipe-OFF (A/B-measured harmful post-QAT). Per-hop re-timing C3 is
armed by the exact-QAT PAIRING at fold time (the trained staircase is the
per-hop twin, parity 1.0000); standalone it stays off — twinless arming breaks
the torch<->deployed-sim parity gate (t0_01 0.8438).
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

# Knobs-dict OFF: affine fold (A/B 2026-07-13 harmful in interaction on the
# trained composition: no_fold 0.9061 vs no_both 0.9307) and retiming, whose
# arm is OWNED by the exact-QAT pairing at fold time (recipe-armed
# lif_exact_qat pairs it; the pair downgrades together on Novena/opt-out).
RECIPE_OFF = ("lif_per_hop_retiming", "lif_affine_fold")
DERIVED_ON = RECIPE_ON + ("lif_per_hop_retiming",)
DERIVED_OFF = ("lif_affine_fold",)


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

    def test_lif_recipe_knobs_leave_the_paired_arms_to_the_fold(self):
        # The knobs dict keeps both False: the affine fold stays off (A/B
        # harmful), and the retiming arm belongs to the exact-QAT pairing so
        # the pair can downgrade together (Novena, explicit opt-out).
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
        for key in DERIVED_ON:
            assert dp[key] is True, key
        for key in DERIVED_OFF:
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
