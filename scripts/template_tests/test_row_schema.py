"""[D7] Tier rows are a CLOSED schema: no key may be silently ignored.

A row key that the generator does not read used to vanish without a word —
`finetune_lr=1e-4` evaporated and an experiment cycle was spent believing a
hypothesis had been tested. Every key must now be either structural (the
generator's own vocabulary) or a registry-known config key routed to the
section the registry declares.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "tier_generate", ROOT / "templates" / "generate.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


class TestUnknownKeysFailLoud:
    def test_typo_is_rejected_with_a_suggestion(self):
        row = {"n": 1, "mode": "lif", "quant": "wq", "wb": 8, "s": 8,
               "vehicle": "lenet5", "finetune_lrr": 1e-4}
        with pytest.raises(ValueError) as excinfo:
            gen.validate_row_keys(row)
        message = str(excinfo.value)
        assert "finetune_lrr" in message
        assert "finetune_lr" in message  # nearest registry match offered

    def test_structural_keys_are_accepted(self):
        gen.validate_row_keys({
            "n": 1, "mode": "mvm", "quant": "wq", "wb": 8, "vehicle": "lenet5",
            "tags": ["sched"], "note": "x", "scheduling": True,
            "extra_dp": {}, "extra_pc": {},
        })

    def test_registry_config_keys_are_accepted(self):
        gen.validate_row_keys({
            "n": 1, "mode": "lif", "quant": "wq", "wb": 8, "s": 8,
            "vehicle": "lenet5", "finetune_lr": 1e-4, "weight_bits": 8,
        })


class TestRoutingBySection:
    def test_deployment_key_lands_in_deployment_parameters(self):
        assert gen.row_config_keys(
            {"finetune_lr": 1e-4}, "deployment_parameters"
        ) == {"finetune_lr": 1e-4}
        assert gen.row_config_keys(
            {"finetune_lr": 1e-4}, "platform_constraints"
        ) == {}

    def test_platform_key_lands_in_platform_constraints(self):
        assert gen.row_config_keys(
            {"activation_bits": 8}, "platform_constraints"
        ) == {"activation_bits": 8}
        assert gen.row_config_keys(
            {"activation_bits": 8}, "deployment_parameters"
        ) == {}

    def test_structural_keys_never_route(self):
        for section in ("deployment_parameters", "platform_constraints"):
            assert gen.row_config_keys({"vehicle": "lenet5", "n": 3}, section) == {}


class TestEveryAuthoredRowValidates:
    @pytest.mark.parametrize("tier", ["T0", "T1", "T2", "T3"])
    def test_authored_rows_carry_only_known_keys(self, tier):
        for row in getattr(gen, tier):
            gen.validate_row_keys(row)  # must not raise
