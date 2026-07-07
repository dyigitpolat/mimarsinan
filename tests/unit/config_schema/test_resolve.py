"""resolve_draft: derived values with WHY, keyed errors, diff-vs-defaults."""

import json
import os

from mimarsinan.config_schema.resolve import resolve_draft

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _t0_01() -> dict:
    path = os.path.join(
        _REPO_ROOT, "test_configs", "tier0", "t0_01_lif_mmixcore_wq_s4.json"
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestResolveTierConfig:
    def test_resolves_clean(self):
        res = resolve_draft(_t0_01())
        assert res.ok
        assert res.unknown_keys == []
        assert res.resolved["activation_quantization"] is True
        assert res.resolved["pipeline_mode"] == "phased"

    def test_derived_chips_carry_why(self):
        res = resolve_draft(_t0_01())
        aq = res.derived["activation_quantization"]
        assert aq["value"] is True
        assert "lif" in aq["why"]
        assert aq["derived_from"]
        assert res.derived["pipeline_mode"]["why"].startswith("phased")

    def test_explicit_keys_and_diff(self):
        res = resolve_draft(_t0_01())
        assert "endpoint_floor_steps" in res.explicit_keys
        by_key = {row["key"]: row for row in res.diff_vs_defaults}
        assert by_key["lr"]["differs"] is True          # 0.003 vs default 0.001
        assert by_key["lr"]["default"] == 0.001
        assert by_key["sanafe_arch_preset"]["differs"] is False


class TestResolveErrors:
    def test_contradicting_aq_is_a_keyed_error(self):
        draft = {
            "pipeline_mode": "phased",
            "deployment_parameters": {
                "spiking_mode": "lif",
                "activation_quantization": False,
                "weight_quantization": True,
            },
        }
        res = resolve_draft(draft)
        assert not res.ok
        assert any(
            e["key"] == "activation_quantization" and e["rule_id"] == "derivation"
            for e in res.errors
        )

    def test_bits_driven_wq_contract_is_a_keyed_error(self):
        draft = {
            "pipeline_mode": "phased",
            "deployment_parameters": {"weight_quantization": False},
            "platform_constraints": {"weight_bits": 5},
        }
        res = resolve_draft(draft)
        assert any(
            e["key"] == "weight_quantization" and e["rule_id"] == "quantization_assembly"
            for e in res.errors
        )

    def test_reserved_s_allocation_mode_attaches_to_its_key(self):
        draft = dict(_t0_01())
        draft["deployment_parameters"] = dict(draft["deployment_parameters"])
        draft["deployment_parameters"]["s_allocation"] = "explicit"
        res = resolve_draft(draft)
        assert any(e["key"] == "s_allocation" for e in res.errors)

    def test_unknown_keys_are_surfaced_not_dropped(self):
        draft = _t0_01()
        draft["deployment_parameters"]["endpoint_floor_wall_s"] = 60
        res = resolve_draft(draft)
        assert res.unknown_keys == ["deployment_parameters.endpoint_floor_wall_s"]

    def test_config_mistakes_never_raise(self):
        res = resolve_draft({"deployment_parameters": {"spiking_mode": "rate"}})
        assert not res.ok
        assert any(e["key"] == "spiking_mode" for e in res.errors)
