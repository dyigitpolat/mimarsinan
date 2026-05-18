"""Round-trip tests for the compilagent plan <-> mimarsinan config codec.

Goal: pin down that any valid configuration the search step might emit can
be encoded into a ``Plan`` and decoded back into the same configuration,
and that malformed interventions raise ``PlanCodecError`` so the backend
can surface a clean retry message to the agent.
"""

from __future__ import annotations

import pytest
from compilagent import Intervention, Plan, Target

from mimarsinan.search.optimizers.compilagent.plan_codec import (
    ARCH_KIND,
    HW_CORE_KIND,
    CodecDefaults,
    PlanCodecError,
    decode_plan,
    encode_plan,
)
from mimarsinan.search.search_space_description import SearchSpaceDescription


def _description(search_mode: str = "joint") -> SearchSpaceDescription:
    return SearchSpaceDescription.from_arch_search(
        search_mode=search_mode,
        arch_options=(
            ("activation", ("ReLU", "LeakyReLU", "GELU")),
            ("fc_w_1", (32, 64, 128)),
        ),
        arch_cfg={
            "num_core_types": 2,
            "core_axons_bounds": [64, 1024],
            "core_neurons_bounds": [64, 1024],
            "core_count_bounds": [50, 500],
        },
        target_tq=32,
    )


def _make_plan(triples):
    interventions = tuple(
        Intervention(target=Target(kind=k, selector=s), payload=p)
        for (k, s, p) in triples
    )
    return Plan(interventions=interventions)


class TestDecode:
    def test_empty_plan_yields_defaults(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        cfg = decode_plan(Plan(), defaults)
        assert cfg["model_config"] == defaults.model_config
        # Defaults snapshot is independent of returned cfg
        assert cfg["platform_constraints"]["cores"] is not defaults.platform_constraints["cores"]

    def test_arch_intervention_overrides_default(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        plan = _make_plan([(ARCH_KIND, "activation", "ReLU")])
        cfg = decode_plan(plan, defaults)
        assert cfg["model_config"]["activation"] == "ReLU"

    def test_hw_core_intervention_overrides_dim(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        plan = _make_plan([
            (HW_CORE_KIND, "0.max_axons", 256),
            (HW_CORE_KIND, "1.count", 400),
        ])
        cfg = decode_plan(plan, defaults)
        assert cfg["platform_constraints"]["cores"][0]["max_axons"] == 256
        assert cfg["platform_constraints"]["cores"][1]["count"] == 400

    def test_later_intervention_wins(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        plan = _make_plan([
            (ARCH_KIND, "fc_w_1", 32),
            (ARCH_KIND, "fc_w_1", 128),
        ])
        cfg = decode_plan(plan, defaults)
        assert cfg["model_config"]["fc_w_1"] == 128

    def test_unknown_kind_raises(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        plan = _make_plan([("made_up", "x", 1)])
        with pytest.raises(PlanCodecError):
            decode_plan(plan, defaults)

    def test_arch_with_empty_selector_raises(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        plan = _make_plan([(ARCH_KIND, "", 1)])
        with pytest.raises(PlanCodecError):
            decode_plan(plan, defaults)

    def test_hw_core_bad_selector_raises(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        for bad in ("foo", "0.bad_dim", "abc.max_axons"):
            plan = _make_plan([(HW_CORE_KIND, bad, 256)])
            with pytest.raises(PlanCodecError):
                decode_plan(plan, defaults)

    def test_hw_core_non_int_payload_raises(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        plan = _make_plan([(HW_CORE_KIND, "0.max_axons", "lots")])
        with pytest.raises(PlanCodecError):
            decode_plan(plan, defaults)


class TestEncode:
    def test_unchanged_config_yields_no_interventions(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        triples = encode_plan(
            {
                "model_config": dict(defaults.model_config),
                "platform_constraints": {
                    "cores": [dict(c) for c in defaults.platform_constraints["cores"]],
                    "target_tq": defaults.platform_constraints["target_tq"],
                    "weight_bits": defaults.platform_constraints["weight_bits"],
                },
            },
            defaults,
            description=d,
        )
        assert triples == ()

    def test_arch_change_emits_one_intervention(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        cfg = {
            "model_config": {**dict(defaults.model_config), "activation": "ReLU"},
            "platform_constraints": defaults.platform_constraints,
        }
        triples = encode_plan(cfg, defaults, description=d)
        assert (ARCH_KIND, "activation", "ReLU") in triples
        # Other arch values stayed at defaults so should not be emitted.
        emitted_keys = [t[1] for t in triples if t[0] == ARCH_KIND]
        assert emitted_keys == ["activation"]

    def test_hw_dim_change_emits_only_changed_dims(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        new_cores = [dict(c) for c in defaults.platform_constraints["cores"]]
        new_cores[0]["max_axons"] = 256
        cfg = {
            "model_config": dict(defaults.model_config),
            "platform_constraints": {
                "cores": new_cores,
                "target_tq": defaults.platform_constraints["target_tq"],
                "weight_bits": defaults.platform_constraints["weight_bits"],
            },
        }
        triples = encode_plan(cfg, defaults, description=d)
        assert (HW_CORE_KIND, "0.max_axons", 256) in triples
        # No other hw triples emitted
        assert [t for t in triples if t[0] == HW_CORE_KIND] == [
            (HW_CORE_KIND, "0.max_axons", 256)
        ]


class TestRoundTrip:
    def test_decode_encode_decode_is_idempotent(self):
        d = _description()
        defaults = CodecDefaults.from_description(d)
        # Construct a non-trivial configuration
        cfg_in = {
            "model_config": {**dict(defaults.model_config), "activation": "GELU", "fc_w_1": 128},
            "platform_constraints": {
                "cores": [
                    {"max_axons": 256, "max_neurons": 512, "count": 100},
                    {"max_axons": 1024, "max_neurons": 1024, "count": 200},
                ],
                "target_tq": defaults.platform_constraints["target_tq"],
                "weight_bits": defaults.platform_constraints["weight_bits"],
            },
        }
        triples = encode_plan(cfg_in, defaults, description=d)
        plan = _make_plan(triples)
        cfg_out = decode_plan(plan, defaults)
        # Arch changes round-tripped
        assert cfg_out["model_config"]["activation"] == "GELU"
        assert cfg_out["model_config"]["fc_w_1"] == 128
        # HW changes round-tripped
        for i, expected in enumerate(cfg_in["platform_constraints"]["cores"]):
            for dim in ("max_axons", "max_neurons", "count"):
                assert cfg_out["platform_constraints"]["cores"][i][dim] == expected[dim]
