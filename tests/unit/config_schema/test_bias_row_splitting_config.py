"""Bias-row splitting is a DEPLOYMENT CONFIG, not a backend special case: the
registry declares it, ``DeploymentPlan`` resolves it, and every consumer reads
the resolved decision.

The platform capability gates it the same way it gates the two-scale projection
— a core with an on-chip bias register has no always-on row to split.
"""

import pytest

from mimarsinan.config_schema.registry import REGISTRY, parse_deployment_document
from mimarsinan.config_schema.registry.types import Category, FieldType
from mimarsinan.mapping.support.bias_rows import (
    BIAS_ROW_SPLITTING_MODES,
    MODE_KEY,
    ROWS_KEY,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    resolve_bias_row_splitting,
    resolve_wq_weight_only_grid,
)


def _config(mode="off", rows=0, has_bias=False, two_scale=False):
    return {
        MODE_KEY: mode,
        ROWS_KEY: rows,
        "wq_two_scale_projection": two_scale,
        "cores": [
            {"max_axons": 128, "max_neurons": 256, "count": 512, "has_bias": has_bias},
        ],
    }


class TestRegistryDeclaration:
    def test_both_keys_are_registered_under_the_bias_owner(self):
        for key in (MODE_KEY, ROWS_KEY):
            assert key in REGISTRY, key
            assert REGISTRY[key].owner == "mapping/bias"
            assert REGISTRY[key].section == "platform_constraints"

    def test_the_schema_literals_do_not_drift_from_the_mapping_ssot(self):
        """The registry cannot import the mapping SSOT (import cycle), so it
        restates the vocabulary; this is the guard that keeps them one."""
        from mimarsinan.config_schema.registry import entries_platform_bias as ep

        assert ep.BIAS_ROW_SPLITTING_MODE_KEY == MODE_KEY
        assert ep.BIAS_ROWS_KEY == ROWS_KEY
        assert ep.BIAS_ROW_SPLITTING_MODES == BIAS_ROW_SPLITTING_MODES

    def test_the_mode_is_an_enumerated_user_choice(self):
        entry = REGISTRY[MODE_KEY]
        assert entry.type is FieldType.ENUM
        assert entry.options == ("off", "auto", "fixed")
        assert entry.exposure == "user"
        assert entry.default == "off"

    def test_the_row_override_is_relevant_only_in_fixed_mode(self):
        entry = REGISTRY[ROWS_KEY]
        assert entry.type is FieldType.INT
        assert entry.category is Category.ADVANCED
        assert entry.relevant is not None
        assert entry.relevant.evaluate({MODE_KEY: "fixed"})
        assert not entry.relevant.evaluate({MODE_KEY: "auto"})
        assert not entry.relevant.evaluate({MODE_KEY: "off"})

    def test_a_document_declaring_the_keys_parses_clean(self):
        parsed = parse_deployment_document({
            "platform_constraints": {MODE_KEY: "auto", ROWS_KEY: 0},
        })
        assert parsed.unknown == []


class TestResolution:
    def test_off_is_the_default_and_is_inactive(self):
        assert not resolve_bias_row_splitting({}).active
        assert not resolve_bias_row_splitting(_config()).active

    def test_auto_activates_on_a_param_encoded_platform(self):
        split = resolve_bias_row_splitting(_config(mode="auto"))
        assert split.active
        assert split.rows_override is None

    def test_an_on_chip_bias_lane_leaves_it_inactive(self):
        # Nothing to split: the bias lives in a register, not a crossbar row.
        assert not resolve_bias_row_splitting(_config(mode="auto", has_bias=True)).active

    def test_fixed_carries_the_declared_row_count(self):
        split = resolve_bias_row_splitting(_config(mode="fixed", rows=8))
        assert split.active
        assert split.rows_override == 8

    def test_fixed_without_a_row_count_fails_loud(self):
        with pytest.raises(ValueError, match=ROWS_KEY):
            resolve_bias_row_splitting(_config(mode="fixed", rows=0))

    def test_an_unknown_mode_fails_loud(self):
        with pytest.raises(ValueError, match="bias_row_splitting"):
            resolve_bias_row_splitting(_config(mode="sometimes"))


class TestWeightOnlyGrid:
    def test_splitting_buys_the_weight_only_grid_on_a_param_encoded_platform(self):
        assert resolve_wq_weight_only_grid(_config(mode="auto"))

    def test_off_keeps_the_shared_grid(self):
        assert not resolve_wq_weight_only_grid(_config())

    def test_the_two_scale_flag_still_owns_the_on_chip_platform(self):
        assert resolve_wq_weight_only_grid(_config(has_bias=True, two_scale=True))
        assert not resolve_wq_weight_only_grid(_config(has_bias=False, two_scale=True))


class TestPlanResolution:
    def test_the_plan_carries_the_resolved_decision(self):
        plan = DeploymentPlan.resolve(_config(mode="fixed", rows=4))
        assert plan.bias_row_splitting.active
        assert plan.bias_row_splitting.rows_override == 4

    def test_a_silent_document_resolves_to_off(self):
        assert not DeploymentPlan.resolve({}).bias_row_splitting.active
