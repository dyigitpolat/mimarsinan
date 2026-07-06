"""TuningPolicy SSOT contract: frozen tuning-loop defaults replace never-configured keys.

The sixteen tuning-loop knobs below were only ever read via ``config.get`` with a
default (no config or template ever set them), so their defaults ARE the behavior.
This module pins (a) the exact frozen values, (b) that the adaptation mixins no
longer read the collapsed keys from config, and (c) that the keys are purged from
every config surface (defaults dict, CONFIG_KEYS_SET, namespaced registry).
"""

import dataclasses
from pathlib import Path

import pytest

from conftest import (
    MockPipeline,
    default_config,
    make_tiny_supermodel,
    override_tuning_policy,
)
from mimarsinan.config_schema.defaults import (
    CONFIG_KEYS_SET,
    DEFAULT_DEPLOYMENT_PARAMETERS,
)
from mimarsinan.config_schema.namespaced_schema import registered_flat_keys
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY, TuningPolicy


EXPECTED_POLICY_VALUES = {
    "checkpoint_scope": "full",
    "checkpoint_location": "device",
    "refind_lr_on_miss": False,
    "recovery_lr_plateau": False,
    "recovery_lr_plateau_factor": 0.3,
    "recovery_lr_plateau_reductions": 2,
    "recovery_check_divisor": 1,
    "rollback_ratchet": False,
    "rollback_cumulative_bound": 0.05,
    "tight_plateau": False,
    "keepbest_certified": False,
    "stabilization_bounded": False,
    "stabilization_ratio": 0.5,
    "use_paired_sensor": False,
    "k_commit": 2.0,
    "global_budget": 0.0,
    "dfq_keepbest_patience": 5,
    "prefix_stage_dfq_iters": 4,
    "prefix_stage_keepbest_interval": 25,
    "prefix_stage_lr": 1e-3,
    # [5u] floor-lifted endpoint LR ceiling (the probe-validated arm).
    "endpoint_floor_lr": 2e-3,
}

COLLAPSED_CONFIG_KEYS = frozenset({
    "tuning_refind_lr_on_miss",
    "tuning_recovery_lr_plateau",
    "tuning_recovery_lr_plateau_factor",
    "tuning_recovery_lr_plateau_reductions",
    "tuning_recovery_check_divisor",
    "tuning_rollback_ratchet",
    "tuning_rollback_cumulative_bound",
    "tuning_tight_plateau",
    "tuning_keepbest_certified",
    "tuning_stabilization_bounded",
    "tuning_stabilization_ratio",
    "tuning_use_paired_sensor",
    "checkpoint_scope",
    "checkpoint_location",
    "k_commit",
    "global_budget",
})


class TestPolicyValues:
    @pytest.mark.parametrize(
        "field,expected", sorted(EXPECTED_POLICY_VALUES.items())
    )
    def test_field_carries_the_effective_default(self, field, expected):
        value = getattr(TUNING_POLICY, field)
        assert value == expected
        assert type(value) is type(expected), (
            f"{field} must keep the default's type ({type(expected).__name__})"
        )

    def test_no_extra_or_missing_fields(self):
        assert {f.name for f in dataclasses.fields(TuningPolicy)} == set(
            EXPECTED_POLICY_VALUES
        )

    def test_module_instance_is_the_default_policy(self):
        assert TUNING_POLICY == TuningPolicy()

    def test_policy_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            TUNING_POLICY.k_commit = 3.0


class TestNoConfigReadsRemain:
    """The adaptation mixins must not read any collapsed key from config."""

    def _sources(self):
        import mimarsinan.tuning.orchestration.smooth_adaptation_cycle as cyc
        import mimarsinan.tuning.orchestration.smooth_adaptation_run as run

        return {
            "smooth_adaptation_cycle.py": Path(cyc.__file__).read_text(),
            "smooth_adaptation_run.py": Path(run.__file__).read_text(),
        }

    @pytest.mark.parametrize("key", sorted(COLLAPSED_CONFIG_KEYS))
    def test_mixins_do_not_config_get_key(self, key):
        for fname, src in self._sources().items():
            assert f'config.get("{key}"' not in src, f"{fname} still reads {key}"
            assert f"config.get('{key}'" not in src, f"{fname} still reads {key}"

    def test_full_transform_probe_stays_a_config_read(self):
        # ConversionPolicy WRITES tuning_full_transform_probe (cascaded recipe),
        # so the cycle mixin must keep reading it from config.
        src = self._sources()["smooth_adaptation_cycle.py"]
        assert 'config.get("tuning_full_transform_probe"' in src


class TestConfigSurfacePurged:
    @pytest.mark.parametrize("key", sorted(COLLAPSED_CONFIG_KEYS))
    def test_absent_from_default_deployment_parameters(self, key):
        assert key not in DEFAULT_DEPLOYMENT_PARAMETERS

    @pytest.mark.parametrize("key", sorted(COLLAPSED_CONFIG_KEYS))
    def test_absent_from_config_keys_set(self, key):
        assert key not in CONFIG_KEYS_SET

    @pytest.mark.parametrize("key", sorted(COLLAPSED_CONFIG_KEYS))
    def test_absent_from_namespaced_registry(self, key):
        assert key not in registered_flat_keys()

    def test_user_tuning_keys_survive(self):
        for key in ("tuning_budget_scale", "tuning_recipe"):
            assert key in DEFAULT_DEPLOYMENT_PARAMETERS, key
            assert key in CONFIG_KEYS_SET, key
        assert "tuning_batch_size" in CONFIG_KEYS_SET


# ── policy → mixin wiring ────────────────────────────────────────────────────


def _build_tuner(tmp_path):
    from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
        SmoothAdaptationTuner,
    )

    class _T(SmoothAdaptationTuner):
        def _update_and_evaluate(self, rate):
            return 0.9

        def _find_lr(self):
            return 0.001

    pipeline = MockPipeline(config=default_config(), working_directory=str(tmp_path))
    return _T(pipeline, make_tiny_supermodel(), target_accuracy=0.9, lr=0.001)


class TestPolicyWiring:
    def test_default_policy_drives_the_cycle_mixin(self, tmp_path):
        tuner = _build_tuner(tmp_path)
        assert tuner._checkpoint_guard.scope == "full"
        assert tuner._checkpoint_guard.location == "device"
        assert tuner._refind_lr_on_miss is False
        assert tuner._recovery_lr_plateau is False
        assert tuner._recovery_lr_plateau_factor == 0.3
        assert tuner._recovery_lr_plateau_reductions == 2
        assert tuner._recovery_check_divisor == 1
        assert tuner._rollback_ratchet is False
        assert tuner._rollback_cumulative_bound == 0.05
        assert tuner._tight_plateau is False
        assert tuner._keepbest_certified is False
        assert tuner._stabilization_bounded is False
        assert tuner._stabilization_ratio == 0.5
        assert tuner._paired_gate is False
        assert tuner._k_commit == 2.0
        assert tuner._global_budget == 0.0
        tuner.close()

    def test_overridden_policy_drives_the_cycle_mixin(self, tmp_path, monkeypatch):
        override_tuning_policy(
            monkeypatch,
            rollback_ratchet=True,
            recovery_check_divisor=4,
            k_commit=3.0,
        )
        tuner = _build_tuner(tmp_path)
        assert tuner._rollback_ratchet is True
        assert tuner._recovery_check_divisor == 4
        assert tuner._k_commit == 3.0
        tuner.close()

    def test_negative_global_budget_still_rejected(self, tmp_path, monkeypatch):
        override_tuning_policy(monkeypatch, global_budget=-0.01)
        with pytest.raises(ValueError, match="global_budget"):
            _build_tuner(tmp_path)
