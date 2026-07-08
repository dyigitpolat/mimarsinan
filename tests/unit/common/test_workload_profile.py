"""The workload-profile injection contract: precedence, folding, resolution."""

import dataclasses

import pytest

from mimarsinan.common.workload_profile import (
    CalibrationSetPolicy,
    DataWorkloadProfile,
    ModelWorkloadProfile,
    ResolvedWorkloadProfile,
    fold_workload_profiles,
)


class TestProfileDefaultsAreAbsent:
    """Absent profile == today's behavior: every registration field is None."""

    @pytest.mark.parametrize(
        "profile_cls",
        [CalibrationSetPolicy, DataWorkloadProfile, ModelWorkloadProfile],
    )
    def test_all_fields_default_to_none(self, profile_cls):
        profile = profile_cls()
        for f in dataclasses.fields(profile):
            assert getattr(profile, f.name) is None, f.name

    def test_default_profiles_produce_no_config_updates(self):
        assert DataWorkloadProfile().config_updates() == {}
        assert ModelWorkloadProfile().config_updates() == {}

    def test_profiles_are_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            DataWorkloadProfile().eval_subsample_target = 1  # type: ignore[misc]


class TestConfigUpdates:
    def test_data_profile_maps_range_to_input_scale_upper_bound(self):
        profile = DataWorkloadProfile(input_value_range=(0.0, 2.75))
        assert profile.config_updates() == {"input_data_scale": 2.75}

    def test_data_profile_maps_budget_fields(self):
        profile = DataWorkloadProfile(
            eval_subsample_target=20000, tuning_step_cap_epochs=1.5
        )
        assert profile.config_updates() == {
            "eval_subsample_target": 20000,
            "tuning_step_cap_epochs": 1.5,
        }

    def test_data_profile_maps_declared_calibration_fields_only(self):
        profile = DataWorkloadProfile(
            calibration=CalibrationSetPolicy(distmatch_bias_iters=20, stat_batches=7)
        )
        assert profile.config_updates() == {
            "calibration_set_policy": {"distmatch_bias_iters": 20, "stat_batches": 7},
        }

    def test_model_profile_maps_its_fields_one_to_one(self):
        profile = ModelWorkloadProfile(
            prefix_stage_lr=5e-4,
            endpoint_floor_lr=1e-3,
            pretrained_weight_source="torchvision",
            proven_recovery_depth=9,
            clamp_cuda_assert_prone=True,
        )
        assert profile.config_updates() == {
            "prefix_stage_lr": 5e-4,
            "endpoint_floor_lr": 1e-3,
            "pretrained_weight_source": "torchvision",
            "proven_recovery_depth": 9,
            "clamp_cuda_assert_prone": True,
        }


class TestFoldPrecedence:
    """Explicit config > model profile > data profile > framework default."""

    def test_all_none_profiles_leave_config_untouched(self):
        config = {"lr": 0.001, "input_data_scale": 3.0}
        fold_workload_profiles(
            config,
            model_profile=ModelWorkloadProfile(),
            data_profile=DataWorkloadProfile(),
        )
        assert config == {"lr": 0.001, "input_data_scale": 3.0}

    def test_explicit_config_beats_the_data_profile(self):
        config = {"input_data_scale": 2.0}
        fold_workload_profiles(
            config, data_profile=DataWorkloadProfile(input_value_range=(0.0, 5.0))
        )
        assert config["input_data_scale"] == 2.0

    def test_data_profile_fills_absent_keys(self):
        config = {}
        fold_workload_profiles(
            config,
            data_profile=DataWorkloadProfile(
                input_value_range=(0.0, 5.0), eval_subsample_target=10000
            ),
        )
        assert config == {"input_data_scale": 5.0, "eval_subsample_target": 10000}

    def test_explicit_config_beats_the_model_profile(self):
        config = {"prefix_stage_lr": 1e-2}
        fold_workload_profiles(
            config, model_profile=ModelWorkloadProfile(prefix_stage_lr=5e-4)
        )
        assert config["prefix_stage_lr"] == 1e-2

    def test_model_profile_is_folded_before_the_data_profile(self):
        applied = []
        config = {}

        class _Recorder(dict):
            def setdefault(self, key, value):
                applied.append(key)
                return super().setdefault(key, value)

        config = _Recorder()
        fold_workload_profiles(
            config,
            model_profile=ModelWorkloadProfile(prefix_stage_lr=5e-4),
            data_profile=DataWorkloadProfile(eval_subsample_target=10000),
        )
        assert applied == ["prefix_stage_lr", "eval_subsample_target"]

    def test_calibration_merge_is_field_wise_with_explicit_winning(self):
        config = {"calibration_set_policy": {"distmatch_bias_iters": 30}}
        fold_workload_profiles(
            config,
            data_profile=DataWorkloadProfile(
                calibration=CalibrationSetPolicy(
                    distmatch_bias_iters=20, stat_batches=7
                )
            ),
        )
        assert config["calibration_set_policy"] == {
            "distmatch_bias_iters": 30,
            "stat_batches": 7,
        }


class TestResolvedWorkloadProfile:
    def test_empty_config_resolves_to_the_workload_neutral_defaults(self):
        resolved = ResolvedWorkloadProfile.from_config({})
        assert resolved == ResolvedWorkloadProfile()
        assert resolved.input_data_scale == 1.0
        assert resolved.calibration == CalibrationSetPolicy()
        assert resolved.clamp_cuda_assert_prone is False
        assert resolved.eval_subsample_target is None
        assert resolved.tuning_step_cap_epochs is None
        assert resolved.prefix_stage_lr is None
        assert resolved.endpoint_floor_lr is None
        assert resolved.pretrained_weight_source is None
        assert resolved.proven_recovery_depth is None

    def test_config_keys_resolve_typed(self):
        resolved = ResolvedWorkloadProfile.from_config({
            "input_data_scale": 5,
            "eval_subsample_target": 10000.0,
            "tuning_step_cap_epochs": 2,
            "calibration_set_policy": {"gauge_batches": 4},
            "prefix_stage_lr": 5e-4,
            "endpoint_floor_lr": 1e-3,
            "pretrained_weight_source": "torchvision",
            "proven_recovery_depth": 9,
            "clamp_cuda_assert_prone": True,
        })
        assert resolved.input_data_scale == 5.0
        assert resolved.eval_subsample_target == 10000
        assert resolved.tuning_step_cap_epochs == 2.0
        assert resolved.calibration.gauge_batches == 4
        assert resolved.calibration.distmatch_bias_iters is None
        assert resolved.prefix_stage_lr == 5e-4
        assert resolved.endpoint_floor_lr == 1e-3
        assert resolved.pretrained_weight_source == "torchvision"
        assert resolved.proven_recovery_depth == 9
        assert resolved.clamp_cuda_assert_prone is True

    def test_unknown_calibration_fields_fail_loud(self):
        with pytest.raises(ValueError, match="calibration_set_policy"):
            ResolvedWorkloadProfile.from_config(
                {"calibration_set_policy": {"nonsense": 1}}
            )

    def test_registration_fold_resolution_round_trip(self):
        config = {}
        fold_workload_profiles(
            config,
            model_profile=ModelWorkloadProfile(endpoint_floor_lr=4e-3),
            data_profile=DataWorkloadProfile(input_value_range=(-1.8, 2.25)),
        )
        resolved = ResolvedWorkloadProfile.from_config(config)
        assert resolved.input_data_scale == 2.25
        assert resolved.endpoint_floor_lr == 4e-3
