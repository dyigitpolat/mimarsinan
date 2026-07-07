"""The per-step QAT transform must not pay module-deepcopy/sync overhead.

The WQ endpoint applies the perceptron transform EVERY optimizer step; the
general mixing path deep-copied the whole perceptron module per perceptron per
step (the profiled 78%-of-wall overhead). Tuners whose previous-transform is a
declared identity (NAPQ: projection-only rungs) take a parameter-clone fast
path that is BIT-IDENTICAL: same parameter bits, same RNG consumption.
"""

from __future__ import annotations

import copy

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)


def _napq_tuner(tmp_path):
    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    torch.manual_seed(0)
    model = make_tiny_supermodel()
    return NormalizationAwarePerceptronQuantizationTuner(
        pipeline, model, quantization_bits=5, target_accuracy=0.5,
        lr=cfg["lr"], adaptation_manager=AdaptationManager(),
    )


def _params(perceptron):
    return {k: v.detach().clone() for k, v in perceptron.named_parameters()}


class TestIdentityPrevFastPath:
    def test_napq_declares_identity_previous_transform(self, tmp_path):
        tuner = _napq_tuner(tmp_path)
        try:
            assert tuner._prev_transform_is_identity is True
        finally:
            tuner.close()

    def test_fast_path_is_bit_identical_to_the_general_path(self, tmp_path):
        tuner = _napq_tuner(tmp_path)
        try:
            perceptron = next(iter(tuner.model.get_perceptrons()))
            ref = copy.deepcopy(perceptron)

            torch.manual_seed(1234)
            tuner._prev_transform_is_identity = False
            cache_general: dict = {}
            tuner._mixed_perceptron_transform(ref, 1.0, cache_general)
            rng_general = torch.random.get_rng_state()

            torch.manual_seed(1234)
            tuner._prev_transform_is_identity = True
            cache_fast: dict = {}
            tuner._mixed_perceptron_transform(perceptron, 1.0, cache_fast)
            rng_fast = torch.random.get_rng_state()

            for (name, got), (_, want) in zip(
                perceptron.named_parameters(), ref.named_parameters(),
            ):
                assert torch.equal(got, want), name
            assert torch.equal(rng_general, rng_fast), (
                "the fast path must consume the RNG stream exactly like the "
                "general path (mask draws preserved)"
            )
        finally:
            tuner.close()

    def test_fast_path_repeated_steps_match_general_path(self, tmp_path):
        # The endpoint case: one mask cache, many steps at rate 1.0.
        tuner = _napq_tuner(tmp_path)
        try:
            perceptron = next(iter(tuner.model.get_perceptrons()))
            ref = copy.deepcopy(perceptron)

            torch.manual_seed(7)
            tuner._prev_transform_is_identity = False
            cache_g: dict = {}
            for _ in range(3):
                tuner._mixed_perceptron_transform(ref, 1.0, cache_g)
            rng_g = torch.random.get_rng_state()

            torch.manual_seed(7)
            tuner._prev_transform_is_identity = True
            cache_f: dict = {}
            for _ in range(3):
                tuner._mixed_perceptron_transform(perceptron, 1.0, cache_f)
            rng_f = torch.random.get_rng_state()

            for (name, got), (_, want) in zip(
                perceptron.named_parameters(), ref.named_parameters(),
            ):
                assert torch.equal(got, want), name
            assert torch.equal(rng_g, rng_f)
        finally:
            tuner.close()

    def test_fast_path_below_full_rate_matches_too(self, tmp_path):
        tuner = _napq_tuner(tmp_path)
        try:
            perceptron = next(iter(tuner.model.get_perceptrons()))
            ref = copy.deepcopy(perceptron)

            torch.manual_seed(99)
            tuner._prev_transform_is_identity = False
            tuner._mixed_perceptron_transform(ref, 0.5, {})
            rng_g = torch.random.get_rng_state()

            torch.manual_seed(99)
            tuner._prev_transform_is_identity = True
            tuner._mixed_perceptron_transform(perceptron, 0.5, {})
            rng_f = torch.random.get_rng_state()

            for (name, got), (_, want) in zip(
                perceptron.named_parameters(), ref.named_parameters(),
            ):
                assert torch.equal(got, want), name
            assert torch.equal(rng_g, rng_f)
        finally:
            tuner.close()

    def test_general_tuners_default_to_the_deepcopy_path(self):
        from mimarsinan.tuning.tuners.perceptron_transform_tuner import (
            PerceptronTransformTuner,
        )

        assert PerceptronTransformTuner._prev_transform_is_identity is False


class TestNapqTransformSyncHygiene:
    def test_quantize_bounds_are_python_scalars(self):
        # Per-call CPU tensor creation fed CUDA kernels one implicit
        # transfer per quantize call; bounds must be plain floats.
        import inspect

        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )

        src = inspect.getsource(NormalizationAwarePerceptronQuantization.transform)
        assert "torch.tensor(" not in src
        assert "builtins.max" not in src

    def test_transform_is_value_identical_after_the_hygiene_pass(self, tmp_path):
        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )

        tuner = _napq_tuner(tmp_path)
        try:
            perceptron = next(iter(tuner.model.get_perceptrons()))
            napq = NormalizationAwarePerceptronQuantization(5, "cpu", rate=1.0)
            napq.transform(perceptron)
            first = _params(perceptron)
            scale_first = perceptron.parameter_scale
            napq.transform(perceptron)  # idempotent on the projected point
            for name, value in _params(perceptron).items():
                assert torch.equal(value, first[name]), name
            assert float(scale_first) == pytest.approx(
                float(perceptron.parameter_scale)
            )
        finally:
            tuner.close()


class TestDeviceResidency:
    """Per-step transform inputs must move WITH the model — a plain-attr tensor
    on the wrong device is one blocking H2D copy per access per step."""

    def test_per_input_scales_is_a_non_persistent_buffer(self):
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

        p = Perceptron(4, 3)
        p.per_input_scales = torch.ones(3)
        assert "per_input_scales" in dict(p.named_buffers())
        assert "per_input_scales" not in p.state_dict(), (
            "non-persistent: cache/golden/parity state must not change layout"
        )

    def test_frozen_stats_move_with_the_module_and_stay_out_of_state_dict(self):
        import torch.nn as nn

        from mimarsinan.models.nn.layers import FrozenStatsNormalization

        bn = nn.BatchNorm1d(3)
        bn.running_mean.uniform_(-1, 1)
        bn.running_var.uniform_(0.5, 2.0)
        frozen = FrozenStatsNormalization(bn)
        x = torch.randn(8, 3)
        want = frozen(x)
        assert "running_mean" in dict(frozen.named_buffers())
        assert "running_mean" not in frozen.state_dict()
        assert "running_var" not in frozen.state_dict()
        got = frozen(x)
        assert torch.equal(got, want)

    def test_norm_affine_params_has_no_unconditional_device_transfer(self):
        import inspect

        from mimarsinan.models.nn.layers import norm_affine_params

        src = inspect.getsource(norm_affine_params)
        assert "device !=" in src or ".device is not" in src, (
            "running stats must only transfer on a measured device mismatch"
        )

    def test_old_cache_pickles_migrate_plain_attrs_to_buffers(self):
        import pickle

        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

        p = Perceptron(4, 3)
        p.per_input_scales = torch.ones(3)
        # Emulate a pre-buffer cache: demote the buffer to a plain attr.
        pis = p._buffers.pop("per_input_scales")
        p.__dict__["per_input_scales"] = pis
        restored = pickle.loads(pickle.dumps(p))
        assert "per_input_scales" in restored._buffers
        assert torch.equal(restored.per_input_scales, pis)

    def test_old_frozen_stats_pickles_migrate_to_buffers(self):
        import pickle

        import torch.nn as nn

        from mimarsinan.models.nn.layers import FrozenStatsNormalization

        frozen = FrozenStatsNormalization(nn.BatchNorm1d(3))
        mean = frozen._buffers.pop("running_mean")
        var = frozen._buffers.pop("running_var")
        frozen.__dict__["running_mean"] = mean
        frozen.__dict__["running_var"] = var
        restored = pickle.loads(pickle.dumps(frozen))
        assert "running_mean" in restored._buffers
        assert "running_var" in restored._buffers
        assert torch.equal(restored.running_mean, mean)


class TestDeferredFiniteChecks:
    """One sync per perceptron transform, not one per raw write — the raise
    still names the offending write (fail-loud contract intact)."""

    def _perceptron(self):
        from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

        return Perceptron(3, 3, normalization=torch.nn.Identity())

    def test_default_mode_raises_immediately(self):
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        p = self._perceptron()
        t = PerceptronTransformer()
        with pytest.raises(RuntimeError, match="non-finite raw weight"):
            t._commit_raw_write(
                p, p.layer.weight, torch.full_like(p.layer.weight, float("nan")),
                "weight",
            )

    def test_deferred_mode_raises_at_flush_naming_the_write(self):
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        p = self._perceptron()
        t = PerceptronTransformer()
        with t.deferred_finite_checks(p):
            with pytest.raises(RuntimeError, match="non-finite raw bias"):
                t._commit_raw_write(p, p.layer.weight, p.layer.weight.data, "weight")
                t._commit_raw_write(
                    p, p.layer.bias, torch.full_like(p.layer.bias, float("inf")),
                    "bias",
                )
                t._flush_finite_checks()

    def test_deferred_mode_flushes_clean_writes_silently(self):
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        p = self._perceptron()
        t = PerceptronTransformer()
        with t.deferred_finite_checks(p):
            t._commit_raw_write(p, p.layer.weight, p.layer.weight.data, "weight")
        assert t._pending_finite == []

    def test_context_exit_flushes_pending_checks(self):
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        p = self._perceptron()
        t = PerceptronTransformer()
        with pytest.raises(RuntimeError, match="non-finite raw weight"):
            with t.deferred_finite_checks(p):
                t._commit_raw_write(
                    p, p.layer.weight,
                    torch.full_like(p.layer.weight, float("nan")), "weight",
                )

    def test_napq_transform_uses_the_deferred_mode(self):
        import inspect

        from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
            NormalizationAwarePerceptronQuantization,
        )

        src = inspect.getsource(NormalizationAwarePerceptronQuantization.transform)
        assert "deferred_finite_checks" in src


class TestHopFrontierStepCap:
    """The hop-frontier ramp caps per-rung training at 40 steps: the family's
    outcome is measured budget-insensitive while every rung step pays the
    O(S x depth) genuine segment forward."""

    def test_hop_frontier_driver_caps_steps_per_rate(self, tmp_path):
        from conftest import MockPipeline, default_config, make_tiny_supermodel
        from mimarsinan.tuning.orchestration.adaptation_manager import (
            AdaptationManager,
        )
        from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY
        from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
            TTFSCycleAdaptationTuner,
        )

        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 8
        cfg["ttfs_genuine_blend_ramp"] = True
        cfg["ttfs_genuine_blend_fast"] = True
        cfg["ttfs_prefix_ramp"] = True
        cfg["ttfs_hop_prefix_ramp"] = True
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=make_tiny_supermodel(), target_accuracy=0.5,
            lr=cfg["lr"], adaptation_manager=AdaptationManager(),
        )
        try:
            if getattr(tuner, "_hop_prefix_levels", None):
                assert (
                    tuner._fast_steps_per_rate
                    <= TUNING_POLICY.hop_stage_steps_per_rate
                )
            else:
                # The tiny fixture's chain may sit below the staging depth;
                # the cap constant still binds the armed path.
                assert TUNING_POLICY.hop_stage_steps_per_rate == 40
        finally:
            tuner.close()
