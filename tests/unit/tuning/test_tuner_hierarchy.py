"""Tests for TunerBase / SmoothAdaptationTuner hierarchy."""

import pytest

from mimarsinan.tuning.unified_tuner import TunerBase, SmoothAdaptationTuner
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner
from mimarsinan.tuning.tuners.activation_adaptation_tuner import ActivationAdaptationTuner
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.tuning.tuners.activation_shift_tuner import ActivationShiftTuner
from mimarsinan.tuning.tuners.noise_tuner import NoiseTuner
from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner
from mimarsinan.tuning.tuners.perceptron_transform_tuner import PerceptronTransformTuner
from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
    NormalizationAwarePerceptronQuantizationTuner,
)


class TestHierarchy:
    """Verify class hierarchy relationships."""

    def test_smooth_adaptation_tuner_extends_tuner_base(self):
        assert issubclass(SmoothAdaptationTuner, TunerBase)

    def test_clamp_tuner_extends_smooth_adaptation(self):
        assert issubclass(ClampTuner, SmoothAdaptationTuner)

    def test_activation_adaptation_tuner_extends_smooth_adaptation(self):
        assert issubclass(ActivationAdaptationTuner, SmoothAdaptationTuner)

    def test_activation_quantization_tuner_extends_smooth_adaptation(self):
        assert issubclass(ActivationQuantizationTuner, SmoothAdaptationTuner)

    def test_noise_tuner_extends_smooth_adaptation(self):
        assert issubclass(NoiseTuner, SmoothAdaptationTuner)

    def test_pruning_tuner_extends_smooth_adaptation(self):
        assert issubclass(PruningTuner, SmoothAdaptationTuner)

    def test_perceptron_transform_tuner_extends_smooth_adaptation(self):
        assert issubclass(PerceptronTransformTuner, SmoothAdaptationTuner)

    def test_weight_quantization_extends_perceptron_transform(self):
        assert issubclass(NormalizationAwarePerceptronQuantizationTuner, PerceptronTransformTuner)

    def test_activation_shift_tuner_extends_tuner_base_not_smooth(self):
        assert issubclass(ActivationShiftTuner, TunerBase)
        assert not issubclass(ActivationShiftTuner, SmoothAdaptationTuner)

    def test_tuner_base_run_raises(self):
        """TunerBase.run() is abstract."""
        with pytest.raises(NotImplementedError):
            TunerBase.run(None)

    def test_smooth_adaptation_update_and_evaluate_raises(self):
        """SmoothAdaptationTuner._update_and_evaluate() is abstract."""
        with pytest.raises(NotImplementedError):
            SmoothAdaptationTuner._update_and_evaluate(None, 0.5)


class TestAdaptationTargetAdjusterDecay:
    """Proportional decay for misses, growth for hits."""

    def test_growth_when_above_target_after_decay(self):
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj = AdaptationTargetAdjuster(0.90)
        adj.update_target(0.80)  # decay first
        lowered = adj.target_metric
        adj.update_target(0.95)  # now above target -> should grow
        assert adj.target_metric > lowered

    def test_growth_capped_at_original(self):
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj = AdaptationTargetAdjuster(0.90)
        for _ in range(100):
            adj.update_target(1.0)
        assert adj.target_metric <= 0.90

    def test_decay_uses_multiplicative_factor(self):
        """A miss applies pure multiplicative decay, regardless of miss size."""
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj = AdaptationTargetAdjuster(0.90, floor_ratio=0.1)
        adj.update_target(0.50)
        expected = max(0.90 * adj.decay, adj.floor)
        assert adj.target_metric == pytest.approx(expected)

    def test_target_never_exceeds_original(self):
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj = AdaptationTargetAdjuster(0.90)
        for _ in range(100):
            adj.update_target(1.0)
        assert adj.target_metric <= 0.90

    def test_target_never_below_floor(self):
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj = AdaptationTargetAdjuster(0.90)
        for _ in range(100):
            adj.update_target(0.0)
        assert adj.target_metric >= adj.floor

    def test_large_miss_same_decay_as_small_miss(self):
        """Decay rate is independent of miss magnitude — only the decay
        factor matters, preventing aggressive target ratcheting."""
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj_small = AdaptationTargetAdjuster(0.90, floor_ratio=0.1)
        adj_large = AdaptationTargetAdjuster(0.90, floor_ratio=0.1)
        adj_small.update_target(0.85)  # small miss
        adj_large.update_target(0.10)  # large miss
        assert adj_small.target_metric == pytest.approx(adj_large.target_metric)

    def test_floor_limits_decay(self):
        """Floor ratio limits how low the target can go via decay."""
        from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
        adj = AdaptationTargetAdjuster(0.90, floor_ratio=0.90)
        adj.update_target(0.10)
        assert adj.target_metric >= adj.floor
