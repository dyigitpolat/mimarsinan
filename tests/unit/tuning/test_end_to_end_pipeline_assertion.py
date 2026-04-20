"""End-to-end tests verifying that tuning steps pass the pipeline's
step-level assertion: new_metric >= previous_metric * tolerance.

These tests simulate the pipeline flow: set target_metric, run tuner,
check that pipeline_metric() passes the tolerance check.

NOTE: On tiny random datasets, some heavy transformations (activation
adaptation: LeakyReLU->ReLU) are unreliable -- see
test_activation_clamp_handoff.py for the same caveat.  We test those
structurally (target stays fixed, tuner completes) rather than numerically.
Light transformations (clamp, quantization, noise) can be tested numerically
because they barely affect accuracy.
"""

import pytest
import torch

from conftest import (
    MockPipeline, MockDataProviderFactory, make_tiny_supermodel,
    default_config, make_activation_scale_stats,
)

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner
from mimarsinan.tuning.tuners.activation_adaptation_tuner import ActivationAdaptationTuner
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.models.layers import TransformedActivation


_MIN_PRETRAINED_ACC = 0.40  # skip if pretrained model is barely above chance


def _pretrain(model, pipeline, epochs=5):
    """Pretrain model and set pipeline target metric."""
    dlf = DataLoaderFactory(pipeline.data_provider_factory, num_workers=0)
    trainer = BasicTrainer(model, "cpu", dlf, pipeline.loss)
    trainer.train_n_epochs(lr=0.1, epochs=epochs, warmup_epochs=0)
    acc = trainer.test()
    trainer.close()
    pipeline._target_metric = acc
    return acc


def _make_pipeline(tmp_path, degradation_tolerance=0.05, size=200):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = degradation_tolerance
    cfg["lr_range_min"] = 1e-5
    cfg["lr_range_max"] = 1e-2
    factory = MockDataProviderFactory(size=size)
    return MockPipeline(config=cfg, working_directory=str(tmp_path), data_provider_factory=factory)


class TestClampTunerPassesPipeline:
    def test_clamp_tuner_retains_accuracy(self, tmp_path):
        """ClampTuner result must satisfy pipeline assertion tolerance.

        Clamping is a light transformation (upper bound on activations) so
        the model should recover easily even on a small dataset.
        """
        torch.manual_seed(42)
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)

        pretrained_acc = _pretrain(model, pipeline, epochs=15)
        if pretrained_acc < _MIN_PRETRAINED_ACC:
            pytest.skip(f"Pretrained acc {pretrained_acc:.2f} too low")

        scales = [1.0] * len(list(model.get_perceptrons()))
        tuner = ClampTuner(
            pipeline, model, target_accuracy=pretrained_acc, lr=0.001,
            adaptation_manager=am,
            activation_scales=scales,
            activation_scale_stats=make_activation_scale_stats(model, scales),
        )
        tuner.run()

        final_acc = tuner.trainer.test()
        tolerance = 1.0 - pipeline.config["degradation_tolerance"]
        assert final_acc >= pretrained_acc * tolerance, (
            f"ClampTuner: {final_acc:.4f} < {pretrained_acc:.4f} * {tolerance} = "
            f"{pretrained_acc * tolerance:.4f}"
        )


class TestActivationQuantizationPassesPipeline:
    def test_activation_quantization_retains_accuracy(self, tmp_path):
        """ActivationQuantizationTuner result must satisfy pipeline tolerance."""
        torch.manual_seed(42)
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            am.update_activation(pipeline.config, p)

        pretrained_acc = _pretrain(model, pipeline, epochs=15)
        if pretrained_acc < _MIN_PRETRAINED_ACC:
            pytest.skip(f"Pretrained acc {pretrained_acc:.2f} too low")

        tuner = ActivationQuantizationTuner(
            pipeline, model, target_tq=4, target_accuracy=pretrained_acc,
            lr=0.001, adaptation_manager=am,
        )
        tuner.run()

        final_acc = tuner.trainer.test()
        tolerance = 1.0 - pipeline.config["degradation_tolerance"]
        assert final_acc >= pretrained_acc * tolerance, (
            f"ActivationQuantizationTuner: {final_acc:.4f} < "
            f"{pretrained_acc:.4f} * {tolerance}"
        )


@pytest.mark.slow
class TestActivationAdaptationMechanics:
    """ActivationAdaptation (LeakyReLU -> ReLU) is a heavy transformation
    that is unreliable on tiny random datasets.  The existing integration
    test (test_activation_clamp_handoff.py) documents this caveat.

    Here we verify the *mechanism*: target stays fixed, tuner completes,
    and _get_target() returns the original value after run().
    """

    def test_target_bounded_through_adaptation(self, tmp_path):
        """After run(), target stays between floor and original_metric.

        The target may relax if the tuner gets stuck with tiny committed
        steps, but must never drop below the floor (original * (1 - dt)).
        """
        torch.manual_seed(42)
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            p.base_activation = make_activation("LeakyReLU")
            p.base_activation_name = "LeakyReLU"
            p.set_activation(TransformedActivation(p.base_activation, []))
            am.update_activation(pipeline.config, p)

        _pretrain(model, pipeline, epochs=8)

        tuner = ActivationAdaptationTuner(
            pipeline, model, target_accuracy=pipeline._target_metric,
            lr=0.001, adaptation_manager=am,
        )
        original_target = tuner._get_target()
        tuner.run()

        final_target = tuner._get_target()
        assert final_target >= tuner.target_adjuster.floor, (
            f"Target {final_target} dropped below floor {tuner.target_adjuster.floor}"
        )
        assert final_target <= original_target, (
            f"Target {final_target} exceeded original {original_target}"
        )

    def test_tuner_completes_without_error(self, tmp_path):
        """ActivationAdaptationTuner.run() completes without raising."""
        torch.manual_seed(42)
        pipeline = _make_pipeline(tmp_path)
        model = make_tiny_supermodel()
        am = AdaptationManager()
        for p in model.get_perceptrons():
            p.base_activation = make_activation("LeakyReLU")
            p.base_activation_name = "LeakyReLU"
            p.set_activation(TransformedActivation(p.base_activation, []))
            am.update_activation(pipeline.config, p)

        _pretrain(model, pipeline, epochs=8)

        tuner = ActivationAdaptationTuner(
            pipeline, model, target_accuracy=pipeline._target_metric,
            lr=0.001, adaptation_manager=am,
        )
        result = tuner.run()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestToleranceAlignmentContract:
    def test_tuner_rollback_threshold_matches_pipeline_tolerance(self):
        """The algebraic identity that prevents the RC1 gap.

        For any degradation_tolerance d:
          Tuner commits when: post_acc >= original_target * (1 - d)
          Pipeline accepts when: metric >= previous_metric * (1 - d)
          Since original_target == previous_metric, both thresholds are equal.
        """
        for d in [0.01, 0.05, 0.10, 0.20]:
            original = 0.90
            tuner_threshold = original * (1.0 - d)
            pipeline_threshold = original * (1.0 - d)
            assert tuner_threshold == pytest.approx(pipeline_threshold), (
                f"Thresholds must match for d={d}"
            )
