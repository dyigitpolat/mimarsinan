"""Unit tests for ClampAdaptationStep.

ClampAdaptationStep always uses ClampTuner — the old "fast path" (no-training
when all activations were already ReLU-compatible) has been removed because it
caused a ~28% accuracy drop in TTFS mode: applying hard clamping without
recovery training degrades a model that was never trained with clamped
activations.

Key contracts verified here:
  - ClampTuner is always created (not None after process()).
  - validate() returns a fresh metric measured after clamping (not the old
    pipeline target).
  - clamp_rate is set to 1.0 after tuning completes.
  - cleanup() does not raise even when called multiple times.
"""

import pytest
import torch
import torch.nn as nn

from conftest import (
    MockDataProviderFactory,
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)

from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.pipelining.pipeline_steps.clamp_adaptation_step import ClampAdaptationStep
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_clamp_step(mock_pipeline, *, target_metric=0.5):
    """Seed a MockPipeline so ClampAdaptationStep can run.  Returns (model, am)."""
    model = make_tiny_supermodel()
    am = AdaptationManager()
    scales = [1.0] * len(model.get_perceptrons())
    scale_stats = make_activation_scale_stats(model, scales, num_batches=2)

    mock_pipeline.config["activation_quantization"] = True
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    mock_pipeline._target_metric = target_metric

    mock_pipeline.seed("model", model, step_name="Activation Adaptation")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Adaptation")
    mock_pipeline.seed("activation_scales", scales, step_name="Activation Analysis")
    mock_pipeline.seed("activation_scale_stats", scale_stats, step_name="Activation Analysis")

    return model, am


def _run_clamp_step(mock_pipeline):
    step = ClampAdaptationStep(mock_pipeline)
    step.name = "Clamp Adaptation"
    mock_pipeline.prepare_step(step)
    step.run()
    return step


def _make_clamp_tuner(mock_pipeline, *, target_metric=0.5):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    scales = [1.0] * len(model.get_perceptrons())
    scale_stats = make_activation_scale_stats(model, scales, num_batches=2)
    mock_pipeline._target_metric = target_metric
    return ClampTuner(
        mock_pipeline,
        model=model,
        target_accuracy=target_metric,
        lr=mock_pipeline.config["lr"],
        adaptation_manager=am,
        activation_scales=scales,
        activation_scale_stats=scale_stats,
    )


# ---------------------------------------------------------------------------
# ClampTuner is always created
# ---------------------------------------------------------------------------

class TestClampAdaptationAlwaysUsesTuner:
    """ClampAdaptationStep must always create a ClampTuner regardless of the
    current activation types on the model."""

    def test_tuner_created_for_relu_compatible_model(self, mock_pipeline):
        """Even when all activations are ReLU-compatible (common after
        ActivationAdaptationStep), ClampTuner must be created."""
        _seed_clamp_step(mock_pipeline)
        step = _run_clamp_step(mock_pipeline)

        assert step.tuner is not None, (
            "ClampAdaptationStep must always create a ClampTuner. "
            "The no-training fast-path has been removed."
        )
        assert isinstance(step.tuner, ClampTuner)

    def test_tuner_created_for_non_relu_model(self, mock_pipeline):
        """When the model has non-ReLU activations, ClampTuner must also be used."""
        from mimarsinan.models.perceptron_mixer.perceptron import make_activation
        from mimarsinan.models.layers import TransformedActivation

        model = make_tiny_supermodel()
        am = AdaptationManager()
        scales = [1.0] * len(model.get_perceptrons())
        scale_stats = make_activation_scale_stats(model, scales, num_batches=2)

        p = model.get_perceptrons()[0]
        p.base_activation = make_activation("LeakyReLU")
        p.base_activation_name = "LeakyReLU"
        p.set_activation(TransformedActivation(p.base_activation, []))

        mock_pipeline.config["activation_quantization"] = True
        mock_pipeline.config["tuning_budget_scale"] = 1.0
        mock_pipeline.seed("model", model, step_name="Activation Adaptation")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Adaptation")
        mock_pipeline.seed("activation_scales", scales, step_name="Activation Analysis")
        mock_pipeline.seed("activation_scale_stats", scale_stats, step_name="Activation Analysis")

        step = _run_clamp_step(mock_pipeline)

        assert step.tuner is not None
        assert isinstance(step.tuner, ClampTuner)


# ---------------------------------------------------------------------------
# validate() contract
# ---------------------------------------------------------------------------

class TestClampAdaptationValidate:
    """validate() must return a fresh metric, not the old pipeline target."""

    def test_validate_returns_fresh_metric_not_old_target(self, mock_pipeline):
        """validate() must NOT return the pre-clamp pipeline target.

        Set an impossibly high target (0.99) and verify the step reports
        something honest for a random untrained model (~0.25 for 4 classes).
        """
        _seed_clamp_step(mock_pipeline, target_metric=0.99)
        step = _run_clamp_step(mock_pipeline)

        result = step.validate()
        assert result != pytest.approx(0.99), (
            "validate() returned the stale pipeline target (0.99) without "
            "measuring the clamped model."
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_validate_returns_float_in_range(self, mock_pipeline):
        _seed_clamp_step(mock_pipeline)
        step = _run_clamp_step(mock_pipeline)

        result = step.validate()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Clamp rate and scale contract
# ---------------------------------------------------------------------------

class TestClampAdaptationClampRate:
    """After tuning completes, clamp_rate must be 1.0."""

    def test_clamp_rate_is_one_after_tuning(self, mock_pipeline):
        model, am = _seed_clamp_step(mock_pipeline)
        _run_clamp_step(mock_pipeline)
        assert am.clamp_rate == pytest.approx(1.0), (
            "ClampAdaptationStep must set clamp_rate=1.0 after tuning."
        )


class TestClampAdaptationScaleAlignment:
    def test_scale_count_mismatch_raises(self, mock_pipeline):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        scales = [1.0] * len(model.get_perceptrons())
        bad_scales = scales[:-1]
        scale_stats = make_activation_scale_stats(model, scales, num_batches=2)

        mock_pipeline.config["activation_quantization"] = True
        mock_pipeline.seed("model", model, step_name="Activation Adaptation")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Adaptation")
        mock_pipeline.seed("activation_scales", bad_scales, step_name="Activation Analysis")
        mock_pipeline.seed("activation_scale_stats", scale_stats, step_name="Activation Analysis")

        step = ClampAdaptationStep(mock_pipeline)
        step.name = "Clamp Adaptation"
        mock_pipeline.prepare_step(step)
        with pytest.raises(ValueError):
            step.run()


class TestClampAdaptationInstantProbe:
    def test_update_and_evaluate_uses_multi_batch_eval_not_train_step(self, mock_pipeline):
        tuner = _make_clamp_tuner(mock_pipeline)
        observed = {}

        def boom(*_args, **_kwargs):
            raise AssertionError("Clamp instant evaluation must not call train_one_step(0)")

        def fake_validate():
            raise AssertionError("Clamp instant evaluation must not use single-batch validate()")

        def fake_validate_n_batches(n_batches):
            observed["n_batches"] = n_batches
            return 0.42

        tuner.trainer.train_one_step = boom
        tuner.trainer.validate = fake_validate
        tuner.trainer.validate_n_batches = fake_validate_n_batches

        acc = tuner._update_and_evaluate(0.5)

        assert acc == pytest.approx(0.42)
        assert observed["n_batches"] == tuner._budget.eval_n_batches

    def test_update_and_evaluate_does_not_mutate_batchnorm_stats(self, mock_pipeline):
        tuner = _make_clamp_tuner(mock_pipeline)
        bn = tuner.model.get_perceptrons()[0].normalization
        running_mean_before = bn.running_mean.clone()
        running_var_before = bn.running_var.clone()

        tuner.trainer.validate_n_batches = lambda _n: 0.5

        tuner._update_and_evaluate(0.5)

        assert torch.allclose(bn.running_mean, running_mean_before)
        assert torch.allclose(bn.running_var, running_var_before)

    def test_scale_metadata_mismatch_raises(self, mock_pipeline):
        model = make_tiny_supermodel()
        am = AdaptationManager()
        scales = [1.0 + i for i, _ in enumerate(model.get_perceptrons())]
        permuted = list(reversed(scales))
        scale_stats = make_activation_scale_stats(model, scales, num_batches=2)

        mock_pipeline.config["activation_quantization"] = True
        mock_pipeline.seed("model", model, step_name="Activation Adaptation")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Adaptation")
        mock_pipeline.seed("activation_scales", permuted, step_name="Activation Analysis")
        mock_pipeline.seed("activation_scale_stats", scale_stats, step_name="Activation Analysis")

        step = ClampAdaptationStep(mock_pipeline)
        step.name = "Clamp Adaptation"
        mock_pipeline.prepare_step(step)
        with pytest.raises(ValueError):
            step.run()


@pytest.mark.slow
class TestClampAdaptationConcatRegression:
    def test_concat_heavy_torch_flow_clamp_step_runs_with_scale_metadata(self, tmp_path):
        from mimarsinan.pipelining.pipeline_steps.activation_analysis_step import (
            ActivationAnalysisStep,
        )
        from mimarsinan.torch_mapping.converter import convert_torch_model

        class TinyFireNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 8, kernel_size=3, padding=1),
                    nn.BatchNorm2d(8),
                    nn.ReLU(),
                )
                self.branch1 = nn.Sequential(
                    nn.Conv2d(8, 4, kernel_size=1),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                )
                self.branch2 = nn.Sequential(
                    nn.Conv2d(8, 4, kernel_size=3, padding=1),
                    nn.BatchNorm2d(4),
                    nn.ReLU(),
                )
                self.pool = nn.MaxPool2d(2)
                self.head = nn.Linear(8 * 4 * 4, 4)

            def forward(self, x):
                x = self.stem(x)
                x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                return self.head(x)

        cfg = default_config()
        raw = TinyFireNet().eval()
        with torch.no_grad():
            raw(torch.randn(1, 3, 8, 8))
        flow = convert_torch_model(raw, input_shape=(3, 8, 8), num_classes=4)

        pipeline = MockPipeline(
            config={**cfg, "input_shape": (3, 8, 8), "num_classes": 4, "activation_quantization": True},
            working_directory=str(tmp_path / "pipeline_cache"),
            data_provider_factory=MockDataProviderFactory(
                input_shape=(3, 8, 8), num_classes=4, size=64
            ),
        )
        am = AdaptationManager()
        pipeline.seed("model", flow, step_name="Activation Adaptation")
        pipeline.seed("adaptation_manager", am, step_name="Activation Adaptation")

        analysis = ActivationAnalysisStep(pipeline)
        analysis.name = "Activation Analysis"
        pipeline.prepare_step(analysis)
        analysis.run()
        analysis.cleanup()

        step = ClampAdaptationStep(pipeline)
        step.name = "Clamp Adaptation"
        pipeline.prepare_step(step)
        step.run()
        assert step.tuner is not None


# ---------------------------------------------------------------------------
# cleanup() safety
# ---------------------------------------------------------------------------

class TestClampAdaptationCleanup:
    def test_cleanup_does_not_raise(self, mock_pipeline):
        _seed_clamp_step(mock_pipeline)
        step = _run_clamp_step(mock_pipeline)
        step.cleanup()  # must not raise

    def test_cleanup_before_run_does_not_raise(self, mock_pipeline):
        _seed_clamp_step(mock_pipeline)
        step = ClampAdaptationStep(mock_pipeline)
        step.name = "Clamp Adaptation"
        mock_pipeline.prepare_step(step)
        step.cleanup()  # must not raise even if process() never ran
