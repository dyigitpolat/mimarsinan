"""Tests for ActivationAdaptationStep and activation_utils.

ActivationAdaptationStep always runs after Activation Analysis. It uses
ActivationAdaptationTuner (SmartSmoothAdaptation) to gradually blend non-ReLU
chip-targeted activations toward ReLU. When all activations are already
ReLU-compatible, it is a no-op.

This step does NOT apply activation_scales -- that is the responsibility of
downstream steps (Clamp Adaptation, etc.).
"""

import pytest

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
)

from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    RELU_COMPATIBLE_TYPES,
    has_non_relu_activations,
)
from mimarsinan.pipelining.pipeline_steps.activation_adaptation_step import (
    ActivationAdaptationStep,
)
from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.models.layers import TransformedActivation


def _model_with_gelu():
    """Model with one GELU perceptron (chip-targeted, can be adapted to ReLU)."""
    model = make_tiny_supermodel()
    perceptrons = model.get_perceptrons()
    assert len(perceptrons) >= 1
    p = perceptrons[0]
    p.base_activation = make_activation("GELU")
    p.base_activation_name = "GELU"
    p.set_activation(TransformedActivation(p.base_activation, []))
    return model


def _model_with_identity_and_leaky_relu():
    """Model containing one host-side Identity perceptron and one adaptable
    LeakyReLU perceptron.

    This reproduces the torch-mapped MLP-Mixer case where ActivationAdaptation
    must adapt the LeakyReLU perceptrons but must NOT rewrite Identity
    perceptrons during the final commit.
    """
    model = make_tiny_supermodel()
    perceptrons = model.get_perceptrons()
    assert len(perceptrons) >= 2

    perceptrons[0].base_activation = make_activation("Identity")
    perceptrons[0].base_activation_name = "Identity"
    perceptrons[0].set_activation(TransformedActivation(perceptrons[0].base_activation, []))

    perceptrons[1].base_activation = make_activation("LeakyReLU")
    perceptrons[1].base_activation_name = "LeakyReLU"
    perceptrons[1].set_activation(TransformedActivation(perceptrons[1].base_activation, []))

    return model, perceptrons[0], perceptrons[1]


class TestActivationUtils:
    """has_non_relu_activations — checks for non-ReLU chip-targeted perceptrons."""

    def test_false_when_all_relu(self):
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.base_activation = make_activation("ReLU")
            p.base_activation_name = "ReLU"
        assert has_non_relu_activations(model) is False

    def test_true_when_any_gelu(self):
        """GELU perceptrons are chip-targeted (adapted to ReLU in ActivationAdaptationStep).
        They are included in get_perceptrons() and trigger activation adaptation."""
        model = _model_with_gelu()
        assert has_non_relu_activations(model) is True

    def test_false_when_all_identity(self):
        """Identity perceptrons are host-side only and must not trigger
        activation adaptation."""
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.base_activation = make_activation("Identity")
            p.base_activation_name = "Identity"
            p.set_activation(TransformedActivation(p.base_activation, []))
        assert has_non_relu_activations(model) is False

    def test_true_when_any_leaky_relu(self):
        model = make_tiny_supermodel()
        perceptrons = model.get_perceptrons()
        perceptrons[0].base_activation = make_activation("LeakyReLU")
        perceptrons[0].base_activation_name = "LeakyReLU"
        perceptrons[0].set_activation(
            TransformedActivation(perceptrons[0].base_activation, [])
        )
        assert has_non_relu_activations(model) is True

    def test_constants_defined(self):
        assert "ReLU" in RELU_COMPATIBLE_TYPES
        assert "LeakyGradReLU" in RELU_COMPATIBLE_TYPES


class TestActivationAdaptationStepRates:
    """ActivationAdaptationStep must not set clamp_rate (stays 0)."""

    def test_clamp_rate_unchanged_after_step(self, mock_pipeline):
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        model = make_tiny_supermodel()
        am = AdaptationManager()
        assert am.clamp_rate == 0.0
        assert am.activation_adaptation_rate == 0.0

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        assert am.clamp_rate == 0.0, (
            "ActivationAdaptationStep must not set clamp_rate; "
            "when act_q is False clamp stays no-op."
        )
        assert am.activation_adaptation_rate == 0.0, (
            "activation_adaptation_rate should be reset to 0 after adaptation."
        )

    def test_scales_not_applied(self, mock_pipeline):
        """ActivationAdaptationStep must NOT apply activation_scales."""
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        model = make_tiny_supermodel()
        am = AdaptationManager()
        n = len(model.get_perceptrons())

        original_scales = [p.activation_scale.item() for p in model.get_perceptrons()]

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        for p, orig in zip(model.get_perceptrons(), original_scales):
            assert p.activation_scale.item() == orig, (
                "ActivationAdaptationStep should not modify activation_scale"
            )


# ---------------------------------------------------------------------------
# Committed-metric contract
# ---------------------------------------------------------------------------

class TestActivationAdaptationCommit:
    """rate=1.0 decorated path and committed ReLU must be forward-equivalent,
    and the step's validate() must return the metric from right after the commit
    without advancing the validation iterator again."""

    def test_rate_one_matches_committed_relu(self):
        """At rate=1.0 the decorator blends entirely to LeakyGradReLU.
        After commit, base_activation is replaced by LeakyGradReLU (make_activation("ReLU")).
        Both forward paths must produce the same output for the same input.
        """
        import torch
        from mimarsinan.tuning.adaptation_manager import AdaptationManager
        from mimarsinan.models.perceptron_mixer.perceptron import make_activation, Perceptron
        from mimarsinan.models.layers import TransformedActivation
        from mimarsinan.models.activations import LeakyGradReLU

        # Build a perceptron with GELU base.
        p = Perceptron(16, 32)
        p.base_activation = make_activation("GELU")
        p.base_activation_name = "GELU"
        p.set_activation(TransformedActivation(p.base_activation, []))

        am = AdaptationManager()
        cfg = default_config()

        # Rate=1.0: fully blend to LeakyGradReLU.
        am.activation_adaptation_rate = 1.0
        am.update_activation(cfg, p)

        x = torch.randn(8, 32)
        p.eval()
        with torch.no_grad():
            out_rate_1 = p(x)

        # Commit: replace base to ReLU (LeakyGradReLU), reset rate to 0.
        p.base_activation = make_activation("ReLU")
        p.base_activation_name = "ReLU"
        am.activation_adaptation_rate = 0.0
        am.update_activation(cfg, p)

        with torch.no_grad():
            out_committed = p(x)

        assert torch.allclose(out_rate_1, out_committed, atol=1e-5), (
            "rate=1.0 decorated output must match committed ReLU output: "
            f"max diff = {(out_rate_1 - out_committed).abs().max().item():.6f}"
        )

    def test_validate_uses_cached_metric(self, mock_pipeline):
        """ActivationAdaptationStep.validate() should return the metric measured
        right after the commit, not advance the validation iterator again.

        Without caching, each validate() call consumes a different minibatch,
        making the reported metric unreliable.  With caching, the pipeline gets
        the same value that was measured immediately after the commit.
        """
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        model = make_tiny_supermodel()
        am = AdaptationManager()

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        # validate() must return a stable value.
        val1 = step.validate()
        val2 = step.validate()
        assert val1 == val2, (
            "ActivationAdaptationStep.validate() must return a stable cached "
            f"metric, not advance the iterator each call. Got {val1} then {val2}."
        )

    def test_validate_does_not_call_tuner_when_metric_is_cached(self, mock_pipeline):
        """validate() must not call tuner.validate() when _committed_metric exists.

        Python evaluates function arguments eagerly, so
        getattr(obj, "_committed_metric", obj.validate()) still calls
        obj.validate() even when _committed_metric is present. That would
        advance the validation iterator and emit a noisy metric even though the
        cached metric is returned.
        """

        class _ExplodingTuner:
            _committed_metric = 0.75

            def validate(self):
                raise AssertionError(
                    "tuner.validate() must not be called when _committed_metric exists"
                )

        step = ActivationAdaptationStep(mock_pipeline)
        step.tuner = _ExplodingTuner()

        assert step.validate() == 0.75

    def test_committed_metric_uses_full_test_set(self, mock_pipeline):
        """_committed_metric must use trainer.test() (full test set), not
        trainer.validate() (single minibatch).

        trainer.validate() reads one batch from val_iter; the batch that
        happens to be current when the commit runs can differ significantly
        from the model's actual accuracy (observed 0.63 vs 0.96 on MNIST).
        That noisy value then becomes the recovery TARGET for ClampTuner,
        causing it to aim for 0.63 instead of 0.96.

        Using trainer.test() iterates the complete test split, giving a
        stable and representative metric for both pipeline tolerance checks
        and downstream tuner targets.
        """
        import torch
        from mimarsinan.tuning.adaptation_manager import AdaptationManager
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
        from mimarsinan.model_training.basic_trainer import BasicTrainer

        # Use a non-ReLU model so the tuner actually runs and sets _committed_metric.
        model = _model_with_gelu()
        am = AdaptationManager()

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        assert step.tuner is not None, "Tuner must be created for non-ReLU model"
        committed = step.tuner._committed_metric

        # Measure independently with a fresh trainer on the test set.
        dlf = DataLoaderFactory(mock_pipeline.data_provider_factory, num_workers=0)
        fresh_trainer = BasicTrainer(model, "cpu", dlf, mock_pipeline.loss)
        fresh_test_acc = fresh_trainer.test()
        fresh_trainer.close()

        # _committed_metric must match the full-set test accuracy, not a
        # noisy single-batch validate().  TinyDataProvider uses batch_size=size
        # for test, so both are deterministic here; the key contract is that
        # _committed_metric comes from test(), not validate().
        assert committed == fresh_test_acc, (
            f"_committed_metric ({committed:.4f}) must equal trainer.test() "
            f"({fresh_test_acc:.4f}) measured on the same model weights.  "
            "A mismatch means _committed_metric was taken from validate() "
            "(single minibatch) rather than the full test set."
        )

    # test_run_does_not_commit_identity_perceptrons removed:
    # Identity perceptrons no longer exist — layers without activation go through
    # ModuleComputeMapper and never enter the adaptation pipeline.
