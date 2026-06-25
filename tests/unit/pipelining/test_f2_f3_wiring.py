"""F2/F3 publication-treatment wiring locks.

The F-harness (``scripts/campaign/experiment_matrix.py``) generates batches that
set two deployment-config axes. These tests LOCK that ``run.py``/the pipeline
actually HONORS those keys (an ignored key silently produces garbage F2/F3 rows
where the treatment arm equals the control arm):

  F2  ``deployment_parameters.activation_scale_quantile`` — the per-perceptron
      ANN->SNN decode/clamp scale quantile. Control 0.99 vs percentile-norm 1.0.
      Wired through ``ActivationAnalysisStep`` (reads ``pipeline.config``).
  F3  ``deployment_parameters.preload_weights`` — from_scratch (False/unset) vs
      pretrained (True). Wired by ``DeploymentPlan.resolve`` deriving
      ``weight_source='torchvision'`` so the existing ``WeightPreloadingStep``
      applies; DEFAULT-OFF: unset => ``weight_source`` unchanged (from_scratch).
"""

import pytest
import torch

from conftest import (
    MockPipeline,
    MockDataProviderFactory,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import (
    ActivationAnalysisStep,
    DEFAULT_SCALE_QUANTILE,
    scale_from_activations,
)
from mimarsinan.pipelining.pipeline_steps.config.weight_preloading_step import (
    WeightPreloadingStep,
)
from mimarsinan.pipelining.pipeline_steps.training.pretraining_step import (
    PretrainingStep,
)

# The exact deployment-config keys the F-harness batches set.
from scripts.campaign.experiment_matrix import (
    QUANTILE_KEY,
    PRELOAD_KEY,
    DEFAULT_ACTIVATION_SCALE_QUANTILE,
    PERCENTILE_NORM_QUANTILE,
)


# ---------------------------------------------------------------------------
# Harness contract: the keys the batches actually emit are the keys we wire.
# ---------------------------------------------------------------------------
class TestHarnessConfigKeys:
    def test_f2_key_is_activation_scale_quantile(self):
        assert QUANTILE_KEY == "deployment_parameters.activation_scale_quantile"

    def test_f3_key_is_preload_weights(self):
        assert PRELOAD_KEY == "deployment_parameters.preload_weights"

    def test_f2_arms_are_default_and_percentile_norm(self):
        assert DEFAULT_ACTIVATION_SCALE_QUANTILE == 0.99
        assert PERCENTILE_NORM_QUANTILE == 1.0


# ---------------------------------------------------------------------------
# F2: the activation-scale quantile treatment actually changes the scale.
# ---------------------------------------------------------------------------
class TestF2QuantileApplies:
    def test_percentile_norm_quantile_differs_from_default(self):
        """q=1.0 (percentile-norm) >= q=0.99 (default) and differs on a skewed dist."""
        # 5 rare outliers in 1000: the 0.99 count-quantile (index ~990) lands in
        # the in-range bulk and CLIPS the outliers, while q=1.0 keeps the max.
        torch.manual_seed(0)
        flat = torch.cat([torch.rand(995), torch.full((5,), 50.0)])
        default_scale = scale_from_activations(
            flat, quantile=DEFAULT_ACTIVATION_SCALE_QUANTILE
        )
        percentile_scale = scale_from_activations(
            flat, quantile=PERCENTILE_NORM_QUANTILE
        )
        assert default_scale < 1.0  # the outliers are clipped at q=0.99
        assert percentile_scale > default_scale
        # q=1.0 == max over active activations (no top-percentile clip).
        assert percentile_scale == pytest.approx(50.0)

    def test_step_reads_activation_scale_quantile_from_config(self, tmp_path):
        """ActivationAnalysisStep honors deployment_parameters.activation_scale_quantile."""
        scales_by_q = {}
        for q in (DEFAULT_ACTIVATION_SCALE_QUANTILE, PERCENTILE_NORM_QUANTILE):
            cfg = default_config()
            cfg["activation_scale_quantile"] = q
            pipeline = MockPipeline(
                config=cfg,
                working_directory=str(tmp_path / f"cache_q{q}"),
                data_provider_factory=MockDataProviderFactory(size=64),
            )
            pipeline.seed("model", make_tiny_supermodel())
            step = ActivationAnalysisStep(pipeline)
            step.name = "ActivationAnalysis"
            pipeline.prepare_step(step)
            step.run()
            stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]
            assert stats["quantile"] == pytest.approx(q)
            scales_by_q[q] = stats["summary"]["max_scale"]

        # q=1.0 never clips below q=0.99: the percentile-norm arm's scales are
        # >= the default arm's (the treatment is not a no-op on the step).
        assert (
            scales_by_q[PERCENTILE_NORM_QUANTILE]
            >= scales_by_q[DEFAULT_ACTIVATION_SCALE_QUANTILE]
        )

    def test_default_unset_quantile_is_byte_identical(self, tmp_path):
        """Key unset => the step uses the historical default 0.99 (control arm)."""
        cfg = default_config()
        assert "activation_scale_quantile" not in cfg
        pipeline = MockPipeline(
            config=cfg,
            working_directory=str(tmp_path / "cache_default"),
            data_provider_factory=MockDataProviderFactory(size=64),
        )
        pipeline.seed("model", make_tiny_supermodel())
        step = ActivationAnalysisStep(pipeline)
        step.name = "ActivationAnalysis"
        pipeline.prepare_step(step)
        step.run()
        stats = pipeline.cache["ActivationAnalysis.activation_scale_stats"]
        assert stats["quantile"] == pytest.approx(DEFAULT_SCALE_QUANTILE)
        assert DEFAULT_SCALE_QUANTILE == DEFAULT_ACTIVATION_SCALE_QUANTILE


# ---------------------------------------------------------------------------
# F3: the preload_weights regime axis selects the preload vs pretrain step.
# ---------------------------------------------------------------------------
class TestF3PreloadWiring:
    def test_preload_true_derives_torchvision_weight_source(self):
        plan = DeploymentPlan.resolve({"preload_weights": True})
        assert plan.weight_source == "torchvision"

    def test_preload_true_activates_weight_preloading_step(self):
        plan = DeploymentPlan.resolve({"preload_weights": True})
        assert WeightPreloadingStep.applies_to(plan) is True
        # Pretrained regime => the from-scratch PretrainingStep is skipped.
        assert PretrainingStep.applies_to(plan) is False

    def test_preload_false_is_from_scratch(self):
        plan = DeploymentPlan.resolve({"preload_weights": False})
        assert plan.weight_source is None
        assert WeightPreloadingStep.applies_to(plan) is False
        assert PretrainingStep.applies_to(plan) is True

    def test_preload_unset_is_byte_identical_from_scratch(self):
        """DEFAULT-OFF: key absent => weight_source unchanged (None) => from_scratch."""
        plan = DeploymentPlan.resolve({})
        assert plan.weight_source is None
        assert WeightPreloadingStep.applies_to(plan) is False
        assert PretrainingStep.applies_to(plan) is True

    def test_explicit_weight_source_wins_over_preload_flag(self):
        """An explicit weight_source is never overridden by the preload flag."""
        plan = DeploymentPlan.resolve(
            {"preload_weights": True, "weight_source": "/tmp/ckpt.pt"}
        )
        assert plan.weight_source == "/tmp/ckpt.pt"

    def test_explicit_weight_source_preserved_when_preload_false(self):
        plan = DeploymentPlan.resolve(
            {"preload_weights": False, "weight_source": "torchvision"}
        )
        assert plan.weight_source == "torchvision"


# ---------------------------------------------------------------------------
# F3 graceful failure: a preload arm on a builder with NO pretrained source must
# raise a CLEAR, EARLY, typed error (the rc=1 root cause), not crash opaquely.
# ---------------------------------------------------------------------------
class TestF3PreloadGracefulFailure:
    def test_step_raises_typed_unsupported_for_native_builder(self, tmp_path):
        """preload_weights=True on a native (no-factory) builder => UnsupportedPreloadError.

        Raised EARLY in process() — before any load/fine-tune — so run.py can map it
        to a clean UNSUPPORTED skip instead of an opaque mid-pipeline rc=1.
        """
        from mimarsinan.model_training.weight_loading import UnsupportedPreloadError

        class NativeBuilder:  # mirrors DeepCNNBuilder: no get_pretrained_factory
            pass

        cfg = default_config()
        cfg["preload_weights"] = True
        cfg["device"] = "cpu"
        pipeline = MockPipeline(
            config=cfg,
            working_directory=str(tmp_path / "cache_preload"),
            data_provider_factory=MockDataProviderFactory(size=8),
        )
        pipeline.seed("model", make_tiny_supermodel())
        pipeline.seed("model_builder", NativeBuilder())

        plan = DeploymentPlan.of(pipeline)
        assert plan.weight_source == "torchvision"  # the ill-posed source
        assert WeightPreloadingStep.applies_to(plan) is True

        step = WeightPreloadingStep(pipeline)
        step.name = "WeightPreloading"
        pipeline.prepare_step(step)
        with pytest.raises(UnsupportedPreloadError, match="get_pretrained_factory"):
            step.process()

    def test_run_headless_maps_unsupported_preload_to_clean_skip(self):
        """run.py's headless loop must catch UnsupportedPreloadError as a CLEAN skip.

        Source-level lock: the typed handler precedes the bare Exception handler and
        records status 'skipped' with exit 0 (so the campaign ledgers a skip, not an
        opaque rc=1). The behavioural raise is locked above; this guards the wiring.
        """
        import ast
        import os

        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        src = open(os.path.join(repo, "run.py")).read()
        assert "UnsupportedPreloadError" in src
        assert '"skipped"' in src
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                names = [
                    h.type.id if isinstance(h.type, ast.Name) else None
                    for h in node.handlers
                ]
                if "UnsupportedPreloadError" in names:
                    assert names.index("UnsupportedPreloadError") < names.index(
                        "Exception"
                    ), "typed UnsupportedPreloadError handler must precede Exception"
                    return
        raise AssertionError("run.py has no UnsupportedPreloadError handler")
