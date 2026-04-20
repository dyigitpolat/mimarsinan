"""Phase A5 regression: tuners must flush enforcement hooks before returning.

Several tuners (PruningTuner, weight quantization tuners, ClampTuner) rely
on forward pre-hooks that enforce invariants (mask application, clamping,
scale rewrite) at the start of every forward pass. Downstream steps that
inspect layer parameters directly -- e.g. normalization fusion reading
``perceptron.layer.weight`` -- would otherwise see the hooks' view of the
parameters as applied the LAST time a forward pass ran, and in some cases
(when training batch-norm or dropout behave differently from eval) that
last forward pass was a training step, not the eval the pipeline uses.

The flush runs one eval-mode forward on a small batch right before
``_after_run`` returns, so all pre-hook-driven buffer values are consistent
with what the next step's ``trainer.test()`` / parameter inspection will
observe.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from conftest import MockPipeline, make_tiny_supermodel
from mimarsinan.tuning.adaptation_manager import AdaptationManager


def test_flush_enforcement_hooks_runs_eval_forward(monkeypatch):
    """``SmoothAdaptationTuner._flush_enforcement_hooks`` must run exactly one
    eval-mode forward with shape matching the pipeline's ``input_shape``.
    """
    from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

    mock = MockPipeline()
    ce = nn.CrossEntropyLoss()
    mock.loss = lambda model, x, y: ce(model(x), y)
    model = make_tiny_supermodel()

    tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
    tuner.pipeline = mock
    tuner.model = model

    calls = {"n_fwd": 0, "mode_train": None, "shape": None}

    real_forward = model.forward

    def _instrumented_forward(x, *a, **kw):
        calls["n_fwd"] += 1
        calls["mode_train"] = model.training
        calls["shape"] = tuple(x.shape)
        return real_forward(x, *a, **kw)

    model.forward = _instrumented_forward

    tuner._flush_enforcement_hooks()

    assert calls["n_fwd"] == 1, "flush must run exactly one forward pass"
    assert calls["mode_train"] is False, (
        "flush forward must be in eval mode to avoid BN stats drift"
    )
    input_shape = mock.config["input_shape"]
    assert calls["shape"][1:] == tuple(input_shape), (
        f"flush forward shape {calls['shape']} must match input_shape={input_shape}"
    )


def test_flush_swallows_forward_errors(tmp_path):
    """If the tuner's model can't be forward-called in eval mode, the flush
    must silently skip rather than failing the whole step.
    """
    from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner

    mock = MockPipeline(working_directory=str(tmp_path))
    tuner = SmoothAdaptationTuner.__new__(SmoothAdaptationTuner)
    tuner.pipeline = mock

    class _Boom(nn.Module):
        def forward(self, *a, **kw):
            raise RuntimeError("intentional")

    tuner.model = _Boom()
    # Must not raise.
    tuner._flush_enforcement_hooks()


def test_clamp_tuner_after_run_flushes(tmp_path):
    """``ClampTuner._after_run`` runs the flush so any decorator-owned buffer
    (e.g. a learnable clamp ceiling in Phase B2) is updated with eval-mode
    inputs before the step metric is measured."""
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner
    from conftest import make_activation_scale_stats

    mock = MockPipeline(working_directory=str(tmp_path))
    ce = nn.CrossEntropyLoss()
    mock.loss = lambda model, x, y: ce(model(x), y)
    model = make_tiny_supermodel()
    am = AdaptationManager()
    scales = [1.0] * len(list(model.get_perceptrons()))

    tuner = ClampTuner(
        pipeline=mock,
        model=model,
        target_accuracy=0.0,
        lr=1e-3,
        adaptation_manager=am,
        activation_scales=scales,
        activation_scale_stats=make_activation_scale_stats(model, scales),
    )

    # Track invocations of _flush_enforcement_hooks.
    flush_calls = {"n": 0}
    real_flush = tuner._flush_enforcement_hooks

    def _track_flush():
        flush_calls["n"] += 1
        return real_flush()

    tuner._flush_enforcement_hooks = _track_flush

    # Fake out the recovery probe so _after_run completes quickly.
    tuner.trainer.validate_n_batches = lambda n: 1.0
    tuner.trainer.train_steps_until_target = lambda *a, **kw: None
    tuner.trainer.test = lambda: (_ for _ in ()).throw(
        AssertionError("trainer.test() must not be called from tuner code")
    )
    tuner.target_adjuster.original_metric = 0.5
    tuner._pipeline_tolerance = 0.05
    tuner._committed_rate = 1.0

    from mimarsinan.tuning.tuning_budget import tuning_budget_from_pipeline
    tuner._budget = tuning_budget_from_pipeline(mock)

    tuner._after_run()

    assert flush_calls["n"] >= 1, "_after_run must invoke _flush_enforcement_hooks"
