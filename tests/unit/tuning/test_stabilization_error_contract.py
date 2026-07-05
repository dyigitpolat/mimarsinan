"""Error contract for the stabilization rollback guards: fail loud, never swallow.

The pre/post validation reads arm the non-destructive rollback guard (and the
between-round improvement reads decide extra stabilization rounds). A swallowed
validation error used to silently disable rollback or truncate rounds — shipping
a possibly-regressed model. These reads must now propagate any failure.

The KD-blend ``_safe_eval`` is the one sanctioned degrade: it feeds only the
``finalize_cliff`` diagnostic report and goes through ``best_effort``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
    SmoothAdaptationTuner,
)


class _GuardTuner(SmoothAdaptationTuner):
    """Concrete tuner with training stubbed out; validation is scripted per test."""

    def _update_and_evaluate(self, rate):
        return 0.9

    def _find_lr(self):
        return 0.001

    def _recovery_training_hooks(self, rate):
        hook = MagicMock()
        hook.remove = MagicMock()
        return [hook]

    def wire(self):
        self.trainer.train_steps_until_target = lambda *a, **k: None
        self.trainer.validate = lambda: 0.9
        self.trainer.test = lambda: pytest.fail(
            "trainer.test() must NEVER be called from tuner internals"
        )
        return self

    def script_evals(self, script):
        """validate_n_batches pops the script; an Exception item is raised."""
        items = iter(script)

        def _val(n=None):
            item = next(items)
            if isinstance(item, Exception):
                raise item
            return item

        self.trainer.validate_n_batches = _val
        return self


@pytest.fixture
def tuner(tmp_path):
    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    t = _GuardTuner(pipeline, make_tiny_supermodel(), 0.9, 0.001).wire()
    t._committed_rate = 1.0
    t._validation_baseline = 0.9
    t._pipeline_hard_floor = None
    t._rollback_tolerance = 0.05
    yield t
    t.close()


class TestStabilizeAtFullRatePropagates:
    def test_pre_eval_error_propagates(self, tuner):
        tuner.script_evals([RuntimeError("val boom")])
        with pytest.raises(RuntimeError, match="val boom"):
            tuner._stabilize_at_full_rate()

    def test_post_eval_error_propagates(self, tuner):
        tuner._max_stabilization_rounds = 1
        tuner.script_evals([0.5, RuntimeError("val boom")])
        with pytest.raises(RuntimeError, match="val boom"):
            tuner._stabilize_at_full_rate()

    def test_between_rounds_eval_error_propagates(self, tuner):
        tuner._max_stabilization_rounds = 3
        tuner.script_evals([0.5, RuntimeError("val boom")])
        with pytest.raises(RuntimeError, match="val boom"):
            tuner._stabilize_at_full_rate()

    def test_rollback_still_restores_on_regression(self, tuner):
        pre_snapshot = {k: v.clone() for k, v in tuner.model.state_dict().items()}

        def _corrupting_train(*a, **k):
            for p in tuner.model.parameters():
                p.data.fill_(999.0)

        tuner.trainer.train_steps_until_target = _corrupting_train
        tuner.script_evals([0.85, 0.40])
        tuner._stabilize_at_full_rate()
        for k, pre in pre_snapshot.items():
            assert torch.allclose(tuner.model.state_dict()[k], pre, atol=1e-6), k


class TestStabilizeBoundedCosinePropagates:
    @pytest.fixture
    def bounded(self, tuner):
        tuner._stabilization_bounded = True
        tuner._stabilization_ratio = 0.5
        tuner._gradual_train_steps = 4
        return tuner

    def test_pre_eval_error_propagates(self, bounded):
        bounded.script_evals([RuntimeError("val boom")])
        with pytest.raises(RuntimeError, match="val boom"):
            bounded._stabilize_at_full_rate()

    def test_post_eval_error_propagates(self, bounded):
        bounded.script_evals([0.5, RuntimeError("val boom")])
        with pytest.raises(RuntimeError, match="val boom"):
            bounded._stabilize_at_full_rate()


def _clamp_tuner(tmp_path):
    from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
        create_adaptation_manager_for_model,
    )
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = default_config()
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    stats = make_activation_scale_stats(model, scales)
    return ClampTuner(pipeline, model, 0.5, 0.001, manager, scales, stats)


class TestEndpointRecoveryPropagates:
    """The P1'' endpoint stage's fp32 guard reads must fail loud, never swallow."""

    def _prep(self, t, script):
        from mimarsinan.tuning.orchestration import dhat_highwater, endpoint_recovery

        dhat_highwater.observe(t.pipeline, 0.99)
        t._phase_seconds = {}
        t._fast_optimizer_steps = 0
        t._rollback_tolerance = 0.02
        items = iter(script)

        def _read(tuner):
            item = next(items)
            if isinstance(item, Exception):
                raise item
            return item

        return endpoint_recovery, _read

    def test_entry_read_error_propagates(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.endpoint_recovery import (
            run_endpoint_recovery,
        )

        t = _clamp_tuner(tmp_path)
        try:
            module, read = self._prep(t, [RuntimeError("val boom")])
            monkeypatch.setattr(module, "_fp32_deployed_read", read)
            with pytest.raises(RuntimeError, match="val boom"):
                run_endpoint_recovery(t, base_steps=1)
        finally:
            t.close()

    def test_exit_read_error_propagates(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.endpoint_recovery import (
            run_endpoint_recovery,
        )
        from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine

        torch.manual_seed(0)
        t = _clamp_tuner(tmp_path)
        try:
            module, read = self._prep(t, [0.5, RuntimeError("val boom")])
            monkeypatch.setattr(module, "_fp32_deployed_read", read)
            monkeypatch.setattr(
                RecoveryEngine, "train_to_target",
                staticmethod(lambda *a, **k: (0.5, 1)),
            )
            with pytest.raises(RuntimeError, match="val boom"):
                run_endpoint_recovery(t, base_steps=1)
        finally:
            t.close()

    def test_rollback_still_restores_on_regression(self, tmp_path, monkeypatch):
        from mimarsinan.tuning.orchestration.endpoint_recovery import (
            run_endpoint_recovery,
        )
        from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine

        torch.manual_seed(0)
        t = _clamp_tuner(tmp_path)
        try:
            module, read = self._prep(t, [0.85, 0.40])
            monkeypatch.setattr(module, "_fp32_deployed_read", read)
            pre_snapshot = {k: v.clone() for k, v in t.model.state_dict().items()}

            def _corrupting_train(*a, **k):
                with torch.no_grad():
                    for p in t.model.parameters():
                        p.fill_(999.0)
                return 0.4, 1

            monkeypatch.setattr(
                RecoveryEngine, "train_to_target", staticmethod(_corrupting_train),
            )
            report = run_endpoint_recovery(t, base_steps=1)
            assert report.rolled_back is True
            for k, pre in pre_snapshot.items():
                assert torch.allclose(t.model.state_dict()[k], pre, atol=1e-6), k
        finally:
            t.close()


class TestKDBlendSafeEvalIsBestEffort:
    """``_safe_eval`` feeds only the finalize-cliff diagnostic: degrade to None."""

    def _stub(self, validate):
        return SimpleNamespace(
            trainer=SimpleNamespace(validate_n_batches=validate),
            _budget=SimpleNamespace(eval_n_batches=2),
        )

    def test_returns_metric_on_success(self):
        stub = self._stub(lambda n: 0.75)
        assert KDBlendAdaptationTuner._safe_eval(stub) == pytest.approx(0.75)

    def test_returns_none_on_failure(self):
        def _boom(n):
            raise RuntimeError("val boom")

        assert KDBlendAdaptationTuner._safe_eval(self._stub(_boom)) is None

    def test_keyboard_interrupt_propagates(self):
        def _interrupt(n):
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            KDBlendAdaptationTuner._safe_eval(self._stub(_interrupt))
