"""LR-explorer refusal propagates to entry-preserving tuner fallbacks (W2 fix C)."""

from types import SimpleNamespace

import pytest

from conftest import MockPipeline, make_tiny_supermodel

from mimarsinan.tuning.orchestration import tuner_base as tuner_base_module
from mimarsinan.tuning.orchestration.rate_tuner_seam import OneShotRateTunerSeamMixin
from mimarsinan.tuning.orchestration.tuner_base import TunerBase


@pytest.fixture
def refusing_tuner(tmp_path, monkeypatch):
    pipeline = MockPipeline(working_directory=str(tmp_path))
    tuner = TunerBase(pipeline, make_tiny_supermodel(), 0.9, 0.001)
    calls = {"n": 0}

    def refuse(*args, **kwargs):
        calls["n"] += 1
        return None

    monkeypatch.setattr(tuner_base_module, "find_lr_range_for_trainer", refuse)
    yield tuner, calls
    tuner.close()


class TestTunerBaseRefusal:
    def test_find_lr_propagates_refusal(self, refusing_tuner):
        tuner, _ = refusing_tuner
        assert tuner._find_lr() is None

    def test_refusal_is_cached_until_invalidated(self, refusing_tuner):
        tuner, calls = refusing_tuner
        assert tuner._get_cached_lr() is None
        assert tuner._get_cached_lr() is None
        assert calls["n"] == 1, "a refusal must not re-run the search every call"
        tuner._invalidate_lr_cache()
        assert tuner._get_cached_lr() is None
        assert calls["n"] == 2

    def test_capped_cached_lr_is_none_on_refusal(self, refusing_tuner):
        tuner, _ = refusing_tuner
        assert tuner._capped_cached_lr() is None

    def test_capped_cached_lr_caps_at_pipeline_lr(self, tmp_path, monkeypatch):
        pipeline = MockPipeline(working_directory=str(tmp_path))
        tuner = TunerBase(pipeline, make_tiny_supermodel(), 0.9, 0.001)
        try:
            monkeypatch.setattr(
                tuner_base_module, "find_lr_range_for_trainer",
                lambda *a, **k: 10.0,
            )
            assert tuner._capped_cached_lr() == pytest.approx(0.001)
        finally:
            tuner.close()


class _RefusingOneShot(OneShotRateTunerSeamMixin):
    """One-shot seam host whose LR search refuses; training must never run."""

    def __init__(self):
        self.trainer = SimpleNamespace(
            train_steps_until_target=self._must_not_train,
            validate_n_batches=lambda n: 0.87,
            validate=lambda: 0.87,
        )
        self._budget = SimpleNamespace(
            max_training_steps=100,
            progress_eval_batches=2,
            check_interval=5,
            accuracy_se=lambda: 0.005,
        )

    @staticmethod
    def _must_not_train(*args, **kwargs):
        raise AssertionError("refused LR search must not reach training")

    def _find_lr(self):
        return None


class TestOneShotSeamRefusal:
    def test_recover_to_skips_training_and_returns_entry_probe(self):
        seam = _RefusingOneShot()
        assert seam.recover_to(0.95) == pytest.approx(0.87)


class TestSmoothCycleRefusalTrace:
    def test_recover_coalesces_refused_lr_for_the_decision_trace(self):
        """A refused sweep leaves _last_recover_lr = None; the trace records
        read float(ctx.lr), so the cycle must coalesce it to 0.0 (no training)."""
        import time

        from mimarsinan.tuning.orchestration.adaptation_driver import CycleContext
        from mimarsinan.tuning.orchestration.smooth_adaptation_cycle import (
            SmoothAdaptationCycleMixin,
        )

        class _Host:
            def __init__(self):
                self._last_recover_lr = None

            def recover_to(self, target, rate=None):
                return 0.5

            def _get_target(self):
                return 0.9

        ctx = CycleContext(rate=1.0, t_cycle_start=time.time())
        SmoothAdaptationCycleMixin._recover(_Host(), ctx)
        assert ctx.lr == 0.0
        float(ctx.lr)  # the DecisionRecord contract
