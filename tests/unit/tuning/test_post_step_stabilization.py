"""Tests for the post-step stabilization phase at rate=1.0.

After ``_after_run`` forces the model to the fully-adapted state
(``rate == 1.0``) and the validation-only safety net has run, a final
stabilization phase trains the committed model at rate=1.0 for
``2 * _budget.max_training_steps`` additional gradient steps. This gives
the optimizer extra runway to consolidate the final adaptation without
exploring the rate schedule, which is the regime where we historically
saw slow but steady accuracy recovery.

Contract:

* Runs exactly once per ``run()``, after ``_after_run()``.
* Uses ``_recovery_training_hooks(1.0)`` so pruning / decorator invariants
  stay enforced throughout.
* Uses the cached LR (no new LR search).
* Preserves best validation via ``train_steps_until_target``'s built-in
  best-state tracking, so it can never make the model worse.
* Never calls ``trainer.test()`` (single-measurement rule).
* Can be disabled by a subclass returning ``None`` from
  ``_stabilization_budget()`` (extensibility escape hatch).
"""

import inspect
from unittest.mock import MagicMock

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


class _StabilizationTuner(SmoothAdaptationTuner):
    """Concrete tuner that records every call to ``train_steps_until_target``."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.train_calls = []
        self.hook_states = []

    def _update_and_evaluate(self, rate):
        return 0.9

    def _find_lr(self):
        return 0.001

    def _recovery_training_hooks(self, rate):
        self.hook_states.append(("install", rate))
        fake_hook = MagicMock()
        fake_hook.remove = MagicMock(
            side_effect=lambda: self.hook_states.append(("remove", rate))
        )
        return [fake_hook]

    def _wire(self):
        tuner = self

        def _fake_train(lr, max_steps, target, *args, **kwargs):
            tuner.train_calls.append(
                {
                    "lr": lr,
                    "max_steps": int(max_steps),
                    "target": target,
                    "kwargs": kwargs,
                }
            )

        self.trainer.train_steps_until_target = _fake_train
        self.trainer.validate = lambda: 0.9
        self.trainer.validate_n_batches = lambda n: 0.9
        self.trainer.test = lambda: pytest.fail(
            "trainer.test() must NEVER be called from tuner internals"
        )


@pytest.fixture
def tuner(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    t = _StabilizationTuner(pipeline, model, target_accuracy=0.9, lr=0.001)
    t._wire()
    return t


class TestStabilizationBudget:
    """The default stabilization budget is exactly 2 * max_training_steps."""

    def test_default_is_two_times_max_training_steps(self, tuner):
        expected = 2 * tuner._budget.max_training_steps
        assert tuner._stabilization_budget() == expected, (
            f"Default stabilization budget must be 2 * max_training_steps; "
            f"got {tuner._stabilization_budget()}, expected {expected}"
        )

    def test_returns_none_disables_stabilization(self, tuner):
        tuner._stabilization_budget = lambda: None
        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None

        tuner._stabilize_at_full_rate()

        assert tuner.train_calls == [], (
            "_stabilization_budget()==None must disable stabilization entirely"
        )

    def test_zero_budget_disables_stabilization(self, tuner):
        tuner._stabilization_budget = lambda: 0
        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None

        tuner._stabilize_at_full_rate()

        assert tuner.train_calls == [], (
            "_stabilization_budget()==0 must disable stabilization entirely"
        )


class TestStabilizationCall:
    """When invoked, ``_stabilize_at_full_rate`` runs exactly one training
    call with the 2x budget at the cached LR, bracketed by install/remove
    of the rate=1.0 recovery hooks."""

    def test_runs_exactly_once_with_correct_budget(self, tuner):
        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None
        tuner._cached_lr = 0.0005

        tuner._stabilize_at_full_rate()

        assert len(tuner.train_calls) == 1
        call = tuner.train_calls[0]
        assert call["max_steps"] == 2 * tuner._budget.max_training_steps
        assert call["lr"] == 0.0005

    def test_installs_and_removes_rate_1_hooks(self, tuner):
        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None

        tuner._stabilize_at_full_rate()

        assert tuner.hook_states == [("install", 1.0), ("remove", 1.0)], (
            "Stabilization must install rate=1.0 recovery hooks, run "
            "training, and then remove them (even on exception)."
        )

    def test_removes_hooks_even_on_exception(self, tuner):
        def _raise(*a, **kw):
            raise RuntimeError("boom")

        tuner.trainer.train_steps_until_target = _raise
        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None

        with pytest.raises(RuntimeError, match="boom"):
            tuner._stabilize_at_full_rate()

        assert tuner.hook_states[0] == ("install", 1.0)
        assert ("remove", 1.0) in tuner.hook_states, (
            "Hooks must be removed even when training raises"
        )

    def test_never_calls_trainer_test(self, tuner):
        """Stabilization uses validation only — test-set isolation."""
        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None

        tuner._stabilize_at_full_rate()

    def test_restores_best_state_on_regression(self, tuner):
        """If the stabilization phase degrades validation, the pre-stabilization
        state is restored so stabilization can never make the model worse."""
        # Snapshot pre-state.
        pre_state = tuner._clone_state()
        pre_snapshot = {k: v.clone() for k, v in tuner.model.state_dict().items()}

        # Make ``train_steps_until_target`` corrupt the model.
        def _fake_train(lr, max_steps, target, *args, **kwargs):
            for p in tuner.model.parameters():
                p.data.fill_(999.0)

        tuner.trainer.train_steps_until_target = _fake_train
        # Validation dips after training. Stabilization now uses
        # ``validate_n_batches`` (multi-batch) rather than the single-batch
        # ``validate()`` whose 3σ ≈ 3.8% on MNIST would otherwise trip the
        # rollback gate on pure noise.
        validation_sequence = [0.85, 0.40]
        tuner.trainer.validate_n_batches = lambda n: (
            validation_sequence.pop(0) if validation_sequence else 0.40
        )

        tuner._committed_rate = 1.0
        tuner._validation_baseline = 0.85
        tuner._pipeline_hard_floor = None
        tuner._rollback_tolerance = 0.02

        tuner._stabilize_at_full_rate()

        for k, pre in pre_snapshot.items():
            assert torch.allclose(
                tuner.model.state_dict()[k], pre, atol=1e-6
            ), f"Stabilization-induced regression must be rolled back (param {k})"


class TestStabilizationIntegrationWithRun:
    """``run()`` must invoke ``_stabilize_at_full_rate`` after
    ``_after_run`` — once, on both the one-shot and gradual exit paths,
    and *after* ``_after_run`` so stabilization sees the final committed
    rate=1.0 model, not the partial-rate model."""

    def test_run_source_calls_stabilize_after_after_run(self):
        # The shared finalize tail (driver path) owns the after_run → stabilize order.
        source = inspect.getsource(SmoothAdaptationTuner._finalize_run)
        assert "_stabilize_at_full_rate" in source, (
            "_finalize_run must invoke _stabilize_at_full_rate"
        )
        after_run_idx = source.index("_after_run")
        stabilize_idx = source.index("_stabilize_at_full_rate")
        assert stabilize_idx > after_run_idx, (
            "_stabilize_at_full_rate must be called *after* _after_run"
        )

    def test_run_calls_stabilize_exactly_once(self, tuner):
        """Integration: run() exits via the one-shot path (success at
        rate=1.0) and must then invoke _stabilize_at_full_rate exactly
        once, so the call count for train_steps_until_target matches
        the expected 2x budget call."""
        stabilize_calls = []
        orig = SmoothAdaptationTuner._stabilize_at_full_rate

        def _counting(self_):
            stabilize_calls.append(1)
            return orig(self_)

        tuner.__class__._stabilize_at_full_rate = _counting
        try:
            tuner.run()
        finally:
            tuner.__class__._stabilize_at_full_rate = orig

        assert len(stabilize_calls) == 1, (
            f"_stabilize_at_full_rate must be called exactly once per run; "
            f"called {len(stabilize_calls)} times"
        )


class TestStabilizationPreconditions:
    """Stabilization must only run when _committed_rate == 1.0 to prevent
    it from ever training a partial-rate model."""

    def test_skips_when_rate_below_one(self, tuner):
        tuner._committed_rate = 0.5
        tuner._validation_baseline = 0.9
        tuner._pipeline_hard_floor = None

        tuner._stabilize_at_full_rate()

        assert tuner.train_calls == [], (
            "Stabilization must be a no-op when _committed_rate < 1.0"
        )


class _RoundsTuner(_StabilizationTuner):
    """Tuner with multi-round stabilization and scripted eval sequence."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.find_lr_calls = 0
        self.eval_sequence = []

    def _find_lr(self):
        # Refreshed LRs below pipeline_lr so the min() clamp keeps them visible.
        self.find_lr_calls += 1
        return 0.001 / (1 + self.find_lr_calls)

    def _wire_evals(self, values):
        seq = iter(values)
        last = values[-1]

        def _val(n=None):
            nonlocal last
            try:
                last = next(seq)
            except StopIteration:
                pass
            return last

        self.trainer.validate_n_batches = _val


class TestStabilizationRounds:
    """KD blend tuners run extra stabilization rounds with LR restarts while
    each round still improves validation by more than accuracy_se()/2; the
    default (rounds=1) keeps the historical single-pass contract."""

    def _make(self, tmp_path, rounds):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        t = _RoundsTuner(pipeline, make_tiny_supermodel(), 0.99, 0.001)
        t._wire()
        t._max_stabilization_rounds = rounds
        t._committed_rate = 1.0
        t._validation_baseline = 0.5
        t._pipeline_hard_floor = None
        t._rollback_tolerance = 0.05
        return t

    def test_improving_rounds_continue_with_fresh_lr(self, tmp_path):
        t = self._make(tmp_path, rounds=3)
        # pre-val 0.50; rounds end at 0.60, 0.70, 0.80 — always improving.
        t._wire_evals([0.50, 0.60, 0.70, 0.80])
        t._stabilize_at_full_rate()
        assert len(t.train_calls) == 3
        # LR re-found between rounds (cached LR invalidated).
        assert t.find_lr_calls >= 2
        lrs = [c["lr"] for c in t.train_calls]
        assert len(set(lrs)) >= 2, f"rounds must use refreshed LRs, got {lrs}"

    def test_plateau_stops_rounds(self, tmp_path):
        t = self._make(tmp_path, rounds=3)
        # pre-val 0.50; round 1 ends at 0.5 (no gain) -> stop after round 1.
        t._wire_evals([0.50, 0.50])
        t._stabilize_at_full_rate()
        assert len(t.train_calls) == 1

    def test_default_rounds_is_single_pass(self, tmp_path):
        t = self._make(tmp_path, rounds=1)
        t._wire_evals([0.50, 0.90])
        t._stabilize_at_full_rate()
        assert len(t.train_calls) == 1

    def test_kd_blend_tuner_requests_multiple_rounds(self):
        from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
            KDBlendAdaptationTuner,
        )

        assert getattr(KDBlendAdaptationTuner, "_max_stabilization_rounds", 1) >= 3


class TestStabilizationRefindsLR:
    """When finalize installed a deployed forward distinct from the ramp (KD
    blend cascaded / cycle-accurate LIF), the cached LR was tuned on the proxy
    forward and is stale; stabilization re-finds it once at entry. Tuners that
    do not swap the forward keep the cached LR (no wasted LR search)."""

    def _tuner(self, tmp_path, refinds):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        t = _RoundsTuner(MockPipeline(config=cfg, working_directory=str(tmp_path)),
                         make_tiny_supermodel(), 0.99, 0.001)
        t._wire()
        t._committed_rate = 1.0
        t._validation_baseline = 0.5
        t._pipeline_hard_floor = None
        t._rollback_tolerance = 0.05
        t._cached_lr = 0.0005
        t._stabilization_refinds_lr = refinds
        return t

    def test_refinds_lr_when_forward_swapped(self, tmp_path):
        t = self._tuner(tmp_path, refinds=True)
        t._wire_evals([0.5, 0.9])
        t._stabilize_at_full_rate()
        assert t.find_lr_calls >= 1, "must re-find LR for the deployed forward"

    def test_keeps_cached_lr_when_no_swap(self, tmp_path):
        t = self._tuner(tmp_path, refinds=False)
        t._wire_evals([0.5, 0.9])
        t._stabilize_at_full_rate()
        assert t.find_lr_calls == 0
        assert t.train_calls[0]["lr"] == 0.0005

    def test_kd_blend_marks_refind_after_installing_forward(self):
        from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
            KDBlendAdaptationTuner,
        )

        class _Stub:
            _installed = None

            # The base _finalize calls these ordered hooks (no-ops by default;
            # LIF overrides them). A bare stub mirrors the base no-ops.
            def _before_finalize_rebuild(self):
                pass

            def _after_finalize_rebuild(self):
                pass

            def _update_target_activations(self):
                pass

            def _finalize_forward(self):
                return object()  # a deployed forward distinct from the ramp

            def _install_forward(self, fwd):
                self._installed = fwd

        stub = _Stub()
        KDBlendAdaptationTuner._finalize(stub)
        assert stub._installed is not None
        assert stub._stabilization_refinds_lr is True
