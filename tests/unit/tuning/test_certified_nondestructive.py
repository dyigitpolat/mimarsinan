"""R7d: the CERTIFIED NON-DESTRUCTIVE controller mode.

The fixed ladder / smooth-adaptation tail can ship a model strictly WORSE than the
best COMMITTED state seen across the whole run: the per-cycle rollback restores the
PRE-CYCLE clone (not the run best), and ``_after_run``'s forced jump to rate 1.0 +
the stabilization pass are only bracketed by LOCAL guards (entry-to-stabilization,
not run best). A long ramp can therefore drift below the best committed cycle.

``tuning_keepbest_certified`` (default off => byte-identical) makes the run
non-destructive end-to-end: every commit whose gate metric beats the running best
snapshots the model STATE, and ``_finalize_run`` restores that best state iff the
finalized gate metric is worse than it. The guard brackets the WHOLE
ramp + after_run + stabilization, mirroring the existing local stabilization guard
but at run scope.

These tests are TESTS-FIRST and non-vacuous: ``test_mutation_removing_restore_*``
proves that without the finalize restore the certified invariant FAILS.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


def _pipeline(tmp_path, **overrides):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    cfg.update(overrides)
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


# A constant "signature" parameter value per rate so a restored state is identifiable.
def _signature_of(model):
    return float(next(model.parameters()).detach().flatten()[0].item())


class _DegradingRunTuner(SmoothAdaptationTuner):
    """A tuner whose committed accuracy PEAKS at an intermediate rate and then
    DEGRADES toward rate 1.0 (a death-spiral-shaped ramp).

    Applying a rate stamps a rate-derived signature into the model parameters and
    makes ``validate_n_batches`` return the scripted accuracy for the currently
    applied rate. This lets a test (a) drive the run loop deterministically and
    (b) verify which committed STATE the finalized model carries (via the stamped
    signature), proving the keep-best restore actually reverted the parameters.
    """

    def __init__(self, pipeline, model, target_accuracy, lr, acc_for_rate):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._acc_for_rate = acc_for_rate
        self._committed_rate = 0.0
        # Validation reads the model's ACTUAL stamped state (the effective rate
        # recovered from the signature), not a side-channel ``_applied_rate``, so a
        # state restore at finalize is reflected in the measured accuracy.
        self.trainer.validate_n_batches = lambda n: self._acc_for_rate(self._effective_rate())
        self.trainer.validate = lambda: self._acc_for_rate(self._effective_rate())
        self.trainer.train_steps_until_target = lambda *a, **k: None
        self.trainer.train_n_steps = lambda *a, **k: None
        self.trainer.test = lambda: self._acc_for_rate(1.0)
        self._stamp(0.0)

    def _effective_rate(self):
        """The rate the current model state encodes (signature == rate + 1.0)."""
        return _signature_of(self.model) - 1.0

    def _stamp(self, rate):
        """Write a rate-derived signature into the model so the restored state is
        physically identifiable (not just a scalar metric)."""
        with torch.no_grad():
            for p in self.model.parameters():
                p.data.fill_(float(rate) + 1.0)

    def _update_and_evaluate(self, rate):
        self._stamp(rate)
        return self._acc_for_rate(float(rate))

    def _find_lr(self):
        return 0.001

    def _stabilization_budget(self):
        return 0  # isolate the ramp/after_run regression from the stabilization pass

    def _after_run(self):
        # Force rate to 1.0 (the worst rung in this fixture) — the unguarded jump
        # the certified mode must defend against.
        self._continue_to_full_rate()
        self._update_and_evaluate(1.0)
        self._committed_rate = 1.0
        return self._acc_for_rate(1.0)


# Accuracy peaks at rate 0.5 (0.95), is mediocre below, and COLLAPSES at rate 1.0.
def _peaked_acc(rate):
    if rate >= 1.0 - 1e-6:
        return 0.40
    if rate >= 0.5 - 1e-6:
        return 0.95
    return 0.90


def _make(tmp_path, certified, acc=_peaked_acc, target=0.9, **cfg):
    pipeline = _pipeline(tmp_path, tuning_keepbest_certified=certified, **cfg)
    model = make_tiny_supermodel()
    tuner = _DegradingRunTuner(pipeline, model, target_accuracy=target, lr=0.001, acc_for_rate=acc)
    return tuner


class TestDefaultPathShipsRegression:
    """The default (flag-off) controller CAN finalize below the best committed
    state — this is exactly the R7 regression the fixed ladder ships."""

    def test_default_finalizes_below_best_committed(self, tmp_path, deterministic_rng):
        tuner = _make(tmp_path, certified=False)
        tuner.run()

        # The run committed a 0.95 cycle (rate 0.5) but finalized at rate 1.0 (0.40).
        best_seen = max(
            r.post_acc for r in tuner._cycle_log.records
            if r.outcome == "commit" and r.post_acc is not None
        )
        final = float(tuner.trainer.validate_n_batches(tuner._budget.eval_n_batches))

        assert best_seen >= 0.95 - 1e-9, f"a 0.95 cycle must have committed; got {best_seen}"
        assert final < best_seen - 0.05, (
            "DEFAULT path must be able to ship strictly below the best committed "
            f"state (the R7 regression): final={final}, best_seen={best_seen}"
        )
        # The finalized parameters carry the rate-1.0 signature (the worse state).
        assert _signature_of(tuner.model) == pytest.approx(2.0)


class TestCertifiedKeepsBest:
    """The certified mode finalizes AT the best committed gate metric — never worse."""

    def test_certified_finalizes_at_best_committed(self, tmp_path, deterministic_rng):
        tuner = _make(tmp_path, certified=True)
        tuner.run()

        best_seen = max(
            r.post_acc for r in tuner._cycle_log.records
            if r.outcome == "commit" and r.post_acc is not None
        )
        final = float(tuner.trainer.validate_n_batches(tuner._budget.eval_n_batches))

        assert best_seen >= 0.95 - 1e-9
        assert final >= best_seen - 1e-9, (
            "CERTIFIED mode must finalize at (or above) the best committed gate "
            f"metric: final={final}, best_seen={best_seen}"
        )
        # The finalized parameters carry the BEST committed cycle's signature
        # (rate 0.5 => signature 1.5), proving the STATE — not just a scalar — was
        # restored.
        assert _signature_of(tuner.model) == pytest.approx(1.5), (
            "certified finalize must restore the best committed STATE's parameters"
        )

    def test_certified_no_restore_when_final_is_best(self, tmp_path, deterministic_rng):
        """When rate 1.0 is the BEST rung, the certified mode must NOT roll the
        model back (no spurious restore) — it ships the genuine final state."""
        def monotone(rate):
            return 0.80 + 0.15 * float(rate)  # best at rate 1.0

        tuner = _make(tmp_path, certified=True, acc=monotone)
        tuner.run()
        final = float(tuner.trainer.validate_n_batches(tuner._budget.eval_n_batches))
        assert final == pytest.approx(0.95)
        assert _signature_of(tuner.model) == pytest.approx(2.0), (
            "with rate 1.0 the best rung, certified mode must ship the rate-1.0 state"
        )


class TestMutationGuard:
    """Non-vacuity: removing the finalize restore must make the certified
    invariant FAIL (the test is sensitive to the keep-best mechanism)."""

    def test_mutation_removing_restore_breaks_certified(self, tmp_path, deterministic_rng, monkeypatch):
        tuner = _make(tmp_path, certified=True)

        # Neuter the finalize restore — the mutation under test (return the result
        # unchanged, i.e. never restore the best state).
        monkeypatch.setattr(
            type(tuner), "_restore_best_committed_state",
            lambda self, finalize_result: finalize_result, raising=True,
        )
        tuner.run()

        final = float(tuner.trainer.validate_n_batches(tuner._budget.eval_n_batches))
        # Without the restore the model ships at rate 1.0 (0.40), below the 0.95 best.
        assert final < 0.95 - 0.05, (
            "MUTATION CHECK FAILED: with the keep-best restore removed the certified "
            f"path still kept the best — the test is vacuous (final={final})"
        )
        assert _signature_of(tuner.model) == pytest.approx(2.0)


class TestDefaultOffByteIdentical:
    """Flag off => the certified machinery is fully inert: no best-state snapshot
    is taken, no restore is attempted, and the finalized model is bit-identical to
    the legacy path."""

    def test_no_snapshot_taken_when_flag_off(self, tmp_path, deterministic_rng):
        tuner = _make(tmp_path, certified=False)
        tuner.run()
        assert getattr(tuner, "_best_committed_state", None) is None, (
            "flag off must never snapshot a best-committed state"
        )

    def test_flag_off_matches_unconfigured_run(self, tmp_path, deterministic_rng):
        """A run with the key absent and a run with the key explicitly False must
        produce bit-identical finalized parameters (no inadvertent path change)."""
        t_absent = _make(tmp_path / "a", certified=False)
        # Drop the key entirely to exercise the registered-default fallback.
        t_absent.pipeline.config.pop("tuning_keepbest_certified", None)
        t_absent.__class__  # no-op; keep model build before run
        t_absent.run()
        sig_absent = {k: v.clone() for k, v in t_absent.model.state_dict().items()}

        t_false = _make(tmp_path / "b", certified=False)
        t_false.run()
        sig_false = t_false.model.state_dict()

        for k, v in sig_absent.items():
            assert torch.equal(v, sig_false[k]), f"param {k} diverged flag-off vs absent"
