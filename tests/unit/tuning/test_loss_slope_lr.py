"""Tests for the cheap loss-slope coarse LR signal (tuning_loss_slope_lr)."""

import pytest

from mimarsinan.tuning.learning_rate_explorer import (
    LRRangeFinder,
    make_loss_slope_signal,
)


class _SurfaceTrainer:
    """Mock trainer whose accuracy/loss both depend on the last probe LR.

    ``loss`` is a 'lower is better' coarse signal; ``acc`` is the fine
    validation signal.  Both are pure functions of the current LR so the
    coarse pass and the full sweep see a consistent surface.
    """

    def __init__(self, acc_fn, loss_fn):
        self._lr = 0.0
        self._acc_fn = acc_fn
        self._loss_fn = loss_fn

    def train_n_steps(self, lr, steps, **kwargs):
        self._lr = lr

    def acc(self):
        return self._acc_fn(self._lr)

    def loss(self):
        return self._loss_fn(self._lr)


def _finder(tr, *, validate_fn, coarse_signal=None, coarse_top_k=3, num_probes=8):
    return LRRangeFinder(
        trainer=tr,
        clone_state=lambda: None,
        restore_state=lambda s: None,
        lr_min=1e-5,
        lr_max=1e-1,
        num_probes=num_probes,
        steps_per_probe=5,
        validate_fn=validate_fn,
        coarse_signal=coarse_signal,
        coarse_top_k=coarse_top_k,
    )


def _counting(fn):
    calls = [0]

    def wrapped():
        calls[0] += 1
        return fn()

    return wrapped, calls


# Surface: accuracy peaks in the mid band, collapses for high LRs; loss is
# (1 - acc)-shaped so the cheap signal and the fine signal agree on ordering.
def _acc_surface(lr):
    if lr < 1e-3:
        return 0.70
    if lr < 1e-2:
        return 0.85
    return 0.10


def _loss_surface(lr):
    return 1.0 - _acc_surface(lr)


class TestLossSlopeCoarseSignal:
    def test_flag_off_is_byte_identical_selection(self):
        """coarse_signal=None must reproduce the full-validation selection."""
        tr_full = _SurfaceTrainer(_acc_surface, _loss_surface)
        full_validate, _ = _counting(tr_full.acc)
        lr_full = _finder(tr_full, validate_fn=full_validate).find_best_lr()

        tr_coarse = _SurfaceTrainer(_acc_surface, _loss_surface)
        coarse_validate, _ = _counting(tr_coarse.acc)
        lr_coarse = _finder(
            tr_coarse,
            validate_fn=coarse_validate,
            coarse_signal=tr_coarse.loss,
        ).find_best_lr()

        assert lr_coarse == lr_full

    def test_coarse_reduces_full_validate_calls(self):
        """The coarse pass must spend fewer full validate_fn calls."""
        tr_full = _SurfaceTrainer(_acc_surface, _loss_surface)
        full_validate, full_calls = _counting(tr_full.acc)
        _finder(tr_full, validate_fn=full_validate).find_best_lr()

        tr_coarse = _SurfaceTrainer(_acc_surface, _loss_surface)
        coarse_validate, coarse_calls = _counting(tr_coarse.acc)
        _finder(
            tr_coarse,
            validate_fn=coarse_validate,
            coarse_signal=tr_coarse.loss,
            coarse_top_k=3,
        ).find_best_lr()

        assert coarse_calls[0] < full_calls[0], (
            f"coarse={coarse_calls[0]} should be < full={full_calls[0]}"
        )

    def test_coarse_only_full_validates_baseline_plus_top_k(self):
        """Exactly 1 baseline + coarse_top_k full validations, no more."""
        tr = _SurfaceTrainer(_acc_surface, _loss_surface)
        validate, calls = _counting(tr.acc)
        _finder(
            tr,
            validate_fn=validate,
            coarse_signal=tr.loss,
            coarse_top_k=3,
            num_probes=8,
        ).find_best_lr()
        assert calls[0] == 1 + 3

    def test_coarse_picks_sane_lr(self):
        """The coarse signal must still land in the non-destructive band."""
        tr = _SurfaceTrainer(_acc_surface, _loss_surface)
        validate, _ = _counting(tr.acc)
        lr = _finder(
            tr,
            validate_fn=validate,
            coarse_signal=tr.loss,
        ).find_best_lr()
        assert 1e-5 < lr < 1e-2, f"coarse LR landed outside the sane band: {lr}"

    def test_coarse_rejects_destructive_high_lr(self):
        """High LRs that destroy accuracy are avoided even when surveyed cheaply."""

        def acc(lr):
            return 0.05 if lr > 0.01 else 0.75

        def loss(lr):
            # Deliberately make the cheap signal *favor* a high LR so the
            # full validate_fn re-check is load-bearing for rejecting it.
            return -lr

        tr = _SurfaceTrainer(acc, loss)
        validate, _ = _counting(tr.acc)
        lr = _finder(
            tr,
            validate_fn=validate,
            coarse_signal=tr.loss,
            coarse_top_k=3,
        ).find_best_lr()
        assert lr <= 0.01, f"Should avoid destructive high LRs, got {lr}"

    def test_top_k_one_validates_once_beyond_baseline(self):
        tr = _SurfaceTrainer(_acc_surface, _loss_surface)
        validate, calls = _counting(tr.acc)
        _finder(
            tr,
            validate_fn=validate,
            coarse_signal=tr.loss,
            coarse_top_k=1,
        ).find_best_lr()
        assert calls[0] == 1 + 1

    def test_restore_called_each_probe_and_finally(self):
        """State must be restored before every probe and once on exit."""
        restores = [0]
        tr = _SurfaceTrainer(_acc_surface, _loss_surface)
        finder = LRRangeFinder(
            trainer=tr,
            clone_state=lambda: object(),
            restore_state=lambda s: restores.__setitem__(0, restores[0] + 1),
            lr_min=1e-5,
            lr_max=1e-1,
            num_probes=8,
            steps_per_probe=5,
            validate_fn=tr.acc,
            coarse_signal=tr.loss,
            coarse_top_k=3,
        )
        finder.find_best_lr()
        # 8 coarse-pass restores + 3 fine-pass restores + 1 finally restore.
        assert restores[0] == 8 + 3 + 1


class TestMakeLossSlopeSignal:
    def test_signal_reads_training_batch_loss(self):
        """make_loss_slope_signal evaluates loss on a fresh training batch."""

        class _T:
            def __init__(self):
                self.batches = 0

            def next_training_batch(self):
                self.batches += 1
                return ("x", "y")

            def evaluate_loss_on_batch(self, batch):
                assert batch == ("x", "y")
                return 0.42

        tr = _T()
        sig = make_loss_slope_signal(tr)
        assert sig() == pytest.approx(0.42)
        assert tr.batches == 1
        sig()
        assert tr.batches == 2
