"""Entry-anchored keep-best in ``train_steps_until_target`` (W2 fix C, the M-guard).

Keep-best seeds from the ENTRY state and metric, never 0.0: a recovery run
that never beats entry restores entry EXACTLY (full state_dict including
buffers) instead of committing a wrecked-but-nonzero checkpoint.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training.basic_trainer_steps import train_steps_until_target


class _BufferedModel(nn.Module):
    """Linear + a raw buffer so entry restore is observable on buffers too."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2, bias=False)
        self.register_buffer("running_stat", torch.zeros(1))

    def forward(self, x):
        return self.lin(x)


class _ScriptedTrainer:
    """Step-trainer duck type whose accuracy is a pure function of steps trained."""

    def __init__(self, acc_fn):
        self.model = _BufferedModel()
        with torch.no_grad():
            self.model.lin.weight.zero_()
        self.device = "cpu"
        self._acc_fn = acc_fn

    @property
    def steps_trained(self) -> int:
        return int(round(float(self.model.lin.weight[0, 0].item())))

    def _get_optimizer_and_scheduler_steps(self, lr, steps, constant_lr=False):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0
        )
        return optimizer, scheduler, None

    def next_training_batch(self):
        return torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)

    def _optimize(self, x, y, optimizer, scaler):
        with torch.no_grad():
            self.model.lin.weight += 1.0
            self.model.running_stat += 1.0

    def validate_n_batches(self, n):
        return float(self._acc_fn(self.steps_trained))

    def _report(self, name, value):
        pass


def _run(trainer, *, target=0.95, max_steps=6, patience=2, **kwargs):
    return train_steps_until_target(
        trainer, 0.1, max_steps, target,
        check_interval=1, patience=patience, **kwargs,
    )


class TestNeverBelowEntry:
    def test_wrecked_training_restores_entry_not_wreck(self):
        """The crater class: every post-step probe reads ~0.1; the step must
        become a no-op at the entry metric, not a committed wreck."""
        tr = _ScriptedTrainer(lambda steps: 0.9 if steps == 0 else 0.1)
        final = _run(tr)
        assert final == pytest.approx(0.9)
        assert tr.steps_trained == 0, "entry weights must be restored exactly"

    def test_degraded_but_nonzero_run_still_restores_entry(self):
        """A 0.5 checkpoint beats the old 0.0 seed but not entry (0.9): the
        M-guard must anchor keep-best at entry, not at zero."""
        tr = _ScriptedTrainer(lambda steps: 0.9 if steps == 0 else 0.5)
        final = _run(tr)
        assert final == pytest.approx(0.9)
        assert tr.steps_trained == 0

    def test_entry_restore_includes_buffers(self):
        tr = _ScriptedTrainer(lambda steps: 0.9 if steps == 0 else 0.1)
        _run(tr)
        assert float(tr.model.running_stat.item()) == 0.0, (
            "the entry restore must cover buffers, not just parameters"
        )

    def test_return_steps_reports_steps_run_after_entry_restore(self):
        tr = _ScriptedTrainer(lambda steps: 0.9 if steps == 0 else 0.1)
        final, steps = _run(tr, return_steps=True)
        assert final == pytest.approx(0.9)
        assert steps >= 1


class TestHealthyPathUnchanged:
    def test_improving_run_keeps_the_best_trained_state(self):
        tr = _ScriptedTrainer(lambda steps: min(0.5 + 0.05 * steps, 0.8))
        final = _run(tr, target=0.99, max_steps=4, patience=10)
        assert final == pytest.approx(0.7)
        assert tr.steps_trained == 4

    def test_target_reached_commits_the_reaching_state(self):
        tr = _ScriptedTrainer(lambda steps: 0.9 if steps == 0 else 0.96)
        final = _run(tr, target=0.95, max_steps=6, patience=3)
        assert final == pytest.approx(0.96)
        assert tr.steps_trained == 1, (
            "the target-reaching checkpoint commits (bonus steps are restored away)"
        )


class TestWallCap:
    """[5u amendment] ``max_seconds``: the floor-lifted endpoint funds a WALL
    budget, not a step count — under pack contention the same steps cost 3-4x
    (measured 402 s for 12k steps on the fba wave), so the stage must stop at
    its funded headroom and keep the best state reached by then."""

    def _fake_clock(self, monkeypatch, seconds_per_tick=1.0):
        import mimarsinan.model_training.basic_trainer_steps as steps_mod

        state = {"t": 0.0}

        def monotonic():
            state["t"] += seconds_per_tick
            return state["t"]

        monkeypatch.setattr(steps_mod.time, "monotonic", monotonic)
        return state

    def test_wall_cap_stops_an_improving_run_and_keeps_best(self, monkeypatch):
        # The clock ticks once per checkpoint; a 5 s cap stops well before the
        # 100-step budget, and keep-best commits the last improved state.
        self._fake_clock(monkeypatch)
        tr = _ScriptedTrainer(lambda steps: min(0.5 + 0.01 * steps, 0.8))
        final, steps = _run(
            tr, target=0.99, max_steps=100, patience=100,
            min_steps=100, max_seconds=5.0, return_steps=True,
        )
        assert steps < 100
        assert final == pytest.approx(0.5 + 0.01 * steps)
        assert tr.steps_trained == steps

    def test_wall_cap_never_ends_below_entry(self, monkeypatch):
        self._fake_clock(monkeypatch)
        tr = _ScriptedTrainer(lambda steps: 0.9 if steps == 0 else 0.1)
        final = _run(
            tr, target=0.99, max_steps=100, patience=100,
            min_steps=100, max_seconds=5.0,
        )
        assert final == pytest.approx(0.9)
        assert tr.steps_trained == 0

    def test_no_cap_runs_the_full_budget(self, monkeypatch):
        self._fake_clock(monkeypatch)
        tr = _ScriptedTrainer(lambda steps: min(0.5 + 0.01 * steps, 0.8))
        final, steps = _run(
            tr, target=0.99, max_steps=8, patience=100,
            min_steps=8, max_seconds=None, return_steps=True,
        )
        assert steps == 8

    def test_target_reach_still_exits_before_the_cap(self, monkeypatch):
        self._fake_clock(monkeypatch)
        tr = _ScriptedTrainer(lambda steps: 0.5 if steps == 0 else 0.96)
        final, steps = _run(
            tr, target=0.95, max_steps=100, patience=100,
            min_steps=100, max_seconds=50.0, return_steps=True,
        )
        assert final == pytest.approx(0.96)
        assert steps <= 4
