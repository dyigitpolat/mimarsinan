"""DeployedRiskFinetune (spiking_deployment_calculus.md §15.4): the generic
terminal stage — origin-KD minimization of the DEPLOYED risk through an
injected deployed forward, keep-best on the injected genuine eval, and
intra-stage checkpoint/resume so the stage owns its window survival.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.tuning.orchestration.deployed_risk_finetune import (
    run_deployed_risk_finetune,
)


class _Quantized(nn.Module):
    """A toy 'deployed composition': linear head behind a hard input grid."""

    def __init__(self, in_f=8, classes=4, levels=8):
        super().__init__()
        self.head = nn.Linear(in_f, classes)
        self.levels = levels

    def forward(self, x):
        grid = torch.round(x * self.levels) / self.levels
        return self.head(grid + (x - x.detach()))  # STE through the grid


def _toy_problem(seed=0, n=256, in_f=8, classes=4):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, in_f, generator=g)
    w = torch.randn(in_f, classes, generator=g)
    y = (x @ w).argmax(1)
    teacher = nn.Linear(in_f, classes)
    with torch.no_grad():
        teacher.weight.copy_(w.T)
        teacher.bias.zero_()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    batches = [(x[i : i + 32], y[i : i + 32]) for i in range(0, n, 32)]
    return batches, teacher


def _accuracy(model, batches):
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            correct += int((model(x).argmax(1) == y).sum())
            total += int(y.numel())
    return correct / total


class TestDeployedRiskFinetune:
    def test_descends_the_deployed_risk_on_a_toy_composition(self):
        torch.manual_seed(1)
        batches, teacher = _toy_problem()
        model = _Quantized()
        entry = _accuracy(model, batches)
        result = run_deployed_risk_finetune(
            model, teacher, model.forward,
            train_batches=batches * 20,
            eval_genuine=lambda: _accuracy(model, batches),
            steps=120, lr=5e-2, eval_every=30,
        )
        assert result.steps_run == 120
        assert result.final_genuine > entry + 0.1
        assert result.best_genuine >= result.final_genuine - 1e-9

    def test_teacher_stays_frozen(self):
        batches, teacher = _toy_problem()
        before = [p.detach().clone() for p in teacher.parameters()]
        run_deployed_risk_finetune(
            _Quantized(), teacher, lambda x: _Quantized()(x),
            train_batches=batches * 4,
            eval_genuine=lambda: 0.5,
            steps=20, lr=1e-2, eval_every=10,
        )
        for p, b in zip(teacher.parameters(), before):
            torch.testing.assert_close(p, b)

    def test_keep_best_restores_the_best_state(self):
        """A rigged eval that peaks early: the returned model must carry the
        state from the PEAK eval point, not the final step."""
        torch.manual_seed(2)
        batches, teacher = _toy_problem()
        model = _Quantized()
        readings = iter([0.9, 0.3, 0.2, 0.1, 0.1, 0.1])
        snapshots = {}

        def rigged_eval():
            value = next(readings)
            snapshots[value] = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            return value

        result = run_deployed_risk_finetune(
            model, teacher, model.forward,
            train_batches=batches * 20,
            eval_genuine=rigged_eval,
            steps=100, lr=5e-2, eval_every=25,
        )
        assert result.best_genuine == 0.9
        for key, value in snapshots[0.9].items():
            torch.testing.assert_close(model.state_dict()[key], value)

    def test_checkpoint_resume_owns_the_window(self, tmp_path):
        """Two invocations with the same checkpoint reach the same total step
        count as one uninterrupted run — the stage owns its resume."""
        batches, teacher = _toy_problem()
        ckpt = str(tmp_path / "stage.ckpt")
        model = _Quantized()
        first = run_deployed_risk_finetune(
            model, teacher, model.forward,
            train_batches=batches * 20,
            eval_genuine=lambda: 0.5,
            steps=40, lr=1e-2, eval_every=20,
            checkpoint_path=ckpt, checkpoint_every=15, stop_after=25,
        )
        assert first.steps_run == 25
        second = run_deployed_risk_finetune(
            model, teacher, model.forward,
            train_batches=batches * 20,
            eval_genuine=lambda: 0.5,
            steps=40, lr=1e-2, eval_every=20,
            checkpoint_path=ckpt, checkpoint_every=15,
        )
        assert second.resumed_from >= 15
        assert second.steps_run == 40
