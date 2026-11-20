"""[U-TGAUGE3] The step rule (the U-TGAUGE2 verify's M-2): every step clears every gradient its backward may write
before that backward, the deployed model's included, in every step loop; the refusal stays for a step whose update
writes nothing, decided by what the update writes and never by the optimizer's membership.

A transform trainer's step differentiates the deployed model and its update consumes those gradients through its
transfer to the trained copies. Its optimizer holds the trained copies only, so clearing the optimizer's gradients left
any gradient another step loop had written on the deployed model (the fast ladder's rung, a gradient read) in place,
and the trainer's first backward added into it (a clipped gradient 0.33 from a fresh one after the WQ tuner's own
rungs; 0.10 and 0.63 after a foreign backward). The rule is ``clear_step_gradients(optimizer, model, writes=...)``:
``writes`` names the parameters of the differentiated model whose gradients the step's update consumes (a plain step:
its optimizer's; a transform trainer: the deployed parameters its transfer moves). Each test reads the gradient the
step clips (where the optimizer reads it) against a fresh backward at the same parameters on the same batch.
"""

from __future__ import annotations

import copy

import pytest
import torch

import unit.tuning.test_step_gradients_never_accumulate as base

from mimarsinan.model_training.training_recipe import UnwrittenStepError, clear_step_gradients
from mimarsinan.models.nn.layers import freeze_batchnorm_running_stats


def _first_step_gap(trainer, optimizer, scaler, monkeypatch):
    """(names holding a gradient when the step's backward starts, the largest gap between the gradient the step clips
    and a fresh one)."""
    batch = trainer.next_training_batch()
    if hasattr(trainer, "_update_and_transform_model"):
        with torch.no_grad():
            trainer._update_and_transform_model()
    probe = copy.deepcopy(trainer.model)
    probe.zero_grad(set_to_none=True)
    probe.train()  # the step's own forward mode (``BasicTrainer._backward_pass_on_loss``)
    if trainer.freeze_bn_stats_in_training:
        freeze_batchnorm_running_stats(probe)
    trainer.loss_function(probe, *batch).backward()
    fresh = {n: p.grad.detach().clone() for n, p in probe.named_parameters() if p.grad is not None}
    spy = base._Spy(trainer.model)
    loss_function = trainer.loss_function
    trainer.loss_function = lambda model, x, y: spy.watch(loss_function(model, x, y))
    seen = {}
    real = torch.nn.utils.clip_grad_norm_

    def clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        names = {id(p): n for n, p in base._trained(trainer).named_parameters()}
        seen.update({names[id(p)]: p.grad.detach().clone() for p in parameters if p.grad is not None and id(p) in names})
        return real(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    try:
        trainer._optimize(*batch, optimizer, scaler)
    finally:
        trainer.loss_function = loss_function
    assert seen and set(seen) <= set(fresh)
    return spy.leftovers, max(float((g - fresh[n]).abs().max()) for n, g in seen.items())


@pytest.mark.parametrize("kind", sorted(base.TRAINERS))
def test_a_trainer_s_step_ignores_a_gradient_another_loop_left_on_its_model(kind, monkeypatch):
    """A foreign backward on the deployed model (another loop's last step, cleared only before that loop's next one),
    then the trainer's first step: the verify's gaps 0.63 (weight trainer) and 0.10 (perceptron trainer)."""
    torch.manual_seed(0)
    trainer = base.TRAINERS[kind]()
    optimizer, _scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-3, 10)
    foreign = trainer.next_training_batch()
    trainer.loss_function(trainer.model, *foreign).backward()
    leftovers, gap = _first_step_gap(trainer, optimizer, scaler, monkeypatch)
    assert leftovers == [[]], leftovers
    assert gap <= 1e-6


def test_the_toolchain_s_own_sequence_the_fast_rungs_then_the_trainer_s_step(tmp_path, monkeypatch):
    """The WQ tuner's fast rungs with the declared default recipe (its rung optimizer over the deployed model), then
    its perceptron trainer's own step on that model: the verify's gap 0.33."""
    tuner = base._napq_tuner(tmp_path, recipe=True)
    try:
        tuner._ensure_fast_optimizer()
        tuner._fast_train_rung(0.5)
        trainer = tuner.trainer
        assert trainer.model is tuner.model
        optimizer, _scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-3, 10)
        leftovers, gap = _first_step_gap(trainer, optimizer, scaler, monkeypatch)
        assert leftovers == [[]], leftovers
        assert gap <= 1e-6
    finally:
        tuner.close()


def test_the_refusal_is_decided_by_what_the_update_writes():
    """A transform trainer's step: its optimizer holds the trained copies only, yet its update writes the deployed
    model through its transfer, so the step clears and does not refuse; an update that writes none of the parameters
    the backward reaches refuses by name, whatever its optimizer holds."""
    model, copies = base.GainNet(), base.GainNet()
    model(torch.rand(2, 8)).sum().backward()
    optimizer = torch.optim.SGD(copies.parameters(), lr=0.1)
    clear_step_gradients(optimizer, model, writes=list(model.parameters()), step="a transfer step")
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(UnwrittenStepError, match="an empty update"):
        clear_step_gradients(torch.optim.SGD(model.parameters(), lr=0.1), model, writes=[], step="an empty update")
