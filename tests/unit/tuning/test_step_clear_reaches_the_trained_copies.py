"""[U-TGAUGE4] The step rule's clear reaches the optimizer's own gradients (the U-TGAUGE3 verify's W30 gap).

A transform trainer's optimizer holds trained copies outside the model its backward differentiates, and its transfer
writes a copy's gradient only where the deployed parameter received one. A perceptron that a step's loss does not reach
receives none, so its copy keeps the last step's gradient unless the clear empties the optimizer's own gradients too:
without that clear the copy is stepped again on a stale gradient. No test had a copy whose deployed parameter received
no gradient in a step.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

import unit.tuning.test_step_gradients_never_accumulate as base

from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer
from mimarsinan.model_training.training_recipe import clear_step_gradients
from mimarsinan.torch_mapping.converter import convert_torch_model


class _TwoHeads(nn.Module):
    """One stem, two heads joined: a loss over the first head's outputs reaches the second head's perceptron not."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.a, self.b = nn.Linear(8, 6), nn.Linear(6, 3), nn.Linear(6, 3)
        self.r0, self.ra, self.rb = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x):
        h = self.r0(self.stem(x.flatten(1)))
        return torch.cat([self.ra(self.a(h)), self.rb(self.b(h))], dim=1)


def test_the_clear_empties_the_optimizer_s_own_gradients():
    model, copies = base.GainNet(), base.GainNet()
    model(torch.rand(2, 8)).sum().backward()
    copies(torch.rand(2, 8)).sum().backward()
    optimizer = torch.optim.SGD(copies.parameters(), lr=0.1)
    clear_step_gradients(optimizer, model, writes=list(model.parameters()), step="a transfer step")
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in copies.parameters())


def test_a_trained_copy_its_step_s_loss_does_not_reach_is_not_stepped_again():
    """Step 1's loss reads both heads; step 2's only the first head's own output (the second head's is outside its
    backward). The second head's trained copies must hold after step 2 what step 1 left (no gradient reached them); a
    stale gradient from step 1 would step them again."""
    torch.manual_seed(0)
    flow = convert_torch_model(_TwoHeads().eval(), (8,), 6, device="cpu", encoding_layer_placement="subsume").train()
    calls = {"n": 0}

    def loss(model, x, y):
        calls["n"] += 1
        held: dict = {}
        first = next(p for p in model.get_perceptrons() if p.name == "a")
        handle = first.register_forward_hook(lambda _m, _i, out: held.setdefault("a", out))
        try:
            out = model(x)
        finally:
            handle.remove()
        return out.square().mean() if calls["n"] == 1 else held["a"].square().mean()

    trainer = PerceptronTransformTrainer(flow, "cpu", base._loader(), loss, lambda perceptron: None,
                                         recipe=base.RECIPE)
    optimizer, _scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-2, 10)
    batches = [trainer.next_training_batch() for _ in range(2)]
    trainer._optimize(*batches[0], optimizer, scaler)
    second_head = next(p for p in trainer.aux_model.get_perceptrons() if p.name == "b")
    after_first = copy.deepcopy({n: p.detach().clone() for n, p in second_head.named_parameters()})
    first_head = next(p for p in trainer.aux_model.get_perceptrons() if p.name == "a")
    first_before = {n: p.detach().clone() for n, p in first_head.named_parameters()}
    trainer._optimize(*batches[1], optimizer, scaler)
    for name, p in second_head.named_parameters():
        assert torch.equal(p.detach(), after_first[name]), name
    assert any(not torch.equal(p.detach(), first_before[n]) for n, p in first_head.named_parameters())
