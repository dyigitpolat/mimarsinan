"""[calculus §16.13] the parity gate reports each disagreeing sample's flip
DIRECTION vs labels when labels are provided (torch-right/sim-right/both-wrong)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.pipelining.core.nf_scm_parity import (
    assert_torch_vs_deployed_sim_parity_or_raise,
)


class _Fixed(nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = preds

    def forward(self, x):
        out = torch.zeros(x.shape[0], 4)
        out[torch.arange(x.shape[0]), self.preds] = 1.0
        return out


def test_flip_directions_are_reported(capsys):
    torch_side = _Fixed(torch.tensor([0, 1, 2, 3]))
    sim_side = _Fixed(torch.tensor([0, 2, 2, 1]))  # flips at idx 1 and 3
    labels = torch.tensor([0, 1, 2, 2])  # idx1: torch right; idx3: both wrong
    x = torch.zeros(4, 3)
    agreement = assert_torch_vs_deployed_sim_parity_or_raise(
        torch_side, sim_side, x, min_agreement=0.4, labels=labels,
    )
    assert abs(agreement - 0.5) < 1e-9
    out = capsys.readouterr().out
    assert "torch-right-sim-wrong=1" in out
    assert "both-wrong=1" in out
    assert "sim-right-torch-wrong=0" in out
