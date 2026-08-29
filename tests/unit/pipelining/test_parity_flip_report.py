"""[calculus §16.13] the readout-drift report names each MOVED decision's
direction vs labels when labels are provided (torch-right/sim-right/both-wrong).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.pipelining.core.nf_scm_parity import (
    measure_readout_decision_drift,
)


class _Fixed(nn.Module):
    def __init__(self, preds):
        super().__init__()
        self.preds = preds

    def forward(self, x):
        out = torch.zeros(x.shape[0], 4)
        out[torch.arange(x.shape[0]), self.preds] = 1.0
        return out


def test_moved_decision_directions_are_reported(capsys):
    torch_side = _Fixed(torch.tensor([0, 1, 2, 3]))
    sim_side = _Fixed(torch.tensor([0, 2, 2, 1]))  # decisions move at idx 1 and 3
    labels = torch.tensor([0, 1, 2, 2])  # idx1: torch right; idx3: both wrong
    x = torch.zeros(4, 3)
    agreement = measure_readout_decision_drift(
        torch_side, sim_side, x, labels=labels,
    )
    # The statistic itself counts READOUT NEURONS, not decisions: the two moved
    # samples disagree on two classes each, 4 of 4x4 = 16 counts.
    assert abs(agreement - 0.75) < 1e-9
    out = capsys.readouterr().out
    assert "decisions_moved=2" in out
    assert "torch-right-sim-wrong=1" in out
    assert "both-wrong=1" in out
    assert "sim-right-torch-wrong=0" in out
