"""The recurrent leaves under ``decompose_leaf_modules``: input projections on the cores, the recurrence on the host.

An ``nn.LSTM``, ``nn.GRU`` or ``nn.RNN`` leaf called on one batch-first activation, its sequence output the one element
read, is lowered at conversion (its FX node stays the leaf, so Wave 0's declared shape rule and the native reference
are the leaf's): per layer, the input-to-hidden map of every step and direction as one Linear over the time axis
(``x W_ih^T + b_ih``, packaged by the converter's Linear rule: a perceptron under MVM), then one host op
(``Recurrence``) carrying ``W_hh``, ``b_hh`` (and ``W_hr``) and the gates, in torch's own order of operations, which
declares its MACs. Every other recurrent call stays the leaf.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

_GATES = {"LSTM": 4, "GRU": 3, "RNN_TANH": 1, "RNN_RELU": 1}


class Recurrence(nn.Module):
    """The hidden-to-hidden recurrence of one layer of a recurrent leaf, every direction, on the precomputed input
    gates ``gx`` (batch, time, directions x gates x hidden); returns ``(output,)``, the output sequence."""

    def __init__(self, leaf: nn.RNNBase, layer: int, steps: int):
        super().__init__()
        self.mode, self.directions = str(leaf.mode), 2 if leaf.bidirectional else 1
        self.hidden, self.proj = int(leaf.hidden_size), int(getattr(leaf, "proj_size", 0) or 0)
        self.steps = int(steps)
        suffixes = [f"_l{layer}"] + ([f"_l{layer}_reverse"] if leaf.bidirectional else [])
        for d, suffix in enumerate(suffixes):
            for kind in ("weight_hh", "bias_hh", "weight_hr"):
                value = getattr(leaf, kind + suffix, None)
                if value is not None:
                    self.register_parameter(f"{kind}_{d}", value)

    @property
    def input_width(self) -> int:
        return self.directions * _GATES[self.mode] * self.hidden

    @property
    def output_width(self) -> int:
        return self.directions * (self.proj or self.hidden)

    def _cell(self, d: int, gates_x: torch.Tensor, h: torch.Tensor, c: Optional[torch.Tensor]):
        hidden = F.linear(h, getattr(self, f"weight_hh_{d}"), getattr(self, f"bias_hh_{d}", None))
        if self.mode == "LSTM":
            i, f, g, o = hidden.add_(gates_x).chunk(4, 1)
            c = (f.sigmoid() * cast(torch.Tensor, c)).add_(i.sigmoid() * g.tanh())
            h = o.sigmoid() * c.tanh()
            weight_hr = getattr(self, f"weight_hr_{d}", None)
            return (h if weight_hr is None else F.linear(h, weight_hr)), c
        if self.mode == "GRU":
            xr, xz, xn = gates_x.chunk(3, 1)
            hr, hz, hn = hidden.chunk(3, 1)
            reset, update = hr.add_(xr).sigmoid_(), hz.add_(xz).sigmoid_()
            new = xn.add(hn.mul_(reset)).tanh_()
            return (h - new).mul_(update).add_(new), None
        pre = hidden.add_(gates_x)
        return (pre.tanh() if self.mode == "RNN_TANH" else pre.relu()), None

    def forward(self, gx: torch.Tensor) -> Tuple[torch.Tensor]:
        width = _GATES[self.mode] * self.hidden
        outputs: List[torch.Tensor] = []
        for d in range(self.directions):
            gates = gx[..., d * width:(d + 1) * width]
            h = gx.new_zeros(gx.shape[0], self.proj or self.hidden)
            c = gx.new_zeros(gx.shape[0], self.hidden) if self.mode == "LSTM" else None
            steps = range(gx.shape[1]) if d == 0 else reversed(range(gx.shape[1]))
            sequence: List[Optional[torch.Tensor]] = [None] * gx.shape[1]
            for t in steps:
                h, c = self._cell(d, gates[:, t], h, c)
                sequence[t] = h
            outputs.append(torch.stack([s for s in sequence if s is not None], dim=1))
        return (torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0],)

    def forward_macs(self) -> int:
        """Per sample: ``T · g · H · H_r`` per direction, plus ``T · H · H_r`` for an LSTM projection."""
        recurrent = self.proj or self.hidden
        per_step = _GATES[self.mode] * self.hidden * recurrent + (self.hidden * self.proj if self.proj else 0)
        return self.steps * per_step * self.directions


def input_projection(leaf: nn.RNNBase, layer: int) -> nn.Linear:
    """The input-to-hidden map of ``layer``, every direction stacked: ``W_ih`` and ``b_ih`` rows of each direction."""
    suffixes = [f"_l{layer}"] + ([f"_l{layer}_reverse"] if leaf.bidirectional else [])
    weight = torch.cat([getattr(leaf, "weight_ih" + s).detach() for s in suffixes], dim=0)
    biases = [getattr(leaf, "bias_ih" + s, None) for s in suffixes]
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=biases[0] is not None, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(weight)
        if linear.bias is not None:
            linear.bias.copy_(torch.cat([b.detach() for b in biases if b is not None]))
    return linear


def lowers_recurrent_leaf(leaf: nn.Module, single_source: bool, carried: Optional[int]) -> bool:
    """A recurrent leaf call the lowering takes (its type registered in ``registry.CONVERSION_LOWERINGS``): batch
    first, one activation, its sequence read alone, no dropout."""
    return bool(getattr(leaf, "batch_first", False)) and single_source and carried == 0 \
        and float(getattr(leaf, "dropout", 0.0)) == 0.0
