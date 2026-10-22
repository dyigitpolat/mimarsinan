"""A Linear on a non-leading axis whose per-sample input does not flatten to its fan-in (review round 13).

A ConvNeXt block permutes to channels-last, then runs LayerNorm, Linear, GELU, Linear and a layer scale on the last
axis of a ``(H, W, C)`` per-sample tensor. Stage 1 is exact; the IR lowering of its value program ended in an anonymous
``ValueError`` ("could not broadcast input array from shape (8,32) into shape (128,32)", ``ir_mapping_class_emit.py``)
on this tree and on w0-base: the FC mapping reads a per-sample input of rank 1 or 2 only. It is refused by name now
(``PerceptronFanInError``, ``perceptron_mapper.py``), naming the Linear's FX node and the shapes.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.norm(y)))) * self.gamma
        return x + y.permute(0, 3, 1, 2)


class TinyConvNeXt(nn.Module):
    """The round-12 reviewer's ``TinyConvNeXt`` (``r12_families.py``)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 2, 2)
        self.b = ConvNeXtBlock()
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        return self.head(self.b(self.stem(x)).mean([-2, -1]))


def test_the_value_program_of_a_channels_last_linear_is_refused_by_name():
    torch.manual_seed(0)
    model = TinyConvNeXt().double().eval()
    with pytest.raises(ToolchainRefusal) as excinfo, contextlib.redirect_stdout(io.StringIO()):
        convert_torch_model(model, (1, 8, 8), 10, packaging=MVM_PACKAGING, value_program_stage=True)
    message = str(excinfo.value)
    assert "identity value program" in message and "PerceptronFanInError" in message, message
    assert "Linear (FX node 'b_pw1'): its per-sample input of shape (4, 4, 8) does not flatten to its fan-in 8" in (
        message), message


def test_the_fc_mapping_refuses_a_rank_3_source_by_name_and_keeps_a_flattening_one():
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronFanInError, fan_in_refusal

    assert issubclass(PerceptronFanInError, ToolchainRefusal) and issubclass(PerceptronFanInError, ValueError)
    sources = np.empty((8, 4, 4), dtype=object)  # the transposed (4, 4, 8) per-sample input of a Linear(8, 32)
    refusal = fan_in_refusal(sources, 8, "b_pw1")
    assert isinstance(refusal, PerceptronFanInError) and refusal.node_name == "b_pw1"
    assert fan_in_refusal(np.empty((8, 1, 1), dtype=object), 8, "fc") is None  # flattens to the fan-in
    assert fan_in_refusal(np.empty((8,), dtype=object), 8, "fc") is None
    assert fan_in_refusal(np.empty((8, 5), dtype=object), 8, "fc") is None  # a rank-2 source: columns
    assert isinstance(fan_in_refusal(np.empty((6,), dtype=object), 8, "fc"), PerceptronFanInError)
