"""[calculus §16.12] gather over batch-mismatched buffers must FAIL LOUD —
a silent broadcast masked the attention double-selection defect (a (197,768)
single-sample output smeared across batch 256)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir.gather_plan import build_gather_plan
from mimarsinan.mapping.ir.source import IRSource


class _FakeNode:
    def __init__(self, sources):
        self.input_sources = np.array(sources, dtype=object)


def test_batch_mismatched_buffer_fails_loud():
    sources = [IRSource(node_id=4, index=i) for i in range(6)]
    plan = build_gather_plan(_FakeNode(sources))
    bad = torch.arange(6.0).reshape(2, 3)  # rows != batch
    x = torch.zeros(5, 4)
    with pytest.raises(ValueError, match="batch"):
        plan.gather(x, {4: bad})


def test_batched_buffers_stay_untouched():
    sources = [IRSource(node_id=7, index=i) for i in range(3)]
    plan = build_gather_plan(_FakeNode(sources))
    buf = torch.rand(4, 3)
    x = torch.zeros(4, 2)
    out = plan.gather(x, {7: buf})
    torch.testing.assert_close(out, buf)
