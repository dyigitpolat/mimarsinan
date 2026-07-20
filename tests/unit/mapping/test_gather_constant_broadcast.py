"""[calculus §16.11] gather over batch-free constant buffers: a parameter
node's buffer (stored without a batch row, e.g. pos-embed (197,768)) must
flatten to (1, features) and broadcast across the batch — the armed-add
first-contact defect on the ViT IR."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir.gather_plan import build_gather_plan
from mimarsinan.mapping.ir.types import IRSource


class _FakeNode:
    def __init__(self, sources):
        self.input_sources = np.array(sources, dtype=object)


def test_batch_free_constant_buffer_broadcasts_across_batch():
    sources = [IRSource(node_id=4, index=i) for i in range(6)]
    plan = build_gather_plan(_FakeNode(sources))
    const = torch.arange(6.0).reshape(2, 3)  # batch-free (2,3) constant
    x = torch.zeros(5, 4)  # batch = 5
    out = plan.gather(x, {4: const})
    assert out.shape == (5, 6)
    expected = torch.arange(6.0).unsqueeze(0).expand(5, 6)
    torch.testing.assert_close(out, expected)


def test_batched_buffers_stay_untouched():
    sources = [IRSource(node_id=7, index=i) for i in range(3)]
    plan = build_gather_plan(_FakeNode(sources))
    buf = torch.rand(4, 3)
    x = torch.zeros(4, 2)
    out = plan.gather(x, {7: buf})
    torch.testing.assert_close(out, buf)
