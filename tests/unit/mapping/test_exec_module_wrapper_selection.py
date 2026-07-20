"""[calculus §16.12] an armed wrapper OWNS output_index: the IR executor must
not re-select on top (double selection sliced the batch on attention ops)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir.types import ComputeOp, IRSource
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper


class _TupleModule(nn.Module):
    def forward(self, x):
        return x * 2.0, None


def _op(module, n=6, output_index=0):
    return ComputeOp(
        id=1, name="t",
        input_sources=np.array([IRSource(node_id=-2, index=i) for i in range(n)],
                               dtype=object),
        op_type="t",
        params={"module": module, "input_shape": (2, 3),
                "module_kwargs": {}, "output_index": output_index},
        input_shape=(2, 3), output_shape=(2, 3),
    )


def test_wrapper_owned_selection_is_not_reapplied():
    wrapped = ScaleNormalizingWrapper(
        _TupleModule(), [torch.tensor([1.0])], torch.tensor([1.0]),
        output_index=0,
    )
    op = _op(wrapped)
    x = torch.rand(4, 6)
    out = op.execute_on_gathered(x)
    assert out.shape == (4, 6)
    torch.testing.assert_close(out, x * 2.0)


def test_bare_module_selection_still_applies():
    op = _op(_TupleModule())
    x = torch.rand(4, 6)
    out = op.execute_on_gathered(x)
    assert out.shape == (4, 6)
    torch.testing.assert_close(out, x * 2.0)
