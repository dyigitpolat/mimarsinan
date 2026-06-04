"""Host ComputeOp execution must run in the module's own floating dtype."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    execute_compute_op_torch,
)


def _op(module, size):
    sources = np.array(
        [IRSource(node_id=7, index=i) for i in range(size)], dtype=object,
    ).reshape(1, size)
    return ComputeOp(
        id=1, name="op", op_type=type(module).__name__,
        input_sources=sources, params={"module": module},
        input_shape=(size,), output_shape=(size,),
    )


def test_double_module_executes_without_dtype_mismatch():
    module = nn.LayerNorm([3]).double()
    op = _op(module, 3)
    buf = {7: torch.tensor([[0.1, -0.4, 0.3]], dtype=torch.float64)}
    out = execute_compute_op_torch(op, torch.zeros(1, 3, dtype=torch.float64), buf)
    assert out.shape == (1, 3)
    assert out.dtype == torch.float64


def test_float32_module_keeps_float32_execution():
    module = nn.LayerNorm([3])
    op = _op(module, 3)
    buf = {7: torch.tensor([[0.1, -0.4, 0.3]], dtype=torch.float64)}
    out = execute_compute_op_torch(op, torch.zeros(1, 3, dtype=torch.float64), buf)
    assert out.dtype == torch.float32
