"""The value domain's one host-op seam (fp64 twins of fp32 host modules), shared by the executor and the golden."""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import execute_compute_op_torch


def execute_host_op(op, x_flat, state_buffer, *, dtype: torch.dtype, twins: Dict[int, nn.Module]):
    """One host ComputeOp in ``dtype``; an fp64 run executes a double twin of the stored fp32 module."""
    original = op.params.get("module")
    if dtype == torch.float64 and original is not None:
        double = twins.get(int(op.id))
        if double is None:
            double = copy.deepcopy(original).double()
            twins[int(op.id)] = double
        op.params["module"] = double
        try:
            return execute_compute_op_torch(
                op, x_flat, state_buffer,
                in_scale=1.0, out_scale=1.0, output_dtype=dtype, gather_dtype=dtype,
            )
        finally:
            op.params["module"] = original
    return execute_compute_op_torch(
        op, x_flat, state_buffer,
        in_scale=1.0, out_scale=1.0, output_dtype=dtype, gather_dtype=dtype,
    )
