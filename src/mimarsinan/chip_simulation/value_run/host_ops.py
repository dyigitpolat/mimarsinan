"""The value domain's one host-op seam, shared by the executor and the golden: [U-DAR] the record's value arithmetic
(``VALUE_ARITHMETIC``: the stored module's float64 twin, inputs gathered at float64, the measurement plane), or a
declared counterfactual value executor's (the H8 arm's float32: the stored module at that dtype)."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import execute_compute_op_torch
from mimarsinan.mapping.ir.deployment_dtype import VALUE_ARITHMETIC


def execute_host_op(op, x_flat, state_buffer, *, dtype: torch.dtype, twins: Dict[int, nn.Module]):
    """One host ComputeOp of a value program in ``dtype``: the record's arithmetic at its core dtype (``twins`` caches
    the float64 twins across calls), else the declared counterfactual at ``dtype``."""
    arithmetic = VALUE_ARITHMETIC if dtype == VALUE_ARITHMETIC.core_dtype else VALUE_ARITHMETIC.counterfactual(dtype)
    return execute_compute_op_torch(
        op, x_flat, state_buffer, arithmetic=arithmetic,
        in_scale=1.0, out_scale=1.0, output_dtype=dtype, twins=twins,
    )
