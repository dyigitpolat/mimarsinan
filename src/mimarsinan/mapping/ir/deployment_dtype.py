"""SSOT for the floating dtype the DEPLOYED executor runs a host ComputeOp at."""

from __future__ import annotations

import itertools

import torch

from mimarsinan.mapping.ir.types import ComputeOp

__all__ = ["computeop_deployment_dtype"]


def computeop_deployment_dtype(op: ComputeOp) -> torch.dtype:
    """The dtype this op's hosted module computes in when deployed.

    The module's own parameters/buffers ARE the deployment's stored precision,
    so they answer the question whenever the module has any. A module that
    carries none is dtype-transparent — it computes in whatever dtype its
    inputs are born in, which is the process dtype the executor's gather
    buffer (``gather_inputs(dtype=None)``) and the value flow both use. Every
    seam that must evaluate the op AS DEPLOYED reads this one function.
    """
    module = (getattr(op, "params", None) or {}).get("module")
    if module is not None and hasattr(module, "parameters"):
        buffers = module.buffers() if hasattr(module, "buffers") else ()
        for tensor in itertools.chain(module.parameters(), buffers):
            if tensor.dtype.is_floating_point:
                return tensor.dtype
    return torch.get_default_dtype()
