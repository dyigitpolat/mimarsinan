"""SSOT for the floating dtype the DEPLOYED executor runs an IR node at.

One question — "what dtype does this node compute in when deployed?" — asked
per node kind. A host ComputeOp answers with its module's own stored precision;
a NeuralCore answers with the value executor's dtype, because a crossbar's
STORED precision (int8 weights, int8 bias) is not the precision its
pre-activation accumulates in.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from mimarsinan.mapping.ir.types import ComputeOp

__all__ = [
    "NEURALCORE_DEPLOYMENT_DTYPE",
    "computeop_deployment_dtype",
    "neuralcore_deployment_dtype",
]

# ``ValueCoreFlow``/``ValueHybridCoreFlow`` default to float32 and only the
# certificate twin is built at float64 (see pipelining/core/gates/value_gates).
# The deployed executor is the float32 one.
NEURALCORE_DEPLOYMENT_DTYPE = np.float32


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


def neuralcore_deployment_dtype() -> type:
    """The numpy dtype the deployed value executor evaluates a NeuralCore at.

    A crossbar's stored weights are int8, so — unlike a host module — storage
    does NOT answer this: the pre-activation ``(x @ W + b) / theta`` accumulates
    in the value executor's dtype, and the constants flowing in are arbitrary
    upstream floats. Every seam that must evaluate a core AS DEPLOYED reads this.
    """
    return NEURALCORE_DEPLOYMENT_DTYPE
