"""Host ComputeOp execution — the ONE evaluator every side of a comparison shares.

A hosted op (a subsumed encoding layer, a normalization the chip does not carry)
runs on the host, and it ends in a staircase: LIF counts, quantized activations.
The step is ``theta/T``, so a pre-activation sitting within float reduction noise
of a step edge resolves to a DIFFERENT step under a different evaluator — and one
step is exactly one spike once the segment boundary transcodes value to counts.
That is why ``device`` is a required argument here and why it rides on
``SpikingDeploymentContract``: the census/HCM reference and every backend that
re-derives the op must land on the same step by construction.
"""

from __future__ import annotations

import itertools
from typing import Dict

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.mapping.ir import ComputeOp, computeop_deployment_dtype
from mimarsinan.mapping.ir.gather_plan import gather_plan_for


def execute_compute_op_torch(
    op: ComputeOp,
    original_input: torch.Tensor,
    state_buffer: Dict[int, torch.Tensor],
    *,
    in_scale: float = 1.0,
    out_scale: float | None = None,
    output_dtype: torch.dtype | None = None,
    gather_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute a host-side ComputeOp; optional in/out activation scales.

    ``gather_dtype=None`` keeps the historical default-dtype gather buffer;
    the value-domain fp64 path passes an explicit dtype so no fp32 round
    enters between stages.
    """
    if out_scale is None:
        out_scale = in_scale

    gathered = op.gather_inputs(original_input, state_buffer, dtype=gather_dtype)
    if gather_dtype is None:
        gathered = gathered.to(computeop_deployment_dtype(op))
    if abs(in_scale - 1.0) > 1e-9:
        gathered = gathered * in_scale

    result = op.execute_on_gathered(gathered)

    if abs(out_scale - 1.0) > 1e-9:
        result = result / out_scale
    if output_dtype is not None and result.dtype != output_dtype:
        result = result.to(output_dtype)
    return result


def _module_home_device(module) -> "torch.device | None":
    """The device a hosted module lives on, or None when it holds no tensors.

    A module borrowed for one evaluation must be handed back where it was
    found: the census flow keeps its ComputeOp modules on the pipeline device
    and evaluates them again for the next sample.
    """
    if module is None:
        return None
    tensors = itertools.chain(
        module.parameters() if hasattr(module, "parameters") else (),
        module.buffers() if hasattr(module, "buffers") else (),
    )
    for tensor in tensors:
        return tensor.device
    return None


def execute_compute_op_numpy(
    op: ComputeOp,
    original_input: np.ndarray,
    state_buffer: Dict[int, np.ndarray],
    *,
    device: "str | torch.device | None",
    in_scale: float = 1.0,
    out_scale: float | None = None,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Execute ComputeOp via torch wrapper; ``dtype=np.float64`` for HCM parity.

    ``device`` is REQUIRED — it decides staircase ties, so no caller may leave
    it to a default. A hosted op ends in a quantized/LIF staircase whose step
    is ``theta/T``; a pre-activation within f32 reduction noise of a step edge
    lands on opposite steps under different reduction orders, and one step IS
    one spike after the segment-boundary transcode (measured: MNIST simple_mlp
    seed 3, hop0 element 5, pre-activation edge distance 1.8e-8 -> one spike).
    Every backend that RE-DERIVES a host op therefore evaluates it on the SAME
    device as the census/HCM flow (``contract.host_compute_device``); pass
    ``None`` only where BOTH sides of the comparison are this CPU evaluator."""
    if out_scale is None:
        out_scale = in_scale

    torch_dtype = (torch.float64 if np.dtype(dtype) == np.float64
                   else torch.float32)
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    x_torch = torch.tensor(original_input, dtype=torch_dtype, device=dev)
    # Convert only the producer buffers this op's gather actually reads.
    referenced = gather_plan_for(op).referenced_node_ids
    buffers_torch = {
        k: torch.tensor(state_buffer[k], dtype=torch_dtype, device=dev)
        for k in referenced
    }
    module = (op.params or {}).get("module") if op.params else None
    moved = module if (module is not None and hasattr(module, "to")) else None
    home = _module_home_device(moved)
    if moved is not None:
        moved.to(dev)
    try:
        result = execute_compute_op_torch(
            op, x_torch, buffers_torch,
            in_scale=in_scale, out_scale=out_scale, output_dtype=torch_dtype,
        )
    finally:
        if moved is not None and home is not None:
            moved.to(home)
    return result.detach().cpu().numpy()
