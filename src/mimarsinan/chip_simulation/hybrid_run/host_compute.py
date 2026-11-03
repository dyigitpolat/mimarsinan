"""Host ComputeOp execution — the ONE evaluator every side of a comparison shares.

A hosted op (a subsumed encoding layer, a normalization the chip does not carry)
runs on the host, and it ends in a staircase: LIF counts, quantized activations.
The step is ``theta/T``, so a pre-activation sitting within float reduction noise
of a step edge resolves to a DIFFERENT step under a different evaluator — and one
step is exactly one spike once the segment boundary transcodes value to counts.
That is why ``device`` is a required argument here and why the re-deriving backends
evaluate on the deployed host (``SpikingDeploymentContract.host_compute_device``):
the census/HCM reference and every backend that re-derives the op must land on the
same step by construction.

[U-DAR] The arithmetic is the record's (``mapping/ir/deployment_dtype.py``), named
by the caller's declared ``arithmetic``: the module the op runs (the value domain's
float64 twin, a spiking domain's stored module), the gather buffer's dtype, the
kernel's dtype and the tie-rule plane. ``evaluate_gathered_host_op`` is the evaluator
after the gather; the constant fold's probe calls it too (``analysis=True``), so the
value the fold derives for a host op is the one every deployed executor computes.

The deployed TIE RULE is decided here, for every executor: a hosted module
runs in the chip-lattice plane (the record's ``HOST_KERNELS`` plane, ``measurement_plane``),
so an armed lattice node (a host-run perceptron of a weight-quantized program, e.g. the
subsumed encoding layer) decides an exact membrane tie by the exact value, as the chip's
integer cores do every cycle, whatever plane its caller has opened (w2-parity fix round:
the C5 T-LIF metric flow read its stem by float dust outside the plane).
"""

from __future__ import annotations

import contextlib
import itertools
from typing import Any, Dict, Iterator

import numpy as np
import numpy.typing as npt
import torch

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.ir.deployment_dtype import (
    DEPLOYED_HOST_DEVICE,
    HOST_KERNELS,
    SPIKING_ARITHMETIC,
    DeployedArithmetic,
    dtype_name,
)
from mimarsinan.mapping.ir.gather_plan import gather_plan_for


@contextlib.contextmanager
def _deployed_kernel(op: ComputeOp, arithmetic: DeployedArithmetic,
                     twins: "Dict[int, Any] | None") -> Iterator[None]:
    """``op`` runs the module the record names for the duration (its twin in the value domain), then its own again."""
    params = op.params if op.params is not None else {}
    original = params.get("module")
    kernel = arithmetic.host_kernel_module(op, twins if twins is not None else {})
    if kernel is original:
        yield
        return
    params["module"] = kernel
    try:
        yield
    finally:
        params["module"] = original


def evaluate_gathered_host_op(
    op: ComputeOp,
    gathered: torch.Tensor,
    *,
    arithmetic: DeployedArithmetic,
    in_scale: float = 1.0,
    out_scale: float | None = None,
    output_dtype: torch.dtype | None = None,
    twins: "Dict[int, Any] | None" = None,
    analysis: bool = False,
) -> torch.Tensor:
    """The deployed host evaluator after the gather: the record's kernel module and dtype, the scales, the record's
    tie-rule plane. ``analysis`` runs the op's analysis seam (``probe_on_gathered``: placement restored, no host-kernel
    replay) instead of the deployment seam; the arithmetic is the same."""
    if out_scale is None:
        out_scale = in_scale
    with _deployed_kernel(op, arithmetic, twins):
        gathered = gathered.to(arithmetic.host_kernel_dtype(op))
        if abs(in_scale - 1.0) > 1e-9:
            gathered = gathered * in_scale
        # The deployed tie rule (module docstring): the plane is the evaluator's, not the caller's.
        with arithmetic.plane(HOST_KERNELS):
            result = op.probe_on_gathered(gathered) if analysis else op.execute_on_gathered(gathered)
    if abs(out_scale - 1.0) > 1e-9:
        result = result / out_scale
    if output_dtype is not None and result.dtype != output_dtype:
        result = result.to(output_dtype)
    return result


def execute_compute_op_torch(
    op: ComputeOp,
    original_input: torch.Tensor,
    state_buffer: Dict[int, torch.Tensor],
    *,
    arithmetic: DeployedArithmetic,
    in_scale: float = 1.0,
    out_scale: float | None = None,
    output_dtype: torch.dtype | None = None,
    twins: "Dict[int, Any] | None" = None,
) -> torch.Tensor:
    """Execute a host-side ComputeOp in the declared ``arithmetic``: gathered at its host-gather dtype, then
    ``evaluate_gathered_host_op``. ``twins`` caches the value domain's float64 twins across calls (the caller's
    executor)."""
    gathered = op.gather_inputs(original_input, state_buffer, dtype=arithmetic.host_gather_dtype)
    return evaluate_gathered_host_op(op, gathered, arithmetic=arithmetic, in_scale=in_scale, out_scale=out_scale,
                                     output_dtype=output_dtype, twins=twins)


def _module_home_device(module) -> "torch.device | None":
    """The device a hosted module lives on, or None when it holds no tensors.

    A module borrowed for one evaluation must be handed back where it was
    found: an executor that evaluates it again for the next sample finds it
    where it left it.
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
    dtype: npt.DTypeLike,
    in_scale: float = 1.0,
    out_scale: float | None = None,
) -> np.ndarray:
    """Execute a spiking program's ComputeOp via the torch evaluator (``SPIKING_ARITHMETIC``) for a numpy backend.

    ``dtype`` (declared, never defaulted) is the dtype the backend's buffers enter and the result leaves in; the
    kernel's inputs are gathered at the record's host-gather dtype whatever it is, so float32 and float64 here carry
    the same values (every input rounds once at the gather). ``device`` is REQUIRED — it decides staircase ties, so
    no caller may leave it to a default. A hosted op ends in a quantized/LIF staircase whose step is ``theta/T``; a
    pre-activation within f32 reduction noise of a step edge lands on opposite steps under different reduction orders,
    and one step IS one spike after the segment-boundary transcode (measured: MNIST simple_mlp seed 3, hop0 element 5,
    pre-activation edge distance 1.8e-8 -> one spike). Every backend that RE-DERIVES a host op therefore evaluates it
    on the deployed host, as the census/HCM flow does (``contract.host_compute_device``); pass ``None`` only where BOTH
    sides of the comparison are this CPU evaluator."""
    torch_dtype = getattr(torch, dtype_name(dtype))
    dev = torch.device(device) if device is not None else torch.device(DEPLOYED_HOST_DEVICE)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device(DEPLOYED_HOST_DEVICE)
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
            op, x_torch, buffers_torch, arithmetic=SPIKING_ARITHMETIC,
            in_scale=in_scale, out_scale=out_scale, output_dtype=torch_dtype,
        )
    finally:
        if moved is not None and home is not None:
            moved.to(home)
    return result.detach().cpu().numpy()
