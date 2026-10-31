"""[U-PS] The declared host-kernel evaluation of a deployed program: on the batch, or one sample at a time.

Theorem 1's host condition asks for deterministic, batch-invariant host kernels on one device and dtype. fp32 host
convolutions and matmuls are batch-dependent at the ulp level (the reduction order follows the batch size), so on their
batch a sample's host values depend on the samples beside it. ``host_kernel_evaluation = per_sample`` makes the program
evaluate every host kernel one sample at a time: a loop over the batch, each sample on its own copy, the results
concatenated in order. A sample's host values are then a function of that sample alone, by construction.

The program carries its mode: ``declare_host_kernel_evaluation`` stamps it on every host op's ``params``, and the one
host seam (``ComputeOp`` in ``mapping/ir/types.py``) reads it through ``evaluate_host_kernel``, so every executor that
evaluates the program (identity and packed flows, the value run, every backend that re-derives a host op, a program
loaded from its pickle) follows it alike. ``batched`` or the key absent stamps nothing: every program is the base's,
byte for byte. A kernel that cannot run on one sample alone is refused by name. The NF twin follows the program's
mode wherever a parity gate compares it with the deployed program (``pipelining/core/nf_scm_parity.py``)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal

HOST_KERNEL_EVALUATION_KEY = "host_kernel_evaluation"
BATCHED, PER_SAMPLE = "batched", "per_sample"
HOST_KERNEL_EVALUATIONS = (BATCHED, PER_SAMPLE)

#: What a host kernel run on one sample alone may raise when it is no per-sample function (memory and interrupts
#: propagate as themselves; a named refusal of the kernel's own propagates unchanged).
_ONE_SAMPLE_ERRORS = (RuntimeError, ValueError, TypeError, IndexError, AssertionError, NotImplementedError)


class PerSampleHostKernelRefusal(ToolchainRefusal, RuntimeError):
    """A host kernel the program declares per sample that cannot be evaluated on one sample alone."""


def declared_evaluation(op: Any) -> str:
    """``per_sample`` where the host op carries that declaration, else ``batched``."""
    params = getattr(op, "params", None) or {}
    return PER_SAMPLE if params.get(HOST_KERNEL_EVALUATION_KEY) == PER_SAMPLE else BATCHED


def declare_host_kernel_evaluation(ir_graph: Any, mode: Any) -> int:
    """Stamp ``mode`` on every host op of ``ir_graph``; the number stamped. ``batched`` or None (the key absent)
    stamps nothing, so the program stays byte for byte the base's; a value outside the vocabulary is refused."""
    if mode is None or mode == BATCHED:
        return 0
    if mode != PER_SAMPLE:
        raise ValueError(f"{HOST_KERNEL_EVALUATION_KEY} must be one of {HOST_KERNEL_EVALUATIONS}, got {mode!r}")
    ops = list(ir_graph.get_compute_ops())
    for op in ops:
        op.params[HOST_KERNEL_EVALUATION_KEY] = PER_SAMPLE
    return len(ops)


def _host_ops(program: Any) -> list:
    """The host ops of an IR graph (``get_compute_ops``) or of a packed program (its stages' ``compute_op``)."""
    if hasattr(program, "get_compute_ops"):
        return list(program.get_compute_ops())
    stages = getattr(program, "stages", None) or ()
    return [st.compute_op for st in stages if getattr(st, "compute_op", None) is not None]


def program_evaluates_per_sample(program: Any) -> bool:
    """Whether the deployed program (an IR graph or a packed program) evaluates its host kernels one sample at a
    time: the mode the NF twin of a parity gate follows."""
    return any(declared_evaluation(op) == PER_SAMPLE for op in _host_ops(program))


def evaluate_host_kernel(op: Any, x: torch.Tensor, kernel: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """``kernel`` on ``x`` as ``op`` declares it: on the batch, or under ``per_sample`` on every sample alone (its
    own copy, so every evaluation sees a fresh buffer whatever its row's place in the batch), concatenated in order.
    An empty batch is no sample and runs as given."""
    if declared_evaluation(op) != PER_SAMPLE or x.dim() == 0 or int(x.shape[0]) == 0:
        return kernel(x)
    rows = []
    for i in range(int(x.shape[0])):
        try:
            out = kernel(x[i:i + 1].clone())
        except ToolchainRefusal:
            raise
        except _ONE_SAMPLE_ERRORS as exc:
            raise PerSampleHostKernelRefusal(
                f"host kernel {getattr(op, 'name', '?')!r} ({getattr(op, 'op_type', 'host')}) cannot be evaluated "
                f"on one sample alone, which {HOST_KERNEL_EVALUATION_KEY}={PER_SAMPLE!r} requires of every host "
                f"kernel of the program: {type(exc).__name__}: {str(exc)[:200]}") from exc
        if out.dim() == 0 or int(out.shape[0]) != 1:
            raise PerSampleHostKernelRefusal(
                f"host kernel {getattr(op, 'name', '?')!r} ({getattr(op, 'op_type', 'host')}) gave "
                f"{tuple(out.shape)} for one sample: no per-sample function, which "
                f"{HOST_KERNEL_EVALUATION_KEY}={PER_SAMPLE!r} requires of every host kernel of the program")
        rows.append(out)
    return torch.cat(rows)


def forward_of(model: Any, samples: torch.Tensor) -> torch.Tensor:
    """The twin's plain forward, as a capture for ``twin_as_program``."""
    return model(samples)


def _concatenated(parts: list) -> Any:
    first = parts[0]
    if isinstance(first, dict):
        return {key: _concatenated([part[key] for part in parts]) for key in first}
    if isinstance(first, torch.Tensor):
        return torch.cat(parts)
    return np.concatenate(parts, axis=0)


def twin_as_program(program: Any, capture: Callable[[Any, torch.Tensor], Any], model: Any,
                    samples: torch.Tensor) -> Any:
    """The NF twin's ``capture(model, samples)`` as the deployed ``program`` evaluates its host kernels: on the batch,
    or where it declares ``per_sample`` one sample at a time, the captures (per-key arrays or tensors, or an output
    tensor) concatenated in order. Every kernel of the twin, its host ones included, then sees each sample alone, as
    the program's host kernels do (each on its own copy), so a parity gate compares like with like at atol 0."""
    n = int(samples.shape[0]) if samples.dim() else 0
    if n == 0 or program is None or not program_evaluates_per_sample(program):
        return capture(model, samples)
    return _concatenated([capture(model, samples[i:i + 1].clone()) for i in range(n)])
