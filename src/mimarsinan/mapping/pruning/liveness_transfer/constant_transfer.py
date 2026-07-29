"""ComputeOp forward constant transfer: ONE generic rule, executed not guessed.

The rule, for every op class alike::

    output o is CONST  iff  every input in region(o) is CONST,
    and its value is obtained by EXECUTING the op's own seam.

``region(o)`` is the W4b-1 ``LivenessTransfer`` relation when the op has one
(elementwise 1:1, index bijections, pooling receptive fields) and the WHOLE
input vector otherwise (LayerNorm, softmax, attention, residual add, unknown
modules). That single definition subsumes every "per-op refinement": an
elementwise op folds per port, a bijection relabels, a region op folds when
its whole receptive field is constant, a multi-input join folds when all
branches are constant, and an add-with-parameter folds because the parameter
lives inside the module. Zero-preservation is NOT required — this is FORWARD
constant flow, so ``sigmoid(0) = 0.5`` propagates like any other value.

Execution is guarded by four bit-exact agreement probes, because a wrong
constant is worse than no constant:

- DETERMINISM — the seam is evaluated twice and must agree;
- BATCH-SIZE INDEPENDENCE — a two-row batch of the same probe must reproduce
  the one-row result (a batch-coupled op, e.g. a batch-statistic normalizer,
  is not a per-sample constant);
- MODE INDEPENDENCE — probing happens under ``module.eval()`` (restored
  afterwards), so the as-deployed mode is probed too and must agree; a module
  left in training mode whose behaviour differs (dropout) is refused;
- DTYPE INDEPENDENCE — the fp32 module and its fp64 twin must agree, because
  the deployed program runs fp32 while the certificate runs an fp64 twin.
  This is the line that keeps the fold bit-exact in EVERY deployment: it
  admits ``sigmoid(0) = 0.5``, ``GELU(0) = 0``, ``LayerNorm(const) = bias``,
  pools, bijections and joins, and refuses e.g. ``GELU(0.3)``, whose value
  genuinely differs between fp32 and fp64 (and which the bit-exact
  certificate would refuse as grid-breaking anyway).

TOP inputs are filled with two DIFFERENT probe values; an output that depends
on a filled position (i.e. a region relation that under-reports its support)
disagrees between the probes and is refused. So an unsound transfer relation
degrades to TOP instead of to a wrong constant.
"""

from __future__ import annotations

import copy
from typing import Dict, FrozenSet, List, Sequence, Tuple

import torch

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.pruning.boundary_policy import _computeop_relays_deadness
from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
    LivenessTransfer,
)

__all__ = ["derive_constant_outputs"]

_PROBE_FILLERS = (0.0, 1.0)


def _module_of(op: ComputeOp):
    return (getattr(op, "params", None) or {}).get("module")


def _regions(
    op: ComputeOp, transfer: LivenessTransfer, n_in: int, n_out_hint: int
) -> Dict[int, FrozenSet[int]]:
    """``region(o)`` per output: the transfer relation, or all inputs."""
    if not transfer.is_opaque and transfer.out_to_ins:
        return {int(o): frozenset(r) for o, r in transfer.out_to_ins.items()}
    every = frozenset(range(n_in))
    return {o: every for o in range(n_out_hint)}


def _relay_constants(
    op: ComputeOp, in_values: Sequence[float | None]
) -> Dict[int, float]:
    """Module-less declared identity relay: output i IS input i."""
    if not _computeop_relays_deadness(op):
        return {}
    return {i: v for i, v in enumerate(in_values) if v is not None}


def _run(op: ComputeOp, probe: List[float], repeats: int, dtype: torch.dtype):
    x = torch.tensor([probe] * repeats, dtype=dtype)
    with torch.no_grad():
        return op.execute_on_gathered(x)


def _eval_mode(module):
    """Temporarily put ``module`` in eval mode; returns a restore callable."""
    was_training = bool(getattr(module, "training", False))
    if was_training and hasattr(module, "eval"):
        module.eval()
        return lambda: module.train()
    return lambda: None


def _probe(
    op: ComputeOp, module, probes: List[List[float]]
) -> "Tuple[torch.Tensor, torch.Tensor] | None":
    """Run the guarded probe battery; None means "refuse, stay TOP"."""
    original = op.params.get("module")
    try:
        double = copy.deepcopy(module).double()
    except (RuntimeError, TypeError, ValueError, AttributeError):
        return None
    restore_double = _eval_mode(double)
    restore = _eval_mode(module)
    try:
        op.params["module"] = double
        y_a = _run(op, probes[0], 1, torch.float64)
        y_b = _run(op, probes[1], 1, torch.float64)
        y_again = _run(op, probes[0], 1, torch.float64)
        y_batch = _run(op, probes[0], 2, torch.float64)
        op.params["module"] = original
        y_f32_eval = _run(op, probes[0], 1, torch.float32)
        restore()
        y_f32_asis = _run(op, probes[0], 1, torch.float32)
    except (RuntimeError, TypeError, ValueError, IndexError, KeyError,
            AttributeError):
        return None
    finally:
        op.params["module"] = original
        restore()
        restore_double()

    if y_a.shape != y_b.shape or y_a.shape != y_again.shape:
        return None
    if not torch.equal(y_a, y_again):
        return None                                   # determinism
    if y_batch.shape[0] != 2 or y_batch.shape[1:] != y_a.shape[1:]:
        return None
    if not torch.equal(y_batch[0], y_batch[1]):
        return None                                   # batch-size coupling
    if not torch.equal(y_batch[0:1], y_a):
        return None
    if y_f32_eval.shape != y_a.shape:
        return None
    if not torch.equal(y_f32_eval, y_f32_asis):
        return None                                   # mode independence
    if not torch.equal(y_f32_eval.double(), y_a):
        return None                                   # dtype independence
    return y_a, y_b


def derive_constant_outputs(
    op: ComputeOp,
    transfer: LivenessTransfer,
    in_values: Sequence[float | None],
) -> Dict[int, float]:
    """CONST outputs of one host ComputeOp (empty dict = everything stays TOP).

    Never raises: an op that cannot be probed exactly is simply not folded.
    """
    n_in = int(len(op.input_sources.flatten()))
    if n_in == 0 or len(in_values) != n_in:
        return {}
    module = _module_of(op)
    if module is None:
        return _relay_constants(op, in_values)

    known = {i: v for i, v in enumerate(in_values) if v is not None}
    if not known:
        return {}

    probes = [
        [known.get(i, filler) for i in range(n_in)] for filler in _PROBE_FILLERS
    ]
    probed = _probe(op, module, probes)
    if probed is None:
        return {}
    y_a, y_b = probed
    flat_a = y_a.flatten().tolist()
    flat_b = y_b.flatten().tolist()
    regions = _regions(op, transfer, n_in, len(flat_a))
    resolved: Dict[int, float] = {}
    for o, value in enumerate(flat_a):
        region = regions.get(o)
        if region is None or not region.issubset(known):
            continue
        if flat_b[o] != value:
            # The relation under-reported output o's support: refuse.
            continue
        resolved[o] = float(value)
    return resolved
