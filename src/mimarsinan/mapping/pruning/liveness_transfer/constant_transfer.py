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
- MODE INDEPENDENCE — probing happens with the WHOLE hosted subtree in eval
  (restored afterwards), so the as-deployed mode is probed too and must agree;
  a module left in training mode whose behaviour differs (dropout) is refused;
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

Probing EXECUTES the deployment's own modules, so it is held to the discipline
of a PURE analysis: running it must leave no observable trace on the process.
Two global side channels exist and both are closed here — the per-module
``training`` flag of the whole hosted subtree (captured and restored verbatim,
because ``nn.Module.train()`` recurses and would clobber a deliberately frozen
child) and the torch RNG (every probe execution runs inside
``torch.random.fork_rng``, so a stochastic host op cannot shift a seeded
experiment's downstream draws).
"""

from __future__ import annotations

import copy
import itertools
from typing import Callable, Dict, FrozenSet, List, Sequence, Tuple

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


def _eval_mode(module) -> Callable[[], None]:
    """Put the WHOLE hosted subtree in eval mode; returns an EXACT restore.

    ``nn.Module.train()``/``.eval()`` recurse into every descendant, so the
    old root-level ``module.train()`` restore silently promoted any child
    deliberately left in another mode (a frozen BatchNorm inside a hosted
    block) to training. The analysis must be pure, so the per-module flag is
    captured for the whole subtree and written back verbatim — which is what
    ``nn.Module.train`` itself does per node, minus the recursion.

    Capturing flags (O(#submodules) booleans) is chosen over probing a
    ``deepcopy`` of the live module: the fp64 twin already pays one deepcopy
    per probe, and a second one would double the cost of every LayerNorm /
    attention probe on the hot fixpoint path for no extra exactness — the
    flag snapshot restores the observable state bit-for-bit either way.
    """
    if not hasattr(module, "eval") or not hasattr(module, "modules"):
        return lambda: None
    try:
        flags = [
            (m, bool(getattr(m, "training", False))) for m in module.modules()
        ]
    except (AttributeError, TypeError):
        return lambda: None
    if not any(training for _, training in flags):
        return lambda: None       # already fully eval: nothing to touch
    module.eval()

    def _restore() -> None:
        for submodule, training in flags:
            submodule.training = training

    return _restore


def _rng_devices(module) -> List[int]:
    """CUDA device indices this module's tensors live on (empty = CPU only)."""
    devices: List[int] = []
    try:
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            device = getattr(tensor, "device", None)
            if device is None or device.type != "cuda":
                continue
            index = device.index
            if index is None:
                index = torch.cuda.current_device()
            if int(index) not in devices:
                devices.append(int(index))
    except (AttributeError, TypeError, RuntimeError):
        return devices
    return devices


def _probe(
    op: ComputeOp, module, probes: List[List[float]]
) -> "Tuple[torch.Tensor, torch.Tensor] | None":
    """Run the guarded probe battery; None means "refuse, stay TOP".

    The whole battery runs inside ``fork_rng`` so that a stochastic host op
    (dropout, a sampling block, a module that draws during ``forward``) cannot
    move the global generator: the analysis must not perturb a seeded run's
    downstream draws. The fork spans the ORIGINAL-module executions too, not
    just the fp64 copies.
    """
    with torch.random.fork_rng(devices=_rng_devices(module), enabled=True):
        return _probe_isolated(op, module, probes)


def _probe_isolated(
    op: ComputeOp, module, probes: List[List[float]]
) -> "Tuple[torch.Tensor, torch.Tensor] | None":
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
