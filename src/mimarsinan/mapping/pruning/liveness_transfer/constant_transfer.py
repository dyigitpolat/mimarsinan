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

Probing runs AT THE DEPLOYMENT DTYPE (``computeop_deployment_dtype``), so the
derived constant is by construction the value the deployed program puts on
that line — there is nothing left to compare across precisions. The value is
then either ON the dyadic grid (grid-certifiable: the cascade certificate can
back it bit-exactly) or not (execution-exact only: the certificate REFUSES the
instance, the documented ``GELU(c != 0)`` pattern, counted on the elimination
ledger rather than hidden). Constants are never snapped onto the grid.

Execution is guarded by three bit-exact agreement probes plus a carriability
gate, because a wrong constant is worse than no constant:

- DETERMINISM — the seam is evaluated twice and must agree;
- BATCH-SIZE INDEPENDENCE — a two-row batch of the same probe must reproduce
  the one-row result; this is what refuses a ``Linear`` (a GEMM reassociates
  its sum differently from the batch-1 GEMV) and any batch-coupled op;
- MODE INDEPENDENCE — probing happens with the WHOLE hosted subtree in eval
  (restored afterwards), so the as-deployed mode is probed too and must agree;
  a module left in training mode whose behaviour differs (dropout) is refused;
- DEPLOYMENT-DTYPE CARRIABILITY — every incoming constant must be exactly
  representable at the deployment dtype. A line the deployment cannot carry
  never held that value, so probing with the ROUNDED value would derive a
  constant for a line that does not exist; refuse instead.

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

import itertools
import time
from typing import Callable, Dict, FrozenSet, List, Sequence, Tuple

import torch

from mimarsinan.mapping.ir import ComputeOp, computeop_deployment_dtype
from mimarsinan.mapping.pruning.boundary_policy import _computeop_relays_deadness
from mimarsinan.mapping.pruning.liveness_transfer.transfer_types import (
    LivenessTransfer,
)

__all__ = ["derive_constant_outputs"]

_PROBE_FILLERS = (0.0, 1.0)


def _module_of(op: ComputeOp):
    return (getattr(op, "params", None) or {}).get("module")


def _any_region_satisfiable(transfer: LivenessTransfer, known, n_in: int) -> bool:
    """Mirrors ``_regions``: no fully-known region == the identical {}, unexecuted."""
    if transfer.is_opaque or not transfer.out_to_ins:
        return len(known) == n_in
    return any(r.issubset(known) for r in transfer.out_to_ins.values())


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
        return op.probe_on_gathered(x)


def _carriable(values: Sequence[float], dtype: torch.dtype) -> bool:
    """Is every constant exactly representable at the deployment dtype?"""
    return all(
        float(torch.tensor(v, dtype=dtype)) == float(v) for v in values
    )


# Every guard on the probe path shares this tuple: a hosted module is arbitrary
# user code, so probing it degrades to "stay TOP" for the failure classes a
# module can plausibly raise. Anything outside it still propagates (fail loud).
_PROBE_FAILURES = (RuntimeError, TypeError, ValueError, IndexError, KeyError,
                   AttributeError)


def _deployment_dtype(op: ComputeOp) -> "torch.dtype | None":
    """The op's deployment dtype; None when the hosted module refuses to say.

    Resolution is never DEFAULTED here: an arbitrary hosted module may raise
    from ``parameters()``, and an unresolved dtype means the analysis cannot
    say what the deployment computes, so it refuses. The deployed executor,
    which must fail loud instead, calls the SSOT directly.
    """
    try:
        return computeop_deployment_dtype(op)
    except _PROBE_FAILURES:
        return None


def _eval_mode(module) -> Callable[[], None]:
    """Put the WHOLE hosted subtree in eval mode; returns an EXACT restore.

    ``nn.Module.train()``/``.eval()`` recurse into every descendant, so the
    old root-level ``module.train()`` restore silently promoted any child
    deliberately left in another mode (a frozen BatchNorm inside a hosted
    block) to training. The analysis must be pure, so the per-module flag is
    captured for the whole subtree and written back verbatim — which is what
    ``nn.Module.train`` itself does per node, minus the recursion.

    Capturing flags (O(#submodules) booleans) is chosen over probing a
    ``deepcopy`` of the live module: a copy per probe would cost a deep clone
    of every LayerNorm / attention host on the hot fixpoint path for no extra
    exactness — the flag snapshot restores the observable state bit-for-bit.
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
    op: ComputeOp, module, probes: List[List[float]], dtype: torch.dtype
) -> "Tuple[torch.Tensor, torch.Tensor] | None":
    """Run the guarded probe battery; None means "refuse, stay TOP".

    The whole battery runs inside ``fork_rng`` so that a stochastic host op
    (dropout, a sampling block, a module that draws during ``forward``) cannot
    move the global generator: the analysis must not perturb a seeded run's
    downstream draws.
    """
    try:
        devices = _rng_devices(module)
    except _PROBE_FAILURES:
        return None
    with torch.random.fork_rng(devices=devices, enabled=True):
        return _probe_isolated(op, module, probes, dtype)


def _probe_isolated(
    op: ComputeOp, module, probes: List[List[float]], dtype: torch.dtype
) -> "Tuple[torch.Tensor, torch.Tensor] | None":
    restore: Callable[[], None] = lambda: None
    try:
        restore = _eval_mode(module)
        y_a = _run(op, probes[0], 1, dtype)
        y_b = _run(op, probes[1], 1, dtype)
        y_again = _run(op, probes[0], 1, dtype)
        y_batch = _run(op, probes[0], 2, dtype)
        restore()
        y_as_deployed = _run(op, probes[0], 1, dtype)
    except _PROBE_FAILURES:
        return None
    finally:
        restore()

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
    if y_as_deployed.shape != y_a.shape:
        return None
    if not torch.equal(y_as_deployed, y_a):
        return None                                   # mode independence
    return y_a, y_b


def derive_constant_outputs(op, transfer, in_values) -> Dict[int, float]:
    t0 = time.perf_counter()
    result = _derive_constant_outputs(op, transfer, in_values)
    dt = time.perf_counter() - t0
    if dt > 1.0:   # slow probes only: the real-scale evidence line
        print(f"[OpProbe] {getattr(op, 'name', op.id)} wall={dt:.1f}s "
              f"known={sum(v is not None for v in in_values)}/{len(in_values)} "
              f"resolved={len(result)}", flush=True)
    return result


def _derive_constant_outputs(
    op: ComputeOp,
    transfer: LivenessTransfer,
    in_values: Sequence[float | None],
) -> Dict[int, float]:
    """CONST outputs of one host ComputeOp (empty dict = everything stays TOP).

    Degrades rather than raising for every failure class a hosted module can
    plausibly raise (``_PROBE_FAILURES``): such an op is simply not folded.
    An exception outside that set still propagates, by fail-loud discipline.
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
    dtype = _deployment_dtype(op)
    if dtype is None or not _carriable(list(known.values()), dtype):
        return {}
    if not _any_region_satisfiable(transfer, known, n_in):
        return {}

    probes = [
        [known.get(i, filler) for i in range(n_in)] for filler in _PROBE_FILLERS
    ]
    probed = _probe(op, module, probes, dtype)
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
