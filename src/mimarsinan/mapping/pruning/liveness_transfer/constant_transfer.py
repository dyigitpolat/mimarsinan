"""ComputeOp forward constant transfer: ONE generic rule, executed not guessed.

The rule, for every op class alike: output ``o`` is CONST iff every input in
``region(o)`` is CONST, its value obtained by EXECUTING the op's own seam.

``region(o)`` is the W4b-1 ``LivenessTransfer`` relation when the op has one
(elementwise 1:1, index bijections, pooling receptive fields) and the WHOLE
input vector otherwise (LayerNorm, softmax, attention, residual add, unknown
modules); that single definition subsumes every "per-op refinement".
Zero-preservation is NOT required — this is FORWARD constant flow, so
``sigmoid(0) = 0.5`` propagates like any other value.

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
- BATCH-SIZE INDEPENDENCE — a two-row batch must reproduce the one-row
  result; refuses ``Linear`` (GEMM vs GEMV reassociation) and batch coupling;
- MODE INDEPENDENCE — probed with the whole hosted subtree in eval (restored
  afterwards) AND as deployed; differing behaviour (dropout) is refused;
- DEPLOYMENT-DTYPE CARRIABILITY — every incoming constant must be exactly
  representable at the deployment dtype; a line the deployment cannot carry
  never held that value, so refuse rather than probe the rounded value.

TOP inputs are filled with two DIFFERENT probe values; an output that depends
on a filled position (i.e. a region relation that under-reports its support)
disagrees between the probes and is refused. So an unsound transfer relation
degrades to TOP instead of to a wrong constant.

Probing EXECUTES the deployment's own modules, so it is held to a PURE
analysis's discipline: no observable trace on the process. Both global side
channels are closed — per-module ``training`` flags (captured and restored
verbatim; ``train()`` recurses) and the torch RNG (``fork_rng`` around every
execution, so a stochastic host op cannot shift a seeded run's draws).
"""

from __future__ import annotations

import itertools
import time
import weakref
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
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


# id(transfer) -> (weakref, (outs, offsets, ins)); frozen transfers hash by
# their (unhashable) mapping fields, so identity + a liveness check stands in.
_CSR_CACHE: Dict[int, tuple] = {}


def _transfer_csr(transfer: LivenessTransfer):
    """``out_to_ins`` lowered once per transfer to CSR arrays."""
    hit = _CSR_CACHE.get(id(transfer))
    if hit is not None and hit[0]() is transfer:
        return hit[1]
    rel = transfer.out_to_ins
    outs = np.fromiter(rel.keys(), dtype=np.int64, count=len(rel))
    sizes = np.fromiter((len(r) for r in rel.values()), dtype=np.int64,
                        count=len(rel))
    offsets = np.zeros(outs.size + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    ins = np.fromiter((i for r in rel.values() for i in r), dtype=np.int64,
                      count=int(offsets[-1]))
    csr = (outs, offsets, ins)
    _CSR_CACHE[id(transfer)] = (weakref.ref(transfer), csr)
    return csr


def _relay_constants(
    op: ComputeOp, in_values: Sequence[float | None]
) -> Dict[int, float]:
    """Module-less declared identity relay: output i IS input i."""
    if not _computeop_relays_deadness(op):
        return {}
    return {i: v for i, v in enumerate(in_values) if v is not None}


def _run(op: ComputeOp, probe, repeats: int, dtype: torch.dtype):
    # from_numpy->to(dtype) rounds double->dtype exactly like the former
    # per-element torch.tensor(list) path; pinned by the frozen-oracle gate.
    base = np.asarray(probe, dtype=np.float64)
    x = torch.from_numpy(np.repeat(base[None, :], repeats, axis=0)).to(dtype)
    with torch.no_grad():
        return op.probe_on_gathered(x)


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

    ``train()``/``.eval()`` recurse, so a root-level restore would promote a
    deliberately-frozen child (e.g. an eval BatchNorm) to training; the
    per-module flag is captured for the whole subtree and written back
    verbatim. Chosen over probing a ``deepcopy``: the flag snapshot restores
    observable state bit-for-bit without cloning attention hosts per probe.
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
    op: ComputeOp, module, probes: Sequence, dtype: torch.dtype
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
    op: ComputeOp, module, probes: Sequence, dtype: torch.dtype
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

    known_mask = np.fromiter(
        (v is not None for v in in_values), dtype=bool, count=n_in
    )
    n_known = int(known_mask.sum())
    if n_known == 0:
        return {}
    vals = np.array(
        [0.0 if v is None else float(v) for v in in_values], dtype=np.float64
    )
    dtype = _deployment_dtype(op)
    if dtype is None:
        return {}
    kt = torch.from_numpy(vals[known_mask])
    if not bool((kt.to(dtype).to(torch.float64) == kt).all()):
        return {}                              # not exactly representable
    outs = sat = None
    mapped = not transfer.is_opaque and bool(transfer.out_to_ins)
    if mapped:
        outs, offsets, ins = _transfer_csr(transfer)
        # segment sums via cumsum: immune to reduceat's zero-length quirk
        c = np.zeros(ins.size + 1, dtype=np.int64)
        np.cumsum(~known_mask[ins], out=c[1:])
        sat = (c[offsets[1:]] - c[offsets[:-1]]) == 0
        if not bool(sat.any()):
            return {}                          # no region fully known
    elif n_known < n_in:
        return {}                              # opaque needs every input

    probes = [np.where(known_mask, vals, filler) for filler in _PROBE_FILLERS]
    probed = _probe(op, module, probes, dtype)
    if probed is None:
        return {}
    y_a, y_b = probed
    ya, yb = y_a.flatten(), y_b.flatten()
    if outs is not None and sat is not None:
        cand = outs[sat & (outs < int(ya.numel()))]
    else:
        cand = np.arange(int(ya.numel()), dtype=np.int64)
    pick = torch.from_numpy(cand)
    idx = cand[(ya[pick] == yb[pick]).numpy()]   # filler disagreement refuses
    resolved_vals = ya[torch.from_numpy(idx)].to(torch.float64).tolist()
    return {int(o): v for o, v in zip(idx.tolist(), resolved_vals)}
