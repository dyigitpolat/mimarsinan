"""The identity semantics of a logical program, and the integer network N it should equal.

``run_reference`` executes the program as written, one logical core per layer, every layer of a segment
in one pass on one absolute cycle grid (the unpartitioned single-pass program of X3a's H0):

- streamed LIF, per cycle: an active core (latency L, window [L, L+T)) adds ``x @ W + b`` to its
  membrane, saturates, fires per the comparator and resets; it reads a segment input's train at its local
  cycle t - L and an in-segment producer's buffer of cycle t - 1, which holds the producer's last fire
  outside its window (LIF memo section 1);
- streamed LIF, per event: the cycle's events (the bias first, then the neuron's own synapses with a
  nonzero weight and an arriving spike, in canonical rank order) are folded one at a time with a
  saturating add and a fire check after each, at most one spike per cycle (D15's ascending-slot serial
  fold);
- synchronized LIF: T integrate cycles with firing disabled, then T emission cycles of fire and reset
  (calculus section 16.3); an in-segment producer delivers its emission train;
- synchronized TTFS: the integer charge C = sum(w * level) + T * b, saturated once, gives the level
  clamp(F(C), 0, T) (TTFS closure section 3.1);
- value: acc = sum(w * x) + b in int64 through the requantization law (arithmetic toggle section 3.7).

``run_network`` evaluates N: the integer law of every layer on its sources' N values (no timing). For
streamed LIF with subtractive reset the reference equals N where window coverage (H2), no terminal
overshoot (H3), chase completion (H4) and the membrane bound (H8) hold (LIF memo Theorem 2); the
reference records each of them per neuron and window in ``diag``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import kernels as K
from .program import STREAMED_LIF, SYNC_LIF, SYNC_TTFS, VALUE, Program

Values = Dict[int, np.ndarray]


def _gather(prog: Program, nid: int, out: Values) -> np.ndarray:
    return np.concatenate([out[s] for s in prog.nodes[nid].sources], axis=1)


def _streamed_segment(prog: Program, seg: List[int], out: Values, diag: Dict[str, Values]) -> None:
    d, T = prog.decl, prog.decl.T
    lat = prog.latency()
    grid = max(lat[n] for n in seg) + T
    B = next(iter(out.values())).shape[0]
    def zeros(n: int, width: int = -1, dtype: Any = np.int64) -> np.ndarray:
        return np.zeros((B, prog.nodes[n].size if width < 0 else width), dtype)

    buf: Values = {n: zeros(n) for n in seg}
    mem: Values = {n: zeros(n) for n in seg}
    cnt: Values = {n: zeros(n) for n in seg}
    run: Values = {n: zeros(n) for n in seg}
    peak: Values = {n: zeros(n) + np.iinfo(np.int64).min for n in seg}
    recv: Values = {n: zeros(n, int(prog.fanin(n).size)) for n in seg}
    sat: Values = {n: zeros(n, dtype=bool) for n in seg}
    trains = {n: {s: K.trains(out[s], T, d.encoder) for s in prog.nodes[n].sources if not prog.in_segment(n, s)}
              for n in seg}
    for t in range(grid):
        new: Values = {}
        for n in seg:
            node, L = prog.nodes[n], lat[n]
            if not L <= t < L + T:
                continue
            parts = [buf[s] if prog.in_segment(n, s) else trains[n][s][t - L] for s in node.sources]
            x = np.concatenate(parts, axis=1)
            recv[n] += x
            if d.granularity == "per_event":
                mem[n], fired = K.event_fold(mem[n], x, node.weights, node.bias, node.theta, d,
                                             range(x.shape[1]))
            else:
                charge = x @ node.weights + node.bias
                run[n] += charge
                peak[n] = np.maximum(peak[n], K.staircase(run[n], node.theta, d.comparator))
                raw = mem[n] + charge
                sat[n] |= K.out_of_range(raw, d.membrane_bits)
                m = K.saturate(raw, d.membrane_bits)
                fired = K.fire(m, node.theta, d.comparator)
                mem[n] = K.reset(m, fired, node.theta, d.reset)
            new[n] = fired.astype(np.int64)
            cnt[n] += new[n]
        buf.update(new)
    for n in seg:
        out[n] = cnt[n]
    for n in seg:
        node = prog.nodes[n]
        produced = _gather(prog, n, out)
        miss = (recv[n] != produced).astype(np.int64) @ (node.weights != 0).astype(np.int64)
        final = K.staircase(run[n], node.theta, d.comparator)
        diag["coverage"][n] = miss > 0
        diag["overshoot"][n] = peak[n] > final
        diag["deficit"][n] = cnt[n] < np.clip(final, 0, T)
        diag["saturation"][n] = sat[n]


def _sync_layer(prog: Program, n: int, out: Values, emit: Values) -> np.ndarray:
    d, T, node = prog.decl, prog.decl.T, prog.nodes[n]
    trains = [emit[s] if prog.in_segment(n, s) else K.trains(out[s], T, d.encoder) for s in node.sources]
    x = np.concatenate(trains, axis=2)
    m = np.zeros((x.shape[1], node.size), np.int64)
    for k in range(T):
        m = K.saturate(m + x[k] @ node.weights + node.bias, d.membrane_bits)
    fires = []
    for _ in range(T):
        f = K.fire(m, node.theta, d.comparator)
        m = K.reset(m, f, node.theta, d.reset)
        fires.append(f.astype(np.int64))
    emit[n] = np.stack(fires)
    return emit[n].sum(axis=0)


def run_reference(prog: Program, inputs: Values) -> tuple:
    """(outputs per node [B, size], diagnostics per check per neural node [B, size] of bool)."""
    d = prog.decl
    out: Values = {k: np.array(v, np.int64) for k, v in inputs.items()}
    diag: Dict[str, Values] = {k: {} for k in ("coverage", "overshoot", "deficit", "saturation")}
    emit: Values = {}
    segs = prog.segments()
    done = set()
    for node in prog.nodes:
        if node.kind == "input" or node.nid in done:
            continue
        if node.kind == "host":
            out[node.nid] = K.host_apply(node.op, [out[s] for s in node.sources], d.hi)
            continue
        if d.semantics == STREAMED_LIF:
            seg = segs[prog.eta()[node.nid]]
            _streamed_segment(prog, seg, out, diag)
            done.update(seg)
            continue
        x = _gather(prog, node.nid, out)
        if d.semantics == SYNC_LIF:
            out[node.nid] = _sync_layer(prog, node.nid, out, emit)
        elif d.semantics == SYNC_TTFS:
            c = K.saturate(x @ node.weights + d.T * node.bias, d.membrane_bits)
            out[node.nid] = K.ttfs_level(c, node.theta, d.comparator, d.T)
        else:
            out[node.nid] = K.value_out(x @ node.weights + node.bias, node.shift, d)
    return out, diag


def network_claimed(prog: Program) -> bool:
    """Whether the box claims reference == N for this declaration (relation C)."""
    d = prog.decl
    if d.semantics == STREAMED_LIF:
        return d.granularity == "per_cycle" and d.reset == "subtract"
    return d.semantics in (SYNC_LIF, SYNC_TTFS, VALUE)


def run_network(prog: Program, inputs: Values) -> Optional[Values]:
    """The integer network N: every layer's law on its sources' N values, or None where not claimed."""
    if not network_claimed(prog):
        return None
    d = prog.decl
    out: Values = {k: np.array(v, np.int64) for k, v in inputs.items()}
    for node in prog.nodes:
        if node.kind == "input":
            continue
        if node.kind == "host":
            out[node.nid] = K.host_apply(node.op, [out[s] for s in node.sources], d.hi)
            continue
        x = _gather(prog, node.nid, out)
        if d.semantics in (STREAMED_LIF, SYNC_LIF):
            out[node.nid] = K.lif_law(x @ node.weights + d.T * node.bias, node.theta, d.comparator, d.T, d.reset)
        elif d.semantics == SYNC_TTFS:
            out[node.nid] = K.ttfs_level(x @ node.weights + d.T * node.bias, node.theta, d.comparator, d.T)
        else:
            out[node.nid] = K.value_out(x @ node.weights + node.bias, node.shift, d)
    return out
