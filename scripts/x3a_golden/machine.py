"""The packed machine: a packed program executed from its slot memory, pass by pass, batch by batch.

Per batch of K inputs, segments run in host-depth order and the host operations of each depth after its
segment. Before a pass the machine writes every non-pinned core of the pass into its slot; pinned cores
were written once at the start. A pass executes what its slots hold (a stale slot runs the wrong core).

Inside a pass (one absolute cycle grid for streamed LIF, latency-group order otherwise) a core gathers
each axon from one of: an in-pass producer (its buffer of cycle t - 1, or its emission train or value),
the carry store (a wire produced by an earlier pass of the same segment), a segment input (the uniform
train or the value of a wire from an earlier depth), or nothing (a wire not yet produced reads 0).

Hypothesis switches (``off``) act here: H5-collapse carries the count (re-encoded at the consumer)
instead of the verbatim raster or level; H5-skew times each pass by its own ALAP latencies and reads a
carried raster at the consumer's local cycle (R1 alone: own latencies, the fused delivery delay kept;
R2 alone: fused latencies, the carried raster read at the consumer's local cycle); H6 clips every part of a fused neuron to
``partial_sum_bits``; H7 folds per-event cores in physical axon order instead of rank order; H8 runs a
declared fp32 value accumulator whatever its bound; R3 lets a segment entry read a stored raster; R6
orders a windowed pass's cores by the stored streamed latency instead of the dependency group.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import numpy as np

from . import kernels as K
from .packer import HardCore
from .program import STREAMED_LIF, SYNC_LIF, SYNC_TTFS, Program
from .schedule import Packed

Values = Dict[int, np.ndarray]


def _local_latency(prog: Program, cores: List[HardCore], streamed: bool) -> Dict[int, int]:
    """Latencies a pass computes over its own layers alone (the H5-skew off arm's timing)."""
    layers = sorted({lay for c in cores for lay, _, _ in c.slots})
    inside = set(layers)
    depth: Dict[int, int] = {}
    for lay in layers:
        src = [depth[s] for s in prog.nodes[lay].sources if s in inside and prog.in_segment(lay, s)]
        depth[lay] = 1 + max(src) if src else 0
    lat = dict(depth)
    if streamed:
        for lay in reversed(layers):
            cons = [c for c in layers if lay in prog.nodes[c].sources and prog.in_segment(c, lay)]
            lat[lay] = max(lat[c] for c in cons) - 1 if cons else depth[lay]
    return {c.hid: lat[c.slots[0][0]] for c in cores}


class _Pass:
    """One pass of one batch: axon routing and the carry store reads shared by every semantics."""

    def __init__(self, prog, cores, seg, out, store, off):
        self.prog, self.cores, self.seg, self.out, self.store, self.off = prog, cores, seg, out, store, off
        self.d = prog.decl
        self.cols: Dict[Tuple[int, int], int] = {}
        for c in cores:
            if not c.partial:
                for lay, n, _ in c.slots:
                    self.cols.setdefault((lay, n), len(self.cols))
        streamed = self.d.semantics == STREAMED_LIF
        local = "H5-skew" in off or "R1" in off
        self.lat = _local_latency(prog, cores, streamed) if local else {c.hid: c.latency for c in cores}
        if "R6" in off and not streamed:
            self.lat = {c.hid: prog.latency()[c.slots[0][0]] for c in cores}
        self.early_reads = 0
        self.route = {c.hid: [self._kind(w) for w in c.axons] for c in cores}
        self.fp32 = False

    def _kind(self, w: int):
        prog = self.prog
        nid = int(prog.owner()[w])
        idx = w - prog.base(nid)
        if prog.nodes[nid].kind == "neural" and prog.eta()[nid] == self.seg:
            if (nid, idx) in self.cols:
                return ("buf", self.cols[(nid, idx)])
            if (self.seg, w) in self.store:
                return ("carry", self.store[(self.seg, w)])
            return ("missing", None)
        if "R3" in self.off and prog.nodes[nid].kind == "neural":
            for (seg, ww), rec in self.store.items():
                if ww == w:
                    return ("stale", rec)
        return ("input", self.out[nid][:, idx])


def _streamed(p: _Pass) -> Dict[Tuple[int, int], dict]:
    d, T, off = p.d, p.d.T, p.off
    B = next(iter(p.out.values())).shape[0]
    P = len(p.cols)
    grid = max(p.lat.values()) + T
    buf, mem, cnt = (np.zeros((B, P), np.int64) for _ in range(3))
    trace = np.zeros((grid, B, P), np.int64)
    local = np.zeros((T, B, P), np.int64)
    trains: Dict[int, np.ndarray] = {}

    def axon(kind, ref, t, L, fused_L, key):
        k = t - L
        if kind == "buf":
            return buf[:, ref]
        if kind == "input":
            tr = trains.setdefault(key, K.trains(ref, T, d.encoder))
            return tr[k] if 0 <= k < T else np.zeros(B, np.int64)
        if kind in ("carry", "stale"):
            if kind == "carry" and "H5-collapse" in off:
                tr = trains.setdefault(key, K.trains(ref["count"], T, d.encoder))
                return tr[k] if 0 <= k < T else np.zeros(B, np.int64)
            if kind == "carry" and not {"H5-skew", "R1", "R2"} & off:  # R1 pins and R2 delays: the fused grid
                tr = ref["trace"]
                return tr[min(t - 1, tr.shape[0] - 1)] if t >= 1 else np.zeros(B, np.int64)
            if kind == "carry" and "R1" in off:  # own timing, the fused delivery delay kept
                k += fused_L - 1 - ref["lat"]
            return ref["local"][k] if 0 <= k < T else np.zeros(B, np.int64)
        return np.zeros(B, np.int64)

    for t in range(grid):
        charges = np.zeros((B, P), np.int64)
        firing = []
        new_buf = buf.copy()
        for c in p.cores:
            L = p.lat[c.hid]
            if not L <= t < L + T:
                continue
            x = np.stack([axon(kind, ref, t, L, c.latency, (c.hid, i)) for i, (kind, ref) in
                          enumerate(p.route[c.hid])], axis=1) if c.axons else np.zeros((B, 0), np.int64)
            cols = [p.cols[(lay, n)] for lay, n, _ in c.slots] if not c.partial else []
            if d.granularity == "per_event":
                order = sorted(range(len(c.axons)), key=lambda i: c.axons[i]) if "H7" not in off else range(len(c.axons))
                m, f = K.event_fold(mem[:, cols], x, c.weights, c.bias, c.scalar, d, order)
                mem[:, cols] = m
                new_buf[:, cols] = f
                continue
            part = x @ c.weights
            for j, (lay, n, g) in enumerate(c.slots):
                col = p.cols.get((lay, n))
                if col is not None:
                    charges[:, col] += K.clip_signed(part[:, j], d.partial_sum_bits) if (g >= 0 and "H6" in off) else part[:, j]
            if not c.partial:
                charges[:, cols] += c.bias
                firing.append((c, cols))
        for c, cols in firing:
            m = K.saturate(mem[:, cols] + charges[:, cols], d.membrane_bits)
            f = K.fire(m, c.scalar, d.comparator)
            mem[:, cols] = K.reset(m, f, c.scalar, d.reset)
            new_buf[:, cols] = f
        for c in p.cores:
            L = p.lat[c.hid]
            if not c.partial and L <= t < L + T:
                cols = [p.cols[(lay, n)] for lay, n, _ in c.slots]
                cnt[:, cols] += new_buf[:, cols]
                local[t - L][:, cols] = new_buf[:, cols]
        buf = new_buf
        trace[t] = buf
    lat = {p.cols[(lay, n)]: c.latency for c in p.cores if not c.partial for lay, n, _ in c.slots}
    return {key: {"count": cnt[:, col], "trace": trace[:, :, col], "local": local[:, :, col], "lat": lat[col]}
            for key, col in p.cols.items()}


def _windowed(p: _Pass) -> Dict[Tuple[int, int], dict]:
    """Synchronized LIF, synchronized TTFS and value: cores in latency-group order, one window each."""
    d, T, off, sem = p.d, p.d.T, p.off, p.d.semantics
    B = next(iter(p.out.values())).shape[0]
    done: Dict[int, dict] = {}
    zero = (lambda: np.zeros((T, B), np.int64)) if sem == SYNC_LIF else (lambda: np.zeros(B, np.int64))

    def window(kind, ref):  # [T, B] trains (synchronized LIF) or [B] levels and values
        if kind == "buf":
            rec = done.get(ref)
            p.early_reads += rec is None
            return zero() if rec is None else rec["train" if sem == SYNC_LIF else "count"]
        if kind in ("carry", "stale"):
            collapse = kind == "carry" and "H5-collapse" in off
            if sem == SYNC_LIF:
                return K.trains(ref["count"], T, d.encoder) if collapse else ref["train"]
            return np.minimum(ref["count"], 1) if collapse and sem == SYNC_TTFS else ref["count"]
        if kind == "input":
            return K.trains(ref, T, d.encoder) if sem == SYNC_LIF else ref
        return zero()

    fp32 = p.fp32

    def part_of(c: HardCore):
        x = np.stack([window(k, r) for k, r in p.route[c.hid]], axis=-1) if c.axons else \
            np.zeros(zero().shape + (0,), np.int64)
        if not fp32:
            return x @ c.weights
        acc = np.zeros((B, len(c.slots)), np.float32)
        for i in range(len(c.axons)):  # sequential float32 accumulation in physical axon order
            acc = (acc + (x[:, i:i + 1] * c.weights[i][None, :]).astype(np.float32)).astype(np.float32)
        return acc

    serving: Dict[Tuple[int, int], List[Tuple[HardCore, int]]] = {}
    for q in p.cores:
        if q.partial:
            for jj, (lay, n, _) in enumerate(q.slots):
                serving.setdefault((lay, n), []).append((q, jj))
    cache: Dict[int, np.ndarray] = {}
    for c in sorted((c for c in p.cores if not c.partial), key=lambda c: (p.lat[c.hid], c.hid)):
        own = part_of(c)
        fused = np.array([g >= 0 for _, _, g in c.slots], bool)
        if "H6" in off:
            own = np.where(fused, K.clip_signed(own, d.partial_sum_bits), own).astype(own.dtype)
        extra = np.zeros_like(own)
        for j, (lay, n, g) in enumerate(c.slots):
            for q, jj in serving.get((lay, n), []) if g >= 0 else []:
                cache.setdefault(q.hid, part_of(q))
                e = cache[q.hid][..., jj]
                e = K.clip_signed(e, d.partial_sum_bits) if "H6" in off else e
                extra[..., j] = (extra[..., j] + e).astype(extra.dtype)
        bias = T * c.bias if sem == SYNC_TTFS else c.bias
        if fp32:
            total = ((bias.astype(np.float32) + own).astype(np.float32) + extra).astype(np.float32)
        else:
            total = own + extra + bias
        if sem == SYNC_LIF:
            m = np.zeros((B, len(c.slots)), np.int64)
            for k in range(T):
                m = K.saturate(m + total[k], d.membrane_bits)
            fires = []
            for _ in range(T):
                f = K.fire(m, c.scalar, d.comparator)
                m = K.reset(m, f, c.scalar, d.reset)
                fires.append(f.astype(np.int64))
            res = {"train": np.stack(fires)}
            res["count"] = res["train"].sum(axis=0)
        elif sem == SYNC_TTFS:
            res = {"count": K.ttfs_level(K.saturate(total, d.membrane_bits), c.scalar, d.comparator, T)}
        else:
            res = {"count": K.value_out(total.astype(np.int64), c.scalar, d)}
        for j, (lay, n, _) in enumerate(c.slots):
            done[p.cols[(lay, n)]] = {key: v[..., j] for key, v in res.items()}
    return {key: done[col] for key, col in p.cols.items() if col in done}


def run_packed(prog: Program, packed: Packed, inputs: Values, off: FrozenSet[str] = frozenset(),
               exact: bool = False) -> tuple:
    """(outputs per node [B, size], machine statistics) of a packed program on ``inputs``.

    ``exact`` runs every accumulator in int64 whatever the declaration (the int64 golden of the identity
    program); otherwise a value declaration's ``accumulator`` is the machine's.
    """
    B = next(iter(inputs.values())).shape[0]
    memory = {slot: hid for hid, slot in packed.pinned.items()}
    stats = {"writes": len(memory), "stale_runs": 0, "carried_reads": 0, "stale_reads": 0, "early_reads": 0}
    segs = sorted({seg for seg, _ in packed.passes})
    host_at: Dict[int, List[int]] = {}
    for node in prog.nodes:
        if node.kind == "host":
            host_at.setdefault(prog.eta()[node.nid], []).append(node.nid)
    results: List[Values] = []
    for b0 in range(0, B, packed.batch_K):
        out: Values = {}
        for node in prog.nodes:
            src = inputs.get(node.nid)
            out[node.nid] = np.array(src[b0:b0 + packed.batch_K], np.int64) if src is not None else \
                np.zeros((min(packed.batch_K, B - b0), node.size), np.int64)
        store: Dict[Tuple[int, int], dict] = {}
        for depth in sorted(set(segs) | set(host_at)):
            for seg, hids in packed.passes:
                if seg != depth:
                    continue
                for h in hids:
                    if h not in packed.pinned:
                        memory[packed.slot_of[h]] = h
                        stats["writes"] += 1
                ran, cores = set(), []
                for h in hids:
                    held = memory[packed.pinned.get(h, packed.slot_of.get(h, -1))]
                    stats["stale_runs"] += held != h
                    if held not in ran:
                        ran.add(held)
                        cores.append(packed.cores[held])
                ps = _Pass(prog, cores, seg, out, store, off)
                ps.fp32 = prog.decl.accumulator == "fp32" and not exact
                kinds = [k for r in ps.route.values() for k, _ in r]
                stats["carried_reads"] += kinds.count("carry")
                stats["stale_reads"] += kinds.count("stale")
                recs = _streamed(ps) if prog.decl.semantics == STREAMED_LIF else _windowed(ps)
                stats["early_reads"] += ps.early_reads
                for (lay, n), rec in recs.items():
                    out[lay][:, n] = rec["count"]
                    store[(seg, prog.base(lay) + n)] = rec
            for h in host_at.get(depth, []):
                node = prog.nodes[h]
                out[h] = K.host_apply(node.op, [out[s] for s in node.sources], prog.decl.hi)
        results.append(out)
    return {nid: np.concatenate([r[nid] for r in results]) for nid in results[0]}, stats
