"""DIRECTION 2: TWO-PHASE pipelined cascade neuron (accumulate -> emit).

The greedy single-spike cascade fires the cycle its RUNNING partial sum crosses
theta. With mixed-sign ReLU weights, early (positive) spikes cross theta before
later (cancelling) spikes arrive -> premature/wrong fire -> the deep death
cascade. The staircase/LIF use the COMPLETE weighted sum and are lossless.

Two-phase fix (implemented in TTFSActivation.set_two_phase): each core LISTENS
over its whole window accumulating the COMPLETE ramp-reconstructed weighted sum
WITHOUT firing, then EMITS a single timing spike encoding that complete sum at
the staircase fire cycle. It stays PIPELINED (depth-staggered, no global
barrier) -- each perceptron-hop occupies [2T*h, 2T*h+2T): listen [.., +T), emit
[+T, +2T). The consumer at hop h+1 listens exactly over the producer's emit
window. So the emitted timing code is byte-identical to the staircase decode ->
a two-phase cascade is lossless BY CONSTRUCTION.

This policy (TwoPhaseTtfsPolicy) reuses the perceptron forward; only the
activation node's fire gate changes (two-phase mode in src). The greedy cascade
window is T per hop; two-phase is 2T per hop (the latency cost).

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/two_phase.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from ft_budget import DEV, build  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc, ttfs_genuine_acc  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.spiking.segment_forward import SegmentForwardDriver, TtfsSegmentPolicy  # noqa: E402
from mimarsinan.spiking.segment_partition import (  # noqa: E402
    is_encoding_perceptron,
    perceptron_of,
)
from mimarsinan.spiking.scale_aware_boundaries import (  # noqa: E402
    propagate_boundary_input_scales,
)


class TwoPhaseTtfsPolicy(TtfsSegmentPolicy):
    """Pipelined accumulate-then-emit cascade. Each cascade node LISTENS over a
    T-window (accumulating the complete ramp-reconstructed weighted sum without
    firing) then EMITS a single timing spike encoding that complete sum over the
    next T cycles. Windows are placed by ``_emit_starts`` (topological T-stagger
    honouring the 1-cycle perc_prev read delay): a producer's emit window is its
    consumer's listen window, so the schedule is depth-staggered with NO global
    barrier (the cascade throughput advantage is preserved; latency is ~2T/hop)."""

    def _ttfs_node_of(self, n):
        p = perceptron_of(n)
        if p is None:
            return None
        ms = [m for m in p.modules() if isinstance(m, TTFSActivation)]
        return ms[0] if ms else None

    def prepare(self, driver):
        super().prepare(driver)
        for _p, mods in self._ttfs_perceptrons(driver._seg_of.keys()):
            for m in mods:
                m.set_two_phase(True)

    def finalize(self, driver):
        for _p, mods in self._ttfs_perceptrons(driver._seg_of.keys()):
            for m in mods:
                m.set_two_phase(False)
        super().finalize(driver)

    def _emit_starts(self, driver, seg_nodes):
        """Per-node first cycle its output is valid (the pipelined T-stagger).

        Topological propagation over the segment exec graph, honouring the
        1-cycle ``perc_prev`` read delay between a spike-producing perceptron and
        its consumer:
          - encoding entry / value boundary: emit_start = 0 (value known up front)
          - transparent route: max over deps of (dep_emit_start + delay(dep))
          - cascade perceptron: listen over T cycles from when its inputs first
            arrive, then emit -> emit_start = listen_start + T
        ``delay(dep) = 1`` iff ``dep`` is a spike-producing perceptron (else 0)."""
        T = driver.T
        seg_set = set(seg_nodes)
        order = sorted(seg_nodes, key=lambda n: driver._index[n])
        emit: dict = {}
        for n in order:
            node = self._ttfs_node_of(n)
            deps = [d for d in driver._deps.get(n, []) if d in seg_set]

            def arrival():
                if not deps:
                    return 0
                return max(emit.get(d, 0) + (1 if perceptron_of(d) is not None else 0)
                           for d in deps)

            if node is not None and node.encoding:
                emit[n] = 0
            elif node is not None:                 # cascade perceptron
                emit[n] = arrival() + T            # listen [arrival, arrival+T)
            else:                                  # transparent route
                emit[n] = arrival()
        return emit

    def run_segment(self, driver, seg_nodes, values, x):
        T = driver.T
        seg_set = set(seg_nodes)
        ext = driver.external_consumed(seg_nodes)
        zeros = self._segment_output_zeros(driver, seg_nodes, values, x)

        emit0 = self._emit_starts(driver, seg_nodes)
        listen0 = {n: emit0[n] - T for n in seg_nodes}   # cascade listen window
        n_cycles = max((emit0[n] + T for n in seg_nodes), default=T)

        boundary_trains: dict = {}

        def boundary_spikes(src, t):
            train = boundary_trains.get(src)
            if train is None:
                train = self._boundary_single_spike_train(values[src], T, n_cycles)
                boundary_trains[src] = train
            return train[t]

        def read(src, out, perc_prev, t, consumer):
            if src not in seg_set:
                if is_encoding_perceptron(consumer):
                    return values[src]
                return boundary_spikes(src, t)
            if perceptron_of(src) is not None:
                return perc_prev.get(src, zeros[src])
            return out[src]

        watch = list(ext)
        if self.node_value_recorder is not None:
            ext_set = set(ext)
            watch += [n for n in seg_nodes
                      if perceptron_of(n) is not None and n not in ext_set]

        # set every node idle initially
        for n in seg_nodes:
            node = self._ttfs_node_of(n)
            if node is not None:
                node.set_two_phase_phase("idle")

        def active_window(n):
            """The T cycles a node carries valid output: [emit0, emit0+T)."""
            return emit0[n], emit0[n] + T

        accum: dict = {}
        latched: dict = {}
        perc_prev: dict = {}
        for t in range(n_cycles):
            out: dict = {}
            for n in seg_nodes:
                node = self._ttfs_node_of(n)
                enc = node is not None and node.encoding
                if node is not None and not enc:
                    if listen0[n] <= t < listen0[n] + T:
                        node.set_two_phase_phase("listen")
                    elif emit0[n] <= t < emit0[n] + T:
                        node.set_two_phase_phase("emit", emit_rel_cycle=t - emit0[n])
                    else:
                        node.set_two_phase_phase("idle")
                        out[n] = zeros[n]
                        continue
                elif enc:
                    if not (0 <= t < T):           # encoding entry only emits [0, T)
                        out[n] = zeros[n]
                        continue
                else:
                    # transparent routing: forward its producer's spikes over the
                    # producer's emit window (1-cycle perc_prev delay included).
                    lo, hi = active_window(n)
                    if not (lo <= t < hi):
                        out[n] = zeros[n]
                        continue
                d = driver._deps.get(n, [])
                if len(d) == 1:
                    inp = read(d[0], out, perc_prev, t, n)
                elif len(d) == 0:
                    inp = x
                else:
                    inp = tuple(read(dep, out, perc_prev, t, n) for dep in d)
                out[n] = n.forward(inp)
            for n in seg_nodes:
                if perceptron_of(n) is not None:
                    perc_prev[n] = out[n]
            for n in watch:
                lo, hi = active_window(n)
                if lo <= t < hi:
                    s = out[n]
                    latched[n] = s if n not in latched else torch.maximum(latched[n], s)
                    accum[n] = latched[n] if n not in accum else accum[n] + latched[n]

        def _decode(n):
            scale = self.decode_scale(driver, n)
            sv = (scale.to(accum[n].device, accum[n].dtype)
                  if isinstance(scale, torch.Tensor) else float(scale))
            return (accum[n] / float(T)) * sv

        for n in ext:
            values[n] = _decode(n)
        if self.node_value_recorder is not None:
            for n in watch:
                if perceptron_of(n) is not None:
                    self.node_value_recorder[n] = (
                        values[n] if n in ext else _decode(n)).detach()
        if driver._output in seg_set:
            return values[driver._output]
        return None


def two_phase_genuine_acc(flow, x, y, S, *, fix_input_scale=True):
    if fix_input_scale:
        propagate_boundary_input_scales(flow)  # input_scale = upstream activation_scale
    drv = SegmentForwardDriver(flow.get_mapper_repr(), S, TwoPhaseTtfsPolicy())
    with torch.no_grad():
        return _accuracy(drv(x.double()), y)


def two_phase_logits(flow, x, S, *, fix_input_scale=True):
    if fix_input_scale:
        propagate_boundary_input_scales(flow)
    drv = SegmentForwardDriver(flow.get_mapper_repr(), S, TwoPhaseTtfsPolicy())
    return drv(x.double())


if __name__ == "__main__":
    print(f"=== TWO-PHASE pipelined cascade vs staircase / greedy (device={DEV}) ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'stair':>7} {'greedy':>7} "
          f"{'2phase':>7} | {'2ph-stair':>9}")
    for depth in (3, 6, 9):
        for S in (8, 16):
            flow, _xtr, _ytr, xte, yte, cont, _teacher, _base = build(depth, S)
            stair = ttfs_staircase_acc(flow, xte, yte, S)
            greedy = ttfs_genuine_acc(flow, xte, yte, S)
            twoph = two_phase_genuine_acc(flow, xte, yte, S)
            print(f"{depth:>2} {S:>3} | {cont:>6.3f} {stair:>7.3f} {greedy:>7.3f} "
                  f"{twoph:>7.3f} | {twoph - stair:>+9.3f}")
