"""Segment-aware TTFS spike-train forward (pre-mapping, differentiable).

The cascaded ``ttfs_cycle_based`` deployment runs each neural segment as a
single-spike, ramp-integrate, fire-once simulation, with value-domain compute
ops between segments and a host-side *encoding layer* (value -> TTFS spike) at
each segment entry. To fine-tune *through* that (not a pointwise surrogate),
this driver walks a ``ModelRepresentation`` exec graph and reproduces it on the
trainable model.

Each node is classified as **spike-producing** or **value-producing** (mirroring
``torch_mapping.encoding_layers``): perceptrons produce spikes; the raw input and
unbounded raw-Linear/Conv ``ComputeOp``s produce values; transparent nodes
(structural reshapes/permutes, bounded compute ops) inherit from their sources.
A *segment* is a maximal connected region of spike-producing nodes. It runs a
``T``-cycle spike sim:

- an entry perceptron (value source) turns its ideal pre-activation into a TTFS
  spike train (``TTFSActivation(encoding=True)``);
- interior perceptrons ramp from arriving single spikes (``TTFSActivation``);
- transparent nodes inside the region apply their (value-invariant) op to the
  per-cycle spikes;
- any region node consumed by a value node (or the output) is decoded
  ``count_of_latched_spikes / T * activation_scale``.

TTFS single-spike timing is causal (a neuron must know its value before placing
its one spike), so segments run sequentially: a segment's inputs are the decoded
values of upstream segments, never their per-cycle spikes. This mirrors
``SpikingHybridCoreFlow`` (the deployed simulator) but on the unmapped model.
"""

from __future__ import annotations

import torch

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.torch_mapping.encoding_layers import _wraps_unbounded_raw_linear_or_conv


def _perceptron_of(node):
    return getattr(node, "perceptron", None)


def _is_encoding_perceptron(node) -> bool:
    p = _perceptron_of(node)
    return p is not None and getattr(p, "is_encoding_layer", False)


def _is_value_boundary(node) -> bool:
    """A node that produces signed/unbounded *values*, never on-chip spikes."""
    if isinstance(node, InputMapper):
        return True
    return isinstance(node, ComputeOpMapper) and _wraps_unbounded_raw_linear_or_conv(node)


def classify_spike_producers(exec_order, deps_map) -> dict:
    """Map each node to whether it carries on-chip spikes (vs host-side values)."""
    produces: dict = {}
    for node in exec_order:  # topological: deps precede node
        if _perceptron_of(node) is not None:
            produces[node] = True
        elif _is_value_boundary(node):
            produces[node] = False
        else:  # transparent (structural / bounded compute op): inherit from sources
            produces[node] = any(produces.get(d, False) for d in deps_map.get(node, []))
    return produces


def partition_spike_segments(exec_order, deps_map):
    """Group spike-producing nodes into segments (maximal connected regions).

    Returns ``({node: segment_root}, produces_spikes)``. Nodes are connected when
    a spike-producing node depends on another spike-producing node.
    """
    produces = classify_spike_producers(exec_order, deps_map)
    parent: dict = {}

    def find(n):
        root = n
        while parent[root] != root:
            root = parent[root]
        while parent[n] != root:
            parent[n], n = root, parent[n]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra is not rb:
            parent[ra] = rb

    spike_nodes = [n for n in exec_order if produces[n]]
    for n in spike_nodes:
        parent.setdefault(n, n)
    for n in spike_nodes:
        # An encoding layer consumes a *value* (its upstream spike region is
        # decoded at this edge), so it starts a fresh segment -- never union it
        # with its source.
        if _is_encoding_perceptron(n):
            continue
        for d in deps_map.get(n, []):
            if produces.get(d, False):
                parent.setdefault(d, d)
                union(n, d)
    return {n: find(n) for n in spike_nodes}, produces


def partition_perceptron_segments(exec_order, deps_map):
    """Perceptron-only view of :func:`partition_spike_segments` (for tests/introspection)."""
    seg_of, _ = partition_spike_segments(exec_order, deps_map)
    return {n: r for n, r in seg_of.items() if _perceptron_of(n) is not None}


class TTFSSegmentForward:
    """Differentiable segment-aware TTFS spike forward over a ``ModelRepresentation``.

    Install as ``model.forward`` during TTFS-cycle fine-tuning (the analog of
    LIF's ``run_cycle_accurate`` install). Perceptron activations must be
    ``TTFSActivation`` with ``encoding`` set to match ``is_encoding_layer``.
    """

    def __init__(self, mapper_repr, T: int):
        self.repr = mapper_repr
        self.T = int(T)
        self.repr._ensure_exec_graph()
        self._exec = self.repr._exec_order
        self._deps = self.repr._deps
        self._output = self.repr.output_layer_mapper
        self._index = {n: i for i, n in enumerate(self._exec)}

        self._seg_of, self._produces = partition_spike_segments(self._exec, self._deps)
        segments: dict = {}
        for node, root in self._seg_of.items():
            segments.setdefault(root, []).append(node)
        for root in segments:
            segments[root].sort(key=lambda n: self._index[n])
        self._segments = segments

        consumers: dict = {n: [] for n in self._exec}
        for node in self._exec:
            for dep in self._deps.get(node, []):
                if dep in consumers:
                    consumers[dep].append(node)
        self._consumers = consumers

    # -- spike-node lifecycle ------------------------------------------------
    def _ttfs_nodes(self, nodes):
        out = []
        for n in nodes:
            p = _perceptron_of(n)
            if p is None:
                continue
            for m in p.modules():
                if isinstance(m, TTFSActivation):
                    out.append(m)
        return out

    # -- boundary classification --------------------------------------------
    def _externally_consumed(self, seg_nodes):
        """Region nodes whose decoded value is needed outside the region."""
        seg_set = set(seg_nodes)
        ext = []
        for n in seg_nodes:
            if n is self._output or any(c not in seg_set for c in self._consumers.get(n, [])):
                ext.append(n)
        return ext

    def _decode_scale(self, node):
        """activation_scale of the perceptron that produced this region node's spikes."""
        p = _perceptron_of(node)
        if p is not None:
            return p.activation_scale
        for d in self._deps.get(node, []):
            if self._produces.get(d, False):
                return self._decode_scale(d)
        return 1.0

    # -- execution -----------------------------------------------------------
    def __call__(self, x):
        ttfs = self._ttfs_nodes(self._seg_of.keys())
        for m in ttfs:
            m.set_cycle_accurate(True)
        try:
            return self._run(x)
        finally:
            for m in ttfs:
                m.set_cycle_accurate(False)
                m.reset_state()

    def _forward_node(self, node, inputs, x):
        d = self._deps.get(node, [])
        if len(d) == 0:
            return node.forward(x)
        if len(d) == 1:
            return node.forward(inputs[d[0]])
        return node.forward(tuple(inputs[dep] for dep in d))

    def _run(self, x):
        values: dict = {}
        done = set()
        remaining = {n: len(self._consumers.get(n, [])) for n in self._exec}

        for node in self._exec:
            if self._produces[node]:
                root = self._seg_of[node]
                if root not in done:
                    self._run_segment(self._segments[root], values, x)
                    done.add(root)
            else:
                values[node] = self._forward_node(node, values, x)

            for dep in self._deps.get(node, []):
                if dep is None or dep is self._output:
                    continue
                remaining[dep] -= 1
                if remaining[dep] == 0 and dep in values:
                    del values[dep]

        return values[self._output]

    def _segment_depths(self, seg_nodes):
        """Per-node cascade latency = perceptron-hops from the segment entry.

        Each perceptron core adds one cycle of propagation delay; transparent
        routing (reshape/permute/...) adds none. Matches ``ChipLatency`` within a
        segment (a core's latency = max source-core latency + 1).
        """
        seg_set = set(seg_nodes)
        depth: dict = {}
        for n in seg_nodes:  # exec/topological order
            in_src = [d for d in self._deps.get(n, []) if d in seg_set]
            if not in_src:
                depth[n] = 0
            else:
                depth[n] = max(depth[s] + (1 if _perceptron_of(s) is not None else 0)
                               for s in in_src)
        return depth

    def _segment_output_zeros(self, seg_nodes, values, x):
        """Per-node output shapes (zero tensors) for not-yet-fired delayed sources."""
        seg_set = set(seg_nodes)
        nodes = self._ttfs_nodes(seg_nodes)
        for m in nodes:
            m.set_cycle_accurate(False)
        zeros: dict = {}
        with torch.no_grad():
            vmode: dict = {}
            for n in seg_nodes:
                d = self._deps.get(n, [])
                if len(d) == 1:
                    inp = vmode[d[0]] if d[0] in seg_set else values[d[0]]
                elif len(d) == 0:
                    inp = x
                else:
                    inp = tuple(vmode[dep] if dep in seg_set else values[dep] for dep in d)
                vmode[n] = n.forward(inp)
                zeros[n] = torch.zeros_like(vmode[n])
        for m in nodes:
            m.set_cycle_accurate(True)
            m.reset_state()
        return zeros

    def _run_segment(self, seg_nodes, values, x):
        seg_set = set(seg_nodes)
        ext = self._externally_consumed(seg_nodes)
        depth = self._segment_depths(seg_nodes)
        zeros = self._segment_output_zeros(seg_nodes, values, x)

        def read(src, out, perc_prev):
            if src not in seg_set:
                return values[src]                      # ideal value (segment input)
            if _perceptron_of(src) is not None:
                return perc_prev.get(src, zeros[src])   # core output, 1-cycle delayed
            return out[src]                             # transparent routing, this cycle

        accum: dict = {}
        latched: dict = {}
        perc_prev: dict = {}
        n_cycles = self.T + max(depth.values(), default=0)
        for t in range(n_cycles):
            out: dict = {}
            for n in seg_nodes:
                # Latency-gated: a core integrates only inside its own window
                # [depth, depth+T); outside it emits nothing (no premature bias firing).
                if t < depth[n] or t >= depth[n] + self.T:
                    out[n] = zeros[n]
                    continue
                d = self._deps.get(n, [])
                if len(d) == 1:
                    inp = read(d[0], out, perc_prev)
                elif len(d) == 0:
                    inp = x
                else:
                    inp = tuple(read(dep, out, perc_prev) for dep in d)
                out[n] = n.forward(inp)
            for n in seg_nodes:
                if _perceptron_of(n) is not None:
                    perc_prev[n] = out[n]
            for n in ext:                               # per-source window [lat, lat+T)
                if depth[n] <= t < depth[n] + self.T:
                    s = out[n]
                    latched[n] = s if n not in latched else torch.maximum(latched[n], s)
                    accum[n] = latched[n] if n not in accum else accum[n] + latched[n]

        for n in ext:
            scale = self._decode_scale(n)
            sv = (scale.to(accum[n].device, accum[n].dtype)
                  if isinstance(scale, torch.Tensor) else float(scale))
            values[n] = (accum[n] / float(self.T)) * sv
