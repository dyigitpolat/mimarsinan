"""Single-spike TTFS segment policy for the unified segment-forward driver."""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.recording import spike_modes
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.perceptron_mixer.perceptron import effective_preactivation_bias
from mimarsinan.spiking.segment_boundary import normalize_ttfs_boundary_value
from mimarsinan.spiking.segment_hop_frontier import run_segment_hop_hybrid
from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of


def segment_series_roots(driver) -> list:
    """Segment roots in execution order — the P4 topological-frontier axis."""
    return sorted(
        driver.segments.keys(),
        key=lambda root: min(driver._index[n] for n in driver.segments[root]),
    )


class TtfsSegmentPolicy:
    """Single-spike TTFS: latency-windowed segment sim with arrival latch and ramp
    decode; entry (encoding) perceptrons fire from the decoded value."""

    node_value_recorder: dict | None = None
    # A float >0 swaps in an STE at offload boundaries; forward (NF↔SCM parity) is unchanged either way.
    boundary_surrogate_temp: float | None = None
    # P4 prefix axis: series indices run genuinely; None = all genuine (deployed default).
    genuine_segments: frozenset | None = None
    # [5v B2] P4 below segments: hops with cascade depth < k run genuinely,
    # deeper hops run the trained proxy on decoded values; None = no frontier.
    genuine_hop_frontier: int | None = None

    def prepare(self, driver):
        for p, mods in self._ttfs_perceptrons(driver._seg_of.keys()):
            bias = effective_preactivation_bias(p)
            for m in mods:
                m.set_cycle_accurate(True)
                m.set_bias(bias)

    def finalize(self, driver):
        # Restore the raw layer.bias reference: the stored bias must stay picklable (no autograd graph).
        for p, mods in self._ttfs_perceptrons(driver._seg_of.keys()):
            bias = getattr(p.layer, "bias", None)
            for m in mods:
                m.set_cycle_accurate(False)
                m.reset_state()
                m.set_bias(bias)

    @staticmethod
    def _ttfs_perceptrons(nodes):
        out = []
        seen = set()
        for n in nodes:
            p = perceptron_of(n)
            if p is None or id(p) in seen:
                continue
            seen.add(id(p))
            mods = [m for m in p.modules() if isinstance(m, TTFSActivation)]
            if mods:
                out.append((p, mods))
        return out

    @classmethod
    def _ttfs_nodes(cls, nodes):
        return [m for _, mods in cls._ttfs_perceptrons(nodes) for m in mods]

    def decode_scale(self, driver, node):
        """activation_scale of the perceptron that produced this region node's spikes."""
        p = perceptron_of(node)
        if p is not None:
            return p.activation_scale
        for d in driver._deps.get(node, []):
            if driver._produces.get(d, False):
                return self.decode_scale(driver, d)
        return 1.0

    def consumer_wire_scale(self, driver, node):
        """``input_activation_scale`` of the perceptron this region node feeds into.

        The wire-domain normalizer for boundary spikes read by ``node``: for a
        perceptron it is its own input scale (what its weight fold multiplies back
        in); a transparent (non-perceptron) node inherits its in-segment consumer's.
        """
        p = perceptron_of(node)
        if p is not None:
            return getattr(p, "input_activation_scale", 1.0)
        seg_root = driver._seg_of.get(node)
        for c in driver._consumers.get(node, []):
            if driver._seg_of.get(c) is seg_root:
                return self.consumer_wire_scale(driver, c)
        return 1.0

    def segment_depths(self, driver, seg_nodes):
        """Per-node cascade latency = perceptron-hops from the segment entry
        (each perceptron core adds one cycle; transparent routing adds none)."""
        seg_set = set(seg_nodes)
        depth: dict = {}
        for n in seg_nodes:  # exec/topological order
            in_src = [d for d in driver._deps.get(n, []) if d in seg_set]
            if not in_src:
                depth[n] = 0
            else:
                depth[n] = max(depth[s] + (1 if perceptron_of(s) is not None else 0)
                               for s in in_src)
        return depth

    def _value_mode_forward(self, driver, seg_nodes, values, x) -> dict:
        """Run the segment's nodes in the trained value domain (staircase proxy).

        Returns the per-node value map; TTFS nodes are toggled out of
        cycle-accurate mode for the walk and restored (state reset) after.
        """
        seg_set = set(seg_nodes)
        nodes = self._ttfs_nodes(seg_nodes)
        for m in nodes:
            m.set_cycle_accurate(False)
        try:
            vmode: dict = {}
            for n in seg_nodes:
                d = driver._deps.get(n, [])
                if len(d) == 1:
                    inp = vmode[d[0]] if d[0] in seg_set else values[d[0]]
                elif len(d) == 0:
                    inp = x
                else:
                    inp = tuple(vmode[dep] if dep in seg_set else values[dep] for dep in d)
                vmode[n] = n.forward(inp)
        finally:
            for m in nodes:
                m.set_cycle_accurate(True)
                m.reset_state()
        return vmode

    def _segment_output_zeros(self, driver, seg_nodes, values, x):
        """Per-node output shapes (zero tensors) for not-yet-fired delayed sources."""
        with torch.no_grad():
            vmode = self._value_mode_forward(driver, seg_nodes, values, x)
        return {n: torch.zeros_like(v) for n, v in vmode.items()}

    def _series_index_of(self, driver, seg_nodes) -> int:
        """Series index (exec order) of the segment owning ``seg_nodes``."""
        cache = getattr(self, "_series_index_cache", None)
        if cache is None or cache[0] is not driver:
            series = {r: i for i, r in enumerate(segment_series_roots(driver))}
            cache = (driver, series)
            self._series_index_cache = cache
        return cache[1][driver._seg_of[seg_nodes[0]]]

    def _run_segment_value_mode(self, driver, seg_nodes, values, x):
        """Suffix member of the P4 prefix hybrid: the trained proxy on decoded values."""
        vmode = self._value_mode_forward(driver, seg_nodes, values, x)
        for n in driver.external_consumed(seg_nodes):
            values[n] = vmode[n]
        if self.node_value_recorder is not None:
            for n in seg_nodes:
                if perceptron_of(n) is not None:
                    self.node_value_recorder[n] = vmode[n].detach()
        if driver._output in set(seg_nodes):
            return vmode[driver._output]
        return None

    def _boundary_single_spike_train(self, value, boundary_scale, T: int) -> torch.Tensor:
        """Single-spike TTFS train of a decoded boundary value at the rising edge of
        HCM's latched encode (``spike_time = round(T(1 - v))`` with the WIRE-normalized
        ``v = clamp(value / boundary_scale, 0, 1)`` — the transcoding SSOT). The train
        is window-relative (length ``T``); consumers read it at ``t - depth``. The
        forward is the exact hard encode; ``boundary_surrogate_temp`` adds an STE backward.
        """
        v = normalize_ttfs_boundary_value(value, boundary_scale)
        latched = torch.stack(
            [spike_modes.to_spikes(v, c, simulation_length=T, spike_mode="TTFS")
             for c in range(T)],
            dim=0,
        ).to(value.dtype)
        train_hard = torch.cat([latched[:1], latched[1:] - latched[:-1]], dim=0)

        temp = self.boundary_surrogate_temp
        if temp is None or not torch.is_grad_enabled():
            return train_hard
        return self._straight_through(train_hard, v, T, float(temp))

    @staticmethod
    def _straight_through(train_hard, v, T, temp):
        """STE: forward == ``train_hard`` exactly; backward flows through a soft
        spike-time ``sigmoid((c - tau)/temp)`` with ``tau = T(1 - v)``."""
        cycles = torch.arange(T, device=v.device, dtype=v.dtype)
        cycles = cycles.reshape((T,) + (1,) * v.dim())
        tau = float(T) * (1.0 - v)
        latched_soft = torch.sigmoid((cycles - tau) / temp)
        train_soft = torch.cat(
            [latched_soft[:1], latched_soft[1:] - latched_soft[:-1]], dim=0
        )
        return train_hard.detach() + (train_soft - train_soft.detach())

    def run_segment(self, driver, seg_nodes, values, x):
        genuine = self.genuine_segments
        if genuine is not None and self._series_index_of(driver, seg_nodes) not in genuine:
            return self._run_segment_value_mode(driver, seg_nodes, values, x)
        k = self.genuine_hop_frontier
        if k is not None:
            return run_segment_hop_hybrid(self, driver, seg_nodes, values, x, int(k))
        return self._run_segment_genuine(driver, seg_nodes, values, x)

    def _run_segment_genuine(self, driver, seg_nodes, values, x):
        T = driver.T
        seg_set = set(seg_nodes)
        ext = driver.external_consumed(seg_nodes)
        depth = self.segment_depths(driver, seg_nodes)
        zeros = self._segment_output_zeros(driver, seg_nodes, values, x)

        boundary_trains: dict = {}
        n_cycles = T + max(depth.values(), default=0)

        def boundary_spikes(src, k, consumer):
            scale = self.consumer_wire_scale(driver, consumer)
            scale_key = (
                float(scale.float().mean())
                if isinstance(scale, torch.Tensor) else float(scale)
            )
            key = (src, scale_key)
            train = boundary_trains.get(key)
            if train is None:
                train = self._boundary_single_spike_train(values[src], scale, T)
                boundary_trains[key] = train
            return train[k]

        def read(src, out, perc_prev, t, consumer):
            if src not in seg_set:
                if is_encoding_perceptron(consumer):
                    return values[src]
                # Window-relative arrival, mirroring the executor's per-core input realignment.
                return boundary_spikes(src, t - depth[consumer], consumer)
            if perceptron_of(src) is not None:
                return perc_prev.get(src, zeros[src])
            return out[src]

        watch = list(ext)
        if self.node_value_recorder is not None:
            ext_set = set(ext)
            watch += [
                n for n in seg_nodes
                if perceptron_of(n) is not None and n not in ext_set
            ]

        accum: dict = {}
        latched: dict = {}
        perc_prev: dict = {}
        for t in range(n_cycles):
            out: dict = {}
            for n in seg_nodes:
                # Latency-gated: a core integrates only inside its window [depth, depth+T).
                if t < depth[n] or t >= depth[n] + T:
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
                if depth[n] <= t < depth[n] + T:
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
                        values[n] if n in ext else _decode(n)
                    ).detach()
        if driver._output in seg_set:
            return values[driver._output]
        return None
