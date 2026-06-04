"""Single-spike TTFS segment policy for the unified segment-forward driver."""

from __future__ import annotations

import torch

from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of


class TtfsSegmentPolicy:
    """Single-spike TTFS: latency-windowed segment sim with arrival latch and
    ramp decode; entry (encoding) perceptrons fire from the decoded value."""

    def prepare(self, driver):
        for m in self._ttfs_nodes(driver._seg_of.keys()):
            m.set_cycle_accurate(True)

    def finalize(self, driver):
        for m in self._ttfs_nodes(driver._seg_of.keys()):
            m.set_cycle_accurate(False)
            m.reset_state()

    @staticmethod
    def _ttfs_nodes(nodes):
        from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

        out = []
        for n in nodes:
            p = perceptron_of(n)
            if p is None:
                continue
            for m in p.modules():
                if isinstance(m, TTFSActivation):
                    out.append(m)
        return out

    def decode_scale(self, driver, node):
        """activation_scale of the perceptron that produced this region node's spikes."""
        p = perceptron_of(node)
        if p is not None:
            return p.activation_scale
        for d in driver._deps.get(node, []):
            if driver._produces.get(d, False):
                return self.decode_scale(driver, d)
        return 1.0

    def segment_depths(self, driver, seg_nodes):
        """Per-node cascade latency = perceptron-hops from the segment entry.

        Each perceptron core adds one cycle of propagation delay; transparent
        routing (reshape/permute/...) adds none. Matches ``ChipLatency`` within a
        segment (a core's latency = max source-core latency + 1).
        """
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

    def _segment_output_zeros(self, driver, seg_nodes, values, x):
        """Per-node output shapes (zero tensors) for not-yet-fired delayed sources."""
        seg_set = set(seg_nodes)
        nodes = self._ttfs_nodes(seg_nodes)
        for m in nodes:
            m.set_cycle_accurate(False)
        zeros: dict = {}
        with torch.no_grad():
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
                zeros[n] = torch.zeros_like(vmode[n])
        for m in nodes:
            m.set_cycle_accurate(True)
            m.reset_state()
        return zeros

    @staticmethod
    def _boundary_single_spike_train(value, T: int, n_cycles: int) -> torch.Tensor:
        """Single-spike TTFS train of a decoded boundary value, at the rising
        edge of HCM's latched encode (``spike_time = round(T(1 - clamp(v)))``).
        The cascade neuron's ramp reconstruction turns the single spike into the
        same linear membrane growth HCM's greedy core gets from the latch."""
        from mimarsinan.chip_simulation.recording import spike_modes

        v = value.clamp(0.0, 1.0)
        latched = torch.stack(
            [spike_modes.to_spikes(v, c, simulation_length=T, spike_mode="TTFS")
             for c in range(n_cycles)],
            dim=0,
        ).to(value.dtype)
        return torch.cat([latched[:1], latched[1:] - latched[:-1]], dim=0)

    def run_segment(self, driver, seg_nodes, values, x):
        T = driver.T
        seg_set = set(seg_nodes)
        ext = driver.external_consumed(seg_nodes)
        depth = self.segment_depths(driver, seg_nodes)
        zeros = self._segment_output_zeros(driver, seg_nodes, values, x)

        boundary_trains: dict = {}
        n_cycles = T + max(depth.values(), default=0)

        def boundary_spikes(src, t):
            train = boundary_trains.get(src)
            if train is None:
                train = self._boundary_single_spike_train(values[src], T, n_cycles)
                boundary_trains[src] = train
            return train[t]

        def read(src, out, perc_prev, t, consumer):
            if src not in seg_set:
                # A subsumed value-encoding entry charges the ideal value directly
                # (bias-mode-agnostic — it mirrors HCM's host ComputeOp); an offload/
                # interior cascade core reads the TTFS-encoded boundary train.
                if is_encoding_perceptron(consumer):
                    return values[src]                  # value->spike entry (ideal value)
                return boundary_spikes(src, t)          # spike entry: TTFS-encoded boundary
            if perceptron_of(src) is not None:
                return perc_prev.get(src, zeros[src])   # core output, 1-cycle delayed
            return out[src]                             # transparent routing, this cycle

        accum: dict = {}
        latched: dict = {}
        perc_prev: dict = {}
        for t in range(n_cycles):
            out: dict = {}
            for n in seg_nodes:
                # Latency-gated: a core integrates only inside its own window
                # [depth, depth+T); outside it emits nothing (no premature bias firing).
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
            for n in ext:                               # per-source window [lat, lat+T)
                if depth[n] <= t < depth[n] + T:
                    s = out[n]
                    latched[n] = s if n not in latched else torch.maximum(latched[n], s)
                    accum[n] = latched[n] if n not in accum else accum[n] + latched[n]

        for n in ext:
            scale = self.decode_scale(driver, n)
            sv = (scale.to(accum[n].device, accum[n].dtype)
                  if isinstance(scale, torch.Tensor) else float(scale))
            values[n] = (accum[n] / float(T)) * sv
        if driver._output in seg_set:
            return values[driver._output]
        return None
