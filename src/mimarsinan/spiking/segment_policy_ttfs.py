"""Single-spike TTFS segment policy for the unified segment-forward driver."""

from __future__ import annotations

import torch

from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of


class TtfsSegmentPolicy:
    """Single-spike TTFS: latency-windowed segment sim with arrival latch and
    ramp decode; entry (encoding) perceptrons fire from the decoded value.

    ``node_value_recorder``: optional dict; when set, every perceptron node's
    decoded value (latch-accumulated over its own window, scaled like a segment
    output) is stored per node — the NF side of the cascaded NF↔SCM parity
    comparison and the per-layer bisect instrument.
    """

    node_value_recorder: dict | None = None
    # Straight-through surrogate temperature for the offload-boundary re-encode.
    # None (default) = the historical contract: the round-based re-encode severs
    # the genuine backward at offload/host-ComputeOp boundaries, so only the last
    # neural segment trains on the deployed cascade. A float >0 enables an STE
    # whose forward is the bit-exact hard train and whose backward flows through a
    # soft spike-time, so the genuine cascade gradient reaches EVERY segment.
    # Forward (hence NF↔SCM parity and deployed accuracy) is unchanged either way.
    boundary_surrogate_temp: float | None = None

    def prepare(self, driver):
        # The walk feeds each node norm(W s_t + b) per cycle; the additive
        # constant of that pre-activation is the norm-folded effective bias,
        # so install it (differentiable, recomputed fresh every drive — no
        # stale references across layer-replacing steps).
        from mimarsinan.models.perceptron_mixer.perceptron import (
            effective_preactivation_bias,
        )

        for p, mods in self._ttfs_perceptrons(driver._seg_of.keys()):
            bias = effective_preactivation_bias(p)
            for m in mods:
                m.set_cycle_accurate(True)
                m.set_bias(bias)

    def finalize(self, driver):
        # Restore the raw layer.bias reference: the picklable stored contract
        # (a drive-time effective bias may carry an autograd graph).
        for p, mods in self._ttfs_perceptrons(driver._seg_of.keys()):
            bias = getattr(p.layer, "bias", None)
            for m in mods:
                m.set_cycle_accurate(False)
                m.reset_state()
                m.set_bias(bias)

    @staticmethod
    def _ttfs_perceptrons(nodes):
        from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

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

    def _boundary_single_spike_train(self, value, T: int, n_cycles: int) -> torch.Tensor:
        """Single-spike TTFS train of a decoded boundary value, at the rising
        edge of HCM's latched encode (``spike_time = round(T(1 - clamp(v)))``).
        The cascade neuron's ramp reconstruction turns the single spike into the
        same linear membrane growth HCM's greedy core gets from the latch.

        The forward train is the exact hard (``round``-based) encode. When
        ``boundary_surrogate_temp`` is set, a straight-through estimator attaches a
        backward path through a soft spike-time so the genuine cascade gradient
        reaches the upstream segment; the forward value is byte-identical."""
        from mimarsinan.chip_simulation.recording import spike_modes

        v = value.clamp(0.0, 1.0)
        latched = torch.stack(
            [spike_modes.to_spikes(v, c, simulation_length=T, spike_mode="TTFS")
             for c in range(n_cycles)],
            dim=0,
        ).to(value.dtype)
        train_hard = torch.cat([latched[:1], latched[1:] - latched[:-1]], dim=0)

        temp = self.boundary_surrogate_temp
        if temp is None or not torch.is_grad_enabled():
            return train_hard
        return self._straight_through(train_hard, v, T, n_cycles, float(temp))

    @staticmethod
    def _straight_through(train_hard, v, T, n_cycles, temp):
        """STE: forward == ``train_hard`` exactly; backward flows through a soft
        spike-time. The hard latch is a Heaviside in cycle index at
        ``tau = T(1 - v)`` (the spike has arrived by cycle ``c`` iff ``c >= tau``);
        softening it to ``sigmoid((c - tau)/temp)`` makes ``d train / d v`` nonzero
        (a larger ``v`` lowers ``tau``, so the spike arrives earlier and the
        downstream ramp integrates a higher decoded value — the correct sign)."""
        cycles = torch.arange(n_cycles, device=v.device, dtype=v.dtype)
        cycles = cycles.reshape((n_cycles,) + (1,) * v.dim())
        tau = float(T) * (1.0 - v)
        latched_soft = torch.sigmoid((cycles - tau) / temp)
        train_soft = torch.cat(
            [latched_soft[:1], latched_soft[1:] - latched_soft[:-1]], dim=0
        )
        return train_hard.detach() + (train_soft - train_soft.detach())

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
            for n in watch:                             # per-source window [lat, lat+T)
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
