"""Per-mode segment-execution policies for the unified segment-forward driver."""

from __future__ import annotations

import torch

from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.spiking.segment_policy_ttfs import TtfsSegmentPolicy

__all__ = ["AnalyticalSegmentPolicy", "LifSegmentPolicy", "TtfsSegmentPolicy"]


def _safe_scale(scale, ref: torch.Tensor):
    if isinstance(scale, torch.Tensor):
        return scale.to(device=ref.device, dtype=ref.dtype).clamp(min=1e-12)
    return max(float(scale), 1e-12)


class LifSegmentPolicy:
    """Signed-IF cascade: perceptrons run per-cycle off upstream trains; entry
    (encoding) perceptrons run once on the decoded value and uniform-encode."""

    def prepare(self, driver):
        from spikingjelly.activation_based import functional

        for p in driver.repr.get_perceptrons():
            functional.reset_net(p)
        self._set_all_cycle_accurate(driver, False)

    def finalize(self, driver):
        self._set_all_cycle_accurate(driver, False)

    @staticmethod
    def _lif_of(perceptron):
        from mimarsinan.spiking.lif_utils import unwrap_lif_activation

        return unwrap_lif_activation(getattr(perceptron, "activation", None))

    @staticmethod
    def _boundary_runs_per_cycle(node, lif) -> bool:
        """Mirror of HCM's ``encode_compute_boundary`` contract: a subsumed
        encoding boundary is cycle-emitted only for a plain LIF perceptron;
        wrapper mappers (Conv1D/Conv2D) stay rate-mode."""
        from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
        from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper

        return lif is not None and not isinstance(
            node, (Conv1DPerceptronMapper, Conv2DPerceptronMapper)
        )

    def _set_all_cycle_accurate(self, driver, mode: bool):
        for p in driver.repr.get_perceptrons():
            lif = self._lif_of(p)
            if lif is not None:
                lif.set_cycle_accurate(mode)

    @staticmethod
    def _record_decoded(driver, perceptron, train):
        """Side-channel: record a perceptron's decoded cascade value (the per-cycle
        train mean = rate*scale, in teacher-activation units) for DFQ calibration.
        Never affects the forward output."""
        recorder = getattr(driver, "_node_value_recorder", None)
        if recorder is not None and perceptron is not None:
            recorder[id(perceptron)] = train.detach().mean(dim=0)

    def run_segment(self, driver, seg_nodes, values, x):
        from spikingjelly.activation_based import functional
        from mimarsinan.spiking.spike_trains import uniform_spike_train

        T = driver.T
        deps_map = driver._deps
        seg_set = set(seg_nodes)
        node_train: dict = {}   # node -> (T, B, ...) per-cycle value (real, = spike*scale)
        node_rate: dict = {}    # node -> rate (real); the inter-stage value

        def rate_of(dep):
            return node_rate[dep] if dep in seg_set else values[dep]

        def train_of(dep):
            """Per-cycle train for ``dep``; encode (uniform, clamped) if only a rate exists."""
            t = node_train.get(dep)
            if t is not None:
                return t
            t = uniform_spike_train(rate_of(dep).clamp(0.0, 1.0), T)
            node_train[dep] = t
            return t

        def forward_node(node, inputs: list):
            d = deps_map.get(node, [])
            if len(d) == 0:
                return node.forward(x)
            if len(d) == 1:
                return node.forward(inputs[0])
            return node.forward(tuple(inputs))

        for node in seg_nodes:
            d = deps_map.get(node, [])
            p = perceptron_of(node)
            if p is not None:
                lif = self._lif_of(p)
                scale = _safe_scale(getattr(lif, "activation_scale", 1.0), x)
                if getattr(p, "is_encoding_layer", False):
                    # Subsumed host encoder: mirror HCM's two outputs — the host
                    # op's rate-mode *value* (ComputeOp consumers) and the
                    # cycle-emitted *train* (``encode_compute_boundary``; plain
                    # LIF perceptrons only — wrappers stay uniform-encoded).
                    if lif is not None:
                        lif.set_cycle_accurate(False)
                    rate_out = forward_node(node, [rate_of(dep) for dep in d])
                    rate_norm = (rate_out / scale).clamp(0.0, 1.0)
                    node_rate[node] = rate_norm
                    if self._boundary_runs_per_cycle(node, lif):
                        lif.set_cycle_accurate(True)
                        functional.reset_net(lif.if_node)
                        dep_trains = [train_of(dep) for dep in d]
                        outs = [
                            forward_node(node, [dt[t] for dt in dep_trains])
                            for t in range(T)
                        ]
                        lif.set_cycle_accurate(False)
                        node_train[node] = torch.stack(outs, dim=0)
                    else:
                        node_train[node] = uniform_spike_train(rate_norm, T) * scale
                else:
                    lif.set_cycle_accurate(True)
                    functional.reset_net(lif.if_node)
                    dep_trains = [train_of(dep) for dep in d]
                    outs = [forward_node(node, [dt[t] for dt in dep_trains]) for t in range(T)]
                    lif.set_cycle_accurate(False)
                    train = torch.stack(outs, dim=0)
                    node_train[node] = train
                    node_rate[node] = (train / scale).mean(dim=0)
                self._record_decoded(driver, p, node_train[node])
            else:
                # Structural (reshape / permute / concat): transparent. Carry a
                # train when every dep has one (per-cycle), and always carry a rate.
                if d and all(node_train.get(dep) is not None for dep in d):
                    dep_trains = [node_train[dep] for dep in d]
                    node_train[node] = torch.stack(
                        [forward_node(node, [dt[t] for dt in dep_trains]) for t in range(T)],
                        dim=0,
                    )
                node_rate[node] = forward_node(node, [rate_of(dep) for dep in d])

        for n in driver.external_consumed(seg_nodes):
            values[n] = node_rate[n]
        if driver._output in seg_set:
            train = node_train.get(driver._output)
            return train.mean(dim=0) if train is not None else node_rate[driver._output]
        return None


class AnalyticalSegmentPolicy:
    """Value-domain analytical execution: every node runs once on ideal values
    (the pointwise-analytical NF of ``ttfs`` / ``ttfs_quantized``)."""

    def prepare(self, driver):
        pass

    def finalize(self, driver):
        pass

    def run_segment(self, driver, seg_nodes, values, x):
        seg_set = set(seg_nodes)
        local: dict = {}

        def val_of(dep):
            return local[dep] if dep in seg_set else values[dep]

        for node in seg_nodes:
            d = driver._deps.get(node, [])
            if len(d) == 0:
                local[node] = node.forward(x)
            elif len(d) == 1:
                local[node] = node.forward(val_of(d[0]))
            else:
                local[node] = node.forward(tuple(val_of(dep) for dep in d))

        for n in driver.external_consumed(seg_nodes):
            values[n] = local[n]
        if driver._output in seg_set:
            return local[driver._output]
        return None
