"""Per-mode segment-execution policies for the unified segment-forward driver."""

from __future__ import annotations

import torch

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.spiking.compute_boundary import normalize_boundary_value
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.spiking.segment_policy_ttfs import TtfsSegmentPolicy
from mimarsinan.spiking.spike_trains import uniform_spike_train

__all__ = ["AnalyticalSegmentPolicy", "LifSegmentPolicy", "TtfsSegmentPolicy"]


def _safe_scale(scale, ref: torch.Tensor):
    if isinstance(scale, torch.Tensor):
        return scale.to(device=ref.device, dtype=ref.dtype).clamp(min=1e-12)
    return max(float(scale), 1e-12)


def _absolute_value_nodes(driver) -> dict:
    """node -> True when its stored value is ABSOLUTE (raw): unarmed host /
    structural chains rooted at the input. Wire nodes (neural producers,
    armed ComputeOps, and their passthroughs) re-encode by clamp alone;
    mixed wire/absolute fan-in at a plain host op fails loud (calculus §11.2)."""
    flags: dict = {}
    for node in driver._exec:
        deps = driver._deps.get(node, [])
        if perceptron_of(node) is not None:
            flags[node] = False
        elif isinstance(node, ComputeOpMapper) and node.output_scale is not None:
            flags[node] = False
        elif not deps:
            flags[node] = True
        else:
            dep_flags = {flags[dep] for dep in deps}
            if len(dep_flags) > 1:
                raise NotImplementedError(
                    "mixed wire/absolute fan-in at a plain host op is not "
                    f"supported ({type(node).__name__})"
                )
            flags[node] = dep_flags.pop()
    return flags


class LifSegmentPolicy:
    """Signed-IF cascade: perceptrons run per-cycle off upstream trains; entry
    (encoding) perceptrons run once on the decoded value and emit the uniform
    wire train of ``clamp(value / theta)`` (the deployed boundary contract).

    ``retime=True`` is the [C3/R5] twin-side per-hop re-encode: every hop's
    emitted train is replaced by the uniform re-encode of its window count
    (count-preserving, ``round((c/T)*T) = c``), matching a deployment mapped
    with per-hop neural segments (``lif_per_hop_retiming``)."""

    # LIF host values travel in the wire domain, so the driver must run host
    # value nodes through the emitted ScaleNormalizingWrapper composition
    # whenever the wrap slots are armed (per-channel theta): the deployed sim
    # decodes ``wire * theta_c`` per channel there and the twin must match.
    wire_domain_host_values = True

    _boundary_scales: dict | None = None
    _absolute_nodes: dict | None = None

    def __init__(self, retime: bool = False, phase_dither: bool = False,
                 synchronized: bool = False):
        self.retime = bool(retime)
        self.phase_dither = bool(phase_dither)
        # [calculus §16] two-window discipline: every hop computes ONCE on the
        # count-decoded values (the strict staircase); no per-cycle loop.
        self.synchronized = bool(synchronized)

    def prepare(self, driver):
        from spikingjelly.activation_based import functional

        from mimarsinan.spiking.scale_aware_boundaries import (
            read_boundary_out_scales,
            stamped_input_boundary_scale,
        )

        for p in driver.repr.get_perceptrons():
            functional.reset_net(p)
        self._set_all_cycle_accurate(driver, False)
        self._boundary_scales = read_boundary_out_scales(
            driver.repr,
            input_data_scale=stamped_input_boundary_scale(driver.repr),
        )
        self._absolute_nodes = _absolute_value_nodes(driver)

    def finalize(self, driver):
        self._set_all_cycle_accurate(driver, False)
        self._boundary_scales = None
        self._absolute_nodes = None

    @staticmethod
    def _lif_of(perceptron):
        return unwrap_lif_activation(getattr(perceptron, "activation", None))

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

    @staticmethod
    def _record_decoded_value(driver, perceptron, value):
        """[§16 sync] same side-channel, from the already-decoded value."""
        recorder = getattr(driver, "_node_value_recorder", None)
        if recorder is not None and perceptron is not None:
            recorder[id(perceptron)] = value.detach()

    def run_segment(self, driver, seg_nodes, values, x):
        from spikingjelly.activation_based import functional

        T = driver.T
        deps_map = driver._deps
        seg_set = set(seg_nodes)
        boundary_scales = self._boundary_scales
        assert boundary_scales is not None, "prepare() must run before run_segment()"
        node_train: dict = {}
        node_rate: dict = {}

        def rate_of(dep):
            return node_rate[dep] if dep in seg_set else values[dep]

        absolute_nodes = self._absolute_nodes
        assert absolute_nodes is not None, "prepare() must run before run_segment()"

        def value_of(dep):
            """[§16 sync] the count-decoded mean of ``train_of(dep)`` without
            materializing the train: same grid (round to counts), same
            absolute/wire dispatch, same producer out-scale."""
            t = node_train.get(dep)
            if t is not None:
                return t.mean(dim=0)
            value = rate_of(dep)
            scale = boundary_scales.get(dep, 1.0)
            if absolute_nodes.get(dep, False):
                rate = normalize_boundary_value(value, scale)
            else:
                rate = value.clamp(0.0, 1.0)
            return torch.round(rate.clamp(0.0, 1.0) * T) / T * scale

        def train_of(dep):
            """Per-cycle train for ``dep``; encode (uniform, clamped) if only a rate exists.

            A rate-only boundary re-encode is value-domain: ``uniform(rate) *
            producer out-scale`` — the deployed IR fold bakes the same scale into
            the consumer's weights (the W1c t0_03 host-op-boundary contract).
            ABSOLUTE (raw, unarmed-chain) producers transcode through the SSOT
            divide-first ``normalize_boundary_value``; wire producers are
            already normalized and only clamp (calculus §11.2).
            """
            t = node_train.get(dep)
            if t is not None:
                return t
            value = rate_of(dep)
            scale = boundary_scales.get(dep, 1.0)
            if absolute_nodes.get(dep, False):
                rate = normalize_boundary_value(value, scale)
            else:
                rate = value.clamp(0.0, 1.0)
            t = uniform_spike_train(rate, T, phase_dither=self.phase_dither)
            if scale != 1.0:
                t = t * scale
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
                    if lif is not None:
                        lif.set_cycle_accurate(False)
                    rate_out = forward_node(node, [rate_of(dep) for dep in d])
                    rate_norm = (rate_out / scale).clamp(0.0, 1.0)
                    node_rate[node] = rate_norm
                    # Mirror of encode_compute_boundary: the deployed boundary is
                    # a uniform wire train; *scale keeps NF value-domain magnitudes.
                    node_train[node] = uniform_spike_train(
                        rate_norm, T, phase_dither=self.phase_dither,
                    ) * scale
                elif self.synchronized:
                    # [calculus §16] two-window discipline: the hop's count is a
                    # function of input counts alone, so it computes ONCE on the
                    # count-decoded values — the strict staircase (the LIF
                    # multi-step forward), bit-equal to the per-cycle
                    # integrate-then-emit chip execution (locked by
                    # test_synchronized_rate).
                    assert lif is not None, (
                        "LifSegmentPolicy: non-encoding perceptron must carry a LIF activation"
                    )
                    lif.set_cycle_accurate(False)
                    functional.reset_net(lif.if_node)
                    out_val = forward_node(node, [value_of(dep) for dep in d])
                    rate_norm = (out_val / scale).clamp(0.0, 1.0)
                    node_rate[node] = rate_norm
                    if node is driver._output:
                        # Only the output needs a train (value-scaled logits);
                        # skipping the rest keeps every host node single-call.
                        node_train[node] = uniform_spike_train(
                            rate_norm, T, phase_dither=self.phase_dither,
                        ) * scale
                    self._record_decoded_value(driver, p, out_val)
                else:
                    assert lif is not None, (
                        "LifSegmentPolicy: non-encoding perceptron must carry a LIF activation"
                    )
                    lif.set_cycle_accurate(True)
                    functional.reset_net(lif.if_node)
                    dep_trains = [train_of(dep) for dep in d]
                    outs = [forward_node(node, [dt[t] for dt in dep_trains]) for t in range(T)]
                    lif.set_cycle_accurate(False)
                    train = torch.stack(outs, dim=0)
                    if self.retime:
                        # STE re-encode: forward = the deployed uniform train of
                        # the window count; backward = the raw cascade's per-cycle
                        # surrogate path (a hard re-encode severs every hop's
                        # gradient — the boundary-grad-severance failure mode).
                        retimed = uniform_spike_train(
                            (train / scale).mean(dim=0).clamp(0.0, 1.0).detach(), T,
                            phase_dither=self.phase_dither,
                        ) * scale
                        train = retimed.detach() + (train - train.detach())
                    node_train[node] = train
                    node_rate[node] = (train / scale).mean(dim=0)
                if node in node_train:
                    self._record_decoded(driver, p, node_train[node])
            else:
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
