"""Per-mode segment-execution policies for the unified segment-forward driver."""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.mappers.source_gauges import read_source_gauges
from mimarsinan.mapping.support.gauge_channels import gauge_axis, gauge_view
from mimarsinan.models.nn.activations.autograd import RoundedStaircaseFunction
from mimarsinan.models.spiking.serial import refuse_cycle_atomic_walk
from mimarsinan.spiking.compute_boundary import normalize_boundary_value
from mimarsinan.spiking.cycle_trains import SegmentReaders, encode_train, scale_train, train_mean
from mimarsinan.spiking.lif_utils import unwrap_lif_activation
from mimarsinan.spiking.segment_currency import (
    MIXED,
    HostCurrencyHooks,
    MixedGatherDitherError,
    absolute_value_nodes,
    consumer_fold,
    mixed_wire_rate,
)
from mimarsinan.spiking.segment_partition import perceptron_of
from mimarsinan.spiking.segment_policy_analytical import AnalyticalSegmentPolicy
from mimarsinan.spiking.segment_policy_lif_serial import run_streamed_lif_cycles
from mimarsinan.spiking.segment_policy_ttfs import TtfsSegmentPolicy

__all__ = ["AnalyticalSegmentPolicy", "LifSegmentPolicy", "MixedGatherDitherError", "TtfsSegmentPolicy"]


def _safe_scale(scale, ref: torch.Tensor):
    if isinstance(scale, torch.Tensor):
        return scale.to(device=ref.device, dtype=ref.dtype).clamp(min=1e-12)
    return max(float(scale), 1e-12)


class LifSegmentPolicy(HostCurrencyHooks):
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
    _source_gauges: dict | None = None
    # Diagnostic: seam terms the sync path skips ({"clamp","grid"}); None = deployed.
    _seam_ablation: tuple | None = None

    def __init__(self, retime: bool = False, phase_dither: bool = False,
                 synchronized: bool = False, *, soma_law: SomaLaw):
        self.retime = bool(retime)
        self.phase_dither = bool(phase_dither)
        # [§16] two-window: one staircase eval per hop, no per-cycle loop.
        self.synchronized = bool(synchronized)
        # The resolved soma point the twin must reproduce. Default-FREE: a
        # policy that silently assumed the default point is exactly how the
        # t0_54 NF ran the per-cycle law against a per-event deployment.
        self.soma_law = soma_law
        refuse_cycle_atomic_walk(
            soma_law, retime=self.retime, synchronized=self.synchronized,
        )

    def prepare(self, driver):
        from spikingjelly.activation_based import functional

        from mimarsinan.spiking.scale_aware_boundaries import (
            boundary_scales_for_walk,
        )

        for p in driver.repr.get_perceptrons():
            functional.reset_net(p)
        self._set_all_cycle_accurate(driver, False)
        # The per-event twin folds the mapper's effective weights, so it owes
        # the gauge-establishment seam its precondition; every default-point
        # walk keeps reading the pure table and mutating nothing.
        self._boundary_scales = boundary_scales_for_walk(
            driver.repr, establish=self.soma_law.is_per_event,
        )
        self._absolute_nodes = absolute_value_nodes(driver)
        self._source_gauges = read_source_gauges(driver.repr)
        self._prepare_currencies(driver)

    def _wire_scale(self, dep, scalar, value):
        """[U-SC2] A wire value's re-encode scale: its per-channel gauge per element where the source walk carries one
        (the aligned vector the deployed core folds), else the boundary table's scalar."""
        view = gauge_view((self._source_gauges or {}).get(dep), value, gauge_axis(dep))
        return scalar if view is None else view

    def finalize(self, driver):
        self._set_all_cycle_accurate(driver, False)
        self._boundary_scales = None
        self._source_gauges = None
        self._absolute_nodes = None
        self._clear_currencies()

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
            recorder[id(perceptron)] = train_mean(train).detach()

    @staticmethod
    def _record_decoded_value(driver, perceptron, value):
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
        # Per-cycle emission MULTIPLICITIES, the currency the per-event fold
        # consumes: a train is ``events * producer scale`` by construction.
        node_events: dict = {}
        # [U-SC3] trains are read per cycle where the walk streams (cycle_trains); one no later reader reads is dropped.
        readers = SegmentReaders(seg_nodes, deps_map, keep=(driver._output,))

        def rate_of(dep):
            return node_rate[dep] if dep in seg_set else values[dep]

        absolute_nodes = self._absolute_nodes
        mixed = self._mixed_currency
        assert absolute_nodes is not None and mixed is not None, "prepare() must run before run_segment()"

        def value_of(dep, consumer=None):
            """[§16 sync] the count-decoded value on the wire grid: same
            grid, dispatch, and producer out-scale; STE rounds so the
            exact-QAT walk trains through the boundary (n7 D3). A MIXED
            gather is read at its ``consumer``'s fold, as the core reads it."""
            t = node_train.get(dep)
            if t is not None and dep not in node_rate:
                return train_mean(t)
            value = rate_of(dep)
            scale = boundary_scales.get(dep, 1.0)
            abl = self._seam_ablation or ()
            if absolute_nodes.get(dep) == MIXED:
                rate, own = mixed_wire_rate(value, mixed[dep], clamp="clamp" not in abl)
                scale = own if consumer is None else consumer_fold(consumer, rate)
            elif absolute_nodes.get(dep, False):
                rate = (value / scale) if "clamp" in abl else (
                    normalize_boundary_value(value, scale)
                )
            else:
                rate, scale = value, self._wire_scale(dep, scale, value)
            if "clamp" not in abl:
                rate = rate.clamp(0.0, 1.0)
            if "grid" in abl:
                return rate * scale
            return RoundedStaircaseFunction.apply(rate, T) * scale

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
            if absolute_nodes.get(dep) == MIXED:
                rate, scale = mixed_wire_rate(value, mixed[dep])
            elif absolute_nodes.get(dep, False):
                rate = normalize_boundary_value(value, scale)
            else:
                rate, scale = value.clamp(0.0, 1.0), self._wire_scale(dep, scale, value)
            t = encode_train(rate, T, phase_dither=self.phase_dither, channels=self.dither_keys(dep, rate))
            node_events[dep] = t
            if isinstance(scale, torch.Tensor) or scale != 1.0:
                t = scale_train(t, scale)
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
                scale = _safe_scale(getattr(lif or p, "activation_scale", 1.0), x)
                if getattr(p, "is_encoding_layer", False):
                    if lif is not None:
                        lif.set_cycle_accurate(False)
                    rate_out = forward_node(node, [rate_of(dep) for dep in d])
                    rate_norm = (rate_out / scale).clamp(0.0, 1.0)
                    node_rate[node] = rate_norm
                    # Mirror of encode_compute_boundary: the deployed boundary is
                    # a uniform wire train; *scale keeps NF value-domain magnitudes.
                    node_events[node] = encode_train(rate_norm, T, phase_dither=self.phase_dither,
                                                     channels=self.dither_keys(node, rate_norm))
                    node_train[node] = scale_train(node_events[node], scale)
                elif self.synchronized:
                    # [§16] two-window discipline: ONE value eval per hop —
                    # LIF hops and their theorem-equal staircase QAT stand-ins
                    # both walk it (locked by test_synchronized_rate).
                    if lif is not None:
                        lif.set_cycle_accurate(False)
                        functional.reset_net(lif.if_node)
                    out_val = forward_node(node, [value_of(dep, p) for dep in d])
                    rate_norm = (out_val / scale).clamp(0.0, 1.0)
                    node_rate[node] = rate_norm
                    if node is driver._output:
                        # Wire-grid logits: the uniform train's mean, STE.
                        node_rate[node] = RoundedStaircaseFunction.apply(
                            rate_norm, T) * scale
                    self._record_decoded_value(driver, p, out_val)
                else:
                    assert lif is not None, (
                        "LifSegmentPolicy: non-encoding perceptron must carry a LIF activation"
                    )
                    dep_trains, dep_events = self._hop_inputs(
                        p, d, rate_of, train_of, node_train, node_events, T)
                    train, events = run_streamed_lif_cycles(
                        self, node=node, perceptron=p, lif=lif,
                        forward_node=forward_node, dep_trains=dep_trains,
                        dep_events=dep_events, T=T, scale=scale, keep_events=readers.read_later(node),
                    )
                    del dep_trains, dep_events
                    readers.read(node, node_train, node_events)  # the hop's inputs, before its own reduction
                    if events is not None:
                        node_events[node] = events
                    node_train[node] = train
                    node_rate[node] = (train / scale).mean(dim=0)
                if node in node_train:
                    self._record_decoded(driver, p, node_train[node])
            else:
                self._gather_node(node, d, forward_node, rate_of, train_of, node_train, node_events, node_rate)
            readers.read(node, node_train, node_events)
            readers.unread(node, node_train, node_events)

        for n in driver.external_consumed(seg_nodes):
            values[n] = node_rate[n]
        if driver._output in seg_set:
            train = node_train.get(driver._output)
            return train_mean(train) if train is not None else node_rate[driver._output]
        return None
