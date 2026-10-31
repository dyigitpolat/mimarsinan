"""The currencies the LIF NF walk reads its host values in (U-SC): absolute, wire, a MIXED layout gather, and the
raw-gauge reads the count-grid rule governs.

``value_domain_map`` classifies every node; gauges are scalar per producer, as the boundary walk's are by contract
(``read_boundary_out_scales``). A MIXED node (a concatenation of an absolute host chain with wire core
outputs, DenseNet-BC's published order) holds each element in its producer's currency, which is what the deployed state
buffers hold, so a domain-owning wrapper reading it decodes every element at its producer's gauge unchanged; a neural
consumer re-encodes each element by its producer's rule (``mixed_wire_rate``), from the per-element currency the walk
records when the gather runs (``gather_currency``). The raw-gauge reads mirror the deployed seam
``count_grid.host_op_for_evaluation`` on the mapper graph (``count_grid_forwards``).
"""

from __future__ import annotations

import functools

import torch

from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.value_domain import MIXED, value_domain_map
from mimarsinan.spiking.count_grid import CountGridModule, CountGridOperandError, count_grid_applies
from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of
from mimarsinan.spiking.spike_trains import uniform_spike_train
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

__all__ = [
    "MIXED",
    "HostCurrencyHooks",
    "MixedGatherDitherError",
    "absolute_value_nodes",
    "consumer_fold",
    "count_grid_forwards",
    "currency_readers",
    "gather_currency",
    "mixed_events",
    "mixed_wire_rate",
]


class MixedGatherDitherError(ToolchainRefusal, NotImplementedError):
    """A phase-dithered re-encode of a mixed-currency gather: the deployed comb keys its rotation per producer slice, and
    the twin does not guess that layout."""


def absolute_value_nodes(driver) -> dict:
    """node -> True when its stored value is ABSOLUTE (raw): unarmed host / structural chains rooted at the input; False
    for wire nodes (neural producers, armed ComputeOps, and their passthroughs), which re-encode by clamp alone; MIXED
    for a layout gather of both. A mixed fan-in at any other plain host op fails loud (calculus §11.2) — the
    gauge-establishment seam repairs it before the walk, or it is a genuinely unsupported graph."""
    flags, joins = value_domain_map(driver._exec, driver._deps)
    if joins:
        raise NotImplementedError(
            "mixed wire/absolute fan-in at a plain host op is not "
            f"supported ({type(joins[0]).__name__})"
        )
    return flags


def gather_currency(node, deps, dep_values, flags: dict, boundary_scales: dict, recorded: dict, *, phase_dither: bool):
    """``(absolute mask, re-encode scale)`` per element of a MIXED layout node: its layout applied to each source's own
    currency (a recorded MIXED source passes its own through)."""
    if phase_dither:
        raise MixedGatherDitherError(
            f"{type(node).__name__} {getattr(node, 'name', None)!r} gathers absolute and wire sources under a "
            "phase-dithered encode: the deployed comb rotates per producer slice, which this twin does not reproduce")
    masks, scales = [], []
    for dep, value in zip(deps, dep_values):
        if flags.get(dep) == MIXED:
            mask, scale = recorded[dep]
        else:
            mask = torch.full_like(value, 1.0 if flags.get(dep) else 0.0)
            scale = torch.full_like(value, float(boundary_scales.get(dep, 1.0)))
        masks.append(mask)
        scales.append(scale)
    if len(deps) == 1:
        return node.forward(masks[0]), node.forward(scales[0])
    return node.forward(tuple(masks)), node.forward(tuple(scales))


def mixed_wire_rate(value: torch.Tensor, currency, *, clamp: bool = True):
    """``(wire rate, re-encode scale)`` of a MIXED value: absolute elements through the divide-first transcode at their
    producer's scale, wire elements as they are, each then clamped as a direct read of its producer is."""
    mask, scale = currency
    absolute = mask > 0.5
    divided = value / scale.clamp(min=1e-12)
    rate = torch.where(absolute, divided.clamp(0.0, 1.0) if clamp else divided, value)
    return (rate.clamp(0.0, 1.0) if clamp else rate), scale


def mixed_events(value: torch.Tensor, currency, simulation_length: int, *, train=None) -> torch.Tensor:
    """The per-cycle spike multiplicities a MIXED input delivers: its gathered train divided back by each element's
    producer scale (exact integers), or, for a host-side gather, the uniform encode of each element's wire rate."""
    if train is not None:
        return torch.round(train / currency[1].clamp(min=1e-12))
    return uniform_spike_train(mixed_wire_rate(value, currency)[0], simulation_length)


def consumer_fold(perceptron, like: torch.Tensor):
    """The fold a deployed core applies to its input spikes (``per_input_scales``, viewed as the weight's input axis):
    a neural consumer of a MIXED gather reads its spikes at this currency, as the deployed core does."""
    fold = PerceptronTransformer().input_scale(perceptron)
    return fold.to(device=like.device, dtype=like.dtype) if isinstance(fold, torch.Tensor) else float(fold)


def _encoding_raw_state(node):
    perceptron = perceptron_of(node)
    assert perceptron is not None, "an encoding node carries a perceptron"
    theta = torch.as_tensor(perceptron.activation_scale).detach().to(torch.float64).reshape(-1)
    if theta.numel() > 1 and not torch.allclose(theta, theta[0].expand_as(theta)):
        return "mixed"
    return None if abs(float(theta.mean()) - 1.0) < 1e-12 else "pure"


def _count_grid_forward(node, simulation_length: int, x):
    # The NF holds raw-gauge values as wire rates (the encoding's ``clamp(value / theta)``): unit divisors.
    module = CountGridModule(node.module, in_divisor=1.0, out_divisor=1.0, simulation_length=simulation_length)
    return node._forward_with_module(module, x)


def count_grid_forwards(driver, simulation_length: int) -> dict:
    """node -> its count-grid forward, for every plain non-commuting host op whose sources are all raw-gauge: a host-run
    spiking perceptron at a non-unit theta (the subsumed encoding layer), or plain ops over such buffers only. The same
    decision the deployed seam takes on the divisor table; a mixed read refuses by name on both sides."""
    state: dict = {}
    forwards: dict = {}
    for node in driver._exec:
        deps = driver._deps.get(node, [])
        if is_encoding_perceptron(node):
            state[node] = _encoding_raw_state(node)
            continue
        armed = isinstance(node, ComputeOpMapper) and node.output_scale is not None
        if perceptron_of(node) is not None or armed or not deps:
            state[node] = None
            continue
        dep_states = {state.get(d) for d in deps}
        if isinstance(node, ComputeOpMapper) and count_grid_applies(getattr(node, "module", None)):
            if dep_states == {None}:
                state[node] = None
                continue
            if dep_states != {"pure"} or node.output_value_offset is not None:
                raise CountGridOperandError(
                    f"host op {getattr(node, 'name', None)!r} ({type(node.module).__name__}) reads raw-gauge buffers "
                    "beside buffers at other gauges, at a per-channel gauge, or with a signed-seam offset: no one "
                    "window-count grid holds its operands")
            state[node] = "pure"
            forwards[node] = functools.partial(_count_grid_forward, node, int(simulation_length))
            continue
        state[node] = "pure" if dep_states == {"pure"} else (None if dep_states == {None} else "mixed")
    return forwards


def currency_readers(driver, flags: dict) -> set:
    """The MIXED nodes whose per-element currency some reader reads: a neural consumer, a MIXED layout node that is
    itself read so, or a MIXED layout node inside a neural segment (it encodes each source by that source's own rule,
    whoever reads it). A host op reads a gather's buffers as they are stored and needs none."""
    consumers: dict = {}
    for node in driver._exec:
        for dep in driver._deps.get(node, []):
            consumers.setdefault(dep, []).append(node)
    produces = driver._produces
    read: set = set()
    for node in reversed(driver._exec):
        if flags.get(node) == MIXED and any(
            perceptron_of(c) is not None or c in read or (flags.get(c) == MIXED and produces.get(c, False))
            for c in consumers.get(node, [])
        ):
            read.add(node)
    return read


class HostCurrencyHooks:
    """The LIF policy's host-value hooks (U-SC): the count-grid forwards (S-2) and the MIXED gathers' currencies (S-3).

    The host requires ``_absolute_nodes``, ``_boundary_scales`` and ``phase_dither``, set by its ``prepare``."""

    _absolute_nodes: dict | None
    _boundary_scales: dict | None
    phase_dither: bool
    _mixed_currency: dict | None = None
    _count_grid: dict | None = None

    _currency_read: set | None = None

    def _prepare_currencies(self, driver) -> None:
        self._mixed_currency = {}
        self._count_grid = count_grid_forwards(driver, driver.T)
        self._currency_read = currency_readers(driver, self._absolute_nodes or {})

    def _clear_currencies(self) -> None:
        self._mixed_currency = None
        self._count_grid = None
        self._currency_read = None

    def host_value_forward(self, driver, node):
        """The count-grid forward of a plain non-commuting op over raw-gauge values, else None."""
        return (self._count_grid or {}).get(node)

    def note_value_node(self, driver, node, values) -> None:
        """Record a host-side MIXED gather's per-element currency while its sources are still held."""
        if self._absolute_nodes is not None and self._absolute_nodes.get(node) == MIXED:
            deps = driver._deps.get(node, [])
            self._record_mixed(node, deps, [values[d] for d in deps])

    def _record_mixed(self, node, deps, dep_values) -> None:
        assert self._absolute_nodes is not None and self._boundary_scales is not None and self._mixed_currency is not None
        if node not in (self._currency_read or ()):
            return  # no reader needs it (host ops decode the buffers as stored), so no per-element currency is held
        self._mixed_currency[node] = gather_currency(
            node, deps, dep_values, self._absolute_nodes, self._boundary_scales, self._mixed_currency,
            phase_dither=self.phase_dither,
        )

    def _hop_inputs(self, perceptron, deps, rate_of, train_of, node_train, node_events, T):
        """A streamed hop's input trains and events: a MIXED source delivers each element's spikes at the consumer core's
        fold (as the deployed core reads them); every other source its own train."""
        assert self._absolute_nodes is not None and self._mixed_currency is not None
        trains, events = [], []
        for dep in deps:
            if self._absolute_nodes.get(dep) == MIXED:
                ev = mixed_events(rate_of(dep), self._mixed_currency[dep], T, train=node_train.get(dep))
                trains.append(ev * consumer_fold(perceptron, ev))
                events.append(ev)
            else:
                trains.append(train_of(dep))
                events.append(node_events.get(dep))
        return trains, events
