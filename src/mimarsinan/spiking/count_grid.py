"""The count-grid rule: ONE boundary transcode for a plain host op that reads a raw-gauge buffer (U-SC S-2).

A host-run spiking perceptron (a subsumed encoding layer) stores its raw value, ``count / T * theta``, in the deployed
state buffer, and a plain host op over it inherits that raw gauge (``boundary_normalization_scales``), while the NF holds
the same values as wire rates, ``count / T``. An op that commutes exactly with a positive division (a selection or a
layout move) computes equal floats in both gauges. An averaging or linear op does not: its result can sit exactly on a
half count (the mean of four integer counts), and each gauge's float error then decides the comb's rounding on its own
side (SD-10c, the ``pool_join_avg`` fixtures). Both twins therefore evaluate such an op through ``CountGridModule``:
decode the buffer once to its exact window counts, evaluate the op in fp64 on that integer grid (exact for a mean, and
order-free for a linear map of small integers), re-encode through the comb rule (``comb_round``, the chip's tie rule),
and return the counts at the op's own gauge, so every later read decodes the same integers. The NF passes unit divisors
(it holds rates); the deployed executors pass the divisor table (``host_op_for_evaluation``). Exactly commuting ops and
domain-owning wrappers (which already read every source at the wire gauge, ``host_op_input_state``) are left as they are.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.recording.spike_modes import comb_round
from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.common.refusal.toolchain_refusal import ToolchainRefusal
from mimarsinan.mapping.ir.gather_plan import gather_plan_for
from mimarsinan.mapping.support.activation_scales import perceptron_wrapped_activation_scale
from mimarsinan.mapping.support.compute_modules import ComputeAdapter, ScaleNormalizingWrapper
from mimarsinan.mapping.support.value_domain import op_preserves_wire_ratio

__all__ = [
    "CountGridModule",
    "CountGridOperandError",
    "comb_round",
    "count_grid_applies",
    "decode_window_counts",
    "host_op_for_evaluation",
    "op_commutes_exactly",
]

#: The largest distance from the integer grid a decoded window count may carry (float32 buffers hold about 1e-6).
_GRID_TOL = 1e-3

# A selection or a layout move of one element: f(x / d) == f(x) / d in float, bit for bit, for every d > 0.
_EXACT_MODULES: tuple[type, ...] = (
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.Identity, nn.Flatten,
)
_EXACT_ADAPTER_FNS = frozenset({"amax", "amin", "flatten", "reshape", "permute", "transpose", "cat"})


class CountGridOperandError(ToolchainRefusal, ValueError):
    """A count-grid read without one exact operand domain: a buffer off the window-count grid, or raw-gauge buffers beside
    others (or at different or per-channel gauges) at one plain non-commuting op."""


def op_commutes_exactly(module) -> bool:
    """Whether a host op commutes bit for bit with a positive division (a selection or a layout move)."""
    if isinstance(module, ComputeAdapter):
        return getattr(module.fn, "__name__", "") in _EXACT_ADAPTER_FNS
    return isinstance(module, _EXACT_MODULES)


def count_grid_applies(module) -> bool:
    """A plain op the rule governs: positively homogeneous (so the scale policy leaves it unarmed, computing on whatever
    gauge its sources carry) and not exactly commuting (an average, a mean, a sum, a bias-free linear map)."""
    return op_preserves_wire_ratio(module) and not op_commutes_exactly(module)


def _straight_through(value: torch.Tensor, exact: torch.Tensor) -> torch.Tensor:
    return value + (exact - value).detach()


def decode_window_counts(value: torch.Tensor, divisor: float, simulation_length: int) -> torch.Tensor:
    """The exact window counts a buffer holds at the gauge ``divisor`` (fp64; the backward passes straight through)."""
    counts = value.to(torch.float64) * (float(simulation_length) / float(divisor))
    exact = torch.round(counts.detach())
    off = float((counts.detach() - exact).abs().max()) if counts.numel() else 0.0
    if off > _GRID_TOL:
        raise CountGridOperandError(
            f"a raw-gauge buffer read by a non-commuting host op is off the window-count grid (|count - round| up to "
            f"{off:.3g} at T {simulation_length}, gauge {float(divisor):.6g}): only a spiking producer's counts, or a "
            "count-grid op's result, decode exactly")
    return _straight_through(counts, exact)


def call_in_fp64(module: nn.Module, *args, **kwargs):
    """``module`` evaluated in fp64 on fp64 operands, its own parameters and buffers untouched (and differentiable)."""
    state = {name: t.to(torch.float64) for name, t in module.named_parameters()}
    state.update({name: (t.to(torch.float64) if t.is_floating_point() else t) for name, t in module.named_buffers()})
    return torch.func.functional_call(module, state, args, kwargs)


class CountGridModule(nn.Module):
    """A plain host op evaluated on the count grid: decode at ``in_divisor``, fp64 op, comb re-encode, ``out_divisor``."""

    def __init__(self, module: nn.Module, *, in_divisor: float, out_divisor: float, simulation_length: int):
        super().__init__()
        self.module = module
        self.in_divisor = float(in_divisor)
        self.out_divisor = float(out_divisor)
        self.simulation_length = int(simulation_length)

    def forward(self, *inputs, **kwargs):
        T = self.simulation_length
        counts = [decode_window_counts(x, self.in_divisor, T) for x in inputs]
        result = call_in_fp64(self.module, *counts, **kwargs)
        if not isinstance(result, torch.Tensor):
            raise CountGridOperandError(
                f"the count-grid rule re-encodes one tensor, and {type(self.module).__name__} returned "
                f"{type(result).__name__}")
        counts_out = _straight_through(result, comb_round(result.detach()))
        return (counts_out * (self.out_divisor / T)).to(inputs[0].dtype)


def _scalar_divisor(divisor, what: str) -> float:
    if isinstance(divisor, np.ndarray) and divisor.size > 1 and not np.allclose(divisor, divisor.reshape(-1)[0]):
        raise CountGridOperandError(
            f"{what} carries a per-channel raw gauge; the count-grid rule reads one gauge per buffer")
    return float(np.mean(divisor))


def host_op_for_evaluation(op, wire_divisors, *, simulation_length: int, spiking_mode: str):
    """The ComputeOp a LIF executor evaluates: ``op`` itself, or, for a plain non-commuting op whose every source is a
    raw-gauge buffer, a twin of it whose module runs on the count grid at the divisor table's gauges."""
    if not wire_divisors or not is_lif(str(spiking_mode)) or int(op.id) not in wire_divisors:
        return op
    module = (op.params or {}).get("module")
    if (module is None or isinstance(module, ScaleNormalizingWrapper)
            or perceptron_wrapped_activation_scale(module) is not None or not count_grid_applies(module)):
        return op
    sources = [int(n) for n in gather_plan_for(op).referenced_node_ids if int(n) >= 0]
    raw = [n for n in sources if n in wire_divisors]
    if not raw:
        return op
    if len(raw) != len(sources):
        raise CountGridOperandError(
            f"host op {op.name!r} ({type(module).__name__}) reads raw-gauge buffers {raw} beside buffers "
            f"{[n for n in sources if n not in wire_divisors]} at other gauges: no one window-count grid holds both")
    gauges = {_scalar_divisor(wire_divisors[n], f"buffer {n}") for n in raw}
    if len(gauges) != 1:
        raise CountGridOperandError(
            f"host op {op.name!r} reads raw-gauge buffers at different gauges {sorted(gauges)}")
    out_divisor = _scalar_divisor(wire_divisors[int(op.id)], f"host op {op.name!r}")
    params = {**op.params, "module": CountGridModule(
        module, in_divisor=gauges.pop(), out_divisor=out_divisor, simulation_length=simulation_length)}
    return dataclasses.replace(op, params=params)
