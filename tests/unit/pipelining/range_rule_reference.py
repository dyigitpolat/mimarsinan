"""[U-BRANGE, U-BRANGE2] An independent reference of the range rule, for the tests: every candidate's squared error read
directly from the grid itself (``quantize_to_value_grid`` or ``quantize_to_lane_grid`` on the seam's values, in
float64), never from the toolchain's own search.

The rule ([U-BRANGE2 (c)]): the candidates are the whole interval up to the calibration maximum, ``RULE_PER_OCTAVE``
geometric spans per octave over ``RULE_OCTAVES`` octaves below the maximum (each an fp32 value, the maximum exact),
written out here; the chosen scale has the least squared error of the site's own grid over the calibration batches,
rounding and clamp both counted; ties go to the least scale.

The former rule (revision 1's interval and R5's scale) is kept for the restore tests only: the largest per-batch 0.999
quantile of the magnitudes (``interval``, ``former_rule``), read here, never by the toolchain.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, List, Sequence, Tuple

import torch

from mimarsinan.models.nn.activations.lane_quantization import quantize_to_lane_grid
from mimarsinan.models.nn.activations.readout_quantizer import snap_channel_groups
from mimarsinan.models.nn.activations.value_quantizer import quantize_to_value_grid

RULE_QUANTILE = 0.999
RULE_CANDIDATES = 257
RULE_PER_OCTAVE = 256
RULE_OCTAVES = 12
#: The relative slack of an error comparison between the toolchain's search and this direct reading.
RELATIVE = 1e-6

Snap = Callable[[torch.Tensor, float], torch.Tensor]


def signed_grid(bits: int) -> Snap:
    """The symmetric signed grid of ``bits``, the boundary SSOT (on the magnitudes, by its odd symmetry)."""
    return lambda values, scale: quantize_to_value_grid(values, float(scale), bits)


def lane_grid(bits: int) -> Snap:
    """The unsigned lane grid of ``bits`` at offset 0."""
    return lambda values, scale: quantize_to_lane_grid(values, float(scale), bits, 0.0)


def exact_quantile(values: torch.Tensor, q: float) -> float:
    """The quantile of ``values`` by a full sort in float64 with torch's linear interpolation."""
    ordered = torch.sort(values.reshape(-1).double()).values
    rank = q * (ordered.numel() - 1)
    below = int(rank)
    above = min(below + 1, ordered.numel() - 1)
    return float(ordered[below] + (ordered[above] - ordered[below]) * (rank - below))


def interval(magnitudes: Sequence[torch.Tensor], q: float = RULE_QUANTILE) -> Tuple[float, float]:
    """The former rule's interval: the largest per-batch ``q`` quantile (``torch.quantile``'s own reading of these
    small seams) and the largest magnitude, as fp32 values."""
    low = max(float(torch.quantile(m.reshape(-1).float(), q)) for m in magnitudes)
    high = max(float(m.max()) for m in magnitudes)
    return float(torch.tensor(low, dtype=torch.float32)), float(torch.tensor(high, dtype=torch.float32))


def candidates(low: float, high: float, count: int = RULE_CANDIDATES) -> List[float]:
    """The former rule's candidates: ``count`` even fp32 scales from ``low`` to ``high``."""
    if not high > low:
        return [low]
    scales = torch.linspace(low, high, count, dtype=torch.float32).double().tolist()
    scales[0], scales[-1] = low, high
    return scales


def whole_interval(high: float) -> List[float]:
    """The rule's candidates: ``high 2^(-k / RULE_PER_OCTAVE)`` for ``k`` from ``RULE_OCTAVES x RULE_PER_OCTAVE`` down
    to 0, each an fp32 value, the maximum exact, ascending and distinct."""
    values = set()
    for k in range(RULE_OCTAVES * RULE_PER_OCTAVE, 0, -1):
        values.add(float(torch.tensor(float(high) * 2.0 ** (-k / RULE_PER_OCTAVE), dtype=torch.float64)
                         .to(torch.float32)))
    values.add(float(torch.tensor(float(high), dtype=torch.float32)))
    return sorted(values)


def grid_error(magnitudes: Sequence[torch.Tensor], scale: float, snap: Snap) -> float:
    """The grid's squared error over every batch, in float64."""
    total = 0.0
    for m in magnitudes:
        v = m.reshape(-1).double()
        total += float(((snap(v, scale) - v) ** 2).sum())
    return total


def reference_errors(magnitudes: Sequence[torch.Tensor], snap: Snap) -> Tuple[List[float], List[float]]:
    """``(candidates, errors)`` of the rule's search over the whole interval, every error read directly from the grid."""
    scales = whole_interval(max(float(m.max()) for m in magnitudes))
    return scales, [grid_error(magnitudes, s, snap) for s in scales]


def least_error(magnitudes: Sequence[torch.Tensor], snap: Snap) -> float:
    return min(reference_errors(magnitudes, snap)[1])


def levels_grid(levels: int) -> Snap:
    """The symmetric grid of ``levels`` positive levels (a consumer group's grid, ``snap_channel_groups``'s form)."""

    def snap(values: torch.Tensor, scale: float) -> torch.Tensor:
        ranges = torch.tensor([float(scale)], dtype=values.dtype)
        return snap_channel_groups(values.reshape(-1, 1), ranges, torch.tensor([int(levels)]), -1).reshape(values.shape)
    return snap


def former_rule(rule, pooled: bool = False):
    """A ``least_squares_ranges`` (``pooled``: a ``least_squares_pooled``) whose every census takes the former rule's
    scale: each site's largest per-batch 0.999 quantile of its magnitudes, read here by hooks on the sites' modules
    (R5's measurement; the former rule read no union), never by the toolchain. The restore tests read the earlier
    programs byte for byte through it."""

    def restored(model, sites, *args, **kwargs):
        batches = list(args[1] if pooled else args[0])
        rest = ((args[0], batches) + tuple(args[2:])) if pooled else ((batches,) + tuple(args[1:]))
        census = rule(model, sites, *rest, **kwargs)
        lows = [0.0] * len(sites)
        handles = []
        for i, site in enumerate(sites):
            def hook(module, hook_args, output=None, i=i, site=site):
                value = output if site.reads == "output" else hook_args[0]
                magnitudes = site.magnitudes(value.detach())
                if magnitudes.numel():
                    lows[i] = max(lows[i], float(torch.quantile(magnitudes.reshape(-1).float(), RULE_QUANTILE)))
            handles.append(site.module.register_forward_hook(lambda m, a, o, h=hook: h(m, a, o))
                           if site.reads == "output" else site.module.register_forward_pre_hook(hook))
        try:
            with torch.no_grad():
                for x in batches:
                    model(x)
        finally:
            for handle in handles:
                handle.remove()
        return [dataclasses.replace(row, scale=low) for row, low in zip(census, lows)]

    return restored
