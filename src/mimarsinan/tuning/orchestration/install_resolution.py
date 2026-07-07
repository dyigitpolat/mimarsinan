"""[MBH-A6] install-resolution pre-flight gauges: static, warn-only (5v)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from mimarsinan.chip_simulation.spiking_semantics import (
    forces_activation_quantization,
    is_lif,
)
from mimarsinan.tuning.orchestration.install_capture import (
    ChannelStatsAccumulator as ChannelStatsAccumulator,
    attach_activation_decorator as attach_activation_decorator,
    collect_channel_stats as collect_channel_stats,
)

MIN_MEDIAN_EFFECTIVE_LEVELS = 2.0
"""A6(i): a hop needs a median of >= 2 usable grid levels per live channel."""

NEAREST_MIN_MEDIAN_EFFECTIVE_LEVELS = 1.0
"""A6(i) for nearest-rounding installs: rounding error is symmetric (<= half a
step both ways), so one representable level suffices — the tier-0.1 corpus
read FAIL on the passing dense ttfsq cells (t01_06 0.97, t01_22 0.99) at the
ceil-kernel threshold."""

STARVED_MASS_WARN = 0.5
"""A6(i): warn when most of a hop's positive activation mass sits under one grid step."""

TEMPORAL_RECOVERY_HEADROOM = 2.0
"""A6(ii): the T-anneal recovery family heals chains whose accumulated
first-fire delay sits below ~2x the window (tier-0.1 corpus: ratios 1.3-1.7
all recovered — t01_02 0.97, t01_16 0.99; ratios >= 3.3 all failed —
t01_01 0.91, t0_01 0.947)."""

PROVEN_RECOVERY_DEPTH = 6
"""Single-segment cascade chains below this depth climbed out of equally-deep
entry craters (t0_22/t0_18/t0_03 at L <= 4-5); at or past it the compounding
kernel binds (t01_12 L=9 read 0.88 with a clean value gauge)."""

_DEAD_DRIVE_DELAY = 1e6
"""A6(ii): a hop with no drive never fires — its first-fire delay saturates."""


def value_gauge_thresholds(spiking_mode: str) -> tuple:
    """(min_median_levels, starved_mass_warn | None) conditioned on the install
    kernel: nearest-rounding (ttfs_quantized) tolerates sub-step mass by
    construction; ceil/floor kernels keep the study thresholds."""
    if str(spiking_mode) == "ttfs_quantized":
        return (NEAREST_MIN_MEDIAN_EFFECTIVE_LEVELS, None)
    return (MIN_MEDIAN_EFFECTIVE_LEVELS, STARVED_MASS_WARN)


def median_effective_levels(
    per_channel_q99: Sequence[float], theta: float, levels: int
) -> float:
    """Median over LIVE channels of ``q99_c / (theta/levels)`` (dead channels are
    pruning artifacts and excluded; a hop with no live channel reads 0.0)."""
    delta = float(theta) / max(int(levels), 1)
    live = [float(q) for q in per_channel_q99 if float(q) > 0.0]
    if not live or delta <= 0.0:
        return 0.0
    return float(statistics.median(q / delta for q in live))


def starved_mass(positive_values: Sequence[float], theta: float, levels: int) -> float:
    """Fraction of positive activation mass below one grid step (empty -> 1.0)."""
    delta = float(theta) / max(int(levels), 1)
    values = [float(v) for v in positive_values if float(v) > 0.0]
    if not values:
        return 1.0
    return sum(1 for v in values if v < delta) / len(values)


@dataclass(frozen=True)
class HopValueGauge:
    """One hop's A6(i) value-kernel starvation read at the target grid."""

    name: str
    depth: int
    theta: float
    median_effective_levels: float
    starved_mass: float
    min_levels: float = MIN_MEDIAN_EFFECTIVE_LEVELS
    mass_warn: Optional[float] = STARVED_MASS_WARN

    @property
    def starved(self) -> bool:
        if self.median_effective_levels < self.min_levels:
            return True
        return self.mass_warn is not None and self.starved_mass > self.mass_warn


def hop_value_gauge(
    *,
    name: str,
    depth: int,
    per_channel_q99: Sequence[float],
    positive_values: Sequence[float],
    theta: float,
    levels: int,
    min_levels: float = MIN_MEDIAN_EFFECTIVE_LEVELS,
    mass_warn: Optional[float] = STARVED_MASS_WARN,
) -> HopValueGauge:
    return HopValueGauge(
        name=str(name),
        depth=int(depth),
        theta=float(theta),
        median_effective_levels=median_effective_levels(per_channel_q99, theta, levels),
        starved_mass=starved_mass(positive_values, theta, levels),
        min_levels=float(min_levels),
        mass_warn=mass_warn,
    )


@dataclass(frozen=True)
class ValueInstallGauge:
    """A6(i) verdict over every hop of an install at ``levels`` grid steps."""

    hops: Tuple[HopValueGauge, ...]
    levels: int

    @property
    def starved_hops(self) -> Tuple[HopValueGauge, ...]:
        return tuple(h for h in self.hops if h.starved)

    @property
    def fails(self) -> bool:
        return bool(self.starved_hops)


def first_fire_delay(*, theta: float, mean_drive: float) -> float:
    """A6(ii) per-hop integration delay estimate ``theta / <drive>`` in cycles."""
    drive = float(mean_drive)
    if drive <= 0.0:
        return _DEAD_DRIVE_DELAY
    return float(theta) / drive


@dataclass(frozen=True)
class TemporalWindowGauge:
    """A6(ii): the chain's accumulated first-fire delay against the window T."""

    window: int
    total_delay: float
    per_depth_delays: Tuple[float, ...]

    @property
    def fails(self) -> bool:
        # Corpus-conditioned: recovery heals below ~2x the window (tier-0.1).
        return self.total_delay >= TEMPORAL_RECOVERY_HEADROOM * self.window


def temporal_window_gauge(delays_by_depth, window: int) -> TemporalWindowGauge:
    """Sum the worst per-depth delay along the chain (``{depth: delay}`` mapping)."""
    per_depth = tuple(
        float(delays_by_depth[d]) for d in sorted(delays_by_depth)
    )
    return TemporalWindowGauge(
        window=int(window),
        total_delay=float(sum(per_depth)),
        per_depth_delays=per_depth,
    )


def needs_quantile_deflation(
    per_channel_q99: Sequence[float], theta: float, levels: int
) -> bool:
    """A6-driven quantile arbitration: a hop whose theta starves the grid
    (median effective levels < 2) must not keep the full-quantile scale."""
    return median_effective_levels(per_channel_q99, theta, levels) < (
        MIN_MEDIAN_EFFECTIVE_LEVELS
    )


def value_grid_levels(spiking_mode: str, config) -> Optional[int]:
    """The value-kernel grid an install will commit (``None`` = continuous).

    AQ-forcing modes install the ``target_tq`` staircase; LIF's deployed rate
    grid is ``theta/T``; analytical ``ttfs`` deploys continuous. The mode
    comes via the caller's ``DeploymentPlan``, never a raw config read.
    """
    if forces_activation_quantization(spiking_mode):
        return int(config["target_tq"])
    if is_lif(spiking_mode):
        return int(config["simulation_steps"])
    return None


def build_value_install_gauge(
    perceptron_stats: Sequence[tuple],
    thetas: Sequence[float],
    depths: dict,
    levels: int,
    *,
    min_levels: float = MIN_MEDIAN_EFFECTIVE_LEVELS,
    mass_warn: Optional[float] = STARVED_MASS_WARN,
) -> ValueInstallGauge:
    """Assemble the A6(i) gauge from per-perceptron channel stats + install thetas."""
    hops = []
    for (perceptron, acc), theta in zip(perceptron_stats, thetas):
        hops.append(hop_value_gauge(
            name=str(getattr(perceptron, "name", type(perceptron).__name__)),
            depth=int(depths.get(id(perceptron), 0)),
            per_channel_q99=acc.per_channel_q99(),
            positive_values=acc.positive_values(),
            theta=float(theta),
            levels=levels,
            min_levels=min_levels,
            mass_warn=mass_warn,
        ))
    return ValueInstallGauge(hops=tuple(hops), levels=int(levels))


def lif_temporal_gauge(
    perceptron_stats: Sequence[tuple],
    depths: dict,
    window: int,
) -> TemporalWindowGauge:
    """A6(ii) for the LIF lockstep kernel: worst per-depth first-fire delay summed.

    Encoding perceptrons are excluded (the analytic encoder uniform-encodes;
    its delay is 0 by construction in the current NF).
    """
    delays_by_depth: dict = {}
    for perceptron, acc in perceptron_stats:
        if getattr(perceptron, "is_encoding_layer", False):
            continue
        delay = first_fire_delay(
            theta=float(perceptron.activation_scale),
            mean_drive=acc.mean_positive(),
        )
        depth = int(depths.get(id(perceptron), 0))
        delays_by_depth[depth] = max(delays_by_depth.get(depth, 0.0), delay)
    return temporal_window_gauge(delays_by_depth, window=window)
