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

STARVED_MASS_WARN = 0.5
"""A6(i): warn when most of a hop's positive activation mass sits under one grid step."""

_DEAD_DRIVE_DELAY = 1e6
"""A6(ii): a hop with no drive never fires — its first-fire delay saturates."""


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

    @property
    def starved(self) -> bool:
        return (
            self.median_effective_levels < MIN_MEDIAN_EFFECTIVE_LEVELS
            or self.starved_mass > STARVED_MASS_WARN
        )


def hop_value_gauge(
    *,
    name: str,
    depth: int,
    per_channel_q99: Sequence[float],
    positive_values: Sequence[float],
    theta: float,
    levels: int,
) -> HopValueGauge:
    return HopValueGauge(
        name=str(name),
        depth=int(depth),
        theta=float(theta),
        median_effective_levels=median_effective_levels(per_channel_q99, theta, levels),
        starved_mass=starved_mass(positive_values, theta, levels),
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
        return self.total_delay >= self.window


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


def gauge_summary(gauge: Optional[ValueInstallGauge]) -> dict:
    """The cache-friendly summary consumed by downstream install policies."""
    if gauge is None:
        return {
            "levels": None,
            "fails": False,
            "starved_hops": [],
            "median_effective_levels": [],
        }
    return {
        "levels": int(gauge.levels),
        "fails": bool(gauge.fails),
        "starved_hops": [h.name for h in gauge.starved_hops],
        "median_effective_levels": [
            float(h.median_effective_levels) for h in gauge.hops
        ],
    }


def emit_value_gauge(context: str, gauge: ValueInstallGauge) -> None:
    """One loud ``[MBH-A6]`` verdict line + one line per starved hop (warn-only)."""
    starved = gauge.starved_hops
    verdict = "FAIL" if gauge.fails else "PASS"
    print(
        f"[MBH-A6] kind=value context={context} levels={gauge.levels} "
        f"hops={len(gauge.hops)} starved_hops={len(starved)} verdict={verdict} "
        f"(pre-flight, warn-only)",
        flush=True,
    )
    for hop in starved:
        print(
            f"[MBH-A6]   starved hop={hop.name} depth={hop.depth} "
            f"theta={hop.theta:.4f} median_levels={hop.median_effective_levels:.2f} "
            f"starved_mass={hop.starved_mass:.2f}",
            flush=True,
        )


def emit_temporal_gauge(context: str, gauge: TemporalWindowGauge) -> None:
    verdict = "FAIL" if gauge.fails else "PASS"
    delays = ", ".join(f"{d:.2f}" for d in gauge.per_depth_delays)
    print(
        f"[MBH-A6] kind=temporal context={context} "
        f"total_first_fire_delay={gauge.total_delay} window={gauge.window} "
        f"verdict={verdict} per_depth=[{delays}] (pre-flight, warn-only)",
        flush=True,
    )


def emit_chain_gauge(
    context: str, *, max_intra_segment_depth: int, s: int, n_segments: int
) -> None:
    """The cascaded install's chain line: hop depth is the compounding exponent."""
    print(
        f"[MBH-A6] kind=chain context={context} "
        f"max_intra_segment_depth={max_intra_segment_depth} S={s} "
        f"spike_segments={n_segments} (pre-flight, warn-only)",
        flush=True,
    )
