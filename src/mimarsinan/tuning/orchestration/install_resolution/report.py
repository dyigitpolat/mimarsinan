"""[MBH-A6] gauge emit/report seams (loud, warn-only)."""

from __future__ import annotations

from typing import Optional

from mimarsinan.tuning.orchestration.install_resolution.gauges import (
    PROVEN_RECOVERY_DEPTH,
    TemporalWindowGauge,
    ValueInstallGauge,
)


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


def chain_gauge_fails(*, max_intra_segment_depth: int, n_segments: int) -> bool:
    """A6 chain verdict: a SINGLE-segment cascade whose hop chain reaches the
    proven-recovery depth binds on the compounding kernel (corpus: t01_12's
    L=9 read 0.88 with a clean value gauge; L <= 4-5 chains all recovered)."""
    return (
        int(n_segments) == 1
        and int(max_intra_segment_depth) + 1 >= PROVEN_RECOVERY_DEPTH
    )


def emit_chain_gauge(
    context: str, *, max_intra_segment_depth: int, s: int, n_segments: int
) -> None:
    """The cascaded install's chain line: hop depth is the compounding exponent."""
    verdict = "FAIL" if chain_gauge_fails(
        max_intra_segment_depth=max_intra_segment_depth, n_segments=n_segments,
    ) else "PASS"
    print(
        f"[MBH-A6] kind=chain context={context} "
        f"max_intra_segment_depth={max_intra_segment_depth} S={s} "
        f"spike_segments={n_segments} verdict={verdict} (pre-flight, warn-only)",
        flush=True,
    )
