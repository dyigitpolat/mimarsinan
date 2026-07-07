"""[MBH-A6] install-resolution pre-flight gauges: capture → math → report.

One warn-only concept in three layers: `capture` (channel-resolved activation
stats at the install anchor, cursor/RNG-isolated), `gauges` (the static A6
value/temporal/chain starvation math with its corpus-conditioned thresholds),
and `report` (the loud ``[MBH-A6]`` emit + cache-summary seams).
"""

from mimarsinan.tuning.orchestration.install_resolution.capture import (
    ChannelStatsAccumulator,
    attach_activation_decorator,
    capture_install_stats,
    collect_channel_stats,
)
from mimarsinan.tuning.orchestration.install_resolution.gauges import (
    MIN_MEDIAN_EFFECTIVE_LEVELS,
    NEAREST_MIN_MEDIAN_EFFECTIVE_LEVELS,
    PROVEN_RECOVERY_DEPTH,
    STARVED_MASS_WARN,
    TEMPORAL_RECOVERY_HEADROOM,
    HopValueGauge,
    TemporalWindowGauge,
    ValueInstallGauge,
    build_value_install_gauge,
    first_fire_delay,
    hop_value_gauge,
    lif_temporal_gauge,
    median_effective_levels,
    needs_quantile_deflation,
    starved_mass,
    temporal_window_gauge,
    value_gauge_thresholds,
    value_grid_levels,
)
from mimarsinan.tuning.orchestration.install_resolution.report import (
    chain_gauge_fails,
    emit_chain_gauge,
    emit_temporal_gauge,
    emit_value_gauge,
    gauge_summary,
)

__all__ = [
    "ChannelStatsAccumulator",
    "HopValueGauge",
    "MIN_MEDIAN_EFFECTIVE_LEVELS",
    "NEAREST_MIN_MEDIAN_EFFECTIVE_LEVELS",
    "PROVEN_RECOVERY_DEPTH",
    "STARVED_MASS_WARN",
    "TEMPORAL_RECOVERY_HEADROOM",
    "TemporalWindowGauge",
    "ValueInstallGauge",
    "attach_activation_decorator",
    "build_value_install_gauge",
    "capture_install_stats",
    "chain_gauge_fails",
    "collect_channel_stats",
    "emit_chain_gauge",
    "emit_temporal_gauge",
    "emit_value_gauge",
    "first_fire_delay",
    "gauge_summary",
    "hop_value_gauge",
    "lif_temporal_gauge",
    "median_effective_levels",
    "needs_quantile_deflation",
    "starved_mass",
    "temporal_window_gauge",
    "value_gauge_thresholds",
    "value_grid_levels",
]
