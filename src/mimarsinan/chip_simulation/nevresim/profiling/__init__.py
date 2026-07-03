"""Synthetic nevresim compile profiling (sweeps, metrics, timing records)."""

from mimarsinan.chip_simulation.nevresim.profiling.compile_profile import (
    NevresimCompileProfile,
    correlate_compile_vs_metric,
    write_profile_rows,
)
from mimarsinan.chip_simulation.nevresim.profiling.mapping_metrics import (
    MappingConnectivityMetrics,
    metrics_from_chip_model,
    metrics_from_hardcore_mapping,
)
from mimarsinan.chip_simulation.nevresim.profiling.profile_runner import profile_mapping_compile
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import (
    build_multi_segment_fanout,
    build_synthetic_mapping,
)

__all__ = [
    "NevresimCompileProfile",
    "correlate_compile_vs_metric",
    "write_profile_rows",
    "MappingConnectivityMetrics",
    "metrics_from_chip_model",
    "metrics_from_hardcore_mapping",
    "profile_mapping_compile",
    "build_multi_segment_fanout",
    "build_synthetic_mapping",
]
