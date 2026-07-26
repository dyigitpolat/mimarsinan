"""Value-domain (MVM) program execution: affine cores, host ops, no cycles."""

from mimarsinan.chip_simulation.value_run.value_execution import (
    run_neural_segment_values,
)
from mimarsinan.chip_simulation.value_run.value_flow import ValueHybridCoreFlow

__all__ = [
    "ValueHybridCoreFlow",
    "run_neural_segment_values",
]
