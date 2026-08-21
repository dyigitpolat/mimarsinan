"""RTL cosimulation of the vendored ODIN core: the instrument behind gate R11a."""

from mimarsinan.chip_simulation.odin_rtl.cosim import (
    CosimPlan,
    CosimResult,
    build_cosim_ops,
    run_cosim,
)
from mimarsinan.chip_simulation.odin_rtl.reference import (
    CycleTrace,
    ReferenceTraceError,
    gather_axon_counts,
    simulate_cycles,
)
from mimarsinan.chip_simulation.odin_rtl.capture import (
    CaptureResult,
    TestbenchFailure,
    parse_capture,
)
from mimarsinan.chip_simulation.odin_rtl.stimulus import (
    Op,
    StimulusError,
    decode_ops,
    encode_ops,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    SimulatorBuildError,
    SimulatorUnavailable,
    available_engine,
    build_testbench,
    unavailable_reason,
)

__all__ = [
    "CaptureResult",
    "CosimPlan",
    "CosimResult",
    "CycleTrace",
    "Op",
    "ReferenceTraceError",
    "SimulatorBuildError",
    "SimulatorUnavailable",
    "StimulusError",
    "TestbenchFailure",
    "available_engine",
    "build_cosim_ops",
    "build_testbench",
    "decode_ops",
    "encode_ops",
    "gather_axon_counts",
    "parse_capture",
    "run_cosim",
    "simulate_cycles",
    "unavailable_reason",
]
