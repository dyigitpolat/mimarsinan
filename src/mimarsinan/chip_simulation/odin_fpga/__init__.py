"""The physical ODIN backend: one device seam, two transports, one program."""

from mimarsinan.chip_simulation.odin_fpga.payload import (
    payload_bytes,
    program_payload,
    program_plan,
    run_plan,
    split_payloads,
    stimulus_ops,
)
from mimarsinan.chip_simulation.odin_fpga.records import (
    BACKEND_NAME,
    OdinFpgaRunRecord,
    OdinSegmentTiming,
    aggregate_walls,
)
from mimarsinan.chip_simulation.odin_fpga.transport import (
    DeviceSession,
    DeviceTransport,
    DeviceTransportError,
    ProgramReceipt,
    TransportRun,
)
from mimarsinan.chip_simulation.odin_fpga.cosim_transport import RtlCosimTransport
from mimarsinan.chip_simulation.odin_fpga.factory import TRANSPORTS, build_transport
from mimarsinan.chip_simulation.odin_fpga.runner import OdinFpgaRunner
from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    OdinFpgaCaptureTruncated,
    OdinFpgaKernelError,
    OdinFpgaProgramTooLarge,
)
from mimarsinan.chip_simulation.odin_fpga.xrt_transport import (
    OdinFpgaDependencyError,
    XrtTransport,
)

__all__ = [
    "BACKEND_NAME",
    "DeviceSession",
    "DeviceTransport",
    "DeviceTransportError",
    "OdinFpgaRunRecord",
    "OdinFpgaRunner",
    "OdinSegmentTiming",
    "OdinFpgaCaptureTruncated",
    "OdinFpgaDependencyError",
    "OdinFpgaKernelError",
    "OdinFpgaProgramTooLarge",
    "ProgramReceipt",
    "RtlCosimTransport",
    "TRANSPORTS",
    "TransportRun",
    "XrtTransport",
    "aggregate_walls",
    "build_transport",
    "payload_bytes",
    "program_payload",
    "program_plan",
    "run_plan",
    "split_payloads",
    "stimulus_ops",
]
