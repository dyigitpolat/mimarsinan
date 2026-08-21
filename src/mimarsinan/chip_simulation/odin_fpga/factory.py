"""Which device a run reaches: the declared transport, resolved from the config."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from mimarsinan.chip_simulation.odin_fpga.cosim_transport import (
    TRANSPORT_NAME as COSIM_TRANSPORT,
    RtlCosimTransport,
)
from mimarsinan.chip_simulation.odin_fpga.transport import DeviceTransportError
from mimarsinan.chip_simulation.odin_fpga.xrt_transport import (
    TRANSPORT_NAME as XRT_TRANSPORT,
    XrtTransport,
)

#: The declarable transports, default first (the registry renders this tuple).
TRANSPORTS: Tuple[str, ...] = (COSIM_TRANSPORT, XRT_TRANSPORT)


def build_transport(config: Mapping[str, Any]) -> Any:
    """The transport the config declares, refusing an unknown name by key."""
    name = str(config.get("odin_fpga_transport") or COSIM_TRANSPORT)
    if name == COSIM_TRANSPORT:
        return RtlCosimTransport()
    if name == XRT_TRANSPORT:
        return XrtTransport(
            xclbin_path=str(config.get("odin_fpga_xclbin_path") or ""),
            device_index=int(config.get("odin_fpga_device_index", 0) or 0),
        )
    raise DeviceTransportError(
        f"odin_fpga_transport={name!r} names no device; declared transports are "
        f"{', '.join(TRANSPORTS)}")
