"""The Alveo U55C transport: the real XRT session structure, lazily imported.

STRUCTURE, NOT SILICON. Every call this module makes against ``pyxrt`` is the
call the board needs — open the device, load the xclbin, resolve the kernel,
read back the geometry the bitstream was compiled with, allocate one buffer
object per kernel argument in that argument's memory bank, DMA the program and
the stimulus, start the kernel, wait, read the status register, sync the capture
back — and its unit coverage drives that sequence through an injected fake. What
the tests prove is the CALL CONTRACT (order, sizes, bytes, refusals); what only
P7b can prove is that a real U55C answers it.

The register map itself lives in ``kernel_registers`` — this module is the
session, not the table.

``pyxrt`` ships with XRT and is not a pip dependency of this project: it is
imported inside the session seam only, and every entry refuses by name when it
is absent, so importing mimarsinan on a machine with no Alveo runtime works.
"""

from __future__ import annotations

import importlib.util
import time
from typing import Any, Dict, Mapping, Sequence

from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    ADDR_CAPTURE_CAPACITY,
    ADDR_PROGRAM_CAPACITY,
    ADDR_STATUS,
    ARG_CAPTURE,
    ARG_PROGRAM,
    ARG_STIMULUS,
    CTRL_OFFSET,
    DEFAULT_CAPTURE_EVENTS,
    KERNEL_NAME,
    WORD_BYTES,
    capture_buffer_bytes,
    capture_words,
    decode_capture,
    require_no_kernel_error,
    require_program_fits,
)
from mimarsinan.chip_simulation.odin_fpga.payload import (
    program_plan,
    payload_bytes,
    run_plan,
    stimulus_ops,
)
from mimarsinan.chip_simulation.odin_fpga.transport import (
    DeviceTransportError,
    ProgramReceipt,
    TransportRun,
)

TRANSPORT_NAME = "xrt"


class OdinFpgaDependencyError(DeviceTransportError):
    """``pyxrt`` (the XRT Python binding) is not importable in this environment."""


def _missing_reason(exc: BaseException) -> str:
    return (
        "pyxrt is not importable, so no Alveo device can be reached "
        f"({exc}). pyxrt ships with the Xilinx Runtime, not with pip: source "
        "the XRT setup script on a board node (see scripts/hacc/RUNBOOK.md) "
        "before selecting odin_fpga_transport='xrt'. The RTL cosimulation "
        "transport runs the same program locally and needs no XRT."
    )


def load_pyxrt() -> Any:
    """The XRT binding, or a refusal that names the dependency and the fix."""
    try:
        # [optional hardware dep] pyxrt ships with the Xilinx Runtime, never
        # with pip: a module-level import would make `import mimarsinan` fail
        # on every machine without an Alveo installation.
        import pyxrt  # type: ignore[import-not-found]  # ships with XRT, never pip
    except ImportError as exc:
        raise OdinFpgaDependencyError(_missing_reason(exc)) from exc
    return pyxrt


def pyxrt_available() -> bool:
    """Whether this environment can reach XRT at all (never raises)."""
    return importlib.util.find_spec("pyxrt") is not None


class XrtTransport:
    """One XRT session against an ODIN kernel on a U55C XDMA shell."""

    name = TRANSPORT_NAME

    def __init__(
        self,
        *,
        xclbin_path: str,
        device_index: int = 0,
        kernel_name: str = KERNEL_NAME,
        capture_events: int = DEFAULT_CAPTURE_EVENTS,
    ) -> None:
        if not xclbin_path:
            raise OdinFpgaDependencyError(
                "odin_fpga_transport='xrt' needs odin_fpga_xclbin_path: the "
                "bitstream is the device's program and there is no default one")
        self.xclbin_path = str(xclbin_path)
        self.device_index = int(device_index)
        self.kernel_name = str(kernel_name)
        self.capture_events = int(capture_events)
        self._xrt: Any = None
        self._device: Any = None
        self._uuid: Any = None
        self._kernel: Any = None
        self._program_bo: Any = None
        self._receipt: ProgramReceipt | None = None
        self._export: Any = None
        self._capture_capacity = 0
        self._program_capacity = 0

    # -- session ---------------------------------------------------------

    def open(self) -> None:
        self._xrt = load_pyxrt()
        self._device = self._xrt.device(self.device_index)
        self._uuid = self._device.load_xclbin(self._xrt.xclbin(self.xclbin_path))
        self._kernel = self._xrt.kernel(
            self._device, self._uuid, self.kernel_name,
            self._xrt.kernel.shared,
        )
        # The two read-only capacity registers: the fabric's RAM depths are
        # compile-time constants of the loaded xclbin, so the host asks the
        # bitstream instead of assuming what it was built with.
        fabric_events = int(self._kernel.read_register(ADDR_CAPTURE_CAPACITY))
        self._program_capacity = int(
            self._kernel.read_register(ADDR_PROGRAM_CAPACITY))
        self._capture_capacity = min(self.capture_events, fabric_events)
        if self._capture_capacity <= 0 or self._program_capacity <= 0:
            raise DeviceTransportError(
                f"{self.name}: the kernel reports a capture capacity of "
                f"{fabric_events} events and a program capacity of "
                f"{self._program_capacity} words — an xclbin built with no "
                f"storage cannot run anything, and reading it as zero would "
                f"turn every run into a silent empty one")

    @property
    def capture_capacity(self) -> int:
        """Events this session can decode: min(host declaration, fabric RAM)."""
        return self._capture_capacity

    @property
    def program_capacity(self) -> int:
        """Words the fabric's program RAM holds, as the bitstream reports it."""
        return self._program_capacity

    def close(self) -> None:
        self._program_bo = None
        self._kernel = None
        self._uuid = None
        self._device = None
        self._receipt = None
        self._export = None
        # The capacities belong to the bitstream that was loaded, not to this
        # object: a reopened session re-reads them rather than trusting them.
        self._capture_capacity = 0
        self._program_capacity = 0

    def _require_session(self, what: str) -> Any:
        if self._kernel is None or self._xrt is None:
            raise OdinFpgaDependencyError(
                f"{self.name}: {what} before a live session — open() loads the "
                f"xclbin and resolves the kernel, and nothing can be written to "
                f"a device that was never opened")
        return self._xrt

    def _buffer(self, nbytes: int, arg: int) -> Any:
        xrt = self._require_session("buffer allocation")
        assert self._kernel is not None and self._device is not None
        return xrt.bo(
            self._device, int(nbytes), xrt.bo.normal,
            self._kernel.group_id(int(arg)),
        )

    def _to_device(self, bo: Any, payload: bytes) -> None:
        xrt = self._require_session("host-to-device DMA")
        bo.write(payload, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(payload), 0)

    # -- the two device operations ---------------------------------------

    def program(self, export: Any) -> ProgramReceipt:
        self._require_session("program()")
        started = time.monotonic()
        plan = program_plan(export)
        payload = payload_bytes(plan.ops)
        self._program_bo = self._buffer(len(payload), ARG_PROGRAM)
        self._to_device(self._program_bo, payload)
        self._export = export
        self._receipt = ProgramReceipt(
            payload=payload, ops=len(plan.ops), cores=plan.n_cores,
            wall_s=time.monotonic() - started,
            basis=(
                "xrt: DMA of the programming payload plus the fabric "
                "sequencer's SPI shifting of every memory and register word"
            ),
        )
        return self._receipt

    def run_samples(
        self,
        per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
        *,
        latencies: Sequence[int],
    ) -> TransportRun:
        self._require_session("run_samples()")
        if self._export is None or self._receipt is None:
            raise OdinFpgaDependencyError(
                f"{self.name}: run_samples() before program() — the cores hold "
                f"no weights, so the counts would be another network's")
        program = program_plan(self._export)
        full = run_plan(self._export, per_cycle_inputs, latencies=latencies)
        stimulus = payload_bytes(stimulus_ops(program, full))
        program_words = len(self._receipt.payload) // WORD_BYTES
        stimulus_words = len(stimulus) // WORD_BYTES
        require_program_fits(
            program_words, stimulus_words, self._program_capacity,
            transport=self.name)
        capture_bytes = capture_buffer_bytes(self._capture_capacity)

        started = time.monotonic()
        stim_bo = self._buffer(len(stimulus), ARG_STIMULUS)
        self._to_device(stim_bo, stimulus)
        capture_bo = self._buffer(capture_bytes, ARG_CAPTURE)
        assert self._kernel is not None
        run = self._kernel(
            self._program_bo, stim_bo, capture_bo,
            program_words, stimulus_words, self._capture_capacity,
        )
        run.wait()
        status = self._kernel.read_register(ADDR_STATUS)
        require_no_kernel_error(status, transport=self.name)
        capture_bo.sync(
            self._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            capture_bytes, 0)
        words = capture_words(capture_bo.read(capture_bytes, 0))
        wall = time.monotonic() - started

        events, device_cycles = decode_capture(words, self._capture_capacity)
        counts: Dict[tuple, int] = {}
        for event in events:
            sample, cycle = full.decode_tag(event.tag)
            key = (sample, cycle, event.core, event.neuron)
            counts[key] = counts.get(key, 0) + 1
        return TransportRun(
            counts=counts, samples=full.samples,
            cycles_per_sample=full.cycles_per_sample,
            wall_s=wall, program_wall_s=self._receipt.wall_s,
            device_cycles=device_cycles,
            detail={
                "xclbin": self.xclbin_path,
                "kernel": self.kernel_name,
                "device_index": self.device_index,
                "stimulus_words": stimulus_words,
                "capture_events": len(events),
                "capture_capacity": self._capture_capacity,
                "program_capacity": self._program_capacity,
                "status": int(status),
                "ctrl": self._kernel.read_register(CTRL_OFFSET),
            },
        )
