"""The Alveo transport: the real XRT session structure, lazily imported.

STRUCTURE, NOT SILICON. Every call this module makes against ``pyxrt`` exists in
``github.com/Xilinx/XRT`` branch ``2024.2``,
``src/python/pybind11/src/pyxrt.cpp`` — and NOTHING else does. In particular
that binding has no ``read_register``/``write_register``, so this session never
touches a control register: it introspects the xclbin's own metadata, opens the
device, resolves the kernel with EXCLUSIVE access, allocates one buffer object
per kernel argument in that argument's memory bank, DMAs the program and the
stimulus, POISONS the capture header with a no-verdict sentinel, starts the
kernel, waits, syncs the capture back and reads the fabric's verdict OUT OF
MEMORY. Its unit coverage drives that sequence through an injected fake.

The protocol itself lives in ``kernel_registers`` — this module is the session,
not the table.

``pyxrt`` ships with XRT and is not a pip dependency of this project: it is
imported inside the session seam only, and every entry refuses by name when it
is absent, so importing mimarsinan on a machine with no Alveo runtime works.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Sequence

from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    ARG_CAPTURE,
    ARG_PROGRAM,
    ARG_STIMULUS,
    DEFAULT_CAPTURE_EVENTS,
    KERNEL_NAME,
    WORD_BYTES,
    KernelCapacity,
    capture_buffer_bytes,
    capture_words,
    decode_capture,
    kernel_arity,
    memory_banks,
    no_verdict_header,
    require_declared_storage,
    require_kernel_in_xclbin,
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

#: 0 is pyxrt's "block until the run completes" (pyxrt.cpp binds wait() to
#: xrt::run::wait(0)); a positive value bounds the wait in milliseconds.
BLOCK_UNTIL_DONE_MS = 0


class OdinFpgaDependencyError(DeviceTransportError):
    """``pyxrt`` (the XRT Python binding) is not importable in this environment."""


def load_pyxrt() -> Any:
    """The XRT binding, or a refusal that names the dependency and the fix."""
    try:
        # [optional hardware dep] pyxrt ships with the Xilinx Runtime, never
        # with pip: a module-level import would make `import mimarsinan` fail
        # on every machine without an Alveo installation.
        import pyxrt  # type: ignore[import-not-found]  # ships with XRT, never pip
    except ImportError as exc:
        raise OdinFpgaDependencyError(
            "pyxrt is not importable, so no Alveo device can be reached "
            f"({exc}). pyxrt ships with the Xilinx Runtime, not with pip: "
            "source the XRT setup script on a board node (see "
            "scripts/hacc/RUNBOOK.md) before selecting "
            "odin_fpga_transport='xrt'. The RTL cosimulation transport runs "
            "the same program locally and needs no XRT.") from exc
    return pyxrt


def require_run_completed(xrt: Any, state: Any, *, transport: str) -> None:
    """``run.wait`` answers with an ``ert_cmd_state``; only one of them is a run."""
    completed = xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    if state != completed:
        raise DeviceTransportError(
            f"{transport}: the kernel run ended in state {state!r}, not "
            f"{completed!r}. XRT never saw ap_done, so nothing was captured and "
            f"there is no verdict in memory to decode")


class XrtTransport:
    """One XRT session against an ODIN kernel on an XDMA shell."""

    name = TRANSPORT_NAME

    def __init__(
        self,
        *,
        xclbin_path: str,
        device_index: int = 0,
        kernel_name: str = KERNEL_NAME,
        capture_events: int = DEFAULT_CAPTURE_EVENTS,
        capacity: KernelCapacity | None = None,
        run_timeout_ms: int = BLOCK_UNTIL_DONE_MS,
    ) -> None:
        if not xclbin_path:
            raise OdinFpgaDependencyError(
                "odin_fpga_transport='xrt' needs odin_fpga_xclbin_path: the "
                "bitstream is the device's program and there is no default one")
        self.xclbin_path = str(xclbin_path)
        self.device_index = int(device_index)
        self.kernel_name = str(kernel_name)
        self.capture_events = int(capture_events)
        self.run_timeout_ms = int(run_timeout_ms)
        self.capacity = capacity if capacity is not None else KernelCapacity()
        self._xrt: Any = None
        self._device: Any = None
        self._uuid: Any = None
        self._kernel: Any = None
        self._program_bo: Any = None
        self._receipt: ProgramReceipt | None = None
        self._export: Any = None
        self._capture_capacity = 0
        #: What the loaded xclbin declares about itself, for the run's detail.
        self.xclbin_kernels: Dict[str, int] = {}
        self.memory_banks: list = []

    # -- session ---------------------------------------------------------

    def open(self) -> None:
        require_declared_storage(self.capacity, transport=self.name)
        self._xrt = load_pyxrt()
        # The xclbin's own metadata, BEFORE it is pushed to a card: a bitstream
        # that does not declare this kernel is not the one this host drives.
        image = self._xrt.xclbin(self.xclbin_path)
        self.xclbin_kernels = kernel_arity(image.get_kernels())
        require_kernel_in_xclbin(
            self.xclbin_kernels, xclbin_path=self.xclbin_path,
            transport=self.name, kernel_name=self.kernel_name)
        self.memory_banks = memory_banks(image.get_mems())
        self._device = self._xrt.device(self.device_index)
        self._uuid = self._device.load_xclbin(image)
        self._kernel = self._xrt.kernel(
            self._device, self._uuid, self.kernel_name,
            # Exclusive access: the deployment owns the board for the whole
            # reservation, and a shared CU would let another job's run interleave
            # with this one's capture buffer.
            self._xrt.kernel.cu_access_mode.exclusive,
        )
        self._capture_capacity = self.capacity.ceiling(self.capture_events)

    @property
    def capture_capacity(self) -> int:
        """Events this session can decode: min(host declaration, declared RAM)."""
        return self._capture_capacity

    def close(self) -> None:
        self._program_bo = None
        self._kernel = None
        self._uuid = None
        self._device = None
        self._receipt = None
        self._export = None
        self._capture_capacity = 0
        self.xclbin_kernels = {}
        self.memory_banks = []

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
            self._device, int(nbytes), xrt.bo.flags.normal,
            self._kernel.group_id(int(arg)),
        )

    def _to_device(self, bo: Any, payload: bytes) -> None:
        xrt = self._require_session("host-to-device DMA")
        bo.write(payload, 0)
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(payload), 0)

    def _capture_buffer(self, nbytes: int) -> Any:
        """A capture buffer whose header carries the host's no-verdict sentinel.

        The fabric overwrites both header words at drain time, so a header that
        comes back untouched is the kernel refusing — the only `err` signal a
        host with no register access can see.
        """
        capture_bo = self._buffer(nbytes, ARG_CAPTURE)
        self._to_device(capture_bo, no_verdict_header())
        return capture_bo

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
        xrt = self._require_session("run_samples()")
        if self._export is None or self._receipt is None:
            raise OdinFpgaDependencyError(
                f"{self.name}: run_samples() before program() — the cores hold "
                f"no weights, so the counts would be another network's")
        program = program_plan(self._export)
        full = run_plan(self._export, per_cycle_inputs, latencies=latencies)
        stimulus = payload_bytes(stimulus_ops(program, full))
        program_words = len(self._receipt.payload) // WORD_BYTES
        stimulus_words = len(stimulus) // WORD_BYTES
        capture_bytes = capture_buffer_bytes(self._capture_capacity)

        started = time.monotonic()
        stim_bo = self._buffer(len(stimulus), ARG_STIMULUS)
        self._to_device(stim_bo, stimulus)
        capture_bo = self._capture_buffer(capture_bytes)
        assert self._kernel is not None
        run = self._kernel(
            self._program_bo, stim_bo, capture_bo,
            program_words, stimulus_words, self._capture_capacity,
        )
        require_run_completed(
            xrt, run.wait(self.run_timeout_ms), transport=self.name)
        capture_bo.sync(
            xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, capture_bytes, 0)
        words = capture_words(capture_bo.read(capture_bytes, 0))
        wall = time.monotonic() - started

        events, device_cycles = decode_capture(
            words, self._capture_capacity, transport=self.name)
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
                "capacity": self.capacity.as_dict(),
                "xclbin_kernels": dict(self.xclbin_kernels),
                "memory_banks": list(self.memory_banks),
                "events_seen": int(words[0]),
            },
        )
