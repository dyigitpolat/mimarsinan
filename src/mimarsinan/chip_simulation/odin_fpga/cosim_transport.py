"""The RTL cosimulation as a device: gate R11a's instrument behind the seam.

Nothing here re-derives a bit layout, an event order or a drain bound. The
transport hands the P5 harness the payloads and hands back the counts it
measured, so "the cosim is the first device" is a wiring claim, not a second
implementation of the chip.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from mimarsinan.chip_simulation.odin_fpga.payload import (
    counts_from_events,
    payload_bytes,
    program_plan,
    run_plan,
)
from mimarsinan.chip_simulation.odin_fpga.transport import (
    DeviceTransportError,
    ProgramReceipt,
    TransportRun,
)
from mimarsinan.chip_simulation.odin_rtl.capture import parse_capture
from mimarsinan.chip_simulation.odin_rtl.stimulus import op_summary, write_stimulus
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    SimulatorUnavailable,
    available_engine,
    build_testbench,
    run_testbench,
)

TRANSPORT_NAME = "rtl_cosim"

#: What the programming wall of a cosimulated device actually measures. The
#: board's own basis (SPI shifting at the configured clock) is a different
#: number and must never be read as this one.
PROGRAM_BASIS = (
    "rtl_cosim: host-side assembly of the programming token payload; the SPI "
    "shifting itself is simulated inside the run and reported in device cycles"
)


class RtlCosimTransport:
    """Program and run one export on the vendored RTL through the P5 harness."""

    name = TRANSPORT_NAME

    def __init__(
        self,
        *,
        engine: str | None = None,
        overlay: bool = False,
        workdir: Path | None = None,
        timeout_s: float = 3600.0,
    ) -> None:
        self._engine = engine
        self._overlay = bool(overlay)
        self._workdir = workdir
        self._timeout_s = float(timeout_s)
        self._export: Any = None
        self._receipt: ProgramReceipt | None = None
        self._opened = False

    def open(self) -> None:
        try:
            self._engine = self._engine or available_engine()
        except SimulatorUnavailable as exc:
            raise DeviceTransportError(
                f"the {self.name!r} device cannot be opened: {exc}") from exc
        self._opened = True

    def close(self) -> None:
        self._opened = False
        self._export = None
        self._receipt = None

    def _require_open(self, what: str) -> None:
        if not self._opened:
            raise DeviceTransportError(
                f"{self.name}: {what} before open() — the session that owns the "
                f"simulator engine was never acquired")

    def program(self, export: Any) -> ProgramReceipt:
        self._require_open("program()")
        started = time.monotonic()
        plan = program_plan(export)
        payload = payload_bytes(plan.ops)
        self._export = export
        self._receipt = ProgramReceipt(
            payload=payload, ops=len(plan.ops), cores=plan.n_cores,
            wall_s=time.monotonic() - started, basis=PROGRAM_BASIS,
        )
        return self._receipt

    def run_samples(
        self,
        per_cycle_inputs: Sequence[Sequence[Mapping[int, Sequence[int]]]],
        *,
        latencies: Sequence[int],
    ) -> TransportRun:
        self._require_open("run_samples()")
        if self._export is None or self._receipt is None:
            raise DeviceTransportError(
                f"{self.name}: run_samples() before program() — the cores hold "
                f"no weights, so the counts would be another network's")
        plan = run_plan(self._export, per_cycle_inputs, latencies=latencies)
        started = time.monotonic()
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(self._workdir) if self._workdir is not None else Path(scratch)
            root.mkdir(parents=True, exist_ok=True)
            stimulus = root / "odin_fpga_stim.hex"
            token_count = write_stimulus(stimulus, plan.ops)
            build = build_testbench(
                n_cores=plan.n_cores, token_count=token_count,
                overlay=self._overlay, engine=self._engine)
            run = run_testbench(build, stimulus, timeout_s=self._timeout_s)
        capture = parse_capture(run.stdout)
        return TransportRun(
            counts=counts_from_events(plan, capture.events),
            samples=plan.samples,
            cycles_per_sample=plan.cycles_per_sample,
            wall_s=time.monotonic() - started,
            program_wall_s=self._receipt.wall_s,
            device_cycles=int(capture.cycles),
            detail={
                "engine": build.engine,
                "build_seconds": build.build_seconds,
                "build_cached": build.cached,
                "sim_seconds": run.seconds,
                "tokens": token_count,
                "barrier_cycles": plan.barrier_cycles,
                "program_ops": self._receipt.ops,
                "program_summary": dict(op_summary(plan.ops)),
            },
        )
