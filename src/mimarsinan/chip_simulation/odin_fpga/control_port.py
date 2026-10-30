"""The wrapper's CONTROL PORT under the orderings AXI4-Lite allows a master.

On the card the kernel's ``s_axi_control`` sits behind the shell's AXI
interconnect, which crosses from the PCIe clock to the kernel clock with one
clock-crossing path PER CHANNEL, so a register write's address and data reach
the kernel apart and in either order. ``hw/tb/tb_odin_fpga_kernel_axil.v`` is a
master that uses those freedoms (W before AW, AW ahead of W, independent random
gaps, AW with W together; B and R back-pressured) against the wrapper and checks
that every argument register holds what was written to ITS address, with one
response per transaction. This module builds and runs it; the [slow] gate is
``tests/integration/test_odin_fpga_kernel_control_port.py``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_CAPTURE_WORDS,
)
from mimarsinan.chip_simulation.odin_fpga.kernel_sim import (
    SHIPPED_FIFO_WORDS,
    kernel_design_sources,
)
from mimarsinan.chip_simulation.odin_rtl.capture import TestbenchFailure
from mimarsinan.chip_simulation.odin_rtl.toolchain import build_testbench, run_testbench

#: The conformance testbench's top module.
KERNEL_AXIL_TB = "tb_odin_fpga_kernel_axil"

#: Its master's modes (``+mode``): W leads, AW leads, independent random gaps,
#: and AW with W together.
CONTROL_PORT_MODES = (0, 1, 2, 3)


@dataclass(frozen=True)
class ControlPortResult:
    """One conformance run: whether it passed, and every verdict line it printed."""

    passed: bool
    lines: Tuple[str, ...]


def run_control_port_conformance(
    *, mode: int, seed: int, n_cores: int = 1, engine: str | None = None,
    timeout_s: float = 600.0, sources: Sequence[Path] | None = None,
) -> ControlPortResult:
    """Drive the wrapper's control port with one master mode and seed."""
    if mode not in CONTROL_PORT_MODES:
        raise ValueError(f"mode {mode!r} is not one of {CONTROL_PORT_MODES}")
    build = build_testbench(
        n_cores=n_cores, token_count=1, engine=engine, tb_name=KERNEL_AXIL_TB,
        rtl_sources=kernel_design_sources() if sources is None else list(sources),
        extra_params={"CAPWORDS": SHIPPED_CAPTURE_WORDS,
                      "FIFOWORDS": SHIPPED_FIFO_WORDS})
    with tempfile.TemporaryDirectory() as scratch:
        # The testbench reads no stimulus; the harness's run line names one.
        unused = Path(scratch) / "unused.hex"
        unused.write_text("00000000\n")
        run = run_testbench(build, unused, timeout_s=timeout_s,
                            plusargs={"mode": int(mode), "seed": int(seed)})
    lines = tuple(line for line in run.stdout.splitlines()
                  if line.startswith(("AXIL_PASS", "AXIL_FAIL")))
    if not any(line == "AXIL_DONE" for line in run.stdout.splitlines()):
        raise TestbenchFailure(
            "the control-port testbench never printed AXIL_DONE: it stopped "
            "before its checks ran")
    return ControlPortResult(
        passed=bool(lines) and all(line.startswith("AXIL_PASS") for line in lines),
        lines=lines)
