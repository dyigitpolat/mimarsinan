"""[ODIN multi-core, targeted pass] the wrapper's control port is AXI4-Lite conformant.

On the card the kernel's s_axi_control sits behind the shell's AXI interconnect,
which crosses from the PCIe clock to the kernel clock with one clock-crossing
path PER CHANNEL. A host's register write therefore reaches the kernel as an
address and a data beat that may arrive in either order, and the next write's
address may arrive before this write's data. That is legal AXI4-Lite. The
Stage 1 wrapper paired each data beat with whichever address it had seen last,
so kernel arguments landed in neighbouring registers: on the NC 4 bitstream the
card streamed the program buffer as a stimulus (0 events after stimulus-words/4
SPI transactions), streamed one word (175 cycles), drained captures into the
program buffer (NO-VERDICT with the fabric's own count right) and never saw
ap_start (hangs). ``stage1_fix_report.md`` holds the evidence.

This gate drives ``hw/tb/tb_odin_fpga_kernel_axil.v``: its master presents W
before AW, AW ahead of W, independent random gaps, and AW with W together, with
B and R back-pressured, and requires every argument register to hold exactly
what was written to ITS address and exactly one response per transaction.
[slow]: one Verilator build of the wrapper at NC 1 (the control port does not
depend on NC).
"""

from __future__ import annotations

import pytest

from integration.odin_rtl_harness import require_simulator

from mimarsinan.chip_simulation.odin_fpga.control_port import (
    CONTROL_PORT_MODES,
    run_control_port_conformance,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]

SEEDS = (1, 2, 3, 7, 11)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("mode", CONTROL_PORT_MODES)
def test_every_register_write_lands_at_its_own_address(mode, seed):
    try:
        require_simulator()
    except Exception as exc:  # pragma: no cover - the loud skip path
        pytest.skip(str(exc))
    result = run_control_port_conformance(mode=mode, seed=seed)
    assert result.passed, "\n".join(result.lines)
    assert result.lines and "writes 240 reads 6" in result.lines[-1]
