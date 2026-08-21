"""Simulating the Vitis RTL kernel: elaboration and the fabric-sequencer smoke run.

The kernel is the P7b build's design; what can be checked LOCALLY is that it
elaborates and that its on-fabric sequencer executes the exporter's token
program with the same semantics as the host-driven P5 testbench. Both are
[slow] gates under ``scripts/hw_tests/``.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.capture import CaptureResult, parse_capture
from mimarsinan.chip_simulation.odin_rtl.stimulus import Op, write_stimulus
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    ENGINE_INTERPRETED,
    REPO_ROOT,
    SimulationRun,
    SimulatorBuildError,
    TestbenchBuild,
    build_testbench,
    design_sources,
    require_tool,
    run_testbench,
)

KERNEL_ROOT = REPO_ROOT / "hw" / "fpga" / "kernel"

#: The kernel testbench's own name; the wrapper elaborates through the top.
KERNEL_TB = "tb_odin_fpga_kernel"
KERNEL_TOP = "odin_fpga_kernel_top"


def kernel_sources() -> List[Path]:
    """The kernel tree, in a stable order (the wrapper last, as the top)."""
    return sorted(KERNEL_ROOT.glob("*.v"))


def kernel_design_sources(*, overlay: bool = False) -> List[Path]:
    """The kernel plus the design it instantiates (the vendored ODIN tree)."""
    return kernel_sources() + design_sources(overlay=overlay)


def elaborate_kernel_top(*, n_cores: int = 1) -> str:
    """Elaborate the Vitis WRAPPER under iverilog; raise with the tool's output.

    The wrapper is what `v++` packages, and its port list is the shell contract,
    so a syntax or width error there is a build failure on HACC hours later.
    This is the local lint that catches it in seconds.
    """
    command = [
        str(require_tool(ENGINE_INTERPRETED)), "-g2005", "-o", "/dev/null",
        "-s", KERNEL_TOP, "-P", f"{KERNEL_TOP}.NC={int(n_cores)}",
    ] + [str(path) for path in kernel_design_sources()]
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SimulatorBuildError(
            f"iverilog failed to elaborate {KERNEL_TOP} (NC={n_cores}):\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return result.stderr


def build_kernel_testbench(
    *, n_cores: int, token_count: int, engine: str | None = None,
    cap_words: int = 65536,
) -> TestbenchBuild:
    """Elaborate the kernel smoke testbench around ``n_cores`` vendored cores."""
    return build_testbench(
        n_cores=n_cores, token_count=token_count, engine=engine,
        tb_name=KERNEL_TB, rtl_sources=kernel_design_sources(),
        extra_params={"CAPWORDS": cap_words},
    )


def run_kernel_program(
    ops: Sequence[Op],
    *,
    n_cores: int = 1,
    engine: str | None = None,
    cap_words: int = 65536,
    timeout_s: float = 3600.0,
    workdir: Path | None = None,
) -> Tuple[CaptureResult, TestbenchBuild, SimulationRun]:
    """Execute one token program on the FABRIC sequencer and parse its capture."""
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(workdir) if workdir is not None else Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        stimulus = root / "odin_kernel_stim.hex"
        tokens = write_stimulus(stimulus, list(ops))
        build = build_kernel_testbench(
            n_cores=n_cores, token_count=tokens, engine=engine,
            cap_words=cap_words)
        run = run_testbench(build, stimulus, timeout_s=timeout_s)
    return parse_capture(run.stdout), build, run
