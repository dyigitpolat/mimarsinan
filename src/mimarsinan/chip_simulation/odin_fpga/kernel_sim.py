"""Simulating the Vitis RTL kernel: elaboration, the fabric run, the WRAPPER run.

The kernel is the P7b build's design; what can be checked LOCALLY is that it
elaborates, that its on-fabric sequencer executes the exporter's token program
with the same semantics as the host-driven P5 testbench, and — through a
behavioural AXI4 memory model — that the WRAPPER's DMA engine moves the bytes
itself: program in, stimulus in, capture out. All are [slow] gates under
``scripts/hw_tests/``.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mimarsinan.chip_simulation.odin_fpga.kernel_registers import (
    SHIPPED_CAPTURE_WORDS,
)
from mimarsinan.chip_simulation.odin_rtl.capture import (
    CaptureResult,
    TestbenchFailure,
    parse_capture,
)
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
#: The WRAPPER testbench: the same program delivered through an AXI4 memory
#: model, so the DMA engine — not a preloaded write port — fills the fabric.
KERNEL_AXI_TB = "tb_odin_fpga_kernel_axi"
KERNEL_TOP = "odin_fpga_kernel_top"


def kernel_sources() -> List[Path]:
    """The kernel tree, in a stable order (the wrapper last, as the top)."""
    return sorted(KERNEL_ROOT.glob("*.v"))


def kernel_design_sources(*, overlay: bool = False) -> List[Path]:
    """The kernel plus the design it instantiates (the vendored ODIN tree)."""
    return kernel_sources() + design_sources(overlay=overlay)


#: The elastic op-stream FIFO's shipped depth, in 32-bit words — the second copy
#: of the ``FIFO_WORDS`` default in ``hw/fpga/kernel/odin_fpga_kernel_top.v``.
#: It is NOT a capacity the host must respect: the stream is bounded only by the
#: word counts the host declares, and this depth only buys latency tolerance.
SHIPPED_FIFO_WORDS = 1024

_BLOCK_RAM_ATTR = 'ram_style = "block"'
_BLOCK_RAM_DECL = re.compile(
    r'\(\*\s*ram_style\s*=\s*"block"\s*\*\)\s*reg\b(?:\s*\[[^\]]*\])?\s*(\w+)\s*\[')


def _indexed_references(body: str, name: str) -> Tuple[int, int]:
    """``(reads, writes)`` of ``name[...]`` in ``body``; a write is one before ``<=``."""
    reads = writes = 0
    for match in re.finditer(rf"\b{re.escape(name)}\s*\[", body):
        index, depth = match.end() - 1, 0
        while index < len(body):
            depth += (body[index] == "[") - (body[index] == "]")
            if depth == 0:
                break
            index += 1
        if body[index + 1:].lstrip().startswith("<="):
            writes += 1
        else:
            reads += 1
    return reads, writes


def block_ram_ports(source: Path) -> Dict[str, Tuple[int, int]]:
    """``{array: (read points, write points)}`` for each ``ram_style="block"`` array.

    A tile is one registered read port and one write port. An array indexed in
    more than one place is one a synthesizer may read as multi-ported and drop
    into distributed RAM instead -- which is exactly what Vivado 2022.2 did to
    the program RAM this kernel no longer has, on the routed U55C build.
    """
    # Comments go first: an array named in prose is not a port. Verilog
    # attributes open with `(*`, so the block-comment strip leaves them alone.
    text = re.sub(r"/\*.*?\*/", " ", source.read_text(), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    body = "\n".join(
        line for line in text.splitlines() if _BLOCK_RAM_ATTR not in line)
    return {
        decl.group(1): _indexed_references(body, decl.group(1))
        for decl in _BLOCK_RAM_DECL.finditer(text)
    }


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
    cap_words: int = SHIPPED_CAPTURE_WORDS,
    fifo_words: int = SHIPPED_FIFO_WORDS,
) -> TestbenchBuild:
    """Elaborate the kernel smoke testbench around ``n_cores`` vendored cores."""
    return build_testbench(
        n_cores=n_cores, token_count=token_count, engine=engine,
        tb_name=KERNEL_TB, rtl_sources=kernel_design_sources(),
        extra_params={"CAPWORDS": cap_words, "FIFOWORDS": fifo_words},
    )


def run_kernel_program(
    ops: Sequence[Op],
    *,
    n_cores: int = 1,
    engine: str | None = None,
    cap_words: int = SHIPPED_CAPTURE_WORDS,
    fifo_words: int = SHIPPED_FIFO_WORDS,
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
            cap_words=cap_words, fifo_words=fifo_words)
        run = run_testbench(build, stimulus, timeout_s=timeout_s)
    return parse_capture(run.stdout), build, run


@dataclass(frozen=True)
class KernelStatus:
    """The wrapper's register reads, as the AXI testbench reports them.

    ``events_seen`` is the fabric's own count of AER-out events — it may exceed
    ``capture_capacity``, which is exactly the truncation the host refuses.
    There is no program capacity: the op stream is not stored on the fabric.
    """

    err: bool
    events_seen: int
    capture_capacity: int
    host_capacity: int
    records_written: int
    program_words: int
    stimulus_words: int
    stream_words: int
    stall_seed: int
    stall_gaps: int
    stall_cycles: int

    @property
    def truncated(self) -> bool:
        """Whether the fabric saw at least as many events as it can hold."""
        return self.events_seen >= min(self.capture_capacity, self.host_capacity)


def parse_kernel_status(stdout: str) -> KernelStatus:
    """The KSTAT/KSPLIT/KSTALL lines of the wrapper testbench; loud when absent."""
    kstat: List[int] = []
    ksplit: List[int] = []
    kstall: List[int] = []
    for line in stdout.splitlines():
        fields = line.split()
        if fields[:1] == ["KSTAT"] and len(fields) == 6:
            kstat = [int(value) for value in fields[1:]]
        elif fields[:1] == ["KSPLIT"] and len(fields) == 4:
            ksplit = [int(value) for value in fields[1:]]
        elif fields[:1] == ["KSTALL"] and len(fields) == 4:
            kstall = [int(value) for value in fields[1:]]
    if not kstat or not ksplit or not kstall:
        raise TestbenchFailure(
            "the wrapper testbench printed no KSTAT/KSPLIT/KSTALL line: its "
            "control reads never completed, so the run has no status to report")
    return KernelStatus(
        err=bool(kstat[0]), events_seen=kstat[1], capture_capacity=kstat[2],
        host_capacity=kstat[3], records_written=kstat[4],
        program_words=ksplit[0], stimulus_words=ksplit[1],
        stream_words=ksplit[2], stall_seed=kstall[0],
        stall_gaps=kstall[1], stall_cycles=kstall[2])


def build_kernel_axi_testbench(
    *, n_cores: int, token_count: int, engine: str | None = None,
    cap_words: int = SHIPPED_CAPTURE_WORDS,
    fifo_words: int = SHIPPED_FIFO_WORDS, split: int, host_capacity: int,
) -> TestbenchBuild:
    """Elaborate the WRAPPER testbench: the DUT is `odin_fpga_kernel_top`."""
    return build_testbench(
        n_cores=n_cores, token_count=token_count, engine=engine,
        tb_name=KERNEL_AXI_TB, rtl_sources=kernel_design_sources(),
        extra_params={
            "CAPWORDS": cap_words,
            "FIFOWORDS": fifo_words,
            "SPLIT": int(split),
            "HOSTCAP": int(host_capacity),
        },
    )


def run_kernel_program_over_axi(
    ops: Sequence[Op],
    *,
    n_cores: int = 1,
    engine: str | None = None,
    cap_words: int = SHIPPED_CAPTURE_WORDS,
    fifo_words: int = SHIPPED_FIFO_WORDS,
    host_capacity: int = 0x000FFFFF,
    stall_seed: int = 0,
    timeout_s: float = 3600.0,
    workdir: Path | None = None,
) -> Tuple[CaptureResult, KernelStatus, TestbenchBuild, SimulationRun]:
    """Execute one token program THROUGH THE WRAPPER's DMA engine.

    The program is placed in an AXI4 memory model split into the two buffers a
    host hands the kernel, and the capture is read back out of that same model —
    nothing is preloaded into the fabric. ``stall_seed`` is a RUN-time plusarg,
    not an elaboration parameter, so every seed of the stall-invariance gate
    reuses one cached build; 0 is the no-stall baseline.
    """
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(workdir) if workdir is not None else Path(scratch)
        root.mkdir(parents=True, exist_ok=True)
        stimulus = root / "odin_kernel_axi_stim.hex"
        tokens = write_stimulus(stimulus, list(ops))
        build = build_kernel_axi_testbench(
            n_cores=n_cores, token_count=tokens, engine=engine,
            cap_words=cap_words, fifo_words=fifo_words,
            split=max(1, tokens // 2), host_capacity=host_capacity)
        run = run_testbench(
            build, stimulus, timeout_s=timeout_s,
            plusargs={"stallseed": int(stall_seed)})
    return (
        parse_capture(run.stdout), parse_kernel_status(run.stdout), build, run)
