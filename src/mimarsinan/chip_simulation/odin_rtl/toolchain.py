"""Simulator discovery, testbench builds (cached per geometry), and runs."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from mimarsinan.common.env import DEFAULT_HW_SIM_BIN_DIR, HW_SIM_BIN_VAR, hw_sim_bin_dir

#: Engines, best first. The fast one is a compiled binary with timing support;
#: the interpreted one is the X-accurate cross-check. Both execute the SAME
#: Verilog-2005 testbench source.
ENGINE_COMPILED = "verilator"
ENGINE_INTERPRETED = "iverilog"
ENGINES: Tuple[str, ...] = (ENGINE_COMPILED, ENGINE_INTERPRETED)

REPO_ROOT = Path(__file__).resolve().parents[4]
HW_ROOT = REPO_ROOT / "hw"
VENDOR_SRC = HW_ROOT / "vendor" / "odin" / "src"
OVERLAY_MEM = HW_ROOT / "fpga" / "mem"
TB_ROOT = HW_ROOT / "tb"
BUILD_CACHE = REPO_ROOT / "build" / "odin_rtl_cache"

#: The tb's program array is a compile-time parameter, so the harness rounds the
#: token count up to one of a few sizes and the build cache stays small.
PROGRAM_SIZE_STEPS: Tuple[int, ...] = (
    1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24,
)


class SimulatorUnavailable(RuntimeError):
    """No RTL simulator was found; the caller must skip LOUDLY, naming the path."""


class SimulatorBuildError(RuntimeError):
    """The testbench did not elaborate."""


def simulator_bin_dir() -> Path:
    """The configured simulator directory, absolute, without checking it exists."""
    configured = Path(hw_sim_bin_dir())
    return configured if configured.is_absolute() else (REPO_ROOT / configured)


def find_tool(name: str) -> Path | None:
    """The named binary in the configured directory, or ``None``."""
    candidate = simulator_bin_dir() / name
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def missing_tool_reason(tools: Sequence[str], *, subject: str, gates: str) -> str:
    """The LOUD skip message: which tools, which directory, which override, which gates."""
    return (
        f"{subject} unavailable: none of {', '.join(tools)} is executable in "
        f"{simulator_bin_dir()} (default {DEFAULT_HW_SIM_BIN_DIR!r}, override with "
        f"{HW_SIM_BIN_VAR}). {gates} cannot run without it "
        f"and are NOT being reported as passing."
    )


def unavailable_reason(tools: Sequence[str]) -> str:
    """The LOUD skip message for the cosimulation gates."""
    return missing_tool_reason(
        tools, subject="RTL simulator", gates="The ODIN cosimulation gates")


def available_engine() -> str:
    """The best engine present, or raise with the full diagnostic."""
    if find_tool("verilator") is not None:
        return ENGINE_COMPILED
    if find_tool("iverilog") is not None and find_tool("vvp") is not None:
        return ENGINE_INTERPRETED
    raise SimulatorUnavailable(unavailable_reason(("verilator", "iverilog+vvp")))


def require_tool(name: str) -> Path:
    path = find_tool(name)
    if path is None:
        raise SimulatorUnavailable(unavailable_reason((name,)))
    return path


def vendor_sources() -> List[Path]:
    """The vendored RTL, in a stable order; the tree is never edited."""
    return sorted(VENDOR_SRC.rglob("*.v"))


def overlay_sources() -> List[Path]:
    """The BRAM overlay, which must precede the vendor tree in the source list."""
    return sorted(OVERLAY_MEM.glob("*.v"))


def design_sources(*, overlay: bool) -> List[Path]:
    """The design's compile order: (optionally) the overlay, then the vendor tree.

    Overlay selection is FILE ORDER: the first declaration of
    ``SRAM_256x128_wrapper`` / ``SRAM_8192x32_wrapper`` wins and the vendored
    behavioural copies inside ``neuron_core.v`` / ``synaptic_core.v`` are
    shadowed. The vendor tree is not touched. Each tool spells that rule its own
    way -- verilator implements it by default, yosys wants ``-nooverwrite``, and
    iverilog rejects duplicate module declarations outright, so an overlay build
    asks for the compiled engine by name.
    """
    return (overlay_sources() if overlay else []) + vendor_sources()


def source_list(tb: Path, *, overlay: bool) -> List[Path]:
    """The simulation compile order: testbench first, then the design sources."""
    return [tb] + design_sources(overlay=overlay)


def program_array_size(token_count: int) -> int:
    """The tb ``PROGWORDS`` parameter for a program of ``token_count`` tokens."""
    for step in PROGRAM_SIZE_STEPS:
        if token_count <= step:
            return step
    raise SimulatorBuildError(
        f"a {token_count}-token program is over the largest supported testbench "
        f"array ({PROGRAM_SIZE_STEPS[-1]} tokens); split the run into samples")


@dataclass(frozen=True)
class TestbenchBuild:
    """A compiled testbench for one geometry."""

    engine: str
    binary: Path
    n_cores: int
    program_words: int
    overlay: bool
    build_seconds: float
    cached: bool

    def run_command(self, stimulus: Path) -> List[str]:
        if self.engine == ENGINE_COMPILED:
            return [str(self.binary), f"+stim={stimulus}"]
        return [str(require_tool("vvp")), str(self.binary), f"+stim={stimulus}"]


def _fingerprint(paths: Sequence[Path], *extra: str) -> str:
    digest = hashlib.sha256()
    for item in extra:
        digest.update(item.encode())
        digest.update(b"\0")
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build_testbench(
    *,
    n_cores: int,
    token_count: int,
    overlay: bool = False,
    engine: str | None = None,
    tb_name: str = "tb_odin_core",
    parameterised: bool = True,
) -> TestbenchBuild:
    """Compile (or reuse) the testbench for one geometry.

    ``parameterised`` is False for testbenches that declare no geometry
    parameters (the memory-overlay harness), so no override is passed for a
    parameter the source does not have.
    """
    engine = engine or available_engine()
    if engine not in ENGINES:
        raise SimulatorBuildError(
            f"unknown engine {engine!r}; known: {', '.join(ENGINES)}")
    if overlay and engine != ENGINE_COMPILED:
        raise SimulatorBuildError(
            f"the BRAM overlay is selected by source-file order, which only the "
            f"{ENGINE_COMPILED!r} engine implements ({ENGINE_INTERPRETED} rejects "
            f"the duplicate module declaration); asked for {engine!r}")
    tb = TB_ROOT / f"{tb_name}.v"
    if not tb.is_file():
        raise SimulatorBuildError(f"no testbench source at {tb}")
    program_words = program_array_size(token_count)
    sources = source_list(tb, overlay=overlay)
    key = _fingerprint(
        sources, engine, tb_name, str(n_cores), str(program_words), str(overlay))
    workdir = BUILD_CACHE / f"{tb_name}_{engine}_{n_cores}c_{program_words}w_{key}"
    binary = workdir / ("vtb" if engine == ENGINE_COMPILED else "tb.vvp")
    if binary.is_file():
        return TestbenchBuild(engine, binary, n_cores, program_words, overlay, 0.0, True)

    workdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if engine == ENGINE_COMPILED:
        overrides = (
            [f"-GNC={n_cores}", f"-GPROGWORDS={program_words}"]
            if parameterised else []
        )
        command = [
            str(require_tool("verilator")), "--binary", "--timing", "-O2",
            "-Wno-fatal", "--default-language", "1364-2005",
            "--top-module", tb_name,
        ] + overrides + [
            "--Mdir", str(workdir / "obj_dir"), "-o", str(binary.resolve()),
        ] + [str(path) for path in sources]
    else:
        overrides = (
            ["-P", f"{tb_name}.NC={n_cores}",
             "-P", f"{tb_name}.PROGWORDS={program_words}"]
            if parameterised else []
        )
        command = [
            str(require_tool("iverilog")), "-g2005", "-o", str(binary),
            "-s", tb_name,
        ] + overrides + [str(path) for path in sources]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0 or not binary.is_file():
        shutil.rmtree(workdir, ignore_errors=True)
        raise SimulatorBuildError(
            f"{engine} failed to elaborate {tb_name} "
            f"(NC={n_cores}, PROGWORDS={program_words}, overlay={overlay}):\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return TestbenchBuild(
        engine, binary, n_cores, program_words, overlay,
        time.monotonic() - started, False)


@dataclass(frozen=True)
class SimulationRun:
    """One testbench execution: its stdout and its wall time."""

    stdout: str
    seconds: float


def run_testbench(build: TestbenchBuild, stimulus: Path,
                  *, timeout_s: float = 3600.0) -> SimulationRun:
    """Execute a built testbench against a stimulus file."""
    started = time.monotonic()
    result = subprocess.run(
        build.run_command(Path(stimulus).resolve()),
        capture_output=True, text=True, timeout=timeout_s,
        cwd=str(Path(stimulus).resolve().parent),
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise SimulatorBuildError(
            f"{build.engine} simulation exited {result.returncode}:\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}")
    return SimulationRun(result.stdout, elapsed)
