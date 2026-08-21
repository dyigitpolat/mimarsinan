"""Yosys synthesis of the vendored ODIN core: the instrument behind plan gate P5.5a."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.synth_census import (
    ResourceTable,
    SynthesisError,
    primitive_cells,
    resource_table,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import (
    REPO_ROOT,
    design_sources,
    find_tool,
    missing_tool_reason,
)

TOP_MODULE = "ODIN"

#: The UltraScale+ family; the U55C's XCU55C part is a member of it (plan F18).
TARGET_FAMILY = "xcup"

SYNTHESIS_TIMEOUT_S = 3600.0


class SynthesisUnavailable(RuntimeError):
    """No synthesis tool was found; the caller must skip LOUDLY, naming the path."""


def yosys_path() -> Path:
    """The yosys binary in the configured toolchain directory, or raise."""
    binary = find_tool("yosys")
    if binary is None:
        raise SynthesisUnavailable(synthesis_unavailable_reason())
    return binary


def synthesis_unavailable_reason() -> str:
    """The LOUD skip message for the synthesizability gate."""
    return missing_tool_reason(
        ("yosys",), subject="RTL synthesis tool",
        gates="The ODIN synthesizability gates (plan §7 row 19)")


def yosys_version() -> str:
    """The exact tool banner, recorded with the numbers it produced."""
    result = subprocess.run(
        [str(yosys_path()), "-V"], capture_output=True, text=True, timeout=120.0)
    if result.returncode != 0:
        raise SynthesisError(f"`yosys -V` exited {result.returncode}: {result.stderr}")
    return result.stdout.strip()


def script_lines(*, sources: Sequence[Path],
                 stat_path: Path | None) -> Tuple[str, str, str]:
    """The three yosys commands the gate runs, with repo-relative source paths.

    ``-nooverwrite`` is yosys's spelling of the first-declaration-wins rule the
    cosimulation already relies on: the `hw/fpga/mem/` wrappers are read first,
    and the vendored behavioural copies inside `neuron_core.v` /
    `synaptic_core.v` are ignored without the vendor tree being edited.
    """
    listed = " ".join(
        path.resolve().relative_to(REPO_ROOT).as_posix() for path in sources)
    stat = f"stat -top {TOP_MODULE} -json"
    return (
        f"read_verilog -nooverwrite {listed}",
        f"synth_xilinx -family {TARGET_FAMILY} -top {TOP_MODULE}",
        stat if stat_path is None else f"tee -q -o {stat_path} {stat}",
    )


@dataclass(frozen=True)
class SynthesisAttempt:
    """One yosys run over the design: its census, or exactly why there is none."""

    overlay: bool
    succeeded: bool
    seconds: float
    script: Tuple[str, str, str]
    stat: Mapping[str, Any] | None
    error_line: str | None
    log_tail: str

    def require_stat(self) -> Mapping[str, Any]:
        if self.stat is None or not self.succeeded:
            raise SynthesisError(
                f"the synthesis (overlay={self.overlay}) produced no census: "
                f"{self.error_line}\n{self.log_tail}")
        return self.stat

    def resource_table(self) -> ResourceTable:
        return resource_table(primitive_cells(self.require_stat()))


def _first_error(log: str) -> str | None:
    for line in log.splitlines():
        if "ERROR" in line:
            return line[line.index("ERROR"):].strip()
    return None


def run_synthesis(*, overlay: bool, workdir: Path) -> SynthesisAttempt:
    """Run `synth_xilinx` over the design and report the census it produced."""
    binary = yosys_path()
    workdir.mkdir(parents=True, exist_ok=True)
    stat_path = workdir / "stat.json"
    log_path = workdir / "yosys.log"
    sources = design_sources(overlay=overlay)
    script_path = workdir / "synth.ys"
    script_path.write_text(
        "\n".join(script_lines(sources=sources, stat_path=stat_path)) + "\n",
        encoding="utf-8")

    started = time.monotonic()
    result = subprocess.run(
        [str(binary), "-q", "-l", str(log_path), "-s", str(script_path)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        timeout=SYNTHESIS_TIMEOUT_S)
    elapsed = time.monotonic() - started

    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    error_line = _first_error(log) or _first_error(result.stderr)
    stat: Mapping[str, Any] | None = None
    if stat_path.is_file():
        stat = json.loads(stat_path.read_text(encoding="utf-8"))
    succeeded = result.returncode == 0 and error_line is None and stat is not None
    if result.returncode != 0 and error_line is None:
        error_line = f"yosys exited {result.returncode} without an ERROR line"
    return SynthesisAttempt(
        overlay=overlay,
        succeeded=succeeded,
        seconds=elapsed,
        script=script_lines(sources=sources, stat_path=None),
        stat=stat if succeeded else None,
        error_line=None if succeeded else error_line,
        log_tail=log[-3000:],
    )
