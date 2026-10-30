"""The committed synthesis evidence: hw/fpga/synth_resources.json and SYNTH_REPORT.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from mimarsinan.chip_simulation.odin_rtl.synth_census import (
    memory_inferences,
    primitive_cells,
    resource_table,
)
from mimarsinan.chip_simulation.odin_rtl.synthesis import (
    TARGET_FAMILY,
    TOP_MODULE,
    SynthesisAttempt,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import HW_ROOT, REPO_ROOT

SCHEMA = "odin_synth_resources/v1"
RESOURCES_JSON = HW_ROOT / "fpga" / "synth_resources.json"
REPORT_MD = HW_ROOT / "fpga" / "SYNTH_REPORT.md"
REGEN_SCRIPT = "scripts/hw_tests/regen_synth_report.py"

#: The sections the [slow] gate compares bit-for-bit. The tool banner is NOT one
#: of them: it is recorded so a mismatch can name the version that moved.
GATED_SECTIONS = (
    "schema", "top", "family", "sources", "script", "per_core", "cells_by_type",
    "memories",
)


def source_names(sources: Sequence[Path]) -> List[str]:
    """The design sources as repo-relative posix paths, in compile order."""
    return [path.resolve().relative_to(REPO_ROOT).as_posix() for path in sources]


def _control_record(control: SynthesisAttempt) -> Dict[str, Any]:
    """What the vendored tree ALONE does: the local witness for plan finding F17."""
    reached: List[str] = []
    if control.succeeded and control.stat is not None:
        reached = [m.module for m in memory_inferences(control.stat)
                   if m.inferred_as_bram]
    return {
        "synthesis_succeeded": control.succeeded,
        "block_ram_reached": bool(reached),
        "memories_in_block_ram": reached,
        "detail": control.error_line,
    }


def build_record(*, attempt: SynthesisAttempt, control: SynthesisAttempt,
                 tool_version: str) -> Dict[str, Any]:
    """The machine-readable artifact: P8's per-core resource input."""
    stat = attempt.require_stat()
    cells = primitive_cells(stat)
    return {
        "schema": SCHEMA,
        "top": TOP_MODULE,
        "family": TARGET_FAMILY,
        "tool": {"name": "yosys", "version": tool_version},
        "sources": _recorded_sources(attempt),
        "script": list(attempt.script),
        "per_core": resource_table(cells).as_record(),
        "cells_by_type": dict(sorted(cells.items())),
        "memories": [m.as_record() for m in memory_inferences(stat)],
        "vendor_only_control": _control_record(control),
    }


def _recorded_sources(attempt: SynthesisAttempt) -> List[str]:
    """The source list the recorded `read_verilog` line actually named."""
    return attempt.script[0].split()[2:]


def gated_view(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Exactly what an unchanged RTL tree must reproduce, byte for byte."""
    view: Dict[str, Any] = {key: record[key] for key in GATED_SECTIONS}
    view["vendor_only_reaches_block_ram"] = (
        record["vendor_only_control"]["block_ram_reached"])
    return view


def load_committed() -> Dict[str, Any]:
    """The committed artifact, or a loud error naming how to regenerate it."""
    if not RESOURCES_JSON.is_file():
        raise FileNotFoundError(
            f"no committed synthesis record at {RESOURCES_JSON}; generate it with "
            f"{REGEN_SCRIPT}")
    return json.loads(RESOURCES_JSON.read_text(encoding="utf-8"))


def write_artifacts(record: Mapping[str, Any]) -> None:
    """Rewrite both evidence files from one record; the gate reads them back."""
    RESOURCES_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESOURCES_JSON.write_text(
        json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_report(record), encoding="utf-8")


def _memory_rows(record: Mapping[str, Any]) -> List[str]:
    rows: List[str] = []
    for memory in record["memories"]:
        cells = memory["bram_cells"] or memory["distributed_cells"] or {}
        rows.append(
            f"| `{memory['module']}` | {memory['geometry']} "
            f"| {memory['words'] * memory['width']:,} "
            f"| {', '.join(f'{k} x{v}' for k, v in cells.items()) or '(none)'} "
            f"| {'**block RAM**' if memory['inferred_as_bram'] else '**NOT block RAM -- FINDING**'} |")
    return rows


def _control_paragraph(record: Mapping[str, Any]) -> str:
    control = record["vendor_only_control"]
    if control["block_ram_reached"]:
        return (
            "The control run reached block RAM **without** the overlay, which "
            "contradicts plan finding F17 and means the overlay is not what makes "
            f"the memories synthesize: {control['memories_in_block_ram']}.")
    if not control["synthesis_succeeded"]:
        return (
            "The same script over the vendored tree **alone** does not complete -- "
            f"yosys reports `{control['detail']}` after mapping the behavioural "
            "arrays toward distributed RAM, whose width/depth combination it then "
            "refuses. That is the local confirmation of plan finding F17 (the stock "
            "RTL is not FPGA-synthesizable as vendored) and the reason "
            "`hw/fpga/mem/` exists.")
    return (
        "The same script over the vendored tree **alone** completes but places "
        "neither memory in a block-RAM tile, which is the local confirmation of "
        "plan finding F17.")


def render_report(record: Mapping[str, Any]) -> str:
    """The committed evidence prose; the fast gate asserts the file equals this."""
    per_core = record["per_core"]
    script = "\n".join(record["script"])
    unclassified = per_core["unclassified"]
    lines = [
        "# ODIN stock core -- local synthesizability proof and per-core resources",
        "",
        "Plan `docs/odin_firing_semantics_and_rtl_export_plan.md` §7 row 19 (P5.5a).",
        "This is the LOCAL half of the synthesis/implementation gate: the vendored",
        "`hw/vendor/odin` core plus the `hw/fpga/mem/` BRAM overlay through yosys",
        f"`synth_xilinx -family {TARGET_FAMILY}` -- the UltraScale+ family the",
        "U55C's XCU55C part belongs to -- with zero errors, both memories in block",
        "RAM, and the per-core resource census below.",
        "",
        "## What produced these numbers",
        "",
        f"- Tool: `{record['tool']['version']}`",
        f"- Top module: `{record['top']}`  |  target family: `{record['family']}`",
        f"- Sources ({len(record['sources'])} files, overlay first so the first "
        f"declaration of each memory wrapper wins):",
        "",
        "```",
        "\n".join(f"  {name}" for name in record["sources"]),
        "```",
        "",
        "- Exact yosys script (run from the repo root):",
        "",
        "```tcl",
        script,
        "```",
        "",
        f"- Regenerate DELIBERATELY with `{REGEN_SCRIPT}`; the [slow] gate",
        "  `tests/integration/test_odin_rtl_synth.py` re-runs the synthesis and",
        f"  requires an exact match against `{RESOURCES_JSON.name}`.",
        "",
        "## Per-core resources (one ODIN core, 256 neurons x 256 axons)",
        "",
        "| Resource | Count |",
        "| --- | ---: |",
        f"| LUTs (`LUT1`..`LUT6`) | {per_core['luts']:,} |",
        f"| Inverters (`INV`, a LUT1 once packed) | {per_core['inverters']:,} |",
        f"| **LUT equivalent** | **{per_core['lut_equivalent']:,}** |",
        f"| Flip-flops (`FD*`) | {per_core['flip_flops']:,} |",
        f"| Carry cells (`CARRY*`) | {per_core['carry']:,} |",
        f"| Wide muxes (`MUXF*`) | {per_core['muxf']:,} |",
        f"| `RAMB36` | {per_core['bram36']:,} |",
        f"| `RAMB18` | {per_core['bram18']:,} |",
        f"| **BRAM tiles (36 kb equivalent)** | **{per_core['bram_tiles']:g}** |",
        f"| `URAM` | {per_core['uram']:,} |",
        f"| Distributed-RAM/SRL cells | {per_core['lutram']:,} |",
        f"| I/O and clock buffers | {per_core['io_buffers']:,} |",
        f"| Unclassified cells | {sum(unclassified.values()):,} |",
        f"| Total primitive cells | {per_core['total_cells']:,} |",
        "",
    ]
    if unclassified:
        lines += [
            f"Unclassified (named, never absorbed): "
            f"{', '.join(f'`{k}` x{v}' for k, v in unclassified.items())}.",
            "",
        ]
    lines += [
        "## BRAM inference outcome",
        "",
        "| Memory | Geometry | Bits | Cells | Outcome |",
        "| --- | --- | ---: | --- | --- |",
        *_memory_rows(record),
        "",
        "## Control: the vendored memories alone (plan finding F17)",
        "",
        _control_paragraph(record),
        "",
        "## The caveat this report must carry",
        "",
        "yosys-synthesizability is **necessary, not sufficient** for Vivado closure",
        "on the U55C shell. This run proves the RTL elaborates, maps to UltraScale+",
        "primitives, and puts both memories in block RAM; it proves **nothing** about",
        "timing closure, placement, routing, the Alveo shell's own resource budget,",
        "or the XRT kernel wrapper. yosys's Xilinx mapping is also generic rather",
        "than exact -- it emits `CARRY4` where UltraScale+ has `CARRY8`, leaves `INV`",
        "unpacked, and does not model LUT pairing -- so the LUT/FF columns are an",
        "ORDER-OF-MAGNITUDE per-core budget for P8's compile-limits study, not Vivado",
        "utilization. Vivado exists only on HACC (owner-gated login), so the",
        "implementation half of §7 row 19 is closed by **P7**'s HACC build, where the",
        "same design is run through Vitis 2022.2 for the U55C shell and the real",
        "utilization report replaces these estimates.",
        "",
    ]
    return "\n".join(lines)
