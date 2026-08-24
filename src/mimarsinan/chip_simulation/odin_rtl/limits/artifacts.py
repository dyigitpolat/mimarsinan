"""The committed compile-limits evidence: hw/fpga/compile_limits.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mimarsinan.chip_simulation.odin_rtl.limits.configurations import (
    STOCK_KEY,
    Configuration,
    configurations,
)
from mimarsinan.chip_simulation.odin_rtl.limits.device import (
    AVAILABILITY_SCENARIOS,
    DEVICE,
    RESOURCE_CLASSES,
    binding_bound,
    census_costs,
    packing_bound,
)
from mimarsinan.chip_simulation.odin_rtl.synth_artifacts import (
    RESOURCES_JSON,
    source_names,
)
from mimarsinan.chip_simulation.odin_rtl.synth_census import (
    primitive_cells,
    resource_table,
)
from mimarsinan.chip_simulation.odin_rtl.synthesis import (
    TARGET_FAMILY,
    SynthesisAttempt,
    run_target,
)
from mimarsinan.chip_simulation.odin_rtl.toolchain import HW_ROOT, REPO_ROOT

SCHEMA = "odin_compile_limits/v1"
LIMITS_JSON = HW_ROOT / "fpga" / "compile_limits.json"
STUDY_MD = REPO_ROOT / "docs" / "odin_fpga_compile_limits_study.md"
REGEN_SCRIPT = "scripts/hw_tests/regen_compile_limits.py"

#: What an unchanged RTL tree must reproduce exactly. The tool banner and the
#: per-run wall are recorded but NOT gated: a version that moved must be able to
#: name itself in the failure rather than be the failure.
GATED_SECTIONS = (
    "schema", "family", "configurations", "wrapper_overhead",
    "capture_ram_slope", "device", "availability", "bounds",
)

#: The census columns the study reports for every configuration.
REPORTED_COLUMNS: Tuple[str, ...] = (
    "lut_equivalent", "flip_flops", "carry", "muxf", "lutram", "bram36",
    "bram18", "bram_tiles", "uram",
)


def _census_of(cells: Mapping[str, int]) -> Dict[str, Any]:
    return resource_table(dict(cells)).as_record()


def measure(configuration: Configuration, *, workdir: Path) -> SynthesisAttempt:
    """Synthesize one configuration through the P5.5a driver, nothing added."""
    return run_target(configuration.target, workdir=workdir / configuration.key)


def stock_row(configuration: Configuration) -> Dict[str, Any]:
    """The stock core's row, READ from the P5.5a artifact rather than re-measured."""
    committed = json.loads(RESOURCES_JSON.read_text(encoding="utf-8"))
    return {
        **configuration.as_record(),
        "measured_by": RESOURCES_JSON.relative_to(HW_ROOT.parent).as_posix(),
        "sources": list(committed["sources"]),
        "script": list(committed["script"]),
        "census": _census_of(committed["cells_by_type"]),
        "cells_by_type": dict(committed["cells_by_type"]),
    }


def measured_row(configuration: Configuration,
                 attempt: SynthesisAttempt) -> Dict[str, Any]:
    """One freshly synthesized configuration's row."""
    cells = primitive_cells(attempt.require_stat())
    return {
        **configuration.as_record(),
        "measured_by": "this run",
        "sources": source_names(list(configuration.target.sources)),
        "script": list(attempt.script),
        "census": _census_of(cells),
        "cells_by_type": dict(sorted(cells.items())),
    }


def _difference(minuend: Mapping[str, Any], subtrahend: Mapping[str, Any],
                ) -> Dict[str, float]:
    return {column: float(minuend[column]) - float(subtrahend[column])
            for column in REPORTED_COLUMNS}


def _row(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    for row in rows:
        if row["key"] == key:
            return row
    raise KeyError(f"no configuration row {key!r} in {[r['key'] for r in rows]}")


def wrapper_overhead_record(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """What the sequencer + DMA + capture cost AROUND one stock core."""
    wrappers = [row for row in rows if row["kind"] == "wrapper"]
    smallest = min(wrappers, key=lambda row: row["geometry"]["cap_words"])
    stock = _row(rows, STOCK_KEY)
    return {
        "wrapper": smallest["key"],
        "minus": STOCK_KEY,
        "at_parameters": dict(smallest["parameters"]),
        "delta": _difference(smallest["census"], stock["census"]),
        "delta_costs": {
            resource: census_costs(smallest["census"])[resource]
            - census_costs(stock["census"])[resource]
            for resource in RESOURCE_CLASSES},
        "meaning": (
            "the wrapper census less the stock core's, i.e. the AXI4-Lite "
            "control block, the AXI4 DMA engine, the token sequencer, the SPI "
            "master, the AER bridge, the program RAM and the capture RAM, at "
            "the depths named in `at_parameters` -- the capture RAM at the "
            "depth the wrapper SHIPS with, the program RAM shrunk from its"
            " shipped `NC * 262144` words"),
    }


def capture_slope_record(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The measured cost of one capture word: two depths, one subtraction."""
    wrappers = sorted(
        (row for row in rows if row["kind"] == "wrapper"),
        key=lambda row: row["geometry"]["cap_words"])
    if len(wrappers) < 2:
        raise ValueError("the capture slope needs two capture depths")
    low, high = wrappers[0], wrappers[-1]
    words = high["geometry"]["cap_words"] - low["geometry"]["cap_words"]
    delta = _difference(high["census"], low["census"])
    return {
        "points": [low["key"], high["key"]],
        "extra_capture_words": words,
        "delta": delta,
        "per_capture_word": {column: value / words for column, value in delta.items()},
        "meaning": (
            "two wrapper points differing ONLY in CAP_WORDS. The flip-flop "
            "slope is what the capture RAM costs per 32-bit word today, and it "
            "is a measurement, not an inference"),
    }


def bounds_record(rows: Sequence[Mapping[str, Any]],
                  overhead: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """`N_max` per resource class, per core configuration, per scenario."""
    fixed = {resource: max(0.0, float(cost))
             for resource, cost in overhead["delta_costs"].items()}
    records: List[Dict[str, Any]] = []
    for row in rows:
        if row["kind"] == "wrapper":
            continue
        per_core = census_costs(row["census"])
        for scenario in AVAILABILITY_SCENARIOS:
            for wrapper_included, fixed_overhead in (
                    (False, {r: 0.0 for r in RESOURCE_CLASSES}), (True, fixed)):
                bounds = packing_bound(
                    per_core=per_core, fixed_overhead=fixed_overhead,
                    availability=scenario)
                binding = binding_bound(bounds)
                records.append({
                    "configuration": row["key"],
                    "scenario": scenario.key,
                    "wrapper_overhead_reserved": wrapper_included,
                    "per_core": per_core,
                    "bounds": [bound.as_record() for bound in bounds],
                    "binding_resource": None if binding is None else binding.resource,
                    "n_max": None if binding is None else binding.n_max,
                })
    return records


def build_record(*, rows: Sequence[Mapping[str, Any]],
                 tool_version: str) -> Dict[str, Any]:
    """The machine-readable study: censuses, deltas, device, bounds."""
    overhead = wrapper_overhead_record(rows)
    return {
        "schema": SCHEMA,
        "family": TARGET_FAMILY,
        "tool": {"name": "yosys", "version": tool_version},
        "reported_columns": list(REPORTED_COLUMNS),
        "configurations": [dict(row) for row in rows],
        "wrapper_overhead": overhead,
        "capture_ram_slope": capture_slope_record(rows),
        "device": dict(DEVICE),
        "availability": [scenario.as_record() for scenario in AVAILABILITY_SCENARIOS],
        "bounds": bounds_record(rows, overhead),
    }


def gated_view(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Exactly what an unchanged tree must reproduce, section by section."""
    return {key: record[key] for key in GATED_SECTIONS}


def load_committed() -> Dict[str, Any]:
    """The committed study record, or a loud error naming how to regenerate it."""
    if not LIMITS_JSON.is_file():
        raise FileNotFoundError(
            f"no committed compile-limits record at {LIMITS_JSON}; generate it "
            f"with {REGEN_SCRIPT}")
    return json.loads(LIMITS_JSON.read_text(encoding="utf-8"))


def write_record(record: Mapping[str, Any]) -> None:
    LIMITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    LIMITS_JSON.write_text(
        json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def measure_all(*, workdir: Path, tool_version: str) -> Dict[str, Any]:
    """Synthesize every configuration but the stock one and build the record."""
    rows: List[Dict[str, Any]] = []
    for configuration in configurations():
        if configuration.key == STOCK_KEY:
            rows.append(stock_row(configuration))
            continue
        rows.append(measured_row(configuration, measure(configuration, workdir=workdir)))
    return build_record(rows=rows, tool_version=tool_version)
