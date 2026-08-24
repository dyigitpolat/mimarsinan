"""The study's tables: one census, one memory, one bound rendered as markdown.

Split from `report.py` so the prose file stays a page of sentences and the
table shapes stay a page of columns; neither derives a number, both read the
record.
"""

from __future__ import annotations

from typing import Any, List, Mapping

from mimarsinan.chip_simulation.odin_rtl.limits.device import (
    CENSUS_TO_DEVICE,
    RESOURCE_CLASSES,
    census_costs,
    lutram_bits,
    lutram_lut_sites,
)

#: One RAMB36E2 tile, in bits.
BRAM36_BITS = 36 * 1024

#: The census columns the measurements table prints, with their headings.
COLUMN_TITLES = (
    ("lut_equivalent", "LUT-equiv"),
    ("flip_flops", "FFs"),
    ("carry", "CARRY4"),
    ("lutram", "LUTRAM cells"),
    ("bram36", "RAMB36E2"),
    ("bram18", "RAMB18E2"),
    ("uram", "URAM"),
)


def rows_of(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return list(record["configurations"])


def row_of(record: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    for row in rows_of(record):
        if row["key"] == key:
            return row
    raise KeyError(f"no configuration {key!r} in the record")


def measurements_table(record: Mapping[str, Any]) -> List[str]:
    header = ("| Configuration | " + " | ".join(t for _, t in COLUMN_TITLES)
              + " | LUT sites (incl. LUTRAM) |")
    rule = "| --- | " + " | ".join("---:" for _ in COLUMN_TITLES) + " | ---: |"
    lines = [header, rule]
    for row in rows_of(record):
        census = row["census"]
        cells = " | ".join(f"{census[column]:,}" for column, _ in COLUMN_TITLES)
        sites = census_costs(census)["lut_sites"]
        lines.append(f"| `{row['key']}` | {cells} | {sites:,.0f} |")
    return lines


def memory_table(record: Mapping[str, Any]) -> List[str]:
    lines = [
        "| Configuration | Array | Words x width | Bits | 36 Kb tiles if in BRAM |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in rows_of(record):
        for memory in row["memories"]:
            bits = int(memory["bits"])
            tiles = -(-bits // BRAM36_BITS)
            lines.append(
                f"| `{row['key']}` | `{memory['array']}` "
                f"| {memory['words']:,} x {memory['width']} | {bits:,} | {tiles} |")
    return lines


def lutram_paragraph(record: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    for row in rows_of(record):
        cells = row["census"]["lutram_cells"]
        if not cells:
            continue
        bits = lutram_bits(row["census"])
        sites = lutram_lut_sites(row["census"])
        declared = int(row["memory_bits"])
        lines.append(
            f"- `{row['key']}`: "
            f"{', '.join(f'{n:,} x `{name}`' for name, n in cells.items())}"
            f" = {bits:,} bits of distributed RAM in {sites:,} LUT6 sites, against "
            f"{declared:,} bits declared by its arrays ({bits / declared:.2f}x -- "
            f"a LUT-RAM column is 64 words deep, so both the depth and the width "
            f"round up). The read multiplexing over those columns is counted "
            f"separately, in the LUT-equivalent column.")
    return lines


def bound_lines(record: Mapping[str, Any], *, configuration: str,
                 wrapper_reserved: bool) -> List[str]:
    lines = [
        "| Scenario | " + " | ".join(
            f"N_max({resource})" for resource in RESOURCE_CLASSES) + " | Binds |",
        "| --- | " + " | ".join("---:" for _ in RESOURCE_CLASSES) + " | --- |",
    ]
    for entry in record["bounds"]:
        if entry["configuration"] != configuration:
            continue
        if entry["wrapper_overhead_reserved"] != wrapper_reserved:
            continue
        by_resource = {b["resource"]: b for b in entry["bounds"]}
        cells = " | ".join(
            "n/a" if by_resource[resource]["n_max"] is None
            else f"{by_resource[resource]['n_max']:,}"
            for resource in RESOURCE_CLASSES)
        binding = entry["binding_resource"] or "nothing"
        lines.append(
            f"| {entry['scenario']} | {cells} | **{binding}** "
            f"-> **{entry['n_max']:,}** |")
    return lines


def availability_lines(record: Mapping[str, Any]) -> List[str]:
    lines = [
        "| Scenario | " + " | ".join(
            CENSUS_TO_DEVICE[r] for r in RESOURCE_CLASSES) + " | Status |",
        "| --- | " + " | ".join("---:" for _ in RESOURCE_CLASSES) + " | --- |",
    ]
    for scenario in record["availability"]:
        cells = " | ".join(
            f"{scenario['available'][CENSUS_TO_DEVICE[r]]:,}"
            for r in RESOURCE_CLASSES)
        lines.append(f"| `{scenario['key']}` | {cells} | {scenario['status']} |")
    return lines


def device_lines(record: Mapping[str, Any]) -> List[str]:
    lines = ["| Class | Published | Value used | Derivation |",
             "| --- | --- | ---: | --- |"]
    for column, entry in record["device"]["totals"].items():
        lines.append(
            f"| `{column}` | {entry['published']} | {entry['value']:,} "
            f"| {entry['derivation']} |")
    return lines


def ratio_lines(record: Mapping[str, Any]) -> List[str]:
    """The measured geometry ratios, stated as ratios and nothing more."""
    generated = [row for row in rows_of(record) if row["kind"] == "generated"]
    base = min(generated, key=lambda row: row["geometry"]["max_axons"]
               * row["geometry"]["max_neurons"])
    lines = [
        "| Configuration | axons x neurons | synapse cells | synapse bits vs base "
        "| LUT sites vs base | FFs vs base |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    base_sites = census_costs(base["census"])["lut_sites"]
    base_bits = base["memories"][0]["bits"]
    for row in generated:
        geometry = row["geometry"]
        cells = geometry["max_axons"] * geometry["max_neurons"]
        lines.append(
            f"| `{row['key']}` | {geometry['max_axons']} x "
            f"{geometry['max_neurons']} | {cells:,} "
            f"| {row['memories'][0]['bits'] / base_bits:.2f}x "
            f"| {census_costs(row['census'])['lut_sites'] / base_sites:.2f}x "
            f"| {row['census']['flip_flops'] / base['census']['flip_flops']:.2f}x |")
    lines.append("")
    lines.append(f"The base row is `{base['key']}`.")
    return lines



def variant_bound_sections(record: Mapping[str, Any]) -> List[str]:
    lines: List[str] = []
    for row in rows_of(record):
        if row["kind"] != "generated":
            continue
        lines += [
            f"### The generated variant `{row['key']}`",
            "",
            "Without any wrapper reserved:",
            "",
            *bound_lines(record, configuration=row["key"], wrapper_reserved=False),
            "",
            "With the measured wrapper overhead reserved once:",
            "",
            *bound_lines(record, configuration=row["key"], wrapper_reserved=True),
            "",
        ]
    return lines
