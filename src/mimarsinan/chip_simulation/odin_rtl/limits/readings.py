"""The study's READINGS: the sentences the record derives rather than tabulates.

Split from `report.py` (which assembles the document) and from `tables.py`
(which shapes its columns) so that a paragraph making an arithmetic claim about
a measured census lives next to the arithmetic. Nothing here measures anything:
every number is read out of the committed record.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.odin_rtl.limits.configurations import STOCK_KEY
from mimarsinan.chip_simulation.odin_rtl.limits.device import (
    LUTRAM_PROVENANCE,
    census_costs,
    lutram_bits,
    lutram_lut_sites,
)
from mimarsinan.chip_simulation.odin_rtl.limits.tables import (
    BRAM36_BITS,
    BRAM36_WORDS,
    row_of,
    rows_of,
    tile_arrays,
    tiles_for,
)


def block_ram_paragraph(record: Mapping[str, Any]) -> List[str]:
    """Per generated row: the tiles measured, against the depth that needs them."""
    lines: List[str] = []
    for row in rows_of(record):
        if row["kind"] != "generated":
            continue
        arrays = tile_arrays(row)
        predicted = sum(tiles_for(memory) for memory in arrays)
        terms = " + ".join(
            f"`{memory['array']}` {memory['words']:,} / {BRAM36_WORDS:,} = "
            f"{tiles_for(memory)}" for memory in arrays)
        lines.append(
            f"- `{row['key']}`: {row['census']['bram36']} RAMB36E2 measured, "
            f"{predicted} needed by the declared depth ({terms}).")
    return lines


def lutram_paragraph(record: Mapping[str, Any]) -> List[str]:
    """Per row with distributed RAM: the cells measured, against what declares them."""
    lines: List[str] = []
    for row in rows_of(record):
        cells = row["census"]["lutram_cells"]
        if not cells:
            continue
        bits = lutram_bits(row["census"])
        sites = lutram_lut_sites(row["census"])
        threshold = next(
            (memory for memory in row["memories"] if memory["array"] == "thr_arr"),
            None)
        declared = int(row["memory_bits"] if threshold is None else threshold["bits"])
        against = "its arrays" if threshold is None else "`thr_arr`"
        lines.append(
            f"- `{row['key']}`: "
            f"{', '.join(f'{n:,} x `{name}`' for name, n in cells.items())}"
            f" = {bits:,} bits of distributed RAM in {sites:,} LUT6 sites, against "
            f"{declared:,} bits declared by {against} ({bits / declared:.2f}x -- "
            f"a LUT-RAM column is 64 words deep, so both the depth and the width "
            f"round up). The read multiplexing over those columns is counted "
            f"separately, in the LUT-equivalent column.")
    return lines


def memory_cross_check_lines(record: Mapping[str, Any]) -> List[str]:
    """Where every declared array actually landed, array by array."""
    stock = row_of(record, STOCK_KEY)
    tiles = stock["census"]["bram36"]
    return [
        "- The STOCK core's two memories are both in block RAM, through the",
        f"  `hw/fpga/mem/` overlay. Its {stock['memories'][0]['bits']:,} + "
        f"{stock['memories'][1]['bits']:,} = {stock['memory_bits']:,} declared "
        f"bits occupy {tiles} RAMB36E2 tiles = {tiles * BRAM36_BITS:,} bits of "
        f"tile ({tiles * BRAM36_BITS / stock['memory_bits']:.2f}x).",
        "  The arithmetic is per memory and not per bit: a RAMB36E2 is 1,024 x 36",
        "  at its true-dual-port width and 512 x 72 in simple-dual-port mode, so",
        "  the 8,192 x 32 synapse memory needs 8,192 / 1,024 = 8 tiles and leaves",
        "  4 of each 36 bits unused, while the 256 x 128 neuron memory needs two",
        "  tiles side by side to make a 128-bit word and then uses only 256 of",
        "  each tile's entries. 8 + 2 is the measured 10.",
        "- Every GENERATED variant puts its SYNAPSE memory in BLOCK RAM too, and",
        "  the tile count is the declared depth and nothing else. That is not a",
        "  synthesis accident either: `hw/gen/odin_gen_core.v.tmpl` declares",
        "  `syn_mem` `ram_style = \"block\"` and gives it ONE synchronous write",
        "  port and ONE REGISTERED read port, whose address is the sweep position",
        "  one cycle ahead -- a tile has no asynchronous read port, and the",
        "  earlier combinational `wire syn_word = syn_mem[syn_index]` could only",
        "  land in SLICEM distributed RAM. The address-ahead read absorbs the",
        "  tile's cycle of latency without moving a count: the per-variant",
        "  cosimulation gates (plan §7 row 20) still hold at zero difference.",
        "  Measured, per variant -- the tiles hold the synapse array and nothing",
        "  else, and the other two arrays are in the two lines after these:",
        "",
        *block_ram_paragraph(record),
        "",
        "- The generated `thr_arr` is now the array in DISTRIBUTED RAM: it is",
        "  read combinationally by the soma, which is what a threshold compare in",
        f"  the same cycle needs. Measured -- {LUTRAM_PROVENANCE}:",
        "",
        *lutram_paragraph(record),
        "",
        "- The generated `vmem_arr` is not in any RAM at all: yosys converts it to",
        "  registers, which is why the 256-neuron variants both carry 4,096",
        "  membrane flip-flops (256 neurons x 16 bits) on top of their control",
        "  state, and the 128-neuron variant 1,024 (128 x 8). It is READ AND",
        "  WRITTEN in one cycle by the soma, which is not a tile access pattern,",
        "  so it is left as it is.",
    ]


def wrapper_tile_lines(record: Mapping[str, Any],
                       overhead: Mapping[str, Any]) -> List[str]:
    """The wrapper's own two RAMs, each against the tiles its depth needs."""
    wrapper = row_of(record, overhead["wrapper"])
    return [
        f"- `{memory['array']}`: {memory['words']:,} x {memory['width']} = "
        f"{memory['bits']:,} bits, {memory['words']:,} / {BRAM36_WORDS:,} = "
        f"{tiles_for(memory)} RAMB36E2"
        for memory in tile_arrays(wrapper)
    ]


def widest_ratio_sentence(record: Mapping[str, Any]) -> str:
    """The widest variant's LUT growth against its synapse growth, from the record."""
    generated = [row for row in rows_of(record) if row["kind"] == "generated"]
    base = min(generated, key=lambda row: row["memories"][0]["bits"])
    widest = max(generated, key=lambda row: row["memories"][0]["bits"])
    bits = widest["memories"][0]["bits"] / base["memories"][0]["bits"]
    sites = (census_costs(widest["census"])["lut_sites"]
             / census_costs(base["census"])["lut_sites"])
    return (f"`{widest['key']}` carries {bits:.2f}x the base row's synapse bits "
            f"on {sites:.2f}x its LUT sites.")


def generated_binding_lines(record: Mapping[str, Any]) -> List[str]:
    """Which generated variant binds on what, read off the bounds section."""
    generated = [row["key"] for row in rows_of(record)
                 if row["kind"] == "generated"]
    by_resource: Dict[str, List[str]] = {}
    for entry in record["bounds"]:
        if entry["configuration"] not in generated:
            continue
        keys = by_resource.setdefault(entry["binding_resource"] or "nothing", [])
        if entry["configuration"] not in keys:
            keys.append(entry["configuration"])
    return [
        f"  - **{resource}**: " + ", ".join(f"`{key}`" for key in keys)
        for resource, keys in sorted(by_resource.items())
    ]


def verdict_lines(record: Mapping[str, Any]) -> List[str]:
    """The study's verdict: which class binds, for which configuration, and why."""
    binding: Dict[str, List[str]] = {}
    all_keys = [entry["configuration"] for entry in record["bounds"]]
    for entry in record["bounds"]:
        binding.setdefault(entry["binding_resource"] or "nothing", []).append(
            entry["configuration"])
    stock = row_of(record, STOCK_KEY)
    return [
        "- For the STOCK core the binding resource is **block RAM**: "
        f"{stock['census']['bram36']} RAMB36E2 tiles per core against the "
        "device's 2,016.",
        "- The GENERATED variants no longer bind on one resource as a family.",
        "  Their synapse memories are in block RAM now, so what binds is whichever",
        "  resource the geometry runs out of first, and the record has both:",
        *generated_binding_lines(record),
        "  A LUT-bound variant is one whose soma datapath and threshold array",
        "  cost more than its synapse tiles do; it is no longer a variant paying",
        "  for a distributed-RAM crossbar.",
        "- Which classes bind, across every configuration and scenario in the",
        "  record: "
        + ", ".join(
            f"`{resource}` for {len(set(keys))} of {len(set(all_keys))} core "
            f"configurations"
            for resource, keys in sorted(binding.items())) + ".",
        "- No bound in this document is a claim about a card until the P7b",
        "  utilization report exists.",
    ]
