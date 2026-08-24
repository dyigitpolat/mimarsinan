"""The committed study prose: docs/odin_fpga_compile_limits_study.md.

Rendered from the record so a number can never drift out of the sentence that
explains it; the fast gate asserts the file on disk equals this rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from mimarsinan.chip_simulation.odin_rtl.limits.configurations import (
    STOCK_KEY,
    WRAPPER_TOP,
)
from mimarsinan.chip_simulation.odin_rtl.limits.device import LUTRAM_PROVENANCE
from mimarsinan.chip_simulation.odin_rtl.limits.tables import (
    BRAM36_BITS,
    COLUMN_TITLES,
    availability_lines,
    bound_lines,
    device_lines,
    lutram_paragraph,
    measurements_table,
    memory_table,
    ratio_lines,
    row_of,
    rows_of,
    variant_bound_sections,
)


def render_study(record: Mapping[str, Any]) -> str:
    """The committed study; the fast gate asserts the file equals this string."""
    stock = row_of(record, STOCK_KEY)
    overhead = record["wrapper_overhead"]
    slope = record["capture_ram_slope"]
    shipped = row_of(record, overhead["wrapper"])["geometry"]
    ff_per_word = slope["per_capture_word"]["flip_flops"]
    shipped_cap_ffs = int(ff_per_word * shipped["shipped_cap_words"])
    shipped_cap_luts = int(
        slope["per_capture_word"]["lut_equivalent"] * shipped["shipped_cap_words"])
    device = record["device"]
    lines: List[str] = [
        "# ODIN on an FPGA: the compile-limits study",
        "",
        "Plan `docs/odin_firing_semantics_and_rtl_export_plan.md` §8, P8 -- the",
        "SYNTHESIS half. It answers one question with measurements: what does one",
        "ODIN-family core cost in FPGA fabric, and how many of them fit an Alveo",
        "U55C. It answers a second question by refusing to: how many fit *after the",
        "Alveo shell takes its share* is NOT measured here and is not guessed at.",
        "",
        "## Method",
        "",
        f"- Tool: `{record['tool']['version']}`, target family `{record['family']}`",
        "  -- the UltraScale+ family the U55C's XCU55C part belongs to.",
        "- Every configuration goes through the SAME driver as the P5.5a gate",
        "  (`chip_simulation/odin_rtl/synthesis.py`), the same `synth_xilinx` call",
        "  and the same `stat -json` census partition",
        "  (`chip_simulation/odin_rtl/synth_census.py`). Nothing is measured a",
        "  second way, and the exact script of every row is in",
        "  `hw/fpga/compile_limits.json`.",
        "- The GENERATED variants are the ones plan §7 row 20 proves at zero",
        "  difference against their nevresim/torch twins. The study reads them from",
        "  the same catalog the cosimulation gates read",
        "  (`mapping/export/odin_gen/variants.py`), so a geometry cannot be costed",
        "  here unless a cosimulation proved it.",
        f"- The stock 256x256 row is NOT re-measured: it is read from",
        f"  `{stock['measured_by']}`, the P5.5a artifact.",
        f"- Regenerate DELIBERATELY with `scripts/hw_tests/regen_compile_limits.py`;",
        "  the [slow] gate `tests/integration/test_odin_compile_limits.py` re-derives",
        "  every non-stock row and requires an exact match.",
        "",
        "## What was measured",
        "",
        *[f"- `{row['key']}` -- {row['label']}" for row in rows_of(record)],
        "",
        *measurements_table(record),
        "",
        "`bram_tiles` counts a RAMB18E2 as half a tile; the LUT-equivalent column",
        "adds unpacked `INV` cells to `LUT1..LUT6`, exactly as the P5.5a report",
        "does. No configuration emitted a DSP, a URAM, or a single cell the",
        "census could not classify -- every `unclassified` bucket in the record is",
        "empty -- so those two device columns cannot bind and are reported as",
        "`n/a` in every bound below.",
        "",
        "## The memory arithmetic, per configuration",
        "",
        "Every array is sized by the spec that generated it (or, for the stock",
        "core, by the overlay wrapper's declared geometry), so the bits column",
        "below is the RTL's own arithmetic and not a second derivation.",
        "",
        *memory_table(record),
        "",
        "Cross-checking that against the censuses above:",
        "",
        "- The STOCK core is the only configuration whose SYNAPSE memory reaches",
        f"  block RAM. Its {stock['memories'][0]['bits']:,} + "
        f"{stock['memories'][1]['bits']:,} = {stock['memory_bits']:,} declared "
        f"bits occupy {stock['census']['bram36']} RAMB36E2 tiles = "
        f"{stock['census']['bram36'] * BRAM36_BITS:,} bits of tile "
        f"({stock['census']['bram36'] * BRAM36_BITS / stock['memory_bits']:.2f}x).",
        "  The arithmetic is per memory and not per bit: a RAMB36E2 is 1,024 x 36",
        "  at its true-dual-port width and 512 x 72 in simple-dual-port mode, so",
        "  the 8,192 x 32 synapse memory needs 8,192 / 1,024 = 8 tiles and leaves",
        "  4 of each 36 bits unused, while the 256 x 128 neuron memory needs two",
        "  tiles side by side to make a 128-bit word and then uses only 256 of",
        "  each tile's entries. 8 + 2 is the measured 10.",
        "- Every GENERATED variant puts its SYNAPSE memory in DISTRIBUTED RAM",
        "  instead. That is not a synthesis accident: the template reads the",
        "  synapse word combinationally (`wire syn_word = syn_mem[syn_index]` in",
        "  `hw/gen/odin_gen_core.v.tmpl`), and a block-RAM tile has no",
        "  asynchronous read port. The stock core reaches BRAM only because the",
        "  `hw/fpga/mem/` overlay gives it a REGISTERED read. Measured, per",
        f"  variant -- {LUTRAM_PROVENANCE}:",
        "",
        *lutram_paragraph(record),
        "",
        "- The generated `thr_arr` is the one generated array that DOES reach a",
        "  tile: the two 256 x 16 threshold memories each take a single RAMB18E2",
        "  (4,096 bits into an 18 Kb tile), because their read address is",
        "  registered where the synapse read is not.",
        "- The generated `vmem_arr` is not in any RAM at all: yosys converts it to",
        "  registers, which is why the 256-neuron variants both carry 4,096",
        "  membrane flip-flops (256 neurons x 16 bits) on top of their control",
        "  state, and the 128-neuron variant 1,024 (128 x 8).",
        "",
        "## How the geometry moves the numbers",
        "",
        "Four points do not support a fitted curve, so this section reports",
        "ratios and nothing else.",
        "",
        *ratio_lines(record),
        "",
        "Read against the geometries: the synapse array (and therefore the LUTRAM",
        "and the LUT-equivalent column that carries its read multiplexing) tracks",
        "axons x neurons x weight_bits, while the flip-flop column tracks",
        "neurons x membrane_bits plus a fixed control block. The sync-fire variant",
        "is the control: same neuron count and same register width as the wide",
        "per-event variant, half the synapse array, and its flip-flop count is",
        "identical while its LUTRAM halves.",
        "",
        "## What the kernel wrapper costs around a core",
        "",
        f"`{WRAPPER_TOP}` at NC=1 was synthesized at two capture depths. The",
        f"overhead below is {overhead['meaning']}:",
        "",
        "| Column | Wrapper overhead (delta vs the stock core) |",
        "| --- | ---: |",
        *[f"| `{column}` | {overhead['delta'][column]:,.0f} |"
          for column, _ in COLUMN_TITLES],
        f"| `lut_sites` (the bound's LUT class) "
        f"| {overhead['delta_costs']['lut_sites']:,.0f} |",
        "",
        "The wrapper's `prog_ram` DOES infer block RAM -- the +4 RAMB36E2 above",
        "is its 131,072 bits at the synthesized depth -- because every one of its",
        "reads lands in a register on the same clock, which is what a tile's",
        "synchronous read port can be. The capture RAM does not, and that is the",
        "next paragraph.",
        "",
        f"The two wrapper points differ ONLY in `CAP_WORDS` "
        f"({slope['extra_capture_words']:,} extra words), which makes the capture",
        "RAM's cost a measurement:",
        "",
        *[f"- `{column}`: {slope['per_capture_word'][column]:+.2f} per capture word"
          for column in ("flip_flops", "lut_equivalent", "bram36")],
        "",
        "**This is the study's sharpest finding.** The capture RAM does not infer",
        "block RAM -- it has two write ports at fixed addresses plus the streaming",
        f"write -- so it costs {ff_per_word:.0f} flip-flops per 32-bit word. The",
        f"wrapper SHIPS with `CAP_WORDS = {shipped['shipped_cap_words']:,}`, which",
        f"is {shipped_cap_ffs:,} flip-flops: "
        f"{shipped_cap_ffs / record['device']['totals']['registers']['value']:.0%} of",
        "the entire device's registers, for the capture buffer alone -- and its",
        f"read multiplexing is another {shipped_cap_luts:,} LUT sites, "
        f"{shipped_cap_luts / record['device']['totals']['luts']['value']:.0%} of",
        "the LUTs. The kernel as written therefore cannot be built at its",
        "shipped capture depth, and",
        "the numbers below are for the SHRUNK depths named in the table. Giving",
        "`cap_ram` a single registered read port and a single write port -- the",
        "same treatment `hw/fpga/mem/` gave the stock memories -- is the fix, and",
        "it is work the BOARD half of P8 owes.",
        "",
        "## The device, with provenance",
        "",
        f"{device['provenance']}",
        "",
        *device_lines(record),
        "",
        "### The number this study does NOT have",
        "",
        f"{device['shell_overhead']}",
        "",
        "## The packing bound",
        "",
        "There is no single headline number, and this section refuses to print",
        "one. The bound is a FUNCTION of which resource you ask about and which",
        "availability you assume:",
        "",
        "```",
        "N_max(resource) = floor( (available(resource) - fixed_overhead(resource))",
        "                         / per_core(resource) )",
        "```",
        "",
        "`available` comes from one of two scenarios, and only the first is",
        "datasheet-grounded:",
        "",
        *availability_lines(record),
        "",
        "`fixed_overhead` is either zero (cores alone, no host interface at all --",
        "not a buildable design, but the cleanest read of the core cost) or the",
        "measured wrapper overhead above.",
        "",
        f"### The stock core (`{STOCK_KEY}`)",
        "",
        "Without any wrapper reserved:",
        "",
        *bound_lines(record, configuration=STOCK_KEY, wrapper_reserved=False),
        "",
        "With the measured wrapper overhead reserved once:",
        "",
        *bound_lines(record, configuration=STOCK_KEY, wrapper_reserved=True),
        "",
        *variant_bound_sections(record),
        "## The verdict",
        "",
        *_verdict_lines(record),
        "",
        "## What the BOARD half of P8 still owes",
        "",
        "- **The Vivado utilization report (B0-adjacent).** `v++` link on HACC for",
        "  `xilinx_u55c_gen3x16_xdma_base_3`, then `report_utilization` on the",
        "  implemented design. That single artifact replaces BOTH the shell",
        "  assumption above and the yosys-vs-Vivado mapping caveat below, and it",
        "  is the only thing that can turn these bounds into a claim about a card.",
        "- **Measured fidelity.** The on-board certificate campaign (plan §7 row",
        "  21, R11b): deployed counts against the nevresim twin at zero",
        "  difference, on silicon rather than in a cosimulation.",
        "- **The measured campaign.** Programming and execution walls from the real",
        "  card, which the deployment record's timing fragment already has a place",
        "  for.",
        "- **The two RTL defects this study found**, both of which change the",
        "  numbers above: the capture RAM that costs flip-flops per word, and the",
        "  generated core's asynchronous synapse read that keeps every variant out",
        "  of block RAM.",
        "",
        "## The caveat this study must carry",
        "",
        "yosys is not Vivado. Its Xilinx mapping is generic: it emits `CARRY4`",
        "where UltraScale+ has `CARRY8`, leaves `INV` unpacked, does not model LUT",
        "pairing, and chooses distributed RAM by its own heuristics rather than",
        "Vivado's. Every LUT and FF number here is an ORDER-OF-MAGNITUDE budget,",
        "the BRAM counts are the most trustworthy column because tile geometry is",
        "discrete, and nothing here says anything about timing closure, placement,",
        "routing across the three SLRs, or the HBM AXI infrastructure. The bounds",
        "are ceilings on a ceiling.",
        "",
    ]
    return "\n".join(lines)


def _verdict_lines(record: Mapping[str, Any]) -> List[str]:
    binding: Dict[str, List[str]] = {}
    all_keys = [entry["configuration"] for entry in record["bounds"]]
    for entry in record["bounds"]:
        binding.setdefault(entry["binding_resource"] or "nothing", []).append(
            entry["configuration"])
    stock = row_of(record, STOCK_KEY)
    lines = [
        "- For the STOCK core the binding resource is **block RAM**: "
        f"{stock['census']['bram36']} RAMB36E2 tiles per core against the "
        "device's 2,016.",
        "- For every GENERATED variant the binding resource is **LUT sites**,",
        "  because the synapse memory never reaches a block-RAM tile and pays for",
        "  itself in SLICEM LUTs instead. That is a property of the emitted RTL,",
        "  not of the geometry, and it is fixable.",
        "- Which classes bind, across every configuration and scenario in the",
        "  record: "
        + ", ".join(
            f"`{resource}` for {len(set(keys))} of {len(set(all_keys))} core "
            f"configurations"
            for resource, keys in sorted(binding.items())) + ".",
        "- No bound in this document is a claim about a card until the P7b",
        "  utilization report exists.",
    ]
    return lines
