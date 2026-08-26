# ODIN on an FPGA: the compile-limits study

Plan `docs/odin_firing_semantics_and_rtl_export_plan.md` §8, P8 -- the
SYNTHESIS half. It answers one question with measurements: what does one
ODIN-family core cost in FPGA fabric, and how many of them fit an Alveo
U55C. It answers a second question by refusing to: how many fit *after the
Alveo shell takes its share* is NOT measured here and is not guessed at.

## Method

- Tool: `Yosys 0.68+118 (git sha1 144c707b7-dirty, Release, Clang /usr/bin/clang++ 21.1.8)`, target family `xcup`
  -- the UltraScale+ family the U55C's XCU55C part belongs to.
- Every configuration goes through the SAME driver as the P5.5a gate
  (`chip_simulation/odin_rtl/synthesis.py`), the same `synth_xilinx` call
  and the same `stat -json` census partition
  (`chip_simulation/odin_rtl/synth_census.py`). Nothing is measured a
  second way, and the exact script of every row is in
  `hw/fpga/compile_limits.json`.
- The GENERATED variants are the ones plan §7 row 20 proves at zero
  difference against their nevresim/torch twins. The study reads them from
  the same catalog the cosimulation gates read
  (`mapping/export/odin_gen/variants.py`), so a geometry cannot be costed
  here unless a cosimulation proved it.
- The stock 256x256 row is NOT re-measured: it is read from
  `hw/fpga/synth_resources.json`, the P5.5a artifact.
- Regenerate DELIBERATELY with `scripts/hw_tests/regen_compile_limits.py`;
  the [slow] gate `tests/integration/test_odin_compile_limits.py` re-derives
  every non-stock row and requires an exact match.

## What was measured

- `stock_a256n256_vendored` -- stock ODIN core, 256 axons x 256 neurons (vendored + BRAM overlay)
- `gen_a128n128_mb8_per_event` -- generated core, 128 axons x 128 neurons, 8-bit unsigned membrane, per-event law
- `gen_a512n256_mb16_per_event` -- generated core, 512 axons x 256 neurons, 16-bit unsigned membrane, per-event law
- `gen_a256n256_mb16s_sync_fire` -- generated core, 256 axons x 256 neurons, 16-bit signed membrane, sync-fire law
- `wrapper_nc1_prog4096_cap16384` -- kernel wrapper `odin_fpga_kernel_top` at NC=1 (PROG_WORDS=4096, CAP_WORDS=16384) -- sequencer + AXI DMA + capture + one stock core
- `wrapper_nc1_prog4096_cap32768` -- kernel wrapper `odin_fpga_kernel_top` at NC=1 (PROG_WORDS=4096, CAP_WORDS=32768) -- sequencer + AXI DMA + capture + one stock core

| Configuration | LUT-equiv | FFs | CARRY4 | LUTRAM cells | RAMB36E2 | RAMB18E2 | URAM | LUT sites (incl. LUTRAM) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock_a256n256_vendored` | 5,659 | 4,362 | 381 | 0 | 10 | 0 | 0 | 5,659 |
| `gen_a128n128_mb8_per_event` | 1,934 | 1,057 | 8 | 4 | 2 | 0 | 0 | 1,966 |
| `gen_a512n256_mb16_per_event` | 6,522 | 4,133 | 13 | 12 | 16 | 0 | 0 | 6,618 |
| `gen_a256n256_mb16s_sync_fire` | 6,474 | 4,133 | 16 | 12 | 8 | 0 | 0 | 6,570 |
| `wrapper_nc1_prog4096_cap16384` | 6,662 | 5,415 | 508 | 0 | 30 | 0 | 0 | 6,662 |
| `wrapper_nc1_prog4096_cap32768` | 6,665 | 5,415 | 508 | 0 | 46 | 0 | 0 | 6,665 |

`bram_tiles` counts a RAMB18E2 as half a tile; the LUT-equivalent column
adds unpacked `INV` cells to `LUT1..LUT6`, exactly as the P5.5a report
does. No configuration emitted a DSP, a URAM, or a single cell the
census could not classify -- every `unclassified` bucket in the record is
empty -- so those two device columns cannot bind and are reported as
`n/a` in every bound below.

## The memory arithmetic, per configuration

Every array is sized by the spec that generated it (or, for the stock
core, by the overlay wrapper's declared geometry), so the bits column
below is the RTL's own arithmetic and not a second derivation. The last
column divides those bits by a tile and is a FLOOR only: tiles are
allocated by DEPTH and WIDTH, not by bit count, and the paragraphs that
follow do that arithmetic per array against what was measured.

| Configuration | Array | Words x width | Bits | 36 Kb tiles of bits (a FLOOR, not the tile count) |
| --- | --- | --- | ---: | ---: |
| `stock_a256n256_vendored` | `SRAM_256x128_wrapper (neuron)` | 256 x 128 | 32,768 | 1 |
| `stock_a256n256_vendored` | `SRAM_8192x32_wrapper (synapse)` | 8,192 x 32 | 262,144 | 8 |
| `gen_a128n128_mb8_per_event` | `syn_mem` | 2,048 x 32 | 65,536 | 2 |
| `gen_a128n128_mb8_per_event` | `thr_arr` | 128 x 8 | 1,024 | 1 |
| `gen_a128n128_mb8_per_event` | `vmem_arr` | 128 x 8 | 1,024 | 1 |
| `gen_a512n256_mb16_per_event` | `syn_mem` | 16,384 x 32 | 524,288 | 15 |
| `gen_a512n256_mb16_per_event` | `thr_arr` | 256 x 16 | 4,096 | 1 |
| `gen_a512n256_mb16_per_event` | `vmem_arr` | 256 x 16 | 4,096 | 1 |
| `gen_a256n256_mb16s_sync_fire` | `syn_mem` | 8,192 x 32 | 262,144 | 8 |
| `gen_a256n256_mb16s_sync_fire` | `thr_arr` | 256 x 16 | 4,096 | 1 |
| `gen_a256n256_mb16s_sync_fire` | `vmem_arr` | 256 x 16 | 4,096 | 1 |
| `wrapper_nc1_prog4096_cap16384` | `prog_ram` | 4,096 x 32 | 131,072 | 4 |
| `wrapper_nc1_prog4096_cap16384` | `cap_ram` | 16,384 x 32 | 524,288 | 15 |
| `wrapper_nc1_prog4096_cap32768` | `prog_ram` | 4,096 x 32 | 131,072 | 4 |
| `wrapper_nc1_prog4096_cap32768` | `cap_ram` | 32,768 x 32 | 1,048,576 | 29 |

Cross-checking that against the censuses above:

- The STOCK core's two memories are both in block RAM, through the
  `hw/fpga/mem/` overlay. Its 32,768 + 262,144 = 294,912 declared bits occupy 10 RAMB36E2 tiles = 368,640 bits of tile (1.25x).
  The arithmetic is per memory and not per bit: a RAMB36E2 is 1,024 x 36
  at its true-dual-port width and 512 x 72 in simple-dual-port mode, so
  the 8,192 x 32 synapse memory needs 8,192 / 1,024 = 8 tiles and leaves
  4 of each 36 bits unused, while the 256 x 128 neuron memory needs two
  tiles side by side to make a 128-bit word and then uses only 256 of
  each tile's entries. 8 + 2 is the measured 10.
- Every GENERATED variant puts its SYNAPSE memory in BLOCK RAM too, and
  the tile count is the declared depth and nothing else. That is not a
  synthesis accident either: `hw/gen/odin_gen_core.v.tmpl` declares
  `syn_mem` `ram_style = "block"` and gives it ONE synchronous write
  port and ONE REGISTERED read port, whose address is the sweep position
  one cycle ahead -- a tile has no asynchronous read port, and the
  earlier combinational `wire syn_word = syn_mem[syn_index]` could only
  land in SLICEM distributed RAM. The address-ahead read absorbs the
  tile's cycle of latency without moving a count: the per-variant
  cosimulation gates (plan §7 row 20) still hold at zero difference.
  Measured, per variant -- the tiles hold the synapse array and nothing
  else, and the other two arrays are in the two lines after these:

- `gen_a128n128_mb8_per_event`: 2 RAMB36E2 measured, 2 needed by the declared depth (`syn_mem` 2,048 / 1,024 = 2).
- `gen_a512n256_mb16_per_event`: 16 RAMB36E2 measured, 16 needed by the declared depth (`syn_mem` 16,384 / 1,024 = 16).
- `gen_a256n256_mb16s_sync_fire`: 8 RAMB36E2 measured, 8 needed by the declared depth (`syn_mem` 8,192 / 1,024 = 8).

- The generated `thr_arr` is now the array in DISTRIBUTED RAM: it is
  read combinationally by the soma, which is what a threshold compare in
  the same cycle needs. Measured -- one `RAM64M8` is eight 64x1 LUT RAMs in one SLICEM, i.e. 512 bits held in 8 LUT6 sites (AMD UG574, UltraScale architecture CLB):

- `gen_a128n128_mb8_per_event`: 4 x `RAM64M8` = 2,048 bits of distributed RAM in 32 LUT6 sites, against 1,024 bits declared by `thr_arr` (2.00x -- a LUT-RAM column is 64 words deep, so both the depth and the width round up). The read multiplexing over those columns is counted separately, in the LUT-equivalent column.
- `gen_a512n256_mb16_per_event`: 12 x `RAM64M8` = 6,144 bits of distributed RAM in 96 LUT6 sites, against 4,096 bits declared by `thr_arr` (1.50x -- a LUT-RAM column is 64 words deep, so both the depth and the width round up). The read multiplexing over those columns is counted separately, in the LUT-equivalent column.
- `gen_a256n256_mb16s_sync_fire`: 12 x `RAM64M8` = 6,144 bits of distributed RAM in 96 LUT6 sites, against 4,096 bits declared by `thr_arr` (1.50x -- a LUT-RAM column is 64 words deep, so both the depth and the width round up). The read multiplexing over those columns is counted separately, in the LUT-equivalent column.

- The generated `vmem_arr` is not in any RAM at all: yosys converts it to
  registers, which is why the 256-neuron variants both carry 4,096
  membrane flip-flops (256 neurons x 16 bits) on top of their control
  state, and the 128-neuron variant 1,024 (128 x 8). It is READ AND
  WRITTEN in one cycle by the soma, which is not a tile access pattern,
  so it is left as it is.

## How the geometry moves the numbers

Four points do not support a fitted curve, so this section reports
ratios and nothing else.

| Configuration | axons x neurons | synapse cells | synapse bits vs base | LUT sites vs base | FFs vs base |
| --- | --- | ---: | ---: | ---: | ---: |
| `gen_a128n128_mb8_per_event` | 128 x 128 | 16,384 | 1.00x | 1.00x | 1.00x |
| `gen_a512n256_mb16_per_event` | 512 x 256 | 131,072 | 8.00x | 3.37x | 3.91x |
| `gen_a256n256_mb16s_sync_fire` | 256 x 256 | 65,536 | 4.00x | 3.34x | 3.91x |

The base row is `gen_a128n128_mb8_per_event`.

Read against the geometries: the synapse array tracks
axons x neurons x weight_bits and is now paid for in TILES, so it has
left the LUT column -- `gen_a512n256_mb16_per_event` carries 8.00x the base row's synapse bits on 3.37x its LUT sites. What the
LUT column still carries is the soma datapath (whose width follows
membrane_bits + weight_bits) and the threshold array's distributed RAM,
neither of which grows with the crossbar. The flip-flop column tracks
neurons x membrane_bits plus a fixed control block. The sync-fire variant
is the control: same neuron count and same register width as the wide
per-event variant, half the synapse array, and its flip-flop count is
identical while its tiles halve and its LUT sites barely move.

## What the kernel wrapper costs around a core

`odin_fpga_kernel_top` at NC=1 was synthesized at two capture depths. The
overhead below is the wrapper census less the stock core's, i.e. the AXI4-Lite control block, the AXI4 DMA engine, the token sequencer, the SPI master, the AER bridge, the program RAM and the capture RAM, at the depths named in `at_parameters` -- the capture RAM at the depth the wrapper SHIPS with, the program RAM shrunk from its shipped `NC * 262144` words:

| Column | Wrapper overhead (delta vs the stock core) |
| --- | ---: |
| `lut_equivalent` | 1,003 |
| `flip_flops` | 1,053 |
| `carry` | 127 |
| `lutram` | 0 |
| `bram36` | 20 |
| `bram18` | 0 |
| `uram` | 0 |
| `lut_sites` (the bound's LUT class) | 1,003 |

Both of the wrapper's own RAMs infer block RAM, and the 20 RAMB36E2 of overhead above is exactly their declared depth:

- `prog_ram`: 4,096 x 32 = 131,072 bits, 4,096 / 1,024 = 4 RAMB36E2
- `cap_ram`: 16,384 x 32 = 524,288 bits, 16,384 / 1,024 = 16 RAMB36E2

Each has ONE synchronous write port and ONE registered read port, which
is what a tile can be. `cap_ram` reaches that shape through a write
arbiter: its two header words (the events the fabric saw, and the cycle
the program ended on) used to be written at fixed addresses from the
sequencer, a third write port that no tile has, and they now go out
through the SAME port as the streaming record, at drain time, when the
streaming writes have stopped.

`prog_ram` reaches it on the READ side, and it took a routed build to
find out that it had not. The sequencer used to index the array in five
places -- the opcode fetch, the three argument fetches and the TAG alias
-- all at `pc`, under two FSM states. yosys merges those into the ONE
read port the census below reports, so every number in this study said
the program RAM was tiles. Vivado 2022.2 read the same source as
multi-ported and put the whole 262,144x32 array into distributed RAM:
163,840 LUTs as RAM and 24 BRAM tiles total on the routed U55C kernel,
none of them the program. The sequencer now funnels every fetch through
ONE clocked read register at ONE address source, at the cost of a cycle
of fetch latency per program word. THIS TABLE CANNOT CONFIRM THAT FIX:
yosys inferred a tile before the change and infers one after, so the
census barely moves, and only the next cluster build's `kernel_util`
report can say whether Vivado now agrees.

The two wrapper points differ ONLY in `CAP_WORDS` (16,384 extra words), which makes the capture
RAM's cost a measurement:

- `flip_flops`: +0.000000 per capture word
- `lut_equivalent`: +0.000183 per capture word
- `bram36`: +0.000977 per capture word

A capture word costs 0 flip-flops: it is bought in TILES, one RAMB36E2 per 1,024 words. That is a
REVERSAL of what the previous revision of this study measured -- with
three write ports (the streaming record plus two fixed-address header
writes) the capture RAM could not infer a tile and cost 32 flip-flops
per word, which at any shippable depth exceeded the device's entire
register budget and made the kernel unbuildable as written. The
wrapper now SHIPS with `CAP_WORDS = 16,384` -- 4,095 event records held in 16 tiles and 0 flip-flops -- and THAT depth is the one measured above, not a shrunk stand-in for it. The program RAM is still shrunk
(4,096 words against the shipped `NC * 262144`), and every row that
carries it says so.

## The device, with provenance

datasheet -- AMD Alveo U55C product table / DS978 (XCU55C, three SLRs, 16 GB HBM2). No figure in this table was measured on hardware.

| Class | Published | Value used | Derivation |
| --- | --- | ---: | --- |
| `luts` | 1,304K | 1,304,000 | product-table LUT count, taken as published |
| `registers` | 2,607K | 2,607,000 | product-table register count, taken as published |
| `bram36_tiles` | 70.9 Mb total block RAM | 2,016 | 70.9 Mib / 36 Kib per RAMB36 tile = 2,016 tiles |
| `uram_blocks` | 270 Mb UltraRAM | 960 | 270 Mib / 288 Kib per URAM288 block = 960 blocks |
| `dsp_slices` | 9,024 | 9,024 | product-table DSP count; no configuration in this study uses a DSP, so it never binds |

### The number this study does NOT have

UNKNOWN UNTIL P7b: the Alveo shell (XDMA/HBM AXI infrastructure, the dynamic-region boundary) consumes device resources this study has not measured. A Vivado `report_utilization` on the linked `xilinx_u55c_gen3x16_xdma_base_3` design is the only source of that number, and Vivado exists only on HACC (owner-gated login), so it is the B0-adjacent step of the BOARD half of P8.

## The packing bound

There is no single headline number, and this section refuses to print
one. The bound is a FUNCTION of which resource you ask about and which
availability you assume:

```
N_max(resource) = floor( (available(resource) - fixed_overhead(resource))
                         / per_core(resource) )
```

`available` comes from one of two scenarios, and only the first is
datasheet-grounded:

| Scenario | luts | registers | bram36_tiles | uram_blocks | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `datasheet_total` | 1,304,000 | 2,607,000 | 2,016 | 960 | datasheet, but NOT achievable -- it ignores the shell entirely |
| `assumed_shell_30pct` | 912,800 | 1,824,900 | 1,411 | 672 | ASSUMPTION, not a measurement. UNKNOWN UNTIL P7b: the Alveo shell (XDMA/HBM AXI infrastructure, the dynamic-region boundary) consumes device resources this study has not measured. A Vivado `report_utilization` on the linked `xilinx_u55c_gen3x16_xdma_base_3` design is the only source of that number, and Vivado exists only on HACC (owner-gated login), so it is the B0-adjacent step of the BOARD half of P8. |

`fixed_overhead` is either zero (cores alone, no host interface at all --
not a buildable design, but the cleanest read of the core cost) or the
measured wrapper overhead above.

### The stock core (`stock_a256n256_vendored`)

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 230 | 597 | 201 | n/a | **bram36** -> **201** |
| assumed_shell_30pct | 161 | 418 | 141 | n/a | **bram36** -> **141** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 230 | 597 | 199 | n/a | **bram36** -> **199** |
| assumed_shell_30pct | 161 | 418 | 139 | n/a | **bram36** -> **139** |

### The generated variant `gen_a128n128_mb8_per_event`

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 663 | 2,466 | 1,008 | n/a | **lut_sites** -> **663** |
| assumed_shell_30pct | 464 | 1,726 | 705 | n/a | **lut_sites** -> **464** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 662 | 2,465 | 998 | n/a | **lut_sites** -> **662** |
| assumed_shell_30pct | 463 | 1,725 | 695 | n/a | **lut_sites** -> **463** |

### The generated variant `gen_a512n256_mb16_per_event`

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 197 | 630 | 126 | n/a | **bram36** -> **126** |
| assumed_shell_30pct | 137 | 441 | 88 | n/a | **bram36** -> **88** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 196 | 630 | 124 | n/a | **bram36** -> **124** |
| assumed_shell_30pct | 137 | 441 | 86 | n/a | **bram36** -> **86** |

### The generated variant `gen_a256n256_mb16s_sync_fire`

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 198 | 630 | 252 | n/a | **lut_sites** -> **198** |
| assumed_shell_30pct | 138 | 441 | 176 | n/a | **lut_sites** -> **138** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 198 | 630 | 249 | n/a | **lut_sites** -> **198** |
| assumed_shell_30pct | 138 | 441 | 173 | n/a | **lut_sites** -> **138** |

## The verdict

- For the STOCK core the binding resource is **block RAM**: 10 RAMB36E2 tiles per core against the device's 2,016.
- The GENERATED variants no longer bind on one resource as a family.
  Their synapse memories are in block RAM now, so what binds is whichever
  resource the geometry runs out of first, and the record has both:
  - **bram36**: `gen_a512n256_mb16_per_event`
  - **lut_sites**: `gen_a128n128_mb8_per_event`, `gen_a256n256_mb16s_sync_fire`
  A LUT-bound variant is one whose soma datapath and threshold array
  cost more than its synapse tiles do; it is no longer a variant paying
  for a distributed-RAM crossbar.
- Which classes bind, across every configuration and scenario in the
  record: `bram36` for 2 of 4 core configurations, `lut_sites` for 2 of 4 core configurations.
- No bound in this document is a claim about a card until the P7b
  utilization report exists.

## What the BOARD half of P8 still owes

- **The Vivado utilization report (B0-adjacent).** `v++` link on HACC for
  `xilinx_u55c_gen3x16_xdma_base_3`, then `report_utilization` on the
  implemented design. That single artifact replaces BOTH the shell
  assumption above and the yosys-vs-Vivado mapping caveat below, and it
  is the only thing that can turn these bounds into a claim about a card.
- **Measured fidelity.** The on-board certificate campaign (plan §7 row
  21, R11b): deployed counts against the nevresim twin at zero
  difference, on silicon rather than in a cosimulation.
- **The measured campaign.** Programming and execution walls from the real
  card, which the deployment record's timing fragment already has a place
  for.
- **A Vivado read of the two memory fixes.** The two RTL defects the
  previous revision of this study found -- the capture RAM's three write
  ports and the generated core's asynchronous synapse read -- are FIXED
  in the tree these numbers were measured on, and every number above
  moved because of it. yosys says both memories now infer tiles; only
  Vivado can say the same about the build that goes on the card.

## The caveat this study must carry

yosys is not Vivado. Its Xilinx mapping is generic: it emits `CARRY4`
where UltraScale+ has `CARRY8`, leaves `INV` unpacked, does not model LUT
pairing, and chooses distributed RAM by its own heuristics rather than
Vivado's. Every LUT and FF number here is an ORDER-OF-MAGNITUDE budget,
the BRAM counts are the most trustworthy column because tile geometry is
discrete, and nothing here says anything about timing closure, placement,
routing across the three SLRs, or the HBM AXI infrastructure. The bounds
are ceilings on a ceiling.
