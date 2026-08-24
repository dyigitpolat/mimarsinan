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
- `wrapper_nc1_prog4096_cap1024` -- kernel wrapper `odin_fpga_kernel_top` at NC=1 (PROG_WORDS=4096, CAP_WORDS=1024) -- sequencer + AXI DMA + capture + one stock core
- `wrapper_nc1_prog4096_cap2048` -- kernel wrapper `odin_fpga_kernel_top` at NC=1 (PROG_WORDS=4096, CAP_WORDS=2048) -- sequencer + AXI DMA + capture + one stock core

| Configuration | LUT-equiv | FFs | CARRY4 | LUTRAM cells | RAMB36E2 | RAMB18E2 | URAM | LUT sites (incl. LUTRAM) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stock_a256n256_vendored` | 5,659 | 4,362 | 381 | 0 | 10 | 0 | 0 | 5,659 |
| `gen_a128n128_mb8_per_event` | 2,274 | 1,064 | 8 | 164 | 0 | 0 | 0 | 3,586 |
| `gen_a512n256_mb16_per_event` | 10,162 | 4,158 | 13 | 1,280 | 0 | 1 | 0 | 20,402 |
| `gen_a256n256_mb16s_sync_fire` | 8,571 | 4,158 | 16 | 640 | 0 | 1 | 0 | 13,691 |
| `wrapper_nc1_prog4096_cap1024` | 25,928 | 38,245 | 508 | 0 | 14 | 0 | 0 | 25,928 |
| `wrapper_nc1_prog4096_cap2048` | 45,459 | 71,013 | 508 | 0 | 14 | 0 | 0 | 45,459 |

`bram_tiles` counts a RAMB18E2 as half a tile; the LUT-equivalent column
adds unpacked `INV` cells to `LUT1..LUT6`, exactly as the P5.5a report
does. No configuration emitted a DSP, a URAM, or a single cell the
census could not classify -- every `unclassified` bucket in the record is
empty -- so those two device columns cannot bind and are reported as
`n/a` in every bound below.

## The memory arithmetic, per configuration

Every array is sized by the spec that generated it (or, for the stock
core, by the overlay wrapper's declared geometry), so the bits column
below is the RTL's own arithmetic and not a second derivation.

| Configuration | Array | Words x width | Bits | 36 Kb tiles if in BRAM |
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
| `wrapper_nc1_prog4096_cap1024` | `prog_ram` | 4,096 x 32 | 131,072 | 4 |
| `wrapper_nc1_prog4096_cap1024` | `cap_ram` | 1,024 x 32 | 32,768 | 1 |
| `wrapper_nc1_prog4096_cap2048` | `prog_ram` | 4,096 x 32 | 131,072 | 4 |
| `wrapper_nc1_prog4096_cap2048` | `cap_ram` | 2,048 x 32 | 65,536 | 2 |

Cross-checking that against the censuses above:

- The STOCK core is the only configuration whose SYNAPSE memory reaches
  block RAM. Its 32,768 + 262,144 = 294,912 declared bits occupy 10 RAMB36E2 tiles = 368,640 bits of tile (1.25x).
  The arithmetic is per memory and not per bit: a RAMB36E2 is 1,024 x 36
  at its true-dual-port width and 512 x 72 in simple-dual-port mode, so
  the 8,192 x 32 synapse memory needs 8,192 / 1,024 = 8 tiles and leaves
  4 of each 36 bits unused, while the 256 x 128 neuron memory needs two
  tiles side by side to make a 128-bit word and then uses only 256 of
  each tile's entries. 8 + 2 is the measured 10.
- Every GENERATED variant puts its SYNAPSE memory in DISTRIBUTED RAM
  instead. That is not a synthesis accident: the template reads the
  synapse word combinationally (`wire syn_word = syn_mem[syn_index]` in
  `hw/gen/odin_gen_core.v.tmpl`), and a block-RAM tile has no
  asynchronous read port. The stock core reaches BRAM only because the
  `hw/fpga/mem/` overlay gives it a REGISTERED read. Measured, per
  variant -- one `RAM64M8` is eight 64x1 LUT RAMs in one SLICEM, i.e. 512 bits held in 8 LUT6 sites (AMD UG574, UltraScale architecture CLB):

- `gen_a128n128_mb8_per_event`: 164 x `RAM64M8` = 83,968 bits of distributed RAM in 1,312 LUT6 sites, against 67,584 bits declared by its arrays (1.24x -- a LUT-RAM column is 64 words deep, so both the depth and the width round up). The read multiplexing over those columns is counted separately, in the LUT-equivalent column.
- `gen_a512n256_mb16_per_event`: 1,280 x `RAM64M8` = 655,360 bits of distributed RAM in 10,240 LUT6 sites, against 532,480 bits declared by its arrays (1.23x -- a LUT-RAM column is 64 words deep, so both the depth and the width round up). The read multiplexing over those columns is counted separately, in the LUT-equivalent column.
- `gen_a256n256_mb16s_sync_fire`: 640 x `RAM64M8` = 327,680 bits of distributed RAM in 5,120 LUT6 sites, against 270,336 bits declared by its arrays (1.21x -- a LUT-RAM column is 64 words deep, so both the depth and the width round up). The read multiplexing over those columns is counted separately, in the LUT-equivalent column.

- The generated `thr_arr` is the one generated array that DOES reach a
  tile: the two 256 x 16 threshold memories each take a single RAMB18E2
  (4,096 bits into an 18 Kb tile), because their read address is
  registered where the synapse read is not.
- The generated `vmem_arr` is not in any RAM at all: yosys converts it to
  registers, which is why the 256-neuron variants both carry 4,096
  membrane flip-flops (256 neurons x 16 bits) on top of their control
  state, and the 128-neuron variant 1,024 (128 x 8).

## How the geometry moves the numbers

Four points do not support a fitted curve, so this section reports
ratios and nothing else.

| Configuration | axons x neurons | synapse cells | synapse bits vs base | LUT sites vs base | FFs vs base |
| --- | --- | ---: | ---: | ---: | ---: |
| `gen_a128n128_mb8_per_event` | 128 x 128 | 16,384 | 1.00x | 1.00x | 1.00x |
| `gen_a512n256_mb16_per_event` | 512 x 256 | 131,072 | 8.00x | 5.69x | 3.91x |
| `gen_a256n256_mb16s_sync_fire` | 256 x 256 | 65,536 | 4.00x | 3.82x | 3.91x |

The base row is `gen_a128n128_mb8_per_event`.

Read against the geometries: the synapse array (and therefore the LUTRAM
and the LUT-equivalent column that carries its read multiplexing) tracks
axons x neurons x weight_bits, while the flip-flop column tracks
neurons x membrane_bits plus a fixed control block. The sync-fire variant
is the control: same neuron count and same register width as the wide
per-event variant, half the synapse array, and its flip-flop count is
identical while its LUTRAM halves.

## What the kernel wrapper costs around a core

`odin_fpga_kernel_top` at NC=1 was synthesized at two capture depths. The
overhead below is the wrapper census less the stock core's, i.e. the AXI4-Lite control block, the AXI4 DMA engine, the token sequencer, the SPI master, the AER bridge, the program RAM and the capture RAM, at the SHRUNK RAM depths named in `at_parameters` -- NOT at the depths the wrapper ships with:

| Column | Wrapper overhead (delta vs the stock core) |
| --- | ---: |
| `lut_equivalent` | 20,269 |
| `flip_flops` | 33,883 |
| `carry` | 127 |
| `lutram` | 0 |
| `bram36` | 4 |
| `bram18` | 0 |
| `uram` | 0 |
| `lut_sites` (the bound's LUT class) | 20,269 |

The wrapper's `prog_ram` DOES infer block RAM -- the +4 RAMB36E2 above
is its 131,072 bits at the synthesized depth -- because every one of its
reads lands in a register on the same clock, which is what a tile's
synchronous read port can be. The capture RAM does not, and that is the
next paragraph.

The two wrapper points differ ONLY in `CAP_WORDS` (1,024 extra words), which makes the capture
RAM's cost a measurement:

- `flip_flops`: +32.00 per capture word
- `lut_equivalent`: +19.07 per capture word
- `bram36`: +0.00 per capture word

**This is the study's sharpest finding.** The capture RAM does not infer
block RAM -- it has two write ports at fixed addresses plus the streaming
write -- so it costs 32 flip-flops per 32-bit word. The
wrapper SHIPS with `CAP_WORDS = 65,536`, which
is 2,097,152 flip-flops: 80% of
the entire device's registers, for the capture buffer alone -- and its
read multiplexing is another 1,249,984 LUT sites, 96% of
the LUTs. The kernel as written therefore cannot be built at its
shipped capture depth, and
the numbers below are for the SHRUNK depths named in the table. Giving
`cap_ram` a single registered read port and a single write port -- the
same treatment `hw/fpga/mem/` gave the stock memories -- is the fix, and
it is work the BOARD half of P8 owes.

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
| datasheet_total | 226 | 589 | 201 | n/a | **bram36** -> **201** |
| assumed_shell_30pct | 157 | 410 | 140 | n/a | **bram36** -> **140** |

### The generated variant `gen_a128n128_mb8_per_event`

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 363 | 2,450 | n/a | n/a | **lut_sites** -> **363** |
| assumed_shell_30pct | 254 | 1,715 | n/a | n/a | **lut_sites** -> **254** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 357 | 2,418 | n/a | n/a | **lut_sites** -> **357** |
| assumed_shell_30pct | 248 | 1,683 | n/a | n/a | **lut_sites** -> **248** |

### The generated variant `gen_a512n256_mb16_per_event`

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 63 | 626 | 4,032 | n/a | **lut_sites** -> **63** |
| assumed_shell_30pct | 44 | 438 | 2,822 | n/a | **lut_sites** -> **44** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 62 | 618 | 4,024 | n/a | **lut_sites** -> **62** |
| assumed_shell_30pct | 43 | 430 | 2,814 | n/a | **lut_sites** -> **43** |

### The generated variant `gen_a256n256_mb16s_sync_fire`

Without any wrapper reserved:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 95 | 626 | 4,032 | n/a | **lut_sites** -> **95** |
| assumed_shell_30pct | 66 | 438 | 2,822 | n/a | **lut_sites** -> **66** |

With the measured wrapper overhead reserved once:

| Scenario | N_max(lut_sites) | N_max(flip_flops) | N_max(bram36) | N_max(uram) | Binds |
| --- | ---: | ---: | ---: | ---: | --- |
| datasheet_total | 93 | 618 | 4,024 | n/a | **lut_sites** -> **93** |
| assumed_shell_30pct | 65 | 430 | 2,814 | n/a | **lut_sites** -> **65** |

## The verdict

- For the STOCK core the binding resource is **block RAM**: 10 RAMB36E2 tiles per core against the device's 2,016.
- For every GENERATED variant the binding resource is **LUT sites**,
  because the synapse memory never reaches a block-RAM tile and pays for
  itself in SLICEM LUTs instead. That is a property of the emitted RTL,
  not of the geometry, and it is fixable.
- Which classes bind, across every configuration and scenario in the
  record: `bram36` for 1 of 4 core configurations, `lut_sites` for 3 of 4 core configurations.
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
- **The two RTL defects this study found**, both of which change the
  numbers above: the capture RAM that costs flip-flops per word, and the
  generated core's asynchronous synapse read that keeps every variant out
  of block RAM.

## The caveat this study must carry

yosys is not Vivado. Its Xilinx mapping is generic: it emits `CARRY4`
where UltraScale+ has `CARRY8`, leaves `INV` unpacked, does not model LUT
pairing, and chooses distributed RAM by its own heuristics rather than
Vivado's. Every LUT and FF number here is an ORDER-OF-MAGNITUDE budget,
the BRAM counts are the most trustworthy column because tile geometry is
discrete, and nothing here says anything about timing closure, placement,
routing across the three SLRs, or the HBM AXI infrastructure. The bounds
are ceilings on a ceiling.
