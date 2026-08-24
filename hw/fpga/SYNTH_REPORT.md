# ODIN stock core -- local synthesizability proof and per-core resources

Plan `docs/odin_firing_semantics_and_rtl_export_plan.md` §7 row 19 (P5.5a).
This is the LOCAL half of the synthesis/implementation gate: the vendored
`hw/vendor/odin` core plus the `hw/fpga/mem/` BRAM overlay through yosys
`synth_xilinx -family xcup` -- the UltraScale+ family the
U55C's XCU55C part belongs to -- with zero errors, both memories in block
RAM, and the per-core resource census below.

## What produced these numbers

- Tool: `Yosys 0.68+118 (git sha1 144c707b7-dirty, Release, Clang /usr/bin/clang++ 21.1.8)`
- Top module: `ODIN`  |  target family: `xcup`
- Sources (20 files, overlay first so the first declaration of each memory wrapper wins):

```
  hw/fpga/mem/SRAM_256x128_wrapper.v
  hw/fpga/mem/SRAM_8192x32_wrapper.v
  hw/vendor/odin/src/IZH_neuron_blocks/izh_calcium.v
  hw/vendor/odin/src/IZH_neuron_blocks/izh_effective_threshold.v
  hw/vendor/odin/src/IZH_neuron_blocks/izh_input_accumulator.v
  hw/vendor/odin/src/IZH_neuron_blocks/izh_neuron_state.v
  hw/vendor/odin/src/IZH_neuron_blocks/izh_stimulation_strength.v
  hw/vendor/odin/src/LIF_neuron_blocks/lif_calcium.v
  hw/vendor/odin/src/LIF_neuron_blocks/lif_neuron_state.v
  hw/vendor/odin/src/ODIN.v
  hw/vendor/odin/src/aer_out.v
  hw/vendor/odin/src/controller.v
  hw/vendor/odin/src/fifo.v
  hw/vendor/odin/src/izh_neuron.v
  hw/vendor/odin/src/lif_neuron.v
  hw/vendor/odin/src/neuron_core.v
  hw/vendor/odin/src/scheduler.v
  hw/vendor/odin/src/sdsp_update.v
  hw/vendor/odin/src/spi_slave.v
  hw/vendor/odin/src/synaptic_core.v
```

- Exact yosys script (run from the repo root):

```tcl
read_verilog -nooverwrite hw/fpga/mem/SRAM_256x128_wrapper.v hw/fpga/mem/SRAM_8192x32_wrapper.v hw/vendor/odin/src/IZH_neuron_blocks/izh_calcium.v hw/vendor/odin/src/IZH_neuron_blocks/izh_effective_threshold.v hw/vendor/odin/src/IZH_neuron_blocks/izh_input_accumulator.v hw/vendor/odin/src/IZH_neuron_blocks/izh_neuron_state.v hw/vendor/odin/src/IZH_neuron_blocks/izh_stimulation_strength.v hw/vendor/odin/src/LIF_neuron_blocks/lif_calcium.v hw/vendor/odin/src/LIF_neuron_blocks/lif_neuron_state.v hw/vendor/odin/src/ODIN.v hw/vendor/odin/src/aer_out.v hw/vendor/odin/src/controller.v hw/vendor/odin/src/fifo.v hw/vendor/odin/src/izh_neuron.v hw/vendor/odin/src/lif_neuron.v hw/vendor/odin/src/neuron_core.v hw/vendor/odin/src/scheduler.v hw/vendor/odin/src/sdsp_update.v hw/vendor/odin/src/spi_slave.v hw/vendor/odin/src/synaptic_core.v
synth_xilinx -family xcup -top ODIN
stat -top ODIN -json
```

- Regenerate DELIBERATELY with `scripts/hw_tests/regen_synth_report.py`; the [slow] gate
  `tests/integration/test_odin_rtl_synth.py` re-runs the synthesis and
  requires an exact match against `synth_resources.json`.

## Per-core resources (one ODIN core, 256 neurons x 256 axons)

| Resource | Count |
| --- | ---: |
| LUTs (`LUT1`..`LUT6`) | 5,209 |
| Inverters (`INV`, a LUT1 once packed) | 450 |
| **LUT equivalent** | **5,659** |
| Flip-flops (`FD*`) | 4,362 |
| Carry cells (`CARRY*`) | 381 |
| Wide muxes (`MUXF*`) | 12 |
| `RAMB36` | 10 |
| `RAMB18` | 0 |
| **BRAM tiles (36 kb equivalent)** | **10** |
| `URAM` | 0 |
| Distributed-RAM/SRL cells | 0 |
| I/O and clock buffers | 36 |
| Unclassified cells | 0 |
| Total primitive cells | 10,460 |

## BRAM inference outcome

| Memory | Geometry | Bits | Cells | Outcome |
| --- | --- | ---: | --- | --- |
| `SRAM_256x128_wrapper` | 256x128 | 32,768 | RAMB36E2 x2 | **block RAM** |
| `SRAM_8192x32_wrapper` | 8192x32 | 262,144 | RAMB36E2 x8 | **block RAM** |

## Control: the vendored memories alone (plan finding F17)

The same script over the vendored tree **alone** does not complete -- yosys reports `ERROR: invalid OPTION_ABITS/WIDTH combination.` after mapping the behavioural arrays toward distributed RAM, whose width/depth combination it then refuses. That is the local confirmation of plan finding F17 (the stock RTL is not FPGA-synthesizable as vendored) and the reason `hw/fpga/mem/` exists.

## The caveat this report must carry

yosys-synthesizability is **necessary, not sufficient** for Vivado closure
on the U55C shell. This run proves the RTL elaborates, maps to UltraScale+
primitives, and puts both memories in block RAM; it proves **nothing** about
timing closure, placement, routing, the Alveo shell's own resource budget,
or the XRT kernel wrapper. yosys's Xilinx mapping is also generic rather
than exact -- it emits `CARRY4` where UltraScale+ has `CARRY8`, leaves `INV`
unpacked, and does not model LUT pairing -- so the LUT/FF columns are an
ORDER-OF-MAGNITUDE per-core budget for P8's compile-limits study, not Vivado
utilization. Vivado exists only on HACC (owner-gated login), so the
implementation half of §7 row 19 is closed by **P7**'s HACC build, where the
same design is run through Vitis 2022.2 for the U55C shell and the real
utilization report replaces these estimates.
