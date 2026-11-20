# Digital SRAM-CIM macro (16 nm) — page-sourced, macro-level profile

`jiang2023a` — the 16 nm digital SRAM compute-in-memory macro (ESSCIRC 2023, full text via
`RA/06_parts/constants.md` sec. 4-5): 128 kB of 6T in 1 kB blocks of 64 x 128 b, 8-bit weights x 4-bit
inputs with 32-bit partial sums and exact digital accumulation (p.2-3); 12.6 TOPS/W non-weight-stationary and
23.8 TOPS/W weight-stationary at 20 MHz, 0.58 V (p.4); local-block read energy 76 to 171 fJ per byte (p.4).

## Validity

Macro-level constants only. The structured validity geometry is the macro's logical core (CL2 reading of p.2-3):
the 128 blocks of 64 rows x 128 b form 16 SR groups of 8 blocks; each group computes one dot product of the 128
4-bit inputs per cycle over one address (one row in each of its 8 blocks, 16 weights each) and holds 64 addresses,
so the macro is one core of 128 axons x 1024 neurons of 8-bit weights (1024 x 128 x 1 B = 128 kB); a fan-in above
128 takes several addresses with a full-precision partial sum (p.3). One macro was measured, so the core count is
1. No aggregate constant is declared, so the geometry bounds nothing priced. (Wave 1 declared one 1 kB block as 64
axons x 16 neurons, up to 128 blocks; a block's 64 rows are addresses read one per cycle against 16 inputs, not 64
inputs, so CL2 corrected it to the group-level core the `sram_cim_macro` target bundle declares.)

## Rows

| constant | value | evidence | basis |
|---|---|---|---|
| `e_mac` | 158.7 fJ, band from 84.0 fJ | derived | 2 / 12.6e12 J per MAC (non-WS, the CIM activation's anchor); 2 / 23.8e12 J weight-stationary |
| `t_cycle` | 703.3 ns - 6.4 us, nominal 6.4 us per core evaluation | derived (DECLARED) | 16 outputs per cycle (p.2) over the core's 1024: 64 cycles at 4-bit inputs, 128 at the 8-bit MAC of Fig. 8c (p.4); clocks 91 MHz to 20 MHz (p.3-4) |
| `e_write_per_cell` | 76 - 342 fJ, nominal 161.2 fJ | derived | constants.md sec. 5: 1x the 20 MHz read byte energy to 2x the 91 MHz one (a write is full swing) |
| `write_verify_iterations` | 1 | derived | a digital write has no verify loop |
| `endurance_cycles` | 1e16 | published | ielmini2018inmemory p.7 |
| `pulses_per_write` | 1 | derived | one SRAM write is one of the write cycles ielmini2018inmemory counts |
| `conductance_levels` | 256 | published | 8-bit weights (p.2-3) |

## Absent, refused by name

Every chip-level row: periphery, NoC, host, static power and area. The
**backing store** is not part of the macro: the profile names it as an external component (`external_components`:
`e_dma_per_byte`, `t_program_per_byte`), and `price_absolute` refuses every absolute term by name
(`absolute_pricing`) until a run declares both, which CL5 and CL7 do as a named `backing_store` override block
(`merge_override_blocks`), so this point is never read without it. houshmand2023benchmarking (Table II and
Fig. 7, p.6) carries an analytical weight-write term that was not read in this round; it is the cross-check the
derived write bracket waits on.

## Cycle time of one evaluation [consts]

One step of the value window is one evaluation of the logical core. The paper states the outputs per cycle and the
clocks, so the cycles per evaluation follow (DECLARED from jiang2023a, the arithmetic shown; PDF pages, printed
pages 409-412):

- "In total 16 32b outputs (SUM[31:0]) are computed each clock cycle for the macro." (p.2) The core's 1024 outputs
  take 1024 / 16 = 64 cycles at the 4-bit input ("The input activation (IA) driver delivers 128 (4b) IA to all 16 SR
  groups", p.2).
- Fig. 8c (p.4) lists the macro at "Precision(I/W/O) 8/8/32" with the footnote "An 8bW*8bA MAC is defined as two
  ops", and its "TP(TOPs) 0.182" closes on 1024 8-bit MACs per cycle (16 x 128 4-bit products) x 2 ops x 91 MHz =
  0.186 TOPS: an 8-bit input takes two 4-bit cycles, 128 per evaluation.
- "The design is functional down to 20MHz at 0.58V VDDH and 91MHz at 1V VDDH." (p.3-4); "clock period time 0.05us
  (20MHz)" (p.4).

The band runs from 64 cycles at 91 MHz (703.3 ns) to 128 cycles at 20 MHz (6.4 us); the nominal is the high end, the
operating point of the energy rows (8-bit MACs at 20 MHz). A partly used core reads fewer addresses (ceil(n / 16) for
n neurons), so the constant is an upper bound per step there; pipeline fill and the activation-buffer load are not
stated and not counted. The 4 bits of an input are multiplied in one cycle (four gates per bit line, the LSB and MSB
2-bit halves accumulated by two adder trees in parallel, p.2), so an 8-bit input is two such cycles.

## Chip Area by the Area Rule (`area_rule`)

[U-AREA] `chip_area_mm2` of this description is priced from its `area_rule` block (the area rule from published silicon per technology type, `platform_physics/area_rule`; design revision 3). Where the rows or notes above describe another chip-area path (a per-core footprint, `area_global_fixed`, or a refusal outside the footprint's validity), the block supersedes them for chip area; the other constants keep their uses.

- **Type and terms.** T2_digital (digital in-memory compute); an anchor: its own published core scaled by the census's cells, converters or bits (rho), its size and word width flagged.
- **Outside part (`outside: type`).** the type table's core-grid share (gamma, flagged); an area_global_fixed beside it is refused by name ('the outside part is stated twice').
- **Node.** 16 nm.
- **Anchor (macro).** core_mm2 `v('T2-JI-02')`; n_pub `s('T2-JI-02', 'scope', 'whole 128 kB macro', 1)`; its own share: published_split `r('types/t2_digital:s')`.
- **Organization.** 128 x 1024, 8 weight bits per cell, 8 stored bits per cell, 0 bits per neuron; term weights {'bits': 1.0}. Citations: rows, cols: CO profiles/sram_cim_macro.json validity.geometry (128 x 1024); stored_bits_per_cell: T2-JI-01 quote (128KB over 128 x 1024 cells); weight_bits_per_cell: CO profiles/sram_cim_macro.json validity.geometry.cell_bits.
- **Cell kind.** `cim_logic_rule_6t_digital`.

Every figure is cited by its evidence id (`area_rule/data/evidence/published_figures.json`); none is typed in this description.
