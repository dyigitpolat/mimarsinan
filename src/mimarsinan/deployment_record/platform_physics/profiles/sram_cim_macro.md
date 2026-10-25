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
| `e_write_per_cell` | 76 - 342 fJ, nominal 161.2 fJ | derived | constants.md sec. 5: 1x the 20 MHz read byte energy to 2x the 91 MHz one (a write is full swing) |
| `write_verify_iterations` | 1 | derived | a digital write has no verify loop |
| `endurance_cycles` | 1e16 | published | ielmini2018inmemory p.7 |
| `conductance_levels` | 256 | published | 8-bit weights (p.2-3) |

## Absent, refused by name

Every chip-level row: periphery, NoC, host, static power, area, and the cycle time of a whole evaluation (the
macro is bit-serial over its 4-bit inputs at 20 MHz, and the cycles per evaluation are not stated). The
**backing store** is not part of the macro: the profile names it as an external component (`external_components`:
`e_dma_per_byte`, `t_program_per_byte`), and `price_absolute` refuses every absolute term by name
(`absolute_pricing`) until a run declares both, which CL5 and CL7 do as a named `backing_store` override block
(`merge_override_blocks`), so this point is never read without it. houshmand2023benchmarking (Table II and
Fig. 7, p.6) carries an analytical weight-write term that was not read in this round; it is the cross-check the
derived write bracket waits on.
