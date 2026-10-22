# Digital SRAM-CIM macro (16 nm) — page-sourced, macro-level profile

`jiang2023a` — the 16 nm digital SRAM compute-in-memory macro (ESSCIRC 2023, full text via
`RA/06_parts/constants.md` sec. 4-5): 128 kB of 6T in 1 kB blocks of 64 x 128 b, 8-bit weights x 4-bit
inputs with 32-bit partial sums and exact digital accumulation (p.2-3); 12.6 TOPS/W non-weight-stationary and
23.8 TOPS/W weight-stationary at 20 MHz, 0.58 V (p.4); local-block read energy 76 to 171 fJ per byte (p.4).

## Validity

Macro-level constants only. The structured validity geometry is one 1 kB block read as 64 axons x 16 neurons of
8-bit weights, up to 128 blocks; no aggregate constant is declared, so the geometry bounds nothing priced.

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
**backing store** is not part of the macro: CL5 and CL7 declare it as a named `backing_store` override block
(`merge_override_blocks`), so this point is never read without it. houshmand2023benchmarking (Table II and
Fig. 7, p.6) carries an analytical weight-write term that was not read in this round; it is the cross-check the
derived write bracket waits on.
