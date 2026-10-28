# NeuRRAM (130 nm RRAM) — page-sourced profile

`wan2022a` (Nature 2022) and `wan2021edge` (arXiv full text): 48 cores of 256 x 256 1T1R RRAM, a weight a
differential pair of cells, selector and peripherals in a 130 nm foundry process (wan2021edge p.2-3, wan2022a
p.10); voltage-mode neuron ADC with 4-bit output.

## Validity

48 cores of 256 x 256 cells of 4-bit weights (the repository's `neurram_like` geometry); no aggregate
constant is declared. The MVM time and energy hold at one operating point of the Nature paper's Extended Data
Table 1: 4-bit inputs, 4-bit weights, 6-bit outputs (the `neurram` bundle's input width).

## Rows

| constant | value | evidence | page |
|---|---|---|---|
| `t_cycle` | 3.9 us per 256 x 256 MVM | published | wan2022a Extended Data Table 1, PDF p.28: "Latency of a 256x256 MVM (us) 1.4 / 1.6 / 3.9 / 10.7" at 1 / 2 / 4 / 8 input bits; the printed energy-delay product 32.0e-15 J s closes on 8.192 nJ x 3.9 us |
| `e_mac` | 125 fJ | published | the same row: "TOPS/W 43 / 40 / 16 / 7"; 16 TOPS/W at 4-bit inputs, 2 / 16e12 J per MAC |
| `write_verify_iterations` | 8.52 | published | wan2022a Methods, PDF p.10: "On average each cell requires 8.52 set/reset pulses." |
| `t_write_per_unit` | 2 - 11 us per cell iteration, nominal 6.57 us | derived | a 1 us pulse plus a 1-10 us verify read; 56 us per cell over 8.52 pulses (wan2022a Methods, PDF p.10) |
| `endurance_cycles` | 1e10 - 1e11 (nominal 3.16e10) | published, device-class proxy | lee2011a p.627, Fig. 2c, a Pt/Ta2O5-x/TaO2-x/Pt bilayer cell (NeuRRAM's TiN/HfOx/TaOx/TiN cell is not measured in either NeuRRAM paper) |
| `pulses_per_write` | 8.52 | published | wan2022a Methods, PDF p.10: each set or reset pulse counted as one endurance cycle (an upper reading of the wear) |
| `conductance_levels` | 16 | published | 4-bit weights |

The write path is the bundle's: one cell at a time ("In the weight-programming mode, individual RRAM cells are
selected for read and write.", wan2022a Methods, PDF p.10), so the `neurram` bundle's cores declare
`write_granularity` cell and `write_units_in_parallel` 1.

## Absent, refused by name

Write energy (not stated; the fetch intake `constants_write_energy.md` sec. 2.2 derives a cell-level band of 12.3 pJ
to 1.32 nJ per cell write under named assumptions, owner decision B3: not declared until a primary value is on
disk), static power, periphery and NoC rows, area and the host group.
