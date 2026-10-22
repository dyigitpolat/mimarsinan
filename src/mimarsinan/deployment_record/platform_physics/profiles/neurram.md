# NeuRRAM (130 nm RRAM) — page-sourced profile

`wan2021edge` (arXiv full text) and `wan2022a` (Nature 2022): 48 cores of 256 x 256 1T1R RRAM, a weight a
differential pair of cells, selector and peripherals in a 130 nm foundry process (wan2021edge p.2-3, wan2022a
p.10); voltage-mode neuron ADC with 4-bit output.

## Validity

48 cores of 256 x 256 cells of 4-bit weights (the repository's `neurram_like` geometry); no aggregate
constant is declared.

## Rows

| constant | value | evidence | page |
|---|---|---|---|
| `t_cycle` | 2.1 us per 256 x 256 MVM | published | wan2021edge p.17 |
| `e_mac` | 125 fJ | published, SECONDARY | 16 TOPS/W, gallo2022a Extended Data Table II p.25; NeuRRAM states its efficiency in figures only |
| `write_verify_iterations` | 8.52 | published | wan2021edge p.9: 99% of cells converge, 8.52 set/reset pulses on average |
| `t_write_per_unit` | 2 - 11 us per cell iteration, nominal 6.57 us | derived | a 1 us pulse plus a 1-10 us verify read (p.9); 56 us per cell over 8.52 pulses (wan2022a p.10) |
| `endurance_cycles` | 1e6 - 1e9 | published, SECONDARY | sridharan2023xformer p.1 until lee2011a is fetched |
| `conductance_levels` | 16 | published | 4-bit weights |

## Absent, refused by name

Write energy (not stated; constants.md sec. 6 bounds the cell-level energy at 767 pJ per cell as an ESTIMATE,
which is not declared), static power, periphery and NoC rows, area and the host group.
