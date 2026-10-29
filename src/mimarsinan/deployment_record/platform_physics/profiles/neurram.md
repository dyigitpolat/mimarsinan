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
| `e_write_per_cell` | 5.33 / 58.9 / 650.8 pJ per pulse into one RRAM device (log-space centre) | derived | low: the 432 fF word line (wan2022a Methods, PDF p.14) at the 3.0 V set gate bias plus the first set pulse at gmin (1.2 V, 1 uS, 1 us; PDF p.10); high: the 4.0 V amplitude ceiling across 40 uS for 1 us plus the word line at the 5.0 V reset gate bias (the code release, `python/wupdate_control.py:79, :90, :371-413`, notebooks' `end_voltage=4.0`, `vset_wl` 3.0 / 3.3) |
| `e_verify_read_per_cell` | 0.04 / 0.8 / 16 pJ per verify read (log-space centre) | derived | 0.2 V across the cell (the code's `vread=1.1, vref=0.9`, `python/adc_control.py:142`) at 1 - 40 uS for 1 - 10 us (wan2022a Methods, PDF p.10) |
| `endurance_cycles` | 1e10 - 1e11 (nominal 3.16e10) | published, device-class proxy | lee2011a p.627, Fig. 2c, a Pt/Ta2O5-x/TaO2-x/Pt bilayer cell (NeuRRAM's TiN/HfOx/TaOx/TiN cell is not measured in either NeuRRAM paper) |
| `pulses_per_write` | 8.52 | published | wan2022a Methods, PDF p.10: each set or reset pulse counted as one endurance cycle (an upper reading of the wear) |
| `conductance_levels` | 16 | published | 4-bit weights |

The write path is the bundle's: one cell at a time ("In the weight-programming mode, individual RRAM cells are
selected for read and write.", wan2022a Methods, PDF p.10), so the `neurram` bundle's cores declare
`write_granularity` cell and `write_units_in_parallel` 1.

The write energy is derived (owner decision B3, from the primaries the owner fetched; fetch intake
`constants_write_energy_b3.md` sec. 2.3): no write current or write energy is printed in the chip paper, its refs. 41
(`zhao2017investigation`) and 52 (`wu2018a`) or its code release (Zenodo 10.5281/zenodo.6558399). The band rests on
four named readings (the full pulse amplitude across the cell, an ohmic cell, one word-line swing per pulse,
inter-wire capacitance excluded); its high corner is a reading, not a bound. One device write prices
(`e_write_per_cell` + `e_verify_read_per_cell`) x 8.52 = 45.7 / 508.5 / 5,681 pJ, replacing the earlier cell-level
12.3 pJ to 1.32 nJ. The rows are per RRAM device; a signed weight is a differential pair of devices
(`python/nn_layers.py:33-37`; "two RRAM cells on adjacent rows along the same column", wan2022a PDF p.10), which the
`neurram` bundle declares (U-C4: `cells_per_weight` 2, `weight_cell_rows` 2): a core holds 128 x 256 weights of the geometry's
256 x 256 devices, and a written weight is priced for both devices (91.5 / 1,017 / 11,362 pJ per signed weight, two
write steps), each device worn by every write of its weight. Paper and code
differ in two places: the reset start (1.5 V in the paper; 1.2 V in range or 2.5 V for a zero target in the code) and
the time-out (30 set-reset polarity reversals in the paper; 50, 30 or 20 iterations in the code). The paper governs
`write_verify_iterations`.

## Absent, refused by name

Static power, periphery and NoC rows, area and the host group. A published write energy is absent too: the declared
band is derived, and the missing quantity is the selector's write current at 3.0 V (set) and 5.0 V (reset) gate bias,
or a measured programming energy.
