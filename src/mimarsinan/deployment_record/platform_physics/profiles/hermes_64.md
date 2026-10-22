# IBM HERMES PCM, 64 cores (14 nm) — page-sourced profile

`gallo2022a` — Le Gallo et al., "A 64-core mixed-signal in-memory compute chip based on phase-change
memory for deep neural network inference", *Nature Electronics* 2023 (arXiv full text). Every row below is
located on a page; what the paper does not state is absent, and every objective that needs it refuses by name.

## Validity

64 cores of 256 x 256 PCM unit cells in 14 nm, chip 12 x 12 mm, core 1.2 x 1.16 mm (p.1-2); 8-bit PWM inputs
and 12-bit ADC outputs (p.9); effective weight precision close to 3 bits with one device and 3 to 4 bits with
two (p.5), so the structured validity geometry declares 4-bit cells: a program whose weights take more bits
lies outside the per-core footprint and its area refuses by name.

## Rows

| constant | value | evidence | page |
|---|---|---|---|
| `e_mac` | 204.9 fJ (one-phase), band to 806.5 fJ (four-phase) | published | Table I p.18: 9.76 and 2.48 TOPS/W, 1 MAC = 2 OPs; the MVM energy aggregates array, ADC and PWM (p.10) |
| `t_cycle` | 133 ns, band to 520 ns | published | Table I p.18: one MVM one-phase and four-phase |
| `area_per_core_total` | 1.392 mm^2 | published | core 1.2 x 1.16 mm (p.1-2) |
| `area_global_fixed` | 54.912 mm^2 | derived | 144 mm^2 chip less 64 cores (p.1-2) |
| `t_write_per_unit` | 2.45 ms per core iteration (lower bound) | derived | p.7: 32 source lines in parallel, 8 programming cycles per diagonal, RESET 125 ns, SET 250 ns, verify 512 + 256 ns |
| `write_verify_iterations` | 1 - 30, nominal 5.48 | published | p.7: "at most 30 iterations"; the mean is not stated |
| `endurance_cycles` | 1e6 - 1e9 | published, SECONDARY | sridharan2023xformer p.1 (NVM in general) until gallo2020an is fetched |

## Absent, refused by name

Write energy (the pulse voltage is not stated), static power, every periphery and NoC row, the host group,
programming DMA and commit energies. The write energy refuses whenever a program rewrites cells; the
lifetime is priced from the endurance band.
