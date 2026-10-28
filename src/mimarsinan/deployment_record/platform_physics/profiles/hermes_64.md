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
| `t_write_setup_per_event` | 1.536 ms per core (1.6384 ms with the 50 ns trailing edge) | derived | published p.689-690: every unit cell's four devices RESET (125 ns) and device 1 SET (250 ns), 32 source lines in parallel: 8,192 + 2,048 programming cycles |
| `t_write_per_unit` | 0.4526 / 0.6492 / 1.0424 ms per core iteration | derived | published p.690: one 125 ns pulse per unit cell (32 in parallel) and one verify round of the 256 diagonals at 1, 2 or 4 reads of 512 + 256 ns |
| `write_verify_iterations` | 1 - 30, nominal 5.48 | published | published p.690: "the maximum number of iterations (30)"; the mean is not stated |
| `endurance_cycles` | 1e8 - 1e12 (nominal 1e10) | published, review | gallo2020an p.4: "> 10^8 SET/RESET cycles" (a storage requirement) to "> 10^12" (a best device); not measured on HERMES-64's doped Ge2Sb2Te5 cell |
| `pulses_per_write` | 3 / 7.48 / 32 | derived | published p.690: RESET, SET and one programming pulse per iteration on the most-written device (2 + write_verify_iterations) |

The write rows cite the published version's pages (*Nature Electronics* 6, 680-693); the MVM and area rows keep
the arXiv full text's, which the fetch intake (`12_tcad_readiness/fetch_intake/constants_write_energy.md` sec. 4)
maps to the published ones. One write step is one core (the `hermes_64` bundle's `write_granularity` core): the
setup once per rewritten core, then the per-iteration step times the iterations. The earlier 2.45 ms per core
iteration was no lower bound and repeated the RESET and SET in every iteration (same file, sec. 3.4).

## Absent, refused by name

Write energy (the pulse voltage is not stated), static power, every periphery and NoC row, the host group,
programming DMA and commit energies. The write energy refuses whenever a program rewrites cells; the
lifetime is priced from the endurance band, worn in programming pulses.
