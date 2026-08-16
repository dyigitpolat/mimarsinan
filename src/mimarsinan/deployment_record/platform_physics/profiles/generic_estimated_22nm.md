# Generic 22 nm digital IMC — ESTIMATED baseline profile

**This describes no real chip.** Every one of its 29 constants is an
`estimated` value, and the profile exists for exactly two reasons:

1. so a first-cut co-optimization study can run before a vendor profile is
   available, and
2. so the `estimated` evidence path is exercised end to end by something no reader
   can mistake for a measurement.

Its `measurement_kind` is `projection` and its display name says **ALL ESTIMATED**,
so the wizard badges it in the warning colour and any cross-platform comparison
against `truenorth` (silicon) or `loihi` (simulation) discloses the difference by
construction.

## How the estimates were made

Every constant is anchored on something published rather than invented from nothing:

- **Digital SRAM cell and state-bit areas** scale TrueNorth's *measured* 0.152 µm²
  6T bitcell from 28 nm by area ∝ node², giving ≈0.094 µm² at 22 nm.
- **`e_mac`** scales Horowitz's (ISSCC 2014) ~0.2 pJ 8-bit MAC at 45 nm / 0.9 V by
  ≈(22/45)·(0.8/0.9)², giving 0.077 pJ.
- **Transport, barrier and leakage constants** bracket the two published digital
  points this repository already holds — TrueNorth at 28 nm and Loihi at 14 nm — so
  the band is a real interval between measured chips rather than a guess around one.
- **`e_dma_per_byte`, `e_readout_per_byte` and `bytes_per_connectivity_entry`**
  restate the framework's own long-standing coefficient bands, so this profile and
  the default-band cost model agree on the coefficients they share. The two transfer
  directions carry the same band here because the underlying figure (Horowitz 2014
  memory-access energy) is stated per access rather than per direction — a target
  whose readout path genuinely differs should override it.

Bands here are **wide on purpose**. Where the two published anchors disagree by 80×
(leakage per core) the band says 80×; where the mechanism is a design choice rather
than a physical limit (`t_cycle`, `t_program_per_byte`) the band spans orders of
magnitude and the note says so. A narrow band on an estimate would be the dishonest
part.

## The two constants to override first

- **`t_cycle`** — it alone converts `latency_steps` into seconds, so it gates
  end-to-end latency and throughput entirely. Its nominal is a design assumption
  (1 µs), not a measurement of anything.
- **`p_static_per_core`** — the widest genuinely physical band in the profile,
  because leakage depends far more on process flavour and power gating than on node.

## Every constant, and its one-line basis

| constant | nominal | basis (see the profile JSON for the full rationale) |
|---|---|---|
| `area_per_cell` | 0.09 um^2 | A 6T SRAM bitcell holding one weight bit. |
| `area_per_cell_per_weight_bit` | 0.09 um^2 | Digital SRAM storage scales linearly with precision — one bitcell per weight bit — so this equals area_per_cell by construction. |
| `e_mac` | 0.08 pJ | An 8-bit multiply-accumulate. |
| `e_neuron_update` | 2.0 pJ | One membrane update: read state, accumulate, compare, write back. |
| `e_leak_per_neuron_step` | 0.2 pJ | A leak decrement is a fraction of a full update: one read-modify-write of the same register without the synaptic accumulation, taken as ~10% of e_neuron_update. |
| `e_intra_tile_packet` | 0.4 pJ | Delivering one spike packet inside a tile. |
| `e_inter_tile_hop` | 2.0 pJ | One packet traversal of one inter-tile link. |
| `t_hop` | 4.0 ns | Latency of one inter-tile link traversal, bracketing the published digital points (TrueNorth 4 ns, Loihi 4. |
| `t_array_read` | 3.0 ns | One 256-row array read. |
| `t_cycle` | 1000.0 ns | The timestep period. |
| `area_per_neuron_logic` | 9.0 um^2 | Per-neuron update logic, excluding state storage. |
| `area_per_state_bit` | 0.09 um^2 | One bit of membrane state in SRAM — the same bitcell as area_per_cell, so it carries the same anchor, band and expected error. |
| `membrane_bits` | 24.0 bit | A 24-bit signed accumulator, matching Loihi's independently characterized width and comfortably above TrueNorth's 20 bits. |
| `area_per_router` | 4000.0 um^2 | One 5-port mesh router. |
| `area_per_tile_fixed` | 8000.0 um^2 | Per-tile overhead that is neither array, neuron logic nor router: schedulers, controllers, local buffers. |
| `area_global_fixed` | 10.0 mm^2 | Whole-chip area scaling with neither cores nor tiles: pads, PLLs, host interface. |
| `e_dma_per_byte` | 0.16 pJ | Moving one byte of programming payload onto the chip. |
| `e_readout_per_byte` | 0.16 pJ | Moving one byte of pass-boundary emissions off the chip. Same off-chip band as the inbound direction, restated so the readout is charged explicitly rather than by omission. |
| `bytes_per_connectivity_entry` | 8.0 B | Wire size of one connectivity (axon source span) entry. |
| `t_program_per_byte` | 80.0 ns | Wall time to move one programming byte. |
| `e_sync_barrier` | 1000.0 pJ | Energy of one chip-wide synchronization barrier. |
| `t_sync_barrier` | 465.0 ns | Wall time of one chip-wide barrier. |
| `p_static_per_core` | 50.0 uW | Static (leakage) power per core. |
| `p_static_global` | 5.0 mW | Whole-chip leakage outside the cores: I/O ring, PLLs, host interface. |
| `conductance_levels` | 256.0 levels | 8-bit digital weights, the precision this profile's array assumes. |
| `write_sigma` | 0.0 fraction | Zero by the profile's own assumption that the array is DIGITAL SRAM, which has no analog programming spread. |
| `read_sigma` | 0.0 fraction | Zero for the same reason as write_sigma: a digital read has no sensing spread. |
| `area_per_adc` | 0.0 um^2 | Zero because this profile assumes a digital array with no analog-to-digital conversion in the datapath. |
| `e_adc_conversion` | 0.0 pJ | Zero for the same digital-array assumption as area_per_adc. |
| `t_adc_conversion` | 0.0 ns | Zero for the same digital-array assumption as area_per_adc. |

## What this profile does not declare

The whole `host` group (`p_host`, `host_compute_rate`, `host_macs_per_s`): those are
facts about the *deployment host machine*, not about a chip, so no chip profile can
state them. Until they are supplied as overrides, energy and end-to-end latency
cannot price the host side of the NeuralOps/ComputeOps boundary — which matters,
because unpriced host work makes moving a layer host-side look free.
