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
| `e_mac` | 0.08 pJ | An 8-bit multiply-accumulate. |
| `e_neuron_update` | 2.0 pJ | One membrane update: read state, accumulate, compare, write back. |
| `e_leak_per_neuron_step` | 0.2 pJ | A leak decrement is a fraction of a full update: one read-modify-write of the same register without the synaptic accumulation, taken as ~10% of e_neuron_update. |
| `e_intra_tile_packet` | 0.4 pJ | Delivering one spike packet inside a tile. |
| `e_inter_tile_hop` | 2.0 pJ | One packet traversal of one inter-tile link. |
| `t_hop` | 4.0 ns | Latency of one inter-tile link traversal, bracketing the published digital points (TrueNorth 4 ns, Loihi 4. |
| `t_array_read` | 3.0 ns | One 256-row array read. |
| `t_cycle` | 1000.0 ns | The timestep period. |
| `membrane_bits` | 24.0 bit | A 24-bit signed accumulator, matching Loihi's independently characterized width and comfortably above TrueNorth's 20 bits. |
| `e_dma_per_byte` | 0.16 pJ | Moving one byte of programming payload onto the chip. |
| `e_program_per_byte` | 0.547723 pJ | Committing one byte into the SRAM weight store, once it has arrived: Horowitz-scale SRAM access (~1.25 pJ/B read at 45 nm, ~0.61 pJ/B scaled to 22 nm) times a 1–3x write ratio, banded 0.1–3.0 pJ/B. |
| `e_core_program` | 512.0 pJ | The payload-independent per-core setup of a program load: the 256-neuron core-state sweep (256 x e_neuron_update) banded [0.5x, 2x] on the anchor's own corners. |
| `e_readout_per_byte` | 0.16 pJ | Moving one byte of pass-boundary emissions off the chip. Same off-chip band as the inbound direction, restated so the readout is charged explicitly rather than by omission. |
| `bytes_per_connectivity_entry` | 8.0 B | Wire size of one connectivity (axon source span) entry. |
| `e_local_mem_read_per_byte` | 0.61 pJ | One byte read from a core-local activation SRAM of 8 to 32 KB (a core's input buffer and the registers that stage a row's inputs into its array), at the Horowitz scale `e_program_per_byte` reads: ~1.25 pJ/B at 45 nm, 0.61 pJ/B at 22 nm, banded 0.15 to 2.5 pJ/B (a wide access to the unscaled 32 KB figure). ESTIMATED: no source states a core-local memory of this projection; expected error a factor of ~4. |
| `e_local_mem_write_per_byte` | 1.056551 pJ | One byte written into the same core-local SRAM (a core's output buffer): the read band times this profile's own stated write-to-read ratio of 1 to 3, corner by corner (0.15 to 7.5 pJ/B, the nominal at the ratio's log-space centre, 0.61 x √3). ESTIMATED from the read band and that ratio, never from a comparison; expected error a factor of ~5. |
| `t_program_per_byte` | 80.0 ns | Wall time to move one programming byte. This profile is a DONOR: the targets that publish no interface rate (Loihi, ODIN, ISAAC) scale this band by `validity.technology_node_nm` to derive their own. |
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

The whole `host` group (`p_host`, `host_macs_per_s`, `t_host_op_overhead`, the link rows; `host_compute_rate` withdrawn [U-HOST]: no profile declares a host row; a host row is admitted only from an identified host description (hosts/<name>.json) as the run's host block, and a measured host wall prices nothing): those are
facts about the *deployment host machine*, not about a chip, so no chip profile can
state them. Until they are supplied as overrides, energy and end-to-end latency
cannot price the host side of the NeuralOps/ComputeOps boundary — which matters,
because unpriced host work makes moving a layer host-side look free.

## Wave 1 rows: validity, the write law and the value NoC (G3)

- **Validity geometry.** 256 x 256 cores of 8-bit cells. The profile declares no aggregate constant, so the
  geometry bounds nothing priced; it is stated so a target bundle can be checked against it.
- **`e_dma_per_byte` and `e_readout_per_byte` corrected to 31 / 160 / 320 pJ.** Both restate the framework DMA
  band (`chip_simulation/weight_reuse_cost_model.py`, `DEFAULT_COEFFICIENT_BAND`: 31e-9 / 160e-9 / 320e-9 mJ
  per byte). The earlier 0.03 / 0.16 / 0.32 pJ wrote those nJ values under the unit pJ, 1000x low; a test now
  ties the profile to the band so the unit cannot drift again.
- **`endurance_cycles` = 1e16** (published, ielmini2018inmemory p.7): embedded SRAM over its expected lifetime.
- **`e_write_per_cell` = 76 - 342 fJ** (derived, jiang2023a p.4 via constants.md sec. 5): one 8-bit cell is a
  byte, bracketed by 1x the measured read byte energy and 2x the 91 MHz one; nominal at the log-space centre.
- **`write_verify_iterations` = 1** (derived): an SRAM write has no verify loop.
- **`pulses_per_write` = 1** (derived): one SRAM write is one of the write cycles ielmini2018inmemory counts.
- **`e_noc_value_per_byte_hop` = 0.5 / 1.0 / 4.0 pJ** (estimated): one value byte over one inter-tile link,
  Horowitz-scale on-chip wire energy (about 0.1 - 0.2 pJ per bit per mm, a 1 mm link) plus the router, banded on
  this profile's own `e_inter_tile_hop` envelope because a value byte and a spike flit cross the same link.

## Chip Area by the Area Rule (`area_rule`)

[U-AREA] `chip_area_mm2` of this description is priced from its `area_rule` block (the area rule from published silicon per technology type, `platform_physics/area_rule`; design revision 3). Where the rows or notes above describe another chip-area path (a per-core footprint, `area_global_fixed`, or a refusal outside the footprint's validity), the block supersedes them for chip area; the other constants keep their uses.

- **Type and terms.** T2_digital (digital in-memory compute); a type rule: the type table's terms, with no anchor of its own.
- **Outside part (`outside: type`).** the type table's core-grid share (gamma, flagged); an area_global_fixed beside it is refused by name ('the outside part is stated twice').
- **Node.** 22 nm (S-SRAM-22 (the push-rule 6T at 22 nm)).
- **Reading as the low edge.** the T1 reading with 24 bits per neuron (R2: the digital spiking reading is the band's low edge).
- **Dropped.** area_per_cell, area_per_cell_per_weight_bit, area_per_neuron_logic, area_per_state_bit, area_per_router, area_per_tile_fixed, area_global_fixed (estimated constants).

Every figure is cited by its evidence id (`area_rule/data/evidence/published_figures.json`); none is typed in this description.
