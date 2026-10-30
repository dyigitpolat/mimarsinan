# ISAAC-CE — platform physics profile

The ISAAC design point of Shafiee et al. (ISCA 2016): 32 nm, analog ReRAM crossbars,
8 × 128×128 arrays of 2-bit cells per IMA, 12 IMAs per tile, 168 tiles. The full
research pass, with verbatim quotes and page references for every number, is
[`docs/research/physics/isaac_constants_research.md`](../../../../../docs/research/physics/isaac_constants_research.md).

This is the **first analog profile**, and the reason the conversion model exists.

## Two things to know before using it

**1. It is a design-space study, not a chip.** `measurement_kind` is `simulation`:
areas and powers come from CACTI 6.5 at 32 nm plus published component models, and no
ISAAC was fabricated. The supply voltage is never stated anywhere in the paper.

**2. Table I publishes POWER, never energy.** Every energy constant here is therefore
`derived`, by multiplying a published power by the paper's own 100 ns cycle and
dividing by the printed instance count. The derivations are shown in full in each
constant's `derivation` field, because a reader must be able to redo them.

## The conversion model — why this profile needed one

An analog accelerator's cost is dominated by data conversion, but no deployment
record counts conversions: the count is a property of the *target's dataflow*. ISAAC
states its dataflow precisely enough to derive one.

- Weights are bit-sliced across **columns**: *"we represent one 16-bit synaptic
  weight with 16/w w-bit cells located in the same row"* → 8 cells per weight at
  w = 2. The `isaac_like` bundle declares the same count for its cores (U-C4:
  `cells_per_weight` 8, `weight_cell_rows` 1), so each 128 × 128 array holds 128 × 16
  weights and a weight write writes its 8 cells; a core type stating another count
  beside this model refuses by name.
- Inputs are bit-sliced across **time**: *"we provide 16 voltage levels sequentially
  … all 16 bits of the input have been handled in 16 cycles"*.
- **Every column is converted every cycle**: *"128 bitline currents are latched in
  128 sample-and-hold circuits. In the next 100 ns cycle, these analog values … are
  fed sequentially to a single 1.28 GSps ADC unit"* — so `adc_sharing_factor = 128`,
  one ADC per array.

Which gives

```
adc_conversions = ceil(macs × cells_per_weight / array_rows) × (input_bits / dac_bits)
                = macs × 8 × 16 / 128
                = macs × 1.0
```

**Exactly one 8-bit conversion per 16-bit MAC.** That result corrected a real defect
in this repository: the conversion model's first formula omitted the column-slicing
factor and would have under-counted conversions — the dominant analog cost — by 8×.
The regression is now pinned by `TestTheIsaacReferenceCase`.

## Notes on individual constants

- **`e_mac` is per CELL-MAC** (one 1-bit input against one 2-bit cell), which is what
  the model's event census counts. A full 16×16 MAC costs 128 of them — 0.234 pJ in
  the array alone, before conversion and periphery.
- **`area_per_cell` carries one significant figure.** Table I prints 0.0002 mm² for
  eight arrays; the five-figure result of dividing it is arithmetic, not precision.
  The number does pass a physical sanity check: the implied 39.1 nm cell pitch is
  4F² at F = 19.5 nm.
- **`area_per_router` is a quarter router per tile.** One router serves four tiles
  ("(shared by 4 tiles)" in Table I: ISAAC's c-mesh is *concentrated*), so [U-A] a tile
  carries 0.151 / 4 mm², as Table I's tile total counts it.
- **HyperTransport is static power, not transport energy.** Dividing its 10.4 W by
  its bandwidth gives a tempting 406 pJ/B, but the paper says the cost is constant:
  *"The HT is a constant overhead of 10 W … but only 16 % of ISAAC chip power."*
  Modelling it per byte would make it vanish on a small workload and explode on a
  large one, so it is declared as `p_static_global` instead.
- **`t_cycle` and `t_array_read` are the same quantity** in ISAAC — the pipeline
  stage *is* the crossbar read — and both carry the paper's own 100–200 ns
  sensitivity band.

## Chip area from components (U-A)

[U-A, owner instruction of 2026-10-28] Table I's area is reproduced from this profile's components, one ISAAC tile
of 12 IMAs x 8 arrays (the bundle's `cores_per_tile` 96, so a NoC tile is an ISAAC tile):

| part | per | Table I | this profile |
|---|---|---|---|
| memristor arrays | cell | 0.0002 mm2 per IMA | `area_per_cell` 0.0015259 um2 x 16,384 cells x 96 arrays = 0.0024 mm2 |
| DAC and IR | row | 0.00017 + 0.0021 mm2 per IMA | `area_per_row_driver` 2.2168 um2 x 128 rows x 96 = 0.0272 mm2 |
| ADC | array | 0.0096 mm2 per IMA | `area_per_adc` 1,200 um2 x 96 = 0.1152 mm2 |
| S+H, S+A, OR | column | 0.00004 + 0.00024 + 0.00077 mm2 per IMA | `area_per_neuron_logic` 1.0254 um2 x 128 columns x 96 = 0.0126 mm2 |
| router | tile | 0.151 mm2 shared by 4 tiles | `area_per_router` 37,750 um2 |
| eDRAM, bus, sigmoid, S+A, maxpool, OR | tile | 0.17725 mm2 | `area_per_tile_fixed` 177,250 um2 |
| tile | | 0.372 mm2 | 0.37244 mm2 |
| 168 tiles | | 62.5 mm2 | 62.57 mm2 |
| 4 HyperTransport links | chip | 22.88 mm2 | `chip_components`: 4 x `hypertransport_link_isaac` (5.72 mm2 each) |
| chip | | 85.4 mm2 | 85.45 mm2 |

The links are a server multi-chip interconnect; an edge chip carries none of them.

## The programming group — writing an analog array, and what is NOT modelled (TS6)

ISAAC is an *inference* architecture: it assumes its weights are already resident, so
the paper publishes no write energy, no write time and no endurance budget. The three
programming constants are authored derivations (`evidence_kind: derived`), each with
its nominal at the band's log-space centre (`sqrt(low x high)`):

- **`e_program_per_byte` = 237.7 / 751.7 / 2377.0 pJ/B.** An analog conductance is
  committed by a **program-and-verify pulse sequence**, one to two orders of magnitude
  above an SRAM write. The library's SRAM class (Loihi 28.90, TrueNorth 22.50, ODIN
  20.65 pJ/B) has geometric mean 23.77 pJ/B, so the band is `[10x, 100x]` that. Per
  2-bit cell (4 cells to the byte) that is 59–594 pJ, the program-verify scale.
  **`e_dma_per_byte` (2.02 pJ/B) is a different thing entirely** — it prices moving a
  byte through the tile's eDRAM buffer, which is also what a carried activation pays
  per inference. Committing and moving are separate constants precisely so an NVM
  target's write cost cannot leak into its per-inference carry. The profile declares
  it a chip-internal buffer row (`transfer_rows`, U-J5): every byte entering the
  tile's eDRAM buffer pays it, a host-sourced payload over the host-chip link
  included, since the link-crossing rule (U-J4) drops only a backing-store row.
- **ENDURANCE IS UNMODELED.** This profile prices the *energy* of a write and says
  nothing about the *wear*: a per-inference-reprogrammed array has a finite write
  budget (10⁶–10⁹ cycles in the device literature) and no quantity in this framework
  counts writes against it. A DOM-style workload that reprograms per sample is priced
  here but **not** feasibility-checked. Treat that as an open modelling gap, not as a
  clean bill of health.
- **`t_program_per_byte` = 5.82 / 326.2 / 1454.5 ns/B.** Donor-scaled along
  `validity.technology_node_nm` from TrueNorth (800 → 914.3 ns/B at 32 nm) and the
  generic 22 nm exemplar (4/80/1000 → 5.82/116.4/1454.5). **Transport only, therefore
  a lower bound here**: the per-cell program-verify wall (microsecond scale in the
  device literature) is not in this figure.
- **`e_core_program` = 1.035 / 2.07 / 4.14 nJ.** ISAAC's "core" is a *tile*, so the
  neuron-sweep anchor the digital profiles use does not apply; the anchor is the tile's
  own eDRAM buffer cycle (20.7 mW × 100 ns = 2.07 nJ — the very quantity
  `e_dma_per_byte` divides), banded `[0.5x, 2x]`. Control work only; the conductance
  writes are `e_program_per_byte`'s.

## What this profile deliberately does not declare

`e_inter_tile_hop` and `t_hop` are the notable absences. Table I gives router
**power** (42 mW) with no stated utilisation, and router **latency** is never
mentioned. Deriving a per-hop energy would require inventing an activity factor the
paper does not supply, so the constant is left absent and the objectives that need it
say so. The research report records the arithmetic that *would* give 4.4–5.3 pJ/flit
if one assumed full utilisation — as an explicitly unusable estimate.

Also absent: the whole `host` group (a property of the deployment host, not the
chip), `e_core_init` / `t_core_init` (ISAAC assumes weights are already resident and
publishes no reset; the *programming* group is now derived — see above), and
`e_sync_barrier` / `t_sync_barrier` (the c-mesh is *statically scheduled*, so there
is no barrier to price).

## Cross-check against what the repository already claims

`mapping/platform/imc_platforms_literature.py::isaac_like` transcribes the geometry
with a verbatim provenance quote — *"The optimal design point has 8 128×128 arrays, 8
ADCs per IMA, and 12 IMAs per tile"* — and it **agrees** with the paper. Its
`claim_eligibility` is already `curve-only`, which is the right classification for a
simulated design point, and this profile's `simulation` measurement kind says the
same thing in the physics layer.

## Wave 1 rows: validity, the write law and the value NoC (G3)

- **Validity geometry.** 128 x 128 arrays of 2-bit cells; no aggregate constant is declared.
- **`endurance_cycles` = 1e10 - 1e11** (published, a device-class proxy): the resistive band of lee2011a p.627
  (Fig. 2c, a TaOx bilayer cell), a same-technology transfer as the write rows below; it replaces the shared NVM band
  of sridharan2023xformer p.1 (SECONDARY). With a declared batch K the write law prices the lifetime against it,
  and a declared lifetime floor judges admissibility.
- **`pulses_per_write` = 8.52** (published, wan2022a Methods, PDF p.10): the wear of one write in endurance cycles,
  NeuRRAM's mean set/reset pulses per cell write, transferred as `write_verify_iterations` is.
- **`write_verify_iterations` = 8.52** (published, wan2021edge p.9) and **`t_write_per_unit` = 2 - 11 us**
  (derived from the 1 us pulse and the 1-10 us verify read): a **same-technology transfer from NeuRRAM**. ISAAC
  publishes no write; its cells are RRAM, so NeuRRAM's write-verify statistics stand in, labelled as such.
- **`e_noc_value_per_byte_hop` = 1.09 - 1.31 pJ** (derived, Table I router row): the e_dma_per_byte pattern
  (power over bytes moved per unit time): 42 mW over 8 ports of 32-bit flits at the paper's conservative 1 GHz
  link (1.2 GHz for the low end). It assumes the router draws its published power with every port busy; it is
  a value-domain byte-hop, which the per-flit spiking hop above deliberately leaves undeclared.
