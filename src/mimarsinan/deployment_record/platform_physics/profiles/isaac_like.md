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
  w = 2.
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
- **`area_per_router` is per ROUTER, and one router serves four tiles.** A floorplan
  charging one router per tile over-counts by 4× — ISAAC's c-mesh is *concentrated*,
  which is what the "(shared by 4 tiles)" in Table I means.
- **HyperTransport is static power, not transport energy.** Dividing its 10.4 W by
  its bandwidth gives a tempting 406 pJ/B, but the paper says the cost is constant:
  *"The HT is a constant overhead of 10 W … but only 16 % of ISAAC chip power."*
  Modelling it per byte would make it vanish on a small workload and explode on a
  large one, so it is declared as `p_static_global` instead.
- **`t_cycle` and `t_array_read` are the same quantity** in ISAAC — the pipeline
  stage *is* the crossbar read — and both carry the paper's own 100–200 ns
  sensitivity band.

## What this profile deliberately does not declare

`e_inter_tile_hop` and `t_hop` are the notable absences. Table I gives router
**power** (42 mW) with no stated utilisation, and router **latency** is never
mentioned. Deriving a per-hop energy would require inventing an activity factor the
paper does not supply, so the constant is left absent and the objectives that need it
say so. The research report records the arithmetic that *would* give 4.4–5.3 pJ/flit
if one assumed full utilisation — as an explicitly unusable estimate.

Also absent: the whole `host` group (a property of the deployment host, not the
chip), `e_core_program` / `e_core_init` / `t_core_init` / `t_program_per_byte` (ISAAC
assumes weights are already resident and does not model programming), and
`e_sync_barrier` / `t_sync_barrier` (the c-mesh is *statically scheduled*, so there
is no barrier to price).

## Cross-check against what the repository already claims

`mapping/platform/imc_platforms_literature.py::isaac_like` transcribes the geometry
with a verbatim provenance quote — *"The optimal design point has 8 128×128 arrays, 8
ADCs per IMA, and 12 IMAs per tile"* — and it **agrees** with the paper. Its
`claim_eligibility` is already `curve-only`, which is the right classification for a
simulated design point, and this profile's `simulation` measurement kind says the
same thing in the physics layer.
