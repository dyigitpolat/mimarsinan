# TrueNorth — platform physics profile

IBM TrueNorth, 28 nm Samsung LPP, 4096 neurosynaptic cores of 256 axons × 256 neurons.
Every constant in `truenorth.json` is either quoted from one of the two primary papers
or derived from them with the arithmetic shown in the profile's `derivation` field.
**No constant in this profile is an estimate.** The full research pass, with verbatim
quotes and page references for every number, is
[`docs/research/physics/truenorth_constants_research.md`](../../../../../docs/research/physics/truenorth_constants_research.md).

## Sources

- `merolla2014a` — Merolla et al., "A million spiking-neuron integrated circuit with a
  scalable communication network and interface", *Science* 345(6197), 2014, **including
  Supplementary Material S1–S13**, where most per-unit physics lives.
- `akopyan2015truenorth` — Akopyan et al., "TrueNorth: Design and Tool Flow of a 65 mW
  1 Million Neuron Programmable Neurosynaptic Chip", *IEEE TCAD* 34(10), 2015.

Both were read in full. Every energy, power and area number below is **measured on
silicon**, which is why this profile is the reference one: it does not rest on
pre-silicon simulation.

## Validity domain

| Quantity | Value |
|---|---|
| Technology | 28 nm Samsung LPP CMOS |
| Supply for `e_mac` and `p_static_per_core` | **0.8 V** — the corner of the Fig. 17 sweep |
| Operating range | 0.70 – 1.05 V, total power 42 – 323 mW |
| Array size assumed | 256 axons × 256 neurons per core, 4096 cores, 64 × 64 mesh |
| Temperature | **not stated by either paper** |

Both priced energy constants come from the *same* measurement at the *same* voltage,
which is the point: they are the slope and the intercept of one published sweep.

## Why 26 pJ per synaptic event is NOT this profile's constant

TrueNorth's famous number — **26 pJ/synaptic-event** — is the measured 72 mW whole-chip
power divided by the events that produced it at 20 Hz and 128 synapses/neuron. It is an
**operating-point average**, so it already contains the static power, and using it as a
per-event constant makes the model a *point* model: exact where it was measured, wrong
everywhere else. Against TrueNorth's own published sweep it gives

| firing rate | 26 pJ/event predicts | published | error |
|---|---|---|---|
| 11.58 Hz | 40.4 mW | 68 mW | **−40.6 %** |
| 20.07 Hz | 70.0 mW | 71 mW | −1.4 % (its own calibration point) |
| 95.93 Hz | 334.8 mW | 94 mW | **+256.1 %** |

So the profile declares the **marginal** energy instead — `e_mac = 2.297 pJ` — with
static power beside it as `p_static_per_core`, both obtained by regressing the published
0.8 V sweep. The held-out middle point then lands at −0.54 %. The arithmetic is in each
constant's `derivation` field and the full working is in
[`docs/research/physics/silicon_correlation_research.md`](../../../../../docs/research/physics/silicon_correlation_research.md).

`e_synaptic_event_total` remains in the vocabulary — it is the right model for a target
that publishes *only* a whole-chip average — but no shipped profile prices from it, and
`deployment_record/correlation/` is the suite that keeps it that way.

**`e_inter_tile_hop` (2.3 pJ/hop) is deliberately not declared here** for the same
reason: `e_mac` is a whole-chip marginal fitted on total power, so it already contains
the spike routing. Declaring the hop energy as well would charge routing twice. The
published figure is recorded in the research note rather than in the profile.

## Notes on individual constants

- **`area_per_core_total` (93 600 µm²)** is the published, measured 240 × 390 µm
  complete-core footprint — crossbar SRAM, neuron block, scheduler, token controller
  and router together. It sits in the `aggregate` group and **supersedes** the
  decomposed area constants: chip area prices as
  `cores × area_per_core_total + area_global_fixed`, and the per-cell/per-neuron
  figures below are kept for decomposition studies, never summed with it.
- **`area_per_cell` (0.152 µm²)** is the published 6T SRAM bitcell. One TrueNorth synapse
  is exactly one bit of that array, so the identification is direct — but it is the
  *storage cell alone* and excludes decoders, sense amps, drivers and the redundant
  rows/columns. For whole-chip area, the superseding `area_per_core_total` is the
  number to trust; `area_global_fixed` is derived from it.
- **`area_per_state_bit` (0.15 µm²)** comes from a completely different sentence than
  `area_per_cell` — "storing neuron state (20 bits) requires an additional area of
  3.0 µm² per neuron" — and lands within 1.3 % of the 6T bitcell. Two independent
  published area facts agreeing is the strongest internal consistency check in this
  profile.
- **`t_cycle`** is banded one-sided on purpose: 1 ms is the chip's defining real-time
  tick, and the low corner (47.6 µs) is the fastest *demonstrated* operation at 21×
  real-time, which the authors state depends on activity, synaptic density and voltage.
  Nominal must stay at 1 ms; a faster tick is a measured possibility, never a guarantee.
- **`t_program_per_byte` (800 ns)** carries a stated assumption: a single serial scan
  path at one bit per 10 MHz cycle. If the chip shifts several chains in parallel, the
  true figure is lower by that factor. The papers do not state the chain count, so the
  assumption is written into the constant's `note` rather than hidden in the number.
- **Zeros are facts, not gaps.** `area_per_adc`, `e_adc_conversion`, `t_adc_conversion`,
  `write_sigma` and `read_sigma` are declared **0.0** because TrueNorth is fully digital
  — there is no converter and no analog programming spread. That is different from an
  undeclared constant, which means "this target has not said", disables the objectives
  that need it, and may never be silently defaulted.
- **`p_static_global` is 0.0** for the same discipline: the entire published 42 mW
  zero-activity corner is already attributed per core, so charging it again chip-wide
  would double count one measurement.

## The programming group — the scan chain, timed and now priced (TS6)

Akopyan Section V-H *times* the configuration path (that is `t_program_per_byte` =
800 ns/B, a published 10 MHz scan chain at one bit per cycle) but never measures its
**energy**. The two energy constants are therefore authored derivations, each stating
its anchor and ratio, each `evidence_kind: derived`, and each with its nominal at the
band's log-space centre (`sqrt(low x high)`), because the centre of a multiplicative
spread is geometric:

- **`e_program_per_byte` = 9.186 / 22.502 / 55.118 pJ/B.** A TrueNorth synapse is
  **binary**, so one synaptic event consults one *bit* of the core's crossbar SRAM and
  a byte of weight store is 8 accesses: the per-byte anchor is `8 x e_mac` =
  18.373 pJ/B. An SRAM write costs ~1–3× its read, so the band is `[0.5x, 3x]` the
  anchor. `e_mac` is the whole-chip *marginal* per-event energy, so it overstates the
  bare SRAM access — which is what the 0.5× low corner is for.
- **`e_core_program` = 293.97 / 587.93 / 1175.86 pJ.** No `e_core_init` exists here, so
  the payload-independent part of a program load is anchored on the **core-state sweep**
  this chip's geometry defines: 256 neurons × `e_mac` = 587.93 pJ, the same
  compartment-sweep bound Loihi's *measured* reset came in ~1.7× below. Band
  `[0.5x, 2x]`.

## What this profile deliberately does not declare

`t_array_read`, `e_neuron_update`, `e_leak_per_neuron_step`,
`e_intra_tile_packet`, `t_hop`, `area_per_router`, `area_per_tile_fixed`,
`e_row_drive`, `area_per_row_driver`, `e_dma_per_byte`, `e_core_init`,
`t_core_init`, `e_sync_barrier` and `t_sync_barrier` are **absent because they are
unpublished**, not because they are zero. Neither paper prints a per-component power or
area breakdown — both show a labelled core floorplan image with no block areas — and hop
latency is architecturally non-deterministic in an asynchronous design. Absence disables
exactly the objectives that need those constants, which is the intended behaviour.

`p_host` and `host_compute_rate` are absent for a different reason: they are facts about
the **deployment host machine**, not about the chip. A chip profile cannot know them, so
they are expected to arrive through `platform_physics_overrides`. Until they do, absolute
energy and end-to-end latency cannot price the host side of the NeuralOps/ComputeOps
boundary — which matters, because unpriced host work makes moving a layer host-side look
free.

## Cross-check against what the repository already claims

The research pass compared this profile against the TrueNorth numbers already in the
tree. Geometry agrees exactly — 256 neurons, 256 axons, 4096 cores, 64 × 64 mesh, one
core per tile — and the `truenorth_like` provenance quote in
`mapping/platform/imc_platforms_literature.py` is a verbatim match to Merolla p. 3.

The **energy numbers do not agree**, and that is a finding rather than a discrepancy to
reconcile here:

- `chip_simulation/sanafe/presets.py::TRUENORTH_PRESET` is commented "Merolla 2014", but
  none of its 16 constants appears in that paper. Its `tile_hop_energy_j = 5.0e-14` is
  **46× below** the published 2.3 pJ/hop, and its `synapse_energy_j = 2.0e-13` reproduces
  0.51 mW against a measured 72 mW.
- The vendored `sana_fe/arch/truenorth.yaml` has every energy and latency set to `0.0`.

Those are simulator inputs, and changing them changes SANA-FE results, so they are
reported rather than edited by this profile. It is also why this profile declares its
published numbers **directly** instead of referencing the presets through `source_ref`:
a reference would inherit the preset's provenance, and the preset does not have any.
