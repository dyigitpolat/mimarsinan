# Boundary-Closure Study — one MLP, four targets, absolute objectives end to end

The first full product of the co-optimization substrate after the
boundary-closure program (N/P/B/L, see `docs/boundary_closure_plan.md`):
a hardware search driven by ABSOLUTE chip-designer objectives — including the
candidate-time NoC census the N-stage added — whose winner deploys to a sealed
record, with `fidelity.json` closing the modeled-vs-measured loop per axis.

## Vehicle and protocol

- **Model**: `simple_mlp` (256/128) on MNIST, LIF synchronized, Tq=4, S=4,
  weight quantization on — the `t0_60` search cell, unchanged.
- **Search**: hardware-only NSGA-II (pop 12, 6 generations, seed 0) over core
  geometry (1 core type; axons ∈ [792, 1024], neurons ∈ [64, 512],
  count ∈ [8, 32]).
- **Objectives** (all five active per candidate, refusals would abort):
  `param_utilization_pct`, `chip_area_mm2`, `energy_per_inference_mj`,
  `e2e_latency_s`, `noc_total_hops`.
- **Declarations**: `platform_physics_profile` per run;
  `activity_factor = 0.05` (the declared switching assumption every
  spike-dependent modeled quantity rests on — undeclared would refuse the
  energy and NoC axes BY NAME at resolution); operator host rates
  (`host_macs_per_s` 10 G/s, `p_host` 20 W, both `estimated` and disclosed)
  because no shipped profile publishes host pricing and the MLP keeps
  host-side work under `subsume`.
- **Targets**: `truenorth` (silicon-derived constants), `loihi` (pre-silicon
  constants + the [L] measured t_cycle band), `isaac_like` (simulation,
  analog, conversion-model priced), `generic_estimated_22nm` (every constant
  estimated, named so nobody mistakes it for a chip).
- **Deployment**: each run's winner deploys through the full phased pipeline
  (train → tune → map → SANA-FE/nevresim verification) to a sealed
  `DeploymentRecord`; `fidelity.json` re-evaluates the candidate view of the
  deployed config and zips it axis-by-axis against the measurement —
  including `noc_total_hops` (wireload model vs sealed link loads).

## Results — the four sealed MLP runs (2026-08-16)

**Different physics chose different chips.** All four searches ran the same
vehicle, the same budget, the same seed — and the per-profile winners differ
by ~3× in declared cell capacity, because each profile's cost surface ranks
the same geometries differently:

| target | basis | winner capacity (cells) | deployed acc | area mm² (priced, record) | e2e s (priced, record) | mJ/sample (measured, SANA-FE) | NoC packets (all intra-tile) |
|---|---|---|---|---|---|---|---|
| truenorth | silicon | 604,160 | 0.981 | 47.35 | 0.0221 | 0.000995 | 667 |
| loihi | pre-silicon sim + measured wall | 1,271,808 | 0.981 | 3.75 | 0.00761 | 0.000971 | 449 |
| isaac_like | simulation (analog) | 1,769,472 | 0.981 | 23.34 | 0.01028 | 0.000964 | 388 |
| generic_estimated_22nm | estimated | 679,680 | 0.981 | 10.39 | 0.00871 | 0.000998 | 667 |

**`fidelity.json` per run** (candidate view of the deployed config vs the
sealed measurement — the loop the program built):

- `total_param_capacity`: **byte-equal on all four** (604,160 / 1,271,808 /
  1,769,472 / 679,680) — the candidate resolves exactly the chip deployment
  built.
- `chip_area_mm2`: **exact** on truenorth and loihi, in-band on generic
  (10.411 vs 10.387), out-of-band by 1.4% on isaac_like (23.67 vs 23.34 —
  the candidate context's conversion-model census vs the record's).
- `noc_total_hops`: **exact (0 = 0)** on loihi and isaac_like; **112 modeled
  vs 0 measured** on truenorth and generic — the layout twin's packer placed
  the winner's cores across a tile boundary the real packer avoided. The
  wireload model is conservative where it diverges, exact where placements
  coincide; this axis is precisely what the fidelity report exists to watch.
- `e2e_latency_s`: modeled µs-scale vs measured ms-scale on every run — the
  measured StageTimer host walls dominate, and the divergence **convicts the
  operator's host declaration**: 10 G/s prices the host ops as if the MACs
  were the cost, while the lab host's per-op dispatch overhead is ~1000×
  larger. On truenorth the chip-side term alone (1 ms tick × 4 steps ≈
  4.02 ms) is a fair fraction of the measured 22.1 ms; the rest is host.
  Fidelity caught a bad operator estimate on its first outing — exactly the
  discipline working.
- `energy_per_inference_mj`: candidate-modeled 0.40–0.61 mJ at the flat
  declared activity 0.05 vs measured 0.96–1.00 mJ/sample — a factor ~2.4,
  reported as a prediction/measurement pair because the RECORD's own priced
  energy honestly refuses (no sealed synaptic-EVENT census yet; the record
  measures spikes, not per-synapse events — a future quantity).

**Cross-platform pricing artifact** (the truenorth winner's sealed census
priced by all four bare shipped profiles;
`scripts/cross_platform_study.py`): area 47.35 / 3.75 / 23.26 / 10.35 mm²
for truenorth / loihi / isaac_like / generic. Bare profiles carry no
operator host declarations, so the host-inclusive axes refuse by name in
this artifact (loihi alone prices e2e via its measured-wall identity) — the
refusal rows and the mixed-basis note are part of the artifact, not
footnotes to it.

**Search-time NoC behaved as an objective**: minimizing `noc_total_hops`
steered every winner to geometries whose deployed traffic is entirely
intra-tile (0 measured inter-tile hops on all four) — the axis the owner
asked for ("NoC quantities are very important during search") demonstrably
shaped the outcome.

## Stretch — LeNet5, pruned, offload (truenorth)

A single-profile repetition on the `t0_02` LeNet5 cell (pruning_fraction
0.5, offload placement) exercising the P-stage boundary live: conv models
expose no perceptron chain pre-conversion, so the candidate keeps unpruned
shapes as a STATED upper bound and fidelity measures the elimination gap.
*(run in flight at the time of writing; artifacts land in
`generated/study_truenorth_lenet5_pruned_phased_deployment_run/`)*

## Evidence disclosure

Every absolute number in this study is priced from declared constants whose
evidence kind travels with it: `truenorth` rows rest on silicon measurements,
`loihi` rows on pre-silicon simulation plus one measured wall,
`isaac_like` on published simulation, `generic_estimated_22nm` on labelled
estimates, and the host share of every row on the two operator-declared
estimated rates above. A comparison mixing these bases says so in the
artifact.
