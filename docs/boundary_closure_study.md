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

## Results

*(pending — runs in flight)*

## Evidence disclosure

Every absolute number in this study is priced from declared constants whose
evidence kind travels with it: `truenorth` rows rest on silicon measurements,
`loihi` rows on pre-silicon simulation plus one measured wall,
`isaac_like` on published simulation, `generic_estimated_22nm` on labelled
estimates, and the host share of every row on the two operator-declared
estimated rates above. A comparison mixing these bases says so in the
artifact.
