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

## Stretch — LeNet5, pruned, offload (truenorth): sealed

A single-profile repetition on the `t0_02` LeNet5 cell (pruning_fraction
0.5, offload placement), same five-axis search. Sealed at deployed accuracy
0.9899, area 55.40 mm², 23.6 µJ/sample measured, 32,312 NoC packets (all
intra-tile), winner capacity 10,927,104 cells. `fidelity.json` was written
**live in-pipeline** on this run — the first end-to-end validation of the
wired emission.

Its first live output caught a twin defect: the rebuild resolved the raw
config's PRE-SEARCH declaration (31.9M cells) instead of the sealed winner
(10.9M). Fixed — the rebuild now takes the chip from the record's own
identity — after which the stretch fidelity reads:

- `total_param_capacity`: **byte-equal** (10,927,104 = 10,927,104).
- `chip_area_mm2`: **exact, in-band** (55.3984 = 55.3984).
- `param_utilization_pct`: 6.33% predicted vs 6.79% measured — with the
  platform twin exact, this pair now ISOLATES the P-stage pruning bound:
  the conv candidate keeps unpruned shapes (no perceptron chain
  pre-conversion, stated), the deployed compaction eliminates more, and the
  gap is 6.7% relative — the upper bound holding, tightly.
- `noc_total_hops`: 192 modeled vs 0 measured — the same conservative
  layout-twin placement divergence as the MLP runs.
- `e2e_latency_s`: 24 ms modeled vs 101 ms measured — host walls again
  (heavier under offload), same operator-declaration lesson.

## Can we optimize software, hardware, and both? — measured (2026-08-16)

The four study runs above are all `search_mode=hardware`. The other two modes
had never been run end-to-end, so they were run: same MNIST MLP family, same
truenorth physics, same declared activity, NSGA-II, all three sealed to real
deployments with measured energy.

| search mode | decision variables | deployed acc | area mm² | e2e s | µJ/sample (measured) | NoC packets | winner cells |
|---|---|---|---|---|---|---|---|
| hardware only | core geometry | 0.9810 | 47.35 | 0.0221 | 995.3 | 667 | 604,160 |
| software only | widths + activation | 0.9786 | 57.83 | 0.0270 | 792.9 | 270 | 31,948,800 (declared) |
| **joint** | **both** | 0.9778 | 48.10 | **0.0204** | **355.6** | **172** | 4,505,600 |

**Joint wins where neither single mode can**, and the mechanism is legible:

- **Hardware-only** moves the chip: best area, 53× smaller than the declared
  platform. But it cannot touch the MAC census, so **energy is FLAT** — the
  whole 72-candidate population spans 0.40195–0.40346 mJ (1.004×, no
  leverage). Measured energy ends up the WORST of the three.
- **Software-only** moves the MAC census: energy improves. But the chip is
  fixed, so **area is FLAT** (57.832 mm² across every candidate) — and area
  ends up the worst of the three.
- **Joint** moves both, and the energy axis that was degenerate under
  geometry search becomes live: **7.93× span** across its population. The
  deployed result is 2.8× less energy than hardware-only and 3.9× less NoC
  traffic, at 0.3 accuracy points.

So the joint space is not the union of the two single spaces — it unlocks an
axis neither can move alone. The per-axis leverage table (span of each
objective across the whole population) is the honest way to see it, and it is
worth running before trusting any search: an axis that is FLAT in your space
is not being optimized, whatever the report says.

The joint front also contains a real trade-off the substrate can now price:
**0.9928 proxy accuracy at 0.102 mJ vs 0.9950 at 0.804 mJ** — 8× the energy
for 0.2 accuracy points. (`estimated_accuracy` is the search-time training
proxy, not deployed accuracy.)

Caveats stated: single runs at small budgets (24 candidates for model/joint,
72 for hardware); the accuracy spread across the three modes (0.9778–0.9810)
is within the proxy's noise; the three searches optimized different objective
sets, so the comparison is of DEPLOYED outcomes, not of optimizer skill.

### The typed constraint did its job

Three joint candidates scored full penalties rather than crashing: the same
model config (LeakyReLU 128/64) puts only **15.26% of parameters on chip**,
below the declared 20% floor. C3's typed on-chip constraint made that a
region of the search space the optimizer can see, exactly as designed —
instead of a pipeline crash after the winner was already chosen.

### What running the option axes cost — and caught

`arch_search.option_axes` (placement, weight_bits, schedule policy) had never
been exercised by any run. The first one died at mapping:
`AssertionError: bias outside its ±15 register range`. Root cause was a
genuine split brain — `QuantizationVerificationStep` captured `q_max` in
`__init__`, at pipeline **assembly**, before the search stamps the winner's
`weight_bits=6`; the quantizer used 6 bits and the gate asserted against the
pre-search 5-bit grid. This is the pipeline-step form of the banned
constructor-shaped `get(key, default)`. Fixed (read where used) plus a
generic ratchet forbidding the shape across every step, with its one
sanctioned exception (contract-shaping reads that decide `requires`) closed
by refusing those keys as search axes.

## Follow-ups the study surfaced

1. A sealed synaptic-EVENT census (the record measures spikes, not
   per-synapse events), so the record-side priced energy stops refusing and
   the energy pair gains an in-band verdict.
2. The layout-twin vs real-packer placement divergence behind the
   112/192-vs-0 hop rows — either feed the twin the real packer's placement
   order or keep the conservative bound and say so per row (current state).
3. Operator host-rate declarations need a measured anchor; the ~1000×
   optimism of a MAC-rate guess against per-op dispatch walls is now a
   documented failure mode the wizard could warn about.

## Evidence disclosure

Every absolute number in this study is priced from declared constants whose
evidence kind travels with it: `truenorth` rows rest on silicon measurements,
`loihi` rows on pre-silicon simulation plus one measured wall,
`isaac_like` on published simulation, `generic_estimated_22nm` on labelled
estimates, and the host share of every row on the two operator-declared
estimated rates above. A comparison mixing these bases says so in the
artifact.
