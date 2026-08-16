# Boundary-Closure Program — candidate-time NoC, deterministic pruned shapes, buffer metric, latency evidence, first study

## Status (2026-08-16)

| Stage | State | Commits |
|---|---|---|
| N0 fragment wiring fix | **DONE** — live candidate views carry physics + quantity_context; `activity_factor`/`simulation_steps` ride the resolved platform | `search(N0)` |
| N1 geometry SSOT | **DONE** — `sanafe/noc_geometry.py`, four readers delegate, pinned | `noc(N1)` |
| N2 fragments | **DONE** — wire census (opt-in walk) + per-pass packer placements + `mapping/noc/` | `noc(N2)` |
| N3 estimator + axes | **DONE** — wireload model; `noc_total_hops` at both completenesses; activity run-gate mirrors the physics gate (registry + step + wizard chip filter) | `noc(N3)` |
| N4 fidelity + granularity | **DONE** — hops zip modeled-vs-measured; message granularity adjudicated on a live SANA-FE sim | `noc(N4)` |
| P pruned shapes + de-search | **DONE** — both knobs shrink the candidate (exact chain twin; mask floor-count bound with IO exemptions + max-propagation); pruning refused as an axis by name | `search(P)` |
| B buffer metric + gate | **DONE** — record axes + `pass_buffer_capacity_bytes` gate at the hard-core mapping step | `record(B)` |
| L latency evidence | **DONE** — loihi `t_cycle` measured band [5.8, 13] µs + `host_compute_rate` identity; `loihi_knn_query_latency` self-consistency case: e2e −22.8%, throughput +16.8% through the real pricer | `physics(L)` |
| E1 executed-window SSOT | **DONE** — `chip_simulation/stage_timesteps.py`; the runner and the candidate size the wall through ONE rule; retired `T x segments` (measured 4 where the record measured 15) | `search(E1)` |
| E2 core-init + reprogramming census | **DONE** — `resident_passes` is the one residency law; candidate `segment_cores`/`reprogrammed_cores`/`reprogrammed_bytes`/`reprogram_passes` match the sealed record on both policies; `cores_allocated` now means what the record means | `search(E2)` |
| E3 directional carry bytes | pending | |
| E4/E5 derived `e_core_init`; cross-pass wires as consumer input traffic | pending | |
| S study | **DONE** — 4 MLP runs + the LeNet5 pruned stretch sealed with live fidelity.json; results + surfaced follow-ups in `docs/boundary_closure_study.md` | `docs(S)` + fixes |

Repo: `mimarsinan` @ `2bb39fb3` (streamed-scheduling G-series closed). This program
closes the honest boundary the substrate discussion named, in the owner's priority
order, and ends with the first real co-optimization study.

## Directives (owner, 2026-08)

1. **N** — NoC quantities are very important during search; needed before anything else.
2. **P** — `pruning_fraction` must produce deterministic mappable shapes that affect the
   final program and the mapping metrics. Pruning is NOT searchable (accuracy impact
   unmodeled) — for now.
3. **B** — buffer capacity is a mapping performance metric (the required buffer for a
   particular program), not a search axis.
4. **L** — latency evidence quality must be improved.
5. **S** — once boundaries close, perform a real study. Thesis work comes later.
6. SSOT discipline is absolute: we cannot afford another SSOT violation.

## Owner decisions (asked 2026-08-16)

| Question | Decision |
|---|---|
| Loihi timestep wall | **Declare measured band** `t_cycle = [5.8, 13] µs` (Frady 2020 §5.1, Pohoiki Springs, workload-dependent, x86-IO-bound — caveats stated in evidence). Add latency reference case(s); same-source cases labeled self-consistency. |
| Buffer capacity | **Record objectives + optional declared-capacity gate**: expose `carried_raster_bytes` / `carry_peak_live_bytes` as record axes AND add an optional platform key; when declared, exceeding it is a typed refusal at deployment. Never searched. |
| Study scope | **MLP × 4 profiles now** (truenorth, loihi, isaac_like, generic_estimated_22nm), NSGA-II, deploy winners, fidelity + cross-platform artifact; **LeNet5 single-profile stretch** afterwards. |

## SSOT map (every formula has exactly one home; pins hold each one)

| Mechanism | SSOT home | Consumers | Pin |
|---|---|---|---|
| XY mesh routing walk | `chip_simulation/noc_geometry.py::xy_route_edges` (extracted; pure) | SANA-FE trace aggregation (`analysis/noc.py` delegates), candidate NoC estimator | identical edge lists on a probe set; delegation asserted |
| Core→tile assignment + tile→(x,y) | `chip_simulation/noc_geometry.py` (extracted from `arch_synth` sequential rule `core // cores_per_tile`, mesh xy) | arch synthesis, estimator | synthesized geometry == geometry helper on real floorplans |
| Floorplan resolution | `arch_synth/floorplan.py::resolve_floorplan` (existing) | arch synthesis, estimator | existing tests; estimator calls it, never re-derives |
| Message granularity | **adjudicated empirically 2026-08-16** on a live SANA-FE sim (2-tile probe): one message per (firing source neuron, destination core) — a neuron with 3 synapses onto one remote core emits ONE message per firing (the trace's `spikes: 3` field is carried synapses); on-somas emit one intra message per cycle; net synthesis fuses duplicate axons per source neuron, matching the census's distinct-cell rule | estimator (documented in `noc_estimate.py`) | synthetic equality pins vs `_aggregate_noc_link_load` / `_summarize_message_trace`; study-stage fidelity closes the numeric loop |
| Pruned shapes | `transformations/pruning/magnitude.py::prune_perceptron_chain` — the DEPLOYED shrink itself runs at candidate time (no twin formula) | soft-core mapping step (deploy), candidate layout resolution (search) | candidate softcore census == deployed softcore census at equal config; realized counts == floor rule |
| Activity assumption | existing `CandidateQuantityContext.activity_factor` (C2) | spikes AND NoC messages | one field, two formulas cross-checked in one test |
| Buffer bytes | existing pass-carry census (`schedule.carry`) | new record axes, deployment gate | axes read the census; gate reads the same census at emission |
| Loihi t_cycle | `platform_physics/profiles/loihi.json` constant (band) | pricer, correlation cases | correlation runner prices through the real pricer |

## Stages

### N — candidate-time NoC census (first)

The record convention (verified in `analysis/noc.py` + `build/from_simulators.py`):
one message → XY route walk (x first) → one count per traversed mesh edge;
`noc_total_hops = Σ link packet_count = Σ_messages manhattan(src_tile, dst_tile)`;
intra-tile messages contribute zero edges; per-segment inter/intra/input-path packet
counts from `analysis/diagnostics.py`.

0. **N0 fragment wiring fix (found during recon, verified empirically)** — the live
   search path never attaches `physics`/`quantity_context` to the candidate view:
   `resolve_active_specs(physics=declared)` ADMITS the absolute axes, then
   `spec.value(view)` raises for every candidate (`layout_hook._static_view` builds
   the view bare; C2's tests stop at spec resolution). Fix: the hook attaches
   `physics` from the candidate's own `pcfg["platform_physics_resolved"]` and a
   `quantity_context` built from pcfg (`simulation_steps`, `weight_bits`, resolved
   floorplan tiles, physical core/neuron/axon censuses) + the onchip/host census +
   the NEW declared `activity_factor` registry key (0 = undeclared → spike-dependent
   modeled quantities stay absent). Pin: a physics-declared problem evaluates
   absolute axes to finite numbers END-TO-END through the hook path.

1. **N1 geometry SSOT** — extract `xy_route_edges`, `tile_and_local_of_core`,
   `xy_of_tile`, and the arch mesh-dims rule (`spec.py`'s packed/explicit-grid
   branches, replicas included) into `chip_simulation/sanafe/noc_geometry.py`;
   `analysis/noc.py`, `net_synth/build.py`, `runner/core.py`, and `arch_synth/spec.py`
   delegate. (`chip_simulation/` top level is at 31 siblings — over budget, so the
   module lives in the sanafe package whose conventions it encodes.)
2. **N2 candidate fragments** — `pack_layout` exposes deterministic per-spec hardcore
   placements; `softcore_spec_adapter` grows a shape-only connectivity summary
   (per (producer core → consumer core) wire counts + input-path wires);
   `CandidateLayout` carries both.
3. **N3 estimator** — one function over (placements, connectivity, resolved floorplan,
   activity, T): messages(P→C) = activity × T × wires(P,C) (modeled) or Σ measured
   per-wire spikes (parity mode); hops weight by the SSOT route walk; outputs
   total/inter/intra/input-path packets + total hops. Candidate quantities gain
   `noc_total_hops`, `noc_intra_tile_packets`, … with provenance `modeled`; the
   existing pricing formulas (`e_inter_tile_hop`, `e_intra_tile_packet`) light up at
   candidate time with no pricer change.
4. **N4 pins + fidelity** — synthetic two-tile equality pin (both sides computed from
   one synthetic spike schedule); sealed-run parity check; NoC axes appear at both
   completenesses so `fidelity.json` zips modeled vs measured automatically.

### P — pruning → deterministic shapes at candidate time; de-searched

1. **P1 reuse, not twin** — candidate layout resolution applies the deployed one-shot
   structural shrink (`prune_perceptron_chain`, criterion `row_col_l1`) with the
   declared `pruning_fraction` on the materialized model BEFORE mapping. Counts are
   weight-independent (`min(floor(out·f), out−1)`, last layer exempt, adjacency-gated
   input shrink) so candidate shapes == deployed shapes by construction; packing,
   feasibility, census, and every cost quantity move with f. Foreign criteria
   (`activation`, `partial_column_group`) stay unpruned at candidate time — stated
   upper bound (the cascade is weights/stats-dependent); fidelity shows the gap.
2. **P2 pins** — realized survivor counts == floor rule on a discriminating vehicle;
   candidate softcore census == deployed softcore census at equal config (structural
   fidelity gate, like pass_count).
3. **P3 de-search** — option-axis derivation refuses `pruning_fraction` by name with a
   written reason (accuracy impact unmodeled; declare it as a run parameter). Dead
   wiring removed: `problem.pruning_fraction` field, `candidate_pruning_fraction`
   (problem + protocol). Registry key stays (it is a run parameter).

### B — buffer as a mapping performance metric

1. Record-side objective axes `carry_peak_live_bytes`, `carried_raster_bytes`
   (min, B, measured) over the sealed pass-carry census — reports, fidelity, Pareto.
2. Optional platform key `pass_buffer_capacity_bytes` (registry; absent = undeclared):
   when declared and the scheduled program's peak live bytes exceed it, deployment
   refuses with a typed error naming both numbers. Never a search axis.

### L — latency evidence

1. Loihi profile declares `t_cycle` = measured band `[5.8e-6, 13e-6] s`
   (frady2020 §5.1; caveats: multi-chip Pohoiki Springs, workload-dependent,
   throttled real-time mode is 1 ms/step per davies2021).
2. ≥1 latency reference case within ±25%: Frady k-NN query search phase
   (T=60 input window + output collection; Fig 8 absolute timing extracted from the
   cached PDF); `independence=false` (self-consistency — same paper as the constants).
   A second case from davies2021 workload tables if the numbers support one honestly.
3. TrueNorth latency case only if a clean non-pipelined published number exists;
   otherwise the refusal reason is recorded (their FPS figures are pipelined
   throughput, not latency).

### S — the study (MLP × 4 profiles; LeNet5 stretch)

1. MNIST MLP vehicle; NSGA-II over core geometry + placement/schedule/weight-bits
   (pruning declared, not searched); objectives: accuracy + absolute
   energy/area/latency + NoC axes.
2. Deploy per-profile winners → sealed records + `fidelity.json` each (now including
   NoC modeled-vs-measured and pruned-shape structural equality).
3. Cross-platform artifact: Pareto fronts, hypervolume, evidence-kind disclosure per
   constant; study report under `mimarsinan_research/docs/research/findings/`.
4. Stretch: single-profile LeNet5 repetition.

## E-series — the cost components that were silently zero (owner, 2026-08-17)

The study's energy decomposition exposed terms the candidate priced at 0 not because
they are negligible but because nothing produced their multiplicand. The governing
rule the owner set:

> if it's reasonable for non-declared cost components to be counted as zero, you can
> leave them as zero. but if declared or derivable, we should not leave them as zero.

| Question | Decision |
|---|---|
| Latency-group depth in the cycle window | A latency group has depth 1; `+own depth` is meaningless. If incomplete groups can co-reside with elements of other groups, the depth must be accounted for — and if that state is reachable at all, it is a correctness problem first. |
| Programming energy per pass | Correct accounting: a pass whose weights are already resident is NOT re-programmed. Never charge every pass as a reprogram. |
| Carry bytes | Charged separately per direction (host→chip vs chip→host): different constants, different multipliers. The COLLAPSE asymmetry is real — outbound counts at `log2(T+1)` bits/wire, the inbound re-emitted train at `T` bits/wire. |
| Cross-pass wires | They ARE input traffic of the consuming pass — correct as-is. |

### E1 — the executed window (DONE)

The candidate priced latency as `timesteps x neural_segment_count`. The runner sizes
each executed stage as `T + max_latency + 1` (cycle-based and cascaded TTFS have their
own rules), and a re-timed program runs one stage PER DEPTH LEVEL. On the sealed MLP
study run those disagreed 4 vs 15 — every latency-multiplying term (static energy,
e2e, throughput) carried the 3.75x shortfall.

`chip_simulation/stage_timesteps.py` is now the one home for the rule; the SANA-FE
runner delegates to it and the candidate applies it to its own pass/level structure
(`mapping/noc/execution_stage_latencies` + `search/problems/joint/candidate_latency_steps`).
A candidate with no layout has no stage structure, so `latency_steps` is ABSENT and the
latency-bearing axes refuse by name rather than pricing a short wall.

Load-bearing detail found while pinning it: **the default LIF recipe re-times.**
`lif_exact_qat` defaults on and pairs `lif_per_hop_retiming` on during resolution, so
the ordinary LIF run executes one stage per depth level. A reader that consults the raw
JSON sees `False` on exactly those runs (this misread cost a debugging cycle) — the
search step therefore reads the RESOLVED config, pinned in
`tests/unit/pipelining/pipeline_steps/test_search_step_carries_firing_semantics.py`.

### E2 — programming and core init (DONE)

Three multiplicands were absent at candidate time, so `e_core_init x
segment_cores`, `e_core_program x reprogrammed_cores` and `e_dma_per_byte x
reprogrammed_bytes` priced as nothing for every searched candidate.

The residency law now has ONE home: `schedule_policy.resident_passes` says that
pass 0 of a bank-clustered segment installs the banks and every later pass runs
on them, while any other composition reprograms all of them — and core INIT is
credited to neither, because a pass resets its cores' neuron state whether or
not the weights stayed (the owner's rule: "no we cannot charge every pass as
reprogram"). `mark_bank_residency` reads that law instead of carrying its own
copy, and `collect_noc_fragments` seals a `PassProgram` per pass (occupied
cores' used-row x used-column cells + residency) so the candidate can price the
program it just laid out. Bytes go through the record's own `params_bytes`.

Checked against the DEPLOYED record built from the same graph, under both
policies, all four quantities equal. Measured effect: on a
`generic_estimated_22nm` candidate the programming energy term now prices
4.6e-08 mJ with an evidence band where it was structurally absent.

Found while wiring it: **`cores_allocated` meant two different things.** The
record means the cores the mapping allocates (Σ over passes); the candidate
was reporting the DECLARED chip's core count — 36 vs 3 on the study MLP. No
formula multiplied it yet, which is exactly why it could drift unnoticed; it is
now the same per-pass count on both sides, pinned against the deployed
`CrossbarUtilizationReport`.

## Verification protocol (every stage)

Tests first; `python -m pytest tests` ≤2 min green; `./scripts/typecheck.sh` zero;
ratchets/budgets clean; templates only via `templates/generate.py`; ARCHITECTURE.md
per touched module; load-bearing guards mutation-checked; per-stage commits, no
AI-attribution trailers.
