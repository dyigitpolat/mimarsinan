# Predictive-Closure Program (R-series) — the full co-optimization claim

## Status (2026-08-17)

| Stage | State | Commits |
|---|---|---|
| R0 diagnosis: the "twin divergence" decomposed | **DONE** — findings below | (this doc) |
| R1 as-mapped event model + committed-cells quantity + occupancy fix | **DONE** — candidate claims `macs`/`cells_used` from the pass structure (== deployed crossbar on the replicated vehicle, both policies); events = as-mapped × T × activity (energy axes now layout-classified); the record mirror divides by the DECLARED chip; probes read ONE fragment→keys table | `search(R1)` |
| R2 host per-invocation overhead | **DONE** — `t_host_op_overhead` (measured percentile band through `execute_compute_op_torch`: 20.5 µs median on this host) × new `compute_op_count` (record: sealed; candidate: the same flow walk, ops metric); rate-only pricing keeps an unpriced-note; measured walls never double-charge. Found: the sealed 19 ms MLP wall is ONE COLD invocation (`invocations=1`) — R5 times more samples so per-pass normalization amortizes it | `physics(R2)` |
| R3 one floorplan for twin/runner/record | **DONE** — `execution_stage_placements` (stage-local re-basing, the runner's own mapping rule); the estimator prices the EXECUTED placement (a re-timed program's twin now models the true 0 the run measured); pass placements stay the fused/stage-blind fallback. Remaining: the sealed placement fragment under-describes per-stage runs (records one stage's arch) — noted for R5's findings | `noc(R3)` |
| R4 estimands + activity loop + gates | **DONE** — record answers `total_sync_barriers` (host segments from sealed stage kinds; zips 1.0==1.0 on the live run); fidelity carries the measured effective activity + declared factor; `activity_warning` fires past 1.5x either way (warn, never gate); `enforce_fidelity_gates` arms the proven-closed axes at emission (both-sides-only, fail loud) | `fidelity(R4)` |
| R5 verification study: MLP + LeNet5 as vehicles, +1 family as the generality witness | pending | |

Repo: `mimarsinan` @ `99a95489`. Goal: upgrade the defensible claim from
"structural/area/chip-side-energy verified against sealed deployments" to
"predictive at candidate time, host included, with every residual named" —
with every mechanism GENERIC over any representable model. MLP and LeNet5 are
verification vehicles only; nothing in the implementation may know a workload.
Deliverables: correctness, performance, elegance — as before.

## R0 — what the diagnosis established (all measured)

The H5 claim table listed "layout-twin placement/allocation divergence
(medium)". Diagnosed on the sealed loihi MLP run, it decomposes into two
honesty bugs and zero twin research:

1. **The twin and the deployed program AGREE.** Both allocate 7 cores and
   commit 68,096 cells; utilization zips at 0.0. The `chip_occupancy_pct`
   "residual" (13.68 vs 15.64) is a RECORD-side defect: the emission computes
   the stats mirror with the ALLOCATED cores as the chip, so record-plane
   occupancy degenerates to utilization (measured occupancy == measured
   utilization == 15.6379 on the same report). The candidate's 13.68% over
   the DECLARED 8 cores is the correct estimand.
2. **Measured hops = 0 is a FALSE ZERO.** The sealed record's `link_loads`
   is EMPTY — the SANA-FE message trace was off (default), nothing was
   measured, and `from_record` claimed hops "measured 0" anyway. The
   113.6-vs-0 "divergence" compared a model against an absent measurement.

Standing gaps from H5, unchanged: the candidate event model multiplies the
LOGICAL MAC census while replicas really fire (~108x on offload LeNet5); the
host wall is per-invocation overhead, not throughput (~2000x past the rate
model); declared activity 0.05 vs measured 0.1118; `total_sync_barriers` /
`timesteps` estimands unadjudicated.

## Owner decisions (asked 2026-08-17)

| Question | Decision |
|---|---|
| Generality witness + R5 scope | **deepcnn + full MLP×4 + LeNet5** (six runs) — conv-scale replication stresses exactly R1's as-mapped model. |
| Activity declaration loop | **Manual declaration + a LOUD deployment-time warning** when the measured effective activity misses the declaration by a significant margin (stated threshold, warning not gate). Fidelity reports the measured anchor; the operator re-declares. |
| Arm numeric fidelity gates now? | **Arm structural + closed terms now** (utilization family, area in-band, capacity, carry known-zeros) as hard checks at emission — fail loud, only when both sides answered; energy/e2e stay report-only until R5 shows their bands hold. |

## R1 — events over as-mapped cells; occupancy honest on both planes

The catalog already declares the right multiplicand: `macs` — "as-mapped MAC
sites: a weight bank replicated across cores counts once per replica — the
ENERGY multiplicand, because replicas really fire." The candidate never
claimed it, and the event model multiplied `onchip_macs` (logical).

1. `CandidateProgramming` gains the committed-cell total per pass (already
   computed as `PassProgram.core_cells`); the candidate claims `macs` and
   `cells_used` from it (they are one figure at candidate completeness —
   stated). Generic: the figure comes from the pass structure of ANY packed
   model.
2. The event model becomes `synaptic_events = macs x timesteps x
   activity_factor` — as-mapped, so replication scales it by construction.
   Unreplicated mappings are numerically unchanged (`macs == onchip_macs`),
   pinned. Events now derive from the LAYOUT: energy axes classify as
   layout-needing via the probes (more honest — events are a mapped-structure
   fact).
3. **Occupancy fix (R0 item 1)**: the record emission computes the stats
   mirror against the DECLARED chip's core types (threaded from the resolved
   platform the emission already holds), and a cross-plane pin asserts
   record `chip_occupancy_pct` == 100 x cells_used / cells_physical ==
   candidate's, on the vehicles.
4. Elegance (H4 deferral, due now that probes grow again): the hand-grown
   probe context and per-fragment null lists become ONE declarative
   fragment→quantity-keys table; E1/E2/H2/R1 each edited two sites by hand —
   the next fragment edits one row.
5. Twin-equality pins on a REPLICATED vehicle (the bank-clustered token graph
   replicates one bank across instances): candidate `macs` == deployed
   crossbar `macs`; modeled events scale with replicas; mutation kills
   (logical-census mutant, replication-dropped mutant).

## R2 — the host wall has a per-invocation cost, measured as the record measures it

1. `calibrate_host.py` gains `t_host_op_overhead`: measured by driving a
   synthetic hybrid program through `run_hybrid_stages` with the SAME
   `StageTimer` that produces the record's `host_ops_s` — the estimand is the
   deployment's own dispatch overhead (segment I/O, tensor conversion), not
   a bare torch-op launch. Synthetic program = framework fixtures only, no
   workload anything.
2. New vocabulary constant `t_host_op_overhead` (TIME, host group,
   multiplicand `compute_op_count`) + new quantity `compute_op_count`:
   record side from the sealed `schedule.compute_op_count`; candidate side
   counted generically during the flow walk the on-chip census already does.
3. Pricer: candidate host time = `compute_op_count x t_host_op_overhead +
   host_macs / host_macs_per_s`; overhead undeclared → the rate-only figure
   prices WITH an unpriced-note naming the overhead (the existing
   unpriced-list mechanism); record plane unchanged (measured wall x identity
   rate). Energy inherits through `p_host x host time` untouched.
4. Pins: formula + refusal notes + calibration block; the e2e convergence
   verdict comes from R5's runs, not from a unit test.

## R3 — one floorplan for the twin, the runner, and the record

REVISED after deeper diagnosis (the "false zero" framing was too small; the
message trace is already always on):

- The record's IDENTITY resolves a 1x2 grid at 4 cores/tile; the twin priced
  hops on that resolution (113.6 modeled — cross-tile pairs exist under it).
- The sealed PLACEMENT fragment shows ONE tile with cores [0..3] — four of
  the schedule's SEVEN cores — and every per-seg inter-tile count is 0.
- So the three readers disagree: the resolution declares one floorplan, the
  per-stage SANA-FE arches ran another (each stage's sim sizes its own arch),
  and the placement fragment describes neither run completely.

R3's job: audit the measured-NoC chain (per-stage arch synthesis → trace →
per-seg inter/intra counts → sealed placement + link loads) and make ALL
readers consume the resolved floorplan — or, where the per-stage arch is the
honest execution (each stage really is its own chip program), make the
RECORD say so (per-stage tile assignment sealed per stage) and make the twin
price per-stage assignments the same way. The modeled-vs-measured hops zip is
only meaningful after the two planes describe the same floorplan; the R5
comparison depends on this stage.

## R4 — one estimand per remaining name + the activity loop

1. `total_sync_barriers`: the record answers the SAME estimand the candidate
   prices — `schedule.sync_count` + the count of maximal compute-stage runs
   (host segments), derived generically from sealed stage kinds; pinned equal
   on the vehicles.
2. `timesteps`: verify `s_global == simulation_steps` across the firing modes
   on sealed fixtures; where a mode makes them differ, the axis reads the one
   its catalog doc means, and the other gets its own name. Pin.
3. The measured effective activity (`synaptic_events / (macs x timesteps)`)
   becomes a first-class fidelity report line — every censused run states its
   own anchor — and the DeploymentRecordStep emits a LOUD warning when the
   measured anchor misses the declared `activity_factor` by more than the
   stated threshold (owner decision: warn, never gate — the declaration is
   an assumption, and an assumption that measurement refutes must be
   impossible to not see). The loop (declare → run → warning/anchor →
   re-declare) is documented; R5's configs re-declare from the H5 anchor.
   No framework constant, no workload knowledge.
4. **Fidelity gates armed** (owner decision): the axes proven closed —
   utilization family, capacity, carry known-zeros (relative tolerance,
   stated) and area (in-band) — become hard checks at fidelity emission,
   failing the run loudly; a gate only fires when BOTH sides answered.
   Energy/e2e stay report-only until R5's bands are in evidence.

## R5 — verification: two known vehicles + one generality witness

1. Re-run with everything above: MLP x 4 profiles + LeNet5 (truenorth) +
   ONE additional representable family (per the owner decision) — all with
   the message trace on, the calibrated host block (rate + overhead), and the
   re-declared activity.
2. Acceptance, per the upgraded claim: candidate dynamic energy within the
   constants' band on unreplicated mappings and within the activity residual
   on replicated ones; candidate e2e within a stated band of measured, host
   included; `macs`/`cells_used`/occupancy zip both planes; hops compared
   honestly for the first time; every non-zipping row carries its basis.
3. The claim table in `docs/flow_health_study.md` gets its upgrade row with
   the measured numbers; program docs + project memory updated.

## Sequencing

R1 → R2 → R3 → R4 → R5 (R1–R4 are independent-ish but small; serial keeps
each commit's gate meaningful).

## Verification protocol (every stage)

Tests first; suite ≤2 min green; typecheck zero; ratchets/budgets clean;
load-bearing guards mutation-checked; generic-only (no workload constants —
the standing rule); ARCHITECTURE.md per touched module; per-stage commits, no
AI-attribution trailers.
