# Flow-Health Study — the estimate↔signoff loop, measured live (H5)

Five runs re-ran the S-study protocol on the H-series machinery (2026-08-17):
`simple_mlp` × {truenorth, loihi, isaac_like, generic_estimated_22nm} and the
LeNet5 pruned-offload stretch (truenorth), NSGA-II pop 12 × 6 generations,
seed 0, five absolute objectives, `activity_factor = 0.05` declared, and —
new this study — the host rate MEASURED on the deployment host
(`scripts/calibrate_host.py`: 21.02 G/s on `sura`, evidence `measured`;
`p_host` stays a disclosed 20 W estimate, RAPL unreadable).

Every run sealed a record with the H1 event census and emitted a
full-surface, term-zipped fidelity report in-pipeline.

## The energy triangle — closed on loihi, within 2.6%

The three legs the program was built to compare, on the loihi run's chip side:

| leg | value (µJ/inference) | what it tests |
|---|---|---|
| candidate-priced dynamic (modeled events = cells × T × declared 0.05) | 0.540 | the multiplicand MODEL |
| record-priced dynamic (MEASURED events × the same constants) | 1.020 | the declared activity |
| SANA-FE measured (its own per-event energy trace) | 0.995 | the CONSTANTS |

Record-priced vs SANA-FE-measured agree within **2.6%** — the analytical
pricer, fed measured multiplicands, reproduces the simulator's independent
energy model on davies2018 constants. The candidate leg is 1.89x low, and the
cause is now a measured number rather than a suspicion:

**Measured effective activity = 0.1118** (events / cells / steps, identical
across all four MLP runs) vs the declared 0.05 — the operator assumption
understates switching 2.24x. The declared constant is the right mechanism
(EDA's switching-activity discipline); its VALUE now has a measured anchor to
re-declare from.

**The event census is geometry-invariant, measured.** All four MLP winners
differ (497,664–1,138,176 cells; different core shapes) yet seal identical
censuses: 30,450 events, 21,577 spikes. Geometry moves cores and routing, not
occupied cells — the E-series "why energy looked flat under geometry" analysis,
now confirmed at the multiplicand level.

## One estimand per name — byte-equal in production

The 2.00x utilization split is gone: `param_utilization_pct`,
`total_param_capacity`, and the wastage family zip at **0.0 relative error**
on every MLP run, and at 0.002% on the LeNet5 stretch (59.6589 vs 59.6578 —
the S-study's 6.7% gap was mostly the estimand fork, not the pruning bound).
`chip_area_mm2` is exact and in-band everywhere.

The new `chip_occupancy_pct` axis does its job in both directions: it carries
the chip-sizing signal by name, and its residual (12.5% on the MLPs, 5.1% on
LeNet5, utilization equal alongside) isolates the layout-twin ALLOCATION
divergence — the deployed builder allocates more cores than the twin's packing
— on its own axis instead of contaminating utilization.

Carry zips as known zeros on every run (single-pass-per-segment programs on
both planes) — present at 0, never refused, exactly the H2 semantics.

## The host term, decomposed at last

The term zip splits what one axis-level error hid. On loihi:

| term | predicted | record-plane | error |
|---|---|---|---|
| `energy_dynamic_mj` | 5.40e-4 | 1.02e-3 | −47% (= the activity, above) |
| `energy_static_mj` | 1.42e-3 | 0.125 | −98.9% (static × the e2e gap) |
| `latency_host_s` | 9.55e-6 | 1.90e-2 | −99.95% |
| `energy_host_mj` | 0.191 | 379.6 | −99.95% |

The host finding: calibrating the RATE (10 → 21.02 G/s measured) was
necessary but nowhere near sufficient — the measured host wall is 19 ms for
~200k MACs, ~2000x the MAC-bound model. **The host wall is per-invocation
overhead, not throughput.** The model needs a `t_host_op_overhead x
compute_op_count` term (measurable by the same calibration harness) before
host-inclusive e2e can converge; until then the term rows keep the chip-side
comparisons clean, which is what they were built for.

A declaration gap this exposed was closed mid-study: only loihi declared the
identity `host_compute_rate`, so only loihi's RECORD plane could price the
measured host wall — the other three refused their headline by that exact
name. All shipped profiles now declare the identity rate (derived, loihi's
own precedent); the four sealed MLP records predate the declaration on three
profiles, so their record-plane headlines refuse honestly in the artifacts,
and the next runs inherit the fix.

## The refusal machinery, fired in production

The LeNet5 stretch sealed `synaptic_events = None`: stages 0 and 2 measured
(2,845,860 + 1,727,328 events) but stage 4's census refused — a span reads a
producer absent from the trace groups — and the None propagated through the
per-stage record, the aggregate, and the sealed energy fragment, leaving the
record-plane energy refused BY NAME. That is the designed behavior working on
the first conv-scale run: a partial count never priced as a full one.
Follow-up (1) below adjudicates the root cause.

## Standing results, re-verified on the new machinery

Different physics still choose different chips (four winners, 2.3x capacity
spread, all at deployed accuracy 0.9810); the LeNet5 stretch sealed at 0.9836
deployed, 52.40 mm² (exact both planes), 23.4 µJ/sample measured. All 8
silicon-correlation cases stay in band; the full suite holds at 11,634 tests
≤60 s, typecheck 0.

## Follow-ups (ranked)

1. **LeNet5 stage-4 census refusal** — ROOT-CAUSED and fixed post-study: the
   finalizer joined `core_to_group`'s GROUP OBJECTS against the name-keyed
   trace counts, so `emissions_of` missed on every inter-core read. Only a
   FUSED multi-latency stage has such reads (per-hop retiming turns MLP
   intra-stage dependencies into stage boundaries), so LeNet5's fused
   classifier was the first production exercise of the path. The join now
   lives beside the census (`census_from_trace_groups`), pinned by a runner
   test with a real inter-core dependency; the refusal machinery behaved
   exactly as designed throughout. RESEALED: the re-run measures every stage
   (2,845,860 + 1,727,328 + 32,220 = 4,605,408 events/inference) with the
   utilization zip holding at 0.002%. The censused record exposed the next
   model refinement: record-plane dynamic energy is ~108x the candidate's
   model, because the offloaded encoder's BANK REPLICATION multiplies
   physical arrivals — modeled events use the LOGICAL (replication-free) MAC
   census, but replicas really fire (the as-mapped-vs-logical distinction the
   quantity catalog already states). The candidate-side fix is to model
   events over its own committed cells (E2/H3 already compute them) rather
   than the partition census — promoted to follow-up 3b.
2. **Host per-invocation overhead**: extend `calibrate_host.py` to measure
   `t_host_op_overhead`; add the `x compute_op_count` term; re-zip e2e.
3. **Re-declare `activity_factor`** from the measured 0.1118 (or declare
   per-workload) — the candidate energy leg then lands within the constants'
   band.
3b. **Model candidate events over AS-MAPPED cells** (the committed rectangle
   the candidate already computes), not the logical partition census: on
   replicated mappings (offload conv) the two diverge by the replication
   factor — measured ~108x combined with the activity gap on LeNet5.
4. **Layout-twin allocation divergence** (`chip_occupancy_pct` residual +
   `noc_total_hops` 113.6 modeled vs 0 measured): the twin packs tighter and
   places differently than the deployed builder — the standing N-series
   follow-up, now measurable per run on its own axes.
5. **`total_sync_barriers` / `timesteps` estimands** (deferred from H3): the
   basis rows now state the mismatch per run; adjudicate on this evidence.


## R-series verification (2026-08-17, six runs, gates ARMED)

The predictive-closure program's acceptance batch — the first study run with
the fidelity gates live, and they earned their keep twice before any row was
written: catching the record-side `tiles` estimand fork (exactly one tile's
router+fixed area on the component-priced profiles) and the multi-pass
utilization fork (the candidate reported one pass's 33.3% against the
record's aggregated 40.8% on the deepcnn witness — a gap no single-pass
vehicle could expose). Both fixed and pinned; the runs resealed on the fixes.

**Candidate dynamic energy is now predictive** (declared activity 0.1118 from
the H5 anchor; measured effective activity 0.109 on every MLP, 0.099 on
LeNet5 — the anchor generalizes):

| run | dyn-energy err (candidate vs record-plane) | hops zip |
|---|---|---|
| truenorth MLP | +2.5% | 0.0 == 0.0 |
| isaac_like MLP | −1.2% | 0.0 == 0.0 |
| loihi MLP | −6.8% | 0.0 == 0.0 |
| **LeNet5 (replicated conv, was ~108x wrong)** | **+12.7%** | 0.0 == 0.0 |
| generic_estimated_22nm MLP | −62.7% (composition gap, see below) | 0.0 == 0.0 |

The **hops zip closes on every run** — R3's executed-placement model prices
the true zero the per-stage remapping produces.

**The generic outlier is a composition gap, not a model error**: its
component-priced dynamic includes terms only the record can back
(`boundary_events`, leak x `neurons_used` — measured-only multiplicands).
Candidate-side `neurons_used` is derivable from the committed rectangle;
named follow-up.

**The host residual is now precisely attributed**: with 5 timed samples
amortizing cold start, the per-op wall is still ~9 ms — data-marshalling
dominated, exactly the R2 estimand note's prediction. The per-byte
marshalling term is the one remaining host follow-up; the term table keeps
it isolated from the chip-side claims.

Claim upgrade, as of this batch: **"predictive at candidate time on the chip
side — dynamic energy within ±13% across four profiles and a replicated conv,
structural axes gated, traffic zipped — with the host wall's marshalling term
and the generic profile's component multiplicands as the two named residuals."**
