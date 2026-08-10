# W3 stage-1 — instrumentation sweep findings (2026-08-10)

The first measured envelope evidence from the new run instrumentation
(`ft_pass_walls.json` incl. endpoint legs + `retention_ledger.json`), swept over
every lifs/lifsync tier-0 cell, fresh dirs, serial on one GPU. Raw table:
`w3_envelope_evidence_20260810.json`. This is the stage-2 (ledger/geometry
rebalance) design input.

## Findings

1. **The I.4 pathology is an ENGAGEMENT failure, not budget starvation.**
   t0_30 (the memo's flagship offload cell, deployed 0.9587 reproduced to the
   digit): the LIF Adaptation step exits **−8.98 pp with ZERO endpoint-ledger
   draw** — the recovery leg at the violating step never engaged — while WQ
   draws 3 738 steps and climbs +3.79 pp. "Recovery must sit where the loss
   occurs" therefore means fixing the *engagement predicate / target capping*
   at the violating step first; splitting the ledger differently cannot help a
   leg that never arms.
2. **The 16 k WQ ledger share never binds on tier-0.** Max total draw across
   the sweep: 11 382 / 17 200 (t0_30); max single WQ draw 4 116. The binding
   constraints are engagement + convergence geometry, not the pool size.
3. **The lifsync family never arms the LIF endpoint leg** (draw = 0 on all 8
   lifsync cells); only streamed cells draw it (up to 1 080). Healthy-cell
   deltas at the estimator are ≈0; the streamed cells exit ≈−1.2 pp.
4. **WQ's leg exits negative on 12/13 cells by the single-batch estimator**
   while the later deployed read recovers — consistent with the §13 curriculum
   (WQ's leg trains the deployed composition whose gain shows at the deployed
   read, not at the step-exit estimator). Ledger analyses must compare like
   observables.
5. **Wall asymmetry measured:** WQ endpoint legs cost 23–964 s vs LIF 0–176 s
   per run (t0_01's WQ leg alone: 963 s).

## Verdicts on the two sweep failures

- **t0_03 (sched)** — a genuine W1.2 program regression, fixed same-day: the
  declared-capacity validation rejected multi-pass schedules whose logical
  core total exceeds the physical chip. The invariant is now per-pass and the
  simulated arch unrolls the physical floorplan into row-stacked replicas
  (`floorplan_replicas`); t0_26 (sched) passed live with SANA-FE parity exact
  over 219 296 windows on the fix.
- **t0_05** — **pre-existing knife-edge FATAL, NOT a program regression**:
  nevresim↔HCM `exact=0.978426 max|dcount|=1` over 788 windows, byte-identical
  at program HEAD, at HEAD-minus-W3s1 (8b41545c), and at the handoff baseline
  (0c5d989b). Four runs across four code states reproduce the same number —
  which simultaneously demonstrates the program's training-path byte-identity
  on this cell. This is the V9/§13 structural tie class (t0_05 was already the
  memo's knob-flip FATAL cell) surfacing in today's environment era. Owner:
  the lif_deployment_exactness V9 thread; fix-path per the assert: theta
  lattice (`quantize_ir_graph`), window record path, or comb drift.

## Protocol lessons (bind future sweeps)

- **Solo per GPU is mandatory, not advisory**: t0_45 measured 0.9775 when run
  concurrently with another cell on one GPU vs the reference **0.9830 solo**
  (reproduced to the digit) — concurrency dust moves training trajectories,
  not just knife-edge certificates.
- **Never resume into a consumed-ledger dir**: a "tail-only" resume re-trained
  from scratch against the leftover `__mbh_endpoint_steps_consumed`, i.e. the
  §13 starvation trap; fresh dirs only.
