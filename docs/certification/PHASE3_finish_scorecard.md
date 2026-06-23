# Frontier Phase 3 — the finish scorecard (absolute AC verdict)

Per-cell × 6-AC verdict on the deployed parity-gated metric, via the A4 absolute-`certify()` overlay. Targets: **AC1 ≥0.96** @S=4, **AC2** lossless vs ANN (~0.984) within 0.5pp by S≤32, **AC3** non-decreasing in S (tol 1pp), **AC4** ≥ Phase-2 baseline, **AC5** ≤300s per FT pass, **AC6** zero-crash.

**6/9 MET the full spec; 3/9 lever-exercised BOUNDED-GAP.**

| cell | config | AC1 | AC2 | AC3 | AC4 | AC5 | AC6 | verdict |
|------|--------|-----|-----|-----|-----|-----|-----|---------|
| matrix_1 | LIF rate | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **MET** |
| matrix_2 | LIF Novena/offload | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **MET** |
| matrix_3 | pruned LIF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | **BOUNDED-GAP** |
| matrix_4 | TTFS analytical | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **MET** |
| matrix_5 | TTFS-quantized | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **MET** |
| matrix_6 | cascaded | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | **BOUNDED-GAP** |
| matrix_7 | synchronized | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **MET** |
| matrix_8 | cascaded offload/no-bias | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | **BOUNDED-GAP** |
| matrix_9 | TTFS vanilla no-WQ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **MET** |

## Measurements

| cell | deployed@S4 | best@S≤32 | ANN | S-curve (S4→32) | per-FT-pass wall |
|------|-------------|-----------|-----|-----------------|------------------|
| matrix_1 | 0.972 | 0.982 | 0.984 | {4: 0.972, 8: 0.966, 16: 0.98, 32: 0.982} | 148.6s |
| matrix_2 | 0.972 | 0.98 | 0.983 | {4: 0.972, 8: 0.978, 16: 0.98, 32: 0.976} | 220.3s |
| matrix_3 | 0.936 | 0.962 | 0.986 | {4: 0.936, 8: 0.956, 16: 0.956, 32: 0.962} | 18.0s |
| matrix_4 | 0.988 | 0.99 | 0.984 | {4: 0.988, 8: 0.986, 16: 0.99, 32: 0.986} | 0.9s |
| matrix_5 | 0.978 | 0.99 | 0.983 | {4: 0.978, 8: 0.978, 16: 0.982, 32: 0.99} | 2.8s |
| matrix_6 | 0.952 | 0.952 | 0.985 | {4: 0.952, 8: 0.948, 16: 0.888, 32: 0.934} | 287.2s |
| matrix_7 | 0.9649 | 0.9795 | 0.984 | {4: 0.9649, 8: 0.9729, 16: 0.9795, 32: 0.975} | 37.8s |
| matrix_8 | 0.938 | 0.938 | 0.984 | {4: 0.938, 8: 0.928, 16: 0.866, 32: 0.836} | 185.3s |
| matrix_9 | 0.982 | 0.988 | 0.984 | {4: 0.982, 8: 0.978, 16: 0.982, 32: 0.988} | 1.0s |

## Bounded-gap dossier (lever exercised, gap quantified)

- **matrix_3 (pruned LIF)** — fails AC1, AC2. deployed@S4=0.936 (AC1 owes 2.4pp); S-curve {4: 0.936, 8: 0.956, 16: 0.956, 32: 0.962}.
- **matrix_6 (cascaded)** — fails AC1, AC2, AC3. deployed@S4=0.952 (AC1 owes 0.8pp); S-curve {4: 0.952, 8: 0.948, 16: 0.888, 32: 0.934}; R2 STE-hedge @S32=[0.816, 0.85, 0.898] (exercised, WORSE).
- **matrix_8 (cascaded offload/no-bias)** — fails AC1, AC2, AC3. deployed@S4=0.938 (AC1 owes 2.2pp); S-curve {4: 0.938, 8: 0.928, 16: 0.866, 32: 0.836}; R2 STE-hedge @S32=[0.768, 0.792, 0.816] (exercised, WORSE).

**Root cause (cascaded matrix_6/8):** greedy single-spike partial-sum firing-gain deficit — S-INDEPENDENT (accuracy *falls* as S rises, AC3), so more temporal resolution cannot un-fire dead neurons; the R2 STE-hedge was RUN and made it worse. Open research axis: per-sample/per-axon firing-gain revival. **matrix_3 (pruned LIF):** pruning removes the capacity to reach AC1 (0.936); AC5 is MET at pass granularity (the 326s step is ~18 passes of ≤18s). These are honest terminal BOUNDED-GAPs, not failures.

Floor book (now carrying absolute AC targets): `docs/certification/regression_floor.json`.

## Research verdicts + close-out

- **R1 / keystone (generalization guard) — LANDED.** A6 implemented the real `CascadeCharacterizer` (4 forward-only probes); the E4 ConversionPolicy can now propose→confirm→**escalate** an off-distribution model to the controller instead of silently shipping the fast recipe (default-off ⇒ byte-identical until enabled).
- **R2 / cascaded lossless — EXERCISED, INSUFFICIENT (bounded-gap).** The STE-hedge was RUN on the real mmixcore at deploy-S and made cascaded *worse* at S=32 (matrix_8 0.84→0.77–0.82). The gap is the S-independent firing-gain deficit, not a trainable optimization gap. Closing it is open research, not a defaults flip.
- **R3 / cost-energy Pareto — INFRA LANDED.** `cost_extraction` + the reserved per-layer-S `temporal_allocation` axis carry the accuracy↔energy↔latency data; the per-layer-S optimizer itself stays reserved (validation now loudly rejects it, A3).
- **R7 / controller revival — KILLED (closed on evidence).** The 6 MET cells reach the spec on the FAST driver; the controller never beats fast on accuracy and its cost gap is only ~1.16×. It does not close any bounded-gap (cascaded is firing-gain-limited, pruned-LIF is capacity-limited — neither is controller-limited). It stays a pure fallback; no `hybrid` arm is warranted.

**Program status — finished under the honest DoD.** Every cell carries a measured absolute verdict on the deployed parity-gated metric: **6 MET the full spec**, **3 are lever-exercised BOUNDED-GAPs** (cascaded ×2 firing-gain deficit; pruned-LIF ×1 capacity) with the gap quantified, the closer run, and the open research axis named. The certification gate now reports the absolute AC verdict alongside the relative non-regression one (A4), so "certified" can no longer be misread as "spec met". No AC is silently mislabeled; no bounded-gap is an unexamined deferral.

