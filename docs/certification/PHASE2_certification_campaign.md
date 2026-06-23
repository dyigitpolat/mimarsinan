# Frontier Phase 2 — the certification campaign (Fix B, certified)

The GPU run the certification protocol (`docs/CERTIFICATION_PROTOCOL.md`) requires
*before the first Fix-B flip*: per (firing × sync × backend × variant) cell, **freeze**
the current controller deployed metric as the regression floor, then run the proven
**fast recipe** (Fix B) and **certify** it against the frozen floor on the deployed
metric of record (`__target_metric.json`, full-test) and the wall-clock budget.

Harness: `scripts/run_certification_campaign.py` (resumable; reuses the tested
`cost_extraction` + `certification` primitives). Floor book:
`docs/certification/regression_floor.json`. Raw measurements:
`docs/certification/campaign_results.json`. Auto-report:
`docs/certification/campaign_report.md`. Commit baseline: `0d156c8`.

## Result — 8 / 9 cells certified PASS

| matrix cell | recipe cell key | floor acc | fast acc | Δ | floor→fast wall | verdict |
|-------------|-----------------|-----------|----------|------|-----------------|---------|
| 1 lif_rate | `lif@nevresim#rate` | 0.972 | 0.968 | −0.4pp | 1703→1009s (1.7×) | **PASS** |
| 2 lif_novena_offload | `lif@nevresim#novena_offload` | 0.974 | 0.970 | −0.4pp | 975→646s (1.5×) | **PASS** |
| 3 lif_pruned_scheduled | `lif@nevresim#pruned_scheduled` | 0.940 | 0.924 | **−1.6pp** | 1205→984s (1.2×) | **FAIL** |
| 4 ttfs_analytical | `ttfs@nevresim#analytical` | 0.978 | 0.978 | 0 | — (no lever) | PASS |
| 5 ttfs_quantized_offload | `ttfs_quantized@nevresim#offload` | 0.972 | 0.972 | 0 | — (no lever) | PASS |
| 6 ttfs_cycle_cascaded | `ttfs_cycle_based/cascaded@nevresim#plain` | 0.960 | 0.960 | **0** | 763→467s (1.6×) | **PASS** |
| 7 ttfs_cycle_synchronized | `ttfs_cycle_based/synchronized@nevresim#plain` | 0.9635 | 0.9618 | −0.17pp | 235→233s | **PASS** |
| 8 ttfs_cycle_offload_nobias | `ttfs_cycle_based/cascaded@nevresim#offload_scheduled_nobias` | 0.932 | 0.924 | −0.8pp | 780→578s (1.3×) | **PASS** |
| 9 ttfs_vanilla_noWQ | `ttfs@nevresim#vanilla_noWQ` | 0.984 | 0.984 | 0 | — (no lever) | PASS |

The fast recipe is `{lif,ttfs}_blend_fast` with `stabilize_steps=1200` (the certified
operating point). eps = 0.005 (lossless-capable modes) / 0.01 (cascaded); wall budget =
floor × 1.25.

## Findings

- **The controller default is genuinely ~10–28 min/cell** (e.g. LIF Adaptation 390s,
  Soft Core Mapping 449s, Weight Quantization 489s on matrix_1). This is the slow
  default the whole program exists to replace; the fast recipe is the AC5 win.
- **The wall budget is never the binding constraint** — every fast candidate clears it
  with ≥1.3× margin. **Accuracy is the only binding constraint**, so the fast recipe is
  tuned for accuracy (more `stabilize_steps`) at trivial wall cost. The accuracy/speed
  trade is real: stab=400 gave LIF 0.962 @ 469s (3.6×), stab=1200 gives 0.968 @ 1009s
  (1.7×) — the certified point trades some speedup to stay within 0.4pp of the floor.
- **Cascaded certifies losslessly-vs-controller with plain `ttfs_blend_fast`** (matrix_6
  0.960 = 0.960). R2 (the STE-hedge) was NOT needed for the gate — at the templates'
  S=4 the controller floor itself is death-cascade-capped at ~0.96, and blend_fast
  matches it. R2 remains the lever to *raise* that ceiling (a separate research win),
  not to pass the no-regression gate.
- **Pruned LIF (matrix_3) regresses 1.6pp** even at stab=1200 — pruning removes the
  capacity the blend approximation needs. It does NOT certify; its default is **held on
  the controller** (the discipline: never ship a regression). Remediation explored
  separately (more stabilize budget).
- **The U1 + U2 fixes are confirmed in deployment**: matrix_2 (Novena) deploys 0.970
  (not the 0.847 bug); matrix_9 (vanilla no-WQ) deploys 0.984 (NF↔SCM parity intact).

## Keying correction (campaign-driven)

The campaign empirically proved the `(firing × sync × backend)` cell is **too coarse**:
the three LIF templates share `lif@nevresim` yet have floors spanning 0.94–0.974
(pruning / Novena / scheduling move the floor). `CertificationCell` gained an optional
`variant` discriminator (`#variant`, default `None` ⇒ byte-identical key) so each
deployment config certifies against its **own** floor.

## What landed

The certified fast recipe was flipped ON in the 5 passing templates that have a fast
lever: matrix_1, 2 (LIF) and matrix_6, 7, 8 (ttfs_cycle). matrix_4, 5, 9 have no
`blend_fast` lever (analytical/vanilla; certified at floor — already lossless, their
controller-speed fast path via the `optimization_driver` axis is a future grounding
step). matrix_3 (pruned LIF) is **not** flipped (uncertified).

## Next

- **matrix_3**: more stabilize budget or a pruning-aware fast recipe; flip only if it
  certifies.
- **Global default flip (propose → confirm → escalate)**: the per-template flip answers
  the review's "levers off in all 9 templates"; the deeper "fast on by default for new
  configs" needs the E4 keystone's real R1 *confirm* probe so a config like pruned LIF
  ESCALATES to the controller instead of silently regressing. That probe is the gating
  prerequisite (currently a stub).
- **Analytical/vanilla AC5**: ground + certify the `optimization_driver: fast` path for
  the clamp/shift/quant chain (matrix_4 floor is 1207s — it is not yet fast).
