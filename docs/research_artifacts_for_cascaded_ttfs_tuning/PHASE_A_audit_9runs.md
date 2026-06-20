# Phase A — Audit of the 9 runs (2026-06-19 12:02 batch)

Source: `generated/*_20260619_12*` (`_RUN_CONFIG/config.json`, `_GUI_STATE/{steps,run_info,console}`).
Extractor: `experiments/audit_runs.py`. Deployed numbers from the `[PROFILE]`/console lines
(nevresim sim on a **500-sample subsample** — noisy to ≈±1pp).

## Headline: 0 of 9 runs completed end-to-end. None exercised the new work.

| # | run | firing/sync | ANN | nevresim deployed (500) | crashed/killed at | total wall |
|---|---|---|---|---|---|---|
| 1 | mnist_hard_all_lif_ca60 (pruned) | LIF | 0.9815 | **0.974** | killed @ SANA-FE | 58 min |
| 2 | mnist_hard_all_lif_strict_cycle_accurate (pruned) | LIF | 0.9852 | **0.960** | killed @ SANA-FE | 63 min |
| 3 | mnist_mmixcore_lif_60 | LIF | 0.981 | **0.974** | killed @ SANA-FE | 55 min |
| 4 | mnist_mmixcore_ttfs_60_offload | TTFS analytical | 0.9821 | — | **CRASH @ SCM** | 7 min |
| 5 | mnist_mmixcore_ttfs_cycle_60_offload | TTFS cycle (cascaded) | 0.9817 | **0.946** | killed @ SANA-FE | 50 min |
| 6 | mnist_mmixcore_ttfs_cycle_60 | TTFS cycle (cascaded) | 0.9846 | **0.960** | killed @ SANA-FE | 67 min |
| 7 | mnist_mmixcore_ttfs_q_30_offload | TTFS analytical | 0.9823 | **0.978** | killed @ SANA-FE | 31 min |
| 8 | mnist_mmixcore_ttfs_q_30 | TTFS analytical | 0.9804 | — | **CRASH @ SCM** | 16 min |
| 9 | regression | TTFS cycle (cascaded) | 0.9844 | — | **CRASH @ SCM** | 17 min |

## D5 — Were the improvements applied? **NO.** (the most important finding)

Every one of the 9 configs has:
- `ttfs_staircase_ste: false`
- `ttfs_gain_correction: false`
- `ttfs_scale_aware_boundaries` / `ttfs_theta_cotrain`: off

→ **All 9 ran the OLD baseline path.** The "horrible results" are NOT the new method
failing — the new method was never turned on. Any conclusion about the new work from
this batch is void; we are looking at a broken/slow baseline pipeline.

## D3 — Crashes

**SCM crash (3/9, all TTFS): a device-mismatch bug in the parity gate.**
`soft_core_mapping_step._run_torch_sim_parity_check`
→ `nf_scm_parity.assert_torch_vs_deployed_sim_parity_or_raise` → `model(samples)`:
```
RuntimeError: Expected all tensors to be on the same device,
  got mat1 on cuda:2, different from other tensors on cpu (wrapper_CUDA_addmm)
[ModelRepresentation] forward failed at node ComputeOpMapper(name='classifier')
```
The pipeline offloads torch models to CPU between steps; the parity gate then runs
`model(samples)` with the model on CPU but `samples` on `cuda:2` (or the model partially
on GPU). Intermittent: ttfs_cycle_60 passed, regression (also ttfs_cycle) crashed →
device placement is not pinned in the gate. **The gate I added is crashing the pipeline.**

**SANA-FE (6/9): the wall.** Every run that passes SCM stops at exactly
`Running 'SANA-FE Simulation'...` with NO further output and NO error logged
(err_lines=0). nevresim Simulation completed just before. So SANA-FE either hangs or is
slow enough that the whole batch was killed there. No SIGFPE in *these* logs (the known
exit-136 is intermittent). Needs a dedicated Phase-B repro.

## D2 — Performance (catastrophic; 5-min/step budget blown 3–5×)

Worst per-step walls across the batch:
- **TTFS Cycle Fine-Tuning: 1390–1461 s (23–24 min)** — 5× over budget.
- **LIF Adaptation: 775–1098 s (13–18 min)** — 3× over.
- Pruning Adaptation: 665–758 s (11–13 min).
- Weight Quantization: up to **1183 s (20 min)**.
- Activation Quantization: 681 s (11 min).
- Soft Core Mapping: up to **1206 s (20 min)**.
- Hard Core Mapping: up to 691 s.
- **Full run ≈ 50–67 min** (LIF/ttfs_cycle). Nine runs ≈ 7–8 h before they died.

## D1 — Accuracy (nevresim, 500 samples; only 6 produced a number)

- **TTFS analytical** (ttfs_q): 0.978 / 0.982 ANN → **best**, ~0.4pp drop.
- **LIF**: 0.96–0.974 (ANN 0.981–0.985) → 1–2.5pp; pruned `strict` worst at 0.96.
- **TTFS cycle (cascaded)**: 0.946–0.960 (ANN 0.982–0.985) → **2–4pp, worst**; offload
  variant 0.946 is **below the 96% floor**.
- Caveat: 500-sample subsample (±~1pp); SANA-FE numbers never obtained → AC unverifiable.

## D4 — Complexity (made concrete)

`deployment_parameters` carries **~90 keys**, including 40+ overlapping
`ttfs_* / lif_* / tuning_*` flags (e.g. `ttfs_blend_fast`, `ttfs_genuine_blend_fast`,
`ttfs_genuine_annealed_ramp`, `ttfs_staircase_ste`, `ttfs_gain_correction`,
`ttfs_scale_aware_boundaries`, `ttfs_theta_cotrain`, `ttfs_boundary_surrogate`,
`ttfs_distmatch_*`, `ttfs_ramp_alpha_*`, ~25 `tuning_*`). This is the flag-thicket to
collapse into contract-driven, composable strategy objects (Phase D).

## Verdict feeding Phases B–E

1. The batch is uninformative about the new method (D5) — **re-run with improvements ON**
   is required before any accuracy conclusion.
2. **Two hard blockers** before any config can finish: SCM device-mismatch gate (clear
   fix) and the SANA-FE wall (needs repro).
3. **Speed is the dominant pain**: tuning steps are 3–5× over the 5-min budget — this is
   why "everything takes forever". The LIF fast-fold (~60 s) path is NOT what these runs
   used (they used `lif_blend_fast`/full controller at 13–18 min).
4. Even the baseline cascaded-TTFS deployed number (0.946–0.96) sits at/below the AC1
   floor — consistent with the known greedy-fire deficit the new work targets.
