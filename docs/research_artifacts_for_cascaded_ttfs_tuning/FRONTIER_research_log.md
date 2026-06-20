# Frontier Program — research-scout findings (round 1)

Grounding probes from the isolated worktree scouts (R1/R2/R3/R7a). **Findings only —
the experimental code stays in the isolated worktrees; nothing landed in `src/`.** Verdicts
feed the next-stage plan; recipes graduate only after certification on the deployed metric.

## R1 — Characterization & auto-policy — **GO**

The characterization signature is **sharply discriminating** (probes ran in seconds, forward-only):
- **Cold genuine cascade is DEAD in every cell** (0.07–0.11 = chance) — revival is always required.
- **LIF ≡ staircase exactly** in every cell; the smooth ceiling **craters with depth at low S**
  (d6/S4 = 0.226, d9/S4 = 0.117) but is **~lossless at S16** (gaps +0.007 / +0.013).
- **Firing-gain split flips** under-fire → mistime as S grows (uf:mt 0.59:0.41 @ low S → 0.10:0.90 @ high S).
- **decode_corr declines** 0.46@d3 → 0.08–0.12@d9.

→ The signature cleanly selects (recipe, driver, train-S, deploy-S) and detects off-distribution
models — the propose→confirm→escalate keystone (E4) has real signal to act on. Next: promote the
4 probe families to a standing deployment-readiness pass + a cached-artifact (no-run) signature reader.

## R2 — Close the cascaded-TTFS lossless gap — **REFINE (headline accuracy)**

The **two-residual model is confirmed**: cold genuine = 0.102 (dead) **AND** staircase ceiling =
0.952–0.980 **with the same weights** ⇒ the cap is **OPTIMIZATION, not capacity**; residual B (firing-gain)
is real and θ revives it. Combo (600 steps, seed0):
- d9 S16 deploy16 = **0.950** (cont 0.965, staircase ceiling 0.952, cold 0.102)
- d9 train16 / **deploy32 = 0.942** (FREE-LUNCH holds) · d9 **train-AT-32 = 0.770** (S-NEGATIVITY −17pp)
- d6 train16 / deploy32 = **0.968** (~lossless) · STE hedge mix=0.5 escapes the pure-genuine plateau (+5.8pp)

→ **The one genuinely-missing piece = DEPLOYED-S allocation.** Split `train_S` vs `deploy_S` in the
ttfs_cycle tuner (today equal): **train at S=16** (the genuine-FT sweet spot — train-at-32 collapses to
0.770), **deploy at the smallest S whose staircase-ceiling(S, depth) ≥ ANN−ε** (from the R1 probe: d6 →
S16–32, d9 → S≥32). Port as `CalibrationPipeline`/`OptimizationDriver` steps on the now-generic axes.
This plausibly clears AC1 for d6 and approaches it for d9; the honest ceiling stays ~0.95–0.96 transferable.

## R3 — accuracy × energy × latency × area Pareto — **GO (most novel)**

Mined **118 SANA-FE-costed runs** into the real cost scatter:
- **2.8× energy span at fixed accuracy ~0.98** by mode/code (rate-ttfs 0.992@0.033 mJ vs
  genuine-synchronized 0.976@0.089 mJ).
- **Genuine cascaded S=4→8→12 is Pareto-DOMINATED everywhere**: latency 28→64→96 steps, energy
  0.040→0.061 mJ, while accuracy DROPS 0.972→0.956→0.914→0.884.
- **Energy is soma-dominated** (per-core corr 1.00 with n_neurons, 0.945 with spikes) ⇒
  **energy ≈ Σ_d neurons_d · S_d** → **per-layer S allocation** (mixed temporal resolution = the temporal
  analogue of mixed-precision) is the headline knob.

→ Promote the cost-extractor to standing infra (keyed to E6's cert cell); the headline experiment is
`set_S_per_layer` on the genuine forward + a "compiler hits a declared budget (≤X mJ / ≤Y steps)" interface.

## R7a — Controller honest re-baseline — **REFINE (incomplete this round)**

Verdict `refine`, but the scout returned no probe numbers — the toy revive→refine controller-vs-ladder
cost comparison did not complete in the worktree. **Action: re-run R7a** (cost-fixes on, deployed-anchored
paired gate, on the revived cascade) to measure the cost ratio + accuracy edge before R7b/R7c. Cheap;
the machinery exists.

---

**Bottom line for the next stage.** R1 (signal confirmed) and R3 (rich, novel front) are GO. R2 has the
decisive insight — the gap is optimization-bound and the missing lever is deployed-S allocation — and a
concrete port plan (still isolated). R7a needs a clean re-run. None of this has changed a number in `src/`;
all of it grounds what the certified Fix-B + research landings will do next.
