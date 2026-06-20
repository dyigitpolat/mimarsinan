# Program spec — generic deployment pipeline: fix accuracy, performance, crashes, complexity

Structured restatement of the requirements, for verification before launching the
engineering+research program. (Acceptance criteria = "Definition of Done".)

## 1. Acceptance criteria (the bar — what "fixed" means)

- **AC1 — S=4 floor.** For ANY ANN with ≥98% MNIST test accuracy, the DEPLOYED
  accuracy (on the chip simulator, not the training proxy) must be **≥96% at S=4**
  (≤2pp drop), for the relevant deployment config.
- **AC2 — Lossless at higher S.** As S increases the deployed accuracy must rise to
  **match the ANN exactly** (e.g. 98% → 98%), within run-to-run noise, at a practical S.
- **AC3 — Monotonicity (correctness invariant).** Deployed accuracy is non-decreasing
  in S; a drop with larger S is a BUG, never a fundamental limit.
- **AC4 — No regression.** For every config, deployed accuracy AND wall-time must be
  **≥ the prior baseline** ("what we had before"). Nothing we add may make a working
  path slower or less accurate.
- **AC5 — Speed (hard budget).** On MNIST, **no fine-tuning step may exceed 5 minutes**
  (the LIF fast-fold ~60s class is the reference for the cheapest path; 5 min is the
  ceiling for any single FT step). Simulation must not be slower than the baseline run.
- **AC6 — Robustness.** Every supported config runs **end-to-end with zero crashes**
  through all steps, including Soft-Core Mapping (SCM), Hard-Core Mapping (HCM),
  nevresim, and SANA-FE.

## 2. Defects to fix definitively (evidence: the last ~9 runs)

- **D1 Accuracy.** Deployed < ANN, below AC1/AC2; some runs worse than prior baselines.
- **D2 Performance.** Tuning cycles take "forever"; simulations slower than before.
- **D3 Crashes.** SCM step fails on some configs; SANA-FE step fails on others.
- **D4 Complexity.** The tuner/pipeline is more complex than the problem warrants.
- **D5 Are the improvements even applied?** Verify, per run, whether the intended
  deployment improvements were actually triggered/wired — or are mis-configured /
  absent / themselves the cause of the regression.

## 3. The generic configuration space (must be supported + AUTO-ASSEMBLED)

The pipeline must, from a declarative deployment configuration, automatically assemble
the *proper tools/steps* for that point in this space — and every axis must compose
with every other (no combinatorial flag-thicket):

| Axis | Options (non-exhaustive; extensible) |
|---|---|
| **Model** | any PyTorch model with NO non-mappable layer |
| **Dataset** | any |
| **Firing mode** | LIF (rate) · TTFS analytical (`ttfs`/`ttfs_quantized`) · TTFS cycle-based |
| **Sync mode** (× any firing mode) | synchronized · cascaded |
| **Encoding behavior** | subsumed · offloaded |
| **Deployment backend** | nevresim · SANA-FE · Lava · HCM/SCM sim · … |
| **Mapping strategy** | core **coalescing** (merge 2 HW cores → wider axonal input channel) · neuron **splitting** (1 SW core → N HW cores → wider neural channel/core) · **sync points** (re-program the same chip for the next set of layers when the model doesn't fit in one pass → multi-pass weight reprogramming) |
| **Weight init** | pretrained · train-from-scratch |
| **Pruning mode** | any |
| **Thresholding mode** | any (`<`, `<=`) |
| **Spike generation mode** | any (TTFS, uniform, stochastic, spike-train) |

## 4. Architectural principle (the north star for every change)

- Any codebase change must be **compatible with the full config space**, **generic**,
  **elegant**, and **mathematically beautiful** — orthogonal concerns stay orthogonal
  and **compose** (firing × sync × encoding × mapping × backend × init × pruning ×
  thresholding × spike-gen), never a combinatorial `if/else` explosion.
- The **deployment configuration declaratively drives the pipeline assembly**: a
  contract resolves which calibration, training-forward, mapping, simulator, and parity
  gates to assemble. Adding a new axis/option must NOT require rewriting the pipeline.
- Extend the codebase's OWN best patterns pipeline-wide: polymorphic policies
  (`spiking/segment_policies`), factories (`FiringStrategyFactory`), derived contracts
  (`deployment_contract.training_forward_kind`), the new `RampStrategy`.
- **Reduce** complexity (D4): collapse flag-thickets into composable strategy/policy
  objects selected by the contract.

## 5. The program (engineering + research) — to launch AFTER you confirm this spec

- **Phase A — Audit the 9 runs.** Extract each run's config (its point in §3) + per-step
  outcomes (metric, wall-time, crash) + whether the §2-D5 improvements were applied.
  Output: a defect table {config → failure mode}.
- **Phase B — Diagnose root causes** of each defect class (SCM crash, SANA-FE crash,
  tuning slowness, sim slowness, accuracy regression, not-wired improvements).
- **Phase C — Fix definitively.** Accuracy (meet AC1/AC2), performance (AC5, no AC4
  regression), crashes (AC6: SCM + SANA-FE robust across configs).
- **Phase D — Refactor for generic config-space support** (§3/§4): contract-driven,
  composable, lower-complexity architecture; verify all axes compose.
- **Phase E — Validate** the representative config matrix end-to-end; confirm AC1–AC6
  and zero regressions.

## 6. Confirmed parameters (signed off 2026-06-19 — the bar is now unambiguous)

1. **Scope of AC1/AC2 = ALL firing/sync modes.** Every firing×sync mode — LIF (rate),
   synchronized + cascaded TTFS (analytical & cycle-based) — must INDEPENDENTLY satisfy
   the ≥96%@S=4 floor and the monotone-to-lossless target. (Cascaded TTFS is the hardest
   and remains the headline, but it is not the only acceptance scope.)
2. **AC2 "higher S" = monotone → ANN-match within noise by S ≤ 32.** No single magic S
   is pinned; the invariant is AC3 monotonicity PLUS reaching ANN-exact (within
   run-to-run noise) at some practical S no larger than 32.
3. **AC6 backends in scope = ALL: nevresim, SANA-FE, HCM/SCM sim, Lava.** Every
   validated config must run end-to-end with zero crashes on all four. Fixing the
   SANA-FE (SIGFPE / exit-136) and SCM crashes from the 9 runs is part of the bar.
   (Lava is LIF-only today; its acceptance scope is the LIF configs.)
4. **AC4 baseline = latest tagged-good LIF run** (LIF fast-fold ≈0.9749 deployed in
   ~60s, from memory `lossless_fast_mnist_campaign`) as the speed/accuracy reference for
   LIF; the cascaded baseline is taken from pre-campaign main (2b8dc2f). Deployed
   accuracy AND wall-time must be ≥ these.
5. **AC5 speed = hard 5-minute ceiling per FT step on MNIST.** No single fine-tuning
   step may exceed 5 minutes on MNIST (LIF fast-fold ~60s is the cheap-path reference);
   simulation wall-time must be ≤ the corresponding baseline run.
6. **Acceptance numbers are measured on MNIST** (the AC1/AC2 reference task); the
   "any dataset" claim in §3 is a config-space *support* requirement, not an accuracy
   bar on every dataset.
