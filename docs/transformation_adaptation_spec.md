# Behavioral Specification: Non-Destructive Gradual Model Transformation via Statistical Predictor–Corrector Continuation

**Document type:** Engineering / behavioral specification
**Status:** Draft for review
**Audience:** ML systems engineers implementing and validating the transformation framework; reviewers responsible for stability sign-off.

---

## 1. Purpose and Scope

This document specifies the required behavior, design, and validation strategy for a generic system that takes a trained neural network and *gradually* migrates its behavior toward a target **behavioral transformation** (e.g., noisy weights, noisy activations, quantized activations, clamped activations, custom-scheme quantized weights), while keeping a performance metric (e.g., accuracy) within a statistically defined budget at every committed step.

The defining requirement is **generality with stability**: a single controller must drive many different transformations — each with different sensitivity curves, different stochasticity, different gradient behavior, and different real-time effects on the metric — to full application (rate `1.0`) without destroying the model and without bespoke per-transformation control logic. The body of this spec is therefore organized around (a) a clean *contract* that isolates transformation-specific behavior, (b) a controller whose correctness rests on a small set of *explicit assumptions and invariants*, and (c) a validation strategy designed to prove those invariants hold across an adversarial range of transformation behaviors before any expensive real run.

Out of scope: the design of individual transformations themselves (only the interface they must satisfy), distributed-training mechanics, and serving/export.

---

## 2. Conceptual Framing

It is worth naming what this system *is*, because the right frame makes the stability requirements obvious rather than ad hoc.

**The process is a predictor–corrector numerical continuation over a homotopy.** Define a homotopy parameter — the **adaptation rate** $\alpha \in [0, 1]$ — that continuously deforms the model from the identity (original behavior, $\alpha = 0$) to the fully transformed behavior ($\alpha = 1$). The goal is to track a *good* set of weights along this path. Each round performs a **predictor** step (increase $\alpha$, which perturbs behavior) followed by a **corrector** step (fine-tune to restore the metric). When the corrector cannot converge within budget, the predictor step is too large and is reduced. This is exactly adaptive step-size control in continuation methods, and it inherits their core stability principle: *keep each step small enough that the corrector stays inside the basin of attraction of an acceptable solution.*

**The rate search within a round is successive approximation.** Each round optimistically attempts the full remaining jump to $\alpha = 1.0$; on failure it bisects the remaining gap (`1.0`, then `committed + gap/2`, then `committed + gap/4`, …), analogous to a successive-approximation-register search converging on the largest feasible increment.

**The acceptance test is a statistical hypothesis test, not a threshold comparison.** Because the metric is estimated from finite, possibly subsampled, possibly stochastic evaluations, every accept/reject decision is a decision under uncertainty. "Acceptable accuracy drop" must be expressed relative to the *sampling distribution* of the metric, and the controller's stability depends on controlling the false-accept and false-reject rates of this test.

Holding these three frames simultaneously — continuation (for step control), successive approximation (for the search), and hypothesis testing (for the gate) — is what lets one controller behave correctly across heterogeneous transformations.

> **Note on the worked example in the brief.** The increment sequence after committing `0.5` was written as `1.0, 0.75, 0.6125, …`; we read the third value as `0.5 + 0.125 = 0.625`. The scheme below defines the sequence precisely so it is unambiguous.

---

## 3. System Overview and Roles

The system is a closed control loop with four cleanly separated responsibilities. Keeping them separate is a hard design requirement: it is what allows the same controller to be reused and independently validated.

- **Plant** — the model plus the transformation applied at a given rate. The plant is the *only* component that is transformation-specific, and it is accessed solely through the Transformation Contract (Section 4).
- **Controller** — the rate scheduler and round/bisection logic (Section 5). It contains no transformation-specific code.
- **Sensor** — the statistical evaluator that estimates the metric and its uncertainty and renders accept/reject/recovered decisions (Section 6).
- **Recovery subsystem** — adaptive learning-rate discovery plus the fine-tuning loop that acts as the corrector (Section 7).
- **Safety / state layer** — checkpointing, rollback, invariants, and termination (Section 9). This layer is what makes the process *non-destructive*: a known-good model is always recoverable.

The controller never touches weights or transformation internals directly; it only issues commands ("apply rate $\alpha$", "probe", "recover to target", "commit", "rollback") and consumes typed decisions from the sensor. This indirection is the backbone of generality.

---

## 4. The Transformation Contract

A transformation is admitted into the framework only if it implements the following contract. The contract exists to push every source of transformation-specific variability behind a uniform interface and to make the controller's assumptions *checkable per transformation* (Section 10).

### 4.1 Required interface

```text
Transformation:
    apply(model_state, alpha) -> plant
        # Returns a runnable plant whose forward pass realizes the
        # transformation at intensity alpha. MUST satisfy:
        #   apply(s, 0.0) is functionally the identity on s
        #   apply(s, 1.0) is the full transformation
        # apply MUST be pure: it does not mutate model_state.

    forward(plant, batch) -> outputs
    backward(plant, loss) -> grads
        # Declares differentiability. If the forward op is non-differentiable
        # (quantization, clamping at saturation), backward MUST supply a named
        # surrogate gradient (e.g., straight-through estimator) and MUST declare it.

    calibrate(plant, calibration_batches) -> plant
        # For transformations with data-dependent internal parameters
        # (e.g., activation quantization scales/zero-points). Deterministic
        # given inputs. No-op if not needed.

    interpolation_mode -> {functional_blend | parameter_path | stochastic_mask}
    monotonicity_guarantee -> {guaranteed | expected | none}
    is_stochastic -> bool
    set_rng(seed)            # makes a stochastic plant reproducible
    tunable_parameters       # transform-owned params that participate in recovery
                             # (e.g., learnable quant scales), distinct from model weights
    descriptor()             # stable hash/string identifying the transform + config
```

### 4.2 Interpolation semantics

How $\alpha$ blends identity into the transformation is the single most consequential design choice for stability, and the contract requires the transformation to declare which of three modes it uses:

1. **Functional blend** — $y = (1-\alpha)\, f_{\text{orig}}(x) + \alpha\, f_T(x)$. This *guarantees* $\alpha = 0$ identity, a continuous and monotone path in output space, and differentiability whenever $f_T$ is. Its costs are extra compute (both branches run) and that intermediate $\alpha$ does not correspond to a "real" partial deployment. Recommended default whenever feasible because it gives the controller the strongest guarantees.
2. **Parameter path** — a transform-internal quantity is scheduled by $\alpha$ (noise std $= \alpha\,\sigma_{\max}$; clamp bound annealed from wide toward target; effective quantization granularity coarsened with $\alpha$). Faithful to a real partial transform, but monotonicity of *distortion* is not automatic and must be declared as `expected` or verified.
3. **Stochastic mask** — apply the full transformation to an $\alpha$-fraction of units/weights/tokens. Useful when no natural continuous path exists; introduces additional variance the sensor must account for.

The contract requires each transformation to state its `monotonicity_guarantee`. `guaranteed` (typically functional blend) lets the controller trust bisection fully. `expected` permits bisection with a monotonicity audit (Section 10). `none` forces the controller into **dense-grid safe mode** (Section 5.5).

### 4.3 Conformance requirements

Every transformation must pass the conformance tests in Section 14.1 before use: identity at $\alpha=0$ within floating-point tolerance, full transform at $\alpha=1$, purity of `apply`, finite declared gradients, deterministic `calibrate`, and working RNG control when `is_stochastic`.

---

## 5. The Rate-Search Controller

### 5.1 State carried across rounds

- `committed_alpha` — the highest rate at which a recovered, confirmed checkpoint exists. Starts at `0.0`.
- `committed_model` — the weights of that checkpoint (already adapted to `committed_alpha`). This, and only this, is the warm-start for every subsequent attempt.
- `baseline_ref` — the fixed reference metric distribution captured once at start (Section 8). Never silently updated.

### 5.2 The round / bisection algorithm

```text
committed_alpha <- 0.0
committed_model <- baseline_checkpoint
baseline_ref    <- sensor.reference(committed_model)      # fixed for the run

while committed_alpha < 1.0 - alpha_tol:
    gap  <- 1.0 - committed_alpha
    step <- gap                                # greedy: try to finish this round
    round_accepted <- false

    while step >= epsilon_step:
        alpha_try <- committed_alpha + step

        # PREDICTOR: apply transform to the *committed* (recovered) weights
        candidate <- transform.apply(committed_model, alpha_try)
        candidate <- transform.calibrate(candidate, calib_batches)
        drop      <- sensor.probe_drop(candidate, baseline_ref)     # fast, paired

        if sensor.probe_gate_pass(drop):       # recoverability gate
            # CORRECTOR: discover LR, then fine-tune to recover the metric
            lr        <- recovery.discover_lr(candidate)            # on a copy
            recovered, converged <- recovery.fine_tune(
                                        candidate,
                                        target = baseline_ref,
                                        lr     = lr,
                                        budget = recovery_budget)

            if converged and sensor.confirm(recovered, baseline_ref):
                committed_model <- recovered
                committed_alpha <- alpha_try
                safety.checkpoint(committed_model, committed_alpha)
                record_probe(alpha_try, drop, outcome=COMMIT)
                round_accepted <- true
                break
            else:
                safety.rollback_to(committed_model)    # bitwise restore
                record_probe(alpha_try, drop, outcome=RECOVERY_FAIL)
        else:
            record_probe(alpha_try, drop, outcome=PROBE_REJECT)

        step <- step / 2

    if not round_accepted:
        return finalize_partial(committed_model, committed_alpha)   # see 5.4

return finalize(committed_model, committed_alpha)
```

Three details are load-bearing and must not be "optimized away":

- **Every attempt warm-starts from `committed_model`, never from a failed attempt.** A failed larger step is rolled back completely before the smaller step is tried. This is what keeps the process non-destructive and makes the search well-defined.
- **The predictor applies the transform at the *absolute* `alpha_try` to the already-adapted committed weights.** As recovery proceeds across rounds, the model's capacity to absorb the transformation grows, so a step that failed earlier may succeed later — this is the mechanism by which the loop converges to `1.0`.
- **Probe is measured pre-tuning; commit is decided post-tuning.** These are distinct decisions with distinct thresholds (Section 6.4).

### 5.3 Why start each round at `1.0` (and when not to)

Restarting each round optimistically at the full jump minimizes the number of *rounds* (each successful big step finishes the process early) at the cost of one wasted probe per round when the transform is currently too sensitive. For smooth transformations this is near-optimal. For cliff-like transformations it wastes a probe every round. The controller therefore supports a configurable `search_policy`:

- `greedy_to_one` (default) — as specified above.
- `last_successful_step` — start each round from the previously accepted step size, growing by one doubling on success; cheaper probes for cliff-like transforms, slightly more rounds for smooth ones.
- `sensitivity_guided` — see 5.6.

### 5.4 Termination and the partial-result contract

The inner loop terminates when `step < epsilon_step` with no acceptance, meaning even the smallest admissible increment cannot be recovered. The system then **finalizes a partial result**: it returns `committed_model` at `committed_alpha`, which by invariant I1 (Section 9) is a fully valid, recovered, non-destructive model — just not at full transformation. This is a *success of a weaker kind*, reported explicitly, never an exception that loses state. A global `max_rounds` and `max_total_compute` budget bound the outer loop; hitting either also finalizes the current partial.

### 5.5 Dense-grid safe mode

If `monotonicity_guarantee == none`, bisection's assumption (a feasible step implies all smaller steps feasible) cannot be trusted, so the controller replaces bisection with a **monotone-increasing dense grid**: it evaluates a fixed grid of increments above `committed_alpha`, takes the *largest* increment that passes the probe gate *and* whose every smaller grid point also passes, then proceeds to recovery. This trades probes for robustness and is the safe default for ill-behaved transforms.

### 5.6 Sensitivity-guided stepping (optional, recommended for production)

Every probe — including failed ones — yields a data point $(\alpha, \widehat{\text{drop}})$. The controller can fit a cheap local sensitivity model (e.g., a monotone secant/Lipschitz estimate of drop vs. $\Delta\alpha$) and choose the *first* step of each round as the predicted maximum feasible increment rather than the full gap, falling back to halving on miss. This directly addresses "varying real-time effects on model performance throughout adaptation": the controller adapts its stepping to the *observed* local sensitivity instead of a fixed schedule, cutting wasted probes on cliff-like transforms while staying greedy on smooth ones. Probe results are cached so a given $\alpha$ is never re-probed against an unchanged committed model.

### 5.7 Complexity

For a single round with gap $g$, bisection issues at most $\lceil \log_2(g/\epsilon_{\text{step}}) \rceil$ probes. The number of *committed* steps to traverse $[0,1]$ is bounded below by the smallest increment that is ever recoverable and above by $1/\epsilon_{\text{step}}$. These bounds feed the compute-budget tests in Section 14.8.

---

## 6. The Statistical Acceptance Sensor

The sensor turns noisy metric measurements into typed, reproducible decisions. Getting this right is the difference between a controller that is "stable across transformations" and one that drifts or stalls because of evaluation noise — and it matters even more for stochastic transformations, where the *plant itself* is a source of metric variance.

### 6.1 What "acceptable drop = 2σ" actually has to mean

Evaluating on subsamples of the eval set yields a *sampling distribution* of the metric with some standard error (SE). A drop smaller than a few SE is not statistically distinguishable from measurement noise. So "acceptable drop within 2 std dev" is a hypothesis test of the null "this step caused no real degradation." The crucial refinements are: (a) the SE must match the evaluation protocol actually used to decide, (b) the test must be **paired** to be powerful, and (c) repeated testing across a run must not inflate false accepts.

### 6.2 Paired difference estimation (use this, not marginal σ)

Comparing the candidate's accuracy distribution to the reference's *marginal* distribution throws away enormous power, because most of the variance is shared across the two models (the same hard examples are hard for both). Evaluate both models **on the same examples** and test the *per-example difference*.

Let $a_i, b_i \in \{0,1\}$ be per-example correctness of reference and candidate over $N$ shared examples, and $d_i = a_i - b_i \in \{-1, 0, 1\}$. The drop estimate and its variance are

$$\widehat{\Delta} = \frac{1}{N}\sum_{i=1}^{N} d_i, \qquad \widehat{\mathrm{Var}}(\widehat{\Delta}) = \frac{1}{N(N-1)}\sum_{i=1}^{N}\left(d_i - \widehat{\Delta}\right)^2 .$$

Equivalently, in McNemar form, only discordant examples contribute: with $b_{10}$ = (reference right, candidate wrong) and $b_{01}$ = (reference wrong, candidate right), $\widehat{\Delta} = (b_{10}-b_{01})/N$ and $\mathrm{SE}(\widehat{\Delta}) \approx \sqrt{(b_{10}+b_{01})}/N$. The acceptance test compares $\widehat{\Delta}$ to $k\cdot \mathrm{SE}(\widehat{\Delta})$. This paired SE is typically several times smaller than the marginal $\sigma$, which both tightens the gate and makes it far less noisy.

### 6.3 Subsampling / bootstrap protocol

To obtain a robust SE without distributional assumptions, the sensor uses a **paired bootstrap**: resample examples with replacement $B$ times (or use $B$ stratified subsamples of size $m$), recompute $\widehat{\Delta}$ on each, and take the bootstrap standard deviation as $\mathrm{SE}(\widehat{\Delta})$. Requirements:

- **Stratify by class** to avoid degenerate subsamples and reduce variance.
- **Fix the subsample partition** (seed) for the run so decisions are reproducible (invariant I6).
- For **stochastic plants**, draw fresh transformation noise for each evaluation pass and *include that variance in the SE* — i.e., the SE must reflect both example sampling and transformation stochasticity, otherwise the gate will be over-confident. Average the metric over $R$ noise draws per example-subsample; report the combined SE.

### 6.4 Two gates: probe (recoverability) vs. confirm (commit)

These are deliberately different decisions and the spec separates them:

- **Probe gate** (pre-tuning, fast): "is the immediate degradation small enough that the corrector is likely to recover it within budget?" This is a *recoverability heuristic*, evaluated cheaply (small $m$, few $B$). Its strictness, `k_probe`, trades search efficiency against wasted recovery attempts. A literal reading of the brief sets `k_probe = 2` (accept only steps within 2 SE pre-tuning); this is safe but conservative and yields many tiny steps. The recommended default is a looser probe gate (e.g., accept pre-tuning drops up to a configured fraction of the global budget) with the tight statistical test reserved for commit. Both behaviors are supported via config; the choice is a documented trade-off, not a silent default.
- **Confirmation gate** (post-tuning, authoritative): the recovered model must pass on a **fresh, larger, independent** eval split, with the paired test at `k_commit` (default `2`). The recovery target is referenced to the **fixed baseline** (Section 8), not the previous committed state.

Separating a cheap noisy *search signal* from an authoritative *commit decision on independent data* is the primary mechanism for controlling false accepts (next).

### 6.5 Multiple-testing control

A full run issues many accept-tests; testing repeatedly against a noise threshold inflates the probability that *some* lucky subsample passes a step that is actually too large. Three controls, in order of importance:

1. **Independent confirmation split** (primary): the commit decision uses data disjoint from all probing, so probe-time "luck" cannot by itself cause a bad commit.
2. **Sufficient $N$ at commit**: size the confirmation set so that the SE at `k_commit` corresponds to a tolerable absolute drop; Section 14.3 calibrates this empirically.
3. **Error budgeting** (optional): apply a sequential / Bonferroni-style tightening of `k_commit` as the number of commits grows, if the empirical false-accept rate from Section 14.3 exceeds target.

### 6.6 Sensor outputs

The sensor returns typed decisions only — `PROBE_PASS/REJECT`, `RECOVERED/NOT_RECOVERED`, `CONFIRMED/UNCONFIRMED` — each accompanied by $\widehat{\Delta}$, $\mathrm{SE}$, the $k$ used, $N$, $B$, $R$, and the seeds, so every decision is auditable and reproducible.

---

## 7. The Recovery (Fine-Tuning) Subsystem

Recovery is the corrector. Its job is to take a model that has just absorbed a predictor step (a metric drop) and restore the metric to the recovery target within a compute budget, *without* destabilizing the already-good behavior carried over from the last commit.

### 7.1 Adaptive learning-rate discovery

Because the loss landscape changes both as $\alpha$ grows and as weights adapt, a fixed learning rate is unsafe across cycles; each recovery cycle re-discovers its LR with a range test (an exponential LR sweep recording loss; pick the LR roughly one order of magnitude below the loss-minimizing point, in the region of steepest stable descent). Mandatory guardrails:

- **Restore after the test.** The range test mutates weights. It must run on a copy, or the model must be restored to the post-predictor checkpoint before real tuning begins. Forgetting this silently corrupts every cycle.
- **Clamp the discovered LR** to `[lr_floor, lr_cap]`. The cap is the single most important guard against catastrophic forgetting: an over-large LR can erase the recovered behavior from prior commits in a few steps. The cap should be derived conservatively from the committed checkpoint's observed stability.
- **Average the range test over stochasticity.** For stochastic plants, average the sweep loss over several noise draws so the chosen LR is not an artifact of one realization.
- **Degenerate-sweep fallback.** If the loss curve is flat or non-monotone (no clear descent region), fall back to the previous cycle's LR scaled down by a fixed factor rather than trusting the test.

### 7.2 Tuning loop and schedule

Within a cycle, use warmup followed by a decaying schedule (one-cycle/cosine) bounded by `recovery_budget` (max optimizer steps). Tune **over the transformation's stochasticity** — for stochastic plants, draw fresh noise each step so the model becomes robust to the *distribution* the transformation induces rather than overfitting a single realization. Transform-owned `tunable_parameters` (e.g., learnable quantization scales) participate in the optimization alongside model weights. Re-`calibrate` data-dependent transform parameters on a cadence (e.g., at cycle start and every K steps) because activation statistics shift during tuning.

### 7.3 Convergence, divergence, and abort

The cycle ends `converged` when the sensor reports the metric within `k_commit` of the baseline on the (recovery-time) eval, evaluated on a schedule (not every step, to save compute). It aborts when any of the following trip, after which the safety layer restores the last committed checkpoint exactly:

- loss is non-finite or exceeds a divergence multiple of its cycle-start value;
- the metric trends *down* past a patience window (the corrector is moving away from the basin);
- `recovery_budget` is exhausted without reaching target.

An abort is **not** a system failure; it is a signal to the controller that this predictor step was too large, handled by reducing `step` (Section 5).

### 7.4 Gradient health for non-differentiable transforms

Quantization and clamping have zero or undefined gradients in regions that matter; the contract requires a declared surrogate (e.g., straight-through estimator; gradient pass-through within clamp bounds). The recovery subsystem monitors **gradient health** each cycle — fraction of saturated/dead units, surrogate-vs-finite-difference agreement on a probe batch, gradient-norm stability — and surfaces it as telemetry. Persistent unhealthy gradients (e.g., most units saturated under clamping) are a leading indicator of an unrecoverable step and feed both alerts and the sensitivity model.

---

## 8. Reference and Budget Management

### 8.1 Fixed baseline (the anti-drift requirement)

The recovery target and all acceptance tests are referenced to a **single baseline distribution captured once** from the original model at $\alpha = 0$. This is non-negotiable and is the defense against slow erosion: if each cycle's target were "back to the previous committed accuracy," measurement noise would let the reference ratchet downward over many rounds, so the model could reach $\alpha = 1$ legitimately "recovered" at each step yet far below the original. Referencing the fixed baseline closes this loophole (invariant I5). The baseline is captured with the same paired protocol and large $N$ used at confirmation.

### 8.2 Global budget

A configurable `global_budget` defines the maximum acceptable end-to-end drop of the final ($\alpha = 1$) model from the fixed baseline. The recovery target at each commit is `baseline − k_commit·SE`, and the global budget is checked at finalization. If recovery can reach $\alpha = 1$ only by spending more than the global budget, the system finalizes the best in-budget partial instead — an explicit, reported outcome.

### 8.3 Non-destructiveness

"Non-destructive" is operationalized as: at all times a checkpoint exists that passes confirmation against the fixed baseline, and any aborted work restores to it bitwise. The system never returns, and never leaves on disk as "current," a model worse than the last committed one.

---

## 9. State Machine and Invariants

### 9.1 States

`INIT` → (`PROBE` ⇄ `BISECT`) → `RECOVER` → `CONFIRM` → `COMMIT` → (loop) → `FINALIZE`, with `ROLLBACK` reachable from `RECOVER`/`CONFIRM`, and `FINALIZE_PARTIAL` reachable from `BISECT` (step underflow) or any budget exhaustion.

- **INIT** — load model; run conformance check on the transformation; capture fixed baseline; run profiling/characterization (Section 10).
- **PROBE** — apply transform at `alpha_try`, calibrate, measure pre-tuning drop, apply probe gate.
- **BISECT** — on probe reject or recovery/confirm failure, halve `step`; if `step < epsilon_step`, go to FINALIZE_PARTIAL.
- **RECOVER** — LR discovery + fine-tune; on abort, go to ROLLBACK.
- **CONFIRM** — independent-split paired test; on fail, go to ROLLBACK.
- **COMMIT** — checkpoint, advance `committed_alpha`, reset round.
- **ROLLBACK** — restore last committed checkpoint exactly; return control to BISECT.
- **FINALIZE / FINALIZE_PARTIAL** — emit final model, committed rate, full decision trace, and budget accounting.

### 9.2 Invariants (testable; these are the stability guarantees)

- **I1 — Committed safety.** Every committed checkpoint passes confirmation against the fixed baseline. The system never commits a regression.
- **I2 — Monotone committed progress.** `committed_alpha` is non-decreasing across commits.
- **I3 — Rollback soundness.** After any abort, model state (weights + transform params + RNG/optimizer state as applicable) equals the last committed checkpoint exactly.
- **I4 — Bounded search.** Probes per round $\le \lceil \log_2(\text{gap}/\epsilon_{\text{step}})\rceil$ (bisection mode); total rounds and total compute are bounded by configured budgets.
- **I5 — Reference fixity.** The baseline used for all acceptance/recovery decisions is fixed for the run; any update is explicit and audit-logged.
- **I6 — Decision reproducibility.** Given identical seeds, checkpoints, and config, every probe/recover/confirm decision reproduces bit-for-bit.
- **I7 — Predictor purity.** A predictor step never mutates the committed model; it always derives from it via pure `apply`.

These invariants are the contract the validation suite (Section 14) exists to verify. Stability of the overall process reduces to: *the assumptions of Section 10 hold for the transformation, and invariants I1–I7 are maintained by the implementation.*

---

## 10. Assumptions and Their Validation (Characterization Phase)

The controller is correct *only under explicit assumptions about the transformation*. Rather than hope they hold, the system runs a one-time **characterization/profiling phase** at INIT that measures each and configures or gates accordingly.

- **A1 — Monotone expected distortion.** Expected pre-tuning drop is non-decreasing in $\alpha$. *Validation:* sweep $\alpha$ on a grid, measure paired drop with SE; flag any statistically significant decrease. On violation: downgrade `monotonicity_guarantee` to `none` → dense-grid safe mode (5.5).
- **A2 — Recoverability.** Sufficient tuning at a committed rate restores the metric. *Validation:* at a few representative $\alpha$, attempt recovery and record whether/when it converges; estimate a recovery-budget requirement and set `recovery_budget` accordingly.
- **A3 — Bounded local sensitivity.** No cliff finer than $\epsilon_{\text{step}}$ that bisection would step over. *Validation:* from the $\alpha$-sweep, estimate the maximum local slope of drop vs. $\alpha$; if a near-vertical cliff exists, reduce $\epsilon_{\text{step}}$ or switch to dense grid near the cliff.
- **A4 — Estimable, unbiased metric.** The sensor's estimator is unbiased and its SE is well estimated (including transform stochasticity). *Validation:* the statistical calibration in 14.3.
- **A5 — Informative surrogate gradient.** For non-differentiable transforms, the declared surrogate correlates with a true descent direction. *Validation:* surrogate-vs-finite-difference agreement on probe batches; gradient-health metrics (7.4).

The characterization phase produces a **transformation profile** (sensitivity curve, monotonicity verdict, recovery-budget estimate, gradient-health summary, stochasticity magnitude) that both configures the controller and is archived with the run for reproducibility.

---

## 11. Stability Across Transformation Types

The central requirement is confidence that one strategy is stable for transformations with different characteristics and different real-time effects. The argument is structured as: the controller's behavior depends on the transformation *only* through the profile of Section 10, so it suffices to show correct behavior across the space of profiles. The failure-mode table makes the mitigations explicit.

| Failure mode | Trigger / transform type | Detection | Mitigation |
|---|---|---|---|
| Non-monotone distortion | Quantization grids that occasionally re-align; some masks | A1 sweep flags significant decrease | Dense-grid safe mode (5.5) |
| Sharp cliff finer than step | Hard clamps, aggressive low-bit quant | A3 slope estimate | Reduce `epsilon_step`; local dense grid; sensitivity-guided stepping (5.6) |
| Decision noise → false accept | Stochastic weights/activations; small eval | Paired SE; 14.3 false-accept estimate | Paired test; independent confirmation split; multiple-testing control (6.5) |
| Over-fit to one noise draw | Any stochastic transform | Eval-vs-train gap under fixed vs fresh noise | Tune & evaluate over fresh noise draws (6.3, 7.2) |
| Catastrophic forgetting in recovery | LR too high, esp. large $\alpha$ | Divergence/down-trend guards (7.3) | LR cap; abort + rollback + reduce step |
| Dead/saturated gradients | Clamping, low-bit quant | Gradient-health metrics (7.4) | Declared surrogate; gradient monitoring; treat as unrecoverable → reduce step |
| Slow reference drift | Any, over many rounds | Confirmation vs fixed baseline | Fixed baseline reference (I5, 8.1) |
| Calibration staleness | Activation quantization | Metric regressions after tuning | Periodic re-`calibrate` (7.2) |
| Non-convergence / stall | Any infeasible transform | Step underflow / budget exhaustion | Finalize partial (5.4) — never crash, never lose state |

The key structural guarantees that make this hold *uniformly*: invariants I1/I3/I7 mean no transformation can ever leave the system with a worse-than-committed model; the profile-driven configuration means transformation-specific quirks are absorbed by configuration rather than by special-case control logic; and the predictor–corrector framing means stability reduces to the well-understood property "small enough steps keep the corrector in-basin," with $\epsilon_{\text{step}}$ as the explicit safety knob.

---

## 12. Configuration Schema

All behavior is governed by an explicit, versioned config. Sensible global defaults are overridable per transformation from its characterization profile.

| Key | Meaning | Default | Notes |
|---|---|---|---|
| `epsilon_step` | Minimum admissible increment | `2^-6` (≈0.0156) | Lower → finer cliffs handled, more probes |
| `alpha_tol` | Treat `committed_alpha` as 1.0 within this | `1e-6` | Termination of outer loop |
| `search_policy` | `greedy_to_one` / `last_successful_step` / `sensitivity_guided` | `greedy_to_one` | 5.3, 5.6 |
| `k_probe` | SE multiplier for probe gate (or fraction-of-budget mode) | budget-fraction | 6.4; literal-spec mode sets `2` |
| `k_commit` | SE multiplier for confirmation | `2` | 6.4 |
| `m`, `B`, `R` | Probe subsample size, bootstrap count, noise draws | small / 200 / 4 | Fast, noisy by design |
| `N_confirm`, `B_confirm` | Confirmation split size, bootstrap count | large / 1000 | Independent of probe data |
| `recovery_budget` | Max optimizer steps per cycle | from A2 profile | Per transformation |
| `lr_floor`, `lr_cap` | LR clamp | from profile | Anti-forgetting cap critical |
| `divergence_mult`, `patience` | Recovery abort guards | `3×`, e.g. 5 evals | 7.3 |
| `global_budget` | Max end-to-end drop at α=1 | e.g. 0.5% abs | 8.2 |
| `max_rounds`, `max_total_compute` | Hard outer bounds | deployment-set | 5.4 |
| `seed_*` | RNG seeds for eval partition, transform noise, tuning | fixed | I6 |

---

## 13. Observability

Every decision must be reconstructable from logs; "stable" is unfalsifiable without this.

- **Decision trace** (per probe/commit): `alpha_try`, outcome, $\widehat{\Delta}$, SE, $k$, $N/B/R$, seeds, LR chosen, recovery steps used, checkpoint hash. This trace is the artifact regression tests snapshot (14.6).
- **Metrics / dashboards:** committed-α over wall-clock and over commits; pre-tuning drop distribution; recovery-step histogram; estimated false-accept rate; gradient-health time series; LR-chosen vs `lr_cap`; bisection depth per round.
- **Alerts:** search stuck (repeated bisection underflow), repeated rollbacks at the same $\alpha$ (sign of A2/A3 violation), LR pinned at cap (forgetting risk), gradient-health collapse, confirmation failing after recovery "converged" (sensor/recovery split disagreement → data-leakage or overfit smell).

---

## 14. Testing and Validation Strategy

Validation is layered so that the cheap, deterministic layers catch the most bugs before any expensive real run. The guiding principle: **prove the controller and sensor correct against synthetic plants with known properties first, then confirm the contract holds for each real transformation.** This is what lets us be *sure* the generalized strategy is stable, rather than discovering instability mid-run.

### 14.1 Transformation conformance tests (per transformation)

Assert the contract of Section 4: $\alpha = 0$ reproduces original outputs within fp tolerance; $\alpha = 1$ equals the full transform; `apply` is pure (input state unchanged, verified by hash); declared gradients are finite where promised and surrogate matches finite differences within tolerance; `calibrate` is deterministic; `set_rng` makes stochastic plants reproducible. A transformation cannot enter the pipeline until these pass.

### 14.2 Controller property tests against a mock-plant zoo (the crux)

Replace the expensive model+training with cheap analytic **mock plants** that expose `accuracy(alpha, tuning_progress)` surfaces with *known* properties, so the controller's logic runs in milliseconds and deterministically. The zoo must be adversarial:

- **Smooth-monotone** — easy baseline; assert it reaches `committed_alpha = 1.0` within the round bound.
- **Cliff** — accuracy falls sharply past $\alpha^\*$; assert bisection locates the edge to within `epsilon_step` and never commits past it.
- **Plateau-then-drop** — assert greedy big-step is correctly rejected then bisected.
- **Stochastic surface** — accuracy has injected noise; assert false-accept/reject behavior matches the statistical design (ties to 14.3).
- **Non-monotone** — violates A1; assert the controller detects it and switches to dense-grid safe mode.
- **Recovery-limited** — corrector recovers slowly or only partially; assert correct use of `recovery_budget`, abort, rollback, and step reduction.
- **Adversarial timing** — sensitivity changes between rounds (a step infeasible early becomes feasible later); assert convergence still completes.

For every mock plant, the suite asserts the **invariants I1–I7 hold throughout** and that the committed trajectory matches the expected outcome and probe-count bound. This is the layer that gives confidence in *stability across characteristics* because the zoo spans the space of profiles by construction.

### 14.3 Statistical validation of the acceptance gate (Monte Carlo)

Simulate per-example correctness as Bernoulli with a *known* true drop, run the full sensor (paired bootstrap, probe/confirm, multiple-testing control), and measure empirical **false-accept** and **false-reject** rates versus nominal across many simulated runs. Use this to calibrate `k`, `m/B/R`, and `N_confirm` to hit target error rates, and to validate the paired-difference SE estimator against ground truth. Repeat with injected transform stochasticity to verify the combined-SE accounting (6.3). Acceptance: empirical error rates within tolerance of nominal for all configured transforms.

### 14.4 LR-discovery tests

On small synthetic objectives and small nets: the range test selects an LR inside the stable descent region; the model is restored after the test (verify by hash); flat/degenerate sweeps trigger the documented fallback; the `lr_cap` prevents a constructed catastrophic-forgetting scenario (a high-LR sweep must not erase prior behavior because the cap binds).

### 14.5 Per-transformation integration tests (the real zoo)

For each real transformation (noisy weights, noisy activations, quantized activations, clamped activations, custom-scheme quantized weights), run the full pipeline on a small model + small dataset end-to-end. Assert: reaches $\alpha = 1$ within budget *or* returns a sensible reported partial; no invariant violated; the recorded sensitivity curve matches the characterization profile; gradient-health stays acceptable. These are slower and run on a schedule rather than per commit.

### 14.6 Regression / golden-trace tests

Snapshot the full decision trace (Section 13) for fixed seeds on the mock zoo and on the small real integrations. Any change in the sequence of probes, accepts, commits, or LRs fails the test and must be explained. This catches unintended behavioral drift in the controller during refactors — exactly the "stability" the spec is about, at the code level.

### 14.7 Metamorphic tests

For a system without ground-truth optimal trajectories, assert relations that must hold regardless of absolutes:

- Tightening `k_commit` must never *increase* final `committed_alpha`.
- Increasing `recovery_budget` must weakly increase (never decrease) final `committed_alpha`.
- Increasing eval set size must reduce decision variance (narrower SE).
- Reducing `epsilon_step` must weakly increase final `committed_alpha` for cliff plants.
- Running the same config twice with the same seeds yields identical traces (I6).

### 14.8 Chaos and budget tests

Inject eval-noise spikes, force a recovery divergence, and kill a process mid-recovery to verify checkpoint integrity and exact rollback (I3). Feed a deliberately non-monotone transform and verify safe-mode engagement. Assert total probe count and optimizer steps stay within the bounds of Section 5.7 / Section 12 (no runaway compute).

### 14.9 System acceptance criteria

The framework itself is considered validated for release when: all conformance, property, statistical, LR, metamorphic, regression, and chaos suites pass; the per-transformation integrations reach target $\alpha$ within budget or report justified partials; and the empirically measured false-accept rate over the statistical suite is at or below its configured target. Each *new* transformation is gated by 14.1 conformance + characterization (Section 10) + a 14.5 integration run.

---

## Appendix A — Reference Orchestrator (consolidated pseudocode)

```text
function run_adaptation(model0, transform, config):
    assert conformance(transform)                       # 14.1
    profile        <- characterize(transform, config)   # Sec 10
    config         <- apply_profile(config, profile)
    baseline_ref   <- sensor.reference(model0, N_confirm)  # fixed (I5)
    committed_model, committed_alpha <- model0, 0.0
    safety.checkpoint(committed_model, 0.0)

    while committed_alpha < 1.0 - alpha_tol and budgets_ok():
        gap  <- 1.0 - committed_alpha
        step <- first_step(config.search_policy, gap, history)   # 5.3/5.6
        accepted <- false
        while step >= epsilon_step:
            a <- committed_alpha + step
            cand <- transform.calibrate(transform.apply(committed_model, a), calib)
            drop <- sensor.probe_drop(cand, baseline_ref)        # paired, 6.2
            if sensor.probe_gate_pass(drop, k_probe):
                lr <- recovery.discover_lr(copy(cand))           # restore-safe, 7.1
                rec, ok <- recovery.fine_tune(cand, baseline_ref, lr, recovery_budget)
                if ok and sensor.confirm(rec, baseline_ref, k_commit):  # 6.4
                    committed_model, committed_alpha <- rec, a
                    safety.checkpoint(committed_model, a)        # I1, I2
                    history.record(a, drop, COMMIT)
                    accepted <- true; break
                else:
                    safety.rollback_to(committed_model)          # I3
                    history.record(a, drop, RECOVERY_FAIL)
            else:
                history.record(a, drop, PROBE_REJECT)
            step <- step / 2
        if not accepted:
            return finalize_partial(committed_model, committed_alpha, history)  # 5.4
    return finalize(committed_model, committed_alpha, history)
```

## Appendix B — Notation and Glossary

- **Adaptation rate $\alpha$** — homotopy parameter; $0$ = original behavior, $1$ = full transformation.
- **Predictor / corrector** — the $\alpha$-increment step / the fine-tuning that restores the metric.
- **Committed model / rate** — the highest rate at which a confirmed, recovered checkpoint exists.
- **Probe gate** — fast pre-tuning recoverability test.
- **Confirmation** — authoritative post-tuning test on an independent split against the fixed baseline.
- **Paired drop $\widehat{\Delta}$** — accuracy difference computed per-example on shared data; its SE drives the gates.
- **Profile** — the characterization output (sensitivity curve, monotonicity verdict, recovery-budget estimate, gradient health, stochasticity) that configures the controller per transformation.
- **Safe mode** — dense-grid stepping used when monotonicity cannot be guaranteed.

## Appendix C — Open Questions / Future Extensions

- **Multi-transformation composition.** Applying several transformations jointly (e.g., quantized *and* clamped) likely needs a multi-dimensional $\alpha$ and a step-control policy over a vector rate; the contract and invariants extend, but the search and statistics do not trivially.
- **Predictive correctors.** Warm-starting recovery with a learned weight-delta predictor from prior cycles could cut recovery cost; must preserve I1/I3.
- **Active eval-set selection.** Spending confirmation budget preferentially on examples near the decision boundary could tighten the paired SE at fixed $N$.
- **Formal continuation guarantees.** Conditions under which the predictor–corrector path provably stays in-basin (a Lipschitz/step-size bound tied to $\epsilon_{\text{step}}$) would upgrade the empirical stability story to a guarantee for the `guaranteed`-monotone class.
