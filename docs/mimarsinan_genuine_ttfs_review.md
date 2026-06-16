# Critical Review — Genuine-TTFS Gradual Fine-Tuning, Controller Robustness & Deployment Parity

**Range:** `c05363d..2b8dc2f` on `origin/main` (since the last follow-up review).
**Size:** 86 files, **+7623 / −617**. Two feature merges (`genuine-ttfs-finetuning`, `coalescing-capability-gate`) plus a cluster of controller-robustness and torch↔deployed-simulator parity commits.
**Method:** read every commit body, the new source (`genuine_probe.py`, `blended_genuine_forward.py`, `scale_aware_boundaries.py`, `distribution_matching.py`, the +434-line `ttfs_cycle_adaptation_tuner.py`, the merge-reconciled `smooth_adaptation_cycle.py`), the `AcceptanceSensor` deltas, and the team's own `docs/tuning_accuracy_retention_report.md`. Verified the D5/D6 dissolution survived the merge and that new behavior is default-off / golden-safe.

---

## 0. The Arc, and the Verdict

The refactored controller (D1–D7) was structurally correct, but the first time it was pointed at the *real* hardware transformation — genuine cascaded-TTFS — it **committed 0/8 cycles and finished at rate 0**. A total failure of the non-destructive premise for that transformation. The investigation is the substance of this range, and it is some of the best engineering in the repository: it correctly attributed the failure to three distinct causes, fixed each at the right layer, and documented the whole retention/cost landscape with unusual honesty (including features it *reverted* and a hypothesis it *refuted*).

The three causes, and why each matters far beyond TTFS:

1. **The acceptance gate was measuring a fooled proxy.** The controller ramped and evaluated the *value-domain* per-perceptron blend, but the deployed behavior is the *genuine single-spike cross-layer cascade*. The proxy looked like it was converging while the genuine forward sat at a ~0.26 cliff. This is the concrete realization of the single biggest risk in the original spec (§6: *the sensor is only as good as the forward it evaluates*).
2. **Two controller-robustness bugs**, surfaced only by a transformation whose rate-0 baseline sits *below* the deployment target: an unachievable hard floor that stalled the ramp to rate 0, and an LR-finder cost that dominated everything.
3. **The transformation itself was a cliff** — its natural rate-path was discontinuous, violating the monotone-homotopy assumption the bisection search rests on (spec §4.2/§10).

**Verdict.** The work is excellent and the fixes are, with one architectural exception, clean and generic. But it also surfaces two uncomfortable truths the team should confront head-on: (a) the *real* win was re-conditioning the transformation, not improving the controller — and once conditioned, an experimental path that **skips the controller entirely** reaches the same accuracy, which raises a legitimate question about how much of the controller's complexity earns its keep; and (b) the genuine ramp is **default-off**, so the *shipping default* for cascaded TTFS still produces the ~0.26 cliff. The non-destructive claim currently holds only with an opt-in recipe enabled. Both are addressed below.

---

## Part A — The Fooled Proxy (vindication of the spec's central risk)

The `tuning_full_transform_probe`, now de-fooled, is the most important correctness addition in the range. `genuine_probe.py` evaluates the *actual* genuine forward on a deepcopy (`genuine_acc_on_clone`: prepare → build forward → evaluate, live model untouched) and the tuner records `proxy_gap = value − genuine`, emitting an explicit **"the value-domain probe WAS FOOLED"** verdict when the proxy converges but the genuine forward does not. This is exactly the instrumentation the original spec argued for: the gate must be anchored to deployed behavior, and a divergence between proxy and genuine must be *visible per-commit*, not discovered once at finalize.

This is a genuine vindication of the spec's emphasis (§6.4 probe-vs-confirm, §6 "measure the real behavior") and, equally, an indictment of every prior review — including mine — that trusted the torch-side metric. The lesson generalizes: **any value-domain interpolation used as a search signal must be validated against the deployed forward before it is trusted as the acceptance basis.** The team has now baked the SSOT that makes the probe identical to the deploy path (`_finalize_forward_for`), which is the right structural guarantee.

---

## Part B — The Real Win Was at the Transformation Layer (and the fast-hack lesson)

The decisive move was not a controller change — it was re-conditioning the homotopy so the cliff disappears. Three mechanism pieces, all `INVARIANT CORE` (always on, even under the fast path):

- **Scale-aware [0,1] boundaries** (`scale_aware_boundaries.py`): forward-propagate `theta_out` so each block normalizes to [0,1] and each downstream `input_scale` un-normalizes — a generic graph walk with one structural exception (the encoding layer, see Part D).
- **DFQ distribution matching** (`distribution_matching.py`): per-neuron bias correction to align the deployed cascade to the teacher ANN's activation distribution.
- **`BlendedGenuineForward`** (`blended_genuine_forward.py`): `out = (1−rate)·teacher(x) + rate·genuine(x)`, rate 0 = the frozen continuous teacher *exactly*, rate 1 = the genuine cascade *exactly*, gradients flowing only through the genuine branch.

That blend is **textbook functional-blend interpolation** (spec §4.2) applied at the *output* level to tame a transformation whose parameter-path was discontinuous. It guarantees rate-0 identity and a continuous, differentiable, monotone path by construction — so `_finalize` deploys the pure cascade with a **cliff of 0 by construction** (the teacher is simply dropped). This is precisely the prescription from the spec: when a transformation's natural path is a cliff, change the interpolation mode rather than asking the controller to bisect a discontinuity.

**The uncomfortable corollary — the fast hack.** Once the homotopy is cliff-free, the *experimental* `ttfs_genuine_blend_fast` reaches genuine **0.41 → 0.9355** by walking a **fixed** rate schedule with **one** optimizer and **no controller at all** — no scheduler, no bisection, no rollback, no recover-to-target, no LR finder. The proper controller path reaches a comparable number but pays minutes in LR-finder tax to do it. This is a real result and it deserves a blunt reading: **for a well-conditioned transformation, almost none of the controller's adaptive machinery is necessary.** The controller's complexity earns its keep only when the feasible step is unknown a priori and recovery can genuinely fail — i.e. for *ill-conditioned* transformations. The engineering payoff was overwhelmingly in conditioning the transformation (characterization + interpolation), exactly where the spec placed it, and only marginally in the controller.

This is not an argument to delete the controller — it is the correct fallback for the unknown/ill-conditioned case, and a new transformation is ill-conditioned until proven otherwise. But it is an argument that the codebase has *empirically discovered* the spec's hierarchy of concerns: **condition the transformation first; the search is then nearly trivial.** That should become an explicit design principle, not an accident buried in an experimental flag (see Part E).

---

## Part C — Controller-Robustness Fixes (correct, generic, and one improvement on the spec)

**The unachievable-floor fix (`9f9c951`) is correct and generic.** `AcceptanceSensor.absolute_floor` now drops a `pipeline_hard_floor` that sits *above* the rate-0 baseline: a deployment target the model cannot reach at any rate can never gate a per-cycle rollback, so including it rolled back every cycle and stalled to rate 0. The shortfall is reported at finalize instead of enforced by stalling. The fix is a two-line guard, model-agnostic, and a no-op in the normal/golden case. This is the right semantics — a per-cycle rollback gate must be anchored to what is *achievable*, while deployment shortfall is a finalize-time concern. It directly fixes a latent bug that any transformation with a sub-target baseline would have hit, not just TTFS.

**The non-stalling ratchet (`ratchet_threshold`) is an improvement on the original spec's anti-drift design.** My spec anchored the recovery target to a fixed baseline to prevent ratchet-down drift, but a pure fixed-floor gate can stall (every step that gives back a little gets rolled back). The new `ratchet_threshold` takes the **max** of three lower bounds — the per-step relative gate (`pre_cycle_acc − tolerance`, so a step *may* give back a little and the ramp keeps climbing), the cumulative cap (`best_committed_acc − cumulative_bound`, which **tightens as the best ratchets up**), and the absolute baseline floor. This keeps the anti-drift guarantee while removing the stall — strictly better than the fixed-floor-only design I specified. Credit where due. It is correctly gated (`tuning_rollback_ratchet`, default off) and the `best_committed_acc` high-water mark is only tracked when enabled.

**The LR-finder tax fix is correct but transformation-specific (and that's fine, scoped).** For the genuine ramp, `_invalidate_lr_cache` becomes a no-op (find LR once) and `_find_lr` uses a coarse 3×10 sweep instead of 8×30. This *relaxes* the spec's §7.1 principle (re-discover LR each cycle because the landscape shifts) — justified here because the teacher anchors the loss landscape so one LR suffices ("the fast hack proves it"), and correctly scoped to the opt-in genuine ramp (golden-safe elsewhere). It is a per-transformation override of a general rule, which is acceptable, but it should be understood as such: it is safe *because this transformation is well-conditioned*, not in general.

I confirmed the **D5/D6 dissolution survived the merge cleanly**: the recovery-quality logic was re-injected into the phase methods (`ratchet_threshold` into `_measure_post`, `_update_best_committed_acc` + refind-LR-on-miss into `_commit_cycle`), the D6 single-pass paired eval is intact, and every new knob is a default-off `getattr`-guarded branch.

---

## Part D — Deployment Faithfulness: the Metric Was Wrong

This range exposed that "accuracy retention" had been measured against the wrong reference, and the fixes here matter for the whole non-destructive premise.

- **The "SCM drop" (0.943 → 0.924) was a metric artifact, not a fidelity loss** (`5568518`/`7458836`): the NF step measured the full 10k test set while the deployment metric subsampled 500. On the *same* 500 samples, torch-vs-sim differ by +0.002 with per-sample argmax agreement 1.0000. The fixes — eval the deployment metric on the full test set by default (`deployment_metric_full_eval`) and add a per-run **torch↔deployed-sim parity gate** (bit-exact modulo rare WQ tie-flips) — are exactly right.
- **The encoding-layer parity bug (`9609cbe`) is a deep, well-debugged find.** The genuine ramp failed NF↔SCM parity at 0.656 because `calibrate_scale_aware_boundaries` retuned *every* block's scale to a teacher quantile, including the encoding block (2.170 vs the data scale 1.0) — violating the fixed input spike-encoding contract, with divergence worst at the *shallow* layers (which refuted the initial DFQ-bias hypothesis via a grad-norm probe). The fix pins the encoding block to `input_data_scale`; parity 0.656 → 1.0000 and tuning accuracy *improved* 0.9245 → 0.9393. Generic, structural, no model-specific logic.

**The lesson extends the original spec.** The spec's "fixed baseline" (§8.1) and acceptance gate (§6) must be measured on the **full** set and on the **deployed** executor, or the anti-drift accounting compares incommensurable numbers and the gate can be fooled. The team has now made both true by default. This is the correct closing of a gap that all four prior reviews (including the spec) missed because they trusted the torch metric.

The adjacent mapping work (`77f343c`: remove the lossy firing partial-sum, gate coalescing as an enforced chip capability that either fuses bit-exactly or raises `WideFanInUnsupportedError`) and the `torch↔sim` fidelity-lock harness (`2ccaa3b`, bit-exact across modes × mapping configs, with `assert_config_triggered` so no cell vacuously passes) are not tuning changes, but they underpin the deployment faithfulness the tuning premise depends on, and they are high quality.

---

## Part E — Architecture Concerns (the real critique)

**E1 — There are now two gradual-tuning engines, and the second bypasses the reviewed controller.** The `INVARIANT CORE` vs `OPTIONAL CONTROLLER` split is principled and clearly labeled, but its consequence is a second non-destructive-tuning path — the fast loop — that is a bespoke `_run` override skipping the entire four-service architecture (scheduler, rollback, recover-to-target, checkpoint guard). This is an architectural fork: the careful controller for the general case, and a parallel hand-rolled loop for the well-conditioned case, sharing only the "core." That is acceptable as a *research* state, but for cohesion the fast path should be **folded into `AdaptationDriver` as a fixed-schedule `RateScheduler` policy** (it is, conceptually, a `uniform_ladder` with rollback disabled and a one-shot LR) rather than a separate method. That keeps one engine with a policy knob instead of two engines, and it lets the fast path inherit the trace/observability and the invariant-core guarantees through the same seam the rest of the system uses. The spec's whole point was *one orchestrator, N transformations*; the fast path quietly reintroduces a second orchestrator.

**E2 — Three parallel genuine ramps + a recovery-quality flag cluster is accreting debt.** There are now three genuine-ramp paths (surrogate-annealed, teacher→genuine blend, blend-fast) and ~10 recovery-quality flags (`refind_lr_on_miss`, `recovery_lr_plateau` + factor + reductions, `rollback_ratchet` + cumulative_bound, `tight_plateau` + check_divisor, `stabilization_bounded` + ratio, `recipe_recovery`, the fast-rate/steps knobs, surrogate-alpha annealing). The project's own discipline is the right answer here, and the team is already applying it: `docs/tuning_accuracy_retention_report.md §6` honestly classifies each as landed / speed / opt-in / **reverted** (terminal stabilization, which *hurt*) / **refuted** (DFQ-bias-as-cause). The next step is to *act* on that taxonomy — graduate the survivors that clear the measured-bake bar, delete the reverted/refuted machinery, and collapse the two annealed/blend ramps toward the one that wins (the report already signals the blend ramp as the practical path). Three opt-in ramps that all reach ~0.94 by different means is two too many to maintain long-term.

**E3 — The shipping default still produces the cliff.** Every genuine improvement is default-off (correctly, until end-to-end-validated and golden-safe). But that means the *default* cascaded-TTFS pipeline still trains the value-domain proxy and eats the ~0.26 finalize cliff. The non-destructive claim, for the headline hardware target, currently holds **only with the opt-in recipe enabled.** This is the most important open item: the genuine ramp (or the folded fast policy from E1) needs a validation campaign and a path to becoming the *default* for cascaded TTFS, with the proxy ramp retained only as a fallback. Until then, "non-destructive gradual transformation" is true for LIF (bit-exact, 0.958 deployed) but aspirational-by-default for cascaded TTFS.

---

## Part F — Code-Level Notes

- **`_BlendGenuineKDLoss.__call__` reaches into `model.__dict__.get("forward")`** to recover the installed blend forward and call its `_genuine` branch. This couples the loss to the forward-install internals (an instance-level `forward` override) and is fragile to changes in how the blend forward is installed. A cleaner contract: have the loss hold the `BlendedGenuineForward` reference directly, or expose a `genuine_logits(x)` method on the model so the loss does not introspect `__dict__`.
- **The fast loop's loss `CE((1−R)·teacher + R·genuine) + 0.3·CE(genuine)`** hard-codes the 0.3 genuine-CE weight as a default in two places (`_BlendGenuineKDLoss.genuine_ce_alpha` and the fast loop). It is the "validated prototype" value; if it stays, make it one named config key rather than two literals.
- **`BlendedGenuineForward` drops its lazy genuine executor on pickle** (teacher snapshot stays light) — good, but means an unpickled blend forward rebuilds the executor on first call; ensure the genuine-probe's deepcopy path accounts for that lazy rebuild cost in the timing budget (it appears to, via `_ensure_executor`).

---

## Part G — Test Coverage

Strong, and aligned with the failure modes discovered. ~100 new unit tests plus the integration fidelity harness. The standouts: `test_genuine_gradual_invariants.py` pins a real deploy-correctness contract on a *meaningful* teacher (the earlier convergence "failure" had used a chance-level random teacher where nothing is measurable — a good catch that the *test fixture*, not the mechanism, was the problem) across five invariants (inert@low-rate, smooth degradation, r=1 == deployment bit-exact, tuning lifts the r=1 endpoint, monotone per-round convergence). `test_torch_sim_fidelity.py` / `_torch_sim_fidelity.py` lock NF == deployed-sim bit-exact across modes × mapping configs with `assert_config_triggered` guarding against vacuous passes. `test_nf_scm_parity_gate.py`, `test_distribution_matching.py`, `test_scale_aware_boundaries.py`, and the recovery-quality / genuine-ramp suites cover each new lever.

Two coverage observations: (a) invariant 5 (strict per-round monotonicity) is explicitly a *controller* guarantee that the fast path does **not** provide (the report and tests are honest that fast "can wobble per-round" but nets convergence) — so the fast path ships with weaker guarantees than the controller, reinforcing E1; (b) the genuine-ramp invariants are validated end-to-end on the real `mlp_mixer` (teacher 0.9731, deployed 0.9111, monotone) per `9f9c951`, which is the right bar — extend that same end-to-end validation to the *default-on* candidate before E3's graduation.

---

## Recommendations (prioritized)

1. **Plan the default graduation for cascaded TTFS (E3).** Run the end-to-end validation campaign for the genuine ramp (or the folded fast policy) and define the criteria to make it the *default*, retaining the proxy ramp as fallback. The non-destructive premise for the headline target depends on this.
2. **Fold the fast path into `AdaptationDriver` as a fixed-schedule policy (E1).** One orchestrator with a `fixed_ladder`/`no_rollback`/`single_lr` policy, not a second engine. It inherits the trace and the invariant-core seam.
3. **Act on the flag taxonomy (E2).** Delete the reverted/refuted machinery (terminal stabilization, the DFQ-bias path), graduate the measured survivors, collapse the annealed/blend ramps toward the winner.
4. **De-fragilize the genuine loss (F).** Replace the `model.__dict__["forward"]` introspection with a direct reference or a `genuine_logits` method; collapse the duplicated 0.3 literal into one config key.
5. **Promote "condition the transformation first" to an explicit design principle (B).** Document that the controller is the fallback for ill-conditioned transformations and that the preferred path is characterization + a cliff-free interpolation followed by a near-trivial fixed ramp — the lesson the genuine-TTFS work proved.

---

## Bottom Line

This range is the most consequential since the original refactor, and it is largely excellent work: it diagnosed a real, total failure of the non-destructive premise on the actual hardware target, attributed it correctly to a fooled proxy, two controller bugs, and a cliff-shaped transformation, and fixed each at the right layer — the controller fixes generic (and the non-stalling ratchet an improvement on the original spec), the transformation fix a textbook functional-blend homotopy, the deployment-parity work closing a gap every prior review missed. The honesty of the accompanying analysis (reverted features, refuted hypotheses, a full cost landscape) is exemplary.

The two things to confront are not defects but consequences: the genuine ramp is **default-off**, so the shipping default for cascaded TTFS still carries the ~0.26 cliff; and the experimental fast path reaching the same accuracy *without the controller* shows that the engineering leverage was in conditioning the transformation, not in the controller's sophistication — which argues for folding the fast path into the one orchestrator and promoting "condition the transformation first" from accident to principle. Address those, prune the flag debt the team has already honestly catalogued, and this becomes a cohesive, deployment-faithful system rather than an excellent research result guarded behind opt-in flags.
