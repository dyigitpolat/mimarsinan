# Final Recommendations — Action Plan + Research Program (R1–R7)

**Context:** the deployment pipeline now has a clean, equivalence-locked structural substrate (the V1–V8 refactor), but every config still runs the slow controller with the fast/lossless levers off, and the optimization-driver/calibration concerns the design doc lists as first-class axes are still fragmented, TTFS-bound, default-off flags. This document is the recommended path from "clean substrate" to "fast + lossless by default, and generalizable" — an action plan (Fix A / keystone / Fix B) and a seven-thread research program, including **R7**, a deliberate re-evaluation of the original non-destructive rollback controller.

**Companions:** the Behavioral Specification, the fast/lossless plan, the four review documents, the team's `DESIGN_GOALS` (10 principles) and `PROGRAM_SPEC` (AC1–AC6).

---

## 0. Executive Recommendation

- **Do Fix A + R1 + R6 immediately and non-negotiably.** They are byte-identical or pure infrastructure, low-risk, and they unblock everything: a pipeline-wide `OptimizationDriver` *and* `CalibrationPipeline` axis (not just TTFS), a characterization-and-policy layer that makes defaults safe, and deployment-faithfulness gates that make every accuracy number trustworthy.
- **Treat Fix B (turn the proven recipes on by default) as a deliberate, per-cell, gated behavior change** — never a blanket flag flip. Roll out lowest-risk-first (LIF → cascaded-TTFS → analytical → synchronized), each certified on the deployed metric against a frozen regression floor.
- **Be honest about the ceiling:** Fix B delivers the *speed* goal (AC5) outright and closes the bulk of the accuracy gap (cascaded TTFS 0.26 → ~0.95), but the best *transferable* cascaded recipe today lands *at or just below* AC1's 0.96 line. The final ~1–2 pp to a clean AC1/AC2 pass is a **research** result (R2, and possibly R7), not a defaults flip.
- **The most novel and defensible research bet is R3** (the accuracy ↔ energy ↔ latency ↔ area Pareto — what a chip compiler ultimately exists to produce). **R2** is the headline accuracy deliverable. **R7** is a cheap, high-information revival of the rollback controller that targets exactly the AC1 residual Fix B leaves open, plus a certified-non-destructive mode.

---

## Part A — The Action Plan

### Fix A — Genericity: unbind the driver *and* calibration into pipeline-wide axes

Endorsed, with two corrections of scope. Lifting the fast-ladder to `SmoothAdaptationTuner` is correct, but **"move the method up" is the easy 20%.** The real deliverable is the **uniform rate-tuner seam** the driver consumes: every rate tuner — LIF, the analytical clamp/shift/quant chain, the manager-rate tuners, the KD-blend tuners — must expose the same three-verb contract, `ramp(rate)` / `recover_to(target)` / `probe()`, against which an `OptimizationDriver` (`controller | fast | …`) is generic. Frame it as: *the driver is an axis that drives an `AdaptationAxis`-shaped tuner, and every tuner implements that shape.* This is the genuinely-generic version of the trifecta, and it stays byte-identical because the axis defaults to `controller` until Fix B.

Two scope corrections to the plan as written:

1. **Unbind `CalibrationPipeline` too, not just the driver.** The design doc lists *both* "driver × ramp" and "calibration" as first-class axes; both are fragmented `ttfs_*` flags today. Gain-correction / θ-cotrain / distmatch are spiking-conversion-health concerns keyed by (firing × sync), not TTFS-tuner internals — if only the driver is unbound, calibration stays a TTFS island and the analytical chain can never receive the conversion-health steps it may need. Unbind both in the same strangler pass.
2. **Do not make the contract a static mode→recipe lookup table.** That is the missing keystone (below); skipping it is precisely where Fix B turns into a silent-regression generator.

`OptimizationDriver` and `CalibrationPipeline` then become declarative axes resolved by `DeploymentPlan` and consumed uniformly by every rate tuner. LIF stops bypassing the driver; the analytical chain finally gets a fast path. Byte-identical, locked by the existing equivalence harness.

### The Keystone — a characterization-and-policy layer (build it as part of Fix A)

Fix B as stated ("`SpikingModePolicy` derives the proven recipe per firing × sync, default it on") uses a **static table by mode**. That table is the right *prior*, but it is unsafe as the whole mechanism for one blunt reason: **every proven recipe was validated on MNIST** (toy digits and mmixcore), while the program promises "any model, any dataset." A hard-coded "cascaded → two-stage revive" will apply a MNIST-tuned recipe to a model where the cascade behaves differently — and because it is now the *default*, that is a regression nobody asked for, found only after the run.

Insert a thin layer the contract consults:

1. The contract **proposes** the recipe by mode (the table — the prior).
2. A **cheap pre-flight characterization confirms it on this specific model** before committing: is the cold cascade dead (revive needed)? is the ramp monotone? what is the staircase/LIF ceiling at this depth and S? is the firing-gain deficit present? These are the probes the research already built in isolated form; they run in seconds, forward-mostly.
3. If the model matches the recipe's assumptions → run fast. If it does not → **escalate to the controller fallback** rather than ship a regression.

This makes "the pipeline is *derived* from the config" true in the strong sense — derived from the config **and the model**, not just declared flags — and it is the transfer-safety guard that lets Fix B ship across the matrix instead of only on the 9 MNIST templates. Build it as part of Fix A (infra, byte-identical, default-off) so Fix B has something safe to switch on. (This layer is also research thread R1.)

### Fix B — Defaults: derive the proven recipe per (firing × sync) and default it on

Endorsed, including the critical judgment **not to blanket-on the levers** — the research is explicit that the STE is refinement-only (revival mandatory), gain-correction is cascaded-specific, and several levers compose badly. So Fix B is "the contract picks the proven recipe per mode," not "turn everything on." Three hard requirements make it safe:

**1. Replace byte-identity with a per-cell Pareto gate — the protocol you do not yet have.** Fix B *breaks every equivalence lock by design* (it changes numbers). Before flipping anything, **freeze the current slow/lossy numbers per matrix cell as the regression floor**, and make the new gate: *deployed accuracy ≥ floor − ε* **and** *wall-clock ≤ budget*, per (firing × sync × backend) cell, enforced in CI. This converts AC4 ("no regression") from a hope into a runtime invariant and gives a clean "this commit changes numbers, here is the new certified baseline" story. Without it, you have traded a test-locked-correct codebase for an un-gated behavior change.

**2. Roll out per cell, lowest-risk first.** LIF fast-fold first (proven lossless-equivalent, ~9×, the safest possible first flip), then cascaded-TTFS two-stage (proxy-revive + stabilize, ~0.95 @ S=4 in ~70 s vs ~24 min), then analytical TTFS (the fast rate-ladder — needs Fix A), then synchronized TTFS separately. Certify each cell before the next; never flip the whole matrix in one commit.

**3. All accuracy is the deployed-forward, full-test, parity-gated number — make it the only number of record.** This saga had four "the torch metric lied" episodes (the SCM-drop artifact, the encoding-layer scale, the SANA-FE version drift, the fooled value-probe). Wire the Fix B certification *exclusively* to the deployed metric behind the torch↔sim parity gate with pinned deps. A proxy win that does not deploy is worse than no change. (This discipline is research thread R6.)

**The honest reality check on Fix B's ceiling.** The best *transferable* cascaded recipe today lands around ~0.95 deployed at S=4 — i.e. at or just below AC1's 0.96 line (the slower controller-proxy reached ~0.956). So Fix B delivers AC5 (speed) outright and closes the vast majority of the accuracy gap (0.26 → ~0.95), but it does **not** by itself clear a clean AC1/AC2 pass for cascaded TTFS. The final ~1–2 pp is a research result (R2's transferred near-lossless recipe + the deployed-S allocation for the quantization floor, and possibly R7's surgical rollback). Communicate Fix B as "fast + dramatically better, AC5 met, AC1 within reach but not guaranteed for cascaded," so nobody reads "defaults shipped" as "program done."

### What the original framing omits

- **Energy / latency / area are first-class for a chip compiler and absent from the ACs.** The sims already report spikes and mJ. "Lossless" for a neuromorphic target is not a scalar — it is a point on an accuracy ↔ energy ↔ latency ↔ cores Pareto front, and the S/code knobs trade directly along it. (Research thread R3.)
- **The cascaded lossless gap is research, not defaults** — keep it out of Fix B and in R2/R7.
- **Generalization beyond MNIST is the real test of the whole edifice** and is currently unproven. (Research thread R4.)

### Sequencing of the action plan

Fix A (unbind driver **and** calibration into pipeline-wide axes + define the uniform rate-tuner seam + build the characterization/policy scaffolding) — all byte-identical → **swap the byte-identity gate for the per-cell Pareto/regression gate** → Fix B per cell (LIF → cascaded → analytical → synchronized), each certified on the deployed metric → research feeds better recipes into the now-generic axes without further plumbing.

### How far to go (the decision framing)

- **Non-negotiable, immediate:** Fix A + R1 + R6. Safety and genericity, byte-identical/infra, low-risk.
- **Deliberate, gated:** Fix B, per cell, behind the Pareto gate. Ship LIF and cascaded once green; be explicit that cascaded lands near but likely just under AC1 until R2/R7 deliver.
- **Research bets:** R2–R5 and R7 (below), with R3 the one to push hardest.

---

## Part B — The Research Program (R1–R7)

Each thread states a thesis/hypothesis, the work, success criteria, and (for the bets) a kill criterion so dead ends die fast.

### R1 — Characterization & auto-policy (the keystone; also ships as infra)

**Thesis.** Robust, safe defaults require deriving the recipe from a *measured profile of the model*, not a static mode-table.
**Work.** Turn the isolated probes into a standing "deployment-readiness" pass emitting, per model: cold-cascade liveness, ramp conditioning/monotonicity, the staircase/LIF ceiling vs depth and S, and the firing-gain profile — and a **policy** that selects (recipe, driver, training-S, deployed-S) and provides the Fix-B transfer-safety guard (propose → confirm → escalate).
**Success.** The policy picks the right path on a held-out set of models and *escalates rather than silently regresses* when a model is off-distribution. This is the bridge to the spec's true "auto-assembled."
**Priority.** Immediate (it gates safe defaults and everything below).

### R2 — Close the cascaded-TTFS lossless gap (the headline accuracy deliverable)

**Thesis.** The cascaded gap is optimization-bound, not capacity-bound; the near-lossless recipe proven on the toy harness transfers.
**Work.** Port the artifact-51 combo (joint per-channel θ-cotrain + progressive shallow→deep unfreeze + continuous-teacher KD, trained at *low* S) onto the real pipeline, and operationalize the **two-residual model**: the quantization floor (LIF ≡ staircase, compounds with depth → set *deployed* S high enough) is one lever; the firing-gain deficit (θ) is the other; *train* at low S because the genuine FT is S-negative.
**Success.** Deployed cascaded ≥ 0.96 at a practical S and lossless-vs-ANN at the S the floor needs, monotone in deployed-S (AC1–AC3), on the deployed metric.
**Kill.** If the combo does not transfer off MNIST after R4's harder models, fall back to "cascaded is fast-but-~0.95" and route the AC1-critical configs to synchronized/LIF.

### R3 — The accuracy ↔ energy ↔ latency ↔ area Pareto (the chip-compiler objective; most novel)

**Thesis.** For a neuromorphic-deployment compiler, "lossless" is a *point on a front*, and the S/code knobs trade along it; optimizing accuracy while silently 4×-ing the time-steps is the wrong objective.
**Work.** Make deployed cost (spikes, mJ, latency in time-steps, cores) co-equal with accuracy. Knobs: **per-layer S allocation** (mixed temporal resolution — the temporal analogue of mixed-precision quantization), and richer codes where the hardware allows (multi-spike/burst, graded spikes). Characterize the front per model; have the compiler accept a constraint ("≤ X mJ" or "≤ Y time-steps") and return the best-accuracy point.
**Success.** A published Pareto frontier per model and a compiler that hits a declared budget. This is the most underexplored axis and the most defensible contribution.
**Kill.** None — even a negative result (the front is flat / one code dominates) is a valuable, publishable characterization.

### R4 — Generalization (any model, any dataset)

**Thesis.** The characterization (R1) and recipe (R2) hold or escalate safely beyond MNIST.
**Work.** Validate on CIFAR, a transformer/attention core, and a deeper net; find where they break (deeper cascades, different activation statistics, attention) and extend.
**Success.** Recipe-by-characterization holds or escalates safely beyond MNIST — the thing that turns "works on our 9 templates" into "works on the config space the spec promises."
**Kill.** If a class of models is fundamentally non-transferable, document it as an explicit unsupported region rather than shipping a silent regression.

### R5 — Co-design / SNN-readiness preconditioning (highest ceiling; longer horizon; parallel)

**Thesis.** A continuation method only reaches a lossless endpoint if a lossless endpoint exists *near the start*; the cold genuine cascade is dead because the pretrained ANN sits in an SNN-hostile basin.
**Work.** Constraint-aware pretraining — quantization + a spiking-friendliness penalty + bounded activations in the loss from the start — so the model is *born* revivable and SNN-friendly.
**Success.** A preconditioned model converts near-trivially (the recipe becomes light or unnecessary) and/or reaches a strictly higher lossless ceiling; quantify the convert-cost / ceiling win vs converting a hostile ANN.
**Kill.** If preconditioning does not measurably ease conversion on the target archs, the basin hypothesis is wrong for them — record and move on.
**Priority.** Run in parallel from the start; it can dominate everything downstream.

### R6 — Deployment-faithfulness as standing infrastructure (cross-cutting; mostly built)

**Thesis.** The deployed-forward full-test number is the only accuracy of record.
**Work.** Institutionalize what the saga taught: torch↔sim parity as a gate on every run; all external deps pinned with version/capability guards at each boundary; drift detection so a silent upgrade or a metric-protocol change fails loud.
**Success.** No accuracy claim ever rests on a proxy or a subsample again; the R2/R3/R7 certifications wire to this exclusively.
**Priority.** Always-on; start immediately alongside Fix A.

### R7 — Reviving the non-destructive rollback controller (the "one more chance" study)

**Thesis.** The original predictor–corrector rollback controller (greedy-to-1.0 + bisect-the-gap + recover-to-target + rollback-to-last-committed) was demoted to "ill-conditioned fallback" on the basis of three measurements that were all **confounded**, and it deserves a clean re-evaluation with everything we now know.

The three confounds in the demotion:
- it was run **cold** (no revive) on a dead cascade → 0/8 commits;
- it was gated on a **fooled value-proxy** and an **unachievable hard floor** → it rolled back everything;
- it was benchmarked with **every cost fix off** (full-model clones, 8×30 LR sweeps, per-cycle eval) → ~388–559 s, making "slow" look intrinsic;
- and it was compared to the fast ladder on the **smooth** case (LIF) where adaptive search is definitionally pointless.

Yet the one time it ran on a **revived, correctly-gated** cascade it reached **0.956 — higher than the fast ladder's 0.95.** That single clean data point says the controller's adaptive rollback may extract residual accuracy the fixed ladder leaves on the table — precisely the 1–2 pp toward AC1/AC2 that Fix B does not close. R7 re-evaluates the controller where it should win, using five insights that change its prospects:

1. **Deployed-anchored, paired (McNemar) gate** (from the fooled-proxy lesson + the spec's paired-difference design) — the controller now measures the *right thing*.
2. **Revive-then-refine** (from the cold-dead-cascade lesson) — start the controller from a *revived* model (proxy/teacher blend), not cold; the controller is the *second* stage, replacing the fixed stabilize.
3. **Characterization (R1)** — localize *where* the ramp is ill-conditioned, so the controller is applied **surgically** (per-layer / per-rate-region) rather than over the whole run.
4. **The cost fixes already built** (scoped/one-shot LR, tunable-scope async checkpoint guard, paired eval) — the controller's overhead may now be competitive, but this was **never re-measured** with them on.
5. **The hard non-destructive invariant** (the spec's I1/I3) — the controller *guarantees* no committed regression; the fast ladder only achieves it empirically via bounded stabilize. For safety-critical / zero-regression-tolerance configs, that guarantee has standalone value.

**Sub-experiments:**

- **R7a — Honest re-baseline.** Run the controller with *all* cost fixes on (loss-slope/one-shot LR, tunable-scope async checkpoint, paired deployed eval, R1-set ε) on the same cells as the fast ladder. *H:* the cost gap is far smaller than the historical ~388 s vs ~70 s. *Kill:* if still >3–4× slower with no accuracy edge anywhere, the controller stays a pure fallback and R7 closes here (cheap to reach).
- **R7b — Revive-then-controller.** Two-stage where stage 2 is the adaptive rollback refiner instead of the fixed stabilize, on cascaded TTFS. *H:* the adaptive refiner beats fixed-stabilize on deployed accuracy (the 0.956 vs 0.95 signal), at a measured cost. Compare on the accuracy/cost Pareto.
- **R7c — Surgical / hybrid driver (the headline experiment).** Fast ladder globally; the rollback controller invoked *only* on the layers / rate-region R1 flags as ill-conditioned (the firing-gain transition, the cliff neighborhood). *H:* this captures the controller's accuracy edge at near-fast-ladder cost — a strict Pareto improvement over both pure drivers for the hard cells. If it works, it becomes a third `OptimizationDriver` arm (`hybrid`) selected by characterization.
- **R7d — Certified non-destructiveness.** Construct/inject cases (noisy eval, a deliberately bad rung) where the fixed ladder commits a regression; show the rollback controller does not. *H:* there exist realistic configs where per-step rollback prevents a regression the ladder ships. *Deliverable:* a `controller` driver positioned as the **certified zero-regression mode** for configs that declare it.
- **R7e — Synthesis / decision.** Across the matrix, does the revived controller ever *strictly dominate* the fast ladder on accuracy × cost (× energy)? Outcomes — all useful: (i) yes for some cells → first-class driver, selected by characterization; (ii) only the hybrid (R7c) dominates → ship `hybrid` as the third arm; (iii) neither → controller remains the safety-net fallback + certified-non-destructive mode.

**Success.** A clear, deployed-metric, Pareto-grounded verdict on the controller's role — first-class / hybrid / fallback — replacing the confounded "it was slow once" demotion. Concretely, a *win* = the hybrid (R7c) or revived controller closes ≥ 1 pp of the cascaded AC1 gap that Fix B leaves, at ≤ 1.5× the fast-ladder cost on the affected cells.
**Kill.** R7a shows no path to competitive cost **and** R7b/R7c show no accuracy edge → document the controller as a pure fallback and stop. Cheap to reach: the machinery exists; these are days-not-weeks runs.
**Why give it one more chance.** It is cheap (machinery built), the one clean data point (0.956 > 0.95) is encouraging, the original demotion was confounded, and it attacks the exact gap (the AC1 residual + a non-destructive guarantee) that the fast-default leaves open.
**Connection to the action plan.** R7 is the empirical study of the `controller` arm of the `OptimizationDriver` axis that **Fix A** creates, and R7c would add a `hybrid` arm. It consumes R1 (localization) and R6 (deployed gate) and reuses the prior cost fixes. It is not nostalgia — it is a candidate path to the last 1–2 pp plus a safety mode.

### Priority order

R6 always-on; **R1 immediate** (unblocks safe defaults and everything else). Then the accuracy pair **R2 + R7** (both attack the cascaded AC1 residual; R7 is cheap and could inform R2 — run them together). Then **R3** (the chip-compiler objective; push hardest — most novel and defensible) → **R4** (generalization) → **R5** (parallel, high-ceiling). R1 and R6 are as much infra as research and start alongside Fix A.

---

## Decision Points

- **What is non-negotiable and immediate:** Fix A (unbind both axes + the uniform seam) + R1 (characterization/policy) + R6 (deployment-faithfulness). All byte-identical or pure infra.
- **What is a deliberate, gated behavior change:** Fix B, per (firing × sync × backend) cell, behind the per-cell Pareto/regression gate, lowest-risk-first. Your call how many cells to flip per release; LIF and cascaded are the obvious first two.
- **What is a research bet (with kill criteria):** R2, R3, R4, R5, R7. R7 is the cheapest bet with the most direct line to the AC1 residual and is worth running early despite being a "revival."
- **The one cross-cutting protocol to establish now:** the transition from byte-identity to a deployed-accuracy + wall-clock certification gate — without it, no behavior change (Fix B or any research recipe) can ship safely.

---

## Bottom Line

The structural refactor built the clean, auditable substrate; the work ahead is to make it *do* something by default. **Fix A** makes the optimization-driver and calibration genuinely pipeline-wide axes (not TTFS islands), with a **characterization/policy keystone** that picks the proven recipe per model and escalates safely when a model is off-distribution. **Fix B** then turns those recipes on, per cell, behind a Pareto/regression gate on the deployed metric — delivering the speed goal outright and most of the accuracy gap, with the honest caveat that the final ~1–2 pp to a clean AC1/AC2 cascaded pass is a research result. The research program closes that residual (**R2**), establishes the chip compiler's true objective (**R3** — the accuracy/energy/latency Pareto), proves it generalizes (**R4**), raises the ceiling at the source (**R5**), keeps every number deployment-faithful (**R6**), and — via **R7** — gives the original non-destructive rollback controller a clean, cheap second chance to earn back a role (first-class, hybrid, or certified-non-destructive fallback) by re-running it un-confounded, revived, deployed-anchored, cost-optimized, and applied surgically where adaptive rollback should actually beat a fixed schedule.
