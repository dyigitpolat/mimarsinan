# Frontier Program — fast + lossless by default, and generalizable

North-star documents (the spec for this program): `docs/mimarsinan_final_refactor_review.md`
(the critical review of the V1–V8 substrate) and `docs/final_recommendations.md` (the
action plan Fix A / keystone / Fix B + the R1–R7 research program). This file tracks
execution and enforces the two cardinal rules.

## Two cardinal rules

1. **Only land well-grounded implementations.** The engineering track (Fix A + keystone
   infra + R6 gates + the certification protocol) is byte-identical / pure-infra and lands
   on `main` (via `frontier-program`). Behavior-changing defaults (Fix B) and research
   recipes land ONLY after they are certified on the deployed metric against a frozen
   per-cell regression floor.
2. **Keep preliminary research isolated in worktrees.** R1–R7 experiments live in
   git worktrees (`isolation: worktree`), never in `src/` on `main`, until grounded and
   promoted as a certified implementation.

## Track E — Engineering foundation (LANDS, byte-identical / infra)

The non-negotiable-immediate tier from the action plan (Fix A + R1-infra + R6), plus the
certification protocol that must exist *before* any Fix B flip.

- **E1 — Uniform rate-tuner seam.** Every rate tuner (LIF, the analytical clamp/shift/quant
  chain, the manager-rate tuners, the KD-blend tuners) exposes the same three-verb contract
  `ramp(rate)` / `recover_to(target)` / `probe()`. The driver is an axis that drives an
  `AdaptationAxis`-shaped tuner; every tuner implements that shape. (Action plan: "the real
  deliverable is the uniform rate-tuner seam," not "move the method up.")
- **E2 — Unbind `OptimizationDriver` → pipeline-wide axis** resolved by `DeploymentPlan`,
  consumed by every rate tuner via E1. LIF stops bypassing it; the analytical chain gains a
  fast path. Default `controller` ⇒ byte-identical.
- **E3 — Unbind `CalibrationPipeline` → pipeline-wide**, keyed by (firing × sync), available
  to all conversion tuners (not just TTFS). Default-off ⇒ byte-identical.
- **E4 — Characterization/policy keystone (R1 infra, default-off).** The propose → confirm →
  escalate layer: the contract proposes a recipe per mode (the prior), a cheap pre-flight
  characterization confirms it on *this* model, and an unmatched model escalates to the
  controller fallback rather than shipping a silent regression. Scaffolding only, no-op
  by default ⇒ byte-identical.
- **E5 — R6 deployment-faithfulness gates.** Institutionalize: torch↔sim parity as a gate on
  every run; all external deps pinned with version/capability guards at each boundary; drift
  detection. (Mostly built — wire + document + audit.)
- **E6 — Certification protocol (the protocol we do not yet have).** A frozen per-cell
  regression-floor format + freezing script + a Pareto/regression gate harness: *deployed
  accuracy ≥ floor − ε AND wall-clock ≤ budget*, per (firing × sync × backend). This REPLACES
  byte-identity as the gate for Fix B. Mechanism lands here; the floor is populated by a
  matrix run before the first Fix B flip.

## Track R — Research threads (ISOLATED worktrees, NOT landed until grounded)

Each scout sets up its isolated harness, runs the cheapest grounding probe, and returns a
findings + go/kill verdict + the full thread plan. Nothing touches `src/` on `main`.

- **R1 — Characterization & auto-policy** (keystone; also E4 infra). Probes: cold-cascade
  liveness, ramp monotonicity, staircase/LIF ceiling vs depth & S, firing-gain profile.
- **R2 — Close the cascaded-TTFS lossless gap** (headline accuracy). Port the artifact-51
  combo (θ-cotrain + progressive unfreeze + continuous-teacher KD, train low-S) +
  two-residual S allocation. Target deployed ≥ 0.96 @ practical S, monotone.
- **R3 — accuracy ↔ energy ↔ latency ↔ area Pareto** (the chip-compiler objective; most
  novel). Per-layer S allocation + richer codes; compiler hits a declared budget.
- **R7 — Revive the non-destructive rollback controller** (cheap, high-info). R7a honest
  re-baseline (cost-fixes on) → R7b revive-then-controller → R7c surgical/hybrid driver →
  R7d certified non-destructiveness → R7e synthesis. Adds a `hybrid` driver arm if it wins.
- **R4 — Generalization** (CIFAR / transformer / deeper) and **R5 — SNN-readiness
  preconditioning** (parallel, high ceiling) — longer-horizon; scoped after R1/R2/R7 ground.

## The transition protocol (C3 in the review)

Fix B breaks every byte-identity lock *by design*. The order is fixed: **build the cert
protocol (E6) → freeze the floor (matrix run) → swap the byte-identity gate for the per-cell
Pareto gate → flip Fix B per cell, lowest-risk-first (LIF → cascaded → analytical →
synchronized), each certified on the deployed metric before the next.** Never a blanket flip.

## Honest ceiling (from the action plan)

Fix B delivers AC5 (speed) outright and closes the bulk of the accuracy gap (cascaded
0.26 → ~0.95), but the best *transferable* cascaded recipe today lands at/just below AC1's
0.96 line. The final ~1–2 pp is a research result (R2, possibly R7), not a defaults flip.
Communicate accordingly: "fast + dramatically better, AC5 met, AC1 within reach but not
guaranteed for cascaded."
