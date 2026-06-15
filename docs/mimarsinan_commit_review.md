# Code Review — `tuning/` Refactor (origin/main, last 21 commits)

**Range reviewed:** `9383d1b..8ebb211` (21 commits on `origin/main`, fetched from origin)
**Scope:** changes related to Non-Destructive Gradual Model Transformation — i.e. the `tuning/` subsystem and its collaborators.
**Size:** 125 files, **+7367 / −1352**; source side **+1845 / −699** across 58 files.
**Method:** read every commit message body, diffstat, and the new/heavily-modified source modules at `8ebb211`; spot-verified live wiring of new capabilities.
**Context:** these commits implement the refactoring plan from the prior review (`mimarsinan_refactoring_report.md`). The commit messages cite its vector labels (P0–P7, W6/W8/W9, IV.x, V6, V9, spec §6.2/§8.2) directly, so this review is graded against that plan.

---

## 0. Verdict

This is an unusually disciplined refactor and it lands the structural core of the plan. The four services (`RateScheduler`, `AcceptanceSensor`, `RecoveryEngine`, `CheckpointGuard`), the `AdaptationAxis` protocol with adapters for all seven tuner families, the in-place `RateBuffer`, the golden-trace harness, and the full Part-IV controller-validation suite (mock zoo, conformance, metamorphic, determinism, chaos, perf gates) are all present, and three of them are now the **sole** code path (legacy loops deleted). The work is honest about cost: behavior-changing knobs were implemented, *ablated*, and either baked in (when <10% cost) or kept opt-in / removed (when not), with a dedicated `docs/tuning_optimization_flags.md` recording the evidence.

The single most important thing this refactor did *right* is refuse to take the plan on faith. Two of the plan's headline predictions were empirically falsified and corrected (see Part C). That is exactly the behavior the plan asked for ("be sure the generalized strategy is stable … across transformations with varying real-time effects") and could not itself guarantee on paper.

The gaps are mostly **latent-capability** issues — modules that are built and unit-tested but not yet wired into the live controller — plus one mislabeled scaling path and one self-neutralizing default. None are correctness regressions (flag-off paths are bit-exact and the goldens hold); all are addressed in Part D.

---

## Part A — Commit-to-Plan Map

| # | Commit | Plan vector | Landed |
|---|---|---|---|
| 1 | `8707b16` P0–P3a | golden harness (IV.1), `AdaptationAxis`, `AcceptanceSensor`, `CheckpointGuard` | ✅ |
| 2 | `0c3c423` P3b+P4 | `RecoveryEngine`, `RateScheduler` + flagged driver path | ✅ |
| 3 | `dc06fcc` P1 routing | route all 7 tuner families through `AdaptationAxis` (flagged, byte-identical) | ✅ |
| 4 | `5ac46bc` P2b | paired McNemar gate (flagged, MC-calibrated) — spec §6.2 | ✅ (opt-in) |
| 5 | `6eb71fd` P4 | extract `AdaptationDriver` class | ◑ (driver path; V6 dissolution deferred) |
| 6 | `ef83680` P5b/P6a | subsample val cache (W8), sensitivity stepping (V8), controller zoo (IV.2) | ✅ / later reverted |
| 7 | `93baf7c` P5c | characterization phase (V9 / spec §10) | ◑ (built+tested, not wired) |
| 8 | `71683de` P5a | in-place `RateBuffer` (W9) | ✅ |
| 9 | `c5e6467` P6/P7 | loss-slope LR (V7), interleaved multi-axis (V10b), Part-IV tests | ✅ / V10b later removed |
| 10–12 | `0b20dd7`,`9737d7b`,`b0561fb` P6 | persistent optimizer + two real-bug fixes | ✅ |
| 13 | `09eef0d` P2b fix | floor paired gate by global budget — spec §8.2 | ✅ (but see D4) |
| 14 | `7116d8b` | graduate refactor: scheduler/axis/guard become sole path; delete legacy | ✅ |
| 15 | `a460fbf` | IV.7 perf gates; document deferred V6 | ✅ |
| 16–17 | `4396939`,`bce6a06` | full-transform probe diagnostic + warning | ✅ (new, beyond plan) |
| 18 | `a49bb7f` | collapse feature flags: remove zombies, bake defaults | ✅ (−809 lines) |
| 19–20 | `0b4271f`,`ab4d1bf` | measure flag cost; bake <10% ones; remove sensitivity stepping | ✅ |
| 21 | `8ebb211` | bake in-place rate buffer with write-through (W9) | ✅ |

Legend: ✅ landed · ◑ partially landed / deferred.

---

## Part B — What Landed Well

**The service extraction is clean and faithful.** `AcceptanceSensor` (`orchestration/acceptance_sensor.py`) is a set of pure static decision functions over the budget's SE; the docstring correctly frames it as a *bit-exact* extraction of math previously inlined in `_adaptation`, with the goldens as the equivalence contract. `CheckpointGuard`, `RecoveryEngine`, and `RateScheduler` are similarly small, single-responsibility, and individually unit-tested (`test_acceptance_sensor.py`, `test_checkpoint_guard.py`, `test_recovery_engine.py`, `test_rate_scheduler.py`). This is the highest-value part of the plan (testable decision math) and it is fully realized.

**The `RateScheduler` correctly collapses three loops into one** and even fixed a real edge bug surfaced by the LIF/TTFS parity tests: the first greedy/ladder jump is always attempted and `epsilon` bounds only the bisection, so a degenerate `epsilon >= gap` no longer commits nothing (`7116d8b`). That is a genuine improvement over the legacy `_continue_to_full_rate`, not just a port.

**The `AdaptationAxis` protocol is the right shape and now the sole rate-application path.** A `runtime_checkable` `Protocol` plus an `AdaptationAxisBase` with benign defaults, adapters for all families (`manager_rate_axis`, `blend_axis`, `activation_shift_axis`, `perceptron_transform_axis`, `pruning_axis`), each declaring `interpolation_mode` / `monotonicity` / `is_stochastic`. The naming choice — `AdaptationAxis` rather than `Transformation`, to avoid colliding with the existing `transformations/` math package — is a good call I did not anticipate and is better than the plan's name in this codebase.

**The in-place `RateBuffer` (W9) is done carefully.** The decorator stack is built once and a ramp step is an O(1) buffer fill; the migration blocker (manager rate *field* going stale while the buffer holds the live value) was resolved with an explicit **write-through** so state queries and the pickled manager never see a stale rate (`8ebb211`). Critically, `test_rate_buffer.py` proves the buffer path is bit-identical to the rebuild path across a rate grid *including* the `0.0`/`1.0` short-circuits **and** identical global torch RNG state after `set_rate(0.0)+forward` — that RNG-equivalence assertion is exactly the subtlety the original code commented about, and pinning it is excellent.

**The persistent-optimizer work is the standout for engineering rigor.** It shipped as a flag, then an ablation on the real MNIST pipeline crashed in the weight-quant tuner (`GradScaler` "no inf checks") because `PerceptronTransformTrainer` replaces model parameters each step, so a held optimizer stepped stale tensors. The fix is layered and correct: `PersistentOptimizerOwner` rebuilds when the parameter-set *identity* changes (`9737d7b`), and a `_supports_persistent_optimizer=False` opt-out for param-transforming families because you *cannot* persist Adam moments across a per-step parameter transform at all (`b0561fb`). Both fixes come with regression tests. This is precisely the kind of transformation-specific pathology the plan warned the generic controller would hit.

**Flag hygiene is exemplary.** `docs/tuning_optimization_flags.md` enumerates every default-off knob with its ablation data, cost, and a recommended collapse order; `a49bb7f` then deletes three genuinely dead flags (`interleave_axes` unread, `clamp_learnable_scale` docstring-only, `per_layer_rate_schedule`) and bakes several behaviors, net −809 lines. Removing a feature you built because it is unused is rare discipline.

**Two useful diagnostics beyond the plan.** The full-transformation probe (`4396939`/`bce6a06`) answers a reviewer's real question — "is the gradual ramp actually pulling the model toward 1.0-viability, or do rate-1.0 attempts keep collapsing independently?" — by recording `committed_acc − full_acc` after each commit and warning if that drop fails to shrink. This is a thoughtful observability addition for the exact failure mode the continuation framing is supposed to prevent.

---

## Part C — Where the Plan Was Empirically Corrected (a strength)

Two of the prior report's recommendations did not survive contact with measurement, and the team was right to override them.

**C1 — Paired stats did *not* reduce cycle count; they thrashed.** The plan predicted the tighter paired (McNemar) SE would let steps grow and cut cycles. In practice the several-fold-smaller SE made a pure `k·SE` gate against the fixed baseline *over-reject* — it rolled back negligible sub-budget drift (e.g. a 0.3% drop well inside a 5% tolerance) and did ~6× more LR searches / rollback cycles (`09eef0d`). The correct fix was to require the drop be **both** statistically significant **and** practically meaningful (exceed the global budget, spec §8.2). Even so the paired path costs +139% tuning wall-clock, so it ships **opt-in**. Net: the plan's *correctness* claim held (more powerful test, real anti-drift) but its *speed* claim was inverted. The commit history documents this honestly rather than quietly shipping the slower path.

**C2 — Sensitivity-guided stepping (V8) gave no measurable benefit and was deleted.** A proper driver-on ablation across 3 seeds showed the real LR-find count was identical (4 = 4) every seed and wall-clock swung ±7% on noise with no consistent gain (`ab4d1bf`). Per the plan's own "remove it if it isn't providing a run-time benefit" instruction, both the flag and the `last_successful_step` policy were removed. Good.

**C3 — Interleaved multi-axis (V10b) was built, found unreachable, and removed** as a zombie flag (`a49bb7f`). Appropriate; it was explicitly the research-grade, highest-risk vector.

These three reversals are the most reassuring thing in the range: the controller-validation layer and ablation harness exist and were actually used to kill plausible-sounding ideas.

---

## Part D — Findings (gaps & nits surfaced by this review)

Ordered by severity. None block; all are latent-capability or labeling issues, not correctness regressions.

**D1 (Medium) — The characterization phase is built and tested but not wired into the live controller.** `orchestration/characterization.py::characterize()` is referenced *only* by its own definition and `test_characterization.py`; `grep` finds no call site in `src/`. So the monotonicity verdict, `epsilon_hint`, and dense-grid safe-mode downgrade never actually configure a real run — the controller still trusts the global `monotonicity="expected"` and a fixed `epsilon`. This matters most for `ActQuantAxis`, which is `is_stochastic=True`, `interpolation_mode="stochastic_mask"`, yet inherits `monotonicity="expected"` — exactly the axis the A1 audit was meant to police. *Recommendation:* call `characterize()` once in `AdaptationDriver` setup over a coarse α grid, feed `epsilon_hint` into the scheduler and switch to a dense grid when `monotonic` is False; gate behind a flag and add a mock-zoo test asserting a `NonMonotone` axis actually flips the live scheduler into safe mode (today only the pure function is asserted).

**D2 (Medium) — Real stochastic-axis decisions are still not reproducible; determinism is validated only at the mock level.** `set_decision_seed` exists in the protocol and as a base no-op, but **no concrete axis overrides it** — including `ActQuantAxis` and `NoiseAxis`. `test_determinism.py` lives in the mock zoo (pure-python), so invariant I6 is proven for the *controller logic* but not for the real stochastic decorators (`RandomMaskAdjustmentStrategy`, `NoisyDropout`), whose `torch.rand` calls remain unseeded per decision. *Recommendation:* implement `set_decision_seed` on the stochastic axes (snapshot/restore RNG or seed the decorator's generator around probe/confirm evals) and add a determinism test on a real stochastic axis, not just mocks. Until then, golden traces over stochastic axes are only as stable as global RNG ordering.

**D3 (Low/Medium) — `CheckpointGuard(location="cpu_pinned")` does not pin and does not overlap.** The snapshot uses `v.detach().to("cpu", copy=True)` (plain pageable CPU copy, no `pin_memory()`), and `restore` uses `v.to(device)` with no `non_blocking=True`; the only sync is a blocking `torch.cuda.synchronize()`. So the advertised async d2h/h2d overlap (the W6 latency benefit) is not realized — the label overpromises. The path is also off by default (`checkpoint_scope="full"`), so impact is latent. *Recommendation:* either implement true pinned-buffer + `non_blocking` copies, or rename to `cpu` to match behavior. The `scope="tunable"` VRAM lever (the more important half of W6) is real and works.

**D4 (Low) — `global_budget` default `0.0` neutralizes the very §8.2 fix that makes the paired gate viable.** `paired_is_rollback` floors the gate at `min_effect=self._global_budget`, but `defaults.py` sets `global_budget: 0.0`. With the floor at zero, `delta > 0.0` is true for essentially any positive drop — re-exposing the thrash `09eef0d` was written to cure. It is masked today only because the paired sensor is itself off by default. *Recommendation:* if the paired sensor is ever enabled, `global_budget` must be set > 0 (e.g. 0.005); consider making the paired-sensor flag refuse to enable with a zero budget, or default the budget to a small positive value. Also reconcile the two commit messages (one says default 0.0, one says 0.5%) — the code says 0.0.

**D5 (Low) — V6 deferred: `_adaptation` is still a ~150-line god-method.** The decision *math* is extracted into pure, testable functions (the high-value win), but the control *flow* still inlines the full sequence (clone → probe → catastrophic → LR → recover → dual/paired rollback gate → commit → target-relax → trace) in one method, now also carrying the paired-gate branch, persist-optimizer selection, and full-transform probe. The commit log is explicit that dissolving this into `AdaptationDriver._cycle` is deferred over "just-stabilized parity-critical code," which is a defensible sequencing call. *Recommendation:* keep it on the roadmap; the services now exist, so the dissolution is mechanical, and it would also remove the slight statistical-basis split noted in D6.

**D6 (Nit) — Mixed statistical basis within one cycle under the paired gate.** When `_paired_gate` is on, *rollback* uses the paired McNemar test but `reached_target` (which drives target relaxation) still uses the marginal `post_acc`. Two estimators decide two halves of the same cycle. Functionally fine, but it means the relaxation state machine and the rollback gate can momentarily disagree about whether the model is "at target." Also: under the paired gate, the cycle runs *both* a marginal `validate_n_batches` (for `post_acc`/`reached_target`) **and** a `validate_correctness_on_indices` pass — part of the +139% cost. *Recommendation:* derive `post_acc` from the paired correctness vector when the paired gate is active (one pass, one basis).

**D7 (Nit) — `CATASTROPHIC_DROP_FACTOR = 0.8` magic constant** survives (flagged minor in the prior report). Now centralized in `AcceptanceSensor.is_catastrophic`, which is the right home; deriving it from the budget SE remains a future tidy-up.

---

## Part E — Defaults Posture (is the shipped behavior the safe one?)

Mostly yes, and deliberately so:

- **On by default (baked):** subsample val cache (W8 — the genuine large-dataset scale fix, after a real seeding-bug fix to use a representative seeded reservoir), in-place rate buffer (W9), persistent optimizer (where param-stable), loss-slope LR ranking. These cleared the team's "<10% cost" bake bar with measurements.
- **Off by default (opt-in, for measured reasons):** paired sensor (+139% wall-clock), `checkpoint_scope="tunable"` / `cpu_pinned` (scale levers, unvalidated end-to-end and — see D3 — partly unimplemented), full-transform probe (diagnostic).
- **Removed:** per-layer rate, interleaved axes, sensitivity stepping, learnable clamp scale.

The risk profile is good: the safe, broadly-beneficial scale fix (subsample cache) is on; the costly or unproven ones are off. The one wrinkle is D4 (a default that quietly disables a correctness floor for an off-by-default feature).

---

## Part F — Test Coverage Assessment

Strong and well-aligned with the plan's Part IV. The range adds the golden harness + Tier-A/Tier-B equivalence goldens, the mock-axis zoo with controller-invariant tests (I1/I2, bounded probes, partial-result on recovery-limited), conformance, metamorphic, determinism, chaos-rollback, paired-sensor Monte-Carlo calibration (false-reject < 0.10 at k=2; paired SE < 0.5× marginal; 8% drop detected > 90%), and deterministic perf gates (cliff cornering is O(log(gap/ε)), greedy one-probe commit, `scope="tunable"` skips the frozen backbone). Suite counts climb cleanly across commits (372 → 815+ in places) with "no new failures" called out.

Coverage gaps that mirror Part D: (i) the controller-validation layer is pure-python/GPU-free by design, so determinism and non-monotone safe-mode are asserted on **mocks**, not on the real stochastic axes or via a live `characterize()` (D1/D2); (ii) the CUDA wall-clock / peak-VRAM budgets on the ViT probe are explicitly deferred to a future integration benchmark (acknowledged in the transformation `ARCHITECTURE.md`), so the large-*model* half of IV.7 is not yet enforced in CI; (iii) `CheckpointGuard` bitwise round-trip is asserted for `scope="full"`; the tunable/cpu path's exactness for the *retained* tensors is less visibly pinned.

---

## Part G — Recommended Next Steps (prioritized)

1. **Wire `characterize()` into the live driver** and add a live non-monotone→safe-mode test (closes D1; activates V9, which is currently latent).
2. **Implement `set_decision_seed` on the stochastic axes** and add a real-axis determinism test (closes D2; makes golden traces meaningful for `ActQuant`/`Noise`).
3. **Fix or rename `cpu_pinned`** (real `pin_memory` + `non_blocking`, or rename to `cpu`) and add a tunable-scope bitwise round-trip test (closes D3).
4. **Guard `global_budget` against the zero-floor footgun** before anyone enables the paired sensor (closes D4).
5. **Land the deferred V6 dissolution** of `_adaptation` into `AdaptationDriver._cycle` now that the services are stable, folding D6 (single eval pass, single statistical basis) in along the way.
6. **Add the CUDA integration benchmark** (wall-clock + peak VRAM on the ViT clamp probe) to make the large-model half of IV.7 a CI gate, not a note.

**One-line summary.** The structural plan landed cleanly and is now the sole code path, the scalability fixes that matter (subsample eval, in-place rate, persistent optimizer) are on by default, the empirical reversals (paired thrash, sensitivity stepping) show the validation harness is doing real work, and the remaining items are latent-capability wiring (characterization, decision-seeding) and one mislabeled path — not correctness debt.
