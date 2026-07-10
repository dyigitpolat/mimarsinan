All load-bearing seams verified against the working tree (main @ 356e228f): `endpoint_recovery.py:90-92/104/116`, `tuning_policy.py:61/68`, `conversion_policy.py:41-44/67-71/88/106`, `basic_trainer_steps.py:141-189`, `generate.py` ledger/caps/validity sections, `segment_runner.py:56-97/124-128`, and the test pins (`test_ssot_flag_collapse.py:67`, `test_wq_demotion.py:161`, `test_deployment_derivation.py:206`). Campaign = 50 cells (25 tier0 + 25 tier0_1, confirmed from `test_configs/`).

# MIMARSINAN TIER0 PASS-ALL CAMPAIGN — IMPLEMENTATION PLAN

Repo root: `/home/yigit/repos/research_stuff/mimarsinan` (branch `main` @ 356e228f). All paths below are under this root unless written in full.

---

## 1. RANKED LEVER LIST

### CONFIRMED — implement (ranked by leverage x dependency position)

**C1. Convergence-stop for armed endpoint/stabilize stages** (merges ttfsq_gap "convergence-stop inside armed geometry" + wall_structure "convergence-stop, budget-scaled horizon"; both verdicts CONFIRMED). Replace `min_steps=budget` at `src/mimarsinan/tuning/orchestration/frontier/endpoint_recovery.py:116` (and the legacy stabilize at `smooth_adaptation_run.py:95,109`) with `min_steps = max(absolute lr-dip cover ~2k, fraction of budget)` + coarse keep-best patience scaled to budget. Reclaims 1.5-2.1 ks/campaign of measured zero-gain burn, defuses the latent tier1/2 full-16k-burn bomb (tier1/2 already run un-capped with patience disarmed), and is the hard prerequisite for C2/C4 and the tier0 2000-cap deletion. Mandatory calibration probe before cap deletion (below).

**C2. Arm-when-engaged — delete the SE entry-gap gate** (ttfsq_gap, CONFIRMED). `endpoint_recovery.py:90-92`: arm the funded geometry whenever `entry < target` and the ledger affords, not only when gap >= 1x SE. Flips t0_14's sterile 210-step +0.0000 branch into a real climb; removes the measured razor-edge arming nondeterminism (t01_06 armed by margin 0.000186). Worst case (sub-SE gap on a 16k family burning the full ledger) is bounded by C1 — land together.

**C3. Divergence guard + warmup + LR-backoff rescue on the floor arm** (mixer_envelope, CONFIRMED). New default-off `TuningPolicy` flag; predicate = (best never beat entry+SE after N=5 checks) OR (current < pipeline_hard_floor for M=3 consecutive checks — required to catch the observed post-progress crater); action = restore live best, rebuild optimizer, restart remaining budget at lr*0.3 with 2% warmup, cosine. Kills the measured 3.5pp identical-config coin-flip (t0_21 0.9316 vs sibling t01_24 0.9671) and the dead-16k floors (t01_05). Note the verdict's correction: this is genuinely NEW code — the existing plateau machinery is a no-op on armed stages and has the wrong semantics (constant LR, no restore, no warmup).

**C4. Grant ttfs_quantized the well-conditioned WQ floor** (mixer_envelope + ttfsq_gap, CONFIRMED twice independently). Add `**_WELL_CONDITIONED_ENDPOINT_FLOOR_KNOBS` to `_TTFS_QUANTIZED_RECIPE_KNOBS` (`conversion_policy.py:67-71`); flip the 3-4 deliberate test pins with the measured justification (proxy→deployed sub-SE: +0.0007/−0.0014/−0.0010 vs SE 0.0092). Do NOT touch `is_bit_parity_lossless_conversion` (semantically false for ttfsq). HARD dependency on C1 (0.98 sits above every envelope; without a convergence stop the floor is a full-budget burner — on tier1/2 ViT ttfsq cells at ViT step cost). Full effect on t01_06 additionally needs the Phase-4 generate.py regen (explicit config keys beat recipe knobs; tier0 pins wq=2000).

**C5. Wave-parallel Lava per-core execution** (wall_structure, CONFIRMED). `src/mimarsinan/chip_simulation/lava_loihi/segment_runner.py:97` serial loop → waves executed in a ProcessPoolExecutor. Two verdict-mandated corrections: (a) derive waves from longest-path levels of the dep graph already built at `segment_runner.py:56-65`, NEVER from `timing.core_latency` (`_align_shiftable_cores` in `src/mimarsinan/mapping/latency/chip.py:114-143` can invert latency order); (b) non-daemonic spawn workers (ProcessPoolExecutor, not multiprocessing.Pool — Lava spawns actor subprocesses; precedent: commit 356e228f). ~790s group-wide (t0_03 832→~470s, t01_19 657→~390s), value scales with tier1/2 core counts. Fully independent — parallel lane.

**C6. Exact-kernel endpoint promotion for ttfsq — CONFIRMED AS DEFERRED.** Measured non-binding (~0.1-0.2pp vs 0.3-2.4pp of denied/sterile climb). Do not implement this campaign; keep as the mbh_t6_sync_exact_kernel X4 follow-up.

### NEEDS_PROBE — run probe before any campaign spend

**P1. Hedged staircase-backward STE for cascaded rung training** (casc_collapse L1).
- **Step 0 (zero GPU, decided by cell names alone):** all 5 failing casc mmixcore cells are S<=8 (t0_16 s8, t01_03 s4, t01_10 s8, t01_12 s8, t01_13 s8) → Phase-C's measured "at S=4 it adds nothing over KD-refine" applies → **drop P1 for the mmixcore family; record in the casc ledger.** The toy-proven regime (S>=16, deep hops) survives only on the deepmlp rung cells (t01_11/t01_14/t0_19, S=16) — lenet5 (t0_17 s32 / t01_18 s16) is WQ-endpoint-bound, i.e. refuted-L2 territory, not rung territory.
- **Probe (worktree, before any campaign cells):** port `_StaircaseSteKDLoss` from `git show campaign-ws1-capabilities:src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py` (~:191) adapted to the CE-only prefix path at `prefix_ramp.make_kd_loss`; register under a NEW key `ttfs_prefix_staircase_ste` (the old `ttfs_staircase_ste` is ratchet-pinned DELETED at `tests/unit/config_schema/test_ssot_flag_collapse.py:67`) or amend COLLAPSED_KEYS with a stated reason; default OFF + flag-off byte-identical unit test. Run 2 cells x 2 seeds x 3 arms at identical step budget: A = status-quo PlainClassificationLoss, B = hedged STE mix=0.5, C = KD-refine (frozen teacher — Phase-C's measured winner, the honest control). Cells: t01_11 (worst deepmlp, 0.7652) + t01_15 (healthy casc deepcnn 0.9914, no-regression control). Instrument per-rung genuine-vs-staircase per-sample logit correlation at rung entry (one no_grad forward pair). **Accept:** B beats BOTH A and C by >2x seed-SE on rung-exit keep-best FT AND within-SE on t01_15 → promote to `_CASCADED_RECIPE_KNOBS` trial. Any other outcome: retire (third real-pipeline refutation is terminal).

**P2. Marginal-slope ledger allocation for starved post-crater endpoints** (mixer_envelope). **Config-only probe, no code:** clone t01_21, t01_08, t0_01 with `wq_endpoint_recovery_steps` 2000→8000 and `endpoint_recovery_steps` 600→2400; clone t01_05 (sync s4, keep `conversion_draws=2` to expose ledger contention). Use `pipeline_resume_from_cached_run` where caches exist. Read [MBH-ENDPOINT] + keep-best traces at 600/1k/2k/4k/8k. **SUPPORTED** iff exit(8k)−exit(2k) >= +2pp on >=2 of 3 LIF WQ endpoints AND a slope>SE stop still captures >=+1.5pp AND wall < the 15-min validity bound; **REFUTED** if <1pp (confirms the FAST-respec family-ceiling adjudication). Genericity leg: one tier1 deep_cnn d8 CIFAR cell (expect ~0pp). Any landed version must reserve the final-WQ base out of the drawable remainder (generate.py:99-105 invariant), define the draws=2 split, and keep the cosine horizon byte-identical when no draw occurs.

**P3. Pretrain keep-best headroom** (mixer_envelope, sub-lever (a) only). **Probe 1 (~45 GPU-min, standalone script):** 15 mixer headers x 3 seeds, build model+trainer exactly as PretrainingStep (recipe=None, legacy 5-epoch GradualWarmup), record per-epoch FULL-val + full-test; metric = test[argmax full-val] − test[last]. **Adopt** iff median >= +0.2pp on the envelope-bound subset (ttfsq, lif s16) and worst >= −0.1pp. **Probe 2 (1-2 GPU-h):** resume full pipelines from keep-best checkpoints (t01_06, t01_02, t0_11); verify deployed hcm tracks envelope delta ~1:1 AND no downstream retention gate (reference = raised pretrain read) flips plateaued siblings (t0_01, t01_05, t0_21) to failure. **Pre-facts (no probe):** drop sub-lever (b) as written — `training_recipe` never reaches PretrainingStep (trainer_factory recipe=None), label_smoothing=0.1 is already hardcoded in `BasicClassificationLoss` (`model_training/training_utilities.py:24`), warmup_ratio is inert under per-epoch scheduler stepping.

**P4. Eval-cadence cut on armed forwards** (wall_structure). **Part 1 (~15 min GPU):** cached-resume t01_09 at the WQ endpoint; cuda-synchronized timing of 42 train steps (bs=32) vs one `validate_n_batches(16)` read (bs=128), 5x after warmup → eval_fraction; repeat on one tier1 armed deep_cnn checkpoint. Rule: fraction >= 25% → adopt the "eval <= 15% of stage wall" multiplier (3-4x); ~13-17% → halve the predicted delta (~300-500s) and re-rank. **Part 2 (CPU):** implement the `TuningPolicy` endpoint check-interval multiplier gated on the `ledger_funded` branch ONLY (`endpoint_recovery.py:104-117`), with unit tests that (i) a non-armed stagnating endpoint still stops at exactly `_RECOVERY_PATIENCE x check_interval` (preserves the mbh_analytical_ttfs_stagnation economics) and (ii) an armed stage runs its full step budget with `floor(budget/(k x interval))` reads and unchanged keep-best restore.

### REFUTED — dropped

1. **casc L2 (hedged STE at both endpoints):** the exact gradient mechanism was swept shut on deep_cnn d6 FMNIST (12 seeds, every mix a wash or −2.3..−3.8pp) and Phase-C measured STE unable to revive collapsed cascades (flat at chance) and worse than plain KD-refine on revived ones; the gating keys are ratchet-pinned deleted.
2. **casc L3 (split-LR theta groups):** PHASE_C already measured the exact w=2e-3/theta=5e-2 split on real-pipeline mmixcore — chance from cold, and with-theta STE-refine (0.9369) LOST to no-theta KD-refine (0.9466); the ratio knob smuggles the toy's weight-LR constant onto paths where it reproduces nothing.
3. **casc L4 (dedup hop-frontier retries):** core premise, delta channel, and risk claim all fail at the cited seams; only the optional zero-GPU log-mining salvage remains.
4. **mixer plateau-adaptive SE-flat floor stop:** replayed on the actual floor traces the rule fires at 5-20% of budget and forfeits 0.6-2.8pp on 3 of 4; it re-instates the measured stagnation failure the min_steps=budget floor was built to fix (C1 is the correct replacement shape).
5. **wall crater guard (arm only if entry >= pipeline_hard_floor):** the floor ratchets DOWN with the crater so it misses 3 of its own 4 cited cells, and at the shared seam it disarms the campaign's most productive FT burns — flipping completed runs (t01_11, t0_19) into retention failures.
6. **wall parity-gated S-invariant train forward:** vacuous where gated (the bit-parity family already trains S-invariant) and ungated where profitable (LIF rate↔cycle-accurate parity does not hold at network level — `cycle_accurate_lif_forward` exists precisely because it doesn't); realizable savings ≈ 0.
7. **wall nevresim compile cache + ttfsq AQ endpoint adoption:** `build_token` is uuid4-per-run so the proposed cache never hits; binaries bake absolute weight/input paths (silent wrong-weights on reuse); the ttfsq `endpoint_recovery_steps` knob is dead behind the `sync_exact_qat_active` gate. Salvage (separate future lever): plumb the EXISTING dormant `NevresimCompileCache` after making binaries relocatable.

---

## 2. IMPLEMENTATION ORDER

Dependency graph: **C1 → {C2, C4, tier0-cap deletion}**; C3 independent (same file, ships in the same change-set); C5 fully independent; probes P1-P4 independent of everything.

**Phase 0 — worktree + tests-first (CPU, ~0.5 day).** Branch off main@356e228f. Write ALL C1-C5 unit tests first (repo discipline). Execute P1 step-0 (outcome already determined above: mmixcore dropped, P1 rescoped to deepmlp; write it into the casc group ledger).

**Phase 1 — targeted micro-probes (GPU, ~2 h, all parallel).**
- C1 dip-scaling: t0_21 NAPQ endpoint at residual budgets {2k, 4k, 8k, 16k} at lr 2e-3; record steps-to-first-new-best. Decides absolute-vs-fractional `min_steps` cover (dip measured ~1.6k absolute → the absolute floor term is expected to win).
- C4 cached-resume: t0_14 + t0_11 with the floor injected at budget 2000; t01_06 with 16000 also injected (quantifies the Phase-4 regen); t0_13 off-mixer proxy↔deployed spot-check. ~10-35 min.
- P4 part-1 timing on t01_09.

**Phase 2 — code (1-2 days).** Land C1+C2+C3+C4 as one coherent endpoint-geometry change-set (they touch the same three files); C5 in a parallel branch (chip_simulation only). Gates: `python -m pytest tests` green <=2 min, `./scripts/typecheck.sh` zero errors, flag-off/dormant paths byte-identical.

**Phase 3 — ONE validation wave adjudicating C1-C5 simultaneously (~16 runs + seeds, ~4-5 GPU-h).** The levers are separable at flag/branch level (C3 dormant unless divergence fires; C4 touches only ttfsq cells; C1/C2 change geometry only when armed), so one wave adjudicates all without confounding:

| Cells | Adjudicates | Accept |
|---|---|---|
| t0_06, t01_12, t0_09 | C1 archetypes (climb / dead-flat / capped) | t0_06 within SE of full-burn 0.9722-0.9748; t01_12 stops <= ~2x its flatten point; t0_09 exit >= 0.9686 |
| t01_23, t01_24, t01_07, t01_09 | C1 no-regression on productive floors | exit_B >= exit_A − 1x SE; group wall saving in the 1.5-2.1 ks band |
| t0_14 | C2 | sterile exit==entry gone; >= +0.2pp at <= +30s; steps_used ~400-600 |
| t0_21 x3 seeds, t01_05 x3 seeds, t01_24 (control), t01_16 (dormancy) | C3 | left tail lifts from ~0.91-0.93 toward the 0.96-0.97 sibling band; healthy replicas byte-identical (guard never fires) |
| t0_11, t01_06, t0_13 (control) | C4 | t0_11 hcm >= 0.966, t01_06 >= 0.958 (2000-budget arm), no rollback, gain survives to hcm (proxy↔deployed sub-SE) |
| t0_03, t01_19 | C5 | wall 832→~470s / 657→~390s; Loihi PROFILE delta stays +0.0000 |
| + 1 tier-1 cell (t1_06-class lif deepcnn CIFAR) | C1/C3 dormancy + genericity before default-on | no truncation of a genuine slow climb; guard dormant |

Any lever failing its row is backed out individually (flag revert / knob revert / condition revert).

**Phase 4 — respec + regen + full campaign.** After the C1 archetype row passes: edit `test_configs/generate.py` (SSOT — never hand-edit JSONs): delete the tier0 `wq_endpoint_recovery_steps=2000` cap for families that now stop by convergence (:508-521), extend the floor-carrying generation to ttfsq mixers, regenerate both matrices, re-run all 50 cells. This run doubles as the C3 default-on graduation A/B and produces the final pass-count measurement.

**Phase 5 — probe lane (parallel, spare GPU throughout Phases 2-4).** P2 (4 cloned configs), P3 probe 1+2, P1 3-arm on t01_11/t01_15, P4 part 2. Each supported probe feeds a follow-up change-set; each refuted one gets a ledger entry.

---

## 3. PER-LEVER IMPLEMENTATION DETAIL

### C1 — Convergence-stop
- **Files:** `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/frontier/endpoint_recovery.py` (:116 min_steps; patience arg at :126), `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/tuning_policy.py` (new frozen fields), `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/smooth_adaptation_run.py` (:95, :109 legacy stabilize), Phase-4: `/home/yigit/repos/research_stuff/mimarsinan/test_configs/generate.py` (:508-521). `basic_trainer_steps.py` patience machinery (:173-189) is reused unchanged.
- **Config keys:** none user-facing. `TuningPolicy.endpoint_floor_min_cover_steps` (absolute, ~2000 pending dip probe) + `endpoint_floor_patience_fraction` (0.25, i.e. patience = ceil(fraction*budget/check_interval)); `endpoint_floor_steps`/`wq_endpoint_recovery_steps` become true ceilings.
- **Tests first** (`tests/unit/tuning/test_endpoint_recovery.py` + new `test_convergence_stop.py`): flat armed trace stops at min_cover+patience and restores entry with ledger charged steps_used only; climbing trace (new best per window) runs full budget; min_cover stays absolute at small residual budgets; stop is step-deterministic; stabilize seam gets identical geometry; per-check (step, progress_acc, best_acc) trajectory appended to the mbh_endpoint event.
- **Adjudicating cells:** t0_06, t01_12, t0_09 (archetypes); t01_23, t01_24, t01_07, t01_09 (no-regression); dip probe on t0_21; one tier-1 deep_cnn CIFAR cell before default-on.

### C2 — Arm-when-engaged
- **Files:** `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/frontier/endpoint_recovery.py` (:90-92 condition; docstring contract at :67-73 updated deliberately).
- **Config keys:** none new (reads `endpoint_floor_steps`/`endpoint_floor_lr`).
- **Tests first:** engaged (entry<target) + ledger-funded → armed geometry (lr=min(pipeline, 2e-3), cosine, ledger-funded); sub-SE gap arms; composition test: sub-SE unreachable target is stopped by C1's patience (not a full-ledger burn) — if the Phase-3 worst-case read shows a full burn anyway, add the fallback cap "sub-SE gaps arm at the 600-step generic budget only, never the 16k grant"; rewrite the "otherwise bit-identical to pre-floor behavior" contract test.
- **Adjudicating cells:** t0_14 (primary), t0_09, worst-case guard on a t01_23-pattern floor cell with a sub-SE residual.

### C3 — Divergence guard + rescue
- **Files:** `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/frontier/endpoint_recovery.py` (wrap the armed `RecoveryEngine.train_to_target` call; thread `tuner._pipeline_hard_floor` — it lives on the run instance, `smooth_adaptation_run.py:209-220`, and can be None), `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/model_training/basic_trainer_steps.py` (dead-run predicate hook — new code; warmup via the existing `warmup_steps` param at :97), `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/tuning_policy.py` (`endpoint_floor_divergence_rescue: bool = False` + rescue factor/warmup fields — a separate flag, NOT the existing `recovery_lr_plateau`, whose semantics differ).
- **Config keys:** the new frozen policy flag only; default-off at landing, default-on flipped by the Phase-3/4 measured evidence.
- **Tests first:** predicate fires on a synthetic never-took-off trace AND on a post-progress crater trace; dormant on a healthy climb; restart geometry exact (restore best, rebuild optimizer, lr*0.3, warmup=max(1, 0.02*remaining), cosine over remaining); guard-off byte-identical; hard_floor=None disables disjunct (b) only.
- **Adjudicating cells:** t0_21 x3 + t01_05 x3 seeds guard-on (lift toward 0.96-0.97 band), t01_24 dormant control, t01_16/t01_15-class deepcnn dormancy; collapse-rate baseline from 6 guard-off replicas with per-check logging (this campaign's traces were lost).

### C4 — ttfsq WQ floor grant
- **Files:** `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/tuning/orchestration/conversion_policy.py` (:67-71 knobs + :117-122 rationale), pin flips: `/home/yigit/repos/research_stuff/mimarsinan/tests/unit/tuning/test_wq_demotion.py` (:161 `test_ttfs_quantized_stays_off_the_generalized_floor` → inverted with stated measured reason), `/home/yigit/repos/research_stuff/mimarsinan/tests/unit/config_schema/test_deployment_derivation.py` (:206/219-222), `/home/yigit/repos/research_stuff/mimarsinan/tests/unit/architecture/test_workload_literals.py` (:230). `normalization_aware_perceptron_quantization_tuner.py:80-87` needs no change (max of the two floor keys). `spiking_semantics.py` untouched except an audit note.
- **Config keys:** recipe-level `wq_endpoint_target_floor=0.98`, `wq_endpoint_recovery_steps=16000`; tier0 config pins (2000) win until the Phase-4 regen (deployment_derivation precedence).
- **Tests first:** ttfsq recipe row carries the floor knobs; explicit config key still beats the recipe; the flipped pins carry the sub-SE-transfer justification in their docstrings.
- **Adjudicating cells:** t0_11 (>= 0.966), t0_14 (0.962-0.966), t01_06 (16k arm, 0.958-0.963), t0_13 (off-mixer proxy↔deployed control); tiers 1/2 (t1_03/t2_02 ViT ttfsq) inherit only after C1 is default-on.

### C5 — Wave-parallel Lava
- **Files:** `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/chip_simulation/lava_loihi/segment_runner.py` (:56-97 — waves = longest-path levels over the `deps` graph at :56-65; keep the source-not-scheduled RuntimeError guard :124-128 loud), `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/chip_simulation/lava_loihi/core_lava.py` (per-core unit unchanged), `/home/yigit/repos/research_stuff/mimarsinan/src/mimarsinan/common/env.py` (worker-count accessor — never read MIMARSINAN_* directly).
- **Config keys:** one env knob (worker count), default os.cpu_count-bounded.
- **Tests first** (CPU, mock the per-core unit): wave levels = longest-path levels on a synthetic dep graph INCLUDING a shifted-core case (source raised to max(consumer)−1 with heterogeneous consumer depths must still precede its consumer); wave union == full core set, no intra-wave edges; parallel-vs-serial output equality (dict-keyed, order-independent aggregation); env knob bounding; executor is spawn-context ProcessPoolExecutor (non-daemonic).
- **Adjudicating cells:** t0_03 (832→~470s), t01_19 (657→~390s); gate: Loihi PROFILE delta = +0.0000 exactly.

---

## 4. PREDICTED FINAL PASS COUNT (all confirmed levers landed, probes not yet)

Assumption: pass = retention gates green AND deployed acceptance read >= 0.97 (N=100) AND wall inside the validity bound (<5 min clean / >15 min invalid, `generate.py`). Baseline on the 50-cell matrix: ~24-26 green (7 hard retention FAILs — 5 casc mmixcore t0_16/t01_03/t01_10/t01_12/t01_13 + 2 casc lenet5 t0_17/t01_18; 4 sub-0.90 completes t0_19/t0_20/t01_11/t01_14; ~14-15 band cells at 0.91-0.969).

- **Flips (high confidence):** t0_09 (0.9686 → 0.972-0.975 via C1 + cap relief).
- **Flips (coin-flip, 40-60%):** t0_11 (C4+regen → ~0.966-0.970), t0_21 (C3 → sibling band 0.964-0.969), t01_01 (cap relief → ~0.96-0.97).
- **Stabilized at-bar (variance kill, not a mean shift):** t01_23, t01_07, t01_09, t01_24 — C3 removes the 3.5pp left tail that currently makes these coin-flips per wave.
- **Materially improved but still sub-bar:** t01_05 (+4-5pp → ~0.95-0.96), t0_14/t01_06 (+0.3-1pp, e2-envelope-capped), t0_01/t01_08/t01_21 (+1-3pp), casc dead floors (wall only).
- **Wall:** t0_03/t01_19 out of the soft/invalid zone; ~2.3-2.9 ks total campaign wall reclaimed (C1 1.5-2.1 ks + C5 ~0.8 ks).
- **Hard retention failures flipped: 0 of 7** — every casc accuracy lever was refuted or probe-gated.

**Predicted: 27-30 / 50 green (median 28)** — up from ~24-26, plus a materially narrower variance band on the ~5 at-bar cells. Under a retention-only pass reading the count stays 43/50 (confirmed levers flip no retention failure; they buy accuracy band, variance, and wall).

**Pass-all is NOT reachable from confirmed levers.** The residual decomposes exactly:
1. **11 casc cells** (7 fails + 4 low): needs a NEW lever family — per the L2/L3 refutation guidance, value-domain revive (data-grounded theta revive / distribution matching) feeding the now-well-conditioned endpoint window; P1 covers only the deepmlp S=16 rung subset (+2 cells at its optimistic end).
2. **~6-8 mixer/ttfsq envelope cells** (0.95-0.966): bound by the e2 pretrain envelope / ~0.97 fbu asymptote → P3 keep-best (+0.2-1pp) and/or the e2→e4 axis respec (a cell redefinition — needs user sign-off, precedent M1).
3. **3-4 LIF S=4 crater cells:** P2 outcome (+0.5-1.5pp realistic per the on-disk slope evidence, +2-4pp claimed).

Optimistic ceiling if all four probes land at their accept bars: ~32-36 / 50. The remaining gap to 50/50 is owned by the casc mmixcore/lenet5 family, for which no surviving lever exists — commissioning the value-domain-revive lever is the single highest-value next research action after this plan ships.