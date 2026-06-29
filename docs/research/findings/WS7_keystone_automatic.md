> ⚠️ **VALIDITY:** deep_mlp results below are an **INVALID host-majority config** (<50% params on-chip; see [VALIDITY_AUDIT.md](VALIDITY_AUDIT.md)). The phenomena may be real but are NOT valid on-chip deployments. Valid trainable-deep vehicle = **deep_cnn**.

# WS7 — The Keystone: Automatic Recipe Selection (propose → confirm → escalate)

**Status (2026-06-24, ESCALATION RE-TEST HARVESTED): the clean re-test of §7 has now run
end-to-end on the trainable, off-distribution, firing-gain-degraded `deep_mlp d8 w64
cascaded` cell (3 seeds, `conversion_policy:true`). HEADLINE: with the keystone on, the
cell deploys 0.9452 (3-seed mean) vs the WS3 fast-ladder baseline 0.8754 — a +7.0pp
RESCUE of the cascaded firing-gain deficit. BUT the ablation (§0.2) proves the rescue is
the CONTROLLER driver, not the keystone or an escalation per se: the keystone's only effect
is to route this cell to `driver=controller` (`propose_recipe` always proposes
`controller`), which vetoes the fast 5-rung ladder and runs the controller's 8-rung
gradual ramp + adaptive post-finalize recovery — and THAT recovery (finalize-cliff ~0.45 →
~0.94) is what closes the gap. The earlier 3-cell validation (§4–§6, `c_keystone_*`)
remains the record for the keep-fast direction.** Read §0 first; it supersedes the
"escalation unvalidated" framing for the d8 cell.

---

## 0. The escalation re-test (the §7 recommendation, executed)

Runs: `runs/campaign/logs/ws7esc_deepmlp_d8_cascaded_s{0,1,2}.log` and the controls
`runs/campaign/logs/{cp_d8_MNIST,ws7esc_nb_MNIST}_DataProvider_cp{False,True}_s{0,1,2}.log`.
Ledger: `cluster:"WS7"`, `kind:"escalation_result"` (3 records).

### 0.1 The headline numbers (all MNIST `deep_mlp d8 w64 cascaded`, S=4, 3-seed mean)

| run group | `conversion_policy` | `ttfs_blend_fast` | driver run | TTFS ladder | deployed (HCM) |
|---|---|---|---|---|---|
| `ws3_depth_mnist_d8_cascaded` (WS3 baseline) | false | true | **fast** | 5-rung [.5→1.0] | **0.8754** (.863/.887/.877) |
| `cp_d8_MNIST` cpFalse | false | true | **fast** | 5-rung | **0.8782** (.873/.887/.875) |
| `cp_d8_MNIST` cpTrue | **true** | true | **controller** | 8-rung [.125→1.0] | **0.9418** (.945/.938/.943) |
| **`ws7esc_deepmlp_d8_cascaded`** (= cpTrue) | **true** | true | **controller** | 8-rung | **0.9452** (.947/.939/.950) |
| `ws7esc_nb_MNIST` cpFalse (no-blend) | false | **false** | **controller** | 8-rung | **0.9396** (.937/.940/.942) |
| `ws7esc_nb_MNIST` cpTrue (no-blend) | true | **false** | **controller** | 8-rung | **0.9417** (.945/.939/.941) |

ANN ref 0.977. NF↔SCM cascaded agreement 1.0 and torch↔sim parity 1.0 on every keystone
run (the deployed metric is faithful, not a sim artifact).

### 0.2 What actually happened — the keystone routes to the controller; the controller rescues

1. **The keystone DOES rescue the cell: 0.8754 → 0.9452 (+7.0pp), reproduced over 3 seeds.**
   `conversion_policy:true` on the exact WS3-degraded d8 cell closes ~65% of the 10.8pp
   ANN gap. This is a real, end-to-end deployment win on a trainable off-distribution cell.

2. **The mechanism is `driver=controller`, NOT a "match→fast" nor a literal escalation.**
   In the CURRENT code, `propose_recipe` (`conversion_policy.py:225`) **always** proposes
   `driver=OPTIMIZATION_DRIVER_CONTROLLER`. So an *enabled* decision — whether it MATCHes
   or ESCALATEs — sets `axis=controller` in `TtfsAdaptationPlan.resolve` (EF3), which
   `fast_enabled = (axis==fast)` → `False` → the fast 5-rung ladder is vetoed and the
   controller's 8-rung gradual ramp runs. The TTFS trace is therefore: 8-rung commit
   ladder (rate 0.125→1.0) → **`finalize_cliff` 0.42–0.54** (installing the genuine cascade
   forward drops val to ~0.42–0.54 — the firing-gain cliff made visible) → **~270–300s
   adaptive recovery** (LR varies 3e-3→1.5e-3→5.8e-4, rollbacks, multi-probe) back to
   ~0.94. That post-finalize stabilization is the controller's, and it is what recovers the
   cliff.

3. **The no-blend ablation ISOLATES the lever: the rescue is the controller, not the
   keystone.** Set `ttfs_blend_fast:false` (fast ladder off in *both* arms): cpFalse (no
   keystone) deploys **0.9396** and cpTrue (keystone) **0.9417** — indistinguishable
   (+0.21pp, within seed sd). The controller alone reaches the rescued accuracy *without*
   the policy. The keystone adds **no accuracy beyond what `driver=controller` already
   delivers**; its sole contribution is to *select* the controller automatically (so a user
   who left `ttfs_blend_fast:true` still gets the controller instead of the lossy fast
   ladder).

### 0.3 Did it ESCALATE or MATCH? — not separable from the log, and it does not matter here

Because `propose_recipe` proposes `controller` for both branches, **MATCH and ESCALATE
both yield `driver=controller`** on this cell — they leave the *same* trace. The keystone
emits no `escalated`/`escalation_reason` to the run log (the decision is not printed), so
the branch cannot be read off the log. To distinguish them on a future run, log
`decision.escalated` + `decision.characterization.probes` (the §7.1 instrumentation, still
not added). What this re-test *does* settle: the keystone's **routing-to-controller path
works end-to-end and recovers accuracy** on a trainable off-distribution cell with a real
firing-gain deficit — the §7 hypothesis in its actionable form.

### 0.4 The WS3 "contradiction" is resolved — it was a fast-vs-controller driver artifact

WS3's d8 cascaded 0.872 used the **fast 5-rung ladder** (no post-finalize recovery; the WQ
step commits at ~0.87 right off the finalize cliff). WS7esc routes to the **controller**
(8-rung + ~300s recovery) → 0.945. Same model, same data, same S — **different driver**.
The cascaded firing-gain deficit is *recoverable by the controller's adaptive
stabilization* on this trainable, non-dead d8 cell — exactly WS3 §4.2's prediction (a
0.87, gradient-bearing deficit is recoverable, not gone). **Actionable: for cascaded
`deep_mlp`, the controller — not the fast ladder — is the correct default, and the keystone
selects it automatically.** (Corroboration: `ws7esc_nb_mmix cpTrue` deploys 0.962 8-rung;
FMNIST nb cpFalse 0.800 ≈ cpTrue 0.805 — same controller==controller no-op when blend off.)

> ⚠️ **§3 below is OUTDATED on the driver labels.** It calls the 8-rung [0.125→1.0] ladder
> the "MATCH → fast" signature, from an earlier model where the proposal was `fast`. In the
> code as run, the 8-rung gradual ladder is the **CONTROLLER** (the keystone proposes
> controller), and the **fast** `_fast_rate_attempt` is the 5-rung [.5,.75,.9,.97,1.0]
> ladder seen only with `conversion_policy:false` + `ttfs_blend_fast:true`. Read §0.2 for
> the corrected discriminator. The §4–§6 `c_keystone_*` keep-fast verdicts are unaffected
> by this label fix (those cells deploy ~0.96 either way).

### 0.5 No-blend cpTrue-vs-cpFalse, both datasets, 3 seeds (`__target_metric.json` basis, 2026-06-24)

A clean 12-run no-blend batch (`ttfs_blend_fast:false`) re-reads the §0.2/§0.3 ablation
from the bare `__target_metric.json` deployed float (the per-item reporting convention),
across **both** datasets, 3 seeds/arm. All 12 runs finalized `rc=0`; NF↔SCM cascaded
agreement and torch↔sim parity = 1.0 on every run (deployed metric faithful).

| dataset | arm | deployed (3-seed mean) | seeds | ANN | sync ceiling | cp lift |
|:--------|:----|-----------------------:|:------|----:|-------------:|--------:|
| mnist  | cpFalse | **0.945** | .941/.953/.941 | 0.977 | 0.964 | — |
| mnist  | cpTrue  | **0.948** | .955/.947/.942 | 0.977 | 0.964 | **+0.30pp** |
| fmnist | cpFalse | **0.7943** | .774/.818/.791 | 0.883 | 0.857 | — |
| fmnist | cpTrue  | **0.8123** | .810/.819/.808 | 0.883 | 0.857 | **+1.80pp** |

**Verdict — REAL-BUT-SMALL cp lift; ESCALATION still NOT isolable (controller==controller).**
With the fast 5-rung ladder vetoed in *both* arms, cpFalse and cpTrue **both** run the
controller's 8-rung gradual ladder [0.125→1.0] + finalize cliff (~0.47–0.49 MNIST,
~0.51–0.65 FMNIST) + ~270–350s adaptive post-finalize recovery. The FMNIST cpFalse trace
shows *explicit* rate exploration (rate 1.0 rollback → 0.5 rollback → 0.25 commit → 1.0
rollback → 0.625 rollback → 0.4375 commit → 1.0 commit @ lr 4.17e-3 — **not** a flat-lr
all-commit ladder), confirming the genuine adaptive search runs in both arms. The cp lift
is therefore a **controller-vs-controller residual**: +0.30pp on MNIST (within ~1.2pp seed
sd — **not significant**) and +1.80pp on FMNIST (real but small). The keystone's measurable
contribution is **routing** (auto-selecting the controller, vetoing the lossy fast ladder),
not extra accuracy beyond what the controller already delivers — consistent with the §0.2
no-blend ablation (+0.21pp) and the prior `escalation_result` ledger record. ESCALATE
remains **inseparable** from MATCH on this cell: `propose_recipe` always proposes
`driver=controller`, so both branches leave the same controller trace.

**Confounds.** (1) **Deployed basis:** `__target_metric.json` (used here per convention)
reads ~1pp **above** the in-log HardCore sim line (e.g. MNIST cpTrue s0 = 0.955 vs in-log
0.945) and the §0.1 ledger means (0.9396/0.9417); read **gaps**, not third decimals.
(2) `max_simulation_samples=1000` → individual deployed values carry ~±1pp sampling
granularity. (3) **No paired synchronized run** in this batch — both prefixes are cascaded
(cp true/false); the sync ceiling (MNIST 0.964, FMNIST 0.857) is the **stated** reference,
not a finalized run here (`synchronized_run_ids` empty in the ledger record). (4) **Not a
chance/untrained artifact:** ANN healthy on both datasets (MNIST 0.977, FMNIST 0.883),
parity 1.0 — a genuine firing-gain/recovery result. The larger FMNIST residual ANN gap
(7pp vs MNIST's 3pp) shows the FMNIST cascade is genuinely harder, but it is still recovered
far above the implied WS3 fast-ladder ~0.77 floor. Runs:
`ws7esc_nb_{MNIST,FashionMNIST}_DataProvider_cp{True,False}_s{0,1,2}`. Ledger:
`cluster:"WS7"`, `kind:"escalation"`, `model:"deep_mlp"`, `depth:8`.

---

### 0.6 The clean negative control — `conversion_policy` is a *no-op* on a near-lossless VALID `lenet5` cascade (2026-06-24)

§0.1–0.5 established the cp lift on the firing-gain-deficient `deep_mlp d8` (an
INVALID host-majority vehicle). The complementary prediction: if the lift is
**firing-gain-deficit-specific** (not a blanket accuracy boost), it must *vanish*
where the cascade is already near-lossless. This is the negative control, run on the
**VALID on-chip-majority `lenet5`** CNN (3 seeds/arm, `max_simulation_samples=1000`,
`ttfs_cycle_based` S=4; all 12 runs `rc=0`, artifact_ok).

| dataset | cpFalse (3-seed mean) | cpTrue | cp lift | cpFalse→ANN gap | deficit regime |
|:--------|----------------------:|-------:|--------:|----------------:|:---------------|
| mnist  | **0.9847** (.984/.983/.987) | **0.9840** (.978/.986/.988) | **−0.07pp** | −0.63pp (near-lossless) | none |
| fashion_mnist | **0.8460** (.834/.852/.852) | **0.8577** (.865/.852/.856) | **+1.17pp** | −7.09pp (mild) | mild |
| *(ref)* deep_mlp d8 mnist | 0.887 | 0.948 | **+6.1pp** | (host-majority, INVALID) | severe |
| *(ref)* deep_mlp d8 fmnist | 0.756 | 0.858 | **+10.2pp** | (host-majority, INVALID) | severe |

**Verdict — CONFIRMED: `conversion_policy` is a deficit-proportional lever, not a
blanket boost.** On **MNIST/lenet5** (cascade already within 0.63pp of the 0.9910 ANN)
cp produces a **−0.07pp** change — a tiny *regression*, i.e. a clean MATCH no-op
exactly where predicted. On **FashionMNIST/lenet5** (only mildly degraded, −7pp vs the
0.9169 ANN) cp lifts **+1.17pp** — real but an **order of magnitude below** the
+6.1pp / +10.2pp rescue it produced on the firing-gain-deficient deep_mlp d8 cells.
The lift **scales with the deficit** (none → none, mild → small, severe → large),
confirming the WS7 controller rescue is firing-gain-deficit-SPECIFIC.

**Confounds.** (1) This is a `conversion_policy` true-vs-false control, **NOT** a
cascaded-vs-synchronized pairing — all 12 runs are cascaded (`ttfs_cycle_based`,
`target_tq=4`, S=4); in the ledger the `cascaded_to_sync_gap_pp` column is repurposed
as `cp_lift = cpTrue − cpFalse`, and `synchronized_run_ids` holds the cpTrue ids.
(2) `max_simulation_samples=1000` → ~±1pp granularity; read the lift, not third
decimals (cpFalse MNIST seed spread .983–.987). (3) ANN refs healthy (MNIST ~0.991,
FMNIST ~0.917) → genuine deployment, not an untrained/chance artifact. (4) The deep_mlp
d8 cp baselines are quoted from the §0 records (those cells separately ledgered as
INVALID host-majority); `lenet5` is the VALID vehicle for this control. Runs:
`cp_lenet_{MNIST,FashionMNIST}_DataProvider_cp{False,True}_s{0,1,2}`. Ledger:
`cluster:"WS7"`, `kind:"escalation"`, `model:"lenet5"`, `depth:"cnn"`.

---

### 0.7 The in-distribution VALID counterpart — removing `blend_fast` leaves the raw `mlp_mixer_core` cascade ROBUST (2026-06-24)

§0.1–0.6 stress the keystone where the cascade is *deficient* (`deep_mlp d8`, INVALID)
or *near-lossless* (`lenet5`, VALID). This row tests the **in-distribution VALID
vehicle the keystone thresholds were calibrated on** — `mlp_mixer_core`, MNIST,
cascaded, S=4 — by removing the `blend_fast` revive→refine recovery
(`ttfs_blend_fast:false`, `conversion_policy:true`) and asking whether the raw cascade
survives *without* that recovery. 6 runs, 3 seeds/arm, all `rc=0`,
`max_simulation_samples=1000`. Ledger: `cluster:"WS7"`, `kind:"escalation"`,
`model:"mlp_mixer_core"`.

| arm | `ttfs_blend_fast` | `conversion_policy` | deployed (3-seed mean) | seeds | ANN | parity |
|:----|:------------------|:--------------------|-----------------------:|:------|----:|:-------|
| blend-OFF (cpTrue) | **false** | true | **0.9547** | .954/.956/.954 | 0.9832 | NF↔SCM 1.0, torch↔sim 1.0 |
| blend-ON baseline (`pm_casc`, cp None) | **true** | None | **0.9523** | .942/.959/.956 | 0.9834 | — |
| **DELTA** (blend_off − blend_on) | | | **+0.23pp** | | | |

**Verdict — MATCH / robust cascade, NO escalation needed.** Removing the `blend_fast`
recovery deploys **0.9547** vs the blend-on baseline **0.9523** — a **FLAT +0.23pp,
within ~1.2pp seed sd** (the blend-on spread 0.942–0.959 is *wider* than the cross-arm
gap). The raw `mlp_mixer_core` cascade therefore **survives architecturally**: its
cascade robustness is *not* dependent on the `blend_fast` revive→refine recovery, so
the keystone MATCHes and **no controller ESCALATE is required** on this in-distribution
cell. This is the in-distribution VALID-vehicle counterpart to the §0.1–0.4 `deep_mlp`
INVALID escalation re-test (where the cascade *did* need the controller's recovery):
exactly where the cascade is healthy, the recovery is a no-op.

**Confounds.** (1) **NOT a cascaded-vs-synchronized pairing** — both arms are cascaded
(`ttfs_cycle_based`, `target_tq=4`, S=4); no finalized synchronized run exists in this
batch, so in the ledger `synchronized_run_ids` holds the blend-ON baseline ids and
`cascaded_to_sync_gap_pp` is repurposed as `blend_off_minus_blend_on = +0.23pp`.
(2) `max_simulation_samples=1000` → ~±1pp sampling granularity; read the gap, not third
decimals. (3) **Not a chance/untrained artifact** — ANN healthy (blend-off 0.9832,
baseline 0.9834), NF↔SCM cascaded agreement 1.0 and torch↔sim parity 1.0 on all 3
blend-off runs ⇒ the deployed metric is faithful, a genuine firing-gain/recovery
result. (4) **ESCALATE-vs-MATCH still not separable** — `propose_recipe` always
proposes `driver=controller` and the decision (`escalated`/`escalation_reason`) is not
printed; the FLAT delta only shows the `blend_fast` recovery is *unnecessary* on this
in-distribution cell (consistent with the §0.2 `deep_mlp` no-blend ablation:
controller==controller no-op when blend off). (5) baseline cp is None (default-off) vs
blend-off cp=True, but the no-blend ablation shows cp is a near no-op when `blend_fast`
is off, so the +0.23pp reflects the `blend_fast` removal, not cp. Runs:
`ws7esc_nb_mmix_cpTrue_s{0,1,2}` (blend-off), `pm_casc_mmix_mnist_s{0,1,2}` (blend-on
baseline).

---

### 0.8 The mmixcore keystone OFF-MNIST — cascade survival is dataset-stable on FMNIST but DEGRADES on KMNIST (`item_id=mmix_blendoff_dataset_axis`, 2026-06-25)

§0.7 established the in-distribution VALID `mlp_mixer_core` keystone MATCH on **MNIST**
(blend-OFF cascade survives, +0.23pp FLAT vs blend-ON, no escalation needed). This row
extends that exact blend-OFF / cpTrue cascade to the **off-MNIST dataset axis**
(FashionMNIST, KMNIST) to ask whether the keystone's near-lossless MNIST survival is
dataset-stable. `mlp_mixer_core`, `ttfs_cycle_based`, `ttfs_cycle_schedule=cascaded`,
`ttfs_blend_fast=false`, `conversion_policy=true`, S=4/`target_tq=4`,
`max_simulation_samples=1000`. 6 runs, 3 seeds/dataset, all `rc=0`. Ledger:
`cluster:"WS7"`, `kind:"escalation"`, `item_id:"mmix_blendoff_dataset_axis"` (2 rows).

| dataset | deployed (3-seed mean) | seeds | ANN | ANN gap | parity | verdict |
|:--------|-----------------------:|:------|----:|--------:|:-------|:--------|
| MNIST (§0.7 ref) | 0.9547 | .954/.956/.954 | 0.9832 | −2.85pp | 1.0 / 1.0 | keystone MATCH (in-distribution) |
| **FashionMNIST** | **0.8547** | .870/.858/.836 | 0.8871 | **−3.25pp** | NF↔SCM 1.0, torch↔sim 1.0 | **MATCH / robust off-MNIST** |
| **KMNIST** | **0.8067** | .815/.798/.807 | 0.9020 | **−9.53pp** (~3× FMNIST) | NF↔SCM 1.0, torch↔sim 1.0 | **DEGRADE off-MNIST** |

**Verdict — BOUNDED-DEGRADE off-MNIST; the §0.7 keystone MATCH holds on FashionMNIST but
NOT on KMNIST.** The blend-OFF cascade survives **architecturally robust on FashionMNIST**
(−3.25pp ANN gap, close to the MNIST keystone's −2.85pp), but on the **harder KMNIST
distribution it opens a real −9.53pp ANN gap (~3× FMNIST)** — the cascade survives
dataset-stably on the easy/mild corner but degrades where the distribution is harder. This
**bounds the §0.7 "no escalation needed" verdict to MNIST + FashionMNIST**: it does not
extend to KMNIST, where a genuine firing-gain deficit appears that the blend-OFF raw
cascade does not survive.

**Confounds.** (1) **NOT a cascaded-vs-synchronized nor a cp true-vs-false pairing** — all 6
finalized runs are cascaded blend-OFF cpTrue (only `pmmixnb_*` exist; **no** blend-ON
`pm_casc_mmix` and **no** cpFalse counterpart on FMNIST/KMNIST), so in the ledger
`cascaded_to_sync_gap_pp` is **repurposed as the ANN gap** (`deployed_mean − ann_mean`) and
`synchronized_deployed_mean`/`synchronized_run_ids` are null/empty. (2) **Not a
chance/untrained artifact** — ANN refs healthy (FMNIST 0.8871, KMNIST 0.9020, both ≫ 0.10
chance) ⇒ genuine firing-gain/deployment result. (3) **Deployed metric faithful** — NF↔SCM
cascaded decision agreement 1.0 and torch↔deployed-sim parity 1.0 on **all 6** runs ⇒ not a
sim artifact. (4) `max_simulation_samples=1000` → ~±1pp granularity; read the ANN gaps not
third decimals (FMNIST seed spread .836–.870 ≈ 3.4pp). (5) All 6 runs `rc=0`, artifact_ok,
3 seeds each. (6) **ESCALATE-vs-MATCH still not separable** — `propose_recipe` always
proposes `controller` and `escalation_reason` is not printed; the KMNIST −9.53pp gap shows
the cascade IS deficient there but **cannot prove** whether the absent controller/blend
recovery would close it. Runs: `pmmixnb_{FashionMNIST,KMNIST}_DataProvider_cpTrue_s{0,1,2}`.

---

## 1. What the keystone is

The keystone is the E4 **CHARACTERIZE → CONFIRM → ESCALATE** layer that picks, per cell,
*who drives the rate ramp* during TTFS-cycle fine-tuning:

- **propose** — `propose_recipe(mode_policy)` proposes the proven fast/lossless recipe
  (`driver=fast`, the fixed rate ladder) for cells where it is expected to hold.
- **confirm** — `CascadeCharacterizer.characterize(model, recipe)` runs four forward-only
  probes on the actual model and returns `matches: bool`.
- **escalate** — on a mismatch, `escalate_to_controller(recipe)` rewrites the recipe to
  `driver=controller` (the adaptive SmoothAdaptation rollback machinery), marking it
  `escalated=True` for provenance.

Wiring (all grounded in source):

- Decision logic + thresholds:
  `src/mimarsinan/tuning/orchestration/characterization.py`
- Policy / propose / escalate / `ConversionDecision`:
  `src/mimarsinan/tuning/orchestration/conversion_policy.py`
- Consumed at `_configure` in
  `src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py`
  (`contract.conversion_policy(... characterizer=_conversion_characterizer_for(self))`,
  ~line 374). The characterizer is built **only** when `conversion_policy:true`
  (`_conversion_characterizer_for`, ~line 92); default-off ⇒ `None` ⇒ the layer is inert
  and names the current controller behavior ⇒ byte-identical.
- The keystone's chosen driver feeds `TtfsAdaptationPlan.resolve` (~line 138 of
  `ttfs_adaptation_plan.py`): an `enabled` decision overrides the optimization-driver
  axis, so `driver=controller` (an escalation) vetoes the fast fork exactly like an
  explicit `controller` axis, and `driver=fast` (a match) lets the fixed-ladder fast
  attempt run. **This is how a decision becomes observable in the log** (§3).

### The CONFIRM verdict (the per-cell decision)

`CascadeCharacterizer.characterize` (characterization.py:314) returns `matches=True`
(keep the **fast** driver) iff **all** of:

- `cold_cascade_live` — ≥ `_LIVE_DEPTH_FRACTION` (0.75) of depths *and* the output layer
  decode a non-trivial value (`> _LIVENESS_FLOOR = 1e-4`). Catches the death-cascade.
- `ramp_monotone` — the output decode grows monotone-non-decreasing across S∈{4,8,16,32}
  (no resolution cliff), via the shared `characterize(... budget=_RAMP_BUDGET=0.02)`.
- `firing_gain ≥ _FIRING_GAIN_FLOOR` (0.1) — deepest-layer
  `genuine_decoded / staircase_decoded`. Near 1 = faithful; → 0 = the firing-gain
  collapse that is the off-distribution signature.

Any failing probe pushes `matches=False` → ESCALATE, with `reason` carrying the human
explanation ("cold cascade dead at depth" / "rate ramp non-monotone" /
"firing gain collapsed (…)").

---

## 2. The dtype crash and its fix (resolved, landed in HEAD)

`CascadeCharacterizer._staircase_means` (characterization.py, ~line 208) calls
`model.double().eval()` for the float64 staircase probe. It originally never restored the
dtype, leaving the model in float64 so the next pipeline forward crashed with
`Input type (c10::Half) and bias type (double)`. Because the keystone runs the
characterizer in `_configure` *before* baseline calibration, `conversion_policy:true`
triggered the leak on every run. The fix captures `orig_dtype` and restores it via
`model.to(orig_dtype)` in a `finally` block that also covers the unrunnable-model path
(`except Exception: means = {}` ⇒ an unrunnable off-distribution model reads as
no-staircase ⇒ the OFF-distribution verdict, never a crash). The fix is in HEAD with
reproduction tests in `tests/unit/tuning/test_characterization.py`. **All three
validation runs got past this — the crash is gone.**

---

## 3. How a decision is read off the log (no verbatim decision is logged)

The keystone does **not** print `ConversionDecision` / `escalated` / the probe readings
to the run log. The decision is inferred from the TTFS-cycle tuner's *driver behavior*,
which is unambiguous:

- **MATCH → fast** shows the **fixed-ladder fast attempt** (`_fast_rate_attempt`): an
  8-rung "Cycle summary" with `rate=0.125 … 1.0`, **all `outcome=commit`**, a **single
  spanning-cosine LR held flat across every rung**, and sub-second-to-few-second rung
  elapsed times. No LR-find probes, no rollback, no recovery-to-target.
- **ESCALATE → controller** would instead show the adaptive controller: per-rung LR
  search (varying `lr=`), `outcome=rollback`/`recover`, `continue_to_full_rate`, and the
  rate-search/rollback trace — *not* a clean monotone commit ladder.

This is a reliable discriminator: the two drivers leave categorically different traces.

---

## 4. What the three runs actually decided

| cell | distribution | model | S | expected | **observed** | deployed | ANN | escalated? |
|---|---|---|---|---|---|---|---|---|
| `c_keystone_matrix_6_cascaded` | in | mlp_mixer_core (has_bias) | 4 | match→fast | **match→fast** | 0.96 (HCM 0.9567) | 0.9815 | no |
| `c_keystone_matrix_8_cascaded` | near | mlp_mixer_core (no-bias) | 4 | match→fast | **match→fast** | 0.9606 | 0.9833 | no |
| `c_keystone_deep_mlp_d16_cascaded` | off | deep_mlp d16 w128 | 4 | **escalate→ctrl** | **match→fast** | 0.10 (HCM 0.1135) | 0.1135 | no |

**All three MATCHed and kept the fast driver. None escalated.** Evidence per cell:

- **matrix_6 (in-distribution):** TTFS "Cycle summary (8 cycles)" rate=0.125→1.0, every
  rung `outcome=commit`, flat `lr=5.79e-4`, rungs 0.6–5.0 s. post_acc 0.982→0.955 across
  the ladder. Then a separate `NormalizationAwarePerceptronQuantizationTuner` (WQ) step
  and Hard Core Mapping (HCM spiking sim 0.9567). The keystone consultation happens **only
  at the TTFS step** (the second tuner is WQ, not a controller escalation). **Correct
  MATCH.**
- **matrix_8 (near-distribution, no-bias):** identical fast-ladder signature (post_acc
  0.9655 @ rate=1.0, flat `lr=1.55e-3`); NF↔SCM cascaded decision agreement 1.0,
  torch↔deployed-sim parity 1.0; deployed **0.9606**. **Correct MATCH.** The `rc=1` is a
  **downstream Hard-Core packing failure** —
  `RuntimeError: Hard-core packing failed … No more hard cores available`
  (`mapping/packing/hybrid_segment.py:118`, the no-bias variant's 576 softcores exhaust
  the core pool) — **not** a keystone/tuning fault. The deployed 0.9606 was measured at
  SoftCoreMapping *before* the crash. The keystone decision was clean.
- **deep_mlp d16 (off-distribution):** **the same fast-ladder trace** — rate=0.125→1.0,
  all `outcome=commit`, flat `lr=1.12e-2`, post_acc **0.1052 throughout**. This is the
  fast driver, *not* the controller. The 0.10 deployment is the **ANN training-floor**
  (the ANN itself is at chance 0.1135), not a decode collapse the keystone could have
  caught. **The keystone did NOT escalate.** See §6.

**Auto-selection on the MATCH path is demonstrated on two real cells** (in-distribution
matrix_6 and near-distribution no-bias matrix_8): the propose→confirm→keep-fast loop fires
correctly, keeps the fast driver, and deploys at the expected accuracy across the no-bias
boundary. That is a genuine, end-to-end win for the keep-fast direction.

---

## 5. Is the escalation path exercised? No.

**The ESCALATE → controller path was never taken on any of the three runs.** It is
exercised only in unit tests (`tests/unit/tuning/test_conversion_policy_escalation.py`),
which build an *artificially dead* cascade (`_build_cascade(dead=True)` prunes ~92% of
every layer's weights) and assert `cold_cascade_live is False` and `firing_gain < 0.1` →
escalate. That confirms the *mechanism* is sound by construction, but **no real pipeline
run has ever escalated.** The one cell that was *supposed* to escalate (d16) MATCHed
instead, for the confound reason in §6.

---

## 6. The d16 off-distribution confound (why it MATCHed instead of escalating)

`c_keystone_deep_mlp_d16_cascaded` is a `deep_mlp` depth-16 width-128 stack of plain
`Linear+ReLU` (no residual / norm). **WS3 already established that this exact architecture
is ANN-untrained-floor:** d16w128 reaches **chance for BOTH cascaded and synchronized**
(ledger `WS3 … training_floor_confound`, ANN test 0.1135). The model never trained.

Why the keystone could not catch it, mechanistically:

- The probes look for a **death-cascade** signature (decode magnitude collapsing toward
  zero with depth). The thresholds are: `cold_cascade_live` needs decoded magnitude
  `> 1e-4` on ≥75% of depths and at the output; `firing_gain < 0.1` to escalate.
- A **degenerate-but-trained-to-chance** model is **not** a death-cascade. Its activations
  have **healthy magnitude** (well above the 1e-4 liveness floor — a chance classifier
  still produces non-trivial, roughly uniform logits), and its genuine cascade decode
  ≈ its staircase decode (so `firing_gain ≈ 1`, comfortably above 0.1). All four probes
  therefore **pass** → MATCH.

So d16 is a **confounded escalation test in both directions**: even if it *had* escalated,
a "correct-looking" ESCALATE would not validate the off-distribution claim (an untrained
model is dead everywhere for reasons unrelated to recipe suitability); and because it is
degenerate rather than a death-cascade, it does not exercise the firing-gain probe at all.
**The d16 result tells us nothing about whether escalation works.** It only tells us the
keystone does not false-escalate on a chance-level model — a safe but uninformative
outcome for the escalation hypothesis.

---

## 7. RECOMMENDATION — a clean escalation re-test (a TRAINABLE off-distribution model)

A valid escalation test needs a model that is (a) **trainable** (ANN well above chance, so
the deployment metric is meaningful), (b) **off the mmixcore distribution** the thresholds
were calibrated on, and (c) **exhibits a real cascaded firing-gain deficit** (so the
death-cascade probes have something to fire on). WS3's own ledger hands us the ideal
vehicle.

### Primary recommendation: `deep_mlp d8 w64 cascaded`

WS3 measured this exact cell: **ANN 0.979, cascaded deployed 0.872 — a ~10.7 pp gap,
verdict `cascaded_firing_gain_degraded`** (ledger `WS3 depth=8 width=64 sched=cascaded`).
This is the textbook escalation target: trainable, off-distribution (a plain deep MLP, not
a mixer), and it actually *has* the firing-gain degradation the keystone is meant to
detect — yet the fast recipe leaves ~10 pp on the table, which is precisely when the
controller should take over. Config (clone the d16 cell, change only the depth and S, keep
`conversion_policy:true`):

```jsonc
// experiments/campaign/c_keystone_deep_mlp_d8_cascaded.json
{
  "seed": 0, "pipeline_mode": "phased",
  "experiment_name": "c_keystone_deep_mlp_d8_cascaded",
  "data_provider_name": "MNIST_DataProvider",
  "platform_constraints": { "max_axons": 784, "max_neurons": 512,
    "target_tq": 4, "simulation_steps": 4, "weight_bits": 5,
    "has_bias": true, "allow_coalescing": true, "allow_neuron_splitting": true,
    "cores": [ {"max_axons":784,"max_neurons":512,"count":60,"has_bias":true},
               {"max_axons":512,"max_neurons":256,"count":60,"has_bias":true} ] },
  "deployment_parameters": {
    "model_type": "deep_mlp",
    "model_config": { "base_activation": "ReLU", "depth": 8, "width": 64 },
    "spiking_mode": "ttfs_cycle_based", "firing_mode": "TTFS",
    "spike_generation_mode": "TTFS", "thresholding_mode": "<=",
    "ttfs_cycle_schedule": "cascaded",
    "ttfs_blend_fast": true, "ttfs_blend_fast_stabilize_steps": 1200,
    "conversion_policy": true,
    "weight_quantization": true, "activation_quantization": false,
    "encoding_layer_placement": "subsume", "max_simulation_samples": 500,
    "enable_nevresim_simulation": true, "enable_sanafe_simulation": false,
    "lr": 0.003, "training_epochs": 10, "batch_size": 128,
    "degradation_tolerance": 0.99, "model_config_mode": "user", "hw_config_mode": "fixed",
    "training_recipe": {"optimizer":"adamw","scheduler":"cosine","weight_decay":0.0001,
      "warmup_ratio":0.05,"grad_clip_norm":1,"layer_wise_lr_decay":1,
      "label_smoothing":0,"betas":[0.9,0.999]},
    "tuning_recipe":   {"optimizer":"adamw","scheduler":"cosine","weight_decay":0.0001,
      "warmup_ratio":0,"grad_clip_norm":1,"layer_wise_lr_decay":1,
      "label_smoothing":0,"betas":[0.9,0.999]} }
}
```

**What to check (the real pass/fail of the escalation hypothesis):**

1. Does the keystone **escalate** (controller trace, not the 8-rung commit ladder)? If
   yes, *which* probe tripped — read it by adding a one-line provenance log of
   `decision.escalation_reason` + `decision.characterization.probes`, or instrument
   `CascadeCharacterizer.characterize` to dump the four readings.
2. If it escalates on `firing_gain < 0.1`: that is the **clean win** — a trainable
   off-distribution cell with a real gain deficit correctly routed to the controller.
3. If it **MATCHes** anyway (fast trace, deployed ≈ 0.87): that is an equally important
   **false-keep-fast** finding — the absolute `firing_gain ≥ 0.1` floor is too lax for
   d8's degradation (0.872/0.979 is a 10 pp deploy gap but the deepest-layer
   `genuine/staircase` ratio may still sit above 0.1). That directly motivates the
   **relative gain criterion** of §8.

### Secondary recommendation: a second, architecturally distinct off-distribution cell

To separate "off-distribution architecture" from "deep-MLP-specific firing gain," add one
non-MLP cell. `lenet5 cascaded` (registered `lenet5`, a conv architecture, expected to
train ~0.99 on MNIST) is the natural pick — but it has **no cascaded deployment precedent
in the ledger**, so **first run it once *without* `conversion_policy` to confirm it trains
and deploys ≥ 0.95 cascaded** before using it as a keystone vehicle. If lenet5 cascaded
deploys near-lossless, it tests the *other* useful direction: a trainable off-distribution
cell where the keystone should **MATCH** (confirming the thresholds don't false-escalate
on a healthy non-mixer). Use d8 for the escalate direction and lenet5 for the
off-distribution-keep-fast direction.

**Do not** re-use any `deep_mlp` depth ≥ 12 (d12/d16/d24/d32 are all WS3
`training_floor_confound` — chance ANN, useless as escalation probes). d8 w64 is the
deepest cascaded deep_mlp WS3 found that is both trainable and firing-gain-degraded.

---

## 8. The threshold-generalization concern (the central open risk)

The four probes use **MNIST-mmixcore-calibrated absolute constants** baked into
`characterization.py`: `_FIRING_GAIN_FLOOR = 0.1`, `_LIVE_DEPTH_FRACTION = 0.75`,
`_RAMP_BUDGET = 0.02`, `_LIVENESS_FLOOR = 1e-4`.

1. **The floors are absolute, not relative to the cell.** A cell whose *healthy* firing
   gain or live fraction sits differently could read as a mismatch (false-escalate, safe
   but wasteful) — or, the dangerous direction, a genuinely degrading cell could read as a
   match (false-keep-fast, shipping fast where the controller was needed). d8 (§7) is the
   exact stress case: a 10 pp deploy gap that may or may not push the deepest-layer ratio
   below 0.1.
2. **Liveness ≠ accuracy.** The d16 confound (§6) shows the converse hole: a model with
   healthy-magnitude-but-useless activations passes every probe. The probes detect a
   *death-cascade*, not a *bad model*; they cannot (and are not meant to) catch a
   training-floor confound. Keep that scope explicit.
3. **No relative / per-cell normalization exists.** There is no "escalate on a *relative*
   genuine-vs-own-staircase drop" — only the absolute floor. If d8 turns up a
   false-keep-fast, add the relative criterion alongside the absolute floor rather than
   retuning a single magic scalar.

**We still cannot state whether the thresholds generalize.** Two real MATCH cells
(matrix_6, matrix_8) is encouraging for the keep-fast direction; the escalate direction
remains entirely unproven on real runs.

---

## 9. Honest limits (one place, no hedging elsewhere)

- **Demonstrated (keep-fast):** the dtype crash is fixed; the keystone MATCHes and keeps
  the proposed driver on the **in-distribution matrix_6** (deployed 0.96) AND the
  **near-distribution no-bias matrix_8** (deployed 0.9606) cells. The auto-selection path
  works end-to-end across the bias boundary.
- **Demonstrated (routing + rescue, §0):** on the trainable off-distribution
  `deep_mlp d8 w64 cascaded` cell, `conversion_policy:true` routes to `driver=controller`
  and deploys **0.9452 (3-seed) vs the WS3 fast-ladder baseline 0.8754 — a +7.0pp rescue**,
  reproduced over 3 seeds with faithful NF↔SCM/sim parity. The keystone's
  routing-to-controller path works end-to-end and recovers the cascaded deficit.
- **Isolated by ablation (§0.2):** the +7pp rescue is the **CONTROLLER driver**, not the
  policy and not an escalation. With `ttfs_blend_fast:false` (fast ladder off in both arms)
  cpFalse 0.9396 ≈ cpTrue 0.9417 — the keystone adds nothing the controller doesn't already
  deliver; it only *selects* the controller (vetoing the lossy fast 5-rung ladder).
- **STILL NOT separable:** ESCALATE vs MATCH on the d8 cell. Because `propose_recipe` always
  proposes `driver=controller`, both branches leave the **same** controller trace, and the
  decision (`escalated`/`escalation_reason`) is not logged. The §7.1 instrumentation
  (`decision.escalated` + `characterization.probes`) is still **not added** — so we know the
  keystone routes correctly and recovers accuracy, but not *via which branch*.
- **Confounded (the d16 cell, §6):** ANN-training-floor (chance ANN, not a death-cascade);
  it MATCHed because a chance model passes all four probes. Tells us nothing about escalation.
- **matrix_8 `rc=1`:** an unrelated Hard-Core packing resource exhaustion (no-bias
  softcore layout), downstream of and irrelevant to the keystone decision; the decision
  and deployed metric (0.9606) were both produced cleanly before the crash.
- **Next instrumentation (to separate ESCALATE from MATCH):** log `decision.escalated` +
  `decision.characterization.probes` at the d8 TTFS step, and (optionally) make
  `propose_recipe` propose `fast` for cells expected to keep-fast, so MATCH→fast and
  ESCALATE→controller leave *distinguishable* traces again.

### §8 — lenet5/FMNIST rescue on a VALID vehicle: PARTIAL, ~5.7pp floor remains; theta_cotrain=TRUE never run (2026-06-24)

The §0 +7pp rescue was demonstrated on the **INVALID** host-majority `deep_mlp d8`.
This batch re-tests the keystone's recoverability claim on the one **VALID**
`lenet5`/FashionMNIST cascaded cell (ANN ~0.917, 99.1% on-chip), 3 seeds/arm,
`max_simulation_samples=1000` (read pp, not 3rd decimals; binomial SE ~0.0092).
Ledger: `cluster:"WS7"`, `kind:"escalation"`, `model:"lenet5"`.

| arm | conversion_policy | theta_cotrain | deployed (3-seed) | ANN gap | lift vs cpFalse | fraction of 7.12pp closed |
|:----|:-----------------:|:-------------:|------------------:|--------:|----------------:|:--------------------------|
| cpFalse (baseline, routing OFF) | false | — | **0.846** | 7.12pp | — | — |
| cpTrue (routing ON) | true | — | **0.8577** | 5.88pp | +1.17pp | ~17% |
| plnrescue cotFalse | true | false | **0.8603** | 5.71pp | +1.43pp | ~20% |

**Verdict — PARTIAL-RESCUE / FLOOR-REMAINS.** The deficit is real and
gradient-bearing (NOT dead, NOT untrained-ANN: ANN ~0.917, logs show a
`finalize_cliff` 0.177 that recovers to ~0.84). But the available controller-routing
lever (`conversion_policy`) closes only ~17% of the ~7pp gap (+1.17pp) and the
`plnrescue` baseline (cp ON, `theta_cotrain` OFF) reaches only 0.8603 (residual
5.71pp). A **hard ~5.6–5.9pp ANN-gap floor remains**. The closeout-10.2 / WS3-4.2
"non-dead, gradient-bearing deficit is recoverable" claim is therefore **NOT
validated to lossless on this VALID on-chip vehicle** — it remains shown only on
the INVALID `deep_mlp d8`. The **WS7 automatic-rescue-on-a-VALID-vehicle cell does
NOT move to MET** on this evidence.

**Confounds.**
1. **NO synchronized arm** exists in these 9 runs — all three arms are
   `ttfs_cycle_schedule=cascaded`; the only rescue/comparison axis is
   `conversion_policy` (plus the `theta_cotrain=false` baseline). No cascaded→sync
   gap is reportable; `synchronized_*` fields are null.
2. **The named rescue lever `theta_cotrain` was NEVER turned ON for this cell.**
   `plnrescue_*cotFalse` has `ttfs_theta_cotrain=false`, and **no `plnrescue_*cotTrue`
   run exists** in `q/done/` or `q/failed/`. So the **upper bound of the rescue is
   UNTESTED here** — recoverability is bounded only by what cpTrue/cotFalse achieve.
3. **Subsampled eval** (`max_simulation_samples=1000`): 3rd decimals are within
   binomial noise; only pp-scale gaps are reliable.
4. NOT confounded by untrained ANN (all ~0.9152–0.9197 ≫ chance 0.10) nor by dead
   neurons (gradient-bearing: cascade fine-tuning recovers from the finalize cliff).

**Open (the missing rescue arm):** run `plnrescue_lenet_FashionMNIST_cotTrue_s{0,1,2}`
(cp ON, `ttfs_theta_cotrain=TRUE`) to establish the actual upper bound of the
per-channel θ-cotrain rescue on this VALID vehicle; without it the keystone's
recoverability claim stays UNDEMONSTRATED on-chip.

Run ids: `cp_lenet_FashionMNIST_DataProvider_{cpFalse,cpTrue}_s{0,1,2}`,
`plnrescue_lenet_FashionMNIST_cotFalse_s{0,1,2}`.

---

### §9 — NO controller-rescue on the FIRST VALID `deep_cnn` d6 onset cell; θ-cotrain is BROKEN on the convnet (`item_id=dcnn_d6_onset_gatefix_rescue`, 2026-06-24)

The §0 +7pp controller rescue and closeout-10.2's positive lift were both measured on
the **INVALID host-majority `deep_mlp d8`**. This is the clean re-test on the **FIRST
VALID `deep_cnn`** firing-gain-deficit vehicle — the d6 onset rung where the within-CNN
cascade first breaks (AC_EVIDENCE §1f: lossless ≤d5, ~5pp deficit ≥d6). `deep_cnn` (w16),
MNIST, `ttfs_cycle_based` S=4, on-chip-majority 99.41%, 3 seeds/arm,
`max_simulation_samples=200`. Two orthogonal rescue knobs are gridded: `conversion_policy`
(cp, the controller revive→refine routing) and `ttfs_theta_cotrain` (cot, the per-channel
θ gain-trim). Ledger: `cluster:"WS7"`, `kind:"escalation"`, `model:"deep_cnn"`, `depth:6`.

| arm | cp | cot | deployed (3-seed) | seeds | ANN | casc→sync gap | rc | verdict |
|:----|:--:|:---:|------------------:|:------|----:|--------------:|:--:|:--------|
| no-policy (baseline) | false | false | **0.9500** | .965/.955/.93 | 0.9938 | +4.04pp | 0 | gradient-bearing deficit, no auto-rescue |
| controller revive→refine | **true** | false | **0.8983** | .985/.94/**.77** | 0.9941 | +9.21pp | 0 | **regresses** (cp lift −5.17pp mean) |
| θ-cotrain (any cp) | — | **true** | **n/a** | — | — | — | **1** | **BROKEN (rc=1 crash)** |

Synchronized ceiling (`pdcnnbc_d6_synchronized_*`, FULL 10k test): **0.9904**
[.9889/.9918/.9906], ANN 0.992.

**Verdict — NO-RESCUE-ON-VALID-VEHICLE; closeout-10.2's controller-rescue lift does NOT
replicate.** On the valid convnet, `conversion_policy` does **not** auto-rescue the d6
firing-gain deficit: the cpFalse→cpTrue lift is **NEGATIVE** (−5.17pp mean, −1.50pp
median), because the policy is high-variance and **catastrophically regresses one seed**
(cpTrue s2 = **0.77**, a genuine finalized collapse). The +2pp s0 lift (0.965→0.985)
cited as the original signal is a **single-seed artifact** that does not survive seeds
(s1 0.94, s2 0.77). The orthogonal θ-cotrain knob is **unusable** here (all 6 cotTrue
runs crash rc=1). So on the convnet there is **no working firing-gain rescue lever**, and
the no-policy cascaded mean **0.950 plateaus ~4.0pp below the 0.9904 synchronized
ceiling** and ~4.4pp below ANN — squarely in the AC_EVIDENCE §1f ~5pp d6 plateau band.
**Synchronized stays the unconditional deep_cnn default.**

**Confounds.**
1. **θ-cotrain BROKEN (decisive).** All 6 cotTrue runs (cpFalse + cpTrue) finalize rc=1
   in `q/failed/` with `RuntimeError: [ModelRepresentation] forward failed at node
   Conv2DPerceptronMapper(name='features_3')` in `converted_model_flow.forward`
   (proximate torch error: a tensor-shape mismatch `28 vs 16 at dim 3`). The 0.99+
   `__target_metric.json` floats on disk for those runs are **stale pre-deployment
   ANN-stage artifacts** (the runs crash before deployment) — **not** valid deployed
   metrics. ⇒ `conversion_policy` is the **only measurable** rescue lever on the convnet.
2. **cpTrue s2 = 0.77 is VARIANCE, not a crash.** It is rc=0 in `q/done/`, on-chip
   99.41%, NF↔SCM cascaded agreement **1.0000**, torch↔deployed-sim parity **1.0000** —
   a **genuine deployed collapse** (revive→refine landed in a bad basin on that seed), so
   the negative mean lift is a real property of the lever.
3. **Eval-set granularity.** Cascaded `max_simulation_samples=200` (0.5% grid → read
   pp-level gaps, not 3rd decimals); the synchronized ceiling is FULL 10k (4 decimals),
   so the ~4pp cpFalse→sync gap is real but its sub-pp digits are not commensurable.
4. **NOT chance / NOT untrained.** ANN ~0.994 and cpFalse cascaded ~0.95 are both
   well-trained (≫ 0.1135 MNIST chance) ⇒ this IS a genuine non-dead, gradient-bearing
   firing-gain-deficit cell, so the no-rescue verdict is meaningful.

Run ids (cascaded): `pdcnnd6fix_cotFalse_cp{False,True}_s{0,1,2}` (rc=0),
`pdcnnd6fix_cotTrue_cp{False,True}_s{0,1,2}` (rc=1, θ-cotrain broken). Synchronized
ceiling: `pdcnnbc_d6_synchronized_s{0,1,2}`. **Open (the missing working lever):** the
d6 firing-gain gate-fix (the AC_EVIDENCE §2b/§1k `plan_stage:25` proposal) — with
`conversion_policy` no-rescue and `theta_cotrain` rc=1-broken on the convnet, no existing
knob closes the d6 onset deficit; a θ-cotrain convnet-forward fix (or a relative-gain
gate-fix) is the only remaining route to a working rescue.

---

### §10 — NO-RESCUE also at the d10 death-cascade rung (REFUTED note only, `item_id=dcnn_d10_gatefix_rescue`, NOT consolidated, 2026-06-25)

A brief note (this item was adjudicated **REFUTED** and is recorded here for context only —
it is **NOT** ledger-consolidated). The §9 NO-RESCUE finding at the d6 onset rung **extends
to the d10 death-cascade rung**: at `deep_cnn` d10 MNIST cascaded (ANN ~0.99, a genuine
firing-gain regime, not at-chance), the keystone gate-fix **does not auto-rescue**.

- **θ-cotrain (cot=true) CRASHES the convnet** — all 6 cotTrue runs finalize `rc=1` with the
  **same `Conv2DPerceptronMapper(name='features_3')` tensor-shape break** (`size of tensor a
  (28) must match b (16) at dim 3`) seen at d6 (§9); no deployed metric exists. The high
  `__target_metric.json` floats on disk for those runs are **stale pre-deployment artifacts**.
- **Gate-fix-only (cp=true, cot=false) deploys ~0.79** (n=2 valid: s0 0.865, s1 0.715; s2 is
  `rc=-9` timed-out with a STALE pre-SCM metric → excluded) — **at or below the ~0.95 cascaded
  baseline and ~20pp under the ~0.992 synchronized ceiling** (faithful: NF↔SCM agreement 1.0,
  torch↔sim 1.0). The cpFalse cpFalse arm has **no valid run** (all 3 `rc=-9` timeouts; their
  high `__target_metric` values 0.94–0.98 are STALE pre-HCM analytical metrics, **not** deployed).
- **Confounds (why REFUTED, not CONFIRMED):** the `pdcnnd10fix` batch has **no synchronized
  arm** (ceiling/baseline imported from the matched `pdcnnbc` d10 batch: sync 0.9917, cascaded
  baseline 0.9517 but **high-variance** 0.907–0.985); the cp:true s1 0.715 is a genuine
  death-cascade collapse (val cliff to 0.19 at finalize, recovered only to 0.71), so the
  gate-fix arm landed on the collapsed tail of a fragile cascade. The decisive, robust facts —
  θ-cotrain `rc=1` convnet-crash (identical to §9) and cp:true-only ≤ the cascaded baseline —
  stand regardless.

Run ids: `pdcnnd10fix_cot{True,False}_cp{True,False}_s{0,1,2}`. **Takeaway (consistent with §9):**
the keystone gate-fix has **no working firing-gain rescue lever on the deep convnet** at either
the d6 onset or the d10 death-cascade rung; **synchronized remains the unconditional deep
default and cascaded-deep is retired.** A θ-cotrain convnet-forward fix (the `features_3`
tensor-shape break) is the prerequisite for any future gate-fix rescue test at depth.

---

### §11 — The convnet-compatible STE replacement ALSO fails: staircase-STE REGRESSES the d6 onset on the DATASET axis (FashionMNIST, KMNIST) (`item_id=dcnn_d6_dataset_ste_gatefix`, 2026-06-25)

§9/§10 left θ-cotrain `rc=1`-broken (the `Conv2DPerceptronMapper(features_3)` tensor-shape
break) as the **only** un-tested firing-gain knob on the convnet, and `conversion_policy` as
the only *measurable* lever (which net-regresses, −5.17pp on MNIST). This batch tests the
**convnet-compatible replacement** for the broken θ-cotrain — `ttfs_staircase_ste`
(`ste_mix=0.5`, the hedged staircase-backward STE) — gridded against `conversion_policy`, and
extends the question to the **harder datasets** at the same d6 onset. `deep_cnn` (w16),
`ttfs_cycle_based` S=4, on-chip-majority VALID, 3 seeds/arm, `max_simulation_samples=200`.
Ledger: `cluster:"WS7"`, `kind:"escalation"`, `item_id:"dcnn_d6_dataset_ste_gatefix"` (2 rows).

| dataset | ste | cp | deployed (3-seed) | seeds | ANN | lever lift | sync ceiling | verdict |
|:--------|:--:|:--:|------------------:|:------|----:|-----------:|:------------:|:--------|
| FashionMNIST | false | false | **0.8433** (baseline) | .875/.825/.83 | 0.932 | — | 0.8962 | no-lever baseline |
| FashionMNIST | **true** | false | **0.7767** | .75/.78/.80 | 0.929 | **ste −6.66pp** | 0.8962 | STE regresses |
| FashionMNIST | false | true | 0.8167 | .795/.83/.825 | 0.931 | cp −2.66pp | 0.8962 | cp net-negative |
| FashionMNIST | true | true | 0.7933 | .745/.815/.82 | 0.931 | ste −2.34pp | 0.8962 | no compose |
| KMNIST | false | false | **0.9083** (baseline) | .895/.935/.895 | 0.974 | — | 0.9619 | no-lever baseline |
| KMNIST | **true** | false | **0.8583** | .895/.805/.875 | 0.966 | **ste −5.0pp** | 0.9619 | STE regresses |
| KMNIST | false | true | 0.8583 | .815/.865/.895 | 0.969 | cp −5.0pp | 0.9619 | cp net-negative |
| KMNIST | true | true | 0.865 | .855/.86/.88 | 0.963 | ste +0.67pp | 0.9619 | within noise |

Synchronized ceiling + cascaded baseline are the SAME d6/S=4/n=200 cell (paired
`pdcnnbcd6data_*`, commensurable): FMNIST sync **0.8962** / cascaded baseline 0.8183;
KMNIST sync **0.9619** / cascaded baseline 0.9167.

**Verdict — NO-FIRING-GAIN-RESCUE-LEVER-EXISTS-ON-CONVNET-d6-ONSET (the dataset axis confirms
§9).** The convnet-compatible `ttfs_staircase_ste` (`ste_mix=0.5`) does **not** rescue and
instead **REGRESSES** the cascade on both harder datasets: FMNIST steTrue best 0.7933 vs
no-lever baseline 0.8433 = **−5.0pp** (and −10.29pp under sync 0.8962); KMNIST steTrue best
0.865 vs baseline 0.9083 = **−4.33pp** (−9.69pp under sync 0.9619). The ste lift at cpFalse is
−6.66pp (FMNIST) / −5.0pp (KMNIST). It does **not compose** with `conversion_policy`, which is
itself net-negative on the convnet (cp lift −2.66pp FMNIST / −5.0pp KMNIST at steFalse) —
identical sign to the §9 MNIST cp regression. Every steTrue arm sits **9.7–10.3pp below the
synchronized ceiling** and **4.3–5.0pp below the no-lever baseline**. So with θ-cotrain `rc=1`
(§9/§10), `conversion_policy` net-regressing (§9 MNIST + here), and now staircase-STE
regressing, there is **NO working config-level firing-gain rescue lever at the d6 convnet
onset on the dataset axis** — **synchronized remains the unconditional deep_cnn default.**

**Confounds.**
1. **All 24 grid runs rc=0** (`q/done/`, finalized, deployed `__target_metric.json` present);
   3 seeds/arm; `max_simulation_samples=200` on EVERY run (grid + paired baseline/sync) →
   1/200=0.5% granularity → **read pp-level gaps, not 3rd decimals**.
2. **NOT chance / NOT untrained.** ANN is healthy on every cell (FMNIST ~0.929–0.932, KMNIST
   ~0.963–0.974, both ≫ 0.10 chance) ⇒ the regression is a genuine firing-gain/gradient
   effect, not an untrained-floor artifact.
3. **Commensurable pairing.** The sync ceiling and cascaded baseline both come from the PAIRED
   `pdcnnbcd6data_*` batch at the SAME d6 onset, SAME S=4, SAME n=200 (unlike the §9 MNIST sync
   which was full-10k). This is an all-cascaded grid (`ttfs_cycle_based`, S=4); ste/cp are
   levers, the synchronized arm is the external `pdcnnbcd6data` ceiling. The KMNIST
   steFalse/cpFalse baseline 0.9083 dips ~1pp below the bc-batch cascaded baseline 0.9167
   (both n=200, within seed/200-sample noise); both well below sync 0.9619.
4. **θ-cotrain (the originally-planned knob) was NOT run here** — it crashes `rc=1` on
   `deep_cnn` (`Conv2DPerceptronMapper features_3`, §9/§10). This grid is its convnet-compatible
   replacement, which **also** fails to rescue.

Run ids (lever, rc=0): `pdcnnd6datastefix_{FashionMNIST,KMNIST}_DataProvider_steTrue_cp{False,True}_s{0,1,2}`;
baselines: `pdcnnd6datastefix_{...}_steFalse_cp{False,True}_s{0,1,2}`; paired ceiling/baseline:
`pdcnnbcd6data_{...}_{synchronized,cascaded}_s{0,1,2}`. **Open (still the only route):** a
θ-cotrain convnet-forward fix (the `features_3` tensor-shape break) — every config-level lever
on the convnet d6 onset is now exhausted (cp net-negative, staircase-STE regressing), so the
remaining firing-gain rescue work is a **code fix to the per-channel θ knob**, not a new schedule.

---

### §12 — NEITHER escalation rescues the n=1000 bigcores-gatefix `deep_cnn` d8 MNIST cell: θ-cotrain CRASHES `rc=1` (unmeasurable) and conversion_policy REGRESSES −2.47pp; the cell is already near-lossless so there is almost no deficit to rescue (`item_id=dcnn_deep_controller_escalation`, 2026-06-25)

§9–§11 exhausted the d6 onset rescue space (θ-cotrain `rc=1`-broken, cp net-negative,
staircase-STE regressing). This item asks the same rescue question one rung deeper, on the
**n=1000-trained big-cores-gatefix `deep_cnn` at d8 (MNIST, S=4, cascaded `ttfs_cycle_based`)**
— the controller-escalation grid `pdcnnbcn1000fix_d8_cot{T,F}_cp{T,F}_s{0,1,2}` (12 runs).
Ledger: `cluster:"WS7"`, `kind:"escalation"`, `item_id:"dcnn_deep_controller_escalation"`.

| arm | cot | cp | deployed (3-seed) | seeds | rc | ANN ref | verdict |
|:----|:--:|:--:|------------------:|:------|:--:|--------:|:--------|
| baseline (pure cascaded) | false | false | **0.9723** | .96/.981/.976 | 0 | ~0.974 | near-lossless (casc→ANN ~0.2pp) |
| conversion_policy escalation | false | true | **0.9477** | .978/.954/**.911** | 0 | ~0.974 | **REGRESSES −2.47pp** (one-seed collapse) |
| θ-cotrain escalation (any cp) | true | — | **n/a** | — | **1** | — | **BROKEN (rc=1 crash, unmeasurable)** |

In-log ANN refs: 0.9704 / 0.9807 / 0.972 (mean ~0.9744, ≫ 0.1135 chance). cot=True runs'
0.99 `__target_metric.json` floats are **stale pre-deployment ANN-stage artifacts** (the runs
crash before deployment), NOT valid deployments.

**Verdict — NEITHER ESCALATION RESCUES; the cell is already near-lossless so there is almost
nothing to rescue.** The **θ-cotrain (cot) escalation CANNOT auto-rescue** the cell because all
6 cotTrue runs crash `rc=1` (the same `Conv2DPerceptronMapper features_3` tensor-shape break as
§9–§11) and are unmeasurable. The **conversion_policy (cp) escalation does NOT rescue either**:
across the 6 finalized rc=0 cotFalse runs, cp leaves accuracy unchanged-to-WORSE — cpFalse
**0.9723** vs cpTrue **0.9477**, a cp lift of **−2.47pp** (a regression dragged by one seed
collapse to 0.911, NOT a lift toward the ~0.974 ANN ceiling). Crucially, the d8 cell is already
**near-lossless on cpFalse** (cascaded→ANN gap only ~0.2pp at n=1000), so there is almost no
firing-gain deficit for an escalation to close. This is the **deeper-rung confirmation of the
§9 MNIST d6 cp NO-OP/REGRESSION and the §9–§11 cotTrue `rc=1` crash** — synchronized remains the
unconditional deep_cnn default and the only remaining rescue route is a code fix to the
per-channel θ-cotrain convnet-forward path.

**Confounds.** (1) **cot escalation unmeasurable:** all 6 cotTrue runs `rc=1` (`q/failed/`),
their 0.99 floats are stale ANN-stage artifacts, not valid deployments. (2) **cp regression is
one-seed-driven:** cpTrue 0.978/0.954/0.911 — the −2.47pp mean is dragged by the s2=0.911
collapse (cpTrue sd 2.77pp vs cpFalse 0.90pp). (3) **`max_simulation_samples=1000`** → read
pp-gaps, not 3rd decimals. (4) **NO at-chance confound:** ANN ~0.974 ≫ 0.1135 → genuine
result. (5) **NO paired synchronized arm** in this batch (only cot/cp axes). (6) The near-
lossless baseline (casc→ANN ~0.2pp) makes this a *negative* rescue cell — it shows the
escalation levers do not HELP and can HURT even where the deficit is small. Run ids: cpFalse
baseline `pdcnnbcn1000fix_d8_cotFalse_cpFalse_s{0,1,2}`; cp escalation
`pdcnnbcn1000fix_d8_cotFalse_cpTrue_s{0,1,2}`; cot-crashed (rc=1)
`pdcnnbcn1000fix_d8_cotTrue_cp{False,True}_s{0,1,2}`.

---

## 9. The conversion_policy rescue ladder — onset-vs-rescue map (2026-06-26)

> ⚠️ **VEHICLE MISLABEL + VALIDITY:** the `cp_lad_*` / `cp_d8_*` cells below are
> `deep_mlp w64`, **NOT** `deep_cnn` — every config is `model_type:deep_mlp` (verified
> in `experiments/campaign/cp_lad_*.json`, 24/24). The exact 24 `cp_lad` ids sit in the
> campaign **VALIDITY host-majority quarantine** (`ledger.jsonl` `kind:quarantine_coverage
> n=292`). d4 ≈19.5% on-chip = **INVALID** (below the 20% gate-v2 floor); d6 ≈28.7% and
> d8 ≈34% = **VALID_FLAGGED_placement** (the host `784→64` encoder Linear is offloadable
> → ~99% on-chip if `encoding_layer_placement=offload`). Read this as the phenomenology of
> the cp lever, not as a valid `deep_cnn` deployment.

**Question.** Across the `deep_mlp w64` S=4 **cascaded** depth ladder, how does the
`conversion_policy` escalation (`cpFalse → cpTrue`, which routes the cell to
`driver=controller` — see §0.2) trade off against the *size* of the within-stack cascaded
firing-gain deficit, as depth and dataset hardness grow?

**Setup.** Pure cp **true/false control**: `ttfs_cycle_schedule=cascaded` and
`ttfs_blend_fast=true` in **both** arms — the only knob is `conversion_policy`. There is
**no synchronized run** here, so per the WS7 escalation convention the ledger's
`cascaded_*` columns hold cpFalse, `synchronized_*` hold cpTrue, and
`cascaded_to_sync_gap_pp` is **repurposed as `cp_lift` (cpTrue − cpFalse)**. 3 seeds/cell,
`max_simulation_samples=200` (→ read pp-gaps, not 3rd decimals). All cells finalized
(`rc=0`); ANNs well-trained (MNIST ~0.977–0.979, FMNIST ~0.886–0.881 ≫ 0.10 chance);
parity clean on every run (NF↔SCM 1.0, torch↔sim 1.0) → the deployed metric is faithful.

### 9.1 The ladder (all `deep_mlp w64`, S=4 cascaded, 3-seed means)

| dataset | depth | cpFalse (cascaded) | cpTrue (escalated) | **cp_lift** | ANN | cpTrue→ANN gap | on-chip | validity |
|---|---|---|---|---|---|---|---|---|
| MNIST  | 4 | 0.9467 | 0.9550 | **+0.83** | 0.9791 | 2.63pp | ~19.5% | INVALID |
| MNIST  | 6 | 0.9433 | 0.9700 | **+2.67** | 0.9794 | 0.91pp | ~28.7% | VALID_FLAGGED |
| MNIST  | 8 | 0.8867 | 0.9483 | **+6.17** | 0.9769 | 2.86pp | ~34% | VALID_FLAGGED |
| FMNIST | 4 | 0.8533 | 0.8750 | **+2.17** | 0.8884 | 1.31pp | ~19.5% | INVALID |
| FMNIST | 6 | 0.7750 | 0.8483 | **+7.33** | 0.8862 | 3.83pp | ~28.7% | VALID_FLAGGED |
| FMNIST | 8 | 0.6950 | 0.8283 | **+13.33** | 0.8806 | 5.23pp | ~34% | VALID_FLAGGED |

(`cp_lad_*` = d4/d6; `cp_d8_*` = d8. Per-seed metric files verified on disk, e.g. FMNIST d6
cpFalse [0.755,0.750,0.820] → cpTrue [0.860,0.850,0.835]; FMNIST d8 cpFalse
[0.710,0.680,0.695] → cpTrue [0.855,0.830,0.800].)

### 9.2 Verdict — the rescue is **firing-gain-deficit-PROPORTIONAL**

The cp escalation lift grows **monotonically with the size of the within-stack cascaded
deficit**: tiny/within-noise where the cascade barely breaks (MNIST d4 +0.83, FMNIST d4
+2.17, MNIST d6 +2.67), and large exactly where it breaks worst (FMNIST d6 +7.33, MNIST d8
+6.17, **FMNIST d8 +13.33**). The deepest-hardest cell (FMNIST d8) — where cpFalse sits
18.56pp below its ANN — gets the **largest** rescue (~2.2× the near-lossless MNIST d8
arm whose cpFalse sits only 9.0pp below ANN). This closes the dataset arm WS7 §0 left open
and establishes a clean **onset-vs-rescue map**: cp escalation pays off in proportion to
the deficit it has to repair.

**Confounds.** (1) **VEHICLE MISLABEL** — both confirmed items titled "VALID deep_cnn";
all cells are `deep_mlp w64` (decisive: this is NOT a deep_cnn result). (2) **VALIDITY** —
d4 INVALID (host-majority, <20% floor); d6/d8 VALID_FLAGGED only because the host encoder
is offloadable; the 24 `cp_lad` ids are in the host-majority quarantine. (3) **NOT a clean
isolation** — cpTrue = the heavier `driver=controller` escalation path (wall 467–973s vs
cpFalse 220–333s, 2–4×), so the lift bundles "more controller compute" with
`conversion_policy`; §0.3's no-blend ablation already showed the *mechanism* is the
controller's post-finalize recovery, not the cp decision per se. (4) **Headline-magnitude
correction** — the source item cited FMNIST d6 as +10.5pp (0.755→0.86); that is the single
**worst** cpFalse seed vs the cpTrue mean — the true 3-seed mean lift is **+7.33pp**.
(5) `max_simulation_samples=200` → gaps reliable to ~1–2pp; the within-noise d4 lifts are
not significant (cpTrue MNIST d4 spans 0.925–0.980). Ledger: `cluster:"WS7"`,
`kind:"escalation"`, `item_id ∈ {ws7_dcnn_controller_rescue_depth_ladder,
ws7_dcnn_d8_fmnist_rescue_completion}` (6 records).

---

### Key file references

- Decision logic + thresholds: `src/mimarsinan/tuning/orchestration/characterization.py`
- Policy / propose / escalate: `src/mimarsinan/tuning/orchestration/conversion_policy.py`
- Keystone consumption / driver gate:
  `src/mimarsinan/tuning/tuners/ttfs_cycle_adaptation_tuner.py` (`_configure`, ~line 374),
  `src/mimarsinan/tuning/orchestration/ttfs_adaptation_plan.py` (~line 138)
- Escalation unit test (the only place escalate fires):
  `tests/unit/tuning/test_conversion_policy_escalation.py`
- Run logs: `runs/campaign/logs/c_keystone_{matrix_6,matrix_8,deep_mlp_d16}_cascaded.log`
- Run configs:
  `experiments/campaign/c_keystone_{matrix_6,matrix_8,deep_mlp_d16}_cascaded.json`
- WS3 trainable/degraded ladder (the re-test vehicle source):
  `docs/research/findings/WS3_depth_firing_gain.md`, `runs/campaign/ledger.jsonl`
  (`WS3 depth=8 width=64 sched=cascaded` → `cascaded_firing_gain_degraded`)
- Per-cell WS7 verdicts: `runs/campaign/ledger.jsonl` (`cluster:"WS7"`, `kind:"keystone_decision"`)
