# TTFS exact-QAT program — generalizing the LIF θ-in-loop exact-QAT to ttfsq + sync

**Status:** mechanism landed (default-off); A/B RESOLVED 2026-07-14 — per-channel
theta is the right QAT lever (forward 0.97-0.98 at S8) but the S8 mixer DEPLOY is
1/S-floored (~0.94-0.96), so do NOT arm; S64 respec stands (§4.2).
**Template:** `lif_exact_qat_program.md` (the LIF path being generalized).
**Theory:** `sync_deployment_exactness.md` (the TTFS composition).

## 1. The gap

The LIF exact-QAT trains the *exact deployed count staircase* with θ trainable
in-loop under a **clamp-gated STE + in-band LSQ θ-gradient**
(`LIFCountStaircaseFunction`), which lifted the LIF mixers +0.6..+5.2pp. The
other two spiking modes did not have this:

- **ttfsq** (analytical `ttfs_quantized`): the AQ stage fell to the plain
  `QuantizeDecorator` — a float shift + floor-quantize proxy with **θ frozen**.
  No exact ceil kernel, no θ training.
- **sync** (`ttfs_cycle_based` synchronized): `sync_exact_qat` trains the exact
  deployed ceil KERNEL (`TTFSCeilStaircaseDecorator`), **but θ is never
  promoted** — `promote_theta_for_exact_qat` was lif/ttfsq-only, and the ceil
  decorator applies θ through plain `÷θ`/`×θ` ops around a plain-STE staircase.
  So sync trains its weights against a frozen θ.

θ-frozen training is the mechanistic reason ttfsq/sync trail LIF at low S: LIF
adapts the grid (θ) to the weights per channel; ttfsq/sync cannot. (The
S-respec confirmed the symmetry at HIGH S: ttfsq S64 0.9795 > LIF S64 0.9776 —
at enough grid levels the modes match; the question is whether θ-in-loop closes
the gap at LOW S without brute-force S.)

## 2. The shared mechanism (landed)

The exact-QAT is now generic across lif/ttfsq/sync — the kernels differ only in
the forward staircase (LIF floor-count vs TTFS ceil first-crossing); everything
else is shared:

- **`TTFSCountStaircaseFunction`** (`models/nn/activations/autograd.py`): forward
  `θ·ttfs_quantized_staircase(z/θ, S)` (the deployed kernel — bit-exact
  identity); backward = the SAME `_gated_lsq_backward` `LIFCountStaircaseFunction`
  now also calls (clamp-gated identity STE on `0<r<1`, in-band LSQ θ-grad
  `q − r·1[0<r<1]`). The TTFS ceil kernel has the same in-band structure as the
  LIF floor kernel (dead below `1/S`, saturated at `r≥1`), so the gate transfers.
- **`TTFSCountStaircaseDecorator`** passes θ INTO the Function (the gated
  gradient trains it), unlike `TTFSCeilStaircaseDecorator` (θ around a plain STE).
- **`promote_theta_for_exact_qat`** (`spiking/theta_cotrain.py`): already
  mode-agnostic — per-channel on R3 matching-axis hops, scalar elsewhere,
  encoder frozen. Reused verbatim by all three modes.
- **`_exact_qat_decorator`** (`adaptation_manager.py`): the shared rate +
  mix/mask envelope wrapping any exact-QAT staircase decorator.

## 3. The three arms

| mode | knob | default | what it adds |
|---|---|---|---|
| lif | `lif_exact_qat` | ARMED | exact count staircase + θ-in-loop + re-timed twin |
| ttfsq | `ttfsq_exact_qat` | off | exact ceil staircase + θ-in-loop (no re-timing — analytical) |
| sync | `sync_exact_qat_theta` | off | promotes θ + gated decorator (kernel already exact) |

TTFS needs **no per-hop re-timing** (LIF-specific — the count back-loading);
TTFS is analytical/synchronized, so its exact-QAT is the θ-in-loop enhancement
alone. All arms are byte-identical when off (golden snapshot unchanged).

## 4. A/B (in progress)

Probes: ttfsq mixer + sync mixer at LOW S (S8, S16), armed vs baseline
(`scripts/_probes/ttfs_exact_qat/xe_*.json`). Hypothesis: θ-in-loop reaches
>0.97 at low S (like LIF), closing the mixer gap without the S-respec crutch.

Verdict rule (same as LIF): if the armed cells gain and no strict cell
regresses, arm `ttfsq_exact_qat` / adopt `sync_exact_qat_theta` in the recipe
and dial back the interim S64 respec; else keep config-armable, document the
refutation (the campaign's post-QAT-inversion discipline).

### 4.1 Scalar theta (slurm/xlog1): REFUTED for the mixer

| cell | exact/theta | baseline | delta |
|---|---|---|---|
| ttfsq S8 | 0.8924 | 0.9561 | -6.4pp |
| ttfsq S16 | 0.9604 | 0.9596 | +0.1pp |
| sync S16 | 0.9635 | 0.9648 | -0.1pp |

Scalar-theta TTFS exact-QAT does NOT help the mixer: ttfsq craters at S8 and is
flat at S16. Mechanistically clear — the mixer's binder is per-channel scale
spread (up to 1870x, M4), which a single scalar theta cannot capture; at S8 the
exact ceil staircase with one global theta starves channels below the float
proxy. Per-channel theta is the right lever; scalar is the wrong one.

### 4.2 Per-channel theta (local, ratchet-hardened): WORKS at QAT, WQ-bound at deploy

The "collapse" of §4.1's follow-up was a MISDIAGNOSIS. Per-channel theta on the
ttfsq S8 mixer trains **healthily** — AQ 0.9702, WQ endpoint recovery val 0.9793
(matching the S64 respec's ~0.9795 but at S=8). What looked like a collapse was
(a) degenerate BN channels being HANDLED (bias routed through normalization beta,
`perceptron_transformer._realize_effective_bias_through_normalization`) — correct
behavior, not failure; (b) the routing print firing every endpoint-recovery step
(6174 lines) as an I/O drag; (c) the 16000-step recovery being infra-killed on the
SHAQ-saturated box (the LIF t0_01 CONTROL was killed the same way at val 0.9787 —
NOT a regression). The ceil kernel quantizes down (q<=r) so its LSQ residual
inflates theta more than LIF's floor kernel -> ~38-40 amplified channels in one
mixer layer (vs LIF's handful), all routed, all recovered.

Fixes landed (generic): the dead-zone theta RATCHET (clamp the dead-zone theta
gradient to >=0 — block the theta-growing runaway, keep the theta-shrinking
revival; the naive full-gate was refuted, routing 7341->10919; ratchet 6174) and
the routing-telemetry gate (`MIMARSINAN_DEGENERATE_ROUTING_DEBUG`, default off).
Ratchet safety (it lives in the SHARED backward): the armed LIF t0_01 mixer is
unregressed — fast-recovery deployed 0.9759 (soft==hard), in the known-good
~0.978 band (the 0.2pp is the 1500-step under-recovery, not the ratchet).

Deployed (theta FROZEN at WQ — witness "in-loop theta frozen on 9 perceptron(s)";
soft-core == hard-core, NF<->SCM parity gate PASSED — the mapping is exact):

| recovery budget | MBH exit | torch test | deployed | vs baseline 0.9561 |
|---|---|---|---|---|
| 1500 steps | 0.9708 | 0.9692 | 0.9595 | +0.34pp |
| 5000 steps (stopped 2016, patience) | 0.9631 | 0.9636 | 0.9427 | -1.34pp |

The torch->spiking gap is honest WQ integer rounding
(`[[mixer_nf_scm_wq_residual_resolved]]` — inherent, do not re-chase). The
recovery is STOCHASTIC (the deployed model is the exit, not the keep-best), so
deployed swings ~0.94-0.96 across runs — NOT robustly above the gate.

**VERDICT (measured): do NOT arm `ttfsq_exact_qat` for the mixer.** Per-channel
theta-in-loop is the right lever at the QAT LEVEL (forward 0.97-0.98 at S8, vs
scalar's 0.8924 crater), but the S8 DEPLOYMENT is floored by physics the QAT
cannot cross: the run's own advisory measures the 1/S per-hop composition
distortion at **-1.91pp at S=8, healed only by S>=16** (the lif-exactness law).
So the deep (L=9) mixer deploys ~0.94-0.96 at S=8 regardless of QAT quality —
below the 0.97 gate and not robust. This is the post-QAT-inversion meta-lesson
(now 8x): the isolated QAT win inverts on the deployed composition. The S64
respec (t0_11, deployed 0.9795) STANDS as the robust gate-passer; the exact-QAT
stays config-armable as the low-S QAT lever, not a deployment fix. Consistent
with "accept mixers at AQ ceiling" (ADV-STAIRCASE-DEPTH).

Integration bugs the A/B found and fixed (real hardening): WQ theta-freeze
generalized to ttfsq/sync (e78fc38c); sync per-channel theta breaks the
synchronized mapper forward -> scalar (a30af1d8); the ratchet + telemetry gate
(0e1d3ec2, plus the env-gated print).

## 5. Generic theta-aware seam + the casc arm (extending to cascaded)

**Generic theta-aware install seam (47092289).** The promote+witness+[TAG]-print
+reporter-event that the three AQ installs (lif/ttfsq/sync) duplicated is one
reusable seam in `spiking/theta_cotrain.py`: `emit_theta_install_witness` +
`install_exact_qat_theta`. Any tuner installs theta-in-loop through it — the
"generically keep theta-aware tuning across tuners" ask.

**casc_exact_qat arm (eb5f3bc1, 282db44f).** The casc training-path map showed
the AQ-stage decorator is DEAD CODE for casc: the cycle tuner subsumes AQ
decorators (`ttfs_active=True`) and the genuine cascade forward reads
`activation_scale` directly (never the decorator chain) — so ttfsq/sync's
AQ-decorator approach would train a probe that does NOT deploy. The arm therefore
lives in STAGE 3 (`TTFSCycleAdaptationTuner`), composing with the genuine
blend-ramp firing-gain correction:
- `TTFSActivation.set_exact_qat_theta(on)` routes the value-mode proxy's theta
  gradient through `TTFSCountStaircaseFunction` (gated-LSQ ratchet) — forward
  bit-exact for theta >= floor (r>1 saturates in both; relu keeps r>=0), and
  TRAINING-ONLY (the cycle-accurate deploy path is untouched -> parity safe).
- `casc_exact_qat` knob forces `theta_cotrain` on, promotes through the shared
  `install_exact_qat_theta` seam (eligibility-filtered per-channel + ComputeOp
  wrap = exact on-chip export), and is mutually exclusive with the gamma gain
  ramp (both own theta).

Casc is fundamentally floored (premature-fire law: d_max ~= 0.56*sqrt(S); tier-0
casc cells read 0.77-0.87 or collapse), so the arm's value is RELATIVE (does the
collapse-hardened + exact-export theta-cotrain beat the plain-STE default?).

**A/B (mmixcore casc S8, local):**

| arm | torch test | deployed |
|---|---|---|
| casc_exact_qat OFF (default casc, no theta-cotrain) | 0.9567 | 0.9567 |
| casc_exact_qat ON | 0.9518 | 0.9539 |

Flat-to-slightly-worse (-0.28pp deployed), WITHIN casc's stochastic noise band
(casc recoveries swing >1pp run-to-run). Note casc_off has torch == deployed
(0.9567) — casc's genuine cascade forward IS the deployed forward, so there is NO
QAT->deploy gap for the theta-in-loop to close (unlike ttfsq/sync's WQ residual).
The arm is NEUTRAL on this floored cell. **VERDICT: keep `casc_exact_qat`
config-armable, default-off** (same as ttfsq/sync) — the mechanism is landed,
generic, and composes correctly with the blend-ramp, but casc's premature-fire
floor leaves no headroom for the theta lever here. Not chasing multi-seed casc
(de-scoped research mode).
