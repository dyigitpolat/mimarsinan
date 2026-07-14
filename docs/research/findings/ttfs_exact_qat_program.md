# TTFS exact-QAT program — generalizing the LIF θ-in-loop exact-QAT to ttfsq + sync

**Status:** mechanism landed (default-off), A/B in progress (2026-07-14).
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

**Results (slurm/xlog1, scalar theta):** REFUTED for the mixer.

| cell | exact/theta | baseline | delta |
|---|---|---|---|
| ttfsq S8 | 0.8924 | 0.9561 | -6.4pp |
| ttfsq S16 | 0.9604 | 0.9596 | +0.1pp |
| sync S8 | (pending) | 0.9603 | - |
| sync S16 | 0.9635 | 0.9648 | -0.1pp |

Scalar-theta TTFS exact-QAT does NOT help the mixer: ttfsq craters at S8
(-6.4pp) and is flat at S16; sync is flat-to-slightly-worse. Mechanistically
clear — the mixer's binder is per-channel scale spread (up to 1870x, M4), which
a single scalar theta cannot capture; at S8 the exact ceil staircase with one
global theta starves channels below the float-proxy baseline. The per-channel
theta that WOULD help is exactly what collapses the mixer (the WQ
degenerate-channel routing OOM, 7341 lines vs 0). **VERDICT: do NOT arm
ttfsq_exact_qat / sync_exact_qat_theta** (fail-toward-measured; keep
config-armable). The mixer benefit requires per-channel theta-in-loop WITH
collapse-hardening (the LIF program's accumulated territory) — the documented
open follow-up. The mechanism (kernel/backward/decorator/theta-promotion) is
correct and generic; scalar theta is simply the wrong lever for this cell.

Three integration bugs the A/B found and fixed (real hardening): WQ theta-freeze
generalized to ttfsq/sync (commit e78fc38c); sync per-channel theta breaks the
synchronized mapper forward -> scalar (a30af1d8); ttfsq per-channel theta
collapses mixer channels -> scalar (c2bdbbac).
