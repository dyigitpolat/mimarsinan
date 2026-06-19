# Final synthesis — a complete theory of the cascaded single-spike TTFS gap

Three phases (two ultracode fan-outs of 7+3+3 agents) on the isolated `cascade_lab.py`
benchmark + real mmixcore validation. The cascaded single-spike TTFS accuracy gap is now
**fully characterized, decomposed, and bounded** — the publishable result.

## The complete causal chain

A single-spike neuron encodes value `v` as fire-time `τ = round(T(1−v))`. A downstream
neuron ramp-integrates an arriving spike to a **triangular** membrane `R(τ) = k(k+1)/2`,
`k = S−τ` — i.e. effective gain `g_eff(τ) = (S−τ+1)/(S+1)`, **quadratic** in the spike's
recency, not the intended linear `v = k/S` (bit-exact to the deployed node). Three
consequences, three different remedies:

| effect | cause | closeable by | irreducible? |
|---|---|---|---|
| **Death cascade** (deep layers decode→0, accuracy→chance) | per-DEPTH mean gain `g_eff` compounds geometrically; deep layers fire late (latency ~1/hop, drift √S/hop); budget **d_max(S)≈0.56·√S** | static per-depth `θ` trim **G** (revive); OR a strong genuine FT | no — structural, recoverable |
| **Per-neuron gain** | fire-time distribution differs by channel | static **per-neuron `θ`** (calib mean/L2): closes 27–93% cold, **+17–46 pts over per-depth G**, bit-exact-equiv to a W/b row scale | no |
| **Per-sample fire-time spread** (the floor, **3–34%** of the residual) | `g_eff(τ)` is **per-sample** — each input fires at a different τ; a static scalar can't invert a per-input-varying gain | only **full genuine FT** (retrains the basin) or an **encode value-warp** / decode change | **yes** to static calibration |

## Two decisive, non-obvious insights

1. **Static gains CANCEL at the readout.** The final classifier is scale-invariant up to
   argmax, so a per-depth/per-neuron `θ` rescale that is uniform-enough across a layer's
   channels mostly cancels — which is *why deployed G gives only ~+0.5pp on mmixcore*
   despite recovering ~85% of the *decoded-value* gap on the toy. **G's real job is
   reviving DEAD layers** (so a downstream FT has gradient), not direct accuracy.
2. **It is NOT capacity-limited, and it IS encode-fixable in principle.** Single-spike
   timing carries `log2(T+1)` bits (= rate). The closed-form **encode value-warp**
   `w(v) = φ_k(v)/S`, `φ_k(v) = (−1+√(1+4vS(S+1)))/2`, inverts `g_eff` so encode→ramp-
   decode = identity: it recovers **~100% of the genuine→ideal-staircase residual at
   S≥8, cold**, and is deployable (a per-neuron monotone input map at the *encode* layer;
   decode untouched, bit-exact, parity-safe). Constraint: a *nonlinear* warp is only
   deployable at the encode/input layer — interior neurons fire via the FIXED ramp and
   admit only an affine `θ/bias` (= G). Exact *interior* linearization needs a **dual-
   spike** code (2× spike traffic — a documented S-constrained fallback, not free).

## Why the deployable levers give only a modest REAL win

On the real mmixcore (a 9-deep cascade), the shipping recipe is the **teacher-blend
genuine ramp** — a strong teacher→genuine *curriculum* that already (a) revives the death
cascade and (b) full-retrains the cascade basin, reaching the ideal staircase on the toy
(full-FT even *exceeds* the underfit toy continuous). So on top of a complete FT:
- G (per-depth) and per-neuron `θ` add little (readout absorbs static gains; FT already
  revived the layers) → measured **+0.5–1.3pp**, near the noise.
- The encode value-warp adds little (FT already reaches the staircase) → its value is the
  **cold / weak-FT / deeper-than-d_max / no-teacher-blend** regime.
- N2 (bias-only phase-advance FT) is **refuted vs full FT** — full FT strictly dominates
  (it rewrites weights into the cascade basin; N2 is capped at the frozen-weight ceiling).

## The bottom line (the honest verdict on LIF-level cascaded TTFS)

- The gap is **fully understood**: a quadratic ramp-decode gain → death cascade
  (`d_max≈0.56√S`) + a per-neuron gain (static-correctable) + an **irreducible per-sample
  fire-time spread** (3–34%, the floor) + the readout-absorption of static gains.
- The genuine cascade, trained through the teacher-blend FT, reaches ~0.93–0.94 on the
  well-trained mmixcore — capped ~3.5pp below LIF (~0.975) by the per-sample spread and
  the harder spiking optimization against a *strong* ANN. The toy "exceeds continuous"
  result is an artifact of the toy's underfit baseline; against a strong ANN the cascade
  caps below.
- **LIF-level cascaded TTFS at practical S is not reachable by these training/calibration
  levers** (the per-sample spread is irreducible without a decode change). The **encode
  value-warp + dual-spike** are the only routes that touch the per-sample term, and the
  exact interior fix (dual-spike) costs 2× spikes. The pragmatic LIF-level path remains
  the **synchronized** schedule (≥0.97); cascaded is the power-optimized variant with a
  now-quantified, well-understood ~3.5pp accuracy trade.

## What ships / what's documented (all default-off, decode bit-exact, parity-safe)

- **G** (`ttfs_gain_correction`, per-depth `θ_d*=γ^d`) — SHIPPED + tested. Keep: revives
  dead/cold/deep layers so a downstream FT has gradient (its real value), modest direct win.
- **Per-neuron `θ` calibration** (calib mean/L2) — the strict upgrade over per-depth G
  (+17–46 pts cold); a clean next implementation if a cold/weak-FT deployment needs it.
- **Encode value-warp** `w(v)=φ_k(v)/S` — the closed-form decode-linearizer (encode layer),
  ~100% of the decode residual cold at S≥8; documented, deployable, neutral on top of a
  full FT.
- **Dual-spike** — the only exact interior linearization; 2× spike cost; documented fallback.
- **N2 / analytical timing proxy / multi-spike anneal** — refuted (documented with evidence).

This is the deliverable: not a single-number accuracy headline, but a **complete,
first-principles, experimentally-grounded theory** of why cascaded single-spike TTFS lags,
exactly what is and isn't recoverable, the deployable primitives, and the irreducible floor
— turning a "fragile dead-end" into a quantified, understood engineering trade.
