# θ-aware WQ tuning — design (decide-before-build)

**Status:** DESIGN ONLY (user: "design only, then decide", 2026-07-14). Not
implemented. This memo scopes the mechanism, seams, candidate payoff, risk, and
a tests-first plan so the guarded WQ θ-freeze is only touched with eyes open.

## 1. What exists today (and why θ is frozen)

The WQ stage (`weight_quantization_step.py`) stamps the scale lattice ONCE at
entry (`compute_per_source_scales(model.get_mapper_repr())`, `:50`) and then
`_freeze_exact_qat_theta(model)` freezes any exact-QAT in-loop θ (`:47`). The
endpoint recovery (`NormalizationAwarePerceptronQuantizationTuner.
_post_stabilization_hook` → `run_endpoint_recovery`) trains a FLOAT aux model and
**reprojects** it to the chip integer grid each rate-step
(`_apply_rate` → `_update_and_transform_model`). The projection
(`NormalizationAwarePerceptronQuantization.transform`) and the effective-weight
fold both read θ via the entry-stamped `per_input_scales`.

The freeze is deliberate. Its docstring records the failure of NOT freezing:
> "a theta trained under the WQ endpoint drifts the lattice out from under the
> projection (torch<->deployed 0.9688 vs 1.0000; leaving it trainable floods the
> projection with degenerate-channel bias routing and OOMs the graph)."

I.e. θ moving while the lattice is stale → the fake-quant forward and the
integer projection disagree → parity breaks. **The freeze is correct GIVEN a
stale lattice.** θ-aware WQ = make the lattice track θ instead of freezing θ.

## 2. The mechanism

Make the freeze a POLICY, default frozen (byte-identical), opt-in θ-aware:

1. `wq_theta_aware` knob (BOOL, default off). When off, `_freeze_exact_qat_theta`
   runs exactly as today.
2. When on: DO NOT freeze the exact-QAT θ. Instead, **re-stamp the lattice
   before every reprojection** — call `compute_per_source_scales(mapper_repr)`
   at the top of `_apply_rate` / `_apply_rate_to` (before the transform), so the
   `per_input_scales` the projection reads always reflect the CURRENT θ. The
   fake-quant float forward and the integer projection then share one θ ⇒ parity
   holds by construction (the same guarantee the sync R7 contract used: one read
   site feeds both sides).
3. θ joins the recovery optimizer (it is already a `requires_grad` Parameter from
   the AQ promote; just skip the freeze) under the ratchet backward — so the
   collapse-hardening carries into WQ.

The re-stamp is the crux: it is exactly the missing step the freeze docstring's
0.9688 failure implies. Frozen-θ WQ = stamp once; θ-aware WQ = re-stamp per
reprojection.

## 3. Candidate payoff (honest, and in tension with a keystone)

The measured ttfsq S8 mixer deploy gap is torch(fake-quant) 0.9692 → spiking
0.9595, with NF↔SCM parity PASSED (soft==hard). That ~1pp is the fake→real
integer-weight-rounding gap. θ is FROZEN across the recovery, so it cannot
re-adapt to the integer weights. θ-aware WQ could let θ absorb some of that
rounding — potentially lifting deploy from ~0.9595 toward ~0.9692.

**Tension to respect:** the keystone `[[mixer_nf_scm_wq_residual_resolved]]`
established this WQ-rounding residual as HONEST/inherent (budget ~0.15), NOT a
per-source-scale STE bug, and told us not to re-chase it. θ-aware WQ is a
DIFFERENT lever (co-adapt θ, not fix a scale bug), so it is not literally the
refuted move — but the keystone's spirit says the payoff is uncertain and may
just re-confirm the residual is irreducible. This is the 8×-measured
post-QAT-inversion risk again: an isolated "θ can absorb rounding" argument may
invert on the trained composition.

**Where it CANNOT help:** the mixer *gate* miss is 1/S composition distortion
(−1.91pp at S=8), which is orthogonal to WQ rounding — θ-aware WQ does not touch
it. So even a best-case WQ-rounding recovery leaves the S8 mixer below gate; the
S64 respec still stands. The realistic win, if any, is a fraction of a point on
cells whose deploy gap is WQ-rounding-dominated and NOT 1/S-floored — a narrow
set (higher-S exact-QAT cells; some non-mixer WQ residuals that are already
near-lossless).

## 4. Risk & cost

- **Parity (primary gate):** the torch↔deployed-sim parity must stay 1.0000. The
  re-stamp-per-reprojection design makes it hold by construction; a test must
  PROVE it on a ttfsq and a sync cell (the sync per-channel-θ mapper-forward
  routing bug still applies — sync θ-aware WQ stays scalar until that is fixed).
- **Degenerate-channel routing:** still fires (handled, telemetry-gated). Not a
  crash; the routing count is a diagnostic.
- **Cost:** `compute_per_source_scales` is an O(graph) walk per reprojection. The
  recovery reprojects per rate-step; re-stamping every step is the exact but
  priciest option. A periodic re-stamp (every N steps) is cheaper but lags θ
  between stamps — and a lagged lattice is exactly the 0.9688 failure mode, so
  periodic re-stamping must be validated against parity, not assumed safe.

## 5. Tests-first plan (if built)

1. `wq_theta_aware` off ⇒ byte-identical (golden snapshot unchanged; θ frozen).
2. On ⇒ θ is NOT frozen at WQ; `compute_per_source_scales` is called before each
   reprojection (spy/assert); the projection reads the re-stamped scales.
3. Parity: torch↔deployed-sim == 1.0000 on a ttfsq cell with θ-aware WQ on (the
   exact-QAT identity must survive the moving lattice).
4. A/B on a WQ-rounding-dominated cell (NOT 1/S-floored) — arm only if it lifts
   deploy with no strict regression (same discipline as every other exact-QAT
   lever this campaign).

## 6. Recommendation

Buildable and low-blast-radius (opt-in, parity-gated, default byte-identical).
But the payoff is genuinely uncertain and structurally capped (cannot cross the
1/S floor that binds the cells we care about), and it re-enters territory a
keystone already flagged as inherent. Suggested order if pursued: build the
opt-in re-stamp policy, prove parity, then a SINGLE A/B on the best WQ-rounding
candidate — and be ready to fail-toward-measured (keep default-off) as with
ttfsq/sync scalar-θ, the KD teacher, and casc_exact_qat.
