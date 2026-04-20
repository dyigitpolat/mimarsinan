# Tuning Subsystem — Future Work

Staged follow-ups to the "smooth-adaptation" refactor. Everything in this
document is **out of scope for the minimal refactor that lives on `main`**;
each item is listed here so it can be picked up as a dedicated, self-contained
project.

Most of this work targets the same underlying issue: the current
rate-mixing formulation computes gradients through the mixed activations /
weights `w_mix = (1 - r) · w_fp + r · Q(w_fp)` and then applies those gradients
to the FP master copy `w_fp`. That is a reasonable heuristic, but the gradient
we apply does **not** correspond to the loss surface the quantized model
actually sits on. The methods below replace or extend the mixing with
formulations where the forward pass and the backward pass are mutually
consistent.

## 1. Straight-Through Estimator (STE) for quantization (highest priority)

**What**: Replace `w_mix` with the true quantized weight `Q(w_fp)` on the
forward pass, and treat quantization as identity on the backward pass
(i.e. `dQ/dw_fp ≈ 1`).

**Why**: The forward pass matches what the deployed model actually does.
The backward pass is a biased-but-stable estimator that avoids the
"which weights do I apply gradients to" ambiguity entirely. STE is the
standard QAT formulation; almost every modern low-bit training paper
builds on it. Replaces the need for rate-mixing on the weight side.

**Adoption path**:

1. Implement `QuantizeSTE(autograd.Function)` in `src/mimarsinan/models/activations.py`.
2. Add a flag `config["quantization_mode"] = "mix" | "ste"` in
   `DeploymentPipeline.default_deployment_parameters`; default stays
   `"mix"` for bit-compatibility.
3. Wire `PerceptronTransformTuner._mixed_perceptron_transform` to select
   `QuantizeSTE` when the flag is set. The rate schedule can still
   interpolate between FP and STE (`(1 - r) · w_fp + r · w_fp.detach() + r · (Q(w_fp) - w_fp).detach_STE`).
4. Contained tests: gradient magnitude remains finite and non-trivial at
   every integer rate; quantized forward is bit-identical to deployment.

## 2. Learnable Step Size Quantization (LSQ / LSQ+)

**What**: Treat the quantizer step size (currently derived from p99
statistics) as an `nn.Parameter` with its own gradient. LSQ+ adds a
learnable zero-point.

**Why**: The p99 statistic is a heuristic decoupled from the loss. LSQ
discovers a step size that is empirically optimal for the downstream
loss. Composes cleanly with STE.

**Adoption path**:

1. The learnable-clamp-scale plumbing (this refactor) gives us the
   infrastructure: `ClampDecorator.scale_param` is already an `nn.Parameter`,
   and `DifferentiableClamp` already routes gradients to its bounds.
2. Extend `QuantizeDecorator` (weight side) with a `step_param: nn.Parameter`
   and a learnable-scale gradient path analogous to `DifferentiableClamp`.
3. Register the extra parameters with the optimiser (they need to be
   visible to `model.parameters()` — the simplest path is to attach them
   to the perceptron as `perceptron.register_parameter(...)` inside
   `AdaptationManager.update_activation`, so `nn.Module.parameters()`
   picks them up automatically).
4. Freeze back to a scalar at the end of the relevant tuner (same pattern
   as `freeze_learnable_scale` in `tuning/tuners/clamp_tuner.py`).
5. Contained tests: starting step size matches p99; final step size
   minimises quantization error more than the scalar baseline on a held-out
   calibration set.

## 3. Differentiable Soft Quantization (DSQ)

**What**: Replace hard rounding with a smooth tanh/sigmoid approximation
whose sharpness β is annealed from soft (near-identity) to hard (near
rounding). Replaces rate-mixing with an annealed non-linearity.

**Why**: The forward pass is fully differentiable at every β; the hard
rounding limit is reached smoothly. Avoids the rate-jump discontinuities
that currently cause loss spikes at high `r`.

**Adoption path**:

1. Add `SoftStaircaseFunction(autograd.Function)` alongside `StaircaseFunction`
   in `src/mimarsinan/models/activations.py`, parameterised by β.
2. Replace `RateAdjustedDecorator` for activation-quantization with a new
   `BetaAnnealedQuantizeDecorator` whose internal β is controlled by the
   `SmartSmoothAdaptation` loop (rate → β scheduler).
3. The orchestration loop stays unchanged; only the rate-to-parameter
   mapping changes. Endpoint invariants (β = 0 ≈ identity,
   β = β_max ≈ hard staircase) are preserved.
4. Contained tests: staircase error decreases monotonically with β;
   hard-limit output matches `StaircaseFunction` output within tolerance.

## 4. AdaRound

**What**: Learn per-weight rounding directions (up or down) via a
relaxation, instead of always rounding-to-nearest.

**Why**: Round-to-nearest minimises per-weight error but not the downstream
loss. AdaRound is a surprisingly large win at aggressive bit widths (≤ 4 bits)
and is data-efficient — it usually needs only a calibration batch.

**Adoption path**:

1. Implement AdaRound as a post-training calibration step (between
   Weight Quantization and Normalization Fusion in the pipeline).
2. Per-layer: freeze the STE-trained integer weights, learn a rounding
   residual `V ∈ [0, 1]` per weight via MSE-to-FP-activations loss,
   produce a final hard round.
3. No change to the smooth-adaptation loop; AdaRound is a one-shot
   refinement on top.
4. Contained tests: calibration set error ≤ round-to-nearest error on the
   same weights.

## 5. BRECQ (Block-Reconstruction Error)

**What**: Replace (or complement) the end-to-end loss during QAT with
block-wise reconstruction: minimise the L2 distance between block outputs
under quantization and under full precision.

**Why**: End-to-end QAT gradients are noisy at low bit widths. Block-wise
reconstruction is a much more stable signal and has been shown to match
end-to-end QAT accuracy with far less calibration data.

**Adoption path**:

1. Add a new pipeline step `BlockReconstructionStep` after Weight
   Quantization. It iterates over "blocks" (for the Mimarsinan supermodel,
   each `Perceptron` is the natural unit).
2. Per block: freeze other weights, minimise L2 reconstruction loss, step
   the block's quantized parameters.
3. Composes with AdaRound (use BRECQ loss as the AdaRound objective).
4. Contained tests: block output MSE strictly decreases under BRECQ
   iterations; end-to-end test accuracy ≥ the STE baseline.

## 6. Self-distillation from the FP master

**What**: Keep the pre-adaptation FP model as a frozen teacher, and add
a KL-divergence distillation loss to every tuning cycle.

**Why**: One of the cheapest and most reliable wins in QAT. The teacher
supplies a smooth target that is *by construction* consistent with the
FP forward surface, which steadies the quantized student during rate
transitions.

**Adoption path**:

1. Capture a frozen copy of the model at the beginning of the first
   quantization-related step (e.g. at the end of Clamp Adaptation).
2. Add a `distillation_loss` term to `CustomClassificationLoss`; its
   weight is a config knob that defaults to 0 (opt-in).
3. Every SmoothAdaptationTuner that trains weights picks up the loss
   automatically via the trainer's recipe.
4. Contained tests: on a saturated-accuracy benchmark, enabling distillation
   raises the post-quantization accuracy by ≥ 0.5 percentage points (the
   threshold can be tuned per dataset).

## 7. Smoother activation-scale estimator

**What**: Replace the hard `max(|x|)` scale estimator (used inside some
quantizer variants) with either:

- an EMA across steps (`scale ← 0.99 · scale_prev + 0.01 · current_max`), or
- a p-norm proxy (`(mean(|x|^p))^(1/p)` with `p ≈ 8–20`).

**Why**: Hard max gives a non-zero gradient only to the argmax element,
which is noisy. A smoother estimator tracks the same distribution but
changes gradually, which composes better with annealed quantizers.

**Adoption path**:

1. Activation-scale is currently computed *offline* (in Activation Analysis)
   from p99, which is already smoother than max. This item is relevant only
   if / when we replace the activation-analysis scale with an online
   estimator during QAT.
2. If adopted, the estimator lives next to `ClampDecorator.scale_param` and
   updates the reference value used by `clamp_scale_regulariser`.

## Cross-cutting design notes

- **Test-set isolation**: All of the above must continue to respect the
  single-measurement rule established by this refactor — `trainer.test()` is
  only ever called from `PipelineStep.pipeline_metric()`, never from inside
  a tuner body.
- **Cross-step accuracy budget**: The `AccuracyBudget` component
  (`src/mimarsinan/pipelining/accuracy_budget.py`) monitors cumulative
  drift; any new quantization formulation must be auditable against it.
- **Per-layer rate schedule**: Already plumbed (opt-in via
  `config["per_layer_rate_schedule"]`). Sensitivity-aware layer warping
  composes cleanly with STE and LSQ.
- **Backward compatibility**: Every item above should ship behind an opt-in
  config flag whose default reproduces the current behaviour. Two reasons:
  (a) reproducibility of existing deployment runs; (b) bisection between
  variants when new methods don't immediately improve accuracy.
