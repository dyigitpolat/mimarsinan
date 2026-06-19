# Direction E — Effective-depth reduction via skip / residual routing (H5)

**Hypothesis (H5).** The cascaded single-spike TTFS death-cascade is exponential
in DEPTH (each hop attenuates; attenuation compounds multiplicatively; deep
layers starve to ~0). If a deep layer ALSO receives a copy of an EARLY layer's
spike — which fired with low latency, long ramp window, faithful value — that
early spike survives the cascade and bypasses the attenuating chain, reducing the
*effective depth* of the surviving signal path and extending the depth budget
`d_max ≈ T`.

**Verdict: PARTIAL.** The mechanism is real and measurable — skips demonstrably
revive the per-layer attenuation of the layers they feed (the death cascade is
broken on the surviving path). But on cold conversion this does NOT recover
end-task accuracy, because the **final decode (classifier) layer always sits at
maximum depth and dies regardless of interior skips** — nothing skips *past* the
decode. The depth-reduction win materialises only when the skip is **trained
through the genuine cascade**: a skip that routes the faithful early spike into
the decode layer, fine-tuned on the deployed dynamics, lifts the PRIMARY
benchmark (depth-3 digits, S=8) from **0.074 (chance) to ~0.83**, and the skip's
contribution *over retraining the classifier alone* is **+0.30**. However, full
end-to-end genuine-cascade fine-tuning recovers even the plain (no-skip) net to
~continuous, so the skip's marginal value shrinks when the whole net can adapt.

Harness: `cascade_lab.py` + `tests/cascade_fixtures.py` + `src` only; float64,
deterministic, CPU. Prototype: `experiments/depth_reduction.py`. Seeds {0,1,2};
S ∈ {8,16,32}; depth ∈ {2..6}. (Run with `OMP_NUM_THREADS=2` — the shared box is
heavily oversubscribed; uncapped torch threads make timings meaningless.)

---

## 1. How a skip maps to deployable hardware (the load-bearing question)

A skip is built as `torch.cat([deep_hidden, early_hidden])` feeding the next
linear layer — **NOT** a value-domain residual add `b + a`.

| construction | converter node | segments | deployed meaning |
|---|---|---|---|
| `torch.cat([h, a])` | `ConcatMapper` (transparent routing) | **1** (verified) | extra fan-in into the consumer core; the early core's already-emitted spike is routed to a 2nd destination |
| `b + a` (value add) | `ComputeOpMapper` (host op) | 2 (CUT) | decode → host add → re-encode: a sync point + a re-encode — **rejected** |

So a concat-skip is, on chip:

- an **extra fan-in / wider weight block** into the consumer core, fed by routing
  the EARLY core's spike to one more destination;
- **no extra spike** (the early neuron still fires exactly once; its single spike
  is fanned out to an additional core);
- **no host op, no per-layer sync** — it lives inside the same spike segment as
  transparent routing;
- bounded only by the per-core max-axon (fan-in) budget — the same resource the
  mapper already coalesces.

**Category:** architecture + trained params. Fully deployable; within standard
crossbar capability (extra columns / extra axons). The strongest variant routes
the *input (encoding-layer) spikes* — which fire earliest and are fully faithful
— to every core; that is just fanning the input axons out more widely.

A **functional** `torch.relu` skip is a trap: the converter does not pattern-match
it, silently degrades the whole net to host ComputeOps (0 perceptrons, segs=0,
*no cascade at all*) and reports a false `gen==cont`. All skip layers must be
`nn.Linear` + `nn.ReLU` **module instances**; the prototype asserts
`perceptrons>0 and segments==1` to guard this.

---

## 2. The mechanism is real: skips revive interior attenuation (random init)

Per-depth attenuation ratio (genuine-cascade decoded mean / continuous-teacher
mean; 1.0 = faithful, → 0 = dead), random init, seed 0, depth 4:

| arch | S=8 | S=16 |
|---|---|---|
| plain | `[0.75, 0.025, 0.0, 0.0]` | `[0.87, 0.26, 0.001, 0.0]` |
| input-skip (hidden) | `[0.76, 0.92, 0.36, 0.0]` | `[0.88, 1.58, 1.06, 0.058]` |
| dense-skip | `[0.70, 0.58, 0.30, 0.0]` | `[0.84, 1.24, 0.94, 0.19]` |
| **input-skip-ALL** | `[0.76, 1.07, 0.34, 0.45]` | `[0.88, 1.79, 1.11, 1.33]` |

Reading: plain dies to 0 by depth 2. A skip pulls the layers it feeds back up
(0.025 → 0.92 at depth 1). `input-skip-ALL` (every layer reads the input) keeps
even the **last** layer alive (0.0 → 0.45 at S=8). At depth 6 / S=8,
input-skip-ALL holds the deepest layer at 0.14 where plain is 0.0. Ratios > 1 are
overshoot from the extra fan-in raising the decoded value above the teacher mean
— a different but faithful (non-attenuated) representation. **The death cascade is
broken on the skip-fed path; effective depth → ~1 for the surviving signal.**

---

## 3. But cold conversion does NOT recover end-task accuracy (E2 / E3)

The mechanism revives interior layers, yet trained-weight cold-conversion gen_acc
stays at chance for depth ≥ 3 at S=8 — because the **classifier (decode) layer is
the deepest hop and still dies**, and accuracy is gated by the decode.

**E2 (depth=3, trained weights, mean of seeds 0–2):**

| arch | S | cont | gen | reten | atten (seed0) |
|---|---|---|---|---|---|
| plain | 8 | 0.782 | 0.077 | 0.10 | `[1.0, 0.28, 0.0]` |
| input-skip | 8 | 0.740 | 0.077 | 0.10 | `[1.0, 0.45, 0.0]` |
| input-skip-ALL | 8 | 0.570 | 0.082 | 0.16 | `[1.0, 0.43, 0.0]` |
| dense-skip | 8 | 0.682 | 0.077 | 0.11 | `[1.0, 0.44, 0.0]` |
| plain | 16 | 0.782 | 0.409 | 0.51 | `[1.0, 0.72, 0.05]` |
| input-skip | 16 | 0.740 | 0.339 | 0.45 | `[1.0, 0.97, 0.10]` |
| dense-skip | 32 | 0.682 | 0.473 | 0.70 | `[1.0, 1.34, 0.43]` |

The middle-layer atten is lifted by skips (e.g. 0.28 → 0.45 at S=8) but the LAST
entry stays ~0 at S=8/16 in every arch → gen stays at chance. The win only shows
at high S where the decode itself survives.

**E3 (depth-budget table, gen_acc, mean seeds 0–2; chance = 0.10):**

| S=8 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|
| plain | 0.609 | 0.077 | 0.077 | 0.077 | 0.077 |
| input-skip | 0.609 | 0.077 | 0.077 | 0.077 | 0.077 |
| input-skip-ALL | 0.651 | 0.082 | 0.077 | 0.077 | 0.077 |
| dense-skip | 0.609 | 0.077 | 0.077 | 0.077 | 0.077 |

| S=16 | d2 | d3 | d4 | d5 | d6 |
|---|---|---|---|---|---|
| plain | 0.666 | 0.409 | 0.077 | 0.077 | 0.077 |
| input-skip | 0.666 | 0.339 | 0.077 | 0.077 | 0.077 |
| input-skip-ALL | 0.672 | **0.129** | 0.129 | 0.077 | 0.077 |
| dense-skip | 0.666 | 0.247 | 0.077 | 0.077 | 0.077 |

**Cold-conversion d_max is NOT meaningfully extended.** At S=8 every arch
collapses for d ≥ 3. At S=16, input-skip-ALL is the only arch nonzero at d4
(0.129 vs 0.077), a marginal +1 extension. The decode-layer bottleneck is the
ceiling for cold conversion.

---

## 4. The real win: train the skip through the genuine cascade (E1)

When the skip is **trained on the deployed dynamics** the picture flips. E1:
freeze a continuous backbone, add a concat skip layer0 → classifier whose skip
sub-weights start at ZERO (continuous output unchanged), then train ONLY those
skip weights through the genuine single-spike cascade (boundary STE). CONTROL =
identical protocol with no skip, retraining the classifier weights instead. The
gap is the skip's specific contribution.

**E1 (depth=3, backbone frozen, mean seeds 0–2):**

| | S=8 | S=16 |
|---|---|---|
| gen_before (dead cascade) | 0.121 | 0.425 |
| **skip_after** (train skip-to-decode) | **0.840** | **0.877** |
| ctrl_after (retrain classifier, no skip) | 0.527 | 0.858 |
| **skip-specific gain (skip − ctrl)** | **+0.314** | +0.019 |

Per-seed at S=8: gen_before 0.074 / 0.208 / 0.082; skip 0.829 / 0.788 / 0.903;
control 0.545 / 0.577 / 0.458.
**The PRIMARY benchmark (depth-3 digits, S=8) goes from 0.074 (chance) to ~0.83.**
The skip gives the decode a faithful short-effective-depth path that retraining
the dead-feature classifier alone cannot reach (+0.30 at S=8). At seed 0 the
skip-revived genuine cascade (0.829) even exceeds the frozen continuous backbone
(0.579) — the skip hands the classifier direct, un-attenuated input features that
the deep backbone was destroying. This is H5 confirmed for the deployable path,
and it is skip-specific (the control isolates "just more training").

---

## 5. Honest caveat: full fine-tuning closes most of the gap WITHOUT skips (E4)

E4: cold-convert, then fine-tune the WHOLE net end-to-end through the genuine
cascade (boundary STE). depth=3, mean seeds 0–2:

| arch | S | cont | gen_cold | gen_ft | ft_lift |
|---|---|---|---|---|---|
| plain | 8 | 0.782 | 0.077 | **0.744** | +0.667 |
| input-skip | 8 | 0.740 | 0.077 | 0.697 | +0.620 |
| input-skip-ALL | 8 | 0.570 | 0.082 | 0.572 | +0.490 |
| dense-skip | 8 | 0.682 | 0.077 | 0.630 | +0.553 |
| plain | 16 | 0.782 | 0.409 | 0.824 | +0.415 |
| input-skip-ALL | 16 | 0.570 | 0.312 | **0.845** | +0.534 |
| dense-skip | 16 | 0.682 | 0.247 | 0.803 | +0.555 |

When every weight can adapt to the cascade, the **plain net recovers to ~its
continuous accuracy** (0.744 @S=8 vs cont 0.782) — the optimizer reshapes deep
layers so they fire usefully, defeating the death cascade without any skip. Skips
do **not** beat plain at depth 3 under full FT (they are ≤ plain at S=8, marginally
> plain at S=16). This is consistent with the field note that the real mmixcore
pipeline trains *through* the genuine forward and only shows ~3pp: full
genuine-cascade FT is already most of the cure at shallow depth.

So the skip's net value is largest in the **limited-adaptation regime** (frozen /
partially-trainable backbone, E1), and modest under full end-to-end FT (E4).
Whether skips beat full-FT at LARGER depth (where full FT may not recover the
exponential death) is the open question (§7) — depth 4+ continuous training in
this fixture is itself poor (final-ReLU classifier saturation, see §6), which
confounds the high-depth comparison and prevented a clean d_max-extension readout.

---

## 6. Confounds / pitfalls found (recorded so they aren't re-stepped-on)

1. **Final-ReLU classifier saturation.** The fixture applies ReLU on logits (every
   layer is a perceptron). A skip routed *directly into the classifier* drives
   pre-ReLU logits strongly negative (mean −5, 7 % positive), so ~47 % of test
   rows have all-zero logits → argmax degenerates → continuous acc craters to
   ~0.47. This is a *continuous-training* pathology, unrelated to the cascade. It
   also makes plain depth-4+ train poorly (cont ≈ 0.46), confounding high-depth
   comparisons. Mitigation: skip into hidden layers, not the classifier (E2); for
   the decode skip, train it (E1) rather than init-and-convert.
2. **Functional-relu skip trap** (§1): silently produces a non-cascade; asserted
   against.
3. **Thread oversubscription:** load-avg ~500 on the shared box; uncapped torch
   (64 threads/proc × parallel agents) made a 1.5 s job take > 120 s. Cap to 2.

---

## 7. Will this transfer to the real mmixcore pipeline?

**Partially — and the most transferable finding is a caution, not a silver bullet.**

- The deployment mechanism is sound and faithful: concat-skips are in-segment
  transparent routing (`ConcatMapper`), no host op, no extra spike, just extra
  fan-in — directly expressible in the existing mapper and within chip capability.
  So a skip-augmented architecture is *deployable as-is*.
- The cold-conversion result (skips revive interior atten but not end-task acc,
  because the decode layer dominates) should transfer structurally: it predicts
  that **adding a skip into the output/decode segment** is the part that matters,
  and that interior skips alone won't move deployed accuracy.
- The E1 win (skip-to-decode trained through the cascade, +0.30 over no-skip)
  should transfer to the regime mmixcore actually faces *if* it has hard-to-
  retrain deep segments (offload/host-ComputeOp boundaries sever the genuine
  backward — see the cascade-boundary-grad-severance note — so deep offload
  segments are exactly the "frozen backbone" regime where skips help most).
- The E4 caveat tempers expectations: mmixcore already fine-tunes through the
  genuine forward and only loses ~3pp, so on *shallow* mmixcore models a skip may
  add little. The toy is shallow too; the depth-4+ comparison that would prove a
  d_max extension is confounded by fixture trainability. The honest expected gain
  on the current shallow real pipeline is small; the lever is most promising for
  **deeper future models and for segments whose genuine gradient is severed**.

**Net:** a real, deployable, first-principles mechanism (effective depth → 1 on
the skip-fed path), with a sharply scoped applicability: route the early/faithful
spike into the **decode segment** and train it through the cascade; expect the big
win where deep segments cannot otherwise adapt, and a modest one where full
genuine-cascade FT is already available.

---

## 8. Next steps

1. **Skip into the decode segment, end-to-end FT** (the E1 mechanism + E4
   training): add an input→classifier concat skip to the PlainMLP and fine-tune
   ALL weights through the cascade; does it beat plain-FT at depth 4–6 where the
   death cascade is steepest? (E4 here only covered depth 3.)
2. **Fix the fixture's high-depth trainability** (drop the final-ReLU on logits in
   a research-only variant, or add BN) so the d_max-extension comparison at depth
   4–6 is not confounded — then re-read E3 as a true depth-budget law.
3. **Graduate to mmixcore:** add a concat skip from the encoding segment into the
   last neural segment (the decode) of a deeper model and measure NF↔SCM and
   deployed accuracy vs the no-skip baseline, focusing on offload/severed-gradient
   segments where E1 predicts the largest gain.

**Best config found:** input→decode concat skip, skip weights trained through the
genuine cascade (E1): depth-3 digits S=8 genuine **0.074 → ~0.83** (+0.72;
+0.30 over the no-skip control). Deployable as extra in-segment fan-in routing.
