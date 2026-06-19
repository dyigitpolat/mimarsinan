# 33 — The static per-neuron limit: how much of the post-G residual is calibration-recoverable?

**Phase 3 question.** After the death cascade is handled (G, the per-depth θ trim),
the genuine cascade still caps below the ideal staircase. The hypothesis (artifact
00/25): the ramp decode applies a **per-SAMPLE** gain `g_eff(τ) = (T−τ+1)/(T+1)` to a
spike arriving at local cycle `τ`; each input drives a neuron to a *different* fire
time `τ`, and the fire-time distribution differs *per neuron*. A static threshold can
invert a fixed gain but **not** a per-sample-varying one. This phase pushes the static
correction to its per-neuron limit and **bounds** what any static per-neuron
calibration can recover — to tell us whether training (N2/F) or a decode/encode
linearization is *required* for the rest.

Prototype: `experiments/per_sample.py`. Reference substrates: `cascade_lab.conversion_gap`
(genuine cascade), `capacity._staircase_acc` (the IDEAL STAIRCASE = optimal linear
single-spike decode), `char_decode_law.analytic_decode`, `closed_form.g_relative`/
`calib_fire`. Digits, float64, multi-seed (0,1,2), S∈{8,16}, depth∈{3,4}.

---

## The four progressively finer STATIC corrections

All keep the deployed ramp decode **bit-exact** (NF↔SCM parity holds). They use only
per-OUTPUT-channel θ (`activation_scale`), set **sequentially** (re-measure each layer
after correcting upstream, so it sees the revived input), then re-install the genuine
node.

| # | level | knob | data access |
|---|---|---|---|
| 1 | **per-DEPTH** (G, baseline) | 1 scalar / layer (`g_relative(S,d)`) | none (closed form) |
| 2 | **per-NEURON mean** | θ_k from each channel's MEAN calib fire-cycle (relative to layer 0; the per-channel `calib_fire`) | calib only |
| 3 | **per-NEURON L2** | θ_k that minimizes the per-sample decode **L2** to the staircase target (trades mean-bias vs the fire-time SPREAD) | calib only |
| 4 | **ORACLE / neuron** | best EVAL accuracy over {mean, L2-on-eval+grid, per-depth acc-grid} — a true upper bound on ANY static per-neuron θ | **eval** (bound only) |

Plus the load-bearing **residual decomposition** per active neuron:
- `bias_resid` = `|mean_genuine − mean_staircase|` → **closable** by a static per-neuron scale.
- `spread_resid` = std over samples of the residual *after the best per-neuron AFFINE
  (slope+offset) fit* to the staircase → **NOT closable** by ANY static per-neuron map;
  this is the irreducible per-sample-τ spread.

And a **re-encode reference** (the ENCODE-CHANGE lever, *not* a static calibration): a
per-hop decode→re-encode host boundary that resets the ramp compounding.

---

## Results (mean over seeds 0,1,2; `closed%` = fraction of the genuine→staircase gap)

```
depth=3 S=8  | cont 0.783  staircase 0.764  baseline 0.077  gap +0.688
  level             acc    closed%   bias_resid  spread_resid
  baseline         0.077      -        1.003       0.122
  per_depth (G)    0.146     10%       1.017       0.127
  per_neuron mean  0.406     48%       0.950       0.154
  per_neuron L2    0.566     71%       0.475       0.109     <- best calib-only
  oracle/neuron    0.745     97%       0.480       0.398     <- static UPPER BOUND
  [encode-change ref] per-hop re-encode genuine 0.536 (its own staircase 0.706)

depth=3 S=16 | cont 0.783  staircase 0.785  baseline 0.260  gap +0.525
  per_depth (G)    0.625     69%
  per_neuron mean  0.748     93%       0.801       0.231     <- best calib-only
  per_neuron L2    0.567     59%       0.491       0.172
  oracle/neuron    0.754     94%

depth=4 S=8  | cont 0.456  staircase 0.440  baseline 0.077  gap +0.364
  per_depth (G)    0.077      0%
  per_neuron mean  0.077      0%
  per_neuron L2    0.175     27%       0.457       0.078     <- best calib-only
  oracle/neuron    0.318     66%       0.629       0.114     <- static UPPER BOUND

depth=4 S=16 | cont 0.456  staircase 0.455  baseline 0.077  gap +0.379
  per_depth (G)    0.090      3%
  per_neuron mean  0.367     77%       1.069       0.173     <- best calib-only
  per_neuron L2    0.203     33%
  oracle/neuron    0.406     87%
```

Per-seed (depth=3 S=8): oracle closes **95 / 99 / 96 %** (seeds 1/0/2); best-calib-only
**72 / 65 / 85 %**. The bound is stable across seeds.

---

## What the numbers say

### 1. A static per-neuron THRESHOLD recovers MOST of the residual — but not all.
The oracle (the best static per-neuron θ, with eval access) closes **66–97 %** of the
genuine→staircase gap. So the post-G residual is *predominantly* a per-neuron GAIN
mismatch, and the per-neuron knob is genuinely finer than per-depth (per_depth alone:
0–69 %; per-neuron levels add 17–46 points). In the **shallow, active** regime
(depth-3) the static per-neuron θ is nearly sufficient (oracle 94–97 %, per-seed up to
99 %). **The deeper the cascade, the more it falls short:** depth-4 S=8 the oracle caps
at **66 %** — 34 % of the gap is *not* recoverable by any static per-neuron threshold.

### 2. The unclosed part is the irreducible per-sample SPREAD (hypothesis confirmed).
The residual decomposition isolates it cleanly. As corrections tighten, **`bias_resid`
collapses** (1.0 → 0.48 at L2/oracle: the per-neuron mean is fixable), but
**`spread_resid` does NOT** — it floors at ≈ 0.10–0.20 and is *unchanged* (even rises)
when the threshold chases accuracy. A static per-neuron θ is a single multiplier; it
moves the whole transfer curve up/down but cannot bend it, so the per-sample dispersion
of `g_eff(τ)` across a neuron's fire-time distribution survives by construction. This is
the direct, measured confirmation of the artifact-25 root-cause hypothesis: the residual
is a per-sample nonlinearity, not a per-neuron offset.

### 3. The DEPLOYABLE (calib-only) static ceiling: ~27–93 % — regime-dependent.
The two calib-only rules are complementary and should be **max-combined**:
- **mean-match** wins in the *healthy* regime (S=16: 77–93 %) — when the cascade is
  alive, the first moment is enough.
- **L2** wins in the *collapsed* regime (S=8: it revives a fully-dead deep cascade where
  mean-match still reads 0 — depth-4 S=8 mean 0 % → L2 27 %), because L2 also inverts the
  bias of the (few) surviving samples to re-ignite the next layer.

`max(mean, L2)` calib-only closes **71 / 93 / 27 / 77 %** across the four cells; it
trails the oracle by **~3–39 points** (smallest where the cascade is active, largest at
depth-4 S=8). So most of the *recoverable* static lever is reachable **without eval
access**, but the last slice of the static ceiling needs the metric (i.e. is not a pure
calibration).

### 4. The threshold lever and the accuracy metric diverge.
The L2 fit minimizes decode error (`bias_resid` → 0.47, the lowest) but is **not** always
accuracy-best; the accuracy-greedy oracle trades per-sample fidelity for argmax
separability (its `spread_resid` *rises* to 0.40–0.53). This is why the oracle is built
as `max-over-strategies` on accuracy, not a single L2 projection — and why a deployable
calibration (which can only see decode fidelity, not the label) cannot reach the full
oracle.

### 5. The OTHER lever (encode change) moves a different residual.
A per-hop decode→re-encode (the ENCODE-side lever, a host op — *not* a static
calibration) lifts genuine to 0.099–0.536 with its own per-segment staircase at
0.52–0.72. It helps in the collapsed regime by resetting the ramp compounding, but each
re-encoded hop is **still a one-hop genuine cascade** with its own `g_eff` distortion, so
re-encode alone does **not** reach the full staircase either. The clean linearization
(re-encode that fully removes the ramp, i.e. a staircase decode) is the only thing that
provably reaches the staircase — but that is a *decode change* (it breaks NF↔SCM
bit-exactness), out of scope for this static-calibration study and noted as the
separate F2/decode-redesign direction.

---

## The bound (the deliverable)

> **A static per-neuron threshold calibration can recover ~66–97 % of the post-death-
> cascade genuine→staircase residual** (95–99 % when the cascade is shallow/active;
> down to 66 % at depth 4). The **deployable, eval-free** ceiling is **~27–93 %**
> (`max(per-neuron mean, per-neuron L2)`, regime-dependent). The remaining **3–34 %
> is the irreducible per-sample fire-time SPREAD** — `spread_resid` never collapses
> under any static θ, while `bias_resid` does. Closing it **REQUIRES** either (a)
> genuine-forward TRAINING (N2/F — the only lever that can reshape the per-sample
> transfer, by moving the fire-time distribution itself), or (b) a decode/encode
> **LINEARIZATION** (re-encode / staircase decode — a decode change that breaks
> bit-exactness). No static per-neuron calibration suffices for the last slice.**

This sharpens the recipe ordering of artifact 25: **G (per-depth) → upgrade the
calibration to per-neuron `max(mean, L2)` (free, +17–46 pts over per-depth) → N2/F for
the irreducible spread.** The per-neuron calibration is a strict, zero-cost upgrade to G
that should ship as the calibration init before fine-tuning.

---

## Deployability — precisely which primitive, and is it allowed

| lever | chip primitive | allowed? |
|---|---|---|
| per-neuron θ (levels 1–4) | per-OUTPUT-channel `activation_scale`. **Verified bit-exact-equivalent** to scaling that neuron's row of W and b by 1/g (`max|Δ|=0` over the test set) — i.e. a per-neuron output-threshold / weight-row trim, a first-class chip parameter the genuine node already broadcasts. Decode **unchanged**. | **YES** — pure calibration; NF↔SCM parity holds. The oracle reads the eval metric so it is a *bound*, not deployable. |
| per-neuron VALUE-domain map (the nonlinear input remap) | would need a per-neuron LUT *between* cascade layers, i.e. a decode→remap→re-encode **host op** per hop. | **NO** (in-segment): a host op is the encode-change lever, not a static threshold. Reported only as the re-encode reference. |
| re-encode boundary | host ComputeOp per hop (decode/re-encode). | separate ENCODE-change direction; bounds the *other* lever, not a calibration. |

**Calib-only knobs (deployable today as the calibration step):** per-neuron `mean` and
`L2` θ, max-combined. Both are pure `set_activation_scale` per channel + re-install;
zero metric access; decode bit-exact.

---

## Caveats / honest scope
- Absolute accuracies are task-ceiling-bound (a plain depth-4 MLP underfits digits,
  cont≈0.46); the transferable metric is the **gap-closed % vs the architecture's own
  staircase**, not the absolute number.
- `closed%` divides by `staircase−baseline`; at depth-4 the baseline is at chance, so a
  few argmax flips swing the percentage — read those cells together with the residual
  decomposition (which is stable).
- The oracle is an *accuracy* upper bound over the static-θ strategies tried; a finer
  per-neuron accuracy search could lift it a little, but the `spread_resid` floor (which
  no static θ touches) is the structural ceiling and is independent of the search.
- Constants in the calib rules are the artifact-25 √S geometry; the per-neuron variants
  are data-grounded (read the calib fire-time / decode directly), so they self-fit and
  do not depend on the fitted `c`.
