# Direction C — Timing-aware differentiable proxy with matched gradients (true D2)

**Hypothesis (H3).** A differentiable proxy whose forward approximates the
*expected ramp-integrate single-spike decode* — modelling sub-window spike
**timing**, not just a pointwise staircase — has a gradient that points in the
genuine cascade's local improvement direction, so training through it and
deploying the genuine cascade beats cold-conversion (unlike the failed staircase
and crude soft-sigmoid STE).

**Verdict: REFUTED for the analytical-proxy family.** A timing-aware proxy with a
*bit-exact forward* exists and is easy to build, but its smoothed gradient is
**mis-conditioned**: training through it leaves a fully-dead cascade unchanged and
**actively regresses** a partially-alive one (depth=2 S=8: cold 0.505 → proxy
0.224; S=16: cold 0.260 → proxy 0.089). The well-conditioned "true D2" is **not a
separate analytical proxy** — it is back-prop through the *genuine cascade
simulation itself* (the project's existing `boundary_surrogate_temp` STE), which
recovers most of the gap (S=16: 0.260 → 0.777; S=32: 0.646 → 0.815).

Artifacts: `experiments/proxy.py` (the proxy + a standalone genuine per-layer
reference; `python proxy.py` reproduces Tier 1). All numbers below are float64,
CPU, sklearn-digits, 3 seeds (0,1,2), no full pipeline.

---

## 1. The decode the proxy must model (first principles, re-derived)

A single TTFS neuron's genuine decode is **not** a pointwise function of its ReLU
pre-activation. Tracing `TTFSActivation` + `TtfsSegmentPolicy` exactly:

* input `i` carries value `v_i`, encoded as ONE spike at the **global** cycle
  `tau_i = round(S·(1 − v_i))` (high value → early spike);
* the consumer opens its integration window at cycle `L` (= its cascade
  latency = perceptron-hops from the segment entry) and runs `S` cycles `[L, L+S)`;
* at local time `s = t − L` the ramp current is
  `R(s) = Σ_i Weff_i · 1[L ≤ tau_i ≤ L+s]` and the membrane is the running
  double-sum `M(s) = Σ_{u≤s} (R(u) + b/θ)`;
* the neuron fires **once** at the first `s` with `M(s) ≥ 1` (normalised θ), then
  latches; the decoded value is `(#cycles latched)/S · θ = (S − s_fire)/S · θ`.

Two structural distortions vs the continuous teacher `relu(Wv+b)`:

* **D-miss (death):** any input whose spike fires *before the window opens*
  (`tau_i < L`) is silently dropped — `1[L ≤ tau_i]` gates it out. Because high
  values fire earliest, the most informative inputs are the first to be lost as
  depth grows. This is the death-cascade, exact.
* **D-shift:** the decode is the *fraction of the window remaining after firing*
  — a ramp-position code — so it is `+L/S`-biased and clips at `v ≤ 1 − L/S`.

Single-spike, single-input verification (`w=1`, `θ=1`) makes D-miss undeniable:

| L | decode(v=0.25) | decode(v=0.5) | decode(v=0.75) | decode(v=1.0) |
|---|---|---|---|---|
| 0 | 0.25 | 0.50 | 0.75 | 1.00 |
| 1 | 0.375 | 0.625 | 0.875 | **0.00** |
| 2 | 0.50 | 0.75 | 1.00 | **0.00** |

`v=1.0` decodes to **0** for any `L>0`: its spike at global cycle 0 is missed.

## 2. The proxy (`timing_proxy_decode`)

Evaluate `M(s)` on the integer local grid `s = 0..S−1` (the cycles HW integrates)
and soften only the two non-differentiable ops:

* **soft arrival / soft death-gate:** `arrived(s,i) = σ(β_a(s − a_i)) · σ(β_a(tau_i − L))`
  with `a_i = tau_i − L` and real-valued `tau_i = S(1−v_i)` (so `tau` is
  differentiable in `v`). The second factor is the smooth death-miss gate.
* **soft fire-once latch:** `latched(s) = 1 − Π_{u≤s}(1 − σ(β_f(M(u)−1)))`
  (smooth monotone running-max; the probability of having fired by `s`).
* `decode = mean_s latched(s) · θ`.

`hard_forward=True` uses `round(tau)`, the hard window gate and a hard `cummax` —
i.e. the bit-exact deployed kernel. Categories: the soft path is a **training-time
proxy**; the deployed decode is unchanged and stays bit-exact with HCM.

## 3. Tier 1 — forward fidelity (the proxy IS faithful)

`mean over a 200×{10→8} batch`, hard proxy vs the standalone genuine layer decode
(itself validated against the project's `TTFSSegmentForward`, §5):

| S | L=0 | L=1 | L=2 | L=3 |
|---|---|---|---|---|
| 8  | **exact** | 0.998 exact, max-err 1/8 | 0.982, 2/8 | 0.971, 3/8 |
| 16 | **exact** | 0.999, 1/16 | 0.998, 1/16 | 0.994, 2/16 |
| 32 | **exact** | **exact** | **exact** | **exact** |

The hard proxy is **bit-exact at L=0** and the only residual at low S is the
`±1/S` rounding tie the genuine forward itself carries (it shrinks geometrically
with S → exact at S=32). **End-to-end** on the trained PRIMARY cascade
(`TimingProxyMLP` hard, flow-calibrated θ), the proxy reproduces the genuine
cascade *exactly*: `proxy_hard_acc = genuine_acc = 0.0742`, **argmax agreement
1.0000**. The forward model of the death cascade is correct.

## 4. Tiers 2-3 — gradient alignment and end-to-end transfer (the proxy FAILS)

### 4a. Gradient alignment is noisy and not decisive

`cos(d loss/dW)` of the soft proxy vs a finite-difference of the genuine layer
decode (single layer, seed-averaged). The genuine FD gradient is **sparse and
quantised** (the decode is a 1/S staircase), so this metric has σ ≈ 0.1-0.3 and
does not cleanly discriminate:

| L | proxy cos | continuous-teacher cos |
|---|---|---|
| 1 | +0.10 … +0.14 | **+0.24 … +0.40** (teacher wins) |
| 2 | +0.48 … +0.70 | +0.39 … +0.66 (≈ tie) |
| 3 | +0.47 … +0.55 | +0.23 … +0.55 (≈ tie) |

The proxy edges the teacher only at deep latency where D-miss dominates, and never
by more than noise. *A single cherry-picked example gave cos +0.56 vs +0.39;
seed-averaging erased it.* The per-layer cosine cannot settle H3 — only the
end-to-end transfer can.

### 4b. End-to-end transfer — the decisive table (3-seed mean, PRIMARY = digits)

Train continuous → (deploy genuine = **cold**) vs fine-tune through each proxy →
deploy genuine. `genuineSTE` = back-prop through the *genuine* cascade forward
(`cascade_forward(grad=True, surrogate_temp=1.0)`), the project's existing
mechanism, the honest "train-through-the-genuine-forward" reference.

depth=2, S=8 (cascade partially alive):

| seed | cont | cold | proxy-SOFT | proxy-STE | genuine-STE |
|---|---|---|---|---|---|
| 0 | 0.662 | 0.508 | 0.278 | 0.265 | **0.657** |
| 1 | 0.785 | 0.510 | 0.226 | 0.223 | **0.764** |
| 2 | 0.720 | 0.497 | 0.169 | 0.141 | **0.712** |

depth=3, S=8 (**PRIMARY**, cascade fully dead — output logits std = 0, argmax
pinned to one class):

| seed | cont | cold | proxy-SOFT | proxy-STE | genuine-STE |
|---|---|---|---|---|---|
| 0 | 0.944 | 0.074 | 0.074 | 0.074 | **0.152** |
| 1 | 0.750 | 0.085 | 0.085 | 0.085 | **0.117** |
| 2 | 0.655 | 0.071 | 0.071 | 0.071 | **0.182** |

Across S (depth=3 mean): cold/proxy/genuine-STE =
`S=8: 0.077 / 0.077 / 0.150`, `S=16: 0.260 / 0.089 / 0.777`,
`S=32: 0.646 / 0.346 / 0.815`. (β-annealing 1→16 changed nothing.)

**Reading the table:**

1. **proxy-SOFT and proxy-STE never help, and usually hurt.** At depth=2 they
   roughly *halve* deploy accuracy; at S=16/32 they regress it toward chance.
   STE (genuine value in the loss, soft gradient) regresses identically, so the
   damage is the **gradient direction**, not a forward-value mismatch.
2. **Fully-dead (depth=3 S=8) is a fixed point for the proxy:** deploy = cold to 4
   d.p. — the analytical gradient cannot escape the dead basin (the death-gate's
   own gradient `∝ σ'(β(tau−L))` vanishes deep in the dead region, exactly where
   a revival signal is needed).
3. **The genuine forward's own surrogate IS well-conditioned** and is the real
   answer: 3× at S=16, strong at S=32, 2× even on fully-dead S=8.

## 5. Why the analytical proxy is mis-conditioned (mechanism)

The soft proxy substitutes a *smooth* death-gate and a *smooth* fire-time for two
hard discontinuities. Smoothing the death-gate has a perverse effect: it tells the
optimizer the dead inputs "almost count", so the loss is minimised by **pushing
their values UP** (making the smooth decode larger) — but higher `v` ⇒ *earlier*
spike ⇒ further outside the window ⇒ *more* dead at deploy. The proxy's descent
direction is anti-correlated with the genuine cascade in precisely the death
regime that matters. The genuine surrogate avoids this because its forward keeps
the hard "missed → 0" fact at every cycle; only the *fire Heaviside* is softened
(ATan), so the gradient never rewards a spike the window will not see.

(Validation note: the standalone `genuine_layer_decode` in `proxy.py` is the same
cycle dynamics as `TtfsSegmentPolicy`; the full `TimingProxyMLP` hard path matches
`TTFSSegmentForward` end-to-end at argmax-agreement 1.0, so the refutation is not
an artifact of an inaccurate reference.)

## 6. Verdict, scope, transfer outlook

* **H3 refuted for analytical timing proxies.** A faithful-forward proxy exists
  (bit-exact), but no soft-gradient variant tested (soft, STE, β-annealed) gives
  a transferable optimum; they degrade deployment. The well-conditioned gradient
  is the genuine simulation's own surrogate, not an analytical stand-in.
* **Actionable redirect:** invest in the genuine-forward STE
  (`boundary_surrogate_temp`) — it already recovers 3× at S=16 here and is the
  mechanism the real `mmixcore` fine-tune uses. Tune *its* surrogate sharpness /
  curriculum rather than building a separate proxy.
* **Transfer to the real pipeline:** the *negative* result transfers cleanly —
  it warns against the tempting "analytical timing surrogate" detour. The
  *positive* control (genuine-forward STE) is exactly the real pipeline's
  existing lever, so its gains here are encouraging evidence that the real fix is
  to strengthen that path (and to attack the death cascade upstream — depth
  reduction / pre-compensation, directions B/E — so the genuine surrogate is not
  asked to revive an already-dead layer).

## 7. Falsified sub-claims / honest caveats

* The fully-dead PRIMARY (depth=3 S=8) is a *degenerate* deploy (constant output);
  no weight-only method that does not first revive the dead layer can move it. So
  depth=2 (partially alive) is the more informative discriminator, and there the
  proxy still clearly **hurts** — the refutation does not hinge on the degenerate
  case.
* A *hybrid* loss = genuine-forward CE + a hinge timing-regulariser
  `Σ relu(v_producer − (1 − L/S))²` (pull upstream values below the death cap so
  their spikes land inside `[L,L+S)`) was tried and made **no difference**
  (S=16: 0.7774 → 0.7774; S=8: 0.1503 → 0.1503). Two reasons: (i) the recorded
  node values are `.detach()`-ed, so the naive penalty had no gradient — a proper
  version needs a differentiable `tau` hook into the genuine walk; (ii) genuine-STE
  already keeps producer values near/under the cap, so the hinge is rarely active.
  The proxy's forward `tau` is exact, so a *gradient-carrying* timing penalty on
  top of genuine STE remains the one constructive use of this artifact — but it is
  not free, and the first quick attempt did not move the needle.
