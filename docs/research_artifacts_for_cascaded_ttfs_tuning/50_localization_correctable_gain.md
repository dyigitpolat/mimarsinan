# 50 — Re-localizing the cascaded-TTFS gap: it is a correctable FIRING-GAIN, not quantization and not a quadratic decode

This round **corrects two central claims of `40_final_synthesis.md`** with direct,
bit-exact measurements on the isolated harness (`experiments/lif_vs_ttfs.py`,
`decode_curve.py`, `mechanism.py`, `theta_calibrate.py`, `revive_ft.py`,
`greedy_theta.py`, `chase_2pp.py`; GPU, `_DeepMLPBN`/`_SingleSegmentMLP` on digits).
The trigger was the user's claim:

> "not fundamental quantization because LIF wouldn't reach 97%."

That claim is **correct**, and pinning down *why* overturns the prior "irreducible
per-sample floor from a quadratic decode" verdict.

---

## Finding 1 — LIF-rate ≡ TTFS-staircase, bit-for-bit. It is NOT quantization.

Same converted flow, same calibration; install LIF rate nodes vs TTFS analytical
staircase (both are T+1-level per-layer quantizers):

```
 d   S |   cont  LIF_rate  TTFS_stair  TTFS_gen |  stair-LIF
 3   8 |  0.983     0.981       0.981     0.102 |     +0.000
 6   8 |  0.981     0.915       0.915     0.106 |     +0.000
 6  16 |  0.981     0.974       0.974     0.106 |     +0.000
 9   8 |  0.965     0.742       0.742     0.102 |     +0.000
 9  16 |  0.965     0.952       0.952     0.102 |     +0.000
```

`stair − LIF = +0.000` at every depth×S. The cold quantizers are *identical*.
So the per-neuron `log2(T+1)`-bit code is shared; whatever makes cascaded TTFS
fail is **not** the quantization. (Note the *true* ceiling is the staircase/LIF
number: at d=9 S=8 it is 0.742 — the T+1-level floor *does* compound with depth at
low S; lossless at d=9 needs S≥16, where staircase=0.952. LIF's real-pipeline 0.97
is at that higher-S operating point.)

## Finding 2 — The deployed ramp decode is LINEAR and lossless, not quadratic.

`decode_curve.py` (A): a 2-layer identity chain, sweep input v; the genuine
single-spike consumer's decoded output is `v` (T+1-quantized), exactly:

```
 v_in  L1_decoded   linear=v   quad=v^2
 0.250     0.2500      0.250      0.062
 0.500     0.5000      0.500      0.250
 0.750     0.7500      0.750      0.562
```

The prior "triangular/quadratic ramp gain `R(τ)=k(k+1)/2`" modelled the
**sub-threshold membrane accumulation**, but the neuron **fires at the threshold
crossing** — its *output* is the fire-time, which decodes linearly. For a single
input the cascade is exact. The quadratic-decode premise of `40_final_synthesis`
(and the encode value-warp `φ_k` built to invert it) is therefore mis-targeted.

## Finding 3 — The real distortion: greedy partial-sum (causal) firing.

`mechanism.py` on a trained d=3 cascade, per layer (genuine vs staircase):

```
layer  gen_mean  stair_mean  median(g/s)   corr
    0    0.7677      0.7677       1.0000  1.000   <- encoding entry: exact
    1    0.3704      1.5669       0.0000  0.812   <- >half the neurons DEAD (g/s median 0)
    2    0.0000      2.5131       0.0000  0.000   <- starved -> totally dead
```

A consumer fires on the **first cycle its RUNNING partial sum crosses θ**. With
multiple inputs arriving at different fire-times and **mixed-sign ReLU weights**,
that crossing happens on an *incomplete* sum — unlike the staircase/LIF, which use
the *complete* weighted sum (LIF: the full sum is present every cycle, which is
exactly why LIF cold ≈ staircase). The result is not random: corr≈0.81 at L1 → a
mostly-**monotonic gain** distortion that under-drives the layer, half its neurons
never reach θ, and the next layer is starved to death. *Depth turns a per-layer
gain deficit into a death cascade.*

## Finding 4 — It is a CORRECTABLE GAIN: oracle per-layer θ recovers d=3 fully.

`mechanism.py` (4): brute-force per-layer threshold-scale γ_d (smaller θ → earlier
fire → higher decode):

```
d=3 S=8: cont=0.944  baseline_genuine=0.074  ->  oracle_theta=0.909  (γ=[0.5,0.35,0.5])
d=4 S=8: cont=0.458  baseline_genuine=0.074  ->  oracle_theta=0.440  (coarse 6-pt grid)
```

A **static, deployable per-layer θ-trim** lifts d=3 from chance (0.074) to **0.909
== the staircase (0.915)**. The collapse carries no lost information — it is a gain
miscalibration. (Prior heuristics capped ~0.41 and the prior synthesis read the
cap as an *irreducible floor*; it is not — the ceiling is the staircase.)

## Finding 5 — Why gradient θ-calibration cannot reach the ceiling (dead-neuron gradient).

`theta_calibrate.py` — freeze weights, optimize per-neuron θ by gradient through
the (differentiable) genuine cascade:

```
 d   S |  stair   cold  theta-calib(grad)  full-FT
 3   8 |  0.981  0.102      0.660            0.944
 6  16 |  0.974  0.106      0.085            0.957
 9   8 |  0.742  0.102      0.074            0.926
```

Gradient θ-calibration stalls (0.66 ≪ oracle 0.909 at d=3) and **collapses at
depth** (≤chance for d≥6). Cause: a **dead neuron emits no spike → no surrogate
gradient → it can never be revived by descent.** The oracle found the recovering θ
precisely because brute force needs no gradient. ⇒ the firing regime must be
restored by a **data-grounded (non-gradient) revive**, *then* FT can polish.

## Finding 6 — Full genuine-cascade FT is the working automatic recipe; revive HURTS.

`revive_ft.py`:

```
 d   S |  stair   cold  revive  revive+FT  FT-alone
 3   8 |  0.981  0.102   0.091      0.900     0.944
 6  16 |  0.974  0.106   0.074      0.074     0.957
 9  16 |  0.952  0.102   0.106      0.074     0.826
```

- **FT-alone (genuine-cascade fine-tune) reaches 0.92–0.96** — the surrogate (ATan)
  gradient flows through non-firing neurons, and *weight/bias* updates (far richer
  than θ-only) revive AND adapt the cascade. This is the deployable automatic recipe.
- **revive HURTS**: `calibrate_revive` over-shrinks (×0.6 up to 40 iters → θ→floor →
  neurons saturate at τ=0 → signal destroyed) and, fatally, it shrinks the LAST
  layer, which the oracle keeps at γ=1.0. revive+FT collapses to chance at d≥6.
- This re-confirms the prior "FT reaches ~0.93–0.95" *number*, but with the corrected
  *mechanism* (firing-gain, not a quadratic-decode floor) and the corrected ceiling
  (the staircase, recoverable — see Finding 7).
- Regularized FT (`chase_2pp.py`, KD+cosine+SWA) lands at the genuine ceiling, not
  past it — d=6 S=16: SWA **0.9425** vs staircase 0.974 (≈ stair − 4pp, the Finding-8
  floor). The regularizers stabilise the optimisation; they do not break the floor.
- At the real depth (d=9 S=16, the 9-deep-mmixcore operating point), FT-alone is
  unstable at 600 steps (0.826) but regularized FT recovers to the genuine ceiling:
  **KD (earlystop) = 0.9258**, SWA = 0.9072 — vs staircase 0.952 (~2.6pp residual).
  KD (continuous-teacher distillation) is the strongest single regulariser at depth.
  The depth instability is optimisation, not capacity.

## Finding 7 — A correctly-found static per-layer θ ALONE matches FT (no weight training).

The Finding-4 oracle was on `_SingleSegmentMLP`; re-run on the **exact ft_budget
`_DeepMLPBN` flow** (d=3 S=8): oracle per-layer θ → **0.944 == FT-alone**, with
**γ=[0.15, 0.35, 1.0]** (last layer untouched — exactly what revive gets wrong).
So a static, deployable per-layer θ calibration, *found correctly*, equals full FT
on the toy WITHOUT training a single weight. The open piece is the SEARCH: brute
force is O(grid^depth); the deployable form is a **greedy O(depth·grid) coordinate
descent** calibrated on a train batch (`greedy_theta.py`).

```
 d   S |  stair   cold  greedy-θ  greedy+FT  FT-alone | gamma
 3   8 |  0.981  0.102     0.944      0.879     0.944 | [0.15,0.35,1.00]
 6  16 |  0.974  0.106     0.794      0.735     0.957 | [0.15,0.15,0.15,0.35,0.50,1.00]
 9  16 |  0.952  0.102     0.108      0.558     0.826 | [1,1,1,1,1,0.15,0.10,1,1]
```

- **d=3: calibration-only greedy-θ == FT-alone (0.944), zero weights trained.** The
  collapse is fully correctable by a static per-layer θ when the cascade is shallow.
- **At depth the correction is JOINT, not greedy.** Coordinate descent hits a local
  min (d=9 stuck at chance — fixing one layer can't revive a coupled death cascade);
  the exponential oracle works, greedy doesn't. Gradient FT optimises all layers
  *jointly*, which is why **FT-alone is the robust automatic recipe (0.92–0.96)**.
- Calibration pre-steps (revive, greedy-θ) **hurt** subsequent FT (they land in a
  basin FT can't improve: greedy+FT < FT-alone everywhere).

## Finding 8 — The residual genuine→staircase gap (~4pp) is S-INDEPENDENT.

Frozen-weight, same-weights, d=3 (greedy-θ genuine vs staircase):

```
  S   stair  greedy-θ  residual
  8   0.981     0.944     0.037
 16   0.985     0.939     0.046
 32   0.981     0.935     0.046
 64   0.983     0.929     0.054
```

The residual does **not** shrink with S — it is a roughly **S-independent ~4pp
structural penalty of greedy partial-sum (causal) firing** vs the complete-sum
staircase. More timing resolution does not buy it back. This is the true floor; it
is closed only by removing the greedy firing: the **synchronized** schedule
(complete sum, ≥0.97) or a **multi-spike** code (averages toward the LIF rate code).

---

## Revised verdict (supersedes 40_final_synthesis §"bottom line")

The prior synthesis's headline NUMBER (cascaded caps ~3.5pp below LIF; synchronized
is the lossless path) **survives** — but its mechanism and its death-cascade verdict
are **corrected**:

- **NOT quantization** (LIF ≡ staircase, bit-exact) and **NOT a quadratic decode**
  (the fire-time decode is LINEAR; the quadratic model measured the sub-threshold
  membrane, not the output). ⇒ the encode value-warp `φ_k`, built to invert that
  quadratic, is **mis-targeted**.
- The gap has **two** parts: (a) a **death cascade** (chance→0.94) that is a
  CORRECTABLE *joint* firing-gain — proven by d=3 calibration-only matching FT; and
  (b) an **S-independent ~4pp greedy-firing residual** (the *real* "irreducible
  floor": per-sample partial-sum vs complete-sum), closed only by synchronized
  scheduling or multi-spike.
- **Deployable recipe for cascaded:** genuine-cascade **FT-alone** (joint, surrogate-
  gradient) reaches 0.92–0.96; θ-calibration pre-steps don't help and can hurt.
  **For true lossless, the path is the synchronized schedule (≥0.97), not cascaded.**
- Why LIF is lossless and single-spike cascaded is not: LIF presents the COMPLETE
  weighted sum every cycle (rate code) → no partial-sum firing penalty → cold ≈
  staircase. Same `log2(T+1)` bits, opposite robustness.

## There is no free middle ground — the real lever is SPIKE COUNT

A tempting idea is a per-layer **fire-delay** (wait for more inputs before firing).
It does not work: delaying the fire also shifts the fire-time, which *is* the decoded
value, so it needs a compensating "accumulate-the-full-membrane-then-emit-a-timing-
spike" decode — and that two-phase neuron **is exactly the synchronized schedule**.
Value-encoded-in-timing fundamentally requires the complete value before emitting the
single spike, so causal single-spike firing and complete-sum accuracy are mutually
exclusive. The only real continuum is the **number of spikes per neuron**:

```
single-spike cascaded  →  k-spike  →  full rate code (= LIF)
  lowest energy            ...          lossless (0.9749 in 60s, shipped)
  ~2.6–4.5pp below LIF                  complete sum every cycle
```

So the deployable map is settled: **LIF is the lossless+fast operating point**;
cascaded single-spike TTFS is the extreme-low-energy end with a now-quantified,
mechanistically-explained structural cost; the **synchronized** single-spike schedule
is the lossless single-spike option (≥0.97) at the cost of `d×` latency. The encode
value-warp / dual-spike interior fixes target the spike-count axis, not a (non-
existent) decode nonlinearity.
