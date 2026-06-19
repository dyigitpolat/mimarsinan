# Direction D (H4) — Multi-spike → single-spike relaxation curriculum

**Hypothesis (H4).** A multi-spike (rate-like) code is depth-robust: the decoded
value is a *spike count*, independent of *when* in the window the spikes land, so
the cascade's window-shortening-with-depth (the death cascade) does not attenuate
it. The deployed single-spike *timing* code is the opposite. **Claim:** train the
continuous teacher into the deployed single-spike basin by ANNEALING the *code*
(rate → timing, `k` spikes → `1`), recovering at each `k`, and this walks the
optimizer into the deployed basin better than S-annealing (D1, which failed). The
**deployed `k=1` forward must stay the bit-exact single-spike cascade** — `k` is a
TRAINING-time curriculum only.

**Verdict: REFUTED as a gap-closer; PARTIAL as a relative improvement over direct
genuine training.** Code-annealing is a much gentler optimization path than direct
`k=1` fine-tuning (which is catastrophic), but its `k=1` endpoint does **not** beat
simply deploying the untrained continuous weights, and at the primary benchmark
(S=8, depth=3) **no schedule, no stop-point moves the deployed accuracy off chance.**
The death cascade's output-layer attenuation is an absolute wall in the single-spike
code; depth-robustness gained at `k>1` evaporates the instant `k` returns to 1.

---

## 1. Method — what was built on `cascade_lab`

`experiments/curriculum.py` is a self-contained, differentiable, **k-spike cascade
simulator** (`KSpikeCascade`) built from the converted flow's perceptron params
(`W`, `b`, `activation_scale`, `input_scale`, latency-by-depth). The whole point of
the curriculum is the `k` knob:

* **`k = 1`** — each neuron may emit a single spike (the deployed timing code).
* **`k > 1`** — the entry encoder and the deep neurons emit up to `k` spikes
  (a multi-spike, rate-like, depth-robust code), via a `min(k−1, 1)`-weighted
  convex blend between a single-spike TTFS train and an evenly-spread rate train,
  plus a `k`-budgeted soft-reset fire on the deep neurons.

**The k=1 endpoint is the deployment contract, and it is enforced bit-exactly.**
Two independent validations (`hard_cascade`, the explicit hard reference, and the
soft `KSpikeCascade(k=1)` forward) match the lab's genuine `cascade_forward` to
`< 1e-9` for **all** S ∈ {4,8,16,32} × depth ∈ {2,3,4}. (Getting this exact
required the entry encoder to fire at `k_fire = ceil(S(1−v))` — the membrane
crossing — not `round`; that single fix took the k=1 diff from 0.11 → 0.0.)

The simulator carries the `k`-knob and the gradient (straight-through fire: forward
is the hard deployed spike, backward flows through a soft membrane/budget). Every
reported **deployed** number is produced by loading the trained weights into a
fresh ReLU MLP, re-converting, **re-calibrating deployment scales**, and running
the lab's genuine float64 single-spike cascade on the full test set — so a "win"
is a real deployed win, not a proxy artifact.

**Curriculum = training-time only.** ✔ Category: a training-time code curriculum;
the deployed decode is unchanged and bit-exact. (No change to the hardware forward.)

---

## 2. The premise is true: the multi-spike code IS depth-robust

On a continuous-trained teacher (depth=3, S=8), measured through `KSpikeCascade`:

| k | k-cascade test acc | mean|out| |
|---|---|---|
| 1 (deployed) | **0.047** (chance) | 0.000 |
| 2 | 0.797 | 0.41 |
| 3 | 0.906 | 0.60 |
| 4 | 0.938 | 0.69 |
| 8 | 0.938 | 0.71 |
| (continuous teacher) | 0.984 | — |

The single-spike code collapses (the death cascade), and **adding spikes revives
the deep layers** almost all the way to continuous. H4's premise holds. (Mechanism
note: in this simulator the multi-spike doesn't change the latch/ramp *decode*; it
changes the *dynamics* — more upstream spikes charge the downstream membrane sooner,
so deep neurons fire earlier and attenuate less. Either way it is depth-robust and
collapses to the deployed single spike at k=1.)

---

## 3. Direct k=1 training is catastrophic; the curriculum is gentler

Matched-compute comparison (same teacher snapshot, same #gradient steps; only the
k-schedule differs). Deployed = genuine single-spike cascade, full test set.

**S=8, depth=3 (PRIMARY), strong teacher (init_cont 0.944, init_gen 0.074):**

| arm | final cont | final deployed gen |
|---|---|---|
| init (deploy continuous weights) | 0.944 | **0.074** |
| direct k=1 (train through deployed forward) | 0.55 | **0.074** |
| curriculum (4→2→1.5→1) | 0.88 | **0.074** |

Deployed gen is **pinned at chance** for every arm. But notice the curriculum
*preserves the continuous representation* far better (cont 0.88 vs direct's 0.55):
direct training through the dead single-spike forward sends near-noise gradients
through the starved deep layers and **destroys** the model; code-annealing does not.

**S=16, depth=3 (room to move), seed 0 (init_cont 0.944, init_gen 0.4286):**

| stage (genuine k=1 deploy after training at that k) | cont | deployed gen |
|---|---|---|
| init | 0.944 | 0.4286 |
| direct k=1 (3 stages) | 0.675 | **0.074** (collapse) |
| curriculum k=4 | 0.928 | 0.323 |
| curriculum k=2 | 0.926 | 0.401 |
| curriculum k=1.5 | 0.924 | **0.447** ← *beats init* |
| curriculum k=1.25 | 0.907 | 0.401 |
| curriculum k=1.1 | 0.889 | 0.397 |
| curriculum **k=1** | 0.874 | **0.30** (drops) |

Two robust facts here: (1) **direct k=1 training collapses the deploy to chance**;
(2) the curriculum tracks the deployed accuracy gracefully all the way down to
`k≈1.25–1.5`, then **the final k=1 stage falls off a cliff** (0.45 → 0.30). The
discontinuity is intrinsic: even `k=1.1` keeps a 10% rate component alive in the
deep layers; at exactly `k=1` the deep layer is pure single-spike and dies.

---

## 4. The "early-stop" refinement: a marginal, seed-dependent, unreliable win

Because the k=1 stage is the cliff, the right move is to anneal down to the best
intermediate `k` and **stop**, deploying *those* weights at genuine k=1 (the deploy
is always single-spike regardless of the training `k`). Deployed genuine k=1 acc
after annealing down to each stop-`k`, 3 seeds, S=16:

| seed | init_gen | k=4 | k=2 | k=1.5 | k=1.25 | k=1 | best stop-k |
|---|---|---|---|---|---|---|---|
| 0 (init_cont 0.94) | 0.4286 | 0.323 | 0.401 | **0.447** | 0.401 | 0.301 | k=1.5 (+0.018) |
| 1 (init_cont 0.75) | 0.1911 | 0.169 | 0.199 | 0.189 | **0.206** | 0.141 | k=1.25 (+0.015) |
| 2 (init_cont 0.66) | 0.1596 | 0.097 | **0.117** | 0.111 | 0.102 | 0.076 | k=2.0 (−0.043) |

* Training **all the way to k=1 always loses** vs init (0.30, 0.14, 0.076).
* The best *stop-k* gives at most ~+1.5pp over init (seeds 0,1) and a *loss* on
  seed 2 — and the optimal stop-k is inconsistent (1.5, 1.25, 2.0).
* Seeds 1,2 have weak teachers (the digits split + init vary by seed), confounding
  their low deployed numbers; only seed 0 is a clean strong-teacher datapoint.

So "early-stop code-annealing" is a real but **marginal and unreliable** effect, not
a gap-closer.

**S=8 (PRIMARY), strong teacher — the absolute wall:** deployed genuine k=1 acc
after annealing down to every stop-k ∈ {8,4,2,1.5,1.25,1}: **0.0742, 0.0742,
0.0742, 0.0742, 0.0742, 0.0742.** Chance at every single stopping point. No code
curriculum moves the S=8 depth=3 deployed cascade.

---

## 5. Mechanism — why the k=1 endpoint is a wall (deployed atten profile, S=8)

| weights | deployed atten by depth `[L0, L1, L2(out)]` |
|---|---|
| init (continuous-trained) | `[1.0, 0.236, 0.0]` |
| curriculum-trained to k=1.5 | `[1.0, 0.071, 0.0]` |

The output layer (depth 2) decodes to **0.0** in both cases: at S=8 the output
neuron, firing at latency ≈2 into a window of length 8, has so little ramp left
that its decode rounds to zero — and **no weight choice in the single-spike code
fixes it** (the `(T−d)/T` depth budget, §2 of the root artifact). Worse, the
curriculum-trained weights make the *middle* layer's single-spike attenuation
*worse* (0.236 → 0.071): the weights get optimized for the multi-spike dynamics,
which are a *different operating point* than the single-spike timing code they must
deploy in. This is the core reason H4 cannot close the gap: **depth-robustness is a
property of the multi-spike code, not of the weights; returning to k=1 returns to
the fragile timing code.**

---

## 6. Honest verdict and what it means

* **H4 premise (multi-spike is depth-robust): CONFIRMED** — k>1 revives the death
  cascade from chance to ~0.94.
* **H4 claim (k-annealing closes the deployed gap): REFUTED.** The k=1 endpoint is
  a discontinuity cliff; the deployed single-spike accuracy after a full anneal is
  **below** init at S=16 and **chance** at S=8 (primary). Best cherry-picked
  stop-k gives ≤ +1.5pp on the one clean seed, a loss on another — not a gap close.
* **Useful negative-space finding (PARTIAL):** code-annealing is a **dramatically
  gentler** optimizer than direct genuine k=1 fine-tuning. Direct k=1 training
  collapses the deploy to chance and destroys the continuous representation
  (cont 0.94 → 0.55); the curriculum preserves it (cont → 0.88) and tracks the
  deployed accuracy down to k≈1.25 before the endpoint cliff. **If any future
  genuine-cascade fine-tuning is attempted, do it via a code-anneal with early
  stop, never by training at k=1 directly.**

**Will it transfer to the real mmixcore pipeline?** The *negative* result transfers
and is actionable: it predicts that direct genuine-TTFS fine-tuning (training *at*
the deployed single-spike forward) is harmful on deep segments — consistent with
the prior D1 S-annealing failure and the `ttfs_cycle finetune↔deploy parity`
memory (genuine cascade must be kept installed, but training *at* it on starved
layers is destructive). The *positive* hope (closing the gap) does **not** transfer:
the wall is the `(T−d)/T` single-spike depth budget, which the real pipeline shares.
The reason mmixcore only shows ~3pp (not collapse) is that it is *shallow* (few
genuine segments) — exactly the regime where the death cascade is mild and a
curriculum has little to fix. On a deep cascade it would hit the same wall.

---

## 7. Next step

Abandon code-annealing as a gap-closer. The data points at the two levers that the
k=1 wall cannot touch and a curriculum cannot manufacture:

1. **Raise the per-layer single-spike depth budget directly** — i.e. Direction B
   (depth-aware analytical pre-compensation: per-depth θ/scale so a late single
   spike's ramp decodes to the teacher's value) or Direction E (effective-depth
   reduction). These attack the `atten→0` output layer at its cause, which no
   training-time code relaxation can.
2. If genuine fine-tuning is used at all, adopt the **early-stop code-anneal** as a
   *gentle* recovery procedure (it strictly dominates direct k=1 training), but pair
   it with a structural fix from (1) so there is a surviving single-spike basin for
   it to land in.

### Artifacts
* Prototype: `docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/curriculum.py`
  (`KSpikeCascade`, `hard_cascade`, `compare_arms`, `train_arm`; k=1 bit-exact
  validated vs the lab's genuine cascade across S×depth).
* This findings doc.
