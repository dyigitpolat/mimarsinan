# WS-mode — Per-mode deployed-accuracy landscape (one model, one dataset)

**Question.** On a single fixed *vehicle* (same `mlp_mixer_core`, same MNIST data,
same time budget `S=4`, same chip), how do the five members of the firing-mode
family rank by **deployed accuracy** — and which of them are *lossless* (deployed
∈ noise of the float ANN reference) versus carrying a real conversion gap? Where
does the genuine single-spike **cascaded** TTFS schedule land relative to its
**synchronized** sibling and to the analytical / LIF references?

**Status: COMPLETE — 15/15 jobs done (rc 0).** 5 modes × 3 seeds. All runs share
one model + dataset + `S=4` + chip; the *only* deliberate differences are the
firing-mode knobs. This is the clean "same-vehicle" mode comparison.

---

## 1. The vehicle (held fixed across all 15 runs)

| knob | value |
|:-----|:------|
| dataset | `MNIST_DataProvider` |
| model | `mlp_mixer_core` (patch_n/m=4, patch_c=128, fc_w 64/128, ReLU) — 78 250 params |
| time budget | `S = target_tq = simulation_steps = 4` |
| chip | 60×(256ax/512n) + 60×(512ax/256n), 5-bit weights, bias, coalescing + neuron-split on |
| weight quant | on (all modes) |
| seeds | 0, 1, 2 |
| backend | nevresim spiking simulation (`enable_nevresim_simulation=true`) |
| run prefix | `pm_<mode>_mmix_mnist_s<seed>` (config glob is `pm_*.json`, **not** `permode_*.json`) |

The five modes are selected purely by the spiking knobs:

| mode label | `spiking_mode` | `firing_mode` | distinguishing knob | placement | act-quant |
|:-----------|:---------------|:--------------|:--------------------|:----------|:----------|
| lif | `lif` | LIF | — | subsume | off |
| ttfs (analytical) | `ttfs` | TTFS | — | subsume | off |
| ttfs_quantized | `ttfs_quantized` | TTFS | — | **offload** | **on** |
| ttfs_cycle cascaded | `ttfs_cycle_based` | TTFS | `ttfs_cycle_schedule=cascaded` | subsume | off |
| ttfs_cycle synchronized | `ttfs_cycle_based` | TTFS | `ttfs_cycle_schedule=synchronized` | subsume | off |

> Note the two non-vehicle deltas that ride along with mode choice (they are
> *properties of the mode*, not separate axes): `ttfs_quantized` is the only mode
> with `activation_quantization=on` + `encoding_layer_placement=offload`; every
> other mode runs `subsume` placement, act-quant off. cascaded vs synchronized
> differ **only** in `ttfs_cycle_schedule`.

---

## 2. The full table — deployed vs ANN reference, by mode

3-seed mean ± sample sd (n=3). **ANN ref** = float source-network test accuracy
*before any spiking fold* (`steps.json → Pretraining.target_metric`). Deployed =
final pipeline `__target_metric.json` (queue artifact; full test set). gap = ANN −
deployed. 95% CI is the two-sided Student-t interval, n=3 (t=4.303). MNIST
const-prediction chance floor ≈ 0.1135.

| mode | deployed (mean ± sd) | 95% CI (±pp) | ANN ref (mean) | gap (pp) | wall (s, mean) |
|:-----|:---------------------|-------------:|:---------------|---------:|---------------:|
| **ttfs (analytical)**       | **0.9807 ± 0.49pp** | 1.23 | 0.9821 | **+0.15** | 1610 |
| **ttfs_quantized**          | **0.9773 ± 0.45pp** | 1.12 | 0.9829 | **+0.55** | 502 |
| ttfs_cycle synchronized     | 0.9607 ± 0.25pp | 0.62 | 0.9826 | +2.19 | 265 |
| lif                         | 0.9600 ± 0.70pp | 1.74 | 0.9829 | +2.29 | 1980 |
| **ttfs_cycle cascaded**     | **0.9523 ± 0.91pp** | 2.25 | 0.9834 | **+3.11** | 664 |

Per-seed (deployed):

| mode | s0 | s1 | s2 |
|:-----|---:|---:|---:|
| ttfs (analytical)       | 0.975 | 0.983 | 0.984 |
| ttfs_quantized          | 0.977 | 0.973 | 0.982 |
| ttfs_cycle synchronized | 0.9582 | 0.9608 | 0.9632 |
| lif                     | 0.960 | 0.953 | 0.967 |
| ttfs_cycle cascaded     | 0.942 | 0.959 | 0.956 |

---

## 3. Which modes are lossless?

"Lossless" = deployed accuracy is within statistical noise of the float ANN ref
on this vehicle (gap small relative to the seed sd / CI).

- **`ttfs (analytical)` — LOSSLESS.** gap **+0.15pp** (98.07 vs 98.21), far
  inside the ±0.49pp seed sd. This is the analytical TTFS mode that propagates
  real-valued membrane voltages across cores (it is *not* a binary-spike
  simulation — see the memory note "ttfs_quantized is analytical, not spiking"),
  so its near-zero gap is expected: it is the least lossy instrument in the family.
- **`ttfs_quantized` — EFFECTIVELY LOSSLESS.** gap **+0.55pp** (97.73 vs 98.29),
  ~1.2× the seed sd — a real but sub-1pp gap, the best of the genuinely
  *quantized* (act-quant on) modes. Also the fastest of the "lossless" pair (~500s).
- **`lif` / `ttfs_cycle synchronized` — NOT lossless on this vehicle.** Both sit
  ~2.2pp below ANN (gap +2.29 / +2.19pp), well outside their seed sd. Note this
  is the **fast-budget** regime (S=4); the LIF memory line records that LIF
  becomes *fully* lossless only when the stabilize step-count is pushed
  400→6000, which this S=4 / fixed-budget config does not do. So "LIF not
  lossless here" is a **budget** statement, not a capacity statement.
- **`ttfs_cycle cascaded` — the lossy outlier (NOT lossless).** gap **+3.11pp**,
  the largest in the family, and the widest seed spread (±0.91pp sd, ±2.25pp CI).

So on this fixed vehicle the loss ranking is:
**ttfs-analytical (lossless) < ttfs_quantized (≈lossless) ≪ sync ≈ lif (~2.2pp) < cascaded (~3.1pp).**

---

## 4. How does cascaded compare to the rest?

**Cascaded is last, by every measure.** It is the only genuine single-spike,
ramp-reconstructed binary-spike schedule, and it pays for that genuineness:

- **−0.84pp vs its synchronized sibling** (95.23 vs 96.07) — the *only* knob
  difference between them is `ttfs_cycle_schedule`, so this gap is a clean,
  confound-free measurement of the **cost of the cascaded greedy partial-sum
  firing schedule** on this 1-segment mlp_mixer vehicle. It corroborates the
  WS3 depth result (cascaded < synchronized at every trainable depth) at a
  shallow operating point.
- **−2.5 to −2.9pp vs the analytical TTFS modes** (cascaded 95.23 vs analytical
  98.07 / ttfsq 97.73). The genuine binary-spike cascade leaves ~2.5–2.9pp on
  the table relative to the real-valued / act-quant TTFS instruments.
- **Highest variance** (±0.91pp sd) — consistent with the known fragility of the
  cold cascade (death-cascade sensitivity to seed / firing-gain).

This is the shallow-depth corner of the same story WS3 told at depth: the
cascaded schedule is the genuine-but-lossy member, and the synchronized schedule
is the one that holds closest to ceiling among the cycle-based modes.

---

## 5. Methodology caveats (read before citing the numbers)

1. **Instrument asymmetry on `sync`.** lif, ttfs-analytical, ttfs_quantized and
   cascaded all report the **nevresim `Simulation`** step as their deployed
   number (it is the last completed step, == the queue artifact). The **sync**
   runs end at **`Hard Core Mapping`** — their deployed number (0.9582/0.9608/
   0.9632) is the **SCM/HCM analytical** deployed metric, *not* a completed
   nevresim Simulation step, even though `enable_nevresim_simulation=true`. Per
   the fidelity memory, synchronized `ttfs_cycle` is SCM↔HCM psum-**bit-exact**
   (psum diff = 0.0 in float64), so the SCM number is a faithful proxy — but it
   is **a different instrument** from the other four. The cascaded−sync gap in §4
   is therefore "nevresim cascaded vs SCM/HCM synchronized"; treat its sign
   (cascaded < sync) as solid and its exact magnitude (0.84pp) as instrument-
   mixed.
2. **`ttfs_quantized` is analytical, not binary-spiking.** Its +0.55pp gap is not
   evidence that genuine spiking is lossless — it is the analytical-V mode with
   act-quant on. The only genuine binary-spike modes here are the two cycle-based
   schedules (cascaded, synchronized) and LIF.
3. **Single vehicle, S=4, n=3.** This is one shallow mlp_mixer on MNIST at a fast
   budget. The "lossless" verdicts are vehicle-specific; the LIF "not lossless"
   result is budget-bound (see §3). Do not extrapolate the ranking to deep
   stacks (that is WS3's job) or to harder datasets without a re-run.

---

## 6. Verdicts appended to ledger

Cluster `WS-mode`, kind `permode`. One per-mode roll-up verdict (×5) plus one
synthesis verdict were appended to `runs/campaign/ledger.jsonl`.

---

## 7. Consolidated multi-seed roll-up — ledger upgrade (2026-06-24)

The earlier ledger left **6 `WS-mode/permode` placeholder records with empty
`run_ids` and `None` means** (the per-mode tables above were correct but not
machine-cited). This round consolidates them into **one 3-seed roll-up record
that cites all 15 run_ids** (`pm_{lif,ttfsanalytic,ttfsq,casc,sync}_mmix_mnist_s{0,1,2}`,
all rc=0), so the director's per-run coverage drops every `permode_*` cell from
`harvest_todo`. The numbers reproduce §2 exactly — nothing in the analysis changes:

| mode | deployed (mean) | ANN gap (pp) | lossless? |
|:-----|----------------:|-------------:|:----------|
| ttfs (analytical) | 0.9807 | +0.15 | **LOSSLESS** (analytical V) |
| ttfs_quantized | 0.9773 | +0.55 | **≈lossless** (analytical V, act-quant on) |
| ttfs_cycle synchronized | 0.9607 | +2.19 | no (budget-bound S=4) |
| lif | 0.9600 | +2.29 | no (budget-bound S=4) |
| **ttfs_cycle cascaded** | **0.9523** | **+3.11** | **LOSSY OUTLIER** |

**Verdict — CONFIRMED (n=3): cascaded-TTFS is the lossy outlier of the firing-mode
family at S=4** (−0.84pp vs its synchronized sibling, +3.11pp ANN gap, widest seed
sd ±0.91pp), while ttfs-analytical is lossless and ttfs_quantized effectively so.
The cascaded→sync sign is corroborated per-seed (sync−casc = +1.62/+0.18/+0.72pp,
all positive). **Confound (magnitude only):** the 0.84pp gap is "nevresim-subsampled
cascaded (1000/10000) vs full-test SCM/HCM synchronized" — see §5.1; sign solid,
magnitude instrument-mixed. Ledger record: `cluster:"WS-mode"`, `kind:"permode"`,
`model:"mlp_mixer_core"`, all 15 `run_ids` cited.
