# E3 — Mapping-Scalability Probe (conv headline: VGG16)

**Question.** Does the bit-exact mapping pipeline — the *attributability keystone*
(per-neuron NF↔SCM/HCM correspondence that lets us say "this physical hard-core
neuron is this logical neuron") — **survive at headline scale**, on the
**mappable on-chip surface** of a real conv net? And where does the greedy
`placement_engine` **break or slow** (the E4 concern: diagnosable failure +
capacity-aware placement + profiled scaling)?

**Mode.** Read-only mapping probe. **NO TRAINING.** Untrained VGG16-BN via the
registered builder `torch_vgg16`, LIF activations, T=32. No production code was
modified (verified `git status`: only docs are untracked). Throwaway probe
scripts:
- `/home/yigit/repos/research_stuff/mimarsinan/.claude/worktrees/wf_d8572b71-220-1/probe_e3_vgg16.py` (CIFAR-scale, 3×32×32)
- `/home/yigit/repos/research_stuff/mimarsinan/.claude/worktrees/wf_d8572b71-220-2/e3_vgg16_mapping_probe.py` (ImageNet-scale, 3×224×224)
- `/home/yigit/repos/research_stuff/mimarsinan/.claude/worktrees/wf_d8572b71-220-2/e3_cores_estimate.py` (core-count estimate)

---

## VERDICT

**The bit-exact attributability keystone does NOT survive at conv-headline scale,
and the greedy placement engine fails fast on the ImageNet-spatial conv net by
budget exhaustion (≈416× overflow). Wave-C conv-headline mapping is NOT de-risked.
E4 placement-engine work — capacity-aware estimation + a scale-robust per-neuron
reassembly bookkeeping — is REQUIRED before spending GPU-weeks on conv-headline
training.**

The result splits cleanly along the user's vision: **genericity holds on the
*value* domain of the mappable surface** (the deployed sim is value-domain
bit-exact, `out_max_abs == 0.0`, wherever it completes), but the **toolset
instruments that certify *per-neuron attributability* and that *place* the net do
not scale**. Those are FLAGGED research gaps (named below), not silent blockers —
the probe diagnoses each one to a specific code seam.

Two regimes, two distinct truths:

| | VGG16 @ 3×32×32 (CIFAR) | VGG16 @ 3×224×224 (ImageNet conv headline) |
|---|---|---|
| **SCM completes** | yes | yes (only with coalescing on) |
| **HCM placement completes** | **yes** (940 cores, 10.4 s) | **NO** — `No more hard cores available` |
| **Value-domain bit-exact** (`out_max_abs`) | **0.0** (rung-3 lock holds) | n/a (placement aborts) |
| **Per-neuron attributability** | **BROKEN** — 1276/65536 mis-attributed | n/a (LIF gate disabled; TTFS gate blows 0.49) |
| **On-chip fraction** | 0.9956 (1792/1800) | **0.0** (every core > 256 axons) |
| **Estimated cores needed** | 940 | ≈416,560 (vs 1000 budget → ≈416×) |

---

## What SURVIVES at scale

**1. IR build (SCM) survives — once coalescing is a declared capability.**
`convert_torch_model → install LIF → IRMapping.map` completes at both scales.
At CIFAR scale: build 1.8 s, convert 0.8 s, IR build 2.3 s → 1792 NeuralCores +
8 host ComputeOps, max latency 14, 6 neural segments / 8 host barriers. At
ImageNet scale (coalescing on): SCM build completes in ~40 s → 87,644 neural
cores, 8 compute ops, max_latency 14, fully representable (FX trace +
representability + Mapper DAG all clean). **The IR keystone is generic.**

**2. SCM (identity-mapped spiking metric) survives at CIFAR scale.** End-to-end,
output shape (2,10). On-chip-majority gate = 1792/1800 = **0.9956 → TRUE**; only
the 8 host ops (input encode, MaxPool barriers, final classifier) are off-chip.

**3. HCM placement survives at CIFAR scale.** No `No more hard cores available`;
placement wall ~10.4 s; **940 hard cores** (coalescing fuses VGG's wide-fan-in FC,
neuron-splitting tiles the wide conv layers) on a realistic 2048-axon/2048-neuron
crossbar budget.

**4. The VALUE-DOMAIN lock holds at CIFAR scale.**
`assert_torch_sim_fidelity` output residual `out_max_abs == 0.0`
(NF-torch == deployed-HCM-sim). **The rung-3 value lock — the claim that the
deployed chip computes the same numbers as torch — survives at VGG-CIFAR scale.**
Spikes are conserved (per-neuron *totals* match exactly, e.g. conv-perceptron 1:
`torch_total == hcm_total == 654`).

---

## Where the keystone BREAKS (the research gaps)

### GAP-1 — Per-neuron attribution does not survive VGG-scale packing (CIFAR)

The per-neuron LIF `k==k` reassembly — the attributability keystone — **fails at
VGG-CIFAR scale under axon-fuse (coalescing) + neuron-splitting**. Conv-perceptron
1: **1276 / 65536 neurons differ** (≈2%), *while the per-neuron total is identical*
(654 == 654) and `out_max_abs == 0.0`.

So: **spikes are conserved, the decode is value-exact, but the *attribution* —
which physical hard-core neuron maps back to which logical neuron — is scrambled
for ≈2% of neurons.** The same harness PASSES per-neuron `k==k` at tiny scale
(wide_dim=64, fan-in ≥ 21, fuse_core_axons=16). This is a **scale-emergent
reassembly-bookkeeping gap**, not a simulator-dynamics bug.

**Root seam.** `tests/integration/_split_reassembly.py::hcm_per_perceptron_counts`
(→ `_reassemble`). The reassembly inverts the packing by: filtering
coalescing-role ∈ {None, master, accum} and psum-role ∈ {None, accum}, sorting
neuron-split fragments by `neuron_range_in_original[0]`, concatenating per IR core
in IR-id order. When **many fused + split fragments of one wide IR core interleave**
at VGG scale, this orig-offset sort / role filter no longer produces the correct
1:1 neuron map. The bookkeeping is sound at small fan-out but does not survive
deep interleaving.

**For LIF this is THE per-neuron lock.** The value-domain
`assert_nf_scm_parity_or_raise` gate is correctly DISABLED for LIF
(`nf_scm_parity_enabled` returns True only for analytic-staircase /
synchronized-TTFS / cascaded; LIF's `training_forward_kind != analytical_staircase`).
For LIF, per-neuron attributability *is* the `k==k` reassembly in
`_split_reassembly.py` — and that is exactly the instrument that broke. There is
no second LIF instrument behind it.

**Blocks:** any per-neuron attributability claim (fault localization, per-neuron
energy/spike accounting, neuron-level provenance) on coalesced+split conv nets at
CIFAR scale and up. Does **not** block the value-domain deployment claim.

### GAP-2 — HCM placement fails fast at ImageNet conv scale (budget exhaustion)

At 3×224×224 with the realistic ImageNet budget
(`[max_axons:256, max_neurons:256, count:1000]`, coalescing on), HCM placement
**FAILS with `RuntimeError: No more hard cores available`**
(`src/mimarsinan/mapping/packing/greedy/pack.py:204`). The **first** early conv
segment `neural_segment_until:features_6` alone emits **50,176 softcores of
(576 axons, 64 neurons)** and exhausts the entire 1000-core budget. Total estimated
requirement **≈416,560 cores → ≈416× overflow**. On-chip fraction = 0/87,644 = **0.0**
(every core exceeds 256 axons; worst 25088×512).

**This failure is fast and diagnosable** (it aborts at the first conv segment by
budget exhaustion, not by slow search), which is the *good* news for E4: the engine
already raises a clean, attributable error. The *missing* capability is a
**capacity-aware pre-estimate** that reports "this net needs ≈416k cores; your
budget is 1000" *before* committing to a greedy pack — so the reviewer sees a
diagnosable failure + a profiled requirement, not a mid-pack crash.

**Blocks:** any ImageNet-spatial conv-headline deployment on a realistic
1000-core / 256-axon budget. The net does not fit, period — this is a chip-budget
reality, not a tool bug, but the tool must *surface* it as a capacity verdict.

### GAP-3 — Wide-flattened-conv-FC is unmappable without coalescing (prerequisite, not a failure)

VGG's first classifier FC has fan-in **25,088** (= 512·7·7 at 224, or the
flattened conv output). On the **shipped default chip** (`allow_coalescing=False`),
SCM build ABORTS at IR-build time with `WideFanInUnsupportedError`
(`src/mimarsinan/mapping/platform/mapping_structure.py:73`, via
`compute_fc_tiling_mode`): a fan-in wider than `max_axons` cannot map without
inter-core membrane partial-sum transfer. **The default 256-axon chip with no
partial-sum transfer cannot map ANY net whose flattened conv feature map exceeds
`max_axons` — effectively every ImageNet-spatial conv net.**

This is a **mapping prerequisite, not a tool failure**: coalescing is a *declared
chip capability* (the realistic ResNet-50-class setting). The probe enabled it,
which is the honest comparison point. Recorded here so Wave-C knows the
conv-headline config **must** declare coalescing, and the "default chip can't map
VGG" result is not mistaken for a regression.

### GAP-4 — The synchronized-TTFS per-neuron gate does not hold at depth/width and is too slow (ImageNet)

The bit-exact NF↔SCM per-neuron parity gate that *is* exercised for TTFS
(synchronized `ttfs_cycle_based`, budget 0.02, documented measured-bit-exact at
small scale) was probed at ImageNet scale (Regime 3). SCM build completes (36.1 s,
87,644 cores); the gate RUNS but **FAILS the bit-exact check HARD**:
**10,120,095 / 20,688,896 values differ** beyond atol=1e-6 → mismatch fraction
**0.4892** (≫ 0.02 budget). Worst: perceptron 13, nf=0.4935 vs scm=0.0625,
|Δ|=0.43.

**CAVEAT (honest).** Weights are **UNTRAINED / random**, so the deep TTFS cascade
dynamics are uncalibrated; 0.4892 sits near the documented ~0.40
"wrong-NF-dynamics" signature and far above the ~0.14 honest-wire-residual ceiling,
so the magnitude is *partly* a no-training artifact. **But** the gate does not
hold bit-exact at this depth/width out of the box, *and* (combined with a prior
9m40s timeout before completion) the synchronized per-neuron gate over 87,644
cores at S=32 is **prohibitively slow** to be a practical attributability check at
headline scale. Two independent reasons it is not a usable instrument here:
correctness (untrained, blows budget) and cost (does not finish in a tractable
window).

### GAP-5 — Deployed-sim wall time is super-linear (CIFAR, secondary reviewer flag)

At CIFAR scale, while placement itself is fast (10.4 s), an **identity-config
deployed sim over 1792 cores did not finish in >180 min CPU**. The **placement
time at ImageNet scale is NOT super-linear** (it fails fast by budget, 85–115 s
wall, GPU shared with a co-tenant probe) — the super-linear wall is the
**deployed-sim** path, not the placer search. This is the practical reviewer flag:
even where mapping *completes*, end-to-end deployed simulation over ~1.8k cores is
not tractable on CPU at the cadence a research loop needs.

---

## Per-gap "what it blocks" summary

| Gap | Instrument / seam | Survives at scale? | Blocks |
|---|---|---|---|
| **GAP-1** | `_split_reassembly.py::hcm_per_perceptron_counts` (LIF per-neuron `k==k`) | NO (≈2% mis-attributed, CIFAR) | per-neuron attributability / fault localization on coalesced+split conv nets |
| **GAP-2** | `greedy/pack.py::run_placement` budget exhaustion | NO (≈416× overflow, ImageNet) | ImageNet-spatial conv deploy on realistic 1000-core budget |
| **GAP-3** | `mapping_structure.py::compute_fc_tiling_mode` (`WideFanInUnsupportedError`) | prerequisite (needs coalescing) | any wide-flattened-FC on the default no-coalescing chip |
| **GAP-4** | `nf_scm_parity.py` synchronized-TTFS per-neuron gate | NO (0.49 mismatch + too slow, ImageNet) | bit-exact per-neuron certification at headline depth/width |
| **GAP-5** | deployed-sim wall (identity config) | NO (>180 min, 1792 cores, CPU) | tractable end-to-end deployed verification in a research loop |

---

## Is Wave-C conv-headline mapping de-risked? — NO

- **Value domain:** de-risked at CIFAR scale (`out_max_abs == 0.0`, 940 cores,
  10.4 s placement). The deployed chip computes the right numbers where it fits.
- **Attributability:** NOT de-risked. The one per-neuron LIF instrument
  (`_split_reassembly` `k==k`) breaks at VGG-CIFAR scale; the TTFS per-neuron gate
  blows budget and is too slow at ImageNet scale.
- **Placement / capacity:** NOT de-risked for ImageNet-spatial conv. The net needs
  ≈416k cores against a 1000-core budget; the placer fails fast (good) but offers no
  capacity-aware pre-estimate (gap).

**Conclusion: do NOT spend GPU-weeks on conv-headline training until the E4
placement-engine work below lands.** CIFAR-scale conv (VGG @ 3×32×32) is a viable
de-risked rung *for the value-domain claim only*; conv-headline at ImageNet spatial
resolution is gated on capacity work that has nothing to do with training.

---

## E4 placement-engine work the probe shows is needed BEFORE GPU-weeks

1. **Capacity-aware pre-estimate (GAP-2).** A fast `estimate_cores_needed(ir, budget)`
   pass that reports the required-vs-available core count (≈416,560 vs 1000) and the
   first segment that overflows (`features_6`, 50,176 softcores) *before* the greedy
   pack starts. Turn the mid-pack `RuntimeError` into an up-front capacity verdict.
   Probe `e3_cores_estimate.py` already computes this offline — productize it.

2. **Scale-robust per-neuron reassembly (GAP-1).** Fix
   `hcm_per_perceptron_counts` so the orig-offset sort / coalescing-role + psum-role
   filtering survives many interleaved fused+split fragments of one wide IR core.
   Add a scale regression test at VGG-CIFAR fan-out (wide conv, fuse + split) that
   asserts per-neuron `k==k`, not just per-neuron totals. This is the attributability
   keystone for LIF and currently has no fallback.

3. **A tractable attributability instrument at scale (GAP-4 / GAP-5).** The
   synchronized-TTFS per-neuron gate over 87k cores at S=32 is too slow and (untrained)
   blows budget. Need either a *sampled* per-neuron check (subset of perceptrons /
   neurons with a statistical bound) or a cheaper deployed-sim path so end-to-end
   verification finishes inside a research-loop window (GAP-5: >180 min today).

4. **(Profiling) Confirm placer scaling on a net that DOES fit.** The ImageNet probe
   failed fast by budget, so it did not stress the greedy search itself. Before
   conv-headline, profile `run_placement` on a net sized to ≈80–95% of a large budget
   to characterize the search's actual scaling curve (the reviewer's "profiled scaling"
   ask) — distinct from the deployed-sim super-linearity already observed.

---

## Environment / reproduction

- Ran from project root with the main-repo venv (`env/bin/activate`) and
  `PYTHONPATH=./src:/home/yigit/repos/research_stuff/mimarsinan/spikingjelly`
  (the worktree's spikingjelly submodule is empty; the package lives in the main
  checkout).
- Builder: `torch_vgg16` → `TorchVGG16Builder` (vgg16_bn, 134,309,962 params; stem
  adapted to 3-channel; 5 maxpools kept at 224). `convert_torch_model` ≈ 2 s (FX
  trace + representability + Mapper DAG, fully representable).
- NF↔SCM "parity" at CIFAR scale via the production fidelity harness
  `tests/integration/_torch_sim_fidelity.py::assert_torch_sim_fidelity`
  (LIF `k==k` + atol=0). The value-domain `assert_nf_scm_parity_or_raise` gate is
  correctly DISABLED for LIF (`nf_scm_parity_enabled` False).
- Realistic budgets probed: CIFAR = 2048-axon / 2048-neuron crossbars
  (`allow_coalescing` + `allow_neuron_splitting`); ImageNet = `256×256×1000`
  with coalescing off (default, GAP-3) and on (Regime 2/3).
- **No production code modified.**
