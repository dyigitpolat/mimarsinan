# mimarsinan ROADMAP — toward a generic, publication-grade SNN-deployment toolchain

*Living artifact. Dependency-ordered. Re-priced as increments land. The companion to
`PROGRAM_CHECKPOINT_v2.md` (state) and `HYPERVOLUME.md` (the axis SSOT).*

## North star (Definition of Done)

The final deliverable is the **toolchain framework**: a composable environment to experiment
with SNN deployments across a **wide deployment-configuration hypervolume**, providing tools that
achieve **publication-grade results** — with genericity **measured and audited, not asserted**.
Energy/accuracy/speed are *per-cell outputs*, not the success metric.

**DoD (all measured, none asserted):**
1. The hypervolume is **honestly + broadly covered** — the screened denominator (every collapse
   artifact-backed) *and* the under-exercised central axes (backend/sim, regime, datasets, scale)
   actually run.
2. Every covered cell carries **comparable per-cell outputs** — cost (defensible + band), accuracy,
   and an explicit **attribution-vs-value-domain** fidelity tag.
3. The headline **capabilities** are built — residual backbone, scheduled-scale realization, and
   either on-chip attention/LN *or* an honestly-scoped conv headline.
4. **Publication rigor** — baselines, CIs/ablations, dual-regime certification, pretrained-near-SOTA
   + ImageNet reachable at a *costed* phase budget.
5. The honesty is **self-defending** (CI-enforced: no collapse-without-artifact, no merged tiers, no
   aged-unowned flags).
6. Then: a **research document** (system, process, quantitative insights, genuine future gaps).

## Current honest state (the baseline this roadmap moves)

- **Honest coverage = 3.75%** (deep_cnn 6 / **160** cells, firing-pinned; measured `coverage_report.py`);
  97 science-valid cells (VALID 43 / VALID_FLAGGED 38 / INVALID 16, tiers never merged, invariant under
  the collapse). *Wave 2 raised it from 0.23% (denominator 2560→160) via two artifact-backed
  FAITHFULNESS-axis collapses (`backend`, `mapping_strategy`) — the keystone working in the legitimate
  direction. The semantic knobs `pruning`/`regime` stay enumerated (no fidelity collapse possible).
  Remaining denominator factors: `pruning`×2 · `regime`×2 · `quantization`×4 = 16× (collapse needs
  real GPU equivalence screens / capability builds, NOT cheap).*
- **Wave-1 landed (main `2462241`, all default-off byte-identical + tested):** A4 self-defense guards
  (`scripts/campaign/guards.py`: base-check / stash-intact / `fcntl` singleton) + gated scheduler
  singleton; B4 SqueezeNet vehicle (region-add — VALID, on-chip frac 1.0 offload, 942/1000 cores
  single-phase, measured by the framework's own instruments); C2 cost-wiring (`reuse_mj_band`
  defensible band + backend cost coordinate, default 0.0 byte-identical); B1 cross-sim parity
  screening instrument (`cross_sim_parity.py` + `justifies_collapse`/`assert_cross_sim_screen_sound`
  honesty gate — the artifact A2 consumes to screen the `backend` axis).
- **Wave-2 landed (main `d7a9515e`, adversarially verified CONFIRMED_CLEAN by a 2-skeptic workflow):**
  C3 GAP-1 attribution fix (joint `(perceptron_output_slice, ir_id)` keying → per-neuron attribution
  bit-exact under coalescing+output-tiling, value-domain byte-identical; GAP-1's "neuron_split"
  framing was sharpened to "output-tiling under compaction reorder"); A2A3 axis-collapse screens
  (`backend`+`mapping_strategy` → SCREENED_COLLAPSED on measured artifacts, fidelity-only with
  capability+cost kept as frontiers; the 38 placement-fixable flags auto-owned `program:placement-offload`;
  honest deep_cnn coverage re-priced **0.23%→3.75%** measured). Introduced the **faithfulness-axis vs
  semantic-knob** distinction (the integrity rule for which axes may collapse on a fidelity artifact).
- **Wave-3 landed (main `f17f0ed2`, dynamic-workflow build→adversarial-verify):** E5 Pareto decision
  layer (cascaded-vs-synchronized **REGIME_DEPENDENT** — cascaded NOT retired, it wins the hard-latency
  budget; cost is a model-estimate-with-band, per-sample energy is the named UNINSTRUMENTED gap) +
  `propose_recipe(budget)`; D3 scheduled-build probe (genuine overflow→scheduling, bit-exact, VGG 16/142
  confirmed-by-mechanism); GAP1R honestly SHARPENED the GAP-1 attribution tag (kept VALUE_DOMAIN_ONLY —
  production gate is identity-mapping-only; only the harness reassembler was fixed). Research-round
  independently corroborated E5 (no working config-level firing-gain rescue lever at the convnet d6 onset).
  **F5 research document** (`RESEARCH_DOCUMENT.md`) drafted, all sections measured-and-cited.
- **Wave-4 landed (main `33afcbeb`):** cost-EMIT (CONFIRMED_CLEAN — the deployment path now emits
  measured `cost_record.json`, closing the cost gap at the source; E5 prefers measured over proxy).
  residual-Tier1 produced a genuine NEGATIVE result kept ISOLATED (`wave4/residual-t1`): an in-segment
  on-chip merge is intrinsically `1/T`-off from host-add (characterized, not a bug). **The cheap-code
  frontier is now ~exhausted** — remaining DoD = capability builds (B3 pretrained bridge · D5 attention
  · D6 timm · D7 baselines) + the GPU-weeks F-layer (F1-F4 ImageNet) = a genuine commitment fork.
- **Wave-5 landed (main `f9012f7f`, both CONFIRMED_CLEAN):** B3 pretrained bridge (`pretrained_bridge.py`
  — real torchvision ResNet-18 mapped+validity-classified; the **two-class VALID_FLAGGED** insight:
  ResNet-18 is flagged for a *placement* reason — residual-boundary host-side encoders, param 0.42 /
  MAC 0.999, `research_gap_ops=[]` — NOT for unsupported ops like ViT; finding doc written) + D7
  percentile-norm baseline (`activation_scale_policy.py`, selectable default-off, numerically verified).
  **The codeable frontier is now exhausted** — every remaining DoD item needs GPU-weeks of campaign
  compute (F1-F4) or a research build (D5 attention). Both deliverables produced; F5 finalized with the
  Wave-4/5 results. Awaiting the GPU-weeks greenlight.
- **Wave-6 landed (main `9a5a8936`, CONFIRMED_CLEAN):** pretrained-vehicle validity sweep — *tested and
  REFUTED* the Wave-5 offload hypothesis (ResNet-18 stays VALID_FLAGGED under both placements; the host
  param-majority is `supported_host` residual shortcuts, not offloadable encoders) and added the
  **ResNet-50 region** (VALID, Bottleneck param-majority, scheduled-feasible). Result: **validity is
  architecture-dependent (BasicBlock-minority vs Bottleneck-majority); VALID_FLAGGED has THREE causes**
  (unsupported-op / offloadable-encoder / structural-host-residual). Testing-not-asserting corrected a
  would-be F5 overclaim — the discipline working.
- **Infra note (Wave 1):** the harness's Workflow `isolation:'worktree'` snapshots a STALE base
  (`bcacfeb`, an old session HEAD — its object DB lacks current `main`); the A4 base-check guard caught
  it. Dispatch pattern is now **manually-created `git worktree` from current `main` + absolute-path
  agents**; re-base worktrees per land (Wave 2 bases on `2462241`).
- **Landed instruments:** E2 static validity pre-check; gate-v2 tiered validity (20%/50%, params+MACs);
  E1 coverage ledger + **P1 self-auditing** (screening-status→denominator, CI guard, flag aging,
  per-region attribution-fidelity); E4 capacity diagnostic + scheduling-aware verdict; **P2 defensible
  GAP-R cost** (VGG@224 ≈13.4 mJ, band 1.5–49.3, weight-reuse 6900×); weight-reuse phase classifier;
  Residual Tier-0; E6 cascaded-rescue quarantine.
- **Frontier mapped + measured:** ImageNet conv = 138K irreducible softcores (no weight sharing),
  feasible-via-Scheduled-path at ~158 phases → ~16 reprogram + ~142 reuse via weight-reuse.

---

## F-LAYER EXECUTION — GREENLIT (Wave 7+, max-parallel isolated)

User **greenlit the GPU-weeks F-layer** (2026-06-25). **ImageNet is constrained to a well-established
FAST recipe, NOT from-scratch-to-SOTA**: target ~**67% top-1 ResNet-50 from-scratch in <1 hour**
(FFCV-style — FFCV claims 67%/30min on one A100; we have 4× RTX PRO 6000 Blackwell). Deployed-SNN
accuracy is then whatever the toolchain retains from that ANN, measured.

**Infra reality (assessed 2026-06-25):** ImageNet is EXTRACTED at `/data/ImageNet` (train/+val/
ImageFolder + devkit); `imagenet_data_provider.py` + cifar10/100 providers exist; `data_handling/ffcv/`
is WIRED but the `ffcv` pip package is NOT installed (fast-loader enablement needed); the ImageNet
tuning-collapse bug was fixed in a prior session; **NO imagenet ledger row yet** (never run to completion).

**Reset dependency order (two groups):**
- **Group 1 — codeable builds (parallel, isolated, no-GPU-to-land) = Wave 7:** D4 pruning×scheduling ·
  D2A residual Component A (shared-HCM-fill NF==HCM `atol=0`) · D5 on-chip attention/LN (research;
  partial+characterized OK per DoD-3's "or honestly-scoped conv headline") · F-harness (F1/F2/F3
  experiment-matrix runner + aggregator) · D6 pretrained DEPLOY bridge · ImageNet fast-recipe (trainer
  CODE; FFCV install + the actual run are a SUPERVISED post-build step to protect the live campaign venv).
- **Group 2 — GPU runs (enqueue → runner drains, parallel):** F1/F2/F3 cells on existing
  vehicles×datasets · B2 CIFAR10/100 · the ImageNet ResNet-50 run → consolidate into F1–F4 → finalize F5
  with measured publication results.

---

## The layers (status · dependency · cost)

Status: ✅ landed · 🔬 isolated/not-merged · ⏳ in-flight · ⬜ open.
Cost: ◦ cheap (code, days) · ◦◦ medium (capability build) · ◦◦◦ GPU-weeks.

### A — Complete the measurement instrument (mostly done; the rest is cheap)
| Item | Status | Dep | Cost |
|---|---|---|---|
| A1 Self-auditing coverage (screening-status denominator, CI guard, flag aging) | ✅ P1 | — | ◦ |
| A2 **P3 screens** — `backend`+`mapping_strategy` ✅ Wave2 (faithfulness axes; measured artifacts: live cross-sim max_abs_diff=0 + bit-exact fidelity lock; fidelity-only). `pruning`/`regime` are SEMANTIC knobs → can't collapse on fidelity; gated on D4/B3 capability + a real GPU equivalence screen (NOT cheap) | partial Wave2 | A1, B1✅ | ◦ / ◦◦ |
| A3 P1↔P3 declare↔execute wiring (a screen mechanically updates `HYPERVOLUME.md` + re-prices) | ✅ Wave2 (A2A3 flipped AXES + updated HYPERVOLUME + re-priced in one unit) | A2 | ◦ |
| A4 Engineering self-defense — base-check guard (stale `bcacfeb` trap), stash-pop guard, scheduler-restart idempotence | ✅ Wave1 (`guards.py`, gated singleton) | — | ◦ |

### B — Raise honest coverage / breadth (the largest open terrain — the deliverable itself)
| Item | Status | Dep | Cost |
|---|---|---|---|
| B1 **Cross-simulator parity** — shared cells across nevresim/SANA-FE/Lava; record agree / disagree(quantified) / inapplicable(capability gap) | ✅ Wave1 (`cross_sim_parity.py` instrument + `justifies_collapse` gate; screen-consumption = A2) | A1 | ◦ |
| B2 Dataset breadth — close the named frontier (SVHN, deeper cells), CIFAR | ⬜ | — | ◦–◦◦ |
| B3 **Regime axis** — the pretrained bridge (timm/torchvision) + 1 small from-scratch↔pretrained cross-screen | bridge ✅ Wave5 (`pretrained_bridge.py`; ResNet-18 ImageNet → VALID_FLAGGED 0.423/0.999, placement-flag not research-gap); the cross-screen RUN is GPU | — | ◦◦ |
| B4 Scale vehicles — SqueezeNet ✅ Wave1 (VALID, frac 1.0 offload, 942/1000); **ResNet-50 ✅ Wave6** (VALID, param 0.666 Bottleneck param-majority, SCHEDULED-feasible 16-17 phases peak 208); ResNet-18 ✅ Wave5 (VALID_FLAGGED structural-host, offload-refuted) | ✅ | B3✅ | ◦–◦◦ |

### C — Per-cell output instrumentation (each cell must carry comparable numbers)
| Item | Status | Dep | Cost |
|---|---|---|---|
| C1 GAP-R defensible cost model + band | ✅ P2 | — | ◦ |
| C2 Wire cost into the production cost path + **backend as a first-class cost coordinate** | ✅ Wave1 (`reuse_mj_band` band + backend coord) **+ Wave4 cost-EMIT** (`sanafe_simulation_step` emits measured `cost_record.json`, exception-isolated + result-byte-identical; E5 prefers measured cost over proxy) | C1 | ◦ |
| C3 **GAP-1 attribution fix** — `(ir_core_id, neuron_range)` joint keying so per-neuron lock survives coalescing+split at scale | ✅ Wave2 (sharpened: it's coalescing+output-tiling under compaction reorder; fixed bit-exact, value-domain unchanged; verifier reproduced the failing-first) | — | ◦ |
| C4 Per-region fidelity recording (value-domain vs attribution) | ✅ P1 | — | ◦ |

### D — Capability contributions that ADD hypervolume regions
| Item | Status | Dep | Cost |
|---|---|---|---|
| D1 Residual Tier-0 (host add, bit-exact) | ✅ | — | ◦ |
| D2 **Residual Tier-1** (on-chip param-free merge) — round 3 = LIF merge-window alignment | 🔬 Wave4: CHARACTERIZED as intrinsically `1/T`-bounded (in-segment IF re-quant ≠ host-add by 1 spike, by construction; not bit-exact to Tier-0). Isolated branch `wave4/residual-t1`; finding doc written. Path fwd: redefine success as `1/T`-characterized OR close Component A (shared-HCM-fill, NF==HCM atol=0) | D1 | ◦◦ |
| D3 **Scheduled-scale realization** — real `_build_scheduled` end-to-end probe (confirm 16/142 + bit-exactness) | ✅ Wave3 (genuine overflow→3 stages `[6,6,2]`, 3 reprogram+33 reuse, bit-exact max\|Δ\|=0; VGG 16/142 confirmed-by-mechanism) | E4✅ | ◦–◦◦ |
| D4 **Pruning × scheduling** — pruning shrinks cores → fewer reprogram phases (attacks the 80% cost term) | ⬜ | C1,D3 | ◦◦ |
| D5 **On-chip attention / LayerNorm** — THE transformer contribution (E7 foreclosed the cheap path) | ⬜ | — | ◦◦◦ |
| D6 timm/torchvision bridge (ResNet-50/ViT-B near-SOTA checkpoints) | ⬜ | B3 | ◦◦ |
| D7 Published baselines (RMP/QCFS/percentile-norm) | percentile-norm ✅ Wave5 (`activation_scale_policy.py`, selectable default-off, numerically verified); the GPU head-to-head is the remaining piece | — | ◦◦ |

### E — Decision & science closure
| Item | Status | Dep | Cost |
|---|---|---|---|
| E5 **Pareto allocator** (consumes C1/C2 cost) → cascaded-vs-synchronized verdict on a valid vehicle | ✅ Wave3 (`pareto.py`; gap +6.06/+7.19/+11.34pp measured; cost=model-estimate+band, per-sample energy UNINSTRUMENTED gap) | C2 | ◦◦ |
| E5b **Retire cascaded-rescue** if synchronized dominates; characterize death-cascade as finished science | ✅ Wave3 — verdict **REGIME_DEPENDENT, NOT retired** (cascaded ~2.7–2.9× lower latency keeps it on the front at hard-latency budgets; synchronized = accuracy default). Research-round corroborates: no working config rescue lever at the convnet d6 onset | E5 | ◦ |
| E5c **Automatic recipe selection** — `propose_recipe` picks (mode, schedule, S, placement) from the budget/Pareto (the "automatic genericity" evidence) | ✅ Wave3 (`propose_recipe(budget)`: accuracy→sync, hard-latency→cascaded) | E5 | ◦◦ |

### F — Publication-grade rigor (the venue bar; gated on A–E trustworthy)
| Item | Status | Dep | Cost |
|---|---|---|---|
| F1 CIs + ablations across the (now honestly-priced) matrix | ⬜ | A,B,C | ◦◦–◦◦◦ |
| F2 Baseline head-to-heads on covered/valid cells | ⬜ | D7 | ◦◦ |
| F3 Dual-regime certification (per-(model,dataset) references) | ⬜ | B3 | ◦◦ |
| F4 **Pretrained near-SOTA + ImageNet** — the headline breadth, reachable via Scheduled-path at a costed phase budget | ⬜ | B3,D3,D6,(D5 for ViT) | ◦◦◦ |
| F5 The **research document** (system, process, quantitative insights, genuine future gaps) | ⬜ | all | ◦◦ |

---

## Critical path (what gates what)

```
A1 ✅ ─┬─► A2/A3 (screens raise honest coverage) ─┐
       └─► B1 cross-sim parity ───────────────────┤
C1 ✅ ─► C2 cost-wiring ─► E5 Pareto ─► E5b retire cascaded / E5c auto-select
C3 GAP-1 ─► (attribution at scale)
D2 Residual T1 ─┐
D3 scheduled-build ─► D4 pruning×scheduling ─┐
B3 pretrained bridge ─► D6 timm ─┬─► F4 ImageNet headline
D5 on-chip attention ────────────┘ (ViT arm)
                                  └─► F1/F2/F3 rigor ─► F5 research document
```

**The single highest-leverage thread:** raising *honest* coverage (A2 screens + B1 cross-sim + B3 regime
+ B2/B4 breadth) — because honest measured coverage *is* the deliverable. Everything in C–E is either an
input to a covered cell or a per-cell output it produces; F is the venue packaging.

## Cheap-first ordering (no GPU-weeks until the cheap layer is trustworthy)
1. **Now (cheap, parallel, isolated):** A2/A3 screens · B1 cross-sim parity · C2 cost-wiring · C3 GAP-1 ·
   A4 self-defense · D3 scheduled-build probe · B4 SqueezeNet.
2. **Then:** E5 Pareto → E5b/E5c · D4 pruning×scheduling · D2 residual Tier-1.
3. **Capability builds:** B3 pretrained bridge · D6 timm · D5 on-chip attention (or scope conv headline) · D7 baselines.
4. **Publication frontier (GPU-weeks):** F1–F4 on covered/valid/instrument-trustworthy cells → **F5 research document**.

Re-price honest coverage after every land; never let a label lead the measurement.
