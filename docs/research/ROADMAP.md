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

- **Honest coverage ≈ 0.3%** (deep_cnn breadth 6 / 2048 cells); 97 science-valid cells
  (VALID 43 / VALID_FLAGGED 38 / INVALID 16, tiers never merged).
- **Landed instruments:** E2 static validity pre-check; gate-v2 tiered validity (20%/50%, params+MACs);
  E1 coverage ledger + **P1 self-auditing** (screening-status→denominator, CI guard, flag aging,
  per-region attribution-fidelity); E4 capacity diagnostic + scheduling-aware verdict; **P2 defensible
  GAP-R cost** (VGG@224 ≈13.4 mJ, band 1.5–49.3, weight-reuse 6900×); weight-reuse phase classifier;
  Residual Tier-0; E6 cascaded-rescue quarantine.
- **Frontier mapped + measured:** ImageNet conv = 138K irreducible softcores (no weight sharing),
  feasible-via-Scheduled-path at ~158 phases → ~16 reprogram + ~142 reuse via weight-reuse.

---

## The layers (status · dependency · cost)

Status: ✅ landed · 🔬 isolated/not-merged · ⏳ in-flight · ⬜ open.
Cost: ◦ cheap (code, days) · ◦◦ medium (capability build) · ◦◦◦ GPU-weeks.

### A — Complete the measurement instrument (mostly done; the rest is cheap)
| Item | Status | Dep | Cost |
|---|---|---|---|
| A1 Self-auditing coverage (screening-status denominator, CI guard, flag aging) | ✅ P1 | — | ◦ |
| A2 **P3 screens** — flip `ASSERTED_UNSCREENED` axes (cross-sim parity = flagship; pruning; mapping_strategy; regime) → `SCREENED_COLLAPSED` w/ artifact | ⬜ | A1 | ◦ (regime needs B3) |
| A3 P1↔P3 declare↔execute wiring (a screen mechanically updates `HYPERVOLUME.md` + re-prices) | ⬜ | A2 | ◦ |
| A4 Engineering self-defense — base-check guard (stale `bcacfeb` trap), stash-pop guard, scheduler-restart idempotence | ⬜ | — | ◦ |

### B — Raise honest coverage / breadth (the largest open terrain — the deliverable itself)
| Item | Status | Dep | Cost |
|---|---|---|---|
| B1 **Cross-simulator parity** — shared cells across nevresim/SANA-FE/Lava; record agree / disagree(quantified) / inapplicable(capability gap) | ⬜ | A1 | ◦ |
| B2 Dataset breadth — close the named frontier (SVHN, deeper cells), CIFAR | ⬜ | — | ◦–◦◦ |
| B3 **Regime axis** — the pretrained bridge (timm/torchvision) + 1 small from-scratch↔pretrained cross-screen | ⬜ | — | ◦◦ |
| B4 Scale vehicles — SqueezeNet (builder exists), ResNet-50 (needs B3 bridge) | ⬜ | B3(ResNet) | ◦–◦◦ |

### C — Per-cell output instrumentation (each cell must carry comparable numbers)
| Item | Status | Dep | Cost |
|---|---|---|---|
| C1 GAP-R defensible cost model + band | ✅ P2 | — | ◦ |
| C2 Wire cost into the production cost path + **backend as a first-class cost coordinate** | ⬜ | C1 | ◦ |
| C3 **GAP-1 attribution fix** — `(ir_core_id, neuron_range)` joint keying so per-neuron lock survives coalescing+split at scale | ⬜ | — | ◦ |
| C4 Per-region fidelity recording (value-domain vs attribution) | ✅ P1 | — | ◦ |

### D — Capability contributions that ADD hypervolume regions
| Item | Status | Dep | Cost |
|---|---|---|---|
| D1 Residual Tier-0 (host add, bit-exact) | ✅ | — | ◦ |
| D2 **Residual Tier-1** (on-chip param-free merge) — round 3 = LIF merge-window alignment | 🔬 | D1 | ◦◦ |
| D3 **Scheduled-scale realization** — real `_build_scheduled` end-to-end probe (confirm 16/142 + bit-exactness) | ⬜ | E4✅ | ◦–◦◦ |
| D4 **Pruning × scheduling** — pruning shrinks cores → fewer reprogram phases (attacks the 80% cost term) | ⬜ | C1,D3 | ◦◦ |
| D5 **On-chip attention / LayerNorm** — THE transformer contribution (E7 foreclosed the cheap path) | ⬜ | — | ◦◦◦ |
| D6 timm/torchvision bridge (ResNet-50/ViT-B near-SOTA checkpoints) | ⬜ | B3 | ◦◦ |
| D7 Published baselines (RMP/QCFS/percentile-norm) | ⬜ | — | ◦◦ |

### E — Decision & science closure
| Item | Status | Dep | Cost |
|---|---|---|---|
| E5 **Pareto allocator** (consumes C1/C2 cost) → cascaded-vs-synchronized verdict on a valid vehicle | ⬜ | C2 | ◦◦ |
| E5b **Retire cascaded-rescue** if synchronized dominates; characterize death-cascade as finished science | ⬜ | E5 | ◦ |
| E5c **Automatic recipe selection** — `propose_recipe` picks (mode, schedule, S, placement) from the budget/Pareto (the "automatic genericity" evidence) | ⬜ | E5 | ◦◦ |

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
