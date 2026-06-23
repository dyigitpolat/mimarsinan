# The Research Organization — operating charter (the concrete plan we follow)

This is the standing structure that executes the publication program of
`docs/mimarsinan_closeout_analysis_v2.md` (ladder + WS0–WS7) as **parallel clusters of
dynamic, multi-round workflows**, sharing one autonomous GPU substrate, coordinated by a
single results ledger. It replaces "one operator hand-feeding the queue."

## The organization

```
                 ┌─────────────────── Research Director (coordinator) ──────────────────┐
                 │  launches clusters · integrates capabilities · synthesizes findings  │
                 └──────────────────────────────────┬───────────────────────────────────┘
   ┌──────────────┬───────────────┬─────────────────┼──────────────┬───────────────────┐
 Cluster A      Cluster B       Cluster C         Cluster D     Capability workflows   (worktree code)
 Depth×gain     Breadth+rigor   Modern ops        Automatic     θ-gate-fix · LeNet/CNN · op-blocks ·
 (WS3, headline)(WS1+WS6)       (WS2)             (WS7)         baselines · Pareto allocator (WS5)
   │              │               │                 │              │
   └──────────────┴───────────────┴──────── shared GPU runner (runs/campaign/q) ────────┘
                                  └──────── shared ledger (runs/campaign/ledger.jsonl) ──┘
```

Each **research cluster** is one dynamic workflow running, for several rounds:
**design → enqueue → wait → analyze → decide(continue|conclude)**, via the CLIs
`scripts/campaign/research_loop.py {enqueue,wait,results,ledger-append,ledger-read}`.
**Capability workflows** (worktree-isolated, test-first, patch→`.finish_patches/`→merge)
build the vehicles a cluster needs. The **GPU substrate** is the never-idle
`campaign_runner` + `gpu_queue` + `campaign_watch` (see `docs/PUBLICATION_CAMPAIGN.md`).

## The difficulty ladder + kill-gate rule (resource efficiency)

T0 diagnostic (deep-but-narrow MLP, single op-block) → T1 small classical (LeNet,
ResNet-20, cheap datasets) → T2 mid modern (ResNet-18/34, small ViT) → T3 SOTA
(ResNet-50/ViT-B @ ImageNet). **Every cluster runs cheap→expensive and escalates ONLY on
a per-stressor pass.** A cheap failure localizes the stressor and *saves* the expensive
run. No cluster spends a GPU-day before its cheap rung passes.

## Cluster charters

| Cluster | Research question (kill-gate) | Vehicle | Stop/escalate criterion |
|---|---|---|---|
| **A — Depth×firing-gain (WS3, HEADLINE)** | Does cascaded single-spike TTFS death-cascade with model DEPTH while synchronized holds? Smoke: d8 cascaded **0.895** vs sync **0.963** (6.8pp, vs 0.8pp shallow) — already suggestive. | `deep_mlp` (built) | Conclude **synchronized = deep-model default** if cascaded falls monotonically with depth + sync holds at tight CIs. Only test θ-cotrain-at-depth AFTER the gate-fix lands; if it doesn't rescue the deep cascade → cascaded scope-bounded. |
| **B — Breadth×rigor (WS1+WS6)** | Does the engine hold across cheap datasets (FMNIST/KMNIST/SVHN) × depths, with multi-seed CIs? | `deep_mlp` + cheap datasets (built) | Build the breadth×CI table; flag any (dataset,depth,mode) that breaks → localize the stressor. |
| **C — Automatic genericity (WS7)** | With the keystone ON (`conversion_policy: true`), does it pick the proven recipe per cell and ESCALATE off-distribution models — no per-config tuning? | A6 keystone (built) | Prove auto-selection + escalation across the cell matrix; re-derive MNIST-calibrated thresholds as data-adaptive. |
| **D — Modern ops (WS2)** | Which modern ops convert on-chip vs host, at what accuracy cost? (residual → LN → GELU → attention) | single-op T0 probes (to build) | Per op: convert / approximate / host-shell + cost. Transformer is the largest lift. |

## Capability backlog (workflows that unblock clusters)

1. **θ-cotrain cascaded gate-fix** (unblocks A's θ-at-depth): per-channel scales through
   `compute_node_output_scales` + the identity hybrid mapping (currently `s.item()` on a
   64-elem per-channel theta crashes the cascaded NF↔SCM faithfulness gate).
2. **LeNet-5 / small-CNN builder** (unblocks B/A at T1 — Conv2d IS supported on-chip).
3. **Single-op probe harness** (unblocks D): residual / LN / GELU / attention blocks.
4. **Baseline layer** (unblocks WS6): RMP/QCFS/SNN-Calibration/TTFS-conversion.
5. **Per-layer-S allocator** (WS5): budget → best-accuracy allocation + the Pareto.

## The ledger = the Definition of Done

Every experiment appends to `runs/campaign/ledger.jsonl` a record with the **measured
absolute verdict** on the deployed parity-gated metric: `{cluster, model, dataset, regime
(pretrained|scratch), schedule, depth, S, seed, deployed_acc, ann_ref, verdict:
MET|BOUNDED-GAP|FAIL, notes}`. A claim is publishable only when its cells carry such
verdicts **with baselines + CIs**. No AC silently mislabeled; no bounded-gap an unexamined
deferral — the discipline that closed the controller arc, carried into the campaign.

## Cadence

Clusters run bounded rounds (≤~4) then report a finding to `docs/research/findings/`; the
Director reviews, integrates capabilities, and launches the next wave (T1→T2, then a single
T3 in both regimes). The GPU runner stays saturated by the union of all clusters' batches;
the watcher alerts the Director at the low-watermark to keep the backlog full.

## Live wave (first)

- **Cluster A** (depth×firing-gain) — analyzing the in-flight depth sweep, deciding the
  deep-model default.
- **Cluster B** (breadth×rigor) — cheap-dataset × depth × seed CIs.
- **Cluster C** (automatic/keystone) — keystone-on auto-selection + escalation.
- **Capability** — θ-cotrain gate-fix + LeNet builder.
