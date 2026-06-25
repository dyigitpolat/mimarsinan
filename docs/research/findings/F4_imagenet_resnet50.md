# F4 — Pretrained near-SOTA + ImageNet (ResNet-50 from-scratch), measured

The headline breadth (DoD-4): **pretrained-near-SOTA + ImageNet reachable at a *costed* phase
budget**. Achieved + measured this campaign.

## 1. The ANN (measured, the headline GPU-weeks result)

**ResNet-50, ImageNet from-scratch, 71.97% top-1** in **61.3 min on 2× RTX PRO 6000 Blackwell**
(`scripts/gpu/train_imagenet_fast.py` via `run_imagenet_leased.py`; 16-epoch one-cycle super-convergence,
batch 512, AMP, channels-last, label smoothing, progressive resize 160→192, eval @224 on the OFFICIAL
val set). This **exceeds** the constrained target (~67% in <~1hr; FFCV's stated 67%/30min on 1×A100),
on 2 GPUs (GPUs 2,3 were held by another tenant) — 4 GPUs would be ~30 min. The torchvision dataloader
was measured-sufficient (~6000 img/s on 2 GPUs); **FFCV was not required**. Checkpoint:
`runs/imagenet/resnet50.pt`.

Trajectory (official-val top-1, the one-cycle profile): epoch 0 **18.2%** → 4 **33.6%** → 8 **34.7%**
(high-LR plateau) → 12 **62.0%** → 15 **71.97%** (cosine-decay convergence). No divergence after the
batch-matched LR fix.

*(Methodology bug caught + fixed mid-run: the provider's class-SORTED index-range 95/5 split trained on
only ~950 classes and made the per-epoch val score at chance — fixed to train on the FULL train + eval
on the OFFICIAL val. The discipline working — see `RESEARCH_DOCUMENT.md` §4 / the ROADMAP.)*

## 2. The deployment — REACHABLE at a costed phase budget (measured)

The trained ResNet-50 is **deployable as an SNN, characterized**: it is **VALID** (Wave-6 measured the
Bottleneck trunk param-majority 0.666; `research_gap_ops=[]` — all conv/bn/relu/pool/linear/residual
mappable). At the native ImageNet input (224×224) it maps to **~O(100K) irreducible soft-cores** (the
single-shot map is genuinely heavy — observed ~132 GB host RAM in the mapping phase, the exact 138K-core
scale wall the program identified for VGG16@224). It is **feasible via the Scheduled path** at a costed
phase budget: ~158 reprogram phases → **~16 reprogram + ~142 reuse** via time-domain weight-reuse
(measured-by-mechanism, D3/E4), priced by the defensible cost model (P2 band). So **ImageNet is reachable
at a costed phase budget — DoD-4 satisfied.**

## 3. The deployed-SNN ACCURACY — what it actually takes (corrected 2026-06-25)

A prior draft of this section punted the deployed accuracy as a "scheduled-path / 138K-core GPU-weeks
frontier," implying the blocker was the per-core sim's memory. **That framing was wrong on two counts,
and both are now corrected by measurement.**

**(a) The memory wall was a standalone-script artifact, not intrinsic.** The 132 GB peak RAM came from
`scripts/gpu/deploy_imagenet_snn.py` materializing all ~138K hard cores at once
(`build_hybrid_hard_core_mapping`). But the deployed value does **not** require that build: the torch-side
**Neuromorphic Forward (NF)** `chip_aligned_segment_forward` is **parity-locked bit-exact** to the
deployed hard-core sim (`out_max_abs == 0.0`, per-neuron `k == k`; `tests/integration/_torch_sim_fidelity.py`,
`test_residual_torch_sim_fidelity.py`) and runs purely on the IR repr — it never builds a hard core. So
the deployed accuracy is measurable **torch-side at full 224 resolution, memory-bounded** (Wave-10 U1
`deploy_via_nf`; and U2 streams the literal per-core HCM one segment at a time, peak RAM = one phase).
The mapping memory is **not** the blocker.

**(b) The real blocker is the LIF ADAPTATION, and a naive conversion is chance — MEASURED.** The NF is a
*measurement* tool, not the conversion. Run on the trained ResNet-50 **without** the LIF-adaptation
pipeline it is at chance, even with textbook activation-scale calibration:

| conversion of the 71.97%-ANN ResNet-50 | T=4 | T=8 | T=16 | T=64 |
|---|---|---|---|---|
| convert + LIF `scale=1.0` (no calibration) | 0.000 | — | 0.031 | 0.078 |
| convert + q=0.99 activation-scale calibration (PTC, no fine-tune) | — | 0.000 | 0.000 | — |
| (float-ANN reference on the same batch) | | | **0.875** | |

(measured this campaign, ImageNet-val subset, float weights). So **post-training conversion of a deep
ResNet-50 to a rate-coded LIF SNN is chance at deployable T** — the depth-driven death-cascade needs the
**gradient-based LIF adaptation** (the pipeline's `LIF Adaptation` fine-tune-with-LIF-in-the-loop step),
not just a per-layer scale. This is the same firing-gain/death-cascade mechanism characterized on the
small vehicles, now confirmed at ResNet-50 scale: **the adaptation is load-bearing, not optional.**

**(c) The genuine number is therefore a bounded GPU run, now precisely specified.** The production
`DeploymentPipeline` already has the spine: `Weight Preloading` (loads `runs/imagenet/resnet50.pt`) →
`Activation Analysis` → **`LIF Adaptation`** (fast-recipe fine-tune) → clamp/quant → `Weight Quantization`
→ `Normalization Fusion` → deployed-accuracy via the **memory-bounded NF** (U1) instead of the 132 GB
`Hard Core Mapping`+`Simulation`. That LIF-adaptation run is the costed, named next step (Wave-11); its
final adapted top-1 is **not yet measured** and is **not faked** here.

## 4. Honesty ledger
- **ANN accuracy** — measured (71.97%, official val, 61 min/2 GPUs). Headline result, exceeds target.
- **Validity / scale** — measured (VALID; ~O(100K) cores @224; scheduled-feasible 16/142 phases; P2 cost band).
- **Memory feasibility of the deployed-accuracy measurement** — RESOLVED (the parity-locked NF measures it
  at full res with no 138K-core build; U2 streams the HCM). The 132 GB wall was a standalone-script artifact.
- **Naive/PTC deployed accuracy** — measured = **chance** (table above). The LIF adaptation is load-bearing.
- **Adapted deployed-SNN accuracy** — NOT yet measured (the bounded LIF-adaptation GPU run, Wave-11);
  honestly the open number. Retention on the VALID small vehicles is measured + lossless (synchronized).
