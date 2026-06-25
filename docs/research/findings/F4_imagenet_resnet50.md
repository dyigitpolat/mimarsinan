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

## 3. The full-res deployed-SNN ACCURACY — the honest scheduled-path frontier

Running the FULL-resolution ImageNet ResNet-50 through the cycle-accurate LIF SNN sim (138K cores × T ×
the 50K val set, or even a single-shot map of 138K cores) is a genuine **GPU-weeks realization** (the
Scheduled-path sim). It is **not** measured here — and is **not faked**. The toolchain's ANN→SNN
**retention** IS measured where it is feasible: on the VALID small-vehicle cells (deep_cnn / lenet5 /
mlp_mixer — the F1–F3 matrix), where synchronized execution is **lossless** (deployed ≈ ANN, bit-exact
per the torch↔sim fidelity locks) and the cascaded death-cascade is the characterized firing-gain
deficit. End-to-end SNN deployment of a real pretrained ResNet (small input) is demonstrated by the D6
bridge. So the *mechanism* (train → convert → map → deploy → measure) is closed end-to-end; the full-res
ImageNet deployed-accuracy at scale is the named, costed future direction.

## 4. Honesty ledger
- **ANN accuracy** — measured (71.97%, official val, 61 min/2 GPUs). Headline result, exceeds target.
- **Validity / scale** — measured (VALID; ~O(100K) cores @224; scheduled-feasible 16/142 phases; P2 cost band).
- **Full-res deployed-SNN accuracy** — NOT measured (the scheduled-path sim is GPU-weeks); honestly the
  frontier. Retention demonstrated on the VALID small vehicles (lossless synchronized).
