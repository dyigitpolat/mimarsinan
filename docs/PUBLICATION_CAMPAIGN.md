# The Publication Campaign — plan, orchestration, and live state

Launched from `docs/mimarsinan_closeout_analysis_v2.md`. The controller arc is
**finished** (6/9 MNIST cells MET the full AC1–AC6 spec, R7 killed). The remaining
program **is** the publication program: prove **automatic genericity at breadth, in
both training regimes, with near-SOTA pretrained conversion, against baselines, with
statistics** — the engine is validated at one point; the missing artifact is the
breadth evidence.

## The difficulty-ladder spectrum (cheap→expensive, kill-gated)

Breadth is a *designed spectrum* run as a complexity ladder; the cheap rungs isolate
the method's known stressors (**depth** → the firing-gain death-cascade; **operator
type** → ReLU-CNN → residual → attention/LayerNorm/GELU; **activation statistics**;
**accuracy regime**) so a cheap failure localizes the cause and *saves* the expensive
run.

| Tier | Vehicles (bold = to build) | Role |
|------|----------------------------|------|
| **T0 diagnostic** | **deep-but-narrow MLP (4→32 layers)** on MNIST/**Fashion-MNIST**; **single residual/attention/LN+GELU block** | isolate ONE stressor; not accuracy claims |
| **T1 small classical** | **LeNet-5**, small CNN/MLP on MNIST/**FMNIST/KMNIST/SVHN**; **ResNet-20**@CIFAR-10 | recipe + keystone workhorse |
| **T2 mid modern** | **ResNet-18/34**, VGG-16, **MobileNetV2**@CIFAR-100/Tiny; **DeiT-Tiny/small ViT**@CIFAR-100 | first real transformer conversion + depth stress |
| **T3 SOTA (headline)** | **ResNet-50, ViT-B/DeiT-B, ConvNeXt-T, Swin-T, EfficientNet**@ImageNet from near-SOTA checkpoints | pretrained-near-SOTA + energy/latency Pareto |

**Kill-gate rule:** run T0→T1→T2→T3; escalate only on per-stressor pass. The
deep-but-narrow MNIST probe (T0) tests the headline's gating risk (depth×firing-gain)
for almost free — answer it there before ImageNet.

## Workstreams → publication claims

| WS | deliverable | unblocks claim |
|----|-------------|----------------|
| WS0 | bank milestone; **freeze controller surface** (no more driver tuning) | — |
| WS1 | the ladder as a dual-mode conversion→deploy→certify campaign; integrate **timm/torchvision** (architectures + near-SOTA checkpoints) + cheap datasets | "across a spectrum of classical→modern models & datasets" |
| WS2 | modern-arch conversion: residual (easy), **attention/softmax, LayerNorm, GELU** (research) | "across modern models (transformers)" |
| WS3 | **depth-scale the cascaded firing-gain fix or settle synchronized as the deep default** (§6) | "pretrained→near-SOTA on deep models" (HEADLINE) |
| WS4 | dual-mode certification: **per-model near-SOTA references** in the floor book; pretrained (tight ε) vs from-scratch (pretraining-amount ablation) | "pretrained near-SOTA + from-scratch profiling" |
| WS5 | the **per-layer-S allocator** (budget → best-accuracy allocation) + the accuracy↔energy↔latency Pareto | "energy/latency-aware neuromorphic deployment" |
| WS6 | **baselines** (RMP/QCFS/SNN-Calibration/TTFS-conversion) + multi-seed CIs + per-lever ablations | reviewer-grade comparison + significance |
| WS7 | **keystone ON**: probe→select→escalate, no per-config tuning; thresholds made data/model-adaptive | "genericity holds, automatically" (CENTRAL) |

**Sequencing:** WS0 now. WS1 cheap-first (stand up T0–T1 before SOTA). WS3 & WS2 in
parallel as the two technical risks, **each first probed at the cheap end** (deep-but-
narrow T0 for WS3; single-block T0 for WS2) so they fail fast. WS4 alongside WS1.
WS5/6/7 once breadth runs populate them. First publishable result = climb the ladder:
lock T0–T1, pass T2 with baselines, escalate to ONE T3 ImageNet model in both regimes.

## Autonomous orchestration (how this runs unattended)

Two layers, so GPUs never idle and code-building stays isolated:

1. **GPU execution — a persistent, never-idle runner** (`scripts/gpu/`): the
   `campaign_runner` drains a crash-safe filesystem `GpuQueue` FOREVER (never exits on
   drain — late enqueues fill GPUs at once), leasing GPUs by class (`free`=profiling
   exclusive / `fit`=correctness, util-agnostic; per-GPU cap), with a `status.json`
   heartbeat. `campaign_watch` wakes the operator at a **low watermark** (refill *before*
   starvation), on drain, stall, or a failure burst. Parametric sweeps (model × dataset
   × seed × cell) are enqueued here and multiplex across the GPU pool.
2. **Capability building — isolated worktree workflows.** New code (model builders,
   datasets, conversion ops, the allocator) is built by parallel worktree-isolated agents
   (disjoint file ownership), test-first, integrated via serialized patches — then the
   runner's plan gains new sweep nodes that use the new capability. Research workstreams
   thus proceed **concurrently and in isolation**; their experiments share the GPU pool.

**Kill-gates** are applied as results land: a stressor failure at tier T prunes the
dependent T+1 jobs before they cost GPU-days. Every (model,dataset,regime) cell carries
a **measured absolute verdict** against a per-model near-SOTA reference (WS4) — the same
DoD discipline that closed the controller arc.

## Live state

- **Controller arc:** finished (Phase-3 scorecard 6/9 MET; floor book carries absolute
  AC targets).
- **Autonomous infra:** built + tested (`campaign_runner`/`gpu_queue`/`campaign_watch`).
- **First batch RUNNING (WS3 cheap probe):** the **θ-cotrain** lever (per-channel
  threshold = the literal firing-gain knob — the close-out's *untested* strong lever,
  distinct from the already-refuted STE-hedge) on cascaded matrix_6/8 × S∈{4,8,16,32} ×
  3 seeds, fast-path and controller-path (48 jobs). Question: does the firing-gain knob
  meet the firing-gain deficit where the STE-hedge could not?
- **Next:** build the T0 deep-but-narrow MLP + cheap datasets (Fashion-MNIST/KMNIST/SVHN)
  + the timm/torchvision bridge (WS1 foundation) so the θ-cotrain probe runs at *depth*.
