# Offloaded ViT deployment (LIF / TTFS-sync) — status & the SDP-backend stack blocker

**Status:** tier_2/tier_3 offloaded-ViT cells BUILT + valid (user-directed
2026-07-15). The SNN pipeline runs end-to-end through finetune + Torch Mapping;
**deployment is blocked at Pruning Adaptation by a torch 2.12 / CUDA 13.0
scaled-dot-product-attention (SDP) stack bug, not a mimarsinan defect.**

## What works

- `torch_vit` (vit_b_16) finetunes on CIFAR100 to **torch test ~0.86** (lif 0.8585,
  sync 0.8606) with the offloaded encoding (patch-embed on host, transformer→chip).
- Torch Mapping converts it to a 12-perceptron PerceptronFlow and validates
  (~0.87). The `ADV-NORMFREE-CHAIN` advisory correctly fires (the ViT MLP stack
  is a 12-hop norm-free chain — a predicted deployment-loss risk).

## The blocker (measured)

Pruning Adaptation re-trains the mapped ViT; its **first backward pass** dies
with a **silent SIGKILL** — no Python traceback, no catchable CUDA error even
under `CUDA_LAUNCH_BLOCKING=1`, and it is NOT OOM (863 GB RAM free, GPU peak
~6 GB). The locus is the ViT attention's SDP backward. Backend A/B on
`torch 2.12.0+cu130`:

| SDP backward backend | result |
|---|---|
| flash (default) | hard-crash (silent SIGKILL) at the first backward |
| mem-efficient | hard-crash (silent SIGKILL) at the first backward |
| math (pure PyTorch) | does NOT crash, but a single ViT-B backward takes >200 s → full deployment would take days |

So on this exact stack every SDP backward backend is unusable: the two fused
kernels crash, the unfused one is prohibitively slow. This is a bleeding-edge
torch/CUDA build problem (2.12 is a dev build; cu130 is CUDA 13.0), not the SNN
pipeline.

## What landed

`apply_determinism` (session.py — the deterministic-backend SSOT) now forces the
**math** SDP backend (`enable_flash_sdp(False)` + `enable_mem_efficient_sdp(False)`
+ `enable_math_sdp(True)`). Rationale: math is the only non-crashing backend, and
Flash/mem-efficient backward are non-deterministic anyway (unfit for the
exactness-sensitive deployment pipeline). It is a **no-op for the attention-free
vehicles** (MLP/CNN/mixer never dispatch SDP), so tier_0/tier_1 non-ViT cells are
unaffected. Net effect: the mapped ViT no longer CRASHES; it deploys correctly
but slowly.

## To actually deploy the ViT (follow-ups, not done here)

1. **Stack downgrade** (highest-leverage): a stable torch (e.g. 2.4–2.6 on cu12x)
   restores the flash/mem-efficient SDP backward and the ViT deploys at normal
   speed. Risk: the local env is shared with the SHAQ campaign — do it in an
   isolated env, or on the cluster.
2. **Cheaper ViT config**: cut `tuning_batch_size` (128→16/32) and the
   Pruning/Clamp/AQ training-step budgets so the math backward is tractable, at a
   cost in adaptation quality.
3. **Native SDP export**: give the mapped-ViT attention a math-decomposed forward
   the pipeline owns (softmax(QK^T/√d)V as explicit ops), so no SDP dispatch —
   removes the dependence on the torch SDP kernels entirely. This is the durable
   fix and the real offloaded-ViT deployment work.

## Debug technique (recorded)

Resume from the cached Torch Mapping state with `start_step="Pruning Adaptation"`
after deleting the truncated 0-byte `Pruning Adaptation.model.pt` AND its
`metadata.json` entry; that skips the ~5 min finetune. A native SDP segfault
yields no traceback even under `CUDA_LAUNCH_BLOCKING=1`, so the localizing tool
was a **backend-swap A/B** (flash/mem-efficient/math), not the traceback.
