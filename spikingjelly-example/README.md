# ANN-to-SNN Knowledge Distillation (VGG9 / CIFAR-10)

Transfer ANN pretrained weights to SNN via activation quantization and knowledge distillation.  
SNN uses **SpikingJelly** `LIFNode`. VGG9 is implemented from scratch.

## Pipeline

```
Step 1: ANN VGG9 (ReLU)              ~89%
   ↓ calibrate + weight-threshold balance
Step 2: ANN VGG9 (ClampFloor T=4)    ~89%
   ↓ knowledge distillation (soft targets)
Step 3: SNN VGG9 (LIF, T=8)          ~87%
```

## Quick Start

```bash
python train.py              # run all 3 steps
python train.py --step 1     # step 1 only
python train.py --step 2     # step 2 only (requires step 1)
python train.py --step 3     # step 3 only (requires step 2)
```

Logs are written to `logs/step{1,2,3}_*.log`.

## What Each Step Does

### Step 1: Train ANN VGG9

Train a standard ANN VGG9 on CIFAR-10 with ReLU activations.

- **Architecture**: 7 Conv3x3+BN+ReLU layers, 3 MaxPool, ConvMLP head (Conv7x7 → Conv1x1 → Linear)
- **Training**: SGD, lr=0.05, cosine schedule, 60 epochs
- **Output**: `checkpoints/ann.pt`

### Step 2: Quantize Activations

Replace ReLU with `ClampFloor(T=4)` which outputs discrete values {0, 0.25, 0.5, 0.75, 1.0} — matching the possible mean firing rates of an SNN with T=4 timesteps.

1. **Calibrate**: forward 2048 samples, collect 99.9th percentile activation per ReLU layer
2. **Weight-threshold balance**: scale BN down and next Conv up so activations fall in [0, 1]
3. **Replace**: ReLU → ClampFloor(T=4)
4. **Finetune**: Adam, lr=1e-3, 30 epochs with STE (straight-through estimator)
- **Output**: `checkpoints/quant.pt`

**Why**: ClampFloor forces the ANN to learn with discrete activations identical to SNN spike rates. Conv kernels trained this way transfer better to SNN than regular ReLU-trained kernels.

### Step 3: Distill to SNN

Use the quantized ANN as teacher, train SNN student via knowledge distillation.

- **Teacher**: Quantized ANN (frozen, provides soft logit targets)
- **Student**: SNN VGG9 with SpikingJelly `LIFNode` (tau=2.0, v_threshold=1.0, ATan surrogate)
- **Weight init**: transfer Conv kernel weights only (not BN) from quantized ANN
  - *Why not BN?* ANN BN has tiny gamma (~0.25) from threshold balancing. SNN needs gamma=1.0 for neurons to fire.
- **T=8**: 8 timesteps gives finer spike rate resolution (9 levels vs 5 at T=4)
- **KD loss**: `0.7 * KL(soft) + 0.3 * CE(hard)`, temperature=3.0
- **CutMix** (p=0.5) + **AutoAugment** for regularization
- **AdamW**, lr=2e-3 with 5-epoch warmup + cosine decay, 100 epochs
- **Output**: `checkpoints/snn.pt`

**Why KD works**: soft targets encode inter-class similarities (e.g., "70% cat, 20% dog") providing much richer gradient signal than one-hot labels. The SNN student can even surpass the ANN teacher.

## Files

```
pretrain-SNN-v2/
├── models.py       # ANN_VGG9, SNN_VGG9, ClampFloor, transfer_weights
├── train.py        # 3-step pipeline
├── checkpoints/    # ann.pt, quant.pt, snn.pt
└── logs/           # step1_ann.log, step2_quantize.log, step3_snn.log
```

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| VGG9, not VGG16 | VGG16 has 5 maxpools (32→1), causing vanishing spikes in SNN. VGG9 has 3 (32→4). |
| Transfer Conv only, not BN | ANN BN gamma~0.25 silences SNN neurons. Fresh BN (gamma=1) lets ~16% of values exceed threshold. |
| Quantized Conv > Regular Conv | Kernels adapted to discrete activations match SNN spike rates better. |
| T=8 for SNN | Finer firing rate resolution reduces temporal quantization error. |
| KD, not direct conversion | Direct ANN→SNN conversion only gives ~16% at T=4. Surrogate gradient training is essential. |

## Requirements

```
torch >= 2.0
torchvision
spikingjelly >= 0.0.0.0.14
```
