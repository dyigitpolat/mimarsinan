# ANN-to-SNN 知识蒸馏 (VGG9 / CIFAR-10)

通过激活量化和知识蒸馏，将 ANN 预训练权重迁移到 SNN。  
SNN 使用 **SpikingJelly** 的 `LIFNode`，VGG9 结构从零实现。

## Pipeline

```
Step 1: ANN VGG9 (ReLU)              ~89%
   ↓ 校准 + 权重-阈值平衡
Step 2: ANN VGG9 (ClampFloor T=4)    ~89%
   ↓ 知识蒸馏 (soft targets)
Step 3: SNN VGG9 (LIF, T=8)          ~87%
```

## 快速开始

```bash
python train.py              # 跑全部 3 步
python train.py --step 1     # 只跑 Step 1
python train.py --step 2     # 只跑 Step 2（依赖 Step 1）
python train.py --step 3     # 只跑 Step 3（依赖 Step 2）
```

日志输出到 `logs/step{1,2,3}_*.log`。

## 每一步做了什么

### Step 1: 训练 ANN VGG9

在 CIFAR-10 上训练标准 ANN VGG9（ReLU 激活）。

- **结构**: 7 层 Conv3x3+BN+ReLU, 3 层 MaxPool, ConvMLP 分类头 (Conv7x7 → Conv1x1 → Linear)
- **训练**: SGD, lr=0.05, cosine 调度, 60 epochs
- **输出**: `checkpoints/ann.pt`

### Step 2: 激活量化

将 ReLU 替换为 `ClampFloor(T=4)`，输出离散值 {0, 0.25, 0.5, 0.75, 1.0}——恰好对应 SNN 在 T=4 时间步的所有可能平均发放率。

1. **校准**: 前向传播 2048 个样本，收集每层 ReLU 后激活的 99.9 分位数
2. **权重-阈值平衡**: 缩小 BN 的 gamma/beta，放大下一层 Conv 权重，使激活值落在 [0, 1]
3. **替换**: ReLU → ClampFloor(T=4)
4. **微调**: Adam, lr=1e-3, 30 epochs，使用 STE（直通估计器）传梯度
- **输出**: `checkpoints/quant.pt`

**为什么这样做**: ClampFloor 迫使 ANN 在与 SNN 发放率完全一致的离散激活下学习。这样训练出的 Conv 核迁移到 SNN 时效果更好（实验验证：epoch 1 准确率 72.63% vs 普通 ReLU 的 67.52%）。

### Step 3: 蒸馏到 SNN

以量化 ANN 为 teacher，通过知识蒸馏训练 SNN student。

- **Teacher**: 量化 ANN（冻结，提供 soft logit 目标）
- **Student**: SNN VGG9，使用 SpikingJelly `LIFNode`（tau=2.0, v_threshold=1.0, ATan 代理梯度）
- **权重初始化**: 只迁移 Conv 核权重（不迁移 BN）
  - *为什么不迁移 BN？* ANN 经过阈值平衡后 BN gamma ≈ 0.25，太小了。SNN 需要 gamma=1.0 才能让足够的神经元超过发放阈值。
- **T=8**: 8 个时间步提供更细的发放率分辨率（9 级 vs T=4 的 5 级）
- **KD 损失**: `0.7 × KL(soft) + 0.3 × CE(hard)`，温度 T=3.0
- **CutMix** (p=0.5) + **AutoAugment** 做正则化
- **AdamW**, lr=2e-3, 5 epoch warmup + cosine 衰减, 100 epochs
- **输出**: `checkpoints/snn.pt`

**为什么 KD 有效**: soft targets 包含类间相似性的"暗知识"（比如"这张图 70% 像猫、20% 像狗"），比 one-hot 标签提供了更丰富的梯度信号。SNN 学生最终可以超越 ANN teacher 的准确率。

## 文件结构

```
pretrain-SNN-v2/
├── models.py       # ANN_VGG9, SNN_VGG9, ClampFloor, transfer_weights
├── train.py        # 三步 pipeline
├── checkpoints/    # ann.pt, quant.pt, snn.pt
└── logs/           # step1_ann.log, step2_quantize.log, step3_snn.log
```

## 关键设计决策

| 决策 | 原因 |
|------|------|
| 用 VGG9 而非 VGG16 | VGG16 有 5 层 MaxPool（32→1），导致 SNN 中脉冲信号消失。VGG9 只有 3 层（32→4）。|
| 只迁移 Conv 权重，不迁移 BN | ANN BN 的 gamma≈0.25 会让 SNN 神经元全部静默。保持 BN 默认初始化（gamma=1）让约 16% 的值超过阈值。|
| 量化 Conv > 普通 Conv | 适应了离散激活的 Conv 核与 SNN 发放率更匹配。|
| SNN 用 T=8 | 更细的发放率分辨率减少时间量化误差。|
| 用 KD 而非直接转换 | 直接 ANN→SNN 转换在 T=4 时只有约 16%。代理梯度训练不可或缺。|

## 依赖

```
torch >= 2.0
torchvision
spikingjelly >= 0.0.0.0.14
```
