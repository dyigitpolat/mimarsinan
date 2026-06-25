"""Pretrained bridge: import stock pretrained CNN classifiers as deployable regions.

Lands the ``regime=pretrained`` REGION as a capability (no training, no GPU
deployment). It imports stock ImageNet-pretrained torchvision residual nets
(ResNet-18, ResNet-50) and re-sizes the classifier head to ``num_classes`` so the
SAME conversion + verification instruments B4/SqueezeNet used
(``classify_validity`` + ``estimate_cores_needed``) can produce an HONEST
pretrained-regime region descriptor. The native 3-channel stem and residual
structure are kept exactly as pretrained -- the bridge does NOT rewrite the
architecture; it only swaps the final ``fc`` Linear so the readout matches the
target task.

Pipeline-native op set: Conv2d / BatchNorm2d (absorbed into conv) / ReLU /
MaxPool2d / AdaptiveAvgPool2d / Linear plus residual ``add`` (host ComputeOp).
No grouped/depthwise conv, no attention, no LayerNorm -- so it carries NO
research-frontier op. The residual ``add`` segment boundaries are what make the
MEASURED verdict param-minority/MAC-majority (VALID_FLAGGED) for the BasicBlock
ResNet-18, the honest cost of mapping a stock residual net. The BottleneckBlock
ResNet-50 keeps the param MAJORITY on-chip (its 1x1/3x3 trunk convs outweigh the
residual-boundary downsample shortcuts) -- so the param-minority verdict is
architecture-dependent, not an intrinsic residual-net property.

Actual pretrained finetune / GPU deployment is a follow-up; this module lands the
capability and its honest, instrument-MEASURED descriptor.
"""

from __future__ import annotations

import torch.nn as nn


def _resize_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap a torchvision classifier ``fc`` Linear to ``num_classes`` outputs, eval()."""
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, int(num_classes))
    return model.eval()


def load_pretrained_resnet18(
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Return a torchvision ResNet-18 with its ``fc`` head re-sized to ``num_classes``.

    ``pretrained=True`` loads the stock ImageNet1K_V1 weights (downloaded/cached by
    torchvision; needs network on first call); ``pretrained=False`` builds the same
    architecture with random weights (offline-safe, for structural checks). Only the
    final ``fc`` Linear is replaced -- the pretrained convolutional trunk is kept
    verbatim, including its native 3-channel stem and residual blocks.

    Raises:
        ImportError: if torchvision is not importable in the environment (reported
            verbatim so the missing dependency is an honest, precise blocker).
    """
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    try:
        import torchvision.models as tvm
        from torchvision.models import ResNet18_Weights
    except ImportError as exc:  # pragma: no cover - dependency-availability gate
        raise ImportError(
            "load_pretrained_resnet18 requires torchvision. The pretrained bridge "
            f"could not import it: {exc}"
        ) from exc

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return _resize_head(tvm.resnet18(weights=weights), num_classes)


def load_pretrained_resnet50(
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Return a torchvision ResNet-50 with its ``fc`` head re-sized to ``num_classes``.

    The bottleneck-block sibling of :func:`load_pretrained_resnet18`: same residual
    op set (conv/bn/relu/pool/linear + residual add, all ``groups==1``), but the
    1x1->3x3->1x1 bottleneck trunk holds the param majority on-chip. ``pretrained``,
    head-resize, and the ``ImportError`` contract match ``load_pretrained_resnet18``.

    Raises:
        ImportError: if torchvision is not importable (reported verbatim).
    """
    if int(num_classes) < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}.")
    try:
        import torchvision.models as tvm
        from torchvision.models import ResNet50_Weights
    except ImportError as exc:  # pragma: no cover - dependency-availability gate
        raise ImportError(
            "load_pretrained_resnet50 requires torchvision. The pretrained bridge "
            f"could not import it: {exc}"
        ) from exc

    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    return _resize_head(tvm.resnet50(weights=weights), num_classes)
