"""Builder that produces a shape-adapted ``torchvision`` Vision Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vision_transformer import VisionTransformer, interpolate_embeddings

from mimarsinan.models.builders.torchvision_builder_utils import (
    parse_image_input_shape,
    resize_conv_input_weights,
)
from mimarsinan.pipelining.model_registry import ModelRegistry

_VIT_DEFAULT_IMAGE_SIZE = 224
_VIT_DEFAULT_PATCH_SIZE = 16
_VIT_IMAGENET_IN_CHANNELS = 3
_VIT_B_NUM_LAYERS = 12
_VIT_B_NUM_HEADS = 12
_VIT_B_HIDDEN_DIM = 768
_VIT_B_MLP_DIM = 3072


def _resolve_vit_patch_size(h: int, w: int) -> int:
    if h != w:
        raise ValueError(
            f"torchvision VisionTransformer expects square images; got {h}x{w}."
        )
    divisors = [d for d in range(1, h + 1) if h % d == 0]
    best = min(divisors, key=lambda d: (abs(d - _VIT_DEFAULT_PATCH_SIZE), -d))
    if best == 1 and h > _VIT_DEFAULT_PATCH_SIZE:
        return min(d for d in divisors if d > 1)
    return best
def _resize_patch_kernel(weight: torch.Tensor, patch_size: int) -> torch.Tensor:
    if weight.shape[-2:] == (patch_size, patch_size):
        return weight.detach().clone()
    flat = weight.detach().reshape(-1, 1, weight.shape[-2], weight.shape[-1])
    resized = F.interpolate(
        flat,
        size=(patch_size, patch_size),
        mode="bicubic",
        align_corners=True,
    )
    return resized.reshape(weight.shape[0], weight.shape[1], patch_size, patch_size)


def _adapt_patch_projection(
    conv: nn.Conv2d, in_channels: int, patch_size: int
) -> nn.Conv2d:
    if (
        conv.in_channels == in_channels
        and conv.kernel_size == (patch_size, patch_size)
        and conv.stride == (patch_size, patch_size)
    ):
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=patch_size,
        stride=patch_size,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        weight = _resize_patch_kernel(conv.weight, patch_size)
        weight = resize_conv_input_weights(weight, in_channels)
        new_conv.weight.copy_(weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _load_compatible_state_dict(model: nn.Module, state_dict) -> None:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    model.load_state_dict(compatible, strict=False)


@ModelRegistry.register("torch_vit", label="Torch ViT", category="torch")
class TorchViTBuilder:
    def __init__(
        self, device, input_shape, num_classes, max_axons, max_neurons, pipeline_config
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def _build_model(self, num_classes: int) -> VisionTransformer:
        c, h, w = parse_image_input_shape(self.input_shape, model_name="ViT")
        patch_size = _resolve_vit_patch_size(h, w)
        model = VisionTransformer(
            image_size=h,
            patch_size=patch_size,
            num_layers=_VIT_B_NUM_LAYERS,
            num_heads=_VIT_B_NUM_HEADS,
            hidden_dim=_VIT_B_HIDDEN_DIM,
            mlp_dim=_VIT_B_MLP_DIM,
            num_classes=num_classes,
        )
        model.conv_proj = _adapt_patch_projection(model.conv_proj, c, patch_size)
        return model

    def build(self, configuration):
        return self._build_model(self.num_classes)

    def get_pretrained_factory(self):
        """Return a callable that creates a pretrained ViT-B/16 (ImageNet weights)."""

        def _factory():
            c, h, w = parse_image_input_shape(self.input_shape, model_name="ViT")
            patch_size = _resolve_vit_patch_size(h, w)
            model = self._build_model(num_classes=1000)
            pretrained = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            state_dict = pretrained.state_dict()
            if h != _VIT_DEFAULT_IMAGE_SIZE or patch_size != _VIT_DEFAULT_PATCH_SIZE:
                state_dict = interpolate_embeddings(
                    image_size=h,
                    patch_size=patch_size,
                    model_state=state_dict,
                    reset_heads=False,
                )
            if c != _VIT_IMAGENET_IN_CHANNELS or patch_size != _VIT_DEFAULT_PATCH_SIZE:
                state_dict["conv_proj.weight"] = resize_conv_input_weights(
                    _resize_patch_kernel(pretrained.conv_proj.weight, patch_size),
                    c,
                )
                if pretrained.conv_proj.bias is not None:
                    state_dict["conv_proj.bias"] = pretrained.conv_proj.bias.detach().clone()
            _load_compatible_state_dict(model, state_dict)
            return model

        return _factory

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "base_activation", "type": "select", "label": "Activation", "options": ["ReLU", "LeakyReLU", "GELU"], "default": "ReLU"},
        ]

    @classmethod
    def validate_config(cls, config, platform_cfg, input_shape):
        """Accept any positive square ``(C, H, W)`` and choose a matching patch size."""
        try:
            _, h, w = parse_image_input_shape(input_shape, model_name="ViT")
            _resolve_vit_patch_size(h, w)
        except (TypeError, ValueError):
            return False
        return True
