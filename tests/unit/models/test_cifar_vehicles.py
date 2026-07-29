"""BC-2 CIFAR-10 checkpoint vehicles (paper decision D2): geometry + registration.

Three torch-category vehicles resolvable by model_type:
- cifar_resnet20: standard CIFAR ResNet-20 (3 stages x 3 BasicBlocks, 16/32/64).
- cifar_vgg8: VGG-8-class net (6 conv+BN+ReLU with 3 pools, avgpool head, 2 FC).
- cifar_vit_leaf: LeafVisionTransformer named config for 32px/patch 4.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.cifar_models import (
    CifarBasicBlock,
    CifarResNet,
    CifarVGG8,
    cifar_resnet20,
    cifar_vgg8,
)
from mimarsinan.models.vit_leaf import LeafVisionTransformer, cifar_vit_leaf


# ── ResNet-20 geometry ───────────────────────────────────────────────────────

class TestCifarResNet20Geometry:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = cifar_resnet20(num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_standard_resnet20_structure(self):
        model = cifar_resnet20()
        blocks = [m for m in model.modules() if isinstance(m, CifarBasicBlock)]
        assert len(blocks) == 9  # 3 stages x 3 BasicBlocks
        assert model.stem_conv.out_channels == 16
        assert [b.conv1.out_channels for b in blocks] == [16] * 3 + [32] * 3 + [64] * 3
        # Downsample projections exactly where geometry changes (stage entries).
        assert [b.shortcut is not None for b in blocks] == [
            False, False, False, True, False, False, True, False, False,
        ]
        assert [b.conv1.stride for b in blocks[0::3]] == [(1, 1), (2, 2), (2, 2)]
        assert model.classifier.in_features == 64
        assert model.classifier.out_features == 10

    def test_every_conv_is_bn_covered_and_ungrouped(self):
        model = cifar_resnet20()
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(convs) == len(bns) == 21  # stem + 9x2 block convs + 2 projections
        assert all(conv.groups == 1 for conv in convs)  # converter-supported subset

    def test_feature_map_reaches_8x8_before_head(self):
        model = cifar_resnet20().eval()
        feats = {}

        def hook(_m, inp, _out):
            feats["shape"] = tuple(inp[0].shape)

        model.head_pool.register_forward_hook(hook)
        with torch.no_grad():
            model(torch.randn(1, 3, 32, 32))
        assert feats["shape"] == (1, 64, 8, 8)

    def test_rejects_bad_input_shapes(self):
        with pytest.raises(ValueError):
            CifarResNet(input_shape=(3, 32), num_classes=10)
        with pytest.raises(ValueError):
            CifarResNet(input_shape=(3, 30, 32), num_classes=10)  # not /4-able
        with pytest.raises(ValueError):
            CifarResNet(input_shape=(3, 32, 32), num_classes=10, blocks_per_stage=0)


# ── VGG-8 geometry ───────────────────────────────────────────────────────────

class TestCifarVGG8Geometry:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = cifar_vgg8(num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_vgg8_structure(self):
        model = cifar_vgg8()
        convs = [m for m in model.features if isinstance(m, nn.Conv2d)]
        pools = [m for m in model.features if isinstance(m, nn.MaxPool2d)]
        bns = [m for m in model.features if isinstance(m, nn.BatchNorm2d)]
        assert len(convs) == 6 and len(bns) == 6 and len(pools) == 3
        assert [c.out_channels for c in convs] == [64, 64, 128, 128, 256, 256]
        fcs = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(fcs) == 2
        assert fcs[0].in_features == 256 * 2 * 2  # avgpool(2) keeps FC fan-in small
        assert fcs[0].out_features == 512
        assert fcs[1].out_features == 10

    def test_max_fan_in_fits_declared_core_geometry(self):
        # Every neuron's fan-in stays <= 2304 (largest: 256-channel 3x3 conv
        # patch = 2304; FC1 = 1024); with the softcore bias row (+1 axon) the
        # BC-2 config's 2560-axon core hosts it without input splitting.
        model = cifar_vgg8()
        fan_ins = [
            m.in_channels * m.kernel_size[0] * m.kernel_size[1]
            for m in model.modules() if isinstance(m, nn.Conv2d)
        ] + [m.in_features for m in model.modules() if isinstance(m, nn.Linear)]
        assert max(fan_ins) <= 2304

    def test_rejects_bad_input_shapes(self):
        with pytest.raises(ValueError):
            CifarVGG8(input_shape=(3, 32), num_classes=10)
        with pytest.raises(ValueError):
            CifarVGG8(input_shape=(3, 12, 12), num_classes=10)  # not /8-able


# ── CIFAR leaf-ViT named config ──────────────────────────────────────────────

class TestCifarViTLeafConfig:
    def test_geometry(self):
        model = cifar_vit_leaf(num_classes=10)
        assert isinstance(model, LeafVisionTransformer)
        assert model.pos_embed.shape == (1, 65, 192)  # (32/4)^2 + cls = 65 tokens
        assert model.patch_embed.kernel_size == (4, 4)
        assert model.patch_embed.stride == (4, 4)
        assert len(model.blocks) == 7
        first = model.blocks[0]
        assert first.attn.num_heads == 3
        assert first.fc1.out_features == 384  # mlp_ratio 2.0
        assert model.head.out_features == 10

    def test_forward_shape(self):
        torch.manual_seed(0)
        model = cifar_vit_leaf(num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)


# ── Registration: DeploymentPlan resolves these by model_type ────────────────

class TestCifarVehicleRegistration:
    @pytest.mark.parametrize("model_type", [
        "cifar_resnet20", "cifar_vgg8", "cifar_vit_leaf",
    ])
    def test_registered_in_torch_category(self, model_type):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        assert model_type in BUILDERS_REGISTRY
        assert ModelRegistry.get_category(model_type) == "torch"
        assert ModelRegistry.get_builder_cls(model_type) is BUILDERS_REGISTRY[model_type]

    @pytest.mark.parametrize("model_type,expected_cls", [
        ("cifar_resnet20", CifarResNet),
        ("cifar_vgg8", CifarVGG8),
        ("cifar_vit_leaf", LeafVisionTransformer),
    ])
    def test_default_build_runs_on_cifar_shape(self, model_type, expected_cls):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        builder = BUILDERS_REGISTRY[model_type](
            device="cpu", input_shape=(3, 32, 32), num_classes=10,
            pipeline_config={},
        )
        model = builder.build({}).eval()
        assert isinstance(model, expected_cls)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)

    def test_resnet20_default_build_is_resnet20(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        model = BUILDERS_REGISTRY["cifar_resnet20"](
            device="cpu", input_shape=(3, 32, 32), num_classes=10,
            pipeline_config={},
        ).build({})
        blocks = [m for m in model.modules() if isinstance(m, CifarBasicBlock)]
        assert len(blocks) == 9

    def test_vit_leaf_default_build_is_cifar_config(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        model = BUILDERS_REGISTRY["cifar_vit_leaf"](
            device="cpu", input_shape=(3, 32, 32), num_classes=10,
            pipeline_config={},
        ).build({})
        assert model.pos_embed.shape == (1, 65, 192)
        assert len(model.blocks) == 7

    def test_validate_config(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        resnet = BUILDERS_REGISTRY["cifar_resnet20"]
        assert resnet.validate_config({}, {}, (3, 32, 32))
        assert not resnet.validate_config({}, {}, (3, 30, 32))
        assert not resnet.validate_config({"blocks_per_stage": 0}, {}, (3, 32, 32))
        assert not resnet.validate_config({}, {}, (3, 32))

        vgg = BUILDERS_REGISTRY["cifar_vgg8"]
        assert vgg.validate_config({}, {}, (3, 32, 32))
        assert not vgg.validate_config({}, {}, (3, 12, 12))
        assert not vgg.validate_config({"base_channels": 0}, {}, (3, 32, 32))

        vit = BUILDERS_REGISTRY["cifar_vit_leaf"]
        assert vit.validate_config({}, {}, (3, 32, 32))
        assert not vit.validate_config({"patch_size": 5}, {}, (3, 32, 32))
        assert not vit.validate_config({"num_heads": 5}, {}, (3, 32, 32))
        assert not vit.validate_config({}, {}, (3, 32, 48))  # non-square
