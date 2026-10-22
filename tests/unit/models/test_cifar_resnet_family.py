"""E294: one CIFAR ResNet family builder over declared options.

Spec `07_engineering_refinements.md` E294: stem width, width multiplier, block
kind (basic, bottleneck), block order (post- or pre-activation), shortcut
(projection, zero-padded identity) and n up to 111, with ``cifar_resnet20``'s
defaults. Sourced counts at 100 classes (MEASURED, mapping audit §4 and
`RA/06_build/audit_v42/arith/runs/h0/macs.json`, the forms of `arith_macs.py`):
ResNet-110 1,736,564; WRN-16-8 post-activation 11,009,332 (zagoruyko2016wide
Table 4 p8 prints 11.0 M); pre-activation ResNet-164 1,726,388 and ResNet-1001
10,350,836 (he2016identity Table 4 p12 prints 1.7 M and 10.2 M).
"""

from __future__ import annotations

import sys
import warnings

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.mapping.verification.verifier.mapping_verifier_soft import verify_soft_core_mapping
from mimarsinan.model_training.training_recipe import _infer_block_depth
from mimarsinan.models.builders import BUILDERS_REGISTRY
from mimarsinan.models.cifar_models import CifarResNet
from mimarsinan.models.families.cifar_resnet import (
    BLOCK_KINDS,
    BLOCK_ORDERS,
    MAX_BLOCKS_PER_STAGE,
    SHORTCUTS,
    CifarResNetFamily,
    ResNetFamilyOptionError,
    resnet_family_depth,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.torch_mapping.converter import convert_torch_model

from unit.mapping.recursive_walk_reference import (
    DEFAULT_RECURSION_LIMIT,
    ir_bytes,
    recursive_walk,
)

C = (3, 32, 32)
KEY = "cifar_resnet_family"
PREACT_BOTTLENECK = {"block_kind": "bottleneck", "block_order": "pre_activation"}


def _params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _family(num_classes=10, shape=C, **cfg):
    return BUILDERS_REGISTRY[KEY]("cpu", shape, num_classes, {}).build(cfg)


def _weight_layers(model: nn.Module) -> int:
    """Convs and Linears outside the shortcuts: the depth the family's names count."""
    return sum(
        isinstance(m, (nn.Conv2d, nn.Linear)) and ".shortcut" not in f".{name}"
        for name, m in model.named_modules()
    )


RESNET20_CONFIGS = [
    {"blocks_per_stage": n, "base_width": w} for n in (2, 3, 5, 7, 9) for w in (8, 16, 24, 32)
]


class TestDefaultsAreResNet20:
    def test_registered_defaults_build_cifar_resnet20(self):
        torch.manual_seed(0)
        family = _family()
        torch.manual_seed(0)
        reference = BUILDERS_REGISTRY["cifar_resnet20"]("cpu", C, 10, {}).build({})
        assert list(family.state_dict()) == list(reference.state_dict())
        x = torch.randn(2, *C, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            assert torch.equal(family.eval()(x), reference.eval()(x))

    @pytest.mark.parametrize("cfg", RESNET20_CONFIGS, ids=lambda c: f"n{c['blocks_per_stage']}w{c['base_width']}")
    def test_every_cifar_resnet20_configuration_is_identical(self, cfg):
        torch.manual_seed(3)
        family = _family(**cfg)
        torch.manual_seed(3)
        reference = BUILDERS_REGISTRY["cifar_resnet20"]("cpu", C, 10, {}).build(cfg)
        assert isinstance(reference, CifarResNet)
        reference_state = reference.state_dict()
        assert list(family.state_dict()) == list(reference_state)
        assert all(torch.equal(v, reference_state[k]) for k, v in family.state_dict().items())
        x = torch.randn(1, *C, generator=torch.Generator().manual_seed(2))
        with torch.no_grad():
            assert torch.equal(family.eval()(x), reference.eval()(x))


class TestSourcedParameterCounts:
    @pytest.mark.parametrize("name,cfg,exact,paper", [
        ("ResNet-110", {"blocks_per_stage": 18}, 1_736_564, None),
        ("WRN-16-8 post-activation", {"blocks_per_stage": 2, "width_multiplier": 8}, 11_009_332, 11.0e6),
        ("ResNet-164 pre-activation", {"blocks_per_stage": 18, **PREACT_BOTTLENECK}, 1_726_388, 1.7e6),
        ("ResNet-1001 pre-activation", {"blocks_per_stage": 111, **PREACT_BOTTLENECK}, 10_350_836, 10.2e6),
        ("WRN-16-8 pre-activation", {"blocks_per_stage": 2, "width_multiplier": 8,
                                     "block_order": "pre_activation"}, 11_007_540, None),
    ])
    def test_counts_equal_the_measured_forms(self, name, cfg, exact, paper):
        count = _params(CifarResNetFamily(C, 100, **cfg))
        print(f"[E294] {name}: {count:,} parameters at 100 classes; paper "
              f"{'n/a' if paper is None else f'{paper / 1e6:.1f} M'}")
        assert count == exact

    @pytest.mark.parametrize("cfg,depth", [
        ({"blocks_per_stage": 3}, 20), ({"blocks_per_stage": 18}, 110),
        ({"blocks_per_stage": 18, **PREACT_BOTTLENECK}, 164),
        ({"blocks_per_stage": 2, "width_multiplier": 8}, 14),
        ({"blocks_per_stage": 111, "block_kind": "bottleneck"}, 1001),
    ])
    def test_depth_is_the_weight_layer_count(self, cfg, depth):
        model = CifarResNetFamily(C, 10, **cfg)
        assert resnet_family_depth(cfg["blocks_per_stage"], cfg.get("block_kind", "basic")) == depth
        assert _weight_layers(model) == depth


class TestDeclaredOptions:
    def test_option_sets(self):
        assert BLOCK_KINDS == ("basic", "bottleneck")
        assert BLOCK_ORDERS == ("post_activation", "pre_activation")
        assert SHORTCUTS == ("projection", "zero_pad")
        assert MAX_BLOCKS_PER_STAGE == 111

    @pytest.mark.parametrize("kind", BLOCK_KINDS)
    @pytest.mark.parametrize("order", BLOCK_ORDERS)
    @pytest.mark.parametrize("shortcut", SHORTCUTS)
    def test_every_option_combination_runs(self, kind, order, shortcut):
        model = CifarResNetFamily((3, 8, 8), 4, blocks_per_stage=1, base_width=4,
                                  block_kind=kind, block_order=order, shortcut=shortcut).eval()
        with torch.no_grad():
            assert model(torch.randn(2, 3, 8, 8)).shape == (2, 4)

    def test_zero_padded_shortcut_is_parameter_free_and_subsamples(self):
        projection = CifarResNetFamily(C, 10, blocks_per_stage=1)
        zero_pad = CifarResNetFamily(C, 10, blocks_per_stage=1, shortcut="zero_pad")
        shortcut_params = sum(p.numel() for n, p in projection.named_parameters() if ".shortcut." in n)
        assert shortcut_params > 0
        assert _params(zero_pad) == _params(projection) - shortcut_params
        block = zero_pad.stages[1][0].eval()
        x = torch.randn(1, 16, 8, 8)
        with torch.no_grad():
            skip = block.shortcut(x)
        assert skip.shape == (1, 32, 4, 4)
        assert torch.equal(skip[:, :16], x[:, :, ::2, ::2]) and not skip[:, 16:].any()

    def test_stem_width_defaults_to_the_base_width(self):
        assert CifarResNetFamily(C, 10, base_width=24).stem_conv.out_channels == 24
        assert CifarResNetFamily(C, 10, base_width=16, width_multiplier=8).stem_conv.out_channels == 16
        assert CifarResNetFamily(C, 10, stem_width=12).stem_conv.out_channels == 12

    def test_pre_activation_has_no_stem_activation_and_a_final_one(self):
        model = CifarResNetFamily(C, 10, block_order="pre_activation")
        assert model.stem_bn is None and model.final_bn is not None

    @pytest.mark.parametrize("kwargs", [
        {"blocks_per_stage": 0}, {"blocks_per_stage": 112}, {"base_width": 0}, {"width_multiplier": 0},
        {"block_kind": "wide"}, {"block_order": "middle"}, {"shortcut": "dense"}, {"stem_width": 0},
        {"stem_width": 32, "shortcut": "zero_pad"},
    ])
    def test_rejects_undeclared_options(self, kwargs):
        with pytest.raises(ResNetFamilyOptionError) as refused:
            CifarResNetFamily(C, 10, **kwargs)
        assert isinstance(refused.value, ToolchainRefusal) and isinstance(refused.value, ValueError)

    def test_parameter_names_keep_layer_wise_decay_inert(self):
        model = CifarResNetFamily((3, 8, 8), 4, blocks_per_stage=2, **PREACT_BOTTLENECK)
        assert all(_infer_block_depth(name) is None for name, _ in model.named_parameters())


class TestRegistration:
    def test_registered_as_a_torch_builder(self):
        assert KEY in BUILDERS_REGISTRY
        assert ModelRegistry.get_category(KEY) == "torch"

    def test_builder_passes_every_option(self):
        model = _family(blocks_per_stage=2, base_width=8, stem_width=4, width_multiplier=2,
                        block_kind="bottleneck", block_order="pre_activation", shortcut="zero_pad")
        assert (model.blocks_per_stage, model.base_width, model.stem_width, model.width_multiplier) == (2, 8, 4, 2)
        assert (model.block_kind, model.block_order, model.shortcut) == ("bottleneck", "pre_activation", "zero_pad")

    def test_validate_config(self):
        builder = BUILDERS_REGISTRY[KEY]
        assert builder.validate_config({"blocks_per_stage": 111, **PREACT_BOTTLENECK}, {}, C)
        assert not builder.validate_config({"blocks_per_stage": 112}, {}, C)
        assert not builder.validate_config({"block_kind": "wide"}, {}, C)
        assert not builder.validate_config({}, {}, (3, 30, 30))


def _convert(model, shape, packaging, *, strict=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return convert_torch_model(model.eval(), shape, 10, packaging=packaging, strict=strict)


class TestConversion:
    @pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
    @pytest.mark.parametrize("cfg", [
        {"blocks_per_stage": 2, **PREACT_BOTTLENECK},
        {"blocks_per_stage": 1, "block_order": "pre_activation"},
        {"blocks_per_stage": 1, "block_order": "pre_activation", "shortcut": "zero_pad"},
        {"blocks_per_stage": 1, "shortcut": "zero_pad"},
    ], ids=["preact_bottleneck", "preact_basic", "preact_zero_pad", "post_zero_pad"])
    def test_converts_within_1e_12_in_fp64(self, packaging, cfg):
        """``strict`` conversion raises unless the fp64 value-parity probe stays within 1e-12."""
        torch.manual_seed(0)
        model = CifarResNetFamily((3, 8, 8), 10, base_width=4, **cfg).double()
        flow = _convert(model, (3, 8, 8), packaging)
        assert flow.conversion_certificate.certified, flow.conversion_certificate.summary()

    def test_resnet164_ir_equals_the_recursive_walk(self):
        """The ResNet-164 graph at a narrow width and small input: the walk order does not depend on either."""
        torch.manual_seed(0)
        flow = _convert(CifarResNetFamily((3, 8, 8), 10, base_width=2, blocks_per_stage=18, **PREACT_BOTTLENECK),
                        (3, 8, 8), MVM_PACKAGING, strict=False)
        walked = ir_bytes(IRMapping(q_max=1.0, firing_mode="Default").map(flow.get_mapper_repr()))
        flow.get_mapper_repr().clear_ir_caches()
        with recursive_walk(50_000):
            reference = ir_bytes(IRMapping(q_max=1.0, firing_mode="Default").map(flow.get_mapper_repr()))
        assert walked == reference


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
def test_resnet1001_converts_maps_and_lays_out_at_the_default_limit(packaging):
    """The full-depth member at a narrow width and an 8 x 8 input (memory scales with both, depth does not)."""
    assert sys.getrecursionlimit() == DEFAULT_RECURSION_LIMIT
    torch.manual_seed(0)
    model = CifarResNetFamily((3, 8, 8), 10, base_width=2, blocks_per_stage=111, **PREACT_BOTTLENECK)
    flow = _convert(model, (3, 8, 8), packaging, strict=False)
    ir = IRMapping(q_max=1.0, firing_mode="Default").map(flow.get_mapper_repr())
    assert ir.get_neural_cores()
    flow.get_mapper_repr().clear_ir_caches()
    soft = verify_soft_core_mapping(flow.get_mapper_repr(), 256, 256)
    assert soft.feasible and soft.error is None, soft.error
    assert LayoutIRMapping(max_axons=256, max_neurons=256).collect_layout_softcores(flow.get_mapper_repr())
    assert sys.getrecursionlimit() == DEFAULT_RECURSION_LIMIT
