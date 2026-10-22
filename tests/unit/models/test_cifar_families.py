"""CIFAR DenseNet-BC and additive PyramidNet: geometry, layer orders, parameter counts, registration.

Sourced counts: DenseNet-BC (L = 100, k = 12) 0.8 M (huang2016densely Table 2
p5); PyramidNet-110 alpha 48 1.7 M and alpha 84 3.8 M (han2016deep Table 4
p7). The papers print one decimal, so a computed count must lie within 0.1 M.
"""

import pytest
import torch
import torch.fx
import torch.nn as nn

from mimarsinan.model_training.training_recipe import _infer_block_depth
from mimarsinan.models.builders import BUILDERS_REGISTRY
from mimarsinan.models.families.densenet_bc import (
    CifarDenseNetBC,
    DenseBottleneckLayer,
    densenet_bc_layers_per_block,
)
from mimarsinan.models.families.pyramidnet import (
    CifarPyramidNet,
    PyramidBasicUnit,
    pyramidnet_units_per_group,
    pyramidnet_widths,
)
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

C100 = (3, 32, 32)


def _params(model):
    return sum(p.numel() for p in model.parameters())


def _bn_producers(model):
    """For every BatchNorm call in the traced graph, the kind of node feeding it."""
    gm = torch.fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    kinds = []
    for node in gm.graph.nodes:
        if node.op == "call_module" and isinstance(modules[node.target], nn.BatchNorm2d):
            src = node.args[0]
            if src.op == "call_module":
                kinds.append(type(modules[src.target]).__name__)
            else:
                kinds.append(getattr(src.target, "__name__", str(src.target)))
    return kinds


class TestDenseNetBCGeometry:
    def test_layers_per_block_rule(self):
        assert densenet_bc_layers_per_block(100) == 16
        assert densenet_bc_layers_per_block(10) == 1
        for bad in (99, 4, 0):
            with pytest.raises(ValueError):
                densenet_bc_layers_per_block(bad)

    @pytest.mark.parametrize("order", ["published", "post_activation"])
    def test_forward_shape(self, order):
        model = CifarDenseNetBC(C100, 100, depth=100, growth_rate=12, layer_order=order).eval()
        with torch.no_grad():
            assert model(torch.randn(2, *C100)).shape == (2, 100)

    def test_channel_bookkeeping_k12(self):
        model = CifarDenseNetBC(C100, 100, depth=100, growth_rate=12)
        assert model.stem_conv.out_channels == 24
        assert model.dense1.out_channels == 24 + 16 * 12 == 216
        assert model.trans1.conv.out_channels == 108
        assert model.dense2.out_channels == 108 + 192 == 300
        assert model.trans2.conv.out_channels == 150
        assert model.classifier.in_features == model.dense3.out_channels == 342
        layers = [m for m in model.modules() if isinstance(m, DenseBottleneckLayer)]
        assert len(layers) == 48
        assert all(l.conv1.out_channels == 48 and l.conv2.out_channels == 12 for l in layers)
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 96 + 2 + 1  # dense convs, transitions, stem
        assert len(convs) + 1 == 100  # plus the classifier: L = 100

    def test_transitions_halve_resolution(self):
        model = CifarDenseNetBC(C100, 100, depth=10, growth_rate=4).eval()
        seen = {}

        def record(_m, inp, _o):
            seen["s"] = tuple(inp[0].shape)

        model.head_pool.register_forward_hook(record)
        with torch.no_grad():
            model(torch.randn(1, *C100))
        assert seen["s"][2:] == (8, 8)

    def test_published_order_normalizes_concatenations(self):
        kinds = _bn_producers(CifarDenseNetBC(C100, 10, depth=10, growth_rate=4, layer_order="published"))
        assert "cat" in kinds and "Conv2d" in kinds

    def test_post_activation_order_puts_every_bn_after_its_conv(self):
        model = CifarDenseNetBC(C100, 10, depth=10, growth_rate=4, layer_order="post_activation")
        kinds = _bn_producers(model)
        assert kinds and set(kinds) == {"Conv2d"}
        assert sum(isinstance(m, nn.BatchNorm2d) for m in model.modules()) == sum(
            isinstance(m, nn.Conv2d) for m in model.modules()
        )

    @pytest.mark.parametrize("kwargs", [
        {"layer_order": "pre"}, {"growth_rate": 0}, {"compression": 0.0}, {"compression": 1.5}, {"depth": 101},
    ])
    def test_rejects_bad_configs(self, kwargs):
        with pytest.raises(ValueError):
            CifarDenseNetBC(C100, 100, **kwargs)


class TestPyramidNetGeometry:
    def test_units_and_widths(self):
        assert pyramidnet_units_per_group(110) == 18
        widths = pyramidnet_widths(110, 48)
        assert len(widths) == 54
        assert widths[-1] == 64 and pyramidnet_widths(110, 84)[-1] == 100  # Table 4 "Output Feat. Dim."
        assert widths == [16 + (48 * j) // 54 for j in range(1, 55)]
        assert widths == sorted(widths)
        for bad in (109, 2):
            with pytest.raises(ValueError):
                pyramidnet_units_per_group(bad)

    @pytest.mark.parametrize("alpha", [48, 84])
    def test_forward_shape(self, alpha):
        model = CifarPyramidNet(C100, 100, depth=110, alpha=alpha).eval()
        with torch.no_grad():
            assert model(torch.randn(2, *C100)).shape == (2, 100)

    def test_downsampling_units_and_shortcuts(self):
        model = CifarPyramidNet(C100, 100, depth=110, alpha=48)
        units = [m for m in model.units if isinstance(m, PyramidBasicUnit)]
        pooled = [i for i, u in enumerate(units) if u.shortcut_pool is not None]
        assert pooled == [18, 36]
        assert [units[i].conv1.stride for i in pooled] == [(2, 2), (2, 2)]
        assert all(u.pad_channels == u.conv1.out_channels - u.conv1.in_channels for u in units)
        assert not any(isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) for m in model.modules())

    def test_branch_output_is_bn_without_relu(self):
        unit = PyramidBasicUnit(16, 20, 1).eval()
        gm = torch.fx.symbolic_trace(unit)
        add = [n for n in gm.graph.nodes if n.op == "call_function" and n.target.__name__ == "add"]
        assert len(add) == 1
        assert add[0].args[0].op == "call_module" and add[0].args[0].target == "bn3"

    def test_zero_padded_shortcut_is_exact_identity_on_kept_channels(self):
        unit = PyramidBasicUnit(4, 6, 1).eval()
        for bn in (unit.bn3,):
            nn.init.zeros_(bn.weight)
            nn.init.zeros_(bn.bias)
        x = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            y = unit(x)
        assert torch.equal(y[:, :4], x) and torch.equal(y[:, 4:], torch.zeros(2, 2, 8, 8))


class TestSourcedParameterCounts:
    @pytest.mark.parametrize("model,sourced,exact", [
        (lambda: CifarDenseNetBC(C100, 100, 100, 12), 0.8e6, 800_032),
        (lambda: CifarPyramidNet(C100, 100, 110, 48), 1.7e6, 1_745_580),
        (lambda: CifarPyramidNet(C100, 100, 110, 84), 3.8e6, 3_864_144),
    ])
    def test_counts_match_the_papers(self, model, sourced, exact):
        count = _params(model())
        assert count == exact
        assert abs(count - sourced) < 0.1e6

    def test_cifar10_densenet_bc_100_is_the_familiar_0_77m(self):
        assert _params(CifarDenseNetBC(C100, 10, 100, 12)) == 769_162

    def test_post_activation_k12_and_published_k24_counts(self):
        assert _params(CifarDenseNetBC(C100, 100, 100, 12, layer_order="post_activation")) == 782_368
        assert _params(CifarDenseNetBC(C100, 100, 100, 24)) == 3_082_156


class TestLlrdStaysInert:
    @pytest.mark.parametrize("model", [
        lambda: CifarDenseNetBC(C100, 10, 10, 4), lambda: CifarPyramidNet(C100, 10, 8, 8),
    ])
    def test_no_parameter_name_matches_a_depth_pattern(self, model):
        assert all(_infer_block_depth(name) is None for name, _ in model().named_parameters())


class TestRegistration:
    def test_registered_as_torch_builders(self):
        for key in ("cifar_densenet_bc", "cifar_pyramidnet"):
            assert key in BUILDERS_REGISTRY
            assert ModelRegistry.get_category(key) == "torch"

    def test_defaults_build_the_paper_configurations(self):
        dn = BUILDERS_REGISTRY["cifar_densenet_bc"]("cpu", C100, 100, {}).build({})
        assert (dn.depth, dn.growth_rate, dn.compression, dn.layer_order) == (100, 12, 0.5, "published")
        pn = BUILDERS_REGISTRY["cifar_pyramidnet"]("cpu", C100, 100, {}).build({})
        assert (pn.depth, pn.alpha) == (110, 48)

    def test_builder_passes_the_layer_order(self):
        dn = BUILDERS_REGISTRY["cifar_densenet_bc"]("cpu", C100, 10, {}).build(
            {"depth": 10, "growth_rate": 4, "layer_order": "post_activation"}
        )
        assert dn.layer_order == "post_activation"

    def test_validate_config(self):
        dn = BUILDERS_REGISTRY["cifar_densenet_bc"]
        assert dn.validate_config({"depth": 100}, {}, C100)
        assert not dn.validate_config({"depth": 99}, {}, C100)
        assert not dn.validate_config({"layer_order": "x"}, {}, C100)
        pn = BUILDERS_REGISTRY["cifar_pyramidnet"]
        assert pn.validate_config({"depth": 110, "alpha": 84}, {}, C100)
        assert not pn.validate_config({"depth": 111}, {}, C100)
        assert not pn.validate_config({}, {}, (3, 30, 30))
