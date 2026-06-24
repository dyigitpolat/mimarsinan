"""Unit tests for the deep_cnn (configurable-depth CNN) model, builder, registration, mapping, and smoke template."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.deep_cnn import DeepCNN, allowed_pool_count
from mimarsinan.models.builders.deep_cnn_builder import DeepCNNBuilder
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


DEVICE = torch.device("cpu")
PIPELINE_CONFIG = {"target_tq": 4, "device": "cpu"}


def _conv_layers(module: nn.Module) -> list[nn.Conv2d]:
    return [m for m in module.modules() if isinstance(m, nn.Conv2d)]


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


def _bn_layers(module: nn.Module) -> list[nn.BatchNorm2d]:
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


def _pool_layers(module: nn.Module) -> list[nn.MaxPool2d]:
    return [m for m in module.modules() if isinstance(m, nn.MaxPool2d)]


class TestAllowedPoolCount:
    def test_28_never_collapses(self):
        # 28 -> 14 -> 7 -> 3 -> 1: a 28x28 input tolerates at most 3 halving pools
        # while keeping spatial >= 2 (the conv-block floor).
        n = allowed_pool_count(28)
        assert n >= 1
        size = 28
        for _ in range(n):
            size //= 2
        assert size >= 2

    def test_32_never_collapses(self):
        n = allowed_pool_count(32)
        assert n >= 1
        size = 32
        for _ in range(n):
            size //= 2
        assert size >= 2

    def test_pools_are_capped(self):
        # A larger input allows strictly more pools than a smaller one (monotone).
        assert allowed_pool_count(32) >= allowed_pool_count(28)


class TestDeepCNN:
    @pytest.mark.parametrize("depth", [4, 8, 12])
    def test_conv_block_count_matches_depth_mnist(self, depth):
        model = DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=depth, width=16)
        convs = _conv_layers(model)
        assert len(convs) == depth, f"expected {depth} conv blocks, got {len(convs)}"
        # one BatchNorm + one ReLU implied per conv block
        assert len(_bn_layers(model)) == depth
        n_relu = sum(isinstance(m, nn.ReLU) for m in model.modules())
        assert n_relu == depth
        # k3 SAME padding on every conv (mapper-critical), no grouped/depthwise conv
        for c in convs:
            assert c.kernel_size == (3, 3)
            assert c.padding == (1, 1)
            assert c.groups == 1
        # exactly one classifier Linear -> n_classes
        linears = _linear_layers(model)
        assert len(linears) == 1
        assert linears[0].out_features == 10

    @pytest.mark.parametrize("depth", [4, 8, 12])
    def test_forward_shape_dummy_mnist(self, depth):
        model = DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=depth, width=16).eval()
        x = torch.randn(5, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (5, 10)

    @pytest.mark.parametrize("depth", [4, 8, 12])
    def test_forward_shape_dummy_svhn(self, depth):
        model = DeepCNN(input_shape=(3, 32, 32), num_classes=10, depth=depth, width=16).eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 10)

    def test_channels_grow_and_cap_at_128(self):
        model = DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=16, width=16)
        convs = _conv_layers(model)
        out_channels = [c.out_channels for c in convs]
        # monotone non-decreasing growth
        assert out_channels == sorted(out_channels)
        # never exceeds the 128 cap
        assert max(out_channels) <= 128
        # first conv takes the dataset's input channels
        assert convs[0].in_channels == 1

    def test_pool_count_respects_cap(self):
        model = DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=12, width=16)
        assert len(_pool_layers(model)) <= allowed_pool_count(28)

    def test_adapts_to_input_channels_and_classes(self):
        model = DeepCNN(input_shape=(3, 32, 32), num_classes=7, depth=6, width=16).eval()
        convs = _conv_layers(model)
        assert convs[0].in_channels == 3
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            assert model(x).shape == (2, 7)

    def test_rejects_non_chw_input_shape(self):
        with pytest.raises(ValueError):
            DeepCNN(input_shape=(784,), num_classes=10, depth=4)

    def test_rejects_out_of_range_depth(self):
        with pytest.raises(ValueError):
            DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=3)
        with pytest.raises(ValueError):
            DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=17)


class TestDeepCNNBuilder:
    def _builder(self, input_shape=(1, 28, 28), num_classes=10):
        return DeepCNNBuilder(
            device=DEVICE,
            input_shape=input_shape,
            num_classes=num_classes,
            pipeline_config=PIPELINE_CONFIG,
        )

    def test_build_returns_deep_cnn_module(self):
        model = self._builder().build({"depth": 8, "width": 16, "base_activation": "ReLU"})
        assert isinstance(model, nn.Module)
        assert isinstance(model, DeepCNN)
        assert len(_conv_layers(model)) == 8
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert model(x).shape == (2, 10)

    def test_build_uses_schema_defaults_when_keys_missing(self):
        """A probe call before the form renders must not raise (schema-default fallback)."""
        model = self._builder().build({})
        assert isinstance(model, DeepCNN)
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert model(x).shape == (2, 10)

    def test_registered_as_deep_cnn_torch_category(self):
        assert ModelRegistry.get_category("deep_cnn") == "torch"
        assert ModelRegistry.get_builder_cls("deep_cnn") is DeepCNNBuilder

    def test_in_builders_registry(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        assert BUILDERS_REGISTRY["deep_cnn"] is DeepCNNBuilder

    def test_config_schema_keys(self):
        keys = {f["key"] for f in DeepCNNBuilder.get_config_schema()}
        assert {"depth", "width", "base_activation"} <= keys


class TestDeepCNNMapping:
    """deep_cnn must convert + map FEASIBLE at depth 8, mirroring test_lenet5."""

    def test_representability_depth8(self):
        from mimarsinan.torch_mapping.converter import check_representability

        model = DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=8, width=16)
        report = check_representability(model, input_shape=(1, 28, 28))
        assert report.is_representable, f"DeepCNN d8 not representable: {report.summary()}"

    def test_convert_and_verify_soft_core_mapping_feasible_depth8(self):
        from mimarsinan.mapping.verification.verifier import (
            verify_soft_core_mapping,
            verify_hardware_config,
        )
        from mimarsinan.mapping.verification.suggester.hw_config_suggester import (
            suggest_hardware_config,
        )
        from mimarsinan.torch_mapping.converter import convert_torch_model

        builder = DeepCNNBuilder(
            device=DEVICE,
            input_shape=(1, 28, 28),
            num_classes=10,
            pipeline_config=PIPELINE_CONFIG,
        )
        raw = builder.build({"depth": 8, "width": 16, "base_activation": "ReLU"})
        raw.eval()
        with torch.no_grad():
            _ = raw(torch.randn(1, 1, 28, 28))
        supermodel = convert_torch_model(
            raw, input_shape=(1, 28, 28), num_classes=10, device="cpu", Tq=4
        )
        model_repr = supermodel.get_mapper_repr()

        result = verify_soft_core_mapping(model_repr, max_axons=1024, max_neurons=1024)
        assert result.feasible, f"deep_cnn d8 soft-core mapping failed: {result.error}"
        assert result.num_neural_cores > 0 or result.host_side_segment_count > 0

        suggestion = suggest_hardware_config(result.softcores)
        if result.softcores:
            assert suggestion.total_cores > 0
            v = verify_hardware_config(result.softcores, suggestion.core_types)
            assert v["feasible"], f"Suggested config not feasible: {v['errors']}"

    def test_produces_conv_neural_cores_depth8(self):
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.torch_mapping.converter import convert_torch_model

        raw = DeepCNN(input_shape=(1, 28, 28), num_classes=10, depth=8, width=16).eval()
        with torch.no_grad():
            _ = raw(torch.randn(1, 1, 28, 28))
        supermodel = convert_torch_model(raw, input_shape=(1, 28, 28), num_classes=10)
        mapper_repr = supermodel.get_mapper_repr()

        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024)
        ir_graph = ir.map(mapper_repr)
        n_nc = len(ir_graph.get_neural_cores())
        n_co = len(ir_graph.get_compute_ops())
        # a depth-8 conv stack must yield several neural cores
        assert n_nc >= 1
        assert n_nc + n_co >= 8


class TestDeepCNNSmokeTemplate:
    def test_smoke_template_parses(self, tmp_path):
        """templates/mnist_deep_cnn_d8_cascaded.json must parse via _parse_deployment_config."""
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        template = repo_root / "templates" / "mnist_deep_cnn_d8_cascaded.json"
        assert template.exists(), f"missing template: {template}"
        config = json.loads(template.read_text())

        assert config["deployment_parameters"]["model_type"] == "deep_cnn"
        assert config["deployment_parameters"]["model_config"]["depth"] == 8
        assert config["deployment_parameters"]["model_config"]["width"] == 16
        assert config["deployment_parameters"]["model_config"]["base_activation"] == "ReLU"
        assert config["deployment_parameters"]["spiking_mode"] == "ttfs_cycle_based"
        assert config["deployment_parameters"]["ttfs_cycle_schedule"] == "cascaded"
        assert config["platform_constraints"]["simulation_steps"] == 4
        assert config["deployment_parameters"]["max_simulation_samples"] <= 200
        assert config["deployment_parameters"]["enable_sanafe_simulation"] is False

        sys.path.insert(0, str(repo_root / "src"))
        try:
            from main import _parse_deployment_config
        except ModuleNotFoundError:
            pytest.skip("src.main not importable in this test context")
        config["generated_files_path"] = str(tmp_path / "generated")
        config["datasets_path"] = str(repo_root / "datasets")
        parsed = _parse_deployment_config(config)
        assert parsed["deployment_parameters"]["model_type"] == "deep_cnn"
        assert parsed["deployment_name"]
