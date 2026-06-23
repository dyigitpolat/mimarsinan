"""Unit tests for the deep_mlp depth-probe model, builder, and registration."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.deep_mlp import DeepMLP
from mimarsinan.models.builders.deep_mlp_builder import DeepMLPBuilder
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


def _count_linear(module: nn.Module) -> int:
    return sum(isinstance(m, nn.Linear) for m in module.modules())


class TestDeepMLP:
    @pytest.mark.parametrize("depth", [4, 8, 16])
    def test_linear_count_and_forward_shape(self, depth):
        """depth hidden layers + 1 classifier => depth+1 Linear; forward => [N, n_classes]."""
        width = 64
        model = DeepMLP(
            input_shape=(1, 28, 28),
            num_classes=10,
            depth=depth,
            width=width,
            base_activation="ReLU",
        )
        assert _count_linear(model) == depth + 1, (
            f"depth={depth}: expected {depth + 1} Linear layers (depth hidden + classifier), "
            f"got {_count_linear(model)}."
        )
        x = torch.randn(3, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (3, 10)

    def test_relu_count_matches_depth(self):
        """One ReLU per hidden layer, none after the classifier."""
        model = DeepMLP(
            input_shape=(1, 28, 28),
            num_classes=10,
            depth=8,
            width=64,
            base_activation="ReLU",
        )
        n_relu = sum(isinstance(m, nn.ReLU) for m in model.modules())
        assert n_relu == 8

    def test_width_is_respected(self):
        model = DeepMLP(
            input_shape=(1, 28, 28),
            num_classes=10,
            depth=4,
            width=64,
            base_activation="ReLU",
        )
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        # First Linear maps flattened input -> width.
        assert linears[0].in_features == 28 * 28
        assert linears[0].out_features == 64
        # Hidden Linears are width -> width.
        for lin in linears[1:-1]:
            assert lin.in_features == 64 and lin.out_features == 64
        # Classifier maps width -> n_classes.
        assert linears[-1].in_features == 64
        assert linears[-1].out_features == 10


class TestDeepMLPBuilder:
    def _builder(self, input_shape=(1, 28, 28), num_classes=10):
        return DeepMLPBuilder(
            device=torch.device("cpu"),
            input_shape=input_shape,
            num_classes=num_classes,
            pipeline_config={"target_tq": 4},
        )

    @pytest.mark.parametrize("depth", [4, 8, 16])
    def test_build_returns_module_with_expected_depth(self, depth):
        model = self._builder().build({"depth": depth, "width": 64, "base_activation": "ReLU"})
        assert isinstance(model, nn.Module)
        assert isinstance(model, DeepMLP)
        assert _count_linear(model) == depth + 1
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 10)

    def test_build_uses_schema_defaults_when_keys_missing(self):
        """A probe call before the form renders must not raise (schema-default fallback)."""
        model = self._builder().build({})
        assert isinstance(model, DeepMLP)
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert model(x).shape == (2, 10)

    def test_registered_as_deep_mlp_torch_category(self):
        assert ModelRegistry.get_category("deep_mlp") == "torch"
        assert ModelRegistry.get_builder_cls("deep_mlp") is DeepMLPBuilder

    def test_in_builders_registry(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        assert BUILDERS_REGISTRY["deep_mlp"] is DeepMLPBuilder

    def test_config_schema_keys(self):
        keys = {f["key"] for f in DeepMLPBuilder.get_config_schema()}
        assert {"depth", "width", "base_activation"} <= keys


class TestDeepMLPConversion:
    """The pure Linear+ReLU stack must convert through the same torch->perceptron path."""

    def test_converted_flow_matches_original(self):
        from mimarsinan.torch_mapping.converter import convert_torch_model

        model = DeepMLP(
            input_shape=(1, 28, 28),
            num_classes=10,
            depth=4,
            width=64,
            base_activation="ReLU",
        )
        model.eval()
        flow = convert_torch_model(model, input_shape=(1, 28, 28), num_classes=10)
        flow.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            orig = model(x)
            conv = flow(x)
        assert orig.shape == conv.shape
        assert torch.allclose(orig, conv, atol=1e-3), (
            f"converted flow diverges (max diff {(orig - conv).abs().max().item():.6f})."
        )


class TestDeepMLPSmokeTemplate:
    def test_smoke_template_parses(self, tmp_path):
        """templates/mnist_deep_mlp_d8_cascaded.json must parse via _parse_deployment_config."""
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        template = repo_root / "templates" / "mnist_deep_mlp_d8_cascaded.json"
        assert template.exists(), f"missing template: {template}"
        config = json.loads(template.read_text())

        assert config["deployment_parameters"]["model_type"] == "deep_mlp"
        assert config["deployment_parameters"]["model_config"]["depth"] == 8
        assert config["deployment_parameters"]["model_config"]["width"] == 64
        assert config["deployment_parameters"]["spiking_mode"] == "ttfs_cycle_based"
        assert config["deployment_parameters"]["ttfs_cycle_schedule"] == "cascaded"
        assert config["deployment_parameters"]["enable_sanafe_simulation"] is False

        sys.path.insert(0, str(repo_root / "src"))
        try:
            from main import _parse_deployment_config
        except ModuleNotFoundError:
            pytest.skip("src.main not importable in this test context")
        config["generated_files_path"] = str(tmp_path / "generated")
        config["datasets_path"] = str(repo_root / "datasets")
        parsed = _parse_deployment_config(config)
        assert parsed["deployment_parameters"]["model_type"] == "deep_mlp"
        assert parsed["deployment_name"]
