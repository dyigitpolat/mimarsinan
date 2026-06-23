"""Unit tests for the lenet5 (LeNet-5) model, builder, registration, mapping, and smoke template."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.lenet5 import LeNet5
from mimarsinan.models.builders.lenet5_builder import LeNet5Builder
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry


DEVICE = torch.device("cpu")
PIPELINE_CONFIG = {"target_tq": 4, "device": "cpu"}


def _conv_layers(module: nn.Module) -> list[nn.Conv2d]:
    return [m for m in module.modules() if isinstance(m, nn.Conv2d)]


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


class TestLeNet5:
    def test_conv_and_linear_structure_mnist(self):
        """Classic LeNet-5: Conv 1->6 k5, Conv 6->16 k5, then FC 120, 84, n_classes."""
        model = LeNet5(input_shape=(1, 28, 28), num_classes=10, base_activation="ReLU")

        convs = _conv_layers(model)
        assert len(convs) == 2
        assert convs[0].in_channels == 1 and convs[0].out_channels == 6
        assert convs[0].kernel_size == (5, 5)
        assert convs[1].in_channels == 6 and convs[1].out_channels == 16
        assert convs[1].kernel_size == (5, 5)
        # No grouped/depthwise conv (hard-unsupported).
        assert convs[0].groups == 1 and convs[1].groups == 1

        linears = _linear_layers(model)
        assert len(linears) == 3
        # MNIST 28x28, k5 pad2 (SAME): 28 -> pool 14 -> 14 -> pool 7 ; flat = 16*7*7 = 784
        assert linears[0].in_features == 16 * 7 * 7
        assert linears[0].out_features == 120
        assert linears[1].in_features == 120 and linears[1].out_features == 84
        assert linears[2].in_features == 84 and linears[2].out_features == 10

        n_pool = sum(isinstance(m, nn.MaxPool2d) for m in model.modules())
        assert n_pool == 2

    def test_forward_shape_on_dummy_mnist_batch(self):
        model = LeNet5(input_shape=(1, 28, 28), num_classes=10).eval()
        x = torch.randn(5, 1, 28, 28)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (5, 10)

    def test_adapts_to_input_channels_and_classes(self):
        model = LeNet5(input_shape=(3, 32, 32), num_classes=7).eval()
        convs = _conv_layers(model)
        assert convs[0].in_channels == 3
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            assert model(x).shape == (2, 7)

    def test_rejects_non_chw_input_shape(self):
        with pytest.raises(ValueError):
            LeNet5(input_shape=(784,), num_classes=10)


class TestLeNet5Builder:
    def _builder(self, input_shape=(1, 28, 28), num_classes=10):
        return LeNet5Builder(
            device=DEVICE,
            input_shape=input_shape,
            num_classes=num_classes,
            pipeline_config=PIPELINE_CONFIG,
        )

    def test_build_returns_lenet5_module(self):
        model = self._builder().build({"variant": "lenet5", "base_activation": "ReLU"})
        assert isinstance(model, nn.Module)
        assert isinstance(model, LeNet5)
        assert len(_conv_layers(model)) == 2
        assert len(_linear_layers(model)) == 3
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert model(x).shape == (2, 10)

    def test_build_uses_schema_defaults_when_keys_missing(self):
        """A probe call before the form renders must not raise (schema-default fallback)."""
        model = self._builder().build({})
        assert isinstance(model, LeNet5)
        x = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            assert model(x).shape == (2, 10)

    def test_registered_as_lenet5_torch_category(self):
        assert ModelRegistry.get_category("lenet5") == "torch"
        assert ModelRegistry.get_builder_cls("lenet5") is LeNet5Builder

    def test_in_builders_registry(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        assert BUILDERS_REGISTRY["lenet5"] is LeNet5Builder

    def test_config_schema_keys(self):
        keys = {f["key"] for f in LeNet5Builder.get_config_schema()}
        assert {"variant", "base_activation"} <= keys


class TestLeNet5Mapping:
    """lenet5 must convert + map FEASIBLE, mirroring test_all_builders_mapping."""

    def test_representability(self):
        from mimarsinan.torch_mapping.converter import check_representability

        model = LeNet5(input_shape=(1, 28, 28), num_classes=10)
        report = check_representability(model, input_shape=(1, 28, 28))
        assert report.is_representable, f"LeNet5 not representable: {report.summary()}"

    def test_convert_and_verify_soft_core_mapping_feasible(self):
        from mimarsinan.mapping.verification.verifier import (
            verify_soft_core_mapping,
            verify_hardware_config,
        )
        from mimarsinan.mapping.verification.suggester.hw_config_suggester import (
            suggest_hardware_config,
        )
        from mimarsinan.torch_mapping.converter import convert_torch_model

        builder = LeNet5Builder(
            device=DEVICE,
            input_shape=(1, 28, 28),
            num_classes=10,
            pipeline_config=PIPELINE_CONFIG,
        )
        raw = builder.build({"variant": "lenet5", "base_activation": "ReLU"})
        raw.eval()
        with torch.no_grad():
            _ = raw(torch.randn(1, 1, 28, 28))
        supermodel = convert_torch_model(
            raw, input_shape=(1, 28, 28), num_classes=10, device="cpu", Tq=4
        )
        model_repr = supermodel.get_mapper_repr()

        result = verify_soft_core_mapping(model_repr, max_axons=1024, max_neurons=1024)
        assert result.feasible, f"lenet5 soft-core mapping failed: {result.error}"
        assert result.num_neural_cores > 0 or result.host_side_segment_count > 0

        suggestion = suggest_hardware_config(result.softcores)
        if result.softcores:
            assert suggestion.total_cores > 0
            v = verify_hardware_config(result.softcores, suggestion.core_types)
            assert v["feasible"], f"Suggested config not feasible: {v['errors']}"

    def test_produces_conv_neural_cores_and_pool_compute_ops(self):
        """LeNet maps the two convs to neural cores and the two pools to ComputeOps."""
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.torch_mapping.converter import convert_torch_model

        raw = LeNet5(input_shape=(1, 28, 28), num_classes=10).eval()
        supermodel = convert_torch_model(raw, input_shape=(1, 28, 28), num_classes=10)
        mapper_repr = supermodel.get_mapper_repr()

        ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=1024, max_neurons=1024)
        ir_graph = ir.map(mapper_repr)
        n_nc = len(ir_graph.get_neural_cores())
        n_co = len(ir_graph.get_compute_ops())
        assert n_co >= 1  # MaxPool / encoding host ops
        assert n_nc + n_co >= 3


class TestLeNet5SmokeTemplate:
    def test_smoke_template_parses(self, tmp_path):
        """templates/mnist_lenet5_synchronized.json must parse via _parse_deployment_config."""
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        template = repo_root / "templates" / "mnist_lenet5_synchronized.json"
        assert template.exists(), f"missing template: {template}"
        config = json.loads(template.read_text())

        assert config["deployment_parameters"]["model_type"] == "lenet5"
        assert config["deployment_parameters"]["model_config"]["variant"] == "lenet5"
        assert config["deployment_parameters"]["model_config"]["base_activation"] == "ReLU"
        assert config["deployment_parameters"]["spiking_mode"] == "ttfs_cycle_based"
        assert config["deployment_parameters"]["ttfs_cycle_schedule"] == "synchronized"
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
        assert parsed["deployment_parameters"]["model_type"] == "lenet5"
        assert parsed["deployment_name"]
