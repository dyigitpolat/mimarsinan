"""Tests for the torch_mapping module.

Covers tracing, representability analysis, Mapper DAG conversion,
and the public converter facade.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.torch_graph_tracer import trace_model, TracingError
from mimarsinan.torch_mapping.representability_analyzer import (
    RepresentabilityAnalyzer,
    RepresentabilityReport,
    RepresentabilityError,
)
from mimarsinan.torch_mapping.mapper_graph_converter import MapperGraphConverter
from mimarsinan.torch_mapping.converter import convert_torch_model, check_representability
from mimarsinan.torch_mapping.converted_model_flow import ConvertedModelFlow


# ── Helper models ────────────────────────────────────────────────────────────

class SimpleMLP(nn.Module):
    """Linear -> ReLU -> Linear."""
    def __init__(self, in_features=16, hidden=32, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


class MLPWithBN(nn.Module):
    """Linear -> BatchNorm1d -> ReLU -> Linear."""
    def __init__(self, in_features=16, hidden=32, out_features=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)


class SimpleConvNet(nn.Module):
    """Conv2d -> BN -> ReLU -> MaxPool -> Flatten -> Linear."""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv(x))))
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    """A model with a residual (add) connection."""
    def __init__(self, features=16, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(features, features)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(features, features)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(features, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        residual = x
        out = self.relu1(self.fc1(x))
        out = self.fc2(out)
        out = self.relu2(out + residual)
        return self.head(out)


class UnsupportedModel(nn.Module):
    """Contains an LSTM which is unsupported."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        return self.fc(out[:, -1, :])


# ── Tests: Tracing ───────────────────────────────────────────────────────────

class TestTracing:
    def test_trace_simple_mlp(self):
        model = SimpleMLP(in_features=16)
        gm = trace_model(model, input_shape=(16,))
        assert gm is not None
        nodes = [n for n in gm.graph.nodes if n.op != "placeholder" and n.op != "output"]
        assert len(nodes) > 0

    def test_trace_conv_net(self):
        model = SimpleConvNet(in_channels=1)
        gm = trace_model(model, input_shape=(1, 8, 8))
        assert gm is not None

    def test_trace_shape_propagation(self):
        model = SimpleMLP(in_features=16)
        gm = trace_model(model, input_shape=(16,))
        has_meta = False
        for node in gm.graph.nodes:
            if "tensor_meta" in node.meta:
                has_meta = True
                break
        assert has_meta, "ShapeProp should annotate at least one node"


# ── Tests: Representability ──────────────────────────────────────────────────

class TestRepresentability:
    def test_simple_mlp_representable(self):
        model = SimpleMLP(in_features=16)
        report = check_representability(model, input_shape=(16,))
        assert report.is_representable

    def test_conv_net_representable(self):
        model = SimpleConvNet(in_channels=1)
        report = check_representability(model, input_shape=(1, 8, 8))
        assert report.is_representable

    def test_residual_representable(self):
        model = ResidualBlock(features=16)
        report = check_representability(model, input_shape=(16,))
        assert report.is_representable

    def test_bn_absorbed(self):
        model = MLPWithBN(in_features=16, hidden=32)
        gm = trace_model(model, input_shape=(16,))
        analyzer = RepresentabilityAnalyzer(gm)
        report = analyzer.analyze()
        assert report.is_representable
        assert len(report.absorption_plan) > 0, "BatchNorm should be in absorption plan"

    def test_unsupported_detected(self):
        model = UnsupportedModel()
        report = check_representability(model, input_shape=(16,))
        assert not report.is_representable
        assert len(report.unsupported_ops) > 0

    def test_report_summary(self):
        model = SimpleMLP(in_features=16)
        report = check_representability(model, input_shape=(16,))
        summary = report.summary()
        assert "Representable: True" in summary


# ── Tests: Conversion ────────────────────────────────────────────────────────

class TestConversion:
    def test_convert_simple_mlp(self):
        model = SimpleMLP(in_features=16, hidden=32, out_features=10)
        supermodel = convert_torch_model(
            model, input_shape=(16,), num_classes=10
        )
        assert supermodel is not None
        perceptrons = supermodel.get_perceptrons()
        assert len(perceptrons) >= 2

    def test_convert_mlp_with_bn(self):
        model = MLPWithBN(in_features=16, hidden=32, out_features=10)
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(2, 16)
            _ = model(dummy)

        supermodel = convert_torch_model(
            model, input_shape=(16,), num_classes=10
        )
        perceptrons = supermodel.get_perceptrons()
        assert len(perceptrons) >= 2

        has_bn = any(
            not isinstance(p.normalization, nn.Identity)
            for p in perceptrons
        )
        assert has_bn, "At least one Perceptron should have absorbed BatchNorm"

    def test_convert_conv_net(self):
        model = SimpleConvNet(in_channels=1, num_classes=10)
        model.eval()
        with torch.no_grad():
            _ = model(torch.randn(1, 1, 8, 8))

        supermodel = convert_torch_model(
            model, input_shape=(1, 8, 8), num_classes=10
        )
        assert supermodel is not None
        perceptrons = supermodel.get_perceptrons()
        assert len(perceptrons) >= 2

    def test_convert_residual(self):
        model = ResidualBlock(features=16, num_classes=10)
        supermodel = convert_torch_model(
            model, input_shape=(16,), num_classes=10
        )
        assert supermodel is not None

    def test_convert_preserves_output_shape(self):
        model = SimpleMLP(in_features=16, hidden=32, out_features=10)
        supermodel = convert_torch_model(
            model, input_shape=(16,), num_classes=10
        )

        supermodel.eval()
        with torch.no_grad():
            dummy = torch.randn(4, 16)
            out = supermodel(dummy)
        assert out.shape == (4, 10)

    def test_convert_unsupported_raises(self):
        model = UnsupportedModel()
        with pytest.raises(RepresentabilityError):
            convert_torch_model(model, input_shape=(16,), num_classes=10)

    def test_mapper_repr_exists(self):
        model = SimpleMLP(in_features=16, hidden=32, out_features=10)
        supermodel = convert_torch_model(
            model, input_shape=(16,), num_classes=10
        )
        mapper_repr = supermodel.get_mapper_repr()
        assert mapper_repr is not None

    def test_weight_transfer(self):
        model = SimpleMLP(in_features=16, hidden=32, out_features=10)
        fc1_weight_sum = model.fc1.weight.data.sum().item()

        supermodel = convert_torch_model(
            model, input_shape=(16,), num_classes=10
        )

        perceptrons = supermodel.get_perceptrons()
        transferred_sums = [p.layer.weight.data.sum().item() for p in perceptrons]
        assert any(
            abs(s - fc1_weight_sum) < 1e-4 for s in transferred_sums
        ), "fc1 weight sum should appear in one of the converted perceptrons"


# ── Tests: ConvertedModelFlow ────────────────────────────────────────────────

class TestConvertedModelFlow:
    def test_get_perceptrons(self):
        model = SimpleMLP(in_features=16)
        supermodel = convert_torch_model(model, input_shape=(16,), num_classes=10)
        flow = supermodel.perceptron_flow
        assert isinstance(flow, ConvertedModelFlow)
        assert len(flow.get_perceptrons()) >= 2

    def test_get_mapper_repr(self):
        model = SimpleMLP(in_features=16)
        supermodel = convert_torch_model(model, input_shape=(16,), num_classes=10)
        flow = supermodel.perceptron_flow
        mapper_repr = flow.get_mapper_repr()
        assert mapper_repr is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
