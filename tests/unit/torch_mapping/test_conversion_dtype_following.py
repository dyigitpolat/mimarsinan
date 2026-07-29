"""Conversion follows the source model's parameter dtype (no silent fp32 downcast).

BN absorption COMPUTES folded weights, so an fp64 source model must land fp64
perceptron parameters and BN stats — otherwise fp64 value-parity carries an
irreducible ~1e-9 fold-rounding floor that belongs to fp32 storage, not to the
conversion. The tracer's ShapeProp example input and the conversion probe
input follow the model dtype too (fp64 models trace and probe cleanly).
"""

import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import Conv2DPerceptronMapper, PerceptronMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model


class _ConvBNLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 8 * 8, 5)

    def forward(self, x):
        return self.fc(self.flatten(self.relu(self.bn(self.conv(x)))))


def _mappers_of(flow, cls):
    seen, stack, found = set(), [flow.get_mapper_repr().output_layer_mapper], []
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, cls):
            found.append(node)
        if hasattr(node, "get_source_mappers"):
            stack.extend(node.get_source_mappers())
        elif getattr(node, "source_mapper", None) is not None:
            stack.append(node.source_mapper)
    return found


class TestConversionDtypeFollowing:
    def test_fp64_model_lands_fp64_perceptrons_and_bn(self):
        torch.manual_seed(0)
        model = _ConvBNLinearNet().double().eval()
        flow = convert_torch_model(
            model, (3, 8, 8), num_classes=5, packaging=MVM_PACKAGING
        )

        convs = _mappers_of(flow, Conv2DPerceptronMapper)
        linears = _mappers_of(flow, PerceptronMapper)
        assert convs and linears
        for mapper in convs + linears:
            assert mapper.perceptron.layer.weight.dtype == torch.float64
            norm = mapper.perceptron.normalization
            if isinstance(norm, (nn.BatchNorm1d, nn.BatchNorm2d)):
                assert norm.running_var is not None
                assert norm.running_var.dtype == torch.float64
                assert norm.weight.dtype == torch.float64

    def test_fp32_model_stays_fp32(self):
        torch.manual_seed(0)
        model = _ConvBNLinearNet().eval()
        flow = convert_torch_model(
            model, (3, 8, 8), num_classes=5, packaging=MVM_PACKAGING
        )
        for mapper in _mappers_of(flow, (Conv2DPerceptronMapper, PerceptronMapper)):
            assert mapper.perceptron.layer.weight.dtype == torch.float32
