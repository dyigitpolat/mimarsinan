"""D19 on a converted value-domain flow: the WQ entry pass and projection keep a signed-input stem's bias.

The flow is built the pipeline's way (MVM packaging, normalization fusion, calibrated boundary grids on
every segment entry); the stem channel is constant-OFF for inputs in [0, 1] but not on its grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
    NormalizationAwarePerceptronQuantization,
)
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from mimarsinan.transformations.perceptron.bias_canonicalization import canonicalize_starved_bias_outliers
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

INPUT_SHAPE = (3, 4, 4)
STEM_BIAS = -0.5
"""Below the unit bound -0.2 of the planted stem channel, far above its grid bound (about -7.5)."""


class _StemNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x):
        return self.fc(torch.flatten(self.relu(self.bn(self.conv(x))), 1))


def _value_flow():
    torch.manual_seed(0)
    flow = convert_torch_model(_StemNet().eval(), INPUT_SHAPE, 3, device="cpu", packaging=MVM_PACKAGING).eval()
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    batches = [torch.randn(64, *INPUT_SHAPE, generator=torch.Generator().manual_seed(1))]
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    calibrate_boundary_scales(flow, entries, install_boundary_quantizers(entries, 5), batches)
    stem = flow.get_perceptrons()[0]
    with torch.no_grad():
        planted = torch.full((27,), -0.1)
        planted[:4] = 0.05
        stem.layer.weight.data[0] = planted
        stem.layer.bias.data[0] = STEM_BIAS
    return flow, stem, batches


def test_value_flow_stem_is_affine_and_reads_a_signed_grid():
    flow, stem, _ = _value_flow()
    assert stem.base_activation_name == "Identity"
    assert float(stem.input_activation.scale) > 1.0


def test_wq_entry_pass_keeps_the_stem_bias_and_the_flow_function():
    flow, stem, batches = _value_flow()
    with torch.no_grad():
        before = [flow(x).clone() for x in batches]
    canonicalize_starved_bias_outliers(flow, batches, bits=4)
    assert float(PerceptronTransformer().get_effective_bias(stem)[0]) == STEM_BIAS
    with torch.no_grad():
        for x, ref in zip(batches, before):
            assert torch.equal(flow(x), ref)


def test_projection_keeps_the_stem_bias_on_its_own_grid():
    _, stem, _ = _value_flow()
    NormalizationAwarePerceptronQuantization(4, "cpu", rate=1.0, two_scale=True).transform(stem)
    bias_scale = float(stem.bias_scale)
    code = float(PerceptronTransformer().get_effective_bias(stem)[0]) * bias_scale
    expected = float(torch.round(torch.tensor(STEM_BIAS * bias_scale)))
    assert abs(code - expected) < 1e-4, (code, expected)
