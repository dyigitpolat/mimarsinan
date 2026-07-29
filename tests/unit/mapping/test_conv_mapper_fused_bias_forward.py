"""Normalization Fusion must survive on BIAS-FREE conv perceptrons (W0.9).

``fuse_into_perceptron`` folds the normalization into the layer, replacing a
bias-free ``nn.Linear`` with a bias-CARRYING one whose bias is the folded BN
term. The conv mappers' ``_forward_impl`` used to gate the additive term on
the construction-time ``self.bias`` flag, which is ``False`` for the CIFAR
VGG-8 / ResNet-20 idiom (``nn.Conv2d(..., bias=False)`` + BN) and stays stale
after fusion -- so the whole folded bias was silently dropped from the forward.

Fusion is a pure refactor of the same function: the forward must be unchanged.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron


def _train_norm(perceptron, x_shape, seed=0):
    """Give the (lazy) normalization real stats and a non-trivial affine."""
    torch.manual_seed(seed)
    perceptron.train()
    with torch.no_grad():
        for _ in range(3):
            perceptron.normalization(torch.randn(*x_shape) * 2.0 + 1.0)
        perceptron.normalization.weight.data = (
            torch.randn(x_shape[1]).abs() + 0.5
        )
        perceptron.normalization.bias.data = torch.randn(x_shape[1]) * 1.5
    perceptron.eval()


class TestConv2DBiasFreeFusionIsForwardPreserving:
    def _mapper(self, *, bias):
        torch.manual_seed(1)
        mapper = Conv2DPerceptronMapper(
            InputMapper((3, 8, 8)),
            in_channels=3, out_channels=4,
            kernel_size=3, stride=1, padding=1,
            bias=bias, use_batchnorm=True,
        )
        _train_norm(mapper.perceptron, (16, 4, 9))
        mapper.eval()
        return mapper

    @pytest.mark.parametrize("bias", [False, True])
    def test_forward_is_unchanged_by_fusion(self, bias):
        mapper = self._mapper(bias=bias)
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            before = mapper.forward(x).clone()
        fuse_into_perceptron(mapper.perceptron, device="cpu")
        with torch.no_grad():
            after = mapper.forward(x)
        assert torch.allclose(before, after, atol=1e-4), (
            f"fusion moved the conv forward by "
            f"{float((before - after).abs().max()):.6g}"
        )

    def test_bias_free_fusion_actually_installs_a_nonzero_bias(self):
        """Guard the fixture: without a real folded bias the test is vacuous."""
        mapper = self._mapper(bias=False)
        assert mapper.perceptron.layer.bias is None
        fuse_into_perceptron(mapper.perceptron, device="cpu")
        assert mapper.perceptron.layer.bias is not None
        assert float(mapper.perceptron.layer.bias.data.abs().max()) > 1e-3

    def test_bias_free_without_normalization_stays_bias_free(self):
        torch.manual_seed(2)
        mapper = Conv2DPerceptronMapper(
            InputMapper((3, 8, 8)),
            in_channels=3, out_channels=4, kernel_size=3, padding=1,
            bias=False, use_batchnorm=False,
        )
        mapper.eval()
        x = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            before = mapper.forward(x).clone()
        fuse_into_perceptron(mapper.perceptron, device="cpu")
        assert mapper.perceptron.layer.bias is None
        with torch.no_grad():
            assert torch.equal(mapper.forward(x), before)


class TestConv1DBiasFreeFusionIsForwardPreserving:
    def _mapper(self, *, bias):
        torch.manual_seed(3)
        mapper = Conv1DPerceptronMapper(
            InputMapper((3, 8)),
            in_channels=3, out_channels=4,
            kernel_size=3, stride=1, padding=1,
            bias=bias, use_batchnorm=True,
        )
        _train_norm(mapper.perceptron, (16, 4, 8))
        mapper.eval()
        return mapper

    @pytest.mark.parametrize("bias", [False, True])
    def test_forward_is_unchanged_by_fusion(self, bias):
        mapper = self._mapper(bias=bias)
        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            before = mapper.forward(x).clone()
        fuse_into_perceptron(mapper.perceptron, device="cpu")
        with torch.no_grad():
            after = mapper.forward(x)
        assert torch.allclose(before, after, atol=1e-4), (
            f"fusion moved the conv1d forward by "
            f"{float((before - after).abs().max()):.6g}"
        )
