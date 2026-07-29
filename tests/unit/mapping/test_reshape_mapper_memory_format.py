"""Shape mappers must be memory-format agnostic (W0.9).

``ReshapeMapper`` / ``SplitLeadingDimMapper`` express a pure LOGICAL reshape of
a batch-major tensor. They used ``Tensor.view``, which additionally demands a
compatible stride layout -- so a channels-last activation (what cuDNN hands
back for some conv configurations) crashed them outright.

The layout reaches them only when nothing in between re-materializes the
tensor: a real ``BatchNorm`` does, ``nn.Identity`` does not. Normalization
Fusion turns every perceptron's normalization into ``nn.Identity``, which is
why the crash appeared the moment the bias-free CIFAR vehicles got past
Quantization Verification into fusion (observed: (128, 64, 2, 2) with stride
(256, 1, 128, 64) at the VGG-8 flatten).

``reshape`` is the correct primitive: identical values and identical output
layout semantics, a free view when the strides allow one and a copy otherwise.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.mappers.leading_dim import SplitLeadingDimMapper
from mimarsinan.mapping.mappers.structural import InputMapper, ReshapeMapper


def _channels_last(t: torch.Tensor) -> torch.Tensor:
    out = t.contiguous(memory_format=torch.channels_last)
    assert not out.is_contiguous(), "fixture failed to produce a strided layout"
    return out


class TestReshapeMapperIsMemoryFormatAgnostic:
    def _mapper(self, output_shape):
        return ReshapeMapper(InputMapper((64, 2, 2)), output_shape)

    def test_channels_last_input_flattens_without_crashing(self):
        x = _channels_last(torch.arange(2 * 64 * 2 * 2, dtype=torch.float32)
                           .reshape(2, 64, 2, 2))
        out = self._mapper((256,)).forward(x)
        assert out.shape == (2, 256)

    def test_values_follow_the_logical_layout_not_the_memory_layout(self):
        """The flatten must reorder by (C, H, W), exactly like the contiguous
        case -- a silent memory-order flatten would permute the features."""
        base = torch.arange(2 * 64 * 2 * 2, dtype=torch.float32).reshape(2, 64, 2, 2)
        expected = self._mapper((256,)).forward(base.contiguous())
        actual = self._mapper((256,)).forward(_channels_last(base))
        assert torch.equal(actual, expected)

    def test_contiguous_input_is_unchanged(self):
        x = torch.randn(3, 64, 2, 2)
        out = self._mapper((256,)).forward(x)
        assert torch.equal(out, x.reshape(3, 256))


class TestSplitLeadingDimMapperIsMemoryFormatAgnostic:
    def test_non_contiguous_2d_input_splits_by_logical_order(self):
        base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        strided = base.t().contiguous().t()
        assert not strided.is_contiguous()
        mapper = SplitLeadingDimMapper(InputMapper((4,)), 3)
        assert torch.equal(mapper.forward(strided), base.reshape(2, 3, 4))
