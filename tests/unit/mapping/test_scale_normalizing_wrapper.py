"""Tests for :class:`ScaleNormalizingWrapper` parity with the legacy Add scaling.

The historical ``_exec_add`` formula was
``(s_a/s_out) * a + (s_b/s_out) * b`` where ``s_out = (s_a + s_b) / 2``.
:class:`ScaleNormalizingWrapper` generalises that to any multi-input
module via ``f(r_i * s_i) / s_out``; for ``f = Add`` it reduces to the
same arithmetic.
"""

from __future__ import annotations

import pytest
import torch

import operator

from mimarsinan.mapping.compute_modules import ComputeAdapter, ScaleNormalizingWrapper


def _add():
    """Test-local helper: binary add as a ``ComputeAdapter``."""
    return ComputeAdapter(operator.add)


class TestAddParityWithLegacyScaling:
    def test_uniform_scales_equals_plain_add(self):
        wrapper = ScaleNormalizingWrapper(
            _add(),
            [torch.tensor([1.0]), torch.tensor([1.0])],
            torch.tensor([1.0]),
        )
        a = torch.tensor([[3.0]])
        b = torch.tensor([[4.0]])
        out = wrapper(a, b)
        assert torch.allclose(out, a + b)

    def test_matches_legacy_add_formula(self):
        """Equivalent to legacy ``(s_a/s_out)*a + (s_b/s_out)*b``."""
        s_a = torch.tensor([2.0, 4.0])
        s_b = torch.tensor([6.0, 8.0])
        s_out = (s_a + s_b) / 2.0   # legacy heuristic

        wrapper = ScaleNormalizingWrapper(_add(), [s_a, s_b], s_out)

        a = torch.tensor([[0.5, 0.25]])
        b = torch.tensor([[0.3, 0.1]])
        out = wrapper(a, b)

        # f(s_a * a, s_b * b) / s_out = (s_a*a + s_b*b) / s_out
        expected = (s_a * a + s_b * b) / s_out
        assert torch.allclose(out, expected, atol=1e-6)

    def test_broadcast_to_last_dim(self):
        """Scalar scales broadcast over channels via ``broadcast_scale_to_dim``."""
        wrapper = ScaleNormalizingWrapper(
            _add(),
            [torch.tensor([2.0]), torch.tensor([4.0])],
            torch.tensor([3.0]),
        )
        a = torch.ones(1, 3)
        b = torch.ones(1, 3)
        out = wrapper(a, b)
        expected = torch.full((1, 3), (1.0 * 2.0 + 1.0 * 4.0) / 3.0)
        assert torch.allclose(out, expected)


class TestPicklable:
    def test_wrapper_roundtrips_through_pickle(self):
        import pickle
        wrapper = ScaleNormalizingWrapper(
            _add(),
            [torch.tensor([2.0, 2.0]), torch.tensor([4.0, 4.0])],
            torch.tensor([3.0, 3.0]),
        )
        loaded = pickle.loads(pickle.dumps(wrapper))
        a = torch.tensor([[1.0, 1.0]])
        b = torch.tensor([[1.0, 1.0]])
        assert torch.allclose(wrapper(a, b), loaded(a, b))


class TestThreeInputModule:
    """The wrapper is op-agnostic — extends naturally past binary ops."""

    def test_three_way_weighted_sum(self):
        class Sum3(torch.nn.Module):
            def forward(self, a, b, c):
                return a + b + c

        wrapper = ScaleNormalizingWrapper(
            Sum3(),
            [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])],
            torch.tensor([2.0]),
        )
        a = torch.tensor([[1.0]])
        b = torch.tensor([[1.0]])
        c = torch.tensor([[1.0]])
        out = wrapper(a, b, c)
        # (1*1 + 1*2 + 1*3) / 2 = 3.0
        assert torch.allclose(out, torch.tensor([[3.0]]))
