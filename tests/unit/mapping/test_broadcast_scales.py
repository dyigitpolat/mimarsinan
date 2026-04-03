"""Tests for _broadcast_scale_pair in per_source_scales.py."""

import pytest
import torch

from mimarsinan.mapping.per_source_scales import _broadcast_scale_pair


class TestBroadcastScalePair:
    def test_equal_length_no_change(self):
        s_a = torch.tensor([1.0, 2.0, 3.0])
        s_b = torch.tensor([4.0, 5.0, 6.0])
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert torch.equal(r_a, s_a)
        assert torch.equal(r_b, s_b)

    def test_shorter_a_divisible(self):
        """When len(s_a) < len(s_b) and n_b % n_a == 0, s_a is repeat_interleaved."""
        s_a = torch.tensor([1.0, 2.0, 3.0])
        s_b = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 6
        assert len(r_b) == 6
        expected_a = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        assert torch.allclose(r_a, expected_a)
        assert torch.equal(r_b, s_b)

    def test_shorter_b_divisible(self):
        """When len(s_b) < len(s_a) and n_a % n_b == 0, s_b is repeat_interleaved."""
        s_a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        s_b = torch.tensor([10.0, 20.0, 30.0])
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 6
        assert len(r_b) == 6
        assert torch.equal(r_a, s_a)
        expected_b = torch.tensor([10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
        assert torch.allclose(r_b, expected_b)

    def test_non_divisible_fallback_to_mean(self):
        """When lengths are not divisible, shorter is replaced with mean-filled tensor."""
        s_a = torch.tensor([1.0, 2.0, 3.0])
        s_b = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0])
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 5
        assert len(r_b) == 5
        assert torch.allclose(r_a, torch.full((5,), 2.0))
        assert torch.equal(r_b, s_b)

    def test_vit_shape_mismatch_3_vs_3072(self):
        """The exact case from the CIFAR-10 ViT bug: 3 (RGB) vs 3072 (MLP hidden)."""
        s_a = torch.tensor([1.0, 1.0, 1.0])
        s_b = torch.ones(3072) * 2.0
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 3072
        assert len(r_b) == 3072
        assert torch.allclose(r_a, torch.ones(3072))
        assert torch.allclose(r_b, torch.ones(3072) * 2.0)

    def test_order_preserved_when_a_shorter(self):
        """Return order matches input order: (expanded_a, original_b)."""
        s_a = torch.tensor([5.0])
        s_b = torch.tensor([1.0, 2.0, 3.0])
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 3
        assert torch.allclose(r_a, torch.full((3,), 5.0))
        assert torch.equal(r_b, s_b)

    def test_order_preserved_when_b_shorter(self):
        """Return order matches input order: (original_a, expanded_b)."""
        s_a = torch.tensor([1.0, 2.0, 3.0])
        s_b = torch.tensor([5.0])
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 3
        assert len(r_b) == 3
        assert torch.equal(r_a, s_a)
        assert torch.allclose(r_b, torch.full((3,), 5.0))

    def test_single_element_vs_many(self):
        """Single-element tensor should broadcast cleanly (always divisible)."""
        s_a = torch.tensor([7.0])
        s_b = torch.ones(100) * 3.0
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        assert len(r_a) == 100
        assert torch.allclose(r_a, torch.full((100,), 7.0))

    def test_combined_average_after_broadcast(self):
        """After broadcasting, element-wise average should work without errors."""
        s_a = torch.tensor([1.0, 2.0, 3.0])
        s_b = torch.ones(3072) * 2.0
        r_a, r_b = _broadcast_scale_pair(s_a, s_b)
        combined = (r_a + r_b) / 2.0
        assert combined.shape == (3072,)
