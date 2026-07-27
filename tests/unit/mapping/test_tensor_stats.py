"""Large-tensor-safe quantile: torch.quantile has a hard 2^24 input cap."""

import pytest
import torch

from mimarsinan.mapping.support.tensor_stats import (
    QUANTILE_ELEMENT_LIMIT,
    safe_quantile,
    subsample_to_limit,
)


class TestSubsampleToLimit:
    def test_small_tensor_keeps_every_element(self):
        x = torch.arange(10.0)
        out = subsample_to_limit(x, limit=100)
        assert out.numel() == x.numel()
        torch.testing.assert_close(out, x)

    @pytest.mark.parametrize("n,limit", [
        (1000, 100),      # exact multiple
        (1050, 100),      # NOT a multiple: a floor stride overshoots to 105
        (77_070_336 // 64, (1 << 24) // 64),  # the ViT patch-embed ratio
        (101, 100),
        (199, 100),
    ])
    def test_result_never_exceeds_the_limit(self, n, limit):
        out = subsample_to_limit(torch.arange(float(n)), limit=limit)
        assert 0 < out.numel() <= limit

    def test_multidimensional_input_is_flattened(self):
        x = torch.randn(4, 5, 6)
        assert subsample_to_limit(x, limit=10_000).dim() == 1


class TestSafeQuantile:
    def test_matches_torch_quantile_under_the_limit(self):
        torch.manual_seed(0)
        x = torch.randn(5000)
        torch.testing.assert_close(
            safe_quantile(x, 0.999), torch.quantile(x, 0.999)
        )

    @pytest.mark.parametrize("q", [0.0, 0.5, 0.9, 0.999, 1.0])
    def test_oversized_path_approximates_the_true_quantile(self, q):
        # The fallback subsamples; on a uniform population the estimate must
        # track the exact quantile closely (calibration-grade, not exact).
        torch.manual_seed(1)
        x = torch.rand(20_000)
        got = safe_quantile(x, q, limit=1000)
        want = torch.quantile(x, q)
        assert abs(float(got) - float(want)) < 0.05

    def test_oversized_path_is_deterministic(self):
        torch.manual_seed(2)
        x = torch.randn(20_000)
        a = safe_quantile(x, 0.99, limit=997)
        b = safe_quantile(x, 0.99, limit=997)
        assert float(a) == float(b)

    def test_default_limit_is_the_torch_cap(self):
        # torch.quantile raises above 2^24 elements; the default must not
        # exceed it or the guard is vacuous.
        assert QUANTILE_ELEMENT_LIMIT <= (1 << 24)

    def test_result_is_a_scalar_tensor_on_the_input_device(self):
        x = torch.randn(2000)
        out = safe_quantile(x, 0.5, limit=100)
        assert out.dim() == 0 and out.device == x.device

    def test_non_float_input_is_promoted(self):
        x = torch.arange(5000, dtype=torch.int64)
        assert float(safe_quantile(x, 0.5, limit=100)) > 0
