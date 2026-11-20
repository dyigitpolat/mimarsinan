"""Large-tensor-safe quantile: torch.quantile has a hard 2^24 input cap.

[U-BRANGE F-3] Above the cap the quantile is read exactly from its order statistics (selection, no sort and no size
cap), never from a subsample: a strided subsample of a flattened (batch, token, channel) tensor whose channel count is
even read even channels only (W38 group 6's diagnosis, m-2: ViT-B/16's GELU seams, 15.5% below to 9.9% above the exact
quantiles).
"""

import pytest
import torch

from mimarsinan.mapping.support.tensor_stats import (
    QUANTILE_ELEMENT_LIMIT,
    safe_quantile,
)
from unit.pipelining.range_rule_reference import exact_quantile


class TestTheOversizedPathIsExact:
    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 0.999, 1.0])
    def test_linear_quantile_equals_the_full_sort_s(self, q):
        x = torch.randn(20_000, generator=torch.Generator().manual_seed(4))
        got = float(safe_quantile(x, q, limit=1000))
        assert got == pytest.approx(exact_quantile(x, q), rel=1e-6, abs=1e-7)

    @pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "midpoint", "nearest"])
    @pytest.mark.parametrize("q", [0.1, 0.5, 0.999])
    def test_every_interpolation_equals_torch_s_on_the_whole_tensor(self, q, interpolation):
        x = torch.rand(30_001, generator=torch.Generator().manual_seed(5))
        got = float(safe_quantile(x, q, limit=999, interpolation=interpolation))
        want = float(torch.quantile(x, q, interpolation=interpolation))
        assert got == pytest.approx(want, rel=1e-6, abs=1e-7)

    def test_a_seam_whose_last_axis_is_even_is_read_on_every_channel(self):
        """(batch, token, channel) with an even channel count: the odd channels carry the large values. A stride of 2
        over the flattened tensor reads even channels only; the exact path reads them all."""
        generator = torch.Generator().manual_seed(6)
        x = torch.rand(8, 16, 64, generator=generator)
        x[..., 1::2] *= 10.0
        limit = x.numel() // 2  # a stride of 2 above it
        got = float(safe_quantile(x.abs(), 0.999, limit=limit))
        assert got == pytest.approx(exact_quantile(x.abs(), 0.999), rel=1e-6)
        assert got > 5.0  # the odd channels' tail, which an even-channel read never sees

    def test_the_exact_path_is_used_only_above_the_limit(self):
        x = torch.randn(5000, generator=torch.Generator().manual_seed(7))
        assert torch.equal(safe_quantile(x, 0.999), torch.quantile(x, 0.999))

    def test_an_out_of_range_q_is_refused_on_both_paths(self):
        x = torch.rand(2000)
        for limit in (10_000, 100):
            with pytest.raises((ValueError, RuntimeError)):
                safe_quantile(x, 1.5, limit=limit)


class TestSafeQuantile:
    def test_matches_torch_quantile_under_the_limit(self):
        torch.manual_seed(0)
        x = torch.randn(5000)
        torch.testing.assert_close(
            safe_quantile(x, 0.999), torch.quantile(x, 0.999)
        )

    @pytest.mark.parametrize("q", [0.0, 0.5, 0.9, 0.999, 1.0])
    def test_oversized_path_approximates_the_true_quantile(self, q):
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

    def test_interpolation_is_forwarded(self):
        x = torch.tensor([0.0, 1.0, 2.0, 3.0])
        assert float(safe_quantile(x, 0.5, interpolation="higher")) == float(
            torch.quantile(x, 0.5, interpolation="higher")
        )

    def test_interpolation_survives_the_oversized_path(self):
        torch.manual_seed(3)
        x = torch.rand(5000)
        got = safe_quantile(x, 0.9, limit=500, interpolation="higher")
        assert abs(float(got) - float(torch.quantile(x, 0.9))) < 0.05
