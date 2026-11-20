"""[U-BRANGE2 (a)] The range rule reads a float64 seam as it reads a float32 one (U-BRANGE verify M-1).

Revision 1's accumulator squared its sorted magnitudes in place when they arrived as float64 (``Tensor.double()``
returns the same tensor), so the cell bounds, the clamp counts and the largest value were then read from squared
values: a float64 boundary got a wrong scale and census without a refusal. The rule never writes a tensor it reads:
a float64 seam takes the float32 seam's scale (within the candidates' fp32 precision) and its census is the direct
count of the seam's own values.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.nn.activations.value_quantizer import value_grid_levels
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from unit.pipelining.test_boundary_range_rule import _DISTRIBUTIONS, _seam_batches

try:
    import mimarsinan.mapping.support.range_rule as range_rule
except ImportError:
    range_rule = None

#: fp32 candidates: one fp32 ulp of the interval's ends, at most.
_FP32 = 2e-6
_CASES = [(d, t, b) for d in sorted(_DISTRIBUTIONS) for t in (False, True) for b in (4, 8)]


def _rule():
    if range_rule is None:
        pytest.fail("no range rule module on this head (R5's boundary scale is the 0.999 quantile)")
    return range_rule


def _census(batches, bits: int):
    _rule()
    perceptron = Perceptron(4, 64).to(batches[0].dtype)
    (quantizer,) = install_boundary_quantizers([perceptron], bits)
    quantizer.to(batches[0].dtype)
    (row,) = calibrate_boundary_scales(perceptron, [perceptron], [quantizer], batches)
    return row


@pytest.mark.parametrize("distribution, tail, bits", _CASES)
def test_a_float64_seam_takes_the_float32_seam_s_scale(distribution, tail, bits):
    single = _seam_batches(distribution, tail)
    double = [b.double() for b in single]
    a, b = _census(single, bits), _census(double, bits)
    assert abs(a.scale - b.scale) <= _FP32 * a.scale, (a.scale, b.scale)
    assert a.clamped == b.clamped and a.entries == b.entries


@pytest.mark.parametrize("distribution, tail, bits", _CASES)
def test_a_float64_census_is_the_direct_count_of_the_seam_s_own_values(distribution, tail, bits):
    batches = [b.double() for b in _seam_batches(distribution, tail)]
    row = _census(batches, bits)
    magnitudes = torch.cat([b.abs().reshape(-1) for b in batches])
    assert row.entries == magnitudes.numel()
    assert row.clamped == int((magnitudes > row.scale).sum())
    largest = float(magnitudes.max())
    assert abs(row.largest_over_scale - largest / row.scale) <= 1e-9 * (largest / row.scale)
    energy = float((magnitudes * magnitudes).sum())
    clamp = float((magnitudes[magnitudes > row.scale] ** 2).sum())
    assert abs(row.clamped_share - clamp / energy) <= 1e-9


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_search_never_writes_a_tensor_it_reads(dtype):
    """The site's magnitudes (here a held tensor, sorted and contiguous: the case where a copy is easiest to skip)
    are the same values after the search."""
    held = torch.sort(_seam_batches("gaussian", True)[0].abs().reshape(-1)).values.to(dtype)
    kept = held.clone()
    probe = torch.nn.Identity()
    rule = _rule()
    site = rule.RangeSite(probe, value_grid_levels(8), lambda _x: held)
    rule.least_squares_ranges(probe, [site], [torch.zeros(1)])
    assert torch.equal(held, kept)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a_site_narrower_than_float32_is_refused_by_name(dtype):
    """[U-BRANGE2 stage 1b, the verify's m-8] The search reads each cell edge in its values' own dtype, so the stated
    resolution (``ERROR_RESOLUTION`` of the site's squared magnitude) holds for float32 and float64 values; on a float16
    seam the census departed from the grid's own error by 1.47e-7 and the choice sat 0.32% above the float16 snap's
    least. A site whose values are narrower than float32 is refused by name (a toolchain refusal naming the dtype),
    never given a range at another resolution; its float32 values take the rule as before."""
    rule = _rule()
    from mimarsinan.common.refusal import ToolchainRefusal

    module = torch.nn.Identity()
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(8)) * 3.0
    with pytest.raises(ToolchainRefusal, match=str(dtype).replace("torch.", "")) as caught:
        rule.least_squares_ranges(module, [rule.RangeSite(module, 127)], [x.to(dtype)])
    assert type(caught.value).__name__ == "RangeSearchError" and isinstance(caught.value, RuntimeError)
    (row,) = rule.least_squares_ranges(module, [rule.RangeSite(module, 127)], [x.to(dtype).float()])
    assert row.scale > 0.0


def test_a_site_of_integer_values_takes_the_rule_as_its_float32_values_do():
    """[U-BRANGE2 stage 1b, sweep SW72] The dtype scope reads floating sites only: a site whose values are integers (the
    search sorts them as float32) is no narrow float, takes the rule, and chooses as its float32 values do."""
    rule = _rule()
    module = torch.nn.Identity()
    x = torch.randint(-40, 41, (64, 32), generator=torch.Generator().manual_seed(9))
    (as_int,) = rule.least_squares_ranges(module, [rule.RangeSite(module, 7)], [x])
    (as_float,) = rule.least_squares_ranges(module, [rule.RangeSite(module, 7)], [x.float()])
    assert as_int.scale == as_float.scale > 0.0 and as_int.squared_error == as_float.squared_error


# --- [U-BRANGE2 stage 1c] the dtype scope reads every call of every site (the delta verify's M-1) --------------------


@pytest.mark.parametrize("order", ["narrow_last", "narrow_first", "narrow_between"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a_site_with_one_call_narrower_than_float32_is_refused_by_name(order, dtype):
    """[U-BRANGE2 stage 1c, the delta verify's M-1] The search's first pass reads the dtype of every call, not of a
    site's first call alone: one narrow call among float32 calls, wherever it falls, refuses the site by name. Red on
    X05 (only a site's first call checked) where the narrow call is not the first."""
    rule = _rule()
    from mimarsinan.common.refusal import ToolchainRefusal

    module = torch.nn.Identity()
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(11)) * 3.0
    calls = {"narrow_last": [x, x.to(dtype)], "narrow_first": [x.to(dtype), x],
             "narrow_between": [x, x.to(dtype), x]}[order]
    with pytest.raises(ToolchainRefusal, match=str(dtype).replace("torch.", "")):
        rule.least_squares_ranges(module, [rule.RangeSite(module, 127)], calls)


def test_a_narrow_call_of_one_member_refuses_its_pooled_union():
    """[U-BRANGE2 stage 1c, the delta verify's M-1] The pooled source (``least_squares_pooled``) is read by the same
    first pass: a union whose second member's values are narrower than float32 is refused by name. Red on X05."""
    rule = _rule()
    from mimarsinan.common.refusal import ToolchainRefusal

    a, b = torch.nn.Identity(), torch.nn.Identity()

    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = a, b

        def forward(self, x):
            return self.a(x) + self.b(x.to(torch.float16)).float()

    sites = [rule.RangeSite(a, 127), rule.RangeSite(b, 127)]
    x = torch.randn(16, 8, generator=torch.Generator().manual_seed(12))
    with pytest.raises(ToolchainRefusal, match="float16"):
        rule.least_squares_pooled(Two(), sites, [[0, 1]], [x])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_an_empty_narrow_call_is_no_call(dtype):
    """[U-BRANGE2 stage 1c, sweep C-T1] The dtype scope reads the values a call carries: an empty call is no call (the
    review's m-3), so an empty narrow call among float32 calls neither refuses the site nor moves its range. Red where
    the dtype is read before the call's size."""
    rule = _rule()
    module = torch.nn.Identity()
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(13)) * 3.0
    (alone,) = rule.least_squares_ranges(module, [rule.RangeSite(module, 127)], [x])
    (with_empty,) = rule.least_squares_ranges(module, [rule.RangeSite(module, 127)], [x, x.to(dtype)[:0]])
    assert with_empty == alone


def test_the_refusal_names_the_narrow_site_among_float32_sites():
    """[U-BRANGE2 stage 1c, sweeps C-T2 and C-T3] Each site's calls are read for their own dtype: in a source of two
    sites whose second alone reads narrow values, the source is refused and the refusal names that site (by its index
    in the source, as the low-end refusal names its site), never the float32 one."""
    rule = _rule()
    from mimarsinan.common.refusal import ToolchainRefusal

    a, b = torch.nn.Identity(), torch.nn.Identity()

    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = a, b

        def forward(self, x):
            return self.a(x), self.b(x.to(torch.bfloat16))

    x = torch.randn(16, 8, generator=torch.Generator().manual_seed(14))
    with pytest.raises(ToolchainRefusal, match="bfloat16") as caught:
        rule.least_squares_ranges(Two(), [rule.RangeSite(a, 127), rule.RangeSite(b, 127)], [x])
    assert "range site(s) [1]" in str(caught.value), str(caught.value)
