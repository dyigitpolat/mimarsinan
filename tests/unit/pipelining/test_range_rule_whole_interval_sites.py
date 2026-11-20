"""[U-BRANGE2 (c)] The integer value path's sites search the whole interval up to the calibration maximum.

Revision 1 searched from the pooled 0.999 quantile up, so at 4 bits on bulk-dominated seams, where the error minimum
lies below that quantile, it returned the quantile (U-BRANGE verify m-2: 1.47 and 1.48 times the least error). The
rule's candidates now cover 12 octaves below the maximum, no quantile read; a minimum at the search's low end is
refused by name, never given silently. Every error here is read directly from the grid (``range_rule_reference``).
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from unit.pipelining.range_rule_reference import candidates, grid_error, interval, signed_grid
from unit.pipelining.test_boundary_range_rule import _seam_batches


def _scale(batches, bits: int) -> float:
    perceptron = Perceptron(4, 64)
    (quantizer,) = install_boundary_quantizers([perceptron], bits)
    calibrate_boundary_scales(perceptron, [perceptron], [quantizer], batches)
    return float(quantizer.grid.scale)


@pytest.mark.parametrize("distribution", ["gaussian", "laplace"])
def test_at_4_bits_on_a_bulk_seam_the_minimum_below_the_quantile_is_found(distribution):
    batches = _seam_batches(distribution, False)
    magnitudes = [b.abs() for b in batches]
    low, high = interval(magnitudes)
    scale = _scale(batches, 4)
    former = min(grid_error(magnitudes, s, signed_grid(4)) for s in candidates(low, high))
    assert scale < low
    assert grid_error(magnitudes, scale, signed_grid(4)) < 0.9 * former


def test_a_minimum_at_the_search_s_low_end_is_refused_by_name(monkeypatch):
    """A site whose error minimum lies below the search's low end (a search narrowed to two octaves, on a Gaussian seam
    whose maximum is one value at 100 times the bulk) is refused, never given the low end silently."""
    try:
        import mimarsinan.mapping.support.range_search as engine
    except ImportError:
        pytest.fail("no range search engine on this head: no low end to refuse")
    monkeypatch.setattr(engine, "RANGE_OCTAVES", 2)
    x = torch.randn(1 << 20, generator=torch.Generator().manual_seed(1))
    x[0] = 100.0  # its clamp at 25 costs 5,625; the bulk's rounding there 3,400 and at 100 about 54,000
    with pytest.raises(engine.RangeSearchError, match="low end") as caught:
        _scale([x.reshape(16384, 64)], 8)
    # [U-BRANGE2 stage 1b, the verify's m-7] the refusal is a toolchain refusal class that keeps its builtin base
    from mimarsinan.common.refusal import ToolchainRefusal

    assert issubclass(engine.RangeSearchError, ToolchainRefusal) and issubclass(engine.RangeSearchError, RuntimeError)
    assert isinstance(caught.value, ToolchainRefusal)


def test_a_minimum_more_than_four_octaves_below_the_maximum_is_found():
    """The search covers its 12 octaves below the maximum (mutant M12 narrowed it to 4). On a 4-bit Gaussian seam whose
    maximum is one value at 512, the error minimum lies about 7.7 octaves below that value (q11_octaves.py: 2.51,
    error 273,051); the error is not unimodal there, so a search of 4 octaves (32 to 512) returns the maximum (error
    1,048,031) without a refusal. The rule finds the minimum and its census's least span is 2^-12 of the maximum."""
    try:
        import mimarsinan.mapping.support.range_search  # noqa: F401
    except ImportError:
        pytest.fail("no range search engine on this head: no octave below the quantile is read")
    x = torch.randn(1 << 20, generator=torch.Generator().manual_seed(3))
    x[0] = 512.0
    batches = [x.reshape(16384, 64)]
    perceptron = Perceptron(4, 64)
    (quantizer,) = install_boundary_quantizers([perceptron], 4)
    (row,) = calibrate_boundary_scales(perceptron, [perceptron], [quantizer], batches)
    magnitudes = [b.abs() for b in batches]
    assert row.scale < 512.0 / 16, row.scale
    assert row.low_span == 512.0 * 2.0 ** -12
    assert grid_error(magnitudes, row.scale, signed_grid(4)) < 0.5 * grid_error(magnitudes, 512.0, signed_grid(4))
