"""[U-BRANGE2, second form] The range rule's search is exact at every declared width with a bounded transient.

The first form (c3) counted each batch into the buckets between every candidate's cell edges at once: its transient was
candidates times levels (the design review's M-1: 11.9 GB and 73 s on one ViT-block seam at 16 bits). The second form
sorts each call's magnitudes once, reads every candidate's clamp part from one rank and its rounding part from one rank
per cell edge, in blocks of a bounded number of (candidate, cell) pairs, and evaluates a candidate only where it can be
the least. Its choice must be the full search's bit for bit: the full search here is the same module with every
candidate evaluated on every call (its selection replaced), and an independent direct reading of the grid
(``range_rule_reference``) checks the arithmetic.
"""

from __future__ import annotations

import math

import pytest
import torch

from mimarsinan.models.nn.activations.value_quantizer import value_grid_levels
from unit.pipelining.range_rule_reference import grid_error, signed_grid

try:  # each clause reads the rule itself, so a head without it fails clause by clause, never at collection
    import mimarsinan.mapping.support.range_rule as range_rule
except ImportError:  # R5: the boundary rule is the 0.999 quantile, no range rule module
    range_rule = None


def _rule():
    if range_rule is None:
        pytest.fail("no range rule module on this head (R5's boundary scale is the 0.999 quantile)")
    return range_rule


def _engine():
    """The module holding the search (the rule's own, or its search module where the rule has one)."""
    try:
        import mimarsinan.mapping.support.range_search as engine
        return engine
    except ImportError:
        return _rule()


def _first_vector(args) -> torch.Tensor:
    return next(a for a in args if torch.is_tensor(a) and a.dim() == 1)


def _candidates(high: float) -> torch.Tensor:
    make = getattr(_engine(), "whole_interval_candidates", None) or getattr(_rule(), "whole_interval_candidates", None)
    if make is None:
        pytest.fail("the rule searches no whole interval (revision 1: 257 candidates from the 0.999 quantile)")
    return make(high)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _review_site(scale: int = 4):
    """The design review's ViT-block seam (rv_cost.py) at ``scale`` samples per batch: a Gaussian bulk with one
    position (token 0, one channel) at -24 to -21 on every sample."""
    g = _gen(7)
    out = []
    for _ in range(2):
        b = torch.randn(scale, 197, 768, generator=g)
        b[:, 0, 259] = -24.0 + torch.rand(scale, generator=g) * 3.0
        out.append(b)
    return out


def _seams():
    """The adversarial seams: two batches each (the second never holding the first's maximum where it matters)."""
    g = _gen(11)
    n = 1 << 17
    heavy = [torch.distributions.StudentT(1.0).sample((n,)) for _ in range(2)]
    heavy2 = [torch.distributions.StudentT(2.0).sample((n,)) for _ in range(2)]
    ties = [torch.randint(0, 11, (n,), generator=g).float() * 0.5 for _ in range(2)]
    constant = [torch.full((n,), 3.7), torch.full((n,), 3.7)]
    tiny = [torch.randn(n, generator=g) * 1e-5 for _ in range(2)]
    signed = [torch.randn(n, generator=g) * torch.where(torch.rand(n, generator=g) < 0.5, 1.0, 3.0) for _ in range(2)]
    outlier = [torch.randn(n, generator=g), torch.randn(n, generator=g)]
    outlier[0][123] = 20_000.0
    late_outlier = [torch.randn(n, generator=g), torch.randn(n, generator=g)]
    late_outlier[1][77] = -900.0
    lone_maximum = [torch.randn(n, generator=g) * 1e-3, torch.randn(n, generator=g)]
    return {"heavy tail (t, 1)": heavy, "heavy tail (t, 2)": heavy2, "ties": ties, "constant": constant,
            "below 1e-3": tiny, "signed": signed, "one far outlier": outlier, "late outlier": late_outlier,
            "lone maximum": lone_maximum, "review site": _review_site()}


SEAMS = _seams()
FAST = [(name, bits) for name in sorted(SEAMS) for bits in (4, 8, 12)]
WIDE = [(name, 16) for name in sorted(SEAMS)]


def _search(batches, bits, full=False, monkeypatch=None):
    """[stage 1] The full search is the engine's own (``full=True``: every candidate evaluated on every call); a head
    without it fails here by name."""
    rule = _rule()
    probe = torch.nn.Identity()
    site = rule.RangeSite(probe, value_grid_levels(bits))
    if full:
        import inspect

        if "full" not in inspect.signature(rule.least_squares_ranges).parameters:
            pytest.fail("the rule holds no full search of its own")
        (row,) = rule.least_squares_ranges(probe, [site], batches, full=True)
        return row
    (row,) = rule.least_squares_ranges(probe, [site], batches)
    return row


def _calls_module():
    """The module holding the second pass's selection (the per-call arithmetic, or the engine before its split)."""
    try:
        import mimarsinan.mapping.support.range_calls as calls
        return calls
    except ImportError:
        return _engine()


def _equal_to_the_full_search(batches, bits, monkeypatch):
    row = _search(batches, bits)
    full = _search(batches, bits, full=True, monkeypatch=monkeypatch)
    assert row == full, (row, full)  # every field bit for bit: the scale, the census and the three errors
    return row


def _direct_check(batches, bits, row):
    """The chosen error is the grid's own (direct reading) and no candidate near it is lower by more than the
    arithmetic's resolution: float64 prefix sums resolve an error to about 1e-12 of the seam's squared magnitude (a
    constant seam at its own scale reads 1e-5 where the grid departs by 0, on 2^18 values at 3.7)."""
    magnitudes = [b.abs() for b in batches]
    snap = signed_grid(bits)
    energy = sum(float((m.double() ** 2).sum()) for m in magnitudes)
    resolution = 1e-10 * energy
    chosen = grid_error(magnitudes, row.scale, snap)
    assert abs(row.squared_error - chosen) <= 1e-6 * chosen + resolution
    scales = _candidates(row.maximum)
    at = int(torch.nonzero(scales == row.scale)[0])
    near = list(range(max(1, at - 24), min(len(scales), at + 25))) + list(range(1, len(scales), 128))
    for i in sorted(set(near)):
        assert chosen <= grid_error(magnitudes, float(scales[i]), snap) * (1 + 1e-6) + resolution


@pytest.mark.parametrize("name, bits", FAST)
def test_the_choice_is_the_full_search_s_bit_for_bit(name, bits, monkeypatch):
    batches = SEAMS[name]
    row = _equal_to_the_full_search(batches, bits, monkeypatch)
    _direct_check(batches, bits, row)


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("name, bits", WIDE)
def test_at_16_bits_the_choice_is_the_full_search_s_bit_for_bit(name, bits, monkeypatch):
    batches = SEAMS[name]
    row = _equal_to_the_full_search(batches, bits, monkeypatch)
    _direct_check(batches, bits, row)


@pytest.mark.slow
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("bits", [8, 12, 16])
def test_on_the_review_s_full_size_site_the_choice_is_the_full_search_s(bits, monkeypatch):
    _equal_to_the_full_search(_review_site(32), bits, monkeypatch)


def _recorded_sizes(monkeypatch, bound=None):
    """Record every rank query's size (searchsorted's queries, bucketize's edges); above ``bound`` fail at once."""
    sizes = []
    real_search, real_bucket = torch.searchsorted, torch.bucketize

    def record(size):
        sizes.append(size)
        if bound is not None and size > bound:
            raise AssertionError(f"a rank query of {size} (candidate, cell) edges against the bound {bound}")

    def search(seq, values, *a, **k):
        record(values.numel())
        return real_search(seq, values, *a, **k)

    def bucket(values, boundaries, *a, **k):
        record(boundaries.numel())
        return real_bucket(values, boundaries, *a, **k)

    monkeypatch.setattr(torch, "searchsorted", search)
    monkeypatch.setattr(torch, "bucketize", bucket)
    return sizes


@pytest.mark.parametrize("bits", [8, 12, 16])
def test_the_transient_is_bounded_independent_of_candidates_times_levels(bits, monkeypatch):
    """No call materializes more (candidate, cell) edges than one block's bound or one edge per candidate, at any
    width: the bound does not grow with candidates times levels."""
    rule = _rule()
    bound = max(getattr(_engine(), "_BLOCK_CELLS", 0), getattr(rule, "RANGE_CANDIDATES", 0))
    sizes = _recorded_sizes(monkeypatch, bound)
    _search(SEAMS["review site"], bits)
    assert sizes and max(sizes) <= bound, (max(sizes), bound)


def test_at_16_bits_only_the_candidates_that_can_be_least_are_evaluated(monkeypatch):
    """The review-site seam at 16 bits: the rounding parts are read for a small share of the full search's
    (candidate, cell) pairs (the clamp part prunes every scale far below the maximum)."""
    full = 2 * getattr(_rule(), "RANGE_CANDIDATES", 3073) * value_grid_levels(16)
    sizes = _recorded_sizes(monkeypatch, full // 20)
    _search(SEAMS["review site"], 16)
    assert sum(sizes) <= full / 20, (sum(sizes), full)


def test_a_candidate_s_error_does_not_depend_on_the_candidates_evaluated_beside_it():
    sorted_type = getattr(_engine(), "_Sorted", None)
    if sorted_type is None:
        pytest.fail("the rule has no per-call sorted form whose candidates are evaluated block by block")
    batch = sorted_type(SEAMS["review site"][0].abs())
    scales = _candidates(float(SEAMS["review site"][0].abs().max()))[-40:]
    levels = value_grid_levels(12)
    if hasattr(batch, "top_parts"):  # the window form: low end 0, no value below it
        kept = batch.top_parts(0.0, scales)[0]
        parts = lambda s, k: batch.rounding_parts(0.0, 0, s, levels, k)  # noqa: E731
    else:
        kept = batch.clamp_parts(scales)[0]
        parts = lambda s, k: batch.rounding_parts(s, levels, k)  # noqa: E731
    together = parts(scales, kept)
    alone = torch.stack([parts(scales[i:i + 1], kept[i:i + 1])[0] for i in range(len(scales))])
    assert torch.equal(together, alone)


def test_a_candidate_the_selection_skipped_is_read_in_a_third_pass_when_it_could_be_least(monkeypatch):
    """The second pass's selection is a selection, never the rule: here it evaluates only the search's two ends on
    every call, and the final check must evaluate whatever could still be the least before choosing."""
    full = _search(SEAMS["signed"], 8, full=True, monkeypatch=monkeypatch)
    consulted = []

    def ends_only(*args, **_k):
        consulted.append(1)
        chosen = torch.zeros(len(_first_vector(args)), dtype=torch.bool)
        chosen[0] = chosen[-1] = True
        return chosen

    rule = _rule()
    monkeypatch.setattr(_calls_module(), "_selection", ends_only, raising=False)
    forwards = []

    class Counted(torch.nn.Identity):
        def forward(self, x):
            forwards.append(1)
            return x

    probe = Counted()
    (row,) = rule.least_squares_ranges(probe, [rule.RangeSite(probe, value_grid_levels(8))], SEAMS["signed"])
    assert consulted, "the rule's second pass has no selection to consult"
    assert len(forwards) == 3 * len(SEAMS["signed"]), f"{len(forwards)} forwards: the third pass did not run"
    assert row == full


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_a_float64_seam_takes_the_float32_seam_s_choice(bits):
    batches = SEAMS["signed"]
    row32 = _search(batches, bits)
    row64 = _search([b.double() for b in batches], bits)
    assert math.isclose(row64.scale, row32.scale, rel_tol=1e-6)
    assert row64.clamped == row32.clamped


def test_the_rule_never_writes_a_tensor_it_reads():
    for dtype in (torch.float32, torch.float64):
        batches = [b.to(dtype) for b in SEAMS["signed"]]
        before = [b.clone() for b in batches]
        _search(batches, 8)
        assert all(torch.equal(a, b) for a, b in zip(before, batches))


def test_no_quantile_is_read(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the rule read a quantile")

    monkeypatch.setattr(_rule(), "safe_quantile", refuse, raising=False)
    assert _search(SEAMS["review site"], 8).scale > 0.0


def test_a_seam_never_reached_keeps_the_scale_0():
    row = _search([torch.zeros(1000), torch.zeros(1000)], 8)
    assert row.scale == 0.0 and row.candidates == 0
