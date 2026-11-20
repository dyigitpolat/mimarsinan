"""[U-BRANGE2 stage 1, the review's m-5] The drive site's row count has a declared budget, and a perceptron that needs
more rows than the budget is refused by name.

The second form's drive rule took the least ``k`` rows whose top code value holds ``max|b| / max|w|`` with no bound:
a perceptron of tiny weights asked for 4,368 always-on rows at 16 bits (design2_review.md, q3_drive.py), more rows than
any core has axons, and nothing refused it. It also took ``k`` rows where the document deploys one: no bias-row
splitting declared (the default ``off``), the projection keeps one row on the shared grid, and a drive chosen for ``k``
rows below the top code widens that grid further than the top code does (the drive unit's own fixture: 16.06 against
the top 20.0, every weight of the perceptron coarser).

The budget is read where the deployment declares it:
* the rows the projection deploys: one row without bias-row splitting (the declared one-row form keeps the top code
  where no code holds the ratio), the computed rows under ``auto``, at least the declared rows under ``fixed``;
* the cores the perceptron maps to: a core holds its always-on rows and at least one data axon, so at most the largest
  declared core's ``max_axons - 1`` rows.
A perceptron whose rows exceed the cores' budget is refused by name at Weight Quantization's entry.
"""

from __future__ import annotations

import pytest
import torch

try:
    import mimarsinan.mapping.support.drive_rule as drive_rule
except ImportError:  # R5 and revision 1: the drive is the top code value, no drive rule module
    drive_rule = None

from unit.transformations.test_bias_row_drive import W_MAX, X_FS, _flow, _stamp, _value_config, _wq


def _budget_type():
    if drive_rule is None:
        pytest.fail("no drive rule module on this head (the drive is its reader grid's top code value)")
    budget = getattr(drive_rule, "RowBudget", None)
    if budget is None:
        pytest.fail("the drive rule reads no row budget: its rows are unbounded and blind to the deployed row count")
    return budget


def _refusal_type():
    _budget_type()
    refusal = getattr(drive_rule, "DriveRowBudgetRefusal", None)
    if refusal is None:
        pytest.fail("the drive rule has no refusal for rows above the budget")
    return refusal


def _grid(bits: int, top: float = 1.0):
    levels = (1 << (bits - 1)) - 1
    return drive_rule.DriveGrid(step=top / levels, offset=0.0, code_max=levels)


def _perceptron(outputs: int, fan_in: int, seed: int, ratio: float):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(outputs, fan_in, generator=g) * 0.05
    b = torch.randn(outputs, generator=g)
    b = b / b.abs().max() * float(w.abs().max()) * ratio
    return w, b


@pytest.mark.parametrize("ratio", [0.4, 3.49, 7.25])
def test_one_declared_row_takes_no_row_beyond_it(ratio):
    """One row deployed: the drive is a code at or above the ratio where one holds it (the rows rule's at one row),
    else the top code value (the one-row form's least widened weight grid); never a code chosen for more rows."""
    budget = _budget_type()
    grid, q_max = _grid(8), 127
    w, b = _perceptron(256, 32, 5, ratio)
    drive, rows = drive_rule.least_error_drive_and_rows(w, b, grid, q_max, fractional=True, budget=budget(split=False))
    assert rows == 1
    if ratio > grid.top:
        assert drive == grid.top
    else:
        split, k = drive_rule.least_error_drive_and_rows(w, b, grid, q_max, fractional=True,
                                                         budget=budget(split=True, axons=256))
        assert (drive, rows) == (split, k)


def test_rows_above_the_cores_budget_are_refused_by_name():
    """The review's case: weights at 1e-5 times normal, a 16-bit reader grid, 4,096 outputs: thousands of rows."""
    budget, refusal = _budget_type(), _refusal_type()
    g = torch.Generator().manual_seed(3)
    w = torch.randn(4096, 256, generator=g) * 1e-5
    b = torch.randn(4096, generator=g) * 0.02
    grid = _grid(16, 1.0)  # max|b| / max|w| about 1,750 over the top code value 1: about 1,750 rows
    with pytest.raises(refusal) as caught:
        drive_rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True, budget=budget(split=True, axons=256))
    text = str(caught.value)
    assert "rows" in text and "255" in text, text
    from mimarsinan.common.refusal import ToolchainRefusal
    assert isinstance(caught.value, ToolchainRefusal)


def test_rows_within_the_budget_are_taken():
    budget = _budget_type()
    w, b = _perceptron(64, 16, 4, 3.49)
    grid = _grid(4)
    drive, rows = drive_rule.least_error_drive_and_rows(w, b, grid, 7, fractional=True,
                                                        budget=budget(split=True, axons=8))
    assert rows == drive_rule.bias_rows(float(b.double().abs().max()) / float(w.abs().max()), grid.top) <= 7
    assert rows * drive >= float(b.double().abs().max()) / float(w.abs().max())


def test_a_declared_row_count_is_a_floor_of_the_rows():
    """``fixed`` rows: the projection codes at least the declared rows, so the rule reads that many."""
    budget = _budget_type()
    w, b = _perceptron(128, 16, 6, 0.4)
    grid = _grid(8)
    drive, rows = drive_rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True,
                                                        budget=budget(split=True, floor=3, axons=256))
    assert rows == 3 and 3 * drive >= float(b.double().abs().max()) / float(w.abs().max())


def test_a_floor_above_the_budget_is_refused_by_name():
    budget, refusal = _budget_type(), _refusal_type()
    w, b = _perceptron(128, 16, 6, 0.4)
    with pytest.raises(refusal):
        drive_rule.least_error_drive_and_rows(w, b, _grid(8), 127, fractional=True,
                                              budget=budget(split=True, floor=40, axons=32))


# --- the step's own entry: the drive unit's fixture (layer 0 reads an entry grid of full scale 20, max|w| 0.05) ----


def test_without_bias_row_splitting_a_bias_beyond_the_full_scale_keeps_the_top_code(monkeypatch):
    """max|b| / max|w| = 30 above the entry grid's top 20, no splitting declared: the projection deploys one row, so the
    drive is the top code value (16.06 under the unbounded rows rule widened the grid to 30 / 16.06 of max|w|)."""
    flow = _flow(b_max=30.0 * W_MAX)
    _wq(monkeypatch, flow, _value_config())
    perceptron = flow.get_perceptrons()[0]
    assert _stamp(perceptron) == X_FS
    assert float(perceptron.parameter_scale) == pytest.approx(127 / (30.0 * W_MAX / X_FS), rel=1e-6)


def test_under_auto_splitting_the_projection_deploys_the_rule_s_rows(monkeypatch):
    from mimarsinan.mapping.support.bias_rows import bias_rows_from_scales
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives

    _budget_type()
    config = _value_config(bias_row_splitting="auto")
    probe = _flow(b_max=30.0 * W_MAX)
    stamps = stamp_projection_drives(probe, config)
    rows = getattr(stamps, "rows", None)
    if rows is None:
        pytest.fail("the stamps count no rows: the row cost is not read")
    name = str(probe.get_perceptrons()[0].name)
    assert rows.get(name) == 2  # ratio 30 over the top 20: two rows hold it
    flow = _flow(b_max=30.0 * W_MAX)
    _wq(monkeypatch, flow, config)
    perceptron = flow.get_perceptrons()[0]
    assert bias_rows_from_scales(perceptron.bias_scale, perceptron.parameter_scale) == 2
    assert 2 * _stamp(perceptron) >= 30.0 * (1 - 1e-6)
    assert float(perceptron.parameter_scale) == pytest.approx(127 / W_MAX, rel=1e-6)  # the weights' own grid


def test_a_perceptron_needing_more_rows_than_its_cores_hold_is_refused_by_name(monkeypatch):
    """max|b| / max|w| = 6,000 over the top 20: 300 rows, above the default cores' 256 axons less one data axon."""
    from mimarsinan.common.refusal import ToolchainRefusal

    _budget_type()
    flow = _flow(b_max=300.0)
    with pytest.raises(ToolchainRefusal) as caught:
        _wq(monkeypatch, flow, _value_config(bias_row_splitting="auto"))
    text = str(caught.value)
    assert str(flow.get_perceptrons()[0].name) in text and "300" in text and "255" in text, text


# --- [U-BRANGE2 stage 1b] the site's reading of the declared rows and the budget's two edges ---------------------------
# The stage's verify (U2/stage1/verify.md): M-1 (VM07, the fixed mode's declared rows not read into the budget), m-1
# (VM09, refused at the budget itself) and m-2 (VM10, the budget reading the axons of cores with a bias register) each
# survived every clause above. The first two clauses are the verify's own (probe/test_v_fixed_rows.py), adopted.


def _fixed_config(rows: int = 3):
    return _value_config(bias_row_splitting="fixed", bias_rows_per_perceptron=rows)


def test_the_fixed_rows_are_the_budget_s_floor():
    """Under ``fixed`` the drive site reads the declared rows as its floor, the rows the projection deploys (Weight
    Quantization passes ``rows_override`` as the tuner's ``bias_rows_floor``). Red on VM07."""
    _budget_type()
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import row_budget

    budget = row_budget(_fixed_config())
    assert budget.split and budget.floor == 3, budget


def test_the_drive_site_reads_the_declared_rows_and_the_projection_deploys_them(monkeypatch):
    """Layer 0's bias needs one row (5 max|w| under the top 20); ``fixed`` declares 3: the stamps count 3 rows, the
    drive is the rule's at a floor of 3 rows, and the projection deploys 3 rows. Red on VM07 (one row read)."""
    _budget_type()
    from mimarsinan.mapping.support.bias_rows import bias_rows_from_scales
    from mimarsinan.mapping.support.reader_grids import _perceptron_nodes, reader_input_grids
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    probe = _flow()
    stamps = stamp_projection_drives(probe, _fixed_config())
    p0 = probe.get_perceptrons()[0]
    assert stamps.rows.get(str(p0.name)) == 3, stamps.rows
    transformer = PerceptronTransformer()
    w, b = transformer.get_effective_weight(p0), torch.as_tensor(transformer.get_effective_bias(p0))
    repr_ = _flow().get_mapper_repr()
    grid = reader_input_grids(repr_, absorbed=frozenset())[id(_perceptron_nodes(repr_)[0])]
    assert abs(grid.top - X_FS) < 1e-9, grid
    want, rows = drive_rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True,
                                                       budget=drive_rule.RowBudget(split=True, floor=3))
    assert rows == 3 and _stamp(p0) == want, (_stamp(p0), want)
    flow = _flow()
    _wq(monkeypatch, flow, _fixed_config())
    p = flow.get_perceptrons()[0]
    assert bias_rows_from_scales(p.bias_scale, p.parameter_scale) == 3


def test_the_budget_s_edge_takes_axons_less_one_rows_and_refuses_as_many_rows_as_axons():
    """m-1 (VM09): a bias needing 5 rows (max|b| / max|w| = 4.5 over the top 1) is taken where the largest core holds
    6 axons (5 rows and one data axon) and refused by name where it holds 5. Red on VM09 at 6 axons."""
    budget, refusal = _budget_type(), _refusal_type()
    w, b = _perceptron(64, 16, 9, 4.5)
    grid = _grid(8)
    ratio = float(b.double().abs().max()) / float(w.abs().max())
    assert drive_rule.bias_rows(ratio, grid.top) == 5, ratio
    drive, rows = drive_rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True,
                                                        budget=budget(split=True, axons=6))
    assert rows == 5 and 5 * drive >= ratio
    with pytest.raises(refusal):
        drive_rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True, budget=budget(split=True, axons=5))


def test_the_budget_reads_the_largest_core_that_carries_the_rows():
    """m-2 (VM10): the rows ride the cores without a bias register, so the budget is the largest such core's axons less
    one; a core with a bias register (more axons here) carries no row. Two row cores (40 and 72 axons) and one register
    core (512): the limit is 71. Red on VM10 (511) and on a budget read from the smallest row core (39)."""
    _budget_type()
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import row_budget

    cores = [{"max_axons": 512, "max_neurons": 64, "count": 1, "has_bias": True},
             {"max_axons": 40, "max_neurons": 64, "count": 1, "has_bias": False},
             {"max_axons": 72, "max_neurons": 64, "count": 1, "has_bias": False}]
    budget = row_budget(_value_config(bias_row_splitting="auto", cores=cores))
    assert budget.split and budget.axons == 72 and budget.limit == 71, budget


# --- [U-BRANGE2 stage 1b] the sweep's survivors at the rows' edges and the projection's row count ------------------


def test_a_ratio_equal_to_the_top_code_value_needs_one_row():
    """Sweep SW42: ``k`` rows at the top code value hold ``max|b| / max|w|`` when ``k top >= ratio``: a ratio equal to
    the top is held by one row (two rows would halve the bias grid's reach for nothing), the next float above it by
    two."""
    _budget_type()
    for top in (1.0, 20.0, 0.375):
        assert drive_rule.bias_rows(top, top) == 1
        assert drive_rule.bias_rows(float(torch.nextafter(torch.tensor(top, dtype=torch.float64),
                                                           torch.tensor(float("inf"), dtype=torch.float64))), top) == 2


def test_a_perceptron_without_weights_keeps_the_top_code_on_one_row():
    """Sweep SW48: a perceptron whose weights are all zero (a pruned layer) and whose bias is not: no ratio exists, so
    the row carries the top code value on one row, under either budget (no rows rule, no candidate code is read)."""
    budget = _budget_type()
    w = torch.zeros(8, 4)
    b = torch.tensor([0.3, -0.2, 0.0, 0.1, 0.05, -0.4, 0.2, 0.0])
    grid = _grid(8)
    for each in (budget(split=False), budget(split=True, axons=16)):
        assert drive_rule.least_error_drive_and_rows(w, b, grid, 127, fractional=True, budget=each) == (grid.top, 1)


def test_the_projection_reads_the_declared_row_count(monkeypatch):
    """Sweep SW50: Weight Quantization's projection reads the declared ``fixed`` row count as its floor (the tuner's
    ``bias_rows_floor``), the count the drive site reads; under ``auto`` no floor is declared."""
    _budget_type()
    captured = _wq(monkeypatch, _flow(), _fixed_config(3))
    assert captured["bias_rows_floor"] == 3, captured
    captured = _wq(monkeypatch, _flow(), _value_config(bias_row_splitting="auto"))
    assert captured["bias_rows_floor"] is None, captured


def test_the_projection_reads_the_declared_scale_rule(monkeypatch):
    """Sweep SW51: Weight Quantization's projection reads the declared scale rule (the tuner's ``fractional_scale``), the
    rule the drive site reads: fractional where declared, the integer floor under ``weight_scale_rule: integer``."""
    _budget_type()
    assert _wq(monkeypatch, _flow(), _value_config())["fractional_scale"] is True
    assert _wq(monkeypatch, _flow(), _value_config(weight_scale_rule="integer"))["fractional_scale"] is False
