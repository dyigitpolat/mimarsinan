"""[U-BRANGE2 (e)] The census of a declared boundary describes the grid it deploys (U-BRANGE verify m-3, route (b)
family check m-1: revision 1's row of a declared boundary carried the calibrated candidate's clamp census, e.g. S1's
boundary 13 printed the share at the rule's scale 8.830 while the deployed grid was 0.904).

A declared row's scale, clamped count, share, largest value over its range end and squared error are those of the
declared grid's own snap (its offset, rounding and codes) on the calibration batches' float seams; the rule's
calibration is kept beside it. An undeclared row is the rule's census, as before. Every number is read directly.
"""

from __future__ import annotations

import pytest
import torch

import mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step as bqs
from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid
from unit.mapping.declared_programs import write_state
from unit.pipelining.test_declared_grid_steps import _boundary_step, _flow

DECLARED = {"scale": 6.0, "offset": 3.0, "rounding": "floor", "codes": [0, 15]}


def _seam(flow):
    """The calibration batch's float seam at the declared entry (``_0``), every grid inert (its own input)."""
    return torch.randn(32, 8, generator=torch.Generator().manual_seed(1)).double()


def _run(monkeypatch, tmp_path):
    report = getattr(bqs, "BOUNDARY_REPORT", None)
    if report is None:
        pytest.fail("Boundary Quantization reports no range census on this head")
    flow = _flow()
    declaration = write_state(tmp_path / "s.json", {"_0": {"entry": dict(DECLARED)}})
    pipeline = _boundary_step(monkeypatch, flow, quantizer_state=declaration)
    rows = {r["name"]: r for r in pipeline.entries[report]["boundaries"]}
    return flow, rows


def test_a_declared_row_is_the_census_of_its_deployed_grid(monkeypatch, tmp_path):
    flow, rows = _run(monkeypatch, tmp_path)
    names = perceptron_fx_names(flow.get_mapper_repr())
    grid = names["_0"].input_activation.grid
    assert isinstance(grid, DeclaredBoundaryGrid)
    (row,) = [r for r in rows.values() if r["declared"]]
    x = _seam(flow)
    low, high = grid.value_range
    out = (x < low) | (x > high)
    assert row["scale"] == grid.scale == row["deployed_scale"]
    assert row["entries"] == x.numel() and row["clamped"] == int(out.sum())
    assert abs(row["clamped_share"] - float((x[out] ** 2).sum()) / float((x * x).sum())) <= 1e-12
    direct = float(((grid.snap(x) - x) ** 2).sum())
    assert abs(row["squared_error"] - direct) <= 1e-9 * max(direct, 1e-12)
    largest = max(float(x.max()) / high, float(x.min()) / low if low < 0 else 0.0)
    assert abs(row["largest_over_scale"] - largest) <= 1e-12 * largest
    assert set(row["calibration"]) >= {"scale", "clamped", "clamped_share", "squared_error"}
    assert row["calibration"]["scale"] != row["scale"]


def test_an_undeclared_row_is_the_rule_s_census(monkeypatch, tmp_path):
    _flow_, rows = _run(monkeypatch, tmp_path)
    plain = [r for r in rows.values() if not r["declared"]]
    assert plain and all("calibration" not in r and r["scale"] == r["deployed_scale"] for r in plain)


def test_the_census_pass_leaves_every_grid_as_installed(monkeypatch, tmp_path):
    flow, _rows = _run(monkeypatch, tmp_path)
    names = perceptron_fx_names(flow.get_mapper_repr())
    assert isinstance(names["_0"].input_activation.grid, DeclaredBoundaryGrid)
    assert names["_0"].input_activation.grid.scale == DECLARED["scale"]
    assert names["_2"].input_activation.grid.armed
