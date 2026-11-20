"""[U-BRANGE2 R3] The always-on row's drive is the code of its reader's input grid whose row grid departs least from
the bias, among the codes at or above ``max|b| / max|w|`` (the top code value where none is).

At drive ``d`` the shared grid sizes ``max(max|w|, max|b| / d)``, the row's code is ``round(b s / d)`` and the bias lies
on a grid of step ``d / s``: a drive below ``max|b| / max|w|`` widens the weight grid, a drive above it coarsens the
bias grid with no gain. Tying the drive to the reader's full scale (the top code) made every reader's bias grid follow
the range of the grid it reads. The fixture is the drive unit's own (``test_bias_row_drive._flow``: layer 0 reads an
entry grid of full scale 20 with max|w| 0.05 and max|b| 0.25, so the ratio is 5; layer 2 reads layer 0's readout grid
of range 8 with max|w| 0.3 and max|b| 0.2, ratio 2/3). Every error is read here from the projected model itself.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of
from unit.mapping.test_bias_row_drive_program import _exact, _inputs, _program
from unit.transformations.test_bias_row_drive import PT, Q_MAX, X_FS, _flow, _stamp, _value_config, _wq


def _row_error(bias: torch.Tensor, w_max: float, drive: float) -> float:
    """The squared error of ``bias`` on one row at ``drive`` under the fractional rule (the fixture's)."""
    b = bias.double()
    peak = max(w_max, float(b.abs().max()) / drive)
    s = Q_MAX / peak
    rows = torch.round(b * (s / drive)).clamp(-(Q_MAX + 1), Q_MAX)
    return float(((b - rows * (drive / s)) ** 2).sum())


def _projected(monkeypatch, **flow_kw):
    flow = _flow(**flow_kw)
    perceptrons = list(flow.get_perceptrons())
    before = [(PT.get_effective_weight(p).detach().clone(), PT.get_effective_bias(p).detach().clone())
              for p in perceptrons]
    _wq(monkeypatch, flow, _value_config())
    return flow, perceptrons, before


def test_the_entry_reader_s_drive_is_a_code_below_its_full_scale_and_at_or_above_the_ratio(monkeypatch):
    _flow_, perceptrons, before = _projected(monkeypatch)
    grid = boundary_grid_of(perceptrons[0].input_activation)
    w, b = before[0]
    ratio = float(b.abs().max()) / float(w.abs().max())
    drive = _stamp(perceptrons[0])
    assert ratio <= drive < X_FS
    code = round(drive / grid.step)
    assert code * grid.step == drive, "a code value of the reader's entry grid"


def test_the_drive_is_the_least_row_error_among_the_codes_at_or_above_the_ratio(monkeypatch):
    _flow_, perceptrons, before = _projected(monkeypatch)
    grid = boundary_grid_of(perceptrons[0].input_activation)
    w, b = before[0]
    w_max, ratio = float(w.abs().max()), float(b.abs().max()) / float(w.abs().max())
    codes = [k for k in range(1, grid.levels + 1) if k * grid.step >= ratio]
    least = min(_row_error(b, w_max, k * grid.step) for k in codes)
    assert _row_error(b, w_max, _stamp(perceptrons[0])) <= least * (1 + 1e-9)


def test_the_weight_grid_is_the_weights_own_and_the_bias_finer_than_at_the_full_scale(monkeypatch):
    _flow_, perceptrons, before = _projected(monkeypatch)
    for perceptron, (w, b) in zip(perceptrons, before):
        drive = _stamp(perceptron)
        assert float(perceptron.parameter_scale) == pytest.approx(Q_MAX / float(w.abs().max()), rel=1e-6)
        projected = PT.get_effective_bias(perceptron).detach().double()
        error = float(((projected - b.double()) ** 2).sum())
        assert error <= _row_error(b, float(w.abs().max()), drive) * (1 + 1e-3) + 1e-12  # the fp32 projection
    grid = boundary_grid_of(perceptrons[0].input_activation)
    w, b = before[0]
    assert _row_error(b, float(w.abs().max()), _stamp(perceptrons[0])) < _row_error(b, float(w.abs().max()),
                                                                                     grid.value_range[1])


def test_a_bias_beyond_the_reader_s_full_scale_keeps_the_top_code(monkeypatch):
    """max|b| / max|w| = 30 above the entry grid's full scale 20: no code holds the bias on the weights' own grid, so
    the row keeps the top code value (the weight grid then widens, as it did)."""
    _flow_, perceptrons, _before = _projected(monkeypatch, b_max=30.0 * 0.05)
    assert _stamp(perceptrons[0]) == float(boundary_grid_of(perceptrons[0].input_activation).value_range[1])


@pytest.mark.parametrize("extra", [pytest.param({}, id="one-row"),
                                   pytest.param({"bias_row_splitting": "auto"}, id="rows-auto")])
def test_the_program_is_exact_at_drives_below_the_full_scale(monkeypatch, extra):
    """NF, the identity program, the packed program and the int64 golden agree at atol 0, with both entry-fed layers
    driven below their entry grids' top codes (the law solver accepts any code of the entry grid)."""
    flow, ir = _program(monkeypatch, **extra)
    for perceptron in flow.get_perceptrons():
        assert _stamp(perceptron) < float(boundary_grid_of(perceptron.input_activation).value_range[1])
    assert any(isinstance(n, NeuralCore) and _stamp(n) not in (1.0,) for n in ir.nodes)
    _exact(flow, ir, _inputs())


def test_core_fed_readers_are_exact_at_codes_below_their_assigned_grid_s_top(monkeypatch):
    """The drive unit's joined fixture (``c`` reads ``cat(a, b)``, ``d`` reads ``a``; the solver joins ``a`` and ``b``):
    each core-fed reader's drive is a code of the grid its cores are assigned, below that grid's top, and the program
    is exact at atol 0 (the law solver accepts any code of its assigned grid)."""
    from mimarsinan.mapping.export.law_grids import assign_core_grids
    from unit.mapping.test_bias_row_drive_program import _joined_flow, _perceptrons_by_index
    from unit.transformations.test_bias_row_drive import BITS

    flow, ir = _program(monkeypatch, flow=_joined_flow())
    names = {str(p.name).split(".")[-1].strip("_"): p for p in flow.get_perceptrons()}
    ranges = {int(i): float(p.scaler.range) for i, p in _perceptrons_by_index(flow).items()}
    grids, _ = assign_core_grids(ir, ranges, BITS)
    index = {id(p): i for i, p in _perceptrons_by_index(flow).items()}
    below = 0
    for name in ("c", "d"):
        drive = _stamp(names[name])
        for core in (n for n in ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == index[id(names[name])]):
            grid = grids[int(core.id)]
            code = round(drive / grid.input_step)
            assert code * grid.input_step == drive and 1 <= code <= grid.input_levels
            below += int(code < grid.input_levels)
    assert below > 0
    _exact(flow, ir, _inputs())


def test_a_drive_that_is_no_code_of_its_assigned_grid_is_refused_by_name(monkeypatch):
    from mimarsinan.common.refusal import ToolchainRefusal

    def tamper(flow):
        perceptron = flow.get_perceptrons()[1]
        perceptron.__dict__["bias_row_drive"] = float(perceptron.__dict__.get("bias_row_drive", 1.0)) * (1.0 + 1e-3)

    with pytest.raises(ToolchainRefusal) as caught:
        _program(monkeypatch, tamper=tamper)
    assert type(caught.value).__name__ == "BiasRowDriveRefusal"
