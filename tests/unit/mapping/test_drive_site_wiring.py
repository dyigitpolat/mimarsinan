"""[U-BRANGE2 stage 1] The drive site's wiring (the first record's m-3, m-5 and m-6).

* m-5: the drive is the least-error code of the bias the projection codes. Weight Quantization's canonicalization (and
  its OFF clip) can shrink a bias after the entry's stamp; a drive sized for the larger bias is still a code and still
  exact, but not the least-error code of the bias deployed. The step stamps again after the canonicalization.
* m-6: one source. The solvers accept any code value of a core's assigned grid, so a core carrying a code that is not
  its perceptron's stamp would pass them; the IR's drive install refuses by name a core already carrying another drive.
* m-3: the on-chip bias register saturates a compensated bias beyond its span as NAPQ saturates it (NF and SCM
  parity); the codes it clips are counted per core (a census, never silent).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of
from unit.mapping.test_bias_row_drive_program import _program
from unit.transformations.test_bias_row_drive import PT, Q_MAX, W_MAX, X_FS_2, _flow, _stamp, _value_config, _wq


def test_a_bias_the_canonicalization_shrinks_gets_its_own_least_error_drive(monkeypatch):
    from mimarsinan.pipelining.pipeline_steps.quantization import weight_quantization_step as wq

    try:
        from mimarsinan.mapping.support.drive_rule import DriveGrid, RowBudget, least_error_drive_and_rows
    except ImportError:
        pytest.fail("no drive rule on this head: the drive is the top code value whatever the bias")
    shrunk = {}

    def shrink(self, model):  # the canonicalization's effect, read alone: layer 0's bias shrunk to a quarter
        perceptron = model.get_perceptrons()[0]
        with torch.no_grad():
            perceptron.layer.bias.mul_(0.25)
        shrunk["bias"] = PT.get_effective_bias(perceptron).detach().clone()
        shrunk["weight"] = PT.get_effective_weight(perceptron).detach().clone()

    flow = _flow(b_max=30.0 * W_MAX)  # at entry the ratio 30 exceeds the top 20 (the top code); shrunk, 7.5 does not
    captured = {}
    monkeypatch.setattr(wq.WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: captured.update(kw))
    monkeypatch.setattr(wq.WeightQuantizationStep, "_canonicalize_starved_bias_outliers", shrink)
    from unit.transformations.test_bias_row_drive import _Pipeline

    wq.WeightQuantizationStep(_Pipeline(_value_config(), {"model": flow, "adaptation_manager": object()})).process()
    perceptron = flow.get_perceptrons()[0]
    grid = boundary_grid_of(perceptron.input_activation)
    drive_grid = DriveGrid(step=float(grid.step), offset=float(grid.entry_offset()), code_max=int(grid.code_range[1]))
    expected, _rows = least_error_drive_and_rows(shrunk["weight"], shrunk["bias"], drive_grid, Q_MAX,
                                                 fractional=True, budget=RowBudget())
    assert _stamp(perceptron) == expected
    assert _stamp(perceptron) < float(grid.value_range[1])  # not the top the entry's larger bias took


def test_a_core_carrying_another_drive_than_its_perceptron_s_stamp_is_refused_by_name(monkeypatch):
    from mimarsinan.mapping.support import bias_drive

    flow, ir = _program(monkeypatch)
    core = next(n for n in ir.nodes if isinstance(n, NeuralCore) and _stamp(n) != 1.0)
    stamp = _stamp(core)
    grid = boundary_grid_of(flow.get_perceptrons()[0].input_activation)
    other = stamp + float(grid.step) if stamp + float(grid.step) <= float(grid.value_range[1]) else stamp - float(grid.step)
    core.__dict__["bias_row_drive"] = other  # another code of a grid, not the perceptron's stamp
    with pytest.raises(ToolchainRefusal) as caught:
        bias_drive.install_bias_row_drives(ir, flow)
    assert type(caught.value).__name__ == "BiasRowDriveRefusal" and "one source" in str(caught.value)
    assert repr(core.name) in str(caught.value)


def test_the_bias_register_s_saturated_codes_are_counted_on_their_core():
    from mimarsinan.mapping.export import chip_quantize

    census = getattr(chip_quantize, "bias_register_census", None)
    if census is None:
        pytest.fail("the export counts no saturated bias register code (a silent clip)")

    def _core(name, bias):  # a core holding only the fields the register's quantization reads
        core = NeuralCore.__new__(NeuralCore)
        core.__dict__.update(name=name, hardware_bias=np.asarray(bias, dtype=np.float64), bias_scale=None)
        return core

    inside, beyond = _core("inside", [0.5, -0.25]), _core("beyond", [0.5, 3.0, -4.0])
    for core in (inside, beyond):
        chip_quantize._scale_hardware_bias(core, 100.0, -128, 127, np.int16, 1e-12)
    graph = type("G", (), {"nodes": [inside, beyond]})()
    assert census(graph) == {"beyond": 2}
    assert int(beyond.hardware_bias.max()) == 127 and int(beyond.hardware_bias.min()) == -128


# --- [U-BRANGE2 stage 1b] the drive site reads the projection's own width and scale rule; the canonicalization reads
# the drive. The stage's verify: M-4 (VM21, 8-bit weights whatever the layer's width) and M-5 (VM22, the integer floor
# scale whatever the declared rule) survived every clause; the two clauses below are the verify's own
# (probe/test_v_drive_site.py), adopted. m-4 (VM24, the stamp the canonicalization reads removed) is the third.


def _site(factor: float, weight_bits: int):
    """The drive fixture with layer 0's weights scaled by ``factor`` (so the projection's ``q_max / max|w|`` has a
    fractional part) under ``weight_bits``: the stamp, the rule at the projection's own width and fractional rule, the
    rule at width 8 (q_max 127) and the rule under the integer floor scale."""
    from mimarsinan.mapping.support import drive_rule
    from mimarsinan.mapping.support.reader_grids import _perceptron_nodes, reader_input_grids
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives

    flow = _flow()
    p0 = flow.get_perceptrons()[0]
    with torch.no_grad():
        p0.layer.weight.mul_(factor)
    repr_ = flow.get_mapper_repr()
    grid = reader_input_grids(repr_, absorbed=frozenset())[id(_perceptron_nodes(repr_)[0])]
    w, b = PT.get_effective_weight(p0), torch.as_tensor(PT.get_effective_bias(p0))
    one = drive_rule.RowBudget(split=False)

    def rule(q, fractional):
        return drive_rule.least_error_drive(w, b, grid, q, fractional=fractional, budget=one)

    q = (1 << (weight_bits - 1)) - 1
    want = rule(q, True)
    stamp_projection_drives(flow, _value_config(weight_bits=weight_bits))
    return _stamp(p0), want, rule(127, True), rule(q, False)


FACTORS = (1.0, 1.7, 2.9, 4.3, 6.1, 7.7)


def test_the_drive_site_reads_the_projection_s_weight_width():
    """4-bit weights: the stamp is the rule's at q_max 7, and a factor exists where q_max 127 picks another code. Red on
    VM21."""
    told = False
    for factor in FACTORS:
        stamped, want, at_127, _floor = _site(factor, 4)
        assert stamped == want, (factor, stamped, want)
        told = told or at_127 != want
    assert told, "fixture: no factor where the 8-bit width picks another code"


def test_the_drive_site_reads_the_projection_s_scale_rule():
    """The fractional rule declared: the stamp is the rule's under it, and a factor exists where the integer floor
    scale picks another code. Red on VM22."""
    told = False
    for factor in FACTORS:
        stamped, want, _at_127, floor = _site(factor, 4)
        assert stamped == want, (factor, stamped, want)
        told = told or floor != want
    assert told, "fixture: no factor where the integer floor scale picks another code"


def test_the_canonicalization_reads_the_bias_row_at_the_rule_s_drive(monkeypatch):
    """m-4: Weight Quantization stamps the drives before its canonicalization, whose rungs read the bias row at its
    drive (``bias_row_values``). Layer 0's channel 1 carries a bias of 400 max|w| that no decision reads (the next
    layer's column 1 is zero). Read at the raw 1.0 the bias starves the shared grid (every weight at code 0), so the
    canonicalization arms and changes the channel; read at the rule's drive (the top code 20: 20 max|w| over the
    weights' grid) about 8% of the weights round to 0, nothing arms and the channel is kept as it was. The
    canonicalization runs live (the library's call at the step's bits on fixed batches; only the trainer's batch
    source is replaced). Red on VM24."""
    from mimarsinan.pipelining.pipeline_steps.quantization import weight_quantization_step as wq
    from mimarsinan.transformations.perceptron.bias_canonicalization import canonicalize_starved_bias_outliers
    from unit.transformations.test_bias_row_drive import BITS, _Pipeline

    flow = _flow()
    p0, p1 = flow.get_perceptrons()[0], flow.get_perceptrons()[1]
    with torch.no_grad():
        p0.layer.bias[1] = 400.0 * W_MAX
        p1.layer.weight[:, 1] = 0.0
    before_w = PT.get_effective_weight(p0).detach().clone()
    before_b = PT.get_effective_bias(p0).detach().clone()
    batches = [torch.randn(32, 16, generator=torch.Generator().manual_seed(s)) * 5.0 for s in (21, 22)]
    reports = []

    def live(self, model):
        reports.append(canonicalize_starved_bias_outliers(model, batches, bits=BITS))

    monkeypatch.setattr(wq.WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: None)
    monkeypatch.setattr(wq.WeightQuantizationStep, "_canonicalize_starved_bias_outliers", live)
    wq.WeightQuantizationStep(_Pipeline(_value_config(), {"model": flow, "adaptation_manager": object()})).process()
    (report,) = reports
    assert not any(report.values()), report
    assert torch.equal(PT.get_effective_weight(p0).detach(), before_w)
    assert torch.equal(PT.get_effective_bias(p0).detach(), before_b)
    assert _stamp(p0) == 20.0


def test_the_step_line_and_the_refusal_name_the_drive_rule_s_codes(capsys):
    """[U-BRANGE2 stage 1b, the verify's m-6] The drives are the rule's least-error codes of their reader's input grid,
    not its full scale: the step's line and the refusal of a pass that reads 1.0 say so (the stage's line printed
    "at their reader's input full scale" over drives 0.0976 to 3.837 on W38's ResNet-18)."""
    from mimarsinan.mapping.support import bias_drive
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_and_report_drives

    flow = _flow()
    stamps = stamp_and_report_drives(flow, _value_config())
    assert stamps is not None and stamps.drives
    line = capsys.readouterr().out
    assert "the drive rule's codes of their reader's input grid" in line and "full scale" not in line, line
    _flow_, ir = _program_ir(flow)
    with pytest.raises(ToolchainRefusal) as caught:
        bias_drive.refuse_bias_row_drives(ir, "a probe")
    text = str(caught.value)
    assert "the drive rule's codes of their reader's input grid" in text and "full scale" not in text, text


def _program_ir(flow):
    """A stand-in IR holding one core that carries the first perceptron's stamp (the refusal reads the cores alone)."""
    core = NeuralCore.__new__(NeuralCore)
    core.__dict__.update(name="core0", bias_row_drive=_stamp(flow.get_perceptrons()[0]))
    return flow, type("G", (), {"nodes": [core]})()


def test_the_drive_site_reads_the_layer_s_own_core_type_width():
    """Sweep SW20: a layer whose core type declares its own weight width (WI-5g: ``install_layer_weight_bits`` stamps
    it on the perceptron) is projected at that width, so the drive site reads it: layer 0 stamped at 4 bits under a
    run width of 8 takes the rule at q_max 7, and a factor exists where q_max 127 picks another code."""
    from mimarsinan.mapping.support import drive_rule
    from mimarsinan.mapping.support.reader_grids import _perceptron_nodes, reader_input_grids
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives
    from mimarsinan.transformations.quantization_bounds import LAYER_WEIGHT_BITS

    told = False
    for factor in FACTORS:
        flow = _flow()
        p0 = flow.get_perceptrons()[0]
        with torch.no_grad():
            p0.layer.weight.mul_(factor)
        setattr(p0, LAYER_WEIGHT_BITS, 4)
        repr_ = flow.get_mapper_repr()
        grid = reader_input_grids(repr_, absorbed=frozenset())[id(_perceptron_nodes(repr_)[0])]
        w, b = PT.get_effective_weight(p0), torch.as_tensor(PT.get_effective_bias(p0))
        one = drive_rule.RowBudget(split=False)
        want = drive_rule.least_error_drive(w, b, grid, 7, fractional=True, budget=one)
        stamp_projection_drives(flow, _value_config())  # the run's width: 8 bits
        assert _stamp(p0) == want, (factor, _stamp(p0), want)
        told = told or drive_rule.least_error_drive(w, b, grid, 127, fractional=True, budget=one) != want
    assert told, "fixture: no factor where the run's width picks another code"


def test_the_drive_site_reads_the_declared_integer_scale_rule():
    """Sweep SW26: the route passes the projection's declared scale rule: under ``weight_scale_rule: integer`` (the
    floor scale) the stamp is the rule's under the floor scale, and a factor exists where the fractional rule picks
    another code."""
    from mimarsinan.mapping.support import drive_rule
    from mimarsinan.mapping.support.reader_grids import _perceptron_nodes, reader_input_grids
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives

    told = False
    for factor in FACTORS:
        flow = _flow()
        p0 = flow.get_perceptrons()[0]
        with torch.no_grad():
            p0.layer.weight.mul_(factor)
        repr_ = flow.get_mapper_repr()
        grid = reader_input_grids(repr_, absorbed=frozenset())[id(_perceptron_nodes(repr_)[0])]
        w, b = PT.get_effective_weight(p0), torch.as_tensor(PT.get_effective_bias(p0))
        one = drive_rule.RowBudget(split=False)
        want = drive_rule.least_error_drive(w, b, grid, 7, fractional=False, budget=one)
        stamp_projection_drives(flow, _value_config(weight_bits=4, weight_scale_rule="integer"))
        assert _stamp(p0) == want, (factor, _stamp(p0), want)
        told = told or drive_rule.least_error_drive(w, b, grid, 7, fractional=True, budget=one) != want
    assert told, "fixture: no factor where the fractional rule picks another code"


def test_a_drive_below_one_is_reported_with_the_others():
    """Sweep SW24: the stamps report every drive other than 1.0 (the step's line prints them; W38's ResNet-18 smoke
    read drives from 0.0976): an entry grid of full scale 0.5 gives layer 0 a drive below 1.0, which the report
    holds."""
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives

    flow = _flow(x_fs=(0.5, X_FS_2), b_max=0.2 * W_MAX)
    stamps = stamp_projection_drives(flow, _value_config())
    p0 = flow.get_perceptrons()[0]
    assert 0.0 < _stamp(p0) < 1.0
    assert stamps is not None and stamps.drives.get(str(p0.name)) == _stamp(p0), stamps.drives


def test_on_a_lane_program_every_reader_s_drive_is_a_code_of_the_grid_it_reads_through_its_lanes():
    """Sweep SW28: the route passes the declared lanes' absorbed nodes to the reader grids. On the residual block with
    clamp, add and pool lanes declared and the bias on the always-on rows, a perceptron that reads a lane through an
    absorbed clamp reads the lane's grid; read without the absorbed nodes its input would be a host value with no
    armed entry grid (refused by name). Every core-mapped perceptron is stamped a code of its reader grid."""
    import contextlib
    import io

    from mimarsinan.mapping.lanes.model_lanes import install_planned_lanes, lane_absorbed
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.mapping.support.reader_grids import _perceptron_nodes, reader_input_grids
    from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives
    from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
        calibrate_boundary_scales,
        install_boundary_quantizers,
    )
    from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import (
        ReadoutLawStep,
        install_readout_quantizers,
    )
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
    from unit.chip_simulation.value_run.integer_programs import residual_block_net
    from unit.mapping.lanes.lane_programs import lane_platform

    platform = lane_platform(("L", "E", "P"), activation_bits=4)
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        flow = convert_torch_model(residual_block_net().eval(), (3, 8, 8), 10, device="cpu",
                                   packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    batches = [torch.randn(32, 3, 8, 8, generator=torch.Generator().manual_seed(1))]
    absorbed = lane_absorbed(flow, platform)
    assert absorbed, "fixture: lanes absorb host ops"
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr(), absorbed))
    calibrate_boundary_scales(flow, entries, install_boundary_quantizers(entries, 4), batches)
    quantizers = install_readout_quantizers(perceptrons, 4)
    lanes = install_planned_lanes(flow, platform, 4)
    every = quantizers + [q for _, q in lanes]
    for quantizer, value in zip(every, ReadoutLawStep._calibrated(flow, perceptrons, every, batches, lanes,
                                                                  absorbed)):
        quantizer.calibrate(value)
    flow.get_mapper_repr().assign_perceptron_indices()
    config = {**_value_config(weight_bits=4, activation_bits=4), **platform}
    stamps = stamp_projection_drives(flow, config)
    assert stamps is not None and not stamps.undetermined, stamps
    repr_ = flow.get_mapper_repr()
    grids = reader_input_grids(repr_, absorbed=absorbed)
    blind = reader_input_grids(repr_, absorbed=frozenset())
    told = False
    for node in _perceptron_nodes(repr_):
        if getattr(node.perceptron, "is_encoding_layer", False):
            continue
        grid = grids[id(node)]
        assert grid is not None and grid.holds(_stamp(node.perceptron)), (node.perceptron.name, grid)
        told = told or blind.get(id(node)) is None
    assert told, "fixture: a perceptron whose grid is read through an absorbed lane op"


def test_an_entry_grid_s_drive_codes_are_its_declared_codes_with_their_offset():
    """Sweep SW53: a reader of host values reads its entry grid; a declared grid with an offset (codes ``0..L``, value
    ``k step - offset``) gives the drive the same code values, its top the grid's own (``value_range``)."""
    from mimarsinan.mapping.support.reader_grids import entry_drive_grid
    from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid

    grid = DeclaredBoundaryGrid(scale=6.0, bits=5, offset=3.0, rounding="floor", code_min=0, code_max=15)
    drive = entry_drive_grid(grid)
    assert drive is not None and drive.code_max == grid.code_range[1] == 15
    assert drive.top == grid.value_range[1] == 15 * grid.step - 3.0
    assert all(drive.value(k) == k * grid.step - grid.entry_offset() for k in range(1, 16))


@pytest.mark.parametrize("unsigned, offset", [(True, 0.0), (True, 0.75), (False, 0.0)])
def test_a_lane_reader_s_drive_grid_is_the_lane_s_own_grid(unsigned, offset):
    """Sweep SW54 and SW55: a reader of a lane reads the grid the lane installer gives its law solver
    (``lanes.install._grid``: ``LaneGrid``): codes ``0..2^b - 1`` on an unsigned lane, ``-L..L`` otherwise, the step
    over that code range, the lane's offset."""
    from mimarsinan.mapping.lanes.install import _grid
    from mimarsinan.mapping.support.reader_grids import _lane_grid
    from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer

    quantizer = ReadoutQuantizer(4)
    quantizer.set_lane(unsigned=unsigned, offset=offset)
    quantizer.calibrate(3.0)
    lane = _grid(quantizer)
    grid = _lane_grid(quantizer, 3.0)
    assert grid is not None
    assert (grid.code_max, grid.step, grid.offset) == (lane.codes(4)[1], lane.step(4), float(lane.offset)), (grid, lane)


def test_the_lane_solver_checks_a_drive_against_its_offset_grid(tmp_path):
    """Sweep SW65: the lane solver checks each core's drive against the input grid it assigns the core, its offset
    included. ``conv1``'s lane is declared on an offset grid (codes 0..31 of step 0.125, offset 2 steps: values -0.25 to
    3.625); ``conv2`` reads it on chip and carries a drive of 31 steps (3.875), a code of the grid without its offset
    but above this grid's top: the lane solver refuses it by name."""
    from pathlib import Path

    import torch.nn as nn

    from mimarsinan.mapping.support.quantizer_state import perceptron_fx_names
    from mimarsinan.transformations.perceptron.bias_drive import set_bias_row_drive
    from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, write_state

    step, offset = 3.875 / 31, 2 * (3.875 / 31)
    platform = lane_platform(("L", "E", "P"), activation_bits=5,
                             readout_law={"adc_bits": 40, "range": "calibrated", "mult_bits": 24})

    def declare(flow, entries, grids, perceptrons, quantizers, lanes):
        platform["quantizer_state"] = write_state(Path(tmp_path) / "state.json",
                                                  {"_0": {"readout_range": 3.875, "readout_offset": offset}})
        set_bias_row_drive(perceptron_fx_names(flow.get_mapper_repr())["_1"], 31 * step)

    torch.manual_seed(3)
    model = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1), nn.Conv2d(6, 8, 3), nn.ReLU(), nn.Flatten(),
                          nn.Linear(8 * 6 * 6, 4)).eval()
    with pytest.raises(ToolchainRefusal, match="no code value of the input grid"):
        build_lane_program(model, (1, 8, 8), num_classes=4, platform=platform, hardware_bias=False, seed=3,
                           weight_bits=4, state_hook=declare)
