"""[U-BRG, W38 bias-row gauge] A program whose always-on rows carry their reader's full-scale code is exact.

The Weight Quantization projection codes each always-on row at its drive ``x_fs`` (the top code value of the grid its
reader reads; ``tests/unit/transformations/test_bias_row_drive.py``). The IR row then holds the row weight ``b / x_fs``,
quantized to ``c = round(b s / x_fs)``, and the readout law's per-column bias term carries the drive:
``B = rnd(c x_fs 2^shift / (theta step))``, so the deployed program computes ``(Delta_in acc + c x_fs) / theta``: the
projected model's bias exactly. The model (NF, reading the stamped law), the identity program, the packed program and
the int64 golden then agree exactly (atol 0) on the program's own route.

A reader of host-sourced axons reads its entry grid; a reader of core outputs reads the grid the law solver assigns it
(the widest range of the producers read together by any one core, ``law_grids.assign_core_grids``, and the lane
solver's ``assign_lane_grids``), and each solver refuses by name a core whose drive is not that grid's top code value.
IR pruning's constant lattice reads an always-on axon at 1.0, so pruning with constant folding under a drive is refused
by name; with the lattice off it reads no always-on value and runs, its kill sets those of the drive 1.

The sequence is the pipeline's (mapping, boundary grids, quantization, laws) through the public functions, the same on
both trees, so a tree without the drive reads its own program.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.chip_quantize import ParameterGrids, quantize_ir_graph
from mimarsinan.mapping.export.law_grids import assign_core_grids
from mimarsinan.mapping.export.law_solver import install_readout_laws
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping, build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.boundary_grids import install_boundary_grids
from mimarsinan.models.nn.activations.value_quantizer import boundary_grid_of
from mimarsinan.pipelining.pipeline_steps.quantization.boundary_quantization_step import (
    calibrate_boundary_scales,
    install_boundary_quantizers,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_law_step import install_readout_quantizers
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import segment_entry_perceptrons
from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron
from unit.transformations.test_bias_row_drive import BITS, LAW, Q_MAX, _flow, _value_config, _wq


def _stamp(holder) -> float:
    """A holder's row drive read without the unit's module (1.0 without a stamp), so the test runs on any tree."""
    return float(holder.__dict__.get("bias_row_drive", 1.0))


def _laws(ir, flow, config) -> None:
    install_boundary_grids(ir, flow)
    quantize_ir_graph(ir, BITS, weight_quantization=True, grids=ParameterGrids(fractional_scale=True))
    install_readout_laws(ir, flow, {"readout_law": dict(LAW), "activation_bits": BITS,
                                    "weight_scale_rule": "fractional", **{k: v for k, v in config.items()
                                                                          if k in ("bias_row_splitting",)}},
                         weight_quantization=True, weight_bits=BITS)


def _map(flow):
    return IRMapping(q_max=Q_MAX, firing_mode="Default", max_axons=None, max_neurons=None, allow_coalescing=False,
                     hardware_bias=False).map(flow.get_mapper_repr())


def _program(monkeypatch, flow=None, tamper=None, **extra):
    flow = _flow() if flow is None else flow
    config = _value_config(**extra)
    _wq(monkeypatch, flow, config)
    if tamper is not None:
        tamper(flow)
    ir = _map(flow)
    _laws(ir, flow, config)
    return flow, ir


def _packed(ir):
    strategy = MappingStrategy.resolve(ChipCapabilities(allow_neuron_splitting=True, allow_coalescing=False,
                                                        allow_scheduling=False, max_schedule_passes=8))
    return build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=[{"max_axons": 64, "max_neurons": 8,
                                                                      "count": 999}],
                                          strategy=strategy, max_schedule_passes=8)


def _inputs(n: int = 16) -> torch.Tensor:
    return (torch.randn(64, n, generator=torch.Generator().manual_seed(11)) * 8.0).double()


def _perceptrons_by_index(flow) -> dict:
    repr_ = flow.get_mapper_repr()
    order, _, _ = repr_._ensure_exec_graph()
    return {int(n.perceptron_index): n.perceptron for n in order
            if getattr(n, "perceptron", None) is not None and getattr(n, "perceptron_index", None) is not None}


def _exact(flow, ir, x) -> None:
    packed = _packed(ir)
    with torch.no_grad():
        identity = ValueHybridCoreFlow(build_identity_hybrid_mapping(ir_graph=ir), dtype=torch.float64)(x)
        packed_out = ValueHybridCoreFlow(packed, dtype=torch.float64)(x)
        nf = flow.double()(x)
    torch.testing.assert_close(identity, packed_out, atol=0, rtol=0)
    torch.testing.assert_close(nf.double(), identity, atol=0, rtol=0)
    codes, steps = run_golden(packed, x.numpy())
    np.testing.assert_array_equal(codes * steps, packed_out.numpy())


# ---------------------------------------------------------------------------
# The law and the twins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", [pytest.param({}, id="one-row"),
                                   pytest.param({"bias_row_splitting": "auto"}, id="rows-auto")])
def test_the_law_bias_term_carries_the_drive(monkeypatch, extra):
    """Each core's law bias is the always-on rows' code sum times the reader's full scale, on the law's resolution:
    ``B = rnd(c x_fs 2^shift / (theta step))``. Without the drive it is ``rnd(c 2^shift / (theta step))`` (the row
    at 1.0)."""
    flow, ir = _program(monkeypatch, **extra)
    by_index = _perceptrons_by_index(flow)
    checked = 0
    for node in ir.nodes:
        if not isinstance(node, NeuralCore):
            continue
        matrix = np.asarray(node.get_core_matrix(ir), dtype=np.float64)
        on = np.array([s.is_always_on() for s in node.input_sources.flatten()])
        if not on.any():
            continue
        c = matrix[on, :].sum(axis=0)
        drive = float(boundary_grid_of(by_index[int(node.perceptron_index)].input_activation).value_range[1])
        law = node.readout_law
        step = np.asarray(law.output_step, dtype=np.float64)
        exact = np.ldexp(c * drive / (float(node.threshold) * step), np.asarray(law.shift))
        error = np.abs(np.asarray(law.bias, dtype=np.float64) - exact)
        assert bool(np.all(error <= 0.5 + 1e-9 * np.abs(exact) + 1e-9)), (node.name, float(error.max()))
        assert np.abs(exact).max() > 1.0, "fixture: a nonzero bias term"
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("extra", [pytest.param({}, id="one-row"),
                                   pytest.param({"bias_row_splitting": "auto"}, id="rows-auto")])
def test_nf_identity_packed_and_golden_agree_exactly_at_the_drive(monkeypatch, extra):
    """Guard (green on both trees): the model with its stamped laws, the identity program, the packed program and the
    int64 golden give the same outputs at atol 0."""
    flow, ir = _program(monkeypatch, **extra)
    _exact(flow, ir, _inputs())


def test_the_static_check_knows_the_drive_form(monkeypatch):
    """The static check (W15) reads a program only through registered forms: a core carrying a row drive must not
    fail Theorem 1 by ``unknown program form`` (the drive reaches the program only through the law's bias term, which
    the ``laws`` condition compares)."""
    from mimarsinan.certification.program_check.forms import unknown_forms

    flow, ir = _program(monkeypatch)
    assert any(_stamp(n) != 1.0 for n in ir.nodes), "fixture: a stamped drive"
    found = [v for v in unknown_forms(_packed(ir), ir) if "bias_row_drive" in v.rule]
    assert found == [], found


# ---------------------------------------------------------------------------
# m-5: a reader of core outputs reads the grid the law solver assigns it
# ---------------------------------------------------------------------------


class _Joined(nn.Module):
    """``a`` and ``b`` read the input; ``c`` reads ``cat(a, b)``, ``d`` reads ``a`` alone. The solver joins ``a`` and
    ``b`` (read together by ``c``'s cores), so ``d`` reads their union's grid, ``b``'s wider range, not ``a``'s."""

    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(16, 6), nn.Linear(16, 6)
        self.c, self.d = nn.Linear(12, 4), nn.Linear(6, 4)

    def forward(self, x):
        a, b = self.a(x), self.b(x)
        return torch.cat([self.c(torch.cat([a, b], dim=1)), self.d(a)], dim=1)


RANGES = {"a": 2.0, "b": 5.0, "c": 3.0, "d": 4.0}


def _joined_flow():
    torch.manual_seed(0)
    flow = convert_torch_model(_Joined().eval(), (16,), 8, device="cpu", packaging=MVM_PACKAGING).eval()
    perceptrons = list(flow.get_perceptrons())
    for perceptron in perceptrons:
        fuse_into_perceptron(perceptron, device="cpu")
    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for perceptron in perceptrons:
            perceptron.layer.bias.copy_((torch.rand(perceptron.layer.bias.shape, generator=gen) * 2 - 1) * 0.6)
    entries = list(segment_entry_perceptrons(flow.get_mapper_repr()))
    grids = install_boundary_quantizers(entries, BITS)
    calibrate_boundary_scales(flow, entries, grids, [torch.randn(32, 16, generator=gen) * 3.0])
    readouts = install_readout_quantizers(perceptrons, BITS)
    for perceptron, quantizer in zip(perceptrons, readouts):
        quantizer.calibrate(RANGES[str(perceptron.name).split(".")[-1].strip("_")])
    flow.get_mapper_repr().assign_perceptron_indices()
    return flow


def test_a_core_fed_reader_drives_its_row_at_the_top_code_of_the_solver_assigned_grid(monkeypatch):
    """[m-5] ``d`` reads core ``a`` alone, but the law solver joins ``a`` with ``b`` (read together by ``c``), so the
    grid ``d``'s cores read spans ``b``'s range 5: its row is driven at that grid's top code value (the widest direct
    producer, ``a``'s 2, is not a code of it), equal to the assigned grid's ``input_step x input_levels``; the program
    stays exact."""
    flow = _joined_flow()
    names = {str(p.name).split(".")[-1].strip("_"): p for p in flow.get_perceptrons()}
    assert set(names) == set(RANGES), f"fixture: perceptrons {sorted(names)}"
    flow, ir = _program(monkeypatch, flow=flow)
    for name in ("c", "d"):
        assert boundary_grid_of(names[name].input_activation) is None, f"fixture: {name} reads core outputs"
    ranges = {int(i): float(p.scaler.range) for i, p in _perceptrons_by_index(flow).items()}
    grids, _ = assign_core_grids(ir, ranges, BITS)
    index = {id(p): i for i, p in _perceptrons_by_index(flow).items()}
    for name in ("c", "d"):
        drive = _stamp(names[name])
        cores = [n for n in ir.nodes if isinstance(n, NeuralCore) and n.perceptron_index == index[id(names[name])]]
        assert cores
        for core in cores:
            grid = grids[int(core.id)]
            assert drive == grid.input_step * grid.input_levels
        assert drive == pytest.approx(RANGES["b"], rel=1e-12)
    _exact(flow, ir, _inputs())


def test_a_row_drive_off_its_assigned_grid_is_refused_by_name(monkeypatch):
    """A core whose stamped drive is not the top code value of the input grid its law solver assigns it is refused by
    name at the law hook, never solved at a drive the reader's DAC does not produce."""
    def tamper(flow):
        perceptron = flow.get_perceptrons()[0]
        perceptron.__dict__["bias_row_drive"] = 2.0 * float(perceptron.__dict__.get("bias_row_drive", 1.0))

    with pytest.raises(ToolchainRefusal) as caught:
        _program(monkeypatch, tamper=tamper)
    assert type(caught.value).__name__ == "BiasRowDriveRefusal"


def test_a_lane_program_drives_its_core_fed_rows_at_their_lane_grid_tops_and_stays_exact():
    """[m-5, lanes] On a residual block with the lanes on chip (adds and clamps in lanes, every perceptron after the
    stem reading core outputs), the stamped drives agree with the lane solver's assigned grids (its check refuses
    otherwise), at least one core-fed core carries a drive other than 1, and NF = SCM = HCM = the golden at atol 0."""
    from mimarsinan.mapping.lanes.model_lanes import lane_absorbed
    from mimarsinan.mapping.support.bias_drive import stamp_bias_row_drives  # absent without the unit
    from unit.chip_simulation.value_run.integer_programs import assert_exact, residual_block_net
    from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform, real_inputs

    platform = lane_platform()
    stamped: dict = {}

    def hook(flow, *_rest):
        stamps = stamp_bias_row_drives(flow, absorbed=lane_absorbed(flow, platform))
        stamped.update(stamps.drives)
        assert not stamps.undetermined, stamps.undetermined

    program = build_lane_program(residual_block_net(), (3, 16, 16), num_classes=10, platform=platform,
                                 state_hook=hook)
    nodes = {int(n.id): n for n in program.ir.nodes}
    core_fed = [n for n in program.ir.nodes if isinstance(n, NeuralCore) and _stamp(n) != 1.0
                and any(s.is_always_on() for s in n.input_sources.flatten())
                and any(s.node_id >= 0 and isinstance(nodes.get(int(s.node_id)), NeuralCore)
                        for s in n.input_sources.flatten())]
    assert stamped and core_fed, "fixture: core-fed readers with a drive"
    x = real_inputs(program, 24)
    with torch.no_grad():
        scm = program.identity()(x)
        assert_exact(program.nf()(x), scm)
        packed = program.packed(axons=256, neurons=32)(x)
    assert_exact(packed, scm)


# ---------------------------------------------------------------------------
# IR pruning under a drive (the liveness pass reads none; the constant lattice is refused)
# ---------------------------------------------------------------------------


def _pruning_step(tmp_path, **config):
    return SimpleNamespace(pipeline=SimpleNamespace(
        config={**_value_config(), "pruning": True, "simulation_steps": 8, **config},
        reporter=SimpleNamespace(report=lambda *a, **k: None), working_directory=str(tmp_path)))


def _pruned_flow():
    flow = _flow()
    with torch.no_grad():  # a dead channel: layer 0's channel 3 (zero weights and bias) feeds no live value
        perceptron = flow.get_perceptrons()[0]
        perceptron.layer.weight[3, :] = 0.0
        perceptron.layer.bias[3] = 0.0
    return flow


def test_ir_pruning_with_constant_folding_under_a_drive_is_refused_by_name(monkeypatch, tmp_path):
    """IR pruning's constant lattice and its carrier fold read an always-on axon at 1.0: under a drive other than 1
    with constant folding in effect (``elimination_constant_folding`` 'full', the default) the combination is refused
    by name before any pruning runs (a boundary of the unit, not a silent fold). Without the drive the pruning runs."""
    from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_ir_pruning import apply_ir_pruning_if_enabled

    flow, ir = _program(monkeypatch, flow=_pruned_flow())
    with pytest.raises(ToolchainRefusal) as caught:
        apply_ir_pruning_if_enabled(_pruning_step(tmp_path), flow, ir, "test")
    assert type(caught.value).__name__ == "BiasRowDriveRefusal"


def _structure(ir) -> list:
    return [(n.name, tuple(np.asarray(n.get_core_matrix(ir)).shape),
             tuple((int(s.node_id), int(s.index)) for s in n.input_sources.flatten()))
            for n in ir.nodes if isinstance(n, NeuralCore)]


def test_ir_pruning_with_constant_folding_off_runs_under_a_drive_with_the_kill_sets_of_drive_one(
        monkeypatch, tmp_path):
    """Guard (green on both trees). With the constant lattice off (``elimination_constant_folding`` 'off': it never
    descends) the zero-only cascade reads no always-on value, so a positive drive changes no kill set: the stamped IR
    prunes to the structure of the same IR with its drives removed, and nothing is refused."""
    from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_ir_pruning import apply_ir_pruning_if_enabled

    flow, ir = _program(monkeypatch, flow=_pruned_flow())
    plain = copy.deepcopy(ir)
    for node in plain.nodes:
        node.__dict__.pop("bias_row_drive", None)
    rows_before = sum(len(s[2]) for s in _structure(ir))
    step = _pruning_step(tmp_path, elimination_constant_folding="off")
    driven = apply_ir_pruning_if_enabled(step, flow, ir, "test")
    undriven = apply_ir_pruning_if_enabled(step, flow, plain, "test")
    assert _structure(driven) == _structure(undriven)
    assert sum(len(s[2]) for s in _structure(driven)) < rows_before, "fixture: the dead channel's reader row is pruned"
