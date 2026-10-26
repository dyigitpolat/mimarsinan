"""[W15] One seeded program fault per named rule the first round left without one (``review_fidelity.md`` F8,
``review_soundness.md`` m1 to m3): each caught by its rule, so every rule of ``RULES`` is shown to bite or is listed
as unmutated with its reason (``test_condition_map``)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENT_CORES

from unit.certification.program_check import programs as P
from unit.certification.program_check.test_mutants import _failed, _scaled_ir, _units
from unit.mapping.residency.executed import vehicles as V

VALUE = dict(accumulation={"value_census": "float64", "int64_golden": "int64"},
             upload={"value_census": "float32"})


def _cored(program):
    for unit in _units(program):
        rows = unit.hard_core_mapping.soft_core_placements_per_hard_core
        for h, records in enumerate(rows):
            if records:
                return unit, h, records
    raise AssertionError("no placed core")


def test_a_placement_record_dropped():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    _, _, records = _cored(bad)
    records.pop()
    report = P.run(bad, ir, P.decl("streamed_lif"))
    _failed(report, "tiles", "coverage")
    _failed(report, "tiles", "unplaced_column")


def test_a_placement_past_the_used_columns():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    unit, h, records = _cored(bad)
    core = unit.hard_core_mapping.cores[h]
    records[0]["neuron_offset"] = int(core.neurons_per_core) - int(core.available_neurons)
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "placement_extent")


def test_a_placement_of_a_node_the_identity_does_not_hold():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    _, _, records = _cored(bad)
    records[0]["ir_node_id"] = 99999
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "foreign_node")


def test_an_output_slice_wider_than_its_node():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    P.neural(bad)[0].output_map[0].size += 1
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "output_width")


def test_a_boundary_scale_of_the_program_changed():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    scales = dict(bad.node_activation_scales)
    key = next(iter(scales))
    scales[key] = 2.0 * float(np.mean(scales[key]))
    bad.node_activation_scales = scales
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "node_scales")


def test_the_network_reading_other_buffers():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    out = list(np.asarray(bad.output_sources).flatten())
    i = next(i for i in range(1, len(out)) if (out[i].node_id, out[i].index) != (out[0].node_id, out[0].index))
    out[0], out[i] = out[i], out[0]
    bad.output_sources = np.array(out, dtype=object)
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "program_outputs")


def _split_level():
    ir = P.integer_ir(V.split_level_repr())
    return V.build(ir, 3, None), ir


def test_a_host_stage_running_another_module():
    program, ir = _split_level()
    bad = copy.deepcopy(program)
    stage = next(s for s in bad.stages if s.kind == "compute" and s.compute_op.name == "host_0")
    op = copy.copy(stage.compute_op)
    op.params = dict(op.params, module=nn.ReLU())
    stage.compute_op = op
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "tiles", "host_op")


def test_a_host_module_left_in_training_mode():
    """A batch-dependent host op (BatchNorm in training mode, W33's t0_01 finding) in both programs."""
    ir = P.integer_ir(V.split_level_repr())
    op = next(n for n in ir.nodes if isinstance(n, ComputeOp) and n.name == "host_0")
    op.params = dict(op.params, module=nn.BatchNorm1d(V.IN).train())
    program = V.build(ir, 3, None)
    _failed(P.run(program, ir, P.decl("streamed_lif")), "tiles", "host_training")
    op.params["module"].eval()
    assert "host_training" not in {v.rule for v in P.run(program, ir, P.decl("streamed_lif")).results[
        "tiles"].violations}


@pytest.mark.parametrize("name", ["activation_scale", "parameter_scale", "input_activation_scale"])
def test_a_stored_singleton_of_the_core_changed(name):
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    core = P.neural(bad)[0].hard_core_mapping.cores[0]
    setattr(core, name, torch.tensor(float(np.mean(np.asarray(getattr(core, name)))) * 2.0 + 1.0))
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H1-theta", name)


def test_a_streamed_program_collapsing_its_carried_slices():
    program, ir = P.streamed_multipass()
    _failed(P.run(program, ir, P.decl("streamed_lif", pass_transfer="collapse")), "H5-collapse", "transfer")


def test_a_delay_on_a_windowed_plain_stage():
    program, ir = P.windowed_multipass(5, 3)
    bad = copy.deepcopy(program)
    stage = next(s for s in P.neural(bad) if not getattr(s, "retimed_level_stages", None))
    stage.replay_delays = {0: -1}
    report = P.run(bad, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=True))
    _failed(report, "H5-collapse", "shifted_count")


def test_a_first_pass_marked_weight_resident():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    P.neural(bad)[0].schedule_weights_resident = True
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H10", "no_programming_pass")


def _pinned(program):
    passes = [s for s in P.neural(program) if getattr(s, SCHEDULE_RESIDENT_CORES, None) is not None]
    assert passes, "the residency plan pins cores"
    return passes


def test_a_pin_of_a_core_the_pass_does_not_hold():
    program, ir = P.residency_planned()
    bad = copy.deepcopy(program)
    setattr(_pinned(bad)[0], SCHEDULE_RESIDENT_CORES, frozenset({999}))
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H10", "pin_out_of_range")


def test_pinned_cores_without_a_declared_pool():
    program, ir = P.residency_planned()
    _failed(P.run(program, ir, P.decl("streamed_lif", core_types=())), "H10", "undeclared_pool")


def test_a_core_holding_neurons_of_two_identity_segments():
    program, ir = _split_level()
    bad = copy.deepcopy(program)
    from mimarsinan.certification.program_check.view import build_view

    iota = build_view(P.identity_of(ir), foreign=-7)
    segment = {p.node: u.program_index for u in iota.units for cv in u.cores for p in cv.placements}
    unit, h, records = _cored(bad)
    other = next(n for n, s in segment.items() if s != segment[int(records[0]["ir_node_id"])])
    records.append(dict(records[0], ir_node_id=other))
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "R3", "mixed_segments")


def test_a_multi_level_windowed_pass_run_streamed_is_a_multi_home_unit():
    for seed in range(12):
        program, ir = P.windowed_multipass(seed, 3)
        bad = copy.deepcopy(program)
        multi = [s for s in P.neural(bad) if len(getattr(s, "retimed_level_stages", None) or ()) >= 2]
        if not multi:
            continue
        multi[0].retimed_level_stages = None
        report = P.run(bad, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=True))
        _failed(report, "R4", "multi_home_unit")
        return
    raise AssertionError("no windowed pass ran two level units")


def test_a_windowed_identity_left_unlevelled():
    program, ir = P.windowed_multipass()
    report = P.run(program, ir, P.decl("windowed_lif"), identity=P.identity_of(ir, retimed=False))
    _failed(report, "H0", "unlevelled_windowed_stage")


def test_level_units_on_a_per_event_core():
    program, ir = P.windowed_multipass()
    report = P.run(program, ir, P.decl("windowed_lif", firing_granularity="per_event"),
                   identity=P.identity_of(ir, retimed=True))
    _failed(report, "H7", "level_unit")


def test_a_split_neuron_on_a_per_event_core():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    _, _, records = _cored(bad)
    records[0]["split_group_id"] = 1
    _failed(P.run(bad, ir, P.decl("streamed_lif", firing_granularity="per_event")), "H7", "split_neuron")


def test_co_resident_soft_cores_on_a_per_event_core(monkeypatch):
    """The projection's refusal switched off (``test_co_resident_soft_cores_at_two_latencies``' vehicle): the token
    family co-packs two soft cores on one core, which a per-event core forbids."""
    import mimarsinan.mapping.packing.pass_timing as pt

    monkeypatch.setattr(pt, "_core_latencies", lambda pl, fused, label: [
        min((fused[n] for n in pl.nodes(h)), default=0) for h in range(len(pl.cores))])
    for count in (2, 3, 4, 6):
        ir = P.integer_ir(V.token_repr(deep=(8,), tokens=2, deep_tokens=(0,)))
        program = V.build(ir, count, None)
        if any(len(records) > 1 for unit in _units(program)
               for records in unit.hard_core_mapping.soft_core_placements_per_hard_core):
            _failed(P.run(program, ir, P.decl("streamed_lif", firing_granularity="per_event")), "H7", "co_residency")
            return
    raise AssertionError("no budget co-packed two soft cores")


def test_a_charge_bound_one_bit_short_under_synchronized_ttfs():
    """The lattice numerator is ``2S (sum |w| + |b|)``: a charge within float64 at ``S`` but not at ``2S``."""
    program, ir = _scaled_ir(float(3 * 2 ** 48))
    assert P.run(program, ir, P.decl("synchronized_lif")).results["H8"].status != "failed"
    _failed(P.run(program, ir, P.decl("synchronized_ttfs")), "H8", "accumulation_bound")


def _value_with(mutate):
    program = P.integer_value()
    mutate(program.ir)
    return program.packed_mapping(axons=32, neurons=32, count=1, passes=4), program.ir


def _first_core(ir):
    return ir.get_neural_cores()[0]


def test_a_weight_a_float32_upload_rounds():
    def raise_weight(ir):
        node = _first_core(ir)
        matrix = np.array(node.core_matrix, dtype=np.float64)
        matrix[0, 0] = float(2 ** 25 + 1)
        node.core_matrix = matrix
    program, ir = _value_with(raise_weight)
    _failed(P.run(program, ir, P.decl("value_integer", **VALUE)), "H8", "upload_precision")


def test_a_law_free_core_in_an_integer_value_program():
    def drop_law(ir):
        _first_core(ir).readout_law = None
    program, ir = _value_with(drop_law)
    _failed(P.run(program, ir, P.decl("value_integer", **VALUE)), "laws", "law_free_core")


def test_a_code_whose_range_no_law_or_grid_declares():
    """A core reading a host op's output with no armed boundary grid of its own: its input range is undeclared."""
    def undeclare(ir):
        hosts = {int(n.id) for n in ir.nodes if isinstance(n, ComputeOp)}
        reader = next(n for n in ir.get_neural_cores() if any(int(s.node_id) in hosts
                                                               for s in n.input_sources.flatten()))
        reader.boundary_grid = None     # it reads a host op's output: no producer law declares the code range
    program, ir = _value_with(undeclare)
    _failed(P.run(program, ir, P.decl("value_integer", **VALUE)), "H8", "input_range")


def test_a_law_whose_magnitude_leaves_float64():
    from mimarsinan.mapping.ir.core_law import CoreLaw

    def huge(ir):
        node = _first_core(ir)
        law = node.readout_law
        node.readout_law = CoreLaw(input_step=law.input_step, m0=np.full_like(np.asarray(law.m0), 2 ** 40),
                                   shift=law.shift, bias=law.bias, lo=law.lo, hi=law.hi,
                                   output_step=law.output_step, adc_shift=law.adc_shift, adc_levels=law.adc_levels)
    program, ir = _value_with(huge)
    _failed(P.run(program, ir, P.decl("value_integer", **VALUE)), "H8", "law_magnitude")


def test_the_network_reading_a_node_no_stage_produces():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    out = list(np.asarray(bad.output_sources).flatten())
    out[0] = IRSource(99999, 0)
    bad.output_sources = np.array(out, dtype=object)
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H9", "output_unproduced")


def test_a_coalescing_group_cut_across_passes():
    program, ir = P.coalesced()
    bad = copy.deepcopy(program)
    group, home = None, None
    for unit in _units(bad):
        for records in unit.hard_core_mapping.soft_core_placements_per_hard_core:
            for r in records:
                if r.get("coalescing_group_id") is not None and group is None:
                    group, home = r["coalescing_group_id"], unit
    other = next(u for u in _units(bad) if u is not home)
    _cored_records = next(rec for rec in other.hard_core_mapping.soft_core_placements_per_hard_core if rec)
    _cored_records[0]["coalescing_group_id"] = group
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H9", "coalescing_group_cut")


def test_compaction_that_drops_a_live_network_output():
    ir = P.integer_ir(V.random_rep(0))
    outputs = {int(s.node_id) for s in np.asarray(ir.output_sources).flatten()}
    node = next(n for n in ir.get_neural_cores() if int(n.id) in outputs)
    rows, cols = node.get_input_count(), node.get_output_count()
    node.pruned_row_mask = [False] * rows
    node.pruned_col_mask = [c == 0 for c in range(cols)]
    matrix = np.array(node.core_matrix, dtype=np.float64)
    matrix[:, 0] = 2.0
    node.core_matrix = matrix
    program = V.build(ir, 3, None)
    _failed(P.run(program, ir, P.decl("synchronized_lif")), "N-identity", "dropped_output")
