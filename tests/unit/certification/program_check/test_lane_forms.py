"""[E6] The lanes' program forms under W15 (WI-5a to WI-5e, W08, the declared value grids). A fused add's operand rows
carry zero crossbar weights, so no fan-in comparison sees them: the executor reads each column's operand code at
the row its lane law names (``value_law.lane_rows``). The laws condition holds every lane law's operand row by the
source it reads (the row index is the placement's, cost-only), and the other lane fields as stamped. A core whose
bias the law resolves (``bias_resolution: law``) holds a real-valued hardware bias no executor reads (the law's integer
bias is the executed one), so H8's integral-bias clause binds only the cores that accumulate their bias. Pool lanes'
reduction cores and producer-side reductions are keyed in ``ir-tiling`` by their lane and pool output."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED, ConditionResult
from mimarsinan.certification.program_check.unbounded import check_ir_tiling

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV


def _failed(report, condition: str, rule: str) -> None:
    result = report.results[condition]
    assert result.status == FAILED, (condition, result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    verdict = report.theorem1 if condition != "N-identity" else report.proposition2
    assert verdict.startswith("failed") and condition in verdict, verdict


def _operand_cores(program):
    """(core, law) of every deployed core whose lane law fuses an operand, the core placed off row 0 first."""
    found = []
    for stage in P.neural(program):
        for core in stage.hard_core_mapping.cores:
            law = getattr(core, "readout_law", None)
            if law is not None and hasattr(law, "operand_row") and (np.asarray(law.operand_row) >= 0).any():
                found.append((core, law))
    return sorted(found, key=lambda pair: -int(np.asarray(pair[1].operand_row).max()))


def test_a_lane_program_with_fused_operands_off_row_zero_is_established():
    program, ir, decl = UV.lanes()
    cores = _operand_cores(program)
    assert cores, "the vehicle fuses an add"
    placed = [p for c, _ in cores for p in c.matrix_placements if p.axon_offset > 0]
    assert placed, "an operand core sits at a nonzero axon offset on its hard core"
    report = P.run(program, ir, decl)
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
    assert report.results["laws"].status == ESTABLISHED and report.results["laws"].checked > 0


def _swap_rows(core, a: int, b: int) -> None:
    sources = list(core.axon_sources)
    sources[a], sources[b] = sources[b], sources[a]
    core.axon_sources = sources
    core._axon_source_spans = None


def test_a_fused_operand_row_reading_another_source():
    """Two columns' operand rows exchange their sources: zero weights, so only the law's operand source sees it."""
    program, ir, decl = UV.lanes()
    bad = copy.deepcopy(program)
    for core, law in _operand_cores(bad):
        rows = sorted({int(r) for r in np.asarray(law.operand_row) if r >= 0})
        if len(rows) >= 2:
            _swap_rows(core, rows[0], rows[1])
            break
    else:
        pytest.skip("no core fuses two operand rows")
    _failed(P.run(bad, ir, decl), "laws", "operand_source")


def test_a_lane_law_naming_another_operand_row():
    program, ir, decl = UV.lanes()
    bad = copy.deepcopy(program)
    core, law = _operand_cores(bad)[0]
    rows = np.asarray(law.operand_row, dtype=np.int64).copy()
    column = int(np.flatnonzero(rows >= 0)[0])
    used = int(core.axons_per_core) - int(core.available_axons)
    rows[column] = (rows[column] + 1) % used
    core.readout_law = law.with_operand_rows(rows)
    _failed(P.run(bad, ir, decl), "laws", "operand_source")


def test_a_lane_field_stamped_differently_is_still_a_stamped_law():
    program, ir, decl = UV.lanes()
    bad = copy.deepcopy(program)
    core, law = _operand_cores(bad)[0]
    core.readout_law = dataclasses.replace(law, operand_m=np.asarray(law.operand_m) + 1)
    _failed(P.run(bad, ir, decl), "laws", "stamped_law")


def test_a_bias_at_the_laws_resolution_is_no_executors_charge():
    """WI-5g's declared grids (fractional weight scale, bias at the law's resolution) on typed cores: the law cores
    hold real-valued hardware biases, and the program is established."""
    program, ir, decl = UV.typed()
    real = [c for s in P.neural(program) for c in s.hard_core_mapping.cores
            if getattr(c, "readout_law", None) is not None and c.hardware_bias is not None
            and not np.array_equal(np.asarray(c.hardware_bias), np.round(np.asarray(c.hardware_bias)))]
    assert real, "the vehicle resolves a bias in its law"
    report = P.run(program, ir, decl)
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
    assert report.results["H8"].status == ESTABLISHED and report.results["H8"].checked > 0


def test_a_core_that_accumulates_its_bias_still_needs_it_integral():
    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    core = P.neural(bad)[0].hard_core_mapping.cores[0]
    bias = np.zeros(core.neurons_per_core) if core.hardware_bias is None else np.array(core.hardware_bias, float)
    bias[0] += 0.5
    core.hardware_bias = bias
    _failed(P.run(bad, ir, P.decl("streamed_lif")), "H8", "off_lattice")


def test_pool_lanes_and_reductions_are_keyed_in_ir_tiling():
    program, ir, decl = UV.lanes()
    pool_cores = [n for n in ir.get_neural_cores() if getattr(n, "lane_outputs", None) is not None]
    assert pool_cores, "the vehicle runs a pool lane"
    report = P.run(program, ir, decl, unbounded_ir=ir)
    tiling = report.results["ir-tiling"]
    assert tiling.status == ESTABLISHED, [v.text() for v in tiling.violations]


def test_a_pool_lane_column_keyed_on_another_pool_output():
    """Two columns of one reduction core exchange their pool outputs in the deployed IR: each now holds another
    output's neuron at unbounded geometry, with that neuron's fan-in."""
    _program, ir, decl = UV.lanes()
    tiled = copy.deepcopy(ir)
    core = next(n for n in tiled.get_neural_cores()
                if getattr(n, "lane_outputs", None) is not None and len(n.lane_outputs) >= 2
                and not np.array_equal(np.asarray(n.get_core_matrix(tiled))[:, 0],
                                       np.asarray(n.get_core_matrix(tiled))[:, 1]))
    outputs = np.asarray(core.lane_outputs).copy()
    outputs[[0, 1]] = outputs[[1, 0]]
    core.lane_outputs = outputs
    result = ConditionResult("ir-tiling", ESTABLISHED)
    check_ir_tiling(tiled, ir, declaration=decl, result=result)
    assert "fan_in" in {v.rule for v in result.violations}, [v.text() for v in result.violations]
