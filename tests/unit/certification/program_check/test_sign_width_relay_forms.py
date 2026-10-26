"""[E6] The closure's per-axon sign rows, WI-5g's per-type weight bits, depth-balancing relays and Swin's columnar axon
sources under W15.

A per-axon sign split gives a mixed-sign row a W+ and a W- row (``SignedRowPlacement.row_signs``); both read the same
source, so the executed fan-in summed per source equals the identity's (tiles). A core of a typed width holds only codes
of its type's grid (H8, ``cell_width``): the declared cell is the narrowest executor of that core. A relay is an IR
neuron like any other to the deployment's conditions, and in ``ir-tiling`` it is keyed by the neuron it relays, so a
relayed IR at the declared geometry compares with the relayed IR at unbounded geometry. A columnar core's axon sources
read back through the same spans every executor reads (tiles)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED, ConditionResult
from mimarsinan.certification.program_check.unbounded import check_ir_tiling
from mimarsinan.mapping.packing.softcore.matrix_placement import SignedRowPlacement
from mimarsinan.mapping.support.spike_source_columns import SpikeSourceColumns
from mimarsinan.transformations.quantization_bounds import CORE_WEIGHT_BITS, quantization_bounds

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV


def _failed(report, condition: str, rule: str) -> None:
    result = report.results[condition]
    assert result.status == FAILED, (condition, result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    assert report.theorem1.startswith("failed") and condition in report.theorem1, report.theorem1


def test_a_per_axon_sign_program_is_established():
    program, ir, decl = UV.per_axon()
    signed = [mp for s in P.neural(program) for c in s.hard_core_mapping.cores for mp in c.matrix_placements or ()
              if isinstance(mp, SignedRowPlacement)]
    assert signed and any(len(set(np.asarray(mp.keep_rows).tolist())) < len(mp.keep_rows) for mp in signed)
    report = P.run(program, ir, decl)
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())


def test_a_sign_row_flipped():
    program, ir, decl = UV.per_axon()
    bad = copy.deepcopy(program)
    core, placement = next((c, mp) for s in P.neural(bad) for c in s.hard_core_mapping.cores
                           for mp in c.matrix_placements or () if isinstance(mp, SignedRowPlacement)
                           and np.asarray(mp.materialize()).any())
    signs = np.array(placement.row_signs, copy=True)
    row = int(np.flatnonzero(np.any(np.asarray(placement.materialize()) != 0, axis=1))[0])
    signs[row] = -signs[row]
    placement.row_signs = signs
    _failed(P.run(bad, ir, decl), "tiles", "fan_in")


def test_a_typed_program_holds_codes_of_each_cores_width():
    program, ir, decl = UV.typed()
    widths = {getattr(c, CORE_WEIGHT_BITS, None) for s in P.neural(program) for c in s.hard_core_mapping.cores}
    assert {2, 4} <= widths
    report = P.run(program, ir, decl)
    assert report.results["H8"].status == ESTABLISHED, [v.text() for v in report.results["H8"].violations]
    assert report.counts["typed_cores"] > 0


def test_a_core_declared_narrower_than_its_codes():
    program, ir, decl = UV.typed()
    bad = copy.deepcopy(program)
    q_min, q_max = quantization_bounds(2)
    core = next(c for s in P.neural(bad) for c in s.hard_core_mapping.cores if getattr(c, CORE_WEIGHT_BITS, None) == 4
                and (np.asarray(c.get_core_matrix()) > q_max).any())
    setattr(core, CORE_WEIGHT_BITS, 2)
    _failed(P.run(bad, ir, decl), "H8", "cell_width")


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("semantics", ["streamed_lif", "windowed_lif"])
def test_a_relayed_program_is_established(count, semantics):
    program, ir, decl = UV.relays(count, semantics)
    assert any(getattr(n, "_depth_relay", False) for n in ir.get_neural_cores())
    report = P.run(program, ir, decl)
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())


def test_a_relayed_ir_compares_with_the_relayed_ir_at_unbounded_geometry():
    program, ir, decl = UV.relays(2)
    report = P.run(program, ir, decl, unbounded_ir=UV.relay_ir(unbounded=True))
    tiling = report.results["ir-tiling"]
    assert tiling.status == ESTABLISHED, [v.text() for v in tiling.violations]
    assert report.reference == "the IR at unbounded geometry"


def _relay(ir):
    return next(n for n in ir.get_neural_cores() if getattr(n, "_depth_relay", False))


def test_a_relay_whose_weight_the_tiling_changed():
    tiled, unbounded = UV.relay_ir(), UV.relay_ir(unbounded=True)
    relay = _relay(tiled)
    matrix = np.array(relay.get_core_matrix(tiled), dtype=np.float64)
    matrix[0, 0] += 1.0
    relay.core_matrix = matrix
    result = ConditionResult("ir-tiling", ESTABLISHED)
    check_ir_tiling(tiled, unbounded, declaration=P.decl("streamed_lif"), result=result)
    assert "fan_in" in {v.rule for v in result.violations}, [v.text() for v in result.violations]


def test_a_relay_relaying_another_neuron():
    """A relay column re-pointed at another source: it is keyed by the neuron it relays, so it holds another relay
    neuron than any of the reference's."""
    tiled, unbounded = UV.relay_ir(), UV.relay_ir(unbounded=True)
    relay = _relay(tiled)
    sources = np.asarray(relay.input_sources, dtype=object).copy()
    first = sources.flat[0]
    sources.flat[0] = type(first)(first.node_id, first.index + 1)
    relay.input_sources = sources
    result = ConditionResult("ir-tiling", ESTABLISHED)
    check_ir_tiling(tiled, unbounded, declaration=P.decl("streamed_lif"), result=result)
    assert {"missing_neuron", "extra_neuron", "duplicate_neuron", "fan_in"} & {v.rule for v in result.violations}, (
        [v.text() for v in result.violations])


def test_a_columnar_program_is_established_and_its_columns_are_read():
    program, ir, decl = UV.columnar()
    columnar = [c for s in P.neural(program) for c in s.hard_core_mapping.cores
                if isinstance(c.axon_sources, SpikeSourceColumns)]
    assert columnar
    assert P.run(program, ir, decl).established("theorem1")
    bad, _ir, _decl = UV.columnar()   # a fresh build: a deep copy packs a core's columns into its pickle form
    core = next(c for s in P.neural(bad) for c in s.hard_core_mapping.cores
                if isinstance(c.axon_sources, SpikeSourceColumns) and (c.axon_sources.kinds() == 1).sum() >= 2)
    sources = core.axon_sources.copy()
    inputs = np.flatnonzero(sources.kinds() == 1)
    sources.neuron[inputs[0]], sources.neuron[inputs[1]] = sources.neuron[inputs[1]], sources.neuron[inputs[0]]
    core.axon_sources = sources
    core._axon_source_spans = None
    _failed(P.run(bad, ir, decl), "tiles", "fan_in")
