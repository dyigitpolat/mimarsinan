"""[U-F4] W15's static check establishes Theorem 1's program conditions on pipelined programs, H10 non-vacuous: each
wavefront group's head loads the layout it states (its dark cores loaded, holding no placement), and every later pass
runs wholly resident, each core on the head core of its stated ordinal, which H10 compares content for content."""

from __future__ import annotations

import copy
import functools

import numpy as np
import pytest
import torch

from mimarsinan.certification.program_check import CheckDeclaration
from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.mapping.support.wavefront_stages import SCHEDULE_CORE_ORDINALS, SCHEDULE_DARK_CORES, SCHEDULE_WAVEFRONT

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.mapping.pipelined_multiplexing import vehicles as V
from unit.mapping.position_multiplexing import vehicles as W02


@functools.lru_cache(maxsize=None)
def _block():
    return V.lane_program("block")


@functools.lru_cache(maxsize=None)
def _pipelined(count: int):
    return V.build(_block().ir, V.cores(count), True)


def _check(program, count: int):
    return P.run(program, _block().ir, UV.value_decl(V.cores(count)))


def _failed(report, rule: str) -> None:
    result = report.results["H10"]
    assert result.status == FAILED, (result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]


@pytest.mark.parametrize("count", [8, 3])
def test_a_pipelined_value_program_is_established_with_h10_on_its_wavefronts(count):
    report = _check(_pipelined(count), count)
    assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
    h10 = report.results["H10"]
    assert h10.status == ESTABLISHED and h10.checked > 0 and "wavefront group" in h10.note, h10.note


def _later(program):
    return next(s for s in V.neural(program) if getattr(s, SCHEDULE_CORE_ORDINALS, None))


def test_a_resident_core_holding_other_weights_than_its_ordinal_s_head_core():
    bad = P.clone(_pipelined(8))
    core = _later(bad).hard_core_mapping.cores[0]
    core.core_matrix = core.core_matrix + 1.0
    _failed(_check(bad, 8), "resident_content")


def test_an_ordinal_the_head_never_loaded():
    bad = copy.deepcopy(_pipelined(8))
    stage = _later(bad)
    ordinals = list(getattr(stage, SCHEDULE_CORE_ORDINALS))
    ordinals[0] = 10 ** 6
    setattr(stage, SCHEDULE_CORE_ORDINALS, tuple(ordinals))
    _failed(_check(bad, 8), "resident_content")


def test_a_head_stating_another_layout_than_it_loads():
    bad = copy.deepcopy(_pipelined(8))
    head = next(s for s in V.neural(bad) if getattr(s, SCHEDULE_WAVEFRONT, None) is not None)
    stated = getattr(head, SCHEDULE_WAVEFRONT)
    setattr(head, SCHEDULE_WAVEFRONT, type(stated)(levels=stated.levels, passes=stated.passes,
                                                   layout_cores=stated.layout_cores + 1))
    _failed(_check(bad, 8), "wavefront_layout")


def test_a_dark_core_holding_a_placement():
    bad = copy.deepcopy(_pipelined(8))
    head = next(s for s in V.neural(bad) if getattr(s, SCHEDULE_DARK_CORES, None))
    placed = next(i for i, ps in enumerate(head.hard_core_mapping.soft_core_placements_per_hard_core) if ps)
    setattr(head, SCHEDULE_DARK_CORES, frozenset(getattr(head, SCHEDULE_DARK_CORES)) | {placed})
    _failed(_check(bad, 8), "wavefront_dark_core")


def test_a_later_pass_that_reprograms_its_cores():
    bad = copy.deepcopy(_pipelined(8))
    _later(bad).schedule_weights_resident = False
    _failed(_check(bad, 8), "wavefront_layout")


def test_a_pipelined_streamed_lif_program_is_established():
    """The LIF chain at integral weights (so H8's integrality binds), on the cycle-accurate streamed discipline."""
    rep = W02.lif_conv_chain(seed=0, widths=(6, 6))
    for perceptron in rep.get_perceptrons():
        with torch.no_grad():
            perceptron.layer.weight.copy_(torch.round(perceptron.layer.weight * 5))
            if perceptron.layer.bias is not None:
                perceptron.layer.bias.fill_(1.0)
            perceptron.base_activation.activation_scale.fill_(6.0)
    ir = W02.lif_ir(rep, 64, 16)
    program = V.build(ir, V.LIF_CORES, True, allow_coalescing=False)
    assert any(getattr(s, SCHEDULE_CORE_ORDINALS, None) for s in V.neural(program))
    decl = CheckDeclaration(semantics="streamed_lif", window=W02.T, pass_transfer=VERBATIM,
                            core_types=tuple(dict(c) for c in V.LIF_CORES))
    report = P.run(program, ir, decl)
    for condition in ("R2", "H5-skew", "H1-latency", "R3", "H9", "R6", "H10", "tiles"):
        assert report.results[condition].status == ESTABLISHED, (condition, report.results[condition].note,
                                                                  report.failures_by_condition())
    assert np.isfinite(report.results["H10"].checked) and report.results["H10"].checked > 0
