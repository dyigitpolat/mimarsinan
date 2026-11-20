"""[thru:w4] W15's static check on a program of input slots (the batch pipeline, PL/thru/design.md (c) 'Executor check',
(d)): the stage's slot field is registered with its readers, so a program of input slots is a known form. Every
condition of Theorem 1 is read on each input slot's projection (its stages in program order, as the reference executors
run them), so producer order and the reads are keyed by slot and node; H10 reads the whole program, a group's first
pass's slot stages one load (the first loads the layout, the others run resident on it). Two rules of H9 are the
slots' own: a stage reads only its own slot's values (``slot_scope``), and a stage's slot lies below B
(``slot_range``), each bitten by one seeded fault in a built program."""

from __future__ import annotations

import copy
import functools

import pytest
import torch

from mimarsinan.certification.program_check import CheckDeclaration
from mimarsinan.certification.program_check.forms import base
from mimarsinan.certification.program_check.report import ESTABLISHED, FAILED
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.mapping.support.wavefront_stages import (
    SCHEDULE_INPUT_SLOT, SCHEDULE_WAVEFRONT, InputSlot, input_slot, program_slots)

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.mapping.pipelined_multiplexing import vehicles as V
from unit.mapping.pipelined_multiplexing.test_layer_copies_programs import LIF_COPY_CORES
from unit.mapping.position_multiplexing import vehicles as W02

COPIES = {"layer_replication": True}


def _slots(b: int) -> dict:
    return {"batch_pipeline_slots": b, "carry_input_slots": b}


@functools.lru_cache(maxsize=None)
def _block():
    return V.lane_program("block")


@functools.lru_cache(maxsize=None)
def _built(count: int, copies: bool, b: int):
    return V.build(_block().ir, V.cores(count), True, **dict(COPIES if copies else {}, **_slots(b)))


def _check(program, count: int):
    return P.run(program, _block().ir, UV.value_decl(V.cores(count)))


@functools.lru_cache(maxsize=None)
def _lif_chain():
    """The LIF chain at integral weights (so H8's integrality binds), as the copies' check reads it."""
    rep = W02.lif_conv_chain(seed=0, widths=(6, 6))
    for perceptron in rep.get_perceptrons():
        with torch.no_grad():
            perceptron.layer.weight.copy_(torch.round(perceptron.layer.weight * 5))
            if perceptron.layer.bias is not None:
                perceptron.layer.bias.fill_(1.0)
            perceptron.base_activation.activation_scale.fill_(6.0)
    return W02.lif_ir(rep, 64, 16)


def _failed(report, rule: str) -> None:
    result = report.results["H9"]
    assert result.status == FAILED, (result.status, result.note)
    assert rule in {v.rule for v in result.violations}, [v.text() for v in result.violations]
    assert report.theorem1.startswith("failed") and "H9" in report.theorem1, report.theorem1


class TestTheSlotField:
    def test_it_is_registered_with_its_readers(self):
        assert base.STAGE[SCHEDULE_INPUT_SLOT] == ("H9", "R3")

    def test_unregistered_the_known_check_fails_by_name(self, monkeypatch):
        monkeypatch.delitem(base.STAGE, SCHEDULE_INPUT_SLOT)
        report = _check(_built(8, True, 2), 8)
        assert report.theorem1.startswith("failed") and "stage.schedule_input_slot" in report.theorem1


class TestEstablished:
    @pytest.mark.parametrize("b", [2, 3, 5])
    @pytest.mark.parametrize("copies", [False, True])
    @pytest.mark.parametrize("count", [8, 3, 1])
    def test_a_value_program_of_input_slots(self, count, copies, b):
        report = _check(_built(count, copies, b), count)
        assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())
        assert report.counts["input_slots"] == b
        h10 = report.results["H10"]
        assert h10.status == ESTABLISHED and h10.checked > 0 and "wavefront group" in h10.note, h10.note
        assert report.results["H9"].checked > 0

    @pytest.mark.parametrize("semantics", ["streamed_lif", "synchronized_ttfs", "cascaded_ttfs"])
    @pytest.mark.parametrize("b", [2, 3])
    def test_a_lif_chain_of_input_slots(self, semantics, b):
        program = V.build(_lif_chain(), LIF_COPY_CORES, True, allow_coalescing=False, **dict(COPIES, **_slots(b)))
        assert program_slots(program.stages) == b
        decl = CheckDeclaration(semantics=semantics, window=W02.T,
                                pass_transfer=VERBATIM if semantics == "streamed_lif" else COLLAPSE,
                                core_types=tuple(dict(c) for c in LIF_COPY_CORES))
        report = P.run(program, _lif_chain(), decl)
        assert report.established("theorem1"), (report.theorem1, report.failures_by_condition())

    def test_the_check_leaves_the_program_byte_identical(self):
        import pickle

        program = _built(3, True, 2)
        before = pickle.dumps(program)
        _check(program, 3)
        assert pickle.dumps(program) == before


def _reads(stage):
    return {int(s.node_id) for s in stage.input_map}


def _outputs(stage):
    return {int(s.node_id) for s in stage.output_map}


# --- seeded faults (``test_mutants.MUTANTS``) -------------------------------------------------------------------------

def test_a_stage_reading_another_slot_s_value():
    """Input slot 1's reader moved before slot 1's producer of a node it reads, after slot 0's: a machine holding
    one value per node hands it slot 0's (counterexample 1 of the proof)."""
    bad = copy.deepcopy(_built(3, True, 2))
    stages = list(bad.stages)
    slot = {i: input_slot(s) for i, s in enumerate(stages)}
    for j, reader in enumerate(stages):
        if reader.kind != "neural" or slot[j].slot != 1:
            continue
        for node in sorted(_reads(reader) - {-2}):
            own = [i for i in range(j) if slot[i] is not None and slot[i].slot == 1 and node in _outputs(stages[i])]
            other = [i for i in range(j) if slot[i] is not None and slot[i].slot == 0
                     and node in _outputs(stages[i])]
            if own and other and other[0] < own[-1]:
                stages.insert(own[-1], stages.pop(j))
                bad.stages = stages
                _failed(_check(bad, 3), "slot_scope")
                return
    pytest.fail("no reader of slot 1 found to move")


def test_a_stage_whose_slot_is_not_below_b():
    bad = copy.deepcopy(_built(3, True, 2))
    stage = next(s for s in V.neural(bad) if input_slot(s).slot == 1)
    setattr(stage, SCHEDULE_INPUT_SLOT, InputSlot(2, 2))
    _failed(_check(bad, 3), "slot_range")


def test_a_stage_stating_another_count_of_slots():
    bad = copy.deepcopy(_built(3, True, 2))
    stage = next(s for s in V.neural(bad) if input_slot(s).slot == 1)
    setattr(stage, SCHEDULE_INPUT_SLOT, InputSlot(1, 3))
    _failed(_check(bad, 3), "slot_range")


def test_a_writer_relabelled_to_the_other_slot():
    """Producer order keyed by slot and node: slot 1's producer of a node relabelled to slot 0 leaves slot 0 two
    producers of it and slot 1's readers none."""
    bad = copy.deepcopy(_built(3, True, 2))
    stages = V.neural(bad)
    writer = next(s for s in stages if input_slot(s).slot == 1
                  and any(_outputs(s) & _reads(r) for r in stages if input_slot(r).slot == 1 and r is not s))
    setattr(writer, SCHEDULE_INPUT_SLOT, InputSlot(0, 2))
    report = _check(bad, 3)
    rules = {v.rule for v in report.results["H9"].violations}
    assert report.results["H9"].status == FAILED and {"produced_twice", "slot_scope"} <= rules, rules


def test_a_later_member_of_a_group_that_loads():
    """H10 reads the group by pass index: a slot stage of a later pass that is not resident on the layout."""
    bad = copy.deepcopy(_built(3, True, 2))
    heads = [i for i, s in enumerate(bad.stages) if getattr(s, SCHEDULE_WAVEFRONT, None) is not None]
    later = next(s for s in bad.stages[heads[0] + 1:] if s.kind == "neural" and input_slot(s).slot == 1)
    later.schedule_weights_resident = False
    report = _check(bad, 3)
    h10 = report.results["H10"]
    assert h10.status == FAILED and "wavefront_layout" in {v.rule for v in h10.violations}, h10.violations
