"""[U-HOST3] m-2 as a class, "a reader that does not apply its writers' rule": every reader of a sealed count applies
the writers' count rule and withholds by name.

The timer writes, per timed host operator, a wall (a finite number of seconds at or above zero, never a bool) and two
counts (its calls and the samples each call carried: positive ints, never a bool). The readers of those sealed fields
are swept (``TestTheReadersAreACensus``): ``stage_timing.host_wall`` (the one function every consumer reads the wall
through: the record's timing fields, its quantities, the objective, the measured plane's diagnostic, the ODIN step)
and ``deployment_record_walls.fold_compute_walls`` (the composed head's rule 7) (the fold of a run's timer rows onto the schedule). Each applies
the writers' rule: ``host_wall`` withholds the per-inference wall by name where a count is not the writers' (the raw
total too where a wall is not), and the fold refuses by name, as it refuses a missing samples count. The second
verify's b1 (six sealed counts) and c4 (a record read end to end) are here, and the verify's surviving wall mutants
x21 (zero calls counted), x22 (a partial count sums the counted ops), x23 (a float count admitted at the timer) and
x26 (the fold takes the largest call count) each have their case.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import pytest

from mimarsinan.chip_simulation.hybrid_run.stage_timing import HostWallCountError, StageTimer, host_wall
from mimarsinan.deployment_record.cost import DeploymentCostModel
from mimarsinan.deployment_record.objectives import OBJECTIVES, DeploymentRecordView
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import ComputeOpRecord
from mimarsinan.deployment_record.schema import record as R
from mimarsinan.pipelining.pipeline_steps.verification import deployment_record_walls as walls
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import fold_compute_walls

from unit.deployment_record.record_fixtures import make_full_record, with_partition

ROOT = Path(__file__).resolve().parents[3]
#: The sealed fields the timer writes per timed host operator.
SEALED = ("wall_s_total", "invocations", "samples_per_call")
#: Counts outside the writers' rule (a positive int, never a bool), with an id.
NOT_COUNTS = [(-4, "negative"), (0, "zero"), (2.7, "float"), (2.0, "integral_float"), ("4", "string"),
              (True, "boolean"), (None, "absent")]
#: Walls outside the writers' rule (a finite number of seconds at or above zero, never a bool).
NOT_WALLS = [(-0.1, "negative"), (float("nan"), "nan"), (float("inf"), "infinite"), ("0.8", "string"),
             (True, "boolean")]


def _row(name="op", wall=0.8, calls=2, per_call=4):
    return {"stage_index": 0, "name": name, "wall_s_total": wall, "invocations": calls, "samples_per_call": per_call}


class TestHostWallAppliesTheCountRule:
    @pytest.mark.parametrize("value", [v for v, _ in NOT_COUNTS], ids=[i for _, i in NOT_COUNTS])
    @pytest.mark.parametrize("field", ["invocations", "samples_per_call"])
    def test_a_count_outside_the_rule_withholds_the_per_inference_wall_by_name(self, field, value):
        """b1: a sealed count of -4, 2.7, '4' or True gave a per-inference wall at revision 2."""
        wall = host_wall([{**_row(name="op_bad"), field: value}])
        assert wall.per_inference_s is None and "op_bad" in (wall.refusal or "") and wall.raw_total_s == 0.8
        with pytest.raises(HostWallCountError, match="op_bad"):
            wall.required()

    def test_zero_calls_is_no_count(self):
        """x21: zero calls counted."""
        wall = host_wall([_row(name="op_zero", calls=0)])
        assert wall.per_inference_s is None and "op_zero" in (wall.refusal or "")

    def test_one_counted_op_beside_an_uncounted_one_withholds_the_whole_wall(self):
        """x22: a partial count summed the counted ops."""
        wall = host_wall([_row(name="op_counted", wall=1.0, calls=1, per_call=10),
                          {"name": "op_uncounted", "wall_s_total": 1.0, "invocations": 1}])
        assert wall.per_inference_s is None and "op_uncounted" in (wall.refusal or "")
        assert "op_counted" not in (wall.refusal or "") and wall.raw_total_s == 2.0

    def test_one_op_over_two_runs_counted_in_one_only_withholds_it(self):
        wall = host_wall([_row(name="op_twice"), {**_row(name="op_twice"), "samples_per_call": None}])
        assert wall.per_inference_s is None and "op_twice" in (wall.refusal or "")

    def test_counts_within_the_rule_give_each_ops_wall_over_its_own_inferences(self):
        rows = [_row(name="a", wall=1.0, calls=10, per_call=1), _row(name="b", wall=4.0, calls=4, per_call=1),
                _row(name="a", wall=1.0, calls=10, per_call=1)]
        assert math.isclose(host_wall(rows).per_inference_s, 2.0 / 20 + 4.0 / 4)


class TestHostWallAppliesTheWallRule:
    @pytest.mark.parametrize("value", [v for v, _ in NOT_WALLS], ids=[i for _, i in NOT_WALLS])
    def test_a_wall_outside_the_rule_withholds_the_raw_total_and_the_wall_by_name(self, value):
        wall = host_wall([_row(name="op_good"), _row(name="op_bad_wall", wall=value)])
        assert wall.per_inference_s is None and wall.raw_total_s is None
        assert "op_bad_wall" in (wall.refusal or "") and "wall_s_total" in (wall.refusal or "")
        with pytest.raises(HostWallCountError, match="op_bad_wall"):
            wall.required()

    def test_a_zero_wall_is_a_wall(self):
        assert host_wall([_row(wall=0.0)]).per_inference_s == 0.0


def _record_with(field, value):
    record = with_partition(make_full_record())
    stages = tuple(dataclasses.replace(s, wall_s_total=0.8, invocations=2, samples_per_call=4)
                   if isinstance(s, ComputeOpRecord) else s for s in record.schedule.stages)
    payload = dataclasses.replace(record, schedule=dataclasses.replace(record.schedule, stages=stages)).to_dict()
    for stage in payload["schedule"]["stages"]:
        if "samples_per_call" in stage:
            stage[field] = value
    return R.DeploymentRecord.from_dict(payload)


class TestARecordReadEndToEnd:
    @pytest.mark.parametrize("value", [-4, 2.5, True, "4"], ids=["negative", "float", "boolean", "string"])
    @pytest.mark.parametrize("field", ["invocations", "samples_per_call"])
    def test_a_hand_edited_sealed_count_never_prints_a_per_inference_wall(self, field, value):
        """c4: the record's host_ops_s quantity printed -0.1 s, 0.2 s or 0.4 s, provenance measured."""
        record = _record_with(field, value)
        assert not from_record(record).has("host_ops_s")
        with pytest.raises(ValueError, match="host_op_wall_s"):  # the objective is unavailable, named
            OBJECTIVES.get("host_op_wall_s").value(DeploymentRecordView(record=record))
        with pytest.raises(HostWallCountError):
            walls.host_ops_wall_s_per_pass(record.schedule)
        terms = {t.name: t for t in DeploymentCostModel().latency(record)}
        assert "host_wall_diagnostic_s" not in terms or terms["host_wall_diagnostic_s"].value is None

    def test_the_writers_counts_read_end_to_end(self):
        record = _record_with("samples_per_call", 4)
        assert from_record(record).get("host_ops_s").value == pytest.approx(0.8 / 8)


def _fold(rows):
    record = make_full_record()
    name = next(s.name for s in record.schedule.stages if isinstance(s, ComputeOpRecord))
    return name, (lambda: fold_compute_walls(record.schedule, [{**row, "name": name} for row in rows]))


class TestTheFoldAppliesTheWritersRule:
    @pytest.mark.parametrize("value", [v for v, _ in NOT_COUNTS], ids=[i for _, i in NOT_COUNTS])
    def test_a_call_count_outside_the_rule_refuses_by_name(self, value):
        name, fold = _fold([_row(calls=value)])
        with pytest.raises(HostWallCountError, match=name):
            fold()

    def test_a_row_without_its_call_count_refuses_by_name(self):
        row = _row()
        row.pop("invocations")
        name, fold = _fold([row])
        with pytest.raises(HostWallCountError, match=name):
            fold()

    @pytest.mark.parametrize("value", [v for v, _ in NOT_WALLS], ids=[i for _, i in NOT_WALLS])
    def test_a_wall_outside_the_rule_refuses_by_name(self, value):
        name, fold = _fold([_row(wall=value)])
        with pytest.raises(HostWallCountError, match=name):
            fold()

    def test_the_fold_sums_one_ops_calls_over_two_runs(self):
        """x26: the fold took the largest call count."""
        name, fold = _fold([_row(wall=1.0, calls=10, per_call=1), _row(wall=1.0, calls=4, per_call=1)])
        (stage,) = [s for s in fold().stages if isinstance(s, ComputeOpRecord) and s.name == name]
        assert stage.invocations == 14 and stage.wall_s_total == 2.0 and stage.samples_per_call == 1


class TestTheTimerAppliesTheRule:
    @pytest.mark.parametrize("value", [2.0, "25", -3, 0, True, None], ids=["float", "string", "negative", "zero",
                                                                          "boolean", "absent"])
    def test_a_samples_count_outside_the_rule_refuses_by_name(self, value):
        """x23: a float count admitted at the timer."""
        timer = StageTimer(clock=lambda: 0.0)
        with pytest.raises(HostWallCountError, match="op_x"):
            with timer.time_compute_stage(0, "op_x", value):
                pass

    def test_the_timers_rows_are_within_the_rule_and_fold(self):
        ticks = iter([0.0, 0.5, 1.0, 1.25])
        timer = StageTimer(clock=lambda: next(ticks))
        for _ in range(2):
            with timer.time_compute_stage(0, "op_x", 3):
                pass
        (row,) = timer.compute_stage_walls()
        assert row["invocations"] == 2 and row["samples_per_call"] == 3 and row["wall_s_total"] == 0.75
        assert host_wall([row]).per_inference_s == pytest.approx(0.75 / 6)


# ---------------------------------------------------------------------------------------------------- the sweep
#: The readers of the sealed wall and counts: (module under src/mimarsinan, function). Each applies the writers' rule
#: (the cases above); a new reader fails here until it is swept.
READERS = {
    ("chip_simulation/hybrid_run/stage_timing.py", "host_wall"),
    ("pipelining/pipeline_steps/verification/deployment_record_walls.py", "fold_compute_walls"),  # rule 7
}


def _reads(tree):
    """(function, field) for every read of a sealed field: an attribute load, a subscript or ``.get``/``getattr``/
    ``_field`` of the field's name as a constant (a dict literal's key, a keyword and a dataclass field are writes)."""
    found = set()

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            if isinstance(child, ast.Attribute) and child.attr in SEALED and isinstance(child.ctx, ast.Load):
                found.add((function, child.attr))
            if isinstance(child, ast.Subscript) and isinstance(child.slice, ast.Constant) and \
                    child.slice.value in SEALED and isinstance(child.ctx, ast.Load):
                found.add((function, child.slice.value))
            if isinstance(child, ast.Call):
                args = [a.value for a in child.args if isinstance(a, ast.Constant)]
                if any(a in SEALED for a in args):
                    found.add((function, next(a for a in args if a in SEALED)))
            visit(child, inner)

    visit(tree, "<module>")
    return found


class TestTheReadersAreACensus:
    def test_every_reader_of_a_sealed_wall_or_count_is_a_swept_site(self):
        sites = set()
        for top in ("src/mimarsinan", "scripts"):
            for path in sorted((ROOT / top).rglob("*.py")):
                rel = path.relative_to(ROOT / "src" / "mimarsinan").as_posix() if top != "scripts" else \
                    path.relative_to(ROOT).as_posix()
                if rel.startswith("deployment_record/schema/"):
                    continue  # the schema declares and serializes the fields
                for function, _field in _reads(ast.parse(path.read_text(encoding="utf-8"))):
                    sites.add((rel, function))
        assert sites == READERS, sorted(sites ^ READERS)
