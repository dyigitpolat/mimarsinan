"""[U-HOST2] One test per refusal clause of the per-inference host wall (verify m-1), and the identity on non-host rows.

The code of revision 1 holds each clause (the verify's adversarial cases b2 to b6); its tests did not pin them, so a
mutant dropping a clause survived (the verify's b1, b3 to b7 and a5). Each test here is green on the rule and red on
its clause's mutant:
- two operators at different counts: each operator's wall over its own inferences, summed (never one global divisor);
- a zero count and a boolean count refused at the timer and at the fold;
- the timer's and the fold's two-count refusals;
- the ODIN FPGA step's refusal of a wall without its count;
- a chip row fed over the host's link seals no host identity (the identity is carried on host rows only).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mimarsinan.chip_simulation.hybrid_run.stage_timing import HostWallCountError, StageTimer, host_wall
from mimarsinan.deployment_record.cost import DeploymentCostModel
from mimarsinan.deployment_record.objectives import OBJECTIVES, DeploymentRecordView
from mimarsinan.deployment_record.platform_physics import available_profiles, get_platform_physics
from mimarsinan.deployment_record.platform_physics.hosts import store as ST
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    host_attach_block, host_description_block, host_descriptions,
)
from mimarsinan.deployment_record.platform_physics.resolve import merge_override_blocks, resolve_platform_physics
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import ComputeOpRecord
from mimarsinan.pipelining.pipeline_steps.verification import deployment_record_walls as walls
from mimarsinan.pipelining.pipeline_steps.verification.deployment_record_assembly import fold_compute_walls

from unit.deployment_record.record_fixtures import make_full_record

#: Two operators: (name, timed calls, samples per call, summed wall in seconds).
OPS = (("op_first", 1, 10, 1.0), ("op_second", 4, 1, 4.0))


def _two_op_record():
    """The fixture record with a second host ComputeOp, each op timed at its own count and folded."""
    record = make_full_record()
    first = next(s for s in record.schedule.stages if isinstance(s, ComputeOpRecord))
    last = max(s.stage_index for s in record.schedule.stages)
    stages = tuple(replace(s, name=OPS[0][0]) if s is first else s for s in record.schedule.stages) + (
        replace(first, name=OPS[1][0], stage_index=last + 1),)
    schedule = replace(record.schedule, stages=stages, compute_op_count=2)
    rows = [{"stage_index": i, "name": name, "wall_s_total": wall, "invocations": calls, "samples_per_call": per}
            for i, (name, calls, per, wall) in enumerate(OPS)]
    return replace(record, schedule=fold_compute_walls(schedule, rows))


#: The definition's value: each op's wall over the inferences its calls carried, summed over the ops.
PER_INFERENCE = sum(wall / (calls * per) for _name, calls, per, wall in OPS)
#: The value a single global divisor would give (sum of walls over sum of inferences): a different number.
GLOBAL = sum(op[3] for op in OPS) / sum(op[1] * op[2] for op in OPS)


class TestTwoOperatorsAtDifferentCounts:
    def test_each_consumer_reads_each_operators_wall_over_its_own_inferences(self):
        record = _two_op_record()
        assert PER_INFERENCE != pytest.approx(GLOBAL)
        assert host_wall(record.schedule.stages).per_inference_s == pytest.approx(PER_INFERENCE)
        assert walls.host_ops_wall_s_per_pass(record.schedule) == pytest.approx(PER_INFERENCE)
        assert from_record(record).get("host_ops_s").value == pytest.approx(PER_INFERENCE)
        assert OBJECTIVES.get("host_op_wall_s").value(DeploymentRecordView(record=record)) == pytest.approx(
            PER_INFERENCE)
        terms = {t.name: t for t in DeploymentCostModel().latency(record)}
        assert terms["host_wall_diagnostic_s"].value == pytest.approx(PER_INFERENCE)


def _fold_one(samples):
    record = make_full_record()
    name = next(s.name for s in record.schedule.stages if isinstance(s, ComputeOpRecord))
    return name, lambda: fold_compute_walls(record.schedule, [
        {"stage_index": 0, "name": name, "wall_s_total": 0.5, "invocations": 1, "samples_per_call": samples}])


class TestACountThatIsNoCount:
    @pytest.mark.parametrize("samples", [0, True], ids=["zero", "boolean"])
    def test_the_timer_refuses_it_by_name(self, samples):
        timer = StageTimer(clock=lambda: 0.0)
        with pytest.raises(HostWallCountError, match="op_x"):
            with timer.time_compute_stage(0, "op_x", samples):
                pass

    @pytest.mark.parametrize("samples", [0, True], ids=["zero", "boolean"])
    def test_the_fold_refuses_it_by_name(self, samples):
        name, fold = _fold_one(samples)
        with pytest.raises(HostWallCountError, match=name):
            fold()


class TestTwoCountsForOneOperator:
    def test_the_timer_refuses_one_stage_timed_at_two_counts(self):
        timer = StageTimer(clock=lambda: 0.0)
        with timer.time_compute_stage(0, "op_x", 2):
            pass
        with pytest.raises(HostWallCountError, match="op_x"):
            with timer.time_compute_stage(0, "op_x", 3):
                pass

    def test_the_fold_refuses_one_operator_folded_at_two_counts(self):
        record = make_full_record()
        name = next(s.name for s in record.schedule.stages if isinstance(s, ComputeOpRecord))
        rows = [{"stage_index": 0, "name": name, "wall_s_total": 1.0, "invocations": 1, "samples_per_call": 10},
                {"stage_index": 0, "name": name, "wall_s_total": 1.0, "invocations": 1, "samples_per_call": 1}]
        with pytest.raises(HostWallCountError, match=name):
            fold_compute_walls(record.schedule, rows)


class TestTheOdinStep:
    def test_a_wall_without_its_count_is_refused_by_name(self):
        """The ODIN FPGA step's fragment: a deployed run whose timed rows carry no samples per call."""
        from mimarsinan.pipelining.pipeline_steps.verification.odin.odin_fpga_simulation_step import _timing_fragment

        rows = [{"stage_index": 0, "name": "op_uncounted", "wall_s_total": 0.5, "invocations": 1}]
        runs = [SimpleNamespace(timings=(), compute_stage_walls=rows)]
        with pytest.raises(HostWallCountError, match="op_uncounted"):
            _timing_fragment(runs, {"programming_s": 0.0, "execution_s": 0.0})


class TestTheIdentityOnNonHostRows:
    def test_a_chip_row_fed_over_the_hosts_link_seals_no_host(self):
        """The host attach row names its host (so the store rule can judge it); the composed chip row is the
        chip's, and its sealed form carries no host identity."""
        name = host_descriptions()[0]
        chip = next(p for p in available_profiles() if get_platform_physics(p).backing_store is not None)
        blocks = {"host": host_description_block(name), "host_attach": host_attach_block(name),
                  "store": ST.store_block(ST.HOST, "a variant (test)")}
        physics = resolve_platform_physics(chip, merge_override_blocks(blocks))
        (key,) = host_attach_block(name)
        assert physics.constants[key].host == ""
        assert "host" not in physics.to_dict()["constants"][key]
