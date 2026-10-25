"""[fix_b] The ladder plans U1 on the builder's segments, so its program is the executed one.

The vehicle is the instance tests' residual net (``design_point/programs.py``): a stem, a basic block, and a basic
block with a projection shortcut, every ReLU, the residual add and the pool on the host. The shortcut's 1x1 conv reads
the block input, so its depth segment (``compute_segment_ids``: a neural core's segment rises across a host
dependency) is that of the block's first conv. In program order, though, it follows the host ReLU after that conv, so
the builder (``partition_ir_graph``: runs of neural cores between host ops) places it in the second conv's segment, and
its capacity split, which cuts a run at its specs' depth segments, runs the shortcut as a pass of its own. A shape-only
plane that groups the specs by depth segments packs the shortcut beside the first conv: a different program (CIFAR
ResNet-18 w1/4 on NeuRRAM at 1/8: 23 planned against 26 executed passes, one per projection shortcut). With the walk's
runs (``residency_rules.spec_runs``, carried on the ladder's census) the ladder plans the builder's passes, and its
period census states the executed program's writes.
"""

from __future__ import annotations

import collections
import contextlib
import io
import warnings
from typing import Any, Dict, List, Tuple

import pytest

import mimarsinan.deployment_record.introspection.scaling_ladder as ladder
from mimarsinan.deployment_record.design_point.timeline import program_writes, writes_summary
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.support.schedule.pass_cut import transfer_for
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.search.problems.joint.candidate_fragments import (
    StageSemantics, candidate_program_facts, with_stage_placements,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from unit.chip_simulation.value_run.integer_programs import residual_block_net
from unit.deployment_record.design_point import programs


def _platform(count: int, **kwargs: Any) -> Dict[str, Any]:
    """The instance tests' platform with the bias on an axon (``has_bias`` false), as the vehicle's integer program
    is built (``build_integer_program``: the IR's bias rides one axon), so both planes tile the same cores."""
    platform = programs.platform(count, **kwargs)
    return {**platform, "cores": [{**core, "has_bias": False} for core in platform["cores"]]}


def _facts(softcores, noc, platform):
    semantics = StageSemantics()
    noc = with_stage_placements(softcores, noc, semantics)
    return noc, candidate_program_facts(softcores, noc, semantics, timesteps=platform.get("simulation_steps"),
                                        weight_bits=platform.get("weight_bits"),
                                        pass_transfer=transfer_for(streamed=False))


@pytest.fixture(scope="module")
def network() -> ladder.LadderNetwork:
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = convert_torch_model(residual_block_net(), (3, 16, 16), 10, packaging=MVM_PACKAGING,
                                   strict=False).eval()
    return ladder.LadderNetwork(name="residual_block", mapper_repr=flow.get_mapper_repr(), resolution=(3, 16, 16))


def _executed_passes(count: int) -> Tuple[Any, List[Tuple[int, ...]]]:
    with contextlib.redirect_stdout(io.StringIO()):
        hybrid = build_hybrid_mapping_for_pipeline(programs.residual_program().ir, _platform(count))
    perceptron = {int(n.id): n.perceptron_index for n in programs.residual_program().ir.get_neural_cores()}
    rows = []
    for stage in hybrid.stages:
        if stage.kind != "neural":
            continue
        held = {perceptron[int(p["ir_node_id"])] for placements in
                stage.hard_core_mapping.soft_core_placements_per_hard_core for p in placements}
        rows.append(tuple(sorted(held)))
    return hybrid, rows


def _ladder_passes(network: ladder.LadderNetwork, census: ladder.ShapeCensus, count: int):
    scheduled = ladder.schedule_program(network, census, _platform(count), _facts)
    rows = [tuple(sorted({census.softcores[s].perceptron_index for s, _h in placements}))
            for placements in scheduled.noc.pass_placements]
    return scheduled, rows


class TestTheLadderPlansTheBuildersSegments:
    def test_the_census_carries_the_walks_runs(self, network):
        census = ladder.shape_census(network, _platform(8))
        assert census.runs is not None and len(census.runs) == len(census.softcores)
        assert census.runs == census.program.spec_runs(len(census.softcores))

    def test_the_vehicle_groups_the_shortcut_apart_on_the_two_segmentations(self, network):
        census = ladder.shape_census(network, _platform(8))
        by_depth: Dict[int, set] = collections.defaultdict(set)
        by_run: Dict[int, set] = collections.defaultdict(set)
        for spec, run in zip(census.softcores, census.runs):
            by_depth[int(spec.segment_id or 0)].add(spec.perceptron_index)
            by_run[int(run)].add(spec.perceptron_index)
        assert sorted(map(sorted, by_depth.values())) != sorted(map(sorted, by_run.values()))

    @pytest.mark.parametrize("count", [8, 64, 256])
    def test_every_planned_pass_holds_the_executed_passs_perceptrons(self, network, count):
        census = ladder.shape_census(network, _platform(count))
        _scheduled, planned = _ladder_passes(network, census, count)
        _hybrid, executed = _executed_passes(count)
        assert planned == executed

    @pytest.mark.parametrize("count", [8, 64])
    def test_the_period_census_states_the_executed_writes(self, network, count):
        platform = _platform(count, k=1)
        census = ladder.shape_census(network, platform)
        priced = ladder.price_program(network, census, platform, _facts)
        hybrid, executed = _executed_passes(count)
        summary = writes_summary(program_writes(hybrid, platform, weight_bits=8))
        assert priced.passes == len(executed)
        assert priced.written_passes == summary["written_passes"]
        assert priced.period_bytes == summary["payload_bytes_per_period"]
        assert priced.max_writes[0] == pytest.approx(summary["max_writes_per_cell_per_inference"])
