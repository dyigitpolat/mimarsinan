"""[thru:w5] A pass outside every wavefront group under the batch pipeline (the conductor's ruling 'The Batch Pipeline
Is Exact and Verified' (a), PL/thru/w5.md): the group rule holds for it as for a group of one pass. It serves the run's
B inputs as slot stages under one load, input slot 0 to B-1 in consecutive passes (its one copy computes one instance per
pass, the group rule's own clause), the first with the pass's own load and every other resident on it, so a run loads
what one input's program loads. Its inputs' producers have ended for every slot: an earlier group's passes, or an
earlier pass of its own, which served every slot first. The pricer and the static check follow by their own rules (a
resident pass writes nothing, H10 reads it on its programming pass), so both planes count its weights once per run.

Vehicles: the lane-fused residual block (a W02 group of owned cores inside its wavefront segment at 1 core, and its
classifier, a segment of its own after the host pool) at 8, 3 and 1 cores, with and without copies, B 2, 3 and 5; and
thru:v2's hand-counted program (two 1x1 convolutions in one group with copies (3, 3) on 6 cores of 32 x 8, the
classifier a W02 group of one owned core), priced on both planes at B absent, 2 and 4 and K 5: per run 4, 6 and 11
passes, the group's layout 132 B and the classifier 48 B written once per run."""

from __future__ import annotations

import copy
import functools
import warnings

import pytest
import torch
import torch.nn as nn

from mimarsinan.certification.program_check.report import FAILED
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import StageOrderError
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.deployment_record.design_point.pricing import executed_record
from mimarsinan.deployment_record.objectives.context import candidate_context_from_platform
from mimarsinan.deployment_record.quantities.from_candidate import from_candidate
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.mapping.support.wavefront_stages import input_slot, stage_groups
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV
from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.lanes.lane_planes import STEPS, TRANSFER, ir_geometry, plane_platform
from unit.mapping.lanes.lane_programs import build_lane_program, lane_platform

from . import vehicles as V
from .test_planes import _ladder_noc

SLOTS = (2, 3, 5)
COPIES = {"layer_replication": True}


def _slots(b) -> dict:
    return {} if b is None else {"batch_pipeline_slots": b, "carry_input_slots": b}


@functools.lru_cache(maxsize=None)
def _block():
    return V.lane_program("block")


@functools.lru_cache(maxsize=None)
def _built(count: int, copies: bool, b):
    return V.build(_block().ir, V.cores(count), True, **dict(COPIES if copies else {}, **_slots(b)))


def _lone(program) -> list:
    """The neural stages outside every wavefront group, in program order."""
    groups = stage_groups(program.stages)
    return [s for i, s in enumerate(program.stages) if s.kind == "neural" and groups[i] is None]


def _nodes(stage) -> list:
    return [sorted((int(p["ir_node_id"]), int(p["axon_offset"]), int(p["neuron_offset"])) for p in ps)
            for ps in stage.hard_core_mapping.soft_core_placements_per_hard_core]


def _loads(program) -> int:
    return sum(1 for s in V.neural(program) if not s.schedule_weights_resident)


class TestTheBuiltProgram:
    @pytest.mark.parametrize("copies", [False, True], ids=["one_copy", "copies"])
    @pytest.mark.parametrize("count", [8, 3, 1])
    @pytest.mark.parametrize("b", SLOTS)
    def test_each_lone_pass_serves_the_slots_in_turn_its_first_stage_loading(self, count, copies, b):
        """Per pass of the one-input program outside every group, B consecutive passes of the slotted program, slot 0
        to B-1, each holding the pass's very soft cores: slot 0's keeps the pass's own load, the others run resident."""
        one, many = _lone(_built(count, copies, None)), _lone(_built(count, copies, b))
        assert one and len(many) == b * len(one)
        for q, stage in enumerate(one):
            turn = many[q * b:(q + 1) * b]
            assert [input_slot(s).slot for s in turn] == list(range(b))
            assert len({s.schedule_pass_index for s in turn}) == b          # one pass per slot: one copy, one input
            assert all(_nodes(s) == _nodes(stage) for s in turn)
            assert turn[0].schedule_weights_resident == stage.schedule_weights_resident
            assert all(s.schedule_weights_resident for s in turn[1:])

    @pytest.mark.parametrize("copies", [False, True], ids=["one_copy", "copies"])
    @pytest.mark.parametrize("count", [8, 3, 1])
    def test_a_run_loads_what_one_input_s_program_loads(self, count, copies):
        """No pass reloads per input: the loading stages of a run of B inputs are the one-input program's."""
        for b in SLOTS:
            assert _loads(_built(count, copies, b)) == _loads(_built(count, copies, None))

    def test_every_resident_lone_core_holds_its_loading_stage_s_weights(self):
        many = _lone(_built(1, True, 3))
        for q in range(0, len(many), 3):
            for stage in many[q + 1:q + 3]:
                for core, loaded in zip(stage.hard_core_mapping.cores, many[q].hard_core_mapping.cores):
                    assert torch.equal(torch.as_tensor(core.get_core_matrix()),
                                       torch.as_tensor(loaded.get_core_matrix()))


# --- thru:v2's hand-counted program ----------------------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(3, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.b = nn.Sequential(nn.Conv2d(8, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        return self.fc(torch.flatten(self.b(self.a(x)), 1))


def _model():
    torch.manual_seed(31)
    net = _Net()
    g = torch.Generator().manual_seed(32)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(m.num_features, generator=g) + 0.5)
    return net.eval()


def _platform(b):
    platform = plane_platform(("L", "E"), axons=32, neurons=8, count=6, passes=1 << 16, coalescing=False,
                              hardware_bias=True, multiplexing=True)
    platform.update(pipelined_multiplexing=True, layer_replication=True, wavefront_group_cores=2,
                    inference_batch_K=5, core_semantics="mvm", activation_bits=4, **_slots(b))
    return platform


@functools.lru_cache(maxsize=None)
def _hand(b):
    """(platform, lane program, built program) of thru:v2's hand-counted program at B (None: absent)."""
    platform = _platform(b)
    geo = ir_geometry(platform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = build_lane_program(_model(), (3, 2, 2), num_classes=3,
                                     platform=lane_platform(("L", "E"), cores=[dict(c) for c in platform["cores"]]),
                                     max_axons=geo["max_axons"], max_neurons=geo["max_neurons"],
                                     hardware_bias=geo["hardware_bias"], allow_coalescing=geo["allow_coalescing"])
        return platform, program, build_hybrid_mapping_for_pipeline(program.ir, platform)


#: Per B, per inference at K 5 (rounds 5, 3 and 2): passes and barriers per run (4 and 3, 6 and 5, 11 and 10) times the
#: rounds over K, the write law's bytes of a period (the group's 132 B and the classifier's 48 B, each once), and the
#: bytes and passes that reprogram per run (the same two loads) times the rounds over K.
HAND = {None: dict(pass_count=4.0, sync_count=3.0, period_rewrite_bytes=180.0, reprogrammed_bytes=180.0,
                   reprogram_passes=2.0),
        2: dict(pass_count=3.6, sync_count=3.0, period_rewrite_bytes=180.0, reprogrammed_bytes=108.0,
                reprogram_passes=1.2),
        4: dict(pass_count=4.4, sync_count=4.0, period_rewrite_bytes=180.0, reprogrammed_bytes=72.0,
                reprogram_passes=0.8)}


def _planes(b):
    platform, program, hybrid = _hand(b)
    record = from_record(executed_record(hybrid, platform, weight_bits=4, label=f"thru:w5 hand {b}", timesteps=STEPS,
                                         pass_transfer=TRANSFER))
    scheduled, _specs = _ladder_noc(program, platform, 3)
    facts = scheduled.facts
    context = candidate_context_from_platform(platform, latency_steps=facts.latency_steps,
                                              programming=facts.programming, carry=facts.carry, passes=facts.passes)
    candidate = from_candidate(layout=None, chip_param_capacity=None, total_params=None, host_side_segment_count=None,
                               context=context)
    return {k: candidate.get(k).value for k in HAND[b]}, {k: record.get(k).value for k in HAND[b]}


class TestTheHandCountedProgram:
    @pytest.mark.parametrize("b", [2, 4])
    def test_its_classifier_serves_the_slots_in_turn_under_one_load(self, b):
        _platform_, _program, hybrid = _hand(b)
        lone = _lone(hybrid)
        assert [input_slot(s).slot for s in lone] == list(range(b))
        assert [s.schedule_weights_resident for s in lone] == [False] + [True] * (b - 1)
        assert len(V.neural(hybrid)) == {2: 8, 4: 16}[b] and _loads(hybrid) == 2   # the group's head, the classifier

    @pytest.mark.parametrize("b", [None, 2, 4])
    def test_both_planes_state_the_hand_counted_prices(self, b):
        candidate, record = _planes(b)
        assert candidate == pytest.approx(HAND[b]) and record == pytest.approx(HAND[b])

    @pytest.mark.parametrize("b", [2, 4])
    def test_every_input_s_outputs_equal_the_identity_alone_and_the_golden(self, b):
        _platform_, program, hybrid = _hand(b)
        identity = program.identity().hybrid_mapping
        for k in (1, b, b + 1, 2 * b + 1):
            x = integer_inputs(program, k, seed=200 + k)
            want = torch.cat([V.value(identity, x[i:i + 1]) for i in range(k)])
            assert_exact(V.value(hybrid, x), want)
            assert_exact(V.value(hybrid, x.flip(0)).flip(0), want)
            codes, steps = run_golden(hybrid, x)
            assert_exact(torch.as_tensor(codes * steps), want)

    @pytest.mark.parametrize("b", [2, 4])
    def test_the_check_establishes_theorem_1(self, b):
        platform, program, hybrid = _hand(b)
        report = P.run(hybrid, program.ir, UV.value_decl(platform["cores"]))
        assert report.established("theorem1"), report.failures_by_condition()


def _reads(stage):
    return {int(s.node_id) for s in stage.input_map}


def _outputs(stage):
    return {int(s.node_id) for s in stage.output_map}


def test_a_slot_stage_of_the_lone_pass_reading_another_slot_s_row():
    """Seeded fault: the classifier's slot-1 stage moved before slot 1's producer of a node it reads, after slot 0's
    (a machine keeping one value per node hands it slot 0's row): the check fails it by H9's slot_scope, and both
    reference executors refuse it by name."""
    platform, program, hybrid = _hand(2)
    bad = copy.deepcopy(hybrid)
    stages = list(bad.stages)
    j = stages.index(_lone(bad)[1])
    reader = stages[j]
    assert input_slot(reader).slot == 1
    for node in sorted(_reads(reader) - {-2}):
        own = [i for i in range(j) if input_slot(stages[i]) is not None and input_slot(stages[i]).slot == 1
               and node in _outputs(stages[i])]
        other = [i for i in range(j) if input_slot(stages[i]) is not None and input_slot(stages[i]).slot == 0
                 and node in _outputs(stages[i])]
        if own and other and other[0] < own[-1]:
            stages.insert(own[-1], stages.pop(j))
            break
    else:
        pytest.fail("no node of the classifier's slot-1 stage whose slot-0 producer runs first")
    bad.stages = stages
    report = P.run(bad, program.ir, UV.value_decl(platform["cores"]))
    h9 = report.results["H9"]
    assert h9.status == FAILED and "slot_scope" in {v.rule for v in h9.violations}, h9.violations
    assert not report.established("theorem1")
    x = integer_inputs(program, 2, seed=7)
    with pytest.raises(StageOrderError, match=r"input slot 1"):
        V.value(bad, x)
    with pytest.raises(StageOrderError, match=r"input slot 1"):
        run_golden(bad, x)
