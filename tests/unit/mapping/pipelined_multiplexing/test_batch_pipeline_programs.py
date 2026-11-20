"""[thru:w2] The batch pipeline's built programs (PL/thru/design.md (c) 'Schedule' and 'Executor check', (d), (e)): a
run of the program serves B inputs. A pass that serves several inputs is one stage per input slot, all with the pass's
index, each carrying its slot (``schedule_input_slot``); the first stage of a group's first pass loads the layout and
every other stage of the group runs resident on it. Both reference executors (the value executor and the int64 golden)
run a slotted program by one rule: a slot stage reads and writes only its slot's rows (input k of a batch is slot
k mod B), and a read of a row that no earlier stage of its slot wrote is refused by name. The other backends refuse a
slotted program by name. Every built program is exact at atol 0 against the identity program run on each input alone
and against the int64 golden, for B 2, 3 and 5 and K inputs not a multiple of B, every input's outputs unchanged when
the inputs come in reverse order: on the value executor (the lane-fused residual block at 8, 3 and 1 cores, with and
without layer copies) and on streamed LIF, rate and synchronized LIF and every TTFS variant on the LIF chains (the
coverage boundary of streamed LIF, PL/thru/v1.md: chains and equal-depth fan-in)."""

from __future__ import annotations

import copy

import pytest
import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import StageOrderError, run_hybrid_stages
from mimarsinan.chip_simulation.value_run.int64_golden import run_golden
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.wavefront_stages import (
    SCHEDULE_CORE_ORDINALS, SCHEDULE_INPUT_SLOT, InputSlot, input_slot, program_slots, refuse_wavefront_backend,
    stage_groups, wavefront_head)

from unit.chip_simulation.value_run.integer_programs import assert_exact, integer_inputs
from unit.mapping.position_multiplexing import vehicles as W02

from . import vehicles as V
from .test_layer_copies_programs import LIF_COPY_CORES

SLOTS = (2, 3, 5)
COPIES = {"layer_replication": True}


def _slots(b: int) -> dict:
    return {"batch_pipeline_slots": b, "carry_input_slots": b}


def _ks(b: int):
    return (1, b, b + 1, 2 * b + 1)


@pytest.fixture(scope="module")
def block():
    return V.lane_program("block")


@pytest.fixture(scope="module")
def built(block):
    """Per (cores, copies, B): the program; B None is the one-input program of the same declaration."""
    out = {}
    for count in (8, 3, 1):
        for copies in (False, True):
            for b in (None,) + SLOTS:
                kw = dict(COPIES if copies else {}, **({} if b is None else _slots(b)))
                out[(count, copies, b)] = V.build(block.ir, V.cores(count), True, **kw)
    return out


def _alone(identity, x) -> torch.Tensor:
    """The identity program run on each input alone."""
    return torch.cat([V.value(identity, x[k:k + 1]) for k in range(x.shape[0])])


def _passes(hybrid) -> int:
    return len({(s.schedule_segment_index, s.schedule_pass_index) for s in V.neural(hybrid)})


class TestTheSchedule:
    @pytest.mark.parametrize("copies", [False, True])
    @pytest.mark.parametrize("b", SLOTS)
    def test_every_stage_carries_its_slot_and_a_pass_of_several_inputs_is_one_stage_per_slot(self, built, copies, b):
        program = built[(8, copies, b)]
        assert program_slots(program.stages) == b
        by_pass = {}
        for stage in V.neural(program):
            slot = input_slot(stage)
            assert isinstance(slot, InputSlot) and slot.slots == b and 0 <= slot.slot < b
            by_pass.setdefault((stage.schedule_segment_index, stage.schedule_pass_index), []).append(slot.slot)
        assert all(slots == sorted(set(slots)) for slots in by_pass.values())   # one stage per slot, slots in order
        assert any(len(slots) > 1 for slots in by_pass.values())                # some pass serves several inputs
        assert {s for slots in by_pass.values() for s in slots} == set(range(b))

    @pytest.mark.parametrize("copies", [False, True])
    def test_the_first_stage_of_a_group_s_first_pass_loads_and_every_other_stage_runs_resident(self, built, copies):
        program = built[(8, copies, 3)]
        heads = V.heads(program)
        assert heads and all(input_slot(h).slot == 0 and not h.schedule_weights_resident for h in heads)
        groups = stage_groups(program.stages)
        for index, stage in enumerate(program.stages):
            if groups[index] is None or groups[index] == index:
                continue
            head = program.stages[groups[index]]
            assert stage.schedule_weights_resident and getattr(stage, SCHEDULE_CORE_ORDINALS, None) is not None
            assert stage.schedule_segment_index == head.schedule_segment_index
            assert head.schedule_pass_index <= stage.schedule_pass_index < (head.schedule_pass_index
                                                                            + wavefront_head(head).passes)

    def test_the_groups_are_read_by_pass_index(self, built):
        """Every stage of a group's passes (several per pass) is its group's: as many group passes as the head
        states, each pass index once per slot it serves."""
        program = built[(8, True, 2)]
        groups = stage_groups(program.stages)
        for h, head in enumerate(program.stages):
            if wavefront_head(head) is None:
                continue
            members = [s for i, s in enumerate(program.stages) if groups[i] == h]
            assert len({s.schedule_pass_index for s in members}) == wavefront_head(head).passes
            assert len(members) > wavefront_head(head).passes

    @pytest.mark.parametrize("count", [8, 3])
    def test_every_resident_core_runs_its_head_s_copy_core_by_core(self, built, count):
        checked = 0
        for stage, core, loaded in V.resident_pairs(built[(count, True, 3)]):
            assert (core.axons_per_core, core.neurons_per_core) == (loaded.axons_per_core, loaded.neurons_per_core)
            assert torch.equal(torch.as_tensor(core.get_core_matrix()), torch.as_tensor(loaded.get_core_matrix()))
            checked += 1
        assert checked > 20

    @pytest.mark.parametrize("copies", [False, True])
    def test_the_inputs_share_passes_so_a_run_of_b_takes_fewer_than_b_runs(self, built, copies):
        one = _passes(built[(8, copies, None)])
        for b in SLOTS:
            assert _passes(built[(8, copies, b)]) < b * one

    def test_without_the_key_no_stage_carries_a_slot(self, built):
        for count in (8, 3, 1):
            for copies in (False, True):
                program = built[(count, copies, None)]
                assert program_slots(program.stages) is None
                assert not any(hasattr(s, SCHEDULE_INPUT_SLOT) for s in program.stages)


class TestValueExactness:
    @pytest.mark.parametrize("copies", [False, True], ids=["one_copy", "copies"])
    @pytest.mark.parametrize("count", [8, 3, 1])
    @pytest.mark.parametrize("b", SLOTS)
    def test_every_input_s_outputs_equal_the_identity_alone_and_the_golden(self, block, built, count, copies, b):
        program = built[(count, copies, b)]
        identity = block.identity().hybrid_mapping
        for k in _ks(b):
            x = integer_inputs(block, k, seed=100 + k)
            want = _alone(identity, x)
            assert_exact(V.value(program, x), want)
            codes, steps = run_golden(program, x)
            assert_exact(torch.as_tensor(codes * steps), want)
            assert_exact(V.value(program, x.flip(0)).flip(0), want)            # inputs in reverse order
            codes, steps = run_golden(program, x.flip(0))
            assert_exact(torch.as_tensor(codes * steps).flip(0), want)

    def test_the_lane_free_ir_is_exact_too(self):
        program = W02.integer_resnet()
        x = integer_inputs(program, 7)
        cores = [{"max_axons": 160, "max_neurons": 32, "count": 64, "has_bias": True}]
        slotted = V.build(program.ir, cores, True, **COPIES, **_slots(3))
        want = _alone(program.identity().hybrid_mapping, x)
        assert_exact(V.value(slotted, x), want)
        codes, steps = run_golden(slotted, x)
        assert_exact(torch.as_tensor(codes * steps), want)

    def test_the_retained_state_holds_every_input_s_rows(self, block, built):
        """run_retaining_state's buffer holds, per node, every input's row as the one-input program's buffer does."""
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow

        x = integer_inputs(block, 7, seed=3).double()
        flow = ValueHybridCoreFlow(built[(3, True, 3)], dtype=torch.float64, device="cpu")
        out, state, _flat = flow.run_retaining_state(x)
        ref_out, ref_state, _ = ValueHybridCoreFlow(built[(3, True, None)], dtype=torch.float64,
                                                    device="cpu").run_retaining_state(x)
        assert_exact(out, ref_out)
        assert set(state) == set(ref_state)
        for node, value in ref_state.items():
            assert torch.equal(state[node], value), node


def _lif_chain(widths, **kw):
    ir = W02.lif_ir(W02.lif_conv_chain(widths=widths), 64, 16)
    return ir, V.build(ir, LIF_COPY_CORES, True, allow_coalescing=False, **kw)


@pytest.fixture(scope="module", params=[(6, 6), (6, 20)], ids=["chain", "split_neurons"])
def lif(request):
    ir = W02.lif_ir(W02.lif_conv_chain(widths=request.param), 64, 16)
    programs = {(copies, b): V.build(ir, LIF_COPY_CORES, True, allow_coalescing=False,
                                     **dict(COPIES if copies else {}, **_slots(b)))
                for copies in (False, True) for b in SLOTS}
    return ir, programs


def _lif_inputs(k, seed):
    return torch.rand(k, *W02.LIF_SHAPE, generator=torch.Generator().manual_seed(seed))


def _flow_alone(flow_of, identity, x):
    with torch.no_grad():
        return torch.cat([flow_of(identity)(x[k:k + 1]) for k in range(x.shape[0])])


def _check_flow(flow_of, ir, programs):
    identity = build_identity_hybrid_mapping(ir_graph=ir)
    for (copies, b), program in sorted(programs.items()):
        for k in _ks(b):
            x = _lif_inputs(k, seed=10 * b + k)
            want = _flow_alone(flow_of, identity, x)
            with torch.no_grad():
                got = flow_of(program)(x)
                back = flow_of(program)(x.flip(0)).flip(0)
            assert torch.equal(got, want), (copies, b, k)
            assert torch.equal(back, want), (copies, b, k)
            assert float(want.std()) > 0 or k == 1


class TestSpikingExactness:
    def test_streamed_lif(self, lif):
        _check_flow(W02.lif_flow, *lif)

    @pytest.mark.parametrize("synchronized", [False, True], ids=["rate", "synchronized_counts"])
    def test_lif_without_the_cycle_accurate_forward(self, lif, synchronized):
        from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

        def flow(hybrid):
            return SpikingHybridCoreFlow(W02.LIF_SHAPE, hybrid, simulation_length=W02.T, spiking_mode="lif",
                                         thresholding_mode="<=", lif_execution_synchronized=synchronized)

        _check_flow(flow, *lif)

    @pytest.mark.parametrize("variant", ["analytical", "quantized", "synchronized", "cascaded"])
    def test_every_ttfs_variant(self, lif, variant):
        from unit.mapping.position_multiplexing.test_programs import _ttfs_flow

        _check_flow(lambda hybrid: _ttfs_flow(hybrid, variant), *lif)


def _slot_stage(program, slot, *, after_pass=0):
    """The program's first neural stage of ``slot`` past pass ``after_pass`` that reads a node a neural stage wrote."""
    written = {int(s.node_id) for st in V.neural(program) for s in st.output_map}
    for index, stage in enumerate(program.stages):
        if (stage.kind == "neural" and input_slot(stage) is not None and input_slot(stage).slot == slot
                and stage.schedule_pass_index > after_pass and {int(s.node_id) for s in stage.input_map} & written):
            return index
    raise AssertionError("no such stage")


class TestTheSlotRule:
    """A read of a row that no earlier stage of its slot wrote is refused by name, by both reference executors."""

    def _faults(self, program):
        early = copy.deepcopy(program)       # a slot-1 stage moved before every other stage of the program
        index = _slot_stage(early, 1)
        stage = early.stages.pop(index)
        early.stages.insert(0, stage)
        relabelled = copy.deepcopy(program)  # a slot-1 stage relabelled as slot 0: slot 1's rows of its nodes unwritten
        index = _slot_stage(relabelled, 1, after_pass=0)
        writer = next(i for i in range(index) if relabelled.stages[i].kind == "neural"
                      and input_slot(relabelled.stages[i]).slot == 1
                      and {int(s.node_id) for s in relabelled.stages[i].output_map}
                      & {int(s.node_id) for s in relabelled.stages[index].input_map})
        setattr(relabelled.stages[writer], SCHEDULE_INPUT_SLOT, InputSlot(0, input_slot(relabelled.stages[writer]).slots))
        return {"early": early, "relabelled": relabelled}

    @pytest.mark.parametrize("fault", ["early", "relabelled"])
    def test_the_value_executor_refuses(self, block, built, fault):
        program = self._faults(built[(8, True, 2)])[fault]
        with pytest.raises(StageOrderError, match=r"input slot \d"):
            V.value(program, integer_inputs(block, 4))

    @pytest.mark.parametrize("fault", ["early", "relabelled"])
    def test_the_golden_refuses(self, block, built, fault):
        program = self._faults(built[(8, True, 2)])[fault]
        with pytest.raises(StageOrderError, match=r"input slot \d"):
            run_golden(program, integer_inputs(block, 4))

    def test_the_streamed_lif_executor_refuses(self, lif):
        ir, programs = lif
        program = self._faults(programs[(True, 2)])["relabelled"]
        with pytest.raises(StageOrderError, match=r"input slot \d"), torch.no_grad():
            W02.lif_flow(program)(_lif_inputs(3, 0))

    def test_a_run_naming_no_slot_refuses_a_slotted_program(self, built):
        """The shared stage loop: an executor that runs a slotted program without naming a slot is refused."""
        with pytest.raises(PositionMultiplexingError, match="input slot"):
            run_hybrid_stages(built[(8, False, 2)], {-2: torch.zeros(1, 1)}, on_neural=lambda *a: None,
                              on_compute=lambda *a: None)


class TestOtherBackends:
    @pytest.mark.parametrize("backend", ["nevresim", "sanafe", "lava"])
    def test_a_slotted_program_is_refused_by_name(self, built, backend):
        with pytest.raises(PositionMultiplexingError, match=f"{backend} backend.*input slots"):
            refuse_wavefront_backend(built[(8, False, 2)].stages, backend)

    def test_a_slotted_program_without_a_wavefront_head_too(self, built):
        stages = copy.deepcopy(built[(8, False, 2)].stages)
        for stage in stages:
            if wavefront_head(stage) is not None:
                delattr(stage, "schedule_wavefront")
        with pytest.raises(PositionMultiplexingError, match="input slots"):
            refuse_wavefront_backend(stages, "nevresim")

    def test_the_backends_replay_gate_refuses_it(self, built):
        from mimarsinan.models.spiking.hybrid.carry import require_backend_replay

        with pytest.raises(PositionMultiplexingError, match="input slots"):
            require_backend_replay(built[(8, False, 2)], "nevresim", transfer="collapse", spike_mode="Uniform",
                                   spiking_mode="lif")
