"""[thru:w4] The batch pipeline's randomized arm (PL/thru/design.md (d) 'The randomized test's extension'): it draws the
X3a network of its stream applied at P positions (every layer a bank of P instances), a semantics where pipelined
multiplexing is legal, the cores as the arm ``on_scheduled`` draws them, both base forms on, layer_replication on or off,
wavefront_group_cores at all, 3/4 or 1/2 of the drawn cores, B absent, 2, 3 or 5 (the chip stating as many carry
slots) and K inputs from 1, B, B + 1 and 2B + 1; it compares every output bit of every input of the built program with
the identity program run on each input alone, and the same with the inputs in reverse order."""

from __future__ import annotations

from collections import Counter

import pytest

from mimarsinan.certification.randomized import batch_arm as BA
from mimarsinan.certification.randomized import program as RP
from mimarsinan.certification.randomized.arms import ARMS, ON_SCHEDULED
from mimarsinan.certification.randomized.cost_draws import legal
from mimarsinan.certification.randomized.generator import generate
from mimarsinan.certification.randomized.runner import EQUAL, REFUSED, instance_for

SEEDS = range(40)


class TestTheDraws:
    def test_the_arm_is_registered_on_its_own_stream(self):
        arm = ARMS[BA.ARM_ID]
        assert arm.stream == BA.ARM_ID and not arm.hypothesis
        assert set(arm.semantics) == {"value", "streamed_lif", "cascaded_ttfs", "synchronized_ttfs", "analytical_ttfs"}
        assert arm.declaration == ARMS[ON_SCHEDULED].declaration

    def test_every_draw_holds_the_base_forms_and_the_four_keys_legally(self):
        seen = Counter()
        for seed in SEEDS:
            _instance, d = BA.draw(seed)
            decl = d.declaration
            assert decl["position_multiplexing"] is True and decl["pipelined_multiplexing"] is True
            assert "residency_plan" not in decl and not legal(decl), legal(decl)
            cores = sum(int(c["count"]) for c in decl["cores"])
            assert all(1 <= int(c["count"]) <= 4 for c in decl["cores"])
            if d.replication:
                assert decl["layer_replication"] is True
                want = max(1, cores * {"all": 4, "3/4": 3, "1/2": 2}[d.limit] // 4)
                assert decl["wavefront_group_cores"] == want
                seen[d.limit] += 1
            else:
                assert "layer_replication" not in decl and "wavefront_group_cores" not in decl
            assert d.slots in (None, 2, 3, 5)
            b = d.slots or 1
            assert d.inputs in (1, b, b + 1, 2 * b + 1)
            assert (decl.get("batch_pipeline_slots"), decl.get("carry_input_slots")) == (d.slots, d.slots)
            assert 2 <= d.positions <= 6
            seen[("slots", d.slots)] += 1
            seen[("replication", d.replication)] += 1
        assert all(seen[("slots", b)] for b in (None, 2, 3, 5)) and seen[("replication", True)] and \
            seen[("replication", False)] and all(seen[limit] for limit in ("all", "3/4", "1/2")), seen

    def test_the_network_is_its_stream_s_x3a_draw(self):
        instance, _d = BA.draw(3)
        assert instance == instance_for(3, ARMS[BA.ARM_ID])

    def test_one_position_is_the_mapper_graph_as_it_was(self):
        instance = generate(11)
        one, plain = RP.mapper_graph(instance, positions=1), RP.mapper_graph(instance)
        ir_one = RP.integer_ir(one, max_axons=None, max_neurons=None, allow_coalescing=False, hardware_bias=True)
        ir_plain = RP.integer_ir(plain, max_axons=None, max_neurons=None, allow_coalescing=False, hardware_bias=True)
        assert [(n.id, tuple(int(s.node_id) for s in n.input_sources.flatten())) for n in ir_one.nodes] == \
            [(n.id, tuple(int(s.node_id) for s in n.input_sources.flatten())) for n in ir_plain.nodes]

    def test_positions_make_every_layer_a_bank_of_p_instances(self):
        instance = generate(11, semantics="value")
        ir = RP.integer_ir(RP.mapper_graph(instance, positions=3), max_axons=None, max_neurons=None,
                           allow_coalescing=False, hardware_bias=True)
        banks = Counter(int(n.weight_bank_id) for n in ir.get_neural_cores() if n.weight_bank_id is not None)
        assert banks and set(banks.values()) == {3}


class TestTheRuns:
    @pytest.fixture(scope="class")
    def verdicts(self):
        return [BA.run_instance(seed) for seed in range(16)]

    def test_no_output_bit_differs_and_refusals_are_named(self, verdicts):
        assert {v.outcome for v in verdicts} <= {EQUAL, REFUSED, BA.VACUOUS}, [
            (v.seed, v.outcome, v.detail) for v in verdicts if v.outcome not in (EQUAL, REFUSED, BA.VACUOUS)]

    def test_the_runs_reach_shared_passes_of_several_inputs_and_copies(self, verdicts):
        equal = [v for v in verdicts if v.outcome == EQUAL]
        assert any(v.slots and v.groups and v.shared_passes for v in equal), [v.to_dict() for v in equal]
        assert any(v.copies for v in equal)
        assert all(v.outputs_compared == 2 * v.inputs * v.output_width for v in equal)
