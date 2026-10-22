"""U1 exactness (plan §20 item 1, superseded by §21 item 1): the key-absent scheduler keeps its capacity split, and
every cut it makes runs equal to the identity run by pass projection (``test_pass_projection``).

The refusal predicate of the earlier rounds (a cut refused where its carried wire or a pass's own timing moved the
fused alignment) is removed: the timing is projected instead. What stays from those rounds: the relay option still
runs exact (it is a user declaration, never a remedy: relays change the declared function), a segment latency is
not a graph latency, a pruned input is no edge, the candidate route prices every cut, and a deep segment plans."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.support.schedule.pass_carry import carried_outputs_by_stage
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes
from mimarsinan.search.problems.joint.candidate_fragments import collect_candidate_noc

from . import runs as R
from . import vehicles as V


def _layers(ir, names):
    """Perceptron indices of the IR cores named by their position in program order (``names``: 1-based)."""
    cores = ir.get_neural_cores()
    return tuple(sorted(cores[i - 1].perceptron_index for i in names))


class TestTheRelayOption:
    """With ``lif_depth_balancing_relays`` (inserted before the IR latency pass, ``SoftCoreMappingStep``) L1's skip of
    skew3 becomes a two-relay chain; U1 deploys it at C 2 and C 3 with carried wires, equal to the identity run of the
    relayed program."""

    @pytest.mark.parametrize("count", [2, 3])
    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.WINDOWED_LIF], ids=["streamed_lif", "windowed_lif"])
    def test_skew3_with_relays_deploys_and_runs_equal_to_the_identity_run(self, count, semantics):
        ir = V.ir_of(V.branchy_repr("skew3"), relays=True)
        assert sum("_depth_relay_" in str(n.name) for n in ir.get_neural_cores()) == 2
        u1 = V.build(ir, count, None, retimed_level_stages=R.retimed(semantics))
        assert len(V.neural_passes(u1)) >= 2 and carried_outputs_by_stage(u1.stages)
        R.assert_runs_equal_to_the_identity_run(ir, count, semantics, mapping=u1)


class TestUOneKeepsTheCapacitySplit:
    """No replan: ``late_skip`` at C 3 is cut where the capacity split first overflows ([A, B, C] | [D, E], carrying B
    to D two levels on), and the chain at C 5 splits layer 2 mid-level; both run equal to the identity run."""

    def test_the_first_overflow_cut_is_planned_and_runs_exact(self):
        ir = V.ir_of(V.branchy_repr("late_skip"))
        assert V.built_pass_layers(V.build(ir, 3, None), ir) == [_layers(ir, [1, 2, 3]), _layers(ir, [4, 5])]
        for semantics in R.ALL_RUNS:
            R.assert_runs_equal_to_the_identity_run(ir, 3, semantics)

    def test_a_chain_is_cut_mid_level(self):
        chain = V.ir_of(V.chain_repr(seed=0))
        assert V.built_pass_layers(V.build(chain, 5, None), chain) == [(1, 2), (2, 3, 4), (5,)]


class TestASegmentLatencyIsNotAGraphLatency:
    """``moved_to_output`` (a short branch of host_0 joined at the output layer to a deep branch through host_1) holds
    the short branch and the output layer in one segment: the output layer reads it one level on within the segment,
    four levels on over the whole graph. The fused latencies are the segment's, so every cut runs exact."""

    @pytest.mark.parametrize("count", [3, 5])
    @pytest.mark.parametrize("semantics", R.ALL_RUNS, ids=R.RUN_IDS)
    def test_a_host_path_into_the_segment_moves_no_cut(self, count, semantics):
        ir = V.ir_of(V.branchy_repr("moved_to_output"))
        assert carried_outputs_by_stage(V.build(ir, count, None).stages)
        R.assert_runs_equal_to_the_identity_run(ir, count, semantics)


class TestAPrunedInputIsNoEdge:
    """The fused graph is read off the packed hard cores, so a skip whose rows packing compaction drops is no edge:
    skew3 with L4's rows from L1 pruned runs as a chain."""

    @staticmethod
    def _pruned_skew3():
        ir = V.ir_of(V.branchy_repr("skew3"))
        cores = ir.get_neural_cores()
        l1, l4 = cores[0], cores[3]
        rows = np.array([getattr(s, "node_id", None) == l1.id for s in l4.input_sources.flatten()])
        l4.core_matrix = np.array(l4.core_matrix)
        l4.core_matrix[rows, :] = 0.0
        l4.pruned_row_mask = rows.tolist()
        l4.pruned_col_mask = [False] * int(l4.core_matrix.shape[1])
        return ir

    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.CASCADED_TTFS], ids=["streamed_lif", "cascaded_ttfs"])
    def test_u1_cuts_it_and_runs_equal_to_the_identity_run(self, semantics):
        ir = self._pruned_skew3()
        u1 = V.build(ir, 2, None)
        assert len(V.neural_passes(u1)) >= 3
        R.assert_runs_equal_to_the_identity_run(ir, 2, semantics, mapping=u1)


class TestTheCandidateRoutePricesEveryCut:
    """The candidate plane plans the capacity split with the wire census and refuses nothing: a skewed cut is
    priced like any other (round 1 raised an unscoped refusal here and aborted the search)."""

    def test_skew3_at_c2_is_priced(self):
        specs, census, _program = V.layout_plane(V.branchy_repr("skew3"))
        priced = collect_candidate_noc(
            softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, 2)], census=census,
            pcfg={"cores": V.cores(2), "allow_scheduling": True, "max_schedule_passes": 8})
        assert len(priced.pass_placements) == 3


class TestADeepSegment:
    def test_a_segment_deeper_than_the_recursion_limit_is_planned(self):
        """A 1,500-level chain in one segment plans iteratively, one pass per 64 cores."""
        n = 1500
        specs = [LayoutSoftCoreSpec(input_count=9, output_count=8, latency_tag=i, segment_id=0) for i in range(n)]
        plan = plan_program_passes(specs, [LayoutHardCoreType(25, 8, 64)])
        assert plan.feasible and plan.total_pass_count == -(-n // 64)
