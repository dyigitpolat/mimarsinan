"""The executed passes carry exactly the plan's live set (acceptance 10), and every cut the key executes runs exact by
pass projection."""

from __future__ import annotations

from typing import Dict, Set

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.activation_semantics import (
    LIF_FAMILY,
    SYNCHRONIZED_VARIANT,
    resolve_activation_semantics,
)
from mimarsinan.deployment_record.build.from_mapping import schedule_record_from_mapping
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.schedule.pass_carry import (
    carried_outputs_by_stage,
    carried_wire_bytes,
    carried_wire_spans,
)
from mimarsinan.mapping.support.schedule.pass_cut import COLLAPSE, VERBATIM
from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

from . import runs as R
from . import templates as TV
from . import vehicles as V


def _ir_carried(hybrid, ir) -> Dict[int, Set[int]]:
    """Independently from the IR: per stage boundary, the producers read by a later pass of the segment."""
    passes = V.neural_passes(hybrid)
    pass_of: Dict[int, int] = {}
    for q, stage in enumerate(passes):
        for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core:
            for p in placements:
                pass_of[int(p["ir_node_id"])] = q
    last: Dict[int, int] = {}
    for core in ir.get_neural_cores():
        for src in np.asarray(core.input_sources, dtype=object).ravel():
            if isinstance(src, IRSource) and src.node_id >= 0 and src.node_id in pass_of:
                if pass_of[src.node_id] < pass_of[int(core.id)]:
                    last[src.node_id] = max(last.get(src.node_id, 0), pass_of[int(core.id)])
    return {b: {n for n, end in last.items() if pass_of[n] <= b < end} for b in range(len(passes) - 1)}


@pytest.mark.parametrize("count", [5, 6])
@pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
def test_per_cut_the_carried_wires_are_the_plans_live_set(count, plan):
    rep = V.chain_repr(seed=1)
    specs, census, program = V.layout_plane(rep)
    ir = V.ir_of(rep)
    hybrid = V.build(ir, count, plan)
    comp = V.shape_only(specs, census, program, count, plan)
    spans = carried_wire_spans(hybrid.stages)
    stage_index = [i for i, s in enumerate(hybrid.stages) if s.kind == "neural"]
    expected = _ir_carried(hybrid, ir)
    for b, (_, end) in enumerate(comp.facts.phases[:-1]):
        live = {(n, w) for n, w, start, stop in spans if start <= stage_index[b] < stop}
        assert {n for n, _ in live} == expected[b]                                 # producer by producer
        assert sum(w for _, w in live) == comp.units.live[end - 1]                 # G5a's live set at the cut
    for transfer in (VERBATIM, COLLAPSE):
        record = schedule_record_from_mapping(hybrid, weight_bits=8, params_reloaded=0, timesteps=V.T,
                                              pass_transfer=transfer).carry
        assert record.transfer == transfer
        assert record.carried_wires == len(spans)
        assert record.carried_bytes == sum(carried_wire_bytes(w, V.T, transfer) for _, w, _, _ in spans)


def _skip_ir(semantics=None):
    """enc -> L1 -> L2, and L3 reads concat(L1, L2): a skip whose consumer sits two levels after its producer.
    [U-TGAUGE5] For a run whose executor reads the window (``runs.per_cycle``) the IR carries the depth-balancing
    relays as Soft Core Mapping builds it."""
    torch.manual_seed(5)
    enc = V._lif(V.IN, V.IN)
    enc.is_encoding_layer = True
    enc.use_cycle_accurate_trains = True
    e = PerceptronMapper(InputMapper((V.IN,)), enc)
    l1 = PerceptronMapper(e, V._lif(8, V.IN))
    l2 = PerceptronMapper(l1, V._lif(8, 8))
    l3 = PerceptronMapper(ConcatMapper([l1, l2]), V._lif(8, 16))
    rep = ModelRepresentation(PerceptronMapper(l3, V._lif(4, 8)))
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    ir = IRMapping(q_max=127.0, firing_mode="Default", max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS).map(rep)
    if semantics is not None and R.per_cycle(semantics):
        from mimarsinan.mapping.latency.depth_balancing import balance_cycle_windows

        balance_cycle_windows(ir, cascade=True, armed=True, thresholding_mode="<=")
    return ir


class TestTheKeyExecutesASkewedCutExactly:
    """U1 pass projection retires the key's skew rule: the executed passes run at the fused latencies and a skewed
    carried wire is replayed at its delivery delay, so the skip at C 2 (L3 reads L1 two levels on) and the two-level
    skew of ``"skew3"`` at C 2 and C 3 build under the key and run equal to the identity run under every deployed run
    (closing round 2 refused them, and the count-domain synchronized twin of the unprojected cut differed on 85 of 96
    outputs)."""

    @pytest.mark.parametrize("plan", [{"step": "stateless"}, {"step": "floor"}])
    @pytest.mark.parametrize("semantics", R.ALL_RUNS, ids=R.RUN_IDS)
    def test_the_skip_cut_runs_equal_to_the_identity_run(self, plan, semantics):
        ir = _skip_ir(semantics)
        mapping = V.build(ir, 2, plan, retimed_level_stages=R.retimed(semantics))
        assert carried_outputs_by_stage(mapping.stages)
        R.assert_runs_equal_to_the_identity_run(ir, 2, semantics, mapping=mapping)

    def test_collapse_is_the_pass_transfer_of_a_synchronized_lif_run(self):
        config = TV.resolve("t0_01")                                         # the lifsync tier-0 template
        semantics = resolve_activation_semantics(config)
        assert (semantics.family, semantics.variant) == (LIF_FAMILY, SYNCHRONIZED_VARIANT)
        assert run_pass_transfer(config) == COLLAPSE

    @pytest.mark.parametrize("count", [2, 3])
    @pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
    def test_skew3_plans_alike_on_both_planes_and_runs_exact(self, count, plan):
        """[U-TGAUGE5] The planes plan the relay-free vehicle alike (the shape-only plane holds no relay cores); each
        per-cycle run reads the vehicle as the mapping builds it (its relays armed) and runs exact under the key."""
        rep = V.branchy_repr("skew3")
        specs, census, program = V.layout_plane(rep)
        ir = V.ir_of(rep)
        built = V.build(ir, count, plan)
        assert V.built_pass_layers(built, ir) == V.fragment_pass_layers(
            V.fragments(specs, census, count, plan, program=program), specs)
        assert [tuple(p) for p in V.shape_only(specs, census, program, count, plan).pass_members()]
        for semantics in (R.STREAMED_LIF, R.CASCADED_TTFS):
            mapped = R.mapped(rep, semantics)
            budget = R.planned_budget(mapped, count, plan)
            R.assert_runs_equal_to_the_identity_run(mapped, budget, semantics, mapping=V.build(mapped, budget, plan))


class TestAPassThatMixesCarriedRastersWithAHostInput:
    """The deep-mismatch vehicle at C 6: the stateless step puts a core reading host_0 into a pass that also reads
    carried rasters. Each slice is encoded by its kind (the raster replayed, the host value encoded as the fused stage
    encodes it), so the key's mixed-input rule is retired and the pass runs exact under every deployed run."""

    @pytest.mark.parametrize("plan", [{"step": "stateless"}, {"step": "floor"}])
    @pytest.mark.parametrize("semantics", R.ALL_RUNS, ids=R.RUN_IDS)
    def test_it_runs_equal_to_the_identity_run(self, plan, semantics):
        ir = V.ir_of(V.branchy_repr("deep_mismatch"))
        mapping = V.build(ir, 6, plan, retimed_level_stages=R.retimed(semantics))
        assert len(V.neural_passes(mapping)) > 3 and carried_outputs_by_stage(mapping.stages)
        R.assert_runs_equal_to_the_identity_run(ir, 6, semantics, mapping=mapping)


class TestTheWindowRule:
    """[U-TGAUGE5] This file's refusal test: the relay-free skip cut (L3 reads L1 two levels on) run by a per-cycle
    executor is refused by name, never read a step off."""

    @pytest.mark.parametrize("semantics", [R.STREAMED_LIF, R.CASCADED_TTFS], ids=["streamed_lif", "cascaded_ttfs"])
    def test_the_relay_free_skip_cut_is_refused_by_name(self, semantics):
        R.assert_refused_at_the_executor(V.build(_skip_ir(), 2, {"step": "floor"}), semantics)
