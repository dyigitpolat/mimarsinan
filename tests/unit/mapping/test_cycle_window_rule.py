"""[U-TGAUGE5] The window rule of every per-cycle cascade: one predicate on the mode policy, read by both relay gates,
the recipe's arming, the registry's relevance and the hybrid flow's dispatch; one edge set; the assertion at mapping
over the executed units; the per-cycle executor's check and its reach; the refusal's causes; the multiplexing
legality of the declared schedules.

The design review's corrections 1 to 6 (FU/utgauge/utgauge5/design_review.md): the arming from the one predicate
(the recipes armed the relays by literals, so the cascaded timing schedule read its unequal-depth joins a step low,
and a refusal said "declared off" where nothing was declared), the re-timed read per hop (it returned no gaps unread),
the executor check's reach (the packed and the reference loop, never the synchronized count path), and the legal sets
of position and pipelined multiplexing (a synchronized timing declaration derived the cascaded recipe).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import unit.spiking.cascade_windows as cw

from mimarsinan.chip_simulation.activation_semantics import fold_spiking_axes
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_point import resolve_spiking_point
from spiking_points import declaring_fire_while_integrating, point_of_mode
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.latency.depth_balancing import UnbalancedFanInError, insert_depth_balancing_relays
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_identity_hybrid_mapping
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy

#: The six declared points of the spiking taxonomy, ``(family, variant)``.
POINTS = [("lif", "streamed"), ("lif", "synchronized"), ("ttfs", "analytical"), ("ttfs", "quantized"),
          ("ttfs", "synchronized"), ("ttfs", "cascaded")]
KEY = "lif_depth_balancing_relays"
T = 8
N = 2


def _config(family, variant) -> dict:
    cfg = {"spiking_family": family, "spiking_variant": variant, "weight_bits": 8}
    fold_spiking_axes(cfg)
    return cfg


# ---- one predicate --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family,variant", POINTS)
def test_the_recipe_arms_the_relays_exactly_where_the_flow_runs_the_per_cycle_executors(family, variant):
    """The hybrid flow runs the per-cycle executors (``_forward_rate``) for the ``count`` decode; the recipe of exactly
    those plans arms the depth-balancing relays (U-TGAUGE4's tree armed them for the LIF family only)."""
    cfg = _config(family, variant)
    point = resolve_spiking_point(declaring_fire_while_integrating(cfg))  # [composed head, rule 12]
    per_cycle = policy_for_spiking_mode(point).decode_mode() == "count"
    recipe = ConversionPolicy.derive(point)
    assert bool(recipe.knobs.get(KEY, False)) is per_cycle, (family, variant)


@pytest.mark.parametrize("family,variant", POINTS)
def test_one_predicate_on_the_mode_policy_equals_the_dispatch_and_the_registry_relevance(family, variant):
    cfg = _config(family, variant)
    plan = DeploymentPlan.resolve(dict(cfg))
    per_cycle = plan.mode_policy().runs_per_cycle_cascade
    assert per_cycle is (plan.mode_policy().decode_mode() == "count")
    assert per_cycle is ((family, variant) in (("lif", "streamed"), ("lif", "synchronized"), ("ttfs", "cascaded")))
    assert REGISTRY[KEY].relevant.evaluate({**cfg, "core_semantics": "spiking"}) is per_cycle


def test_a_value_domain_plan_runs_no_cascade_and_its_key_does_not_exist():
    cfg = {"core_semantics": "mvm", "weight_bits": 8}
    assert DeploymentPlan.resolve(dict(cfg)).mode_policy().runs_per_cycle_cascade is False
    assert REGISTRY[KEY].relevant.evaluate({"core_semantics": "mvm", "spiking_family": "lif"}) is False


def test_a_relay_scan_on_a_plan_that_is_no_cascade_leaves_the_graph_byte_for_byte():
    from mimarsinan.mapping.latency.depth_balancing import balance_cycle_windows

    graph = _join_graph(shallow_w=0.0)
    sources = [s for n in graph.nodes for s in n.input_sources.flatten()]
    assert balance_cycle_windows(graph, cascade=False, armed=True, thresholding_mode="<=") == 0
    assert [s for n in graph.nodes for s in n.input_sources.flatten()] == sources


# ---- the assertion at mapping, its causes and the executed units ----------------------------------------------------

def _core(node_id, name, sources, matrix):
    return NeuralCore(id=node_id, name=name, input_sources=np.asarray(sources, dtype=object),
                      core_matrix=np.asarray(matrix, dtype=np.float64), threshold=1.0, latency=None,
                      parameter_scale=torch.tensor(0.0))


def _join_graph(*, shallow_w=0.35) -> IRGraph:
    """A -> B1 -> B2 -> C and the direct edge A -> C: A is read at two depths."""
    eye = np.eye(N)
    a = _core(0, "A", [IRSource(-2, i) for i in range(N)], eye)
    b1 = _core(1, "B1", [IRSource(0, i) for i in range(N)], eye)
    b2 = _core(2, "B2", [IRSource(1, i) for i in range(N)], eye)
    c = _core(3, "C", [IRSource(2, i) for i in range(N)] + [IRSource(0, i) for i in range(N)],
              np.vstack([eye * 0.4, eye * shallow_w]))
    return IRGraph(nodes=[a, b1, b2, c], output_sources=np.asarray([IRSource(3, i) for i in range(N)], dtype=object))


def _scm(family, variant, relays):
    return cw.scm_step(cw.plan_config(family, variant, relays=relays, steps=T))


@pytest.mark.parametrize("relays,cause", [(False, "declared off"), (None, "no mapping armed")],
                         ids=["declared_off", "not_armed"])
def test_the_mapping_names_the_cause_of_a_gap(relays, cause):
    """Declared off only where the key is declared false; else not armed (no recipe folded the key, as a config
    folded without its recipe holds it)."""
    step = _scm("ttfs", "cascaded", relays)
    if relays is None:
        step.pipeline.config.pop(KEY)
    graph = _join_graph()
    assert cw.scm_before_quantization(step, graph) is False
    with pytest.raises(UnbalancedFanInError, match=cause):
        cw.scm_on_final_ir(step, graph, False)


def test_the_relays_armed_by_the_recipe_leave_no_gap_and_a_plan_that_is_no_cascade_is_not_read():
    step = _scm("ttfs", "cascaded", True)
    graph = _join_graph()
    assert cw.scm_before_quantization(step, graph) is True
    cw.scm_on_final_ir(step, graph, True)
    assert sum(bool(getattr(n, "_depth_relay", False)) for n in graph.nodes) == 2
    analytical = _scm("ttfs", "analytical", True)
    gapped = _join_graph()
    assert cw.scm_before_quantization(analytical, gapped) is False
    cw.scm_on_final_ir(analytical, gapped, False)  # the dependency-order executors hold no window
    assert len(gapped.nodes) == 4


def test_a_re_timed_program_is_read_per_hop():
    """Under per-hop re-timing each depth level is its own executed unit, so a join across levels holds no in-unit
    edge: the assertion reads the units the hybrid build executes (it returned no gaps unread)."""
    from mimarsinan.mapping.latency.cycle_windows import ir_window_gaps

    graph = _join_graph()
    # the alignment raises A to one below C (its deepest reader), so B1 reads A a cycle ahead of its own window
    assert [tuple(g) for g in ir_window_gaps(graph)] == [(0, 1, -1)]
    assert ir_window_gaps(graph, per_hop=True) == []
    step = _scm("lif", "synchronized", False)
    assert step.pipeline.config["lif_per_hop_retiming"] is True
    cw.scm_before_quantization(step, graph)
    cw.scm_on_final_ir(step, graph, False)


# ---- the per-cycle executor's check and its reach ---------------------------------------------------------------------

def _flow(graph, **kwargs):
    return SpikingHybridCoreFlow(input_shape=(N,), hybrid_mapping=build_identity_hybrid_mapping(ir_graph=graph),
                                 simulation_length=T, preprocessor=nn.Identity(), firing_mode="Default",
                                 spike_mode="Uniform", thresholding_mode="<=", spiking_mode="lif",
                                 cycle_accurate_lif_forward=True, **kwargs).eval()


def test_the_executor_refuses_a_gapped_stage_on_the_packed_and_on_the_reference_path():
    x = torch.full((1, N), 0.5)
    with pytest.raises(UnbalancedFanInError, match="no mapping armed"):
        with torch.no_grad():
            _flow(_join_graph())(x)
    with pytest.raises(UnbalancedFanInError, match="no mapping armed"):
        with torch.no_grad():
            _flow(_join_graph()).forward_with_recording(x)
    relayed = _join_graph()
    insert_depth_balancing_relays(relayed, thresholding_mode="<=")
    with torch.no_grad():
        _flow(relayed)(x)


def test_a_pass_reading_a_carried_producer_at_a_delivery_delay_is_refused():
    """A pass of a fused segment reads an earlier pass's producer as a carried slice delayed ``d = g - 1`` cycles,
    so a nonzero delay on a live read is a read at a gap other than one, which no in-pass edge shows: the relayed skew
    vehicle cut at C 3 runs (every delay zero); the same program with one carried slice delayed a cycle (a program
    built outside the mapping) is refused by name, never read a cycle off."""
    from mimarsinan.mapping.packing.pass_timing import carried_nodes

    from unit.mapping.residency.executed import runs as R
    from unit.mapping.residency.executed import vehicles as V

    program = V.build(V.ir_of(V.branchy_repr("skew3"), relays=True), 3, None)
    x = torch.rand(6, V.IN, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        R.flow(program, **R.STREAMED_LIF)(x)
    stage = next(s for s in program.stages if s.kind == "neural" and carried_nodes(program.stages, s))
    carried = carried_nodes(program.stages, stage)
    k = next(k for k, s in enumerate(stage.input_map) if int(s.node_id) in carried)
    stage.replay_delays = {k: -1}
    R.assert_refused_at_the_executor(program, R.STREAMED_LIF)


def test_the_synchronized_count_path_reads_whole_windows_and_runs_no_check():
    """The synchronized LIF twin reads each core's window count in dependency order (no window to misread), so it
    returns before the per-cycle executors and their check."""
    with torch.no_grad():
        out = _flow(_join_graph(), lif_execution_synchronized=True)(torch.full((1, N), 0.5))
    assert out.shape == (1, N)


# ---- the multiplexing legality of the declared schedules ---------------------------------------------------------------

@pytest.mark.parametrize("key", ["position_multiplexing", "pipelined_multiplexing"])
@pytest.mark.parametrize("variant,legal", [("synchronized", (False, True)), ("cascaded", (False,))])
def test_a_timing_declaration_s_multiplexing_legal_set_follows_its_schedule(key, variant, legal):
    """The shape-only plane cannot see relay cores, so a cascaded declaration (its recipe arms the relays) keeps
    neither multiplexing; a synchronized one keeps both. The legality derived the recipe with the schedule the
    variant declares (a variant-only document read the cascaded default)."""
    doc = {"spiking_family": "ttfs", "spiking_variant": variant, "position_multiplexing": True,
           "allow_scheduling": True}
    assert tuple(REGISTRY[key].legal_values(doc)) == legal


# ---- a deploy path outside Soft Core Mapping -------------------------------------------------------------------------

def test_the_pretrained_bridge_deploys_a_dense_join_through_the_window_pass(monkeypatch):
    """The pretrained bridge maps its LIF program outside Soft Core Mapping; it runs the mapping's window pass, so a
    converter-built dense join deploys with relay cores and runs (it deployed relay-free, its shallow read a step
    low, and the per-cycle executor would refuse that program by name)."""
    import mimarsinan.mapping.packing.hybrid_hardcore_mapping as hybrid_module
    from mimarsinan.models import pretrained_bridge

    captured = {}
    original = hybrid_module.build_hybrid_hard_core_mapping

    def capture(*args, **kwargs):
        captured["ir"] = kwargs["ir_graph"]
        return original(*args, **kwargs)

    monkeypatch.setattr(hybrid_module, "build_hybrid_hard_core_mapping", capture)
    model, shape = cw.make("fanout_entry", 1)
    n_out = int(model(torch.zeros(1, *shape)).shape[1])
    x = torch.rand(4, *shape, generator=torch.Generator().manual_seed(0))
    result = pretrained_bridge.deploy_and_eval(model, shape, n_out, x, torch.zeros(4, dtype=torch.long),
                                               simulation_length=T)
    assert sum(bool(getattr(n, "_depth_relay", False)) for n in captured["ir"].nodes) >= 1
    assert torch.isfinite(result.logits).all()


# ---- the early notice and the causes ---------------------------------------------------------------------------------

@pytest.mark.parametrize("schedule,fires", [("cascaded", True), ("synchronized", False)])
def test_the_fan_in_notice_reads_the_predicate(schedule, fires):
    """The mapper-graph notice of an unequal-depth join with the relays declared off reads the one predicate: the
    cascaded timing schedule is a per-cycle cascade (it read the LIF family only), the synchronized one is not."""
    from mimarsinan.advisories.engine import evaluate_graph_advisories

    from unit.advisories.test_rules_graph import _unbalanced_join_repr

    config = {"spiking_mode": "ttfs_cycle_based", "ttfs_cycle_schedule": schedule, "simulation_steps": 4,
              "activation_quantization": True, KEY: False}
    ids = {a.id for a in evaluate_graph_advisories(_unbalanced_join_repr(), config)}
    assert ("ADV-FANIN-DEPTH-IMBALANCE" in ids) is fires


def test_the_three_causes_of_a_gap_at_mapping():
    from mimarsinan.mapping.latency.cycle_windows import DECLARED_OFF, NOT_ARMED, RELAYS_LEFT_GAP, mapping_cause

    assert mapping_cause(False, armed=False) == DECLARED_OFF
    assert mapping_cause(None, armed=False) == NOT_ARMED
    assert mapping_cause(True, armed=True) == RELAYS_LEFT_GAP
    assert len({DECLARED_OFF, NOT_ARMED, RELAYS_LEFT_GAP}) == 3
