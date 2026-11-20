"""[U-TGAUGE6] The dead-read pass removes the cores it leaves unread (U-TGAUGE5's check, M-2), and the window rule's
minors m-1 to m-4.

Turning a dead in-segment read off can leave its producer with no reader. Under per-hop re-timing (the windowed LIF
semantics) such a core is a level stage without outputs, and the per-cycle executor failed on it
with an unnamed error. The one rule: the pass that turns dead in-segment reads off also removes every core it leaves
with no reader and no output source, with that core's own reads, to a fixed point, in the same pass, by the IR's
whole-node deletion (``IRGraph.remove_nodes``, the call IR pruning removes its dead cores with). A removed core is read
by nothing, so no remaining read changes and the relay pass, the latency engine and the assertion read one edge set.

The minors: the mapping step's own derivation refuses a gapped cascade with the relays declared off (m-1); the relay
key's registry relevance equals the one predicate over the semantics' own lists of modes and schedules (m-2); the
pretrained bridge reads the predicate (m-3); the mapping step's record states the dead reads turned off and the cores
removed only where there are any (m-4).
"""

from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np
import pytest
import torch

import unit.spiking.cascade_windows as cw

from mimarsinan.chip_simulation.activation_axes import LIF_VARIANTS, SPIKING_FAMILIES, TTFS_VARIANTS
from mimarsinan.chip_simulation.activation_semantics import fold_spiking_axes
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from spiking_points import point_of_mode
from mimarsinan.chip_simulation.spiking_semantics import ALL_SPIKING_MODES, TTFS_CYCLE_SCHEDULES
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency import cycle_windows
from mimarsinan.mapping.latency.cycle_windows import ir_window_gaps, silence_dead_segment_reads
from mimarsinan.mapping.latency.depth_balancing import (
    UnbalancedFanInError, balance_cycle_windows, find_intra_segment_depth_gaps)
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

N = 2
KEY = "lif_depth_balancing_relays"


def _core(node_id, name, sources, matrix, **groups):
    return NeuralCore(id=node_id, name=name, input_sources=np.asarray(sources, dtype=object),
                      core_matrix=np.asarray(matrix, dtype=np.float64), threshold=1.0, latency=None,
                      parameter_scale=torch.tensor(0.0), **groups)


def _reads(node_id):
    return [IRSource(node_id, i) for i in range(N)]


def _graph(nodes, outputs) -> IRGraph:
    return IRGraph(nodes=list(nodes), output_sources=np.asarray([s for o in outputs for s in _reads(o)], dtype=object))


EYE = np.eye(N)
LIVE = EYE * 0.4
DEAD = np.zeros((N, N))


def _dead_read(chain: int = 1) -> IRGraph:
    """A over the input; B over A; a chain Q1..Qk over B; C over A (live) and Qk (every row at zero); outputs B, C."""
    nodes = [_core(0, "A", [IRSource(-2, i) for i in range(N)], EYE), _core(1, "B", _reads(0), EYE)]
    for k in range(chain):
        nodes.append(_core(2 + k, f"Q{k + 1}", _reads(1 + k), EYE))
    last = 1 + chain
    nodes.append(_core(last + 1, "C", _reads(0) + _reads(last), np.vstack([LIVE, DEAD])))
    return _graph(nodes, [1, last + 1])


def _ids(graph) -> list:
    return [int(n.id) for n in graph.nodes]


def _unread(graph) -> list:
    read = {int(s.node_id) for n in graph.nodes for s in n.input_sources.flatten() if s.node_id >= 0}
    read |= {int(s.node_id) for s in graph.output_sources.flatten() if s.node_id >= 0}
    return sorted(set(_ids(graph)) - read)


# ---- M-2: the one rule ----------------------------------------------------------------------------------------------

def test_a_core_the_dead_reads_leave_unread_goes_with_its_own_reads():
    graph = _dead_read()
    done = silence_dead_segment_reads(graph)
    assert (done.reads, done.cores) == (N, 1)
    assert _ids(graph) == [0, 1, 3] and _unread(graph) == [] and graph.validate() == []
    c = graph.get_node_by_id(3)
    assert [s.node_id for s in c.input_sources.flatten()] == [0] * N + [-1] * N


@pytest.mark.parametrize("chain", [2, 3])
def test_a_chain_read_only_through_a_dead_read_goes_to_a_fixed_point(chain):
    graph = _dead_read(chain)
    done = silence_dead_segment_reads(graph)
    assert (done.reads, done.cores) == (N, chain)
    assert _ids(graph) == [0, 1, chain + 2] and _unread(graph) == []


def test_a_producer_still_read_elsewhere_stays():
    """Input columns at zero from a producer that another core reads live: the reads go off, no core goes."""
    a = _core(0, "A", [IRSource(-2, i) for i in range(N)], EYE)
    b = _core(1, "B", _reads(0), EYE)
    c = _core(2, "C", _reads(1) + _reads(0), np.vstack([LIVE, DEAD]))
    graph = _graph([a, b, c], [2])
    done = silence_dead_segment_reads(graph)
    assert (done.reads, done.cores) == (N, 0) and _ids(graph) == [0, 1, 2]


def test_an_output_source_is_never_removed():
    graph = _dead_read()
    graph.output_sources = np.asarray(list(graph.output_sources) + _reads(2), dtype=object)
    done = silence_dead_segment_reads(graph)
    assert (done.reads, done.cores) == (N, 0) and 2 in _ids(graph)


@pytest.mark.parametrize("attr", ["psum_group_id", "coalescing_group_id"])
def test_a_group_goes_whole_or_stays(attr):
    """One removal for the whole fixed point: a group (two partial cores and the core that sums them) with no reader
    outside it goes whole; a group one member of which another core still reads stays whole (never a split the IR's
    deletion would refuse)."""
    def build(extra_reader: bool) -> IRGraph:
        a = _core(0, "A", [IRSource(-2, i) for i in range(N)], EYE)
        b = _core(1, "B", _reads(0), EYE)
        q1 = _core(2, "Q1", _reads(1), EYE, **{attr: 7})
        q2 = _core(3, "Q2", _reads(1), EYE, **{attr: 7})
        q = _core(4, "Q", _reads(2) + _reads(3), np.vstack([EYE, EYE]), **{attr: 7})
        c = _core(5, "C", _reads(0) + _reads(4), np.vstack([LIVE, DEAD]))
        nodes, outputs = [a, b, q1, q2, q, c], [1, 5]
        if extra_reader:
            nodes.append(_core(6, "D", _reads(3), LIVE))
            outputs.append(6)
        return _graph(nodes, outputs)

    whole = build(False)
    assert silence_dead_segment_reads(whole) == (N, 3) and _ids(whole) == [0, 1, 5]
    kept = build(True)
    assert silence_dead_segment_reads(kept) == (N, 0) and _ids(kept) == [0, 1, 2, 3, 4, 5, 6]


def test_a_host_operation_is_never_removed():
    """Only cores go: a host operation the removal leaves unread stays (a host stage costs no core)."""
    a = _core(0, "A", [IRSource(-2, i) for i in range(N)], EYE)
    b = _core(1, "B", _reads(0), EYE)
    host = ComputeOp(id=2, name="host", input_sources=np.asarray(_reads(1), dtype=object), op_type="identity",
                     params={"module": torch.nn.Identity()})
    q = _core(3, "Q", _reads(2), EYE)
    c = _core(4, "C", _reads(3) + _reads(1), np.vstack([DEAD, LIVE]))
    graph = _graph([a, b, host, q, c], [4])
    done = silence_dead_segment_reads(graph)
    assert (done.reads, done.cores) == (N, 1) and _ids(graph) == [0, 1, 2, 4]


def test_a_program_without_dead_reads_is_left_byte_for_byte():
    """No read turned off, nothing removed: not even a core that nothing read before the pass (the rule's scope is
    the cores its own reads leave unread)."""
    graph = _dead_read()
    graph.get_node_by_id(3).core_matrix = np.vstack([LIVE, LIVE])
    graph.nodes.append(_core(4, "U", _reads(1), EYE))
    before = pickle.dumps(graph)
    assert silence_dead_segment_reads(graph) == (0, 0)
    assert pickle.dumps(graph) == before


def test_the_removal_is_one_call_of_the_irs_whole_node_deletion(monkeypatch):
    calls = []
    original = IRGraph.remove_nodes

    def recorded(self, node_ids):
        calls.append(sorted(int(n) for n in node_ids))
        return original(self, node_ids)

    monkeypatch.setattr(IRGraph, "remove_nodes", recorded)
    graph = _dead_read(3)
    silence_dead_segment_reads(graph)
    assert calls == [[2, 3, 4]]


@pytest.mark.parametrize("per_hop", [False, True])
def test_one_edge_set_for_the_relay_pass_the_latency_engine_and_the_assertion(per_hop):
    """After the pass every remaining read is one the three readers share: no read names a removed core, the relay
    pass and the assertion find no gap, and the latency engine runs on the same graph."""
    graph = _dead_read(2)
    before = {int(n.id): [(s.node_id, s.index) for s in n.input_sources.flatten()] for n in graph.nodes}
    balance_cycle_windows(graph, cascade=True, armed=False)
    assert graph.validate() == [] and _ids(graph) == [0, 1, 4]
    for node in graph.nodes:  # only the dead reads changed: no read of a remaining core moved
        want = [(-1, 0) if source in (2, 3) else (source, index) for source, index in before[int(node.id)]]
        assert [(s.node_id, s.index) for s in node.input_sources.flatten()] == want
    assert find_intra_segment_depth_gaps(graph) == []
    assert ir_window_gaps(graph, per_hop=per_hop) == []
    assert IRLatency(graph).calculate() >= 1


# ---- m-4: the mapping step's record ---------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["streamed", "synchronized"])
def test_the_mapping_record_states_the_dead_reads_only_where_there_are_any(variant):
    step = cw.scm_step(cw.plan_config("lif", variant, relays=None, steps=8))
    plan = DeploymentPlan.resolve(step.pipeline.config)
    _enabled, record = step._apply_depth_balancing_relays(_dead_read(2), plan)
    assert record == {"relay_cores_inserted": 0, "dead_reads_off": N, "unread_cores_removed": 2}
    clean = _dead_read()
    clean.get_node_by_id(3).core_matrix = np.vstack([LIVE, LIVE])
    _enabled, record = step._apply_depth_balancing_relays(clean, plan)
    assert record == {"relay_cores_inserted": 2}


@pytest.mark.parametrize("family,variant", [("ttfs", "analytical"), ("ttfs", "synchronized")])
def test_a_plan_that_is_no_per_cycle_cascade_keeps_its_dead_reads(family, variant):
    """The dependency-order executors hold no window: Soft Core Mapping's pass leaves their programs byte for byte,
    dead reads and their producers included, and records nothing of them."""
    step = cw.scm_step(cw.plan_config(family, variant, relays=None, steps=8))
    graph = _dead_read(2)
    before = pickle.dumps(graph)
    enabled, record = step._apply_depth_balancing_relays(graph, DeploymentPlan.resolve(step.pipeline.config))
    assert (enabled, record) == (False, {"relay_cores_inserted": 0})
    assert pickle.dumps(graph) == before


def test_the_record_fragment_keeps_its_bytes_without_dead_reads():
    entries = {}
    step = cw.scm_step(cw.plan_config("lif", "synchronized", relays=None, steps=8))
    step.add_entry = lambda key, value, strategy: entries.__setitem__(key, (value, strategy))
    graph = _dead_read()
    step._emit_deployment_record_scm(graph, {"relay_cores_inserted": 3}, 4)
    value, strategy = entries["deployment_record_scm"]
    assert list(value) == ["reuse_plan", "relay_cores_inserted", "ir_max_latency"] and strategy == "basic"
    step._emit_deployment_record_scm(graph, {"relay_cores_inserted": 0, "dead_reads_off": 2,
                                             "unread_cores_removed": 1}, 4)
    value, _ = entries["deployment_record_scm"]
    assert list(value) == ["reuse_plan", "relay_cores_inserted", "dead_reads_off", "unread_cores_removed",
                           "ir_max_latency"]
    assert (value["dead_reads_off"], value["unread_cores_removed"]) == (2, 1)


# ---- m-1: the mapping step's own derivation ------------------------------------------------------------------------

def _derive(relays):
    flow, shape = cw._converted("smallest", 1, None)
    torch.manual_seed(12)
    gauges = cw._gauges(flow, list(flow.get_perceptrons()), 1, torch.rand(16, *shape))
    cw.timing_activations(flow, gauges, S=11)
    step = cw.scm_step(cw.plan_config("ttfs", "cascaded", relays=relays, steps=11))
    with contextlib.redirect_stdout(io.StringIO()):
        return step._derive_ir(flow, IRMapping(q_max=127, firing_mode="TTFS", max_axons=2048, max_neurons=2048),
                               DeploymentPlan.resolve(step.pipeline.config), {"weight_bits": 8}, 8,
                               phase=lambda _name: contextlib.nullcontext(), report=False)


def test_the_mapping_steps_derivation_refuses_a_gapped_cascade_with_the_relays_declared_off():
    """The window assertion is the last call of ``_derive_ir`` (U-TGAUGE5's mutant Y22 survived: no test reached the
    call through the derivation)."""
    with pytest.raises(UnbalancedFanInError, match=r"^SoftCoreMappingStep: .*declared off"):
        _derive(False)


def test_the_mapping_steps_derivation_returns_the_window_pass_record():
    graph, record, _latency = _derive(None)
    assert record["relay_cores_inserted"] >= 1 and cw.relay_cores(graph) == record["relay_cores_inserted"]
    assert ir_window_gaps(graph) == []


# ---- m-2: the registry's relevance over the semantics' own lists ---------------------------------------------------

def _declared_points():
    for family in SPIKING_FAMILIES:
        for variant in {"lif": LIF_VARIANTS, "ttfs": TTFS_VARIANTS}[family]:
            yield {"spiking_family": family, "spiking_variant": variant}
    for mode in sorted(ALL_SPIKING_MODES):
        for schedule in sorted(TTFS_CYCLE_SCHEDULES):
            yield {"spiking_mode": mode, "ttfs_cycle_schedule": schedule}


def test_the_relay_key_is_relevant_exactly_where_the_one_predicate_holds():
    """Every declared point of the semantics' own lists (each family with its variants, each mode with each schedule)
    folds to its axes; the key's relevance equals ``runs_per_cycle_cascade`` of the policy its mode and schedule
    resolve to (a new per-cycle mode lands here without a list to extend)."""
    seen = 0
    for declared in _declared_points():
        cfg = {**declared, "weight_bits": 8}
        fold_spiking_axes(cfg)
        cascade = policy_for_spiking_mode(point_of_mode(cfg["spiking_mode"], cfg.get("ttfs_cycle_schedule"))).runs_per_cycle_cascade
        assert cascade is DeploymentPlan.resolve(dict(cfg)).mode_policy().runs_per_cycle_cascade, declared
        authored = {k: cfg[k] for k in ("spiking_family", "spiking_variant")}
        assert REGISTRY[KEY].relevant.evaluate({**authored, "core_semantics": "spiking"}) is cascade, declared
        seen += 1
    assert seen == sum(len(v) for v in (LIF_VARIANTS, TTFS_VARIANTS)) + len(ALL_SPIKING_MODES) * len(
        TTFS_CYCLE_SCHEDULES)


# ---- m-3: the pretrained bridge reads the predicate ----------------------------------------------------------------

@pytest.mark.parametrize("cascade", [True, False])
def test_the_pretrained_bridge_passes_the_one_predicate(cascade, monkeypatch):
    import mimarsinan.chip_simulation.spiking_mode_policy as policies
    import mimarsinan.mapping.latency.depth_balancing as balancing
    from mimarsinan.models import pretrained_bridge

    seen = []
    real_policy, real_balance = policies.policy_for_spiking_mode, balancing.balance_cycle_windows

    class Policy:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        runs_per_cycle_cascade = cascade

    monkeypatch.setattr(policies, "policy_for_spiking_mode", lambda *a, **k: Policy(real_policy(*a, **k)))

    def balance(ir, **kwargs):
        seen.append(kwargs["cascade"])
        return real_balance(ir, **kwargs)

    monkeypatch.setattr(balancing, "balance_cycle_windows", balance)
    model, shape = cw.make("single_entry", 1)
    n_out = int(model(torch.zeros(1, *shape)).shape[1])
    x = torch.rand(2, *shape, generator=torch.Generator().manual_seed(0))
    pretrained_bridge.deploy_and_eval(model, shape, n_out, x, torch.zeros(2, dtype=torch.long), simulation_length=4)
    assert seen == [cascade]


def test_the_window_module_exports_the_pass_record():
    assert "DeadReads" in cycle_windows.__all__
    assert cycle_windows.DeadReads(0, 0).record() == {}
    assert cycle_windows.DeadReads(3, 1).record() == {"dead_reads_off": 3, "unread_cores_removed": 1}
