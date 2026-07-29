"""[W4b-2b] the constant analysis is PURE: no module-mode and no RNG side effects.

Constant propagation is the one analysis in the elimination framework that
EXECUTES the deployment's own modules. That makes two global side channels
reachable from a pass that is supposed to only read the program:

- the per-module ``training`` flag — ``nn.Module.train()``/``.eval()`` recurse
  into every descendant, so a root-level restore promotes a deliberately
  frozen child (a BatchNorm pinned to eval inside a hosted block) back to
  training just by running the analysis;
- the global torch RNG — a stochastic host op draws from it on every probe
  execution, so an analysis pass shifts every downstream random draw of a
  seeded, multi-seed experiment.

Both are pinned here as OBSERVABLE invariants of a full cascade with
``elimination_constant_folding="full"``, together with proof that the probe
really ran (the fold is still derived, the module really was executed).

The third pin is registry parity: ``elimination_constant_folding`` is a
user-facing axis, so it must be representable in the config-schema registry —
the configurability SSOT — in lockstep with the constant-policy constants.

The fourth pin closes the LOW design risk the W4b-2 verifier flagged: a
RECORDED ``CONST(c != 0)`` outranks the "eliminated => 0" wiring rule in
``source_constant``, so IF a folded line were orphan-killed while a live
reader still read it, the lattice would keep returning ``c`` while the
deployed program rewires that axon OFF and reads 0. It cannot happen —
``_orphan_neurons`` only kills a column once EVERY consumer axon is pruned —
and that is pinned directly, certificate-backed, rather than argued.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.config_schema.registry import REGISTRY, FieldType
from mimarsinan.mapping.ir import ComputeOp, IRSource, NeuralCore
from mimarsinan.mapping.pruning.certificate import certify_cascade_equivalence
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
)
from mimarsinan.mapping.pruning.liveness_transfer.constant_policy import (
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    ELIMINATION_CONSTANT_FOLDING_KEY,
    ELIMINATION_CONSTANT_FOLDING_MODES,
    resolve_elimination_constant_folding,
)

from unit.mapping.constant_vehicles import (
    bias_only_collapse_graph,
    residual_join_graph,
    sigmoid_chain_graph,
)

STEM_DEAD = {0: (set(), {0, 1, 2, 3})}
SIGMOID_FOLD = {j: 0.5 for j in range(4)}


def _arm(graph, *, folding="full", seeds=None, mode="cascade"):
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    return compute_global_pruned_sets(
        graph,
        initial_per_node=seeds,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
        mode=mode,
        elimination_constant_folding=folding,
        spiking_mode=INERT_SPIKING_MODE,
    )


def _host_graph(module):
    """The sigmoid vehicle with ``module`` hosted in place of ``nn.Sigmoid``."""
    graph = sigmoid_chain_graph()
    op = next(n for n in graph.nodes if isinstance(n, ComputeOp))
    op.params["module"] = module
    return graph


class _FrozenNormHost(nn.Module):
    """A hosted block whose normalizer is deliberately FROZEN in eval mode.

    This is the ordinary shape of a fine-tuning / conversion checkpoint: the
    block trains, its BatchNorm does not. Its ``forward`` is still exactly the
    sigmoid vehicle's (the identity-configured norm passes 0 through), so the
    lattice must derive the very same CONST(0.5) fold.
    """

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        with torch.no_grad():
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
            self.norm.running_mean.zero_()
            self.norm.running_var.fill_(1.0)
        self.train()          # the block is in training mode ...
        self.norm.eval()      # ... and its normalizer is deliberately not

    def forward(self, x):
        return torch.sigmoid(self.norm(x))


class _ModeProbeHost(nn.Module):
    """Records the mode its CHILD was executed in (a pure observation point)."""

    def __init__(self) -> None:
        super().__init__()
        self.child = nn.Sigmoid()

    def forward(self, x):
        _CHILD_MODES.append(bool(self.child.training))
        return self.child(x)


# Module-level, so a ``deepcopy`` of the host (the fp64 twin) shares it and
# every probe execution is counted, copies included.
_CHILD_MODES: list = []
_PROBE_EXECUTIONS: list = []


class _RngDrawingHost(nn.Module):
    """A host whose ``forward`` draws from the GLOBAL torch generator.

    The output is deterministic (the draw is discarded), so the probe battery
    passes every agreement gate and the lattice still folds — which is what
    makes this a clean measurement of the RNG side effect alone rather than of
    a refusal. ``live_calls`` counts only executions of THIS instance; the
    module-level list counts the fp64 copies too.
    """

    def __init__(self) -> None:
        super().__init__()
        self.live_calls = 0

    def forward(self, x):
        self.live_calls += 1
        _PROBE_EXECUTIONS.append(1)
        torch.rand(4, dtype=torch.float32)
        return torch.sigmoid(x)


class TestModuleModeIsRestoredPerSubmodule:
    """FINDING 1: a root-level ``module.train()`` clobbers the whole subtree."""

    def test_a_frozen_child_keeps_its_mode_across_a_full_cascade(self):
        host = _FrozenNormHost()
        graph = _host_graph(host)
        before = {name: m.training for name, m in host.named_modules()}
        assert before[""] is True and before["norm"] is False, (
            "vehicle precondition: parent trains, child is frozen"
        )

        result = _arm(graph, seeds=STEM_DEAD)

        after = {name: m.training for name, m in host.named_modules()}
        assert after == before, (
            "running the constant analysis mutated the hosted module's "
            f"training flags: {before} -> {after}"
        )
        assert result.constant_folds.folded_rows[2] == SIGMOID_FOLD, (
            "the probe must actually have executed the module"
        )

    def test_an_all_eval_module_is_never_touched(self):
        host = _FrozenNormHost().eval()
        graph = _host_graph(host)
        _arm(graph, seeds=STEM_DEAD)
        assert not any(m.training for m in host.modules())

    def test_the_probe_covers_a_training_child_under_an_eval_root(self):
        """The eval probe is about the AS-DEPLOYED mode, so it must reach the
        subtree: a root already in eval whose child still trains was skipped
        entirely by the root-level guard."""
        host = _ModeProbeHost()
        host.eval()
        host.child.train()
        _CHILD_MODES.clear()

        result = _arm(_host_graph(host), seeds=STEM_DEAD)

        assert _CHILD_MODES, "the host was never executed"
        assert False in _CHILD_MODES, (
            "the eval-mode probe must cover descendants, not just the root"
        )
        assert host.training is False and host.child.training is True, (
            "the deliberate mode split must survive the analysis"
        )
        assert result.constant_folds.folded_rows[2] == SIGMOID_FOLD


class TestGlobalRngIsUnperturbed:
    """FINDING 2: probes execute host modules, twice on the ORIGINAL module."""

    def test_the_cpu_generator_is_byte_identical_across_an_analysis(self):
        host = _RngDrawingHost()
        graph = _host_graph(host)
        _PROBE_EXECUTIONS.clear()
        torch.manual_seed(1234)
        before = torch.get_rng_state().clone()

        result = _arm(graph, seeds=STEM_DEAD)

        after = torch.get_rng_state()
        assert len(_PROBE_EXECUTIONS) >= 6, (
            "the probe battery must have executed the stochastic host "
            f"(saw {len(_PROBE_EXECUTIONS)} executions)"
        )
        assert torch.equal(before, after), (
            "the constant analysis consumed the global torch generator; every "
            "downstream draw of a seeded run would shift"
        )
        assert result.constant_folds.folded_rows[2] == SIGMOID_FOLD

    def test_the_downstream_draw_sequence_is_unchanged(self):
        """The end-to-end statement: an analysis run in the middle of a seeded
        program does not move a single sample."""
        torch.manual_seed(99)
        reference = [torch.rand(3) for _ in range(3)]

        torch.manual_seed(99)
        _arm(_host_graph(_RngDrawingHost()), seeds=STEM_DEAD)
        observed = [torch.rand(3) for _ in range(3)]

        assert all(torch.equal(a, b) for a, b in zip(reference, observed))

    def test_isolation_covers_the_original_module_not_only_the_copies(self):
        """The fp64 twin is a deepcopy, but two of the six probe executions run
        on the LIVE module — forking only around the copies would still leak."""
        host = _RngDrawingHost()
        graph = _host_graph(host)
        torch.manual_seed(7)
        before = torch.get_rng_state().clone()
        _arm(graph, seeds=STEM_DEAD)
        assert torch.equal(before, torch.get_rng_state())
        # Two fp32 executions run on `host` ITSELF (the eval probe and the
        # as-is probe); the other four run on its fp64 deepcopy.
        assert host.live_calls == 2, (
            "expected the live module to be executed twice, saw "
            f"{host.live_calls}"
        )


class TestConstantFoldingIsRegistryRepresentable:
    """FINDING 3: a user-facing axis lives in the configurability SSOT."""

    def test_the_axis_is_a_first_class_registry_entry(self):
        entry = REGISTRY[ELIMINATION_CONSTANT_FOLDING_KEY]
        assert entry.type is FieldType.ENUM
        assert entry.group == "mapping_strategy"
        assert entry.exposure == "user"

    def test_options_and_default_are_locked_to_the_policy_ssot(self):
        entry = REGISTRY[ELIMINATION_CONSTANT_FOLDING_KEY]
        assert tuple(entry.resolved_options()) == tuple(
            ELIMINATION_CONSTANT_FOLDING_MODES
        )
        assert entry.derived_default is not None
        assert entry.derived_default({}) == DEFAULT_ELIMINATION_CONSTANT_FOLDING
        assert DEFAULT_ELIMINATION_CONSTANT_FOLDING == "full"
        assert set(ELIMINATION_CONSTANT_FOLDING_MODES) == {"full", "off"}

    def test_every_registered_option_resolves_through_the_resolve_path(self):
        """No drift: the wizard cannot offer a value the resolver refuses, and
        the resolver's silent default is the wizard's rendered default."""
        entry = REGISTRY[ELIMINATION_CONSTANT_FOLDING_KEY]
        for option in entry.resolved_options():
            assert resolve_elimination_constant_folding(
                {ELIMINATION_CONSTANT_FOLDING_KEY: option}
            ) == option
        assert resolve_elimination_constant_folding({}) == entry.derived_default({})

    def test_it_sits_beside_its_sibling_axis(self):
        sibling = REGISTRY["elimination_propagation"]
        entry = REGISTRY[ELIMINATION_CONSTANT_FOLDING_KEY]
        assert (entry.group, entry.owner, entry.section, entry.category) == (
            sibling.group, sibling.owner, sibling.section, sibling.category
        )
        assert entry.relevant.to_json() == sibling.relevant.to_json()


def _live_readers_of_pruned_constants(graph, result):
    """(reader, port, value) for every LIVE axon reading a pruned CONST(c!=0).

    This is EXACTLY the hazard: the lattice answers ``c`` for such a line
    while the deployed program rewires the axon OFF and reads 0.
    """
    lattice = result.constant_folds.lattice
    offenders = []
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        pruned = result.pruned_rows_per_node.get(node.id, set())
        for i, src in enumerate(node.input_sources.flatten()):
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            if i in pruned:
                continue
            port = (src.node_id, src.index)
            value = lattice.values.get(port)
            if value is None or value == 0.0:
                continue
            if src.index in result.pruned_cols_per_node.get(
                src.node_id, set()
            ):
                offenders.append(((node.id, i), port, value))
    for src in graph.output_sources.flatten():
        if not isinstance(src, IRSource) or src.node_id < 0:
            continue
        port = (src.node_id, src.index)
        value = lattice.values.get(port)
        if value is None or value == 0.0:
            continue
        if src.index in result.pruned_cols_per_node.get(src.node_id, set()):
            offenders.append((("model_output", -1), port, value))
    return offenders


class TestFoldedThenOrphanedLinesAgreeWithTheDeployedProgram:
    """The LOW risk, proven absent rather than argued away.

    ``_orphan_neurons`` kills a column only when EVERY consumer axon is
    already pruned, and a CONST(c != 0) line's readers are pruned exactly by
    folding ``c`` away — so no live reader can survive the orphan kill. Pinned
    on every vehicle, plus a bit-exact certificate on the one vehicle where a
    non-zero constant really is folded and then orphan-killed.
    """

    @pytest.mark.parametrize(
        "make,seeds",
        [
            (sigmoid_chain_graph, STEM_DEAD),
            (residual_join_graph, STEM_DEAD),
            (bias_only_collapse_graph, None),
        ],
    )
    def test_no_live_axon_reads_an_orphan_killed_nonzero_constant(
        self, make, seeds
    ):
        graph = make()
        result = _arm(graph, seeds=seeds)
        assert _live_readers_of_pruned_constants(graph, result) == []

    def test_the_bias_only_vehicle_really_exercises_the_risk_case(self):
        """Guard against the pin passing vacuously: this vehicle DOES record a
        non-zero constant on a column that is then orphan-killed."""
        graph = bias_only_collapse_graph()
        result = _arm(graph)
        value = result.constant_folds.lattice.values[(0, 0)]
        assert value != 0.0
        assert 0 in result.pruned_cols_per_node[0], (
            "the constant column must be orphan-killed for this to be the "
            "risk case at all"
        )
        head = next(n for n in graph.nodes if n.id == 2)
        readers = [
            i for i, src in enumerate(head.input_sources.flatten())
            if isinstance(src, IRSource) and (src.node_id, src.index) == (0, 0)
        ]
        assert readers, "vehicle precondition: the constant line has a reader"
        assert set(readers) <= result.pruned_rows_per_node[2], (
            "every reader of the orphan-killed constant must be folded away"
        )

    def test_the_orphan_killed_constant_is_bit_exact_on_the_executor(self):
        report = certify_cascade_equivalence(
            bias_only_collapse_graph(), batches=4, batch_size=8,
        )
        assert report.passed is True
        assert report.max_abs_delta == 0.0
        assert report.pruned_cells < report.reference_cells

    def test_a_synthetic_live_reader_is_detected_by_the_probe(self):
        """Mutation guard: the invariant probe above is not vacuous — inject a
        live axon reading an orphan-killed non-zero constant and it fires."""
        graph = bias_only_collapse_graph()
        result = _arm(graph)
        head = next(n for n in graph.nodes if n.id == 2)
        head.input_sources = np.array(
            list(head.input_sources.flatten()) + [IRSource(node_id=0, index=0)],
            dtype=object,
        )
        assert _live_readers_of_pruned_constants(graph, result) == [
            ((2, 4), (0, 0), result.constant_folds.lattice.values[(0, 0)])
        ]
