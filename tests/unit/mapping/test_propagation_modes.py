"""W3 propagation-mode axis: masked | closure | cascade.

The three arms pin the paper's attribution claim (C1/P2):

- ``masked``   — allocation-naive LOWER BOUND: exactly the seeded,
  exemption-filtered sets, no structural reasoning at all.
- ``closure``  — one-hop seed-group coupling (the DepGraph/torch-pruning
  equivalent baseline): a seeded neuron kill removes the corresponding axon
  row in every direct consumer, and a seeded axon-row kill removes its paired
  producer neuron when that neuron has no other live reader. NO emergent
  deadness (a neuron whose inputs all died stays alive), NO iteration.
- ``cascade``  — the existing bidirectional liveness fixpoint (DEFAULT; the
  default path must remain byte-identical).

Invariant under test: kills(masked) <= kills(closure) <= kills(cascade),
per node and per bank, and each mode is semantics-preserving for its own
kill set (certificate green).
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    DEFAULT_ELIMINATION_PROPAGATION,
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
    ELIMINATION_PROPAGATION_MODES,
    require_elimination_propagation,
    resolve_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.pruning_propagation import (
    compute_propagated_pruned_rows_cols,
    matrix_one_step_deaths,
)
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph

MASKED = ELIMINATION_PROPAGATION_MASKED
CLOSURE = ELIMINATION_PROPAGATION_CLOSURE
CASCADE = ELIMINATION_PROPAGATION_CASCADE


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


def make_emergent_chain():
    """A -> B -> C chain where cascade provably kills MORE than closure.

    Seeding A.col0 starves B.col0 (its only weighted axon is B.row0, which
    reads A.0), whose death then frees C.row0 — emergent deadness at depth 2+
    that one-hop closure must NOT discover.
    """
    w_a = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]], dtype=np.float64)
    w_b = np.array([[10.0, 0.0], [0.0, 11.0], [0.0, 1.0]], dtype=np.float64)
    w_c = np.array([[5.0, 0.0], [0.0, 6.0], [0.0, 1.0]], dtype=np.float64)
    a = NeuralCore(
        id=0, name="A", input_sources=_src([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=w_a, threshold=1.0, latency=0,
    )
    b = NeuralCore(
        id=1, name="B", input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
        core_matrix=w_b, threshold=1.0, latency=1,
    )
    c = NeuralCore(
        id=2, name="C", input_sources=_src([(1, 0), (1, 1), (-3, 0)]),
        core_matrix=w_c, threshold=1.0, latency=2,
    )
    graph = IRGraph(nodes=[a, b, c], output_sources=_src([(2, 0), (2, 1)]))
    kwargs = dict(
        initial_per_node={0: (set(), {0})},
        exempt_rows_per_node={0: frozenset({0, 1}), 1: frozenset(), 2: frozenset()},
        exempt_cols_per_node={0: frozenset(), 1: frozenset(), 2: frozenset({0, 1})},
    )
    return graph, kwargs


def make_backward_coupling_pair():
    """A -> B where the seed kills B.row0 so closure must kill its paired
    producer A.col0 (sole reader) — but masked must not."""
    w_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    w_b = np.array([[5.0, 6.0], [7.0, 8.0], [0.0, 1.0]], dtype=np.float64)
    a = NeuralCore(
        id=0, name="A", input_sources=_src([(-2, 0), (-2, 1)]),
        core_matrix=w_a, threshold=1.0, latency=0,
    )
    b = NeuralCore(
        id=1, name="B", input_sources=_src([(0, 0), (0, 1), (-3, 0)]),
        core_matrix=w_b, threshold=1.0, latency=1,
    )
    graph = IRGraph(nodes=[a, b], output_sources=_src([(1, 0), (1, 1)]))
    kwargs = dict(
        initial_per_node={1: ({0}, set())},
        exempt_rows_per_node={0: frozenset({0, 1}), 1: frozenset()},
        exempt_cols_per_node={0: frozenset(), 1: frozenset({0, 1})},
    )
    return graph, kwargs


def make_bank_token_graph(n_tokens=3, seed=13):
    """n_tokens bank-backed cores sharing one (5x4) bank -> owned head."""
    rng = np.random.default_rng(seed)
    bank = WeightBank(
        id=0, core_matrix=rng.standard_normal((5, 4)).astype(np.float64)
    )
    nodes = []
    for tok in range(n_tokens):
        srcs = _src([(-2, tok * 4 + i) for i in range(4)] + [(-3, 0)])
        nodes.append(NeuralCore(
            id=tok, name=f"tok{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, 4), threshold=1.0,
            latency=0,
        ))
    head_srcs = _src(
        [(tok, j) for tok in range(n_tokens) for j in range(4)] + [(-3, 0)]
    )
    head = NeuralCore(
        id=n_tokens, name="head", input_sources=head_srcs,
        core_matrix=rng.standard_normal((n_tokens * 4 + 1, 2)).astype(np.float64),
        threshold=1.0, latency=1,
    )
    graph = IRGraph(
        nodes=nodes + [head],
        output_sources=_src([(n_tokens, 0), (n_tokens, 1)]),
        weight_banks={0: bank},
    )
    kwargs = dict(initial_per_bank={0: (set(), {1})})
    return graph, kwargs


class TestModeSSOT:
    def test_modes_tuple_and_default(self):
        assert ELIMINATION_PROPAGATION_MODES == ("masked", "closure", "cascade")
        assert DEFAULT_ELIMINATION_PROPAGATION == "cascade"

    def test_require_accepts_every_mode(self):
        for m in ELIMINATION_PROPAGATION_MODES:
            assert require_elimination_propagation(m) == m

    def test_require_rejects_unknown_mode_loudly(self):
        with pytest.raises(ValueError, match="bogus"):
            require_elimination_propagation("bogus")

    def test_resolve_defaults_to_cascade_when_absent(self):
        assert resolve_elimination_propagation({}) == "cascade"

    def test_resolve_reads_and_validates_the_config_key(self):
        assert resolve_elimination_propagation(
            {"elimination_propagation": "closure"}
        ) == "closure"
        with pytest.raises(ValueError, match="elimination_propagation"):
            resolve_elimination_propagation({"elimination_propagation": "x"})


class TestKernelModes:
    """The single-matrix kernel over the mode axis."""

    def _matrix(self):
        # col 2 seeded dead; row 1 feeds ONLY col 2 (cascade kills it).
        return np.array(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=np.float64,
        )

    def test_masked_returns_exactly_the_seeds(self):
        rows, cols = compute_propagated_pruned_rows_cols(
            self._matrix(), initial_zero_rows=set(), initial_zero_cols={2},
            mode=MASKED,
        )
        assert rows == set()
        assert cols == {2}

    def test_closure_equals_masked_at_matrix_granularity(self):
        """One-hop coupling is a cross-matrix phenomenon; within a single
        matrix closure adds nothing (and discovers no emergent deadness)."""
        seeded = dict(initial_zero_rows=set(), initial_zero_cols={2})
        m_r, m_c = compute_propagated_pruned_rows_cols(
            self._matrix(), mode=MASKED, **seeded
        )
        c_r, c_c = compute_propagated_pruned_rows_cols(
            self._matrix(), mode=CLOSURE, **seeded
        )
        assert (m_r, m_c) == (c_r, c_c)

    def test_default_mode_is_cascade(self):
        seeded = dict(initial_zero_rows=set(), initial_zero_cols={2})
        d_r, d_c = compute_propagated_pruned_rows_cols(self._matrix(), **seeded)
        k_r, k_c = compute_propagated_pruned_rows_cols(
            self._matrix(), mode=CASCADE, **seeded
        )
        assert (d_r, d_c) == (k_r, k_c)
        assert 1 in d_r, "cascade must starve row 1"

    def test_kernel_subset_ordering(self):
        seeded = dict(initial_zero_rows=set(), initial_zero_cols={2})
        results = {
            m: compute_propagated_pruned_rows_cols(self._matrix(), mode=m, **seeded)
            for m in ELIMINATION_PROPAGATION_MODES
        }
        assert results[MASKED][0] <= results[CLOSURE][0] <= results[CASCADE][0]
        assert results[MASKED][1] <= results[CLOSURE][1] <= results[CASCADE][1]

    def test_kernel_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="bogus"):
            compute_propagated_pruned_rows_cols(self._matrix(), mode="bogus")

    def test_exemptions_hold_in_every_mode(self):
        for m in ELIMINATION_PROPAGATION_MODES:
            _, cols = compute_propagated_pruned_rows_cols(
                self._matrix(), initial_zero_rows=set(), initial_zero_cols={2},
                exempt_cols={2}, mode=m,
            )
            assert 2 not in cols


class TestMatrixOneStep:
    def test_one_step_discovers_only_first_generation_deaths(self):
        w = np.array(
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        rows, cols = matrix_one_step_deaths(
            w, pruned_rows=set(), pruned_cols={2},
        )
        assert rows == {1}
        assert cols == set()

    def test_one_step_col_starvation(self):
        w = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        rows, cols = matrix_one_step_deaths(
            w, pruned_rows={0, 1}, pruned_cols=set(),
        )
        assert cols == {0}
        assert rows == set()

    def test_one_step_respects_exemptions_and_implicit_sources(self):
        w = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        _, cols = matrix_one_step_deaths(
            w, pruned_rows={0, 1}, pruned_cols=set(), exempt_cols={0},
        )
        assert cols == set()
        _, cols = matrix_one_step_deaths(
            w, pruned_rows={0, 1}, pruned_cols=set(),
            cols_with_implicit_source={0},
        )
        assert cols == set()

    def test_quiescent_state_yields_nothing(self):
        w = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        rows, cols = matrix_one_step_deaths(
            w, pruned_rows=set(), pruned_cols=set(),
        )
        assert rows == set() and cols == set()


class TestGlobalModeSemantics:
    def test_masked_is_exactly_the_seed_set(self):
        graph, kwargs = make_emergent_chain()
        res = compute_global_pruned_sets(graph, mode=MASKED, **kwargs)
        assert res.pruned_cols_per_node[0] == {0}
        assert res.pruned_rows_per_node[1] == set()
        assert res.pruned_cols_per_node[1] == set()
        assert res.pruned_rows_per_node[2] == set()

    def test_closure_adds_consumer_coupling_but_no_emergent_deadness(self):
        graph, kwargs = make_emergent_chain()
        res = compute_global_pruned_sets(graph, mode=CLOSURE, **kwargs)
        assert res.pruned_cols_per_node[0] == {0}
        assert res.pruned_rows_per_node[1] == {0}, (
            "seeded neuron kill must remove the direct consumer's axon row"
        )
        assert res.pruned_cols_per_node[1] == set(), (
            "closure must NOT discover emergent deadness: B.col0's inputs all "
            "died but one-hop coupling keeps it alive"
        )
        assert res.pruned_rows_per_node[2] == set()

    def test_cascade_kills_strictly_more_than_closure(self):
        graph, kwargs = make_emergent_chain()
        res = compute_global_pruned_sets(graph, mode=CASCADE, **kwargs)
        assert res.pruned_cols_per_node[0] == {0}
        assert res.pruned_rows_per_node[1] == {0}
        assert res.pruned_cols_per_node[1] == {0}, "emergent: B.col0 starved"
        assert res.pruned_rows_per_node[2] == {0}, "emergent: C.row0 freed"

    def test_backward_coupling_kills_the_paired_producer(self):
        graph, kwargs = make_backward_coupling_pair()
        masked = compute_global_pruned_sets(graph, mode=MASKED, **kwargs)
        closure = compute_global_pruned_sets(graph, mode=CLOSURE, **kwargs)
        assert 0 not in masked.pruned_cols_per_node[0]
        assert 0 in closure.pruned_cols_per_node[0], (
            "seeded axon-row kill must couple to its paired producer neuron"
        )

    def test_backward_coupling_spares_producers_with_live_readers(self):
        """A producer neuron read by a NON-seeded consumer row must survive
        closure (killing it would drop live signal)."""
        graph, kwargs = make_backward_coupling_pair()
        # Add a second consumer of A.0 whose row is not seeded.
        w_d = np.array([[1.0], [0.5]], dtype=np.float64)
        d = NeuralCore(
            id=2, name="D", input_sources=_src([(0, 0), (-3, 0)]),
            core_matrix=w_d, threshold=1.0, latency=1,
        )
        graph.nodes.append(d)
        graph.output_sources = _src([(1, 0), (1, 1), (2, 0)])
        kwargs["exempt_rows_per_node"] = dict(kwargs["exempt_rows_per_node"])
        kwargs["exempt_rows_per_node"][2] = frozenset()
        kwargs["exempt_cols_per_node"] = dict(kwargs["exempt_cols_per_node"])
        kwargs["exempt_cols_per_node"][2] = frozenset({0})
        closure = compute_global_pruned_sets(graph, mode=CLOSURE, **kwargs)
        assert 0 not in closure.pruned_cols_per_node[0]

    @pytest.mark.parametrize("factory", [
        make_emergent_chain, make_backward_coupling_pair, make_bank_token_graph,
    ])
    def test_arm_ordering_invariant(self, factory):
        """kills(masked) <= kills(closure) <= kills(cascade), node and bank."""
        graph, kwargs = factory()
        results = {
            m: compute_global_pruned_sets(graph, mode=m, **kwargs)
            for m in ELIMINATION_PROPAGATION_MODES
        }
        for lo, hi in ((MASKED, CLOSURE), (CLOSURE, CASCADE)):
            a, b = results[lo], results[hi]
            for nid in a.pruned_rows_per_node:
                assert a.pruned_rows_per_node[nid] <= b.pruned_rows_per_node[nid]
                assert a.pruned_cols_per_node[nid] <= b.pruned_cols_per_node[nid]
            for bid in a.pruned_rows_per_bank:
                assert a.pruned_rows_per_bank[bid] <= b.pruned_rows_per_bank[bid]
                assert a.pruned_cols_per_bank[bid] <= b.pruned_cols_per_bank[bid]

    def test_default_call_is_byte_identical_to_explicit_cascade(self):
        graph, kwargs = make_emergent_chain()
        default = compute_global_pruned_sets(graph, **kwargs)
        explicit = compute_global_pruned_sets(graph, mode=CASCADE, **kwargs)
        assert default.pruned_rows_per_node == explicit.pruned_rows_per_node
        assert default.pruned_cols_per_node == explicit.pruned_cols_per_node
        assert default.pruned_rows_per_bank == explicit.pruned_rows_per_bank
        assert default.pruned_cols_per_bank == explicit.pruned_cols_per_bank

    def test_global_rejects_unknown_mode(self):
        graph, kwargs = make_emergent_chain()
        with pytest.raises(ValueError, match="bogus"):
            compute_global_pruned_sets(graph, mode="bogus", **kwargs)

    def test_fixpoint_iteration_count_per_mode(self):
        graph, kwargs = make_emergent_chain()
        assert compute_global_pruned_sets(
            graph, mode=MASKED, **kwargs
        ).fixpoint_iterations == 0
        assert compute_global_pruned_sets(
            graph, mode=CLOSURE, **kwargs
        ).fixpoint_iterations == 1
        cascade = compute_global_pruned_sets(graph, mode=CASCADE, **kwargs)
        assert cascade.fixpoint_iterations >= 2, (
            "the chain needs multiple global sweeps to reach the fixpoint"
        )


class TestBankUnionRuleAllModes:
    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_bank_union_rule_respected(self, mode):
        from mimarsinan.mapping.pruning.certificate import (
            check_shared_bank_union_rule,
        )
        graph, kwargs = make_bank_token_graph()
        res = compute_global_pruned_sets(graph, mode=mode, **kwargs)
        checked = check_shared_bank_union_rule(graph, res)
        assert checked >= 1, "the seeded bank column must be verified"

    def test_closure_couples_bank_neuron_kills_forward(self):
        graph, kwargs = make_bank_token_graph()
        closure = compute_global_pruned_sets(graph, mode=CLOSURE, **kwargs)
        # Bank col 1 seeded -> every token's local col 1 dead (aliasing) ->
        # the head rows reading (tok, 1) die by one-hop coupling.
        head_id = 3
        assert closure.pruned_rows_per_node[head_id] >= {1, 5, 9}
        masked = compute_global_pruned_sets(graph, mode=MASKED, **kwargs)
        assert masked.pruned_rows_per_node[head_id] == set()


class TestPruneIrGraphModes:
    def _two_core_graph(self):
        w_a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        w_b = np.array([[5.0, 6.0], [0.0, 7.0]], dtype=np.float64)
        a = NeuralCore(
            id=0, name="A", input_sources=_src([(-2, 0), (-2, 1)]),
            core_matrix=w_a, threshold=1.0, latency=0,
        )
        b = NeuralCore(
            id=1, name="B", input_sources=_src([(0, 0), (0, 1)]),
            core_matrix=w_b, threshold=1.0, latency=1,
        )
        return IRGraph(nodes=[a, b], output_sources=_src([(1, 0), (1, 1)]))

    def test_masked_compacts_seeds_only_and_rewires_consumers_to_off(self):
        graph = self._two_core_graph()
        prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False, False], [True, False])},
            elimination_propagation=MASKED,
        )
        a = next(n for n in graph.nodes if n.id == 0)
        b = next(n for n in graph.nodes if n.id == 1)
        assert a.core_matrix.shape == (2, 1)
        assert b.core_matrix.shape == (2, 2), (
            "masked must not cascade into the consumer's axon rows"
        )
        assert b.input_sources.flatten()[0].is_off()

    def test_cascade_also_kills_the_consumer_row(self):
        graph = self._two_core_graph()
        prune_ir_graph(
            graph,
            initial_pruned_per_node={0: ([False, False], [True, False])},
            elimination_propagation=CASCADE,
        )
        b = next(n for n in graph.nodes if n.id == 1)
        assert b.core_matrix.shape == (1, 2)

    def test_default_prune_matches_explicit_cascade(self):
        g1 = self._two_core_graph()
        g2 = self._two_core_graph()
        seeds = {0: ([False, False], [True, False])}
        prune_ir_graph(g1, initial_pruned_per_node=seeds)
        prune_ir_graph(
            g2, initial_pruned_per_node=seeds,
            elimination_propagation=CASCADE,
        )
        for n1, n2 in zip(g1.nodes, g2.nodes):
            assert np.array_equal(n1.core_matrix, n2.core_matrix)


def _dyadic(rng, shape, fraction_bits=4, span=8):
    ints = rng.integers(-span, span + 1, size=shape).astype(np.float64)
    return np.ldexp(ints, -fraction_bits)


def _dyadic_owned_two_core_graph(seed=11):
    """NC0 (5x6, 4 data axons + bias row) -> NC1 (7x3) -> 3 output logits."""
    rng = np.random.default_rng(seed)
    core0 = NeuralCore(
        id=0, name="c0",
        input_sources=_src([(-2, 0), (-2, 1), (-2, 2), (-2, 3), (-3, 0)]),
        core_matrix=_dyadic(rng, (5, 6)), threshold=1.0, latency=0,
    )
    core1 = NeuralCore(
        id=1, name="c1",
        input_sources=_src([(0, j) for j in range(6)] + [(-3, 0)]),
        core_matrix=_dyadic(rng, (7, 3)), threshold=1.0, latency=1,
    )
    out = _src([(1, 0), (1, 1), (1, 2)])
    return IRGraph(nodes=[core0, core1], output_sources=out)


def _dyadic_bank_token_graph(seed=13, n_tokens=3):
    """n_tokens bank-backed cores (bank 5x4) -> owned head NC -> 2 logits."""
    rng = np.random.default_rng(seed)
    bank = WeightBank(id=0, core_matrix=_dyadic(rng, (5, 4)))
    nodes = []
    for tok in range(n_tokens):
        srcs = _src([(-2, tok * 4 + i) for i in range(4)] + [(-3, 0)])
        nodes.append(NeuralCore(
            id=tok, name=f"tok{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, 4), threshold=1.0,
            latency=0,
        ))
    head = NeuralCore(
        id=n_tokens, name="head",
        input_sources=_src(
            [(tok, j) for tok in range(n_tokens) for j in range(4)] + [(-3, 0)]
        ),
        core_matrix=_dyadic(rng, (n_tokens * 4 + 1, 2)), threshold=1.0,
        latency=1,
    )
    return IRGraph(
        nodes=nodes + [head],
        output_sources=_src([(n_tokens, 0), (n_tokens, 1)]),
        weight_banks={0: bank},
    )


class TestCertificateGreenUnderEveryMode:
    """Each mode must be semantics-preserving for its own kill set."""

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_owned_vehicle_green(self, mode):
        from mimarsinan.mapping.pruning.certificate import (
            certify_cascade_equivalence,
        )
        graph = _dyadic_owned_two_core_graph()
        seeds = {0: ([False] * 5, [j == 2 for j in range(6)])}
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_node=seeds, batches=2, batch_size=4,
            elimination_propagation=mode,
        )
        assert report.passed is True

    @pytest.mark.parametrize("mode", ELIMINATION_PROPAGATION_MODES)
    def test_bank_vehicle_green(self, mode):
        from mimarsinan.mapping.pruning.certificate import (
            certify_cascade_equivalence,
        )
        graph = _dyadic_bank_token_graph()
        seeds = {0: ([False] * 5, [j == 1 for j in range(4)])}
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_bank=seeds, batches=2, batch_size=4,
            elimination_propagation=mode,
        )
        assert report.passed is True
