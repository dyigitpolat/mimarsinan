"""W1 cascade equivalence certificate: bit-exact value parity, preconditions, bank union rule, mutation trip."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.certificate import (
    CascadeCertificateError,
    CascadeCertificatePreconditionError,
    certify_cascade_equivalence,
    check_shared_bank_union_rule,
    derive_cols_with_implicit_source,
    is_zero_preserving_host_op,
    snap_ir_graph_to_dyadic_grid,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_core import compute_global_pruned_sets


def _srcs(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _dyadic(rng, shape, fraction_bits=4, span=8):
    """Random matrix exactly on the 2^-fraction_bits grid."""
    ints = rng.integers(-span, span + 1, size=shape).astype(np.float64)
    return np.ldexp(ints, -fraction_bits)


def _owned_two_core_graph(seed=11):
    """NC0 (5x6, 4 data axons + bias row) -> NC1 (7x3) -> 3 output logits."""
    rng = np.random.default_rng(seed)
    w0 = _dyadic(rng, (5, 6))
    w1 = _dyadic(rng, (7, 3))
    src0 = _srcs([(-2, 0), (-2, 1), (-2, 2), (-2, 3), (-3, 0)])
    core0 = NeuralCore(id=0, name="c0", input_sources=src0, core_matrix=w0,
                       threshold=1.0, latency=0)
    src1 = _srcs([(0, j) for j in range(6)] + [(-3, 0)])
    core1 = NeuralCore(id=1, name="c1", input_sources=src1, core_matrix=w1,
                       threshold=1.0, latency=1)
    out = _srcs([(1, 0), (1, 1), (1, 2)])
    return IRGraph(nodes=[core0, core1], output_sources=out)


def _bank_token_graph(seed=13, n_tokens=3):
    """n_tokens bank-backed cores (bank 5x4) -> owned head NC -> 2 logits."""
    rng = np.random.default_rng(seed)
    bank = WeightBank(id=0, core_matrix=_dyadic(rng, (5, 4)))
    nodes = []
    for tok in range(n_tokens):
        srcs = _srcs([(-2, tok * 4 + i) for i in range(4)] + [(-3, 0)])
        nodes.append(NeuralCore(
            id=tok, name=f"tok{tok}", input_sources=srcs, core_matrix=None,
            weight_bank_id=0, weight_row_slice=(0, 4), threshold=1.0, latency=0,
        ))
    head_srcs = _srcs(
        [(tok, j) for tok in range(n_tokens) for j in range(4)] + [(-3, 0)]
    )
    head = NeuralCore(
        id=n_tokens, name="head", input_sources=head_srcs,
        core_matrix=_dyadic(rng, (n_tokens * 4 + 1, 2)), threshold=1.0,
        latency=1,
    )
    out = _srcs([(n_tokens, 0), (n_tokens, 1)])
    return IRGraph(nodes=nodes + [head], output_sources=out,
                   weight_banks={0: bank})


def _relu_sandwich_graph(act_module, seed=17):
    """NC0 -> activation ComputeOp -> NC1 -> 2 logits."""
    rng = np.random.default_rng(seed)
    core0 = NeuralCore(
        id=0, name="c0", input_sources=_srcs([(-2, 0), (-2, 1), (-3, 0)]),
        core_matrix=_dyadic(rng, (3, 4)), threshold=1.0, latency=0,
    )
    op = ComputeOp(
        id=1, name="act", input_sources=_srcs([(0, j) for j in range(4)]),
        op_type=type(act_module).__name__,
        params={"module": act_module, "input_shape": (4,)},
        input_shape=(4,), output_shape=(4,),
    )
    core1 = NeuralCore(
        id=2, name="c1", input_sources=_srcs([(1, j) for j in range(4)] + [(-3, 0)]),
        core_matrix=_dyadic(rng, (5, 2)), threshold=1.0, latency=1,
    )
    out = _srcs([(2, 0), (2, 1)])
    return IRGraph(nodes=[core0, op, core1], output_sources=out)


def _conv_vehicle():
    """Small conv mvm vehicle via the real torch converter (bank-backed conv)."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2), nn.Flatten(), nn.Linear(64, 10),
    ).eval()
    fused = convert_torch_model(
        model, (1, 8, 8), 10, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    for p in fused.get_perceptrons():
        fuse_into_perceptron(p, device="cpu")
    repr_ = fused.get_mapper_repr()
    repr_.assign_perceptron_indices()
    return IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=256, max_neurons=64
    ).map(repr_)


def _op_with_module(module, n_inputs=4):
    return ComputeOp(
        id=9, name="probe", input_sources=_srcs([(0, j) for j in range(n_inputs)]),
        op_type=type(module).__name__,
        params={"module": module, "input_shape": (n_inputs,)},
        input_shape=(n_inputs,), output_shape=(n_inputs,),
    )


class TestZeroPreservingRegistry:
    @pytest.mark.parametrize("module", [
        nn.ReLU(), nn.LeakyReLU(), nn.GELU(), nn.Identity(),
    ])
    def test_known_activations_are_zero_preserving(self, module):
        assert is_zero_preserving_host_op(_op_with_module(module)) is True

    def test_avgpool_is_zero_preserving(self):
        op = ComputeOp(
            id=9, name="pool", input_sources=_srcs([(0, j) for j in range(16)]),
            op_type="AvgPool2d",
            params={"module": nn.AvgPool2d(2), "input_shape": (1, 4, 4)},
            input_shape=(1, 4, 4), output_shape=(1, 2, 2),
        )
        assert is_zero_preserving_host_op(op) is True

    @pytest.mark.parametrize("module", [nn.Sigmoid(), nn.Softmax(dim=-1)])
    def test_sigmoid_and_softmax_are_not_zero_preserving(self, module):
        assert is_zero_preserving_host_op(_op_with_module(module)) is False

    def test_unknown_module_fails_loud_naming_the_op(self):
        class MysteryOp(nn.Module):
            def forward(self, x):
                return x + 1.0

        with pytest.raises(CascadeCertificatePreconditionError, match="MysteryOp"):
            is_zero_preserving_host_op(_op_with_module(MysteryOp()))

    def test_registered_op_failing_numeric_check_fails_loud(self, monkeypatch):
        """A registry entry whose act(0) != 0 must raise, not pass: the check is real."""
        import mimarsinan.mapping.pruning.certificate.zero_preserving as zp

        class LyingRelu(nn.Module):
            def forward(self, x):
                return x + 1.0

        monkeypatch.setattr(
            zp, "ZERO_PRESERVING_HOST_OP_TYPES",
            zp.ZERO_PRESERVING_HOST_OP_TYPES | frozenset({LyingRelu}),
        )
        with pytest.raises(CascadeCertificatePreconditionError, match="LyingRelu"):
            is_zero_preserving_host_op(_op_with_module(LyingRelu()))

    def test_identity_op_without_module_is_zero_preserving(self):
        op = ComputeOp(
            id=9, name="relay", input_sources=_srcs([(0, 0), (0, 1)]),
            op_type="identity",
        )
        assert is_zero_preserving_host_op(op) is True

    def test_derive_cols_with_implicit_source_marks_sigmoid_consumers(self):
        graph = _relu_sandwich_graph(nn.Sigmoid())
        exempt = derive_cols_with_implicit_source(graph)
        assert exempt == {0: frozenset({0, 1, 2, 3})}

    def test_derive_cols_with_implicit_source_empty_for_relu(self):
        graph = _relu_sandwich_graph(nn.ReLU())
        assert derive_cols_with_implicit_source(graph) == {}


class TestCascadeCertificateGreen:
    def test_owned_vehicle_green_and_actually_eliminates(self):
        graph = _owned_two_core_graph()
        seeds = {0: ([False] * 5, [j == 2 for j in range(6)])}
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_node=seeds, batches=3, batch_size=8,
        )
        assert report.passed is True
        assert report.outputs_compared == 3 * 8 * 3
        assert report.pruned_cells < report.reference_cells
        assert report.max_abs_delta == 0.0

    def test_relu_sandwich_green(self):
        graph = _relu_sandwich_graph(nn.ReLU())
        seeds = {0: ([False, False, False], [j == 1 for j in range(4)])}
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_node=seeds, batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.pruned_cells < report.reference_cells

    def test_bank_vehicle_green_and_union_rule_checked(self):
        graph = _bank_token_graph()
        seeds = {0: ([False] * 5, [j == 1 for j in range(4)])}
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_bank=seeds, batches=3, batch_size=4,
        )
        assert report.passed is True
        assert report.bank_columns_checked >= 1
        assert report.pruned_cells < report.reference_cells

    def test_value_only_pruning_green_without_seeds(self):
        graph = _owned_two_core_graph()
        graph.nodes[0].core_matrix[:, 4] = 0.0  # exactly-zero column: value seed
        report = certify_cascade_equivalence(graph, batches=2, batch_size=4)
        assert report.passed is True
        assert report.pruned_cells < report.reference_cells

    def test_conv_vehicle_green(self):
        """DoD: FATAL cert green on one small conv vehicle (transformer deferred, W0.5)."""
        graph = snap_ir_graph_to_dyadic_grid(_conv_vehicle(), fraction_bits=8)
        n_cols = graph.weight_banks[0].core_matrix.shape[1]
        seeds = {0: ([False] * graph.weight_banks[0].core_matrix.shape[0],
                     [j == 1 for j in range(n_cols)])}
        report = certify_cascade_equivalence(
            graph, initial_pruned_per_bank=seeds, batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.bank_columns_checked >= 1
        assert report.pruned_cells < report.reference_cells


class TestCascadeCertificateRefusals:
    def test_refuses_sigmoid_activation(self):
        graph = _relu_sandwich_graph(nn.Sigmoid())
        with pytest.raises(CascadeCertificatePreconditionError, match="Sigmoid"):
            certify_cascade_equivalence(graph, batches=1, batch_size=2)

    def test_refuses_unknown_host_op(self):
        class MysteryOp(nn.Module):
            def forward(self, x):
                return x * 2.0

        graph = _relu_sandwich_graph(MysteryOp())
        with pytest.raises(CascadeCertificatePreconditionError, match="MysteryOp"):
            certify_cascade_equivalence(graph, batches=1, batch_size=2)

    def test_refuses_grid_breaking_gelu(self):
        """GELU is zero-preserving but maps dyadic values off-grid: bit-exact
        comparison is ill-posed, so the certificate refuses rather than flake."""
        graph = _relu_sandwich_graph(nn.GELU())
        with pytest.raises(CascadeCertificatePreconditionError, match="GELU"):
            certify_cascade_equivalence(graph, batches=1, batch_size=2)

    def test_refuses_off_grid_weights(self):
        graph = _owned_two_core_graph()
        graph.nodes[0].core_matrix[0, 0] = 0.1  # not dyadic
        with pytest.raises(CascadeCertificatePreconditionError, match="dyadic"):
            certify_cascade_equivalence(graph, batches=1, batch_size=2)

    def test_refuses_non_power_of_two_threshold(self):
        graph = _owned_two_core_graph()
        graph.nodes[1].threshold = 3.0
        with pytest.raises(CascadeCertificatePreconditionError, match="threshold"):
            certify_cascade_equivalence(graph, batches=1, batch_size=2)

    def test_refuses_per_node_seed_on_bank_backed_node(self):
        graph = _bank_token_graph()
        seeds = {0: ([False] * 5, [True, False, False, False])}
        with pytest.raises(
            CascadeCertificatePreconditionError, match="initial_pruned_per_bank"
        ):
            certify_cascade_equivalence(
                graph, initial_pruned_per_node=seeds, batches=1, batch_size=2,
            )

    def test_snap_helper_makes_float_graph_certifiable(self):
        graph = _owned_two_core_graph()
        graph.nodes[0].core_matrix[0, 0] = 0.1
        graph.nodes[1].threshold = 3.0
        snapped = snap_ir_graph_to_dyadic_grid(graph, fraction_bits=8)
        report = certify_cascade_equivalence(snapped, batches=1, batch_size=2)
        assert report.passed is True
        # The original instance is untouched (snap works on a deep copy).
        assert graph.nodes[0].core_matrix[0, 0] == 0.1


class TestSharedBankUnionRule:
    def test_union_rule_holds_on_bank_fixpoint(self):
        graph = _bank_token_graph()
        result = compute_global_pruned_sets(
            graph, initial_per_bank={0: (set(), {1})},
        )
        checked = check_shared_bank_union_rule(graph, result)
        assert checked >= 1

    def test_union_rule_violation_raises_naming_bank_col_and_node(self):
        graph = _bank_token_graph()
        result = compute_global_pruned_sets(
            graph, initial_per_bank={0: (set(), {1})},
        )
        # Tamper: pretend instance tok0 still needs bank column 1.
        result.pruned_cols_per_node[0].discard(1)
        with pytest.raises(CascadeCertificateError, match="bank 0.*column 1.*node 0"):
            check_shared_bank_union_rule(graph, result)

    def test_row_union_violation_raises(self):
        graph = _bank_token_graph()
        result = compute_global_pruned_sets(
            graph, initial_per_bank={0: (set(), {1})},
        )
        result.pruned_rows_per_bank[0].add(2)
        with pytest.raises(CascadeCertificateError, match="row 2"):
            check_shared_bank_union_rule(graph, result)


class TestCertificateTripsOnMutation:
    def test_corrupted_rewire_sources_trips_certificate(self, monkeypatch):
        """A certificate that cannot fail is not a certificate: corrupt the
        rewiring step and the value comparison must trip."""
        import mimarsinan.mapping.pruning.ir_pruning_core as core_mod

        original = core_mod._rewire_sources

        def corrupted(graph, pruned_cols_per_node):
            original(graph, pruned_cols_per_node)
            for node in graph.nodes:
                if not hasattr(node, "input_sources"):
                    continue
                flat = node.input_sources.flatten()
                live = [
                    i for i, s in enumerate(flat)
                    if isinstance(s, IRSource) and s.node_id >= 0
                ]
                for a in live:
                    for b in live:
                        if (
                            flat[a].node_id == flat[b].node_id
                            and flat[a].index != flat[b].index
                        ):
                            flat[a], flat[b] = flat[b], flat[a]
                            node.input_sources = flat.reshape(
                                node.input_sources.shape
                            )
                            return
            raise AssertionError("mutation found nothing to corrupt")

        monkeypatch.setattr(core_mod, "_rewire_sources", corrupted)

        graph = _owned_two_core_graph()
        seeds = {0: ([False] * 5, [j == 2 for j in range(6)])}
        with pytest.raises(CascadeCertificateError, match="differ"):
            certify_cascade_equivalence(
                graph, initial_pruned_per_node=seeds, batches=3, batch_size=8,
            )
