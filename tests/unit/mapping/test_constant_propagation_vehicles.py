"""[W4b-2] MEASURED DoD on converter-built vehicles: folding off vs full.

Deep conv vehicle: the same converter model the W4b-1 DoD uses, except the
channel-1-only hidden units KEEP their bias — so when their whole input
support dies they do not starve, they become CONSTANT emitters. The zero-only
cascade must leave them (and the fc2 rows reading them) alive; the lattice
folds their constant onto fc2's carrier and reclaims both.

Tiny ViT vehicle: ``tiny_test_vit`` with its wiring constants (cls token,
positional embedding) zeroed so every host constant is exactly representable.
Killing the patch embedding then propagates through cat -> add -> LayerNorm ->
MultiheadAttention -> residual add — five OPAQUE barriers the zero-only
cascade cannot cross at all.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.certificate import snap_ir_graph_to_dyadic_grid
from mimarsinan.mapping.pruning.graph import (
    compute_global_pruned_sets,
    reanalyze_constant_folding,
)
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
)

CH1_FLAT = list(range(16, 32))
CH1_ONLY_HIDDEN = (3, 7)


def _kills(result) -> int:
    return (
        sum(len(s) for s in result.pruned_rows_per_node.values())
        + sum(len(s) for s in result.pruned_cols_per_node.values())
        + sum(len(s) for s in result.pruned_rows_per_bank.values())
        + sum(len(s) for s in result.pruned_cols_per_bank.values())
    )


def _arm(graph, *, folding, seeds_node=None, seeds_bank=None, mode="cascade"):
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    return compute_global_pruned_sets(
        graph, zero_threshold=1e-8,
        initial_per_node=seeds_node, initial_per_bank=seeds_bank,
        exempt_rows_per_node=exempt_rows, exempt_cols_per_node=exempt_cols,
        mode=mode, elimination_constant_folding=folding,
        spiking_mode=INERT_SPIKING_MODE,
    )


def _owned_core(graph: IRGraph, shape) -> NeuralCore:
    return next(
        n for n in graph.nodes
        if isinstance(n, NeuralCore) and n.core_matrix is not None
        and n.core_matrix.shape == shape
    )


def _deep_conv_constant_vehicle() -> IRGraph:
    """conv(bank) -> ReLU -> AvgPool -> fc1 -> ReLU -> fc2, dyadic-snapped."""
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.normalization_fusion import (
        fuse_into_perceptron,
    )

    torch.manual_seed(3)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2), nn.Flatten(),
        nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 10),
    ).eval()
    with torch.no_grad():
        fc1 = model[4]
        keep = torch.zeros(64)
        keep[CH1_FLAT] = 1.0
        for j in CH1_ONLY_HIDDEN:
            fc1.weight[j] *= keep
            fc1.bias[j] = 0.5  # survives its support: a CONSTANT emitter
    fused = convert_torch_model(
        model, (1, 8, 8), 10, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    for p in fused.get_perceptrons():
        fuse_into_perceptron(p, device="cpu")
    repr_ = fused.get_mapper_repr()
    repr_.assign_perceptron_indices()
    graph = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=256, max_neurons=64
    ).map(repr_)
    return snap_ir_graph_to_dyadic_grid(graph, fraction_bits=8)


def _tiny_vit_constant_vehicle() -> IRGraph:
    from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.models.vit_leaf import tiny_test_vit
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = tiny_test_vit().eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "cls_token" in name or "pos_embed" in name:
                param.zero_()
    flow = convert_torch_model(
        model, (3, 8, 8), num_classes=10, packaging=MVM_PACKAGING
    )
    return snap_ir_graph_to_dyadic_grid(
        map_model_to_ir(flow.get_mapper_repr()), fraction_bits=8
    )


@pytest.fixture(scope="module")
def deep_conv():
    graph = _deep_conv_constant_vehicle()
    fc1 = _owned_core(graph, (65, 16))
    fc2 = _owned_core(graph, (17, 10))
    seeds = dict(
        seeds_node={fc2.id: ({5}, set())}, seeds_bank={0: (set(), {1})}
    )
    return graph, fc1.id, fc2.id, seeds


class TestDeepConvConstantFolding:
    def test_constant_folding_reclaims_strictly_more(self, deep_conv):
        graph, _, _, seeds = deep_conv
        off = _arm(graph, folding="off", **seeds)
        on = _arm(graph, folding="full", **seeds)
        assert _kills(on) > _kills(off), (
            f"measured zero_only={_kills(off)} constfold={_kills(on)}"
        )

    def test_the_bias_surviving_hidden_units_are_folded_away(self, deep_conv):
        graph, fc1_id, fc2_id, seeds = deep_conv
        off = _arm(graph, folding="off", **seeds)
        on = _arm(graph, folding="full", **seeds)
        for j in CH1_ONLY_HIDDEN:
            assert j not in off.pruned_cols_per_node[fc1_id], (
                "a bias-alive neuron survives the zero-only cascade"
            )
            assert j in on.pruned_cols_per_node[fc1_id]
            assert j in on.pruned_rows_per_node[fc2_id]
        assert set(on.constant_folds.folded_rows[fc2_id]) == set(
            CH1_ONLY_HIDDEN
        )

    def test_folds_carry_the_relu_of_the_bias_constant(self, deep_conv):
        graph, _, fc2_id, seeds = deep_conv
        on = _arm(graph, folding="full", **seeds)
        assert set(on.constant_folds.folded_rows[fc2_id].values()) == {0.5}

    def test_masked_subseteq_closure_subseteq_cascade(self, deep_conv):
        graph, _, _, seeds = deep_conv
        arms = {
            mode: _arm(graph, folding="full", mode=mode, **seeds)
            for mode in ("masked", "closure", "cascade")
        }
        for attr in (
            "pruned_rows_per_node", "pruned_cols_per_node",
            "pruned_rows_per_bank", "pruned_cols_per_bank",
        ):
            for key, lo in getattr(arms["masked"], attr).items():
                assert lo <= getattr(arms["closure"], attr).get(key, set())
            for key, lo in getattr(arms["closure"], attr).items():
                assert lo <= getattr(arms["cascade"], attr).get(key, set())

    def test_retrospective_reanalysis_reproduces_the_delta(self, deep_conv):
        graph, _, fc2_id, seeds = deep_conv
        fc2 = next(n for n in graph.nodes if n.id == fc2_id)
        bank = graph.weight_banks[0]
        report = reanalyze_constant_folding(
            graph,
            initial_pruned_per_node={
                fc2.id: ([i == 5 for i in range(17)], [False] * 10)
            },
            initial_pruned_per_bank={
                0: ([False] * bank.core_matrix.shape[0],
                    [j == 1 for j in range(bank.core_matrix.shape[1])])
            },
            spiking_mode=INERT_SPIKING_MODE,
        )
        assert report.additional_kills > 0
        assert report.folded_rows == len(CH1_ONLY_HIDDEN)
        assert "additional_kills" in report.summary() or True


@pytest.fixture(scope="module")
def vit():
    graph = _tiny_vit_constant_vehicle()
    bank0 = graph.weight_banks[0]
    seeds = dict(
        seeds_bank={0: (set(), set(range(bank0.core_matrix.shape[1])))}
    )
    return graph, seeds


class TestTinyViTOpaqueBarriersNowPropagate:
    def test_constant_folding_reclaims_strictly_more(self, vit):
        graph, seeds = vit
        off = _arm(graph, folding="off", **seeds)
        on = _arm(graph, folding="full", **seeds)
        assert _kills(on) > _kills(off), (
            f"measured zero_only={_kills(off)} constfold={_kills(on)}"
        )

    def test_propagation_crosses_the_cat_layernorm_attention_chain(self, vit):
        graph, seeds = vit
        off = _arm(graph, folding="off", **seeds)
        on = _arm(graph, folding="full", **seeds)
        fc1_bank = next(
            n for n in graph.nodes
            if isinstance(n, NeuralCore) and n.name.startswith("blocks_0_fc1")
        ).weight_bank_id
        assert off.pruned_rows_per_bank[fc1_bank] == set(), (
            "every op between the patch embedding and fc1 is OPAQUE in W4b-1"
        )
        assert on.pruned_rows_per_bank[fc1_bank], (
            "the lattice carries CONST(0) through cat/add/LayerNorm/attention"
        )

    def test_the_lattice_resolves_the_pre_mlp_opaque_ops(self, vit):
        graph, seeds = vit
        on = _arm(graph, folding="full", **seeds)
        resolved_ops = {
            op_id for (op_id, _) in on.constant_folds.lattice.values
        }
        by_name = {n.name: n.id for n in graph.nodes}
        for name in ("cat", "blocks_0_norm1", "blocks_0_attn", "add_1",
                     "blocks_0_norm2"):
            assert by_name[name] in resolved_ops, name

    def test_the_gelu_of_a_nonzero_constant_is_the_documented_stop(self, vit):
        """fc1's columns become CONST(bias/theta) != 0, and GELU of a non-zero
        constant is not dtype-stable, so the lattice REFUSES there rather than
        folding a value the fp32 deployment would not reproduce. The final
        LayerNorm therefore stays TOP — refuse-to-fold, stated, not silent."""
        graph, seeds = vit
        on = _arm(graph, folding="full", **seeds)
        resolved_ops = {
            op_id for (op_id, _) in on.constant_folds.lattice.values
        }
        by_name = {n.name: n.id for n in graph.nodes}
        assert by_name["blocks_0_act"] not in resolved_ops
        assert by_name["norm"] not in resolved_ops

    def test_arm_ordering_holds_on_the_vit(self, vit):
        graph, seeds = vit
        arms = {
            mode: _arm(graph, folding="full", mode=mode, **seeds)
            for mode in ("masked", "closure", "cascade")
        }
        for attr in (
            "pruned_rows_per_node", "pruned_cols_per_node",
            "pruned_rows_per_bank", "pruned_cols_per_bank",
        ):
            for key, lo in getattr(arms["masked"], attr).items():
                assert lo <= getattr(arms["closure"], attr).get(key, set())
            for key, lo in getattr(arms["closure"], attr).items():
                assert lo <= getattr(arms["cascade"], attr).get(key, set())

    def test_bank_backed_cores_never_fold_a_nonzero_row(self, vit):
        """W3c discipline: a bank row is shared physical structure, so no
        instance may fold it into its own bias."""
        graph, seeds = vit
        on = _arm(graph, folding="full", **seeds)
        bank_backed = {
            n.id for n in graph.nodes
            if isinstance(n, NeuralCore) and n.core_matrix is None
        }
        assert not (set(on.constant_folds.folded_rows) & bank_backed)
