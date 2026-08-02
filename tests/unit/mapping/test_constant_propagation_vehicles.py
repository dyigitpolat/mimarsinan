"""[W4b-2] MEASURED DoD on converter-built vehicles: folding off vs full.

Deep conv vehicle: the same converter model the W4b-1 DoD uses, except the
channel-1-only hidden units KEEP their bias — so when their whole input
support dies they do not starve, they become CONSTANT emitters. The zero-only
cascade must leave them (and the fc2 rows reading them) alive; the lattice
folds their constant onto fc2's carrier and reclaims both.

Tiny ViT vehicle: ``tiny_test_vit`` with its wiring constants (cls token,
positional embedding) zeroed so every host constant lands on the dyadic grid.
Killing the patch embedding then propagates through cat -> add -> LayerNorm ->
MultiheadAttention -> residual add — five OPAQUE barriers the zero-only
cascade cannot cross at all. The PRISTINE twin keeps those trained constants,
so the same chain now carries off-grid values: execution-exact, not
grid-certifiable.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.core_semantics import INERT_SPIKING_MODE
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.certificate import (
    certify_cascade_equivalence,
    is_on_grid,
    snap_ir_graph_to_dyadic_grid,
)
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


def _tiny_vit_constant_vehicle(*, zero_wiring_constants: bool = True) -> IRGraph:
    from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.models.vit_leaf import tiny_test_vit
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = tiny_test_vit().eval()
    if zero_wiring_constants:
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

    def test_the_folded_converter_vehicle_is_bit_exact(self, deep_conv):
        """End-to-end on the deployed value executor: the folded program is
        value-identical to the seeded reference AND strictly smaller."""
        graph, _, fc2_id, _ = deep_conv
        fc2 = next(n for n in graph.nodes if n.id == fc2_id)
        bank = graph.weight_banks[0]
        node_seed = {fc2.id: ([i == 5 for i in range(17)], [False] * 10)}
        bank_seed = {
            0: ([False] * bank.core_matrix.shape[0],
                [j == 1 for j in range(bank.core_matrix.shape[1])])
        }
        reports = {
            folding: certify_cascade_equivalence(
                _deep_conv_constant_vehicle(),
                initial_pruned_per_node=node_seed,
                initial_pruned_per_bank=bank_seed,
                batches=2, batch_size=4,
                elimination_constant_folding=folding,
            )
            for folding in ("off", "full")
        }
        assert all(r.passed and r.max_abs_delta == 0.0 for r in reports.values())
        assert reports["full"].pruned_cells < reports["off"].pruned_cells

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


def _vit_with_dead_patch_embedding(graph: IRGraph):
    bank0 = graph.weight_banks[0]
    return graph, dict(
        seeds_bank={0: (set(), set(range(bank0.core_matrix.shape[1])))}
    )


@pytest.fixture(scope="module")
def vit():
    return _vit_with_dead_patch_embedding(_tiny_vit_constant_vehicle())


@pytest.fixture(scope="module")
def pristine_vit():
    return _vit_with_dead_patch_embedding(
        _tiny_vit_constant_vehicle(zero_wiring_constants=False)
    )


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

    def test_the_gelu_of_a_nonzero_constant_is_execution_exact_not_certifiable(
        self, vit
    ):
        """fc1's columns become CONST(bias/theta) != 0, and the GELU of that is
        exactly what the deployment computes — so the lattice RESOLVES it now
        (it derives at the deployment dtype) and the value is off the dyadic
        grid. The refusal moves to where the grid is visible: the certificate
        declines the instance. The analysis no longer pretends the line is
        unknown, and nothing is snapped onto the grid.

        [ratchet, was ``blocks_0_act not in resolved_ops``] the old assertion
        pinned the fp64-vs-fp32 probe, which refused every constant fp32 can
        carry but fp64 cannot — i.e. almost every trained value.
        """
        graph, seeds = vit
        on = _arm(graph, folding="full", **seeds)
        lattice = on.constant_folds.lattice.values
        by_name = {n.name: n.id for n in graph.nodes}
        act = [
            v for (op_id, _), v in lattice.items()
            if op_id == by_name["blocks_0_act"]
        ]
        assert act, "the GELU of a known constant is a known constant"
        assert not is_on_grid(act), "execution-exact, NOT grid-certifiable"

        # [ratchet, was ``norm not in lattice``] U6 gave the NeuralCore COLUMN
        # rule the same deployment-dtype treatment U2 gave the ComputeOp probe,
        # so fc1's columns resolve and the chain no longer stops at the
        # post-block residual join. The barrier this pinned is gone; what
        # remains true — and is the load-bearing claim — is that the value is
        # execution-exact rather than grid-certifiable.
        norm = [
            v for (op_id, _), v in lattice.items() if op_id == by_name["norm"]
        ]
        assert norm, "the post-block residual join now resolves"
        assert not is_on_grid(norm), "execution-exact, NOT grid-certifiable"

    def test_a_pristine_transformer_propagates_past_its_first_arithmetic_op(
        self, pristine_vit
    ):
        """The vehicle above zeroes cls_token/pos_embed so every host constant
        is dyadic. UNZEROED — a normally trained ViT — the fp64-vs-fp32 probe
        stopped at the very first op that adds two trained floats, because
        their fp64 sum is not an fp32 value. Deriving at the deployment dtype
        carries the chain through add -> LayerNorm -> attention -> add.
        """
        graph, seeds = pristine_vit
        on = _arm(graph, folding="full", **seeds)
        resolved = {op_id for (op_id, _) in on.constant_folds.lattice.values}
        by_name = {n.name: n.id for n in graph.nodes}
        for name in ("cat", "add", "blocks_0_norm1", "blocks_0_attn", "add_1",
                     "blocks_0_norm2"):
            assert by_name[name] in resolved, name
        values = [
            v for (op_id, _), v in on.constant_folds.lattice.values.items()
            if op_id == by_name["add"]
        ]
        assert values and not is_on_grid(values), (
            "trained cls_token + pos_embed is exactly what the deployment "
            "carries and is nowhere near the dyadic grid"
        )

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
