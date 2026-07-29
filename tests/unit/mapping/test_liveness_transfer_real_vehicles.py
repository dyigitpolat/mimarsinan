"""W4b measured DoD on converter-built vehicles.

Conv vehicle (real converter, bank-backed conv + owned FCs, bank AND owned
seeds): cascade STRICTLY exceeds closure exceeds masked, the elimination
ledger reconciles (arm ordering + depth replay traverse the same transfer
maps), and the cascade certificate stays bit-exact on the dyadic-snapped
instance.

Tiny ViT vehicle: fc1 <-> fc2 propagation through the GELU ComputeOp is
nonzero in BOTH directions (forward from killed fc1 columns; backward from
killed fc2 rows — a cascade-level kill because fc1 is weight-bank-shared
across tokens), while LayerNorm / attention / residual-add boundaries still
confine the reach.

Note on the 2-layer certificate conv vehicle (`test_cascade_certificate.py::
_conv_vehicle`): with only conv -> pool -> logits, every cascade-EXCLUSIVE
operator is structurally blocked for any seed (first-layer rows are
model-input-exempt, logit columns output-exempt), so strict cascade>closure
separation requires the deeper vehicle below — same converter, one hidden FC.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.pruning.certificate import (
    certify_cascade_equivalence,
    snap_ir_graph_to_dyadic_grid,
)
from mimarsinan.mapping.pruning.elimination_ledger import (
    compute_elimination_ledger,
)
from mimarsinan.mapping.pruning.graph import compute_global_pruned_sets
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _boundary_policy_exemptions,
)
from mimarsinan.mapping.pruning.liveness_transfer import (
    COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY,
)


def _total_kills(res) -> int:
    return (
        sum(len(s) for s in res.pruned_rows_per_node.values())
        + sum(len(s) for s in res.pruned_cols_per_node.values())
        + sum(len(s) for s in res.pruned_rows_per_bank.values())
        + sum(len(s) for s in res.pruned_cols_per_bank.values())
    )


# ── conv vehicle (converter-built; bank-backed conv + two owned FCs) ─────────

CH1_FLAT = list(range(16, 32))  # conv channel 1's pooled plane after flatten
CH1_ONLY_HIDDEN = (3, 7)  # hidden units depending ONLY on channel 1


def _deep_conv_vehicle() -> IRGraph:
    """conv(bank) -> ReLU op -> AvgPool op -> fc1 -> ReLU op -> fc2 (logits).

    Two fc1 units are structurally restricted to conv channel 1's pooled
    plane (a structured mask, applied pre-conversion exactly as a pruning
    criterion would), so killing that channel starves them — an emergent,
    cascade-only kill two transfer hops downstream of the seed.
    """
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

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
            fc1.bias[j] = 0.0
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


def _owned_core(graph: IRGraph, shape) -> NeuralCore:
    return next(
        n for n in graph.nodes
        if isinstance(n, NeuralCore)
        and n.core_matrix is not None
        and n.core_matrix.shape == shape
    )


@pytest.fixture(scope="module")
def arms():
    graph = _deep_conv_vehicle()
    fc1 = _owned_core(graph, (65, 16))
    fc2 = _owned_core(graph, (17, 10))
    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    results = {
        mode: compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node={fc2.id: ({5}, set())},
            initial_per_bank={0: (set(), {1})},
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode=mode,
        )
        for mode in ("masked", "closure", "cascade")
    }
    return graph, fc1.id, fc2.id, results


class TestDeepConvVehicleArms:
    """Bank seed (conv channel 1) + owned seed (fc2 axon row 5), all arms."""

    def test_cascade_strictly_exceeds_closure_exceeds_masked(self, arms):
        _, _, _, results = arms
        masked = _total_kills(results["masked"])
        closure = _total_kills(results["closure"])
        cascade = _total_kills(results["cascade"])
        assert masked < closure < cascade, (
            f"measured masked={masked} closure={closure} cascade={cascade}"
        )

    def test_arm_ordering_is_setwise_not_just_counted(self, arms):
        _, _, _, results = arms
        for attr in (
            "pruned_rows_per_node", "pruned_cols_per_node",
            "pruned_rows_per_bank", "pruned_cols_per_bank",
        ):
            for key, lo in getattr(results["masked"], attr).items():
                assert lo <= getattr(results["closure"], attr).get(key, set())
            for key, lo in getattr(results["closure"], attr).items():
                assert lo <= getattr(results["cascade"], attr).get(key, set())

    def test_closure_reaches_fc1_through_relu_and_pool(self, arms):
        """Forward one hop: dead conv channel 1 -> its 16 pooled-plane fc1
        rows, THROUGH the ReLU and AvgPool ComputeOps (transfer chain)."""
        _, fc1_id, _, results = arms
        assert results["closure"].pruned_rows_per_node[fc1_id] >= set(CH1_FLAT)
        assert results["masked"].pruned_rows_per_node[fc1_id] == set()

    def test_closure_couples_backward_through_the_fc_relu(self, arms):
        """Backward one hop: seed-dead fc2 axon 5 orphans fc1 column 5
        through the elementwise ReLU op between the FCs."""
        _, fc1_id, _, results = arms
        assert 5 in results["closure"].pruned_cols_per_node[fc1_id]

    def test_cascade_only_emergent_starvation_and_its_forward_echo(self, arms):
        """The channel-1-only hidden units starve once their entire input
        support is dead (cascade-exclusive), and their deaths propagate
        forward into fc2 axon rows through the ReLU op."""
        _, fc1_id, fc2_id, results = arms
        for j in CH1_ONLY_HIDDEN:
            assert j not in results["closure"].pruned_cols_per_node[fc1_id]
            assert j in results["cascade"].pruned_cols_per_node[fc1_id]
            assert j not in results["closure"].pruned_rows_per_node[fc2_id]
            assert j in results["cascade"].pruned_rows_per_node[fc2_id]

    def test_identity_only_kill_switch_confines_the_cascade(self, arms):
        graph, fc1_id, fc2_id, results = arms
        exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
        old = compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node={fc2_id: ({5}, set())},
            initial_per_bank={0: (set(), {1})},
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode="cascade",
            computeop_liveness_transfers=(
                COMPUTEOP_LIVENESS_TRANSFERS_IDENTITY_ONLY
            ),
        )
        assert old.pruned_rows_per_node[fc1_id] == set(), (
            "identity_only must reproduce the pre-W4b ComputeOp barrier"
        )
        assert _total_kills(old) < _total_kills(results["cascade"])


class TestDeepConvVehicleLedgerAndCertificate:
    def test_elimination_ledger_reconciles_with_transfer_propagation(self):
        """Arm ordering + depth replay traverse the SAME transfer maps as
        production (the by-construction divergence guard), and the emergent
        kills are attributed beyond closure."""
        graph = _deep_conv_vehicle()
        fc2 = _owned_core(graph, (17, 10))
        row_seed = [i == 5 for i in range(17)]
        bank = graph.weight_banks[0]
        bank_cols = [j == 1 for j in range(bank.core_matrix.shape[1])]
        ledger = compute_elimination_ledger(
            graph,
            initial_pruned_per_node={fc2.id: (row_seed, [False] * 10)},
            initial_pruned_per_bank={
                0: ([False] * bank.core_matrix.shape[0], bank_cols)
            },
            elimination_propagation="cascade",
        )
        fc1_record = next(r for r in ledger.per_node if r.n_axons == 65)
        assert fc1_record.counts.closure_rows >= 16
        assert fc1_record.counts.emergent_cols >= len(CH1_ONLY_HIDDEN)
        # Emergent kills sit strictly deeper than the coupled wave.
        emergent_depths = [
            fc1_record.col_depths[j] for j in CH1_ONLY_HIDDEN
        ]
        assert all(d >= 2 for d in emergent_depths)

    def test_cascade_certificate_green_on_the_deep_conv_vehicle(self):
        """The new through-op kills are bit-exact on the deployed executor."""
        graph = snap_ir_graph_to_dyadic_grid(
            _deep_conv_vehicle(), fraction_bits=8
        )
        fc2 = _owned_core(graph, (17, 10))
        bank = graph.weight_banks[0]
        report = certify_cascade_equivalence(
            graph,
            initial_pruned_per_node={
                fc2.id: ([i == 5 for i in range(17)], [False] * 10)
            },
            initial_pruned_per_bank={
                0: ([False] * bank.core_matrix.shape[0],
                    [j == 1 for j in range(bank.core_matrix.shape[1])])
            },
            batches=2, batch_size=4,
        )
        assert report.passed is True
        assert report.max_abs_delta == 0.0
        assert report.pruned_cells < report.reference_cells


# ── tiny ViT vehicle (GELU MLP, weight-bank-shared per token) ────────────────


def _tiny_vit_graph() -> IRGraph:
    from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
    from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
    from mimarsinan.models.vit_leaf import tiny_test_vit
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(0)
    model = tiny_test_vit().eval()
    flow = convert_torch_model(
        model, (3, 8, 8), num_classes=10, packaging=MVM_PACKAGING
    )
    return map_model_to_ir(flow.get_mapper_repr())


@pytest.fixture(scope="module")
def vit():
    graph = _tiny_vit_graph()
    fc1_nodes = [
        n for n in graph.nodes
        if isinstance(n, NeuralCore) and n.name.startswith("blocks_0_fc1")
    ]
    fc2_nodes = [
        n for n in graph.nodes
        if isinstance(n, NeuralCore) and n.name.startswith("blocks_0_fc2")
    ]
    assert fc1_nodes and fc2_nodes
    gelu = next(
        n for n in graph.nodes
        if isinstance(n, ComputeOp) and n.op_type == "GELU"
    )
    assert gelu is not None
    exempt = _boundary_policy_exemptions(graph)
    return graph, fc1_nodes, fc2_nodes, exempt


class TestTinyViTGELUPropagation:
    def _run(self, vit, mode, bank_seeds):
        graph, _, _, (exempt_rows, exempt_cols) = vit
        return compute_global_pruned_sets(
            graph,
            zero_threshold=1e-8,
            initial_per_node=None,
            initial_per_bank=bank_seeds,
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode=mode,
        )

    def test_forward_fc1_cols_kill_fc2_rows_through_gelu(self, vit):
        graph, fc1_nodes, fc2_nodes, _ = vit
        fc1_bank = fc1_nodes[0].weight_bank_id
        fc2_bank = fc2_nodes[0].weight_bank_id
        seeds = {fc1_bank: (set(), {2, 3})}
        masked = self._run(vit, "masked", seeds)
        cascade = self._run(vit, "cascade", seeds)
        for node in fc2_nodes:
            assert masked.pruned_rows_per_node[node.id] == set()
            assert cascade.pruned_rows_per_node[node.id] >= {2, 3}
        assert cascade.pruned_rows_per_bank[fc2_bank] >= {2, 3}
        forward_reach = _total_kills(cascade) - _total_kills(masked)
        assert forward_reach >= 2 * len(fc2_nodes) + 2  # rows/token + bank rows

    def test_backward_fc2_rows_kill_fc1_cols_through_gelu(self, vit):
        graph, fc1_nodes, fc2_nodes, _ = vit
        fc1_bank = fc1_nodes[0].weight_bank_id
        fc2_bank = fc2_nodes[0].weight_bank_id
        seeds = {fc2_bank: ({4, 5}, set())}
        closure = self._run(vit, "closure", seeds)
        cascade = self._run(vit, "cascade", seeds)
        for node in fc1_nodes:
            # Weight-bank-shared producers orphan only at cascade level (the
            # shared-bank union rule needs every token's view dead).
            assert closure.pruned_cols_per_node[node.id] == set()
            assert cascade.pruned_cols_per_node[node.id] >= {4, 5}
        assert cascade.pruned_cols_per_bank[fc1_bank] >= {4, 5}

    def test_layernorm_attention_and_residual_still_confine(self, vit):
        graph, fc1_nodes, fc2_nodes, _ = vit
        fc1_bank = fc1_nodes[0].weight_bank_id
        fc2_bank = fc2_nodes[0].weight_bank_id
        cascade = self._run(
            vit, "cascade",
            {fc1_bank: (set(), {2, 3}), fc2_bank: ({4, 5}, set())},
        )
        mlp_ids = {n.id for n in fc1_nodes} | {n.id for n in fc2_nodes}
        mlp_banks = {fc1_bank, fc2_bank}
        # fc2 columns feed the residual add (opaque): never pruned.
        for node in fc2_nodes:
            assert cascade.pruned_cols_per_node[node.id] == set()
        # fc1 rows read LayerNorm outputs (opaque): never pruned.
        for node in fc1_nodes:
            assert cascade.pruned_rows_per_node[node.id] == set()
        # Nothing outside the MLP pair moves: patch embed, attention side,
        # head all stay untouched.
        for nid, rows in cascade.pruned_rows_per_node.items():
            if nid not in mlp_ids:
                assert rows == set(), f"leak into node {nid}"
        for nid, cols in cascade.pruned_cols_per_node.items():
            if nid not in mlp_ids:
                assert cols == set(), f"leak into node {nid}"
        for bid in graph.weight_banks:
            if bid not in mlp_banks:
                assert cascade.pruned_rows_per_bank[bid] == set()
                assert cascade.pruned_cols_per_bank[bid] == set()

    def test_no_seeds_no_kills(self, vit):
        cascade = self._run(vit, "cascade", None)
        assert _total_kills(cascade) == 0
