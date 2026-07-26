"""[wsm V1] FC bank emission: token-instanced Linears share one WeightBank."""

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model


def _ir_for(model, input_shape, *, max_axons=64, max_neurons=64,
            allow_coalescing=False):
    flow = convert_torch_model(
        model.eval(), input_shape, 4, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=max_axons, max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
    ).map(repr_)
    return flow, ir


def _fc_cores(ir):
    return [n for n in ir.nodes if type(n).__name__ == "NeuralCore"]


class TestTokenInstancedFCSharesOneBank:
    def test_three_tokens_one_bank(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 10)).eval()
        _, ir = _ir_for(model, (3, 6))
        cores = _fc_cores(ir)
        assert len(cores) == 3
        bank_ids = {c.weight_bank_id for c in cores}
        assert len(bank_ids) == 1 and None not in bank_ids
        assert all(c.core_matrix is None for c in cores)
        assert sorted(c.perceptron_output_column for c in cores) == [0, 1, 2]
        assert len(ir.weight_banks) == 1

    def test_bank_matrix_carries_weights_and_bias_row(self):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(6, 10)).eval()
        _, ir = _ir_for(model, (3, 6))
        core = _fc_cores(ir)[0]
        mat = core.get_core_matrix(ir)
        lin = model[0]
        np.testing.assert_allclose(
            mat[:-1, :], lin.weight.detach().numpy().T, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            mat[-1, :], lin.bias.detach().numpy(), rtol=0, atol=0
        )

    def test_output_tiled_tokens_share_per_tile_banks(self):
        torch.manual_seed(2)
        model = nn.Sequential(nn.Linear(6, 10)).eval()
        _, ir = _ir_for(model, (3, 6), max_neurons=4)
        cores = _fc_cores(ir)
        # 3 tiles (0-4, 4-8, 8-10) x 3 tokens.
        assert len(cores) == 9
        assert len(ir.weight_banks) == 3
        by_slice = {}
        for c in cores:
            assert c.weight_bank_id is not None
            by_slice.setdefault(c.perceptron_output_slice, set()).add(
                c.weight_bank_id
            )
        assert set(by_slice) == {(0, 4), (4, 8), (8, 10)}
        # Every tile's tokens share exactly one bank.
        assert all(len(v) == 1 for v in by_slice.values())

    def test_single_instance_fc_stays_owned(self):
        torch.manual_seed(3)
        model = nn.Sequential(nn.Linear(6, 10)).eval()
        _, ir = _ir_for(model, (6,))
        cores = _fc_cores(ir)
        assert len(cores) == 1
        assert cores[0].weight_bank_id is None
        assert cores[0].core_matrix is not None
        assert len(ir.weight_banks) == 0

    def test_wide_fanin_coalescing_tokens_stay_owned_v1(self):
        # The mapper-level coalescing x bank interaction is the flagged
        # unknown; V1 pins the owned fallback for wide-fan-in token FCs.
        torch.manual_seed(4)
        model = nn.Sequential(nn.Linear(100, 8)).eval()
        _, ir = _ir_for(model, (2, 100), max_axons=64, allow_coalescing=True)
        cores = _fc_cores(ir)
        assert cores, "wide fan-in must still map"
        assert all(c.weight_bank_id is None for c in cores)
        assert len(ir.weight_banks) == 0


class TestValueEquivalenceWithBanks:
    def test_identity_program_matches_fused_flow(self):
        from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
        from mimarsinan.mapping.packing.hybrid_build_pool import (
            build_identity_hybrid_mapping,
        )
        from mimarsinan.transformations.normalization_fusion import (
            fuse_into_perceptron,
        )

        torch.manual_seed(5)
        model = nn.Sequential(
            nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 4)
        ).eval()
        flow = convert_torch_model(
            model, (3, 6), 4, device="cpu", packaging=MVM_PACKAGING
        ).eval()
        for p in flow.get_perceptrons():
            fuse_into_perceptron(p, device="cpu")
        repr_ = flow.get_mapper_repr()
        repr_.assign_perceptron_indices()
        ir = IRMapping(
            q_max=127.0, firing_mode="Default", max_axons=64, max_neurons=64
        ).map(repr_)
        assert len(ir.weight_banks) >= 2  # both linears bank-shared
        vflow = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=ir), dtype=torch.float64
        )
        x = torch.randn(4, 3, 6)
        with torch.no_grad():
            got = vflow(x)
            want = flow.double()(x.double()).reshape(4, -1)
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)


class TestWeightReuseSeesFCBanks:
    def test_reuse_phases_count_token_instances(self):
        from mimarsinan.mapping.weight_reuse import weight_reuse_plan_from_graph

        torch.manual_seed(6)
        model = nn.Sequential(nn.Linear(6, 10)).eval()
        _, ir = _ir_for(model, (5, 6))
        plan = weight_reuse_plan_from_graph(ir)
        assert plan.reprogram_passes == 1
        assert plan.reuse_passes == 4
