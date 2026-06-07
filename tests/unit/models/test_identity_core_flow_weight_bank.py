"""Tests for the rung-2 identity executor with weight-bank-backed cores."""

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow


def _inp(idx):
    return IRSource(node_id=-2, index=idx)


def _on():
    return IRSource(node_id=-3, index=0)


def _placement_bank_counts(flow):
    """Return (num_banks, bank_backed_placements, owned_placements) for the flow."""
    banks: dict[int, object] = {}
    bank_backed = 0
    owned = 0
    for segment in flow.hybrid_mapping.get_neural_segments():
        banks.update(segment.weight_banks)
        for placements in segment.soft_core_placements_per_hard_core:
            for placement in placements:
                if placement.get("weight_bank_id") is not None:
                    bank_backed += 1
                else:
                    owned += 1
    return len(banks), bank_backed, owned


def _build_conv_like_graph(n_positions=4, axons=3, neurons=2):
    """Build a tiny graph simulating a conv: one bank, N cores."""
    rng = np.random.RandomState(0)
    bank_mat = rng.randn(axons + 1, neurons).astype(np.float32)
    bank = WeightBank(
        id=0, core_matrix=bank_mat,
        activation_scale=torch.tensor(1.0),
        parameter_scale=torch.tensor(1.0),
        input_activation_scale=torch.tensor(1.0),
    )

    nodes = []
    all_out = []
    for i in range(n_positions):
        srcs = np.array([_inp(j) for j in range(axons)] + [_on()])
        core = NeuralCore(
            id=i, name=f"conv_pos{i}",
            input_sources=srcs,
            core_matrix=None,
            weight_bank_id=0,
            weight_row_slice=(0, neurons),
        )
        nodes.append(core)
        all_out.extend([IRSource(node_id=i, index=k) for k in range(neurons)])

    return IRGraph(
        nodes=nodes,
        output_sources=np.array(all_out),
        weight_banks={0: bank},
    )


class TestBankParamRegistration:
    def test_single_bank_many_cores(self):
        graph = _build_conv_like_graph(n_positions=16, axons=5, neurons=3)
        flow = build_identity_spiking_flow(
            input_shape=(5,), ir_graph=graph, simulation_length=4,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        # All 16 cores share the single bank; none are owned (no per-core matrix).
        num_banks, bank_backed, owned = _placement_bank_counts(flow)
        assert num_banks == 1
        assert owned == 0
        assert bank_backed == 16


class TestForwardWithBanks:
    def test_ttfs_continuous(self):
        graph = _build_conv_like_graph(n_positions=4, axons=3, neurons=2)
        flow = build_identity_spiking_flow(
            input_shape=(3,), ir_graph=graph, simulation_length=4,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        x = torch.rand(2, 3)
        out = flow(x)
        assert out.shape == (2, 8)

    def test_rate_mode(self):
        graph = _build_conv_like_graph(n_positions=2, axons=3, neurons=2)
        flow = build_identity_spiking_flow(
            input_shape=(3,), ir_graph=graph, simulation_length=4,
            preprocessor=nn.Identity(), spiking_mode="rate",
        )
        x = torch.rand(2, 3)
        out = flow(x)
        assert out.shape == (2, 4)


class TestMixedOwnedAndBank:
    def test_forward(self):
        bank_mat = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        bank = WeightBank(id=0, core_matrix=bank_mat)

        nodes = [
            NeuralCore(
                id=0, name="shared0",
                input_sources=np.array([_inp(0), _inp(1), _on()]),
                core_matrix=None, weight_bank_id=0, weight_row_slice=(0, 2),
            ),
            NeuralCore(
                id=1, name="owned_fc",
                input_sources=np.array([
                    IRSource(node_id=0, index=0),
                    IRSource(node_id=0, index=1),
                ]),
                core_matrix=np.array([[1.0], [1.0]], dtype=np.float32),
            ),
        ]
        graph = IRGraph(
            nodes=nodes,
            output_sources=np.array([IRSource(node_id=1, index=0)]),
            weight_banks={0: bank},
        )
        flow = build_identity_spiking_flow(
            input_shape=(2,), ir_graph=graph, simulation_length=4,
            preprocessor=nn.Identity(), spiking_mode="ttfs",
        )
        x = torch.tensor([[0.5, 0.3]])
        out = flow(x)
        assert out.shape == (1, 1)

        num_banks, bank_backed, owned = _placement_bank_counts(flow)
        assert num_banks == 1
        assert bank_backed == 1
        assert owned == 1
