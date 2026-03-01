"""
Tests for TTFS (Time-to-First-Spike) spiking simulation.

Migrated from tests/test_ttfs.py to pytest format.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.ir import NeuralCore, IRGraph, IRSource
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.hybrid_core_flow import SpikingHybridCoreFlow
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_relu_model(in_dim, hidden_dim, out_dim):
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=True),
    )
    with torch.no_grad():
        model[0].weight.data = torch.abs(model[0].weight.data) * 0.5
        model[0].bias.data = torch.abs(model[0].bias.data) * 0.1
        model[2].weight.data = torch.abs(model[2].weight.data) * 0.5
        model[2].bias.data = torch.abs(model[2].bias.data) * 0.1
    return model


def _relu_model_to_ir_graph(model, in_dim, *, quantize=False, weight_bits=8):
    w1 = model[0].weight.data.numpy()
    b1 = model[0].bias.data.numpy()
    w2 = model[2].weight.data.numpy()
    b2 = model[2].bias.data.numpy()

    hidden_dim = w1.shape[0]
    out_dim = w2.shape[0]
    q_max = (2 ** (weight_bits - 1)) - 1

    core1_matrix = np.vstack([w1.T, b1.reshape(1, -1)])
    abs_max_1 = max(np.abs(core1_matrix).max(), 1e-12)
    ps1 = q_max / abs_max_1

    input_sources_1 = np.array(
        [IRSource(node_id=-2, index=i) for i in range(in_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )

    if quantize:
        core1_matrix_q = np.round(core1_matrix * ps1)
        threshold_1 = ps1
    else:
        core1_matrix_q = core1_matrix
        threshold_1 = 1.0

    core1 = NeuralCore(
        id=0, name="hidden",
        input_sources=input_sources_1,
        core_matrix=core1_matrix_q,
        threshold=threshold_1,
        parameter_scale=torch.tensor(1.0) if quantize else torch.tensor(ps1),
        latency=0,
    )

    core2_matrix = np.vstack([w2.T, b2.reshape(1, -1)])
    abs_max_2 = max(np.abs(core2_matrix).max(), 1e-12)
    ps2 = q_max / abs_max_2

    input_sources_2 = np.array(
        [IRSource(node_id=0, index=i) for i in range(hidden_dim)]
        + [IRSource(node_id=-3, index=0)],
        dtype=object,
    )

    if quantize:
        core2_matrix_q = np.round(core2_matrix * ps2)
        threshold_2 = ps2
    else:
        core2_matrix_q = core2_matrix
        threshold_2 = 1.0

    core2 = NeuralCore(
        id=1, name="output",
        input_sources=input_sources_2,
        core_matrix=core2_matrix_q,
        threshold=threshold_2,
        parameter_scale=torch.tensor(1.0) if quantize else torch.tensor(ps2),
        latency=1,
    )

    output_sources = np.array(
        [IRSource(node_id=1, index=i) for i in range(out_dim)],
        dtype=object,
    )
    return IRGraph(nodes=[core1, core2], output_sources=output_sources)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTTFSEncoding:
    def test_ttfs_encode_input(self):
        T = 16
        ir_graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        flow = SpikingUnifiedCoreFlow(
            input_shape=(5,), ir_graph=ir_graph, simulation_length=T,
            preprocessor=nn.Identity(), firing_mode="TTFS",
            spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        activations = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]])
        spike_train = flow._ttfs_encode_input(activations)

        for i in range(5):
            spike_cycles = (spike_train[:, 0, i] > 0).nonzero(as_tuple=True)[0].tolist()
            expected_time = round(T * (1.0 - activations[0, i].item()))
            if expected_time < T:
                assert len(spike_cycles) == 1, \
                    f"act={activations[0,i]:.2f}: expected 1 spike, got {len(spike_cycles)}"
                assert spike_cycles[0] == expected_time
            else:
                assert len(spike_cycles) == 0


class TestTTFSUnifiedCoreFlow:
    def test_argmax_matches_relu(self):
        torch.manual_seed(42)
        np.random.seed(42)

        in_dim, hidden_dim, out_dim = 8, 16, 4
        T = 64

        model = _make_simple_relu_model(in_dim, hidden_dim, out_dim)
        x = torch.rand(8, in_dim)

        with torch.no_grad():
            relu_preds = model(x).argmax(dim=1)

        ir_graph = _relu_model_to_ir_graph(model, in_dim, quantize=False)
        flow = SpikingUnifiedCoreFlow(
            input_shape=(in_dim,), ir_graph=ir_graph, simulation_length=T,
            preprocessor=nn.Identity(), firing_mode="TTFS",
            spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        with torch.no_grad():
            ttfs_preds = flow(x).argmax(dim=1)

        agreement = (relu_preds == ttfs_preds).float().mean().item()
        assert agreement == 1.0, f"Agreement {agreement*100:.0f}% < 100%"


class TestTTFSHybridCoreFlow:
    def test_argmax_matches_relu(self):
        torch.manual_seed(42)
        np.random.seed(42)

        in_dim, hidden_dim, out_dim = 8, 16, 4
        T = 64

        model = _make_simple_relu_model(in_dim, hidden_dim, out_dim)
        x = torch.rand(8, in_dim)

        with torch.no_grad():
            relu_preds = model(x).argmax(dim=1)

        ir_graph = _relu_model_to_ir_graph(model, in_dim, quantize=False)
        cores_config = [{"max_axons": 256, "max_neurons": 256, "count": 10}]
        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph, cores_config=cores_config,
        )

        flow = SpikingHybridCoreFlow(
            input_shape=(in_dim,), hybrid_mapping=hybrid_mapping,
            simulation_length=T, preprocessor=nn.Identity(),
            firing_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        with torch.no_grad():
            ttfs_preds = flow(x).argmax(dim=1)

        agreement = (relu_preds == ttfs_preds).float().mean().item()
        assert agreement == 1.0


class TestTTFSQuantized:
    def test_quantized_agreement(self):
        torch.manual_seed(42)
        np.random.seed(42)

        in_dim, hidden_dim, out_dim = 16, 32, 10
        T = 32

        model = _make_simple_relu_model(in_dim, hidden_dim, out_dim)
        x = torch.rand(32, in_dim)

        with torch.no_grad():
            relu_preds = model(x).argmax(dim=1)

        ir_graph = _relu_model_to_ir_graph(model, in_dim, quantize=True, weight_bits=8)
        unified = SpikingUnifiedCoreFlow(
            input_shape=(in_dim,), ir_graph=ir_graph, simulation_length=T,
            preprocessor=nn.Identity(), firing_mode="TTFS",
            spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        with torch.no_grad():
            ttfs_preds = unified(x).argmax(dim=1)

        agreement = (relu_preds == ttfs_preds).float().mean().item()
        assert agreement >= 0.8, f"Quantized agreement {agreement*100:.0f}% < 80%"


class TestTTFSFireOnce:
    def test_output_nonnegative(self):
        torch.manual_seed(0)
        in_dim, hidden_dim, out_dim = 4, 8, 2
        T = 32

        model = _make_simple_relu_model(in_dim, hidden_dim, out_dim)
        ir_graph = _relu_model_to_ir_graph(model, in_dim)

        flow = SpikingUnifiedCoreFlow(
            input_shape=(in_dim,), ir_graph=ir_graph, simulation_length=T,
            preprocessor=nn.Identity(), firing_mode="TTFS",
            spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        x = torch.rand(4, in_dim)
        with torch.no_grad():
            out = flow(x)

        assert (out >= -1e-6).all(), f"Min output: {out.min().item()}"


class TestUnifiedVsHybridTTFS:
    def test_agreement(self):
        torch.manual_seed(42)
        np.random.seed(42)

        in_dim, hidden_dim, out_dim = 8, 16, 4
        T = 64

        model = _make_simple_relu_model(in_dim, hidden_dim, out_dim)
        x = torch.rand(4, in_dim)

        ir_graph = _relu_model_to_ir_graph(model, in_dim)

        unified = SpikingUnifiedCoreFlow(
            input_shape=(in_dim,), ir_graph=ir_graph, simulation_length=T,
            preprocessor=nn.Identity(), firing_mode="TTFS",
            spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        cores_config = [{"max_axons": 256, "max_neurons": 256, "count": 10}]
        hybrid_mapping = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph, cores_config=cores_config,
        )
        hybrid = SpikingHybridCoreFlow(
            input_shape=(in_dim,), hybrid_mapping=hybrid_mapping,
            simulation_length=T, preprocessor=nn.Identity(),
            firing_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        with torch.no_grad():
            u_out = unified(x)
            h_out = hybrid(x)

        u_preds = u_out.argmax(dim=1)
        h_preds = h_out.argmax(dim=1)

        agreement = (u_preds == h_preds).float().mean().item()

        assert agreement >= 0.75, f"Agreement {agreement*100:.0f}% < 75%"


class TestTTFSRealisticMixedWeights:
    def test_mixed_weight_quantized_agreement(self):
        torch.manual_seed(123)
        np.random.seed(123)

        in_dim, hidden_dim, out_dim = 16, 32, 10
        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
        x = torch.rand(64, in_dim)

        with torch.no_grad():
            relu_preds = model(x).argmax(dim=1)

        ir_graph = _relu_model_to_ir_graph(model, in_dim, quantize=True, weight_bits=8)
        flow = SpikingUnifiedCoreFlow(
            input_shape=(in_dim,), ir_graph=ir_graph, simulation_length=32,
            preprocessor=nn.Identity(), firing_mode="TTFS",
            spike_mode="TTFS", thresholding_mode="<=", spiking_mode="ttfs",
        )

        with torch.no_grad():
            ttfs_preds = flow(x).argmax(dim=1)

        agreement = (relu_preds == ttfs_preds).float().mean().item()
        assert agreement >= 0.7, f"Mixed weight agreement {agreement*100:.0f}% < 70%"
