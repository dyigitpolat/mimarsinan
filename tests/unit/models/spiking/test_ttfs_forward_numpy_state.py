"""_forward_ttfs keeps a numpy state buffer: no per-stage torch round-trips (W3 wall)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    gather_final_output_numpy,
)
from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow


def _two_core_ir(in_dim=6, hidden=5, out_dim=3, seed=7):
    rng = np.random.default_rng(seed)
    w1 = np.abs(rng.normal(size=(in_dim + 1, hidden))) * 0.5
    w2 = np.abs(rng.normal(size=(hidden + 1, out_dim))) * 0.5
    core1 = NeuralCore(
        id=0,
        name="hidden",
        core_matrix=w1,
        input_sources=np.array(
            [IRSource(node_id=-2, index=i) for i in range(in_dim)]
            + [IRSource(node_id=-3, index=0)],
            dtype=object,
        ),
        threshold=1.0,
        latency=0,
    )
    core2 = NeuralCore(
        id=1,
        name="output",
        core_matrix=w2,
        input_sources=np.array(
            [IRSource(node_id=0, index=i) for i in range(hidden)]
            + [IRSource(node_id=-3, index=0)],
            dtype=object,
        ),
        threshold=1.0,
        latency=1,
    )
    output_sources = np.array(
        [IRSource(node_id=1, index=i) for i in range(out_dim)], dtype=object,
    )
    return IRGraph(nodes=[core1, core2], output_sources=output_sources)


def _make_flow(spiking_mode="ttfs_quantized", T=4, in_dim=6):
    return build_identity_spiking_flow(
        input_shape=(in_dim,),
        ir_graph=_two_core_ir(in_dim=in_dim),
        simulation_length=T,
        preprocessor=nn.Identity(),
        firing_mode="TTFS",
        spike_mode="TTFS",
        thresholding_mode="<=",
        spiking_mode=spiking_mode,
    )


class TestForwardMatchesContractRunner:
    def test_forward_bit_matches_the_canonical_contract_path(self):
        """flow(x) for B=1 must equal the pure-numpy contract runner's gather
        exactly — same functions, same float64 buffers, no round-trip drift."""
        T = 4
        in_dim = 6
        flow = _make_flow(T=T, in_dim=in_dim)
        rng = np.random.default_rng(11)
        for i in range(3):
            x_np = rng.uniform(0, 1, size=(1, in_dim)).astype(np.float64)
            with torch.no_grad():
                got = flow(torch.tensor(x_np, dtype=torch.float32))
            contract = run_ttfs_hybrid_contract(
                flow.hybrid_mapping,
                x_np,
                simulation_length=T,
                spiking_mode="ttfs_quantized",
                sample_index=i,
            )
            want = gather_final_output_numpy(
                flow.hybrid_mapping.output_sources,
                contract.state_buffer,
                x_np,
                1,
            ) * float(T)
            np.testing.assert_allclose(
                got.cpu().numpy(), want, rtol=0, atol=0,
            )

    def test_batched_forward_matches_per_sample_rows(self):
        T = 4
        in_dim = 6
        flow = _make_flow(T=T, in_dim=in_dim)
        rng = np.random.default_rng(13)
        x_np = rng.uniform(0, 1, size=(5, in_dim)).astype(np.float64)
        x = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            batched = flow(x)
            rows = torch.cat([flow(x[i : i + 1]) for i in range(x.shape[0])])
        np.testing.assert_allclose(
            batched.cpu().numpy(), rows.cpu().numpy(), rtol=0, atol=1e-12,
        )

    def test_forward_output_dtype_and_scale(self):
        flow = _make_flow()
        x = torch.rand(2, 6)
        with torch.no_grad():
            out = flow(x)
        assert out.dtype == torch.float32
        assert out.shape == (2, 3)
