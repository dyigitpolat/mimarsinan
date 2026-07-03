"""Tests for the on-chip attention/LayerNorm mappability frontier (D5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.onchip_attention import (
    build_centering_matrix,
    OnchipLayerNormCentering,
    affine_realizability_report,
    AffineRealizability,
)


def _run_ir_graph(graph: IRGraph, x: torch.Tensor) -> dict[int, torch.Tensor]:
    """Execute every NeuralCore once in id order; return per-node buffers (one pass: rails read only the chip input)."""
    buffers: dict[int, torch.Tensor] = {}
    for node in graph.nodes:
        buffers[node.id] = node.execute(x, buffers)
    return buffers


def test_centering_matrix_is_mean_subtraction():
    N = 7
    C = build_centering_matrix(N)
    x = torch.randn(5, N, dtype=torch.float64)
    ref = x - x.mean(dim=-1, keepdim=True)
    got = x @ torch.tensor(C.T, dtype=torch.float64)
    assert torch.allclose(ref, got, atol=1e-12), (ref - got).abs().max().item()


def test_centering_matrix_row_sums_zero():
    C = build_centering_matrix(9)
    assert np.allclose(C.sum(axis=1), 0.0, atol=1e-12)


def test_onchip_centering_emits_only_neural_cores():
    mapper = OnchipLayerNormCentering(num_features=6)
    graph = mapper.to_ir_graph()
    assert isinstance(graph, IRGraph)
    assert graph.validate() == []
    assert graph.get_compute_ops() == []
    cores = graph.get_neural_cores()
    assert len(cores) == 2
    for core in cores:
        assert isinstance(core, NeuralCore)


def test_onchip_centering_is_deterministic_bit_exact():
    N = 8
    mapper = OnchipLayerNormCentering(num_features=N, clamp_ceiling=1e6)
    graph = mapper.to_ir_graph()
    x = torch.randn(11, N, dtype=torch.float64) * 0.3
    got_a = mapper.reconstruct_centered(_run_ir_graph(graph, x))
    got_b = mapper.reconstruct_centered(_run_ir_graph(graph, x))
    assert torch.equal(got_a, got_b)


def test_onchip_centering_bounded_to_math_reference():
    N = 8
    mapper = OnchipLayerNormCentering(num_features=N, clamp_ceiling=1e6)
    graph = mapper.to_ir_graph()
    x = torch.randn(11, N, dtype=torch.float64) * 0.3
    buffers = _run_ir_graph(graph, x)
    got = mapper.reconstruct_centered(buffers)
    assert got.dtype == torch.float32

    ref = x - x.mean(dim=-1, keepdim=True)
    max_abs = (ref.to(torch.float32) - got).abs().max().item()
    assert max_abs <= 1.2e-6, max_abs


def test_onchip_centering_idempotent_double_apply():
    N = 5
    C = build_centering_matrix(N)
    assert np.allclose(C @ C, C, atol=1e-12)


def test_centering_clamp_ceiling_is_a_characterized_bound():
    N = 4
    ceiling = 0.5
    mapper = OnchipLayerNormCentering(num_features=N, clamp_ceiling=ceiling)
    assert mapper.exact_input_bound() == ceiling
    graph = mapper.to_ir_graph()
    x = torch.full((1, N), 0.0, dtype=torch.float64)
    x[0, 0] = 10.0
    buffers = _run_ir_graph(graph, x)
    got = mapper.reconstruct_centered(buffers)
    ref = (x - x.mean(dim=-1, keepdim=True)).to(torch.float32)
    assert not torch.allclose(ref, got, atol=1e-6)


def test_qkt_is_bilinear_not_affine():
    report = affine_realizability_report("qk_score")
    assert report.realizable is False
    assert report.reason_code == "bilinear_data_dependent"

    d = 4
    q = torch.randn(d, dtype=torch.float64, requires_grad=True)
    k = torch.randn(d, dtype=torch.float64, requires_grad=True)
    score = (q * k).sum()
    grad_q = torch.autograd.grad(score, q, create_graph=True)[0]
    cross_hessian = torch.autograd.grad(grad_q.sum(), k)[0]
    assert torch.allclose(cross_hessian, torch.ones(d, dtype=torch.float64))


def test_softmax_is_not_piecewise_linear():
    report = affine_realizability_report("softmax")
    assert report.realizable is False
    assert report.reason_code == "nonlinear_exp_normalize"

    z = torch.randn(5, dtype=torch.float64, requires_grad=True)
    s = torch.softmax(z, dim=-1)
    j0 = torch.autograd.functional.jacobian(lambda v: torch.softmax(v, -1), z)
    j1 = torch.autograd.functional.jacobian(lambda v: torch.softmax(v, -1), z * 3.0)
    assert not torch.allclose(j0, j1, atol=1e-6)


def test_layernorm_variance_division_is_scale_invariant_not_linear():
    report = affine_realizability_report("layernorm_variance")
    assert report.realizable is False
    assert report.reason_code == "scale_invariant_division"

    def ln(v):
        c = v - v.mean(-1, keepdim=True)
        return c / (c.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()

    x = torch.randn(3, 6, dtype=torch.float64)
    assert torch.allclose(ln(x), ln(3.0 * x), atol=1e-5)


def test_layernorm_centering_is_the_realizable_part():
    report = affine_realizability_report("layernorm_centering")
    assert report.realizable is True
    assert report.reason_code == "fixed_linear_projection"


def test_report_covers_every_attention_subop():
    realizable = {
        name
        for name in AffineRealizability.KNOWN_OPS
        if affine_realizability_report(name).realizable
    }
    assert realizable == {"layernorm_centering"}


def test_unknown_op_raises():
    with pytest.raises(KeyError):
        affine_realizability_report("not_a_real_op")
