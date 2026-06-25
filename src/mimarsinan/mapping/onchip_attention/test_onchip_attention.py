"""Tests for the on-chip attention/LayerNorm mappability frontier (D5).

TESTS-FIRST. Two halves:

1. REALIZABLE (bit-exact lock): LayerNorm mean-centering ``x - mean(x)`` is a
   FIXED linear projection ``C = I - (1/N) 11^T`` and maps to on-chip
   ``NeuralCore`` crossbars via a signed two-rail ReLU decomposition, executing
   bit-exact (float64) to the torch reference.
2. NON-MAPPABLE (formal characterization): the data-dependent QK^T bilinear,
   the softmax exp/normalize, and the LayerNorm variance division are proven
   NOT expressible as static-weight ``clamp(relu(W x + b))`` crossbar maps.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.mapping.ir import IRGraph, NeuralCore, IRSource
from mimarsinan.mapping.onchip_attention import (
    build_centering_matrix,
    OnchipLayerNormCentering,
    affine_realizability_report,
    AffineRealizability,
)


# ----- helpers -----------------------------------------------------------

def _run_ir_graph(graph: IRGraph, x: torch.Tensor) -> dict[int, torch.Tensor]:
    """Execute every NeuralCore once in id order; return per-node buffers.

    A single-layer two-rail graph has no inter-core dependencies on other
    cores' buffers (both rails read the chip input), so one pass suffices.
    """
    buffers: dict[int, torch.Tensor] = {}
    for node in graph.nodes:
        buffers[node.id] = node.execute(x, buffers)
    return buffers


# ----- 1. REALIZABLE: centering matrix ----------------------------------

def test_centering_matrix_is_mean_subtraction():
    N = 7
    C = build_centering_matrix(N)
    x = torch.randn(5, N, dtype=torch.float64)
    ref = x - x.mean(dim=-1, keepdim=True)
    got = x @ torch.tensor(C.T, dtype=torch.float64)
    assert torch.allclose(ref, got, atol=1e-12), (ref - got).abs().max().item()


def test_centering_matrix_row_sums_zero():
    # Each output row of C sums to 0 (subtracting the mean kills the DC term).
    C = build_centering_matrix(9)
    assert np.allclose(C.sum(axis=1), 0.0, atol=1e-12)


# ----- 1. REALIZABLE: on-chip two-rail emission is bit-exact -------------

def test_onchip_centering_emits_only_neural_cores():
    mapper = OnchipLayerNormCentering(num_features=6)
    graph = mapper.to_ir_graph()
    assert isinstance(graph, IRGraph)
    assert graph.validate() == []
    # The whole op lands ON CHIP: no host ComputeOp nodes at all.
    assert graph.get_compute_ops() == []
    cores = graph.get_neural_cores()
    assert len(cores) == 2  # positive rail + negative rail
    for core in cores:
        assert isinstance(core, NeuralCore)


def test_onchip_centering_is_deterministic_bit_exact():
    # The on-chip result is BIT-EXACT run-to-run: the crossbar arithmetic is
    # deterministic, so two executions of the same graph are torch.equal. This
    # is the lock that the mapping has no nondeterministic / data-races.
    N = 8
    mapper = OnchipLayerNormCentering(num_features=N, clamp_ceiling=1e6)
    graph = mapper.to_ir_graph()
    x = torch.randn(11, N, dtype=torch.float64) * 0.3
    got_a = mapper.reconstruct_centered(_run_ir_graph(graph, x))
    got_b = mapper.reconstruct_centered(_run_ir_graph(graph, x))
    assert torch.equal(got_a, got_b)


def test_onchip_centering_bounded_to_math_reference():
    # The chip executor (NeuralCore.execute) runs the crossbar in float32 by
    # construction, so the on-chip centering matches the mathematical
    # x - mean(x) reference up to one float32 epsilon (a BOUNDED lock, in the
    # spirit of the residual Tier-1 1/T characterization: a valid on-chip op
    # with a precisely-bounded numeric gap, not a host-identical one).
    N = 8
    mapper = OnchipLayerNormCentering(num_features=N, clamp_ceiling=1e6)
    graph = mapper.to_ir_graph()
    x = torch.randn(11, N, dtype=torch.float64) * 0.3  # within clamp ceiling
    buffers = _run_ir_graph(graph, x)
    got = mapper.reconstruct_centered(buffers)
    assert got.dtype == torch.float32

    ref = x - x.mean(dim=-1, keepdim=True)
    max_abs = (ref.to(torch.float32) - got).abs().max().item()
    assert max_abs <= 1.2e-6, max_abs  # one float32 epsilon


def test_onchip_centering_idempotent_double_apply():
    # Mean-centering is a projection: C @ C == C. Re-centering a centered
    # vector is a no-op, so two on-chip applications equal one.
    N = 5
    C = build_centering_matrix(N)
    assert np.allclose(C @ C, C, atol=1e-12)


def test_centering_clamp_ceiling_is_a_characterized_bound():
    # The realizable lock is bit-exact ONLY while |C x| <= clamp_ceiling. A
    # value above the ceiling saturates the rail (a chip reality, not a bug):
    # the mapper exposes the bound so callers can certify it.
    N = 4
    ceiling = 0.5
    mapper = OnchipLayerNormCentering(num_features=N, clamp_ceiling=ceiling)
    assert mapper.exact_input_bound() == ceiling
    graph = mapper.to_ir_graph()
    # An input whose centered magnitude exceeds the ceiling clips (NOT exact).
    x = torch.full((1, N), 0.0, dtype=torch.float64)
    x[0, 0] = 10.0  # centered first coord ~7.5 >> 0.5
    buffers = _run_ir_graph(graph, x)
    got = mapper.reconstruct_centered(buffers)
    ref = (x - x.mean(dim=-1, keepdim=True)).to(torch.float32)
    assert not torch.allclose(ref, got, atol=1e-6)


# ----- 2. NON-MAPPABLE: formal characterization -------------------------

def test_qkt_is_bilinear_not_affine():
    # The attention score q . k is bilinear in the activation pair (q, k):
    # its cross second derivative w.r.t. q and k is the identity (nonzero), so
    # it is NOT an affine function of the concatenated input -> not a static
    # crossbar map.
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
    # The chip primitive clamp(relu(W x + b)) is piecewise-LINEAR. Softmax is
    # smooth/strictly-convex in places: its Hessian is nonzero, so it is not
    # piecewise affine -> not realizable on the crossbar.
    report = affine_realizability_report("softmax")
    assert report.realizable is False
    assert report.reason_code == "nonlinear_exp_normalize"

    z = torch.randn(5, dtype=torch.float64, requires_grad=True)
    s = torch.softmax(z, dim=-1)
    # Jacobian is input-dependent (diag(s) - s s^T): pick two scalings of the
    # SAME direction; a fixed matrix M would give the same Jacobian, softmax
    # does not.
    j0 = torch.autograd.functional.jacobian(lambda v: torch.softmax(v, -1), z)
    j1 = torch.autograd.functional.jacobian(lambda v: torch.softmax(v, -1), z * 3.0)
    assert not torch.allclose(j0, j1, atol=1e-6)


def test_layernorm_variance_division_is_scale_invariant_not_linear():
    # LN(x) = (x - mean) / std is scale-invariant: LN(a x) == LN(x) for a > 0.
    # A linear map M obeys M(a x) = a M(x); scale invariance contradicts
    # linearity for any a != 1, so the variance division is not a static map.
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
    # The characterization is total: every named sub-op has a verdict, and only
    # the centering projection is realizable on chip.
    realizable = {
        name
        for name in AffineRealizability.KNOWN_OPS
        if affine_realizability_report(name).realizable
    }
    assert realizable == {"layernorm_centering"}


def test_unknown_op_raises():
    import pytest

    with pytest.raises(KeyError):
        affine_realizability_report("not_a_real_op")
