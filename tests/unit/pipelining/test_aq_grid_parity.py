"""[mvm AQ R-edge] the certificate is judged in grid units + exact decisions."""

import numpy as np
import pytest
import torch

from mimarsinan.certification.value_certificate import VALUE_R_EDGE_AQ_LSB_BOUND
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.support.boundary_grids import widest_boundary_step
from mimarsinan.models.nn.activations.value_quantizer import BoundaryGrid
from mimarsinan.pipelining.core.gates.value_gates import _assert_aq_grid_parity


def _ir_with_scales(scales, bits=8):
    nodes = []
    for i, s in enumerate(scales):
        nodes.append(NeuralCore(
            id=i, name=f"c{i}",
            input_sources=np.array([IRSource(-2, 0)], dtype=object),
            core_matrix=np.ones((1, 1), dtype=np.float64),
            boundary_grid=(BoundaryGrid(scale=float(s), bits=bits) if s else None),
        ))
    return IRGraph(nodes=nodes,
                   output_sources=np.array([IRSource(0, 0)], dtype=object),
                   weight_banks={})


class TestWidestBoundaryStep:
    def test_none_when_no_core_carries_a_grid(self):
        assert widest_boundary_step(_ir_with_scales([0.0, 0.0])) is None

    def test_widest_armed_grid_sets_the_step(self):
        # 8 bits -> 127 positive levels; the widest grid is the coarsest.
        step = widest_boundary_step(_ir_with_scales([0.5, 2.54, 1.0]))
        assert step == pytest.approx(2.54 / 127)

    def test_mixed_armed_and_float_boundaries(self):
        step = widest_boundary_step(_ir_with_scales([0.0, 1.27, 0.0]))
        assert step == pytest.approx(1.27 / 127)


class TestGridParityAssertion:
    def _outputs(self, delta, flip=False):
        want = torch.tensor([[3.0, 1.0], [0.5, 2.0]], dtype=torch.float64)
        got = want.clone()
        got[0, 0] += delta
        if flip:  # push sample 0's argmax onto the other class
            got[0, 0] = want[0, 1] - 1.0
        return got, want

    def test_sub_lsb_divergence_passes(self):
        lsb = 0.02
        got, want = self._outputs(0.4 * lsb)
        _assert_aq_grid_parity(got, want, 0.4 * lsb, lsb, 2)  # must not raise

    def test_beyond_one_lsb_is_fatal(self):
        lsb = 0.02
        got, want = self._outputs(1.5 * lsb)
        with pytest.raises(RuntimeError, match="grid parity FAILED"):
            _assert_aq_grid_parity(got, want, 1.5 * lsb, lsb, 2)

    def test_decision_flip_is_fatal_even_when_small(self):
        # A tiny delta that changes the argmax must still fail: decisions are
        # what a deployed classifier promises.
        got, want = self._outputs(0.0, flip=True)
        delta = float((got - want).abs().max())
        with pytest.raises(RuntimeError, match="DECISION parity FAILED"):
            _assert_aq_grid_parity(got, want, delta, 1.0, 2)

    def test_bound_is_exactly_one_lsb(self):
        assert VALUE_R_EDGE_AQ_LSB_BOUND == 1.0
