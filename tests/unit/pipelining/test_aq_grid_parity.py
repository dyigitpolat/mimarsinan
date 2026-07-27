"""[mvm AQ R-edge] the certificate is judged in grid units + exact decisions."""

import numpy as np
import pytest
import torch

from mimarsinan.certification.value_certificate import VALUE_R_EDGE_AQ_LSB_BOUND
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.pipelining.core.gates.value_gates import (
    _assert_aq_grid_parity,
    boundary_grid_lsb,
)


def _ir_with_scales(scales):
    nodes = []
    for i, s in enumerate(scales):
        nodes.append(NeuralCore(
            id=i, name=f"c{i}",
            input_sources=np.array([IRSource(-2, 0)], dtype=object),
            core_matrix=np.ones((1, 1), dtype=np.float64),
            input_activation_scale=torch.tensor(float(s)),
        ))
    return IRGraph(nodes=nodes,
                   output_sources=np.array([IRSource(0, 0)], dtype=object),
                   weight_banks={})


class TestBoundaryGridLsb:
    def test_none_when_aq_is_off(self):
        assert boundary_grid_lsb(None, _ir_with_scales([2.0])) is None
        assert boundary_grid_lsb(0, _ir_with_scales([2.0])) is None

    def test_widest_armed_scale_sets_the_step(self):
        # 8 bits -> 127 positive levels; the widest grid is the coarsest.
        lsb = boundary_grid_lsb(8, _ir_with_scales([0.5, 2.54, 1.0]))
        assert lsb == pytest.approx(2.54 / 127)

    def test_unarmed_cores_are_ignored(self):
        assert boundary_grid_lsb(8, _ir_with_scales([0.0, 0.0])) is None


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
