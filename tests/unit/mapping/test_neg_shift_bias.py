"""Negative-value shift bias correction: W·(x+s)+B' ≡ W·x+B.

A ComputeOp that emits negative values is shifted into the non-negative domain
(F(x) → F(x)+s) so the spike encoder (which clamps rates to [0,1]) is lossless.
The consuming core's bias absorbs the shift: B' = B − W·s, so the on-chip result
is identical to the unshifted model. This pins that algebra (the encode-side
application and stats-derived `s` are tested separately).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.mapping.support.bias_compensation import apply_negative_shift_bias


def _perceptron(W, B):
    out_f, in_f = W.shape
    p = Perceptron(out_f, in_f, normalization=nn.Identity())
    p.layer.weight.data = W.clone()
    p.layer.bias.data = B.clone()
    return p


@pytest.mark.parametrize("seed", range(8))
def test_shift_bias_preserves_output(seed):
    torch.manual_seed(seed)
    out_f, in_f = 5, 7
    W = torch.randn(out_f, in_f)
    B = torch.randn(out_f)
    s = torch.rand(in_f) * 2.0  # per-axon non-negative shift

    p = _perceptron(W, B)
    apply_negative_shift_bias(p, s)

    x = torch.randn(3, in_f)
    unshifted = x @ W.T + B                  # W·x + B   (original)
    shifted = (x + s) @ W.T + p.layer.bias.data  # W·(x+s) + B'
    torch.testing.assert_close(shifted, unshifted, atol=1e-5, rtol=0.0)


def test_scalar_shift():
    torch.manual_seed(0)
    W = torch.randn(4, 6)
    B = torch.randn(4)
    p = _perceptron(W, B)
    apply_negative_shift_bias(p, 1.5)  # scalar shift on every axon
    x = torch.randn(2, 6)
    torch.testing.assert_close(
        (x + 1.5) @ W.T + p.layer.bias.data, x @ W.T + B, atol=1e-5, rtol=0.0
    )


def test_idempotent():
    torch.manual_seed(0)
    W = torch.randn(4, 6)
    B = torch.randn(4)
    s = torch.rand(6)
    p = _perceptron(W, B)
    apply_negative_shift_bias(p, s)
    baked_once = p.layer.bias.data.clone()
    apply_negative_shift_bias(p, s)  # second call must be a no-op
    torch.testing.assert_close(p.layer.bias.data, baked_once, atol=0.0, rtol=0.0)


def test_negative_shifts_from_min():
    import numpy as np
    from mimarsinan.mapping.support.bias_compensation import negative_shifts_from_min

    mins = {
        5: np.array([-0.4, 0.1, -1.2]),   # two negative channels → shift
        7: np.array([0.0, 0.3, 0.5]),     # all >= 0 → dropped
    }
    out = negative_shifts_from_min(mins)
    assert set(out) == {5}                # node 7 dropped (no shift needed)
    np.testing.assert_allclose(out[5], [0.4, 0.0, 1.2])
