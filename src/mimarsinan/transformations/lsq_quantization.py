"""Learnable Step-Size Quantization (LSQ) for weights.

LSQ replaces the "rate-mixed FP+Q weights" approach of
``NormalizationAwarePerceptronQuantization`` with a differentiable
quantizer that uses a *learnable* step size and a straight-through
estimator (STE) for the rounding operation.

Why log-space for the scale
---------------------------
The step size ``s = exp(log_scale)`` is positive by construction under
any optimiser update, no matter how large a gradient hits it.  The
legacy ``scale = q_max / max(|w|)`` loses its positivity guarantee the
moment gradient descent is applied to the raw scale.

Why STE
-------
``round`` is piecewise-constant, so its true derivative is zero almost
everywhere and a Dirac at the bin boundaries.  Plugging that gradient
into the optimiser freezes every weight instantly.  STE pretends the
rounding was identity on the backward pass, which is the standard QAT
formulation from the LSQ / LSQ+ papers.

Why the quantizer is an ``nn.Module``
-------------------------------------
``BasicTrainer`` builds its optimiser from ``self.model.parameters()``.
Storing ``log_scale`` inside a child module ensures every per-perceptron
quantizer is picked up automatically -- no special optimiser wiring
required.  It also makes standard PyTorch save/load round-trip the
learnt scale alongside the weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _STERound(torch.autograd.Function):
    """Round with a straight-through gradient.

    Forward: ``torch.round(x)``.  Backward: identity on the gradient,
    which gives LSQ its characteristic "learn through rounding"
    behaviour.  Clamping to the integer grid happens *outside* this
    function so the clamp contributes its own (sub-)gradient instead of
    being swallowed by the STE.
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return _STERound.apply(x)


class LSQQuantizer(nn.Module):
    """LSQ-style symmetric weight quantizer.

    Attributes
    ----------
    bits : int
        Bit-width of the target representation.  For an ``N``-bit signed
        representation the integer grid is ``[-2**(N-1), 2**(N-1) - 1]``.
    q_min, q_max : int
        Lower / upper integer bounds of the quantization grid.
    log_scale : torch.nn.Parameter
        Log-space learnable step size.  ``step = exp(log_scale)`` is the
        floating-point value of one integer unit in the grid.

    Usage
    -----
    ::

        q = LSQQuantizer(bits=8)
        q.init_from_tensor(w)            # seed step from max-abs statistic
        w_q = q(w)                       # forward quantisation (STE)
        loss = loss_fn(w_q)
        loss.backward()                  # gradients flow to w *and* log_scale
    """

    def __init__(self, bits: int):
        super().__init__()
        if bits < 2:
            raise ValueError(f"LSQQuantizer requires bits >= 2; got {bits}")
        self.bits = int(bits)
        self.q_min = -(2 ** (bits - 1))
        self.q_max = (2 ** (bits - 1)) - 1
        # Initialise to "unit step" so the quantizer is usable even
        # before ``init_from_tensor`` has been called.  Tuners should
        # always reseed it from a representative tensor before training.
        self.log_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    @torch.no_grad()
    def init_from_tensor(self, tensor: torch.Tensor) -> None:
        """Seed ``log_scale`` from the max-abs of ``tensor``.

        The legacy closed-form scale is ``q_max / max(|w|)`` -- i.e. the
        integer grid's maximum absolute value divided by the weight
        tensor's maximum absolute value.  The reciprocal is the step
        size, which is what we store.  A floor of ``1e-12`` guards
        against all-zero tensors (pruned layers) without producing
        ``inf`` or ``NaN``.
        """
        p_max = float(torch.max(torch.abs(tensor)).item())
        p_max = max(p_max, 1e-12)
        step = p_max / float(self.q_max)
        self.log_scale.data.fill_(math.log(step))

    def step(self) -> torch.Tensor:
        """Return the live step size ``exp(log_scale)`` -- a 0-d tensor
        that tracks gradients when ``log_scale.requires_grad`` is True."""
        return torch.exp(self.log_scale)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # Convert the bounded integer range into the floating-point
        # range that clamp() operates on, so clamp contributes its own
        # sub-gradient to the optimiser instead of being swallowed by
        # STE.  Doing the clamp in the floating-point domain (rather
        # than after STE) also means the ceiling/floor values track
        # changes to log_scale smoothly.
        step = self.step()
        q_min = float(self.q_min) * step
        q_max = float(self.q_max) * step
        w_clamped = torch.clamp(w, q_min, q_max)
        # STE-rounded integer index, then scaled back to the original
        # floating-point domain.  ``ste_round`` is identity on the
        # backward pass; ``clamp`` contributes the sign-dependent
        # masking sub-gradient.
        integer = ste_round(w_clamped / step)
        return integer * step
