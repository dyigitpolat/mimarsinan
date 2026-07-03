"""Differentiable single-spike TTFS activation (Stanojevic et al., Nat. Commun. 2024)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction


class TTFSCycleActivation(nn.Module):
    """Single-spike TTFS activation: the deployment staircase kernel on
    ``clamp(relu(x)/θ, 0, 1)`` scaled by θ, with STE grad."""

    _VALID_THRESHOLDING_MODES = ("<", "<=")

    def __init__(
        self,
        T: int,
        activation_scale: nn.Parameter | torch.Tensor | float,
        thresholding_mode: str = "<=",
        firing_mode: str = "TTFS",
    ):
        super().__init__()
        self.T = int(T)
        if isinstance(activation_scale, (int, float)):
            activation_scale = nn.Parameter(
                torch.tensor(float(activation_scale)), requires_grad=False
            )
        self.activation_scale = activation_scale

        if thresholding_mode not in self._VALID_THRESHOLDING_MODES:
            raise ValueError(
                f"TTFSCycleActivation thresholding_mode must be one of "
                f"{self._VALID_THRESHOLDING_MODES!r}; got {thresholding_mode!r}"
            )
        self.thresholding_mode = thresholding_mode
        self.firing_mode = firing_mode

    @property
    def activation_type(self) -> str:
        return "TTFS"

    def extra_repr(self) -> str:
        return f"T={self.T}, thresholding_mode={self.thresholding_mode!r}"

    def _safe_scale(self, x: torch.Tensor) -> torch.Tensor | float:
        scale = self.activation_scale
        if isinstance(scale, torch.Tensor):
            return scale.to(device=x.device, dtype=x.dtype).clamp(min=1e-12)
        return max(float(scale), 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self._safe_scale(x)
        r = (F.relu(x) / scale).clamp(0.0, 1.0)
        return TTFSStaircaseFunction.apply(r, self.T) * scale
