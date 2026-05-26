"""Small ``nn.Module`` wrappers for host-side ComputeOps.

Each former specialized ``Mapper`` class (``MeanMapper``, ``SelectMapper``,
``ConstantAddMapper``, ``ConstantPrependMapper``, ``AddMapper``, ...) is
replaced by a torch ``nn.Module`` here.  The mapping side constructs
``ComputeOpMapper(source, <module>)``; the IR side stores the module in
``ComputeOp.params["module"]`` and the shared ``_exec_module`` executor
runs it host-side during simulation.

``ScaleNormalizingWrapper`` is the generic carrier for the per-source rate→
scale rescaling pass.  ``compute_per_source_scales`` stamps it around any
multi-input ``ComputeOpMapper.module`` whose source rates carry diverging
activation scales.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mimarsinan.mapping.scale_broadcast import broadcast_scale_to_dim


class Select(nn.Module):
    """``x[:, index, ...]`` along the first non-batch axis.

    Lifted from the former ``SelectMapper`` / ``ComputeOp._exec_select``
    pair.  Used for CLS-token extraction in ViTs.
    """

    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = int(index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.index]


class Mean(nn.Module):
    """``x.mean(dim=dim)`` — reduces one batch-relative axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class ConstantAdd(nn.Module):
    """``x + constant`` with a frozen learnable constant (e.g. positional embedding)."""

    def __init__(self, constant: torch.Tensor) -> None:
        super().__init__()
        # Stored as a non-trainable Parameter so it survives pickling and
        # rides .to(device) with the host-side module.
        self.constant = nn.Parameter(
            constant.detach().clone(), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.constant


class ConstantPrepend(nn.Module):
    """Prepend a constant token along the sequence axis (CLS token in ViT).

    Forward shape: ``(B, S, D)`` → ``(B, S+1, D)``.  The stored constant
    has shape ``(1, 1, D)`` and is expanded over the batch.
    """

    def __init__(self, constant: torch.Tensor, dim: int = 1) -> None:
        super().__init__()
        const = constant.detach().clone()
        # Normalise to (1, 1, D) for broadcast over batch and sequence.
        if const.dim() == 1:
            const = const.view(1, 1, -1)
        elif const.dim() == 2:
            const = const.unsqueeze(0)
        self.constant = nn.Parameter(const, requires_grad=False)
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        const = self.constant.expand(B, -1, -1)
        return torch.cat([const, x], dim=self.dim)


class Add(nn.Module):
    """Plain two-input element-wise add (``a + b``).

    Per-source scale handling lives in :class:`ScaleNormalizingWrapper`
    — this module stays pure so the same ``Add()`` instance can sit
    inside or outside a wrapper unchanged.
    """

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class ScaleNormalizingWrapper(nn.Module):
    """Per-source rate→absolute and absolute→rate rescaling around a host module.

    Each input source slice ``r_i`` is a rate stream representing real values
    ``v_i = r_i · s_i``; downstream perceptrons read this rate stream against
    a single combined ``s_out``.  This wrapper computes
    ``f(r_1 · s_1, …, r_N · s_N) / s_out`` so the underlying module ``f``
    operates in absolute units and the output is rebased to ``s_out``.

    Buffers are registered (not stored as Python lists) so the wrapper
    pickles cleanly and follows ``.to(device)``.
    """

    def __init__(
        self,
        module: nn.Module,
        input_scales: Sequence[torch.Tensor],
        output_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.module = module
        self._num_inputs = len(input_scales)
        for i, scale in enumerate(input_scales):
            self.register_buffer(
                f"input_scale_{i}", torch.as_tensor(scale, dtype=torch.float32)
            )
        self.register_buffer(
            "output_scale", torch.as_tensor(output_scale, dtype=torch.float32)
        )

    def _input_scale(self, i: int) -> torch.Tensor:
        return getattr(self, f"input_scale_{i}")

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != self._num_inputs:
            raise ValueError(
                f"ScaleNormalizingWrapper: expected {self._num_inputs} inputs, "
                f"got {len(inputs)}"
            )
        absolute_inputs = []
        for i, x in enumerate(inputs):
            scale = self._input_scale(i)
            broadcast = broadcast_scale_to_dim(
                scale.to(dtype=x.dtype, device=x.device), x.shape[-1]
            )
            absolute_inputs.append(x * broadcast)
        absolute_out = self.module(*absolute_inputs)
        out_scale = broadcast_scale_to_dim(
            self.output_scale.to(dtype=absolute_out.dtype, device=absolute_out.device),
            absolute_out.shape[-1],
        )
        return absolute_out / out_scale
