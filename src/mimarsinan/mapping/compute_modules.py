"""Host-side ComputeOp payload classes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from mimarsinan.mapping.scale_broadcast import broadcast_scale_to_dim


def _cat_along(x: torch.Tensor, prefix: torch.Tensor, *, dim: int) -> torch.Tensor:
    return torch.cat([prefix, x], dim=dim)


class ComputeAdapter(nn.Module):
    """Generic host-side ComputeOp payload wrapping a picklable callable.

    Bound tensors are stored batch-stripped; ``forward`` expands them to the
    input batch size, so IR-time shape inference always probes at batch=1.
    """

    def __init__(
        self,
        fn,
        *,
        bound_tensors: Sequence[torch.Tensor] = (),
        extra_args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.extra_args = tuple(extra_args)
        self.kwargs = dict(kwargs) if kwargs else {}
        self._bound_count = len(bound_tensors)
        for i, tensor in enumerate(bound_tensors):
            self.register_parameter(
                f"bound_{i}",
                nn.Parameter(tensor.detach().clone(), requires_grad=False),
            )

    @property
    def display_name(self) -> str:
        fn = self.fn
        module = getattr(fn, "__module__", "")
        name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
        if name is None:
            return type(fn).__name__
        return f"{module}.{name}" if module and module != "builtins" else name

    def _bound_tensors(self) -> list[torch.Tensor]:
        return [getattr(self, f"bound_{i}") for i in range(self._bound_count)]

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs[0].shape[0] if inputs else 1
        expanded_bound = [
            t.unsqueeze(0).expand(batch_size, *t.shape) for t in self._bound_tensors()
        ]
        return self.fn(
            *inputs, *expanded_bound, *self.extra_args, **self.kwargs,
        )

    @classmethod
    def from_fx_node(cls, node, fn) -> "ComputeAdapter":
        import torch.fx as fx
        extra_args = tuple(a for a in node.args[1:] if not isinstance(a, fx.Node))
        kwargs = {k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)}
        return cls(fn, extra_args=extra_args, kwargs=kwargs)


class ScaleNormalizingWrapper(nn.Module):
    """Per-source rate→absolute→rate rescaling around a wrapped module.

    Computes ``f(r_1·s_1, ..., r_N·s_N) / s_out`` so ``f`` operates in absolute
    units while inputs/outputs travel as rates.
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
