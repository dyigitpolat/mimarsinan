"""Host-side ComputeOp payload classes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.mapping.support.scale_broadcast import broadcast_scale_to_dim


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

    @staticmethod
    def _leading_tensor(obj) -> torch.Tensor | None:
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (tuple, list)):
            for item in obj:
                found = ComputeAdapter._leading_tensor(item)
                if found is not None:
                    return found
        return None

    def forward(self, *inputs) -> torch.Tensor:
        lead = self._leading_tensor(inputs)
        batch_size = lead.shape[0] if lead is not None else 1
        expanded_bound = [
            t.unsqueeze(0).expand(batch_size, *t.shape) for t in self._bound_tensors()
        ]
        return self.fn(
            *inputs, *expanded_bound, *self.extra_args, **self.kwargs,
        )

    @classmethod
    def from_fx_node(cls, node, fn) -> "ComputeAdapter":
        extra_args = tuple(a for a in node.args[1:] if not isinstance(a, fx.Node))
        kwargs = {k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)}
        return cls(fn, extra_args=extra_args, kwargs=kwargs)


class ScaleNormalizingWrapper(nn.Module):
    """Per-source rate→absolute→rate rescaling around a wrapped module.

    Computes ``f(r_1·s_1, ..., r_N·s_N) / s_out`` so ``f`` operates in absolute
    units while inputs/outputs travel as rates.
    """

    output_scale: torch.Tensor

    output_offset: torch.Tensor | None

    def __init__(
        self,
        module: nn.Module,
        input_scales: Sequence[torch.Tensor],
        output_scale: torch.Tensor,
        output_offset: torch.Tensor | None = None,
        module_kwargs: dict | None = None,
        output_index: int | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        # Transparent to the wrapped module's calling convention: keyword args
        # (MultiheadAttention's need_weights) and a tuple return selected by
        # output_index — the value twin applies both, so the wire twin must too.
        self.module_kwargs = dict(module_kwargs) if module_kwargs else {}
        self.output_index = output_index
        self._num_inputs = len(input_scales)
        for i, scale in enumerate(input_scales):
            self.register_buffer(
                f"input_scale_{i}", torch.as_tensor(scale, dtype=torch.float32)
            )
        self.register_buffer(
            "output_scale", torch.as_tensor(output_scale, dtype=torch.float32)
        )
        # [sigma-in-the-op] pre-training signed-seam lift: part of the op's
        # own function, so every representation carries it uniformly.
        self.register_buffer(
            "output_offset",
            None if output_offset is None
            else torch.as_tensor(output_offset, dtype=torch.float32),
        )

    def _input_scale(self, i: int) -> torch.Tensor:
        return getattr(self, f"input_scale_{i}")

    def forward(self, *inputs: torch.Tensor, **call_kwargs) -> torch.Tensor:
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
        # Call-site kwargs (the IR executor's) WIN on collision: the wrapper
        # must behave exactly as the bare module did for that caller; the walk
        # passes none, so the constructor-owned kwargs still govern it.
        merged = {**self.module_kwargs, **call_kwargs}
        absolute_out = self.module(*absolute_inputs, **merged)
        if self.output_index is not None:
            absolute_out = absolute_out[self.output_index]
        if self.output_offset is not None:
            absolute_out = absolute_out + self.output_offset.to(
                dtype=absolute_out.dtype, device=absolute_out.device,
            )
        out_scale = broadcast_scale_to_dim(
            self.output_scale.to(dtype=absolute_out.dtype, device=absolute_out.device),
            absolute_out.shape[-1],
        )
        return absolute_out / out_scale
