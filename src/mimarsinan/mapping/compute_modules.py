"""Host-side ComputeOp payloads.

The framework synthesises an ``nn.Module`` payload for every non-neural
``ComputeOp`` so the shared ``_exec_module`` executor can run it host-side
during simulation.  There are exactly two payload kinds:

* :class:`ComputeAdapter` — generic adapter for any picklable callable
  (functions / methods / get_attr-bound ops) with optional bound tensor
  constants and extra args/kwargs.  Bound constants are stored
  batch-stripped; the adapter prepends batch dim 0 and expands them to
  match the input's batch size at forward time.  Replaces every former
  per-op wrapper.

* :class:`ScaleNormalizingWrapper` — dynamic per-source rate→absolute→rate
  rescaling stamped onto multi-input ``ComputeOpMapper``s whose source
  scales diverge.  Different category; lives here to keep all host-side
  payload classes in one file.

A small picklable utility ``_cat_along`` is exposed for ops whose call
structure does not fit the generic ``fn(*inputs, *bound)`` signature
(``torch.cat`` takes a list-of-tensors first arg).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from mimarsinan.mapping.scale_broadcast import broadcast_scale_to_dim


def _cat_along(x: torch.Tensor, prefix: torch.Tensor, *, dim: int) -> torch.Tensor:
    """``torch.cat([prefix, x], dim=dim)`` as a flat-positional callable.

    Used for constant-prepend patterns (CLS token in ViT and similar).
    Module-level + picklable so it can be stored in
    ``ComputeAdapter.fn`` and survive IR pickling.
    """
    return torch.cat([prefix, x], dim=dim)


class ComputeAdapter(nn.Module):
    """Generic host-side ComputeOp payload for any picklable callable.

    Tensor constants are stored as non-trainable ``nn.Parameter``\\ s
    with the batch dim stripped.  At forward time, the adapter prepends
    batch dim 0 and expands every bound tensor to match the input's
    batch size, then calls
    ``fn(*inputs, *bound_expanded, *extra_args, **kwargs)``.

    This invariant means IR-time shape inference (``probe_module_io_shapes``)
    does not need to reason about batch dim — it always probes at
    batch=1 — and the wrapper still works for any runtime batch size,
    including future batched soft-core execution.

    The adapter subsumes the former specialized wrappers ``Add``,
    ``Mean``, ``Select``, ``ConstantAdd``, ``ConstantPrepend`` and the
    converter-local ``_FunctionWrapper`` — there is one host-side
    payload class for every non-neural op.

    Display: ``op_type`` labelling in the IR uses ``display_name``,
    which reports the wrapped callable's qualified name so GUI / DOT
    visualisations stay readable.
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
        """Readable callable name for IR ``op_type`` / GUI labels."""
        fn = self.fn
        module = getattr(fn, "__module__", "")
        name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
        if name is None:
            return type(fn).__name__
        return f"{module}.{name}" if module and module != "builtins" else name

    def _bound_tensors(self) -> list[torch.Tensor]:
        return [getattr(self, f"bound_{i}") for i in range(self._bound_count)]

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if inputs:
            batch_size = inputs[0].shape[0]
        else:
            batch_size = 1
        expanded_bound: list[torch.Tensor] = []
        for tensor in self._bound_tensors():
            expanded_bound.append(
                tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
            )
        return self.fn(
            *inputs, *expanded_bound, *self.extra_args, **self.kwargs,
        )

    @classmethod
    def from_fx_node(cls, node, fn) -> "ComputeAdapter":
        """Build a ``ComputeAdapter`` from a ``call_function`` FX node.

        ``args[0]`` is the input tensor (handled by the mapper source);
        remaining non-Node positional args are captured as ``extra_args``,
        and all non-Node kwargs are captured.  Replaces the former
        ``_FunctionWrapper.from_fx_node``.
        """
        import torch.fx as fx
        extra_args = tuple(a for a in node.args[1:] if not isinstance(a, fx.Node))
        kwargs = {k: v for k, v in node.kwargs.items() if not isinstance(v, fx.Node)}
        return cls(fn, extra_args=extra_args, kwargs=kwargs)


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
