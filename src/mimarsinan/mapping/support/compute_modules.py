"""Host-side ComputeOp payload classes."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch
import torch.fx as fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.support.call_template import (
    ArgTemplate,
    CallPartition,
    SourceArg,
    partition_call,
)
from mimarsinan.mapping.support.scale_broadcast import broadcast_scale_along


def _cat_along(x: torch.Tensor, prefix: torch.Tensor, *, dim: int) -> torch.Tensor:
    """Concatenate a batch-stripped ``prefix`` before ``x``, replicated per sample (cat never broadcasts)."""
    batched = prefix.unsqueeze(0).expand(x.shape[0], *prefix.shape)
    return torch.cat([batched, x], dim=dim)


def cat_per_sample(tensors: Sequence[torch.Tensor], *, dim: int) -> torch.Tensor:
    """``torch.cat`` of batched activations and batch-stripped bound tensors (one rank lower, replicated per sample)."""
    rank = max(t.dim() for t in tensors)
    batch = next(t.shape[0] for t in tensors if t.dim() == rank)
    return torch.cat([t.unsqueeze(0).expand(batch, *t.shape) if t.dim() < rank else t for t in tensors], dim=dim)


class HostOpWriteError(ToolchainRefusal, RuntimeError):
    """A host op of a call outside the declared operator set wrote in place an input that another node of the program
    reads (review round 15, ``operand_writes``): the converted model computes every value once, at its own node, so
    the other reader would see the value at another moment than natively; the converted forward refuses the input."""

    def __init__(self, operator: str, node_name: str, index: int):
        self.operator, self.node_name, self.index = operator, node_name, index
        super().__init__(f"{operator} (FX node {node_name!r}): it wrote its input {index} in place at run time, and "
                         "another node of the program reads that input; the conversion computes every value once, at "
                         "its own node, so the converted model refuses this input")


class HostOpAliasError(ToolchainRefusal, RuntimeError):
    """A host op of a call torch's schema lets share its operand's storage, which the traced program's run showed
    copying and on which the rule of a write relies (review round 16, ``operand_writes.COPY_GUARD``), met at run time
    an operand of another layout than the run's, or returned a value sharing its operand's storage: the conversion took
    the call as a copy, so the converted forward refuses the input."""

    def __init__(self, operator: str, node_name: str, reason: str):
        self.operator, self.node_name, self.reason = operator, node_name, reason
        super().__init__(f"{operator} (FX node {node_name!r}): {reason}; the conversion takes this call as a copy, as "
                         "the traced program's run at batch 1 showed it, so the converted model refuses this input")


_IS_ALIAS_OF = getattr(torch._C, "_is_alias_of")


def checked_copy(out: Any, inputs: Sequence[Any], guard: tuple, node_name: str) -> None:
    """``out`` of a host op guarded as a copy (``guard``: the operand's input index, its strides in the traced
    program's run at batch 1, the operator): its operand has those strides on every axis of extent above 1, and
    ``out`` shares no storage with it; else ``HostOpAliasError``."""
    index, strides, operator = guard
    operand = inputs[index]
    if not isinstance(out, torch.Tensor) or not isinstance(operand, torch.Tensor):
        return
    found = tuple(operand.stride())
    if len(found) != len(strides) or any(a != b for a, b, n in zip(found, strides, operand.shape) if n != 1):
        raise HostOpAliasError(operator, node_name, f"its input {index} has the strides {found} at run time, and the "
                                                    f"traced program's run at batch 1 showed {tuple(strides)}")
    if out is operand or bool(_IS_ALIAS_OF(out, operand)):
        raise HostOpAliasError(operator, node_name, f"it returned a value sharing the storage of its input {index}")


def write_counter(tensor: torch.Tensor) -> Any:
    """What a write in place of ``tensor`` changes: its version counter, or, for an inference tensor (created under
    ``torch.inference_mode()``, which keeps no version counter), its bytes (review round 16: reading ``_version`` of an
    inference tensor raised)."""
    if not tensor.is_inference():
        return tensor._version
    flat = tensor.detach().reshape(-1).contiguous()
    return bytes(flat.view(torch.uint8).cpu().numpy()) if flat.numel() else b""


def _state(value: Any) -> Any:
    return (write_counter(value), tuple(value.shape), tuple(value.stride())) if isinstance(value, torch.Tensor) else None


def unwritten_call(call: Callable[[Sequence[Any]], Any], inputs: Sequence[Any], kept: Sequence[int], operator: str,
                   node_name: str) -> Any:
    """``call(inputs)`` on the inputs themselves, refused by name (``HostOpWriteError``) when it wrote one of the
    ``kept`` inputs in place. [changed in review round 16, targeted pass: an input every other reader reads before the
    call was given as a copy, ``Tensor.clone``, which keeps the values and not the layout; the call now runs after
    those readers, ``Mapper.run_after``]"""
    before = [_state(inputs[i]) for i in kept]
    out = call(inputs)
    changed = [i for i, state in zip(kept, before) if _state(inputs[i]) != state]
    if changed:
        raise HostOpWriteError(operator, node_name, changed[0])
    return out


def read_after(value: Any, *calls: Any) -> Any:
    """``value`` itself, read after ``calls`` ran (review round 16, targeted pass): the host op through which every
    reader of an operand after a call whose write the traced run observed reads it (``operand_writes``), so the
    converted model runs it after the call, on the written operand, as natively."""
    return value


class ComputeAdapter(nn.Module):
    """Generic host-side ComputeOp payload wrapping a picklable callable.

    Bound tensors are stored batch-stripped and broadcast against the inputs by
    trailing alignment. ``arg_template`` keeps the traced argument order; without
    one the call is ``fn(*inputs, *bound, *extra_args, **kwargs)``.
    """

    # Class-level default: an adapter in the canonical layout stores no template,
    # so its pickled state is the state adapters had before templates existed.
    arg_template: ArgTemplate | None = None
    # Per input, the traced dtype of a non-floating activation operand (a boolean mask, an index), else None: a value
    # program carries every value in a floating buffer, so the call reads such an operand back in its traced dtype; a
    # no-op where the operand has that dtype (Wave 1 G1). The round trip is exact for a boolean, and for an integer of
    # magnitude at most 2**24 in a float32 buffer, 2**53 in a float64 one (the significand); the fp64 probe certifies
    # the fp64 value run, so an fp32 run of an integer operand above 2**24 is outside what it shows.
    operand_dtypes: tuple | None = None

    def __init__(
        self,
        fn,
        *,
        bound_tensors: Sequence[torch.Tensor | nn.Module] = (),
        extra_args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
        arg_template: ArgTemplate | None = None,
    ) -> None:
        super().__init__()
        self.fn = fn
        self.extra_args = tuple(extra_args)
        self.kwargs = dict(kwargs) if kwargs else {}
        self._bound_count = len(bound_tensors)
        for i, tensor in enumerate(bound_tensors):
            if isinstance(tensor, nn.Module):  # computed at every call from its own state (``batch_free_value``)
                self.add_module(f"bound_{i}", tensor)
                continue
            self.register_parameter(
                f"bound_{i}",
                nn.Parameter(tensor.detach().clone(), requires_grad=False),
            )
        if arg_template is not None:
            self.arg_template = arg_template

    @classmethod
    def from_partition(
        cls, fn, partition: CallPartition, *, restore_native_shapes: bool = False,
    ) -> "ComputeAdapter":
        """The adapter evaluating ``partition`` in its traced order (native bound shapes on request)."""
        template = partition.template
        if restore_native_shapes and partition.strips_bound_shapes:
            template = partition.explicit_template(restore_native_shapes=True)
        return cls(
            fn,
            bound_tensors=partition.payloads(),
            extra_args=partition.literals,
            kwargs=partition.literal_kwargs,
            arg_template=template,
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
        found = [getattr(self, f"bound_{i}") for i in range(self._bound_count)]
        return [value() if isinstance(value, nn.Module) else value for value in found]

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

    def _bound_values(self, inputs) -> list[torch.Tensor]:
        bound = self._bound_tensors()
        if self._leading_tensor(inputs) is None:
            # No input to broadcast against: the batch axis is a single sample.
            return [t.unsqueeze(0) for t in bound]
        return bound

    def forward(self, *inputs) -> torch.Tensor:
        if self.operand_dtypes is not None:
            inputs = tuple(x.to(dtype) if dtype is not None and isinstance(x, torch.Tensor) else x
                           for x, dtype in zip(inputs, self.operand_dtypes))
        bound = self._bound_values(inputs)
        if self.arg_template is None:
            return self.fn(*inputs, *bound, *self.extra_args, **self.kwargs)
        args, slot_kwargs = self.arg_template.resolve(inputs, bound, self.extra_args)
        return self.fn(*args, **self.kwargs, **slot_kwargs)

    @classmethod
    def from_fx_node(cls, node: fx.Node, fn) -> "ComputeAdapter":
        """Every FX-node argument becomes an input, in traced order."""
        partition = partition_call(fn, node.args, node.kwargs, SourceArg)
        return cls.from_partition(fn, partition)


class ScaleNormalizingWrapper(nn.Module):
    """Per-source rate→absolute→rate rescaling around a wrapped module.

    Computes ``f(r_1·s_1, ..., r_N·s_N) / s_out`` so ``f`` operates in absolute
    units while inputs/outputs travel as rates.
    """

    output_scale: torch.Tensor

    output_offset: torch.Tensor | None
    # [U-SC2] the batched axis a per-channel slot runs along (1: a channel-first input); None, and every pickle from
    # before it, broadcasts along the last axis.
    channel_axis: int | None = None

    def __init__(
        self,
        module: nn.Module,
        input_scales: Sequence[torch.Tensor],
        output_scale: torch.Tensor,
        output_offset: torch.Tensor | None = None,
        module_kwargs: dict | None = None,
        output_index: int | None = None,
        channel_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        if channel_axis is not None:
            self.channel_axis = int(channel_axis)
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
            broadcast = broadcast_scale_along(scale.to(dtype=x.dtype, device=x.device), x, self.channel_axis)
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
        out_scale = broadcast_scale_along(
            self.output_scale.to(dtype=absolute_out.dtype, device=absolute_out.device), absolute_out, self.channel_axis,
        )
        return absolute_out / out_scale
