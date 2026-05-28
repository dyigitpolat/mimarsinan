"""PerceptronMapper, ComputeOpMapper, ModuleMapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.shape_probe import probe_module_io_shapes
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


from mimarsinan.mapping.mappers.base import Mapper


class ShapeMismatchError(RuntimeError):
    """Multi-input ComputeOpMapper received broadcast-incompatible inputs."""

class ComputeOpMapper(Mapper):
    """Unified host-side ComputeOp mapper for any ``nn.Module``.

    Branches internally on ``len(sources)`` (unary vs. multi-input) and
    on ``src_arr.ndim`` (multi-column per-instance vs. whole-tensor).
    Shapes are probed via ``probe_module_io_shapes`` when not supplied.
    ``compute_per_source_scales`` may populate ``per_source_scales`` /
    ``output_scale`` to trigger ``ScaleNormalizingWrapper`` at emission.
    """

    def __init__(
        self,
        sources,
        module: nn.Module,
        *,
        input_shape: Sequence[int] | None = None,
        input_shapes: Sequence[Sequence[int]] | Sequence[int] | None = None,
        output_shape: Sequence[int] | None = None,
        module_kwargs: dict | None = None,
        output_index: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if isinstance(sources, Mapper):
            self._sources_list = [sources]
        else:
            self._sources_list = list(sources)
        if not self._sources_list:
            raise ValueError("ComputeOpMapper: at least one source is required")
        self.module = module
        if input_shapes is None and input_shape is not None:
            input_shapes = input_shape
        self.input_shapes = self._normalize_input_shapes(input_shapes)
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.module_kwargs = dict(module_kwargs) if module_kwargs else {}
        self.output_index = output_index
        self.name = name

        self.per_source_scales: list[torch.Tensor] | None = None
        self.output_scale: torch.Tensor | None = None

    @property
    def sources(self) -> list[Mapper]:
        return list(self._sources_list)

    def get_source_mappers(self):
        return [m for m in self._sources_list if m is not None]

    def combine_source_scales(
        self, source_scales: list[torch.Tensor]
    ) -> torch.Tensor:
        stacked = torch.stack([s.to(dtype=torch.float32) for s in source_scales])
        return stacked.mean(dim=0)

    def _forward_impl(self, x):
        if len(self._sources_list) == 1:
            inputs: tuple = (x,)
        else:
            inputs = tuple(x) if isinstance(x, (tuple, list)) else (x,)
            self._check_broadcastable(inputs)
        out = self.module(*inputs, **self.module_kwargs)
        if self.output_index is not None:
            out = out[self.output_index]
        return out

    def _check_broadcastable(self, inputs: tuple) -> None:
        tensor_shapes = [
            tuple(t.shape) for t in inputs if isinstance(t, torch.Tensor)
        ]
        if len(tensor_shapes) < 2:
            return
        try:
            torch.broadcast_shapes(*tensor_shapes)
        except RuntimeError as exc:
            raise ShapeMismatchError(
                f"ComputeOpMapper(name={self.name!r}): inputs do not broadcast. "
                f"observed_shapes={tensor_shapes}, "
                f"recorded_input_shapes={self.input_shapes}"
            ) from exc

    def _map_to_ir(self, ir_mapping):
        source_arrays = [src.map_to_ir(ir_mapping) for src in self._sources_list]
        module = self._maybe_wrap_for_scales()
        if len(source_arrays) == 1:
            return self._emit_unary(ir_mapping, source_arrays[0], module)
        return self._emit_multi(ir_mapping, source_arrays, module)

    def _maybe_wrap_for_scales(self) -> nn.Module:
        if self.per_source_scales is None or self.output_scale is None:
            return self.module
        if len(self.per_source_scales) != len(self._sources_list):
            raise ValueError(
                f"ComputeOpMapper: per_source_scales length "
                f"({len(self.per_source_scales)}) does not match source count "
                f"({len(self._sources_list)})"
            )
        return ScaleNormalizingWrapper(
            self.module, self.per_source_scales, self.output_scale,
        )

    def _emit_unary(self, ir_mapping, src_arr, module):
        from mimarsinan.mapping.layout.layout_source_view_ops import stack_source_views

        if src_arr.ndim == 2 and self._is_per_instance_module(module) and self.input_shapes is None:
            oriented = self._orient_2d_for_columns(src_arr.transpose(), module)
            col_input_shape = (int(oriented.shape[0]),)
            input_shape, output_shape = self._resolve_shapes_for_input(col_input_shape)
            col_count = int(oriented.shape[1])
            outputs = []
            for i in range(col_count):
                col_sources = oriented[:, i]
                if hasattr(col_sources, "flatten"):
                    col_sources = col_sources.flatten()
                col_out = ir_mapping.add_compute_op(
                    input_sources=col_sources,
                    op_type=self._module_label(module),
                    params=self._build_params(module, input_shape=input_shape),
                    input_shape=input_shape,
                    output_shape=output_shape,
                    name=(f"{self.name}_col{i}" if self.name else None),
                )
                outputs.append(
                    col_out.flatten() if hasattr(col_out, "flatten") else col_out
                )
            result = stack_source_views(outputs, axis=1)
            return result.transpose()

        if self.input_shapes is not None:
            input_shape_in = self.input_shapes[0]
            flat_sources = src_arr.flatten() if hasattr(src_arr, "flatten") else src_arr
        elif src_arr.ndim >= 2:
            input_shape_in = tuple(int(d) for d in src_arr.shape)
            flat_sources = src_arr.flatten() if hasattr(src_arr, "flatten") else src_arr
        elif src_arr.ndim == 1:
            input_shape_in = (int(src_arr.shape[0]),)
            flat_sources = src_arr
        else:
            input_shape_in = None
            flat_sources = src_arr

        input_shape, output_shape = self._resolve_shapes_for_input(input_shape_in)
        return ir_mapping.add_compute_op(
            input_sources=flat_sources,
            op_type=self._module_label(module),
            params=self._build_params(module, input_shape=input_shape),
            input_shape=input_shape,
            output_shape=output_shape,
            name=self.name,
        )

    def _emit_multi(self, ir_mapping, source_arrays, module):
        from mimarsinan.mapping.layout.layout_source_view_ops import concat_source_views

        flat_sources = [
            arr.flatten() if hasattr(arr, "flatten") else arr
            for arr in source_arrays
        ]
        input_shapes = self._resolved_multi_input_shapes(source_arrays, module)
        output_shape = self.output_shape or self._probe_multi(module, input_shapes)
        return ir_mapping.add_compute_op(
            input_sources=concat_source_views(flat_sources),
            op_type=self._module_label(module),
            params=self._build_params(
                module, input_shapes=[tuple(s) for s in input_shapes],
            ),
            input_shape=None,
            output_shape=tuple(output_shape),
            name=self.name,
        )

    def _build_params(
        self,
        module: nn.Module,
        *,
        input_shape: tuple | None = None,
        input_shapes: list[tuple] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"module": module}
        if input_shape is not None:
            params["input_shape"] = tuple(input_shape)
        if input_shapes is not None:
            params["input_shapes"] = input_shapes
        if self.module_kwargs:
            params["module_kwargs"] = dict(self.module_kwargs)
        if self.output_index is not None:
            params["output_index"] = self.output_index
        return params

    def _resolve_shapes_for_input(
        self, input_shape: tuple[int, ...] | None,
    ) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
        if self.output_shape is not None or input_shape is None:
            return input_shape, self.output_shape

        probed = probe_module_io_shapes(
            self.module,
            input_shape,
            module_kwargs=self.module_kwargs or None,
            output_index=self.output_index,
        )
        return input_shape, probed.output_shape

    def _resolved_multi_input_shapes(
        self, source_arrays, module: nn.Module,
    ) -> list[tuple[int, ...]]:
        if self.input_shapes is not None:
            return [tuple(s) for s in self.input_shapes]
        return [tuple(int(d) for d in arr.shape) for arr in source_arrays]

    def _probe_multi(
        self, module: nn.Module, input_shapes: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        probed = probe_module_io_shapes(
            self.module,
            input_shapes,
            module_kwargs=self.module_kwargs or None,
            output_index=self.output_index,
        )
        return probed.output_shape

    @staticmethod
    def _normalize_input_shapes(
        input_shapes,
    ) -> tuple[tuple[int, ...] | None, ...] | None:
        if input_shapes is None:
            return None
        if len(input_shapes) == 0:
            return None
        first = input_shapes[0]
        if first is None or isinstance(first, (tuple, list)):
            return tuple(
                tuple(int(d) for d in s) if s is not None else None
                for s in input_shapes
            )
        return (tuple(int(d) for d in input_shapes),)

    @staticmethod
    def _module_label(module: nn.Module) -> str:
        display = getattr(module, "display_name", None)
        if isinstance(display, str):
            return display
        return type(module).__name__

    @staticmethod
    def _is_per_instance_module(module: nn.Module) -> bool:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return True
        if isinstance(module, nn.Sequential) and len(module) > 0:
            return isinstance(module[0], (nn.Linear, nn.Conv1d, nn.Conv2d))
        # Duck-type Perceptron without importing it (would pull in models/).
        if hasattr(module, "layer") and isinstance(
            getattr(module, "layer", None), nn.Linear,
        ):
            return True
        return False

    @staticmethod
    def _orient_2d_for_columns(src_arr, module: nn.Module):
        in_features = getattr(module, "in_features", None)
        if in_features is None and hasattr(module, "0"):
            in_features = getattr(module[0], "in_features", None)
        if in_features is not None:
            if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                return src_arr.T
        return src_arr

