"""PerceptronMapper, ComputeOpMapper, ModuleComputeMapper, and ModuleMapper."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.shape_probe import probe_module_io_shapes
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer


class PerceptronMapper(Mapper):
    def __init__(self, source_mapper, perceptron):
        super(PerceptronMapper, self).__init__(source_mapper)
        self.perceptron = perceptron

    def _map_to_ir(self, ir_mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        layer_sources = self.source_mapper.map_to_ir(ir_mapping)
        layer_sources = layer_sources.transpose()

        # Encoding-layer Perceptrons run host-side as ComputeOps wrapping
        # the full Perceptron forward (including the tunable TransformedActivation
        # chain). This covers both plain MLP (single-column source) and Mixer
        # token-grid layouts (multi-column: one ComputeOp per token position).
        if getattr(self.perceptron, "is_encoding_layer", False):
            return self._map_to_ir_as_encoding_compute_op(
                ir_mapping, layer_sources, layer_weights, layer_biases
            )

        normalization = getattr(self.perceptron, "normalization", None)
        normalization_type = type(normalization).__name__ if normalization is not None else None
        activation_type = resolve_activation_type(self.perceptron)

        output_shape = np.array([layer_weights.shape[0], layer_sources.shape[-1]])
        layer_sources = ir_mapping.map_fc(
            layer_sources,
            output_shape,
            layer_weights,
            layer_biases,
            self.perceptron.activation_scale,
            self.perceptron.parameter_scale,
            self.perceptron.input_activation_scale,
            name=getattr(self.perceptron, "name", None),
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=getattr(self, "perceptron_index", None),
        )

        return layer_sources.transpose()

    def _map_to_ir_as_encoding_compute_op(self, ir_mapping, layer_sources, layer_weights, _layer_biases):
        """Host-side full ``Perceptron`` forward as one ``ComputeOp(module)``.

        ``layer_sources`` is shaped ``(in_features,)`` for single-column
        (plain MLP) or ``(in_features, num_instances)`` for multi-column
        (e.g. Mixer token grid). In the multi-column case we emit one
        ComputeOp per token position, all sharing the same Perceptron module
        (so a single set of tunable parameters / activation decorators drives
        every column) — same pattern as ``ModuleComputeMapper._map_to_ir``.
        """
        in_features = int(layer_weights.shape[1])
        out_features = int(layer_weights.shape[0])

        from mimarsinan.mapping.layout.layout_source_view import stack_source_views
        src_arr = layer_sources
        if src_arr.ndim == 2 and src_arr.shape[1] > 1:
            # Multi-column: one ComputeOp per column. ``src_arr[:, i]`` has
            # shape ``(in_features,)``.
            num_instances = int(src_arr.shape[1])
            outputs = []
            base_name = getattr(self.perceptron, "name", None)
            for i in range(num_instances):
                col_sources = src_arr[:, i]
                if hasattr(col_sources, "flatten"):
                    col_sources = col_sources.flatten()
                col_out = ir_mapping.add_compute_op(
                    input_sources=col_sources,
                    op_type=type(self.perceptron).__name__,
                    params={"module": self.perceptron, "input_shape": (in_features,)},
                    input_shape=(in_features,),
                    output_shape=(out_features,),
                    name=(f"{base_name}_col{i}" if base_name else None),
                )
                outputs.append(col_out.flatten() if hasattr(col_out, "flatten") else col_out)
            result = stack_source_views(outputs, axis=1)  # (out_features, num_instances)
            return result.transpose()  # (num_instances, out_features)

        flat_in = src_arr.flatten()
        out = ir_mapping.add_compute_op(
            input_sources=flat_in,
            op_type=type(self.perceptron).__name__,
            params={"module": self.perceptron, "input_shape": (in_features,)},
            input_shape=(in_features,),
            output_shape=(out_features,),
            name=getattr(self.perceptron, "name", None),
        )
        return out.transpose()

    def _forward_impl(self, x):
        return self.perceptron(x)

    def owned_perceptron_groups(self):
        return [[self.perceptron]]


class ComputeOpMapper(Mapper):
    """Unified host-side ComputeOp mapper for any ``nn.Module``.

    Replaces the former ``ModuleComputeMapper`` (unary) and
    ``_MultiInputModuleComputeMapper`` (multi-input).  ``sources`` is a
    single Mapper or a sequence of Mappers; the class branches internally
    on ``len(sources)`` and on ``source_array.ndim`` (multi-column
    expansion for 2D unary sources).

    Shape inference: when ``input_shapes`` / ``output_shape`` are not
    supplied, :func:`probe_module_io_shapes` runs a single zeros-tensor
    forward pass through ``module`` to fill them in.

    Per-source scale handling: when ``compute_per_source_scales`` decides
    that incoming sources carry diverging activation scales, it sets
    ``per_source_scales`` and ``output_scale`` on this instance; at
    emission time the module is transparently wrapped in
    :class:`mimarsinan.mapping.compute_modules.ScaleNormalizingWrapper`.
    Unary mappers and multi-input mappers with coherent source scales
    emit the bare module.
    """

    def __init__(
        self,
        sources,
        module: nn.Module,
        *,
        input_shape: Sequence[int] | None = None,  # back-compat alias
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
            input_shapes = input_shape  # treated as a single shape below
        self.input_shapes = self._normalize_input_shapes(input_shapes)
        self.output_shape = tuple(output_shape) if output_shape is not None else None
        self.module_kwargs = dict(module_kwargs) if module_kwargs else {}
        self.output_index = output_index
        self.name = name

        # Filled in by ``compute_per_source_scales`` when source scales diverge.
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
        """Choose the combined output scale for a multi-input op.

        Default policy: arithmetic mean of input scales (matches the
        historical ``AddMapper`` behaviour).  Subclasses or callers may
        replace this for ops with different semantics.
        """
        stacked = torch.stack([s.to(dtype=torch.float32) for s in source_scales])
        return stacked.mean(dim=0)

    def _forward_impl(self, x):
        if len(self._sources_list) == 1:
            inputs: tuple = (x,)
        else:
            inputs = tuple(x) if isinstance(x, (tuple, list)) else (x,)
        out = self.module(*inputs, **self.module_kwargs)
        if self.output_index is not None:
            out = out[self.output_index]
        return out

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
        """Single-source emission.  Handles 1D, 2D-per-instance, and structured ND sources.

        Two 2D paths:

        * **Per-instance** (``Linear``/``Conv``/``Sequential`` of one):
          each column is an independent token / instance — emit one
          ComputeOp per column sharing the same module.  Mirrors the
          former ``ModuleComputeMapper`` / ``PerceptronMapper.
          is_encoding_layer`` multi-column expansion.

        * **Whole-tensor** (``LayerNorm``, ``Mean``, ``Select``,
          ``ConstantAdd``, ``ConstantPrepend``, ...): the module
          consumes the structured shape directly.  Flatten the source
          to ``(N,)`` and store the original ``input_shape`` so the
          ``_exec_module`` executor reshapes back before applying the
          module.
        """
        from mimarsinan.mapping.layout.layout_source_view import stack_source_views

        # Per-instance expansion fires only for genuinely 2D
        # ``(num_instances, in_features)`` sources feeding a per-instance
        # module.  Higher-dimensional sources carry a structured shape
        # (e.g. Conv input ``(C, H, W)``) that the module consumes whole.
        if src_arr.ndim == 2 and self._is_per_instance_module(module) and self.input_shapes is None:
            oriented = self._orient_2d_for_columns(src_arr.transpose(), module)
            col_input_shape = (int(oriented.shape[0]),)
            input_shape, output_shape = self._resolve_shapes_for_input(
                col_input_shape,
            )
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

        # Whole-tensor path: pass the source's full structured shape via
        # ``input_shape`` so ``_exec_module`` reshapes the flat (B, N) gather
        # before calling the module.
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
        """Multi-source emission: concat sources, emit one ComputeOp."""
        from mimarsinan.mapping.layout.layout_source_view import concat_source_views

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
        """Return ``(input_shape, output_shape)`` for one unary emission.

        ``output_shape`` is taken from ``self.output_shape`` when set, else
        derived from a zeros-tensor probe of the underlying ``self.module``.
        Wrapping with :class:`ScaleNormalizingWrapper` preserves shape, so
        the probe runs against the bare module.
        """
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
    ) -> tuple[tuple[int, ...], ...] | None:
        if input_shapes is None:
            return None
        if len(input_shapes) == 0:
            return None
        first = input_shapes[0]
        if isinstance(first, (tuple, list)):
            return tuple(tuple(int(d) for d in s) for s in input_shapes)
        return (tuple(int(d) for d in input_shapes),)

    @staticmethod
    def _module_label(module: nn.Module) -> str:
        """Display label for an emitted ComputeOp.

        ``ComputeAdapter`` (and any module exposing ``display_name``)
        provides a readable callable name (e.g. ``"operator.add"``,
        ``"torch.mean"``).  Plain ``nn.Module`` instances fall back to
        their class name (``"LayerNorm"``, ``"MaxPool2d"``, etc.).
        """
        display = getattr(module, "display_name", None)
        if isinstance(display, str):
            return display
        return type(module).__name__

    @staticmethod
    def _is_per_instance_module(module: nn.Module) -> bool:
        """True if ``module`` is applied independently per token / instance.

        ``Linear`` / ``Conv1d`` / ``Conv2d`` (and ``nn.Sequential`` starting
        with one) consume one row at a time when fed a 2D
        ``(num_instances, in_features)`` source — one ComputeOp per
        instance is the correct expansion.

        All other modules (``LayerNorm``, ``Mean``, ``Select``,
        ``ConstantAdd``, ``MultiheadAttention``, ...) operate on the
        whole structured tensor at once; emit one ComputeOp with
        ``input_shape`` set to the source's structured shape.

        ``Perceptron`` is treated as per-instance: the
        encoding-layer path in :class:`PerceptronMapper` expands one
        ComputeOp per token (see ``_map_to_ir_as_encoding_compute_op``).
        """
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            return True
        if isinstance(module, nn.Sequential) and len(module) > 0:
            return isinstance(module[0], (nn.Linear, nn.Conv1d, nn.Conv2d))
        # Perceptron-from-models: detect by attribute rather than import to
        # avoid the heavy ``models`` dependency from this mapping file.
        if hasattr(module, "layer") and isinstance(
            getattr(module, "layer", None), nn.Linear,
        ):
            return True
        return False

    @staticmethod
    def _orient_2d_for_columns(src_arr, module: nn.Module):
        """Pre-existing heuristic: align ``src_arr[:, i]`` with module in_features.

        Sequential modules (or modules with an explicit ``in_features``) may
        receive sources in either ``(features, instances)`` or
        ``(instances, features)`` orientation depending on the upstream
        StackMapper/EinopsRearrange path.  This mirrors the legacy
        ``ModuleComputeMapper._map_to_ir`` heuristic.
        """
        in_features = getattr(module, "in_features", None)
        if in_features is None and hasattr(module, "0"):
            in_features = getattr(module[0], "in_features", None)
        if in_features is not None:
            if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                return src_arr.T
        return src_arr


class ModuleMapper(Mapper):
    """
    Forward-only module application mapper.
    For mapping (chip compilation), this acts as identity (passes sources through).
    """

    def __init__(self, source_mapper, module: nn.Module):
        super(ModuleMapper, self).__init__(source_mapper)
        self.module = module

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

    def _forward_impl(self, x):
        return self.module(x)
