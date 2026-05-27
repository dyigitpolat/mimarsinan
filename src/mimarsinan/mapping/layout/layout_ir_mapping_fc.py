from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from mimarsinan.mapping.layout.layout_source_view_ops import (
    concat_source_views,
    stack_source_views,
)
from mimarsinan.mapping.platform.mapping_structure import compute_fc_tiling_mode
from mimarsinan.mapping.layout.layout_ir_mapping_fc_psum import map_fc_with_psum


class _LayoutIRMappingFC:

    def map_fc(
        self,
        input_tensor_sources: np.ndarray,
        output_shape: np.ndarray,
        fc_weights: Any,
        fc_biases: Any = None,
        activation_scale: Any = None,
        parameter_scale: Any = None,
        input_activation_scale: Any = None,
        name: Optional[str] = None,
        normalization_type: Optional[str] = None,
        activation_type: Optional[str] = None,
        perceptron_index: Optional[int] = None,
        psum_group_id: Optional[int] = None,
        psum_role: Optional[str] = None,
        coalescing_group_id: Optional[int] = None,
        coalescing_role: Optional[str] = None,
    ) -> np.ndarray:
        """Decide tiling mode and dispatch to ``add_neural_core`` (or the
        psum / output-tiled helpers).  All structural decisions live here."""
        out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
        in_features = int(getattr(fc_weights, "shape", [0, 0])[1])

        src_arr = input_tensor_sources

        # Batched FC: (in_features, core_count) -> dispatch column-by-column.
        if src_arr.ndim == 2:
            if src_arr.shape[0] != in_features and src_arr.shape[1] == in_features:
                src_arr = src_arr.T
            if src_arr.shape[0] != in_features:
                raise ValueError(
                    f"map_fc: input sources first dim must match in_features "
                    f"({src_arr.shape} vs in_features={in_features})"
                )
            core_count = int(src_arr.shape[1])
            outs = []
            for i in range(core_count):
                col_sources = src_arr[:, i]
                if hasattr(col_sources, "flatten"):
                    col_sources = col_sources.flatten()
                outs.append(
                    self.map_fc(
                        col_sources,
                        np.array([out_features, 1]),
                        fc_weights,
                        fc_biases,
                        activation_scale,
                        parameter_scale,
                        input_activation_scale,
                        name=(f"{name}_col{i}" if name else None),
                        normalization_type=normalization_type,
                        activation_type=activation_type,
                        perceptron_index=perceptron_index,
                        psum_group_id=psum_group_id,
                        psum_role=psum_role,
                        coalescing_group_id=coalescing_group_id,
                        coalescing_role=coalescing_role,
                    ).flatten()
                )
            out = stack_source_views(outs, axis=1)
            return out.reshape(tuple(output_shape))

        has_bias = fc_biases is not None
        mode = compute_fc_tiling_mode(
            in_features, out_features,
            self.max_axons, self.max_neurons,
            has_bias, self.hardware_bias, self.allow_coalescing,
        )

        if mode == "psum":
            return map_fc_with_psum(
                self,
                input_sources=src_arr.flatten(),
                fc_weights=fc_weights,
                fc_biases=fc_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=name,
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
            )

        if mode == "coalescing" and coalescing_group_id is None:
            coalescing_group_id = self._coalescing_group_counter
            self._coalescing_group_counter += 1
            coalescing_role = "master"

        wide_and_output_tiled = (
            mode == "coalescing"
            and self.max_neurons is not None
            and out_features > self.max_neurons
        )
        if mode == "output_tiled" or wide_and_output_tiled:
            return self._map_fc_output_tiled(
                src_arr=src_arr,
                fc_weights=fc_weights,
                fc_biases=fc_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=name,
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                psum_group_id=psum_group_id,
                psum_role=psum_role,
                coalescing_group_id=coalescing_group_id,
                coalescing_role=coalescing_role,
            )

        # "single" or non-output-tiled "coalescing"
        fc_input = src_arr.T if src_arr.ndim > 1 else src_arr
        return self.add_neural_core(
            input_sources=fc_input,
            weights=fc_weights,
            biases=fc_biases,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            name=name,
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            psum_group_id=psum_group_id,
            psum_role=psum_role,
            coalescing_group_id=coalescing_group_id,
            coalescing_role=coalescing_role,
        )

    def _map_fc_output_tiled(
        self,
        *,
        src_arr,
        fc_weights: Any,
        fc_biases: Any,
        activation_scale: Any,
        parameter_scale: Any,
        input_activation_scale: Any,
        name: Optional[str],
        normalization_type: Optional[str],
        activation_type: Optional[str],
        perceptron_index: Optional[int],
        psum_group_id: Optional[int],
        psum_role: Optional[str],
        coalescing_group_id: Optional[int],
        coalescing_role: Optional[str],
    ):
        out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
        chunk_size = int(self.max_neurons)

        output_sources_list = []
        start = 0
        while start < out_features:
            end = min(start + chunk_size, out_features)
            tile_weights = fc_weights[start:end, :] if fc_weights is not None else None
            tile_biases = fc_biases[start:end] if fc_biases is not None else None

            tile_sources = self.add_neural_core(
                input_sources=src_arr.flatten() if hasattr(src_arr, "flatten") else src_arr,
                weights=tile_weights,
                biases=tile_biases,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=(f"{name}_tile_{start}_{end}" if name else None),
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                perceptron_output_slice=(start, end),
                psum_group_id=psum_group_id,
                psum_role=psum_role,
                coalescing_group_id=coalescing_group_id,
                coalescing_role=coalescing_role,
            )
            output_sources_list.append(tile_sources)
            start = end
        return concat_source_views(output_sources_list)

