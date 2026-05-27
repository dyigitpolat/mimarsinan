from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mimarsinan.mapping.platform.mapping_structure import compute_psum_params


def map_fc_with_psum(
    mapper,
    *,
    input_sources: np.ndarray,
    fc_weights: Any,
    fc_biases: Any,
    activation_scale: Any,
    parameter_scale: Any,
    input_activation_scale: Any,
    name: Optional[str],
    normalization_type: Optional[str],
    activation_type: Optional[str],
    perceptron_index: Optional[int],
) -> np.ndarray:
    out_features = int(getattr(fc_weights, "shape", [0, 0])[0])
    in_features = int(getattr(fc_weights, "shape", [0, 0])[1])

    pp = compute_psum_params(
        in_features, out_features,
        int(mapper.max_axons), mapper.max_neurons,
        fc_biases is not None, mapper.hardware_bias,
    )

    src_arr = np.array(input_sources, dtype=object).flatten()
    group_id = mapper._psum_group_counter
    mapper._psum_group_counter += 1

    all_output_sources: list[np.ndarray] = []
    a = 0
    while a < out_features:
        b = min(out_features, a + pp.out_block_size)
        block = b - a
        w_block = fc_weights[a:b, :] if fc_weights is not None else None
        b_block = fc_biases[a:b] if fc_biases is not None else None

        partial_pos_sources: list[np.ndarray] = []
        partial_neg_sources: list[np.ndarray] = []
        for t_idx, (ta, tb) in enumerate(pp.tile_slices):
            w_tile = w_block[:, ta:tb] if w_block is not None else None
            tile_src = src_arr[ta:tb]

            pos_out = mapper.add_neural_core(
                input_sources=tile_src,
                weights=w_tile,
                biases=None,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=(f"{name}_psum_pos_g{group_id}_t{t_idx}_o{a}_{b}" if name else None),
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                perceptron_input_slice=(ta, tb),
                perceptron_output_slice=(a, b),
                psum_group_id=group_id,
                psum_role="partial_pos",
            )
            neg_out = mapper.add_neural_core(
                input_sources=tile_src,
                weights=w_tile,
                biases=None,
                activation_scale=activation_scale,
                parameter_scale=parameter_scale,
                input_activation_scale=input_activation_scale,
                name=(f"{name}_psum_neg_g{group_id}_t{t_idx}_o{a}_{b}" if name else None),
                normalization_type=normalization_type,
                activation_type=activation_type,
                perceptron_index=perceptron_index,
                perceptron_input_slice=(ta, tb),
                perceptron_output_slice=(a, b),
                psum_group_id=group_id,
                psum_role="partial_neg",
            )
            partial_pos_sources.append(pos_out)
            partial_neg_sources.append(neg_out)

        acc_input_list = []
        for t_idx in range(pp.tile_count):
            for n in range(block):
                acc_input_list.append(partial_pos_sources[t_idx][n])
        for t_idx in range(pp.tile_count):
            for n in range(block):
                acc_input_list.append(partial_neg_sources[t_idx][n])

        from mimarsinan.mapping.platform.mapping_structure import build_psum_accumulator_weights

        acc_w = build_psum_accumulator_weights(block, pp.tile_count, parameter_scale)

        acc_out = mapper.add_neural_core(
            input_sources=np.array(acc_input_list, dtype=object),
            weights=acc_w,
            biases=b_block,
            activation_scale=activation_scale,
            parameter_scale=parameter_scale,
            input_activation_scale=input_activation_scale,
            name=(f"{name}_psum_accum_g{group_id}_o{a}_{b}" if name else None),
            normalization_type=normalization_type,
            activation_type=activation_type,
            perceptron_index=perceptron_index,
            perceptron_output_slice=(a, b),
            psum_group_id=group_id,
            psum_role="accum",
        )
        all_output_sources.append(acc_out)
        a = b

    return np.concatenate(all_output_sources)
