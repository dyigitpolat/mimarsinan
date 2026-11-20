"""[U-PFOLD] The cores of a perceptron whose fold is held per position (``transformations.perceptron.position_fold``).

The program maps a linear over leading positions to one core per position and a convolution to one core per output
position, so each position's core folds its own row. Positions whose weight row and bias are equal share one weight bank
and its shared cores (the banked path a perceptron of one fold takes over every position); every other row has its own
bank. A bank serves positions of one fold by construction (each distinct row its own bank), so the fold's refusal, kept
for positions sharing one core (``assert_one_fold_per_core``), is an assertion no program the mappers build reaches.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mimarsinan.mapping.layout.layout_source_view_ops import concat_source_views, stack_source_views
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import (
    PER_INPUT,
    PER_POSITION,
    ROW_AXIS,
    PositionFoldError,
    assert_one_fold_per_core,
    effective_bias_rows,
    effective_weight_rows,
    outputs_first,
    position_folds,
)

__all__ = ["position_banks", "map_linear_positions"]


def position_banks(perceptron: Any) -> tuple[torch.Tensor, torch.Tensor, list]:
    """``(weights (rows, out, in), biases (positions, out), groups)``: each group a list of positions of one
    weight row and one bias, in order of first position (one bank set each), the fold asserted one per group."""
    weights, index = effective_weight_rows(perceptron)
    biases = effective_bias_rows(perceptron, PerceptronTransformer().shared_effective_bias(perceptron))
    stamp = position_folds(perceptron)  # each position's fold row, the one reader's
    folds, held = stamp.detach().cpu().numpy(), biases.detach().cpu().numpy()
    keys: dict = {}
    groups: list = []
    for p in range(int(index.numel())):
        key = folds[p].tobytes() + held[p].tobytes()  # one fold and one bias: the rows of one bank
        if key not in keys:
            keys[key] = len(groups)
            groups.append([])
        groups[keys[key]].append(p)
    for group in groups:
        assert_one_fold_per_core(getattr(perceptron, "name", None), stamp[group])
    return weights, biases, groups


def map_linear_positions(mapper: Any, ir_mapping: Any, layer_sources: np.ndarray, row_values: Any, common: dict):
    """A linear's positions (the columns of ``layer_sources``, ``(in_features, positions)``) each on a core of its
    row's bank (``map_fc``'s banked path where the platform tiles banks, else each position an owned core), named and
    numbered as the shared path numbers them; returns ``(out, positions)``."""
    perceptron = mapper.perceptron
    weights, biases, groups = position_banks(perceptron)
    fan_in, out = int(weights.shape[-1]), int(weights.shape[PER_POSITION.weight])
    positions = int(layer_sources.shape[-1]) if layer_sources.ndim == 2 else 1
    if layer_sources.ndim != 2 or int(layer_sources.shape[0]) != fan_in or positions != sum(map(len, groups)):
        raise PositionFoldError(
            f"{common.get('name')!r}: a fold of {sum(map(len, groups))} position rows over an input of shape "
            f"{tuple(layer_sources.shape)}: its rows run along the input's leading positions, its last axis the fan-in")
    name, scales = common.pop("name"), common.pop("scales")
    _rows, index = effective_weight_rows(perceptron)
    bias_rows = row_values(biases)
    bias_rows_count = ir_mapping.param_encoded_bias_rows(common.get("bias_scale"), scales[1], name=name)
    tiles = ir_mapping._fc_bank_tiles(fan_in, out, True, bias_rows_count)
    columns: list = [None] * positions
    for group in groups:
        # the bank's layout (the IR's: output neuron first) from the row's per-input view
        weight = outputs_first(weights.select(ROW_AXIS, int(index[group[0]])), PER_INPUT.weight)
        bias = outputs_first(bias_rows.select(ROW_AXIS, group[0]), PER_INPUT.bias)
        if tiles is None:
            for p in group:
                columns[p] = ir_mapping.map_fc(
                    _column(layer_sources, p), np.array([out, 1]), weight, bias, *scales,
                    name=f"{name}_col{p}" if name else None, perceptron_output_column=p, **common).flatten()
            continue
        full = (0, out)
        banks = [ir_mapping.register_weight_bank(
            weights=weight if (s, e) == full else weight[s:e, :],
            biases=bias if (s, e) == full else bias[s:e],
            activation_scale=scales[0], parameter_scale=scales[1], input_activation_scale=scales[2],
            perceptron_index=common.get("perceptron_index"), bias_scale=common.get("bias_scale")) for s, e in tiles]
        tiled = len(tiles) > 1
        for p in group:
            col_name = f"{name}_col{p}" if name else None
            views = [ir_mapping.add_shared_neural_core(
                input_sources=_column(layer_sources, p), weight_bank_id=bank, has_bias=True,
                name=f"{col_name}_tile_{s}_{e}" if (tiled and col_name) else col_name,
                normalization_type=common.get("normalization_type"), activation_type=common.get("activation_type"),
                perceptron_index=common.get("perceptron_index"),
                perceptron_output_slice=(s, e) if tiled else (full if common.get("perceptron_index") is not None
                                                              else None),
                perceptron_output_column=p) for (s, e), bank in zip(tiles, banks)]
            columns[p] = (views[0] if len(views) == 1 else concat_source_views(views)).flatten()
    return stack_source_views(columns, axis=1).reshape((out, positions))


def _column(layer_sources: np.ndarray, p: int):
    column = layer_sources[:, p]
    return column.flatten() if hasattr(column, "flatten") else column
