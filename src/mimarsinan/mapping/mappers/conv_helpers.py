"""Shared helpers for the shared-weight convolution mappers."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, cast

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.mappers.base import resolve_activation_type
from mimarsinan.mapping.mappers.position_cores import position_banks
from mimarsinan.transformations.perceptron.bias_drive import bias_row_values
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_fold import (
    PER_INPUT, ROW_AXIS, effective_weight_rows, holds_positions, outputs_first)


def _chunk_sizes(total: int, chunk: int):
    assert chunk > 0
    sizes = []
    remaining = int(total)
    while remaining > 0:
        sizes.append(min(chunk, remaining))
        remaining -= sizes[-1]
    return sizes


def pad_source_grid(
    input_sources: np.ndarray,
    pad_width: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Pad an IRSource object grid with OFF sources (np.pad's stubs reject object fills)."""
    off_source = IRSource(node_id=-1, index=0)
    return np.pad(
        input_sources,
        pad_width,
        mode="constant",
        constant_values=cast(Any, off_source),
    )


def perceptron_conv_padding(conv: Any) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    """``(padding, None)``: the numeric zero padding the perceptron patch gather applies; ``(None, reason)`` if none.

    Without padding the ``padding_mode`` has no effect, so any mode with zero padding amounts is zero padding 0.
    """
    padding, reason = _numeric_padding(conv)
    if padding is not None and any(padding) and conv.padding_mode != "zeros":
        return None, f"padding_mode {conv.padding_mode!r} would be converted as zero padding"
    return padding, reason


def _numeric_padding(conv: Any) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    padding = conv.padding
    if padding == "valid":
        return tuple(0 for _ in conv.kernel_size), None
    if padding == "same":
        totals = [int(d) * (int(k) - 1) for d, k in zip(conv.dilation, conv.kernel_size)]
        if any(total % 2 for total in totals):
            return None, (
                f"padding 'same' needs asymmetric zero padding (totals {totals} per axis, odd), which the "
                "perceptron patch gather cannot express"
            )
        return tuple(total // 2 for total in totals), None
    if isinstance(padding, str):
        return None, f"string padding {padding!r} is not supported"
    return tuple(int(p) for p in padding), None


def conv_output_size(
    in_size: Sequence[int],
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
) -> tuple[int, ...]:
    """Per spatial axis, torch's ConvNd output length."""
    return tuple(
        (int(i) + 2 * int(p) - int(d) * (int(k) - 1) - 1) // int(s) + 1
        for i, k, s, p, d in zip(in_size, kernel_size, stride, padding, dilation, strict=True)
    )


def gather_conv_patches(
    padded_sources: np.ndarray,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    output_size: Sequence[int],
) -> np.ndarray:
    """im2col of a padded (C, *spatial) grid: one row per row-major output position, columns in
    the (C, k_1, ..., k_n) order of ``ConvNd.weight.reshape(out_channels, -1)``."""
    n_axes = len(kernel_size)
    rank = 2 * n_axes + 1
    channels = int(padded_sources.shape[0])
    channel_shape = [1] * rank
    channel_shape[n_axes] = channels
    index = [np.arange(channels).reshape(channel_shape)]
    for axis in range(n_axes):
        taps = (
            np.arange(int(output_size[axis]))[:, None] * int(stride[axis])
            + np.arange(int(kernel_size[axis]))[None, :] * int(dilation[axis])
        )
        tap_shape = [1] * rank
        tap_shape[axis] = int(output_size[axis])
        tap_shape[n_axes + 1 + axis] = int(kernel_size[axis])
        index.append(taps.reshape(tap_shape))
    patches = padded_sources[tuple(np.broadcast_arrays(*index))]
    n_positions = int(np.prod(output_size))
    return patches.reshape(n_positions, channels * int(np.prod(kernel_size)))


def _position_label(position: int, output_size: Sequence[int]) -> str:
    coords = []
    for extent in reversed(output_size):
        position, coord = divmod(position, int(extent))
        coords.append(str(coord))
    return "_".join(reversed(coords))


def map_shared_weight_conv(
    mapper: Any,
    ir_mapping: Any,
    input_sources: np.ndarray,
    *,
    kernel_size: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    layout: str,
) -> np.ndarray:
    """Lower a shared-weight ConvNd perceptron mapper: one weight bank per output-channel group and
    one core per (output position, group) reading that position's patch; returns (C_out, *out)."""
    n_axes = len(kernel_size)
    if len(input_sources.shape) != n_axes + 1:
        raise ValueError(
            f"{mapper.name}: expects {n_axes + 1}D input sources {layout}, got {input_sources.shape}"
        )
    c_in, *in_size = input_sources.shape
    if int(c_in) != mapper.in_channels:
        raise ValueError(
            f"{mapper.name}: expected in_channels={mapper.in_channels}, got {c_in}"
        )
    out_size = conv_output_size(in_size, kernel_size, stride, padding, dilation)
    out_channels = mapper.out_channels
    perceptron = mapper.perceptron

    if getattr(perceptron, "is_encoding_layer", False):
        flat_in = np.array(input_sources, dtype=object).flatten()
        out = ir_mapping.add_compute_op(
            input_sources=flat_in,
            op_type="module",
            # Two equal tuples, not one shared object: the IR pickles as before the shared lowering.
            params={"module": mapper, "input_shape": (c_in, *in_size)},
            input_shape=(c_in, *in_size),
            output_shape=(out_channels, *out_size),
            name=mapper.name,
        )
        return out.reshape(out_channels, *out_size)

    if any(p > 0 for p in padding):
        input_sources = pad_source_grid(
            input_sources, ((0, 0),) + tuple((int(p), int(p)) for p in padding)
        )

    if holds_positions(perceptron):
        # [U-PFOLD] one bank set per distinct patch fold (and bias): each output position's core folds its own patch;
        # a weight row (and a position's bias) taken on the row axis is in the per-input form.
        weights, biases, groups = position_banks(perceptron)
        _rows, row_of = effective_weight_rows(perceptron)
        bank_inputs = [(outputs_first(weights.select(ROW_AXIS, int(row_of[g[0]])), PER_INPUT.weight),
                        bias_row_values(perceptron, outputs_first(biases.select(ROW_AXIS, g[0]), PER_INPUT.bias)))
                       for g in groups]
        group_of = {p: i for i, g in enumerate(groups) for p in g}
    else:
        # The always-on row's weight: the bias over the row's drive (``transformations.perceptron.bias_drive``).
        weight, bias = PerceptronTransformer().get_effective_parameters(perceptron)  # the per-input reader's views
        bank_inputs = [(outputs_first(weight, PER_INPUT.weight),
                        bias_row_values(perceptron, outputs_first(bias, PER_INPUT.bias)))]
        group_of = None
    has_bias = bank_inputs[0][1] is not None
    activation_type = resolve_activation_type(perceptron)
    perceptron_index = getattr(mapper, "perceptron_index", None)

    if mapper.max_neurons is None:
        group_sizes = [out_channels]
    else:
        group_sizes = _chunk_sizes(out_channels, int(mapper.max_neurons))

    bank_sets: list[list[int]] = []
    bank_channel_ranges: list[tuple[int, int]] = []
    for full_w, full_b in bank_inputs:
        bank_ids: list[int] = []
        bank_channel_ranges = []
        start_idx = 0
        for g in group_sizes:
            end_idx = start_idx + g
            bank_ids.append(ir_mapping.register_weight_bank(
                weights=full_w[start_idx:end_idx, :],
                biases=full_b[start_idx:end_idx] if has_bias else None,
                activation_scale=perceptron.activation_scale,
                parameter_scale=perceptron.parameter_scale,
                input_activation_scale=perceptron.input_activation_scale,
                perceptron_index=perceptron_index,
                bias_scale=getattr(perceptron, "bias_scale", None),
            ))
            bank_channel_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        bank_sets.append(bank_ids)

    patches = gather_conv_patches(input_sources, kernel_size, stride, dilation, out_size)

    all_output_sources = []
    for pos in range(patches.shape[0]):
        label = _position_label(pos, out_size)
        position_outputs = []
        for g_idx, bank_id in enumerate(bank_sets[0 if group_of is None else group_of[pos]]):
            position_outputs.append(ir_mapping.add_shared_neural_core(
                input_sources=patches[pos],
                weight_bank_id=bank_id,
                has_bias=has_bias,
                perceptron_output_slice=bank_channel_ranges[g_idx],
                perceptron_output_column=pos,
                name=f"{mapper.name}_pos{label}_g{g_idx}",
                activation_type=activation_type,
                perceptron_index=perceptron_index,
            ))
        all_output_sources.append(np.concatenate(position_outputs))

    output_array = np.stack(all_output_sources, axis=0).T
    return output_array.reshape(out_channels, *out_size)
