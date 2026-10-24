"""Stamp the solved core laws onto the model's readout quantizers, so NF and the chip read one law."""

from __future__ import annotations

from typing import Any, Dict, Hashable, List, Mapping, Tuple

import numpy as np
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.entry_offsets import core_entry_offset, offset_multipliers, unfold_entry_offset
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.models.nn.activations.readout_quantizer import (
    LAW_FIELDS,
    OFFSET_MULTIPLIER,
    ReadoutQuantizer,
    ReadoutStamp,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.quantization_bounds import layer_weight_bits, quantization_bounds


class ReadoutLawPerceptronFormError(ToolchainRefusal, ValueError):
    """A perceptron whose forward is not the fused affine op its cores compute, so NF cannot read the law."""


class ReadoutLawChannelError(ToolchainRefusal, ValueError):
    """Cores carrying one perceptron channel whose laws differ (the model reads one law per output channel)."""


def readout_quantizers_by_perceptron(model_repr) -> Dict[int, Tuple[Any, Any, ReadoutQuantizer]]:
    """{perceptron index: (owning mapper, perceptron, its readout quantizer)}."""
    order, _, _ = model_repr._ensure_exec_graph()
    found = {}
    for node in order:
        perceptron = getattr(node, "perceptron", None)
        index = getattr(node, "perceptron_index", None)
        if perceptron is not None and index is not None and isinstance(perceptron.scaler, ReadoutQuantizer):
            found[int(index)] = (node, perceptron, perceptron.scaler)
    return found


def core_channels(core: NeuralCore) -> np.ndarray:
    """The perceptron output channels a core's columns compute, in column order (IR pruning included)."""
    count = int(core.get_output_count())
    start, end = core.perceptron_output_slice or (0, count)
    channels = np.arange(int(start), int(end))
    mask = getattr(core, "pre_pruning_col_mask", None)
    if core.weight_bank_id is None and mask is not None and len(mask) == channels.size:
        channels = channels[~np.asarray(mask, dtype=bool)]
    if channels.size != count:
        raise ValueError(f"core {core.name!r}: {count} columns but {channels.size} perceptron channels")
    return channels


def channel_keys(core: NeuralCore) -> Tuple[Hashable, ...]:
    """Per column, the channel the model reads its law from: ``(perceptron index, output channel)``, or the core's
    own column where it belongs to no perceptron. Cores sharing a key must carry one law in that column."""
    if core.perceptron_index is None:
        return tuple(("core", int(core.id), j) for j in range(int(core.get_output_count())))
    return tuple((int(core.perceptron_index), int(c)) for c in core_channels(core))


def differing_fields(law, column: int, other, other_column: int, names) -> List[str]:
    """The law fields ``names`` whose values differ between two columns (floats by value)."""
    return [name for name in names if getattr(law, name)[column] != getattr(other, name)[other_column]]


def _parameter_name(module: nn.Module, parameter) -> str:
    for name, candidate in module.named_parameters():
        if candidate is parameter:
            return name
    raise ReadoutLawPerceptronFormError(f"{type(module).__name__} does not own the perceptron's parameter")


def _channel_law(cores: List[NeuralCore], laws: Mapping[int, CoreLaw], channels: int, ir_graph):
    """(per-channel law arrays, per-channel theta, input step) assembled from a perceptron's cores; an entry
    offset's folded term is undone per core (its positions differ) and carried as ``OFFSET_MULTIPLIER``."""
    arrays = {name: np.array(getattr(CoreLaw.blank(channels), name)) for name in LAW_FIELDS}
    theta = np.zeros(channels)
    digests: Dict[int, Tuple[str, CoreLaw, int, str]] = {}
    steps = set()
    for core in cores:
        law = unfold_entry_offset(laws[int(core.id)], core, ir_graph)
        offset = core_entry_offset(core)
        if offset != 0.0:
            multipliers = offset_multipliers(law, float(core.threshold), offset)
            column_multipliers = arrays.setdefault(OFFSET_MULTIPLIER, np.zeros(channels, dtype=np.int64))
            column_multipliers[core_channels(core)] = multipliers
        for column, (channel, digest) in enumerate(zip(core_channels(core), law.column_digests())):
            seen = digests.setdefault(int(channel), (digest, law, column, core.name))
            if seen[0] != digest:
                fields = differing_fields(law, column, seen[1], seen[2], LAW_FIELDS)
                raise ReadoutLawChannelError(
                    f"core {core.name!r}: channel {channel} disagrees with core {seen[3]!r}'s law in {fields}: the "
                    f"model reads one law per channel")
            for name in LAW_FIELDS:
                arrays[name][channel] = getattr(law, name)[column]
            theta[channel] = float(core.threshold)
        steps.update(float(s) for s in law.input_step)
    if len(steps) != 1:
        raise ValueError(f"cores {[c.name for c in cores][:4]} of one perceptron read {len(steps)} input grids")
    return arrays, theta, steps.pop()


def _check_form(perceptron, index: int) -> np.ndarray:
    """The perceptron's effective weight (fp64), refused unless it is the layer's own weight."""
    if not isinstance(perceptron.normalization, nn.Identity):
        raise ReadoutLawPerceptronFormError(f"perceptron {index}: normalization is not fused into its layer")
    effective = PerceptronTransformer().get_effective_weight(perceptron)
    if not bool((effective == perceptron.layer.weight.data).all()):
        raise ReadoutLawPerceptronFormError(
            f"perceptron {index}: its effective weight differs from its layer weight (input or activation scale)"
        )
    return effective.detach().double().cpu().numpy()


#: Public readings for the lane stamp (WI-5e).
perceptron_layer_weight = _check_form
parameter_name = _parameter_name


def stamp_model_laws(model, ir_graph, laws: Mapping[int, CoreLaw], *, weight_bits: int) -> int:
    """Stamp every perceptron's channel law and integer weight grid; unmapped perceptrons drop their quantizer."""
    by_perceptron: Dict[int, List[NeuralCore]] = {}
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore) and node.perceptron_index is not None:
            by_perceptron.setdefault(int(node.perceptron_index), []).append(node)
    stamped = 0
    for index, (mapper, perceptron, quantizer) in readout_quantizers_by_perceptron(model.get_mapper_repr()).items():
        cores = by_perceptron.get(index)
        if not cores:
            perceptron.set_scaler(nn.Identity())
            continue
        effective = _check_form(perceptron, index)
        arrays, theta, input_step = _channel_law(cores, laws, int(effective.shape[0]), ir_graph)
        shaped = theta.reshape((-1,) + (1,) * (effective.ndim - 1))
        p_min, p_max = quantization_bounds(layer_weight_bits(perceptron, weight_bits))  # [WI-5g] its type's grid
        grid = np.where(shaped > 0, np.clip(np.round(effective * shaped), p_min, p_max), 0.0)
        bias = perceptron.layer.bias
        quantizer.stamp(ReadoutStamp(
            owner=(mapper, perceptron),
            weight_name=_parameter_name(mapper, perceptron.layer.weight),
            bias_name=None if bias is None else _parameter_name(mapper, bias),
            weight_grid=grid, input_step=input_step, law=arrays,
            channel_axis=int(perceptron.output_channel_axis),
        ))
        perceptron.append_input_wire_op(quantizer.input_tap())
        stamped += 1
    return stamped
