"""Stamp the solved core laws onto the model's readout quantizers, so NF and the chip read one law."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.models.nn.activations.readout_quantizer import (
    LAW_FIELDS,
    ReadoutQuantizer,
    ReadoutStamp,
)
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.quantization_bounds import quantization_bounds


class ReadoutLawPerceptronFormError(ToolchainRefusal, ValueError):
    """A perceptron whose forward is not the fused affine op its cores compute, so NF cannot read the law."""


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


def _parameter_name(module: nn.Module, parameter) -> str:
    for name, candidate in module.named_parameters():
        if candidate is parameter:
            return name
    raise ReadoutLawPerceptronFormError(f"{type(module).__name__} does not own the perceptron's parameter")


def _channel_law(cores: List[NeuralCore], laws: Mapping[int, CoreLaw], channels: int):
    """(per-channel law arrays, per-channel theta, input step) assembled from a perceptron's cores."""
    arrays = {name: np.array(getattr(CoreLaw.blank(channels), name)) for name in LAW_FIELDS}
    theta = np.zeros(channels)
    digests: Dict[int, str] = {}
    steps = set()
    for core in cores:
        law = laws[int(core.id)]
        for column, (channel, digest) in enumerate(zip(core_channels(core), law.column_digests())):
            if digests.setdefault(int(channel), digest) != digest:
                raise ValueError(f"core {core.name!r}: channel {channel} disagrees with another core's law")
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


def stamp_model_laws(model, ir_graph, laws: Mapping[int, CoreLaw], *, weight_bits: int) -> int:
    """Stamp every perceptron's channel law and integer weight grid; unmapped perceptrons drop their quantizer."""
    q_min, q_max = quantization_bounds(weight_bits)
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
        arrays, theta, input_step = _channel_law(cores, laws, int(effective.shape[0]))
        shaped = theta.reshape((-1,) + (1,) * (effective.ndim - 1))
        grid = np.where(shaped > 0, np.clip(np.round(effective * shaped), q_min, q_max), 0.0)
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
