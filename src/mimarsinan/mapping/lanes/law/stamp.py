"""Stamp a lane program's laws on the model (WI-5e), so NF reads every core's law: a perceptron's per channel with
its lane clauses, a pool lane's on its :class:`PoolStamp`."""

from __future__ import annotations

import dataclasses
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

import numpy as np
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.entry_offsets import offset_multipliers
from mimarsinan.mapping.export.law_stamp import (
    channel_keys,
    core_channels,
    parameter_name,
    perceptron_layer_weight,
    readout_quantizers_by_perceptron,
)
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.lane_law import LaneLaw
from mimarsinan.mapping.lanes.ir.fusion import LANE_OWNER, LaneFusion
from mimarsinan.mapping.lanes.plan import LanePlan
from mimarsinan.mapping.lanes.ir.pool_cores import LANE_OUTPUTS
from mimarsinan.mapping.lanes.ir.reduce_cores import PRODUCER_THETA
from mimarsinan.mapping.lanes.law.solve import data_row_weight_sums
from mimarsinan.models.nn.activations.lane_quantization import LanePoolModule, PoolStamp
from mimarsinan.models.nn.activations.readout_quantizer import (
    LAW_FIELDS,
    OFFSET_MULTIPLIER,
    OPERAND_M,
    OPERAND_OFFSET,
    OPERAND_STEP,
    OUTPUT_OFFSET,
    ReadoutStamp,
)
from mimarsinan.transformations.quantization_bounds import layer_weight_bits, quantization_bounds

class LaneStampError(ToolchainRefusal, ValueError):
    """A lane program whose per-core laws the model cannot read per channel (one law per output channel)."""


#: The lane clauses a perceptron's stamp carries per channel (the operand's only on a fused add).
_OUTPUT_KEYS = {OUTPUT_OFFSET: "output_offset"}
_OPERAND_KEYS = {OPERAND_M: "operand_m", OPERAND_STEP: "operand_step", OPERAND_OFFSET: "operand_offset"}


def _unfolded(law: LaneLaw, core: NeuralCore, ir_graph) -> Tuple[LaneLaw, np.ndarray]:
    """The per-channel law the core's was folded from, and its offset multipliers (zeros without an offset)."""
    offset = float(law.input_offset[0]) if law.columns else 0.0
    if offset == 0.0:
        return law, np.zeros(law.columns, dtype=np.int64)
    multipliers = offset_multipliers(law, float(core.threshold), offset)
    sums = data_row_weight_sums(core, ir_graph)
    return dataclasses.replace(law, bias=law.bias + multipliers * sums), multipliers


def _channel_arrays(cores: List[NeuralCore], laws: Mapping[int, LaneLaw], channels, ir_graph, *, operand: bool,
                    column_channels) -> Tuple[Dict[str, np.ndarray], float, float]:
    keys = list(LAW_FIELDS) + list(_OUTPUT_KEYS) + (list(_OPERAND_KEYS) if operand else [])
    fields = {**{name: name for name in LAW_FIELDS}, **_OUTPUT_KEYS, **_OPERAND_KEYS}
    arrays = {key: np.array(getattr(LaneLaw.blank(channels), fields[key])) for key in keys}
    offsets = np.zeros(channels, dtype=np.int64)
    digests: Dict[int, str] = {}
    steps, input_offsets = set(), set()
    for core in cores:
        law, multipliers = _unfolded(laws[int(core.id)], core, ir_graph)
        for column, (channel, digest) in enumerate(zip(column_channels(core), law.column_digests())):
            # The operand row differs across a perceptron's tiles; the lane function per channel does not.
            if digests.setdefault(int(channel), digest) != digest:
                differing = [key for key in keys if arrays[key][channel] != getattr(law, fields[key])[column]]
                if differing:
                    raise LaneStampError(f"core {core.name!r}: channel {channel} disagrees with another core's law in "
                                         f"{differing}{_cause(differing)}: the model reads one law per channel")
            for key in keys:
                arrays[key][channel] = getattr(law, fields[key])[column]
            offsets[channel] = multipliers[column]
        steps.update(float(s) for s in law.input_step)
        input_offsets.update(float(o) for o in law.input_offset)
    if len(steps) != 1 or len(input_offsets) != 1:
        raise ValueError(f"cores {[c.name for c in cores][:4]} of one lane read {len(steps)} input grids")
    if offsets.any():
        arrays[OFFSET_MULTIPLIER] = offsets
    return arrays, steps.pop(), input_offsets.pop()


def _cause(differing) -> str:
    """The configuration a channel's differing lane fields point at (an operand of two lane grids)."""
    if set(differing) & set(_OPERAND_KEYS):
        return (" (a fused operand reading two lane grids within one channel, as a concatenation of two lanes along "
                "height or width)")
    return ""


def _reduced_arrays(cores: List[NeuralCore], laws: Mapping[int, LaneLaw], size: int):
    """A reduction's law per pool output element (flat, channel-major), as the IR solved and folded it (its bias holds
    each window's present instances and its input offset's term), the one input grid its cores read and the
    producer's theta."""
    keys = list(LAW_FIELDS) + list(_OUTPUT_KEYS)
    fields = {**{name: name for name in LAW_FIELDS}, **_OUTPUT_KEYS}
    arrays = {key: np.array(getattr(LaneLaw.blank(size), fields[key])) for key in keys}
    filled = np.zeros(size, dtype=bool)
    steps, offsets, thetas = set(), set(), set()
    for core in cores:
        law, outputs = laws[int(core.id)], np.asarray(getattr(core, LANE_OUTPUTS))
        for key in keys:
            arrays[key][outputs] = getattr(law, fields[key])
        filled[outputs] = True
        steps.update(float(v) for v in law.input_step)
        offsets.update(float(v) for v in law.input_offset)
        thetas.add(float(getattr(core, PRODUCER_THETA)))
    if not filled.all() or len(steps) != 1 or len(offsets) != 1 or len(thetas) != 1:
        raise ValueError(f"reduced cores {[c.name for c in cores][:4]} cover {int(filled.sum())} of {size} outputs "
                         f"with {len(steps)} input grids and {len(thetas)} thresholds")
    return arrays, steps.pop(), offsets.pop(), thetas.pop()


def _stamp_perceptrons(model, ir_graph, laws, plan: LanePlan, weight_bits: int,
                       fusion: Optional[LaneFusion] = None) -> int:
    adds = {id(lane.holder.perceptron) for lane in plan.perceptron_lanes if lane.add is not None}
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
        effective = perceptron_layer_weight(perceptron, index)
        form = None if fusion is None else fusion.reduce_forms.get(index)
        if form is not None:  # a producer-side reduction: the law per pool output, theta the producer's
            size = sum(core.get_output_count() for core in cores)
            arrays, input_step, input_offset, producer_theta = _reduced_arrays(cores, laws, size)
            theta = np.full(int(effective.shape[0]), producer_theta)
        else:
            arrays, input_step, input_offset = _channel_arrays(
                cores, laws, int(effective.shape[0]), ir_graph, operand=id(perceptron) in adds,
                column_channels=core_channels)
            theta = np.zeros(int(effective.shape[0]))
            for core in cores:
                theta[core_channels(core)] = float(core.threshold)
        shaped = theta.reshape((-1,) + (1,) * (effective.ndim - 1))
        p_min, p_max = quantization_bounds(layer_weight_bits(perceptron, weight_bits))  # [WI-5g] its type's grid
        grid = np.where(shaped > 0, np.clip(np.round(effective * shaped), p_min, p_max), 0.0)
        bias = perceptron.layer.bias
        stamp = ReadoutStamp(
            owner=(mapper, perceptron), weight_name=parameter_name(mapper, perceptron.layer.weight),
            bias_name=None if bias is None else parameter_name(mapper, bias), weight_grid=grid,
            input_step=input_step, law=arrays, channel_axis=int(perceptron.output_channel_axis),
        )
        stamp.lane_input_offset = input_offset  # pyright: ignore[reportAttributeAccessIssue] -- declared-only
        if form is not None:
            stamp.lane_reduce = (form.kernel, form.stride, form.padding)  # pyright: ignore[reportAttributeAccessIssue]
        quantizer.stamp(stamp)
        perceptron.append_input_wire_op(quantizer.input_tap())
        stamped += 1
    return stamped


def _pool_lanes(ir_graph, plan: LanePlan, first_owner: int):
    """Per pool lane: its owner, its reduction cores, its channel count and each core's per-column channels."""
    for k, lane in enumerate(plan.pool_lanes):
        owner = first_owner + k
        cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore) and getattr(n, LANE_OWNER, None) == owner]
        shapes = lane.pool.input_shapes
        first = shapes[0] if shapes else None
        channels = int(first[0]) if first else 0
        per_channel = max(1, sum(n.get_output_count() for n in cores) // max(channels, 1))

        def column_channels(core, per_channel=per_channel):
            return np.asarray(getattr(core, LANE_OUTPUTS)) // per_channel

        yield owner, lane, cores, channels, column_channels


def lane_channel_keys(ir_graph, plan: LanePlan, fusion: LaneFusion, first_owner: int):
    """Per core, per column, the channel the stamps read one law from: a pool lane's channel, a reduction's pool
    output (its law is per output), a perceptron's output channel (``law_stamp.channel_keys``)."""
    pools = {int(c.id): (owner, fn) for owner, _lane, cores, _n, fn in _pool_lanes(ir_graph, plan, first_owner)
             for c in cores}

    def keys(core: NeuralCore) -> Tuple[Hashable, ...]:
        pool = pools.get(int(core.id))
        if pool is not None:
            return tuple(("pool", pool[0], int(c)) for c in pool[1](core))
        if core.perceptron_index is not None and int(core.perceptron_index) in fusion.reduce_forms:
            return tuple(("reduced", int(core.perceptron_index), int(o)) for o in getattr(core, LANE_OUTPUTS))
        return channel_keys(core)

    return keys


def _stamp_pools(ir_graph, laws, plan: LanePlan, fusion: LaneFusion, first_owner: int) -> int:
    for owner, lane, cores, channels, column_channels in _pool_lanes(ir_graph, plan, first_owner):
        arrays, input_step, input_offset = _channel_arrays(cores, laws, channels, ir_graph, operand=False,
                                                           column_channels=column_channels)
        form = fusion.pool_forms[owner]
        module = lane.pool.module
        if not isinstance(module, LanePoolModule):
            raise TypeError(f"pool lane {lane.pool.name!r} carries no lane end (Readout Law installs it)")
        module.stamp = PoolStamp(
            input_step=input_step, input_offset=input_offset, kernel=form.kernel, stride=form.stride,
            padding=form.padding, law={**arrays, OFFSET_MULTIPLIER: arrays.get(OFFSET_MULTIPLIER,
                                                                                np.zeros(channels, dtype=np.int64))},
        )
    return len(plan.pool_lanes)


def stamp_lane_laws(model, ir_graph, laws: Mapping[int, LaneLaw], plan: LanePlan, fusion: LaneFusion, *,
                    first_owner: int, weight_bits: int) -> int:
    """Stamp every perceptron's channel law and every pool lane's; the number of stamps."""
    return _stamp_perceptrons(model, ir_graph, laws, plan, weight_bits, fusion) + _stamp_pools(
        ir_graph, laws, plan, fusion, first_owner)
