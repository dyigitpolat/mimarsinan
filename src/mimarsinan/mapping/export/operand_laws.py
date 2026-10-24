"""N2's operand banks on the readout-law surface: output groups, payload steps, thresholds, floors and stamps."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Dict, Mapping, Optional, Set, cast

import numpy as np

from mimarsinan.mapping.export.law_grids import OutputGroups
from mimarsinan.mapping.export.law_stamp import readout_quantizers_by_perceptron
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.ir.dynamic_bank import OperandBindingError, bound_instances
from mimarsinan.mapping.mappers.operand import OperandContractionMapper, OperandStamp, operand_mappers
from mimarsinan.mapping.mappers.operand.consumer_groups import bank_reads, channels_of
from mimarsinan.mapping.support.operand_form import B_STATIONARY
from mimarsinan.models.nn.activations.integer_law import grid_levels
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer


def output_groups_of(model_repr) -> Dict[int, OutputGroups]:
    """{perceptron index: its consumer groups} of every grouped readout quantizer."""
    out: Dict[int, OutputGroups] = {}
    for index, (_mapper, _perceptron, quantizer) in readout_quantizers_by_perceptron(model_repr).items():
        if quantizer.channel_groups is not None:
            out[index] = OutputGroups(channels=np.asarray(quantizer.channel_groups),
                                      ranges=tuple(float(r) for r in quantizer.group_ranges),
                                      levels=tuple(quantizer.group_levels))
    return out


def operand_ranges(model_repr) -> Dict[int, float]:
    """{bank perceptron index: its calibrated output range}."""
    return {int(op.perceptron_index): float(op.scaler.range) for m in operand_mappers(model_repr)
            for op in m.banks if isinstance(op.scaler, ReadoutQuantizer) and op.perceptron_index is not None}


def operand_floors(model_repr) -> Dict[int, int]:
    """{bank perceptron index: 0} for every bank whose product absorbed a ReLU."""
    return {int(op.perceptron_index): 0 for m in operand_mappers(model_repr) if m.relu for op in m.banks
            if op.perceptron_index is not None}


def _step(steps: Mapping, key) -> float:
    return float(steps[key] if key in steps else steps[key[0]])


def _bank_of_outputs(producer: OperandContractionMapper, flat: np.ndarray) -> Set[int]:
    """The producer banks that compute the given flat per-sample output elements."""
    form = producer.lowering.form
    canonical = (*form.batch, form.m, form.n)
    out_shape = tuple(canonical[p] for p in form.out_perm)
    coords = np.unravel_index(np.asarray(flat, dtype=np.int64), out_shape)
    by_axis = {form.out_perm[i]: coords[i] for i in range(len(out_shape))}
    if not form.batch:
        return {0}
    batch = tuple(by_axis[a] for a in range(len(form.batch)))
    return {int(b) for b in np.unique(np.ravel_multi_index(batch, form.batch))}


def _levels(groups: Mapping[int, OutputGroups], key, activation_bits: int) -> int:
    found = groups.get(key[0])
    return int(found.levels[key[1]]) if found is not None else grid_levels(activation_bits)


def _payload_step(model_repr, mapper: OperandContractionMapper, bank: int, steps: Mapping,
                  groups: Mapping[int, OutputGroups], activation_bits: int) -> float:
    """The bank's payload step: its re-snap grid, or its producer's one output grid; either at ``payload_bits``
    (a producer grid of another width refuses by name, so the law's accumulator bound holds as declared)."""
    operand = mapper.banks[bank]
    width = grid_levels(operand.payload_bits)
    if operand.payload_quantizer is not None:
        grid = operand.payload_quantizer.grid
        if not grid.armed:
            raise OperandBindingError(f"{operand.name}: its payload grid was never calibrated")
        found_steps, levels = {float(grid.step)}, {int(grid.levels)}
    else:
        role = 1 if mapper.lowering.stationary == B_STATIONARY else 0
        root = mapper.get_source_mappers()[mapper.operand_roots[role]]
        flat = bank_reads(mapper, role)[bank]
        if isinstance(root, OperandContractionMapper):
            keys = {(int(cast(int, root.banks[b].perceptron_index)), 0) for b in _bank_of_outputs(root, flat)}
        else:
            index = int(cast(int, getattr(root, "perceptron_index")))
            channels = channels_of(getattr(root, "perceptron"), mapper.root_shapes[mapper.operand_roots[role]], flat)
            found = groups.get(index)
            keys = {(index, int(g)) for g in (np.unique(found.channels[channels]) if found is not None else [0])}
        found_steps = {_step(steps, key) for key in keys}
        levels = {_levels(groups, key, activation_bits) for key in keys}
    if len(found_steps) != 1:
        raise OperandBindingError(f"{operand.name}: its payload spans {len(found_steps)} producer grids; a payload "
                                  "written as codes lies on one grid")
    if levels != {width}:
        raise OperandBindingError(f"{operand.name}: its payload lies on a grid of {sorted(levels)} positive levels, "
                                  f"but its operand core type declares payload_bits {operand.payload_bits} ({width} "
                                  "levels): a payload cell holds the declared width only")
    return found_steps.pop()


def bind_payload_steps(ir_graph, model_repr, steps: Mapping, groups: Mapping[int, OutputGroups],
                       activation_bits: int) -> None:
    """Each bound bank's payload step, and its instances' threshold θ = 1 / (payload step x absorbed scale)."""
    instances = bound_instances(ir_graph)
    by_index = {int(op.perceptron_index): (m, j) for m in operand_mappers(model_repr)
                for j, op in enumerate(m.banks) if op.perceptron_index is not None}
    for bank, binding in list(ir_graph.dynamic_bank_bindings.items()):
        cores = instances[int(bank)]
        mapper, j = by_index[int(cores[0].perceptron_index)]
        step = _payload_step(model_repr, mapper, j, steps, groups, activation_bits)
        ir_graph.dynamic_bank_bindings[bank] = replace(binding, payload_step=step)
        for core in cores:
            core.threshold = 1.0 / (step * mapper.scale)


def stamp_operand_laws(model_repr, ir_graph, laws: Mapping[int, CoreLaw]) -> int:
    """Stamp every bank perceptron with its instances' one law, its streamed step and its payload step."""
    instances = bound_instances(ir_graph)
    by_index: Dict[int, int] = {}
    for bank, cores in instances.items():
        for core in cores:
            by_index[int(core.perceptron_index)] = bank
    stamped = 0
    for mapper in operand_mappers(model_repr):
        for operand in mapper.banks:
            bank: Optional[int] = by_index.get(int(operand.perceptron_index)) if operand.perceptron_index is not None \
                else None
            if bank is None:
                continue
            cores = instances[bank]
            law = laws[int(cores[0].id)]
            if any(laws[int(c.id)].digest() != law.digest() for c in cores):
                raise OperandBindingError(f"{operand.name}: its instances carry different laws")
            operand.stamp = OperandStamp(
                law={f.name: np.asarray(getattr(law, f.name)) for f in fields(CoreLaw)},
                input_step=float(law.input_step[0]),
                payload_step=float(ir_graph.dynamic_bank_bindings[bank].payload_step),
            )
            stamped += 1
    return stamped
