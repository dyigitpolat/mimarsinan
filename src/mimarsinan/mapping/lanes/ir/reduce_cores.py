"""Reduced cores of a producer-side reduction (WI-5e, D11): an average pool its perceptron alone feeds, summed in
that perceptron's accumulator (arith_toggle.md section 3.4: ``W·Σ_p q_p = Σ_p W·q_p``, one rounding).

A pool output's column stacks, per present window instance, the producer column's rows and weights (the crossbar
runs every instance on the producer's own codes and the accumulator adds them), so every weight stays on the
producer's grid; ``theta`` is the window size times the producer's (the ``1/N`` of the average goes into ``M0``), and
the bias is the producer's once per present instance (zero padding of the pool adds nothing and counts in ``N``).
Outputs whose windows read the same producer cores share one reduced core.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import numpy as np

from mimarsinan.mapping.ir import ComputeOp, IRSource, NeuralCore
from mimarsinan.mapping.lanes.forms import PoolForm
from mimarsinan.mapping.lanes.ir.pool_cores import LANE_OUTPUTS, pool_windows
from mimarsinan.mapping.lanes.ir.refusal import LaneFusionError
from mimarsinan.mapping.support.neuron_thresholds import core_threshold

#: The attribute a reduced core carries: per column, the window instances it sums (a padded window has fewer).
REDUCED_INSTANCES = "lane_reduced_instances"
#: The attribute a reduced core carries: its producer's theta (the core's own is the window size times it).
PRODUCER_THETA = "lane_producer_theta"
_MASKS = ("pruned_row_mask", "pruned_col_mask", "pre_pruning_row_mask", "pre_pruning_col_mask")


def refuse_unreducible(core: NeuralCore, op: ComputeOp) -> None:
    """A producer core a reduction cannot stack: pruned (its masks carry no instance rows), a partial-sum or
    coalescing part (the accumulator is split), or a core another lane already rewrote."""
    masks = [name for name in _MASKS if getattr(core, name, None) is not None]
    if masks:
        raise LaneFusionError(f"core {core.name!r} under reduction {op.name!r} is pruned ({masks[0]}): a "
                              "producer-side reduction of a pruned core is refused")
    if core.psum_group_id is not None or core.coalescing_group_id is not None:
        raise LaneFusionError(f"core {core.name!r} under reduction {op.name!r} is a partial-sum or coalescing part: "
                              "a reduction sums whole accumulators")
    if core.perceptron_index is None:
        raise LaneFusionError(f"core {core.name!r} under reduction {op.name!r} belongs to no perceptron")


def reduced_lane_cores(op: ComputeOp, sources: np.ndarray, cores: Dict[int, NeuralCore], ir_graph, *,
                       form: PoolForm, first_id: int) -> Tuple[List[NeuralCore], np.ndarray]:
    """The reduced cores of pool ``op`` over its producer elements ``sources`` (core outputs), and the source of each
    pool output (flat, channel-major)."""
    shape = tuple(int(d) for d in (op.input_shape or ()))
    out_shape, windows = pool_windows(form, shape)  # pyright: ignore[reportArgumentType] -- rank 3, checked
    if op.output_shape is not None and tuple(int(d) for d in op.output_shape) != out_shape:
        raise LaneFusionError(f"reduction {op.name!r}: windows give {out_shape}, the op writes {op.output_shape}")
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for output, window in enumerate(windows):
        groups.setdefault(tuple(int(sources[k].node_id) for k in window), []).append(output)
    matrices = {core_id: np.asarray(cores[core_id].get_core_matrix(ir_graph)) for core_id in
                {c for key in groups for c in key}}
    built: List[NeuralCore] = []
    outputs = np.empty(len(windows), dtype=object)
    for g, (key, members) in enumerate(groups.items()):
        thetas = {core_threshold(cores[c], f"reduction {op.name!r}") for c in key}  # [U-LIVE r2] one register each
        if len(thetas) != 1:
            raise LaneFusionError(f"reduction {op.name!r} sums producer cores of thresholds {sorted(thetas)}: one "
                                  "accumulator, one theta")
        blocks, rows, bias = [], [], None
        for instance, core_id in enumerate(key):
            core = cores[core_id]
            columns = [int(sources[windows[m][instance]].index) for m in members]
            blocks.append(matrices[core_id][:, columns])
            rows.extend(core.input_sources.flatten().tolist())
            if core.hardware_bias is not None:
                part = np.asarray(core.hardware_bias, dtype=np.float64)[columns]
                bias = part if bias is None else bias + part
        first, theta = cores[key[0]], thetas.pop()
        reduced = dataclasses.replace(
            first, id=first_id + g, name=f"{op.name}_reduce{g}", input_sources=np.array(rows, dtype=object),
            core_matrix=np.concatenate(blocks, axis=0), weight_bank_id=None, weight_row_slice=None,
            threshold=float(form.divisor) * theta, hardware_bias=bias, latency=None,
            perceptron_output_slice=None, perceptron_output_column=None, perceptron_input_slice=None,
            layout_softcore_index=None,
        )
        setattr(reduced, LANE_OUTPUTS, np.asarray(members, dtype=np.int64))
        setattr(reduced, REDUCED_INSTANCES, np.full(len(members), len(key), dtype=np.int64))
        setattr(reduced, PRODUCER_THETA, theta)
        for column, member in enumerate(members):
            outputs[member] = IRSource(node_id=first_id + g, index=column)
        built.append(reduced)
    return built, outputs
