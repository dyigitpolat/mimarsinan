"""Lanes in the model at Readout Law (WI-5e): lane grids on the readout quantizers, the lane ends installed, the
ranges measured at each lane's end."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from mimarsinan.mapping.lanes.declaration import lanes_executed
from mimarsinan.mapping.lanes.plan import LanePlan, plan_lanes
from mimarsinan.mapping.support.tensor_stats import safe_quantile
from mimarsinan.models.nn.activations.lane_quantization import LaneAddModule, LanePoolModule, LaneReduceModule
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer


def install_model_lanes(plan: LanePlan, activation_bits: int) -> List[Tuple[str, ReadoutQuantizer]]:
    """Each perceptron lane's readout quantizer takes its lane grid (unsigned where the lane clamps) and, on a fused
    add or a reduction, rounds at that op's end (its payload becomes a :class:`LaneAddModule` or a
    :class:`LaneReduceModule`); each pool lane gets its own quantizer behind a :class:`LanePoolModule`. Returns
    ``[(pool FX name, quantizer)]`` of the pool lanes in plan order."""
    for lane in plan.perceptron_lanes:
        quantizer = lane.holder.perceptron.scaler
        if not isinstance(quantizer, ReadoutQuantizer):
            raise TypeError(f"perceptron {lane.holder.perceptron.name!r} carries no readout quantizer for its lane")
        if lane.add is None and lane.clamp is None and lane.reduce is None:
            continue
        quantizer.set_lane(unsigned=lane.unsigned, continues=lane.add is not None or lane.reduce is not None)
        if lane.add is not None and not isinstance(lane.add.module, LaneAddModule):
            lane.add.module = LaneAddModule(lane.add.module, quantizer, lane.operand_index)
        if lane.reduce is not None and not isinstance(lane.reduce.module, LaneReduceModule):
            lane.reduce.module = LaneReduceModule(lane.reduce.module, quantizer)
    pools = []
    for lane in plan.pool_lanes:
        if not isinstance(lane.pool.module, LanePoolModule):
            quantizer = ReadoutQuantizer(activation_bits)
            quantizer.set_lane(unsigned=lane.unsigned)
            lane.pool.module = LanePoolModule(lane.pool.module, quantizer)
        pools.append((str(lane.pool.name), lane.pool.module.quantizer))
    return pools


def install_planned_lanes(model, config, activation_bits: int) -> Optional[List[Tuple[str, ReadoutQuantizer]]]:
    """The Readout Law hook: plan and install the declared lanes (``None`` when no lane runs, nothing changed)."""
    declaration = lanes_executed(config)
    if declaration is None:
        return None
    return install_model_lanes(plan_lanes(model.get_mapper_repr(), declaration), activation_bits)


def lane_absorbed(model, config) -> frozenset:
    """``id`` of every host-op mapper the declared lanes absorb (empty when no lane runs): Boundary Quantization's
    segment entries leave them out (a perceptron reading a lane reads a core)."""
    declaration = lanes_executed(config)
    return frozenset() if declaration is None else plan_lanes(model.get_mapper_repr(), declaration).absorbed()


def pool_lane_quantizers(model_repr) -> List[Tuple[str, ReadoutQuantizer]]:
    """``[(pool FX name, quantizer)]`` of every installed pool lane, in execution order."""
    order, _, _ = model_repr._ensure_exec_graph()
    return [(str(node.name), node.module.quantizer) for node in order
            if isinstance(getattr(node, "module", None), LanePoolModule)]


def _seam(quantizer: ReadoutQuantizer):
    """The module whose input is the lane's last value: the seam of a fused add or a reduction, else the
    quantizer."""
    return quantizer.lane_seam if quantizer.lane_continues else quantizer


def _lane_value(quantizer: ReadoutQuantizer, value: torch.Tensor) -> torch.Tensor:
    """The magnitude the lane's range covers: the clamp's output on an unsigned grid at code 0 = 0, else
    ``|value|``."""
    return torch.relu(value) if quantizer.lane_unsigned and quantizer.lane_offset == 0.0 else value.abs()


def measure_lane_ranges(model, quantizers: Sequence[ReadoutQuantizer], batches: Iterable[torch.Tensor],
                        quantile: float) -> List[float]:
    """The largest per-batch quantile of each lane's last value (``measure_seam_quantiles`` at the lane's end)."""
    maxima = {id(q): 0.0 for q in quantizers}
    handles = []
    for quantizer in quantizers:
        def hook(module, args, q=quantizer):
            value = safe_quantile(_lane_value(q, args[0].detach()), quantile).item()
            maxima[id(q)] = max(maxima[id(q)], value)
        handles.append(_seam(quantizer).register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return [maxima[id(q)] for q in quantizers]


def lane_clipping_census(model, named: Sequence[Tuple[str, ReadoutQuantizer]],
                         batches: Iterable[torch.Tensor]) -> Dict[str, dict]:
    """Per lane: its range and the fraction of its last values beyond it on the calibration batches."""
    totals = {id(q): [0, 0] for _, q in named}
    handles = []
    for _, quantizer in named:
        def hook(module, args, q=quantizer):
            value = _lane_value(q, args[0].detach())
            if float(q.range) > 0.0:
                totals[id(q)][0] += int((value > float(q.range)).sum())
            totals[id(q)][1] += int(value.numel())
        handles.append(_seam(quantizer).register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for x in batches:
                model(x)
    finally:
        for handle in handles:
            handle.remove()
    return {name: {"range": float(q.range), "clipped": totals[id(q)][0], "outputs": totals[id(q)][1],
                   "clipped_fraction": totals[id(q)][0] / max(totals[id(q)][1], 1)} for name, q in named}
