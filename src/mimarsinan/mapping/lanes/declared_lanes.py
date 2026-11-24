"""Declared lane grids (WI-5e): readout offsets on perceptron lanes and the ranges of pool lanes, from the map; and
the refusal of a declared readout range on the holder of an add that stays on the host."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from mimarsinan.mapping.lanes.declaration import lanes_executed
from mimarsinan.mapping.lanes.plan import plan_lanes
from mimarsinan.mapping.support.quantizer_state import (
    DECLARED_OFFSET_ATTRIBUTE,
    DECLARED_STATE_ATTRIBUTE,
    QuantizerState,
    QuantizerStateError,
    perceptron_fx_names,
)


class HostSumRangeError(QuantizerStateError):
    """A declared readout range on a perceptron whose only reader is an add that the declared class E would put on
    chip but that stays on the host: the range would round the branch before the add, never the sum."""


def refuse_host_sum_ranges(model, platform: Mapping[str, Any], state: QuantizerState) -> None:
    """Under executed lanes, refuse by name a declared readout range on the holder of an add the lane plan leaves on
    the host (``LanePlan.host_adds``: its operand a host value, or its holder a segment entry). Such a holder's lane
    ends at its own output, so the range would round the branch and the host would add the other input exactly, where
    an on-chip addition rounds the sum once on that range. Without a declared range nothing changes."""
    declaration = lanes_executed(platform)
    if declaration is None or not state.readout_ranges:
        return
    repr_ = model.get_mapper_repr()
    names = {id(p): name for name, p in perceptron_fx_names(repr_).items()}
    for host in plan_lanes(repr_, declaration).host_adds:
        name = names.get(id(host.holder.perceptron))
        if name in state.readout_ranges:
            raise HostSumRangeError(
                f"quantizer_state declares the readout range of {name!r}, whose only reader is the add "
                f"{host.add.name!r}, which stays on the host ({host.reason}): the range would round the branch "
                f"{name!r} and the host would add the other input exactly, never rounding the sum; an add rounds its "
                f"sum once only on chip (its operand and its holder on chip), else declare no readout range for "
                f"{name!r}")


def _offset(quantizer, name: str, offset: float, sha256: str) -> None:
    if quantizer.lane_unsigned:
        raise QuantizerStateError(f"quantizer_state declares a readout_offset for {name!r}, whose lane clamps at 0: "
                                  f"an offset grid is a signed lane's")
    quantizer.set_lane(unsigned=True, offset=float(offset), continues=quantizer.lane_continues)
    setattr(quantizer, DECLARED_OFFSET_ATTRIBUTE, sha256)


def declare_lane_state(model, perceptrons: list, quantizers: list, pools: Sequence[Tuple[str, Any]],
                       state: QuantizerState) -> Dict[str, int]:
    """After the declared readout ranges: each declared readout offset on its perceptron's lane quantizer, each
    declared pool lane range (and offset) on its lane's quantizer; counts for the step report."""
    names = perceptron_fx_names(model.get_mapper_repr())
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    for name, offset in state.readout_offsets.items():
        _offset(by_perceptron[id(names[name])], name, offset, state.sha256)
    by_name = dict(pools)
    unknown = sorted(set(state.lane_ranges) - set(by_name))
    if unknown:
        raise QuantizerStateError(f"quantizer_state declares lanes {unknown}, which are no pool lane's FX node name "
                                  f"(pool lanes: {sorted(by_name)})")
    for name, value in state.lane_ranges.items():
        quantizer = by_name[name]
        quantizer.calibrate(value)
        setattr(quantizer, DECLARED_STATE_ATTRIBUTE, state.sha256)
        if name in state.lane_offsets:
            _offset(quantizer, name, state.lane_offsets[name], state.sha256)
    return {"declared_offsets": len(state.readout_offsets) + len(state.lane_offsets),
            "declared_lane_ranges": len(state.lane_ranges)}
