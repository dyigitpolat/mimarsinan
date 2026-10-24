"""Declared lane grids (WI-5e): readout offsets on perceptron lanes and the ranges of pool lanes, from the map."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from mimarsinan.mapping.support.quantizer_state import (
    DECLARED_OFFSET_ATTRIBUTE,
    DECLARED_STATE_ATTRIBUTE,
    QuantizerState,
    QuantizerStateError,
    perceptron_fx_names,
)


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
