"""[U-BRANGE2 (d)] The declared quantizer state's operand-bank and consumer-group sites (``quantizer_state`` loads the
map): an operand bank's host-fed entry grid and its readout range (N2), keyed by the bank perceptron's own name, and a
perceptron's consumer-group ranges, each group named by its readers. A grid a checkpoint was trained under is declared
at its site, never recalibrated: each declaration runs after its step's calibration and replaces the calibrated grid.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from mimarsinan.mapping.mappers.operand.consumer_groups import consumer_group_readers
from mimarsinan.mapping.mappers.operand.operand_mapper import operand_perceptrons
from mimarsinan.mapping.support.quantizer_state import (
    DECLARED_STATE_ATTRIBUTE,
    QuantizerState,
    QuantizerStateError,
    perceptron_fx_names,
)
from mimarsinan.models.nn.activations.value_quantizer import DeclaredBoundaryGrid, ValueGridQuantizer

__all__ = ["declare_bank_entry_grids", "declare_bank_readout_ranges", "declare_group_ranges"]


def _bank_perceptrons_by_name(model) -> Dict[str, Any]:
    """{name: perceptron} of the model's operand bank perceptrons (N2)."""
    return {str(p.name): p for p in operand_perceptrons(model.get_mapper_repr())}


def _known_banks(model, names: Iterable[str], what: str) -> Dict[str, Any]:
    banks = _bank_perceptrons_by_name(model)
    unknown = sorted(set(names) - set(banks))
    if unknown:
        raise QuantizerStateError(f"quantizer_state declares a {what} for {unknown}, which are no operand bank's name")
    return banks


def declare_bank_entry_grids(model, entries: list, quantizers: list, state: QuantizerState, *,
                             unsigned_dac: bool = False) -> int:
    """[U-BRANGE2 (d)] Install each declared bank entry grid (an operand bank's host-fed streamed operand, N2) on its
    bank's boundary quantizer, after its calibration: a trained grid declared, never recalibrated."""
    banks = _known_banks(model, state.bank_entry_grids, "bank entry")
    by_perceptron = {id(p): q for p, q in zip(entries, quantizers)}
    for name, entry in state.bank_entry_grids.items():
        quantizer = by_perceptron.get(id(banks[name]))
        if not isinstance(quantizer, ValueGridQuantizer):
            raise QuantizerStateError(f"quantizer_state declares a bank entry grid for {name!r}, whose streamed "
                                      f"operand is not host-fed (no entry grid)")
        low, high = entry.codes if entry.codes is not None else (None, None)
        quantizer.declare(DeclaredBoundaryGrid(
            scale=entry.scale, bits=quantizer.activation_bits, offset=entry.offset, rounding=entry.rounding,
            code_min=low, code_max=high, state_sha256=state.sha256, unsigned_dac=unsigned_dac,
        ))
    return len(state.bank_entry_grids)


def declare_bank_readout_ranges(model, perceptrons: list, quantizers: list, state: QuantizerState) -> int:
    """[U-BRANGE2 (d)] Set each declared bank readout range on its bank's readout quantizer, tagged with the map's
    digest."""
    banks = _known_banks(model, state.bank_readout_ranges, "bank readout range")
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    for name, value in state.bank_readout_ranges.items():
        quantizer = by_perceptron.get(id(banks[name]))
        if quantizer is None:
            raise QuantizerStateError(f"quantizer_state declares a readout range for bank {name!r}, which carries no "
                                      f"readout quantizer")
        quantizer.calibrate(value)
        setattr(quantizer, DECLARED_STATE_ATTRIBUTE, state.sha256)
    return len(state.bank_readout_ranges)


def declare_group_ranges(model, perceptrons: list, quantizers: list, state: QuantizerState) -> int:
    """[U-BRANGE2 (d)] Set each declared consumer-group range on its perceptron's grouped readout quantizer, the group
    found by its readers' names (``consumer_group_readers``); a group key no group of the perceptron carries is refused
    by name. Returns the groups declared."""

    if not state.group_ranges:
        return 0
    repr_ = model.get_mapper_repr()
    known = perceptron_fx_names(repr_)
    readers = consumer_group_readers(repr_)
    by_perceptron = {id(p): q for p, q in zip(perceptrons, quantizers)}
    declared = 0
    for name, groups in state.group_ranges.items():
        perceptron = known.get(name)
        quantizer = None if perceptron is None else by_perceptron.get(id(perceptron))
        if quantizer is None or getattr(quantizer, "channel_groups", None) is None:
            raise QuantizerStateError(f"quantizer_state declares consumer groups for {name!r}, which carries no "
                                      f"grouped readout quantizer")
        names = readers[id(perceptron)]
        ranges = [float(r) for r in quantizer.group_ranges.reshape(-1)]
        for key, value in groups.items():
            if key not in names:
                raise QuantizerStateError(f"quantizer_state declares a group of {name!r} read by {list(key)}, which no "
                                          f"group of it is")
            ranges[names.index(key)] = value
            declared += 1
        quantizer.calibrate_groups(ranges)
        setattr(quantizer, DECLARED_STATE_ATTRIBUTE, state.sha256)
    return declared

