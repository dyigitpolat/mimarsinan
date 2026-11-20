"""[U-RSHIFT] The received shift's one writer: a reader's correction is its own linear response to the field it receives.

``received_field.received_shifts`` carries each source's field to its readers. ``correct_for_received_shift`` writes a
correction of the class into a reader's bias (a core or host perceptron, or a host bias carrier), in one of three forms
decided by exact equality of the field rows (tolerance 0):

* ``uniform``: every row is the source's own value on every axon (on every slot its patch reads, for a reader whose
  positions hold their own fold), which is today's bake: today's arithmetic, bytes and record;
* ``one``: every row equals one other row: one bias, that row recorded;
* ``positions``: the rows differ: each position its own bias through the fold unit's position form (``weight_charges``
  per position, ``add_bias_corrections``), the stamp held per position, the field recorded per element of the input.

Rows, not corrections, decide: a correction is linear in the field for every weight the reader will hold, so equal rows
stay equal under every later weight write. Sources compose: a reader records the sources it compensates (``SOURCES``)
and the sum of their fields (``input_shift``); a call whose source a reader already compensates writes nothing for it,
and a new source adds its response. A weight write re-derives the correction from the record (``weight``: the change).
The affine fold writes at the opposite sign and records nothing. ``bake_received_shift`` is what the graph-level sites
call; it prints its census only where some reader's field differs from today's bake.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.scale_propagation import assign_per_input_scales
from mimarsinan.mapping.support.received_field import ReceivedShift, ReceivedShiftRefusal, received_shifts
from mimarsinan.transformations.perceptron.input_shift import BAKED_FLAG, record_input_shift, recorded_input_shift
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.perceptron.position_bias import add_bias_corrections, weight_charges
from mimarsinan.transformations.perceptron.position_fold import (
    PER_POSITION, POSITION_BIAS, SLOT_SOURCE, bias_offsets, effective_weight_rows, holds_positions, shift_rows)

__all__ = ["EVERY_AXON", "SOURCES", "ReceivedShift", "ReceivedShiftRefusal", "bake_received_shift",
           "compensate_projected_weights", "correct_for_received_shift", "every_axon_shift", "received_for",
           "received_shifts", "recorded_received_shift", "recorded_sources"]

SOURCES = "received_shift_sources"
"""The names of the sources a reader's bias compensates (a plain attribute beside its record ``input_shift``)."""

EVERY_AXON = "every axon"
"""The source a direct call names: one shift baked over every axon (``apply_negative_shift_bias``)."""


def recorded_sources(reader: Any) -> tuple:
    return tuple(getattr(reader, SOURCES, ()) or ())


def _today(perceptron: Any, value: Any) -> Optional[torch.Tensor]:
    """Today's every-axon field of ``value`` as ``perceptron``'s positions read it: the value on every axon (on every
    slot its patch reads where its positions hold their own fold), float64; None where no axon order holds it."""
    v = torch.as_tensor(value).detach().to(torch.float32)
    if holds_positions(perceptron):
        return shift_rows(perceptron, v.to(perceptron.layer.weight.device)).squeeze(PER_POSITION.weight).double()
    flat, width = v.reshape(-1), int(perceptron.input_features)
    return flat.expand(width).reshape(1, width).double() if flat.numel() in (1, width) else None


def every_axon_shift(perceptron: Any, shift: Any, source: Optional[str] = EVERY_AXON) -> ReceivedShift:
    """A direct call's field: ``shift`` (one value, or one per input) on every axon of ``perceptron``, today's bake, as
    ``source`` (None: a correction no source names, written however often it is called)."""
    rows = _today(perceptron, shift)
    if rows is None:
        raise ReceivedShiftRefusal(f"{getattr(perceptron, 'name', None)!r}: a shift of "
                                   f"{torch.as_tensor(shift).numel()} values over {perceptron.input_features} inputs")
    return ReceivedShift(rows=rows, sources=() if source is None else (source,), uniform=torch.as_tensor(shift))


def _total(parts: list) -> torch.Tensor:
    total = parts[0]
    for part in parts[1:]:
        total = total + part
    return total


def received_for(node: Any, fields: dict, own: dict, names: Optional[tuple] = None) -> ReceivedShift:
    """The field ``node`` receives from ``names`` (of ``received_shifts``' entry): a perceptron's rows by its own patch
    table (``patch_gather``, a padding slot at zero) or its leading positions; a host carrier's rows per operand."""
    names = tuple(fields) if names is None else tuple(names)
    perceptron = getattr(node, "perceptron", None)
    if perceptron is None:
        parts = [fields[n] for n in names]
        operands = tuple(_total([p[i] for p in parts]).reshape(-1, parts[0][i].shape[-1]) for i in range(len(parts[0])))
        return ReceivedShift(rows=operands[0], sources=names, operands=operands)
    total = _total([fields[n] for n in names])
    elements, flat, slots = total[0], total.reshape(-1), None
    if hasattr(node, "patch_gather"):
        shape = tuple(int(d) for d in elements.shape)
        slots = (np.asarray(node.patch_gather(shape).slot_source_index, dtype=np.int64), int(flat.numel()))
        rows = torch.cat([flat, flat.new_zeros(1)])[torch.as_tensor(slots[0], device=flat.device)]
    else:
        rows = flat.reshape(-1, int(perceptron.input_features))
    uniform = None
    if len(names) == 1 and not recorded_sources(perceptron) and recorded_input_shift(perceptron) is None:
        today = _today(perceptron, own[names[0]])
        if today is not None and bool((rows == today.to(rows.device)).all()):
            uniform = own[names[0]]
    return ReceivedShift(rows=rows, elements=elements, sources=names, uniform=uniform, slots=slots)


def recorded_received_shift(perceptron: Any) -> Optional[ReceivedShift]:
    """The record a weight write re-derives its correction from (``input_shift``: one value, one per input, or one per
    element of the input), or None where the bias compensates no received shift."""
    record = recorded_input_shift(perceptron)
    if record is None:
        return None
    return ReceivedShift(rows=record, sources=recorded_sources(perceptron), uniform=record)


def _record_slots(perceptron: Any, record: torch.Tensor) -> torch.Tensor:
    """A record as each core position's slots read it, ``(positions, 1, in_features)``: a field per element (its
    per-sample shape, rank 2 or more) by the perceptron's own slot table (a linear's leading positions), else one value
    or one per input on the slots each patch reads (``shift_rows``)."""
    if record.dim() < 2:
        return shift_rows(perceptron, record)
    held = perceptron.__dict__.get(SLOT_SOURCE)
    flat = record.reshape(-1)
    if held is None:
        return flat.reshape(-1, int(perceptron.input_features)).unsqueeze(PER_POSITION.weight)
    table = torch.as_tensor(np.asarray(held[0]), device=flat.device)
    return torch.cat([flat, flat.new_zeros(1)])[table].unsqueeze(PER_POSITION.weight)


def _subtracts(sign: float):
    return (lambda b, c: b - c) if sign < 0 else (lambda b, c: b + c)


def _write_one(perceptron: Any, tr: Any, value: torch.Tensor, sign: float, weight: Any = None) -> None:
    """One bias: ``B <- B + sign * (W . value)`` with ``value`` one shift, one per input (today's arithmetic)."""
    w = tr.get_effective_weight(perceptron) if weight is None else weight
    step, c = _subtracts(sign), (w * value.to(dtype=w.dtype, device=w.device)).sum(dim=-1)
    tr.apply_effective_bias_transform(perceptron, lambda b, c=c: step(b, c))


def _write_rows(perceptron: Any, tr: Any, slots: torch.Tensor, sign: float, weights: Any = None) -> None:
    """Each position's bias: ``B_p <- B_p + sign * (W_p . slots_p)`` (``slots`` ``(positions, 1, in)``)."""
    if weights is None:
        rows, index = effective_weight_rows(perceptron)
        weights = rows[index]
    charges = weight_charges(weights, slots.to(dtype=weights.dtype, device=weights.device))
    add_bias_corrections(perceptron, -sign * charges, lambda f: tr.write_shared_bias(perceptron, f))


def _write_positions(perceptron: Any, tr: Any, received: ReceivedShift, sign: float) -> None:
    """The rows differ: each position's correction from its own weight; a perceptron that read one fold for every
    position holds it once per position (through the stamp's one writer), and its positions keep their own bias."""
    rows = received.rows
    slots = rows.unsqueeze(PER_POSITION.weight)
    if holds_positions(perceptron):
        _write_rows(perceptron, tr, slots, sign)
    else:
        scale = tr.input_scale(perceptron)
        w = tr.get_effective_weight(perceptron)
        _write_rows(perceptron, tr, slots, sign, w.unsqueeze(0).expand(int(rows.shape[0]), *w.shape))
        if received.slots is not None:
            perceptron.__dict__[SLOT_SOURCE] = received.slots
        if bias_offsets(perceptron) is None:  # every correction alike today: the positions keep their own bias
            zeros = w.new_zeros(int(rows.shape[0]), int(w.shape[0]))
            perceptron.register_buffer(POSITION_BIAS, zeros, persistent=False)
        vector = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        assign_per_input_scales(perceptron, vector.expand(int(perceptron.input_features)).clone())


def _record(perceptron: Any, received: ReceivedShift, form: str) -> None:
    """The record a later weight write re-derives from, and the sources it compensates: today's shift (``uniform``), the
    one row (``one``), or the field per element of the input; a second source adds to the record per element."""
    held, like = recorded_input_shift(perceptron), perceptron.layer.weight
    if held is None and form == "uniform":
        value = torch.as_tensor(received.uniform).to(dtype=like.dtype, device=like.device)
    elif held is None and form == "one":
        value = received.rows[0].to(dtype=like.dtype, device=like.device)
    else:
        if received.elements is None:
            raise ReceivedShiftRefusal(f"{getattr(perceptron, 'name', None)!r}: a second source over a record that "
                                       f"no field per element holds")
        value = received.elements.to(dtype=like.dtype, device=like.device)  # its per-sample shape: rank 2 or more
        if held is not None:
            prior = held.to(value)
            if prior.dim() == 0 or prior.shape == value.shape:
                value = value + prior
            elif prior.dim() == 1 and received.slots is None and int(value.shape[-1]) == prior.numel():
                value = value + prior  # one row over a linear's positions
            else:
                raise ReceivedShiftRefusal(f"{getattr(perceptron, 'name', None)!r}: a record of {prior.numel()} values "
                                           f"names no element of an input of shape {tuple(value.shape)}")
    setattr(perceptron, BAKED_FLAG, True)
    record_input_shift(perceptron, value)
    setattr(perceptron, SOURCES, recorded_sources(perceptron) + tuple(received.sources))


def _correct_host(module: Any, received: ReceivedShift, sign: float) -> str:
    """A host bias carrier: its one bias per operand projection takes ``sign * W . row`` (an attention module's query,
    key and value rows each from its own operand's field); positions that read different fields refuse by name, and
    today's every-axon shift keeps today's arithmetic (``sigma * W`` summed)."""
    held = recorded_sources(module)
    if received.sources and all(name in held for name in received.sources):
        return "compensated"
    operands = received.operands or (received.rows,)
    if len(operands) not in (1, 3) or not all(bool((rows == rows[:1]).all()) for rows in operands):
        raise ReceivedShiftRefusal(f"{type(module).__name__}: a host bias carrier whose positions or operands read "
                                   f"fields no one host bias holds")
    linear = isinstance(module, nn.Linear)
    weight, bias = (module.weight, module.bias) if linear else (module.in_proj_weight, module.in_proj_bias)
    with torch.no_grad():
        if received.uniform is not None:
            step = _subtracts(sign)
            bias.copy_(step(bias, float(torch.as_tensor(received.uniform).reshape(-1)[0]) * weight.sum(dim=1)))
        else:
            width = int(weight.shape[0]) // (1 if linear else 3)
            for part in range(1 if linear else 3):
                row = operands[part if len(operands) == 3 else 0][0].to(weight)
                block = slice(part * width, (part + 1) * width)
                bias[block] += sign * (weight[block] @ row)
    setattr(module, SOURCES, held + tuple(received.sources))
    return "host"


def compensate_projected_weights(perceptron: Any, transformer: Any, weight_before: torch.Tensor) -> None:
    """``B' <- B' - (W~_after - W~_before) s``: the bias keeps compensating its recorded field through the effective
    weights a projection just installed (each position's from its own projected weight row), at any rate."""
    received = recorded_received_shift(perceptron)
    if received is None:
        return
    if holds_positions(perceptron):
        rows, index = effective_weight_rows(perceptron)
        change = (rows - weight_before)[index]
    else:
        change = transformer.get_effective_weight(perceptron) - weight_before
    correct_for_received_shift(perceptron, received, weight=change, transformer=transformer)


def correct_for_received_shift(reader: Any, received: ReceivedShift, *, weight: Any = None, sign: float = -1.0,
                               record: bool = True, transformer: Any = None) -> str:
    """The one writer of the class: ``B_p <- B_p + sign * (W_p . rows_p)`` per core position of ``reader`` (a
    perceptron, or a host bias carrier, ``rows`` per operand). ``weight``: a weight write's change, whose response to the
    record the bias follows (no form decided, nothing recorded). Returns the form written, or ``compensated`` where the
    reader already compensates every source of ``received``."""
    tr = transformer if transformer is not None else PerceptronTransformer()
    if not hasattr(reader, "layer"):
        return _correct_host(reader, received, sign)
    if weight is not None:
        if holds_positions(reader):
            _write_rows(reader, tr, _record_slots(reader, received.rows.to(weight)), sign, weight)
        else:
            _write_one(reader, tr, received.rows, sign, weight)
        return "weight"
    held = recorded_sources(reader)
    if received.sources and all(name in held for name in received.sources):
        return "compensated"
    rows = received.rows
    form = ("uniform" if received.uniform is not None and not held
            else "one" if bool((rows == rows[:1]).all()) else "positions")
    if form == "uniform" and holds_positions(reader):
        _write_rows(reader, tr, shift_rows(reader, torch.as_tensor(received.uniform).to(reader.layer.weight)), sign)
    elif form == "uniform":
        _write_one(reader, tr, torch.as_tensor(received.uniform), sign)
    elif holds_positions(reader) or bias_offsets(reader) is not None or form == "positions":
        _write_positions(reader, tr, received, sign)
    else:
        _write_one(reader, tr, rows[0], sign)
    if record:
        _record(reader, received, form)
    return form


def bake_received_shift(model_repr: Any, sources: Optional[dict] = None, *, sign: float = -1.0, record: bool = True,
                        readers: Any = None) -> dict:
    """``received_shifts`` then ``correct_for_received_shift`` on every reader reached, each with the sources it does
    not yet compensate. Returns the census: the readers reached, each form's count, and the readers whose received
    field differs from today's bake (``differs``), printed only where there is one."""
    census: dict = {"readers": 0, "uniform": 0, "one": 0, "positions": 0, "host": 0, "compensated": 0, "differs": []}
    for node, fields, own in received_shifts(model_repr, sources, readers=readers).values():
        target = getattr(node, "perceptron", None)
        target = target if target is not None else node.module
        names = tuple(n for n in fields if n not in recorded_sources(target)) or tuple(fields)
        received = received_for(node, fields, own, names)
        form = correct_for_received_shift(target, received, sign=sign, record=record)
        census["readers"] += 1
        census[form] += 1
        if form in ("one", "positions") or (form == "host" and not all(
                bool((o == o.reshape(-1)[:1]).all()) for o in received.operands)):
            census["differs"].append(str(getattr(node, "name", None)))
    if census["differs"] and record:
        print(f"[RECEIVED-SHIFT] {len(census['differs'])} shifted reader(s) whose received field differs from the "
              f"uniform bake (one bias, or a bias per position): {census['differs']}", flush=True)
    return census
