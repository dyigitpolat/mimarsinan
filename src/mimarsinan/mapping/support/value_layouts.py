"""[U-PCH3] The layout of every per-core and per-node value, declared by name by its holder, and the one reading of it.

A holder class (an IR node, a weight bank, a hard core) declares, per value name, what the value's entries index
(``VALUE_LAYOUTS``):
- ``PER_NEURON``: the holder's own neuron columns (a threshold register per neuron, a hardware bias);
- ``PER_CHANNEL``: the output channels of the holder's perceptron, every node of the perceptron holding the whole vector
  (a per-channel activation scale); a holder with no perceptron identity indexes its own columns;
- ``PER_INPUT_ROW``: the holder's input rows (an entry currency per input);
- ``WHOLE``: nothing (one value or object for the holder: a value a hard core stores once and its residents share).

A value holds one entry (every member of its index reads it) or exactly one entry per member of its declared index. Any
other size is malformed and refused by name (``ValueLayoutError``), and so is a name its holder does not declare. An
object that carries its own columns (a readout law, ``PER_NEURON``) is read at the declared columns by its own
``take_columns``, which holds no column it lacks (an index error, never a shorter read). Every reader reads a per-core
or per-node value through its holder's declaration, never by the value's size: ``placement_value`` on a placement's
columns of a hard core, ``neuron_entries`` on an IR node's neurons, ``row_entries`` on its input rows."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal

__all__ = ["HELD", "LAYOUTS_ATTR", "ValueLayout", "ValueLayoutError", "declared_layout", "declared_names",
           "neuron_entries", "one_entry", "placement_value", "row_entries"]

#: The class attribute a holder declares its values' layouts by.
LAYOUTS_ATTR = "VALUE_LAYOUTS"


class ValueLayout(Enum):
    """What a per-core or per-node value's entries index."""

    PER_NEURON = "per output neuron"
    PER_CHANNEL = "per output channel of its perceptron"
    PER_INPUT_ROW = "per input row"
    WHOLE = "whole"


class ValueLayoutError(ToolchainRefusal, ValueError):
    """A per-core or per-node value read outside its holder's declared layout (a name the holder does not declare, or
    a size that is neither one entry nor one per member of its declared index), refused by name."""


class _Held:
    """Marker: read the value the holder holds under the name (a reader may pass a value its holder derives)."""


HELD = _Held()
Channels = Union[None, np.ndarray, Callable[[], Optional[np.ndarray]]]


def declared_layout(holder: Any, name: str) -> ValueLayout:
    """The layout ``holder``'s class declares for its value ``name``; an undeclared name is refused by name."""
    layout = getattr(type(holder), LAYOUTS_ATTR, {}).get(name)
    if layout is None:
        raise ValueLayoutError(
            f"{type(holder).__name__} declares no layout for its value {name!r} ({LAYOUTS_ATTR}): a reader reads a "
            f"per-core or per-node value only by its holder's declaration")
    return layout


def declared_names(holder_type: type, layout: ValueLayout) -> tuple:
    """The names ``holder_type`` declares in ``layout``, in declaration order."""
    return tuple(name for name, held in getattr(holder_type, LAYOUTS_ATTR, {}).items() if held is layout)


def _flat(value: Any) -> np.ndarray:
    return np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=np.float64).reshape(-1)


def one_entry(value: Any) -> bool:
    """Whether ``value`` holds one entry (every member of its declared index reads it)."""
    return int(value.numel() if hasattr(value, "numel") else np.size(value)) == 1


def _refuse(holder: Any, name: str, layout: ValueLayout, size: int, want: str) -> ValueLayoutError:
    label = getattr(holder, "name", None) or getattr(holder, "id", "?")
    return ValueLayoutError(
        f"{type(holder).__name__} {label!r} holds {name!r} {layout.value} with {size} entries, neither one nor "
        f"{want}: a malformed value, never read in another layout")


def placement_value(holder: Any, name: str, lo: int, hi: Optional[int], value: Any = HELD) -> Any:
    """Value ``name`` of ``holder`` (a hard core) as a placement on its neuron columns ``lo:hi`` holds it: a
    ``PER_NEURON`` value at those columns (an object by its own ``take_columns``, a readout law; one held entry as it
    is held, byte for byte), any other value whole (the one value the core stores, which its residents share)."""
    held = getattr(holder, name, None) if value is HELD else value
    if declared_layout(holder, name) is not ValueLayout.PER_NEURON or held is None:
        return held
    if hasattr(held, "take_columns"):
        return held.take_columns(np.arange(int(lo), int(holder.get_output_count()) if hi is None else int(hi)))
    if one_entry(held):
        return held
    entries = np.asarray(held.detach().cpu() if hasattr(held, "detach") else held).reshape(-1)  # its own dtype
    count = int(holder.get_output_count())
    if entries.size != count:
        raise _refuse(holder, name, ValueLayout.PER_NEURON, entries.size, f"one per neuron ({count})")
    return entries[int(lo):None if hi is None else int(hi)].copy()


def neuron_entries(holder: Any, name: str, columns: np.ndarray, *, width: int, channels: Channels = None,
                   value: Any = HELD) -> Any:
    """Value ``name`` of ``holder`` (an IR node of ``width`` columns) as its neurons ``columns`` read it, by the
    declared layout: one held entry broadcast; ``PER_NEURON`` at the columns; ``PER_CHANNEL`` at the columns' channels
    (``channels``: each column's perceptron channel as the reader derives it, or a function giving it on first need;
    None for a holder with no perceptron identity, whose columns are its channels); ``WHOLE`` as it is. A
    ``PER_INPUT_ROW`` value has no entry per neuron (``row_entries`` reads it). A ``PER_NEURON`` object reads its own
    columns (``take_columns``, a readout law)."""
    held = getattr(holder, name, None) if value is HELD else value
    layout = declared_layout(holder, name)
    if held is None or layout is ValueLayout.WHOLE:
        return held
    if layout is ValueLayout.PER_NEURON and hasattr(held, "take_columns"):
        return held.take_columns(np.asarray(columns, dtype=np.int64))
    if layout is ValueLayout.PER_INPUT_ROW:
        raise ValueLayoutError(f"{type(holder).__name__}'s {name!r} is {layout.value}: its neurons read it by the "
                               f"rows they read (row_entries), never per neuron")
    flat = _flat(held)
    columns = np.asarray(columns, dtype=np.int64)
    if flat.size == 1:
        return np.full(len(columns), float(flat[0]))
    table = channels() if callable(channels) else channels
    if layout is ValueLayout.PER_NEURON or table is None:
        if flat.size != int(width):
            raise _refuse(holder, name, layout, flat.size, f"one per column ({int(width)})")
        return flat[columns]
    index = np.asarray(table, dtype=np.int64)[columns]
    if len(index) and int(index.max()) >= flat.size:
        raise _refuse(holder, name, layout, flat.size, f"one per channel up to channel {int(index.max())}")
    return flat[index]


def row_entries(holder: Any, name: str, rows: np.ndarray, *, value: Any = HELD) -> Optional[np.ndarray]:
    """Value ``name`` of ``holder`` (a ``PER_INPUT_ROW`` value) at its input rows ``rows``: one held entry broadcast,
    else one entry per input row of the holder (``get_input_count``)."""
    held = getattr(holder, name, None) if value is HELD else value
    layout = declared_layout(holder, name)
    if layout is not ValueLayout.PER_INPUT_ROW:
        raise ValueLayoutError(f"{type(holder).__name__}'s {name!r} is {layout.value}, not per input row")
    if held is None:
        return None
    flat = _flat(held)
    rows = np.asarray(rows, dtype=np.int64)
    if flat.size == 1:
        return np.full(len(rows), float(flat[0]))
    count = int(holder.get_input_count())
    if flat.size != count:
        raise _refuse(holder, name, layout, flat.size, f"one per input row ({count})")
    return flat[rows]


def layouts(**groups: tuple) -> Mapping[str, ValueLayout]:
    """A holder's declaration from names grouped by layout member (``PER_NEURON=(...)``, ``WHOLE=(...)``, ...)."""
    return {name: ValueLayout[member] for member, names in groups.items() for name in names}
