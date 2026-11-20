"""Optional blocks of a serialized dataclass: each block's field declares how it is read, so the class's ``to_dict``
and ``from_dict`` name no block (U-AREA's fix round, MA-4: the module budget holds by a split, never an allowlist).

A block is serialized only when present (a single block when it is not None, a tuple of blocks or values when it is
not empty), in the order of the class's fields. It is written by its own ``to_dict`` (a plain value as it is; a
tuple element by element, as a list) and read by the reader its field names (``many``: element by element into a
tuple; ``named``: the reader also takes the described object's name, for its refusal texts).
"""

from __future__ import annotations

from dataclasses import field, fields
from typing import Any, Callable, Dict, MutableMapping, NamedTuple

#: The field metadata key of an optional block.
OPTIONAL_BLOCK = "optional_block"


class BlockReader(NamedTuple):
    read: Callable[..., Any]
    many: bool = False
    named: bool = False


def optional_block(read: Callable[..., Any], *, many: bool = False, named: bool = False) -> Any:
    """A dataclass field of a block serialized only when present: absent, None (a single block) or () (``many``)."""
    return field(default=() if many else None, metadata={OPTIONAL_BLOCK: BlockReader(read, many, named)})


def _written(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    return value if to_dict is None else to_dict()


def blocks_to_dict(obj: Any) -> Dict[str, Any]:
    """The present optional blocks of ``obj`` by field name, in field order."""
    out: Dict[str, Any] = {}
    for f in fields(obj):
        reader, value = f.metadata.get(OPTIONAL_BLOCK), getattr(obj, f.name)
        if reader is None or (not value if reader.many else value is None):
            continue
        out[f.name] = [_written(v) for v in value] if reader.many else _written(value)
    return out


def blocks_from_dict(cls: type, kwargs: MutableMapping[str, Any], name: str) -> None:
    """Read every optional block of ``cls`` in ``kwargs`` in place (an absent block takes its absent value)."""
    for f in fields(cls):
        reader = f.metadata.get(OPTIONAL_BLOCK)
        if reader is None:
            continue
        raw = kwargs.get(f.name)
        if reader.many:
            kwargs[f.name] = tuple(reader.read(v) for v in raw or ())
        else:
            kwargs[f.name] = None if raw is None else reader.read(raw, name) if reader.named else reader.read(raw)
