"""One value schema for every declared value (scalars, JSON keys, core-type fields) and the one loop that judges it."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

#: Kinds are the registry ``FieldType`` values (a str enum); ``ANY`` admits every value.
ANY = "any"
_UNIVERSAL = "universal"
_INT, _FLOAT, _BOOL, _ENUM = "int", "float", "bool", "enum"
_STRING_KINDS = frozenset({"str", "enum", "path"})
SCALAR_KINDS = frozenset({_INT, _FLOAT, _BOOL}) | _STRING_KINDS


class DeclarationValueError(ToolchainRefusal, ValueError):
    """A declared value outside its registry schema, or a key of the other core-semantics domain."""


@dataclass(frozen=True)
class RequiredIff:
    """``field`` is required exactly when ``when`` holds one of ``values`` (and refused otherwise)."""

    field: str
    when: str
    values: Tuple[Any, ...]


@dataclass(frozen=True)
class ValueSchema:
    """What one declared value may be: a scalar, an object of fields, a map of entries, or a list of objects."""

    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    low_exclusive: bool = False
    high_exclusive: bool = False
    #: The legal set of a closed key: what its consumer accepts and runs.
    options: Optional[Tuple[Any, ...]] = None
    #: Values the consumer names but no run implements: no legal value, refused by name as reserved.
    reserved: Tuple[Any, ...] = ()
    #: The minimum length of a string (characters) or a list (items).
    min_length: int = 0
    fields: Mapping[str, "ValueSchema"] = field(default_factory=dict)
    required: Tuple[str, ...] = ()
    required_iff: Tuple[RequiredIff, ...] = ()
    #: A map: every entry value is judged against this schema (keys are free).
    entries: Optional["ValueSchema"] = None
    #: A list: every item is judged against this schema.
    item: Optional["ValueSchema"] = None
    #: A closed object refuses fields it does not declare; an open one leaves them to the parser.
    closed: bool = True
    #: The decision class of a nested field (``decision_classes.py`` vocabulary).
    decision_class: Optional[str] = None
    #: What the field changes (the declaration table's reason) and what its absence means.
    doc: str = ""
    absent_means: str = ""

    def __post_init__(self) -> None:
        # A FieldType member is a str enum; hold its plain value so messages read the same on every Python.
        object.__setattr__(self, "kind", str(self.kind.value if isinstance(self.kind, Enum) else self.kind))
        if self.reserved and (self.options is None or set(self.reserved) & set(self.options)):
            raise ValueError("reserved values are named beside a closed legal set, never inside it")


def _type_matches(kind: str, value: Any) -> bool:
    """Python conflates bool with int; the registry declares them distinct, so bool is not an int."""
    if kind == _BOOL:
        return isinstance(value, bool)
    if kind == _INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == _FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, str)


def _bound_error(schema: ValueSchema, value: Any, path: str) -> Optional[str]:
    low, high = schema.low, schema.high
    if low is not None and (value <= low if schema.low_exclusive else value < low):
        if schema.low_exclusive:
            return f"{path}={value!r} must be above {low} (exclusive bound)"
        return f"{path}={value!r} is below the minimum {low}"
    if high is not None and (value >= high if schema.high_exclusive else value > high):
        if schema.high_exclusive:
            return f"{path}={value!r} must be below {high} (exclusive bound)"
        return f"{path}={value!r} is above the maximum {high}"
    return None


def _scalar_errors(schema: ValueSchema, value: Any, path: str) -> List[str]:
    if not _type_matches(schema.kind, value):
        return [f"{path} must be {schema.kind}, got {type(value).__name__} ({value!r})"]
    if schema.options is not None and value in schema.reserved:
        legal = ", ".join(repr(option) for option in schema.options)
        verb = "is" if len(schema.options) == 1 else "are"
        return [f"{path}={value!r} is reserved/not implemented; only {legal} {verb} supported"]
    if schema.options is not None:
        if value not in schema.options:
            return [f"{path}={value!r} is not a valid option; choose one of {list(schema.options)}"]
        return []
    if schema.kind in _STRING_KINDS:
        if len(value) < schema.min_length:
            return [f"{path} must be a string of at least {schema.min_length} character(s)"]
        return []
    if schema.kind == _BOOL:
        return []
    if schema.kind == _FLOAT and isinstance(value, float) and not math.isfinite(value):
        # Every comparison with NaN is False, so a bound alone would admit it. An int is always
        # finite and compares exactly with a bound, even beyond float range (where isfinite raises).
        return [f"{path}={value!r} must be a finite number"]
    error = _bound_error(schema, value, path)
    return [] if error is None else [error]


def _presence_errors(schema: ValueSchema, value: Mapping[str, Any], path: str) -> List[str]:
    present = {name for name, item in value.items() if item is not None}
    errors = [f"{path}.{name} is required" for name in schema.required if name not in present]
    for rule in schema.required_iff:
        holds = value.get(rule.when) in rule.values
        if holds and rule.field not in present:
            errors.append(f"{path}.{rule.field} is required when {path}.{rule.when} is "
                          f"one of {list(rule.values)}")
        elif not holds and rule.field in present:
            errors.append(f"{path}.{rule.field} is declared only when {path}.{rule.when} is "
                          f"one of {list(rule.values)}")
    return errors


def _object_errors(schema: ValueSchema, value: Any, path: str) -> List[str]:
    if not isinstance(value, Mapping):
        return [f"{path} must be an object, got {type(value).__name__} ({value!r})"]
    errors: List[str] = []
    for name in value:
        if name not in schema.fields and schema.closed:
            errors.append(f"{path}.{name} is not a declared field; known: {sorted(schema.fields)}")
    errors.extend(_presence_errors(schema, value, path))
    for name, sub in schema.fields.items():
        if value.get(name) is not None:
            errors.extend(value_errors(sub, value[name], f"{path}.{name}"))
    return errors


def value_errors(schema: ValueSchema, value: Any, path: str) -> List[str]:
    """Every way ``value`` violates ``schema``, each message starting with its dotted path."""
    if schema.kind == ANY:
        return []
    if schema.kind in SCALAR_KINDS:
        return _scalar_errors(schema, value, path)
    if schema.item is not None:
        if not isinstance(value, list):
            return [f"{path} must be a list, got {type(value).__name__} ({value!r})"]
        if len(value) < schema.min_length:
            return [f"{path} must be a list of at least {schema.min_length} item(s)"]
        return [e for i, item in enumerate(value)
                for e in value_errors(schema.item, item, f"{path}[{i}]")]
    if schema.entries is not None:
        if not isinstance(value, Mapping):
            return [f"{path} must be an object, got {type(value).__name__} ({value!r})"]
        return [e for name, item in value.items()
                for e in value_errors(schema.entries, item, f"{path}.{name}")]
    return _object_errors(schema, value, path)


def nested_schemas(schema: ValueSchema, prefix: str) -> Iterator[Tuple[str, ValueSchema]]:
    """``(dotted path, schema)`` of every nested field, depth first; a map entry's fields sit under ``*``
    and a list item's fields under the list's own path."""
    if schema.item is not None:
        yield from nested_schemas(schema.item, prefix)
    if schema.entries is not None:
        yield from nested_schemas(schema.entries, f"{prefix}.*")
    for name, sub in schema.fields.items():
        yield f"{prefix}.{name}", sub
        yield from nested_schemas(sub, f"{prefix}.{name}")


def schema_of(entry: Any) -> Optional[ValueSchema]:
    """The entry's declared value schema, else the scalar schema its type, bounds and options imply."""
    if entry.value_schema is not None:
        return entry.value_schema
    if entry.type not in SCALAR_KINDS:
        return None
    low, high = entry.bounds if entry.bounds is not None else (None, None)
    low_exclusive, high_exclusive = entry.exclusive_bounds
    return ValueSchema(
        kind=entry.type, low=low, high=high,
        low_exclusive=low_exclusive, high_exclusive=high_exclusive,
        options=entry.resolved_options() if entry.type == _ENUM else None,
    )


def enum_legal_set(legal: Iterable[Any], reserved: Iterable[Any] = ()) -> Dict[str, Any]:
    """The registry attributes of a closed key whose legal set is its consumer's: the options the key
    names (the legal values, then the reserved ones) and the value schema both doors judge it by."""
    legal, reserved = tuple(legal), tuple(reserved)
    return {"options": legal + reserved,
            "value_schema": ValueSchema(kind=_ENUM, options=legal, reserved=reserved)}


def entry_value_errors(entry: Any, value: Any) -> List[str]:
    """``value_errors`` of one registry entry's declared value, keyed by its flat key."""
    schema = schema_of(entry)
    return [] if schema is None else value_errors(schema, value, entry.flat_key)


def domain_errors(entry: Any, *, run_domain: str) -> List[str]:
    """The entry's registry domain tag judged against the run's domain (``value`` or ``event``)."""
    if entry.domain in (_UNIVERSAL, run_domain):
        return []
    return [f"{entry.flat_key} is {entry.domain}-domain configuration and is not declarable "
            f"in this {run_domain}-domain deployment. Remove the key or change core_semantics."]


def judge_declaration(entry: Any, value: Any, *, run_domain: Optional[str]) -> Any:
    """``value`` unchanged when its schema (and, given a run domain, its domain tag) admits it, else a
    refusal naming every violation."""
    errors = [] if run_domain is None else domain_errors(entry, run_domain=run_domain)
    errors += entry_value_errors(entry, value)
    if errors:
        raise DeclarationValueError("; ".join(errors))
    return value


def _number(x: float) -> str:
    return f"{x:g}"


def _bound_text(schema: ValueSchema) -> str:
    kind, fmt = schema.kind.upper(), _number
    low, high = schema.low, schema.high
    if low is not None and high is not None:
        left, right = "(" if schema.low_exclusive else "[", ")" if schema.high_exclusive else "]"
        return f"{kind} in {left}{fmt(low)}, {fmt(high)}{right}"
    if low is not None:
        return f"{kind} {'>' if schema.low_exclusive else '>='} {fmt(low)}"
    if high is not None:
        return f"{kind} {'<' if schema.high_exclusive else '<='} {fmt(high)}"
    return kind


def describe(schema: ValueSchema) -> str:
    """The legal values of ``schema`` as one phrase (the declaration table's cell)."""
    if schema.options is not None:
        text = f"{schema.kind.upper()} " + " | ".join(str(o) for o in schema.options)
        if schema.reserved:
            text += " (reserved: " + ", ".join(str(o) for o in schema.reserved) + ")"
        return text
    if schema.kind in SCALAR_KINDS:
        text = _bound_text(schema)
        return text + (" (non-empty)" if schema.min_length else "")
    if schema.kind == ANY:
        return "any value"
    if schema.entries is not None:
        return "JSON {key: " + describe(schema.entries) + "}"
    if schema.item is not None:
        return "list of " + describe(schema.item) + (" (non-empty)" if schema.min_length else "")
    return "JSON {" + ", ".join(schema.fields) + "}"
