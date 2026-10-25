"""A small closed schema language for instance documents: typed fields, refused unknown fields, JSON Schema export.

A spec is a ``Scalar`` (str, int, number, bool, any), an ``Enum``, a ``Nullable``, a ``ListOf``, a ``MapOf`` (string
keys) or an ``Obj`` (required and optional fields, every other field refused). ``errors(document, spec)`` lists every
violation with its path; ``json_schema(spec)`` writes the same rules as JSON Schema (draft 2020-12), so a reader in
another language checks the document against the rules the toolchain checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

#: JSON Schema type of each scalar kind (``any`` has none).
_JSON_TYPES = {"str": "string", "int": "integer", "number": "number", "bool": "boolean"}


@dataclass(frozen=True)
class Scalar:
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in (*_JSON_TYPES, "any"):
            raise ValueError(f"unknown scalar kind {self.kind!r}")


@dataclass(frozen=True)
class Enum:
    options: Tuple[Any, ...]


@dataclass(frozen=True)
class Nullable:
    spec: "Spec"


@dataclass(frozen=True)
class ListOf:
    item: "Spec"
    length: Union[int, None] = None


@dataclass(frozen=True)
class MapOf:
    value: "Spec"


@dataclass(frozen=True)
class Obj:
    required: Mapping[str, "Spec"]
    optional: Mapping[str, "Spec"] = field(default_factory=dict)
    doc: str = ""


Spec = Union[Scalar, Enum, Nullable, ListOf, MapOf, Obj]

STR, INT, NUMBER, BOOL, ANY = Scalar("str"), Scalar("int"), Scalar("number"), Scalar("bool"), Scalar("any")
#: A priced band, corner-aligned: (low, nominal, high).
BAND = ListOf(NUMBER, length=3)


def _scalar_errors(value: Any, kind: str, path: str) -> List[str]:
    if kind == "any":
        return []
    if kind == "bool":
        ok = isinstance(value, bool)
    elif kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = isinstance(value, str)
    return [] if ok else [f"{path}: expected {kind}, got {type(value).__name__}"]


def errors(value: Any, spec: Spec, path: str = "$") -> List[str]:
    """Every violation of ``spec`` in ``value`` (empty: the document conforms)."""
    if isinstance(spec, Scalar):
        return _scalar_errors(value, spec.kind, path)
    if isinstance(spec, Enum):
        return [] if value in spec.options else [f"{path}: {value!r} is not one of {list(spec.options)}"]
    if isinstance(spec, Nullable):
        return [] if value is None else errors(value, spec.spec, path)
    if isinstance(spec, ListOf):
        if not isinstance(value, list):
            return [f"{path}: expected a list, got {type(value).__name__}"]
        out = [] if spec.length is None or len(value) == spec.length else [
            f"{path}: expected {spec.length} items, got {len(value)}"]
        for index, item in enumerate(value):
            out.extend(errors(item, spec.item, f"{path}[{index}]"))
        return out
    if isinstance(spec, MapOf):
        if not isinstance(value, dict):
            return [f"{path}: expected an object, got {type(value).__name__}"]
        out = [f"{path}: key {key!r} is not a string" for key in value if not isinstance(key, str)]
        for key, item in value.items():
            out.extend(errors(item, spec.value, f"{path}.{key}"))
        return out
    if not isinstance(value, dict):
        return [f"{path}: expected an object, got {type(value).__name__}"]
    out = [f"{path}: missing required field {name!r}" for name in spec.required if name not in value]
    known = {**spec.required, **spec.optional}
    for name, item in value.items():
        if name not in known:
            out.append(f"{path}: unknown field {name!r}")
            continue
        out.extend(errors(item, known[name], f"{path}.{name}"))
    return out


def json_schema(spec: Spec) -> Dict[str, Any]:
    """``spec`` as a JSON Schema fragment (closed objects: ``additionalProperties`` false)."""
    if isinstance(spec, Scalar):
        return {} if spec.kind == "any" else {"type": _JSON_TYPES[spec.kind]}
    if isinstance(spec, Enum):
        return {"enum": list(spec.options)}
    if isinstance(spec, Nullable):
        return {"anyOf": [{"type": "null"}, json_schema(spec.spec)]}
    if isinstance(spec, ListOf):
        out: Dict[str, Any] = {"type": "array", "items": json_schema(spec.item)}
        if spec.length is not None:
            out.update(minItems=spec.length, maxItems=spec.length)
        return out
    if isinstance(spec, MapOf):
        return {"type": "object", "additionalProperties": json_schema(spec.value)}
    properties = {name: json_schema(item) for name, item in {**spec.required, **spec.optional}.items()}
    out = {"type": "object", "properties": properties, "required": list(spec.required),
           "additionalProperties": False}
    if spec.doc:
        out["description"] = spec.doc
    return out


def field_paths(spec: Spec, prefix: str = "$") -> Sequence[str]:
    """Every field path the spec names (lists and maps read as ``[]`` and ``{}``), for docs and tests."""
    if isinstance(spec, Nullable):
        return field_paths(spec.spec, prefix)
    if isinstance(spec, ListOf):
        return field_paths(spec.item, f"{prefix}[]")
    if isinstance(spec, MapOf):
        return field_paths(spec.value, f"{prefix}{{}}")
    if not isinstance(spec, Obj):
        return [prefix]
    out: List[str] = []
    for name, item in {**spec.required, **spec.optional}.items():
        out.extend(field_paths(item, f"{prefix}.{name}"))
    return out
