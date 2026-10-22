#!/usr/bin/env python3
"""Print the declaration table: one row per registry key and nested field, sorted by decision class."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

from mimarsinan.config_schema.registry import REGISTRY, Category
from mimarsinan.config_schema.registry.decision_classes import DECISION_CLASSES
from mimarsinan.config_schema.registry.value_schema import ValueSchema, describe, schema_of

_COLUMNS = ("key", "section", "group", "legal_values", "absent_means", "decision_class", "reason")
_FROZEN = "consumer frozen default"


def _text(value: Any) -> str:
    """A value as the table prints it: in full, never cut (a string verbatim, anything else as JSON)."""
    return value if isinstance(value, str) else json.dumps(value)


def _key_absent(entry: Any) -> str:
    """What an absent key means, from the registry alone: the stated meaning, the schema default, why no
    value can be declared, the value the consumer falls back to, else the wizard's own phrase."""
    if entry.empty_means:
        return entry.empty_means
    if entry.has_default():
        return f"default {_text(entry.default)}"
    if entry.category is Category.RUNTIME:
        return "runtime fact; not declarable"
    if entry.category is Category.DERIVED:
        source = "derived from " + ", ".join(entry.derived_from)
        return source if entry.declarable else f"{source}; not declarable"
    fallback = None if entry.derived_default is None else entry.derived_default({})
    if fallback is not None:
        label = "frozen default" if entry.provenance == _FROZEN else f"{entry.provenance} default"
        return f"{label} {_text(fallback)}"
    return f"unset — {entry.owner} decides at resolve"


def _field_absent(parent: ValueSchema, name: str, schema: ValueSchema) -> str:
    if name in parent.required:
        return "required"
    for rule in parent.required_iff:
        if rule.field == name:
            return f"required iff {rule.when} is {' or '.join(map(str, rule.values))}"
    return schema.absent_means or "—"


def _walk(schema: ValueSchema, prefix: str) -> Iterator[Tuple[str, ValueSchema, str, ValueSchema]]:
    """``(path, declaring object, field name, field schema)`` of every nested field."""
    while schema.item is not None:
        schema = schema.item
    if schema.entries is not None:
        yield from _walk(schema.entries, f"{prefix}.*")
        return
    for name, sub in schema.fields.items():
        yield f"{prefix}.{name}", schema, name, sub
        yield from _walk(sub, f"{prefix}.{name}")


def rows() -> List[Dict[str, str]]:
    """Every registry key and nested field with its legal values, absent meaning and class."""
    table: List[Dict[str, str]] = []
    for key, entry in REGISTRY.items():
        schema = schema_of(entry)
        table.append({
            "key": key, "section": entry.section, "group": entry.group,
            "legal_values": describe(schema) if schema is not None else entry.type.value.upper(),
            "absent_means": _key_absent(entry),
            "decision_class": str(entry.decision_class),
            "reason": str(entry.decision_reason),
        })
        for path, parent, name, sub in _walk(entry.value_schema, key) if entry.value_schema else ():
            table.append({
                "key": path, "section": entry.section, "group": entry.group,
                "legal_values": describe(sub),
                "absent_means": _field_absent(parent, name, sub),
                "decision_class": str(sub.decision_class),
                "reason": sub.doc,
            })
    table.sort(key=lambda row: (DECISION_CLASSES.index(row["decision_class"]), row["key"]))
    return table


def _markdown(table: List[Dict[str, str]]) -> str:
    lines = ["| " + " | ".join(_COLUMNS) + " |", "|" + "---|" * len(_COLUMNS)]
    for row in table:
        cells = [row[c].replace("|", "\\|") for c in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the rows as JSON")
    args = parser.parse_args(argv)
    table = rows()
    print(json.dumps(table, indent=1) if args.json else _markdown(table))
    return 0


if __name__ == "__main__":
    sys.exit(main())
