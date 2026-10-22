"""Declarations read as their consumers read them: the one reading both doors apply before the judging loop.

A key whose consumer normalizes what it reads declares the consumer's own function as its
``consumer_reading`` (registry ``types.py``). Both doors read every declaration through it first, so a
spelling the consumer normalizes resolves to the value the consumer runs, and the one judging loop judges
that value: the document door through ``parse_deployment_document`` and ``validate_deployment_config``, the
run door through ``build_platform_constraints_resolved``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.config_schema.registry.build import REGISTRY

#: The schema defaults a run merges beneath every document: a reading sees them under the declarations.
SCHEMA_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {key: entry.default for key, entry in REGISTRY.items() if entry.has_default()})

#: The keys whose consumer normalizes what it reads, with the consumer's reading.
_READINGS: Tuple[Tuple[str, Any], ...] = tuple(
    (key, entry.consumer_reading) for key, entry in REGISTRY.items() if entry.consumer_reading is not None)


def read_like_consumers(declared: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of ``declared`` in which every present value of a key whose consumer normalizes it is the
    value the consumer reads (its ``consumer_reading`` over the schema defaults and the declarations).

    A reading replaces a value and never adds or drops a key. A value the consumer refuses (its reading
    raises ``ArithmeticError``, ``TypeError`` or ``ValueError``, ``int(inf)`` included) stays as written, so
    the one judging loop refuses it by name.
    """
    view = {**SCHEMA_DEFAULTS, **declared}
    read = dict(declared)
    for key, reading in _READINGS:
        if declared.get(key) is None:
            continue
        try:
            read[key] = reading(view)
        except (ArithmeticError, TypeError, ValueError):
            continue
    return read
