"""Declarations read as their consumers read them: the one reading both doors apply before the judging loop.

A key whose consumer normalizes what it reads declares the consumer's own function as its
``consumer_reading`` (registry ``types.py``). Both doors read every declaration through it first, so a
spelling the consumer normalizes resolves to the value the consumer runs, and the one judging loop judges
that value: the document door through ``parse_deployment_document`` and ``validate_deployment_config``, the
run door through ``build_platform_constraints_resolved``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Set, Tuple

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.config_schema.registry.build import REGISTRY
from mimarsinan.config_schema.registry.value_schema import schema_of

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


class ReadRecorder(Mapping[str, Any]):
    """A read-only view of a state that records every key a judgment reads through it (a legal set, a relevance,
    a consumer's reading), so a judge knows which declarations the judgment rests on: one that reads a declaration
    refused by its own schema is undetermined, and that refusal is the message."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self._state = state
        self.read: Set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return self._state[key]

    def __contains__(self, key: object) -> bool:
        self.read.add(str(key))
        return key in self._state

    def __iter__(self) -> Iterator[str]:
        self.read.update(self._state)
        return iter(self._state)

    def __len__(self) -> int:
        return len(self._state)


def consumer_refusal(declared: Mapping[str, Any], refusal: BaseException) -> str:
    """The message refusing declarations no value schema judges, by the consumer that read them: each declared
    key with its value, then the consumer's own reason (a KeyError's text unquoted)."""
    reason = refusal.args[0] if isinstance(refusal, KeyError) and refusal.args else refusal
    named = ", ".join(f"{key}={value!r}" for key, value in declared.items()) or "a declaration"
    return f"{named}: refused by its consumer: {reason}"


def workload_reading_errors(config: Mapping[str, Any], declared: Mapping[str, Any]) -> List[str]:
    """The refusal of the workload reading (``ResolvedWorkloadProfile.from_config``, which the plan and the tuners
    read) of a config: a value it refuses names each of the ``declared`` keys it read that no value schema judges,
    with its reason (``consumer_refusal``)."""
    reads = ReadRecorder(config)
    try:
        ResolvedWorkloadProfile.from_config(reads)
    except (TypeError, ValueError, KeyError, AttributeError) as refusal:
        return [consumer_refusal({key: declared[key] for key in sorted(reads.read) if key in declared
                                  and key in REGISTRY and schema_of(REGISTRY[key]) is None}, refusal)]
    return []
