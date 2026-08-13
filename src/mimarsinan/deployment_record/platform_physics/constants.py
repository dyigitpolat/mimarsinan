"""The physics-constant vocabulary: what a target MAY declare, declared once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

from mimarsinan.deployment_record.platform_physics import vocabulary
from mimarsinan.deployment_record.units import DIMENSIONS, unit_for

PHYSICS_GROUPS: Tuple[str, ...] = vocabulary.GROUPS


@dataclass(frozen=True)
class PhysicsConstantSpec:
    """One declarable constant: its group, dimension, expected unit and multiplicand."""

    key: str
    group: str
    dimension: str
    display_unit: str
    multiplicand: str
    doc: str

    def __post_init__(self) -> None:
        if self.group not in PHYSICS_GROUPS:
            raise ValueError(
                f"{self.key}: group {self.group!r} is not one of {list(PHYSICS_GROUPS)}"
            )
        if self.dimension not in DIMENSIONS:
            raise ValueError(
                f"{self.key}: dimension {self.dimension!r} is not one of {list(DIMENSIONS)}"
            )
        if unit_for(self.display_unit).dimension != self.dimension:
            raise ValueError(
                f"{self.key}: display unit {self.display_unit!r} is not a "
                f"{self.dimension} unit"
            )


def _build() -> Dict[str, PhysicsConstantSpec]:
    specs: Dict[str, PhysicsConstantSpec] = {}
    for key, group, dimension, unit, multiplicand, doc in vocabulary.ROWS:
        if key in specs:
            raise ValueError(f"physics constant {key!r} is declared twice")
        specs[key] = PhysicsConstantSpec(key, group, dimension, unit, multiplicand, doc)
    return specs


PHYSICS_CONSTANTS: Mapping[str, PhysicsConstantSpec] = _build()

#: Aggregate -> the decomposed constants it already contains (see vocabulary.SUPERSEDES).
SUPERSEDES: Mapping[str, Tuple[str, ...]] = vocabulary.SUPERSEDES


def superseded_by(key: str) -> Tuple[str, ...]:
    """Constants ``key`` already accounts for, which must not be priced beside it."""
    return SUPERSEDES.get(key, ())


def resolve_supersessions(keys: Iterable[str]) -> Tuple[str, ...]:
    """Which of ``keys`` a pricer may charge, once aggregates absorb what they contain.

    A profile may legitimately declare an aggregate AND its decomposition — both are
    published facts. Charging both double counts, so the aggregate wins and the terms it
    already contains drop out. Declaration order never matters.
    """
    declared = set(keys)
    absorbed = {
        member
        for aggregate in declared & set(SUPERSEDES)
        for member in superseded_by(aggregate)
    }
    return tuple(key for key in sorted(declared) if key not in absorbed)


def spec_for(key: str) -> PhysicsConstantSpec:
    """The declaration for ``key``, or a loud error naming the whole vocabulary."""
    try:
        return PHYSICS_CONSTANTS[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a declarable physics constant; the vocabulary is "
            f"{sorted(PHYSICS_CONSTANTS)}"
        ) from None


def keys_in_group(group: str) -> Tuple[str, ...]:
    """Vocabulary order within one group — also the wizard panel's row order."""
    if group not in PHYSICS_GROUPS:
        raise KeyError(f"{group!r} is not a physics group; expected {list(PHYSICS_GROUPS)}")
    return tuple(key for key, spec in PHYSICS_CONSTANTS.items() if spec.group == group)
