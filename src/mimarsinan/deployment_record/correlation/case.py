"""A reference case: one published chip measurement and the census that produced it.

The case is the unit of silicon correlation. It carries the published number WITH its
quote, the workload census that the target's own paper determines, and a declaration of
whether the measurement is independent of the constants being tested — because a chip
reproducing the operating point its constants were divided out of proves only
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from mimarsinan.deployment_record.quantities.spec import quantity_spec
from mimarsinan.deployment_record.schema.serde import strict_kwargs

#: independent = the published value did not produce the constants being tested;
#: self_consistency = it did, so the case checks arithmetic, not generalization.
INDEPENDENCE_KINDS = ("independent", "self_consistency")


@dataclass(frozen=True)
class PublishedValue:
    """One number a paper prints, with the sentence it prints it in."""

    value: float
    citation: str
    quote: str

    def __post_init__(self) -> None:
        for field in ("citation", "quote"):
            if not getattr(self, field):
                raise ValueError(
                    f"PublishedValue.{field} must be stated: a published number "
                    f"without its {field} cannot be checked by a reader"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "citation": self.citation, "quote": self.quote}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PublishedValue":
        return cls(**strict_kwargs(cls, data))


@dataclass(frozen=True)
class ReferenceCase:
    """A published operating point, priceable by a declared profile."""

    name: str
    profile: str
    citation: str
    description: str
    independence: str
    operating_conditions: Mapping[str, Any]
    census: Mapping[str, float]
    census_sources: Mapping[str, str]
    overrides: Mapping[str, float]
    published: Mapping[str, PublishedValue]
    tolerance_pct: float

    def __post_init__(self) -> None:
        if self.independence not in INDEPENDENCE_KINDS:
            raise ValueError(
                f"{self.name}: independence must be one of {INDEPENDENCE_KINDS}, "
                f"got {self.independence!r}"
            )
        for key in self.census:
            quantity_spec(key)
        missing = sorted(set(self.census) - set(self.census_sources))
        if missing:
            raise ValueError(
                f"{self.name}: census_sources is missing {missing}; every census "
                f"number must say where it came from, or it is a fitted parameter"
            )
        if not self.published:
            raise ValueError(
                f"{self.name}: a case must publish at least one measured value"
            )
        if self.tolerance_pct <= 0:
            raise ValueError(
                f"{self.name}: tolerance_pct must be positive, got {self.tolerance_pct}"
            )

    @property
    def is_independent(self) -> bool:
        return self.independence == "independent"

    @property
    def axes(self) -> Tuple[str, ...]:
        return tuple(sorted(self.published))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "profile": self.profile,
            "citation": self.citation,
            "description": self.description,
            "independence": self.independence,
            "operating_conditions": dict(self.operating_conditions),
            "census": dict(self.census),
            "census_sources": dict(self.census_sources),
            "overrides": dict(self.overrides),
            "published": {k: v.to_dict() for k, v in self.published.items()},
            "tolerance_pct": self.tolerance_pct,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReferenceCase":
        kwargs = strict_kwargs(cls, data)
        kwargs["published"] = {
            key: value if isinstance(value, PublishedValue)
            else PublishedValue.from_dict(value)
            for key, value in kwargs["published"].items()
        }
        kwargs["census"] = {k: float(v) for k, v in kwargs["census"].items()}
        kwargs["overrides"] = {k: float(v) for k, v in kwargs["overrides"].items()}
        return cls(**kwargs)
