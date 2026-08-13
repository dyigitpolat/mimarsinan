"""Deployment options as search decision variables, described by the config SSOT.

A deployment option (where the encoder is placed, which schedule policy runs, how
wide a weight is, how much is pruned) is a first-class axis of the deployment
configuration space — it changes cost as much as the chip's geometry does. Promoting
one is therefore a DECLARATION, not new code: the config-key registry already states
each key's section, legal values, bounds and documentation, so an axis derives itself
from there and nothing here duplicates the configurability SSOT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from mimarsinan.config_schema.registry import REGISTRY, FieldType

#: Which sub-document a decoded option belongs in — the registry's own answer.
SECTION_PLATFORM = "platform_constraints"
SECTION_DEPLOYMENT = "deployment_parameters"

_NUMERIC_TYPES = (FieldType.INT, FieldType.FLOAT)

#: A declaration is either a bare list of keys (take the registry's whole range) or a
#: map key -> narrowing: a list of choices, or ``{"bounds": [lo, hi]}``.
OptionAxisDeclaration = Union[Sequence[str], Mapping[str, Any], None]


@dataclass(frozen=True)
class OptionAxis:
    """One deployment option promoted to a decision variable.

    Exactly one shape: ``choices`` (a discrete option, index-coded like the arch
    options) or ``bounds`` (a numeric range). Both would be two encodings of one
    axis, and neither would be an axis at all.
    """

    key: str
    section: str
    choices: Tuple[Any, ...]
    bounds: Optional[Tuple[float, float]]
    integral: bool
    label: str
    doc: str

    def __post_init__(self) -> None:
        if bool(self.choices) == (self.bounds is not None):
            raise ValueError(
                f"{self.key}: an option axis declares exactly one of choices or "
                f"bounds, got choices={self.choices!r} bounds={self.bounds!r}"
            )
        if self.section not in (SECTION_PLATFORM, SECTION_DEPLOYMENT):
            raise ValueError(f"{self.key}: unknown section {self.section!r}")

    @property
    def is_choice(self) -> bool:
        return bool(self.choices)

    @property
    def lower(self) -> float:
        """The encoded box's floor — a choice index, or the numeric bound."""
        return 0.0 if self.is_choice else float(self._bounds()[0])

    @property
    def upper(self) -> float:
        if self.is_choice:
            return float(len(self.choices) - 1)
        return float(self._bounds()[1])

    def _bounds(self) -> Tuple[float, float]:
        if self.bounds is None:
            raise ValueError(f"{self.key} is a choice axis and declares no bounds")
        return self.bounds


def decode_option_value(axis: OptionAxis, raw: float) -> Any:
    """The value this axis' encoded coordinate means — clipped, never wrapped."""
    if axis.is_choice:
        index = int(round(float(raw)))
        index = max(0, min(len(axis.choices) - 1, index))
        return axis.choices[index]
    low, high = axis._bounds()
    value = max(low, min(high, float(raw)))
    return int(round(value)) if axis.integral else value


def candidate_option(configuration: Mapping[str, Any], key: str, declared: Any) -> Any:
    """This candidate's value for a deployment option — searched, else declared.

    The one reader of ``deployment_options``, so a candidate that searched an option
    and a run that merely declared one are never told apart by hand at a call site.
    """
    options = configuration.get("deployment_options") or {}
    return options.get(key, declared)


def _entry(key: str):
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not a config key, so it cannot be a search axis"
        ) from None


def _registry_shape(key: str) -> Tuple[Tuple[Any, ...], Optional[Tuple[float, float]]]:
    """(choices, bounds) as the config registry declares them for ``key``."""
    entry = _entry(key)
    options = entry.resolved_options()
    if options:
        return tuple(options), None
    if entry.type in _NUMERIC_TYPES and entry.bounds is not None:
        low, high = entry.bounds
        if low is None or high is None:
            raise ValueError(
                f"{key}: the registry declares a half-open range {entry.bounds!r}; a "
                f"search axis needs both ends, so declare explicit bounds"
            )
        return (), (float(low), float(high))
    raise ValueError(
        f"{key}: the config registry describes neither choices nor bounds for it "
        f"(type {entry.type}), so it is not searchable — declare a key that is"
    )


def _narrow(
    key: str,
    narrowing: Any,
    choices: Tuple[Any, ...],
    bounds: Optional[Tuple[float, float]],
) -> Tuple[Tuple[Any, ...], Optional[Tuple[float, float]]]:
    """Apply a declaration's narrowing, refusing anything the registry disallows."""
    if narrowing is None or narrowing == {} or narrowing == []:
        return choices, bounds
    if isinstance(narrowing, Mapping):
        declared = narrowing.get("bounds")
        if declared is None:
            return choices, bounds
        if bounds is None:
            raise ValueError(f"{key}: bounds declared for a choice axis")
        low, high = float(declared[0]), float(declared[1])
        if low < bounds[0] or high > bounds[1] or low > high:
            raise ValueError(
                f"{key}: declared bounds {(low, high)} fall outside the registry's "
                f"bounds {bounds}"
            )
        return choices, (low, high)
    narrowed = tuple(narrowing)
    if not choices:
        raise ValueError(f"{key}: choices declared for a numeric axis")
    unknown = [value for value in narrowed if value not in choices]
    if unknown:
        raise ValueError(
            f"{key}: {unknown} are not legal values; the registry declares "
            f"{list(choices)}"
        )
    return narrowed, None


def build_option_axes(declaration: OptionAxisDeclaration) -> Tuple[OptionAxis, ...]:
    """The declared option axes, each described by the config registry.

    ``declaration`` is a list of keys (search each key's whole declared range) or a
    map key -> narrowing. Anything the registry cannot describe as a range fails
    loud here, at declaration time, rather than as a candidate that never varies.
    """
    if not declaration:
        return ()
    items = (
        list(declaration.items()) if isinstance(declaration, Mapping)
        else [(key, None) for key in declaration]
    )
    axes: list[OptionAxis] = []
    seen: set[str] = set()
    for key, narrowing in items:
        key = str(key)
        if key in seen:
            raise ValueError(f"option axis {key!r} is declared twice")
        seen.add(key)
        choices, bounds = _narrow(key, narrowing, *_registry_shape(key))
        entry = _entry(key)
        axes.append(OptionAxis(
            key=key,
            section=entry.section,
            choices=choices,
            bounds=bounds,
            integral=entry.type is FieldType.INT,
            label=entry.label,
            doc=entry.doc,
        ))
    return tuple(axes)


__all__ = [
    "SECTION_DEPLOYMENT",
    "SECTION_PLATFORM",
    "OptionAxis",
    "OptionAxisDeclaration",
    "build_option_axes",
    "candidate_option",
    "decode_option_value",
]
