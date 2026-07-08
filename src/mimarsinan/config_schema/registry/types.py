"""ConfigKeySchema: the typed per-key record of the configurability SSOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

from mimarsinan.config_schema.registry.relevance import Relevance


class FieldType(str, Enum):
    """Widget/value type of one config key (drives schema-driven rendering)."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    STR = "str"
    PATH = "path"
    RECIPE = "recipe"
    CORES = "cores"
    JSON = "json"
    INT_LIST = "int_list"
    SHAPE = "shape"


class Category(str, Enum):
    """Wizard altitude: BASIC renders by default, ADVANCED behind a drawer,
    DERIVED as read-only chips, RUNTIME only in run views."""

    BASIC = "basic"
    ADVANCED = "advanced"
    DERIVED = "derived"
    RUNTIME = "runtime"


class _NoDefault:
    """Sentinel: the key has no schema default (absence is meaningful)."""

    def __repr__(self) -> str:  # pragma: no cover - repr only
        return "NO_DEFAULT"


NO_DEFAULT = _NoDefault()

SECTIONS = ("top", "deployment_parameters", "platform_constraints")

# SSOT-source vocabulary for derived-by-default values: WHERE the effective
# value comes from when the document is silent. Rendered generically by the
# wizard next to every green derived value.
PROVENANCE_SOURCES = frozenset({
    "TUNING_POLICY",            # frozen tuning-loop constant table
    "provider registration",    # DataWorkloadProfile (data-loader contract)
    "builder profile",          # ModelWorkloadProfile (model-builder contract)
    "ConversionPolicy recipe",  # the mode -> proven-recipe fold
    "derivation rule",          # config_schema/deployment_derivation.py
    "consumer frozen default",  # a named consumer's workload-neutral literal
})

OptionsSpec = Union[Tuple[str, ...], Callable[[], Tuple[str, ...]], None]


@dataclass(frozen=True)
class ConfigKeySchema:
    """One flat deployment/platform/top-level key: provenance, type, docs, UI.

    Absorbs the KeySpec provenance fields (group/owner/derivation/exposure)
    and the display metadata (label/doc/effect/type); defaults are injected
    from ``config_schema.defaults`` by the registry builder, never declared.
    """

    flat_key: str
    group: str
    owner: str
    type: FieldType
    category: Category
    label: str
    doc: str
    section: str = "deployment_parameters"
    derivation: str = "default"
    exposure: str = "system"
    default: Any = NO_DEFAULT
    effect: Optional[str] = None
    unit: Optional[str] = None
    options: OptionsSpec = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    relevant: Relevance = field(default_factory=Relevance.always)
    # Mode-aware prominence: an ADVANCED key renders as PRIMARY while this
    # predicate holds (a knob that is the point of the current mode is never
    # "advanced"). None = the declared category always applies.
    promote_when: Optional[Relevance] = None
    # What an empty/absent value means, shown verbatim next to the widget
    # (only for keys without a schema default; defaulted keys show the default).
    empty_means: Optional[str] = None
    # Ownership transfer: while this key is NOT relevant, its value is
    # produced by this other concern group — cards render an ownership chip
    # where the hand field would be instead of silently dropping the concern.
    provided_by: Optional[str] = None
    derived_from: Tuple[str, ...] = ()
    why: Optional[Callable[[dict], str]] = None
    # Machine-readable companion to ``why`` for DERIVED keys: extra structured
    # facts about the derivation (e.g. a backend enable's mode-capability
    # ``supported`` flag) served with the resolve payload's derived rows.
    meta: Optional[Callable[[dict], dict]] = None
    declarable: bool = True
    important: bool = False
    # SSOT source of the derived-by-default value (PROVENANCE_SOURCES); None
    # for keys whose default is a plain schema default.
    provenance: Optional[str] = None
    # Derivation-owned keys the UI must not render on ANY surface (no field,
    # no chip); they remain config data (declarable escape) where marked.
    hidden: bool = False

    def __post_init__(self) -> None:
        if self.section not in SECTIONS:
            raise ValueError(f"{self.flat_key!r}: unknown section {self.section!r}")
        if self.type is FieldType.ENUM and self.options is None:
            raise ValueError(f"{self.flat_key!r}: enum keys must declare options")
        if self.bounds is not None and self.type not in (FieldType.INT, FieldType.FLOAT):
            raise ValueError(f"{self.flat_key!r}: bounds require a numeric type")
        if (self.category is Category.DERIVED) != (self.derivation == "derived"):
            raise ValueError(f"{self.flat_key!r}: DERIVED category must pair with derivation='derived'")
        if (self.category is Category.RUNTIME) != (self.derivation == "runtime"):
            raise ValueError(f"{self.flat_key!r}: RUNTIME category must pair with derivation='runtime'")
        if self.category is Category.DERIVED and not self.derived_from:
            raise ValueError(f"{self.flat_key!r}: derived keys must name derived_from inputs")
        if self.promote_when is not None and self.category is not Category.ADVANCED:
            raise ValueError(f"{self.flat_key!r}: promote_when only applies to ADVANCED keys")
        if self.provided_by is not None and self.relevant.op == "always":
            raise ValueError(
                f"{self.flat_key!r}: provided_by requires a conditional relevance "
                "(an always-relevant key is never owned elsewhere)"
            )
        if self.provenance is not None and self.provenance not in PROVENANCE_SOURCES:
            raise ValueError(
                f"{self.flat_key!r}: unknown provenance {self.provenance!r} "
                f"(vocabulary: {sorted(PROVENANCE_SOURCES)})"
            )
        if self.hidden and self.category is not Category.DERIVED:
            raise ValueError(
                f"{self.flat_key!r}: hidden requires a derivation-owned "
                "(DERIVED) category — a knob must render"
            )

    def resolved_options(self) -> Optional[Tuple[str, ...]]:
        """Options with any registry-derived callable resolved to a tuple."""
        if callable(self.options):
            return tuple(self.options())
        return self.options

    def has_default(self) -> bool:
        return not isinstance(self.default, _NoDefault)
