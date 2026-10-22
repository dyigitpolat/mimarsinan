"""ConfigKeySchema: the typed per-key record of the configurability SSOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from mimarsinan.config_schema.registry.relevance import Relevance
from mimarsinan.config_schema.registry.value_schema import ValueSchema


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

# Which core-semantics DOMAIN a key belongs to. EVENT keys exist only under
# core_semantics='spiking' (spike physics, temporal grids); VALUE keys only
# under 'mvm' (value-grid boundaries); UNIVERSAL keys under both. The domain
# drives validation (domain_rules), wizard existence (relevance injection),
# and emission stripping from ONE tag.
DOMAINS = ("universal", "event", "value")

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

# The SSOT deriver of a key's value while the document is silent, evaluated
# against the current config state (so a mode-aware fallback renders its
# concrete value). Consumers read it through ``effective_value``.
DerivedDefaultFn = Callable[[Mapping[str, Any]], Any]
# The key's LEGAL VALUE SET for the current config state. |legal| == 1 LOCKS the
# field; |legal| > 1 offers exactly those options; anything else is a keyed error.
# ``None`` means "legality does not apply in this state" (the builder was not
# consulted): the field is neither locked nor judged — it renders as an ordinary
# widget. The distinction from an EMPTY set (a consulted registration that admits
# nothing) is load-bearing.
LegalValuesFn = Callable[[Mapping[str, Any]], Optional[Tuple[Any, ...]]]

# Discrete-valued types whose legal set is a well-defined enumeration.
_LEGAL_VALUE_TYPES = frozenset({FieldType.ENUM, FieldType.STR, FieldType.BOOL})


def frozen_default(value: Any) -> DerivedDefaultFn:
    """A named consumer's frozen literal, declared once for both readers."""
    return lambda cfg: value


def registered_elsewhere(cfg: Mapping[str, Any]) -> None:
    """A workload-profile registration (provider/builder) injects this value, or
    a consumer computes it from state the registry cannot see: the wizard reads
    the registered facts and renders '—' when nothing registers it."""
    return None


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
    # The value that source produces for a given config state (the CONCRETE
    # green in-field text, and the number the consumer reads).
    derived_default: Optional[DerivedDefaultFn] = None
    # Legality that depends on other config (see LegalValuesFn).
    legal_values: Optional[LegalValuesFn] = None
    # Derivation-owned keys the UI must not render on ANY surface (no field,
    # no chip); they remain config data (declarable escape) where marked.
    hidden: bool = False
    # Core-semantics domain membership (see DOMAINS above).
    domain: str = "universal"
    # The value schema the one judging loop reads (registry/value_schema.py); None = the
    # scalar schema implied by type, bounds and options.
    value_schema: Optional[ValueSchema] = None
    # Per-end exclusivity of ``bounds`` (False = inclusive, the historical meaning).
    exclusive_bounds: Tuple[bool, bool] = (False, False)
    # The value schema a declaration also passes where ``applied_when`` holds: the state in which the consumer
    # ``applied_by`` names applies the value (the schema's ``doc`` names why); None = the key's own schema
    # everywhere. Both doors judge it through the one judge (registry/declaration_judge.py), and the target
    # expansion names a bundle key whose ``applied_when`` fails "<applied_by> not applied".
    applied_schema: Optional[ValueSchema] = None
    applied_when: Optional[Relevance] = None
    applied_by: Optional[str] = None
    # A top-level key the run also lays beneath this section's declarations, where its flat-config consumers read
    # it (the derivation, the plan): every reader reads the run's layout of it (config_schema/runtime.py).
    laid_beneath: Optional[str] = None
    # The resolver carries this platform key verbatim onto platform_constraints_resolved,
    # after judging its value and domain.
    passthrough: bool = False
    # The consumer's own reading of this key from a config (a reference to the consumer's function,
    # never a copy), declared where the consumer normalizes what it reads: both doors read a declared
    # value through it before the one judging loop (registry/consumer_reading.py).
    consumer_reading: Optional[DerivedDefaultFn] = None
    # Injected by the registry builder from decision_classes.py, never declared by an entry.
    decision_class: Optional[str] = None
    decision_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.section not in SECTIONS:
            raise ValueError(f"{self.flat_key!r}: unknown section {self.section!r}")
        if self.domain not in DOMAINS:
            raise ValueError(
                f"{self.flat_key!r}: unknown domain {self.domain!r} "
                f"(vocabulary: {list(DOMAINS)})"
            )
        if self.type is FieldType.ENUM and self.options is None:
            raise ValueError(f"{self.flat_key!r}: enum keys must declare options")
        if self.bounds is not None and self.type not in (FieldType.INT, FieldType.FLOAT):
            raise ValueError(f"{self.flat_key!r}: bounds require a numeric type")
        if any(self.exclusive_bounds) and self.bounds is None:
            raise ValueError(f"{self.flat_key!r}: exclusive_bounds require bounds")
        if self.value_schema is not None and self.value_schema.kind != self.type.value:
            raise ValueError(f"{self.flat_key!r}: a value_schema is of its key's own type")
        if self.value_schema is not None and self.value_schema.options is not None and (
                self.resolved_options() != self.value_schema.options + self.value_schema.reserved):
            raise ValueError(f"{self.flat_key!r}: the options name the value schema's legal, then "
                             "reserved, values")
        applied = (self.applied_schema, self.applied_when, self.applied_by)
        if any(part is not None for part in applied) and (
                self.applied_schema is None or self.applied_when is None or not self.applied_by
                or self.applied_schema.kind != self.type.value or self.applied_when.op == "always"
                or not self.applied_schema.doc):
            raise ValueError(f"{self.flat_key!r}: an applied_schema is of its key's own type, judged where a "
                             "conditional applied_when holds, by the consumer applied_by names, and names why in "
                             "its doc")
        if self.laid_beneath is not None and (self.section != "top" or self.laid_beneath not in SECTIONS[1:]):
            raise ValueError(f"{self.flat_key!r}: laid_beneath names the section a top-level key is laid beneath")
        if self.consumer_reading is not None and self.value_schema is None:
            raise ValueError(f"{self.flat_key!r}: a consumer reading needs the value schema both doors "
                             "judge the read value by")
        if self.passthrough and self.section != "platform_constraints":
            raise ValueError(f"{self.flat_key!r}: only a platform key rides the resolved surface")
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
        if self.category is Category.DERIVED and self.provenance is None:
            raise ValueError(
                f"{self.flat_key!r}: a DERIVED key must name its provenance "
                f"source (vocabulary: {sorted(PROVENANCE_SOURCES)})"
            )
        if self.derived_default is not None and self.provenance is None:
            raise ValueError(
                f"{self.flat_key!r}: a derived_default must name the provenance "
                "source that produces it"
            )
        if self.legal_values is not None and self.type not in _LEGAL_VALUE_TYPES:
            raise ValueError(
                f"{self.flat_key!r}: legal_values requires a discrete type "
                f"({sorted(t.value for t in _LEGAL_VALUE_TYPES)}) — the legal set "
                "is an enumeration of concrete values"
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
