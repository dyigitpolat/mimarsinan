"""The pretrained-weight-set registration contract and its pure regime predicates."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Tuple

WEIGHT_SETS_KEY = "pretrained_weight_sets"
WEIGHT_SET_KEY = "pretrained_weight_set"


@dataclass(frozen=True)
class PretrainedWeightSet:
    """ONE pretrained artifact a MODEL BUILDER registers.

    A builder does not have *a* pretrained source: it has a set of weight sets,
    each trained on some dataset for some task at some input geometry and class
    count. Builders declare; the framework never enumerates workloads.
    ``adapts_*`` states what the builder can project onto a different workload,
    and ``model_config_requires`` is its declarative applicability predicate
    over the architecture config.
    """

    id: str
    label: str
    task: str
    dataset: str
    input_shape: Tuple[int, ...]
    num_classes: int
    source: str
    expected_accuracy: Optional[float] = None
    license: Optional[str] = None
    num_parameters: Optional[int] = None
    recipe: Optional[str] = None
    preprocessing: Optional[Mapping[str, Any]] = None
    adapts_input_shape: bool = False
    adapts_num_classes: bool = False
    model_config_requires: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.id or not self.label:
            raise ValueError("a pretrained weight set must declare an id and a label")
        if not self.source:
            raise ValueError(f"{self.id!r}: a weight set must declare a source")
        if not self.input_shape:
            raise ValueError(f"{self.id!r}: a weight set must declare its input_shape")
        if int(self.num_classes) < 1:
            raise ValueError(f"{self.id!r}: num_classes must be >= 1")

    def as_dict(self) -> Dict[str, Any]:
        """The JSON-safe record: the config injection AND the panel's facts."""
        record: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "input_shape":
                value = [int(dim) for dim in value]
            elif isinstance(value, Mapping):
                value = dict(value)
            record[f.name] = value
        return record


# The regime, as PURE functions of the injected records. ``None`` everywhere
# means "no builder was consulted in this config view" (a raw document); it is
# never the same as an EMPTY registration ("this builder has no pretrained
# weights"), which is what disables the regime.


def registered_weight_sets(cfg: Mapping[str, Any]) -> Optional[Tuple[Dict[str, Any], ...]]:
    """The builder's declared weight sets as injected; ``None`` = not consulted."""
    raw = cfg.get(WEIGHT_SETS_KEY)
    if raw is None:
        return None
    return tuple(dict(record) for record in raw)


def weight_set_mismatch(record: Mapping[str, Any], cfg: Mapping[str, Any]) -> Optional[str]:
    """Why this weight set cannot be used here, or ``None`` when it can.

    Geometry and class count are judged only against facts the config carries:
    the data provider writes them at run time, so a raw draft never filters on
    them. A set the builder can adapt is never a mismatch — the torchvision
    builders project conv weights and rebuild the head by construction.
    """
    for key, value in (record.get("model_config_requires") or {}).items():
        declared = (cfg.get("model_config") or {}).get(key)
        if declared != value:
            return (
                f"needs model_config {key}={value!r} "
                f"(this configuration declares {declared!r})"
            )

    shape = cfg.get("input_shape")
    if shape is not None and not record.get("adapts_input_shape"):
        provider_shape = tuple(int(dim) for dim in shape)
        native = tuple(int(dim) for dim in record["input_shape"])
        if provider_shape != native:
            return (
                f"expects input {native} and the builder declares no input "
                f"adaptation (the data provider serves {provider_shape})"
            )

    classes = cfg.get("num_classes")
    if classes is not None and not record.get("adapts_num_classes"):
        if int(classes) != int(record["num_classes"]):
            return (
                f"expects {int(record['num_classes'])} classes and the builder "
                f"declares no head adaptation (the data provider serves {int(classes)})"
            )
    return None


def applicable_weight_sets(cfg: Mapping[str, Any]) -> Optional[Tuple[Dict[str, Any], ...]]:
    """The registered sets that apply to this config; ``None`` = not consulted."""
    sets = registered_weight_sets(cfg)
    if sets is None:
        return None
    return tuple(r for r in sets if weight_set_mismatch(r, cfg) is None)


def legal_weight_set_ids(cfg: Mapping[str, Any]) -> Optional[Tuple[str, ...]]:
    """THE legal value set of ``pretrained_weight_set`` for this config state."""
    sets = applicable_weight_sets(cfg)
    return None if sets is None else tuple(str(r["id"]) for r in sets)


def legal_preload_values(
    cfg: Mapping[str, Any], *, source_declared: bool = False
) -> Optional[Tuple[bool, ...]]:
    """THE legal value set of ``preload_weights``.

    ``source_declared`` (the resolver's config-data escape: an explicit
    ``weight_source``) locks the flag ON — the pipeline preloads either way, so
    a false flag would be a lie. A builder with no applicable weight set admits
    only OFF, which is what makes the regime error the resolver raises
    unauthorable from the configurator. The forbidden ``weight_source`` decision
    flag is read only by the resolver / registry layers that pass it here.
    """
    if source_declared:
        return (True,)
    ids = legal_weight_set_ids(cfg)
    if ids is None:
        return None
    return (False, True) if ids else (False,)


def preload_unavailable_reason(cfg: Mapping[str, Any]) -> Optional[str]:
    """Why the preload regime is unavailable here, or ``None`` when it is not."""
    sets = registered_weight_sets(cfg)
    if sets is None or (applicable_weight_sets(cfg) or ()):
        return None
    model = cfg.get("model_type") or "this model builder"
    if not sets:
        return f"{model} registers no pretrained weights"
    mismatches = "; ".join(
        f"{record['id']}: {weight_set_mismatch(record, cfg)}" for record in sets
    )
    return (
        f"{model} registers {len(sets)} pretrained weight set(s), none applicable "
        f"to this configuration — {mismatches}"
    )


def select_weight_set(cfg: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The registered weight set the preload regime selects: an explicit
    applicable ``pretrained_weight_set`` id, else the builder's default (its
    first applicable registration). ``None`` when the regime is off or nothing
    is selectable.

    The regime here is the ``preload_weights`` FLAG; an explicit ``weight_source``
    (a checkpoint path/URL, not a registered set) is resolved by
    ``resolve_weight_source`` in the DeploymentPlan and never lands here.
    """
    if not cfg.get("preload_weights"):
        return None
    applicable = applicable_weight_sets(cfg) or ()
    declared = cfg.get(WEIGHT_SET_KEY)
    if declared:
        return next((r for r in applicable if r["id"] == declared), None)
    return applicable[0] if applicable else None


def derived_weight_set_id(cfg: Mapping[str, Any]) -> Optional[str]:
    """The id the derivation resolves ``pretrained_weight_set`` to (the green value)."""
    chosen = select_weight_set(cfg)
    return None if chosen is None else str(chosen["id"])


def selected_source(cfg: Mapping[str, Any]) -> Optional[str]:
    """The loader-facing source of the chosen registered weight set, or ``None``.

    This is the REGISTERED source only; the explicit-``weight_source`` override
    (config-data escape) is applied by ``resolve_weight_source`` in the
    DeploymentPlan resolver, which owns the forbidden decision-flag read.
    """
    chosen = select_weight_set(cfg)
    return None if chosen is None else str(chosen["source"])


def preload_regime_error(cfg: Mapping[str, Any]) -> ValueError:
    """THE regime's fail-loud message; the wizard renders the same text keyed.

    Every remedy it names is reachable: the configurator's Pretrained switch, its
    weight-set selector, or an explicit ``weight_source`` in the config document.
    """
    declared = cfg.get(WEIGHT_SET_KEY)
    if declared:
        record = next(
            (r for r in registered_weight_sets(cfg) or () if r["id"] == declared), None
        )
        detail = (
            weight_set_mismatch(record, cfg) if record is not None
            else f"the builder registers {list(legal_weight_set_ids(cfg) or ())}"
        )
        return ValueError(
            f"pretrained_weight_set={declared!r} cannot be used here: {detail}. "
            f"Choose a registered weight set, or turn the preload regime off."
        )
    reason = preload_unavailable_reason(cfg) or (
        "the model builder registers no pretrained weight set applicable to this "
        "configuration"
    )
    return ValueError(
        f"preload_weights=true but {reason} "
        f"(ModelWorkloadProfile.pretrained_weight_sets). Turn the preload regime "
        f"off, or declare an explicit weight_source (checkpoint path or URL) as "
        f"config data."
    )
