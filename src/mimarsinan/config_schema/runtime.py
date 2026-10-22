"""Build flat runtime config dicts for pipeline, wizard, and API preview."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.common.workload_profile import fold_workload_profiles
from mimarsinan.config_schema.defaults import apply_preset
from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
    derive_pipeline_runtime_parameters,
    derive_platform_constraints,
    enforce_quantization_assembly_contract,
    legal_value_error,
)
from mimarsinan.config_schema.registry.build import REGISTRY
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers, workload_reading_errors
from mimarsinan.config_schema.registry.declaration_judge import (
    declared_sections_errors, legality_rows, sections_run_layout,
)
from mimarsinan.config_schema.registry.declaration_places import declared, laid_top_keys, run_layers, same_value
from mimarsinan.config_schema.registry.types import Category, FieldType
# Bound as modules (resolved from sys.modules while they initialize): target_expansion imports the registry, whose
# package imports this module first when target_expansion is the first import; the run door and the model
# registry are the run's own readers after the derivation, which the document door states too.
import mimarsinan.mapping.platform.target_expansion as target_expansion
import mimarsinan.pipelining.core.platform_constraints_resolver as platform_constraints_resolver
import mimarsinan.pipelining.core.registry.model_registry as model_registry


def build_flat_pipeline_config(
    deployment_parameters: Optional[Dict[str, object]] = None,
    platform_constraints: Optional[Dict[str, object]] = None,
    *,
    pipeline_mode: str = "vanilla",
) -> Dict[str, object]:
    """Merge defaults + overrides the same way ``DeploymentPipeline`` does (without device I/O): the preset and
    the pipeline mode are laid beneath the deployment parameters first, the mode as no declaration (the mode of a
    document that declares none), so a declared target expands under the mode the run takes; then ``run_config``,
    the run's own expansion and derivation."""
    dp = dict(deployment_parameters or {})
    apply_preset(pipeline_mode, dp)
    laid = () if dp.get("pipeline_mode") is not None else ("pipeline_mode",)
    if laid:
        dp["pipeline_mode"] = pipeline_mode
    return run_config(dp, platform_constraints or {}, laid=laid)


class DerivationRefusal(ToolchainRefusal, ValueError):
    """The run's derivation refuses a document's judged declarations (a pairing, the quantization assembly
    contract, a derived key's legal set, a declared derived key it would replace, a value a registration or a
    consumer refuses), with the run's message."""


def replaced_declaration_errors(declarations: Mapping[str, Any], config: Mapping[str, Any]) -> List[str]:
    """Each declarable derived key declared (``declarations``, read like their consumers) whose value the derivation
    replaced in ``config``, refused by name with the declarations it derives it from: the derivation never
    overwrites a declaration silently (a contradiction its own rules name is refused before, by them)."""
    errors: List[str] = []
    for key, entry in REGISTRY.items():
        if (entry.category is not Category.DERIVED or not entry.declarable or key not in declarations
                or key not in config or same_value(declarations[key], config[key])):
            continue
        basis = ", ".join(f"{name}={declarations[name]!r}" for name in entry.derived_from if name in declarations)
        errors.append(f"{key}={declarations[key]!r} is declared, and the run derives {key}={config[key]!r} from "
                      f"{basis or 'the declarations it derives from'}: declare {key}={config[key]!r} or drop one "
                      "of them")
    return errors


def derive_run_config(deployment_parameters: Mapping[str, Any], platform_constraints: Mapping[str, Any], *,
                      laid: Iterable[str] = ()) -> dict:
    """The run's flat config of two judged sections (``merge_pipeline_config`` after its judge and expansion):
    the deployment parameters, then the platform constraints, over their default layers (``run_layers``: a null
    declaration is unset, as both doors read it), read as their consumers read them, then derived: the core
    maxima of the declared cores and the Tq | S contract (``derive_platform_constraints``), the quantization flags
    and the pipeline mode, the runtime parameters. The derivation's explicit keys are the declarations (``laid``: keys
    laid beneath them, which are none), and a declared derived key it replaces is refused by name
    (``replaced_declaration_errors``). A derivation refusal is a ``DerivationRefusal``."""
    config = read_like_consumers(run_layers(deployment_parameters, platform_constraints, defaults=True))
    explicit = (set(declared(deployment_parameters)) | set(declared(platform_constraints))) - set(laid)
    declarations = {key: copy.deepcopy(config[key]) for key in explicit if key in config}
    try:
        derive_platform_constraints(config, cores_declared="cores" in declared(platform_constraints))
        derive_deployment_parameters(config, explicit_keys=explicit)
        derive_pipeline_runtime_parameters(config)
    except DerivationRefusal:
        raise
    except ValueError as refusal:
        raise DerivationRefusal(str(refusal)) from refusal
    if errors := replaced_declaration_errors(declarations, config):
        raise DerivationRefusal("; ".join(errors))
    return config


def run_config(deployment_parameters: Mapping[str, Any], platform_constraints: Mapping[str, Any], *,
               laid: Iterable[str] = ()) -> dict:
    """The run's flat config of two judged sections, the one derivation every reader takes (the merge after its
    judge, the session's pipeline mode, the document door, the wizard's preview): the target expansion with the
    two labels a flat config carries (``expand_target``), then ``derive_run_config``."""
    dp, pc = target_expansion.expand_target(deployment_parameters, platform_constraints, write_derived=True)
    return derive_run_config(dp, pc, laid=laid)


def unset(key: str, value: Any) -> bool:
    """Whether a run reads a top-level declaration as unset: a null, as at both doors, or an empty path (a PATH
    key's empty string, which names no directory)."""
    entry = REGISTRY.get(key)
    return value is None or (value == "" and entry is not None and entry.type is FieldType.PATH)


def top_declaration(document: Mapping[str, Any], key: str, default: Any) -> Any:
    """A top-level declaration a run reads with a default, which an unset one (``unset``) takes."""
    value = document.get(key)
    return default if unset(key, value) else value


def run_sections(document: Mapping[str, Any]) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    """``(deployment parameters, platform constraints, layout)`` a run takes from a document whose target is
    expanded: each top-level key the run lays (``laid_beneath``: the seed, the pipeline mode) laid beneath the
    deployment parameters where neither section declares it, as declared, else its schema default (an undeclared
    pipeline mode lays nothing: the derivation derives it; a section's declaration is the run's, and a differing
    top-level one is the judge's to refuse), under hardware search the platform's search space merged
    (``merge_search_space``), and the run's layout of the two sections (``sections_run_layout``), the one reading
    every reader of the run takes (the data provider's seed, preprocessing and batch size). Neither input is
    changed."""
    dp = copy.deepcopy(dict(document["deployment_parameters"]))
    sections = {**declared(document.get("platform_constraints")), **declared(dp)}
    for key in laid_top_keys():
        default = REGISTRY[key].default if REGISTRY[key].has_default() else None
        if key not in sections and (value := top_declaration(document, key, default)) is not None:
            dp[key] = value
    pc = merge_search_space(document.get("platform_constraints", {}), dp)
    return dp, pc, sections_run_layout(dp, pc)


def _search_section(sections: Iterable[MutableMapping[str, Any]], key: str) -> Optional[MutableMapping[str, Any]]:
    """The section whose ``key`` the run's layout reads (the platform constraints' first), else None."""
    return next((section for section in sections if section.get(key) is not None), None)


def merge_search_space(platform_constraints: Any, deployment_parameters: Dict[str, Any]) -> Any:
    """In hardware search mode (the run's layout of the two sections), the platform constraints without the
    search space the layout reads, merged beneath the ``arch_search`` the layout reads (the deployment parameters'
    unless the platform constraints declare one); the platform input is not changed. Else the platform constraints
    as given. An ``arch_search`` or a search space that is no object is refused by name (``DerivationRefusal``),
    the first with the document door's presence message."""
    if not isinstance(platform_constraints, dict):
        return platform_constraints
    if sections_run_layout(deployment_parameters, platform_constraints).get("hw_config_mode", "fixed") != "search":
        return platform_constraints
    platform = copy.deepcopy(platform_constraints)
    holder = _search_section((platform, deployment_parameters), "search_space")
    space = {} if holder is None else holder.pop("search_space")
    target = _search_section((platform, deployment_parameters), "arch_search") or deployment_parameters
    if target.get("arch_search") is None:
        target["arch_search"] = {}
    arch_cfg = target["arch_search"]
    if not isinstance(arch_cfg, dict):
        raise DerivationRefusal("Search is active but arch_search is missing or not a dict")
    if not isinstance(space, dict):
        raise DerivationRefusal(f"search_space={space!r} must be an object: hardware search merges it into "
                                "arch_search")
    for key, value in space.items():
        arch_cfg.setdefault(key, value)
    return platform


def enforce_run_contract(deployment_parameters: Mapping[str, Any], platform_constraints: Any) -> None:
    """The quantization assembly contract over the run's layout of the two sections (``sections_run_layout``:
    each key it reads where the run reads it, the pipeline mode the derivation reads included); a violation is a
    ``DerivationRefusal``."""
    layout = sections_run_layout(deployment_parameters, platform_constraints)
    try:
        enforce_quantization_assembly_contract(layout, layout, pipeline_mode=layout.get("pipeline_mode"))
    except ValueError as refusal:
        raise DerivationRefusal(str(refusal)) from refusal


def registration_updates(model_profile: Any, data_profile: Any) -> Dict[str, Any]:
    """The config keys the workload registrations provide (``config_updates``), the model's over the data's, as
    ``fold_workload_profiles`` folds them beneath the declarations; None for no registration."""
    return {**(data_profile.config_updates() if data_profile is not None else {}),
            **(model_profile.config_updates() if model_profile is not None else {})}


def registration_errors(config: MutableMapping[str, Any], declared_values: Mapping[str, Any],
                        profiles: Iterable[Any]) -> List[str]:
    """Fold the workload registrations beneath a run's flat config (``fold_workload_profiles``: explicit > model >
    data, ``profiles`` in that order, None for none) and state the run's refusals of what they meet, each by name:
    a declared legality-bearing value the registrations and the dataset facts make illegal (the pretrained
    regime's legal sets read the builder's weight sets and the provider's input shape and classes, which only the
    run holds), then the workload reading's refusal (``workload_reading_errors``). The merge's judge has refused
    by then what the builder's registration alone refuses (``merge_pipeline_config``)."""
    model_profile, data_profile = profiles
    fold_workload_profiles(config, model_profile=model_profile, data_profile=data_profile)
    errors = [str(legal_value_error(key, value, legal)) for key, value, legal in legality_rows(config, declared_values)]
    return errors or workload_reading_errors(config, declared_values)


def model_registration(config: Mapping[str, Any]) -> Any:
    """The builder registration (``ModelWorkloadProfile``) of the model a config names, None for an unknown one."""
    return model_registry.ModelRegistry.get_workload_profile(str(config.get("model_type") or ""))


def run_derivation_errors(document: Mapping[str, Any], *, builder_registration: bool = True) -> List[str]:
    """The run's own refusal of a judged document's declarations, as messages (empty when it derives), in the
    run's order: the quantization assembly contract; the merge's judge under the builder registration
    (``declared_sections_errors`` with the model's ``registration_updates``: a registration-owned key declared
    otherwise, a pretrained value the builder's weight sets make illegal); the merge's derivation (pairings, the
    legality of a derived value); and the run door's reading of the derived config (the resolver: the workload
    reading and the other consumers of the keys no value schema judges; no data provider is read here, so its
    registration and the dataset facts are the run's alone).
    ``builder_registration`` False leaves the builder's registration out (the wizard's pretrained panel states
    those refusals, keyed with their remedies)."""
    if not isinstance(document.get("deployment_parameters"), Mapping):
        return []
    try:
        dp, pc, layout = run_sections(document)
        enforce_run_contract(dp, pc)
        builder = model_registration(layout) if builder_registration else None
        if errors := declared_sections_errors(dp, pc, registration=registration_updates(builder, None)):
            return errors
        config = run_config(dp, pc if isinstance(pc, Mapping) else {})
        platform_constraints_resolver.build_platform_constraints_resolved(config)
    except ValueError as refusal:
        return [str(refusal)]
    return []
