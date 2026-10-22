"""The one judge of declared values both doors call (S0 amendment B, owner decision 8; closed jointly with G4).

``declaration_errors`` judges every present key but the runtime facts by the schema the document door's registry
loop judges it with (``schema_of``), one row per key, after its domain tag where a key with a value schema or the
passthrough flag is declared; ``run_door_errors`` adds the platform bias rule, each refusal once. The run door
(``build_platform_constraints_resolved``) judges a flat config with it.

A document reaches the run door only after the run has read it, so every entry point judges the document's
declarations as written before anything else reads one: ``merge_pipeline_config`` calls ``declared_sections_errors``
on the two sections, and ``parse_deployment_config`` and ``validate_deployment_config`` call
``document_declaration_errors`` on the document (each after the target expansion, which reads only its own keys,
judges them by their schemas, and supplies the core semantics this judge's domain rule reads).

Every reader of a run reads one layout of a document (``declaration_places.run_layers``: the top-level keys laid
beneath the deployment parameters, the deployment parameters, the platform default layer, the platform constraints;
a null declaration is unset, so it lays nothing). So the judge first refuses by name a declaration no reader reads
and a key declared in two places the run reads with differing readings (``declaration_places``). It then judges the
two sections twice: in the run's layout (``sections_run_layout``) and each section's own declarations as written
(``section_declarations``, the document door's view of a flat document), so a section's own declaration that a later
copy shadows is judged too; a shadowed copy outside its key's own section (a platform key, or a laid top-level key,
in the deployment parameters under a platform declaration) is judged by its value schema with its place
(``reading_errors``, review round 3). Each declaration is judged by its value schema, by the document domain rule
(``mvm_document_errors`` over the declarations a run reads), by its legal set under the run's state
(``legality_rows``) and by the schema its consumer applies where it applies it (``applied_errors``); the layout
carries the platform bias rule and the floorplan pairing. The run's state holds only the declarations their schemas
admit, and a legal set or a condition that reads a declaration its schema refuses is undetermined there: that
refusal is the one message (``ReadRecorder``). On a flat document the judge so states every refusal of the document
door's registry loop, legal-value-set law and domain rules, and of the run's layout; the wizard's structural platform
shapes (``mode`` user or auto) nest a body only the document door's parser reads.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import floorplan_config_errors
from mimarsinan.config_schema.deployment_derivation import legal_value_error
from mimarsinan.config_schema.registry.build import REGISTRY, passthrough_keys
from mimarsinan.config_schema.registry.consumer_reading import SCHEMA_DEFAULTS, ReadRecorder, read_like_consumers
from mimarsinan.config_schema.registry.declaration_places import (
    declared, placement_errors, read_sections, reading_errors, registered_declaration_errors, run_layers,
    section_placement_errors,
)
from mimarsinan.config_schema.registry.domain_rules import mvm_document_errors
from mimarsinan.config_schema.registry.entries_write import platform_bias_errors
from mimarsinan.config_schema.registry.types import Category
from mimarsinan.config_schema.registry.value_schema import domain_errors, entry_value_errors, value_errors

#: The section of the keys the session reads from a document's top level, beside the two sections a run lays flat.
TOP_SECTION = "top"
DP, PC = "deployment_parameters", "platform_constraints"


def domain_of(config: Mapping[str, Any]) -> str:
    """``value`` for a value-domain (MVM) config, else ``event``; an unknown core_semantics raises ValueError."""
    return "value" if is_mvm_core_semantics(config.get("core_semantics")) else "event"


def declaration_errors(pipeline_config: Mapping[str, Any], run_domain: Optional[str]) -> List[str]:
    """Why a present key refuses, one row per key as the document door's registry loop states it: its value by
    the schema the document door judges it with (``schema_of``: the declared value schema, else the scalar
    schema of its type, bounds and options), after its domain tag against ``run_domain`` where it carries a
    value schema or the passthrough flag and its presence is a declaration (no schema default; None judges
    values only). A runtime fact the run writes (the device, the dataset's shape and classes) is no declaration:
    no document declares it, and the document door refuses one that does."""
    carried = set(passthrough_keys())
    errors: List[str] = []
    for key, entry in REGISTRY.items():
        value = pipeline_config.get(key)
        if value is None or entry.category is Category.RUNTIME:
            continue
        if (run_domain is not None and not entry.has_default()
                and (key in carried or entry.value_schema is not None)):
            errors += domain_errors(entry, run_domain=run_domain)
        if messages := entry_value_errors(entry, value):
            errors.append("; ".join(messages))
    return errors


def _once(*messages: str) -> List[str]:
    out: List[str] = []
    for message in messages:
        if message not in out:
            out.append(message)
    return out


def _layout_errors(config: Mapping[str, Any], run_domain: Optional[str]) -> List[str]:
    """``declaration_errors``, then the cross-key rules of a layout's declarations: the platform bias rule (which
    judges each bias declaration by its schema itself) and the floorplan pairing over the declarations their
    schemas admit (a value outside its schema is the schema's to name, once)."""
    admitted = {key: value for key, value in config.items()
                if key not in REGISTRY or not entry_value_errors(REGISTRY[key], value)}
    return _once(*declaration_errors(config, run_domain), *platform_bias_errors(config, REGISTRY["has_bias"]),
                 *floorplan_config_errors(admitted))


def run_door_errors(pipeline_config: Mapping[str, Any]) -> List[str]:
    """Why the run door refuses a flat config read like its consumers, else []: every present declaration by its
    value schema, with the domain tag where its presence is a declaration (``declaration_errors``), then the
    platform bias rule both doors call (``platform_bias_errors``: each declaration as a boolean, then their
    agreement). Each refusal is stated once, in that order."""
    try:
        run_domain: Optional[str] = domain_of(pipeline_config)
    except ValueError:  # an unknown core_semantics, which its value schema names below
        run_domain = None
    return _layout_errors(pipeline_config, run_domain)


def sections_run_layout(deployment_parameters: Any, platform_constraints: Any) -> Dict[str, Any]:
    """The declarations a run reads from a document's two sections, laid as ``merge_pipeline_config`` lays them
    (``run_layers`` without the default layers): the deployment parameters, then the platform constraints' default
    layer (which replaces an earlier declaration of a key it holds), then the platform constraints. A later layer
    replaces a value whole (no deep merge), a null declaration is unset and lays nothing, and a section that is no
    object lays nothing."""
    return run_layers(deployment_parameters, platform_constraints, defaults=False)


def section_declarations(deployment_parameters: Any, platform_constraints: Any) -> Dict[str, Any]:
    """Each section's own declarations as written (the keys the registry places in that section), side by side:
    the document door's view of the two sections (its parser's known keys), and the declarations a consumer takes
    out of their section itself even where a later section's copy replaces them in the run's layout. A section
    that is no object declares nothing."""
    own: Dict[str, Any] = {}
    for name, section in ((DP, deployment_parameters), (PC, platform_constraints)):
        if isinstance(section, Mapping):
            own.update({key: value for key, value in section.items()
                        if key in REGISTRY and REGISTRY[key].section == name})
    return own


def legality_rows(state: Mapping[str, Any], declared_values: Mapping[str, Any], *,
                  refused: Iterable[str] = ()) -> List[Tuple[str, Any, Tuple[Any, ...]]]:
    """``(key, value, legal set)`` of every declared legality-bearing value outside the legal set the registry
    gives under ``state`` (``legal_values``; a set that does not apply, None, judges nothing). A value its own
    schema refuses is the schema's to name, once (``entry_value_errors``), so the set judges only the rest; and a
    set that reads one of the ``refused`` declarations (each refused by its own schema) is undetermined, so it
    judges nothing either."""
    refused = frozenset(refused)
    rows: List[Tuple[str, Any, Tuple[Any, ...]]] = []
    for key, entry in REGISTRY.items():
        value = declared_values.get(key)
        if entry.legal_values is None or value is None or entry_value_errors(entry, value):
            continue
        reads = ReadRecorder(state)
        legal = entry.legal_values(reads)
        if legal is not None and not reads.read & refused and value not in (legal := tuple(legal)):
            rows.append((key, value, legal))
    return rows


def applied_errors(state: Mapping[str, Any], declared_values: Mapping[str, Any], *,
                   refused: Iterable[str] = ()) -> List[str]:
    """Why a declared value its own schema admits fails the schema its consumer applies (a key's
    ``applied_schema`` where its ``applied_when`` holds under ``state``), each with the reason the entry names;
    a condition that reads one of the ``refused`` declarations is undetermined and judges nothing."""
    refused = frozenset(refused)
    errors: List[str] = []
    for key, entry in REGISTRY.items():
        value, schema, when = declared_values.get(key), entry.applied_schema, entry.applied_when
        if schema is None or when is None or value is None or entry_value_errors(entry, value):
            continue
        reads = ReadRecorder(state)
        if when.evaluate(reads) and not reads.read & refused:
            errors += [f"{message}: {schema.doc}" for message in value_errors(schema, value, key)]
    return errors


def _domain_rule_errors(deployment_parameters: Any, platform_constraints: Any) -> List[str]:
    """The document domain rule over the declarations a run reads in the two sections (a null one is unset),
    under the core semantics the run's layout carries (a core_semantics declared in the platform constraints is
    the run's too); an unknown core_semantics names no domain (its value schema refuses it)."""
    if not isinstance(deployment_parameters, Mapping):
        return []
    layout = sections_run_layout(deployment_parameters, platform_constraints)
    try:
        return mvm_document_errors(read_sections(deployment_parameters, platform_constraints), run_layout=layout)
    except ValueError:
        return []


def _sections_view(deployment_parameters: Any, platform_constraints: Any, top: Optional[Mapping[str, Any]],
                   registration: Optional[Mapping[str, Any]] = None
                   ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], FrozenSet[str]]:
    """The run's layout of the two sections and each section's own declarations, read like their consumers; the
    state a run judges them under: the schema defaults, the workload ``registration`` folded beneath the
    declarations (the builder's registered weight sets, where a caller reads them), the top-level declarations
    ``top``, the layout, each declaration its own schema admits (a null one is unset); and the declarations the
    state leaves out because their schemas refuse them."""
    layout = read_like_consumers({key: value for key, value in sections_run_layout(
        deployment_parameters, platform_constraints).items() if key not in REGISTRY
        or REGISTRY[key].section != TOP_SECTION or REGISTRY[key].laid_beneath is not None})
    own = read_like_consumers(section_declarations(deployment_parameters, platform_constraints))
    stated = {**declared(top or {}), **layout}
    admitted = {key: value for key, value in stated.items()
                if key not in REGISTRY or not entry_value_errors(REGISTRY[key], value)}
    state = {**SCHEMA_DEFAULTS, **(registration or {}), **admitted}
    return layout, own, state, frozenset(stated) - frozenset(admitted)


def applied_view_errors(deployment_parameters: Any, platform_constraints: Any, *,
                        top: Optional[Mapping[str, Any]] = None) -> List[str]:
    """``applied_errors`` of the run's layout of two sections under the state a run judges them in
    (``_sections_view``): the document door's judgement of a structural platform body by the schemas its consumers
    apply, over the parser's flat view of it."""
    layout, _, state, refused = _sections_view(deployment_parameters, platform_constraints, top)
    return applied_errors(state, layout, refused=refused)


def declared_legality_rows(deployment_parameters: Any, platform_constraints: Any, *,
                           top: Optional[Mapping[str, Any]] = None, registration: Optional[Mapping[str, Any]] = None
                           ) -> List[Tuple[str, Any, Tuple[Any, ...]]]:
    """``legality_rows`` of the run's layout of the two sections and of each section's own declarations, under
    the state a run judges them in (``_sections_view``), each row once."""
    layout, own, state, refused = _sections_view(deployment_parameters, platform_constraints, top, registration)
    rows: List[Tuple[str, Any, Tuple[Any, ...]]] = []
    for row in (*legality_rows(state, layout, refused=refused), *legality_rows(state, own, refused=refused)):
        if row not in rows:
            rows.append(row)
    return rows


def declared_sections_errors(deployment_parameters: Any, platform_constraints: Any, *,
                             top: Optional[Mapping[str, Any]] = None,
                             registration: Optional[Mapping[str, Any]] = None) -> List[str]:
    """The verdict on the declarations of a document's two sections, read like their consumers, before the
    derivation or any consumer reads one: a declaration no reader reads and a key with two differing readings
    (``declaration_places``, with the top-level declarations ``top``); the layout a run gives the sections (a key
    declared in the other section included) by every value schema and the platform bias rule, each section's own
    declarations as written (``section_declarations``, read together, as the document door reads them) by their
    value schemas, the document domain rule, a registration-owned key declared otherwise
    (``registered_declaration_errors``, where a caller gives the workload ``registration``), and the
    legal-value-set law (``declared_legality_rows``) and the schemas consumers apply (``applied_errors``) under
    the run's state. Each refusal is stated once."""
    layout, own, state, refused = _sections_view(deployment_parameters, platform_constraints, top, registration)
    illegal = [str(legal_value_error(key, value, legal)) for key, value, legal in
               declared_legality_rows(deployment_parameters, platform_constraints, top=top,
                                      registration=registration)]
    return _once(*section_placement_errors(deployment_parameters, platform_constraints),
                 *reading_errors(top or {}, deployment_parameters, platform_constraints),
                 *_layout_errors(layout, None), *declaration_errors(own, None),
                 *_domain_rule_errors(deployment_parameters, platform_constraints),
                 *registered_declaration_errors(layout, registration or {}), *illegal,
                 *applied_errors(state, layout, refused=refused))


def document_declaration_errors(document: Mapping[str, Any]) -> List[str]:
    """The run door's verdict on every declaration of a deployment document before a run reads any: every
    declaration no reader reads (``placement_errors``), each top-level key the session reads (section ``top``) by
    its schema, then ``declared_sections_errors`` over the two sections with the top-level declarations. Each
    refusal is stated once."""
    top = {key: document[key] for key, entry in REGISTRY.items()
           if entry.section == TOP_SECTION and key in document}
    return _once(*placement_errors(document), *declaration_errors(top, None),
                 *declared_sections_errors(document.get(DP), document.get(PC), top=top))
