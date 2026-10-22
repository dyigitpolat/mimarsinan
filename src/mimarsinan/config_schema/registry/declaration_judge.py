"""The one judge of declared values both doors call (S0 amendment B, owner decision 8).

``declaration_errors`` judges every present key but the runtime facts by the schema the document door's registry
loop judges it with (``schema_of``), one row per key, after its domain tag where a key with a value schema or the
passthrough flag is declared; ``run_door_errors`` adds the platform bias rule, each refusal once. The run door
(``build_platform_constraints_resolved``) judges its flat config with it.

A run reads some declarations before that flat config exists: the derivation re-derives or casts them
(``weight_quantization: "false"`` reads as true), the session casts the top-level seed, and the session reads some
deployment parameters from the section itself (the data provider's preprocessing and batch size, the hardware mode
that decides the search-space merge), where a platform-constraints copy of the key replaces them in the flat config.
So ``declared_sections_errors`` judges a document's two sections twice: in the layout a run gives them
(``sections_run_layout``: ``merge_pipeline_config``'s layers), and each section's own declarations as written
(``section_declarations``: the keys the registry places in that section, the document door's view of a flat
document); ``document_declaration_errors`` adds the top-level keys the session reads. A caller so judges every
declaration before anything reads it: on a flat document the judge states every refusal of the document door's
registry loop and of the run's layout (the wizard's structural platform shapes, ``mode`` user or auto, nest a body
only the document door's parser reads, and the merge lays its containers as unknown keys). A key declared in the
other section is judged where the run lays it; one a later layer replaces there (a key of another section in the
deployment parameters, under the platform default layer or a platform-constraints copy) is no section's own
declaration: the merge reads it nowhere, and neither door nor the judge judges it (the session reads one such
declaration, the deployment parameters' ``pipeline_mode`` where the top level declares none, for its
quantization-assembly check: a G4 finding). A null declaration is unset, as at both doors, so no judge states one;
``nulls_read_as_unset`` is the run's reading of it (both for G4's merge, interface notes in the S0 amendment B
report).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from mimarsinan.chip_simulation.core_semantics import is_mvm_core_semantics
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.config_schema.registry.build import REGISTRY, passthrough_keys
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry.entries_write import platform_bias_errors
from mimarsinan.config_schema.registry.types import Category
from mimarsinan.config_schema.registry.value_schema import domain_errors, entry_value_errors

#: The section of the keys the session reads from a document's top level, beside the two sections a run lays flat.
TOP_SECTION = "top"


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


def run_door_errors(pipeline_config: Mapping[str, Any]) -> List[str]:
    """Why the run door refuses a flat config read like its consumers, else []: every present declaration by its
    value schema, with the domain tag where its presence is a declaration (``declaration_errors``), then the
    platform bias rule both doors call (``platform_bias_errors``: each declaration as a boolean, then their
    agreement). Each refusal is stated once, in that order."""
    try:
        run_domain: Optional[str] = domain_of(pipeline_config)
    except ValueError:  # an unknown core_semantics, which its value schema names below
        run_domain = None
    return _once(*declaration_errors(pipeline_config, run_domain),
                 *platform_bias_errors(pipeline_config, REGISTRY["has_bias"]))


def sections_run_layout(deployment_parameters: Any, platform_constraints: Any) -> Dict[str, Any]:
    """The declarations a run reads from a document's two sections, laid as ``merge_pipeline_config`` lays them
    over its default layers: the deployment parameters, then the platform constraints' default layer (which
    replaces an earlier declaration of a key it holds), then the platform constraints. A later layer replaces a
    value whole (no deep merge; a null and a non-object replace too); a section that is no object lays nothing."""
    layout: Dict[str, Any] = {}
    for defaults, section in ((get_default_deployment_parameters(), deployment_parameters),
                              (get_default_platform_constraints(), platform_constraints)):
        for key in defaults:
            layout.pop(key, None)
        if isinstance(section, Mapping):
            layout.update(section)
    return layout


def section_declarations(deployment_parameters: Any, platform_constraints: Any) -> Dict[str, Any]:
    """Each section's own declarations as written (the keys the registry places in that section), side by side:
    the document door's view of the two sections (its parser's known keys), and the declarations a consumer takes
    out of their section itself even where a later section's copy replaces them in the run's layout. A section
    that is no object declares nothing."""
    own: Dict[str, Any] = {}
    for name, section in (("deployment_parameters", deployment_parameters),
                          ("platform_constraints", platform_constraints)):
        if isinstance(section, Mapping):
            own.update({key: value for key, value in section.items()
                        if key in REGISTRY and REGISTRY[key].section == name})
    return own


def declared_sections_errors(deployment_parameters: Any, platform_constraints: Any) -> List[str]:
    """The verdict on the declarations of a document's two sections, read like their consumers, before the
    derivation or any consumer reads one: the run door's (``run_door_errors``) on the layout a run gives them, a
    key declared in the other section included, then each section's own declarations as written
    (``section_declarations``, read together, as the document door reads them), by their value schemas. Each
    refusal is stated once."""
    layout = sections_run_layout(deployment_parameters, platform_constraints)
    own = section_declarations(deployment_parameters, platform_constraints)
    return _once(*run_door_errors(read_like_consumers(layout)),
                 *declaration_errors(read_like_consumers(own), None))


def document_declaration_errors(document: Mapping[str, Any]) -> List[str]:
    """The run door's verdict on every declaration of a deployment document before a run reads any: each
    top-level key the session reads (section ``top``) by its schema, then ``declared_sections_errors`` over the
    two sections. A top-level key is judged on its own: the session reads it there even where a section declares
    the key too (the top-level seed seeds the data provider). Each refusal is stated once."""
    top = {key: document[key] for key, entry in REGISTRY.items()
           if entry.section == TOP_SECTION and key in document}
    return _once(*declaration_errors(top, None),
                 *declared_sections_errors(document.get("deployment_parameters"),
                                           document.get("platform_constraints")))


def nulls_read_as_unset(config: MutableMapping[str, Any], defaults: Mapping[str, Any]) -> None:
    """Read every null of a laid config as unset, as both doors read a null declaration: a key whose laid value
    is None takes the value of the default layers beneath (``defaults``), and leaves the config where they hold
    none, so the derivation and every consumer read it as they read the key undeclared. A null that replaced an
    earlier section's declaration keeps replacing it, as in the run layout both doors judge."""
    for key in [key for key, value in config.items() if value is None]:
        if key in defaults:
            config[key] = defaults[key]
        else:
            del config[key]
