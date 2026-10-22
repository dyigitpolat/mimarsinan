"""Where a run reads each declaration: the run's one layout of a document, and the two rules that keep every reader
on it (the S0-B and G4 joint closure, review round 1).

A run lays a document's declarations once (``run_layers``): the top-level keys it lays beneath the deployment
parameters (``laid_beneath``), the deployment parameters, the platform default layer (which replaces a
deployment-parameters declaration of each key it holds), the platform constraints. Each layer replaces a value
whole, and a null declaration is unset, so it lays nothing. Every reader reads this layout: the merge and the
derivation, the session's data provider, hardware-search branch and pipeline mode, the quantization assembly
contract. So a declaration has one reading, or the one judge refuses it by name:

- **Unread** (``placement_errors``): a declaration no reader reads. A section key at the document's top level; a
  top-level key the run does not lay, in a section (the session reads it at the top level only); a
  deployment-parameters key the platform default layer replaces; a registry key inside an object the parser does
  not read, at any depth (the nested bias object's own ``has_bias`` excepted: the bias rule reads it, unless the
  platform constraints' nested object replaces the deployment parameters' one). A nested bias object that is no
  object is refused by name too: the bias rule would read nothing in it.
- **Two readings** (``reading_errors``): a key declared, each value admitted by its schema, in two places the run
  reads (the top level and a section, or both sections) whose consumer readings differ. The layout reads one of
  them, so the other would be a silent misread. A copy its schema refuses is no reading: the rule refuses it by its
  schema, prefixed with its place, where no other judgement states it (a deployment-parameters copy the platform
  constraints' copy shadows, outside the key's own section; review round 3), so no copy is dropped unjudged.

A declaration in the other section that the run reads (``platform_constraints.lr``, ``platform_constraints.
hw_config_mode``) is neither: the layout reads it, and every reader reads the layout.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.config_schema.defaults import get_default_deployment_parameters, get_default_platform_constraints
from mimarsinan.config_schema.registry.build import REGISTRY
from mimarsinan.config_schema.registry.consumer_reading import read_like_consumers
from mimarsinan.config_schema.registry.entries_write import PLATFORM_BIAS_PATHS
from mimarsinan.config_schema.registry.parse import parse_deployment_document
from mimarsinan.config_schema.registry.types import Category
from mimarsinan.config_schema.registry.value_schema import entry_value_errors

TOP, DP, PC = "top", "deployment_parameters", "platform_constraints"
#: The nested bias object a section may hold (``PLATFORM_BIAS_PATHS``): the bias rule reads its own ``has_bias``.
_BIAS_OBJECT, _BIAS_KEY = next(path.split(".") for path in PLATFORM_BIAS_PATHS if "." in path)


def declared(section: Any) -> Dict[str, Any]:
    """A section's declarations: a null one is unset, and a section that is no object declares nothing."""
    return {key: value for key, value in section.items() if value is not None} if isinstance(section, Mapping) else {}


def run_layers(deployment_parameters: Any, platform_constraints: Any, *, defaults: bool) -> Dict[str, Any]:
    """The run's layout of two sections: the deployment parameters, then the platform default layer (which
    replaces a deployment-parameters declaration of each key it holds), then the platform constraints; each layer
    replaces a value whole, and a null declaration lays nothing. ``defaults`` lays the two default layers too (the
    merge's flat config, in the merge's key order), else the declarations alone."""
    layout: Dict[str, Any] = {}
    for layer, section in ((get_default_deployment_parameters(), deployment_parameters),
                           (get_default_platform_constraints(), platform_constraints)):
        if defaults:
            layout.update(layer)
        else:
            for key in layer:
                layout.pop(key, None)
        layout.update(declared(section))
    return layout


def laid_top_keys() -> Tuple[str, ...]:
    """The top-level keys a run lays beneath the deployment parameters (``laid_beneath``)."""
    return tuple(key for key, entry in REGISTRY.items() if entry.laid_beneath is not None)


def not_declarable_message(key: str) -> Optional[str]:
    """Why a document may not declare ``key`` (a runtime fact, or a derived key its owner derives), else None."""
    entry = REGISTRY[key]
    if entry.category is Category.RUNTIME:
        owner_kind = "the runtime"
    elif entry.category is Category.DERIVED and not entry.declarable:
        owner_kind = "the derivation"
    else:
        return None
    return f"{key} is not declarable in a config document — {owner_kind} ({entry.owner}) owns it. Remove the key."


def _home(key: str) -> str:
    section = REGISTRY[key].section
    return "a top-level key" if section == TOP else f"a {section} key"


def _remedy(key: str) -> str:
    entry = REGISTRY[key]
    if entry.category is Category.RUNTIME or not entry.declarable:
        return f"remove it ({entry.owner} owns it)"
    return "declare it at the top level" if entry.section == TOP else f"declare it in {entry.section}"


def _unread(path: str, key: str, why: str = "nothing reads it here") -> str:
    return f"{path}: {key} is {_home(key)}, and {why}: {_remedy(key)}"


def _scan(value: Any, path: str, *, bias_read: bool) -> List[str]:
    """The refusal of every registry key inside an object nothing reads, at any depth (a found key's value is
    its own, so the scan stops there); ``bias_read``: the nested bias object's own ``has_bias`` is read."""
    found: List[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            here = f"{path}.{key}"
            if key not in REGISTRY:
                found += _scan(item, here, bias_read=bias_read)
            elif not (bias_read and key == _BIAS_KEY and path in (f"{DP}.{_BIAS_OBJECT}", f"{PC}.{_BIAS_OBJECT}")):
                found.append(_unread(here, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += _scan(item, f"{path}[{index}]", bias_read=bias_read)
    return found


def _section_placement(section: str, key: str, value: Any, platform_constraints: Any) -> List[str]:
    """The refusal of a declaration at ``<section>.<key>`` the parser does not read there."""
    path = f"{section}.{key}"
    entry = REGISTRY.get(key)
    if entry is None and key == _BIAS_OBJECT and value is not None and not isinstance(value, Mapping):
        return [f"{path} must be an object holding {_BIAS_KEY}, got {type(value).__name__} ({value!r})"]
    if entry is None:  # an object the parser does not read: its bias object is replaced by a platform one
        replaced = section == DP and declared(platform_constraints).get(key) is not None
        return _scan(value, path, bias_read=not replaced)
    if value is None:
        return []
    if entry.section == TOP and entry.laid_beneath is None:
        return [_unread(path, key, "the session reads it at the top level only")]
    if section == DP and key in get_default_platform_constraints():
        return [_unread(path, key, "nothing reads it here (the platform default layer replaces it)")]
    return []


def _section_unread(deployment_parameters: Any, platform_constraints: Any) -> List[Tuple[str, List[str]]]:
    """``(<section>.<key>, refusals)`` of every declaration the parser does not read in its section."""
    sections = {DP: deployment_parameters, PC: platform_constraints}
    found: List[Tuple[str, List[str]]] = []
    for path in parse_deployment_document(sections).unknown:
        segments = path.split(".")
        if len(segments) == 2 and segments[0] in sections and isinstance(sections[segments[0]], Mapping):
            section = sections[segments[0]]
            found.append((path, _section_placement(segments[0], segments[1], section.get(segments[1]),
                                                   platform_constraints)))
    return found


def section_placement_errors(deployment_parameters: Any, platform_constraints: Any) -> List[str]:
    """``placement_errors`` of the two sections a run lays (the merge's view: no top level)."""
    return [message for _, messages in _section_unread(deployment_parameters, platform_constraints)
            for message in messages]


def unread_section_paths(deployment_parameters: Any, platform_constraints: Any) -> Tuple[str, ...]:
    """The ``<section>.<key>`` declarations no reader reads (``section_placement_errors`` refuses each)."""
    return tuple(path for path, messages in _section_unread(deployment_parameters, platform_constraints)
                 if any(message.startswith(f"{path}: ") for message in messages))


def placement_errors(document: Mapping[str, Any]) -> List[str]:
    """Every declaration of a document no reader reads, refused by name: a section key, or a registry key inside
    an object, at the top level, and in the two sections as ``section_placement_errors`` refuses them."""
    errors: List[str] = []
    for key, value in document.items():
        if key in (DP, PC) or str(key).startswith("_") or value is None:
            continue
        if key not in REGISTRY:
            errors += _scan(value, str(key), bias_read=False)
        elif REGISTRY[key].section != TOP:
            errors.append(_unread(str(key), key))
    return [*errors, *section_placement_errors(document.get(DP), document.get(PC))]


def _read_places(top: Mapping[str, Any], deployment_parameters: Any, platform_constraints: Any
                 ) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
    """The places a run reads, with their declarations: the top level (its top-level keys), each section (a key
    the platform default layer replaces, or a top-level key the run does not lay, is no declaration it reads)."""
    replaced = get_default_platform_constraints()

    def read_here(key: str, section: str) -> bool:
        entry = REGISTRY.get(key)
        if entry is None:
            return False
        if section == TOP:
            return entry.section == TOP
        return not ((entry.section == TOP and entry.laid_beneath is None) or (section == DP and key in replaced))

    places = ((TOP, declared(top)), (DP, declared(deployment_parameters)), (PC, declared(platform_constraints)))
    return tuple((name, {k: v for k, v in values.items() if read_here(k, name)}) for name, values in places)


def read_sections(deployment_parameters: Any, platform_constraints: Any) -> Dict[str, Dict[str, Any]]:
    """Each section's declarations a run reads there (registry keys; a null one is unset)."""
    return dict(_read_places({}, deployment_parameters, platform_constraints)[1:])


def _where(place: str) -> str:
    return "at the top level" if place == TOP else f"in {place}"


def _canonical(value: Any) -> str:
    """One JSON text per value as its consumers read it: types count at every depth, mapping order does not."""
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):  # unorderable keys, or a cycle
        return repr(value)


def same_value(first: Any, second: Any) -> bool:
    """Whether two values are the same JSON value (types count at every depth, mapping order does not)."""
    return _canonical(first) == _canonical(second)


def registered_declaration_errors(declared_values: Mapping[str, Any], registration: Mapping[str, Any]) -> List[str]:
    """A declaration of a key no document declares (``not_declarable_message``) that a workload registration
    provides with another value, refused by name: the registration owns it, and the run would read the
    declaration in its place (an equal one is the registration's own value)."""
    return [message for key, value in registration.items()
            if declared_values.get(key) is not None and key in REGISTRY
            and (message := not_declarable_message(key)) is not None and not same_value(declared_values[key], value)]


def _judged_elsewhere(key: str, place: str, last: str) -> bool:
    """Whether another judgement states the schema refusal of ``key``'s copy in ``place``: the layout's (``last``, the
    place whose copy the layout reads), a key's own section's (``section_declarations``; a top-level copy is always its
    key's own, which the session's top-level judgement judges)."""
    return place in (last, REGISTRY[key].section)


def reading_errors(top: Mapping[str, Any], deployment_parameters: Any, platform_constraints: Any) -> List[str]:
    """Every copy of a key in a place the run reads, judged: a copy its schema refuses is no reading, and is
    refused by its schema, prefixed with its place, where no other judgement states it (``_judged_elsewhere``: a
    deployment-parameters copy the platform constraints' copy shadows, outside the key's own section); then every
    key declared, each value admitted by its schema, in two places whose consumer readings differ (the layout reads
    one of them), refused by name with each reading."""
    places = _read_places(top, deployment_parameters, platform_constraints)
    context = {**declared(top), **run_layers(deployment_parameters, platform_constraints, defaults=False)}
    errors: List[str] = []
    for key, entry in REGISTRY.items():
        declaring = [place for place, values in places if key in values]
        readings = []
        for place, values in places:
            if key in values:
                read = read_like_consumers({**context, key: values[key]})[key]
                if not (refusal := entry_value_errors(entry, read)):
                    readings.append((place, values[key], read))
                elif not _judged_elsewhere(key, place, declaring[-1]):
                    errors.append(f"{place}.{key}: " + "; ".join(refusal))
        if len({_canonical(read) for _, _, read in readings}) > 1:
            listed = " and ".join(f"{_where(place)} as {read!r}" for place, _, read in readings)
            errors.append(f"{key} is declared {listed}, and a run reads one {key}: declare it once")
    return errors
