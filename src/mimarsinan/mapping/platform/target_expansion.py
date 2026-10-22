"""The target expansion: the one function every config entry point calls to write a declared target's keys.

``expand_target`` keeps ``target``, writes each key the bundle sets that the declarations leave out into its
registry section, keeps an explicit key that declares the bundle's value (so the registry still judges it as
written), refuses a differing one unless ``target_overrides`` replaces the key with a note, refuses an override
that departs from nothing, and refuses an unknown name listing the known ones; twice equals once.

- **One reading at every door.** Every key the expansion reads (``target``, ``target_overrides``, the two labels
  and, under a target, every key the bundle sets) is read in its registry section, from the declarations as the
  run lays them (``sections_run_layout``); a declaration in the other section that the run reads refuses by name,
  at the run door from the two sections, at the document door from the parser's unknown paths (one no reader
  reads is the one judge's to refuse: ``declaration_places.placement_errors``, the document's top level and an
  object the parser does not read, such as the nested bias object, included). The selecting keys and the labels
  are read in the flat ``platform_constraints`` only: the run door does not read the wizard's nested user/auto
  shapes, so a declaration in one (in its body, or beside its mode) refuses by name at every door, located by the
  document parser.
- **Not applied.** A key the bundle sets whose registry relevance fails under the declarations, and that no
  other concern provides, is named "<key> not applied" in the faithfulness label (the relevance the wizard's
  dormant rows read); one whose consumer applies it only where ``applied_when`` holds, and it fails, is named
  "<applied_by> not applied" (its other consumers read it as declared).
- **Same declaration.** Two values declare the same thing when their consumers read the same JSON value: types
  count at every depth (4.0 is not 4, True is not 1), spellings a consumer normalizes do not.
- **Labels.** ``write_derived`` adds a flat config's two labels, never a document's, and refuses a declared label
  the expansion does not derive (an equal one is kept, so a flat config re-merges to itself).
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY, parse_deployment_document
from mimarsinan.config_schema.registry.consumer_reading import SCHEMA_DEFAULTS, read_like_consumers
from mimarsinan.config_schema.registry.declaration_judge import sections_run_layout
from mimarsinan.config_schema.registry.declaration_places import unread_section_paths
from mimarsinan.config_schema.registry.parse import ParsedDocument
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError, entry_value_errors
# Bound as a module (resolved from sys.modules while it initializes): target_bundles imports the registry, whose
# package imports the entry points and so this module first when target_bundles is the first import.
import mimarsinan.mapping.platform.target_bundles as target_bundles

DP, PC = "deployment_parameters", "platform_constraints"
#: The keys that select a target and depart from it.
TARGET_KEYS = ("target", "target_overrides")
#: The two labels the expansion derives on a flat config.
LABELS = ("target_claim_class", "target_faithfulness_class")
#: The keys the expansion reads whatever the target: read in the flat platform constraints at every door.
READ_KEYS = (*TARGET_KEYS, *LABELS)


class TargetConflictError(ToolchainRefusal, ValueError):
    """A declaration differs from a value the target sets, or hands one of its keys to another concern."""


class TargetOverrideError(ToolchainRefusal, ValueError):
    """A ``target_overrides`` entry the expansion cannot apply, or one that departs from nothing."""


class TargetDeclarationError(ToolchainRefusal, ValueError):
    """A key the expansion reads declared outside its registry section, or a derived label declared."""


def same_declaration(first: Any, second: Any) -> bool:
    """Whether two values are the same JSON value: types count at every depth, mapping order does not."""
    return _json(first) == _json(second)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):  # unorderable keys, or a cycle
        return repr(value)


def _declares(key: str, explicit: Any, wanted: Any, flat: Mapping[str, Any]) -> bool:
    """Whether ``explicit`` declares ``wanted`` for ``key``: the same value as the key's consumers read both."""
    read = (read_like_consumers({**flat, key: value})[key] for value in (explicit, wanted))
    return same_declaration(*read)


def _refuse_misplaced(keys: Iterable[str], declared_paths: Iterable[str], unread: Iterable[str] = ()) -> None:
    """Refuse each of ``keys`` found at ``<other section>.<key>`` among ``declared_paths`` where the run reads it
    (``unread``: the paths no reader reads, which the one judge refuses by name)."""
    wrong = {f"{DP if REGISTRY[key].section == PC else PC}.{key}": key for key in keys}
    if found := sorted(path for path in set(declared_paths) - set(unread) if path in wrong):
        raise TargetDeclarationError("; ".join(
            f"{path}: {wrong[path]} is a {REGISTRY[wrong[path]].section} key, read there at every door; "
            "declare it there" for path in found))


def _refuse_nested(paths: Mapping[str, Tuple[str, ...]], unknown: Iterable[str]) -> None:
    """Refuse each of ``READ_KEYS`` the document parser locates in a nested platform shape: in its body (a
    container path below ``platform_constraints``) or beside its mode (an unknown ``platform_constraints.<key>``,
    which a flat platform never holds, since each of them is a platform key)."""
    beside = {f"{PC}.{key}": key for key in READ_KEYS}
    found = {".".join((*paths[key], key)): key for key in READ_KEYS if key in paths and tuple(paths[key]) != (PC,)}
    found.update((path, beside[path]) for path in set(unknown) if path in beside)
    if found:
        raise TargetDeclarationError("; ".join(
            f"{path}: {found[path]} is read in the flat platform_constraints at every door (the run door does not "
            "read the nested user/auto platform shapes): "
            f"{'remove it, the expansion derives it' if found[path] in LABELS else 'declare it there'}"
            for path in sorted(found)))


def _section_paths(dp: Mapping[str, Any], pc: Mapping[str, Any]) -> List[str]:
    return [f"{DP}.{key}" for key in dp] + [f"{PC}.{key}" for key in pc]


def _refuse_declared_labels(pc: Mapping[str, Any], derived: Mapping[str, str], name: Any) -> None:
    """Refuse a declared label (by presence: a declared null is a declaration) the expansion does not derive."""
    errors = [
        f"{label}={pc[label]!r} is derived by the target expansion and no target is declared: remove it"
        if label not in derived else
        f"{label}={pc[label]!r} differs from {label}={derived[label]!r}, which target={name!r} derives: remove it"
        for label in LABELS
        if label in pc and not (label in derived and same_declaration(pc[label], derived[label]))
    ]
    if errors:
        raise TargetDeclarationError("; ".join(errors))


def _judge_overrides(name: str, values: Mapping[str, Any], overrides: Mapping[str, Any],
                     flat: Mapping[str, Any]) -> None:
    if stray := sorted(set(overrides) - set(values)):
        raise TargetOverrideError(f"target_overrides {stray}: target={name!r} sets no such key (it sets "
                                  f"{sorted(values)}); declare them as ordinary keys")
    if inert := sorted(key for key in overrides if _declares(key, overrides[key]["value"], values[key], flat)):
        raise TargetOverrideError("; ".join(
            f"target_overrides.{key}.value equals the {key} target={name!r} sets, so nothing departs: drop the "
            "override" for key in inert))


def _not_applied(key: str, state: Mapping[str, Any]) -> Optional[str]:
    """The departure of a bundle key no consumer applies under ``state`` (its relevance fails), or that the consumer
    ``applied_by`` names does not apply (its ``applied_when`` fails; the key's other consumers read it)."""
    entry = REGISTRY[key]
    if not entry.relevant.evaluate(state):
        return f"{key} not applied"
    if entry.applied_when is not None and not entry.applied_when.evaluate(state):
        return f"{entry.applied_by} not applied"
    return None


def target_assignments(dp: Mapping[str, Any], pc: Mapping[str, Any], *, write_derived: bool,
                       bundles: Optional[Mapping[str, target_bundles.TargetBundle]] = None,
                       declared_paths: Optional[Iterable[str]] = None,
                       unread: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
    """``{section: {key: value}}`` the expansion writes (empty without a target).

    ``declared_paths`` are the ``<section>.<key>`` paths the declarations hold outside the two sections as
    given (the document parser's unknown paths), and ``unread`` those no reader reads; None reads them from
    ``dp`` and ``pc``, the sections as a run lays them, whose nested platform shape the parser locates.
    ``target`` and ``target_overrides`` are judged by their registry schemas first, and a key whose registry
    relevance hands it to another concern (``provided_by``) is not the bundle's to set."""
    paths = list(_section_paths(dp, pc) if declared_paths is None else declared_paths)
    _refuse_misplaced(READ_KEYS, paths)
    if declared_paths is None:
        nested = parse_deployment_document({DP: dp, PC: pc})
        _refuse_nested(nested.paths, nested.unknown)
    flat = sections_run_layout(dp, pc)
    for key in TARGET_KEYS:
        if flat.get(key) is not None and (errors := entry_value_errors(REGISTRY[key], flat[key])):
            raise DeclarationValueError("; ".join(errors))
    name, overrides = flat.get("target"), dict(flat.get("target_overrides") or {})
    if name is None:
        if overrides:
            raise TargetOverrideError(f"target_overrides {sorted(overrides)} declared without a target: "
                                      "declare target, or the keys themselves")
        if write_derived:
            _refuse_declared_labels(pc, {}, None)
        return {}
    bundle = target_bundles.get_target_bundle(name, bundles)
    values = bundle.values  # a fresh copy (target_bundles.TargetBundle.values)
    _refuse_misplaced(values, paths, unread_section_paths(dp, pc) if unread is None else unread)
    _judge_overrides(name, values, overrides, flat)
    out: Dict[str, Dict[str, Any]] = {DP: {}, PC: {}}
    state = {**SCHEMA_DEFAULTS, **flat, **{key: overrides[key]["value"] if key in overrides else value
                                         for key, value in values.items() if flat.get(key) is None}}
    for key, value in values.items():
        if (entry := REGISTRY[key]).provided_by and not entry.relevant.evaluate(state):
            raise TargetConflictError(f"target={name!r} sets {key}, which {entry.provided_by} provides under these "
                                      "declarations: drop target, or the declaration that hands the key over")
        wanted = overrides[key]["value"] if key in overrides else value
        if (explicit := flat.get(key)) is None:
            out[entry.section][key] = copy.deepcopy(wanted)
        elif not _declares(key, explicit, wanted, flat):
            source = f"target_overrides.{key}.value" if key in overrides else f"target={name!r}"
            raise TargetConflictError(f"{key}={explicit!r} differs from {source}, which sets {key}={wanted!r}: "
                                      f"drop {key}, or declare it in target_overrides with a note")
    if write_derived:
        # A key another concern provides was refused above wherever its relevance fails (Rule R).
        inert = [departure for key in values if (departure := _not_applied(key, state))]
        derived = {"target_claim_class": bundle.claim_class,
                   "target_faithfulness_class": bundle.faithfulness.with_declared(overrides).with_departures(
                       inert).label()}
        _refuse_declared_labels(pc, derived, name)
        out[PC].update(derived)
    return out


def expand_target(dp: Mapping[str, Any], pc: Mapping[str, Any], *, write_derived: bool,
                  bundles: Optional[Mapping[str, target_bundles.TargetBundle]] = None
                  ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The two sections with the target's keys written in (copies; the inputs are not changed)."""
    written = target_assignments(dp, pc, write_derived=write_derived, bundles=bundles)
    return {**dp, **written.get(DP, {})}, {**pc, **written.get(PC, {})}


def _document_assignments(document: Mapping[str, Any], parsed: ParsedDocument, *, write_derived: bool,
                          bundles: Optional[Mapping[str, target_bundles.TargetBundle]]) -> Dict[str, Dict[str, Any]]:
    """``target_assignments`` of a document as ``parse_deployment_document`` parsed it; empty without a target."""
    _refuse_misplaced(READ_KEYS, parsed.unknown)
    _refuse_nested(parsed.paths, parsed.unknown)
    if parsed.pc.get("target") is None and not parsed.pc.get("target_overrides"):
        return {}
    return target_assignments(parsed.dp, parsed.pc, write_derived=write_derived, bundles=bundles,
                              declared_paths=parsed.unknown,
                              unread=unread_section_paths(document.get(DP), document.get(PC)))


def expand_target_document(document: Mapping[str, Any], *, write_derived: bool = False,
                           bundles: Optional[Mapping[str, target_bundles.TargetBundle]] = None) -> Mapping[str, Any]:
    """A deployment document with the target's keys written into its two sections (the target is read in the
    flat platform constraints only); the document itself without a target."""
    written = _document_assignments(document, parse_deployment_document(document), write_derived=write_derived,
                                    bundles=bundles)
    if not written:
        return document
    out = copy.deepcopy(dict(document))
    for section in (DP, PC):
        if isinstance(out.get(section), dict):
            out[section].update(written.get(section, {}))
    return out


def target_core_semantics(document: Mapping[str, Any]) -> Optional[str]:
    """The core semantics a run of ``document`` takes: the declared one, else the target's, else None (the
    default). A read of the expansion that writes nothing and never refuses: a refused target, a nested one
    included, leaves the declarations; the parser reads any draft, a container that is no object an unknown
    path."""
    parsed = parse_deployment_document(document)
    try:
        written = _document_assignments(document, parsed, write_derived=False, bundles=None)
    except ToolchainRefusal:
        written = {}
    value = {**parsed.dp, **written.get(DP, {})}.get("core_semantics")
    return None if value is None else str(value)
