"""Target bundles: one name selects a published target's declared keys, claim class and faithfulness class.

A bundle is data, ``targets/<name>.json``: a geometry (a registered ``IMCPlatform`` by name, or declared with
its provenance and claim class), the core fields its cores take beside the geometry's (``core_fields``: the cores'
write path), registry keys judged by the value schema at load (the chip's permissions and input width among them),
and a faithfulness class (exact; approximate or refused, with what departs named). This module loads and reads the
bundles; ``target_expansion`` writes a declared target's keys at every config entry point.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY, Category
from mimarsinan.config_schema.registry.build import declared_paths, effective_class
from mimarsinan.config_schema.registry.decision_classes import COMPOSITE, FUNCTION_BEARING, class_of_path
from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
from mimarsinan.config_schema.registry.value_schema import entry_value_errors, value_errors
from mimarsinan.mapping.platform.imc_platforms import IMCPlatform, get_imc_platform, imc_platform_names
from mimarsinan.mapping.platform.weight_cells import CELLS_PER_WEIGHT, WEIGHT_CELL_ROWS, weight_capacity

DP, PC = "deployment_parameters", "platform_constraints"
_TARGETS_DIRNAME = "targets"
_BUNDLE_FIELDS = frozenset({"name", "display_name", "geometry", "keys", "core_fields", "faithfulness", "notes"})
_GEOMETRY_FIELDS = frozenset({"name", "cores", "weight_bits", "provenance", "claim_class"})
_FAITHFULNESS_FIELDS = frozenset({"class", "named", "note"})
FAITHFULNESS_KINDS = ("exact", "approximate", "refused")
#: Keys no bundle sets: the physics override block is composed whole from named blocks (merge_override_blocks).
UNBUNDLED_KEYS = ("platform_physics_overrides",)


class TargetBundleError(ToolchainRefusal, ValueError):
    """A bundle file the loader refuses: an unknown or undeclarable key, an illegal value, a reused name."""


class UnknownTargetError(ToolchainRefusal, ValueError):
    """``target`` names no bundle; the message lists the known names."""


@dataclass(frozen=True)
class Faithfulness:
    """How closely a declaration follows the published target: a kind and the named departures."""

    kind: str
    named: Tuple[str, ...] = ()

    def with_declared(self, keys: Iterable[str]) -> "Faithfulness":
        """The class once ``keys`` are overridden: an exact bundle becomes approximate, each key named."""
        return self.with_departures(f"{key} declared" for key in sorted(keys))

    def with_departures(self, departures: Iterable[str]) -> "Faithfulness":
        """The class once a deployment departs from the bundle in ``departures``: an exact bundle becomes
        approximate, each departure named after the bundle's own."""
        if not (named := tuple(departures)):
            return self
        return Faithfulness("approximate" if self.kind == "exact" else self.kind, self.named + named)

    def label(self) -> str:
        return self.kind if not self.named else f"{self.kind}: {'; '.join(self.named)}"


@dataclass(frozen=True)
class TargetBundle:
    """One named target: every registry key it sets (its geometry's included) and its two classes.

    The keys are held as JSON text (bundle values are JSON values), so the cached bundle is immutable: each read
    of ``values`` is a fresh copy, and a caller that edits one changes no later read or expansion."""

    name: str
    geometry: str
    claim_class: str
    faithfulness: Faithfulness
    values_json: str

    @property
    def values(self) -> Dict[str, Any]:
        """``{key: value}`` of every key the bundle sets, in bundle order: a fresh copy at each read."""
        return json.loads(self.values_json)


def _refuse(source: str, message: str) -> TargetBundleError:
    return TargetBundleError(f"target bundle {source}: {message}")


def _closed(source: str, where: str, data: Any, allowed: frozenset, required: Iterable[str]) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise _refuse(source, f"{where} must be an object, got {type(data).__name__}")
    unknown, missing = sorted(set(data) - allowed), sorted(set(required) - set(data))
    if unknown or missing:
        raise _refuse(source, f"{where} has unknown fields {unknown} or lacks {missing}")
    return data


def _geometry(source: str, spec: Any) -> Tuple[str, str, Dict[str, Any]]:
    """``(name, claim class, {cores, weight_bits})`` of a registered or declared geometry (``IMCPlatform``)."""
    if isinstance(spec, str):
        try:
            platform = get_imc_platform(spec)
        except KeyError as exc:
            raise _refuse(source, str(exc.args[0])) from None
    else:
        data = _closed(source, "geometry", spec, _GEOMETRY_FIELDS, _GEOMETRY_FIELDS)
        if data["name"] in imc_platform_names():
            raise _refuse(source, f"geometry {data['name']!r} declares a second geometry under a registered name")
        try:
            platform = IMCPlatform(name=str(data["name"]), cores=tuple(data["cores"]),
                                   weight_bits=data["weight_bits"], provenance=str(data["provenance"]),
                                   claim_eligibility=str(data["claim_class"])).validate_for_run()
        except (KeyError, TypeError, ValueError) as exc:
            raise _refuse(source, f"geometry: {exc}") from None
    values = {"cores": [dict(core) for core in platform.cores], "weight_bits": platform.weight_bits}
    return platform.name, platform.claim_eligibility, values


def _bundle_key_errors(key: str, value: Any) -> List[str]:
    """Why a bundle may not set ``key`` to ``value``: the registry's value schema and declarability."""
    entry = REGISTRY.get(key)
    if entry is None:
        return [f"{key} is not a registry key"]
    if key in UNBUNDLED_KEYS:
        return [f"{key} is declared through its named override blocks, never by a bundle"]
    if entry.category is Category.RUNTIME or (entry.category is Category.DERIVED and not entry.declarable):
        return [f"{key} is not declarable"]
    if entry.decision_class == COMPOSITE or entry.section not in (DP, PC):
        return [f"{key} cannot be set by a bundle (a composite or top-level key)"]
    return entry_value_errors(entry, value)


def _faithfulness(source: str, spec: Any) -> Faithfulness:
    data = _closed(source, "faithfulness", spec, _FAITHFULNESS_FIELDS, ("class",))
    kind, named = data["class"], tuple(data.get("named", ()))  # the note is read by people only
    if kind not in FAITHFULNESS_KINDS or (kind == "exact") != (not named):
        raise _refuse(source, f"faithfulness {kind!r} {list(named)}: one of {FAITHFULNESS_KINDS}, "
                              "exact with nothing named, approximate or refused with what departs named")
    return Faithfulness(kind, named)


def _with_core_fields(source: str, cores: List[Dict[str, Any]], fields: Any) -> List[Dict[str, Any]]:
    """[U-C C-2] The geometry's cores with the bundle's declared core fields (its cores' write path) on every core;
    a field the geometry sets refuses by name (the registry's ``cores`` schema judges the rest below). [U-C4] The
    geometry counts cells: a declared ``cells_per_weight`` turns them into the cores' weight capacity
    (``weight_cells.weight_capacity``), and a count the cells cannot hold as whole weights refuses by name."""
    if not isinstance(fields, Mapping):
        raise _refuse(source, f"core_fields must be an object, got {type(fields).__name__}")
    if clash := sorted(f for f in fields if any(f in core for core in cores)):
        raise _refuse(source, f"core_fields {clash} are set by the geometry")
    if unknown := sorted(set(fields) - set(CORE_TYPE_SCHEMA.fields)):  # the cores schema is open; a bundle's is not
        raise _refuse(source, f"core_fields {unknown} are no core-type field of the registry")
    if errors := [m for key in (CELLS_PER_WEIGHT, WEIGHT_CELL_ROWS) if key in fields
                  for m in value_errors(CORE_TYPE_SCHEMA.fields[key], fields[key], f"core_fields.{key}")]:
        raise _refuse(source, "; ".join(errors))
    cores, contradiction = weight_capacity([{**core, **copy.deepcopy(dict(fields))} for core in cores])
    if contradiction is not None:
        raise _refuse(source, contradiction)
    return cores


def load_bundle(path: Path) -> TargetBundle:
    """One bundle file, judged: closed fields, a valid geometry, and every key legal for a bundle."""
    source = path.name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise _refuse(source, f"not JSON: {exc}") from None
    data = _closed(source, "the bundle", data, _BUNDLE_FIELDS, ("name", "geometry", "keys", "faithfulness"))
    if data["name"] != path.stem:
        raise _refuse(source, f"name {data['name']!r} differs from the file name {path.stem!r}")
    geometry, claim_class, values = _geometry(source, data["geometry"])
    values["cores"] = _with_core_fields(source, values["cores"], data.get("core_fields", {}))
    if not isinstance(data["keys"], Mapping):
        raise _refuse(source, f"keys must be an object, got {type(data['keys']).__name__}")
    for key, value in data["keys"].items():
        if errors := [f"{key} is set by the geometry"] if key in values else _bundle_key_errors(key, value):
            raise _refuse(source, "; ".join(errors))
        values[key] = copy.deepcopy(value)
    if errors := [m for key in ("cores", "weight_bits") for m in _bundle_key_errors(key, values[key])]:
        raise _refuse(source, "; ".join(errors))
    return TargetBundle(data["name"], geometry, claim_class, _faithfulness(source, data["faithfulness"]),
                        json.dumps(values))


_CACHE: Dict[Path, Mapping[str, TargetBundle]] = {}


def load_bundles(directory: Path) -> Mapping[str, TargetBundle]:
    """Every bundle in ``directory``, by name; a geometry bundled twice refuses by name. Cached per directory,
    as a read-only mapping of immutable bundles."""
    directory = Path(directory)
    if directory not in _CACHE:
        bundles: Dict[str, TargetBundle] = {}
        for path in sorted(directory.glob("*.json")):
            bundle = load_bundle(path)
            if owner := next((b.name for b in bundles.values() if b.geometry == bundle.geometry), None):
                raise _refuse(path.name, f"geometry {bundle.geometry!r} is already bundled by {owner!r}")
            bundles[bundle.name] = bundle
        _CACHE[directory] = MappingProxyType(bundles)
    return _CACHE[directory]


def target_bundles() -> Mapping[str, TargetBundle]:
    """The shipped bundles (``targets/`` beside this module), by name."""
    return load_bundles(Path(__file__).with_name(_TARGETS_DIRNAME))


def get_target_bundle(name: Any, bundles: Optional[Mapping[str, TargetBundle]] = None) -> TargetBundle:
    if name not in (known := target_bundles() if bundles is None else bundles):
        raise UnknownTargetError(f"target={name!r} names no target bundle; the known targets are {sorted(known)}")
    return known[name]


_BLOCKS: Dict[int, Tuple[Mapping[str, TargetBundle], Dict[str, Tuple[Tuple[str, int, int], ...]]]] = {}


def technology_blocks(bundles: Optional[Mapping[str, TargetBundle]] = None
                      ) -> Dict[str, Tuple[Tuple[str, int, int], ...]]:
    """[U-CR C-3] Per technology (the physics profile a chip description binds), the weight-cell block each chip
    description binding it states on its cores: ``(description, cells_per_weight, weight_cell_rows)``. A technology
    whose descriptions store one weight per cell is absent (``weight_cells.technology_block_errors`` reads it)."""
    known = target_bundles() if bundles is None else bundles
    if (cached := _BLOCKS.get(id(known))) is not None and cached[0] is known:
        return dict(cached[1])
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    for name, bundle in sorted(known.items()):
        values = bundle.values
        stated = {(int(c[CELLS_PER_WEIGHT]), int(c[WEIGHT_CELL_ROWS])) for c in values["cores"]
                  if int(c.get(CELLS_PER_WEIGHT, 1)) > 1}  # a count above 1 states its rows (the loader's rule)
        if (profile := values.get("platform_physics_profile")) is not None:
            out.setdefault(str(profile), []).extend((name, cells, rows) for cells, rows in sorted(stated))
    blocks = {profile: tuple(entries) for profile, entries in out.items() if entries}
    _BLOCKS[id(known)] = (known, blocks)
    return dict(blocks)


def target_decision_class(name: str) -> str:
    """The decision class of ``target: <name>``: the join over every key path the bundle sets."""
    return effective_class(declared_paths(get_target_bundle(name).values))


def _path_value(values: Mapping[str, Any], path: str) -> Any:
    key, *fields = path.split(".")
    value = values.get(key)
    for name in fields:
        value = (tuple(item.get(name) if isinstance(item, Mapping) else None for item in value)
                 if isinstance(value, list) else value.get(name) if isinstance(value, Mapping) else None)
    return value


def function_bearing_differences(first: TargetBundle, second: TargetBundle) -> Tuple[str, ...]:
    """The function-bearing key paths whose values differ between two bundles (absent meanings included)."""
    paths = set(declared_paths(first.values)) | set(declared_paths(second.values))
    return tuple(path for path in sorted(paths)
                 if class_of_path(REGISTRY, path) == FUNCTION_BEARING
                 and _path_value(first.values, path) != _path_value(second.values, path))


def target_departures(platform: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Each overridden bundle key of a resolved platform beside the bundle's value: key, bundle, declared, note."""
    if platform.get("target") is None:
        return ()
    values = get_target_bundle(platform["target"]).values  # a fresh copy: the rows share nothing with the cache
    return tuple({"key": key, "bundle": values.get(key), "declared": copy.deepcopy(entry["value"]),
                  "note": entry["note"]}
                 for key, entry in sorted((platform.get("target_overrides") or {}).items()))


def _same(first: Any, second: Any) -> bool:
    return json.dumps(first, sort_keys=True, default=repr) == json.dumps(second, sort_keys=True, default=repr)


def declared_departures(name: str, declared: Optional[Mapping[str, Any]] = None,
                        core_fields: Optional[Mapping[str, Any]] = None) -> Tuple[Dict[str, Any], ...]:
    """[U-C C-5] Each key a declaration beside ``target=<name>`` sets to another value than the bundle's, and each core
    field it sets to another value than the bundle's cores carry (``cores.<field>``): key, bundle, declared. A key the
    bundle does not set departs from nothing (the declaration completes the chip description)."""
    values = get_target_bundle(name).values
    out = [{"key": key, "bundle": values[key], "declared": copy.deepcopy(value)}
           for key, value in sorted((declared or {}).items()) if key in values and not _same(values[key], value)]
    cores = values.get("cores") or []
    for field, value in sorted((core_fields or {}).items()):
        stated = [core[field] for core in cores if field in core]
        if stated and not all(_same(v, value) for v in stated):
            out.append({"key": f"cores.{field}", "bundle": stated[0], "declared": copy.deepcopy(value)})
    return tuple(out)


def absolute_pricing_refusal(platform: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """``(name, reason)`` of the absolute-pricing refusal for a resolved platform whose target declares no
    physics (the pricer builds its own refusal record from the pair); None otherwise."""
    name = platform.get("target")
    if name is None or platform.get("platform_physics_resolved") is not None:
        return None
    return ("absolute_pricing", f"target={name!r} declares no physics profile ("
            f"{get_target_bundle(name).faithfulness.label()}): no area, energy or latency is priced")
