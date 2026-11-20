"""Target bundles: one name selects a published target's declared keys, claim class and faithfulness class.

A bundle is data, ``targets/<name>.json``: a geometry (a registered ``IMCPlatform`` by name, or declared with
its provenance and claim class), the core fields its cores take beside the geometry's (``core_fields``: the cores'
write path), registry keys judged by the value schema at load (the chip's permissions and input width among them),
and a faithfulness class (exact; approximate or refused, with what departs named), its computation statements and
produced realizations ([U-CHIPKEYS] ``descriptions``). This module loads and reads the bundles;
``target_expansion`` writes a declared target's keys at every config entry point. [U-CHIPKEYS3] A description
states its pricing (a profile, or ``pricing_absent`` with why, read at the run door: ``departures``). [U-CHIPKEYS4]
Where it states none, its label names the pricing a run resolves (``departures.run_faithfulness``), never its file.
[U-CHIPKEYS5/6] One grammar (``descriptions.label_clauses``) writes the label, typed texts as quoted fields.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.build import declared_paths, effective_class
from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
from mimarsinan.config_schema.registry.value_schema import entry_value_errors, value_errors
from mimarsinan.mapping.platform.descriptions import computation_keys, label_clauses, realizations
from mimarsinan.mapping.platform.descriptions.computation_keys import (  # moved: re-exported
    function_bearing_differences as function_bearing_differences)
from mimarsinan.mapping.platform.descriptions.departures import (  # moved: re-exported
    PricingAbsentError as PricingAbsentError, absolute_pricing_refusal as absolute_pricing_refusal,
    declared_departures as declared_departures, declared_pricing_refusal as declared_pricing_refusal,
    judged_faithfulness as judged_faithfulness, pricing_absence as pricing_absence,
    pricing_departures as pricing_departures, run_faithfulness as run_faithfulness, run_pricing as run_pricing,
    states_pricing as states_pricing, target_departures as target_departures)
from mimarsinan.mapping.platform.imc_platforms import IMCPlatform, get_imc_platform, imc_platform_names
from mimarsinan.mapping.platform.weight_cells import CELLS_PER_WEIGHT, WEIGHT_CELL_ROWS, weight_capacity

DP, PC = "deployment_parameters", "platform_constraints"
_TARGETS_DIRNAME = "targets"
_BUNDLE_FIELDS = frozenset({"name", "display_name", "geometry", "keys", "core_fields", "faithfulness", "notes",
                            "computation", "realizations", "pricing_absent"})
_GEOMETRY_FIELDS = frozenset({"name", "cores", "weight_bits", "provenance", "claim_class"})
_FAITHFULNESS_FIELDS = frozenset({"class", "named", "given", "note"})
FAITHFULNESS_KINDS = ("exact", "approximate", "refused")
#: Keys no bundle sets: the physics override block is composed whole from named blocks (merge_override_blocks).
UNBUNDLED_KEYS = ("platform_physics_overrides",)


class TargetBundleError(ToolchainRefusal, ValueError):
    """A bundle file the loader refuses: an unknown or undeclarable key, an illegal value, a reused name."""


class ChipKeyUnstatedError(TargetBundleError):
    """A bundle that neither states a key that changes the computation from its source nor names it unstated."""


class UnknownTargetError(ToolchainRefusal, ValueError):
    """``target`` names no bundle; the message lists the known names."""


@dataclass(frozen=True)
class Faithfulness:
    """How closely a declaration follows the published target: a kind, the named departures, and [U-CHIPKEYS2] the
    configuration the class rests on (``given``: the chip's settings a deployment holds for the class to hold)."""

    kind: str
    named: Tuple[str, ...] = ()
    given: Tuple[str, ...] = ()
    #: [U-CHIPKEYS4] The head of ``named`` that names the pricing a run resolves (a description stating none).
    pricing: Tuple[str, ...] = ()

    def with_declared(self, keys: Iterable[str]) -> "Faithfulness":
        """The class once ``keys`` are overridden: an exact bundle becomes approximate, each key named."""
        return self.with_departures(label_clauses.clause("declared", key=key) for key in sorted(keys))

    def with_departures(self, departures: Iterable[str]) -> "Faithfulness":
        """The class once a deployment departs from the bundle in ``departures``: an exact bundle becomes
        approximate, each departure named after the bundle's own; the configuration it rests on is kept."""
        if not (named := tuple(departures)):
            return self
        return replace(self, kind="approximate" if self.kind == "exact" else self.kind, named=self.named + named)

    def with_pricing(self, pricing: Iterable[str]) -> "Faithfulness":
        """[U-CHIPKEYS4] The class with ``pricing`` named in place of the pricing it names (first); a pricing named is
        a departure (an exact class becomes approximate), and the rest is kept."""
        pricing = tuple(pricing)
        kind = "approximate" if self.kind == "exact" and pricing else self.kind
        return replace(self, kind=kind, named=pricing + self.named[len(self.pricing):], pricing=pricing)

    def label(self) -> str:
        """``<kind>[, given "<setting>" and given "<setting>"][: <departure>; <departure>]`` ([U-CHIPKEYS6] each given
        setting a quoted field, as each typed departure is in ``named``)."""
        given = " and ".join(label_clauses.field(label_clauses.GIVEN, setting) for setting in self.given)
        return f"{self.kind}{', ' + given if given else ''}" + (f": {'; '.join(self.named)}" if self.named else "")


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
    computation_json: str = "{}"
    realizations_json: str = "{}"
    #: [U-CHIPKEYS3] The stated pricing absence (why no physics profile prices this description), "" where none.
    pricing_absent: str = ""

    @property
    def values(self) -> Dict[str, Any]:
        """``{key: value}`` of every key the bundle sets, in bundle order: a fresh copy at each read."""
        return json.loads(self.values_json)

    @property
    def statements(self) -> Dict[str, Dict[str, Any]]:
        """[U-CHIPKEYS] ``{key path: statement}``: each computation key stated from its source or named unstated."""
        return json.loads(self.computation_json)


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
    if REGISTRY.get(key) is not None and key in UNBUNDLED_KEYS:
        return [f"{key} is declared through its named override blocks, never by a bundle"]
    if (reason := computation_keys.unsettable_reason(key)) is not None:  # [U-CHIPKEYS] one rule, the sweep's too
        return [reason]
    return entry_value_errors(REGISTRY[key], value)


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
    if data["name"] != path.stem or not realizations.NAME.fullmatch(str(data["name"])):  # [U-CHIPKEYS3] one pattern
        raise _refuse(source, f"name {data['name']!r} differs from the file name {path.stem!r} or is not letters, "
                              "digits and underscores")
    if (shown := label_clauses.text_error("display_name", data.get("display_name"))) is not None:
        raise _refuse(source, shown)  # [U-CHIPKEYS5] printed beside the label: no clause the label derives
    geometry, claim_class, values = bundle_values(source, data)
    statements = judged_statements(source, data.get("computation"), values)
    absent = pricing_absence(source, data.get("pricing_absent"), values)  # [U-CHIPKEYS4] its pricing, then the label's
    faith = judged_faithfulness(source, data["faithfulness"], computation_keys.departures(statements),
                                pricing_departures(states_pricing(values, absent), run_pricing(values)))
    return TargetBundle(data["name"], geometry, claim_class, faith, json.dumps(values), json.dumps(statements),
                        json.dumps(realizations.judged(source, data, geometry, values)), absent)


def bundle_values(source: str, data: Mapping[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """``(geometry, claim class, values)`` of a bundle's data: its geometry's cores with the core fields and its keys,
    each judged; the computation statements are judged on these values."""
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
    return geometry, claim_class, values


def judged_statements(source: str, block: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    """[U-CHIPKEYS] The one loader rule (``computation_keys``): a key with no statement refuses by name."""
    unstated, errors, statements = computation_keys.statement_errors(block, values)
    if unstated or errors:
        raise (ChipKeyUnstatedError if unstated else TargetBundleError)(f"target bundle {source}: " + (
            f"computation keys neither stated nor named unstated with a note: {'; '.join(unstated)}" if unstated
            else "; ".join(errors)))
    return statements


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
    """The bundle ``name`` selects: a bundle, or a declared realization of one (``<bundle>@<realization>``)."""
    if (found := realizations.lookup(known := target_bundles() if bundles is None else bundles, name)) is None:
        raise UnknownTargetError(f"target={name!r} names no target bundle; the known targets are {target_names(known)}")
    return found


def target_names(bundles: Optional[Mapping[str, TargetBundle]] = None) -> List[str]:
    """Every name a ``target`` takes: each bundle's, then each declared realization's (``<bundle>@<name>``)."""
    return realizations.names(target_bundles() if bundles is None else bundles)


_BLOCKS: Dict[int, Tuple[Mapping[str, TargetBundle], Dict[str, Tuple[Tuple[str, int, int], ...]]]] = {}


def technology_blocks(bundles: Optional[Mapping[str, TargetBundle]] = None
                      ) -> Dict[str, Tuple[Tuple[str, int, int], ...]]:
    """[U-CR C-3] Per technology (the physics profile a chip description binds), the weight-cell block each chip
    description binding it states on its cores: ``(description, cells_per_weight, weight_cell_rows)``, every target
    name read through the one lookup ([U-CHIPKEYS2]: a realization binds its own profile or none). A technology whose
    descriptions store one weight per cell is absent (``weight_cells.technology_block_errors`` reads it)."""
    known = target_bundles() if bundles is None else bundles
    if (cached := _BLOCKS.get(id(known))) is not None and cached[0] is known:
        return dict(cached[1])
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    for name in target_names(known):  # [U-CHIPKEYS2] every name a target takes, read through the one lookup
        values = get_target_bundle(name, known).values
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
