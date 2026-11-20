"""How a deployment departs from the chip description it names: overridden keys, declarations beside the target and
the absolute-pricing refusal of a description without physics. Moved unchanged from ``target_bundles`` ([U-CHIPKEYS]:
that module's line budget), which re-exports every name here; each reads the description through the one lookup
(``target_bundles.get_target_bundle``), so a realization's departures are read like its bundle's.

[U-CHIPKEYS3] M-2: a stated pricing absence is read at the run door. A pricing a run declares on a target whose
description or realization states pricing absent (a physics profile or constants beside the target or in its
overrides, or a core type's technology among the cores it takes) refuses by name at the expansion
(``declared_pricing_refusal``), and the pricer's entry refuses absolute pricing by the statement whatever physics a
platform carries (``absolute_pricing_refusal``), so no label is printed for a pricing the statement excludes.

[U-CHIPKEYS4] 3(f): the label is derived from the pricing the run resolves, one source. A description states its
pricing (a physics profile among its keys, or its absence with why) or states none; the faithfulness it states never
names its pricing (``judged_faithfulness``, moved here from ``target_bundles``: the module's line budget). Where it
states none, the label names the pricing its run resolves, read as the resolver reads it (``run_pricing``: the
resolver's ``priced_under`` over its ``physics_declarations``, or a resolved platform's physics), first among the
departures: "no physics profile" where none resolves, the resolved physics by name where a run declares one
(``pricing_departures``). Every label site takes it from ``run_faithfulness``: the run door's label, the
absolute-pricing refusal, the scaling ladder's target spec and the design point instance's chip block.

[U-CHIPKEYS5] M-1 as its class: the pricing clauses are written from the label grammar's templates
(``label_clauses.clause``), and a text the label prints in or beside it (a named departure, a given setting, the note)
holding any clause the label derives, read back by the same templates, refuses by name (``judged_faithfulness``); m-3:
the absolute-pricing refusal and the label it prints read one platform (its resolved physics, absent read as none).

[U-CHIPKEYS6] The class closed by construction (``label_clauses``): every typed text the label prints in or beside it (a
named departure, a given setting, the note, the profile a description names) is judged by its words at load, and the
label prints a named departure as a quoted field (``named "<text>"``), never where a derived clause stands.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
# Bound as a module (resolved from sys.modules while it initializes): target_bundles imports this module first.
import mimarsinan.mapping.platform.target_bundles as bundles
# [U-CHIPKEYS4] Bound as a module as well: the resolver's package reaches this module's loader while it
# initializes, so its names are read at call time (the label's and the resolver's one reading of the pricing).
import mimarsinan.deployment_record.platform_physics.resolve as resolver
from mimarsinan.mapping.platform.descriptions import label_clauses, realizations
from mimarsinan.mapping.platform.weight_cells import TECHNOLOGY_FIELDS

#: [U-CHIPKEYS4] The departure a description that states no pricing names where its run resolves none.
NO_PHYSICS = label_clauses.clause("no_pricing")
#: The resolved physics a resolved platform carries (``build_platform_constraints_resolved``).
RESOLVED = "platform_physics_resolved"


class PricingAbsentError(ToolchainRefusal, ValueError):
    """[U-CHIPKEYS3] A pricing declared on a target whose description or realization states pricing absent."""


def target_departures(platform: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Each overridden bundle key of a resolved platform beside the bundle's value: key, bundle, declared, note."""
    if platform.get("target") is None:
        return ()
    values = bundles.get_target_bundle(platform["target"]).values  # a fresh copy: the rows share nothing with the cache
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
    values = bundles.get_target_bundle(name).values
    out = [{"key": key, "bundle": values[key], "declared": copy.deepcopy(value)}
           for key, value in sorted((declared or {}).items()) if key in values and not _same(values[key], value)]
    cores = values.get("cores") or []
    for field, value in sorted((core_fields or {}).items()):
        stated = [core[field] for core in cores if field in core]
        if stated and not all(_same(v, value) for v in stated):
            out.append({"key": f"cores.{field}", "bundle": stated[0], "declared": copy.deepcopy(value)})
    return tuple(out)


def pricing_declarations(values: Mapping[str, Any], prefix: str = "") -> Tuple[str, ...]:
    """[U-CHIPKEYS3] Each pricing ``values`` declare: a pricing key with a value (the resolver's ``PHYSICS_KEYS``), and
    each core type's named technology (``weight_cells.TECHNOLOGY_FIELDS``), as ``<prefix><key>=<value>``."""
    out: List[str] = [f"{prefix}{key}={values[key]!r}" for key in resolver.PHYSICS_KEYS if values.get(key)]
    cores = values.get("cores")
    for index, core in enumerate(cores if isinstance(cores, list) else ()):
        out += [f"{prefix}cores[{index}].{field}={core[field]!r}" for field in TECHNOLOGY_FIELDS
                if isinstance(core, Mapping) and core.get(field)]
    return tuple(out)


def pricing_absence(source: str, absent: Any, values: Mapping[str, Any]) -> str:
    """[U-CHIPKEYS3] A description's stated pricing absence ("" where it states none): a non-empty reason, and then no
    pricing among its values (``pricing_declarations``: a profile, constants, a core type's technology). [U-CHIPKEYS6]
    The profile it names is a typed text the label prints (a realization's pricing clause, the chip block)."""
    if (shown := label_clauses.text_error(f"keys.{realizations.PRICING_KEY}", values.get(realizations.PRICING_KEY))):
        raise bundles._refuse(source, shown)
    if absent is None:
        return ""
    if not isinstance(absent, str) or not absent.strip():
        raise bundles._refuse(source, f"pricing_absent {absent!r}: a stated pricing absence says why")
    if declared := pricing_declarations(values):
        raise bundles._refuse(source, f"pricing_absent is stated and the description declares {', '.join(declared)}: "
                                      "it states its pricing once, a physics profile or its absence")
    return absent


def declared_pricing_refusal(name: str, absent: str, declared: Mapping[str, Any],
                             overrides: Mapping[str, Any]) -> Optional[str]:
    """[U-CHIPKEYS3] Why a run may not declare its pricing on ``target=<name>`` (None when it may): the description
    states pricing ``absent``, and the run declares a pricing beside the target (``declared``, the flat sections:
    a pricing key, or a technology among the cores it declares) or in ``target_overrides`` (each entry's value)."""
    if not absent:
        return None
    found = pricing_declarations({**{key: declared.get(key) for key in resolver.PHYSICS_KEYS},
                                  "cores": declared.get("cores")})
    found += pricing_declarations({key: entry.get("value") for key, entry in overrides.items()
                                   if isinstance(entry, Mapping)}, "target_overrides.")
    if not found:
        return None
    return (f"target={name!r} states pricing absent ({absent}), and the run declares {', '.join(found)}: drop the "
            "pricing, or name a target whose description states its own")


def _stated_absence(name: Any) -> str:
    """The pricing absence the description ``name`` selects states ("" for none, or for a name it does not select)."""
    found = realizations.lookup(bundles.target_bundles(), name)
    return "" if found is None else found.pricing_absent


def absolute_pricing_refusal(platform: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """``(name, reason)`` of the absolute-pricing refusal for a resolved platform whose target declares no
    physics (the pricer builds its own refusal record from the pair); None otherwise. [U-CHIPKEYS3] A target whose
    description states pricing absent refuses by that statement whatever physics the platform carries. [U-CHIPKEYS4]
    The reason prints the run's label (``run_faithfulness`` over the platform)."""
    name = platform.get("target")
    if name is None:
        return None
    if absent := _stated_absence(name):
        return ("absolute_pricing", f"target={name!r} states pricing absent ({absent}): no area, energy or latency "
                                    "is priced")
    if (resolved := platform.get(RESOLVED)) is not None:
        return None
    return ("absolute_pricing", f"target={name!r} declares no physics profile ("  # [U-CHIPKEYS5] m-3: its platform's
            f"{run_faithfulness(bundles.get_target_bundle(name), {RESOLVED: resolved}).label()}): no area, energy or "
            "latency is priced")


# --- [U-CHIPKEYS4] 3(f): the pricing in the label, derived from the pricing the run resolves -----------------------------

def states_pricing(values: Mapping[str, Any], absent: Any) -> bool:
    """Whether a description states its pricing: a physics profile among its keys, or its absence with why."""
    return bool(values.get(realizations.PRICING_KEY)) or bool(absent)


def pricing_departures(stated: bool, priced: Optional[str]) -> Tuple[str, ...]:
    """The pricing a label names for a description: nothing where it states its pricing (its own profile, or its
    absence, which the run door enforces), else the physics its run resolves: ``NO_PHYSICS`` for none, the physics by
    name for one (a profile, or the resolver's custom profile for declared constants alone)."""
    if stated:
        return ()
    return (NO_PHYSICS,) if priced is None else (label_clauses.clause("pricing", name=priced),)


def run_pricing(config: Mapping[str, Any]) -> Optional[str]:
    """The physics a run is priced under, as the run door resolves it: a resolved platform's physics by name (None
    where none resolved), else the resolver's own reading of the declarations (``priced_under`` over
    ``physics_declarations``), which ``resolve_platform_physics`` prices by."""
    if RESOLVED in config:
        return None if config[RESOLVED] is None else str(config[RESOLVED]["name"])
    return resolver.priced_under(*resolver.physics_declarations(config))


def run_faithfulness(bundle: Any, config: Mapping[str, Any]) -> Any:
    """The faithfulness a run on ``bundle`` prints: the description's own (its stated class, its statements' departures)
    with the pricing ``config``'s run resolves (the declarations, or a resolved platform) in place of the pricing the
    description resolves alone. The one source of every label site."""
    return bundle.faithfulness.with_pricing(pricing_departures(states_pricing(bundle.values, bundle.pricing_absent),
                                                               run_pricing(config)))


def judged_faithfulness(source: str, spec: Any, departures: Tuple[str, ...] = (),
                        pricing: Tuple[str, ...] = ()) -> Any:
    """The faithfulness a description states (moved from ``target_bundles``), with the pricing it resolves alone
    (``pricing_departures``) named first and the departures its statements state named after its own
    (``computation_keys.departures``): an exact class names nothing of its own, so a departing statement refuses it;
    an approximate or refused class names what departs, its pricing included. [U-CHIPKEYS5] A typed clause (a named
    departure, a given setting) holds no label separator, and no text the label prints in or beside it (those and the
    note) holds a clause the label derives (``label_clauses.text_error``, the grammar read back)."""
    data = bundles._closed(source, "faithfulness", spec, bundles._FAITHFULNESS_FIELDS, ("class",))
    typed = {label_clauses.NAMED: data.get("named", []), label_clauses.GIVEN: data.get("given", [])}
    listed = {field: isinstance(texts, list) and all(isinstance(t, str) and t.strip() for t in texts)
              for field, texts in typed.items()}
    # [U-CHIPKEYS6] each typed text judged by its words first (the class's refusal names the clause it holds)
    if errors := [error for field, texts in (*typed.items(), ("note", [data.get("note")])) if listed.get(field, True)
                  for text in texts if (error := label_clauses.text_error(field, text))]:
        raise bundles._refuse(source, "faithfulness " + "; ".join(errors))
    for field, what in (("named", "what departs"), ("given", "the settings the class rests on")):
        if not listed[field] or any(set(t) & label_clauses.SEPARATORS for t in typed[field]):
            raise bundles._refuse(source, f"faithfulness {field} {typed[field]!r}: a list of {what}, each named "
                                          "without ':' or ';' (the label's separators)")
    kind, named = data["class"], tuple(typed["named"]) + tuple(departures)  # the note is read by people only
    if kind not in bundles.FAITHFULNESS_KINDS or (bool(named) if kind == "exact" else not (named or pricing)):
        raise bundles._refuse(source, f"faithfulness {kind!r} {list(named)}: one of {bundles.FAITHFULNESS_KINDS}, "
                                      "exact with nothing named, approximate or refused with what departs named (its "
                                      "statements' and its pricing's too)")
    printed = tuple(label_clauses.field(label_clauses.NAMED, t) for t in typed["named"]) + tuple(departures)
    return bundles.Faithfulness(kind, printed, tuple(typed["given"])).with_pricing(pricing)
