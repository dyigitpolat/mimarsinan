"""[U-CHIPKEYS, ruling D-5] A chip description's produced realizations: each a declared geometry of the bundle (the
cores and weight grid the produced design holds), the keys the design changes, the statements of every computation
key it changes (from the record that produced it), and that record. A realization is addressed as
``<bundle>@<realization>``; it inherits every other key, statement and the faithfulness of its bundle, and its own
values are judged by the same loader rule as the bundle's.

[U-CHIPKEYS2] A realization states its pricing: its own physics profile (``keys.platform_physics_profile``), or
``pricing_absent`` with why, so a run on it refuses absolute pricing by name; it never inherits its bundle's profile,
which prices another technology (the bundle's chip as built).

[U-CHIPKEYS3] The lookup carries the stated absence to the run door (``departures``); a realization that states its
bundle's profile as its own refuses at load, as does an absence beside a pricing among its values (a core type's
technology inherited from its bundle's core fields); its faithfulness is its bundle's stated class with the
departures of its own statements.

[U-CHIPKEYS4] 3(e): the record that produced a realization is judged by form at load, each field (``RECORD_FORMS``):
the report's path in its repository, its 64-hex sha256, the design, session and result as non-empty texts. What cannot
be judged at load is carried, not judged: that the report at that path holds that digest (the report lives in the
research repository, which no load reads) and what the texts say; no run door reads the record.

[U-CHIPKEYS5] m-2: where its bundle states no pricing, a realization's label names the pricing the realization states
(its own profile, or none for a stated absence: the label grammar's pricing clause), so a class that rests on its
bundle's derived pricing alone rests on the realization's own statement and loads; its display name holds no clause
the label derives (``label_clauses.text_error``).
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Mapping, Optional

# Bound as a module (resolved from sys.modules while it initializes): target_bundles imports this module first, and
# its judges (geometry, core fields, keys, statements) are read at call time.
import mimarsinan.mapping.platform.target_bundles as bundles
from mimarsinan.mapping.platform.descriptions import computation_keys, label_clauses

SEPARATOR = "@"
#: [U-CHIPKEYS3] The name of a bundle and of a realization: letters, digits and underscores (never the separator).
NAME = re.compile(r"[A-Za-z0-9_]+")
#: The key that names the technology a description is priced under (a physics profile).
PRICING_KEY = "platform_physics_profile"
_FIELDS = frozenset({"display_name", "geometry", "keys", "computation", "record", "notes", "pricing_absent"})
_REQUIRED = ("geometry", "computation", "record")
#: The record that produced a realization: its report (path and sha256), the design it ran and what it read.
_RECORD_FIELDS = frozenset({"report", "sha256", "design", "session", "result"})
#: [U-CHIPKEYS4] A report's path in its repository: names of letters, digits, '_', '.' and '-' joined by '/'.
_PATH = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")
_DIGEST = re.compile(r"[0-9a-f]{64}")
#: [U-CHIPKEYS4] 3(e) Each record field's form (what it is, the judge of a value), judged at load where present.
RECORD_FORMS = {
    "report": ("the report's path in its repository (names of letters, digits, '_', '.' and '-' joined by '/', none of "
               "them '.' or '..')", lambda v: isinstance(v, str) and _PATH.fullmatch(v) is not None
               and not set(v.split("/")) & {".", ".."}),
    "sha256": ("the report's 64-hex sha256", lambda v: isinstance(v, str) and _DIGEST.fullmatch(v) is not None),
    "design": ("a non-empty text", lambda v: isinstance(v, str) and bool(v.strip())),
    "session": ("a non-empty text", lambda v: isinstance(v, str) and bool(v.strip())),
    "result": ("a non-empty text", lambda v: isinstance(v, str) and bool(v.strip())),
}
_CACHE: Dict[tuple, Any] = {}


def judged(source: str, data: Mapping[str, Any], geometry: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    """``{name: {display_name, geometry, claim_class, values, statements, record}}`` of a bundle file's declared
    realizations, each judged: closed fields, a declared geometry of its own, legal keys and the loader rule."""
    specs = data.get("realizations") or {}
    if not isinstance(specs, Mapping):
        raise bundles._refuse(source, f"realizations must be an object, got {type(specs).__name__}")
    out = {}
    for name, spec in specs.items():
        where = f"{source} realization {name!r}"
        if not NAME.fullmatch(str(name)):
            raise bundles._refuse(where, "a realization name is letters, digits and underscores")
        spec = bundles._closed(where, "the realization", spec, _FIELDS, _REQUIRED)
        if (shown := label_clauses.text_error("display_name", spec.get("display_name"))) is not None:
            raise bundles._refuse(where, shown)  # [U-CHIPKEYS5] printed beside the label: no derived clause
        if not isinstance(spec["geometry"], Mapping):
            raise bundles._refuse(where, "a realization declares its geometry (cores, weight_bits, provenance, "
                                         "claim_class), never a registered one")
        own, claim_class, geometry_values = bundles._geometry(where, spec["geometry"])
        if own == geometry:
            raise bundles._refuse(where, f"geometry {own!r} is the bundle's own: a realization declares its own")
        record = bundles._closed(where, "record", spec["record"], _RECORD_FIELDS, ("report", "sha256", "result"))
        if errors := [f"record.{field} is {form}, got {record[field]!r}" for field, (form, judge) in RECORD_FORMS.items()
                      if field in record and not judge(record[field])]:  # [U-CHIPKEYS4] 3(e) each field by its form
            raise bundles._refuse(where, "; ".join(errors))
        own_keys = spec.get("keys") or {}
        absent = spec.get("pricing_absent")
        stated_absent = isinstance(absent, str) and bool(absent.strip())
        if not ((PRICING_KEY in own_keys and absent is None) or (PRICING_KEY not in own_keys and stated_absent)):
            raise bundles._refuse(where, f"a realization states its pricing: its own physics profile (keys."
                                         f"{PRICING_KEY}) or pricing_absent with why, one of them, never its "
                                         "bundle's profile by inheritance")
        if PRICING_KEY in own_keys and own_keys[PRICING_KEY] == values.get(PRICING_KEY):
            raise bundles._refuse(where, f"keys.{PRICING_KEY} {own_keys[PRICING_KEY]!r} is its bundle's profile, "
                                         "which prices the bundle's chip as built: a realization states its own "
                                         "technology's profile or pricing_absent with why")
        merged = {k: v for k, v in copy.deepcopy(dict(values)).items() if k != PRICING_KEY}
        merged["cores"] = bundles._with_core_fields(where, geometry_values["cores"], data.get("core_fields", {}))
        merged["weight_bits"] = geometry_values["weight_bits"]
        for key, value in own_keys.items():
            if errors := [f"{key} is set by the geometry"] if key in ("cores", "weight_bits") else \
                    bundles._bundle_key_errors(key, value):
                raise bundles._refuse(where, "; ".join(errors))
            merged[key] = copy.deepcopy(value)
        block = {**(data.get("computation") or {}), **(spec.get("computation") or {})}
        statements = bundles.judged_statements(where, block, merged)
        # a realization states its pricing (above), so its label names no pricing of a run's ([U-CHIPKEYS4] 3(f));
        # [U-CHIPKEYS5] m-2: on a bundle stating none, it names the pricing the realization states, written once here
        faith = bundles.judged_faithfulness(where, data.get("faithfulness"), computation_keys.departures(statements),
                                            bundles.pricing_departures(bundles.states_pricing(
                                                values, data.get("pricing_absent")), bundles.run_pricing(merged)))
        out[str(name)] = {"display_name": spec.get("display_name"), "geometry": own, "claim_class": claim_class,
                          "values": merged, "statements": statements, "record": dict(record),
                          "pricing_absent": bundles.pricing_absence(where, absent, merged),
                          "faithfulness": {"kind": faith.kind, "named": list(faith.named), "given": list(faith.given)}}
    return out


def names(known: Mapping[str, Any]) -> List[str]:
    """Every name a target takes in ``known``: each bundle's, then each declared realization's."""
    return sorted(known) + sorted(f"{name}{SEPARATOR}{real}" for name, bundle in known.items()
                                  for real in json.loads(getattr(bundle, "realizations_json", "{}") or "{}"))


def lookup(known: Mapping[str, Any], name: Any) -> Optional[Any]:
    """The bundle ``name`` selects in ``known``: a bundle by its name, or ``<bundle>@<realization>`` (the
    realization's geometry, claim class, values and statements, its bundle's faithfulness); None for neither."""
    if not isinstance(name, str):
        return None
    if name in known or name.count(SEPARATOR) != 1:
        return known.get(name)
    base, real = name.split(SEPARATOR)
    bundle = known.get(base)
    if bundle is None:
        return None
    key = (id(known), name, bundle.realizations_json)
    if key not in _CACHE:
        found = json.loads(bundle.realizations_json or "{}").get(real)
        _CACHE[key] = None if found is None else type(bundle)(
            name=name, geometry=found["geometry"], claim_class=found["claim_class"],
            faithfulness=bundles.Faithfulness(found["faithfulness"]["kind"], tuple(found["faithfulness"]["named"]),
                                              tuple(found["faithfulness"]["given"])),
            values_json=json.dumps(found["values"]), computation_json=json.dumps(found["statements"]),
            pricing_absent=found["pricing_absent"])
    return _CACHE[key]
