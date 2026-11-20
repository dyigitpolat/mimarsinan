"""[U-HOST2] The one admission rule for a row that names a host: it is judged on its value, never on its label.

[U-HOST] A host row prices the deployment host, a machine the toolchain does not run on, from its published data (the
owner's rule: a Jetson-class device's data, never the development machine's or a GPU's). [U-HOST2] So a row naming its
host (``HOST_IDENTITY_FIELD``: ``'<description> <mode>'``) is admitted only as
  (a) the named mode's own row, as the description resolves it: its band at each corner, its unit and its evidence
      class equal, and its energy plane and link ends the description's statement for the row; or
  (b) a derivation of the mode's rows declared in the one form (``hosts.derivation``), which this rule recomputes from
      the description and compares (the band in SI, the unit, the class ``derived``, the plane and ends of the row;
      [U-HOST3] a derivation text that contradicts the declaration refused);
else it is refused by name. A HOST-group row naming no host, a host no read description declares (``hosts.registry``:
the shipped descriptions and any registered through its contract), a row the named mode does not declare (with the
description's own reason where it states one) and rows of two hosts in one physics are refused by name. The rule is
judged at the loader's three entries (``loader.profile_from_dict``: a profile's own rows; ``loader.apply_overrides``:
a run's declared rows, each before any composition; ``PlatformPhysics.from_dict``: a sealed physics read back), so no
other value reaches a pricer under a host's name. A development machine's calibration is a labelled diagnostic.
[U-HOST3] Every judgment reads the description held under a digest (``hosts.registry.held_description``): a declared
row's is the digest of the description readable by its name now, a sealed row's is its physics' sealed origin
(``host_origin``), never the description that holds the name now; one physics rests on one description (one origin).
[U-HOST4] A description is identified by its digest, never by its path (``hosts.registry.HostOrigin``'s identity):
rows judged against one digest read at two paths rest on one description.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from mimarsinan.deployment_record.platform_physics.constants import constant_from_declaration, spec_for
from mimarsinan.deployment_record.platform_physics.hosts.derivation import (
    DERIVATION_FIELD, HostDerivationError, combined, judged, parsed, same_band, text_refusal,
)
from mimarsinan.deployment_record.platform_physics.hosts.description import (
    ADMITTED_CLASSES as ADMITTED_CLASSES, ENERGY_PLANE_FIELD, HOST_IDENTITY_FIELD, SYSTEM, HostDescriptionError,
    HostRowAdmissionError as HostRowAdmissionError, host_description, host_descriptions, judged_description,
    select_mode,
)
from mimarsinan.deployment_record.platform_physics.hosts.link_ends import LINK_ENDS_FIELD, judged_link_ends
from mimarsinan.deployment_record.platform_physics.hosts.registry import (
    HostOrigin, HostOriginError, HostRegistryError, held_description, read_description,
    register_host_description as _register, unregister_host_description as unregister_host_description,
)
from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST
from mimarsinan.deployment_record.units import to_canonical

Band3 = Tuple[float, float, float]
#: [U-HOST3] The sealed physics' field naming the description its host rows were judged against.
ORIGIN_FIELD = "host_origin"


@dataclass(frozen=True)
class Described:
    """What one host description resolves for one of its modes: each row it declares there, the energy plane and the
    link ends it states per row, and its reasons for the rows the mode refuses; [U-HOST3] and its origin."""

    identity: str
    rows: Mapping[str, Any]
    planes: Mapping[str, str]
    ends: Mapping[str, Mapping[str, Tuple[str, ...]]]
    refused: Mapping[str, str]
    #: Each row's band in SI (low, nominal, high), read once.
    si: Mapping[str, Tuple[float, float, float]]
    origin: HostOrigin


_CACHE: Dict[Tuple[Any, ...], Described] = {}


def register_host_description(path: Path) -> str:
    """Register a host description (``<name>.json`` beside its ``.md``) through the registry's one contract: judged
    whole (``host_description``) before it is read; its name is returned."""
    return _register(path, host_description)


def read_hosts() -> Dict[str, Tuple[str, ...]]:
    """Every read description and its modes (for a refusal's text; one no longer readable says so)."""
    hosts: Dict[str, Tuple[str, ...]] = {}
    for name in host_descriptions():
        try:
            hosts[name] = tuple(sorted(host_description(name).modes))
        except (HostDescriptionError, HostRegistryError) as why:
            hosts[name] = (f"unreadable: {why}",)
    return hosts


def described(identity: str, origin: Optional[HostOrigin] = None) -> Tuple[Optional[Described], Optional[str]]:
    """(what ``identity`` resolves to, None) or (None, why it does not resolve). [U-HOST3] Judged against the
    description held under ``origin`` (a sealed physics'), or, without one, under the digest of the description
    readable by its name now."""
    name, _, mode = str(identity).partition(" ")
    try:
        if origin is None:
            found = read_description(name)
            origin = None if found is None else found[2]
        if origin is None:  # [U-HOST3] an identity without a mode names no mode: refused below, by name
            return None, f"names host {identity!r}, which no read host description declares (read: {read_hosts()})"
        if origin.name != name:
            return None, f"names host {identity!r}, and its physics' {ORIGIN_FIELD} is the {origin.label}"
        raw = held_description(origin)  # held by digest, else refused by name
    except (HostOriginError, HostRegistryError) as why:
        return None, f"names host {identity!r}: {why}"
    key = (origin, mode)  # [U-HOST3] keyed by the digest: a cached judgment is always of the same bytes ([U-HOST4]
    if key not in _CACHE:  # the origin's identity, kind, name and digest: one judgment per description, any path)
        try:
            description = judged_description(name, raw, origin.label, None)
        except (HostDescriptionError, ValueError) as why:  # a held file that is no description refuses by name
            return None, f"names host {identity!r}: {why}"
        if mode not in description.modes:
            return None, (f"names host {identity!r}: {name!r} declares no mode {mode!r} (its modes: "
                          f"{sorted(description.modes)})")
        facts = description.modes[mode]
        raw_rows = {**description.constants, **(facts.get("constants") or {})}
        where = f"host description {name!r}"
        rows = {k: constant_from_declaration(k, row) for k, row in raw_rows.items()}
        _CACHE[key] = Described(
            identity=f"{name} {mode}", rows=rows, planes=dict(description.energy_attribution),
            ends={k: judged_link_ends(k, v, where) for k, v in description.link_ends.items()},
            refused={str(k): str(v) for k, v in (facts.get("refused") or {}).items()},
            si={k: _si(v) for k, v in rows.items()}, origin=origin)
    found = _CACHE[key]  # [U-HOST4] its origin states where this read found the description (the path is text)
    return (found if found.origin.path == origin.path else replace(found, origin=origin)), None


def _si(value: Any) -> Band3:
    band = value.band
    return band.low, band.nominal, band.high


def _undeclared(rows: Iterable[str], d: Described) -> str:
    rows = sorted(rows)
    reasons = "; ".join(f"{row}: {d.refused[row]}" for row in rows if row in d.refused)
    return (f"host {d.identity!r} declares no {rows} in its mode" + (f" (the description: {reasons})" if reasons else "")
            + "; a row the description does not declare is no row of that host")


def _expected(key: str, value: Any, d: Described) -> Tuple[Optional[Tuple[Band3, str, str, str]], Optional[str]]:
    """((the band in SI, unit, class the rule expects, what it is), None) or (None, why it cannot be stated)."""
    if not value.derived_from:
        if key not in d.rows:
            return None, _undeclared([key], d)
        row = d.rows[key]
        return (d.si[key], row.unit, row.evidence_kind, f"host {d.identity!r}'s own {key}"), None
    try:
        terms, combine = parsed(key, value.derived_from)
    except HostDerivationError as why:
        return None, str(why)
    names = [row for row, _factor in terms]
    if spec_for(key).group == HOST and key not in names:
        return None, (f"a host row's derivation is a derivation of that row: {DERIVATION_FIELD} {value.derived_from} "
                      f"does not name {key}")
    if missing := [row for row in names if row not in d.rows]:
        return None, _undeclared(missing, d)
    if why := text_refusal(key, value.derivation, terms):  # [U-HOST3] the words never contradict the declaration
        return None, why
    band = combined(terms, combine, {row: d.si[row] for row in names})
    unit = d.rows[key].unit if key in d.rows else d.rows[names[0]].unit
    return (band, unit, "derived", f"the declared derivation {value.derived_from} of host {d.identity!r}"), None


def value_refusal(key: str, value: Any, system: bool = False,
                  ends: Optional[Mapping[str, Tuple[str, ...]]] = None,
                  origin: Optional[HostOrigin] = None) -> Optional[str]:
    """Why the row ``key`` (a ``PhysicsConstantValue``; ``system``: the physics states it system-level; ``ends``: the
    link ends the physics states for it; [U-HOST3] ``origin``: the description it was sealed under, None for the one
    readable now) is not admitted, or None: not a host row and naming no host, or admitted."""
    if not value.host:
        if spec_for(key).group == HOST:
            return f"{key} names no host (an identified host description's block names it: '<description> <mode>')"
        return f"{key}: {DERIVATION_FIELD} derives from a host's rows and the row names no host" \
            if value.derived_from else None
    d, why = described(value.host, origin)
    if d is None:
        return f"{key} {why}"
    expected, why = _expected(key, value, d)
    if expected is None:
        return f"{key} of host {value.host!r}: {why}"
    band, unit, klass, what = expected
    found = []
    if not same_band(_si(value), band):
        found.append(f"its band ({value.low:.6g}, {value.nominal:.6g}, {value.high:.6g}) {value.unit} is not {what} "
                     f"({', '.join(f'{v:.6g}' for v in band)} in SI)")
    if value.unit != unit:
        found.append(f"its unit {value.unit!r} is not {unit!r}")
    if value.evidence_kind != klass:
        found.append(f"its class {value.evidence_kind!r} is not {klass!r} (a host row is never a measurement of the "
                     "machine running the toolchain)")
    if system != (d.planes.get(key) == SYSTEM):
        found.append(f"its {ENERGY_PLANE_FIELD} is {'system' if system else 'attributable or none'}, the description "
                     f"states {d.planes.get(key)!r}")
    if (dict(ends) if ends else None) != (dict(d.ends[key]) if key in d.ends else None):
        found.append(f"its {LINK_ENDS_FIELD} {dict(ends) if ends else None} are not the description's "
                     f"{dict(d.ends[key]) if key in d.ends else None}")
    if not found:
        return None
    return (f"{key} of host {value.host!r}: " + "; ".join(found) + " (a row naming a host is its description's own row "
            "or a declared derivation of it, never another value under its name)")


def admit_host_rows(constants: Mapping[str, Any], where: str, system_rows: Iterable[str] = (),
                    link_ends: Optional[Mapping[str, Mapping[str, Tuple[str, ...]]]] = None,
                    also_named: Iterable[str] = (), judged: Iterable[str] = (),
                    origin: Optional[HostOrigin] = None) -> Optional[HostOrigin]:
    """Refuse by name every row of ``constants`` the rule does not admit (with the physics' system-level rows and link
    ends), and rows naming two hosts (``also_named``: hosts named by declared rows composed into chip rows;
    ``judged``: rows the caller judged as declared, counted for their host and not judged again). [U-HOST3] Rows not
    judged as declared are judged against ``origin`` (the description they were sealed under; None: the one readable
    now); rows judged against two descriptions refuse by name. Returns the origin the rows naming a host rest on."""
    system, ends, done = set(system_rows), dict(link_ends or {}), set(judged)
    why: Dict[str, str] = {}
    hosts = set(also_named)
    origins = {_origin_of(host, None) for host in hosts}
    resting = set()
    for key, value in constants.items():
        if key not in done and (reason := value_refusal(key, value, key in system, ends.get(key), origin)) is not None:
            why[key] = reason
        elif value.host:
            hosts.add(value.host)
            resting.add(_origin_of(value.host, None if key in done else origin))
    refuse_named(why, where, hosts)
    if len(hosts) > 1:
        raise HostRowAdmissionError(f"{where}: one physics has one host: its rows name {sorted(hosts)}")
    if len(origins | resting) > 1:
        raise HostRowAdmissionError(f"{where}: one physics rests on one description: its host rows were judged "
                                    f"against {sorted(o.label for o in origins | resting)}")
    return next(iter(resting), None)


def _origin_of(identity: str, origin: Optional[HostOrigin]) -> HostOrigin:
    """The origin an admitted row naming ``identity`` was judged against."""
    d, why = described(identity, origin)
    if d is None:
        raise HostRowAdmissionError(f"host {identity!r}: {why}")
    return d.origin


def admit_sealed(constants: Mapping[str, Any], where: str, system_rows: Iterable[str],
                 link_ends: Mapping[str, Mapping[str, Tuple[str, ...]]], origin: Optional[HostOrigin]) -> None:
    """[U-HOST3] A sealed physics' host rows judged against the description held under its sealed origin. Rows naming a
    host without an origin (a physics sealed before the origin seal) refuse by name: the description their values
    came from cannot be named again, so they are never judged against the description that holds the name now; an
    origin beside no row naming a host refuses by name."""
    named = sorted({value.host for value in constants.values() if value.host})
    if named and origin is None:
        raise HostOriginError(f"{where}: its rows name host {named} and it seals no {ORIGIN_FIELD}: the description its "
                              "values were judged against cannot be named again (a physics sealed before the origin "
                              "seal); a re-read is a new record")
    if origin is not None and not named:
        raise HostOriginError(f"{where}: its {ORIGIN_FIELD} names the {origin.label}, and no row of it names a host")
    admit_host_rows(constants, where, system_rows, link_ends, origin=origin)


def refuse_named(why: Mapping[str, str], where: str, hosts: Iterable[str] = ()) -> None:
    """Raise the rule's refusal naming every row in ``why`` (none: nothing)."""
    if not why:
        return
    hosts = sorted(set(hosts))
    beside = f" beside the rows of host {hosts}" if hosts else ""
    raise HostRowAdmissionError(
        f"{where}: rows {sorted(why)} are not admitted{beside}: a row naming a host is admitted only as an identified "
        "host description's own row or a declared derivation of its rows (hosts/<name>.json, declared as the run's "
        "host block); " + "; ".join(why[k] for k in sorted(why)))


def host_derivation_block(name: str, mode_rule: str, derivations: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """The override rows of ``derivations`` (per row, a factor or a declaration of the one form) on the rows of
    ``name``'s mode ``mode_rule`` selects: each band recomputed from the description, class ``derived``, its host,
    its declaration, and the plane and link ends the description states for the row."""
    identity = f"{name} {select_mode(host_description(name), mode_rule)}"
    d, why = described(identity)
    if d is None:
        raise HostRowAdmissionError(f"host {identity!r}: {why}")
    block: Dict[str, Dict[str, Any]] = {}
    for key, spec in derivations.items():
        declaration = dict(spec) if isinstance(spec, Mapping) else {"of": {key: spec}}
        terms, combine = judged(key, declaration)
        if missing := [row for row, _factor in terms if row not in d.rows]:
            raise HostRowAdmissionError(f"{key} of host {identity!r}: {_undeclared(missing, d)}")
        unit = d.rows[key].unit if key in d.rows else d.rows[terms[0][0]].unit
        scale = to_canonical(1.0, unit)
        low, nominal, high = (v / scale for v in combined(terms, combine, {r: d.si[r] for r, _f in terms}))
        parts = " and ".join(f"{factor:g} x {row} ({d.rows[row].evidence_kind})" for row, factor in terms)
        block[key] = {
            "low": low, "nominal": nominal, "high": high, "unit": unit, "evidence_kind": "derived",
            "citation": "; ".join(dict.fromkeys(d.rows[r].citation for r, _f in terms if d.rows[r].citation)),
            "derivation": f"the {combine} of {parts} of host {identity}, corner by corner (hosts.derivation)",
            HOST_IDENTITY_FIELD: identity, DERIVATION_FIELD: {"of": dict(terms), "combine": combine},
            **({ENERGY_PLANE_FIELD: d.planes[key]} if key in d.planes else {}),
            **({LINK_ENDS_FIELD: {c: list(e) for c, e in d.ends[key].items()}} if key in d.ends else {})}
    return block
