""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

import numpy as np

SCHEMA = "resource_states/1"
MW = 1e-3
ENDS = ("low", "nominal", "high")


class StateRuleError(Exception):
    ""

    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


GAP_FORMS = ("per_input", "per_pass", "per_burst")


def rests_per_inference(form, q, K, launched=False):
    ""
    q = np.asarray(q, dtype=float)
    K = np.asarray(K, dtype=float)
    if form == "per_input":
        return q / 1.0
    if form == "per_pass":
        return q / K
    if form == "per_burst":
        return 1.0 + (1.0 if launched else 0.0) * np.maximum(q - 1.0, 0.0) / K
    raise StateRuleError("GAP_FORM", f"{form!r} is not a form of the map {GAP_FORMS}")


def rest_charge(P_busy, Ps, tee, tfit, t_period, t_busy, n_gaps, hold=0.0):
    ""
    t = np.asarray(t_period, dtype=float)
    busy = np.asarray(t_busy, dtype=float)
    ng = np.maximum(np.asarray(n_gaps, dtype=float), 1e-300)
    shape = np.broadcast(t, busy, ng).shape
    t, busy, ng = (np.broadcast_to(x, shape) for x in (t, busy, ng))
    if hold:
        busy = np.minimum(busy + ng * float(hold), t)
    Ps = np.asarray(Ps, dtype=float)
    tee = np.asarray(tee, dtype=float)
    tfit = np.asarray(tfit, dtype=float)
    gap = (t - busy) / ng
    fits = tfit[None, :] <= gap.reshape(-1, 1) + 0.0
    k = (len(Ps) - 1 - np.argmax(fits[:, ::-1], axis=1)).reshape(shape)
    above = np.maximum(k - 1, 0)
    te = tee[k]
    E = P_busy * busy + ng * (Ps[k] * (gap - te) + Ps[above] * te)
    with np.errstate(divide="ignore", invalid="ignore"):
        Pw = np.where(t > 0, E / np.where(t > 0, t, 1.0), 0.0)
    return dict(energy_J=E, power_W=Pw, state=k, gap_s=gap, engaged_s=busy)


TOP_FIELDS = OrderedDict([
    ("schema", "the schema's name"), ("what", "text"), ("resource", "the resource's name"),
    ("case", "the declared configuration the table is read under"), ("power_ends", "the band ends of the powers"),
    ("busy_state", "the state during a burst"), ("busy_basis", "why the busy state"),
    ("hold_s", "the time in the busy state after each burst before a rest may begin"), ("hold_basis", "its basis"),
    ("gap_form", "the rests per period"), ("states", "the ladder, shallowest first"),
    ("excluded_states", "states the ladder leaves out, each with its reason"),
    ("generated_by", "text: the generator"), ("inputs", "text: the generator's inputs by sha256"),
    ("variant_of", "text: the table a variant copies"), ("swapped", "text: what a variant changes"),
    ("sensitivities", "text: named sensitivities")])
INFORMATIONAL = ("what", "generated_by", "inputs", "variant_of", "swapped", "sensitivities", "busy_basis",
                 "hold_basis")
STATE_FIELDS = OrderedDict([
    ("name", "the state's name"), ("power_mW", "the state's power at each band end"),
    ("power_basis", "the basis of the power"), ("power_class", "the class of the power's source"),
    ("entry_s", "the time entering, charged at the state above"), ("entry_basis", "its basis"),
    ("exit_s", "the time leaving, charged at the state above"), ("exit_basis", "its basis"),
    ("min_residency_s", "the least time in the state, inside the fit test"), ("min_residency_basis", "its basis"),
    ("zero_time_reason", "why a time is zero where the other is not")])
POWER_END_FIELDS = ("low", "nominal", "high", "values", "basis")
GAP_FORM_FIELDS = ("nominal", "values", "basis")
EXCLUDED_FIELDS = ("name", "reason")


def _unknown(d, allowed, where):
    bad = [k for k in d if k not in allowed]
    if bad:
        raise StateRuleError("TABLE_FIELD_UNKNOWN", f"{where}: fields {bad} are not declared (declared: {list(allowed)})")


def _num(v, what):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
        raise StateRuleError("STATE_POWER", f"{what} {v!r} refused")
    return float(v)


def load_resource_table(path):
    ""
    raw = open(path, "rb").read()
    t = json.loads(raw)
    if t.get("schema") != SCHEMA:
        raise StateRuleError("SCHEMA", f"{path}: schema {t.get('schema')!r} is not {SCHEMA}")
    _unknown(t, TOP_FIELDS, "the table")
    for k in ("resource", "case", "power_ends", "busy_state", "hold_s", "gap_form", "states"):
        if k not in t:
            raise StateRuleError("TABLE_FIELD", f"{path}: lacks {k}")
    _unknown(t["power_ends"], POWER_END_FIELDS, "power_ends")
    _unknown(t["gap_form"], GAP_FORM_FIELDS, "gap_form")
    for e in t.get("excluded_states", []):
        _unknown(e, EXCLUDED_FIELDS, f"excluded state {e.get('name')!r}")
        if not str(e.get("reason", "")).strip():
            raise StateRuleError("TABLE_FIELD", f"excluded state {e.get('name')!r} without its reason")
    _num(t["hold_s"], "hold_s")
    if t["hold_s"] and not str(t.get("hold_basis", "")).strip():
        raise StateRuleError("STATE_BASIS", "hold_s without its basis")
    pe = t["power_ends"]
    for e in ENDS:
        if pe.get(e) not in pe.get("values", []):
            raise StateRuleError("TABLE_FIELD", f"power_ends.{e} {pe.get(e)!r} not among its values")
    if t["gap_form"].get("nominal") not in t["gap_form"].get("values", []):
        raise StateRuleError("TABLE_FIELD", "gap_form.nominal not among its values")
    for g in t["gap_form"].get("values", []):
        if g not in GAP_FORMS:
            raise StateRuleError("GAP_FORM", f"{path}: gap form {g!r} is not a form of the map {GAP_FORMS}")
    names = []
    for s in t["states"]:
        _unknown(s, STATE_FIELDS, f"state {s.get('name')!r}")
        for k in ("name", "power_mW", "power_basis", "entry_s", "entry_basis", "exit_s", "exit_basis"):
            if k not in s:
                raise StateRuleError("STATE_FIELD", f"state {s.get('name')!r} lacks {k}")
        for e in pe["values"]:
            _num(s["power_mW"].get(e), f"{s['name']} power at {e}")
        for k in ("entry_s", "exit_s", "min_residency_s"):
            if k in s:
                _num(s[k], f"{s['name']} {k}")
        for kb in ("power_basis", "entry_basis", "exit_basis"):
            if not str(s[kb]).strip():
                raise StateRuleError("STATE_BASIS", f"{s['name']}: {kb} empty")
        if s.get("min_residency_s", 0.0) and not str(s.get("min_residency_basis", "")).strip():
            raise StateRuleError("STATE_BASIS", f"{s['name']}: min_residency_s without its basis")
        if (s["entry_s"] == 0) != (s["exit_s"] == 0) and not str(s.get("zero_time_reason", "")).strip():
            raise StateRuleError("STATE_ZERO", f"{s['name']}: a zero time without zero_time_reason")
        names.append(s["name"])
    if len(set(names)) != len(names):
        raise StateRuleError("STATE_FIELD", "state names repeat")
    if t["busy_state"] not in names:
        raise StateRuleError("TABLE_FIELD", f"busy state {t['busy_state']!r} is not a state of the table")
    for e in pe["values"]:
        L = resource_ladder(t, e)
        if L[0][2] != 0 or L[0][3] != 0:
            raise StateRuleError("STATE_ORDER", f"{e}: the shallowest state {L[0][0]!r} has a non-zero time")
        for (n0, p0, t0, f0), (n1, p1, t1, f1) in zip(L, L[1:]):
            if p1 >= p0 or t1 < t0 or f1 < f0:
                raise StateRuleError("STATE_ORDER", f"{e}: {n1!r} after {n0!r} is not deeper (power {p0} to {p1} W, "
                                                    f"entry plus exit {t0} to {t1} s, fit {f0} to {f1} s)")
    t["_sha256"] = hashlib.sha256(raw).hexdigest()
    t["_path"] = path
    return t


def resource_ladder(table, end):
    ""
    if end not in table["power_ends"]["values"]:
        raise StateRuleError("POWER_END", f"{end!r} is not a declared band end")
    return [(s["name"], float(s["power_mW"][end]) * MW, float(s["entry_s"]) + float(s["exit_s"]),
             float(s["entry_s"]) + float(s.get("min_residency_s", 0.0)) + float(s["exit_s"]))
            for s in table["states"]]


def state_power(table, name, end):
    for s in table["states"]:
        if s["name"] == name:
            return float(s["power_mW"][end]) * MW
    raise StateRuleError("STATE_NAME", f"no state {name!r}")


def resource_charge(table, end, t_period, t_busy, n_gaps, name="resource"):
    ""
    t = np.asarray(t_period, dtype=float)
    busy = np.asarray(t_busy, dtype=float)
    if np.any(busy > t) or np.any(busy < 0):
        raise StateRuleError("OVERRUN", f"{name}: a busy time exceeds its period or is below zero: the period must be "
                                        "at least the resource's occupancy (the period composer), never clipped here")
    pe = table["power_ends"]
    e = pe[end] if end in ENDS else end
    L = resource_ladder(table, e)
    return rest_charge(state_power(table, table["busy_state"], e), [x[1] for x in L], [x[2] for x in L],
                       [x[3] for x in L], t, busy, n_gaps, hold=float(table["hold_s"]))
