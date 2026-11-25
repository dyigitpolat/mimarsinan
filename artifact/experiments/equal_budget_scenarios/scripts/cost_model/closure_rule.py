""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from collections import OrderedDict

PP_REL = "deployment_record/platform_physics"
TGT_REL = "mapping/platform/targets"
_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:json|md|py|sh|txt|v|sdc|mk|pdf|rpt|tex))(?![\w/.-])")
_TICK = re.compile(r"`([A-Za-z0-9_]+)`")
_SKIP_TEXT_KEYS = ("name",)


class ClosureRefusal(Exception):
    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _load(path):
    with open(path) as fh:
        return json.load(fh, object_pairs_hook=OrderedDict)


class Walk:
    ""

    def __init__(self, src_root):
        self.root = src_root
        self.pp = os.path.join(src_root, PP_REL)
        self.tg = os.path.join(src_root, TGT_REL)
        self.objects = OrderedDict()
        self.files = OrderedDict()
        self._done = set()
        self._by = None
        self.blocks = self._block_index()


    def _block_index(self):
        idx = OrderedDict()
        for kind, pat in (("profile", os.path.join(self.pp, "profiles", "*.json")),
                          ("component", os.path.join(self.pp, "area_components", "*.json")),
                          ("lane_block", os.path.join(self.pp, "lane_blocks", "*.json")),
                          ("host", os.path.join(self.pp, "hosts", "*.json")),
                          ("target", os.path.join(self.tg, "*.json"))):
            for p in sorted(glob.glob(pat)):
                nm = os.path.splitext(os.path.basename(p))[0]
                if nm == "node_scaling":
                    continue
                idx.setdefault(nm, []).append((kind, p))
        return idx

    def rel(self, path):
        return os.path.relpath(path, self.root)

    def add(self, oid, path, where, value, kind):
        if oid not in self.objects:
            self.objects[oid] = OrderedDict(id=oid, kind=kind, file=self.rel(path) if path else None, where=where,
                                            value=_short(value), profiles=[])
        if self._by and self._by not in self.objects[oid]["profiles"]:
            self.objects[oid]["profiles"].append(self._by)

    def _file(self, path):
        if not os.path.isfile(path):
            raise ClosureRefusal("CLOSURE_FILE_MISSING", f"{self.rel(path)} is named and absent")
        self.files[self.rel(path)] = sha(path)

    def _texts(self, obj, path):
        ""
        for s in _strings(obj):
            for m in _PATH.finditer(s):
                p = m.group(1)
                if p.startswith(("http", "www")) or "." not in p:
                    continue
                self.add(f"source:{p}", path, "cited in a text field", p, "source")


    def profile(self, name):
        self._by = name
        p = os.path.join(self.pp, "profiles", f"{name}.json")
        self._file(p)
        d = _load(p)
        pid = f"profile:{name}"
        for k, v in d.items():
            if k in ("validity", "constants", "carry_buffer", "chip_components", "backing_store"):
                continue
            self.add(f"{pid}:{k}", p, k, v, "declaration")
        for k, v in d["validity"].items():
            if k == "geometry":
                for gk, gv in v.items():
                    self.add(f"{pid}:validity.geometry.{gk}", p, f"validity.geometry.{gk}", gv, "declaration")
            else:
                self.add(f"{pid}:validity.{k}", p, f"validity.{k}", v, "declaration")
        for k, v in d["constants"].items():
            self.add(f"{pid}:constants.{k}", p, f"constants.{k}", _ends(v), "constant")
        cb = d.get("carry_buffer", {})
        for r in cb.get("rows", []):
            self.add(f"{pid}:carry_buffer.rows[{r['bytes']}]", p, "carry_buffer.rows",
                     _ends(r["e_carry_buffer_per_byte"]), "constant")
        for k, v in cb.items():
            if k != "rows":
                self.add(f"{pid}:carry_buffer.{k}", p, f"carry_buffer.{k}", v, "declaration")
        comps = []
        for c in d.get("chip_components", []):
            self.add(f"{pid}:chip_components.{c['component']}", p, "chip_components",
                     OrderedDict(count=c.get("count"), role=c.get("role")), "component")
            comps.append(c["component"])
            for lk, lv in (c.get("link") or {}).items():
                self.add(f"{pid}:chip_components.{c['component']}.link.{lk}", p, "chip_components[].link", lv,
                         "declaration")
        for k, v in d.get("backing_store", {}).items():
            self.add(f"{pid}:backing_store.{k}", p, f"backing_store.{k}", "declared store", "component")
        self._texts(d, p)
        rc = d["constants"].get("area_per_router", {}).get("component")
        for c in comps + ([rc] if rc else []):
            self.component(c)
        for c in d.get("chip_components", []):
            if c.get("link", {}).get("host"):
                self.host(c["link"]["host"])
        if d.get("description_file"):
            self.description(os.path.join(self.pp, "profiles", d["description_file"]))
        for t in sorted(glob.glob(os.path.join(self.tg, "*.json"))):
            td = _load(t)
            if td.get("keys", {}).get("platform_physics_profile") == name:
                self.target(t)
        self._by = None

    def component(self, name):
        key = ("component", name)
        p = os.path.join(self.pp, "area_components", f"{name}.json")
        self._file(p)
        d = _load(p)
        if key in self._done:
            for k in d:
                self.add(f"component:{name}:{k}", p, k, None, "declaration")
            if d.get("scaling"):
                self.add(f"node_scaling:rules.{d['scaling']}", None, "rules", None, "declaration")
            return
        self._done.add(key)
        for k, v in d.items():
            self.add(f"component:{name}:{k}", p, k, _ends(v) if isinstance(v, dict) else v,
                     "constant" if k == "area" else "declaration")
        self._texts(d, p)
        if d.get("scaling"):
            ns = os.path.join(self.pp, "area_components", "node_scaling.json")
            self._file(ns)
            rules = _load(ns)["rules"]
            if d["scaling"] not in rules:
                raise ClosureRefusal("CLOSURE_RULE_MISSING", f"{name} names scaling {d['scaling']!r}, absent")
            self.add(f"node_scaling:rules.{d['scaling']}", ns, f"rules.{d['scaling']}",
                     rules[d["scaling"]].get("exponent"), "declaration")

    def host(self, name):
        key = ("host", name)
        p = os.path.join(self.pp, "hosts", f"{name}.json")
        self._file(p)
        d = _load(p)
        if key in self._done:
            for oid in [o for o in self.objects if o.startswith(f"host:{name}:")]:
                self.add(oid, p, None, None, None)
            return
        self._done.add(key)
        hid = f"host:{name}"
        for k, v in d.items():
            if k in ("constants", "modes"):
                continue
            self.add(f"{hid}:{k}", p, k, v, "declaration")
        for k, v in d.get("constants", {}).items():
            self.add(f"{hid}:constants.{k}", p, f"constants.{k}", _ends(v), "constant")
        for mn, m in d.get("modes", {}).items():
            for k, v in m.items():
                if k == "constants":
                    for ck, cv in v.items():
                        self.add(f"{hid}:modes.{mn}.constants.{ck}", p, f"modes.{mn}.constants.{ck}", _ends(cv),
                                 "constant")
                elif k == "refused":
                    for rk, rv in v.items():
                        self.add(f"{hid}:modes.{mn}.refused.{rk}", p, f"modes.{mn}.refused.{rk}", rv, "declaration")
                else:
                    self.add(f"{hid}:modes.{mn}.{k}", p, f"modes.{mn}.{k}", v, "declaration")
        self._texts(d, p)
        if d.get("description_file"):
            self.description(os.path.join(self.pp, "hosts", d["description_file"]))

    def target(self, path):
        name = os.path.splitext(os.path.basename(path))[0]
        key = ("target", name)
        self._file(path)
        d = _load(path)
        tid = f"target:{name}"
        if key in self._done:
            for oid in [o for o in self.objects if o.startswith(tid + ":")]:
                self.add(oid, path, None, None, None)
            return
        self._done.add(key)
        for k, v in d.items():
            if k == "geometry":
                for gk, gv in v.items():
                    self.add(f"{tid}:geometry.{gk}", path, f"geometry.{gk}", gv, "declaration")
            elif k == "keys":
                for kk, kv in v.items():
                    if kk == "onchip_arithmetic":
                        for ak, av in kv.items():
                            self.add(f"{tid}:keys.onchip_arithmetic.{ak}", path, f"keys.onchip_arithmetic.{ak}", av,
                                     "declaration")
                    else:
                        self.add(f"{tid}:keys.{kk}", path, f"keys.{kk}", kv, "declaration")
            else:
                self.add(f"{tid}:{k}", path, k, v, "declaration")
        self._texts(d, path)
        lb = d.get("keys", {}).get("onchip_arithmetic", {}).get("lane_block")
        if lb:
            self.lane_block(lb)

    def lane_block(self, name):
        key = ("lane_block", name)
        p = os.path.join(self.pp, "lane_blocks", f"{name}.json")
        self._file(p)
        d = _load(p)
        lid = f"lane_block:{name}"
        if key in self._done:
            for oid in [o for o in self.objects if o.startswith(lid + ":")]:
                self.add(oid, p, None, None, None)
            return
        self._done.add(key)
        for k, v in d.items():
            if k == "constants":
                for ck, cv in v.items():
                    self.add(f"{lid}:constants.{ck}", p, f"constants.{ck}", _ends(cv), "constant")
            else:
                self.add(f"{lid}:{k}", p, k, v, "declaration")
        self._texts(d, p)

    def description(self, path):
        self._file(path)
        rel = os.path.basename(path)
        with open(path) as fh:
            text = fh.read()
        for nm in sorted(set(_TICK.findall(text))):
            if nm in self.blocks:
                kinds = [k for k, _ in self.blocks[nm]]
                self.add(f"description:{rel}:names.{nm}", path, "a block named in backticks", ",".join(kinds),
                         "reference")
                for kind, bp in self.blocks[nm]:
                    if kind == "lane_block":
                        self.lane_block(nm)
                    elif kind == "host":
                        self.host(nm)
                    elif kind == "component":
                        self.component(nm)
                    elif kind == "target" and _load(bp).get("keys", {}).get("platform_physics_profile") == self._by:
                        self.target(bp)


        for m in _PATH.finditer(text):
            p = m.group(1)
            if "." in p and not p.startswith(("http", "www")):
                self.add(f"source:{p}", path, "cited in the description", p, "source")


def _ends(v):
    if not isinstance(v, dict):
        return v
    return OrderedDict((e, v[e]) for e in ("low", "nominal", "high", "unit", "component") if e in v)


def _short(v):
    ""
    return v


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            if k not in _SKIP_TEXT_KEYS:
                yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)


def walk(src_root, profiles):
    w = Walk(src_root)
    for p in profiles:
        w.profile(p)
    return w


def match(oid, table):
    ""
    import fnmatch
    rows = table["rows"]
    hits = [k for k in rows if "*" in k and fnmatch.fnmatchcase(oid, k)]
    if oid in rows:
        if hits:
            other = [k for k in hits if rows[k]["disposition"] != rows[oid]["disposition"]]
            if other:
                raise ClosureRefusal("CLOSURE_ROW_UNDER_PATTERN",
                                     f"{oid}: its exact row ({rows[oid]['disposition']}) and the pattern(s) {other} of "
                                     f"another disposition both match")
            raise ClosureRefusal("CLOSURE_REDUNDANT_ROW", f"{oid}: its exact row and the pattern(s) {hits} of the same "
                                                          f"disposition both match")
        return oid
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ClosureRefusal("CLOSURE_NO_ROW", f"{oid} has no row in the declared table of dispositions")
    raise ClosureRefusal("CLOSURE_TWO_ROWS", f"{oid} matches {hits}")


def load_table(path, pin):
    ""
    got = sha(path)
    if got != str(pin):
        raise ClosureRefusal("CLOSURE_PIN", f"{os.path.basename(path)} sha256 {got[:16]} against the declared pin "
                                            f"{str(pin)[:16]}")
    return _load(path)
