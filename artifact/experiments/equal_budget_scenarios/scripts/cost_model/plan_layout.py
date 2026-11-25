""
from __future__ import annotations

import gzip
import hashlib
import json

import numpy as np

SCHEMA = "placement/1"
ORDERS = ("window_first", "residents_first")


class LayoutRefusal(Exception):
    ""

    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


def unit_pass(n_units, cuts):
    ""
    cuts = sorted(int(c) for c in cuts)
    if any(c <= 0 or c >= n_units for c in cuts) or len(set(cuts)) != len(cuts):
        raise LayoutRefusal("LAYOUT_CUTS", f"cuts {cuts} not inside 1..{n_units - 1}")
    up = np.zeros(int(n_units), dtype=np.int64)
    for c in cuts:
        up[c:] += 1
    return up


def pinned_counts(pass_tiles, r, C):
    ""
    r, C = int(r), int(C)
    forced = [max(0, int(t) - r) if r < C else 0 for t in pass_tiles]
    spare = max(0, C - r - sum(forced))
    out = []
    for t, k in zip(pass_tiles, forced):
        extra = min(spare, int(t) - k)
        spare -= extra
        out.append(k + extra)
    return out


def pinned_of(payloads, k):
    ""
    order = sorted(range(len(payloads)), key=lambda i: (-int(payloads[i]), i))
    return sorted(order[:int(k)])


def plan_layout(unit_tiles, tile_payload, cuts, r, C, order):
    ""
    if order not in ORDERS:
        raise LayoutRefusal("LAYOUT_ORDER", f"order {order!r} not in {ORDERS}")
    unit_tiles = [int(t) for t in unit_tiles]
    F = sum(unit_tiles)
    up = unit_pass(len(unit_tiles), cuts) if cuts else np.zeros(len(unit_tiles), dtype=np.int64)
    tp = np.repeat(up, unit_tiles)
    Q = int(up.max()) + 1 if len(up) else 1
    R = [int(np.sum(tp == q)) for q in range(Q)]
    if max(R) > int(C):
        raise LayoutRefusal("LAYOUT_PASS_EXCEEDS", f"a pass of {max(R)} tiles on {C} positions")
    k = pinned_counts(R, r, C)
    res = np.zeros(F, dtype=bool)
    for q in range(Q):
        idx = np.flatnonzero(tp == q)
        res[idx[pinned_of(np.asarray(tile_payload)[idx], k[q])]] = True
    pos = np.full(F, -1, dtype=np.int64)
    ridx = np.flatnonzero(res)
    P = int(ridx.size)
    if order == "residents_first":
        pos[ridx] = np.arange(ridx.size)
        for q in range(Q):
            w = np.flatnonzero((tp == q) & ~res)
            pos[w] = P + np.arange(w.size)
        return tp, res, pos
    nmax = max(int(np.sum((tp == q) & ~res)) for q in range(Q))
    for q in range(Q):
        w = np.flatnonzero((tp == q) & ~res)
        pos[w] = np.arange(w.size)
    pos[ridx] = nmax + np.arange(ridx.size)
    return tp, res, pos


def verify(tp, res, pos, n_positions, tiles_per_router):
    ""
    faults = []
    tp, res, pos = np.asarray(tp), np.asarray(res, dtype=bool), np.asarray(pos)
    if pos.shape != tp.shape or res.shape != tp.shape:
        return [f"PLACE_SHAPE: {pos.shape} positions for {tp.shape} tiles"]
    if pos.dtype.kind not in "iu":
        return [f"PLACE_TYPE: positions of dtype {pos.dtype}"]
    if np.any(pos < 0):
        faults.append(f"PLACE_MISSING: {int(np.sum(pos < 0))} tiles without a position")
    if np.any(pos >= int(n_positions)):
        faults.append(f"PLACE_OUTSIDE: {int(np.sum(pos >= int(n_positions)))} positions beyond {n_positions}")
    rp = set(pos[res].tolist())
    if len(rp) != int(res.sum()):
        faults.append("PLACE_RESIDENT_SHARED: two resident tiles on one position")
    for q in np.unique(tp):
        w = pos[(tp == q) & ~res]
        on = np.concatenate([pos[res], w])
        if np.unique(on).size != on.size:
            faults.append(f"PLACE_TWICE: pass {int(q)} puts two tiles on one position (residents held)")
        if set(w.tolist()) & rp:
            faults.append(f"PLACE_ON_RESIDENT: pass {int(q)} writes a tile on a resident's position")
        rt = np.bincount(on[on >= 0] // int(tiles_per_router))
        if rt.size and rt.max() > int(tiles_per_router):
            faults.append(f"PLACE_ROUTER: pass {int(q)} puts {int(rt.max())} tiles on one router")
    return faults


def file_sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _no_duplicates(pairs):
    ""
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise LayoutRefusal("PLACEMENT_DUPLICATE", f"the key {k!r} twice in one object")
        seen[k] = v
    return seen


def load_placement(path, pin, tiles_per_router, mesh):
    ""
    got = file_sha256(path)
    if got != str(pin):
        raise LayoutRefusal("PLACEMENT_PIN", f"{path}: sha256 {got[:16]} differs from its declared pin {str(pin)[:16]}")
    with gzip.open(path, "rt") as fh:
        doc = json.load(fh, object_pairs_hook=_no_duplicates)
    if doc.get("schema") != SCHEMA:
        raise LayoutRefusal("PLACEMENT_SCHEMA", f"{path}: schema {doc.get('schema')!r}")
    if type(doc.get("tiles_per_router")) is not int or doc["tiles_per_router"] != int(tiles_per_router):
        raise LayoutRefusal("PLACEMENT_TPR", f"{path}: tiles_per_router {doc.get('tiles_per_router')!r}, declared "
                                             f"{int(tiles_per_router)}")
    if doc.get("mesh") != mesh:
        raise LayoutRefusal("PLACEMENT_MESH", f"{path}: written for the mesh {doc.get('mesh')!r}, read under {mesh!r}")
    for dk, steps in doc.get("designs", {}).items():
        for sk, lst in steps.items():
            if not isinstance(lst, list) or any(type(x) is not int for x in lst):
                raise LayoutRefusal("PLACE_TYPE", f"{path}: {dk} step {sk} holds a position that is not an integer")
    return doc


def design_key(label, geom, C, Ce):
    return f"{label}|{geom}|{int(C)}|{int(Ce)}"


def step_key(r, ce, q):
    return f"{int(r)},{int(ce)},{int(q)}"
