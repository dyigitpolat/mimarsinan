""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MESH_FORMS = ("exact", "near_square")


class NocRefusal(Exception):
    ""

    def __init__(self, code, msg):
        super().__init__(f"{code}: {msg}")
        self.code = code


def mesh_dims(n_routers, form="exact"):
    ""
    n = max(1, int(n_routers))
    if form not in MESH_FORMS:
        raise NocRefusal("NOC_FLOORPLAN", f"mesh form {form!r} not in {MESH_FORMS}")
    h = 1
    for c in range(int(math.isqrt(n)), 0, -1):
        if n % c == 0:
            h = c
            break
    if form == "exact" or h > 1:
        return n // h, h
    w = int(math.isqrt(n))
    w = w if w * w == n else w + 1
    return w, -(-n // w)


@dataclass(frozen=True)
class Floorplan:
    tiles: int
    tiles_per_router: int
    routers: int
    width: int
    height: int

    @property
    def diameter(self):
        return (self.width - 1) + (self.height - 1)

    @property
    def directed_links(self):
        return 2 * ((self.width - 1) * self.height + self.width * (self.height - 1))

    @property
    def mean_hops_to_attach(self):
        ""
        return (self.width - 1) / 2.0 + (self.height - 1) / 2.0


def floorplan(tiles, tiles_per_router, form="exact"):
    if int(tiles_per_router) <= 0:
        raise NocRefusal("NOC_FLOORPLAN", f"tiles_per_router {tiles_per_router} refused")
    n = -(-int(tiles) // int(tiles_per_router))
    w, h = mesh_dims(n, form)
    return Floorplan(int(tiles), int(tiles_per_router), n, w, h)


def router_of(index, fp):
    return np.asarray(index) // fp.tiles_per_router


def xy(router, fp):
    r = np.asarray(router)
    return r // fp.height, r % fp.height


def flits(payload_bytes, packets, flit_bytes, header_flits):
    ""
    pb = np.asarray(payload_bytes, dtype=float)
    pk = np.asarray(packets, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(pk > 0, np.ceil(pb / np.maximum(pk, 1e-300) / flit_bytes - 1e-12), 0.0)
    return pk * (per + header_flits)


def link_loads(xs, ys, xd, yd, fl, width, height):
    ""
    xs, ys, xd, yd = (np.asarray(a, dtype=int) for a in (xs, ys, xd, yd))
    fl = np.asarray(fl, dtype=float)
    E = np.zeros((height, width + 1))
    Wd = np.zeros((height, width + 1))
    N = np.zeros((width, height + 1))
    S = np.zeros((width, height + 1))
    e, w = xd > xs, xd < xs
    np.add.at(E, (ys[e], xs[e]), fl[e])
    np.add.at(E, (ys[e], xd[e]), -fl[e])
    np.add.at(Wd, (ys[w], xd[w]), fl[w])
    np.add.at(Wd, (ys[w], xs[w]), -fl[w])
    n, s = yd > ys, yd < ys
    np.add.at(N, (xd[n], ys[n]), fl[n])
    np.add.at(N, (xd[n], yd[n]), -fl[n])
    np.add.at(S, (xd[s], yd[s]), fl[s])
    np.add.at(S, (xd[s], ys[s]), -fl[s])
    loads = [np.cumsum(a, axis=1)[:, :-1].ravel() for a in (E, Wd, N, S)]
    allv = np.concatenate(loads)
    return allv if allv.size else np.zeros(1)


def place(src_index, dst_index, fp):
    ""
    s = np.where(np.asarray(src_index) < 0, 0, src_index)
    d = np.where(np.asarray(dst_index) < 0, 0, dst_index)
    rs, rd = router_of(s, fp), router_of(d, fp)
    xs, ys = xy(rs, fp)
    xd, yd = xy(rd, fp)
    return rs, rd, xs, ys, xd, yd, np.abs(xs - xd) + np.abs(ys - yd)


def phase_loads(phase, n_phases, rs, rd, xs, ys, xd, yd, hops, fl, fp, exact_links):
    ""
    port, avg, mx = np.zeros(n_phases), np.zeros(n_phases), np.zeros(n_phases)
    nl = max(1, fp.directed_links)
    for u in range(n_phases):
        m = phase == u
        if not m.any():
            continue
        o = np.bincount(rs[m], weights=fl[m], minlength=fp.routers)
        i = np.bincount(rd[m], weights=fl[m], minlength=fp.routers)
        port[u] = max(o.max(), i.max())
        avg[u] = float(np.sum(fl[m] * hops[m])) / nl
        if exact_links:
            mx[u] = link_loads(xs[m], ys[m], xd[m], yd[m], fl[m], fp.width, fp.height).max()
    return port, avg, mx


def energy(byte_hops, e_byte_hop):
    return float(byte_hops) * float(e_byte_hop)
