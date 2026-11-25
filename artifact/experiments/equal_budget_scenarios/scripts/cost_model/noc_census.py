""
from __future__ import annotations

import math
import re
from collections import OrderedDict

import numpy as np

import noc_rule as NR

_BLOCK = re.compile(r"^(?P<blk>s\d+b\d+)(?P<part>c1|c2|sc)$")
BLOCK_PARTS = ("c1", "c2", "sc")
ATTACH = -1
STRIDED_FORMS = ("consumer", "all")
BITS_PER_BYTE = int(np.iinfo(np.uint8).bits)


def _groups(n, width):
    return [(j * width, min((j + 1) * width, n)) for j in range(math.ceil(n / width))]


def _rowgroup_channels(cin, k, R, g):
    ""
    lo, hi = g * R, min((g + 1) * R, cin * k * k)
    return lo // (k * k), (hi - 1) // (k * k) + 1


def dataflow(units):
    ""
    names = [u.name for u in units]
    edges = []
    x = None
    i = 0
    while i < len(units):
        nm = names[i]
        if nm == "stem":
            edges.append(("host", i, "conv"))
            x = i
            i += 1
            continue
        if nm == "fc":
            if x is None:
                raise NR.NocRefusal("NOC_DATAFLOW", "fc before any block")
            edges.append((x, i, "pool"))
            edges.append((i, "host", "out"))
            x = i
            i += 1
            continue
        mt = _BLOCK.match(nm)
        if not mt or mt.group("part") != BLOCK_PARTS[0]:
            raise NR.NocRefusal("NOC_DATAFLOW", f"unit {nm!r} is outside the declared dataflow (stem, s#b#c1/c2/sc, fc)")
        blk = mt.group("blk")
        seen = []
        while len(seen) < len(BLOCK_PARTS) and i + len(seen) < len(units) \
                and names[i + len(seen)] == blk + BLOCK_PARTS[len(seen)]:
            seen.append(names[i + len(seen)])
        if len(seen) < len(BLOCK_PARTS) - 1:
            raise NR.NocRefusal("NOC_DATAFLOW", f"{blk}: c1 without its c2")
        c1, c2 = i, i + seen.index(blk + "c2")
        sc = i + seen.index(blk + "sc") if blk + "sc" in seen else None
        edges.append((x, c1, "conv"))
        edges.append((c1, c2, "conv"))
        if sc is not None:
            edges.append((x, sc, "conv"))
            edges.append((sc, c2, "skip"))
        else:
            edges.append((x, c2, "skip"))
        x = c2
        i += len(seen)
    return edges


def traffic(units, splits, R, value_bits, psum_bits, io, strided="consumer"):
    ""
    if strided not in STRIDED_FORMS:
        raise NR.NocRefusal("NOC_MODEL", f"strided form {strided!r} not in {STRIDED_FORMS}")
    base = np.cumsum([0] + [r * c for r, c, w in splits])[:-1]

    def tid(u, i, j):
        r, c, w = splits[u]
        return int(base[u] + j * r + i)
    rec = OrderedDict((k, []) for k in ("phase", "src", "dst", "bytes", "packets", "kind", "copies_of_value"))
    copy_records = []
    copy_units = []

    def add(phase, src, dst, nbytes, packets, kind, copies):
        rec["phase"].append(phase)
        rec["src"].append(src)
        rec["dst"].append(dst)
        rec["bytes"].append(float(nbytes))
        rec["packets"].append(float(packets))
        rec["kind"].append(kind)
        rec["copies_of_value"].append(copies)
    vb, pb = value_bits / float(BITS_PER_BYTE), psum_bits / float(BITS_PER_BYTE)
    for u, un in enumerate(units):
        r, c, w = splits[u]
        if r > 1:
            for j, (lo, hi) in enumerate(_groups(un.cout, w)):
                for i in range(r - 1):
                    add(u, tid(u, i, j), tid(u, i + 1, j), (hi - lo) * pb * un.pos, un.pos, "psum", 1)
    for p, v, kind in dataflow(units):
        if kind == "out":
            un = units[p]
            r, c, w = splits[p]
            for j, (lo, hi) in enumerate(_groups(un.cout, w)):
                add(p, tid(p, r - 1, j), ATTACH, (hi - lo) * vb * un.pos, un.pos, "host_out", 1)
            continue
        vu = units[v]
        rv, cv, wv = splits[v]
        if p == "host":
            ch_groups = [(0, vu.cin)]
            ppos = int(round(io[0] / (vu.cin * vb)))
            src_of = lambda j: ATTACH
            phase = v
        else:
            pu = units[p]
            rp, cp, wp = splits[p]
            ch_groups = _groups(pu.cout, wp)
            if kind == "pool":
                ppos = 1
            elif strided == "consumer" and kind == "conv" and vu.k == 1:
                ppos = min(pu.pos, vu.pos)
            else:
                ppos = pu.pos
            src_of = lambda j, p=p, rp=rp: tid(p, rp - 1, j)
            phase = p
            if pu.cout != vu.cin:
                raise NR.NocRefusal("NOC_DATAFLOW", f"{pu.name} -> {vu.name}: {pu.cout} channels into {vu.cin}")
        if kind == "skip":
            dst_groups = _groups(vu.cout, wv)
            for j, (lo, hi) in enumerate(ch_groups):
                for j2, (lo2, hi2) in enumerate(dst_groups):
                    s = max(0, min(hi, hi2) - max(lo, lo2))
                    if s:
                        add(phase, src_of(j), tid(v, rv - 1, j2), s * vb * ppos, ppos, "skip", 1)
            continue
        kk = 1 if kind == "pool" else vu.k
        cin = vu.cin
        rg = [_rowgroup_channels(cin, kk, R, g) for g in range(rv)]
        copies = np.zeros(cin, dtype=int)
        for (a, b) in rg:
            copies[a:b] += cv
        for j, (lo, hi) in enumerate(ch_groups):
            for g, (a, b) in enumerate(rg):
                s = max(0, min(hi, b) - max(lo, a))
                if not s:
                    continue
                cp = int(copies[max(lo, a):min(hi, b)].max())
                for j2 in range(cv):
                    add(phase, src_of(j), tid(v, g, j2), s * vb * ppos, ppos, "out" if p != "host" else "host_in", cp)
        if p != "host":
            for ch in range(cin):
                copy_records.append((vb * ppos, int(copies[ch])))
                copy_units.append((int(p), int(v)))
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["copy_units"] = copy_units
    out["n_tiles"] = int(sum(r * c for r, c, w in splits))
    out["base"] = base
    out["splits"] = list(splits)
    out["pos"] = np.array([u.pos for u in units])
    out["copy_records"] = copy_records
    return out


STORE = ATTACH - 1

FLOORS = ("unicast", "fifth")
CARRY_FORMS = ("attach", "mesh")
SITES = ("router0", "median")


def site_router(fp, site):
    ""
    if site == "router0":
        return 0
    if site == "median":
        xm = int(np.floor(np.median(np.arange(fp.width))))
        ym = int(np.floor(np.median(np.arange(fp.height))))
        r = xm * fp.height + ym
        if r >= fp.routers:
            raise NR.NocRefusal("NOC_FLOORPLAN", f"the median router {r} is an empty place of the mesh "
                                                f"({fp.routers} routers on {fp.width} x {fp.height})")
        return int(r)
    raise NR.NocRefusal("NOC_MODEL", f"store site {site!r} not in {SITES}")


def mean_hops_to(fp, router):
    ""
    x0, y0 = int(router) // fp.height, int(router) % fp.height
    sx = sum(abs(x - x0) for x in range(fp.width))
    sy = sum(abs(y - y0) for y in range(fp.height))
    return sx / fp.width + sy / fp.height


def entries(tr, tile_pass=None):
    ""
    if tile_pass is None:
        pairs = {(int(s), int(d)) for s, d in zip(tr["src"], tr["dst"]) if d != ATTACH}
        return len(pairs)
    tp = np.asarray(tile_pass)
    s, d = np.asarray(tr["src"]), np.asarray(tr["dst"])
    m = d != ATTACH
    s, d = s[m], d[m]
    cross = (s != ATTACH) & (tp[np.where(s == ATTACH, 0, s)] != tp[d])
    s = np.where(cross, STORE, s)
    span = int(tp.size) - STORE
    return int(np.unique((s - STORE) * span + d).size)


def lower_bound_bh(tr, tpr, form="per_channel", unit_pass=None, keep=None):
    ""
    by = tr["bytes"]
    lo = 0.0
    for u, (r, c, w) in enumerate(tr["splits"]):
        if r > 1:
            m = (tr["kind"] == "psum") & (tr["phase"] == u)
            per_transfer = by[m].sum() / max(1, m.sum())
            lo += per_transfer * c * (math.ceil(r / tpr) - 1)
    if form == "per_channel":
        cu = tr.get("copy_units")
        for k, (b, cp) in enumerate(tr["copy_records"]):
            if unit_pass is not None and unit_pass[cu[k][0]] != unit_pass[cu[k][1]]:
                continue
            lo += b * max(0, cp - (tpr - 1))
        return float(lo)
    if form == "slice":
        mo = tr["kind"] == "out"
        if keep is not None:
            mo = mo & keep
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(tr["copies_of_value"] > 0, np.maximum(0, tr["copies_of_value"] - (tpr - 1))
                            / np.maximum(tr["copies_of_value"], 1), 0.0)
        return float(lo + float(np.sum(by[mo] * frac[mo])))
    raise NR.NocRefusal("NOC_MODEL", f"lower-bound form {form!r} not declared")


def carried_legs(tr, xi, fl_all):
    ""
    src, dst = np.asarray(tr["src"]), np.asarray(tr["dst"])
    kind, cop = np.asarray(tr["kind"]), np.asarray(tr["copies_of_value"])
    by, ph = np.asarray(tr["bytes"], dtype=float), np.asarray(tr["phase"])
    base = np.asarray(tr["base"])
    tot = OrderedDict()
    for t in xi.tolist():
        c = max(int(cop[t]), 1) if kind[t] == "out" else 1
        key = (int(src[t]), int(np.searchsorted(base, dst[t], side="right") - 1))
        if key not in tot:
            tot[key] = [0.0, 0.0, int(ph[t])]
        tot[key][0] += float(by[t]) / c
        tot[key][1] += float(fl_all[t]) / c
    per_tile = OrderedDict()
    for (s, _e), (b, f, p) in tot.items():
        cur = per_tile.get(s)
        if cur is None or b > cur[0]:
            per_tile[s] = (b, f, p)
    out = (np.array(list(per_tile.keys()), dtype=np.int64), np.array([v[2] for v in per_tile.values()], dtype=np.int64),
           np.array([v[0] for v in per_tile.values()], dtype=float),
           np.array([v[1] for v in per_tile.values()], dtype=float))
    v_unit = np.searchsorted(base, dst[xi], side="right") - 1
    inn = (dst[xi].astype(np.int64), v_unit.astype(np.int64), by[xi].astype(float), np.asarray(fl_all)[xi].astype(float))
    return out, inn


def evaluate(tr, Ce, C, decl, model, links="exact", lower="per_channel", mesh="exact", layout=None, floor="fifth",
             carry="attach"):
    ""
    if floor not in FLOORS:
        raise NR.NocRefusal("NOC_MODEL", f"floor form {floor!r} not in {FLOORS}")
    if carry not in CARRY_FORMS:
        raise NR.NocRefusal("NOC_MODEL", f"carry form {carry!r} not in {CARRY_FORMS}")
    fp = NR.floorplan(C, decl["tiles_per_router"], mesh)
    fb, hf = decl["flit_bytes"], decl["header_flits"]
    tpr = decl["tiles_per_router"]
    s_router = site_router(fp, decl.get("store_site", "router0"))
    fl_all = NR.flits(tr["bytes"], tr["packets"], fb, hf)
    if layout is None:
        src = np.where(tr["src"] == ATTACH, ATTACH, tr["src"] % Ce)
        dst = np.where(tr["dst"] == ATTACH, ATTACH, tr["dst"] % Ce)
        keep = np.ones(len(tr["src"]), dtype=bool)
        up = None
    else:
        pos, tpass, up = layout[0], layout[1], layout[2]
        s0 = np.where(tr["src"] == ATTACH, 0, tr["src"])
        d0 = np.where(tr["dst"] == ATTACH, 0, tr["dst"])
        src = np.where(tr["src"] == ATTACH, ATTACH, np.asarray(pos)[s0])
        dst = np.where(tr["dst"] == ATTACH, ATTACH, np.asarray(pos)[d0])
        ps = np.where(tr["src"] == ATTACH, -1, np.asarray(tpass)[s0])
        pd = np.where(tr["dst"] == ATTACH, -1, np.asarray(tpass)[d0])
        keep = (ps == pd) | (ps < 0) | (pd < 0)
    cross = ~keep
    cross_bytes, cross_flits = float(np.sum(tr["bytes"][cross])), float(np.sum(fl_all[cross]))
    T = {k: np.asarray(tr[k])[keep] for k in ("phase", "bytes", "packets", "kind", "copies_of_value")}
    src, dst = src[keep], dst[keep]
    fl = fl_all[keep]
    legs = carry == "mesh" and layout is not None and bool(cross.any())
    n_out = n_in = 0
    if legs:
        (o_t, o_p, o_b, o_f), (i_t, i_p, i_b, i_f) = carried_legs(tr, np.flatnonzero(cross), fl_all)
        n_out, n_in = int(o_t.size), int(i_t.size)
        T = {"phase": np.concatenate([T["phase"], o_p, i_p]),
             "bytes": np.concatenate([T["bytes"], o_b, i_b]),
             "kind": np.concatenate([T["kind"], np.array(["carry_out"] * n_out + ["carry_in"] * n_in)]),
             "copies_of_value": np.concatenate([T["copies_of_value"], np.ones(n_out + n_in, dtype=int)])}
        src = np.concatenate([src, np.asarray(pos)[o_t], np.full(n_in, STORE)])
        dst = np.concatenate([dst, np.full(n_out, STORE), np.asarray(pos)[i_t]])
        fl = np.concatenate([fl, o_f, i_f])
    at_site = s_router * int(tpr)
    rs, rd, xs, ys, xd, yd, hops = NR.place(np.where(src == STORE, at_site, src), np.where(dst == STORE, at_site, dst),
                                            fp)
    hp = hops.astype(float)
    by = T["bytes"]
    bh_p0 = float(np.sum(by * hp))
    lo = lower_bound_bh(tr, tpr, lower, unit_pass=up, keep=None if layout is None else keep)
    bh_hi = float(np.sum(by)) * fp.diameter
    n_units = len(tr["splits"])
    ph_occ, ph_lo = np.zeros(n_units), np.zeros(n_units)
    nsp = float(decl.get("store_ports", 1))
    fs, fd = src == STORE, dst == STORE
    if model == "lane-shared":
        if not legs:
            port, avg, mx = NR.phase_loads(T["phase"], n_units, rs, rd, xs, ys, xd, yd, hp, fl, fp, links == "exact")
        else:
            port, avg, mx = np.zeros(n_units), np.zeros(n_units), np.zeros(n_units)
            nl = max(1, fp.directed_links)
            for u in range(n_units):
                m = T["phase"] == u
                if not m.any():
                    continue
                o = np.bincount(rs[m & ~fs], weights=fl[m & ~fs], minlength=fp.routers)
                i = np.bincount(rd[m & ~fd], weights=fl[m & ~fd], minlength=fp.routers)
                port[u] = max(o.max(), i.max(), fl[m & fd].sum() / nsp, fl[m & fs].sum() / nsp)
                avg[u] = float(np.sum(fl[m] * hp[m])) / nl
                if links == "exact":
                    mx[u] = NR.link_loads(xs[m], ys[m], xd[m], yd[m], fl[m], fp.width, fp.height).max()
        if links == "exact":
            lk = mx
        elif links == "average":
            lk = avg
        else:
            raise NR.NocRefusal("NOC_MODEL", f"links form {links!r} not declared")
        occ = occ_lo = 0.0
        for ph in np.unique(T["phase"]):
            u = int(ph)
            m = T["phase"] == ph
            v = max(port[u], lk[u])
            occ += v
            ph_occ[u] = v
            srcs = src[m]
            once = fl[m] / np.maximum(T["copies_of_value"][m], 1)
            once = np.where(np.isin(T["kind"][m], ("out",)), once, fl[m])
            if not legs:
                t_out = np.bincount(np.where(srcs == ATTACH, 0, srcs + 1), weights=once)
                t_in = np.bincount(np.where(dst[m] == ATTACH, 0, dst[m] + 1), weights=fl[m])
                tile_max = max(t_out[1:].max() if t_out.size > 1 else 0.0, t_in[1:].max() if t_in.size > 1 else 0.0)
                spread = max(once.sum(), fl[m].sum()) / fp.routers
            else:
                t0 = -STORE
                sb, db = srcs - STORE, dst[m] - STORE
                t_out = np.bincount(sb, weights=once, minlength=t0)
                t_in = np.bincount(db, weights=fl[m], minlength=t0)
                tile_max = max(t_out[t0:].max() if t_out.size > t0 else 0.0,
                               t_in[t0:].max() if t_in.size > t0 else 0.0)

                spread = max(float(fl[m & ~fs].sum()), float(fl[m & ~fd].sum())) / fp.routers
            b = max(tile_max, spread)
            if floor == "unicast":
                if not legs:
                    u_out = np.bincount(np.where(srcs == ATTACH, 0, srcs + 1), weights=fl[m])
                    b = max(b, float(u_out.max()) if u_out.size else 0.0, float(t_in.max()) if t_in.size else 0.0)
                else:
                    u_out = np.bincount(sb, weights=fl[m], minlength=t0)
                    b = max(b, float(u_out[1:].max()), float(t_in[1:].max()), float(u_out[0]) / nsp,
                            float(t_in[0]) / nsp)
            occ_lo += b
            ph_lo[u] = b
        occ_hi = occ
    elif model == "crossbar-parallel":
        out_r = np.bincount(rs, weights=fl, minlength=fp.routers)
        in_r = np.bincount(rd, weights=fl, minlength=fp.routers)
        link = float(np.sum(fl * hp)) / max(1, fp.directed_links)
        occ = max(fl.sum() / fp.routers, link)
        once = np.where(T["kind"] == "out", fl / np.maximum(T["copies_of_value"], 1), fl)
        occ_lo = max(once[~fs].sum(), 0.0) / fp.routers
        if floor == "unicast":
            occ_lo = max(occ_lo, max(float(fl[~fs].sum()), float(fl[~fd].sum())) / fp.routers)
        occ_hi = max(out_r.max(), in_r.max(), link)
        if legs:
            sp = max(float(fl[fd].sum()), float(fl[fs].sum())) / nsp
            occ, occ_lo, occ_hi = max(occ, sp), max(occ_lo, sp), max(occ_hi, sp)
    else:
        raise NR.NocRefusal("NOC_MODEL", f"compute model {model!r} not declared")
    intra = (rs == rd) & (src != ATTACH) & (dst != ATTACH) & (src != STORE) & (dst != STORE)
    kinds = T["kind"]
    return OrderedDict(occ=float(occ), occ_lo=float(occ_lo), occ_hi=float(occ_hi), bh=bh_p0, bh_lo=float(lo),
                       bh_hi=bh_hi, bytes=float(np.sum(by)), flits=float(np.sum(fl)),
                       psum_bytes=float(np.sum(by[kinds == "psum"])),
                       out_bytes=float(np.sum(by[np.isin(kinds, ("out", "skip"))])),
                       io_bytes=float(np.sum(by[np.isin(kinds, ("host_in", "host_out"))])),
                       intra_router_bytes=float(np.sum(by[intra])),
                       routers=fp.routers, width=fp.width, height=fp.height, diameter=fp.diameter,
                       mean_hops_to_attach=float(mean_hops_to(fp, s_router)), ph_occ=ph_occ, ph_lo=ph_lo,
                       cross_bytes=cross_bytes, cross_flits=cross_flits,
                       carry_leg_bytes=float(np.sum(by[np.isin(kinds, ("carry_out", "carry_in"))])),
                       carry_leg_byte_hops=float(np.sum((by * hp)[np.isin(kinds, ("carry_out", "carry_in"))])),
                       carry_out_bytes=float(np.sum(by[kinds == "carry_out"])),
                       carry_in_bytes=float(np.sum(by[kinds == "carry_in"])),
                       carry_out_legs=n_out, carry_in_legs=n_in, store_router=int(s_router))


def leg(ev, model, form, scale, c_units, cc_balanced):
    ""
    if model == "lane-shared":
        N = np.asarray(ev["ph_lo"] if form == "lower" else ev["ph_occ"]) * scale
        return float(np.sum(np.maximum(np.asarray(c_units), N)))
    occ = {"p0": ev["occ"], "lower": ev["occ_lo"], "upper": ev["occ_hi"]}[form] * scale
    return float(max(cc_balanced, occ))
