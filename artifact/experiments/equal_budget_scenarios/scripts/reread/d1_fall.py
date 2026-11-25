""
import json
import os
import sys
from collections import OrderedDict

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon7 as C

B = C.load_tree()
import rlib7 as L

SEL = B.r5_spec("P0")
F = B.FORMS["ruled"]
cache = os.path.join(C.CACHE, "rows")
RD = B.Rows(None, cache_dir=cache, name="declared", r5=SEL)
RC = B.Rows(B.K.CAPS8, cache_dir=cache, name="check", r5=SEL)
CNT = L.MEAS["correct"]
LAB = B.LABELS
Kk, am, beta = 64, 0.25, 1.0
ZS = OrderedDict(p_tile="zero", p_global_W=0.0, p_dram_bit="zero", store=dict(mode="none"), link=dict(mode="none"))
P0 = B.point_P0(SEL)
pts = OrderedDict(zero=L.make_point(B, ZS, SEL), P0=P0,
                  P0_gaps_per_pass=L.make_point(B, OrderedDict(p_tile="nominal",
                                                                p_global_W=P0["p_global"], p_dram_bit="nominal",
                                                                store=dict(gap="per_pass"), link={}), SEL))
pt30 = dict(P0)
pt30["p_tile"] = 30e-6
pts["p_tile_30uW"] = pt30
s = B.S_AU_CONV
out = OrderedDict()
for nm, pt in pts.items():
    U = B.units(RD, pt, F)
    Eb = beta * U[Kk]
    pr = RD.price(pt, s, F)
    combos = []
    for f in B.K.F_BAND:
        for a_m in B.PM.AREA_MODELS:
            area = RD.area(a_m, s, F)
            pool = (RD.Kv == Kk) & (area <= am * B.A_L + 1e-9)
            adm = pool & (pr[("E", f)] <= Eb)

            def best(mask):
                ix = np.flatnonzero(mask)
                if ix.size == 0:
                    return None
                lab = max({LAB[RD.li[i]] for i in ix}, key=lambda l_: (CNT[l_], -LAB.index(l_)))
                ixl = [i for i in ix if LAB[RD.li[i]] == lab]
                i = min(ixl, key=lambda i: (pr[("E", f)][i], area[i]))
                return OrderedDict(network=lab, correct=CNT[lab], geometry=RD.meta[i]["geom"], tiles=RD.meta[i]["C"],
                                   passes=int(pr["Q"][i]), E_over_budget=float(pr[("E", f)][i] / Eb),
                                   period_us=float(pr["t"][i]) * 1e6, static_uJ=float(pr["static"][i]) * 1e6)
            j, fo = best(adm), best(adm & RD.fits)
            members = OrderedDict()
            for lab in ("R34 w1/2 b4444", "R18 w1/sqrt2 b4444", "R18 w1/2 b4444"):
                ix = np.flatnonzero(pool & (RD.li == LAB.index(lab)))
                if ix.size:
                    i = min(ix, key=lambda i: pr[("E", f)][i])
                    members[lab] = OrderedDict(least_E_over_budget=float(pr[("E", f)][i] / Eb),
                                               passes=int(pr["Q"][i]), period_us=float(pr["t"][i]) * 1e6,
                                               static_uJ=float(pr["static"][i]) * 1e6, tiles=RD.meta[i]["C"],
                                               geometry=RD.meta[i]["geom"])
            combos.append(OrderedDict(f=float(f), area_model=a_m, J=j, FO=fo,
                                      gain_pp=(None if not (j and fo) else (j["correct"] - fo["correct"]) * 100.0
                                               / L.MEAS["total"]), members=members))
    out[nm] = OrderedDict(budget_uJ=Eb * 1e6, unit_uJ=U[Kk] * 1e6, combos=combos,
                          gain_pp=min(c["gain_pp"] for c in combos if c["gain_pp"] is not None))
    RD._pc.clear()
print("output", C.dump(out, sys.argv[1]))
