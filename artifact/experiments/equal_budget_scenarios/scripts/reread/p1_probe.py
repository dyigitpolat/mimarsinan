""
import os
import sys
import time
from collections import OrderedDict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon7 as C

T0 = time.time()
OUT = sys.argv[1]
W = OrderedDict()


def say(m):
    W[m] = round(time.time() - T0, 1)
    print(f"[{time.time() - T0:7.1f} s] {m}", flush=True)


B = C.load_tree()
import rlib7 as L
say(f"import, eval cache {B.eval_cache(os.path.join(C.EVC, 'main'))}")
K = B.K
sel = B.r5_spec("P0")
pt = B.point_P0(sel)
F = B.FORMS["ruled"]
cache = os.path.join(C.CACHE, "P0")
RD = B.Rows(None, cache_dir=cache, name="declared", r5=sel)
say(f"rows declared cached {RD.cached} bad {RD.bad} evc {B.eval_cache_stats()}")
RC = B.Rows(K.CAPS8, cache_dir=cache, name="check", r5=sel)
say(f"rows check cached {RC.cached} bad {RC.bad} evc {B.eval_cache_stats()}")
e, mk, rd = L.energy_read(B, RD, RC, pt, F, detail=True, sel=sel)
say(f"energy_read P0 detail units {e['units_uJ']} cells {[(c, v['gain'], v['J']) for c, v in e['cells'].items()]}")
e2, mk, rd = L.energy_read(B, RD, RC, pt, F, detail=False, sel=sel)
say("energy_read P0 again, no detail")
pts = L.box_points(B)
p = pts[len(pts) // 2]
pt2 = L.make_point(B, p, sel)
e3, mk, rd = L.energy_read(B, RD, RC, pt2, F, detail=False, sel=sel)
say(f"energy_read box point {p['tag']} units {e3['units_uJ']}")
cn, mod = K.CTX_COUNTED[0], K.MODELS[0]
G, u, info = B.lat_group(cn, mod, 1, pt, float(B.S_AU_CONV), F, ta=8)
say(f"lat_group {cn} {mod} K1 P0 first, unit {u * 1e6}")
G, u, info = B.lat_group(cn, mod, 1, pt, float(B.S_AU_CONV), F, ta=8)
say("lat_group same again")
G2, u2, info2 = B.lat_group(cn, mod, 1, pt2, float(B.S_AU_CONV), F, ta=8)
say(f"lat_group same group at the box point, unit {u2 * 1e6}")
t = time.time()
n = 0
for am in K.AREA_MULT:
    for lb in K.LBUD:
        for f in (1.0, 2.0):
            for ap in K.A_P:
                B.lat_cell(G, cn, mod, 1, am * K.A_L, 8, "per-phase", f, ap, u, lb=lb, unit_form="composer",
                           rule="cap", s_au=float(B.S_AU_CONV), forms=F)
                n += 1
say(f"lat_cell x{n} in {time.time() - t:.2f} s")
for cn_ in K.CTX_COUNTED:
    for mod_ in K.MODELS:
        G, u, info = B.lat_group(cn_, mod_, 64, pt, float(B.S_AU_CONV), F, ta=8)
        say(f"lat_group {cn_} {mod_} K64 P0, unit {u * 1e6:.4f}")
say(f"evc {B.eval_cache_stats()}")
B.eval_cache_flush()
say("flushed")
C.dump(OrderedDict(wall=W, axes=OrderedDict(CTX_COUNTED=list(K.CTX_COUNTED), MODELS=list(K.MODELS), KS=list(K.KS),
                                             AREA_MULT=list(K.AREA_MULT), LBUD=list(K.LBUD), A_P=list(K.A_P),
                                             F_BAND=list(K.F_BAND), BETAS=list(B.PV.BETAS)),
                   chip_wide_axes=L.chip_wide_axes(B), n_box=len(pts), p0=e), OUT)
