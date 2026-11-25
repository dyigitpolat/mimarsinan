""
import itertools
import json
import os
import sys
import time
from collections import OrderedDict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon7 as C

T0 = time.time()


def say(m):
    print(f"[{time.time() - T0:8.1f} s] {m}", flush=True)


B = C.load_tree()
import rlib7 as L
import gridlib as G

SHARD, NSH, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
SEL = B.r5_spec("P0")
F = B.FORMS["ruled"]
cache = os.path.join(C.CACHE, "rows")
RD = B.Rows(None, cache_dir=cache, name="declared", r5=SEL)
RC = B.Rows(B.K.CAPS8, cache_dir=cache, name="check", r5=SEL)
if not (RD.cached and RC.cached):
    raise SystemExit("P0's rows are not in the step's row cache: run P0's whole reading first")
say(f"rows of P0 read from the cache, bad {RD.bad, RC.bad}")
J5 = "active standby with refresh, ports idle"
HIGH = ("joint", os.path.join(L.STATE_VARIANTS, "store_states_r3_ports_high.json"))
STORES = [("none", dict(mode="none"), None), ("rule_25C", dict(temperature="25C"), None),
          ("rule_45C_P0", dict(), None), ("rule_max", dict(temperature="max"), None),
          ("J5_ports_high", dict(mode="state", state=J5), HIGH)]


def grid_points():
    PR = B.PARAMS["p_tile"]
    lo, nom, hi = (B.SR.band_value(PR, e) for e in ("low", "nominal", "high"))
    tiles = G.log_axis(lo, hi, 97, exact=(nom,))
    cw = L.chip_wide_axes(B)
    exact = [cw[f"case B {k}"]["p_global_without_link_mW"] * 1e-3
             for k in ("low", "nominal", "high", "high with the upper PLL")]
    glob = [0.0] + G.log_axis(0.5e-3, 1.0, 73, exact=exact)
    pts = []
    for si, (sn, skw, swap) in enumerate(STORES):
        for ti, pt_ in enumerate(tiles):
            for gi, pg in enumerate(glob):
                pts.append(OrderedDict(tag=f"g {si} {ti} {gi}", grid=[si, ti, gi], store_name=sn,
                                       spec=OrderedDict(p_tile=pt_, p_global_W=pg, p_dram_bit="nominal", store=skw,
                                                        link={}),
                                       swap=(None if swap is None else dict(store=list(swap)))))
    for p in L.box_points(B):
        spec = OrderedDict((k, p[k]) for k in ("p_tile", "p_global_W", "p_dram_bit", "store", "link"))
        pts.append(OrderedDict(tag=p["tag"].replace(" ", "_"), grid=None, store_name=p["store_label"], spec=spec,
                               swap=(None if p["table_swap"] is None else dict(store=list(p["table_swap"])))))
    pts.append(OrderedDict(tag="P0", grid=None, store_name="P0", spec="P0", swap=None))
    return pts, dict(tiles=tiles, glob=glob, stores=[s[0] for s in STORES])


if len(sys.argv) > 4:
    PTS = json.load(open(sys.argv[4]), object_pairs_hook=OrderedDict)
    AX = None
else:
    PTS, AX = grid_points()
mine = [p for k, p in enumerate(PTS) if k % NSH == SHARD]
mine.sort(key=lambda p: json.dumps(p["swap"], sort_keys=True))
say(f"{len(PTS)} points, this shard {len(mine)}")
done = set()
if os.path.isfile(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["tag"])
        except Exception:
            pass
fh = open(OUT, "a")
if SHARD == 0 and AX is not None:
    C.dump(AX, OUT.replace(".jsonl", "_axes.json"))
n = 0
for swap_key, grp in itertools.groupby(mine, key=lambda p: json.dumps(p["swap"], sort_keys=True)):
    grp = list(grp)
    sw = json.loads(swap_key)
    st = None if not (sw and sw.get("store")) else tuple(sw["store"])
    with L.Swap(B, [RD, RC], store=st):
        for p in grp:
            if p["tag"] in done:
                continue
            pt = B.point_P0(SEL) if p["spec"] == "P0" else L.make_point(B, p["spec"], SEL)
            e, mk, rd = L.energy_read(B, RD, RC, pt, F, detail=False, sel=SEL)
            g = OrderedDict((k, e[k]) for k in ("units_uJ", "refs", "cells", "rule_rr", "rule_rr_mean", "pairs",
                                                "cells48", "cells48_mean", "masks", "untrained"))
            RD._pc.clear()
            RC._pc.clear()
            rec = OrderedDict(tag=p["tag"], grid=p["grid"], store_name=p["store_name"],
                              p_tile=pt["p_tile"], p_global=pt["p_global"], read=g)
            fh.write(json.dumps(rec, default=C._np) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
                say(f"{n} of {len(mine)}")
fh.close()
say(f"done, {n} points written")
