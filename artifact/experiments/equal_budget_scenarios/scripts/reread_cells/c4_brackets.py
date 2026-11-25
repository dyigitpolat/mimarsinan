""
import json
import os
import sys
import time
from collections import OrderedDict

sys.dont_write_bytecode = True
RRS = ("records/tiers/"
       "paper_v6/fix5/study_build/reread/scripts")
sys.path.insert(0, RRS)
import rcommon7 as C

T0 = time.time()
SCR = "/scratch/tcad/study_r7/p0_cells"
ROWSC = SCR + "/rows"
VDIR = SCR + "/variants/temp"
OUTF = C.BR + "/reread_cells/out/brackets_costliest.json"
THD = C.OUT + "/thresh"
B = C.load_tree()
import rlib7 as L

B.eval_cache(SCR + "/evc_main")
sel = B.r5_spec("P0")
RD = B.Rows(None, cache_dir=ROWSC, name="declared", r5=sel)
RC = B.Rows(B.K.CAPS8, cache_dir=ROWSC, name="check", r5=sel)
if not (RD.cached and RC.cached):
    raise SystemExit("REFUSED: this step's rows are not in its scratch rows cache (c1_cells.py first)")
pt0 = B.point_P0(sel)
forms = B.FORMS["ruled"]
SK = B.PM.skey(B.PM.S_AU_CONVENTION)
DECL = list(B.PM.DECLARED_READ)
T_ROWS = ((25.0, "25C"), (45.0, "45C"), (75.0, "75C"), (85.0, "max"))
J12 = ("self refresh, ports asleep", "self refresh, ports idle")
BASE = json.load(open(B.STATES_FILE), object_pairs_hook=OrderedDict)


def sr_mW(Tc):
    pts = [(tt, float(BASE["states"][[s_["name"] for s_ in BASE["states"]].index(J12[0])]["device_mW"][kk]))
           for tt, kk in T_ROWS]
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= Tc <= t1:
            return p0 + (p1 - p0) * (Tc - t0) / (t1 - t0)
    raise SystemExit(f"temperature {Tc} outside 25 to 85 C")


def temp_table(Tc):
    tb = json.loads(json.dumps(BASE), object_pairs_hook=OrderedDict)
    p = sr_mW(Tc)
    for s_ in tb["states"]:
        if s_["name"] in J12:
            s_["device_mW"]["45C"] = p
    os.makedirs(VDIR, exist_ok=True)
    path = os.path.join(VDIR, f"store_states_r3_T{Tc:.6f}.json")
    with open(path, "w") as fh:
        json.dump(tb, fh, indent=1)
    return path


def costliest(combos):
    least = min(r[3] for r in combos)
    att = [i for i, r in enumerate(combos) if r[3] == least]
    dk = [i for i, r in enumerate(combos) if list(r[0]) == DECL][0]
    k = dk if dk in att else att[0]
    return OrderedDict(k=k, least=least, attaining=att, J=combos[k][1], FO=combos[k][2])


def read_at(axis, x):
    sw = None
    p = dict(pt0)
    if axis.startswith("p_tile"):
        p["p_tile"] = float(x)
    elif axis.startswith("p_global"):
        p["p_global"] = float(x)
    else:
        sw = ("joint", temp_table(x))
    with L.Swap(B, [RD, RC], store=sw):
        ee, mk_, rd_ = L.energy_read(B, RD, RC, p, forms, detail=False, sel=sel)
    RD._pc.clear()
    RC._pc.clear()
    out = OrderedDict()
    for c in L.ORDER:
        v = ee["cells"][c]
        cb = [[r_["combo"], r_["J_x10"], r_["FO_x10"], r_["gap_x10"]]
              for r_ in rd_["cells"][c]["by_s_au"][SK]["per_combo"]]
        out[c] = OrderedDict(sig=[v["gain"], v["J"], v["FO"]], sig_mean=[v["gain_mean"], v["J_mean"], v["FO_mean"]],
                             J_passes=v["J_passes"], time_ratio=v["time_ratio"], combos=cb, costliest=costliest(cb))
    return out


SCALE = {"uW": 1e6, "mW": 1e3, "C": 1.0}
OUT = OrderedDict()
for axis in ("p_tile_up", "p_global_up", "temp_up"):
    Tj = json.load(open(os.path.join(THD, f"{axis}.json")), object_pairs_hook=OrderedDict)
    sc = SCALE[Tj["unit"]]
    for item, b in Tj["bisected"].items():
        if not item.startswith("cell "):
            continue
        c = item[5:].replace(" mean", "")
        mean = item.endswith(" mean")
        a = read_at(axis, b["last_same"] / sc)
        z = read_at(axis, b["first_changed"] / sc)
        key = "sig_mean" if mean else "sig"
        ca, cz = a[c]["costliest"], z[c]["costliest"]
        OUT[f"{axis} | {item}"] = OrderedDict(
            threshold=b["threshold"], unit=Tj["unit"], record_P0_item=b["P0_item"], record_other_item=b["other_item"],
            reproduces_record=(a[c][key] == list(b["P0_item"]) and z[c][key] == list(b["other_item"])),
            costliest_last_same=ca, costliest_first_changed=cz,
            costliest_networks_change=([ca["J"], ca["FO"]] != [cz["J"], cz["FO"]]),
            costliest_least_changes=(ca["least"] != cz["least"]),
            declared_networks_change=(a[c]["sig"][1:] != z[c]["sig"][1:]),
            combos_last_same=a[c]["combos"], combos_first_changed=z[c]["combos"])
        print(f"[{time.time() - T0:7.1f} s] {axis} {item}: {b['threshold']:.6g} {Tj['unit']}, reproduces "
              f"{OUT[f'{axis} | {item}']['reproduces_record']}, costliest k {ca['k']} {ca['J']}/{ca['FO']} {ca['least']}"
              f" -> k {cz['k']} {cz['J']}/{cz['FO']} {cz['least']}, declared changes "
              f"{OUT[f'{axis} | {item}']['declared_networks_change']}", flush=True)
inputs = OrderedDict((os.path.relpath(p, C.RA), C.sha(p)) for p in
                     [os.path.join(THD, f"{a}.json") for a in ("p_tile_up", "p_global_up", "temp_up")]
                     + [RRS + "/rlib7.py", RRS + "/t_thresh.py", os.path.abspath(__file__)])
res = OrderedDict(step="p0:cells c4", brackets=OUT, inputs=inputs, eval_cache=B.eval_cache_stats(),
                  wall_s=round(time.time() - T0, 1))
print(f"output {C.dump(res, OUTF)}, evc {B.eval_cache_stats()}")
B.eval_cache_flush()
