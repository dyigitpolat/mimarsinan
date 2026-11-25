""
import hashlib
import json
import os
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True
P = "/work/research/tcad_paper"
RA = P + "/research_artifacts"
BR = RA + "/12_tcad_readiness/tiers/paper_v6/fix5/study_build"
OUTD = BR + "/reread_cells/out"
CBF = OUTD + "/cells_by_combination.json"
OUTF = OUTD + "/keys_costliest.json"
THD = BR + "/reread/out/thresh"
RP = json.load(open(RA + "/06_build/headline_v46_pairs/rules_pairs.json"))
PAIRS = RP["pairs"]
TRAINED = sorted(RP["jobs"]["members"].keys())
NETS = {"R18 w1 b4444": r"ResNet-18 w1", "R18 w1/sqrt2 b4444": r"ResNet-18 w1/$\sqrt{2}$",
        "R18 w1/2 b4444": r"ResNet-18 w1/2", "R18 w1/(2sqrt2) b4444": r"ResNet-18 w1/(2$\sqrt{2}$)",
        "R10 w1 b4444": r"ResNet-10 w1", "R10 w1/2 b4444": r"ResNet-10 w1/2", "R34 w1 b4444": r"ResNet-34 w1",
        "R34 w1/2 b4444": r"ResNet-34 w1/2", "R34 w1/sqrt2 b4444": r"ResNet-34 w1/$\sqrt{2}$",
        "R34 w1/sqrt2 b3333": r"ResNet-34 w1/$\sqrt{2}$ with 3-bit weights"}
CELLNAME = {"K64": "the batch-64 cell", "H10.1": "the batch-16 cell", "R1'": "the loose batch-1 cell",
            "R1L": "the tight batch-1 cell"}


def sha(path, n=64):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:n]


CB = json.load(open(CBF), object_pairs_hook=OrderedDict)
keys = OrderedDict()


def key(name, value, display, field):
    if name in keys:
        raise SystemExit(f"key {name} twice")
    keys[name] = OrderedDict(name=name, value=value, display=display, record_field=field)


def pp(x):
    return f"{x:.2f}\\,pp"


def times(x):
    return f"{x:.2f}"


def cf(path):
    return "BR/reread_cells/out/cells_by_combination.json#" + "/".join(str(p) for p in path)


c = "R1L"
cell = CB["cells"][c]
k = cell["costliest"]["k"]
x = cell["combinations"][k]
b = ["cells", c, "combinations", k]
jd, fd = x["J_design"], x["FO_design"]
t = "ronel"
key(f"pzgain{t}c", x["gap"], pp(x["gap"]), cf(b + ["gap"]) + " (the cell's least gain over its 8 combinations)")
key(f"pzsel{t}c", x["J"], NETS[x["J"]], cf(b + ["J"]))
key(f"pzsingle{t}c", x["FO"], NETS[x["FO"]], cf(b + ["FO"]))
key(f"pzpasses{t}c", x["J_passes"], str(x["J_passes"]), cf(b + ["J_passes"]))
key(f"pztiles{t}c", x["J_tiles"], str(x["J_tiles"]), cf(b + ["J_tiles"]))
key(f"pztime{t}c", x["time_ratio"], times(x["time_ratio"]), cf(b + ["time_ratio"]))
st = jd["store"]
key(f"pzstore{t}c", st["state"], "self refresh with the memory ports asleep"
    if st["state"] == "self refresh, ports asleep" else st["state"], cf(b + ["J_design", "store", "state"]))
key(f"pzstoremw{t}c", st["store_W"] * 1e3, f"{st['store_W'] * 1e3:.0f}\\,mW", cf(b + ["J_design", "store", "store_W"]))
key(f"pzstoregap{t}c", st["gap_us"], f"{st['gap_us']:.0f}\\,$\\mu$s", cf(b + ["J_design", "store", "gap_us"]))
key(f"pzlink{t}c", jd["link"]["state"], jd["link"]["state"], cf(b + ["J_design", "link", "state"]))
key(f"pzlinkgap{t}c", jd["link"]["gap_us"], f"{jd['link']['gap_us']:.0f}\\,$\\mu$s",
    cf(b + ["J_design", "link", "gap_us"]))
key(f"pzlinkgapfo{t}c", fd["link"]["gap_us"], f"{fd['link']['gap_us']:.0f}\\,$\\mu$s",
    cf(b + ["FO_design", "link", "gap_us"]))
key(f"pzmem{t}c", jd["beside_MB"], f"{jd['beside_MB']:.2f}\\,MB", cf(b + ["J_design", "beside_MB"]))
key(f"pzeshare{t}c", jd["energy_share"], f"{100 * jd['energy_share']:.0f}\\%", cf(b + ["J_design", "energy_share"]))
key(f"pzashare{t}c", jd["area_share"], f"{100 * jd['area_share']:.0f}\\%", cf(b + ["J_design", "area_share"]))
eq = [cc for cc, p in PAIRS.items() if p["J"] == x["J"] and p["FO"] == x["FO"]]
key(f"pzpair{t}c", eq, ("the batch-64 pair of record" if eq == ["K64"] else
                        "the batch-16 pair of record" if eq == ["H10.1"] else "PENDING"),
    "06_build/headline_v46_pairs/rules_pairs.json#pairs against " + cf(b + ["J"]) + " and FO")
key(f"pztrained{t}c", [x["J"] in TRAINED, x["FO"] in TRAINED],
    "both members trained at five seeds" if (x["J"] in TRAINED and x["FO"] in TRAINED) else "not both",
    "06_build/headline_v46_pairs/rules_pairs.json#jobs/members against " + cf(b + ["J"]) + " and FO")


TH = CB["thresholds_R1L"]
for ax_, nm in (("p_tile_up", "thtile"), ("p_global_up", "thchip"), ("temp_up", "thtemp")):
    T = json.load(open(os.path.join(THD, f"{ax_}.json")), object_pairs_hook=OrderedDict)
    ft = f"BR/reread/out/thresh/{ax_}.json"
    unit = T["unit"]
    h = TH[ax_]
    bb = T["bisected"].get("cell R1L")
    base = nm + t
    name = base + "c"
    note = f" (the costliest row's first change: {cf(['thresholds_R1L', ax_])})"
    if bb is None:
        key(name, None, "none in the band", ft + f"#bisected (no 'cell R1L': no change to {T['end']:g} {unit})" + note)
        continue
    val = bb["threshold"]
    disp = (f"{val:.0f}\\,$\\mu$W" if unit == "uW" else f"{val:.0f}\\,mW" if unit == "mW" else f"{val:.1f}\\,C")
    key(name, val, disp, ft + "#bisected/cell R1L/threshold" + note)

    far = h["first_changed"]["R1L"]["combos"]
    least = min(r[3] for r in far)
    att = [i for i, r in enumerate(far) if r[3] == least]
    dk = [i for i, r in enumerate(far) if r[0] == CB["cells"][c]["declared_combination"]][0]
    kf = dk if dk in att else att[0]
    key(base + "gainc", least, pp(least), cf(["thresholds_R1L", ax_, "first_changed", "R1L", "combos", kf, 3]))
    key(base + "selc", far[kf][1], NETS[far[kf][1]], cf(["thresholds_R1L", ax_, "first_changed", "R1L", "combos", kf, 1]))
    key(base + "ratioc", val / T["P0_value"], f"{val / T['P0_value']:.2f}", ft + "#bisected/cell R1L/threshold over "
                                                                               "P0_value")


tr = CB["time_range"]
mx, mn = tr["costliest_max"], tr["costliest_min"]
key("pztimemaxc", mx["value"], times(mx["value"]), cf(["time_range", "costliest_max", "value"]))
key("pztimemaxcellc", mx["cell"], CELLNAME[mx["cell"]], cf(["time_range", "costliest_max", "cell"]))
key("pztimeminc", mn["value"], times(mn["value"]), cf(["time_range", "costliest_min", "value"]))
key("pztimemincellc", mn["cell"], CELLNAME[mn["cell"]], cf(["time_range", "costliest_min", "cell"]))

out = OrderedDict(generated_by="BR/reread_cells/scripts/c2_keys.py", cells_by_combination_sha256=sha(CBF),
                  keys=list(keys.values()))
tmp = OUTF + ".part"
with open(tmp, "w") as fh:
    json.dump(out, fh, indent=1)
    fh.write("\n")
os.replace(tmp, OUTF)
print(f"{len(keys)} keys, {OUTF} {sha(OUTF, 8)}")
for v in keys.values():
    print(f"  {v['name']}: {v['value']!r} -> {v['display']}")
