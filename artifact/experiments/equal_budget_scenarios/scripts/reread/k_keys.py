""
import json
import os
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon7 as C

AN = os.path.join(C.OUT, "analysis.json")
A = json.load(open(AN), object_pairs_hook=OrderedDict)
TAGS = (("k", "K64"), ("h", "H10.1"), ("ronep", "R1'"), ("ronel", "R1L"))
NETS = {"R18 w1 b4444": r"ResNet-18 w1", "R18 w1/sqrt2 b4444": r"ResNet-18 w1/$\sqrt{2}$",
        "R18 w1/2 b4444": r"ResNet-18 w1/2", "R18 w1/(2sqrt2) b4444": r"ResNet-18 w1/(2$\sqrt{2}$)",
        "R10 w1 b4444": r"ResNet-10 w1", "R10 w1/2 b4444": r"ResNet-10 w1/2", "R34 w1 b4444": r"ResNet-34 w1",
        "R34 w1/2 b4444": r"ResNet-34 w1/2", "R34 w1/sqrt2 b4444": r"ResNet-34 w1/$\sqrt{2}$",
        "R34 w1/sqrt2 b3333": r"ResNet-34 w1/$\sqrt{2}$ with 3-bit weights"}
keys = OrderedDict()


def get(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def key(name, value, display, field):
    if name in keys:
        raise SystemExit(f"key {name} twice")
    keys[name] = OrderedDict(name=name, value=value, display=display, record_field=field)


def rd(i):
    p = os.path.join(C.OUT, "read", f"{i}.json")
    return json.load(open(p), object_pairs_hook=OrderedDict), f"BR/reread/out/read/{i}.json"


def th(axis):
    p = os.path.join(C.OUT, "thresh", f"{axis}.json")
    return json.load(open(p), object_pairs_hook=OrderedDict), f"BR/reread/out/thresh/{axis}.json"


def pp(x):
    return f"{x:.2f}\\,pp"


def times(x):
    return f"{x:.2f}"


def af(path):
    return "BR/reread/out/analysis.json#" + "/".join(str(p) for p in path)


for t, c in TAGS:
    b = ["P0_full", c]
    v = get(A, b)
    key(f"pzgain{t}", v["gain_pp"], pp(v["gain_pp"]), af(b + ["gain_pp"]))
    key(f"pzsel{t}", v["J"], NETS[v["J"]], af(b + ["J"]))
    key(f"pzsingle{t}", v["FO"], NETS[v["FO"]], af(b + ["FO"]))
    key(f"pzpasses{t}", v["J_passes"], str(v["J_passes"]), af(b + ["J_passes"]))
    key(f"pztiles{t}", v["J_tiles"], str(v["J_tiles"]), af(b + ["J_tiles"]))
    key(f"pztime{t}", v["time_ratio"], times(v["time_ratio"]), af(b + ["time_ratio"]))
    key(f"pzstore{t}", v["J_store"]["state"], "self refresh with the memory ports asleep"
        if v["J_store"]["state"] == "self refresh, ports asleep" else v["J_store"]["state"],
        af(b + ["J_store", "state"]))
    key(f"pzstoremw{t}", v["J_store"]["store_W"] * 1e3, f"{v['J_store']['store_W'] * 1e3:.0f}\\,mW",
        af(b + ["J_store", "store_W"]))
    key(f"pzstoregap{t}", v["J_store"]["gap_us"], f"{v['J_store']['gap_us']:.0f}\\,$\\mu$s",
        af(b + ["J_store", "gap_us"]))
    key(f"pzlink{t}", v["J_link"]["state"], v["J_link"]["state"], af(b + ["J_link", "state"]))
    key(f"pzlinkgap{t}", v["J_link"]["gap_us"], f"{v['J_link']['gap_us']:.0f}\\,$\\mu$s",
        af(b + ["J_link", "gap_us"]))
    key(f"pzlinkgapfo{t}", v["FO_link"]["gap_us"], f"{v['FO_link']['gap_us']:.0f}\\,$\\mu$s",
        af(b + ["FO_link", "gap_us"]))
    key(f"pzmem{t}", v["J_memory_beside_MB"], f"{v['J_memory_beside_MB']:.2f}\\,MB", af(b + ["J_memory_beside_MB"]))
    key(f"pzeshare{t}", v["J_energy_share"], f"{100 * v['J_energy_share']:.0f}\\%", af(b + ["J_energy_share"]))
    key(f"pzashare{t}", v["J_area_share"], f"{100 * v['J_area_share']:.0f}\\%", af(b + ["J_area_share"]))
    eq = v["equals_pair_of_record_of"]
    key(f"pzpair{t}", eq, ("the batch-64 pair of record" if eq == ["K64"] else
                           "the batch-16 pair of record" if eq == ["H10.1"] else "PENDING"),
        af(b + ["equals_pair_of_record_of"]))
    key(f"pztrained{t}", [v["J_trained_five"], v["FO_trained_five"]],
        "both members trained at five seeds" if (v["J_trained_five"] and v["FO_trained_five"]) else "not both",
        af(b + ["J_trained_five"]))
key("pzdevices", A["P0_full"]["K64"]["J_devices"], str(A["P0_full"]["K64"]["J_devices"]),
    af(["P0_full", "K64", "J_devices"]))
for kk, t in (("1", "one"), ("16", "h"), ("64", "k")):
    u = A["P0_units_uJ"][kk]
    key(f"pzunit{t}", u, f"{u:.1f}\\,$\\mu$J", af(["P0_units_uJ", kk]))

prr = json.load(open(os.path.join(C.HP, "pairs_result_record.json")), object_pairs_hook=OrderedDict)
for c, t in (("K64", "k"), ("H10.1", "h")):
    m = prr["cells"][c]["mean_pp"]
    key(f"pzfive{t}", m, f"+{m:.2f}\\,pp", f"06_build/headline_v46_pairs/pairs_result_record.json#cells/{c}/mean_pp "
                                            f"(the ledger's tr{t}pairmean)")

for t, c in TAGS:
    g = A["four_cells_zero"][c]["gain"]
    key(f"zsgain{t}", g, pp(g), af(["four_cells_zero", c, "gain"]))
    s = A["four_cells_zero"][c]["J"]
    key(f"zssel{t}", s, NETS[s], af(["four_cells_zero", c, "J"]))

pv = json.load(open(os.path.join(C.NEW7, "params_composed.json")), object_pairs_hook=OrderedDict)["parameters"]
key("tilelow", pv["p_tile"]["low"], f"{pv['p_tile']['low']:.2f}\\,$\\mu$W",
    "CD/v47_composed_r7/params_composed.json#parameters/p_tile/low")
key("tilenom", pv["p_tile"]["nominal"], f"{pv['p_tile']['nominal']:.0f}\\,$\\mu$W",
    "CD/v47_composed_r7/params_composed.json#parameters/p_tile/nominal")
key("tilehigh", pv["p_tile"]["high"], f"{pv['p_tile']['high']:,.0f}\\,$\\mu$W",
    "CD/v47_composed_r7/params_composed.json#parameters/p_tile/high")
key("tilenomclass", pv["p_tile"]["class_nominal"], "a declared generic 22\\,nm estimate",
    "CD/v47_composed_r7/params_composed.json#parameters/p_tile/class_nominal")
key("tilehighclass", pv["p_tile"]["class_high"], "the study's own routed tile on a predictive library",
    "CD/v47_composed_r7/params_composed.json#parameters/p_tile/class_high")
ax = A["grid_axes"]
key("chiplow", ax["glob"][0], "0\\,W", af(["grid_axes", "glob", 0]))
key("chiphigh", ax["glob"][-1], "1\\,W", af(["grid_axes", "glob", len(ax["glob"]) - 1]))
key("gridpoints", A["grid"]["points"], f"{A['grid']['points']:,}", af(["grid", "points"]))
key("boxpoints", len(A["box"]), str(len(A["box"])), af(["box"]))
p0, f0 = rd("P0")
pg = p0["point"]["p_global"]
key("pzchipwide", pg * 1e3, f"{pg * 1e3:.2f}\\,mW", f0 + "#point/p_global")

for ax_ in ("p_tile_up", "p_global_up", "temp_up"):
    T, ft = th(ax_)
    unit = T["unit"]
    for t, c in TAGS:
        b = T["bisected"].get(f"cell {c}")
        nm = {"p_tile_up": "thtile", "p_global_up": "thchip", "temp_up": "thtemp"}[ax_] + t
        if b is None:
            key(nm, None, "none in the band", ft + f"#bisected (no 'cell {c}': no change to {T['end']:g} {unit})")
            continue
        val = b["threshold"]
        disp = (f"{val:.0f}\\,$\\mu$W" if unit == "uW" else f"{val:.0f}\\,mW" if unit == "mW" else f"{val:.1f}\\,C")
        key(nm, val, disp, ft + f"#bisected/cell {c}/threshold")
        og = b["other_item"]
        key(nm + "gain", og[0], pp(og[0]), ft + f"#bisected/cell {c}/other_item")
        key(nm + "sel", og[1], NETS[og[1]], ft + f"#bisected/cell {c}/other_item")
        key(nm + "ratio", val / T["P0_value"], f"{val / T['P0_value']:.2f}", ft + f"#bisected/cell {c}/threshold "
                                                                                f"over P0_value")
for ax_ in ("p_tile_down", "p_global_down", "temp_down"):
    T, ft = th(ax_)
    four = [k for k in T["first_change_at_scan"] if k.startswith("cell ")]
    key({"p_tile_down": "nonetiledown", "p_global_down": "nonechipdown", "temp_down": "nonetempdown"}[ax_],
        four, "no change of any of the four cells", ft + "#first_change_at_scan (no cell item)")
T, ft = th("temp_up")
key("thtempronepsr", T["bisected"]["cell R1'"]["self_refresh_device_mW"],
    f"{T['bisected']['cell R1' + chr(39)]['self_refresh_device_mW']:.1f}\\,mW",
    ft + "#bisected/cell R1'/self_refresh_device_mW")

fz = A["cells48_static_vs_zero"]["down"]
key("fallcount", len(fz), str(len(fz)), af(["cells48_static_vs_zero", "down"]))
cz = fz[0]
key("fallcell", cz[0], "batch 64 at 0.25$A_L$ and 1$c_{\\mathrm{ref}}$", af(["cells48_static_vs_zero", "down", 0, 0]))
key("fallzero", cz[1], pp(cz[1]), af(["cells48_static_vs_zero", "down", 0, 1]))
key("fallpz", cz[2], pp(cz[2]), af(["cells48_static_vs_zero", "down", 0, 2]))
key("fallpzsize", round(cz[1] - cz[2], 4), pp(cz[1] - cz[2]), af(["cells48_static_vs_zero", "down", 0]))
f48 = A["grid"]["cells48_falls_vs_zero"]["K64 0.25A_L 1c_ref"]
key("fallmin", f48["min"], pp(f48["min"]), af(["grid", "cells48_falls_vs_zero", "K64 0.25A_L 1c_ref", "min"]))
key("fallminsize", round(f48["zero"] - f48["min"], 4), pp(f48["zero"] - f48["min"]),
    af(["grid", "cells48_falls_vs_zero", "K64 0.25A_L 1c_ref"]))
key("fallgridpts", f48["n"], f"{f48['n']:,}", af(["grid", "cells48_falls_vs_zero", "K64 0.25A_L 1c_ref", "n"]))
key("falltilemax", f48["p_tile"][1] * 1e6, f"{f48['p_tile'][1] * 1e6:.0f}\\,$\\mu$W",
    af(["grid", "cells48_falls_vs_zero", "K64 0.25A_L 1c_ref", "p_tile", 1]))
T, ft = th("p_tile_down")
fd = T["first_change_at_scan"]["48 K64 0.25A_L 1c_ref"]
key("falltiledown", fd["before"], f"{fd['before']:.1f}\\,$\\mu$W", ft + "#first_change_at_scan/48 K64 0.25A_L 1c_ref/before")
T, ft = th("p_global_up")
fu = T["first_change_at_scan"]["48 K64 0.25A_L 1c_ref"]
key("fallchipup", fu["after"], f"{fu['after']:.0f}\\,mW", ft + "#first_change_at_scan/48 K64 0.25A_L 1c_ref/after")
gp, fg = rd("gaps_per_pass")
key("fallperpass", gp["energy"]["cells48"]["K64 0.25A_L 1c_ref"]["gain"],
    pp(gp["energy"]["cells48"]["K64 0.25A_L 1c_ref"]["gain"]), fg + "#energy/cells48/K64 0.25A_L 1c_ref/gain")
D = json.load(open(os.path.join(C.OUT, "d1_fall.json")), object_pairs_hook=OrderedDict)
mz = D["zero"]["combos"][2]["members"]["R34 w1/2 b4444"]["least_E_over_budget"]
mp = D["P0"]["combos"][2]["members"]["R34 w1/2 b4444"]["least_E_over_budget"]
key("fallrthreezero", mz, f"{mz:.2f}", "BR/reread/out/d1_fall.json#zero/combos/2/members/R34 w1/2 b4444")
key("fallrthreepz", mp, f"{mp:.2f}", "BR/reread/out/d1_fall.json#P0/combos/2/members/R34 w1/2 b4444")

mt = max((v["max_time_ratio"] for v in A["grid"]["four_cells"].values()), key=lambda x: x[0])
key("timemax", mt[0], times(mt[0]), af(["grid", "four_cells"]) + " (the largest max_time_ratio)")
key("timemaxsel", mt[5], NETS[mt[5]], af(["grid", "four_cells"]))
key("timemaxpasses", mt[6], str(mt[6]), af(["grid", "four_cells"]))

LS = A["latency_statements"]
key("latstatic", LS["static_power"]["lowers"], str(LS["static_power"]["lowers"]),
    af(["latency_statements", "static_power", "lowers"]))
key("latstaticraises", LS["static_power"]["raises"], str(LS["static_power"]["raises"]),
    af(["latency_statements", "static_power", "raises"]))
key("latcells", LS["static_power"]["denominator_P0"]["determinable"],
    str(LS["static_power"]["denominator_P0"]["determinable"]),
    af(["latency_statements", "static_power", "denominator_P0", "determinable"]))
key("latwm", LS["working_memory"]["lowers"], str(LS["working_memory"]["lowers"]),
    af(["latency_statements", "working_memory", "lowers"]))
dn = LS["working_memory"]["downs"]
key("latwmmin", min(a - b for _, a, b in dn), pp(min(a - b for _, a, b in dn)),
    af(["latency_statements", "working_memory", "downs"]))
key("latwmmax", max(a - b for _, a, b in dn), pp(max(a - b for _, a, b in dn)),
    af(["latency_statements", "working_memory", "downs"]))
_lb = max(v["lat_vs_zero"]["down"] for v in A["box"].values())
key("latboxlowers", _lb, str(_lb),
    af(["box"]) + " (the largest lat_vs_zero/down over the 228 points)")

for i, nm in (("hop1", "hopone"), ("hop2", "hoptwo"), ("A", "hopfour"), ("hop8", "hopeight"),
              ("chip_clock", "hopchip")):
    for t, c in TAGS:
        v = A["points"][i]["cells"][c]
        key(f"{nm}gain{t}", v["gain"], pp(v["gain"]), af(["points", i, "cells", c, "gain"]))
        key(f"{nm}sel{t}", v["J"], NETS[v["J"]], af(["points", i, "cells", c, "J"]))
key("hopmeasured", 762.294, "762\\,MHz", "CD/v47_composed_r7/params_composed.json#network/router/measured_fmax_hz")

for i, nm in (("A", "carrymesha"), ("carry_once_A", "carryoncea")):
    g = A["points"][i]["cells"]["R1L"]["gain"]
    key(nm, g, pp(g), af(["points", i, "cells", "R1L", "gain"]))

for i, nm in (("link_L1_low", "linkonelow"), ("link_L1_high", "linkonehigh"), ("link_L0_bound", "linkzero")):
    g = A["points"][i]["cells"]["R1'"]["gain"]
    key(nm, g, pp(g), af(["points", i, "cells", "R1'", "gain"]))
lp = A["points"]["lane_pass"]["lat_vs_P0"]
key("lanepassslowup", sum(v for k, v in lp["up_by_ctx"].items() if "91 MHz" in k),
    str(sum(v for k, v in lp["up_by_ctx"].items() if "91 MHz" in k)), af(["points", "lane_pass", "lat_vs_P0", "up_by_ctx"]))
key("lanepassslowdown", sum(v for k, v in lp["down_by_ctx"].items() if "91 MHz" in k),
    str(sum(v for k, v in lp["down_by_ctx"].items() if "91 MHz" in k)),
    af(["points", "lane_pass", "lat_vs_P0", "down_by_ctx"]))
key("lanepassfastup", lp["up_by_ctx"].get("DDR4, 1.5 GHz", 0), str(lp["up_by_ctx"].get("DDR4, 1.5 GHz", 0)),
    af(["points", "lane_pass", "lat_vs_P0", "up_by_ctx", "DDR4, 1.5 GHz"]))
key("lanepassfastdown", lp["down_by_ctx"].get("DDR4, 1.5 GHz", 0), str(lp["down_by_ctx"].get("DDR4, 1.5 GHz", 0)),
    af(["points", "lane_pass", "lat_vs_P0", "down_by_ctx", "DDR4, 1.5 GHz"]))
for i, nm in (("launch_record", "launchrecorddown"), ("launch_high", "launchhighdown"), ("host_off", "hostoffmoves"),
              ("host_high", "hosthighmoves")):
    v = A["points"][i]["lat_vs_P0"]
    val = v["down"] if "down" in nm else v["up"] + v["down"]
    key(nm, val, str(val), af(["points", i, "lat_vs_P0"]))

gmin = min(A["P0_full"][c]["J_store"]["gap_us"] for c in ("K64", "H10.1", "R1'", "R1L"))
key("prechargens", 20.8, "20.8\\,ns", "BR/sources_r3/sources_r3.md:696-701 (F3-m3: tRTP plus tRP)")
key("pzgapmin", gmin, f"{gmin:.0f}\\,$\\mu$s", af(["P0_full"]) + " (the least J_store/gap_us)")
key("prechargeshare", 20.8e-3 / gmin, f"{100 * 20.8e-3 / gmin:.3f}\\%", "prechargens over pzgapmin")

r34 = A["seed0_note"]["members"]["R34 w1/sqrt2 b3333"]
key("bthreeseedzero", r34["s0"], f"{r34['s0']:,} of 5,000", af(["seed0_note", "members", "R34 w1/sqrt2 b3333", "s0"]))
key("bthreeseeds", [min(r34["seeds"]), max(r34["seeds"])], f"{min(r34['seeds']):,} to {max(r34['seeds']):,}",
    af(["seed0_note", "members", "R34 w1/sqrt2 b3333", "seeds"]))
for lab, v in A["untrained"]["cost"].items():
    nm = {"R34 w1 b4444": "rthreefourone", "R34 w1/sqrt2 b4444": "rthreefoursq", "R34 w1/sqrt2 b3333": "rthreefoursqb",
          "R34 w1/2 b4444": "rthreefourhalf"}[lab]
    key(nm + "seedgpu", v["one_seed_gpu_h"], f"{v['one_seed_gpu_h']:.2f}\\,GPU hours", af(["untrained", "cost", lab]))
    key(nm + "fivegpu", v["five_seeds_gpu_h"], f"{v['five_seeds_gpu_h']:.2f}\\,GPU hours", af(["untrained", "cost", lab]))

for i, nm in (("joint_energy_low", "jointe"), ("joint_latency_low", "jointl")):
    if i not in A["points"]:
        continue
    for t, c in TAGS:
        v = A["points"][i]["cells"][c]
        key(f"{nm}gain{t}", v["gain"], pp(v["gain"]), af(["points", i, "cells", c, "gain"]))
    lv = A["points"][i]["lat_vs_P0"]
    key(f"{nm}latdown", lv["down"], str(lv["down"]), af(["points", i, "lat_vs_P0", "down"]))
    key(f"{nm}latup", lv["up"], str(lv["up"]), af(["points", i, "lat_vs_P0", "up"]))

for t, c in TAGS:
    f = A["P0_fronts"][c]
    key(f"figbudget{t}", f["energy_budget_uJ"], f"{f['energy_budget_uJ']:.1f}\\,$\\mu$J",
        af(["P0_fronts", c, "energy_budget_uJ"]))

bt = json.load(open(os.path.join(C.NEW7, "data", "store_states_r3.json")), object_pairs_hook=OrderedDict)
key("templow", 25.0, "25\\,C", "CD/v47_composed_r7/data/store_states_r3.json#board_temperature/low (the 25 C row)")
key("temphigh", 85.0, "85\\,C", "CD/v47_composed_r7/data/store_states_r3.json#board_temperature/high ('max', the "
                               "sheet's -40 to 85 C maximum, placed at 85 C for the sweep by t_thresh.py)")
key("tempnom", 45.0, "45\\,C", "CD/v47_composed_r7/data/store_states_r3.json#board_temperature/nominal")
key("cellsother", len(A["points"]["P0"]["cells"]) and len(rd("P0")[0]["energy"]["cells48"]),
    str(len(rd("P0")[0]["energy"]["cells48"])), "BR/reread/out/read/P0.json#energy/cells48 (its count)")
th_ = json.load(open(os.path.join(C.NEW7, "params_composed.json")), object_pairs_hook=OrderedDict)["network"]["t_hop"]
for nm, e in (("hopnsone", "low"), ("hopnstwo", "sweep_2ns"), ("hopnsfour", "nominal"), ("hopnseight", "high")):
    key(nm, th_[e], f"{th_[e]:g}\\,ns", f"CD/v47_composed_r7/params_composed.json#network/t_hop/{e}")
grp = list(rd("P0")[0]["latency"]["groups"].keys())
key("ctxslow", grp[0].split(" | ")[0], "91\\,MHz", "BR/reread/out/read/P0.json#latency/groups (the first context)")
key("ctxfast", [g for g in grp if "1.5 GHz" in g][0].split(" | ")[0], "1.5\\,GHz",
    "BR/reread/out/read/P0.json#latency/groups (the 1.5 GHz context)")

for t, c in TAGS:
    for pl in ("nmap", "aware"):
        key(f"place_{pl}_{t}", None, "PENDING", "a later addendum reads the named placements from placement files")
print(len(keys), "keys", C.dump(OrderedDict(generated_by="BR/reread/scripts/k_keys.py", analysis_sha256=C.sha(AN),
                                            keys=list(keys.values())), sys.argv[1]))
