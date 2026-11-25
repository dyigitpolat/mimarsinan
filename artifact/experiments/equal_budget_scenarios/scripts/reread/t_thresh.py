""
import json
import math
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

AXIS, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SEL = B.r5_spec("P0")
F = B.FORMS["ruled"]
cache = os.path.join(C.CACHE, "rows")
RD = B.Rows(None, cache_dir=cache, name="declared", r5=SEL)
RC = B.Rows(B.K.CAPS8, cache_dir=cache, name="check", r5=SEL)
if not (RD.cached and RC.cached):
    raise SystemExit("P0's rows are not in the step's row cache")
PT0 = B.point_P0(SEL)
P_TILE = B.PARAMS["p_tile"]
T_ROWS = ((25.0, "25C"), (45.0, "45C"), (75.0, "75C"), (85.0, "max"))
J12 = ("self refresh, ports asleep", "self refresh, ports idle")
VDIR = os.path.join(C.SCR, "variants", "temp")
os.makedirs(VDIR, exist_ok=True)
BASE = json.load(open(B.STATES_FILE), object_pairs_hook=OrderedDict)


def sr_mW(T):
    ""
    pts = [(t, float(BASE["states"][[s["name"] for s in BASE["states"]].index(J12[0])]["device_mW"][k]))
           for t, k in T_ROWS]
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        if t0 <= T <= t1:
            return p0 + (p1 - p0) * (T - t0) / (t1 - t0)
    raise SystemExit(f"temperature {T} outside 25 to 85 C")


def temp_table(T):
    ""
    t = json.loads(json.dumps(BASE), object_pairs_hook=OrderedDict)
    p = sr_mW(T)
    for s in t["states"]:
        if s["name"] in J12:
            s["device_mW"]["45C"] = p
    path = os.path.join(VDIR, f"store_states_r3_T{T:.6f}.json")
    with open(path, "w") as fh:
        json.dump(t, fh, indent=1)
    return path, p


def point_at(x):
    ""
    if AXIS.startswith("p_tile"):
        pt = dict(PT0)
        pt["p_tile"] = float(x)
        return pt, None
    if AXIS.startswith("p_global"):
        pt = dict(PT0)
        pt["p_global"] = float(x)
        return pt, None
    path, _ = temp_table(x)
    return dict(PT0), ("joint", path)


def read(x):
    pt, sw = point_at(x)
    with L.Swap(B, [RD, RC], store=sw):
        e, mk, rd = L.energy_read(B, RD, RC, pt, F, detail=False, sel=SEL)
    RD._pc.clear()
    RC._pc.clear()
    return e


def sig(e):
    ""
    s = OrderedDict()
    for c, v in e["cells"].items():
        s[f"cell {c}"] = (v["gain"], v["J"], v["FO"])
        s[f"cell {c} mean"] = (v["gain_mean"], v["J_mean"], v["FO_mean"])
    for ld, v in e["rule_rr"].items():
        s[f"rule RR {ld}"] = (tuple(v["cell"]) if isinstance(v["cell"], list) else v["cell"], v["regret"])
    for k, v in e["cells48"].items():
        s[f"48 {k}"] = v["gain"]
    for c, v in e["pairs"].items():
        s[f"pair {c}"] = (v["J_adm_all"], v["FO_res_all"])
    s["untrained"] = tuple(e["untrained"])
    return s


def brief(e):
    return OrderedDict(units=e["units_uJ"], cells=OrderedDict(
        (c, OrderedDict((k, v[k]) for k in ("gain", "J", "FO", "J_passes", "time_ratio", "J_E_over_budget",
                                             "gain_mean", "J_mean"))) for c, v in e["cells"].items()),
        rule_rr=e["rule_rr"], untrained=e["untrained"])


if AXIS.startswith("p_tile"):
    x0 = PT0["p_tile"]
    x1 = B.SR.band_value(P_TILE, "low" if AXIS.endswith("down") else "high")
    scan = [x0 * (x1 / x0) ** (k / N) for k in range(1, N + 1)]
    scan[-1] = x1
    unit, scale = "uW", 1e6
elif AXIS.startswith("p_global"):
    x0 = PT0["p_global"]
    if AXIS.endswith("down"):
        lo = 1e-9
        scan = [x0 * (lo / x0) ** (k / N) for k in range(1, N + 1)] + [0.0]
    else:
        scan = [x0 * (1.0 / x0) ** (k / N) for k in range(1, N + 1)]
        scan[-1] = 1.0
    unit, scale = "mW", 1e3
else:
    x0 = 45.0
    x1 = 25.0 if AXIS.endswith("down") else 85.0
    scan = [x0 + (x1 - x0) * k / N for k in range(1, N + 1)]
    unit, scale = "C", 1.0
e0 = read(x0) if not AXIS.startswith("temp") else read(45.0)
s0 = sig(e0)
say(f"{AXIS}: P0 read at {x0 * scale:g} {unit}, {len(scan)} scan points")
first = OrderedDict()
prev = x0
for x in scan:
    e = read(x)
    s = sig(e)
    for k in s0:
        if k not in first and s[k] != s0[k]:
            first[k] = OrderedDict(before=prev * scale, after=x * scale, P0=s0[k], value_after=s[k])
    prev = x
say(f"{AXIS}: scan done, {len(first)} items change")
bis = OrderedDict()
for k, v in first.items():
    if not (k.startswith("cell ") or k.startswith("rule RR")):
        continue
    a, b = v["before"] / scale, v["after"] / scale
    for _ in range(80):
        if abs(b - a) <= 1e-7 * max(abs(a), abs(b)):
            break
        m = math.sqrt(a * b) if (a > 0 and b > 0 and not AXIS.startswith("temp")) else 0.5 * (a + b)
        if sig(read(m))[k] == s0[k]:
            a = m
        else:
            b = m
    th = 0.5 * (a + b)
    near = read(a)
    far = read(b)
    bis[k] = OrderedDict(threshold=th * scale, unit=unit, last_same=a * scale, first_changed=b * scale,
                         relative_to_P0=(th / x0 if x0 else None), side_P0=brief(near), side_other=brief(far),
                         P0_item=s0[k], other_item=sig(far)[k])
    if AXIS.startswith("temp"):
        bis[k]["self_refresh_device_mW"] = sr_mW(th)
    say(f"{AXIS} {k}: threshold {th * scale:.8g} {unit}")
out = OrderedDict(axis=AXIS, n_scan=len(scan), unit=unit, P0_value=x0 * scale, end=scan[-1] * scale,
                  P0=brief(e0), first_change_at_scan=first, bisected=bis,
                  temperature_rows=(None if not AXIS.startswith("temp") else
                                    [[t, k, sr_mW(t)] for t, k in T_ROWS]))
say(f"output {C.dump(out, OUT)}")
