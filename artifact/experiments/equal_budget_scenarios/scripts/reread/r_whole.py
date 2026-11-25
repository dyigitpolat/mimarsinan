""
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon7 as C

T0 = time.time()


def say(m):
    print(f"[{time.time() - T0:8.1f} s] {m}", flush=True)


B = C.load_tree()
import rlib7 as L

JOBS = json.load(open(sys.argv[1]), object_pairs_hook=OrderedDict)["jobs"]
EVC = sys.argv[2]
IDS = sys.argv[3:]
if EVC != "none":
    say(f"the step evaluations' disk cache: {B.eval_cache(os.path.join(C.EVC, EVC))}")
ROWS = OrderedDict()
os.makedirs(os.path.join(C.OUT, "read"), exist_ok=True)


def rows_of(sel):
    if sel not in ROWS:
        t = time.time()
        cache = os.path.join(C.CACHE, "rows")
        RD = B.Rows(None, cache_dir=cache, name="declared", r5=sel)
        RC = B.Rows(B.K.CAPS8, cache_dir=cache, name="check", r5=sel)
        ROWS[sel] = (RD, RC)
        say(f"rows {sel.router_clock}/{sel.t_hop_end}: cached {RD.cached, RC.cached}, bad {RD.bad, RC.bad} "
            f"({time.time() - t:.1f} s), evc {B.eval_cache_stats()}")
    return ROWS[sel]


def step_network(sel, mm, j):
    ""
    m = B.CF.member(B.FAM[mm["label"]], B.PV.GEOMS[mm["geom"]])
    cen = B.X9.census(m)
    Cn = mm["C"]
    lane = sum(u_.pos * max(1, -(-n // Cn)) for u_, n in zip(m.units, cen.tiles))
    cc = B.T_ACT * lane
    ctx = B.ctx_of(B.CTX_E, sel)
    f = ctx["f"]
    if cen.F <= Cn:
        net = B.r5_net(m, mm["geom"], Cn, Cn, B._TC["model"], ctx, sel, cc)
        return OrderedDict(step=None, plan_passes=1, leg_us=net["leg"] / f * 1e6, co_leg_us=net["leg"] / f * 1e6,
                           e_net_uJ=net["e_net"] * 1e6, e_lane_uJ=net["e_lane"] * 1e6, lane_us=net["lane_cycles"] / f * 1e6)
    live = tuple(u_.live_after for u_ in m.units)
    pl = B.X9.plan_steps(cen.tiles, live, Cn)
    stk = [tuple(s) for s in pl["steps"]] + [(Cn, pl["stateless"][0], pl["stateless"][1])]
    net = B.r5_net(m, mm["geom"], Cn, Cn, B._TC["model"], ctx, sel, cc, steps=stk)
    ns = B.step_net(net, stk[j])
    return OrderedDict(step=list(stk[j]), plan_passes=int(ns.get("passes_of_plan", stk[j][2])),
                       leg_us=ns["leg"] / f * 1e6, co_leg_us=net["leg"] / f * 1e6, co_step=net.get("co_step"),
                       cross_bytes=ns.get("cross_bytes"), e_net_step_uJ=ns.get("e_net_step", 0.0) * 1e6,
                       e_net_delta_uJ=ns.get("e_net_delta", 0.0) * 1e6, net_occ_us=ns["occ"] / f * 1e6,
                       e_lane_uJ=net["e_lane"] * 1e6, lane_us=ns.get("lane_cycles", net["lane_cycles"]) / f * 1e6)


def run_thresh(job):
    ""
    import subprocess
    t = time.time()
    os.makedirs(os.path.join(C.OUT, "thresh"), exist_ok=True)
    tout = os.path.join(C.OUT, "thresh", f"{job['axis']}.json")
    lg = os.path.join(C.RR, "logs", f"t_{job['axis']}.log")
    with open(lg, "a") as fh:
        rc = subprocess.run([sys.executable, "-I", "-B", os.path.join(HERE, "t_thresh.py"), job["axis"],
                             str(job["n"]), tout], stdout=fh, stderr=subprocess.STDOUT).returncode
    C.dump(OrderedDict(id=job["id"], axis=job["axis"], exit=rc, output=os.path.relpath(tout, C.RA),
                       output_sha8=(C.sha(tout, 8) if os.path.isfile(tout) else None),
                       wall_s=round(time.time() - t, 1)), os.path.join(C.OUT, "read", f"{job['id']}.json"))
    say(f"{job['id']}: exit {rc} ({time.time() - t:.1f} s)")


def run(job):
    if job["kind"] == "thresh":
        return run_thresh(job)
    t = time.time()
    r5 = job["r5"]
    sel = B.r5_spec(r5["setting"], **r5["over"]) if r5 else None
    RD, RC = rows_of(sel)
    forms = B.FORMS[job["forms"]]
    if job["spec"] == "P0":
        pt = B.point_P0(sel)
    else:
        pt = L.make_point(B, job["spec"], sel)
    swap = job.get("swap") or {}
    st = None if not swap.get("store") else tuple(swap["store"])
    lk = swap.get("link")
    out = OrderedDict(id=job["id"], note=job.get("note", ""), meta={k: v for k, v in job.items()
                                                                     if k not in ("id", "note")},
                      selection=(sel._asdict() if sel else None),
                      point=OrderedDict((k, (list(v) if isinstance(v, tuple) else
                                             (v._asdict() if hasattr(v, "_asdict") else v))) for k, v in pt.items()),
                      rests_on=B.rests_on(pt))
    with L.Swap(B, [RD, RC], store=st, link=lk):
        e, mk, rd = L.energy_read(B, RD, RC, pt, forms, detail=job["detail"], sel=sel)
        say(f"{job['id']}: units {[round(x, 4) for x in e['units_uJ'].values()]}, " + ", ".join(
            f"{c} {v['gain']} {v['J']} Q{v['J_passes']} t{v['time_ratio']}" for c, v in e["cells"].items())
            + f", RR {[x['cell'] for x in e['rule_rr'].values()]}, masks {e['masks']} ({time.time() - t:.1f} s)")
        if job["detail"] and sel is not None:
            for c, v in e["cells"].items():
                for side in ("J_design", "FO_design"):
                    d = v.get(side)
                    if d:
                        d["step_net"] = step_network(sel, RD.meta[d["row"]], d["step"])
            for Kk, r in e["refs"].items():
                i = RD.ref_index(int(Kk))
                r["step_net"] = step_network(sel, RD.meta[i], r["step"])
                r["routers_W"] = float(B.router_watts(pt, RD.meta[i]["C"], RD.meta[i]["geom"]))
        out["energy"] = e
        if job.get("fronts"):
            U = OrderedDict((int(k), v * 1e-6) for k, v in e["units_uJ"].items())
            out["fronts"] = L.fronts(B, RD, pt, forms, U)
        if job.get("units"):
            uf = [(u, {"composer": "composer", "l_ref": "record_l_ref"}[u]) for u in job["units"]]
            out["latency"] = L.latency_read(B, pt, forms, unit_forms=uf)
    out["wall_s"] = round(time.time() - t, 1)
    out["eval_cache"] = B.eval_cache_stats()
    path = os.path.join(C.OUT, "read", f"{job['id']}.json")
    say(f"{job['id']}: output {C.dump(out, path)} ({out['wall_s']} s)")


BYID = OrderedDict((j["id"], j) for j in JOBS)
for i in IDS:
    if i not in BYID:
        raise SystemExit(f"no job {i!r}")
CLAIMS = os.path.join(C.SCR, "claims")
POOL = os.path.join(C.OUT, "plan", "pool.json")


def claim(i):
    os.makedirs(CLAIMS, exist_ok=True)
    try:
        os.mkdir(os.path.join(CLAIMS, i))
    except FileExistsError:
        return False
    with open(os.path.join(CLAIMS, i, "by"), "w") as fh:
        fh.write(f"{os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    return True


queue = list(IDS)
if EVC == "main" and os.path.isfile(POOL):
    queue += [i for i in json.load(open(POOL))["pool"] if i not in queue]
for i in queue:
    if os.path.isfile(os.path.join(C.OUT, "read", f"{i}.json")):
        if i in IDS:
            say(f"{i}: output exists, kept")
        continue
    if not claim(i):
        if i in IDS:
            say(f"{i}: claimed by another process, left to it")
        continue
    run(BYID[i])
B.eval_cache_flush()
say(f"done, evc {B.eval_cache_stats()}")
