""
import hashlib
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
OUT = sys.argv[1]


def say(m):
    print(f"[{time.time() - T0:7.1f} s] {m}", flush=True)


B = C.load_tree()
import rlib7 as L

PM, K, PV, K42 = B.PM, B.K, B.PV, B.K42
RES = L.RES
TABLE = RES["measured_table"]
out = OrderedDict()


src_states = os.path.join(C.SR3, "store_states_r3.json")
src_states_A = os.path.join(C.SR3, "store_states_r3_variants", "store_states_r3_caseA.json")
src_link = os.path.join(C.SR4, "link_states_r4.json")
t_src = B.load_states(src_states)
t_srcA = B.load_states(src_states_A)


def same_table(a, b):
    ka = {k: v for k, v in a.items() if not k.startswith("_")}
    kb = {k: v for k, v in b.items() if not k.startswith("_")}
    return json.dumps(ka, sort_keys=True) == json.dumps(kb, sort_keys=True)


lk_new = json.load(open(B.LINK_STATES_FILE))
lk_src = json.load(open(src_link))
diff_link = sorted(k for k in set(lk_new) | set(lk_src) if lk_new.get(k) != lk_src.get(k))
out["LOAD"] = OrderedDict(
    sources_table=dict(path=os.path.relpath(src_states, C.RA), sha256=C.sha(src_states)),
    declared_states_file=dict(path=os.path.relpath(B.STATES_FILE, C.RA), sha256=C.sha(B.STATES_FILE),
                              pin=B._BS["states_sha256"]),
    loaded_STATES_sha256=B.STATES["_sha256"], loaded_STATES_path=B.STATES.get("_path"),
    bytes_equal=C.sha(src_states) == C.sha(B.STATES_FILE) == B.STATES["_sha256"],
    fields_equal_by_the_builds_loader=same_table(t_src, B.STATES),
    case_A=dict(sources_sha256=C.sha(src_states_A), declared_sha256=C.sha(B.STATES_FILE_A),
                loaded=B.STATES_A["_sha256"], fields_equal=same_table(t_srcA, B.STATES_A)),
    point_P0_store=list(B.point_P0()["store"]), point_P0_link=list(B.point_P0()["link"]),
    link=dict(declared_file=os.path.relpath(B.LINK_STATES_FILE, C.RA), sha256=C.sha(B.LINK_STATES_FILE),
              loaded=B.LINK_STATES["_sha256"], fourth_form_file=os.path.relpath(src_link, C.RA),
              fourth_form_sha256=C.sha(src_link), top_level_fields_differing=diff_link),
    p0_chip_wide_without_link_mW=B._HL["p0_chip_wide_without_link_mW"],
    fourth_form_p_global_without_link_mW=L.PS4["derived"]["P0"]["p_global_without_link_mW"])
say(f"LOAD {dict(out['LOAD'])}")

t = time.time()
RD = B.Rows(None, cache_dir=C.CACHE, name="declared")
RC = B.Rows(K.CAPS8, cache_dir=C.CACHE, name="check")
say(f"rows (no round-5 selection) cached {RD.cached, RC.cached} bad {RD.bad, RC.bad} ({time.time() - t:.1f} s)")
out["G_ROWS"] = OrderedDict(mismatches=RD.bad + RC.bad, rows=[len(RD.meta), len(RC.meta)])
REC = B.FORMS["record"]
U0 = B.units(RD, B.ZERO, REC)
txt = B.dumps(B.masks(RD, RC, B.ZERO, REC, U=U0))
rec_bytes = open(os.path.join(B.HV, "masks.json"), "rb").read()
R0 = PM.read(json.loads(txt, object_pairs_hook=OrderedDict), TABLE)
reg = PM.regress(json.loads(txt, object_pairs_hook=OrderedDict))
out["G1"] = OrderedDict(byte_equal=txt.encode() == rec_bytes, sha256=hashlib.sha256(txt.encode()).hexdigest(),
                        record_sha256=hashlib.sha256(rec_bytes).hexdigest())
out["G2"] = OrderedDict(read_equal=json.dumps(R0, sort_keys=True) == json.dumps(RES["measured"], sort_keys=True))
out["G3"] = OrderedDict(equal=reg["equal"], of=reg["of"])
say(f"G1 {out['G1']['byte_equal']}, G2 {out['G2']['read_equal']}, G3 {reg['equal']} of {reg['of']}")
FIELDS = ("label", "geom", "C", "F", "K", "E", "E_stateless", "E_c0", "area", "Q", "fits", "param_bits", "tile_bits")
g4 = OrderedDict()
for lname, R, caps in (("declared", RD, None), ("check", RC, K.CAPS8)):
    Dstudy = K.energy_designs(B.T, B.TECH, caps=caps)
    bad = 0
    for f in K.F_BAND:
        for am in PM.AREA_MODELS:
            for s in K.S_AU:
                A = K.inflate(K.scaled(Dstudy, f, am), s)
                Bb = B.design_rows(R, B.ZERO, f, am, s, REC)
                bad += sum(1 for a, b in zip(A, Bb) if any(a[k] != b[k] for k in FIELDS)) + abs(len(A) - len(Bb))
    g4[lname] = OrderedDict(rows=len(Dstudy), settings=12, rows_differing=bad)
out["G4"] = g4
g5 = OrderedDict()
for lname, R in (("declared", RD), ("check", RC)):
    a, b = R.price(B.ZERO, 1.0, REC), R.price(B.ZERO, 1.0, B.FORMS["ruled"])
    g5[lname] = bool(all(np.array_equal(a[k], b[k]) for k in (("E", 1.0), ("E", 2.0), ("Esl", 1.0), ("Esl", 2.0),
                                                               "E_c0", "step")))
out["G5"] = g5
say(f"G4 {g4}, G5 {g5}")
out["energy_all"] = bool(RD.bad + RC.bad == 0 and out["G1"]["byte_equal"] and out["G2"]["read_equal"]
                         and reg["equal"] == reg["of"] and all(v["rows_differing"] == 0 for v in g4.values())
                         and all(g5.values()))


out["FRONTS_record"] = L.fronts(B, RD, B.ZERO, REC, U0)
out["units_record_uJ"] = {str(k): v * 1e6 for k, v in U0.items()}
say("fronts at the record's setting")


MEAS = TABLE
CNT = [MEAS["correct"][l] for l in K.LABELS]
COLS = ("per_v41", "E_v41", "per_f1", "E_f1", "per_f2", "E_f2", "per_inf", "E_inf", "per_sl", "E_sl")


def gain(P):
    J = [int(i) for i in np.flatnonzero(P["J"]["set"])] if P["J"]["set"] is not None else []
    FO = [int(i) for i in np.flatnonzero(P["FO"]["set"])] if P["FO"]["set"] is not None else []
    if not J or not FO:
        return None
    return (max(CNT[i] for i in J) - max(CNT[i] for i in FO)) * 100.0 / MEAS["total"]


def diffs(Pa, Pb):
    o = []
    for nm in L.PROCS:
        a, b = Pa[nm].get("set"), Pb[nm].get("set")
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            o.append(nm)
    return o


res = OrderedDict()
B._load_caches()
for s_au in (1.0, float(B.S_AU_CONV)):
    tag = f"s_AU {s_au:g}"
    acc = OrderedDict(moved=[], columns=OrderedDict(), reads=0, no_J_new=0, no_J_rec=0)
    for cn in K.CTX_COUNTED:
        groups, _ = K.price_lat(ctxs=(cn,), t_acts=(8,))
        for mod in K.MODELS:
            for Kk in K.KS:
                Grec = groups[(0, K.MODELS.index(mod), Kk)]
                G6, u6, i6 = B.lat_group(cn, mod, Kk, B.ZERO, s_au, REC, ta=8)
                cm = []
                for c in COLS:
                    a = np.nan_to_num(Grec[f"{c}|8|per-phase"], posinf=1e300)
                    bb = np.nan_to_num(G6[c], posinf=1e300)
                    if a.shape != bb.shape or not np.array_equal(a, bb):
                        cm.append(f"{c} ({int(np.sum(a != bb)) if a.shape == bb.shape else 'shape'} rows)")
                if cm:
                    acc["columns"][f"{cn} | {mod} | K{Kk}"] = cm
                for am in K.AREA_MULT:
                    for lb in K.LBUD:
                        L_b = lb * K42.l_ref(mod, 8)
                        d6, g_rec, g_new, j_new, j_rec = set(), [], [], [], []
                        for f in (1.0, 2.0):
                            for ap in K.A_P:
                                Pr = K.lat_cell(Grec, cn, mod, Kk, am * K.A_L, L_b, 8, "per-phase", f, ap, rule="cap",
                                                s_au=s_au)
                                P6 = B.lat_cell(G6, cn, mod, Kk, am * K.A_L, 8, "per-phase", f, ap, u6, lb=lb,
                                                unit_form="record_l_ref", rule="cap", s_au=s_au, forms=REC)
                                acc["reads"] += 1
                                d6 |= set(diffs(Pr, P6))
                                g_rec.append(gain(Pr))
                                g_new.append(gain(P6))
                                j_new.append(P6["J"]["set"] is not None and bool(np.any(P6["J"]["set"])))
                                j_rec.append(Pr["J"]["set"] is not None and bool(np.any(Pr["J"]["set"])))
                        acc["no_J_new"] += int(not any(j_new))
                        acc["no_J_rec"] += int(not any(j_rec))
                        if d6:
                            gr = None if any(g is None for g in g_rec) else round(min(g_rec), 4)
                            gn = None if any(g is None for g in g_new) else round(min(g_new), 4)
                            acc["moved"].append([cn, mod, Kk, am, lb, sorted(d6), gr, gn, any(j_new)])
                say(f"{tag} {cn} {mod} K{Kk}: moved so far {len(acc['moved'])}")
    acc["n_moved"] = len(acc["moved"])
    acc["n_moved_JFORB"] = sum(1 for c in acc["moved"] if set(c[5]) & {"J", "FO", "RB"})
    acc["gains_changed"] = [c for c in acc["moved"] if c[6] != c[7]]
    res[tag] = acc
out["G_LAT"] = res
out["wall_s"] = round(time.time() - T0, 1)
say(f"output {C.dump(out, OUT)}")
