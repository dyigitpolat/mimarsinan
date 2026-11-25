""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cl5_readings import nevresim_cell, reuse_reading
from cl5_head import state_label

out, census_path, run = sys.argv[1], sys.argv[2], sys.argv[3]
regime = json.load(open(os.path.join(out, "regime_map.json")))
counts = json.load(open(os.path.join(out, "capacity_counts.json")))
points = regime["points"]


def fmt(v, n=3):
    if v is None:
        return "-"
    if isinstance(v, (list, tuple)):
        return " / ".join(fmt(x, n) for x in v)
    if isinstance(v, float):
        return f"{v:.{n}g}"
    return str(v)


def line(*cells):
    print("| " + " | ".join(str(c) for c in cells) + " |")


status = regime["meta"].get("partial") or {}
print("### Completeness of the run\n")
print(f"{status.get('label', 'no job status (the build ran without the run dir job lists)')} (run code state "
      f"{status.get('code_state')}).\n")
line("tier", "jobs", "complete", "missing (by name)")
line(*["---"] * 4)
for tier, t in sorted((status.get("by_tier") or {}).items()):
    missing = t.get("missing") or []
    line(tier, t["jobs"], t["complete"], ", ".join(missing[:40]) + (f" and {len(missing) - 40} more" if len(missing) > 40
                                                                   else "") if missing else "-")
print()


facts = (regime["meta"].get("plane_facts") or {})
sampled = facts.get("sampled") or {}
print("### Planes: a sampled point per (network, target) and the fit-rule labels\n")
print(f"Plane facts of {facts.get('pairs', 0)} pairs (code state {state_label(facts.get('code_state'))}): "
      "U1 at 1/8 of F, K = 1, the executed program's census against the exploration plane's, field by field; the builder's "
      "single pool tested below F. MEASURED pass counts and bytes, ESTIMATE energies; provisional.\n")
line("network", "target", "departure", "cores", "planes agree", "fields that differ", "builder holds below F at", "refusal")
line(*["---"] * 8)
for key, r in sorted(sampled.items()):
    n, t = key.split("|", 1)
    sp = r.get("sampled_point") or {}
    pl = sp.get("planes") or {}
    differ = [f["field"] for f in pl.get("fields", []) if not f.get("agree")]
    ref = r.get("refusal") or sp.get("candidate_refusal") or sp.get("plane_refusal")
    line(n, t, "coalescing declared" if r.get("departure") else "-", fmt(sp.get("cores")),
         "-" if not pl else ("yes" if pl.get("agree") else "no"), ", ".join(differ) or "-",
         ", ".join(fmt(b) for b in (r.get("builder_holds_below_F") or [])) or "none",
         "-" if not ref else (ref.get("name") if isinstance(ref, dict) else str(ref))[:60])
labels = Counter()
for p in points:
    lab = p.get("planes_fit")
    if lab:
        labels[lab["label"].split(":")[0]] += 1
sampled_counts = Counter()
for p in points:
    lab = p.get("planes_sampled")
    if lab:
        sampled_counts[lab["label"].split(" (")[0]] += 1
print("\nRegime points by their pair's sampled planes (executed against candidate at the sampled point):\n")
line("label", "points")
line("---", "---")
for k, v in sorted(sampled_counts.items()):
    line(k, v)
print("\nRegime points by their fit-rule label (the shape-only plane against the builder):\n")
line("label", "points")
line("---", "---")
for k, v in sorted(labels.items()):
    line(k, v)
print()


w09_path = os.path.join(out, "w09_deploy.json")
if os.path.exists(w09_path):
    w09 = json.load(open(w09_path))
    import hashlib
    sha = hashlib.sha256(open(w09_path, "rb").read()).hexdigest()[:16]
    print(f"### W09 deployment gate (w09_deploy.json sha256 {sha}, code state "
          f"{state_label(w09.get('code_state'))}; failed rows: {w09.get('failed') or 'none'})\n")
    line("member", "target", "variant", "cores", "x_packed", "replication", "neural passes", "certificate",
         "planes (executed against candidate)", "write energy (executed / candidate)")
    line(*["---"] * 10)
    for r in w09["rows"]:
        if r.get("route") != "value":
            continue
        fields = {f["field"]: f for f in (r.get("planes") or {}).get("fields", [])}
        e = fields.get("energy_write_mj") or {}
        er = fields.get("energy_write_mj_refusal") or {}
        energy = (f"{fmt(e.get('executed'))} mJ on both planes" if e.get("executed") is not None and e.get("agree")
                  else f"executed {fmt(e.get('executed'))}, candidate {fmt(e.get('candidate'))} mJ" if e.get("executed") is not None
                  else "refused by name on both planes" if er.get("agree") and er.get("executed") else
                  f"{er.get('executed')} / {er.get('candidate')}")
        cert = r.get("certificate") or {}
        line(r["member"], r["target"], r.get("variant"), fmt(r.get("cores")), fmt(r.get("x_packed")),
             fmt(r.get("replication")), fmt(r.get("neural_passes")),
             "exact at atol 0" if r.get("certified") else "NOT certified",
             "agree" if (r.get("planes") or {}).get("agree") else
             "PlaneDisagreement: " + ", ".join(f["field"] for f in (r.get("plane_refusal") or {}).get("fields", [])),
             energy)
    print()
    line("member", "target", "variant", "cores", "x_packed", "replication", "neural passes", "count certificate (HCM)",
         "per-layer census (spikes; distinct inputs of 16)", "unrecorded layers", "nevresim")
    line(*["---"] * 11)
    for r in w09["rows"]:
        if r.get("route") != "spiking":
            continue
        if r.get("refusal"):
            line(r["member"], r["target"], r.get("variant"), "-", "-", "-", "-",
                 f"refused by name: {r['refusal']['name']}", "-", "-", "-")
            continue
        census = "; ".join(f"{k}: {v['spikes']}, {v['distinct_inputs']}" for k, v in (r.get("layer_census") or {}).items())
        nev_text = nevresim_cell(r.get("nevresim"))
        verdict = ("certified" if r.get("certified") else "NOT certified: " + (
            f"VacuousCertificate (silent {r['vacuity_refusal']['silent_layers']}, input-independent "
            f"{r['vacuity_refusal']['input_independent_layers']})" if r.get("vacuity_refusal") else "mismatch"))
        line(r["member"], r["target"], r.get("variant"), fmt(r.get("cores")), fmt(r.get("x_packed")),
             fmt(r.get("replication")), fmt(r.get("neural_passes")), verdict, census or "-",
             ", ".join(r.get("layers_unrecorded") or []) or "-", nev_text)
    print()


print("### Coverage of the priced tier\n")
line("target", "points priced", "refused", "above budget", "networks")
line(*["---"] * 5)
cov = defaultdict(Counter)
nets = defaultdict(set)
for p in points:
    key = p["target"] + (" (coalescing declared)" if p.get("departure") else "")
    cov[key]["refused" if p.get("refusal") else "above" if p.get("above_budget") else "priced"] += 1
    nets[key].add(p["network"])
for t in sorted(cov):
    line(t, cov[t]["priced"], cov[t]["refused"], cov[t]["above"], len(nets[t]))
ref = Counter((p["target"], p["refusal"]["name"], p["refusal"].get("stage", "")) for p in points if p.get("refusal"))
print("\nRefusals by name (target, refusal, stage: points):")
for (t, n, s), c in sorted(ref.items()):
    print(f"- {t}, {n}, {s or 'schedule'}: {c}")
print()


p110, full = counts["probe_110"], counts["full_set"]
print("### Capacity counts\n")
line("set", "pairs", "x_fluid measured", "exceed at x_fluid", "x_packed measured", "exceed at x_packed")
line(*["---"] * 6)
line("probe's 110 pairs (10 ImageNet members x 11 chips)", p110["pairs"], p110["measured"], p110["exceed_fluid"],
     p110.get("packed_measured"), p110["exceed_packed"])
line("full set", full["pairs"], full["measured"], full["exceed_fluid"], full.get("packed_measured"),
     full["exceed_packed"])
line("probe ESTIMATE (fluid bound, weight count)", 110, 110, 72, "-", "-")
print("\nNot measured among the 110:", "; ".join(f"{k}: {v[:90]}" for k, v in list(p110["not_measured"].items())[:20]))
print()


print("### x_fluid and x_packed at the declared core count (capacity tier)\n")
rows = []
for f in sorted(glob.glob(os.path.join(run, "capacity", "*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        if "network" in r:
            rows.append(r)
sel = ("neurram", "hermes_64", "isaac_like", "generic_estimated_22nm", "sram_cim_macro", "truenorth", "loihi",
       "puma_node_like", "ibm_analog_ai_512x512x34")
by = {(r["network"], r["target"]): r for r in rows}
networks = sorted({r["network"] for r in rows})
reuse = reuse_reading(rows)
line("network", "R (MACs/weight)", *[f"{t} x_f / x_p" for t in sel])
line(*["---"] * (2 + len(sel)))
for n in networks:
    rr = [by.get((n, t)) for t in sel]
    line(n, reuse[n],
         *[("refused: " + ((r.get("refusal") or {}).get("name", "")) if r and r.get("x_fluid") is None else
            f"{fmt(r.get('x_fluid'))} / {fmt(r.get('x_packed'))}") if r else "-" for r in rr])
print()


print("### Regime readings of U1 points (Pi/K per band corner; counts of points)\n")
reg = defaultdict(Counter)
for p in points:
    if p.get("arm") not in (None, "absent"):
        continue
    for axis in ("energy", "time"):
        r = (p.get(axis) or {}).get("regime")
        if r:
            reg[(p["target"] + ("" if p.get("variant") in (None, "host") else "+" + p["variant"][5:]), axis,
                 p["inference_batch_K"])][r["reading"]] += 1
labels = ["capacity-free", "informative", "resident-dominant", "straddles", "resident (no per-inference weight writes)"]
line("target (variant)", "axis", "K", *labels)
line(*["---"] * (3 + len(labels)))
for k in sorted(reg):
    line(*k, *[reg[k].get(x, 0) for x in labels])
print()


print("### Residency arms side by side (programmed bytes per period, K = 1, host variant)\n")
rows_s = [r for r in regime.get("side_by_side", []) if r["inference_batch_K"] == 1 and r.get("variant") in (None, "host")
          and all(a in r["arms"] for a in ("absent", "floor", "stateless"))]
fl_st = [r["floor_over_stateless_bytes"] for r in rows_s if r.get("floor_over_stateless_bytes") is not None]
fl_u1 = [r["floor_over_u1_bytes"] for r in rows_s if r.get("floor_over_u1_bytes") is not None]
bc4 = [r["arms"]["absent"].get("bc4_saving_closed_form") for r in rows_s]
print(f"- rows with all three arms: {len(rows_s)}")
if fl_st:
    print(f"- floor over stateless bytes: median {statistics.median(fl_st):.3g}, min {min(fl_st):.3g}, max {max(fl_st):.3g}")
if fl_u1:
    print(f"- floor over U1 bytes: median {statistics.median(fl_u1):.3g}; floor below U1 on {sum(v < 1 for v in fl_u1)},"
          f" equal on {sum(v == 1 for v in fl_u1)}, above on {sum(v > 1 for v in fl_u1)}")
print()
line("network", "target", "budget", "U1 bytes", "floor bytes", "stateless bytes", "floor/stateless",
     "BC4 closed form, P = the floor's passes (P)")
line(*["---"] * 8)
for r in rows_s[:40]:
    a = r["arms"]
    line(r["network"], r["target"], fmt(float(r["budget"]) if r["budget"] != "declared" else "declared"),
         fmt(a["absent"]["period_bytes"]), fmt(a["floor"]["period_bytes"]), fmt(a["stateless"]["period_bytes"]),
         fmt(r.get("floor_over_stateless_bytes")),
         f"{fmt(a['floor'].get('bc4_saving_closed_form'))} ({a['floor'].get('passes')})")
print()


print("### Minimum batch on the write-limited targets (census K* beside the closed form; K = 1 rows)\n")
line("network", "target", "arm", "budget", "cores", "x_fluid", "x_packed", "writes/cell/inf",
     "K_end census (E low/nom/high; batch = ceiling)",
     "K_end closed x_packed", "K_bw (t_w low/nom/high)", "K_min fav/nom/adv", "binding nom", "E_cross census", "ratio split")
line(*["---"] * 15)
for p in points:
    if p["target"] not in ("neurram", "hermes_64", "isaac_like") or p.get("inference_batch_K") != 1 or p.get("refusal") \
            or p.get("above_budget") or p.get("budget") not in (0.125, 0.5, "declared"):
        continue
    mb, ks = p.get("min_batch") or {}, p.get("k_star") or {}
    km = mb.get("k_min_census") or {}
    line(p["network"], p["target"], p.get("arm"), p["budget"], p["cores"], fmt(p["x_fluid"]), fmt(p["x_packed"]),
         fmt(ks.get("max_writes_per_cell_per_inference")),
         " / ".join(str(math.ceil(v)) for v in mb.get("k_endurance_census") or [] if v is not None) or "-",
         fmt(mb.get("k_endurance_closed_packed")), fmt(mb.get("k_bandwidth")),
         " / ".join(str(math.ceil(km[c])) if km.get(c) is not None else "-" for c in ("favourable", "nominal", "adverse")),
         (mb.get("binding") or {}).get("nominal", "-"), fmt(mb.get("e_cross_census")),
         f"{fmt(ks.get('factor_packing_replication'))} x {fmt(ks.get('factor_rotation'))}" if ks.get("ratio") else "-")
print()


print("### Latency semantics (22 nm point with HBM2, U1, budget 1/8)\n")
line("network", "K", "time per inference s", "batch period s", "request latency s", "throughput /s",
     "carry buffer B", "layer-pipelined bound s")
line(*["---"] * 8)
for p in points:
    if p["target"] == "generic_estimated_22nm" and p.get("variant") == "host+hbm2" and p.get("arm") == "absent" \
            and p.get("budget") in (0.125, 1.0) and not p.get("refusal"):
        ls = p.get("latency_semantics") or {}
        line(p["network"] + f" @{p['budget']}", p["inference_batch_K"], fmt(ls.get("time_per_inference_s")),
             fmt(ls.get("batch_period_s")), fmt(ls.get("request_latency_s")),
             fmt(ls.get("throughput_inferences_per_s")), fmt(ls.get("carry_buffer_bytes")),
             fmt(ls.get("layer_pipelined_bound_s")))
print()


print("### M against the closed form (nominal corner)\n")
for arm, form in (("floor", "m_rel_diff_floor"), ("stateless", "m_rel_diff_stateless"), ("absent", "m_rel_diff_stateless")):
    for axis in ("energy", "time"):
        v = [p[axis][form][1] for p in points if p.get("arm") == arm and (p.get(axis) or {}).get(form)]
        if v:
            print(f"- {arm} arm, {axis}, toolchain M against the {form[11:]} closed form: n {len(v)}, median "
                  f"{statistics.median(v):.3g}, min {min(v):.3g}, max {max(v):.3g}")
print()


if os.path.exists(census_path):
    c = json.load(open(census_path))
    print("### CL5b sentences\n")
    for s in c.get("sentences", []):
        print("- " + s)
    print("\nEndurance demonstration (cells whose admitted set the lifetime floor changes):")
    for tag, demo in c.get("endurance", {}).items():
        print(f"- {tag}: " + "; ".join(
            f"{t}: cells changed {d['count']} (joint {d.get('joint_changed')}, fill {d.get('comparator_changed', {}).get('fill')}, "
            f"reference {d.get('comparator_changed', {}).get('reference')}), (cell, chip, network) admissions removed "
            f"{d.get('chip_admissions_removed')}" for t, d in demo.items()))
    print("\nPer target (nominal, with host, lifetime floor):")
    for t, v in c["census"]["nominal|with host"]["lifetime_floor"].items():
        print(f"- {t}: {v.get('axes', '')}; priced cells {v.get('priced_cells')}, fill holds {v.get('holds_fill')}, "
              f"reference holds {v.get('holds_reference')}, both {v.get('holds_both')}, fill boundary "
              f"{len(v.get('boundary_fill', []))}, reference boundary {len(v.get('boundary_reference', []))}, "
              f"fill chip {v.get('fill_chip')}")
    print("\nHost-bound (the reduction count differs with and without host energy):", c.get("host_bound_targets"))


if os.path.exists(census_path):
    import hashlib

    c = json.load(open(census_path))
    sha = hashlib.sha256(open(census_path, "rb").read()).hexdigest()[:16]
    tag = "nominal|with host"
    demo = c.get("endurance", {}).get(tag, {})
    floored = c["census"][tag]["lifetime_floor"]
    n = sum(t.get("holds_both", 0) for t in floored.values())
    total = sum(t.get("priced_cells", 0) for t in floored.values())
    fill_b = sum(len(t.get("boundary_fill", [])) for t in floored.values())
    ref_b = sum(len(t.get("boundary_reference", [])) for t in floored.values())
    changed = ", ".join(f"{d['count']} on {t} ({d.get('joint_changed')} joint)" for t, d in sorted(demo.items()))
    removed = ", ".join(f"{d.get('chip_admissions_removed')} on {t}" for t, d in sorted(demo.items()))
    print(f"\n### Report text: endurance demonstration ({tag}; census.json sha256 {sha})\n")
    print(f"- The lifetime floor changes the admitted set in {changed} cells.")
    print(f"- It removes {removed} (cell, chip, network) admissions.")
    print(f"\n### Report text: CL5b acceptance row (census.json sha256 {sha})\n")
    print(f"Nominal, with host: holds in {n} of {total} priced cells; boundary cells fill {fill_b}, reference {ref_b}. "
          f"Endurance changes the admitted set in {changed} cells, and removes {removed} chip admissions "
          f"(census.json sha256 {sha}, provisional).")
