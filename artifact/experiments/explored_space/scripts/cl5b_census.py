""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

CNN = ("resnet18_w1q", "resnet20", "resnet56", "resnet110", "resnet164_preact", "vgg16_bn", "densenet_bc100")
REFERENCE = "resnet18_w1q"
WRITE_LIMITED = ("neurram", "hermes_64", "isaac_like")
ENDS = ("low", "nominal", "high")
GRID = 8
FLOOR_H = 43800.0


def grid(values):
    ""
    lo, hi = min(values), max(values)
    if lo <= 0 or hi <= 0:
        return None
    if lo == hi:
        return [lo] * GRID
    inner = [math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / (GRID - 1)) for i in range(1, GRID - 1)]
    return [lo] + [min(max(v, lo), hi) for v in inner] + [hi]


def priced(points, end, host):
    ""
    i = ENDS.index(end)
    out = defaultdict(list)
    for p in points:
        if p.get("tier") != "census" or p.get("network") not in CNN or p.get("refusal") or p.get("above_budget"):
            continue
        lat = ((p.get("latency_semantics") or {}).get("time_per_inference_s"))
        dual = p.get("energy_dual") or {}
        energy = dual.get("with_host_mj") if host else dual.get("without_host_mj")
        if lat is None and energy is None:
            continue
        life = ((p.get("write_law") or {}).get("terms") or {}).get("lifetime_hours")
        life_ok = True if life is None else life[2 - i] >= FLOOR_H
        key = p["target"] if p.get("variant") in (None, "host") else f"{p['target']}+{p['variant']}"
        chip = (str(p["budget"]), int(p["inference_batch_K"]), int(p.get("cores") or 0))
        out[(key, p["network"], chip)].append((None if lat is None else lat[i], None if energy is None else energy[i],
                                               life_ok))
    return out


def within(value, cap):
    ""
    return cap is None or (value is not None and value <= cap)


def admits(entries, lat_cap, en_cap, floor):
    return any(within(l, lat_cap) and within(e, en_cap) and (ok or not floor) for l, e, ok in entries)


def census(points, end, host, floor):
    table = priced(points, end, host)
    by_target = defaultdict(lambda: defaultdict(dict))
    for (target, net, chip), entries in table.items():
        by_target[target][chip][net] = entries
    out = {}
    for target, chips in sorted(by_target.items()):
        all_l = [l for c in chips.values() for es in c.values() for l, _, _ in es]
        all_e = [e for c in chips.values() for es in c.values() for _, e, _ in es]
        lats, ens = [l for l in all_l if l is not None], [e for e in all_e if e is not None]


        time_axis, energy_axis = len(lats) == len(all_l), len(ens) == len(all_e)
        lg = grid(lats) if time_axis else [None]
        eg = grid(ens) if energy_axis else [None]
        if not (time_axis or energy_axis) or lg is None or eg is None:
            out[target] = {"absent": "no axis is priced at every point, or a priced band touches zero"}
            continue
        declared = [c for c in chips if c[0] == "declared"]
        fill = max(declared, key=lambda c: c[1]) if declared else None
        cells = []
        for lc in lg:
            for ec in eg:
                joint = {n for c in chips for n, es in chips[c].items() if admits(es, lc, ec, floor)}
                if not joint:
                    continue
                fill_set = set() if fill is None else {n for n, es in chips[fill].items() if admits(es, lc, ec, floor)}
                ref_ok = [c for c in chips if REFERENCE in chips[c] and admits(chips[c][REFERENCE], lc, ec, floor)]
                ref = min(ref_ok, key=lambda c: (min((l if l is not None else e) for l, e, ok in chips[c][REFERENCE]
                                                     if within(l, lc) and within(e, ec)), c[2], -c[1])) if ref_ok else None
                ref_set = set() if ref is None else {n for n, es in chips[ref].items() if admits(es, lc, ec, floor)}
                cells.append({"latency_cap_s": lc, "energy_cap_mj": ec, "joint": sorted(joint),
                              "fill": {"chip": fill, "admits": sorted(fill_set), "holds": joint <= fill_set,
                                       "boundary": not fill_set},
                              "reference": {"chip": ref, "admits": sorted(ref_set), "holds": joint <= ref_set,
                                            "boundary": ref is None}})
        out[target] = {"latency_grid_s": lg, "energy_grid_mj": eg, "fill_chip": fill, "cells": cells,
                       "axes": ("time and energy" if time_axis and energy_axis else
                                "time only: energy refused by name at %d of %d points" % (len(all_e) - len(ens), len(all_e))
                                if time_axis else
                                "energy only: latency refused by name at %d of %d points" % (len(all_l) - len(lats), len(all_l))),
                       "priced_cells": len(cells),
                       "holds_fill": sum(c["fill"]["holds"] for c in cells),
                       "holds_reference": sum(c["reference"]["holds"] for c in cells),
                       "holds_both": sum(c["fill"]["holds"] and c["reference"]["holds"] for c in cells),
                       "boundary_fill": [i for i, c in enumerate(cells) if c["fill"]["boundary"]],
                       "boundary_reference": [i for i, c in enumerate(cells) if c["reference"]["boundary"]]}
    return out


def main() -> int:
    regime = json.load(open(sys.argv[1]))
    points = regime["points"]
    result = {"meta": {**regime["meta"], "cnn_roster": list(CNN), "reference_member": REFERENCE,
                       "write_limited": list(WRITE_LIMITED), "grid_points_per_axis": GRID,
                       "lifetime_floor_hours": FLOOR_H, "rule_text": __doc__}, "census": {}, "endurance": {}}
    sentences = []
    for end in ENDS:
        for host in (True, False):
            tag = f"{end}|{'with host' if host else 'without host'}"
            plain = census(points, end, host, floor=False)
            floored = census(points, end, host, floor=True)
            result["census"][tag] = {"no_lifetime_floor": plain, "lifetime_floor": floored}
            n = sum(t.get("holds_both", 0) for t in floored.values())
            total = sum(t.get("priced_cells", 0) for t in floored.values())
            line = (f"[{tag}] the reduction holds (both comparators admit every network the joint search admits) in "
                    f"{n} of {total} priced cells over CNN families under exact accumulation; boundary cells "
                    f"(a comparator admits nothing): fill {sum(len(t.get('boundary_fill', [])) for t in floored.values())}, "
                    f"reference {sum(len(t.get('boundary_reference', [])) for t in floored.values())} (provisional)")
            sentences.append(line)
            demo = {}
            for target in floored:
                if target.split("+")[0] not in WRITE_LIMITED or "cells" not in floored[target]:
                    continue
                a = {(c["latency_cap_s"], c["energy_cap_mj"]): c for c in plain.get(target, {}).get("cells", [])}
                b = {(c["latency_cap_s"], c["energy_cap_mj"]): c for c in floored[target]["cells"]}
                changed, comparators = [], {"fill": 0, "reference": 0}
                for key in sorted(set(a) | set(b), key=str):
                    ca, cb = a.get(key, {}), b.get(key, {})
                    ja, jb = ca.get("joint", []), cb.get("joint", [])
                    moved = {rule: (ca.get(rule, {}).get("admits", []), cb.get(rule, {}).get("admits", []))
                             for rule in ("fill", "reference")}
                    for rule, (x, y) in moved.items():
                        comparators[rule] += x != y
                    if ja != jb or any(x != y for x, y in moved.values()):
                        changed.append({"latency_cap_s": key[0], "energy_cap_mj": key[1], "joint_without_floor": ja,
                                        "joint_with_floor": jb,
                                        **{f"{r}_without_floor": x for r, (x, _) in moved.items()},
                                        **{f"{r}_with_floor": y for r, (_, y) in moved.items()}})

                removed = 0
                table_t = {k: v for k, v in priced(points, end, host).items() if k[0] == target}
                for c in b:
                    for (_, net, chip), entries in table_t.items():
                        removed += admits(entries, c[0], c[1], False) and not admits(entries, c[0], c[1], True)
                demo[target] = {"cells_changed_by_endurance": changed, "count": len(changed),
                                "joint_changed": sum(1 for c in changed if c["joint_without_floor"] != c["joint_with_floor"]),
                                "comparator_changed": comparators, "chip_admissions_removed": removed}
            result["endurance"][tag] = demo
    host_bound = []
    for end in ENDS:
        w, wo = result["census"][f"{end}|with host"]["lifetime_floor"], result["census"][f"{end}|without host"]["lifetime_floor"]
        for t in w:
            if w[t].get("holds_both") != wo.get(t, {}).get("holds_both"):
                host_bound.append({"end": end, "target": t, "with_host": w[t].get("holds_both"),
                                   "without_host": wo.get(t, {}).get("holds_both")})
    result["host_bound_targets"] = host_bound
    result["sentences"] = sentences
    with open(sys.argv[2], "w") as handle:
        json.dump(result, handle, indent=1, sort_keys=True, default=str)
    for line in sentences:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
