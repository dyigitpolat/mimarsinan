""
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PAPER = HERE.parent
sys.path.insert(0, str(PAPER / "figures_v2" / "scripts"))
import _ledger as L

NAME = "fig_space"
F2_DATA = PAPER / "figures_v2" / "data" / "fig_space_toolchain.json"


CHIP_LABEL = {
    "generic_estimated_22nm": "Generic 22 nm",
    "hermes_64": "HERMES-64",
    "neurram": "NeuRRAM",
    "isaac_like": "ISAAC-like",
    "sram_cim_macro": "SRAM macro",
    "study_compass": "COMPASS-geometry study",
    "study_isaac_like": "ISAAC-like study",


    "study_vr_like": "Multi-bit-cell study",
    "loihi": "Loihi",
    "truenorth": "TrueNorth",
    "odin": "ODIN",
}


KIND = {"ViT.T-VAL8-INT.b": ("host attention", "rbcountvitlite")}
STATUS = {"priced": "estimated", "above budget": "over_budget", "refused": "refused"}
KEYS = ("clfivepoints", "clfiverefused", "clfiveabove", "clfivechips", "clfivenetworks", "dseconfigs",
        "dsenetworks", "dsechips", "lifefloor", "planefacts", "planesamplednets", "figfourchecked", "eimages")


def number(v) -> float:
    return float(str(v).replace(",", "").split()[0])


def front(points):
    ""
    out, best = [], math.inf
    for t, e in sorted(points):
        if e < best:
            out.append([t, e])
            best = e
    return out


def orders(vals) -> float:
    return math.log10(max(vals) / min(vals))


def two_digits(v: float) -> str:
    ""
    r = float(f"{v:.2g}")
    return (f"{int(round(r)):,}" if r >= 10 else f"{r:g}") + "\u00d7"


def factor(lo: float, hi: float) -> dict:
    return {"value": hi / lo, "display": two_digits(hi / lo), "of": [lo, hi],
            "rule": "greatest over least, two significant digits"}


FLOOR_CHIP = "neurram"
VERDICT = {"admitted": "accepted", "straddles": "corner_dependent", "inadmissible": "ruled_out"}
VERDICT_LABEL = {"accepted": "Accepted", "corner_dependent": "Corner-dependent",
                 "ruled_out": "Ruled out"}


def floor_blocks(vals, floor_h):
    ""
    path = L.record("clfivepoints", "out/regime_map.json")
    rm = L.load_json(path)
    rrel, rsha = L.rel(path), L.sha256(path)
    meta_floor = float(rm["meta"]["assumptions"]["min_lifetime_hours"])
    rate = float(rm["meta"]["assumptions"]["inference_rate_hz"])
    if meta_floor != floor_h:
        raise L.LedgerError(f"service-life floor: the record states {meta_floor} h, the ledger {floor_h} h")
    if len(rm["points"]) != number(vals["clfivepoints"]["value"]):
        raise L.LedgerError(f"regime map: {len(rm['points'])} points against clfivepoints")
    F = "points[target={chip}]"
    per = {}
    for p in rm["points"]:
        terms = (p.get("write_law") or {}).get("terms") or {}
        life = terms.get("lifetime_hours")
        if not life:
            continue
        lo, nom, hi = (float(v) for v in life)
        if not lo <= nom <= hi:
            raise L.LedgerError(f"{p['target']}: a lifetime triple out of order {life}")
        v = [x for x in p["write_law"]["verdicts"] if x["name"] == "lifetime_admissible"]
        if len(v) != 1 or v[0]["verdict"] not in VERDICT:
            raise L.LedgerError(f"{p['target']}: no single lifetime_admissible verdict: {p['write_law']['verdicts']}")
        verdict = VERDICT[v[0]["verdict"]]

        want = "accepted" if lo >= floor_h else ("ruled_out" if hi < floor_h else "corner_dependent")
        if verdict != want:
            raise L.LedgerError(f"{p['target']}: verdict {verdict} against the lifetime triple {life}")
        per.setdefault(p["target"], []).append({
            "K": int(p["inference_batch_K"]), "life": (lo, nom, hi), "verdict": verdict,
            "network": p["network"], "budget": p["budget"], "cores": p.get("cores"), "passes": p.get("passes"),
            "least_batch": (p.get("k_star") or {}).get("k_star_census_batch")})

    facts = {}
    for chip in sorted(per, key=lambda c: CHIP_LABEL[c]):
        ps = per[chip]
        under = [q for q in ps if q["life"][1] < floor_h]
        corner_only = [q for q in ps if q["life"][1] >= floor_h and q["life"][0] < floor_h]
        facts[chip] = {
            "label": CHIP_LABEL[chip], "drawn": chip == FLOOR_CHIP,
            "points_with_a_lifetime_reading": len(ps),
            "under_the_floor_at_the_nominal_corner": len(under),
            "under_the_floor_at_some_corner_only": len(corner_only),
            "under_the_floor_at_every_corner": sum(q["verdict"] == "ruled_out" for q in ps),
            "verdicts": {k: sum(q["verdict"] == k for q in ps) for k in VERDICT.values()},
            "least_lifetime_under_the_floor_h": min((q["life"][1] for q in under), default=None),
            "greatest_lifetime_under_the_floor_h": max((q["life"][1] for q in under), default=None),
            "by_batch": {str(K): {"points": sum(q["K"] == K for q in ps),
                                  "under_the_floor_at_the_nominal_corner": sum(q["K"] == K for q in under),
                                  **{k: sum(q["K"] == K and q["verdict"] == k for q in ps) for k in VERDICT.values()}}
                         for K in sorted({q["K"] for q in ps})},
            "networks": len({q["network"] for q in ps}),
            "source": {
                "record": rrel, "sha256": rsha[:16],
                "points_with_a_lifetime_reading": f"{F.format(chip=chip)}.write_law.terms.lifetime_hours not null",
                "under_the_floor_at_the_nominal_corner": "write_law.terms.lifetime_hours[1] < "
                                                         "meta.assumptions.min_lifetime_hours",
                "under_the_floor_at_some_corner_only": "lifetime_hours[1] >= the floor and lifetime_hours[0] < the floor",
                "under_the_floor_at_every_corner": "write_law.verdicts[name=lifetime_admissible].verdict = "
                                                   "'inadmissible' (lifetime_hours[2] < the floor)",
                "verdicts": "write_law.verdicts[name=lifetime_admissible].verdict: admitted = accepted, straddles = "
                            "corner-dependent, inadmissible = ruled out",
                "least_lifetime_under_the_floor_h": "min of lifetime_hours[1] under the floor",
                "greatest_lifetime_under_the_floor_h": "max of lifetime_hours[1] under the floor",
                "by_batch": "the same counts per inference_batch_K",
            },
        }


    ps = per[FLOOR_CHIP]
    batches = sorted({q["K"] for q in ps})
    by_k = {K: sorted(q["life"][1] for q in ps if q["K"] == K) for K in batches}
    base = by_k[batches[0]]

    for K in batches[1:]:
        if len(by_k[K]) != len(base) or any(abs(b / (K / batches[0] * a) - 1.0) > 1e-9 for a, b in zip(base, by_k[K])):
            raise L.LedgerError(f"{FLOOR_CHIP}: lifetimes at batch {K} are not {K}/{batches[0]} of those at batch "
                                f"{batches[0]}")
    first = [q for q in ps if q["K"] == batches[0]]
    for q in first:
        if q["least_batch"] != math.ceil(floor_h * batches[0] / q["life"][1] - 1e-9):
            raise L.LedgerError(f"{FLOOR_CHIP}: least batch {q['least_batch']} against lifetime {q['life'][1]}")
    lines = Counter(q["life"][1] for q in first)


    top = batches[-1] / batches[0]
    cross = sorted((q for q in first if q["life"][1] < floor_h <= q["life"][1] * top),
                   key=lambda q: (q["life"][1], q["cores"] or 0, q["network"], str(q["budget"])))
    if not cross:
        raise L.LedgerError(f"{FLOOR_CHIP}: no configuration crosses the floor between batch {batches[0]} and "
                            f"{batches[-1]}")
    ex = cross[len(cross) // 2]
    series = {k: {"label": VERDICT_LABEL[k], "batch": [], "lifetime_h": []} for k in VERDICT.values()}
    for q in ps:
        series[q["verdict"]]["batch"].append(q["K"]); series[q["verdict"]]["lifetime_h"].append(q["life"][1])
    fc = facts[FLOOR_CHIP]
    panel = {
        "chip": FLOOR_CHIP, "chip_label": CHIP_LABEL[FLOOR_CHIP],
        "choice": "of the three chip descriptions with points under the floor (" + ", ".join(
            f"{facts[c]['label']} {facts[c]['under_the_floor_at_the_nominal_corner']} of "
            f"{facts[c]['points_with_a_lifetime_reading']}" for c in facts
            if facts[c]["under_the_floor_at_the_nominal_corner"]) + "), the one where the greatest explored batch "
            "takes the most points over the floor",
        "points": len(ps), "under_the_floor": fc["under_the_floor_at_the_nominal_corner"],
        "batches": batches, "by_batch": fc["by_batch"], "verdicts": fc["verdicts"],
        "lifetime_h": [min(q["life"][1] for q in ps), max(q["life"][1] for q in ps)],
        "series": series,
        "connectors": {"rule": f"lifetime at batch K = K times the lifetime at batch {batches[0]} (checked on every "
                               f"point of the record)", "lifetime_h_at_least_batch": sorted(lines),
                       "configurations": [lines[k] for k in sorted(lines)]},
        "crossing": {"configurations": len(first), "under_at_least_batch": sum(q["life"][1] < floor_h for q in first),
                     "over_at_greatest_batch": sum(q["life"][1] * top >= floor_h for q in first),
                     "crossing_between": len(cross)},
        "example": {"network": ex["network"], "budget": ex["budget"], "cores": ex["cores"], "passes": ex["passes"],
                    "batch": batches, "lifetime_h": [ex["life"][1] * K / batches[0] for K in batches],
                    "least_batch": ex["least_batch"],
                    "rule": "the configuration of median lifetime among those under the floor at the least batch and "
                            "over it at the greatest (nominal corner)",
                    "source": f"{rrel} points[target={FLOOR_CHIP}, network={ex['network']}, budget={ex['budget']}, "
                              f"inference_batch_K={batches[0]}]: write_law.terms.lifetime_hours[1], "
                              f"k_star.k_star_census_batch"},
        "rate_hz": rate,
        "source": f"{rrel} (sha256 {rsha[:16]}) points[target={FLOOR_CHIP}]: inference_batch_K, "
                  f"write_law.terms.lifetime_hours[1] (nominal corner), write_law.verdicts[name=lifetime_admissible]",
    }

    best = max((c for c in facts if facts[c]["under_the_floor_at_the_nominal_corner"]),
               key=lambda c: (facts[c]["by_batch"][str(batches[-1])]["points"]
                              - facts[c]["by_batch"][str(batches[-1])]["under_the_floor_at_the_nominal_corner"]))
    if best != FLOOR_CHIP:
        raise L.LedgerError(f"panel (d) draws {FLOOR_CHIP}, the record's choice by the stated rule is {best}")
    for c in facts:
        if facts[c]["drawn"]:
            facts[c]["example_least_batch"] = {"value": ex["least_batch"], "network": ex["network"],
                                               "budget": ex["budget"],
                                               "source": "k_star.k_star_census_batch of the example point "
                                                         "(floor_panel.example.source)"}
            facts[c]["least_batch_over_the_drawn_configurations"] = {
                "least": min(q["least_batch"] for q in first), "greatest": max(q["least_batch"] for q in first),
                "source": f"{F.format(chip=c)}.k_star.k_star_census_batch at inference_batch_K = {batches[0]}"}
    return panel, facts, {"path": rrel, "sha256": rsha[:16]}


def main():
    vals = L.values(*KEYS)
    for k, v in vals.items():
        if v["form"] == "pending" or v["value"] is None:
            raise L.LedgerError(f"{k} is pending: the figure has no value to draw")


    summary_path = L.record("clfivepoints", "out/summary.json")
    summary = L.load_json(summary_path)
    facts_path = L.record("clfivepoints", "out/plane_facts.json")
    facts = L.load_json(facts_path)
    srel, frel = L.rel(summary_path), L.rel(facts_path)
    chips = {}
    for name, n in summary["status_per_target"].items():
        chip, status = name.split("|")
        row = chips.setdefault(chip, {"id": chip, "label": CHIP_LABEL[chip], "estimated": 0, "over_budget": 0,
                                      "refused": 0, "source": f"{srel} status_per_target/{chip}|*"})
        row[STATUS[status]] += int(n)
    built = Counter()
    sampled = Counter()
    for r in facts["rows"]:
        sampled[r["target"]] += 1
        if ((r.get("sampled") or {}).get("planes") or {}).get("agree") is True:
            built[r["target"]] += 1


    m = re.fullmatch(r"(\d+) of (\d+)", vals["planefacts"]["value"].strip())
    shown = vals["planefacts"]["display"]
    if not m or "0 disagree" not in shown or "0 refused by the builder" not in shown:
        raise L.LedgerError(f"planefacts: the display {shown!r} does not state that every sampled point that is "
                            f"not built is a named refusal before a build")
    if sum(sampled.values()) != int(m.group(2)):
        raise L.LedgerError(f"sampled points: the record gives {sum(sampled.values())}, the ledger {m.group(2)}")
    for chip, row in chips.items():
        row["total"] = row["estimated"] + row["over_budget"] + row["refused"]
        row["samples"] = sampled[chip]
        row["samples_equal"] = built[chip]
        row["samples_source"] = f"{frel} rows[target={chip}].sampled.planes.agree"
    rows = sorted(chips.values(), key=lambda r: (-r["estimated"], r["label"]))
    total = sum(r["total"] for r in rows)
    checks = {
        "points": (total, number(vals["clfivepoints"]["value"])),
        "refused": (sum(r["refused"] for r in rows), number(vals["clfiverefused"]["value"])),
        "over_budget": (sum(r["over_budget"] for r in rows), number(vals["clfiveabove"]["value"])),
        "chips": (len(rows), number(vals["clfivechips"]["value"])),
        "samples_equal": (sum(r["samples_equal"] for r in rows), number(vals["planefacts"]["value"])),
    }
    for what, (got, want) in checks.items():
        if got != want:
            raise L.LedgerError(f"{what}: the records give {got}, the ledger {want}")


    F2 = json.loads(F2_DATA.read_text())
    f2rel = L.rel(F2_DATA)
    meta = L.meta()
    if F2["ledger_meta"]["ledger_sha256"] != meta["ledger_sha256"]:
        print(f"note: {f2rel} was built at ledger {F2['ledger_meta']['ledger_sha256'][:12]}, the ledger now is "
              f"{meta['ledger_sha256'][:12]}; the counts below are checked against the ledger of now")
    member_map = L.record("clfivepoints", "out/member_map.json")
    carried = [p for p in F2["points"] if (p.get("accuracy") or {}).get("rule") == "carried"]
    if len(carried) != number(vals["dseconfigs"]["value"]):
        raise L.LedgerError(f"carried configurations: the data gives {len(carried)}, the ledger "
                            f"{vals['dseconfigs']['value']}")
    chips_carried = sorted({p["chip"] for p in carried})
    if len(chips_carried) != number(vals["dsechips"]["value"]):
        raise L.LedgerError(f"chips of the carried configurations: {chips_carried} against dsechips")
    images = int(number(vals["eimages"]["value"]))
    floor_h = number(vals["lifefloor"]["value"])
    networks = []
    for nid, net in F2["networks"].items():
        pts = [p for p in carried if p["network"] == nid]
        if not pts:
            continue
        acc = {(p["accuracy"]["count"], p["accuracy"]["images"]) for p in pts}
        if len(acc) != 1:
            raise L.LedgerError(f"{nid}: more than one carried count {acc}")
        count, n_img = acc.pop()
        dep = L.value(net["deployed"]["key"])
        if int(number(dep["value"])) != count or n_img != images:
            raise L.LedgerError(f"{nid}: carried count {count} of {n_img} against {dep['key']} = {dep['value']} "
                                f"of {images}")
        pts.sort(key=lambda p: (p["latency_s"], p["energy_mj"]))
        t = [p["latency_s"] for p in pts]
        e = [p["energy_mj"] for p in pts]
        life = [p["lifetime_h"] for p in pts]
        with_life = [v for v in life if v is not None]
        checked = [{"budget": r["budget"], "K": r["K"], "time_s": r["latency_s"], "energy_mj": r["energy_mj"],
                    "count": r["count"], "images": r["images"], "row": r["row"]}
                   for r in F2["checked_points"]["data"] if r["member"] == net["member"]
                   and r["chip"] in chips_carried and r["count"] == count]
        label = net["label"]
        if nid in KIND:
            kind, kind_key = KIND[nid]
            if dep["key"] != kind_key:
                raise L.LedgerError(f"{nid}: named the {kind} by {kind_key}, its accuracy key is {dep['key']}")
            label = f"{label}, {kind}"
        networks.append({
            "id": nid, "label": label, "chip": chips_carried[0], "chip_label": CHIP_LABEL[chips_carried[0]],
            "accuracy": {"count": count, "images": n_img, "pct": round(100.0 * count / n_img, 2),
                         "key": dep["key"], "source": dep["source"],
                         "rule": "carried from the deployments of record that agree on every property that "
                                 "changes the computation"},
            "configurations": len(pts),
            "time_s": t, "energy_mj": e, "lifetime_h": life,
            "smaller_chip": [bool(p["x_packed"] > 1.0) for p in pts],
            "batch": [p["K"] for p in pts],
            "passes": [p["passes"] for p in pts],
            "front": front(list(zip(t, e))),
            "span": {"time_orders": round(orders(t), 2), "energy_orders": round(orders(e), 2),
                     "time_factor": factor(min(t), max(t)), "energy_factor": factor(min(e), max(e)),
                     "floor_margin_factor": factor(floor_h, min(with_life)),
                     "time_s": [min(t), max(t)], "energy_mj": [min(e), max(e)],
                     "lifetime_h": [min(with_life), max(with_life)],
                     "lifetime_configurations": len(with_life),
                     "no_cell_written": len(life) - len(with_life),
                     "floor_margin_orders": round(math.log10(min(with_life) / floor_h), 2)},
            "checked": checked,

            "checked_summary": {"deployments": len({r["row"] for r in checked}),
                                "batches": sorted({r["K"] for r in checked}),
                                "cost_points": len(checked),
                                "of_deployments": int(number(vals["figfourchecked"]["value"])),
                                "key": "figfourchecked"},
            "source": f"{f2rel} points[network={nid}, accuracy.rule=carried] (from {L.rel(member_map)} "
                      f"points[*].write_law.terms, nominal corner; time per inference = batch period / K)",
            "checked_source": "ledger:figfourchecked",
        })
    networks.sort(key=lambda n: -n["configurations"])
    years = re.search(r"\((\d+) years\)", vals["lifefloor"]["display"])
    if not years:
        raise L.LedgerError(f"lifefloor: no year count in the display {vals['lifefloor']['display']!r}")
    floor = dict(vals["lifefloor"], hours=floor_h, years=int(years.group(1)))

    floor_panel, floor_facts, regime_src = floor_blocks(vals, floor_h)

    gaps = [
        "No area estimate per configuration: no record of the exploration states an area per point, so no area "
        "axis is drawn.",
        "Built and checked cost points exist for ResNet-18 w1 alone among the three networks (key "
        "figfourchecked holds none for DenseNet-BC k12 or ViT-Lite-7/4): their panels hold estimates only. "
        "The diamonds are cost points of deployments of record, one per batch size, not one deployment each.",
        "Lifetime is stated only where a cell is written per inference: "
        + ", ".join(f"{n['label']} {n['span']['lifetime_configurations']} of {n['configurations']}"
                    for n in networks) + " configurations; the built deployments state none.",
        "The checked accuracy carries on one chip description (key dsechips); the other chip descriptions of "
        "panel (a) hold random-weight points with cost alone.",
    ]
    data = {
        "figure": NAME, "label": "fig:space", "section": "VI-A",
        "built_by": L.rel(Path(__file__)),
        "built_at": subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip(),
        "ledger_meta": meta,
        "ledger": vals,
        "chips": rows,
        "chips_total": {"points": total, "key": "clfivepoints", "display": vals["clfivepoints"]["display"]},
        "samples_per_chip": {"value": int(number(vals["planesamplednets"]["value"])), "key": "planesamplednets"},
        "samples_total": {"equal": int(m.group(1)), "sampled": int(m.group(2)),
                          "refused_before_build": int(m.group(2)) - int(m.group(1)),
                          "display": vals["planefacts"]["value"], "key": "planefacts"},
        "networks": networks,
        "floor": floor,
        "floor_panel": floor_panel,
        "floor_facts": floor_facts,
        "sources": {
            "summary": {"path": srel, "sha256": L.sha256(summary_path)[:16]},
            "plane_facts": {"path": frel, "sha256": L.sha256(facts_path)[:16]},
            "regime_map": regime_src,
            "carried_points": {"path": f2rel, "sha256": L.sha256(F2_DATA)[:16],
                               "built_from": F2["source"]},
        },
        "data_gaps": gaps,
    }
    out = HERE / "data" / f"{NAME}.json"
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {L.rel(out)} ({out.stat().st_size / 1e3:.0f} kB): {len(rows)} chip descriptions, {total} points; "
          + "; ".join(f"{n['label']} {n['configurations']} configurations at {n['accuracy']['pct']}%, "
                      f"{len(n['checked'])} built and checked cost points" for n in networks))


if __name__ == "__main__":
    main()
