""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
sys.path.insert(0, str(F3.parent / "figures_v2" / "scripts"))
import _ledger as LG

HOURS_PER_YEAR = 8760.0


def src(key: str, suffix: str) -> dict:
    ""
    hits = [s for s in LG.sources(key) if str(s.get("json_path", "")).endswith(suffix)]
    if len(hits) != 1:
        raise LG.LedgerError(f"{key!r}: {len(hits)} source fields end with {suffix!r}")
    s = hits[0]
    path = LG.source_path(s)
    in_record = LG.json_at(LG.load_json(path), s["json_path"])
    if in_record != s["value"]:
        raise LG.LedgerError(f"{key!r}: {s['json_path']} is {in_record!r} in the record, {s['value']!r} in the ledger")
    return {"value": s["value"], "key": key, "source": f"{LG.rel(path)} :: {s['json_path']}"}


def record_of(key: str):
    path = LG.source_path(LG.sources(key)[0])
    return path, LG.load_json(path)


def led(key: str) -> dict:
    v = LG.value(key)
    return {k: v[k] for k in ("key", "source", "value", "display", "unit", "class", "status", "form")}


def band(key: str, prefix: str) -> dict:
    ""
    out = {c: src(key, f"{prefix}/{c}") for c in ("low", "nominal", "high")}
    return out


def rate_line(key: str, prefix: str, batch) -> dict:
    ""
    w = band(key, f"{prefix}/exposed_write_s")
    return {
        "batch": batch,
        "write_s": w,

        "rate": {"low": 1.0 / w["high"]["value"], "nominal": 1.0 / w["nominal"]["value"],
                 "high": 1.0 / w["low"]["value"]},
        "derivation": "1 / write time per inference",
        "key": key,
    }


def same_at_every_batch(key: str, prefixes: list) -> dict:
    lines = [rate_line(key, p, src(key, f"{p}/K")["value"]) for p in prefixes]
    first = lines[0]["rate"]
    if any(ln["rate"] != first for ln in lines[1:]):
        raise LG.LedgerError(f"{key!r}: the write time differs between batches")
    one = dict(lines[0])
    one["batch"] = "every"
    one["batches"] = [ln["batch"] for ln in lines]
    return one


def main():
    floor = led("lifefloor")
    floor_h = src("lifefloor", "min_lifetime_hours")
    rate = led("rate")
    rate_hz = src("rate", "inference_rate_hz")
    years = floor_h["value"] / HOURS_PER_YEAR
    if abs(years - round(years)) > 1e-9:
        raise LG.LedgerError(f"the floor {floor_h['value']} h is not a whole number of years")


    p1, r1 = record_of("figonerowonelife")
    life1 = {
        "low": src("figonerowoneverdict", "rows/0/lifetime_hours/low"),
        "nominal": src("figonerowonelife", "rows/0/lifetime_hours/nominal"),
        "high": src("figonerowoneverdict", "rows/0/lifetime_hours/high"),
    }
    lines1 = [rate_line("figonerowoneunmet", f"priced_writes/rows/{i}",
                        src("figonerowoneunmet", f"priced_writes/rows/{i}/K")["value"]) for i in (0, 4, 8)]
    budget = r1["declaration"]["budget_attempts"][0]
    cores1 = src("attnsegments", "declaration/cores/0/count")
    if budget["cores"] != cores1["value"] or not budget["built"]:
        raise LG.LedgerError("row 1: the built core budget is not the declared core count")
    row1 = {
        "id": "sram_macro",
        "name": "SRAM macro",
        "tone": "accent",
        "fit": {
            "cores": cores1,
            "passes": src("attnsegments", "pass_timeline/packed/max_passes_per_segment"),
            "cores_per_tile": {"value": budget["fraction"],
                               "source": f"{LG.rel(p1)} :: declaration/budget_attempts/0/fraction"},
            "widest_segment_tiles": {"value": r1["declaration"]["widest_segment_tiles"],
                                     "source": f"{LG.rel(p1)} :: declaration/widest_segment_tiles"},
        },
        "lifetime_h": life1,
        "lifetime": led("figonerowonelife"),
        "verdict": led("figonerowoneverdict"),
        "rate_lines": lines1,
        "rate_print": led("figonerowonerate"),
        "rate_reading": led("figonerowoneunmet"),
    }


    p2, r2 = record_of("lifemixed")
    declared = r2["row"]["cores_declared"]
    tiles = r2["row"]["tiles_unbounded"]
    n_cores, n_tiles = sum(c["count"] for c in declared), sum(tiles.values())
    per_tile = {"value": n_cores / n_tiles,
                "source": f"{LG.rel(p2)} :: row/cores_declared/*/count over row/tiles_unbounded"}
    passes2 = src("figonerowtwopasses", "row/packed/max_passes_per_segment")
    mixed = "row/write_verdicts/neurram static + sram_cim_macro operand/lifetime_hours"
    allnvm = "row/write_verdicts/all neurram/lifetime_hours"
    row2 = {
        "id": "resistive_sram",
        "name": "Resistive + SRAM",
        "tone": "accent",
        "fit": {"cores_per_tile": per_tile, "passes": passes2},
        "lifetime_h": {"low": src("figonerowtwoverdict", f"{mixed}/low"),
                       "nominal": src("lifemixed", f"{mixed}/nominal"),
                       "high": src("figonerowtwoverdict", f"{mixed}/high")},
        "lifetime": led("lifemixed"),
        "verdict": led("figonerowtwoverdict"),
        "rate_lines": [same_at_every_batch("figonerowtwounmet", [f"declarations/0/rows/{i}" for i in (18, 19, 20)])],
        "rate_print": led("figonerowtworate"),
        "rate_reading": led("figonerowtwounmet"),
    }
    row3 = {
        "id": "resistive_only",
        "name": "Resistive only",
        "tone": "quiet",
        "fit": {"cores_per_tile": per_tile},
        "lifetime_h": band("lifeallnvm", allnvm),
        "lifetime": led("lifeallnvm"),
        "verdict": led("figonerowthreeverdict"),
        "rate_lines": [same_at_every_batch("figonerowthreeunmet", [f"declarations/0/rows/{i}" for i in (0, 1, 2)])],
        "rate_print": led("figonerowthreerate"),
        "rate_reading": led("figonerowthreeunmet"),
    }


    for row in (row1, row2, row3):
        lo, hi = row["lifetime_h"]["low"]["value"], row["lifetime_h"]["high"]["value"]
        reading = ("accepted" if lo >= floor_h["value"] else "refused" if hi < floor_h["value"]
                   else "dependent on the corner")
        if not str(row["verdict"]["value"]).startswith(reading):
            raise LG.LedgerError(f"{row['id']}: the band reads {reading!r}, the ledger {row['verdict']['value']!r}")

        row["verdict"]["label"] = {"accepted": "Accepted", "refused": "Refused",
                                   "dependent on the corner": "Corner-dependent"}[reading]

    data = {
        "figure": "fig_block",
        "label": "fig:block",
        "section": "VI-A",
        "message": "For the attention block, the write law (lifetime and sustainable rate), not the fit, "
                   "decides which chip description is accepted.",
        "floor": {"hours": floor_h, "years": years, "ledger": floor,
                  "label": f"Required life, {round(years)} years"},
        "rate": {"per_second": rate_hz, "ledger": rate, "label": "Declared rate"},
        "rows": [row1, row2, row3],
        "not_drawn": {
            "lifetime_at_larger_batches": {
                "why": "panel (a) draws the first description's lifetime at batch 1 (the ledger's key, the shortest "
                       "of its batches); the record holds the longer lifetimes of the larger batches",
                "rows": [{"batch": r1["priced_writes"]["rows"][i]["K"],
                          "hours": r1["priced_writes"]["rows"][i]["lifetime_hours"]["nominal"],
                          "source": f"{LG.rel(p1)} :: priced_writes/rows/{i}/lifetime_hours/nominal"}
                         for i in (4, 8)],
            },
            "energy_per_inference": {
                "why": "priced for the first chip description only; the other two carry the ledger's stated "
                       "boundary, so three comparable values do not exist",
                "rows": [led("figonerowonecost"), led("figonerowtwocost"), led("figonerowthreecost")],
            },
        },
    }
    out = LG.write("fig_block", data, F3 / "data")
    print(f"wrote {LG.rel(out)}")
    for row in data["rows"]:
        lt = row["lifetime_h"]
        print(f"  {row['name']:18s} lifetime {lt['low']['value']:.4g} / {lt['nominal']['value']:.4g} / "
              f"{lt['high']['value']:.4g} h  {row['verdict']['label']:17s} cores per tile "
              f"{row['fit']['cores_per_tile']['value']:.3g}")
        for ln in row["rate_lines"]:
            r = ln["rate"]
            print(f"      batch {ln['batch']!s:6s} rate {r['low']:.4g} / {r['nominal']:.4g} / {r['high']:.4g} per second")


if __name__ == "__main__":
    main()
