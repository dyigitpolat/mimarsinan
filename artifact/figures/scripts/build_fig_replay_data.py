""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
F3 = HERE.parent
PAPER = F3.parent
REPO = F3.parents[2]
sys.path.insert(0, str(PAPER / "figures_v2" / "scripts"))
import _ledger as LG

NAME = "fig_replay"
OLD = PAPER / "figures_v2" / "data" / "fig_exact.json"
COUNT_KEY = "gfivefuzz"
TWIN_KEY = "frozenfuzz"


def count() -> dict:
    v = LG.value(COUNT_KEY)
    fields = {}
    for s in LG.sources(COUNT_KEY):
        rec = LG.load_json(LG.source_path(s))
        name = s["json_path"].rsplit("/", 1)[-1]
        got = LG.json_at(rec, s["json_path"])
        if got != s["value"]:
            raise SystemExit(f"{COUNT_KEY}: the record's {s['json_path']} is {got}, the ledger quotes {s['value']}")
        fields[name] = {"value": got, "source": f"{s['path']}:{s['json_path']}", "key": COUNT_KEY}
    differ, total = fields["differ"]["value"], fields["N"]["value"]
    if v["display"] != f"{differ} of {total:,}":
        raise SystemExit(f"{COUNT_KEY}: display {v['display']!r} is not {differ} of {total:,}")
    twin = LG.value(TWIN_KEY)
    if twin["display"] != v["display"]:
        raise SystemExit(f"{TWIN_KEY} ({twin['display']}) and {COUNT_KEY} ({v['display']}) disagree")
    e = LG.entry(COUNT_KEY)
    return {"display": v, "differ": fields["differ"], "programs": fields["N"],
            "what": e["what"], "same_value_under": TWIN_KEY}


def example() -> dict:
    old = json.loads(OLD.read_text())
    ex = old["replay"]["schematic_example"]
    src = f"{LG.rel(OLD)}:replay/schematic_example"

    def val(k):
        return ex[k]["value"]

    layers, passes = val("layers"), val("passes")
    win, sends, hop = val("windows"), val("spike_steps"), val("hop_delay")
    first = passes[0]
    later = [n for p in passes[1:] for n in p]
    carried = [e for e in val("main_path") + [val("skip")] if e[0] in first and e[1] in later]
    stored = sorted({t for a, _ in carried for t in sends[a]})

    skip = val("skip")
    span = layers.index(skip[1]) - layers.index(skip[0])
    deliveries = [{"src": a, "sent": t, "dst": b, "at": t + hop, "skip": False}
                  for a, b in val("main_path") for t in sends[a]]
    deliveries += [{"src": skip[0], "sent": t, "dst": skip[1], "at": t + span * hop, "skip": True}
                   for t in sends[skip[0]]]
    arrive = val("stored_arrival_steps")
    for d in deliveries:
        assert win[d["dst"]][0] <= d["at"] <= win[d["dst"]][1], d
        if d["src"] in first and d["dst"] in later:
            assert d["at"] in arrive[f"to_{d['dst']}"], d
    assert sum(val("cores_per_layer")) > val("chip_cores")
    return {
        "status": ex["status"],
        "note": ex["note"],
        "mechanism_record": ex["mechanism_record"],
        "source": src,
        "layers": layers,
        "cores_single_pass": sum(val("cores_per_layer")),
        "cores_chip": val("chip_cores"),
        "steps": val("segment_steps"),
        "hop": hop,
        "windows": win,
        "spikes": sends,
        "passes": passes,
        "stored_steps": stored,
        "deliveries": deliveries,
        "output_layer": layers[-1],
    }


def main():
    data = {"id": NAME, "paths_relative_to": "the repository root",
            "built_by": LG.rel(Path(__file__)),
            "message": "A later pass replays the stored boundary spikes at their original steps, so a multi-pass "
                       "run equals the single-pass run.",
            "count": count(), "example": example()}
    out = LG.write(NAME, data, F3 / "data")
    print(out, data["count"]["display"]["display"], data["example"]["stored_steps"])


if __name__ == "__main__":
    main()
