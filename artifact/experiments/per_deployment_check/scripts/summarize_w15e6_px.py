""
import collections
import json
import os
import re
import statistics
import sys

R = sys.argv[1]
SEMS = ("streamed_lif", "windowed_lif", "synchronized_lif", "cascaded_ttfs", "synchronized_ttfs")


def rows(path):
    return [json.loads(line) for line in open(path)] if os.path.exists(path) else []


def verdict(t1):
    if not t1:
        return "none"
    if t1.startswith("established on the program against the IR at unbounded geometry"):
        return "established (unbounded)"
    if t1.startswith("established on the program against the IR at the declared geometry"):
        return "established (declared)"
    if t1.startswith("conditional"):
        return "conditional: " + t1.split(":", 1)[1].split(";")[0].replace(" stay with the sampled run", "").strip()
    if t1.startswith("outside_scope"):
        return "outside scope"
    if t1.startswith("failed"):
        return "failed: " + t1.split(":", 1)[1].split(";")[0].strip()
    return t1.split(":")[0]


NOT_RUN = ("check_error", "window_build", "refused", "sampled", "error")


def outcome_class(cell, field):
    ""
    value = (cell or {}).get(field)
    if value is not None:
        return str(value)
    for why in NOT_RUN:
        text = (cell or {}).get(why)
        if why != field and isinstance(text, str) and text and text != "equal":
            cls = text.split(":")[0]
            return f"not run: {cls}" if why == "sampled" else f"not run: {why} {cls}"
    return "absent"


def dump_whole(obj, path):
    ""
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if not isinstance(k, str):
                    raise SystemExit(f"REFUSED: KEY_NOT_TEXT: {k!r} in the summary")
                keys(v)
        elif isinstance(o, list):
            for v in o:
                keys(v)
    keys(obj)
    text = json.dumps(obj, indent=1, sort_keys=True, default=str)
    with open(path, "w") as f:
        f.write(text)


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def rules(row):
    ""
    out = []
    for cond, texts in (row.get("violations") or {}).items():
        if (row.get("statuses") or {}).get(cond) == "failed":
            for t in texts:
                m = re.match(r"\[(\S+) ([^\]]+)\]", t)
                if m:
                    out.append(f"{m.group(1)} {m.group(2)}")
    return sorted(set(out))


def row_class(r):
    if "theorem1" in r:
        return verdict(r["theorem1"])
    return "refused" if "refused" in r else ("error" if "error" in r else "skipped")


summary = {}
cfg = rows(f"{R}/configs/rows.jsonl")
templates = [r for r in cfg if r["config"].startswith("t0_")]
fixtures = [r for r in cfg if not r["config"].startswith("t0_")]
print("## Tier-0 templates and fixtures at random init (the IR at unbounded geometry given)\n")
print("| set | rows | Theorem 1 verdicts | Proposition 2 verdicts | failures by condition | unknown or unread forms | "
      "median check / compile | median check s |")
print("|---|---|---|---|---|---|---|---|")
for name, sel in (("templates", templates), ("fixtures", fixtures)):
    t1 = collections.Counter(row_class(r) for r in sel)
    p2 = collections.Counter(verdict(r.get("proposition2")) if "proposition2" in r else row_class(r) for r in sel)
    fails = collections.Counter(c for r in sel for c in (r.get("failures") or {}))
    forms = sum(1 for r in sel if r.get("unknown_forms"))
    ratio = med([r["w15_check_s"] / r["compile_s"] for r in sel if r.get("compile_s") and "w15_check_s" in r])
    print(f"| {name} | {len(sel)} | {dict(t1)} | {dict(p2)} | {dict(fails)} | {forms} | {ratio} | "
          f"{med([r.get('w15_check_s') for r in sel])} |")
    summary[name] = {"rows": len(sel), "theorem1": dict(t1), "proposition2": dict(p2), "failures": dict(fails),
                     "unknown_form_rows": forms, "median_check_over_compile": ratio}
print()
print("| config | semantics | Theorem 1 | Proposition 2 | failed rules | ir-tiling | passes | compile s | check s | "
      "ir-tiling s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in cfg:
    print(f"| {r['config']} | {r.get('semantics')} | {row_class(r)} | "
          f"{verdict(r.get('proposition2')) if 'proposition2' in r else ''} | {'; '.join(rules(r))} | "
          f"{r.get('ir_tiling')} | {r.get('passes')} | {r.get('compile_s')} | {r.get('w15_check_s')} | "
          f"{r.get('ir_tiling_s')} |")

fz = rows(f"{R}/fuzz_base/rows.jsonl")
cells = [(s, r[s]) for r in fz for s in SEMS if s in r]
print(f"\n## Fuzz grid (G5's generator, base family): {len({r['seed'] for r in fz})} seeds, {len(fz)} (seed, budget) "
      f"rows, {len(cells)} checks\n")
print("| semantics | checks | Theorem 1 verdicts | failures by condition | sampled (own identity) | "
      "sampled (unbounded identity) | established and differing | median check ms |")
print("|---|---|---|---|---|---|---|---|")
for sem in SEMS:
    mine = [c for s, c in cells if s == sem]
    t1 = collections.Counter(verdict(c.get("t1")) for c in mine)
    fails = collections.Counter(k for c in mine for k in (c.get("fail") or {}))
    sampled = collections.Counter(outcome_class(c, "sampled") for c in mine)
    unb = collections.Counter(outcome_class(c, "sampled_unbounded") for c in mine)
    bad = sum(1 for c in mine if verdict(c.get("t1")).startswith("established") and
              (c.get("sampled") != "equal" or c.get("sampled_unbounded") != "equal"))
    print(f"| {sem} | {len(mine)} | {dict(t1)} | {dict(fails)} | {dict(sampled)} | {dict(unb)} | {bad} | "
          f"{med([1000 * c['check_s'] for c in mine if 'check_s' in c])} |")
    summary[f"fuzz_{sem}"] = {"checks": len(mine), "theorem1": dict(t1), "failures": dict(fails),
                              "sampled": dict(sampled), "sampled_unbounded": dict(unb), "established_and_differing": bad}
built = sum(1 for r in fz if "passes" in r)
print(f"\nbuilt programs {built}, multi-pass {sum(1 for r in fz if r.get('multipass'))}, refused "
      f"{sum(1 for r in fz if any(k.startswith('refused') for k in r))}, vehicle errors "
      f"{sum(1 for r in fz if 'vehicle_error' in r)}, check errors {sum(1 for _s, c in cells if 'check_error' in c)}; "
      f"float IR streamed {dict(collections.Counter(verdict(r['float_ir_streamed'].get('t1')) for r in fz if 'float_ir_streamed' in r))}")

print("\n## Unit programs\n")
print("| program | build | passes (resident, pinned) | Theorem 1 | Proposition 2 | failures | ir-tiling | compile s | "
      "check s | check / compile |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name in ("lanes_r18/rows.jsonl", "vit/softmax.jsonl", "vit/relu.jsonl", "mux/rows.jsonl", "w03/rows.jsonl",
             "lenet/rows.jsonl"):
    for r in rows(f"{R}/{name}"):
        if "build" not in r:
            print(f"| {r.get('vehicle')} | {r.get('budget', '')} | | {r.get('refused') or r.get('unbounded_ir_note', '')} "
                  "| | | | | | |")
            continue
        ratio = round(r["check_s"] / r["compile_s"], 3) if r.get("compile_s") and r.get("check_s") else None
        print(f"| {r.get('vehicle')} | {r.get('build')} | {r.get('passes')} ({r.get('resident_passes')}, "
              f"{r.get('pinned_passes')}) | {row_class(r)} | "
              f"{verdict(r.get('proposition2')) if 'proposition2' in r else ''} | {r.get('failures')} "
              f"{'; '.join(rules(r))} | {(r.get('statuses') or {}).get('ir-tiling', r.get('ir_tiling'))} | "
              f"{r.get('compile_s')} | {r.get('check_s')} | {ratio} |")
        summary.setdefault("units", []).append({k: r.get(k) for k in ("vehicle", "build", "passes", "resident_passes",
                                                                      "theorem1", "proposition2", "failures",
                                                                      "compile_s", "check_s", "error", "refused")})
dump_whole(summary, f"{R}/summary.json")
