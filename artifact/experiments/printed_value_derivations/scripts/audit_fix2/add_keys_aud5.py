""
import hashlib
import json
import re
import sys
from pathlib import Path

P = Path(sys.argv[1]).resolve()
TR = "records/12_tcad_readiness"
BR = f"{TR}/tiers/paper_v6/fix5/study_build/reread"
HERE = Path(__file__).resolve().parent
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def sha(rel):
    return hashlib.sha256((P / rel).read_bytes()).hexdigest()


sums = {}
for line in (P / BR / "SHA256SUMS").read_text().splitlines():
    if line.strip():
        h, p = line.split(None, 1)
        sums["records/" + p.strip().lstrip("*")] = h


def pinned(rel):
    h = sha(rel)
    if sums.get(rel) != h:
        sys.exit(f"add_keys_aud5.py: {rel} has sha256 {h[:16]}, not the re-read's SHA256SUMS {str(sums.get(rel))[:16]}: stopped")
    return h


def load(rel):
    return json.loads((P / rel).read_text())


def inp(rel, field, value, h=None, **kw):
    return dict(file=rel, sha256=h or pinned(rel), field=field, value=value, **kw)


out = {"what": "Derived values of step aud5:assemble (the second fix round's assembly, 2026-11-20), by group (study: the "
               "re-read of record on the seventh build of the study cost model; decl: declared settings). Each value "
               "names its inputs (file relative to the paper folder, sha256, field or line, value) and its arithmetic. "
               "Written by add_keys_aud5.py beside this file; never edited by hand.",
       "study": {}, "decl": {}}

design = f"{TR}/e2e/design.md"
dh = sha(design)
if not dh.startswith("0345ae09"):
    sys.exit(f"add_keys_aud5.py: {design} has sha256 {dh[:16]}, not the pinned 0345ae09: stopped")
lines = (P / design).read_text().splitlines()
for key, ln, label in (("budgethalf", 256, "B1 capacity schedule"), ("budgeteighth", 257, "B2 deep schedule")):
    text = lines[ln - 1]
    if label not in text:
        sys.exit(f"add_keys_aud5.py: {design}:{ln} does not name {label!r}: stopped")
    m = re.search(r"count ceil\(F / (\d+)\)", text)
    if not m:
        sys.exit(f"add_keys_aud5.py: {design}:{ln} holds no 'count ceil(F / n)': stopped")
    out["decl"][key] = dict(value=int(m.group(1)),
                            inputs=[inp(design, f"line {ln}", text.strip(), h=dh)],
                            arithmetic=f"the divisor n of '{label} | count ceil(F / n)' on line {ln}")

an_rel = f"{BR}/out/analysis.json"
an = load(an_rel)
ls = an["latency_statements"]["static_power"]["lowers"]
out["study"]["latstaticdown"] = dict(value=ls, words=WORDS.get(ls, str(ls)),
                                     inputs=[inp(an_rel, "latency_statements/static_power/lowers", ls)],
                                     arithmetic="the record's count of latency cells whose gain static power lowers")

kj_rel = f"{BR}/keys.json"
kl = load(kj_rel)["keys"]
kv = {k["name"]: k for k in kl}
cells = ["k", "h", "ronep", "ronel"]
ins = [inp(kj_rel, f"keys[name=hop{e}gain{c}]/value", kv[f"hop{e}gain{c}"]["value"],
           record_field=kv[f"hop{e}gain{c}"].get("record_field")) for e in ("one", "eight") for c in cells]
falls = sum(1 for c in cells if kv[f"hopeightgain{c}"]["value"] < kv[f"hoponegain{c}"]["value"])
rises = sum(1 for c in cells if kv[f"hopeightgain{c}"]["value"] > kv[f"hoponegain{c}"]["value"])
out["study"]["hopfalls"] = dict(value=falls, of=len(cells), words=WORDS.get(falls, str(falls)), inputs=ins,
                                arithmetic=f"of the {len(cells)} cells, gains lower at 8 ns than at 1 ns: {falls} "
                                           f"(higher: {rises}, hopraises' count)")

box = sorted((P / BR / "out" / "read").glob("box_*.json"))
counts, bins = [], []
for f in box:
    rel = str(f.relative_to(P))
    c48 = load(rel)["energy"]["cells48"]
    n = sum(1 for v in c48.values() if isinstance(v.get("gain"), (int, float)) and v["gain"] > 0)
    counts.append(n)
    bins.append(inp(rel, "<count:energy/cells48:gain>", n))
if len(box) != len(an["box"]):
    sys.exit(f"add_keys_aud5.py: {len(box)} box files, analysis.json's box holds {len(an['box'])}: stopped")
out["study"]["cellsgainbox"] = dict(least=min(counts), largest=max(counts), of=48, files=len(box), inputs=bins,
                                    arithmetic=f"least and largest, over the {len(box)} box files, of the count of "
                                               f"energy/cells48 entries with gain > 0")

p0 = load(f"{BR}/out/read/P0.json")
zr = load(f"{BR}/out/read/zero_ruled.json")
checks = dict(
    fotilesk=dict(value=p0["energy"]["cells"]["K64"]["FO_tiles"], file=f"{BR}/out/read/P0.json",
                  sha256=pinned(f"{BR}/out/read/P0.json"), field="energy/cells/K64/FO_tiles"),
    cellsgainzs=dict(value=[sum(1 for v in zr["energy"]["cells48"].values()
                                if isinstance(v.get("gain"), (int, float)) and v["gain"] > 0),
                            len(zr["energy"]["cells48"])], file=f"{BR}/out/read/zero_ruled.json",
                     sha256=pinned(f"{BR}/out/read/zero_ruled.json"),
                     field="<count:energy/cells48:gain>; <len:energy/cells48>"),
    fallcell=dict(value=kl[133]["value"], display=kl[133]["display"], name=kl[133]["name"], file=kj_rel,
                  sha256=pinned(kj_rel), field="keys/133/value; keys/133/display",
                  record_field=kl[133].get("record_field")))
if kl[133]["name"] != "fallcell":
    sys.exit("add_keys_aud5.py: keys.json's entry 133 is not fallcell: stopped")
if an["cells48_static_vs_zero"]["down"][0][0] != kl[133]["value"] or len(an["cells48_static_vs_zero"]["down"]) != 1:
    sys.exit("add_keys_aud5.py: fallcell is not analysis.json's one cell whose gain static power lowers: stopped")
out["read_by_local_keys"] = checks
(HERE / "derived_values.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for g in ("decl", "study"):
    for k, v in out[g].items():
        print(g, k, {x: v[x] for x in v if x not in ("inputs", "arithmetic")})
for k, v in checks.items():
    print("read by local_keys_r4.py:", k, v["value"], v.get("display", ""))
print("derived_values.json", hashlib.sha256((HERE / "derived_values.json").read_bytes()).hexdigest()[:16])
