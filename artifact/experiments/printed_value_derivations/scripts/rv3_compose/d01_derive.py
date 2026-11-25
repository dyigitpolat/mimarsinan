""
import hashlib
import json
import sys

P = "/work/research/tcad_paper"
TRR = "records/12_tcad_readiness"
FZ = TRR + "/frozen_sources"
W48 = FZ + "/f_w48_deployed_t1_20261121T003451/records/" + TRR + "/e2e/w48/f_w48/644b4c3bc5d7/run/w48_pairs.json"
ROWS = FZ + "/f_w48_deployed_t1_20261121T003451/records/" + TRR + "/e2e/routeb/rows_r20/"
ROWB0 = FZ + "/routeb_tint_supplement_t1_20261121T001425/records/" + TRR + "/e2e/routeb/rows_r20/C1.T-INT.b.B0.256x256/result.json"
W38 = FZ + "/w38_imagenet_t1_20261120T224345/records/" + TRR + "/imagenet/records/63d7c3274503/"
P0 = TRR + "/tiers/paper_v6/fix5/study_build/reread/out/read/P0.json"
PINS = {W48: "9a99f7e2ab985c20", ROWS + "C1.T-INT.b.B5.256x64/result.json": "56e16c7a1dcc3038",
        ROWS + "C1.T-INT.b.B5m.256x64/result.json": "599565eae5889368", ROWB0: "bb3461b6acb314fd",
        P0: "134c2343da0146b1"}
GROUPS = ("resnet50.neurram.W", "resnet18.neurram.W", "mobilenet_v2.neurram.W", "densenet121.neurram.W",
          "vit_b_16.neurram.W")
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(rel):
    d = sha(f"{P}/{rel}")
    if rel in PINS and not d.startswith(PINS[rel]):
        sys.exit(f"d01: {rel} has sha256 {d[:16]}, not the pinned {PINS[rel]}: stopped")
    with open(f"{P}/{rel}", encoding="utf-8") as f:
        return json.load(f), d


def inp(rel, digest, field, value):
    return {"file": rel, "sha256": digest, "field": field, "value": value}


def half_up(x):
    return int(x + 0.5)


def one_copy():
    w, dw = load(W48)
    arms = w["members"]["C1"]["arms"]
    a, b = arms["B5"], arms["B5m@8192"]
    if a["row"] != "C1.T-INT.b.B5.256x64" or b["row"] != "C1.T-INT.b.B5m.256x64":
        sys.exit(f"d01: the arms' rows are {a['row']} and {b['row']}: stopped")
    ca = a["declared"]["cores"][0]; cb = b["declared"]["cores"][0]
    if (ca["count"], ca["max_axons"], ca["max_neurons"]) != (cb["count"], cb["max_axons"], cb["max_neurons"]):
        sys.exit("d01: B5 and B5m declare different cores: stopped")
    i_pass, i_wr, i_lat, i_area, i_out, ratios, passes, written = [], [], [], [], [], {}, set(), set()
    for K in ("K1", "K16", "K64"):
        x, y = a["candidate"]["shipped|" + K], b["candidate"]["shipped|" + K]
        if x["area_rule_mm2"][1] != y["area_rule_mm2"][1]:
            sys.exit(f"d01: {K}: the two areas differ: stopped")
        ratios[K] = x["e2e_latency_s"][1] / y["e2e_latency_s"][1]
        passes.add(y["passes"]); written.add(y["written_passes"])
        for arm, c in (("B5", x), ("B5m@8192", y)):
            i_lat.append(inp(W48, dw, f"members/C1/arms/{arm}/candidate/shipped|{K}/e2e_latency_s/1", c["e2e_latency_s"][1]))
            i_area.append(inp(W48, dw, f"members/C1/arms/{arm}/candidate/shipped|{K}/area_rule_mm2/1", c["area_rule_mm2"][1]))
        i_pass.append(inp(W48, dw, f"members/C1/arms/B5m@8192/candidate/shipped|{K}/passes", y["passes"]))
        i_wr.append(inp(W48, dw, f"members/C1/arms/B5m@8192/candidate/shipped|{K}/written_passes", y["written_passes"]))
    if len(passes) != 1 or len(written) != 1:
        sys.exit(f"d01: B5m's passes {passes} or written passes {written} differ across batches: stopped")
    rm, dm = load(ROWS + "C1.T-INT.b.B5m.256x64/result.json")
    r5, d5 = load(ROWS + "C1.T-INT.b.B5.256x64/result.json")
    r0, d0 = load(ROWB0)
    npass = passes.pop()
    if rm["deployment"]["neural_stages"] != npass or not rm["design"]["toolchain"]["position_multiplexing"]:
        sys.exit("d01: the B5m row's neural stages or its position multiplexing disagree with the priced schedule: stopped")
    if r5["design"]["toolchain"]["position_multiplexing"]:
        sys.exit("d01: the B5 row multiplexes positions: stopped")
    i_pass.append(inp(ROWS + "C1.T-INT.b.B5m.256x64/result.json", dm, "deployment/neural_stages", rm["deployment"]["neural_stages"]))
    outs = {n: r["K9"]["outputs_sha256"] for n, r in (("B0", r0), ("B5", r5), ("B5m", rm))}
    k45 = rm["K4_K5_pipeline"]
    gates = (k45["K4_value_parity_gate"]["verdict"], k45["K4_value_parity_gate"]["atol"],
             k45["K5_value_twin_certificate"]["verdict"], k45["K5_value_twin_certificate"]["atol"])
    static = rm["static_program_check"]["lines"][0]
    if len(set(outs.values())) != 1 or gates != ("PASS", 0.0, "PASS", 0.0) or "Theorem 1 established" not in static:
        sys.exit(f"d01: the B5m row's outputs {outs}, gates {gates} or static check do not certify equal outputs: stopped")
    for rel, dg, r in ((ROWS + "C1.T-INT.b.B5m.256x64/result.json", dm, rm),
                       (ROWS + "C1.T-INT.b.B5.256x64/result.json", d5, r5), (ROWB0, d0, r0)):
        i_out.append(inp(rel, dg, "K9/outputs_sha256", r["K9"]["outputs_sha256"]))
        i_out.append(inp(rel, dg, "K9/images", r["K9"]["images"]))
    lo, hi = min(ratios.values()), max(ratios.values())


    return {"cores": ca["count"],
            "passes": {"value": npass, "inputs": i_pass},
            "written_passes": {"value": written.pop(), "inputs": i_wr},
            "speed": {"least": half_up(lo), "largest": half_up(hi),
                      "exact": {k: round(v, 4) for k, v in ratios.items()}, "inputs": i_lat,
                      "arithmetic": "B5's nominal e2e latency over B5m@8192's at each batch, the least and largest "
                                    "rounded half up"},
            "batches": {"least": 1, "largest": 64, "of": ["K1", "K16", "K64"]},
            "area": {"equal": True, "mm2": round(a["candidate"]["shipped|K1"]["area_rule_mm2"][1], 4), "inputs": i_area},
            "outputs_equal": {"images": rm["K9"]["images"], "digest": outs["B5m"][:16], "K4_K5": "PASS at atol 0",
                              "static": "Theorem 1 established", "inputs": i_out},
            "what": "ResNet-18 w1 with one kernel copy (B5m) on B5's 242 cores of 256 x 64 and its area, generic "
                    "22 nm, the toolchain's estimate; passes and written passes are the priced schedule's, equal at "
                    "every batch"}


def inet():
    ins, imgs, cores4, cores1 = [], set(), set(), set()
    for g in GROUPS:
        rel = W38 + g + ".group.json"
        d, dg = load(rel)
        for i, c in enumerate(d["K10"]["comparisons"]):
            if not c["passed"] or c["outputs_differ"] != 0:
                sys.exit(f"d01: {g}: K10 comparison {i} did not pass: stopped")
            imgs.add(c["images"])
            ins.append(inp(rel, dg, f"K10/comparisons/{i}/images", c["images"]))
        for i, r in enumerate(d["rows"]):
            if r.get("budget") == "chipx4+pm" and r.get("state") == "complete":
                cores4.add(r["declared_cores"]); ins.append(inp(rel, dg, f"rows/{i}/declared_cores", r["declared_cores"]))
            if r.get("budget") == "chip" and r.get("state") == "complete":
                cores1.add(r["declared_cores"]); ins.append(inp(rel, dg, f"rows/{i}/declared_cores", r["declared_cores"]))
    if len(imgs) != 1 or len(cores4) != 1 or len(cores1) != 1 or cores4.copy().pop() % cores1.copy().pop():
        sys.exit(f"d01: ImageNet images {imgs}, 4-chip cores {cores4} or chip cores {cores1} not single: stopped")
    n = cores4.pop() // cores1.pop()
    return {"compared_images": imgs.pop(), "chips": n, "chips_word": WORDS[n], "inputs": ins,
            "arithmetic": "chips: the chipx4+pm rows' declared cores over the chip rows' declared cores; the "
                          "compared images: every K10 comparison of the five groups, each passed with no output differing"}


def cells():
    d, dg = load(P0)
    c = d["energy"]["cells48"]
    a0 = {}
    for cell in d["fronts"].values():
        for side in ("energy_multi", "energy_single"):
            for net, row in cell[side].items():
                if net in a0 and a0[net] != row[-1]:
                    sys.exit(f"d01: {net}'s seed-0 accuracy differs between fronts: stopped")
                a0[net] = row[-1]
    trained = {n for n in a0 if n.endswith("b4444")} - set(d["energy"]["untrained"])
    win = {k: v for k, v in c.items() if v.get("gain") is not None and v["gain"] > 0}
    pairs = {}
    for k, v in win.items():
        cand = sorted({(j, f) for j in v["J"] for f in trained if f != j and abs(a0[j] - v["gain"] - a0[f]) < 0.006})
        if len(cand) != 1:
            sys.exit(f"d01: cell {k}: {len(cand)} consistent pairs, not one: stopped")
        pairs.setdefault(cand[0], []).append(k)
    return {"cells": len(c), "gain": len(win), "pairs": len(pairs),
            "by_pair": {f"{j} | {f}": len(v) for (j, f), v in sorted(pairs.items(), key=lambda x: -len(x[1]))},
            "inputs": [inp(P0, dg, "energy/cells48/*/gain and J", len(win)),
                       inp(P0, dg, "fronts/*/energy_multi|energy_single/*/-1 (seed-0 accuracy)", len(a0))],
            "arithmetic": "per cell with gain > 0, the one (J, FO) with J in the cell's list and FO a trained b4444 "
                          "network whose seed-0 accuracy is J's less the gain within 0.006; the distinct pairs counted",
            "what": "distinct pairs of selected networks (multi-pass, single-pass) over the cost-budget cells that gain"}


def main():
    out = {"format": "rv3-derived/1", "written_by": TRR + "/tiers/paper_locks/cut/rv3_compose/steps/d01_derive.py",
           "h": {"one_copy": one_copy()}, "q3": {"inet": inet()}, "q4": {"cells": cells()}}
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    h, q3, q4 = out["h"]["one_copy"], out["q3"]["inet"], out["q4"]["cells"]
    print("d01: wrote", sys.argv[1], "| one copy: cores", h["cores"], "passes", h["passes"]["value"], "written",
          h["written_passes"]["value"], "speed", {k: h["speed"][k] for k in ("least", "largest", "exact")}, "outputs", h["outputs_equal"]["images"], "| inet: compared",
          q3["compared_images"], "chips", q3["chips"], q3["chips_word"], "| cells:", q4["gain"], "of", q4["cells"],
          "pairs", q4["pairs"])


if __name__ == "__main__":
    main()
