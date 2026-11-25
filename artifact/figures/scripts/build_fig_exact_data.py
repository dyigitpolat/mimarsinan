""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
F3 = HERE.parent
PAPER = F3.parent
sys.path.insert(0, str(PAPER / "figures_v2" / "scripts"))
import _ledger as LG

NUM = r"([0-9][0-9,]*)"


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _led(key: str) -> dict:
    ""
    v = LG.value(key)
    return {"key": key, "display": v["display"], "form": v["form"], "class": v["class"], "tier": v["tier"],
            "head": v.get("head"), "pending": v["value"] is None, "what": LG.entry(key).get("what", "")}


def _parse(key: str, pattern: str, groups=(1,)):
    ""
    v = _led(key)
    if v["pending"]:
        return v, None
    m = re.search(pattern, v["display"])
    if not m:
        raise SystemExit(f"fig_exact: ledger key {key!r} prints {v['display']!r}, which does not match {pattern!r}")
    return v, tuple(_int(m.group(g)) for g in groups)


def _source_sum(key: str, suffix: str):
    ""
    vals = [s["value"] for s in LG.sources(key)
            if str(s.get("json_path") or "").endswith(suffix) and isinstance(s.get("value"), (int, float))]
    return sum(vals) if vals else None


def _check(name: str, parsed, recorded):
    if recorded is not None and parsed != recorded:
        raise SystemExit(f"fig_exact: {name}: the ledger prints {parsed}, its sources record {recorded}")


def k_of_n(key: str):
    ""
    v, got = _parse(key, NUM + r" of " + NUM, (1, 2))
    return (v, None, None) if got is None else (v, got[0], got[1])


def family(fid: str, label: str, unit: str, differ, compared, keys: list, note: str = "", derived: str = ""):
    pend = differ is None or compared is None
    return {"id": fid, "label": label, "unit": unit, "differ": differ, "compared": compared,
            "status": "pending" if pend else "measured", "keys": keys, "note": note, "derived": derived}


def long_window_range(key: str):
    ""
    what = LG.entry(key).get("what", "")
    m = re.search(r"\bT (\d+) to (\d+)\b", what)
    if not m:
        raise SystemExit(f"fig_exact: the ledger's words for {key!r} state no range of T: {what!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    basis = [f"ledger words of {key}: {m.group(0)}"]
    seen = False
    for s in LG.sources(key):
        top = str(s.get("json_path") or "").split("/")[0]
        if not top:
            continue
        scope = LG.load_json(LG.source_path(s)).get(top + "_scope")
        if not isinstance(scope, str):
            continue
        r = re.search(r"\bT in (\d+)-(\d+), beyond", scope)
        if not r:
            raise SystemExit(f"fig_exact: the scope line of {key}'s record states no range of T: {scope!r}")
        if (int(r.group(1)), int(r.group(2))) != (lo, hi):
            raise SystemExit(f"fig_exact: {key}: the ledger says T {lo} to {hi}, its record says {r.group(0)!r}")
        if not seen:
            basis.append(f"{s['path']} at {top}_scope: T in {r.group(1)}-{r.group(2)}")
        seen = True
    if not seen:
        raise SystemExit(f"fig_exact: no source record of {key} holds a scope line to check the range of T against")
    return lo, hi, basis


SEMANTICS = {"proptwosynclif": "synchronized LIF", "proptwottfs": "synchronized TTFS",
             "proptwovalue": "value domain"}
PROP_TWO = [
    ("proptwo_lif", "proptwosynclif", "LIF neuron windows", "neuron-windows"),
    ("proptwo_ttfs", "proptwottfs", "TTFS neuron windows", "neuron-windows"),
    ("proptwo_value", "proptwovalue", "Integer neuron windows", "output windows"),
]


def build_families() -> list:
    out = []


    v, k, n = k_of_n("routebinvariance")
    out.append(family("cifar", "CIFAR-100 output codes", "output codes", k, n, [v],
                      note="trained networks across core budgets, development split"))


    v, k, n = k_of_n("ditherwindows")
    out.append(family("tiling", "Tiled against untiled IR", "neuron-windows", k, n, [v]))


    for fid, key, label, unit in PROP_TWO:
        v, got = _parse(key, NUM + r" of " + NUM + r" instances, " + NUM + " " + unit, (1, 2, 3))
        differ = compared = None
        if got is not None:
            _check(key + " windows", got[2], _source_sum(key, "/windows"))
            _check(key + " differ", got[0], _source_sum(key, "/differ"))
            if got[0] != 0:
                raise SystemExit(f"fig_exact: {key} prints {got[0]} differing instances: its row is not a zero row "
                                 f"(a count of instances does not print over a count of {unit})")
            differ, compared = got[0], got[2]
        out.append(family(fid, label, unit, differ, compared, [v],
                          note="identity program against the integer network (Proposition 2), " + SEMANTICS[key],
                          derived=f"compared = the {unit} {key} prints; differing = the differing instances {key} "
                                  f"prints (zero instances of {'pending' if got is None else got[1]}, so zero {unit})"))


    vw, got = _parse("cardwindows", r"^" + NUM + r"$")
    vc, passed, checks = k_of_n("cardchecks")
    if got is None or passed is None:
        differ = compared = None
    else:
        compared = got[0]
        _check("cardwindows", compared, _source_sum("cardwindows", "/windows_total"))
        if passed != checks:
            raise SystemExit(f"fig_exact: card checks {passed} of {checks}: the card row is not a zero row")
        differ = checks - passed
    out.append(family("card", "FPGA card, neuron windows", "neuron-windows", differ, compared, [vw, vc],
                      note="two ODIN realizations on the card",
                      derived="differing = cardchecks total minus passing (every per-pass check passes); compared = cardwindows"))


    v, k, n = k_of_n("staticdisagree")
    out.append(family("static", "Static part against sampled run", "checks", k, n, [v]))


    v, k, n = k_of_n("lenetmismatch")
    out.append(family("lenet", "LeNet-5 outputs", "outputs", k, n, [v]))


    v, k, n = k_of_n("gfivefuzz")
    out.append(family("fuzz", "Random multi-pass programs", "programs", k, n, [v]))


    v, eq, n = k_of_n("xbackendfuzz")
    out.append(family("simulators", "Simulator runs", "programs", None if eq is None else n - eq, n, [v],
                      derived="differing = compared minus equal, both printed by xbackendfuzz"))


    v, got = _parse("screenlong", NUM + r" deployments, " + NUM + r" counterexamples", (1, 2))
    if got is not None:
        _check("screenlong accepted", got[0], _source_sum("screenlong", "/admitted"))
    lo, hi, basis = long_window_range("screenlong")
    out.append(family("long", f"Randomized test, {lo} to {hi} steps", "deployments",
                      None if got is None else got[1], None if got is None else got[0], [v]))
    out[-1]["window_range"] = {"lo": lo, "hi": hi, "basis": basis}

    done = [f for f in out if f["status"] == "measured"]
    pend = [f for f in out if f["status"] != "measured"]
    return sorted(done, key=lambda f: -f["compared"]) + pend


ARMS = [
    ("H0", ("H0", ""), "Declared semantics", "armhzero"),
    ("H1-theta", ("H1", "\u03b8"), "Equal stored values", "armhonetheta"),
    ("R4", ("R4", ""), "Input encoding", "armrfour"),
    ("H5-collapse", ("H5", "c"), "Delivery across passes", "armhfivecollapse"),
    ("H5-skew", ("H5", "s"), "Pass timing", "armhfiveskew"),
    ("R6", ("R6", ""), "Producer order", "armrsix"),
    ("H8", ("H8", ""), "Exact counts, levels, sums", "armheight"),
]


ZERO_REASON = {"H5-skew": "Fan-ins balanced by relay cores", "R6": "Fan-ins balanced by relay cores"}


def arm_count(cid: str, key: str, vper: dict, per):
    ""
    v = _led(key)
    if not v["pending"] and " of " in v["display"]:
        _, k, n = k_of_n(key)
        _check(key + " accepted", n, _source_sum(key, f"arms/{cid}/admitted"))
        keys = [v]
    else:
        _, got = _parse(key, r"^" + NUM + r"$")
        k, n, keys = None if got is None else got[0], per, [v, vper]
    if k is not None:
        _check(key, k, _source_sum(key, f"per_hypothesis/{cid}/code_teeth"))
    return k, n, keys


REPLAY_RULES = ("R1", "R6")
REPLAY_TAG = (" to ".join(REPLAY_RULES), "")
REPLAY_RECORD = "records/10_wave1/g5_residency_planner/u1_design.md, section 5.1, rules 1 to 6"


def control(cid: str, label: str, tag, differ, of, keys: list, unit: str, number: str, note: str = "",
            derived: str = ""):
    ""
    pend = differ is None or of is None
    return {"id": cid, "label": label, "tag_stem": tag[0], "tag_sub": tag[1], "differ": differ, "of": of,
            "unit": unit, "number": number, "share_pct": None if pend else 100.0 * differ / of,
            "status": "pending" if pend else "measured", "keys": keys, "note": note, "derived": derived}


def build_controls() -> dict:

    vc, got_c = _parse("screencex", r"^" + NUM + r"$")
    va, got_a = _parse("screenaccepted", r"^" + NUM + r"$")
    if got_a is not None:
        _check("screenaccepted", got_a[0], _source_sum("screenaccepted", "hypothesis_on/admitted"))
    baseline = control("none", "All kept", ("", ""), None if got_c is None else got_c[0],
                       None if got_a is None else got_a[0], [vc, va], "accepted deployments", "count",
                       note="the code-level screen with every condition kept",
                       derived="differing = screencex, accepted = screenaccepted; share = 100 x differing / accepted")

    vper, got_per = _parse("armaccepted", r"^" + NUM + r" per arm$")
    per = None if got_per is None else got_per[0]
    vsw, got_sw = _parse("hypswitched", r"^" + NUM + r" plus " + NUM + r"$", (1, 2))
    if got_sw is not None and got_sw[0] != len(ARMS):
        raise SystemExit(f"fig_exact: the ledger counts {got_sw[0]} switched conditions, this figure draws {len(ARMS)}")
    rows = []
    for cid, tag, label, key in ARMS:
        k, n, keys = arm_count(cid, key, vper, per)
        rows.append(control(cid, label, tag, k, n, keys, "accepted deployments", "count",
                            note="accepted deployments that differ, each with an equal paired control",
                            derived=(f"share = 100 x differing / accepted, both printed by {key}" if len(keys) == 1
                                     else f"share = 100 x {key} / armaccepted")))

        rows[-1]["reason"] = ZERO_REASON.get(cid, "") if rows[-1]["status"] == "measured" and k == 0 else ""


    v, k, n = k_of_n("replayneeded")
    other = [control("replay", "Pass replay", REPLAY_TAG, k, n, [v], "random multi-pass programs", "count",
                     note="random multi-pass programs of an earlier version without pass replay, streamed LIF; "
                          "not one of the screen's switched conditions. The gray ids after the name: the earlier "
                          f"version lacks the six multi-pass mapping rules {REPLAY_RULES[0]} to {REPLAY_RULES[1]} "
                          f"at once ({REPLAY_RECORD})",
                     derived="bar length = 100 x differing / compared, both printed by replayneeded; "
                             "the label prints the ledger's form")]

    done = sorted([r for r in rows if r["status"] == "measured"], key=lambda r: -r["share_pct"])
    pend = [r for r in rows if r["status"] != "measured"]


    for words in sorted({r["reason"] for r in done if r.get("reason")}):
        run = [r for r in done if r.get("reason") == words]
        if len(run) > 1:
            lead = "Both" if len(run) == 2 else f"All {len(run)}"
            for r in run:
                r["reason"] = f"{lead}: {words[:1].lower()}{words[1:]}"
    return {"baseline": baseline, "removed": done + pend, "other": other, "switched": vsw, "per_condition": vper}


def main():
    fams = build_families()
    ctrl = build_controls()
    data = {
        "figure": "fig_exact",
        "label": "fig:exact",
        "message": "Zero differences over every evidence family, while a removed condition does produce differences.",
        "panels": {
            "a": {"title": "All conditions kept", "xlabel": "Compared instances"},
            "b": {"title": "One condition switched off", "xlabel": "Differing share (%)"},
        },
        "families": fams,
        "controls": ctrl,

        "caption_values": {"armaccepted": ctrl["per_condition"]},

        "caption_mock_refs": {"theory:tab:conditions": "II"},
        "gaps": [f["label"] for f in fams if f["status"] != "measured"]
                + [c["label"] for c in [ctrl["baseline"]] + ctrl["removed"] + ctrl["other"]
                   if c["status"] != "measured"],
    }
    path = LG.write("fig_exact", data, F3 / "data")
    print(f"wrote {LG.rel(path)}: {len(fams)} families, {1 + len(ctrl['removed'])} control rows, "
          f"{len(ctrl['other'])} other row, {len(data['gaps'])} pending")
    for f in fams:
        print(f"  (a) {f['label']:36s} {f['differ']} of {f['compared']}  [{', '.join(k['key'] for k in f['keys'])}]")
    b = ctrl["baseline"]
    print(f"  (b) {b['label']:36s} {b['differ']} of {b['of']}  [{', '.join(k['key'] for k in b['keys'])}]")
    for c in ctrl["removed"] + ctrl["other"]:
        print(f"  (b) {c['label']:36s} {c['differ']} of {c['of']}  [{', '.join(k['key'] for k in c['keys'])}]")


if __name__ == "__main__":
    main()
