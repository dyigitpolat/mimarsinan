""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PT = HERE.parent
RA = PT.parent.parent
RR = RA / "12_tcad_readiness" / "tiers" / "paper_v6" / "fix5" / "study_build" / "reread"

INPUTS = OrderedDict(
    P0=(RR / "out" / "read" / "P0.json", "134c2343da0146b167abf1a66f9dddc8c53c975e64eec0972a186a32e16d1254"),
    analysis=(RR / "out" / "analysis.json", "caaba7dee52f931688e1f3c787564c3006045dbdf145b528088e94d81930285e"),
    rules_pairs_json=(PT / "rules_pairs.json", "bf7eac6f9c28c8a593457d2e316b22d68d8b542c69416f8d4a829c9f30c8e13c"),
)
OUT = HERE / "selections_r3.json"

CELLS = OrderedDict([
    ("K64", "batch 64"),
    ("H10.1", "batch 16"),
    ("R1'", "batch 1, loose budgets"),
    ("R1L", "batch 1, tight budgets"),
])

AREA = {0: "fakeram", 1: "equal area per bit"}
DECLARED_READ = (2.0, 0, 0.05)


REREAD_TABLE = OrderedDict([
    ("K64", dict(J="R18 w1 b4444", J_geometry="COMPASS", J_tiles=204, J_passes=5, FO="R18 w1/2 b4444", FO_tiles=204,
                 gain_pp=4.70, time_ratio=1.2252, equals=["K64"])),
    ("H10.1", dict(J="R18 w1 b4444", J_geometry="COMPASS", J_tiles=204, J_passes=5, FO="R18 w1/2 b4444", FO_tiles=204,
                   gain_pp=4.70, time_ratio=1.3596, equals=["K64"])),
    ("R1'", dict(J="R18 w1/sqrt2 b4444", J_geometry="COMPASS", J_tiles=242, J_passes=9, FO="R18 w1/2 b4444",
                 FO_tiles=204, gain_pp=1.96, time_ratio=1.9618, equals=["H10.1"])),
    ("R1L", dict(J="R18 w1/2 b4444", J_geometry="ISAAC-like", J_tiles=171, J_passes=7, FO="R10 w1/2 b4444",
                 FO_tiles=171, gain_pp=4.42, time_ratio=2.5718, equals=[])),
])

E01_TOTAL = 5000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(name: str):
    p, pin = INPUTS[name]
    got = sha256(p)
    if got != pin:
        raise SystemExit(f"REFUSED: {p} has sha256 {got}, pinned {pin}")
    return json.loads(p.read_text(), object_pairs_hook=OrderedDict)


def count_of(acc_pct: float) -> int:
    c = round(acc_pct * E01_TOTAL / 100.0)
    if abs(acc_pct * E01_TOTAL / 100.0 - c) > 1e-6:
        raise SystemExit(f"REFUSED: accuracy {acc_pct} is not a count of {E01_TOTAL}")
    return int(c)


def member_accuracies(p0: dict) -> OrderedDict:
    ""
    acc = OrderedDict()
    for cell, fr in p0["fronts"].items():
        for side in ("energy_multi", "energy_single"):
            for m, row in fr[side].items():
                a = float(row[4])
                if m in acc and acc[m] != a:
                    raise SystemExit(f"REFUSED: {m}: accuracy {acc[m]} and {a} in two fronts")
                acc[m] = a
    return acc


def main() -> int:
    p0 = load("P0")
    an = load("analysis")
    rp = load("rules_pairs_json")
    if p0.get("id") != "P0" or p0.get("meta", {}).get("kind") != "full":
        raise SystemExit("REFUSED: P0.json is not the full reading of P0")
    acc = member_accuracies(p0)
    jobs = OrderedDict()
    for c, p in rp["pairs"].items():
        for side in ("J", "FO"):
            m, j = p[side], p[f"{side}_job"]
            if jobs.get(m, j) != j:
                raise SystemExit(f"REFUSED: {m} has two jobs")
            jobs[m] = j

            if count_of(acc[m]) != p[f"s0_{side}_correct"]:
                raise SystemExit(f"REFUSED: {m}: P0.json gives {acc[m]} pp, rules_pairs.json {p[f's0_{side}_correct']}")
    pairs_of_record = OrderedDict((c, (p["J"], p["FO"])) for c, p in p0["energy"]["pairs"].items())
    for c, (j, fo) in pairs_of_record.items():
        if (rp["pairs"][c]["J"], rp["pairs"][c]["FO"]) != (j, fo):
            raise SystemExit(f"REFUSED: P0.json's pair of record {c} differs from rules_pairs.json")

    def net(m: str) -> OrderedDict:
        return OrderedDict(network=m, job=jobs.get(m), trained_s1_to_s5=m in jobs,
                           dev_accuracy_pct_s0=acc[m], dev_correct_s0=count_of(acc[m]), dev_total=E01_TOTAL)

    out_cells = OrderedDict()
    problems = []
    for c, words in CELLS.items():
        e = p0["energy"]["cells"][c]
        a = an["P0_full"][c]
        z = an["four_cells_zero"][c]
        fr = p0["fronts"][c]
        J, FO = e["J"], e["FO"]

        for k_e, k_a in (("J", "J"), ("FO", "FO"), ("J_tiles", "J_tiles"), ("J_passes", "J_passes"),
                         ("FO_tiles", "FO_tiles"), ("FO_passes", "FO_passes"), ("time_ratio", "time_ratio"),
                         ("gain", "gain_pp"), ("J_geometry", "J_geometry")):
            if e[k_e] != a[k_a]:
                problems.append(f"{c}: P0.json {k_e} {e[k_e]!r} against analysis.json {k_a} {a[k_a]!r}")
        combos = [OrderedDict(f=k[0], area=AREA[k[1]], a_p=k[2], J=j, FO=fo, gap_pp=g) for k, j, fo, g in e["combos"]]
        gmin = min(x["gap_pp"] for x in combos)
        if abs(gmin - e["gain"]) > 1e-9:
            problems.append(f"{c}: the gain {e['gain']} is not the minimum over the 8 combinations ({gmin})")
        dec = [x for x in combos if (x["f"], {v: k for k, v in AREA.items()}[x["area"]], x["a_p"]) == DECLARED_READ]
        if len(dec) != 1 or (dec[0]["J"], dec[0]["FO"]) != (J, FO) or abs(dec[0]["gap_pp"] - e["declared_gap"]) > 1e-9:
            problems.append(f"{c}: the named selection is not the declared read's")
        diff_pp = (count_of(acc[J]) - count_of(acc[FO])) / (E01_TOTAL / 100.0)
        if abs(diff_pp - e["declared_gap"]) > 1e-9:
            problems.append(f"{c}: the selected networks' seed-0 difference {diff_pp} differs from declared_gap")
        attaining = [x for x in combos if abs(x["gap_pp"] - gmin) <= 1e-9]
        sel = OrderedDict(
            cell=c, words=words,
            energy_budget_uJ=e["budget_uJ"], area_budget_mm2=e["J_design"]["area_budget_mm2"],
            selected=OrderedDict(net(J), geometry=e["J_geometry"], tiles=e["J_tiles"], passes=e["J_passes"],
                                 period_us=e["J_period_us"]),
            single_pass=OrderedDict(net(FO), geometry=e["FO_design"]["geometry"], tiles=e["FO_tiles"],
                                    passes=e["FO_passes"], period_us=e["FO_period_us"]),
            time_ratio=e["time_ratio"],
            gain_at_convention_pp=e["gain"],
            gain_definition=("the cell's gain at seed 0: the minimum of N* - N*_1 over the cell's 8 combinations (f x "
                             "ISAAC-like tile area x port band) at s_AU 1.13, the reading's quantity of headline_v46 "
                             "(headline_rules.md section 6, 'robust gap at the convention')"),
            gain_attained_at=[OrderedDict((k, x[k]) for k in ("f", "area", "a_p", "J", "FO")) for x in attaining],
            declared_read=OrderedDict(f=2.0, area="fakeram", a_p=0.05, s_AU=1.13),
            declared_read_gap_pp=e["declared_gap"],
            selected_minus_single_dev_s0_pp=round(diff_pp, 4),
            gain_mean_pp=e["gain_mean"], J_mean=e["J_mean"], FO_mean=e["FO_mean"],
            combinations=combos,
            figure_setting=OrderedDict(setting=fr["setting"], multi=fr["at_budget_multi"], single=fr["at_budget_single"],
                                       gap_pp=fr["gap_at_budget"]),
            without_static_power=OrderedDict(J=z["J"], FO=z["FO"], gain_pp=z["gain"]),
            pair_of_record_of_this_cell=OrderedDict(J=pairs_of_record[c][0], FO=pairs_of_record[c][1]),
        )
        eq = [k for k, (j, fo) in pairs_of_record.items() if (j, fo) == (J, FO)]
        eq_rev = [k for k, (j, fo) in pairs_of_record.items() if (fo, j) == (J, FO)]
        sel["equals_pair_of_record"] = eq
        sel["equals_pair_of_record_reversed"] = eq_rev
        if eq != a["equals_pair_of_record_of"]:
            problems.append(f"{c}: equals {eq}, analysis.json {a['equals_pair_of_record_of']}")
        sel["own_pair_of_record_selected"] = c in eq
        sel["without_static_power_is_own_pair_of_record"] = (z["J"], z["FO"]) == pairs_of_record[c]

        sel["ties"] = OrderedDict(
            selected=[m for m in acc if m != J and count_of(acc[m]) == count_of(acc[J])],
            single_pass=[m for m in acc if m != FO and count_of(acc[m]) == count_of(acc[FO])])
        out_cells[c] = sel

        t = REREAD_TABLE[c]
        mine = dict(J=J, J_geometry=e["J_geometry"].split()[0], J_tiles=e["J_tiles"], J_passes=e["J_passes"], FO=FO,
                    FO_tiles=e["FO_tiles"], gain_pp=round(e["gain"], 2), time_ratio=e["time_ratio"], equals=eq)
        for k, v in t.items():
            if mine[k] != v:
                problems.append(f"{c}: {k} {mine[k]!r}, the re-read's table {v!r}")


    r1l = out_cells["R1L"]
    fig_single = r1l["figure_setting"]["single"][1]
    fo = r1l["single_pass"]["network"]
    r1l["two_settings"] = OrderedDict(
        nominal_full_reading=OrderedDict(
            setting="the declared read (f 2, fakeram, a_p 0.05, s_AU 1.13), whose picks the full reading names",
            selected=r1l["selected"]["network"], single_pass=fo,
            single_pass_dev_correct_s0=count_of(acc[fo]), gap_pp=r1l["declared_read_gap_pp"]),
        figure_drawing_setting=OrderedDict(
            setting="Fig. 15's drawing setting (f 2, equal area per bit, s_AU 1.13)",
            selected=r1l["figure_setting"]["multi"][1], single_pass=fig_single,
            single_pass_dev_correct_s0=count_of(acc[fig_single]), gap_pp=r1l["figure_setting"]["gap_pp"],
            single_pass_tiles=p0["fronts"]["R1L"]["energy_single"][fig_single][2],
            single_pass_passes=p0["fronts"]["R1L"]["energy_single"][fig_single][1]),
        tie=count_of(acc[fo]) == count_of(acc[fig_single]),
        difference_images_of_5000=count_of(acc[fig_single]) - count_of(acc[fo]),
        why=("N*_1 is the most accurate member admissible resident in the combination (headline_rules.md section 6). "
             "At the declared read (fakeram tile area) both selections run on 171 ISAAC-like tiles (91.7% of the area "
             "budget), and ResNet-18 w1/(2 sqrt2), though more accurate than ResNet-10 w1/2, is not the single-pass "
             "pick, so by the definition it is not admissible resident there (its one-pass design in the figure's "
             "front uses 242 tiles). ResNet-10 w1/2 runs in one pass on 171 tiles. At equal area per bit the "
             "ISAAC-like tile is smaller, ResNet-18 w1/(2 sqrt2) runs in one pass on 242 tiles within the same area "
             "budget, and it is the more accurate of the two (3,262 against 3,203 of 5,000 at seed 0), so it is the "
             "single-pass network there. The cell's gain at the convention (4.42 pp, the minimum over the 8 "
             "combinations) is attained at equal area per bit against ResNet-18 w1/(2 sqrt2). The declared read's "
             "pair differs by 5.60 pp at seed 0."),
    )

    declared = OrderedDict(
        name="R1L-P0", cell="R1L", words="the fourth contrast: the tight batch-1 cell's selection at P0",
        J=r1l["selected"]["network"], J_job=r1l["selected"]["job"],
        FO=r1l["single_pass"]["network"], FO_job=r1l["single_pass"]["job"],
        J_design=OrderedDict(geometry=r1l["selected"]["geometry"], tiles=r1l["selected"]["tiles"],
                             passes=r1l["selected"]["passes"]),
        FO_design=OrderedDict(geometry=r1l["single_pass"]["geometry"], tiles=r1l["single_pass"]["tiles"],
                              passes=r1l["single_pass"]["passes"]),
        rule=("the pair of the nominal point's full reading (the declared read's picks), as revision 2 fixed every "
              "pair (rules_pairs.md section 3: 'the fixed pair is the declared read's')"),
        s0_dev_J=r1l["selected"]["dev_correct_s0"], s0_dev_FO=r1l["single_pass"]["dev_correct_s0"],
        s0_dev_difference_images_of_5000=r1l["selected"]["dev_correct_s0"] - r1l["single_pass"]["dev_correct_s0"],
        cell_gain_at_convention_pp=r1l["gain_at_convention_pp"],
    )
    if not (declared["J_job"] and declared["FO_job"]):
        problems.append("the fourth contrast's members are not both trained jobs of the track")

    out = OrderedDict(
        label=("the four selections at the point of record P0 (seed 0, the development split), revision 3 of the pairs "
               "track, from development data only"),
        inputs=OrderedDict((k, OrderedDict(path=str(p.relative_to(RA)), sha256=pin)) for k, (p, pin) in INPUTS.items()),
        script="select_r3.py",
        cells=out_cells,
        pairs_of_record=OrderedDict((c, OrderedDict(J=j, FO=fo)) for c, (j, fo) in pairs_of_record.items()),
        fourth_contrast=declared,
        matches_reread_table=not problems,
        problems=problems,
    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"selections_r3.json sha256 {sha256(OUT)}")
    for c, s in out_cells.items():
        print(f"{c}: N* {s['selected']['network']} ({s['selected']['geometry']}, {s['selected']['tiles']} tiles, "
              f"{s['selected']['passes']} passes, {s['selected']['dev_accuracy_pct_s0']}) against N*_1 "
              f"{s['single_pass']['network']} ({s['single_pass']['tiles']} tiles, {s['single_pass']['dev_accuracy_pct_s0']}); "
              f"gain {s['gain_at_convention_pp']} pp (declared read {s['declared_read_gap_pp']}); equals {s['equals_pair_of_record']}; "
              f"ties {dict(s['ties'])}")
    print("R1L two settings:", json.dumps(r1l["two_settings"], indent=None)[:900])
    if problems:
        print("DIFFERS FROM THE RE-READ'S TABLE OR THE RECORD:\n  " + "\n  ".join(problems))
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
