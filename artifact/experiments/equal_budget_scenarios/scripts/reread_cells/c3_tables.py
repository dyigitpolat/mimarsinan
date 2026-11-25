""
import json
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True
BR = ("records/tiers/"
      "paper_v6/fix5/study_build")
CB = json.load(open(BR + "/reread_cells/out/cells_by_combination.json"), object_pairs_hook=OrderedDict)
AM = {0: "fakeram", 1: "eq/bit"}


def f4(x):
    return "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


print("== rows ties", json.dumps(CB["rows"]))
print("== record energy read equal", CB["record_energy_read_equal_P0_json"])
print("== eval cache", CB["eval_cache"])
print("== findings", CB["findings"])
for c, cell in CB["cells"].items():
    print(f"\n== {c}: gain {cell['gain']} (mean {cell['gain_mean']}), attaining {cell['attaining']}, declared attains "
          f"{cell['declared_attains']}, distinct printed rows among attaining {cell['attaining_distinct_printed_rows']}")
    print("   costliest:", cell["costliest"]["k"], cell["costliest"]["why"])
    print("| k | f | tile area | a_p | N* | N*_1 | gap pp (means) | N* design | N*_1 design | periods us | time ratio | "
          "N* E/budget | attains |")
    for x in cell["combinations"]:
        jd, fd = x["J_design"], x["FO_design"]
        print(f"| {x['k']} | {x['combo'][0]:g} | {AM[x['combo'][1]]} | {x['combo'][2]:g} | {x['J']} | {x['FO']} | "
              f"{x['gap']} ({x['gap_mean']}) | {jd['geometry']} {jd['tiles']} t {jd['passes']} p | {fd['geometry']} "
              f"{fd['tiles']} t {fd['passes']} p | {jd['period_us']} / {fd['period_us']} | {x['time_ratio']} | "
              f"{x['J_E_over_budget']} | {x['attains_least']} |")
    print("   twins:", [(tw["f"], tw["area_model"], tw["combinations"], tw["masks_rows_equal"],
                         tw["rows_equal_every_field"], tw["printed_fields_differing"]) for tw in cell["port_twins"]])
    print("   ties per combination:", [(x["k"], x["ties"]["sets_and_selection_recomputed"], x["ties"]["gap_recomputed"],
                                       x["ties"]["masks_designs"], x["ties"]["P0_json_combo"],
                                       x["ties"]["J_memory_beside"]["memory_beside_at_combination_form_equal"],
                                       x["ties"]["FO_memory_beside"]["memory_beside_at_combination_form_equal"])
                                      for x in cell["combinations"]])
    dt = CB["ties"][c]["declared_read"]
    print("   declared-read ties:", {k: (v["here_equals_P0_json"], v["here_equals_record_in_process"])
                                     for k, v in dt.items()})
    drow, crow = cell["declared_row"], cell["costliest"]["row"]
    print("   printed fields, declared read | costliest:")
    for k in drow:
        mark = "" if drow[k] == crow[k] else "   <== differs"
        print(f"     {k}: {f4(drow[k])} | {f4(crow[k])}{mark}")

    for x in cell["combinations"]:
        if x["attains_least"]:
            dif = [k for k in drow if x["printed"][k] != drow[k]]
            if dif:
                print(f"   attaining k {x['k']} differs from the declared read in {dif}: "
                      f"{ {k: x['printed'][k] for k in dif} }")
print("\n== R1L costliest designs in full")
x = CB["cells"]["R1L"]["combinations"][CB["cells"]["R1L"]["costliest"]["k"]]
for side in ("J_design", "FO_design"):
    d = x[side]
    print(f"  {side}: " + json.dumps({k: v for k, v in d.items()}))
print("\n== fronts tie R1L", json.dumps(CB["fronts_tie_R1L"]))
print("\n== thresholds R1L")
for ax, h in CB["thresholds_R1L"].items():
    if ax == "same_count_difference_pairs":
        print("  same count difference pairs:", h)
        continue
    print(f"  {ax}: bracket {h['record_bracket'] and {k: h['record_bracket'][k] for k in ('threshold', 'last_same', 'first_changed', 'P0_item', 'other_item')}}, reproduces {h['reproduces_record']}")
    for side in ("last_same", "first_changed", "at_end"):
        if side in h:
            r = h[side]["R1L"]
            print(f"    {side}: sig {r['sig']}, declared passes {r['J_passes']}, time ratio {r['time_ratio']}")
            for cb in r["combos"]:
                print(f"      {cb}")
print("\n== time range", json.dumps(CB["time_range"]))
print("\n== wall", json.dumps(CB["wall_s"]))
