""
import json
import sys

U = "records/tiers/fidelity/ufidfpga"
R = json.load(open(f"{U}/readings/readings.json"))["realizations"]
out = {}


def cells(real, arm, ncs=None):
    return [c for c in R[real]["cells"].values() if c["arm"] == arm and c["nc"] != "all" and (ncs is None or c["nc"] in ncs)]


def band(cs, form="P-A"):
    return min(c[form]["min"] for c in cs), max(c[form]["max"] for c in cs)


for real in R:
    lo, hi = band(cells(real, "as_run"))
    out[f"E1 {real} as_run P-A"] = {"min": lo, "max": hi, "held": -0.10 <= lo and hi <= 0.0}
    lo, hi = band(cells(real, "resident"))
    out[f"E2 {real} resident P-A"] = {"min": lo, "max": hi, "held": hi < 0.0}
    for key, inv in R[real]["nc_invariance"].items():
        out[f"E3 {real} {key}"] = {"max_relative_spread": inv["max_relative_spread"], "max_round_spread": inv["max_round_spread"],
                                   "samples": inv["samples"], "held": inv["max_relative_spread"] <= 0.005}
    read = [c for c in R[real]["campaigns"] if c["read"]]
    out[f"E4 {real} census"] = {"mismatches": R[real]["census_mismatches"], "campaigns_read": len(read),
                                "campaigns_not_read": [c["name"] for c in R[real]["campaigns"] if not c["read"]],
                                "held": R[real]["census_mismatches"] == 0}
    pr = R[real]["programming_runs"]
    if pr:
        lo, hi = min(c["P-A"]["min"] for c in pr.values()), max(c["P-A"]["max"] for c in pr.values())
        floor = -0.10 if real == "stock" else -0.30
        out[f"E5 {real} programming run P-A"] = {"min": lo, "max": hi, "band": [floor, 0.0], "held": floor <= lo and hi <= 0.0}
json.dump(out, open(f"{U}/readings/expectations.json", "x"), indent=1, sort_keys=True)
for k, v in out.items():
    print(k, json.dumps(v, sort_keys=True))
sys.exit(0)
