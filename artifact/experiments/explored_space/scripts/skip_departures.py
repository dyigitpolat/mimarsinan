""
import glob
import json
import os
import sys

d = sys.argv[1]
for f in sorted(glob.glob(os.path.join(d, "*__*.jsonl"))):
    if "+coalescing" in f or not any(f"__{t}" in os.path.basename(f) for t in ("truenorth", "loihi", "odin")):
        continue
    rows = [json.loads(line) for line in open(f)]
    if not rows or not rows[-1].get("trailer"):
        continue
    priced = sum(1 for r in rows if "network" in r and not r.get("refusal") and not r.get("above_budget"))
    stem = f[:-len(".jsonl")]
    for suffix in ("+coalescing.jsonl", "+coalescing_arms.jsonl"):
        dep = stem + suffix if not stem.endswith("_arms") else None
        if dep is None or os.path.exists(dep) or not priced:
            continue
        if suffix == "+coalescing_arms.jsonl" and not os.path.exists(stem + "_arms.jsonl"):
            continue
        with open(dep, "w") as handle:
            handle.write(json.dumps({"header": True, "skipped": f"the as-declared job {os.path.basename(f)} priced "
                                     f"{priced} points: the bundle's own permissions map the member"}) + "\n")
            handle.write(json.dumps({"trailer": True, "rows": 0, "skipped": True}) + "\n")
        print("skip", os.path.basename(dep))
