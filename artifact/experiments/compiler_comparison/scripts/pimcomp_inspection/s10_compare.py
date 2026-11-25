""
import sys, json, glob, numpy as np

W = sys.argv[1]
out = {}
for name in ("regnet_y_400mf", "regnet_y_800mf"):
    for rng in ("norm", "raw"):
        rows = sorted(sum((json.load(open(p)) for p in glob.glob(f"{W}/replay/{name}_{rng}_*.json")), []),
                      key=lambda r: r["i"])
        if not rows:
            continue
        a = np.array([r["replay_logits"] for r in rows]); b = np.load(f"{W}/ort_logits_{name}_{rng}.npy")[[r["i"] for r in rows]]
        norm = np.abs(a - b).max(1) / np.abs(b).max(1)
        out[f"{name}_{rng}"] = {"n": len(rows), "elements": int(a.size),
                                "differ_1e-6": int((np.abs(a - b) > 1e-6 * np.abs(b)).sum()),
                                "max_normwise_rel": float(norm.max()), "median_normwise_rel": float(np.median(norm)),
                                "top1_changed": int((a.argmax(1) != b.argmax(1)).sum()),
                                "ort_top1_agrees_with_saved": int(sum(r["ort_top1"] == int(b[k].argmax()) for k, r in enumerate(rows))),
                                "first_product_slot_nonzero_max": max(r["first_product_slot_nonzero"] for r in rows),
                                "s_per_input_median": float(np.median([r["s"] for r in rows]))}
        print(name, rng, json.dumps(out[f"{name}_{rng}"]), flush=True)
json.dump(out, open(f"{W}/replay_compare.json", "w"), indent=1)
