""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "driver"))
import w38common as c

BATCH = 50
N_FULL_VAL = 50_000


def clopper_pearson(k: int, n: int, level: float) -> tuple[float, float]:
    from scipy.stats import beta

    a = 1.0 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--images", type=int, default=5000)
    a = ap.parse_args()
    c.install_data_guard()
    import numpy as np
    import torch
    from torchvision import models

    torch.set_num_threads(a.threads)
    row = c.members()[a.member]
    wf = row["weights_file"]
    if c.sha256_file(wf["file"]) != wf["file_sha256"]:
        raise SystemExit(f"REFUSED: {wf['file']} differs from members.json")
    weights = c.weights_of(row)
    model = models.get_model(a.member, weights=weights).eval()
    digest = c.state_digest(model.state_dict())
    if digest != wf["canonical_state_digest"]:
        raise SystemExit("REFUSED: canonical state digest differs from members.json")
    entries = c.manifest_entries(a.images)
    tf = weights.transforms()
    out = HERE / a.member
    out.mkdir(parents=True, exist_ok=True)
    logits, labels = [], []
    t0 = time.time()
    t_load = t_model = 0.0
    with torch.no_grad():
        for i in range(0, len(entries), BATCH):
            ta = time.time()
            batch = entries[i: i + BATCH]
            x = torch.stack([tf(c.load_image(e)) for e in batch])
            tb = time.time()
            logits.append(model(x).float().numpy())
            t_load += tb - ta
            t_model += time.time() - tb
            labels += [e["class_index"] for e in batch]
    wall = time.time() - t0
    z = np.concatenate(logits)
    y = np.asarray(labels, dtype=np.int64)
    pred = z.argmax(1)
    top5 = np.argsort(-z, axis=1)[:, :5]
    k1, k5, n = int((pred == y).sum()), int((top5 == y[:, None]).any(1).sum()), len(y)
    srt = np.sort(z, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    pub = row["published_acc1"] / 100.0
    p = k1 / n
    fpc_se = math.sqrt(p * (1 - p) / n * (N_FULL_VAL - n) / (N_FULL_VAL - 1))
    ci95, ci99 = clopper_pearson(k1, n, 0.95), clopper_pearson(k1, n, 0.99)
    rec = {"format": "w38-float/1", "member": a.member, "weights_enum": row["weights_enum"],
           "weights_file_sha256": wf["file_sha256"], "canonical_state_digest": digest, "images": n,
           "dtype": "float32", "device": "cpu", "threads": a.threads, "torch": torch.__version__,
           "transform": repr(tf), "top1_correct": k1, "top1": p, "top5_correct": k5, "top5": k5 / n,
           "published_acc1": row["published_acc1"], "published_acc5": row["published_acc5"],
           "clopper_pearson_95": ci95, "clopper_pearson_99": ci99,
           "published_inside_95": ci95[0] <= pub <= ci95[1], "published_inside_99": ci99[0] <= pub <= ci99[1],
           "finite_population_se": fpc_se, "difference_pp": 100 * (p - pub),
           "top1_margin_below_1e-4": int((margin < 1e-4).sum()), "top1_exact_ties": int((margin == 0).sum()),
           "wall_s": wall, "s_per_image_total": wall / n, "s_per_image_model": t_model / n,
           "s_per_image_load_and_transform": t_load / n, "peak_rss_kb": c.peak_rss_kb(),
           "manifest_sha256": c.sha256_file(c.MANIFEST),
           "files": {"logits.npy": c.save_npy(out / "logits.npy", z), "pred.npy": c.save_npy(out / "pred.npy", pred),
                     "labels.npy": c.save_npy(out / "labels.npy", y)},
           "logits_sha256": c.sha256_array(z), "script_sha256": c.sha256_file(Path(__file__)),
           "written": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    rec["transform_equality"] = {
        "subset_first_16": c.check_transform_equality(row, c.manifest_entries(16)),
        "calibration_first_16": c.check_transform_equality(row, c.calibration_entries(16))}
    c.atomic_write_json(out / "float.json", rec)
    print(json.dumps({k: rec[k] for k in ("member", "top1_correct", "top1", "published_acc1", "clopper_pearson_99",
                                          "published_inside_99", "s_per_image_model", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
