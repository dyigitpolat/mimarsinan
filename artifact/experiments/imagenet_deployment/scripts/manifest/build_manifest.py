""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path("/data/ImageNet")
VAL, TRAIN = ROOT / "val", ROOT / "train"
META_BIN, DEVKIT = ROOT / "meta.bin", ROOT / "ILSVRC2012_devkit_t12.tar.gz"
PER_CLASS = 5
ROUNDS = PER_CLASS
SALT_SUBSET, SALT_ROUND = "W38-subset-v1", "W38-round-v1"
SALT_CALIB, SALT_CALIB_ORDER = "W38-calib-v1", "W38-calib-order-v1"
FORMAT = "w38-imagenet-manifest/1"


def key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sorted_wnids() -> list[str]:
    wnids = sorted(d for d in os.listdir(VAL) if (VAL / d).is_dir())
    train = sorted(d for d in os.listdir(TRAIN) if (TRAIN / d).is_dir())
    if len(wnids) != 1000 or wnids != train:
        raise SystemExit(f"REFUSED: val has {len(wnids)} class folders; train folders equal: {wnids == train}")
    return wnids


def check_class_index(wnids: list[str]) -> dict:
    ""
    import numpy as np
    import scipy.io
    import torch

    w2c, val_wnids = torch.load(META_BIN, weights_only=True)
    out = {"meta_bin": str(META_BIN), "meta_bin_sha256": sha256_file(META_BIN),
           "meta_bin_classes": len(w2c), "meta_bin_val_images": len(val_wnids),
           "meta_bin_wnids_equal_folders": sorted(w2c) == wnids}
    with tarfile.open(DEVKIT, "r:gz") as tar:
        names = tar.getnames()
        meta_name = next(n for n in names if n.endswith("data/meta.mat"))
        gt_name = next(n for n in names if n.endswith("data/ILSVRC2012_validation_ground_truth.txt"))
        meta = scipy.io.loadmat(io.BytesIO(tar.extractfile(meta_name).read()), squeeze_me=True)["synsets"]
        gt = [int(v) for v in tar.extractfile(gt_name).read().decode().split()]
    ids = [int(s[0]) for s in meta]
    devkit_wnids = [str(s[1]) for s in meta]
    num_children = [int(s[4]) for s in meta]
    leaf = {i: w for i, w, k in zip(ids, devkit_wnids, num_children) if k == 0}
    by_id = [leaf[i] for i in range(1, 1001)]
    out.update({"devkit": str(DEVKIT), "devkit_sha256": sha256_file(DEVKIT), "devkit_meta_member": meta_name,
                "devkit_ground_truth_member": gt_name, "devkit_leaf_synsets": len(leaf),
                "devkit_wnid_set_equals_folders": sorted(by_id) == wnids,


                "devkit_id_order_equals_sorted_wnids": by_id == wnids,
                "devkit_first_ids": {str(i): by_id[i - 1] for i in (1, 2, 3)},
                "devkit_val_labels": len(gt),
                "devkit_ground_truth_equals_meta_bin": [by_id[g - 1] for g in gt] == list(val_wnids)})
    out["torchvision_categories"] = torchvision_category_check(wnids, w2c)
    out["passed"] = bool(out["meta_bin_wnids_equal_folders"] and out["devkit_wnid_set_equals_folders"]
                         and out["devkit_ground_truth_equals_meta_bin"] and len(gt) == 50000
                         and out["torchvision_categories"]["passed"])
    out["_val_wnids"] = list(val_wnids)
    out["_w2c"] = {k: list(v) for k, v in w2c.items()}
    del np
    return out


def torchvision_category_check(wnids: list[str], w2c: dict) -> dict:
    ""
    import torchvision
    from torchvision.models import ResNet50_Weights

    cats = list(ResNet50_Weights.IMAGENET1K_V1.meta["categories"])

    def norm(s: str) -> str:
        return s.lower().replace("-", " ").replace("'", "").strip()

    exact = [i for i, w in enumerate(wnids) if cats[i] in w2c[w]]
    loose = [i for i, w in enumerate(wnids) if any(norm(cats[i]) == norm(n) or norm(cats[i]) in norm(n)
                                                  for n in w2c[w])]
    shifted = sum(1 for i, w in enumerate(wnids[:-1]) if cats[i + 1] in w2c[w])
    unmatched = [{"index": i, "wnid": w, "torchvision": cats[i], "meta_bin": list(w2c[w])}
                 for i, w in enumerate(wnids) if i not in set(loose)]
    return {"torchvision": torchvision.__version__, "categories": len(cats), "exact_name_matches": len(exact),
            "name_matches_after_normalization": len(loose), "matches_if_shifted_by_one": shifted,
            "unmatched": unmatched, "passed": len(cats) == 1000 and len(loose) >= 990 and shifted < 50}


def build_subset(wnids: list[str], val_wnids: list[str]) -> list[dict]:
    picks = {}
    for c, wnid in enumerate(wnids):
        files = sorted(os.listdir(VAL / wnid))
        if len(files) != 50:
            raise SystemExit(f"REFUSED: {VAL / wnid} holds {len(files)} files, not 50")
        ranked = sorted(files, key=lambda f: key(SALT_SUBSET, wnid, f))[:PER_CLASS]
        picks[wnid] = ranked
    entries = []
    for r in range(ROUNDS):
        for wnid in sorted(wnids, key=lambda w: key(SALT_ROUND, str(r), w)):
            f = picks[wnid][r]
            number = int(f.split("_")[-1].split(".")[0])
            if val_wnids[number - 1] != wnid:
                raise SystemExit(f"REFUSED: {f} lies in {wnid} but meta.bin labels it {val_wnids[number - 1]}")
            path = VAL / wnid / f
            entries.append({"order": len(entries), "round": r, "wnid": wnid, "class_index": wnids.index(wnid),
                            "file": f"val/{wnid}/{f}", "val_number": number, "bytes": path.stat().st_size,
                            "sha256": sha256_file(path)})
    return entries


def build_calibration(wnids: list[str]) -> list[dict]:
    rows = []
    for wnid in sorted(wnids, key=lambda w: key(SALT_CALIB_ORDER, w)):
        files = sorted(os.listdir(TRAIN / wnid))
        f = min(files, key=lambda name: key(SALT_CALIB, wnid, name))
        path = TRAIN / wnid / f
        rows.append({"order": len(rows), "wnid": wnid, "class_index": wnids.index(wnid), "file": f"train/{wnid}/{f}",
                     "class_train_files": len(files), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    t0 = time.time()
    wnids = sorted_wnids()
    check = check_class_index(wnids)
    if not check["passed"]:
        raise SystemExit(f"REFUSED: class index check failed: { {k: v for k, v in check.items() if not k.startswith('_')} }")
    val_wnids, w2c = check.pop("_val_wnids"), check.pop("_w2c")
    subset = build_subset(wnids, val_wnids)
    calib = build_calibration(wnids)
    if {e["file"] for e in subset} & {e["file"] for e in calib}:
        raise SystemExit("REFUSED: subset and calibration overlap")
    rule_sha = sha256_file(Path(__file__))
    common = {"root": str(ROOT), "rule_file": "manifest/build_manifest.py", "rule_sha256": rule_sha,
              "salts": {"subset": SALT_SUBSET, "round": SALT_ROUND, "calibration": SALT_CALIB,
                        "calibration_order": SALT_CALIB_ORDER},
              "built": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    manifest = {"format": FORMAT, "kind": "evaluation subset", **common, "per_class": PER_CLASS, "images": len(subset),
                "classes": len(wnids), "class_index_rule": "position of the wnid in sorted order",
                "wnids_sorted_sha256": hashlib.sha256("\n".join(wnids).encode()).hexdigest(),
                "class_names": {w: w2c[w] for w in wnids}, "entries": subset}
    calibration = {"format": FORMAT, "kind": "calibration list (training split)", **common, "images": len(calib),
                   "entries": calib}
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("manifest.json", manifest), ("calibration.json", calibration)):
        (out / name).write_text(json.dumps(obj, indent=1, sort_keys=True))
    check["wall_s"] = time.time() - t0
    (out / "class_index_check.json").write_text(json.dumps(check, indent=1, sort_keys=True))
    with open(out / "manifest.sha256", "w") as f:
        for e in subset:
            f.write(f"{e['sha256']}  {ROOT / e['file']}\n")
    with open(out / "calibration.sha256", "w") as f:
        for e in calib:
            f.write(f"{e['sha256']}  {ROOT / e['file']}\n")
    per_class = {}
    for e in subset:
        per_class[e["class_index"]] = per_class.get(e["class_index"], 0) + 1
    print(json.dumps({"subset": len(subset), "classes_with_5": sum(v == 5 for v in per_class.values()),
                      "first_1000_distinct_classes": len({e["class_index"] for e in subset[:1000]}),
                      "calibration": len(calib), "check_passed": check["passed"], "wall_s": round(check["wall_s"], 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
