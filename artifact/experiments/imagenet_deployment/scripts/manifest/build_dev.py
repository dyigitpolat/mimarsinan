""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/data/ImageNet")
VAL = ROOT / "val"
META_BIN = ROOT / "meta.bin"
SALT_DEV, SALT_DEV_ORDER = "W38-dev-v1", "W38-dev-order-v1"
IMAGES = 256
FORMAT = "w38-imagenet-manifest/1"


def key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    manifest = json.loads((out / "manifest.json").read_text())
    calibration = json.loads((out / "calibration.json").read_text())
    subset_files = {e["file"] for e in manifest["entries"]}
    wnids = sorted(d for d in os.listdir(VAL) if (VAL / d).is_dir())
    if len(wnids) != 1000 or hashlib.sha256("\n".join(wnids).encode()).hexdigest() != manifest["wnids_sorted_sha256"]:
        raise SystemExit("REFUSED: the validation class folders are not the manifest's 1,000 sorted wnids")
    _, val_wnids = torch.load(META_BIN, weights_only=True)
    candidates = {}
    for wnid in wnids:
        files = sorted(os.listdir(VAL / wnid))
        if len(files) != 50:
            raise SystemExit(f"REFUSED: {VAL / wnid} holds {len(files)} files, not 50")
        rest = [f for f in files if f"val/{wnid}/{f}" not in subset_files]
        if len(rest) != 45:
            raise SystemExit(f"REFUSED: {wnid} keeps {len(rest)} non-subset files, not 45")
        candidates[wnid] = min(rest, key=lambda f: key(SALT_DEV, wnid, f))
    entries = []
    for wnid in sorted(wnids, key=lambda w: key(SALT_DEV_ORDER, w))[:IMAGES]:
        f = candidates[wnid]
        number = int(f.split("_")[-1].split(".")[0])
        if val_wnids[number - 1] != wnid:
            raise SystemExit(f"REFUSED: {f} lies in {wnid} but meta.bin labels it {val_wnids[number - 1]}")
        path = VAL / wnid / f
        entries.append({"order": len(entries), "wnid": wnid, "class_index": wnids.index(wnid),
                        "file": f"val/{wnid}/{f}", "val_number": number, "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    files = {e["file"] for e in entries}
    if files & subset_files or files & {e["file"] for e in calibration["entries"]}:
        raise SystemExit("REFUSED: the development list overlaps the subset or the calibration list")
    if len({e["class_index"] for e in entries}) != IMAGES:
        raise SystemExit("REFUSED: the development list repeats a class")
    doc = {"format": FORMAT, "kind": "development list (validation split, outside the subset)", "root": str(ROOT),
           "rule_file": "manifest/build_dev.py", "rule_sha256": sha256_file(Path(__file__)),
           "salts": {"dev": SALT_DEV, "dev_order": SALT_DEV_ORDER}, "images": len(entries),
           "classes": len({e["class_index"] for e in entries}), "subset_sha256": sha256_file(out / "manifest.json"),
           "disjoint_from_subset": True, "disjoint_from_calibration": True,
           "built": time.strftime("%Y-%m-%d %H:%M:%S %z"), "entries": entries}
    (out / "dev.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    with open(out / "dev.sha256", "w") as f:
        for e in entries:
            f.write(f"{e['sha256']}  {ROOT / e['file']}\n")
    print(json.dumps({"dev": len(entries), "classes": doc["classes"], "built": doc["built"],
                      "dev_json_sha256": sha256_file(out / "dev.json")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
