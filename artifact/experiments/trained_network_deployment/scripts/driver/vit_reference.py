""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import common
import inventory


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    start = common.driver_code_sha256()
    member = inventory.MEMBERS["ViT"]
    out = Path(a.out) / "ViT"
    out.mkdir(parents=True, exist_ok=True)
    labels = np.load(common.RECORDS / "wi1" / "ViT" / "e01_labels.npy")
    rec = {"member": "ViT", "source": "vit_gate per-sample predictions (no re-evaluation, design.md WI-2)", "jobs": {}}
    for role, rel in member.harness["predictions"].items():
        src = common.B / rel
        record = json.loads((src.parent / "result.json").read_text())
        kind = next(iter(record["predictions"]))
        listed = record["predictions"][kind]
        digest = common.sha256_file(src)
        if digest != listed["sha256"] or src.name != listed["file"]:
            raise SystemExit(f"REFUSED: {rel} differs from its record's predictions entry")
        pred = np.load(src).astype(np.int64)
        correct = int((pred == labels).sum())
        if correct != int(record["counts"][kind]):
            raise SystemExit(f"REFUSED: {rel} counts {correct}, its record {record['counts'][kind]}")
        dst = out / f"gate_pred_{role}.npy"
        shutil.copyfile(src, dst)
        rec["jobs"][role] = {"file": rel, "sha256": digest, "copy_sha256": common.sha256_file(dst),
                             "count": correct, "record_count": int(record["counts"][kind]),
                             "record": str((src.parent / "result.json").relative_to(common.RA))}
    common.refuse_if_changed(start, "the ViT reference")
    rec["driver_code_sha256"] = start
    common.atomic_write_json(out / "vit_reference.json", rec)
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
