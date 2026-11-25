""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "driver"))
import w38common as c

BATCH = 50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--threads", type=int, default=1)
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
    entries = c.dev_entries()
    tf = weights.transforms()
    out = c.float_reference_dir(a.member, "dev")
    out.mkdir(parents=True, exist_ok=True)
    logits, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(entries), BATCH):
            batch = entries[i: i + BATCH]
            logits.append(model(torch.stack([tf(c.load_image(e)) for e in batch])).float().numpy())
            labels += [e["class_index"] for e in batch]
    z = np.concatenate(logits)
    y = np.asarray(labels, dtype=np.int64)
    pred = z.argmax(1)
    srt = np.sort(z, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    rec = {"format": "w38-float-dev/1", "member": a.member, "split": "dev", "weights_enum": row["weights_enum"],
           "weights_file_sha256": wf["file_sha256"], "canonical_state_digest": digest, "images": len(y),
           "dtype": "float32", "device": "cpu", "threads": a.threads, "torch": torch.__version__,
           "transform": repr(tf), "top1_correct": int((pred == y).sum()),
           "top5_correct": int((np.argsort(-z, axis=1)[:, :5] == y[:, None]).any(1).sum()),
           "top1_exact_ties": int((margin == 0).sum()), "top1_margin_below_1e-4": int((margin < 1e-4).sum()),
           "wall_s": time.time() - t0, "dev_sha256": c.sha256_file(c.DEV),
           "files": {"logits.npy": c.save_npy(out / "logits.npy", z), "pred.npy": c.save_npy(out / "pred.npy", pred),
                     "labels.npy": c.save_npy(out / "labels.npy", y)},
           "script_sha256": c.sha256_file(Path(__file__)), "written": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    rec["transform_equality_dev_first_16"] = c.check_transform_equality(row, entries[:16])
    c.atomic_write_json(out / "float.json", rec)
    print(json.dumps({k: rec[k] for k in ("member", "images", "top1_correct", "top1_exact_ties", "wall_s")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
