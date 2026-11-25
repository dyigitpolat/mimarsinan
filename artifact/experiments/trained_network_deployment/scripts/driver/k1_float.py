""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common
import inventory
import loader
from e01_provider import E01Cifar100Provider

HARNESS_FIRST256_SHA256 = "d38d35d3a08bb73cd059d1df6f339ff72a56df110e29daae844901b9d316e3b6"


def e01_inputs(member) -> tuple[torch.Tensor, np.ndarray, dict]:
    provider = E01Cifar100Provider(str(common.DATASETS_PATH), preprocessing=inventory.preprocessing(member))
    ds = provider._get_test_dataset()
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs).contiguous()
    return x, np.asarray(ys, dtype=np.int64), provider.digests()


def evaluate(model, x: torch.Tensor, dtype=torch.float32, batch: int = 500) -> np.ndarray:
    model = model.to(dtype).eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            out.append(model(x[i: i + batch].to(dtype)).double())
    return torch.cat(out).numpy()


def run_member(member, out: Path) -> dict:
    t0 = time.time()
    rec = {"member": member.id, "label": member.label, "model_type": member.model_type,
           "model_config": member.model_config, "params": member.params}
    x, y, digests = e01_inputs(member)
    rec["provider"] = digests
    rec["inputs_fp32_sha256"] = common.sha256_bytes(x.numpy().tobytes())
    if member.family == "cnn":
        first = common.sha256_bytes(x[:256].numpy().tobytes())
        rec["first256_equal_harness_inputs"] = first == HARNESS_FIRST256_SHA256
    model, split, k0 = loader.load_checkpoint(member, member.float_ckpt)
    rec["K0_float"] = k0
    logits = evaluate(model, x)
    pred = logits.argmax(1)
    correct = int((pred == y).sum())
    top2 = np.sort(logits, axis=1)[:, -2:]
    rec["K1"] = {"recorded": member.float_ckpt.e01_correct, "cpu_fp32_correct": correct,
                 "difference": correct - member.float_ckpt.e01_correct,
                 "passed": correct == member.float_ckpt.e01_correct,
                 "near_ties_below_1e-4": int(((top2[:, 1] - top2[:, 0]) < 1e-4).sum())}
    if member.family == "vit":
        gate = np.load(common.B / member.harness["predictions"]["float"]).astype(np.int64)
        rec["K1"]["agree_gate_float_predictions"] = int((pred == gate).sum())
    rec["files"] = {"float_pred_fp32.npy": common.save_npy(out / member.id / "float_pred_fp32.npy", pred),
                    "float_logits_fp32.npy": common.save_npy(out / member.id / "float_logits_fp32.npy",
                                                             logits.astype(np.float32)),
                    "e01_labels.npy": common.save_npy(out / member.id / "e01_labels.npy", y)}
    rec["K0_qat"] = {}
    for role, ckpt in member.qat.items():
        _, qsplit, qk0 = loader.load_checkpoint(member, ckpt)
        rec["K0_qat"][role] = {**qk0, "lane_state": qsplit.lanes if qsplit.lanes else None,
                               "grid_steps": len(qsplit.grids)}
    rec["wall_s"] = time.time() - t0
    rec["peak_rss_kb"] = common.peak_rss_kb()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out)
    for mid in a.members:
        member = inventory.MEMBERS[mid]
        start = common.driver_code_sha256()
        rec = {"toolchain": common.toolchain_provenance(), "driver_code_sha256": start,
               "torch": torch.__version__, "threads": torch.get_num_threads(), **run_member(member, out)}
        common.refuse_if_changed(start, f"the {mid} K0 and K1 checks")
        common.atomic_write_json(out / mid / "k0_k1.json", rec)
        k1 = rec["K1"]
        print(f"{mid}: K1 {k1['cpu_fp32_correct']} vs recorded {k1['recorded']} ({k1['difference']:+d}); "
              f"K0 float {rec['K0_float']['module_keys']} keys; QAT {list(rec['K0_qat'])}; {rec['wall_s']:.0f} s",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
