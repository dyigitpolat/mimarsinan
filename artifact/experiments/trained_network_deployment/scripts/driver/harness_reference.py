""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common
import inventory

COPY = common.DRIVER / "harness_copy"
KEEP_ON_COPY = ("HERE", "SNAPSHOT_DIR", "SNAPSHOT_SRC", "SNAPSHOT_MANIFEST")


def bind_copy(harness: str):
    ""
    sys.path.insert(0, str(COPY / harness))
    import screen_common as sc

    remapped = {}
    for name, value in list(vars(sc).items()):
        if not isinstance(value, Path) or name in KEEP_ON_COPY:
            continue
        try:
            rel = value.resolve().relative_to(COPY)
        except ValueError:
            continue
        setattr(sc, name, common.B / rel)
        remapped[name] = str(common.B / rel)
    if Path(sc.HERE).resolve() != (COPY / harness).resolve():
        raise SystemExit(f"REFUSED: screen_common imported from {sc.HERE}, not the copy")
    return sc, remapped


def build_screen(sc, member, rules):
    from lane_quant import lane_quantize
    from screen_models import build_member, count_params

    spec = sc.arm_spec(rules, member.harness["arm"])
    q = rules["quantization"]
    torch.manual_seed(int(rules["recipe"]["seed"]))
    model = build_member(spec, rules["num_classes"])
    if count_params(model) != int(spec["params"]):
        raise SystemExit("REFUSED: parameter count differs from the declared arm")
    gm, census = lane_quantize(model, int(q["weight_bits"]), int(q["output_bits"]))
    expected = rules["lane_census"][spec["id"]]
    if {k: census.get(k) for k in expected} != expected or not census["rounding_as_declared"]:
        raise SystemExit(f"REFUSED: lane census {census} differs from the declared {expected}")
    return gm, census, {"arm": spec["id"], "params": int(spec["params"])}


def build_headline(sc, member, rules):
    import train_headline
    from lane_quant import lane_quantize
    from stage_bits import assign_stage_bits, bits_by_stage

    spec = sc.job_spec(rules, member.harness["job"])
    arch = sc.arch_spec(rules, spec["arch"])
    q = rules["quantization"]
    torch.manual_seed(int(rules["recipe"]["seed"]))
    model, params, census_params = train_headline.build_checked(arch, rules)
    gm, census = lane_quantize(model, int(q["walk_weight_bits"]), int(q["output_bits"]))
    stage = assign_stage_bits(gm, arch["units"], spec["bits"])
    set_bits = {s: sorted(v) for s, v in bits_by_stage(gm, arch["units"]).items()}
    if set_bits != {s + 1: [int(b)] for s, b in enumerate(spec["bits"])}:
        raise SystemExit(f"REFUSED: per-stage weight bits {set_bits} differ from the declared {spec['bits']}")
    expected = arch["lane_census"]
    if {k: census.get(k) for k in expected} != expected or not census["rounding_as_declared"]:
        raise SystemExit(f"REFUSED: lane census {census} differs from the declared {expected}")
    return gm, census, {"job": spec["id"], "params": params, "census_params": census_params,
                        "weight_bits_by_stage": stage}


def run(member, out: Path) -> dict:
    t0 = time.time()
    import copies

    copy_digests = copies.verify()
    harness = member.harness["harness"]
    sc, remapped = bind_copy(harness)
    if harness == "screen_v42p":
        digests = sc.verify_lock(sc.sha256_file(sc.RULES_MD))
    else:
        digests = sc.verify_lock(sc.launcher_rules_sha256())
    sc.verify_snapshot()
    mm = sc.import_snapshot_toolchain()
    rules = sc.load_rules()
    from lane_quant import lane_quantizers, set_lane_mode
    from screen_data import CifarArrayDataset, load_split, load_train_batch, make_loader, transforms_for

    ckpt = member.qat["qat"]
    if sc.sha256_file(ckpt.file) != ckpt.sha256:
        raise SystemExit("REFUSED: checkpoint sha256 differs from the inventory")
    state = torch.load(ckpt.file, map_location="cpu", weights_only=True)

    def build_loaded():
        ""
        model, census_, ident_ = (build_screen if harness == "screen_v42p" else build_headline)(sc, member, rules)
        load_ = model.load_state_dict(state, strict=True)
        set_lane_mode(model, "quantize")
        return model.eval(), census_, ident_, load_

    gm, census, ident, load = build_loaded()
    record = json.loads((common.RA / ckpt.record).read_text())
    recorded_lanes = {row["quantizer"]: row for row in record["lanes"] if row.get("quantizer")}
    lanes = {name: {"theta": float(m.act_theta.detach()), "shift": float(m.act_shift.detach()),
                    "signed": bool(m.signed), "calibrated": bool(m.calibrated), "rounding": m.rounding}
             for name, m in lane_quantizers(gm)}
    lane_mismatch = sorted(n for n, v in lanes.items() if n not in recorded_lanes
                           or v["theta"] != recorded_lanes[n]["theta"] or v["shift"] != recorded_lanes[n]["shift"]
                           or v["signed"] != recorded_lanes[n]["signed"])

    images, labels = load_train_batch()
    _, val_idx = load_split(labels)
    loader = make_loader(CifarArrayDataset(images, labels, val_idx, transforms_for("val")), 500, shuffle=False,
                         generator=None, workers=0, drop_last=False)
    xs, ys = zip(*[(x, y) for x, y in loader])
    x_all, y_all = torch.cat(xs), torch.cat(ys).numpy().astype(np.int64)

    def evaluate(model, dtype):
        with torch.no_grad():
            return torch.cat([model(x_all[i: i + 500].to(dtype)).double() for i in range(0, len(x_all), 500)]).numpy()

    t = time.time()
    logits32 = evaluate(gm, torch.float32)
    wall32 = time.time() - t
    t = time.time()
    logits64 = evaluate(build_loaded()[0].double(), torch.float64)
    wall64 = time.time() - t
    p32, p64 = logits32.argmax(1), logits64.argmax(1)
    top2 = np.sort(logits32, axis=1)[:, -2:]
    d = out / member.id
    files = {"harness_pred_fp32.npy": common.save_npy(d / "harness_pred_fp32.npy", p32),
             "harness_pred_fp64.npy": common.save_npy(d / "harness_pred_fp64.npy", p64),
             "harness_logits_fp32.npy": common.save_npy(d / "harness_logits_fp32.npy", logits32.astype(np.float32)),
             "harness_logits_fp64.npy": common.save_npy(d / "harness_logits_fp64.npy", logits64),
             "e01_labels.npy": common.save_npy(d / "e01_labels.npy", y_all),
             "e01_indices.npy": common.save_npy(d / "e01_indices.npy", np.asarray(val_idx, dtype=np.int64))}
    recorded = int(record["qat_val_correct"])
    c32, c64 = int((p32 == y_all).sum()), int((p64 == y_all).sum())
    return {
        "member": member.id, "label": member.label, "harness": harness, **ident, "checkpoint": ckpt.path,
        "checkpoint_sha256": ckpt.sha256, "copies_verified": len(copy_digests),
        "copies_listing_sha256": common.sha256_file(COPY / "COPIES.sha256"), "remapped_paths": remapped,
        "lock_verified": True, "lock_code_sha256": digests["code_sha256"], "snapshot_verified": True,
        "toolchain_snapshot": str(Path(mm.__file__).resolve()), "load_strict": str(load), "lane_census": census,
        "lanes": lanes, "lanes_equal_record": not lane_mismatch, "lane_mismatch": lane_mismatch,
        "torch": torch.__version__, "threads": torch.get_num_threads(), "e01_images": int(len(y_all)),
        "inputs_fp32_sha256": common.sha256_bytes(x_all.numpy().tobytes()),
        "first256_inputs_fp32_sha256": common.sha256_bytes(x_all[:256].contiguous().numpy().tobytes()),
        "K2": {"recorded": recorded, "cpu_fp32_correct": c32, "cpu_fp64_correct": c64,
               "fp32_minus_recorded": c32 - recorded, "passed": c32 == recorded,
               "fp32_vs_fp64_pred_differ": int((p32 != p64).sum()),
               "fp32_near_ties_below_1e-4": int(((top2[:, 1] - top2[:, 0]) < 1e-4).sum())},
        "eval_wall_s": {"fp32": wall32, "fp64": wall64}, "files": files, "wall_s": time.time() - t0,
        "peak_rss_kb": common.peak_rss_kb(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "mimarsinan" in sys.modules:
        raise SystemExit("REFUSED: mimarsinan imported before the harness snapshot")
    torch.set_num_threads(1)
    sys.path.insert(0, str(COPY))
    member = inventory.member(a.member)
    out = Path(a.out)
    start = common.driver_code_sha256()
    rec = run(member, out)
    common.refuse_if_changed(start, f"the {member.id} harness reference")
    rec["driver_code_sha256"] = start
    common.atomic_write_json(out / member.id / "harness_reference.json", rec)
    k2 = rec["K2"]
    print(f"{member.id}: recorded {k2['recorded']}, CPU fp32 {k2['cpu_fp32_correct']}, fp64 {k2['cpu_fp64_correct']}, "
          f"fp32/fp64 differ {k2['fp32_vs_fp64_pred_differ']}; lanes equal record {rec['lanes_equal_record']}; "
          f"{rec['wall_s']:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
