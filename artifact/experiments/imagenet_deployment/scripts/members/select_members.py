""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driver"))
from w38common import state_digest

CENSUS = Path(__file__).resolve().parents[3] / "10_wave1" / "cl_cov" / "census_run_r4" / "records.jsonl"
MODULES = {"resnet": "residual", "densenet": "dense concatenation",
           "mobilenetv2": "inverted residual, depthwise convolution on the host",
           "vision_transformer": "attention, host MACs"}
REQUIRED = "resnet50"
CONVERTING = ("converts", "converts with CPU share")
FORMAT = "w38-members/1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census_verdicts() -> dict:
    out = {}
    for line_no, line in enumerate(open(CENSUS), 1):
        r = json.loads(line)
        if r["population"] == "torchvision" and r["check"] == "verdict":
            out[r["member"]] = {"class": r["outcome"]["class"], "host_mac_share": r["outcome"].get("host_mac_share"),
                                "relative_error": r["outcome"].get("relative_error"), "side": r["side"],
                                "record": f"10_wave1/cl_cov/census_run_r4/records.jsonl:{line_no}"}
    return out


def transforms_spec(weights) -> dict:
    t = weights.transforms()
    return {"class": type(t).__name__, "crop_size": [int(v) for v in t.crop_size],
            "resize_size": [int(v) for v in t.resize_size], "interpolation": str(t.interpolation.value),
            "antialias": bool(t.antialias), "mean": [float(v) for v in t.mean], "std": [float(v) for v in t.std],
            "repr": repr(t)}


def select() -> tuple[list, dict]:
    import torchvision
    from torchvision import models

    verdicts = census_verdicts()
    candidates = {}
    for name in models.list_models(module=models):
        builder = models.get_model_builder(name)
        module = builder.__module__.rsplit(".", 1)[-1]
        if module not in MODULES and name != REQUIRED:
            continue
        try:
            enum = models.get_model_weights(name)
        except (ValueError, KeyError):
            continue
        default = enum.DEFAULT
        if "ImageNet-1K" not in default.meta.get("_metrics", {}):
            continue
        candidates[name] = {"module": module, "params": int(default.meta["num_params"]),
                            "census": verdicts.get(name), "default": default}
    picks = [REQUIRED]
    trace = {}
    for module in MODULES:
        pool = sorted((v["params"], n) for n, v in candidates.items()
                      if v["module"] == module and n != REQUIRED and v["census"]
                      and v["census"]["class"] in CONVERTING)
        trace[module] = [{"name": n, "params": p, "census_class": candidates[n]["census"]["class"]} for p, n in pool]
        excluded = sorted(n for n, v in candidates.items() if v["module"] == module and n != REQUIRED
                          and not (v["census"] and v["census"]["class"] in CONVERTING))
        trace[module + "_excluded_not_converting"] = excluded
        if not pool:
            raise SystemExit(f"REFUSED: module {module} has no converting member with ImageNet-1K weights")
        picks.append(pool[0][1])
    if candidates[REQUIRED]["census"]["class"] not in CONVERTING:
        raise SystemExit("REFUSED: ResNet-50 does not convert in census r4")
    rows = []
    for name in picks:
        c = candidates[name]
        w = c["default"]
        m = w.meta["_metrics"]["ImageNet-1K"]
        rows.append({"member": name, "module": c["module"], "class": MODULES.get(c["module"], "residual (required)"),
                     "weights_enum": f"{type(w).__name__}.{w.name}", "weights_url": w.url,
                     "published_acc1": float(m["acc@1"]), "published_acc5": float(m["acc@5"]),
                     "num_params": c["params"], "gflops": w.meta.get("_ops"), "recipe": w.meta.get("recipe"),
                     "min_size": list(w.meta.get("min_size", ())), "transforms": transforms_spec(w),
                     "census_r4": c["census"], "torchvision": torchvision.__version__})
    return rows, trace


def canonical_state(member: str, weights) -> dict:
    ""
    from torchvision import models

    model = models.get_model(member, weights=weights)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def fetch(row: dict, download: bool) -> dict:
    import torch
    from torchvision import models

    enum_name, member = row["weights_enum"].split(".")
    w = getattr(models, enum_name)[member]
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    path = hub / os.path.basename(w.url)
    existed = path.exists()
    t0 = time.time()
    if download:
        w.get_state_dict(progress=False, check_hash=True)
    elif not existed:
        raise SystemExit(f"REFUSED: {path} absent; run with --fetch")
    state = canonical_state(row["member"], w)
    fresh = models.get_model(row["member"], weights=None)
    result = fresh.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in fresh.parameters())
    return {"file": str(path), "file_sha256": sha256_file(path), "file_bytes": path.stat().st_size,
            "hash_prefix_in_name": path.stem.rsplit("-", 1)[-1], "cached_before": existed,
            "fetch_wall_s": round(time.time() - t0, 2), "canonical_state_keys": len(state),
            "canonical_state_digest": state_digest(state),
            "canonical_state_rule": "get_model(member, weights=<enum>).state_dict(), digest by state_digest()",
            "strict_load_missing": list(result.missing_keys), "strict_load_unexpected": list(result.unexpected_keys),
            "parameters_counted": int(n_params), "parameters_equal_meta": int(n_params) == row["num_params"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    rows, trace = select()
    for row in rows:
        row["weights_file"] = fetch(row, a.fetch)
        if not row["weights_file"]["file_sha256"].startswith(row["weights_file"]["hash_prefix_in_name"]):
            raise SystemExit(f"REFUSED: {row['weights_file']['file']} sha256 does not start with its name's prefix")
    doc = {"format": FORMAT, "rule_file": "members/select_members.py", "rule_sha256": sha256_file(Path(__file__)),
           "census": str(CENSUS), "census_sha256": sha256_file(CENSUS), "selection_trace": trace,
           "written": time.strftime("%Y-%m-%d %H:%M:%S %z"), "members": rows}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps([{k: r[k] for k in ("member", "weights_enum", "published_acc1", "num_params")}
                      | {"sha256": r["weights_file"]["file_sha256"][:16]} for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
