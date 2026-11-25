""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

import common

STEP_MODELS = ("Weight Preloading", "Torch Mapping", "Readout Law", "Weight Quantization")


def delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach().double(), b.detach().double()
    if a.shape != b.shape:
        return {"shape_equal": False, "max_abs": None, "elements_differ": None, "elements": int(a.numel())}
    d = (a - b).abs()
    return {"shape_equal": True, "max_abs": float(d.max()) if d.numel() else 0.0,
            "elements_differ": int((d > 0).sum()), "elements": int(d.numel())}


def _equal(entry: dict) -> bool:
    return bool(entry["shape_equal"] and entry["elements_differ"] == 0)


def row_passed(row: dict) -> bool:
    return _equal(row["weight"]) and _equal(row["bias"]) and row.get("scales_equal", True)


def compare_states(declared: dict, loaded: dict) -> dict:
    ""
    per_key = {k: delta(declared[k], loaded[k]) for k in sorted(declared) if k in loaded}
    keys_equal = sorted(declared) == sorted(loaded)
    return {"keys": len(declared), "keys_equal": keys_equal,
            "tensors_differ": sorted(k for k, v in per_key.items() if not _equal(v)),
            "max_abs": max(((v["max_abs"] or 0.0) for v in per_key.values()), default=0.0),
            "passed": keys_equal and all(_equal(v) for v in per_key.values())}


def _pairs(a, b):
    pa, pb = a.get_perceptrons(), b.get_perceptrons()
    if len(pa) != len(pb):
        raise SystemExit(f"REFUSED: the step models hold {len(pa)} and {len(pb)} perceptrons")
    return zip(pa, pb)


def compare_unchanged(before, after) -> list:
    ""
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    tr = PerceptronTransformer()
    with torch.no_grad():
        return [{"perceptron": str(q.name), "weight": delta(tr.get_effective_weight(p), tr.get_effective_weight(q)),
                 "bias": delta(tr.get_effective_bias(p), tr.get_effective_bias(q))} for p, q in _pairs(before, after)]


def compare_projection(pre, post, *, bits: int, two_scale: bool, fractional_scale: bool = False,
                       float_bias: bool = False) -> list:
    ""
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization)
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
    from mimarsinan.transformations.quantization_bounds import LAYER_WEIGHT_BITS

    tr = PerceptronTransformer()
    proj = copy.deepcopy(pre)
    napq = NormalizationAwarePerceptronQuantization(int(bits), "cpu", rate=1.0, two_scale=bool(two_scale),
                                                    fractional_scale=bool(fractional_scale),
                                                    float_bias=bool(float_bias))
    rows = []
    with torch.no_grad():
        for p, q in _pairs(proj, post):
            if getattr(q, LAYER_WEIGHT_BITS, None) is not None:
                setattr(p, LAYER_WEIGHT_BITS, getattr(q, LAYER_WEIGHT_BITS))
        for p in proj.get_perceptrons():
            napq.transform(p)
        for p, q in _pairs(proj, post):
            scales = [[float(q.parameter_scale), float(p.parameter_scale)], [float(q.bias_scale), float(p.bias_scale)]]
            rows.append({"perceptron": str(q.name),
                         "weight": delta(tr.get_effective_weight(p), tr.get_effective_weight(q)),
                         "bias": delta(tr.get_effective_bias(p), tr.get_effective_bias(q)),
                         "parameter_scale": scales[0], "bias_scale": scales[1],
                         "scales_equal": scales[0][0] == scales[0][1] and scales[1][0] == scales[1][1]})
    return rows


def _load(work: Path, step: str):
    obj = torch.load(work / f"{step}.model.pt", map_location="cpu", weights_only=False)
    return obj[0] if isinstance(obj, tuple) else obj


def _summary(rows: list) -> dict:
    return {"perceptrons": len(rows), "differ": [r["perceptron"] for r in rows if not row_passed(r)],
            "max_abs_weight": max(((r["weight"]["max_abs"] or 0.0) for r in rows), default=0.0),
            "max_abs_bias": max(((r["bias"]["max_abs"] or 0.0) for r in rows), default=0.0),
            "passed": all(row_passed(r) for r in rows)}


def check_work(work: Path, weight_source: Path, weight_source_sha256: str, *, bits: int, two_scale: bool,
               fractional_scale: bool = False, float_bias: bool = False, stdout_log: Path | None = None) -> dict:
    ""
    work = Path(work)
    missing = [s for s in STEP_MODELS if not (work / f"{s}.model.pt").exists()]
    if missing and missing != ["Readout Law"]:
        return {"applicable": False, "reason": f"no cached step model for {missing}", "passed": None}
    if common.sha256_file(weight_source) != weight_source_sha256:
        raise SystemExit(f"REFUSED: {weight_source} differs from the session's weight_source digest")
    if missing == ["Readout Law"]:
        return _spiking_check(work, weight_source, stdout_log)
    rec = {"applicable": True, "bits": int(bits), "two_scale": bool(two_scale)}
    if fractional_scale or float_bias:
        rec.update(fractional_scale=bool(fractional_scale), float_bias=bool(float_bias))
    declared = torch.load(weight_source, map_location="cpu", weights_only=True)
    rec["a_preload_equals_checkpoint"] = compare_states(declared, _load(work, "Weight Preloading").state_dict())
    del declared
    tm, pre, post = (_load(work, s) for s in STEP_MODELS[1:])
    rows_b = compare_unchanged(tm, pre)
    rows_c = compare_projection(pre, post, bits=bits, two_scale=two_scale, fractional_scale=fractional_scale,
                                float_bias=float_bias)
    rec["b_torch_mapping_to_readout_law_unchanged"] = _summary(rows_b)
    rec["c_weight_quantization_equals_rate1_projection"] = _summary(rows_c)
    ledger = work / "Weight Quantization.adaptation_ledger.json"
    rec["weight_quantization_ledger_totals"] = json.loads(ledger.read_text()).get("totals") if ledger.exists() else None
    rec["d_no_training_step"] = training_steps(work, stdout_log)
    rec["passed"] = bool(rec["a_preload_equals_checkpoint"]["passed"] and rec["d_no_training_step"]["passed"]
                         and rec["b_torch_mapping_to_readout_law_unchanged"]["passed"]
                         and rec["c_weight_quantization_equals_rate1_projection"]["passed"])
    rec["per_perceptron"] = {"b": rows_b, "c": rows_c}
    return rec


def ledger_steps(work: Path) -> dict:
    ""
    out = {}
    for path in sorted(Path(work).glob("*.adaptation_ledger.json")):
        totals = json.loads(path.read_text()).get("totals") or {}
        out[path.name] = {k: totals.get(k) for k in ("total_steps", "endpoint_steps", "recovery_steps")}
    return {"ledgers": out, "passed": bool(out) and all(not any(v.values()) for v in out.values())}


def resolved_config(stdout_log: Path) -> dict | None:
    ""
    import ast

    if stdout_log is None or not Path(stdout_log).is_file():
        return None
    lines = Path(stdout_log).read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("Deployment pipeline")), None)
    if start is None:
        return None
    out = {}
    for ln in lines[start + 1:]:
        if not ln.startswith("  ") or ": " not in ln:
            break
        key, _, value = ln.strip().partition(": ")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def cycle_summaries(stdout_log: Path | None) -> dict:
    ""
    import re

    if stdout_log is None or not Path(stdout_log).is_file():
        return {"ladders": {}, "skipped": []}
    text = Path(stdout_log).read_text().splitlines()
    ladders, skipped, cur = {}, [], None
    for ln in text:
        m = re.match(r"\[(\w+)\] Cycle summary \((\d+) cycles\):", ln)
        if m:
            cur = ladders.setdefault(m[1], [])
            continue
        m = re.match(r"\s+\[\d+\] rate=([0-9.]+) committed=[0-9.]+ post_acc=([0-9.]+) lr=([0-9.e+-]+) outcome=(\w+)", ln)
        if m and cur is not None:
            cur.append({"rate": float(m[1]), "post_acc": float(m[2]), "lr": float(m[3]), "outcome": m[4]})
            continue
        cur = None
        m = re.search(r"\[MBH-GATE\] tuner=(\w+) entry_fast_path", ln)
        if m:
            skipped.append(m[1])
    return {"ladders": ladders, "skipped": sorted(set(skipped))}


def ladder_steps_per_rate(config: dict, tuner: str) -> int | None:
    ""
    from mimarsinan.tuning.orchestration.lif_exact_qat import lif_subsumed_ladder_steps
    from mimarsinan.tuning.orchestration.tuning_policy import FAST_LADDER_STEPS_PER_RATE as FAST

    get = config.get
    if tuner == "ClampTuner":
        return lif_subsumed_ladder_steps(config, "clamp_rate", int(get("clamp_fast_steps_per_rate", FAST)))
    if tuner == "ActivationQuantizationTuner":
        return lif_subsumed_ladder_steps(config, "quantization_rate",
                                         int(get("manager_rate_fast_steps_per_rate", FAST)))
    if tuner == "NoiseTuner":
        return lif_subsumed_ladder_steps(config, "noise_rate", int(get("manager_rate_fast_steps_per_rate", FAST)))
    if tuner == "ActivationAdaptationTuner":
        return int(get("activation_adaptation_fast_steps_per_rate", FAST))
    if tuner == "NormalizationAwarePerceptronQuantizationTuner":
        return int(get("wq_fast_steps_per_rate", 0))
    if tuner == "LIFAdaptationTuner":
        from mimarsinan.tuning.orchestration.lif_adaptation_plan import LifAdaptationPlan

        return int(LifAdaptationPlan.resolve(config).blend_fast_steps_per_rate)
    return None


_ENDPOINT_FIELDS = {"target": float, "entry": float, "exit": float, "budget": int, "steps_used": int,
                    "engaged": "bool", "reached": "bool", "rolled_back": "bool", "target_floor": float,
                    "floor_lifted": "bool", "armed": "bool", "divergence_rescued": "bool", "decoupled": "bool"}


def endpoint_legs(stdout_log: Path | None) -> list:
    ""
    import re

    if stdout_log is None or not Path(stdout_log).is_file():
        return []
    out = []
    for ln in Path(stdout_log).read_text().splitlines():
        if not ln.startswith("[MBH-ENDPOINT] "):
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", ln))
        leg = {"tuner": kv.get("tuner")}
        for key, kind in _ENDPOINT_FIELDS.items():
            if key in kv:
                leg[key] = (kv[key] == "True") if kind == "bool" else kind(kv[key])
        out.append(leg)
    return out


def training_legs(work: Path, stdout_log: Path | None) -> dict:
    ""
    rec = training_steps(work, stdout_log)
    rec.pop("passed", None)
    rec.pop("ledgers_passed", None)
    rec["endpoint_legs"] = endpoint_legs(stdout_log)
    return rec


def check_preload(work: Path, weight_source: Path, weight_source_sha256: str) -> dict:
    ""
    work = Path(work)
    if not (work / "Weight Preloading.model.pt").exists():
        return {"applicable": False, "reason": "no cached Weight Preloading model", "passed": None}
    if common.sha256_file(weight_source) != weight_source_sha256:
        raise SystemExit(f"REFUSED: {weight_source} differs from the session's weight_source digest")
    declared = torch.load(weight_source, map_location="cpu", weights_only=True)
    return compare_states(declared, _load(work, "Weight Preloading").state_dict())


def training_steps(work: Path, stdout_log: Path | None) -> dict:
    ""
    rec = ledger_steps(work)
    walls = Path(work) / "ft_pass_walls.json"
    passes = json.loads(walls.read_text()).get("passes", []) if walls.exists() else []
    rec["fine_tune_passes"] = [{"label": p.get("label"), "wall_s": p.get("wall_s")} for p in passes]
    config = resolved_config(stdout_log)
    cycles = cycle_summaries(stdout_log)
    ladders = {}
    for tuner, rows in sorted(cycles["ladders"].items()):
        steps = None if config is None else ladder_steps_per_rate(config, tuner)
        ladders[tuner] = {"cycles": len(rows), "steps_per_rate": steps,
                          "ladder_steps": None if steps is None else steps * len(rows),
                          "lr_per_cycle": [r["lr"] for r in rows], "post_acc_per_cycle": [r["post_acc"] for r in rows]}
    rec["fast_ladders"] = ladders
    rec["ladders_skipped_at_entry"] = cycles["skipped"]
    rec["resolved_config_read"] = config is not None
    rec["ledgers_passed"] = rec.pop("passed")
    rec["passed"] = bool(rec["ledgers_passed"] and not passes and config is not None
                         and all(v["steps_per_rate"] == 0 for v in ladders.values()))
    return rec


def _spiking_check(work: Path, weight_source: Path, stdout_log: Path | None = None) -> dict:
    ""
    declared = torch.load(weight_source, map_location="cpu", weights_only=True)
    rec = {"applicable": True, "spiking": True,
           "a_preload_equals_checkpoint": compare_states(declared, _load(work, "Weight Preloading").state_dict()),
           "b_c": "not applicable: a spiking pipeline has no Readout Law step; its adaptation steps move weights by "
                  "their rules at rate 1",
           "d_no_training_step": training_steps(work, stdout_log)}
    rec["passed"] = bool(rec["a_preload_equals_checkpoint"]["passed"] and rec["d_no_training_step"]["passed"])
    rec["per_perceptron"] = {}
    return rec


def check_row(row_dir: Path) -> dict:
    ""
    session = json.loads((row_dir / "session.json").read_text())
    config = json.loads((row_dir / "config.json").read_text())
    if common.sha256_file(row_dir / "config.json") != session["config_sha256"]:
        raise SystemExit("REFUSED: config.json differs from the session's digest")
    return check_work(Path(session["working_directory"]), Path(session["weight_source"]["path"]),
                      session["weight_source"]["sha256"], **declared_grids(config),
                      stdout_log=row_dir / "pipeline_stdout.log")


def declared_grids(config: dict) -> dict:
    ""
    plat, dp = config["platform_constraints"], config["deployment_parameters"]
    over = plat.get("target_overrides", {}).get("weight_bits")
    bits = int(over["value"]) if over else int(plat.get("weight_bits", 8))
    return {"bits": bits, "two_scale": bool(dp.get("wq_two_scale_projection", False)),
            "fractional_scale": plat.get("weight_scale_rule") == "fractional",
            "float_bias": plat.get("bias_resolution") == "law"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True)
    ap.add_argument("--rows", default=str(common.ROWS), help="the rows directory holding <row>/")
    a = ap.parse_args()
    torch.set_num_threads(1)
    start = common.driver_code_sha256()
    row_dir = Path(a.rows) / a.row
    rec = {"row": a.row, **check_row(row_dir)}
    common.refuse_if_changed(start, "the no-update check")
    rec["driver_code_sha256"] = start
    rec["peak_rss_kb"] = common.peak_rss_kb()
    common.atomic_write_json(row_dir / "no_update.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_perceptron", "driver_code_sha256")}, indent=1))
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
