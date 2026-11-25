""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import w38common as c

STEP_MODELS = ("Weight Preloading", "Torch Mapping", "Readout Law", "Weight Quantization")


def delta(a, b) -> dict:
    a, b = a.detach().double(), b.detach().double()
    if a.shape != b.shape:
        return {"shape_equal": False, "max_abs": None, "elements_differ": None, "elements": int(a.numel())}
    d = (a - b).abs()
    return {"shape_equal": True, "max_abs": float(d.max()) if d.numel() else 0.0,
            "elements_differ": int((d > 0).sum()), "elements": int(d.numel())}


def _equal(entry: dict) -> bool:
    return bool(entry["shape_equal"] and entry["elements_differ"] == 0)


def _passed(row: dict) -> bool:
    return (_equal(row["weight"]) and _equal(row["bias"]) and row.get("scales_equal", True)
            and row.get("drive_equal", True))


def _pairs(a, b):
    pa, pb = a.get_perceptrons(), b.get_perceptrons()
    if len(pa) != len(pb):
        raise SystemExit(f"REFUSED: the step models hold {len(pa)} and {len(pb)} perceptrons")
    return zip(pa, pb)


def _load(work: Path, step: str):
    import torch

    obj = torch.load(work / f"{step}.model.pt", map_location="cpu", weights_only=False)
    return obj[0] if isinstance(obj, tuple) else obj


def _summary(rows: list) -> dict:
    return {"perceptrons": len(rows), "differ": [r["perceptron"] for r in rows if not _passed(r)][:50],
            "max_abs_weight": max(((r["weight"]["max_abs"] or 0.0) for r in rows), default=0.0),
            "max_abs_bias": max(((r["bias"]["max_abs"] or 0.0) for r in rows), default=0.0),
            "passed": all(_passed(r) for r in rows)}


def projection_settings(flat: dict) -> dict:
    ""
    from mimarsinan.pipelining.core.platform_constraints_resolver import resolve_bias_mode, resolve_wq_two_scale_projection

    out = {"bits": int(flat["weight_bits"]), "two_scale": bool(resolve_wq_two_scale_projection(dict(flat))),
           "two_scale_declared": bool(flat.get("wq_two_scale_projection", False))}
    try:
        from mimarsinan.mapping.support.grid_declarations import projection_grids
    except ImportError:
        declared = {k: flat.get(k) for k in ("weight_scale_rule", "bias_resolution") if flat.get(k) is not None}
        if declared:
            raise SystemExit(f"REFUSED: the config declares {declared}, but this head has no grid declarations")
        out.update(fractional_scale=False, float_bias=False, grids_source="none at this head (integer rule)")
        return out
    grids = projection_grids(dict(flat), two_scale=out["two_scale_declared"],
                             on_chip_bias=lambda: resolve_bias_mode(dict(flat)) == "on_chip")
    out.update(fractional_scale=bool(grids.fractional_scale), float_bias=bool(grids.law_bias),
               grids_source="mapping.support.grid_declarations.projection_grids")
    return out


def make_projection(settings: dict):
    ""
    from mimarsinan.transformations.normalization_aware_perceptron_quantization import (
        NormalizationAwarePerceptronQuantization)

    params = inspect.signature(NormalizationAwarePerceptronQuantization.__init__).parameters
    kwargs = {}
    for key in ("fractional_scale", "float_bias"):
        if key in params:
            kwargs[key] = bool(settings[key])
        elif settings[key]:
            raise SystemExit(f"REFUSED: {key} is declared, but this head's projection takes no {key}")
    return NormalizationAwarePerceptronQuantization(int(settings["bits"]), "cpu", rate=1.0,
                                                    two_scale=bool(settings["two_scale"]), **kwargs)


def _drive(perceptron) -> float:
    ""
    return float(perceptron.__dict__.get("bias_row_drive", 1.0))


def stamp_replay(model, flat: dict) -> dict:
    ""
    try:
        from mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route import stamp_projection_drives
    except ImportError:
        return {"source": "none at this head (no always-on row drive)", "route": False, "drives": {},
                "undetermined": []}
    stamps = stamp_projection_drives(model, dict(flat))
    return {"source": "mimarsinan.pipelining.pipeline_steps.quantization.bias_row_drive_route.stamp_projection_drives",
            "route": stamps is not None, "drives": dict(stamps.drives) if stamps is not None else {},
            "undetermined": list(stamps.undetermined) if stamps is not None else []}


def compare_projection(pre, post, settings: dict, *, flat: dict | None = None) -> dict:
    ""
    import torch
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    tr = PerceptronTransformer()
    with torch.no_grad():
        proj = copy.deepcopy(pre)
        replay = stamp_replay(proj, flat) if flat is not None else None
        napq = make_projection(settings)
        for p in proj.get_perceptrons():
            napq.transform(p)
        rows = []
        for p, q in _pairs(proj, post):
            scales = [[float(q.parameter_scale), float(p.parameter_scale)], [float(q.bias_scale), float(p.bias_scale)]]
            rows.append({"perceptron": str(q.name),
                         "weight": delta(tr.get_effective_weight(p), tr.get_effective_weight(q)),
                         "bias": delta(tr.get_effective_bias(p), tr.get_effective_bias(q)),
                         "scales_equal": scales[0][0] == scales[0][1] and scales[1][0] == scales[1][1],
                         "drive": [_drive(q), _drive(p)], "drive_equal": _drive(q) == _drive(p)})
    out = _summary(rows)
    out["drives_differ"] = [r["perceptron"] for r in rows if not r["drive_equal"]][:50]
    if replay is not None:
        out["replay_drives"] = replay
    return out


def check_work(work: Path, weight_source: Path, weight_source_sha256: str, *, flat: dict) -> dict:
    ""
    import torch
    from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer

    work = Path(work)
    missing = [s for s in STEP_MODELS if not (work / f"{s}.model.pt").exists()]
    if missing:
        return {"applicable": False, "reason": f"no cached step model for {missing}", "passed": None}
    if c.sha256_file(weight_source) != weight_source_sha256:
        raise SystemExit(f"REFUSED: {weight_source} differs from the session's weight_source digest")
    settings = projection_settings(flat)
    rec = {"applicable": True, "projection_settings": settings, "bits": settings["bits"],
           "two_scale": settings["two_scale"]}
    declared = torch.load(weight_source, map_location="cpu", weights_only=True)
    loaded = _load(work, "Weight Preloading").state_dict()
    per_key = {k: delta(declared[k], loaded[k]) for k in sorted(declared) if k in loaded}
    rec["a_preload_equals_weight_source"] = {
        "keys": len(declared), "keys_equal": sorted(declared) == sorted(loaded),
        "tensors_differ": sorted(k for k, v in per_key.items() if not _equal(v))[:50],
        "passed": sorted(declared) == sorted(loaded) and all(_equal(v) for v in per_key.values())}
    del declared, loaded
    tm, pre, post = (_load(work, s) for s in STEP_MODELS[1:])
    tr = PerceptronTransformer()
    with torch.no_grad():
        rows_b = [{"perceptron": str(q.name), "weight": delta(tr.get_effective_weight(p), tr.get_effective_weight(q)),
                   "bias": delta(tr.get_effective_bias(p), tr.get_effective_bias(q))} for p, q in _pairs(tm, pre)]
    rec["b_torch_mapping_to_readout_law_unchanged"] = _summary(rows_b)
    rec["c_weight_quantization_equals_rate1_projection"] = compare_projection(pre, post, settings, flat=flat)
    ledger = work / "Weight Quantization.adaptation_ledger.json"
    rec["weight_quantization_ledger_totals"] = json.loads(ledger.read_text()).get("totals") if ledger.exists() else None
    rec["passed"] = bool(rec["a_preload_equals_weight_source"]["passed"]
                         and rec["b_torch_mapping_to_readout_law_unchanged"]["passed"]
                         and rec["c_weight_quantization_equals_rate1_projection"]["passed"])
    return rec
