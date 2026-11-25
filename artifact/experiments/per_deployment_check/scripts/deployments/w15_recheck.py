""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from mimarsinan.certification import program_check
from mimarsinan.mapping.ir.core_law import graph_carries_laws

work, label, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])


def resolve(cfg: dict) -> dict:
    from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline, merge_pipeline_config
    from mimarsinan.pipelining.session import _resolve_platform_constraints
    dp = dict(cfg["deployment_parameters"])
    dp.setdefault("seed", int(cfg.get("seed", 0)))
    DeploymentPipeline.apply_preset(cfg.get("pipeline_mode", "phased"), dp)
    pc = _resolve_platform_constraints(cfg, dp)
    config = merge_pipeline_config(dp, pc if isinstance(pc, dict) else {})
    config.update(input_shape=(3, 32, 32), num_classes=100, input_size=3 * 32 * 32, device="cpu")
    return config


def load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def fields_of(d: dict) -> dict:
    conds = d.get("conditions") or {}
    tiling = conds.get("ir-tiling") or {}
    return {"semantics": d.get("semantics"), "theorem1": d.get("theorem1"), "proposition2": d.get("proposition2"),
            "failures": d.get("failures_by_condition"), "unknown_forms": d.get("unknown_forms"),
            "unread_forms": d.get("unread_forms"), "reference": d.get("reference"), "in_scope": d.get("in_scope"),
            "statuses": {c: v.get("status") for c, v in sorted(conds.items())},
            "checked": {c: v.get("checked") for c, v in sorted(conds.items())},
            "ir_tiling": tiling.get("status"), "ir_tiling_violations": tiling.get("violation_count")}


row = {"label": label, "run": str(work)}
pipeline = json.loads((work / "static_program_check.json").read_text())
row["pipeline"] = fields_of(pipeline)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        config = resolve(json.loads((work / "_RUN_CONFIG" / "config.json").read_text()))
    pcfg = json.loads((work / "Model Configuration.platform_constraints_resolved.json").read_text())
    t0 = time.perf_counter()
    ir = load(work / "Soft Core Mapping.ir_graph.pickle")
    unb_path = work / "Soft Core Mapping.ir_graph_unbounded.pickle"
    entry = load(unb_path) if unb_path.exists() else None
    unbounded = entry.get("ir") if isinstance(entry, dict) else entry
    row["unbounded_note"] = entry.get("note") if isinstance(entry, dict) else None
    program = load(work / "hybrid_mapping.pickle")
    row["load_s"] = round(time.perf_counter() - t0, 1)
    row["unbounded_ir"] = "given" if unbounded is not None else "absent"
    decl = program_check.declaration_from_config(config, cores=pcfg["cores"], has_laws=graph_carries_laws(ir),
                                                 backends={})
    t1 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        report = program_check.check_deployment(program, ir_graph=ir, declaration=decl, compile_source="loaded",
                                                unbounded_ir=unbounded)
    row["check_s"] = round(time.perf_counter() - t1, 1)
    rerun = json.loads(json.dumps(report.to_dict(), default=str))
    row["rerun"] = fields_of(rerun)
    row["passes"] = sum(1 for s in program.stages if s.kind == "neural")
    differ = sorted(k for k in row["pipeline"] if row["pipeline"][k] != row["rerun"][k])
    row["fields_differing"] = differ
    row["rows_differing_in_total"] = len(differ)
except Exception as exc:
    import traceback
    row["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    row["trace"] = traceback.format_exc()[-2500:]
p = row["pipeline"]
row["established"] = bool(str(p["theorem1"]).startswith("established") and str(p["proposition2"]).startswith("established")
                          and not p["unknown_forms"] and not p["unread_forms"] and not p["failures"])
out.write_text(json.dumps(row, indent=1, default=str))
print(json.dumps({k: row.get(k) for k in ("label", "established", "rows_differing_in_total", "fields_differing", "error",
                                         "passes", "check_s")}, default=str))
