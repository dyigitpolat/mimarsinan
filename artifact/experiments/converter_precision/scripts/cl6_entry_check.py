""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cl6_common as K

HOOKED = {"Model Configuration.platform_constraints_resolved"}


REBUILT = {"Model Configuration.model_config", "Model Configuration.model_builder"}


def static_table(member: str, snap: Path) -> dict:
    from mimarsinan.pipelining.session import PipelineSession
    meta = json.loads((snap / "snapshot.json").read_text())
    cfg = dict(meta["config"])
    cfg["_working_directory"] = str(snap / "static_probe_run")
    session = PipelineSession.from_config(cfg)
    pipeline = session.pipeline
    pipeline.set_up_requirements()
    names = [n for n, _ in pipeline.steps]
    start = K.start_step(member)
    i0 = names.index(start)
    produced = {}
    for name, step in pipeline.steps[:i0]:
        for key in list(step.promises) + list(step.updates):
            produced[f"{name}.{key}"] = name
    consumers: dict = {}
    for name, step in pipeline.steps[i0:]:
        for req in step.requires:
            real = pipeline.key_translations[name][req]
            if real in produced:
                consumers.setdefault(real, []).append(name)
    rows = [{"entry": e, "producer": produced[e], "consumers_at_or_after_start": consumers.get(e, [])}
            for e in sorted(produced)]
    steps = [{"step": n, "requires": list(s.requires), "promises": list(s.promises), "updates": list(s.updates),
              "clears": list(s.clears), "before_start": i < i0} for i, (n, s) in enumerate(pipeline.steps)]
    session.finish()
    import shutil
    shutil.rmtree(snap / "static_probe_run", ignore_errors=True)
    return {"start_step": start, "steps": steps, "entries": rows}


def _equal(a, b, path="") -> list:
    ""
    import numpy as np
    import torch
    if isinstance(a, torch.nn.Module) and isinstance(b, torch.nn.Module):
        out = []
        if type(a) is not type(b):
            out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
        if repr(a) != repr(b):
            out.append(f"{path}: module tree differs")
        sa, sb = a.state_dict(), b.state_dict()
        if list(sa) != list(sb):
            out.append(f"{path}: state_dict keys differ")
        for k in sa:
            if k in sb and not (sa[k].shape == sb[k].shape and torch.equal(sa[k], sb[k])):
                out.append(f"{path}: {k} differs")
        return out
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return [] if (a.shape == b.shape and torch.equal(a, b)) else [f"{path}: tensor differs"]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return [] if np.array_equal(a, b) else [f"{path}: array differs"]
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present in one only")
            else:
                out.extend(_equal(a[k], b[k], f"{path}.{k}"))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_equal(x, y, f"{path}[{i}]"))
        return out
    if hasattr(a, "__dict__") and hasattr(b, "__dict__") and type(a) is type(b) and not isinstance(a, type):
        return _equal(vars(a), vars(b), f"{path}<{type(a).__name__}>")
    try:
        same = a == b
        if isinstance(same, bool):
            return [] if same else [f"{path}: {str(a)[:80]!r} vs {str(b)[:80]!r}"]
    except Exception:
        pass
    try:
        return [] if pickle.dumps(a) == pickle.dumps(b) else [f"{path}: pickles differ"]
    except Exception as e:
        return [f"{path}: not comparable ({type(e).__name__})"]


def compare_snapshots(snap_a: Path, snap_b: Path) -> list:
    from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
    ca, cb = PipelineCache(), PipelineCache()
    ca.load(str(snap_a / "run"))
    cb.load(str(snap_b / "run"))
    fa = json.loads((snap_a / "snapshot.json").read_text())["files_sha256"]
    fb = json.loads((snap_b / "snapshot.json").read_text())["files_sha256"]
    meta = json.loads((snap_a / "run" / "metadata.json").read_text())
    rows = []
    for key in sorted(set(ca.keys()) | set(cb.keys())):
        if key not in ca.cache or key not in cb.cache:
            rows.append({"entry": key, "equal": False, "differences": ["present in one snapshot only"]})
            continue
        diffs = _equal(ca.get(key), cb.get(key), key)
        fname = None
        if key in meta:
            fname = next((f for f in fa if f.startswith(meta[key][1] + ".")), None)
        rows.append({"entry": key, "equal": not diffs, "differences": diffs[:12], "n_differences": len(diffs),
                     "file_bytes_equal": (fname is not None and fa.get(fname) == fb.get(fname)) if fname else None})
    return rows


def main() -> None:
    member, snap_a, snap_b, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    snap_a2 = Path(sys.argv[5]) if len(sys.argv) > 5 else None
    table = static_table(member, snap_a)
    rows = compare_snapshots(snap_a, snap_b)
    by_entry = {r["entry"]: r for r in rows}
    rr = {r["entry"]: r for r in compare_snapshots(snap_a, snap_a2)} if snap_a2 else {}
    verdict_rows = []
    failures = []
    for e in table["entries"]:
        cmp = by_entry.get(e["entry"])
        varies = e["entry"] in rr and not rr[e["entry"]]["equal"]
        dependent = cmp is not None and not cmp["equal"] and not varies
        handled = "re-resolved by the hook" if e["entry"] in HOOKED else (
            "re-run and asserted by the hook" if e["entry"] in REBUILT else "")
        verdict = ("varies run to run (dependence not measurable here)" if varies else
                   "independent" if not dependent else ("dependent, " + handled if handled else "dependent"))
        if dependent and not handled and e["consumers_at_or_after_start"]:
            failures.append(e["entry"])
        verdict_rows.append({**e, "verdict": verdict, "differences": cmp["differences"] if cmp else ["missing"],
                             "run_to_run_differences": rr[e["entry"]]["differences"] if varies else []})
    run_scoped = [r for r in rows if not any(r["entry"] == e["entry"] for e in table["entries"])]
    for r in run_scoped:
        r["varies_run_to_run"] = r["entry"] in rr and not rr[r["entry"]]["equal"]
        if not r["equal"] and not r["varies_run_to_run"]:
            failures.append(r["entry"])
    meta_a = json.loads((snap_a / "snapshot.json").read_text())
    meta_b = json.loads((snap_b / "snapshot.json").read_text())
    K.write_json(out, {
        "member": member, "start_step": table["start_step"], "steps": table["steps"], "entries": verdict_rows,
        "run_scoped_entries": run_scoped, "failures": failures, "pass": not failures,
        "snapshot_a": {k: meta_a[k] for k in ("arm", "adc_bits", "cores", "stopped_after", "src_digest")},
        "snapshot_b": {k: meta_b[k] for k in ("arm", "adc_bits", "cores", "stopped_after", "src_digest")},
        "snapshot_a2": str(snap_a2) if snap_a2 else None,
        "run_to_run_varying": sorted(k for k, r in rr.items() if not r["equal"]),
    })
    print(json.dumps({"member": member, "pass": not failures, "failures": failures,
                      "entries": len(verdict_rows), "run_scoped": len(run_scoped)}))


if __name__ == "__main__":
    main()
