""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPIKING_RUNS = ("streamed_lif", "windowed_lif", "synchronized_lif", "cascaded_ttfs", "synchronized_ttfs",
                "analytical_ttfs")

OUTSIDE_THEOREM = ("analytical_ttfs",)


def clopper_pearson_upper(n: int, alpha: float = 0.05) -> float:
    ""
    return 1.0 - alpha ** (1.0 / n) if n else 1.0


def _class(value) -> str:
    v = str(value or "")
    if v in ("equal", "DIFFERS"):
        return v
    return v.split(" ")[0] if v else "absent"


def comparison(rows: list, key: str) -> dict:
    counts = collections.Counter(_class(r.get(key)) for r in rows)
    n = counts.get("equal", 0) + counts.get("DIFFERS", 0)
    differ = counts.get("DIFFERS", 0)
    out = {"N": n, "differ": differ, "classes": dict(counts)}
    if n and not differ:
        out.update(rule_of_three=round(3.0 / n, 6), clopper_pearson_95=round(clopper_pearson_upper(n), 6))
    return out


def ranges(rows: list) -> dict:
    out = {}
    for field in ("perceptrons", "host_ops", "ir_cores", "unbounded_ir_cores", "max_fan_in", "max_width", "layers"):
        vals = [r["vehicle"][field] for r in rows if isinstance(r.get("vehicle"), dict) and field in r["vehicle"]]
        if vals:
            out[field] = [min(vals), max(vals)]
    passes = [len(r["passes"]) for r in rows if r.get("passes")]
    if passes:
        out["passes"] = [min(passes), max(passes)]
    for field in ("count", "axons", "neurons", "window"):
        vals = [r[field] for r in rows if field in r]
        if vals:
            out[field] = sorted(set(vals)) if len(set(vals)) <= 8 else [min(vals), max(vals)]
    kinds = collections.Counter(r["vehicle"].get("kind") for r in rows if isinstance(r.get("vehicle"), dict)
                                and r["vehicle"].get("kind"))
    if kinds:
        out["kinds"] = dict(kinds)
    return out


def summarize(name: str, rows: list, script: str) -> dict:
    vehicle_errors = [r for r in rows if "vehicle_error" in r]
    rows = [r for r in rows if "vehicle_error" not in r]
    build_key = "build" if script == "value" else "u1"
    builds = collections.Counter(r.get(build_key) for r in rows)
    refusals = collections.Counter(r.get("refusal_class", "?") for r in rows if r.get(build_key) == "refused")
    built = [r for r in rows if r.get(build_key) == "built"]
    multi = [r for r in built if r.get("multipass")]
    out = {"job": name, "rows": len(rows) + len(vehicle_errors), "vehicle_errors": len(vehicle_errors),
           "seeds": len({r["seed"] for r in rows}), "builds": dict(builds), "refusal_classes": dict(refusals),
           "build_errors": [r.get("reason", "")[:160] for r in rows if r.get(build_key) == "error"][:10],
           "built": len(built), "multipass": len(multi), "ranges": ranges(rows), "runs": {}}
    if script == "value":
        out["runs"]["value"] = {"against_identity": comparison(multi, "value"),
                                "against_int64_golden": comparison(multi, "value_golden"),
                                "single_pass_against_identity": comparison([r for r in built if not r.get("multipass")],
                                                                           "value")}
        out["identity_against_golden"] = dict(collections.Counter(bool(r.get("golden_identity")) for r in rows))
    else:
        for run in SPIKING_RUNS:
            if not any(run in r for r in multi):
                continue
            out["runs"][run] = {"against_tiled_identity": comparison(multi, run),
                                "against_unbounded_identity": comparison(multi, run + "_unbounded"),
                                "nondiscriminating": sum(1 for r in multi if r.get(run) == "equal"
                                                         and not r.get(run + "_disc")),
                                "outside_theorem_by_name": run in OUTSIDE_THEOREM}
    differs = [r for r in multi if any(_class(r.get(k)) == "DIFFERS" for k in r
                                       if k in SPIKING_RUNS or k.endswith("_unbounded") or k in ("value", "value_golden"))]
    out["admitted_and_inexact"] = len(differs)
    out["admitted_and_inexact_seeds"] = [(r["seed"], r["count"]) for r in differs[:20]]
    out["theorem_programs"] = theorem_programs(multi, script, [k for k in out["runs"] if k not in OUTSIDE_THEOREM])
    return out


def theorem_programs(multi: list, script: str, runs: list) -> dict:
    ""
    keys = ("value", "value_golden") if script == "value" else tuple(run + "_unbounded" for run in runs)
    n = d = 0
    programs = set()
    for r in multi:
        outcomes = [_class(r.get(k)) for k in keys]
        if not any(o in ("equal", "DIFFERS") for o in outcomes):
            continue
        programs.add((r["seed"], r["count"]))
        n += 1
        d += int("DIFFERS" in outcomes)
    out = {"N": n, "differ": d, "distinct_programs": len(programs), "runs": list(runs) if script != "value" else ["value"],
           "unit": "multi-pass programs, each once; a trial differs if any Theorem 1 comparison on it differs"}
    if n and not d:
        out.update(rule_of_three=round(3.0 / n, 6), clopper_pearson_95=round(clopper_pearson_upper(n), 6))
    return out


def load(directory: Path) -> list:
    return [json.loads(line) for f in sorted(directory.glob("chunk_*.jsonl")) for line in f.read_text().splitlines()
            if line.strip()]


def markdown(summary: dict) -> str:
    lines = [f"# W13 Summary ({summary['run_dir']})", "", f"Manifest sha256 `{summary['manifest_sha256']}`; "
             f"fraction {summary['fraction']}; code state {summary['code_state']}.", "",
             "| job | seeds | rows | built | multi-pass | run | against | N | differ | 3/N |", "|---|---|---|---|---|---|---|---|---|---|"]
    for job in summary["jobs"]:
        for run, comps in job["runs"].items():
            for against, c in comps.items():
                if not isinstance(c, dict):
                    continue
                lines.append(f"| {job['job']} | {job['seeds']} | {job['rows']} | {job['built']} | {job['multipass']} | "
                             f"{run}{' (outside the theorem by name)' if comps.get('outside_theorem_by_name') else ''} | "
                             f"{against} | {c['N']} | {c['differ']} | {c.get('rule_of_three', '-')} |")
    lines += ["", "Theorem programs: each multi-pass program of a job counted once (a trial differs if any Theorem 1 run "
              "on it differs); the runs of one job run the same programs, so their comparisons are never summed into "
              "one 3/N, and a job that reruns another job's programs (another encode or window) is its own reading.", "",
              "| job | refusal classes | build errors | admitted and inexact | theorem programs N, differ, 3/N | measured ranges |",
              "|---|---|---|---|---|---|"]
    for job in summary["jobs"]:
        t = job["theorem_programs"]
        lines.append(f"| {job['job']} | {job['refusal_classes']} | {len(job['build_errors'])} | {job['admitted_and_inexact']} | "
                     f"{t['N']}, {t['differ']}, {t.get('rule_of_three', '-')} | {json.dumps(job['ranges'])} |")
    lines += ["", "Declared generator ranges:", ""] + [f"- {k}: {v}" for k, v in summary["generators"].items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--manifest", default=str(HERE / "w13_manifest.json"))
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    manifest = json.loads(Path(a.manifest).read_text())
    record = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    jobs, by_name = [], {}
    for job in manifest["jobs"]:
        if not (run_dir / job["name"]).is_dir():
            continue
        rows = load(run_dir / job["name"])
        by_name[job["name"]] = (rows, job["script"])
        jobs.append(summarize(job["name"], rows, job["script"]))
    for view in manifest.get("views", []):
        if view["of"] in by_name:
            rows, script = by_name[view["of"]]
            s0, s1 = view["seeds"]
            kept = [r for r in rows if s0 <= r["seed"] < s1 and (r.get("count") in view["counts"] or "vehicle_error" in r)]
            jobs.append(summarize(f"{view['name']} (view of {view['of']})", kept, script))
    summary = {"run_dir": str(run_dir), "manifest_sha256": record.get("manifest_sha256"), "fraction": record.get("fraction"),
               "code_state": record.get("code_state_start"), "generators": manifest["generators"], "jobs": jobs,
               "chunks_failed": [c for c in record.get("chunks", []) if c.get("exit") != 0]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    (run_dir / "summary.md").write_text(markdown(summary))
    print(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
