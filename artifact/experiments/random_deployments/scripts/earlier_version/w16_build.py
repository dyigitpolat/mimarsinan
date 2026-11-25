""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

X = sys.argv[1]
ROOT = "/work/research/"
RUNS = os.path.join(X, "runs16", "main")

RUNS2 = os.path.join(X, "runs16", "arms2")
RA = "records/research_artifacts"
XR = f"{RA}/11_wave2/exact"
NUM = "records/meeting_20261021/draft/shared/numbers.tex"
W0 = f"{RA}/09_wave0/wave0_report.md"
SIG = f"{RA}/08_significance"
G5 = f"{RA}/10_wave1/g5_residency_planner/u1_design.md"
MRG = f"{RA}/10_wave1/merge/targeted/log.md"
FIX = f"{RA}/11_wave2/closure/fix_report.md"
TGT = f"{RA}/11_wave2/closure/targeted_report.md"
SCREEN = f"{RA}/11_wave2/closure/fix_round/gates/screen/report.json"
HYP = f"{RA}/11_wave2/closure/fix_round/thm/hypothesis_table.md"
EXG = "thesis/figures/data/ex_g_defect_matrix.json"
EXG_CAT = "mimarsinan_research/experiments/thesis_campaigns/ex_g_defects.py"
CMAP = f"{XR}/w15_condition_map.md"
PC = "src/mimarsinan/certification/program_check"


def rows_of(pattern, base=None):
    out = []
    for path in sorted(glob.glob(os.path.join(base or RUNS, pattern))):
        for line in open(path):
            row = json.loads(line)
            if "mode" in row and "argv" in row:
                continue
            out.append(row)
    return out


def rel(path):
    return os.path.relpath(path, ROOT)


def failed(brief):
    return isinstance(brief, dict) and str(brief.get("t1", "")).startswith("failed")


def hist(briefs):
    h = collections.Counter()
    for b in briefs:
        for c in (b or {}).get("fail", {}) or {}:
            h[c] += 1
    return dict(sorted(h.items()))


def arm_summary(path_glob, label, base=None):
    base = base or RUNS2
    rows = rows_of(path_glob, base)
    s = collections.Counter()
    caught, missed, flagged_equal, ctrl_fail, gates_dec, gates_out = [], [], [], [], 0, 0
    agree = 0
    in_out_differ = 0
    samples = 0
    for r in rows:
        run = r.get("arm_run") or {}
        out = run.get("outcome", "harness_error" if "harness_error" in r else "missing")
        s[out] += 1
        agree += out == r.get("closure_outcome")
        if out not in ("counterexample", "equal", "vacuous"):
            continue
        s["admitted"] += 1
        ctrl = r.get("control_run") or {}
        if failed(ctrl.get("check_in")):
            ctrl_fail.append(r["seed"])
        if str((run.get("check_in") or {}).get("t1", "")).startswith("outside"):
            s["outside_scope"] += 1
        if (run.get("check_in") or {}).get("t1") != (run.get("check_out") or {}).get("t1"):
            in_out_differ += 1
        if out == "counterexample":
            teeth = ctrl.get("outcome") == "equal"
            s["teeth"] += teeth
            g = run.get("gate") or {}
            samples += int(g.get("samples", 0))
            gates_dec += int(g.get("decisions", 0)) > 0
            gates_out += int(g.get("outputs", 0)) > 0
            (caught if failed(run.get("check_in")) else missed).append(r["seed"])
        elif failed(run.get("check_in")):
            flagged_equal.append(r["seed"])
    cx = [r for r in rows if (r.get("arm_run") or {}).get("outcome") == "counterexample"]
    return {
        "label": label, "instances": len(rows), "outcomes": dict(s), "closure_outcome_agreement": agree,
        "counterexamples": len(cx), "static_failed_on_counterexamples": len(caught),
        "static_established_on_counterexamples": len(missed), "missed_seeds": missed[:20],
        "static_conditions_on_counterexamples": hist([(r["arm_run"] or {}).get("check_in") for r in cx]),
        "static_failed_on_equal": len(flagged_equal),
        "static_conditions_on_equal": hist([(r["arm_run"] or {}).get("check_in") for r in rows
                                            if (r.get("arm_run") or {}).get("outcome") == "equal"]),
        "controls_failed": len(ctrl_fail), "control_failed_seeds": ctrl_fail[:20],
        "controls_checked": sum(1 for r in rows if (r.get("control_run") or {}).get("check_in")),
        "check_inside_vs_outside_engagement_differ": in_out_differ,
        "gate_counterexamples_with_a_decision_change": gates_dec,
        "gate_counterexamples_with_an_output_change": gates_out,
        "gate_samples_on_counterexamples": samples,
        "same_ir_counterexamples_departing_from_their_own_identity": sum(
            1 for r in cx if int(((r["arm_run"] or {}).get("same_ir") or {}).get("deployment_vs_identity", 0)) > 0),
        "same_ir_counterexamples_whose_identity_departs_from_the_reference": sum(
            1 for r in cx if int(((r["arm_run"] or {}).get("same_ir") or {}).get("identity_vs_reference", 0)) > 0),
        "same_ir_errors": sum(1 for r in cx if "error" in ((r["arm_run"] or {}).get("same_ir") or {})),
        "same_ir_counterexamples_caught_among_program_departures": sum(
            1 for r in cx if int(((r["arm_run"] or {}).get("same_ir") or {}).get("deployment_vs_identity", 0)) > 0
            and failed((r["arm_run"] or {}).get("check_in"))),
        "logs": sorted(rel(p) for p in glob.glob(os.path.join(base, path_glob))),
    }


def fuzz_summary(path_glob, label):
    rows = [r for r in rows_of(path_glob) if "seed" in r and "count" in r]
    multi = [r for r in rows if r.get("multipass")]
    sems = sorted({k for r in multi for k, v in r.items() if isinstance(v, dict) and "integer" in v})
    out = {"label": label, "cells": len(rows), "built_integer": sum(1 for r in rows if "passes" in r),
           "multipass": len(multi), "pinned_programs": sum(1 for r in multi if r.get("pinned")),
           "delayed_programs": sum(1 for r in multi if r.get("delayed")), "semantics": {},
           "logs": sorted(rel(p) for p in glob.glob(os.path.join(RUNS, path_glob)))}
    for sem in sems:
        cells = [r[sem] for r in multi if sem in r]
        f_diff = sum(1 for c in cells if (c.get("float") or {}).get("sampled") == "DIFFERS")
        i_diff = [c for c in cells if (c.get("integer") or {}).get("sampled") == "DIFFERS"]
        i_eq = [c for c in cells if (c.get("integer") or {}).get("sampled") == "equal"]
        errs = sum(1 for c in cells if str((c.get("integer") or {}).get("sampled", "")).startswith("run_error"))
        f_errs = sum(1 for c in cells if str((c.get("float") or {}).get("sampled", "")).startswith("run_error"))
        out["semantics"][sem] = {
            "programs": len(cells), "float_ir_differs": f_diff, "float_ir_run_errors": f_errs,
            "integer_ir_differs": len(i_diff), "integer_ir_equal": len(i_eq), "integer_ir_run_errors": errs,
            "static_failed": sum(1 for c in cells if failed(c.get("static"))),
            "static_failed_on_differing": sum(1 for c in i_diff if failed(c.get("static"))),
            "static_established_on_differing": sum(1 for c in i_diff if (c.get("static") or {}).get("t1")
                                                   == "established"),
            "static_failed_on_equal": sum(1 for c in i_eq if failed(c.get("static"))),
            "static_conditions": hist([c.get("static") for c in cells]),
            "static_conditions_on_differing": hist([c.get("static") for c in i_diff]),
            "static_errors": sum(1 for c in cells if "error" in (c.get("static") or {})),
            "gate_differing_with_a_decision_change_integer": sum(
                1 for c in i_diff if int(((c.get("integer") or {}).get("gate") or {}).get("decisions", 0)) > 0),
            "gate_differing_with_a_decision_change_float": sum(
                1 for c in cells if (c.get("float") or {}).get("sampled") == "DIFFERS"
                and int(((c.get("float") or {}).get("gate") or {}).get("decisions", 0)) > 0),
        }
    return out


def u1_summary(path_glob, label):
    rows = [r for r in rows_of(path_glob) if "seed" in r and "count" in r]
    built = [r for r in rows if r.get("u1") == "built"]
    multi = [r for r in built if r.get("multipass")]
    sems = ("streamed_lif", "windowed_lif", "cascaded_ttfs", "analytical_ttfs", "synchronized_lif")
    return {"label": label, "cells": len(rows), "built": len(built), "multipass": len(multi),
            "refused": sum(1 for r in rows if r.get("u1") == "refused"),
            "semantics": {s: dict(collections.Counter(str(r.get(s)).split(" ")[0] for r in multi)) for s in sems},
            "logs": sorted(rel(p) for p in glob.glob(os.path.join(RUNS, path_glob)))}


def exg_summary():
    base = {r["program"]: r for r in rows_of("exg/worktree.jsonl") if "program" in r}
    out = {"programs": len(base), "base_errors": sorted(k for k, r in base.items() if "error" in r),
           "base_verdicts": dict(collections.Counter(v["t1"] for r in base.values()
                                                     for v in (r.get("reports") or {}).values())),
           "resident_passes": {k: r.get("resident_passes") for k, r in base.items() if "resident_passes" in r},
           "base_value_runs": {k: r.get("value_run") for k, r in base.items() if "value_run" in r},
           "trees": {}, "log_worktree": rel(os.path.join(RUNS, "exg/worktree.jsonl"))}
    for path in sorted(glob.glob(os.path.join(RUNS, "exg/exg_*.jsonl"))):
        name = os.path.basename(path)[4:-6]
        mine = {r["program"]: r for r in rows_of(f"exg/{os.path.basename(path)}") if "program" in r}
        changed = sorted(k for k in base if json.dumps(base[k].get("reports"), sort_keys=True)
                         != json.dumps((mine.get(k) or {}).get("reports"), sort_keys=True))
        value_changed = sorted(k for k in base if "value_run" in base[k]
                               and (base[k]["value_run"] or {}).get("equal") != ((mine.get(k) or {}).get("value_run")
                                                                                 or {}).get("equal"))
        out["trees"][name] = {"programs": len(mine), "static_reports_changed": changed,
                              "errors": sorted(k for k, r in mine.items() if "error" in r),
                              "value_run_changed": value_changed,
                              "value_runs": {k: r.get("value_run") for k, r in mine.items() if "value_run" in r},
                              "log": rel(path)}
    return out


def reseed_summary(name):
    rows = rows_of(f"reseed/{name}.jsonl", RUNS2)
    s = arm_summary(f"reseed/{name}.jsonl", name)
    s["control_counterexamples"] = sum(1 for r in rows if (r.get("control_run") or {}).get("outcome")
                                       == "counterexample")
    return s


def trace_sites():
    rows = rows_of("trace.jsonl")
    executed = {}
    for r in rows:
        if "executed" in r:
            executed = {"src/" + k: set(v) for k, v in r["executed"].items()}
    return executed


SITES = {
    "D1": [("src/mimarsinan/torch_mapping/converter_handlers/batchnorm_lowering.py", 1, 400),
           ("src/mimarsinan/torch_mapping/representability_analyzer.py", 239, 280)],
    "D2": [("src/mimarsinan/torch_mapping/converter_handlers/conv_mixin.py", 1, 400)],
    "D8": [("src/mimarsinan/torch_mapping/representability_analyzer.py", 239, 280)],
    "D9": [("src/mimarsinan/mapping/mappers/conv_helpers.py", 80, 200)],
    "D10": [("src/mimarsinan/torch_mapping/converter_handlers/batchnorm_lowering.py", 109, 140),
            ("src/mimarsinan/torch_mapping/converter_handlers/conv_mixin.py", 1, 400)],
    "D11": [("src/mimarsinan/mapping/support/call_template.py", 1, 400),
            ("src/mimarsinan/mapping/support/compute_modules.py", 112, 300)],
    "exg": None,
    "F3.delay": [("src/mimarsinan/models/spiking/hybrid/carry.py", 67, 90),
                 ("src/mimarsinan/models/spiking/hybrid/stage_io.py", 1, 80)],
    "F3.walk": [("src/mimarsinan/models/spiking/hybrid/executors/sync_counts.py", 24, 90)],
    "G5.store": [("src/mimarsinan/models/spiking/hybrid/carry.py", 45, 50),
                 ("src/mimarsinan/models/spiking/hybrid/rate_forward.py", 1, 400)],
    "G5.pins": [("src/mimarsinan/mapping/packing/pass_timing.py", 121, 128)],
    "G5.delays": [("src/mimarsinan/mapping/packing/pass_timing.py", 130, 166),
                  ("src/mimarsinan/models/spiking/hybrid/carry.py", 52, 90)],
    "G5.walk": [("src/mimarsinan/chip_simulation/value_run/value_execution.py", 25, 60),
                ("src/mimarsinan/chip_simulation/ttfs/ttfs_segment.py", 132, 150)],
    "G5.positions": [("src/mimarsinan/mapping/packing/pass_positions.py", 1, 200)],
    "G5.levels": [("src/mimarsinan/mapping/packing/retimed_levels.py", 1, 300)],
    "C.count": [("src/mimarsinan/models/spiking/wire_semantics.py", 147, 190),
                ("src/mimarsinan/models/spiking/hybrid/executors/sync_counts.py", 60, 70)],
    "C.staircase": [("src/mimarsinan/models/spiking/wire_semantics.py", 37, 80)],
    "C.levels": [("src/mimarsinan/mapping/layout/segmentation.py", 66, 130)],
    "C.lattice": [("src/mimarsinan/chip_simulation/ttfs/membrane_lattice.py", 1, 200),
                  ("src/mimarsinan/chip_simulation/ttfs/ttfs_segment.py", 180, 215)],
    "C.float": [("src/mimarsinan/chip_simulation/ttfs/float_membrane.py", 1, 200)],
}


def site_cover(executed, sites):
    out = []
    for path, lo, hi in sites or ():
        lines = executed.get(path, set())
        out.append({"site": f"{path}:{lo}-{hi}", "file_lines_executed": len(lines),
                    "range_lines_executed": sum(1 for n in lines if lo <= n <= hi)})
    return out


def main() -> int:
    summary = {"arms": {}, "fuzz": {}, "prefix": {}, "u1": {}, "reseed": {}}
    for arm in ("H0", "H1-theta", "H5-collapse", "H5-skew", "H8", "R4", "R6"):
        summary["arms"][arm] = arm_summary(f"arms/{arm}.jsonl", arm)
    for sw in ("pins_delays_off", "pins_off", "delays_off", "store_unscoped", "walks_latency"):
        summary["fuzz"][sw] = fuzz_summary(f"fuzz/{sw}_*.jsonl", sw)
    for t in ("prefix", "prefix_delay", "prefix_walk"):
        summary["prefix"][t] = fuzz_summary(f"prefix/{t}_[0-9]*.jsonl", t)
        summary["u1"][t] = u1_summary(f"u1/{t}_[0-9]*.jsonl", t)
    summary["u1"]["head"] = u1_summary("u1/head_[0-9]*.jsonl", "head")
    for name in ("count_rebuild_on_long_window", "retiming_dropped_on", "retiming_dropped_on_sign_split",
                 "retiming_dropped_on_sign_split_scheduled"):
        summary["reseed"][name] = reseed_summary(name)
    summary["exg"] = exg_summary()
    summary["residency"] = {}
    for path in sorted(glob.glob(os.path.join(X, "runs16", "residency", "*.jsonl"))):
        summary["residency"][os.path.basename(path)[:-6]] = {
            "rows": [r for r in (json.loads(line) for line in open(path)) if "vehicle" in r], "log": rel(path)}
    conv = [r for r in rows_of("conv1d.jsonl") if "vehicle" in r]
    summary["conv1d"] = conv[0] if conv else {}
    executed = trace_sites()
    summary["trace"] = {"files_executed": len(executed), "lines_executed": sum(len(v) for v in executed.values()),
                        "log": rel(os.path.join(RUNS, "trace.jsonl"))}
    exg_files = {}
    for path in sorted(glob.glob(os.path.join(X, "runs16", "tree_diffs", "*.diff"))):
        name = os.path.basename(path)[:-5]
        files = [line.split()[1].split("/", 1)[1] for line in open(path) if line.startswith("+++ ")]
        exg_files[name] = files
    summary["trace"]["sites"] = {k: site_cover(executed, v) for k, v in SITES.items() if v}
    summary["trace"]["tree_files"] = {k: {f: len(executed.get(f, set())) for f in files}
                                      for k, files in exg_files.items()}
    summary["trace"]["program_check_files"] = {k: len(v) for k, v in executed.items() if k.startswith(PC)}
    with open(os.path.join(X, "runs16", "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in summary.items()})[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
