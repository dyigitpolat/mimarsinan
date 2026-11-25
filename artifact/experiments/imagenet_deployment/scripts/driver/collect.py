""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import w38common as c
import configs
import lockcheck
import preflight
import run_row

FORMAT = "w38-group/2"
CHECK_FORMAT = "w38-structure-check/1"


def _row(records: Path, group: str, budget: str) -> dict:
    d = records / f"{group}.{budget}"
    if (d / "result.json").exists():
        r = json.loads((d / "result.json").read_text())
        if r.get("format") != "w38-row-result/1" or r.get("status") != "complete":
            raise SystemExit(f"REFUSED: {d.name}: format {r.get('format')} status {r.get('status')}")
        for name, digest in r["K9"]["files"].items():
            if c.sha256_file(d / name) != digest:
                raise SystemExit(f"REFUSED: {d / name} does not verify")
        return {"budget": budget, "state": "complete", "result": r, "dir": d, "mode": r.get("mode"),
                "toolchain": r["inputs"]["toolchain"]}
    if (d / "session.json").exists():
        s = json.loads((d / "session.json").read_text())
        return {"budget": budget, "state": "session " + str(s.get("status")), "error": s.get("error"),
                "refusal": s.get("refusal"), "steps_run": s.get("steps_run"), "wall_s": s.get("wall_s"), "peak_rss_kb": s.get("peak_rss_kb"),
                "dir": d, "mode": s.get("mode"), "toolchain": s.get("toolchain")}
    return {"budget": budget, "state": "absent", "dir": d}


def certificates(rr: dict) -> dict:
    ""
    k4 = rr["K4_K5_pipeline"].get("K4_value_parity_gate") or {}
    k5 = rr["K4_K5_pipeline"].get("K5_value_twin_certificate") or {}
    k6, k14 = rr.get("K6") or {}, rr.get("K14") or {}
    return {"K0": {"verdict": rr["K0"]["passed"], "denominator": f"{rr['K0']['keys']} tensors"},
            "K1": {"verdict": rr["K1"]["passed"], "denominator": f"{rr['K1']['images']} images"},
            "K4": {"verdict": k4.get("verdict"), "denominator": f"{k4.get('samples')} sampled images",
                   "kind": "sampled check"},
            "K5": {"verdict": k5.get("verdict"), "denominator": f"{k5.get('samples')} sampled images",
                   "kind": "sampled check"},
            "K6": {"verdict": k6.get("passed"), "denominator": f"{k6.get('images')} images, {k6.get('outputs')} outputs",
                   "kind": "sampled check"},
            "K11": {"verdict": rr["K11"]["passed"], "denominator": "record identity"},
            "K14": {"verdict": k14.get("passed"), "difference": k14.get("difference"),
                    "denominator": f"{k14.get('census_images')} images, near-tie bound {k14.get('near_tie_images')}"},
            "NU": {"verdict": rr["no_update"]["passed"], "denominator": "every perceptron, exact"},
            "input_path": {"verdict": (rr.get("input_path_check") or {}).get("passed"),
                           "denominator": f"{(rr.get('input_path_check') or {}).get('images')} images"}}


def structure_provenance(structure: dict, rows: list[dict]) -> dict:
    ""
    mine = structure.get("toolchain") or {}
    differ = []
    for r in rows:
        if r.get("mode") is not None and r["mode"] != structure.get("mode"):
            differ.append({"against": f"row {r['budget']}", "mode (structure, row)": [structure.get("mode"), r["mode"]]})
        if r.get("toolchain"):
            d = c.provenance_differ(mine, r["toolchain"])
            if d:
                differ.append({"against": f"row {r['budget']}", "toolchain (structure, row)": d})
    of_record = structure.get("mode") != "smoke"
    if of_record:
        try:
            run_row.head_of_record_check(mine, structure.get("group"))
        except SystemExit as exc:
            differ.append({"against": "head of record", "refusal": str(exc)[:1500]})
    return {"passed": not differ, "differ": differ, "keys": list(c.PROVENANCE_KEYS),
            "rows_compared": [r["budget"] for r in rows if r.get("toolchain")], "head_of_record_compared": of_record}


def structure_check(records: Path, group: str) -> dict:
    ""
    path = records / f"{group}.structure.json"
    out_path = records / f"{group}.structure_check.json"
    if not path.is_file():
        raise SystemExit(f"REFUSED: no structural pack {path}: the structure-only check reads a pack written by "
                         "structure.py; nothing was written")
    structure = json.loads(path.read_text())
    if out_path.exists() and structure.get("mode") != "smoke":
        raise SystemExit(f"REFUSED: {out_path} exists: a pack is checked once; nothing was written")
    status = structure.get("status")
    check = structure_provenance(structure, []) if status == "complete" else None
    compared = bool(check is not None and check["head_of_record_compared"])
    passed = bool(status == "complete" and check is not None and check["passed"] and compared)
    reason = (None if passed else f"the pack's status is {status!r}, not complete" if status != "complete" else
              f"the pack (mode {structure.get('mode')!r}) was not compared with the head of record" if not compared
              else "the pack's provenance differs from the head of record")
    member, bundle, arm = group.split(".")
    packs = [{"token": "chip", "file": path.name, "sha256": c.sha256_file(path), "status": status,
              "provenance": check, "passed": passed, "refused_by_name": None if status == "complete" else
              preflight.named_refusal(structure), "stop": preflight.stopped_pack(structure)}]
    stopped = []
    if packs[0]["stop"]:
        stopped.append((structure, path.name))
    for token in configs.structure_tokens(member, bundle, arm)[1:]:
        kp = records / f"{group}.structure.{token}.json"
        if not kp.is_file():
            packs.append({"token": token, "file": kp.name, "sha256": None, "status": "absent", "provenance": None,
                          "passed": False, "refused_by_name": None, "stop": None})
            continue
        krec = json.loads(kp.read_text())
        kcheck = structure_provenance(krec, []) if krec.get("status") == "complete" else None
        kpassed = bool(kcheck is not None and kcheck["passed"] and kcheck["head_of_record_compared"])
        packs.append({"token": token, "file": kp.name, "sha256": c.sha256_file(kp), "status": krec.get("status"),
                      "provenance": kcheck, "passed": kpassed, "refused_by_name": None if krec.get("status") ==
                      "complete" else preflight.named_refusal(krec), "stop": preflight.stopped_pack(krec)})
        if packs[-1]["stop"]:
            stopped.append((krec, kp.name))
    first = next((s for s in stopped if s[0].get("in_flight")), stopped[0] if stopped else None)
    stop_rec = None if first is None else {"exit": first[0].get("exit"), "stop": preflight.stopped_pack(first[0]),
                                           "token": first[0].get("token"), "file": first[1]}
    others = [k for k in packs[1:] if not (k["passed"] or k["refused_by_name"] or k["stop"])]
    if passed and others:
        passed = False
        reason = "; ".join(f"the {k['token']} pack ({k['file']}) is {k['status']}" + (
            ", its provenance differs from the head of record or was not compared with it"
            if k["status"] == "complete" else "") for k in others)
    rec = {"format": CHECK_FORMAT, "group": group, "structure_file": path.name,
           "structure_sha256": c.sha256_file(path), "structure_mode": structure.get("mode"),
           "structure_status": status, "provenance": check, "passed": passed, "reason": reason,
           "refused_by_name": None if status == "complete" else preflight.named_refusal(structure),
           "stopped": stop_rec, "packs": packs,
           "rule": "structure_provenance with no rows: the pack's mode and toolchain digests against the locked head "
                   "of record outside a smoke (protocol.md section 9.2); the rows-equal-pack check stays at the "
                   "group's end (collect)",
           "written": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    c.atomic_write_json(out_path, rec)
    return rec


def _scheduled_multiple(records: Path, group: str, rows: list[dict]) -> dict:
    ""
    member, bundle, arm = group.split(".")
    try:
        tokens = configs.structure_tokens(member, bundle, arm)[1:]
    except configs.RowRefusal as exc:
        return {"token": None, "state": f"structure tokens not read: {exc}"[:600]}
    if not tokens:
        return {"token": None, "state": "no K: passes and reloads from the chip pack (x.shape_only)"}
    token = tokens[0]
    path = records / f"{group}.structure.{token}.json"
    rec = json.loads(path.read_text()) if path.is_file() else None
    if rec is None or rec.get("status") != "complete":
        return {"token": token, "state": "absent" if rec is None else rec.get("status"),
                "error": None if rec is None else rec.get("error"),
                "refused_by_name": None if rec is None else preflight.named_refusal(rec),
                "stop": None if rec is None else preflight.stopped_pack(rec)}
    check = structure_provenance(rec, rows)
    if not check["passed"]:
        return {"token": token, "state": "withheld: the pack is not on the rows' head or the head of record",
                "provenance": check, "file": path.name, "sha256": c.sha256_file(path)}
    sched = rec.get("schedule") or {}
    return {"token": token, "C": rec["shape"]["C"], "F": rec["shape"]["packed_tiles_F"],
            "passes": sched.get("passes"), "reprogram_passes": sched.get("reprogram_passes"),
            "reprogrammed_bytes_per_inference": sched.get("reprogrammed_bytes_per_inference"),
            "programmed_bytes_once": sched.get("programmed_bytes_once"), "schedule_refusal": sched.get("refusal"),
            "label": "shape-only plane, at K x 48 cores (K NeuRRAM chips' cores, a declared departure)",
            "file": path.name, "sha256": c.sha256_file(path), "provenance": check}


def _cell(text) -> str:
    ""
    return str(text).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _executed_x(rows: list[dict], chip_pack: dict | None, shape_only: dict) -> dict:
    ""
    done = [r for r in rows if r["state"] == "complete"]
    if not done:
        return {"state": "no row reached packing"}
    res = next((r for r in done if r["budget"] == "res"), None)
    if res is None:
        return {"state": "withheld: the res row did not reach packing", "rows_reached_packing": [r["budget"] for r in done]}
    F = res["result"]["deployment"]["physical_cores"]
    if chip_pack is not None:
        C, source = chip_pack["shape"]["C"], "chip pack"
    else:
        chip = next((r for r in done if r["budget"] in ("chip", "chip+pm")), None)
        if chip is None:
            return {"state": "withheld: no C (the chip pack did not complete and no chip or chip+pm row is complete)",
                    "F": F}
        C, source = chip["result"]["deployment"]["declared_count"], f"row {chip['budget']}"
    ex = {"F": F, "C": C, "x": F / C, "C_source": source,
          "rule": "x = F / C, F the physical cores of the resident pack (the res row), C the chip's core count",
          "equals_shape_only": None, "x_equals_shape_only": None}
    if "x" in shape_only:
        ex["equals_shape_only"] = F == shape_only["F"]
        ex["x_equals_shape_only"] = ex["x"] == shape_only["x"]
    return ex


EXECUTED_PASSES = {"passes": ("pass_count", "passes"), "reprogram_passes": ("reprogram_passes", "reprogram_passes")}


NOT_COMPARABLE = {"reprogrammed_bytes_per_inference": (
    "params_reloaded", "reprogrammed_bytes_per_inference",
    "the row record's schedule.params_reloaded is the weight count Soft Core Mapping plans to load once per bank "
    "(planned@scm, the same at every core budget), and the pack's is the bytes the priced schedule rewrites per "
    "inference at C; the deployment record holds no executed bytes at C to compare")}

P_PRINT_BOUND_LOG10 = -300


def paired_p(b: int, cc: int) -> dict:
    ""
    b, cc = int(b), int(cc)
    m = b + cc
    if m == 0:
        return {"p": 1.0, "log10": 0.0, "text": "1"}
    from scipy.stats import binom

    p = float(min(1.0, 2.0 * binom.cdf(min(b, cc), m, 0.5)))
    k = min(b, cc)
    terms = [math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) - m * math.log(2.0) for i in range(k + 1)]
    top = max(terms)
    log_tail = top + math.log(sum(math.exp(t - top) for t in terms))
    log10 = min(0.0, (math.log(2.0) + log_tail) / math.log(10.0))
    if p > 0.0 and math.log10(p) >= P_PRINT_BOUND_LOG10:
        text = f"{p:.4g}"
    else:
        text = f"below 10^{P_PRINT_BOUND_LOG10} (log10 p = {log10:.2f})"
    return {"p": p, "log10": log10, "text": text}


def gib(kib) -> str:
    ""
    return f"{int(kib) / 1024 ** 2:.2f} GiB"


def _executed_passes(records: Path, group: str, rows: list[dict]) -> list[dict]:
    ""
    member, bundle, arm = group.split(".")
    try:
        tokens = configs.structure_tokens(member, bundle, arm)
    except configs.RowRefusal:
        tokens = []
    out = []
    for r in rows:
        if r["state"] != "complete":
            continue
        sched = (r["result"].get("record") or {}).get("schedule") or {}
        e = {"budget": r["budget"], **{k: sched.get(row_key) for k, (row_key, _) in EXECUTED_PASSES.items()},
             **{row_key: sched.get(row_key) for row_key, _, _ in NOT_COMPARABLE.values()},
             "pack": None, "equal": None,
             "not_comparable": {k: {"row_key": row_key, "row": sched.get(row_key), "pack": None, "reason": why}
                                for k, (row_key, _, why) in NOT_COMPARABLE.items()}}
        if r["budget"] in tokens:
            name = f"{group}.structure.json" if r["budget"] == tokens[0] else f"{group}.structure.{r['budget']}.json"
            pack = json.loads((records / name).read_text()) if (records / name).is_file() else None
            if pack is not None and pack.get("status") == "complete":
                ps = pack.get("schedule") or {}
                e["pack"] = name
                e["equal"] = {k: None if e[k] is None or ps.get(pack_key) is None else e[k] == ps.get(pack_key)
                              for k, (_, pack_key) in EXECUTED_PASSES.items()}
                for k, (_, pack_key, _) in NOT_COMPARABLE.items():
                    e["not_comparable"][k]["pack"] = ps.get(pack_key)
        out.append(e)
    return out


def collect(records: Path, group: str, budgets: list[str]) -> dict:
    import numpy as np

    rows = [_row(records, group, b) for b in budgets]
    done = [r for r in rows if r["state"] == "complete"]
    member, bundle, arm = group.split(".")
    out = {"format": FORMAT, "group": group, "budgets": budgets, "rows": [], "departures": configs.departures(arm)}
    if done:
        first = done[0]["result"]
        for r in done[1:]:
            rr = r["result"]
            if rr["inputs"]["driver_code_sha256"] != first["inputs"]["driver_code_sha256"]:
                raise SystemExit(f"REFUSED: {rr['row']} ran other driver code than {first['row']}")
            if c.provenance_differ(rr["inputs"]["toolchain"], first["inputs"]["toolchain"]):
                raise SystemExit(f"REFUSED: {rr['row']} ran another toolchain state than {first['row']}")
        ref = next((r for r in done if r["budget"] == "res"), done[0])
        ref_out = np.load(ref["dir"] / "packed_prefix_fp64.npy")
        k10 = {"reference_budget": ref["budget"], "comparisons": []}
        for r in done:
            if r is ref:
                continue
            mine = np.load(r["dir"] / "packed_prefix_fp64.npy")
            m = min(len(mine), len(ref_out))
            d = np.abs(mine[:m] - ref_out[:m])
            k10["comparisons"].append({"budget": r["budget"], "images": int(m), "outputs": int(d.size),
                                       "outputs_differ": int((d > 0).sum()), "max_abs_delta": float(d.max()),
                                       "passed": bool(d.max() == 0.0)})
        k10["passed"] = bool(k10["comparisons"]) and all(x["passed"] for x in k10["comparisons"])
        out["K10"] = k10
    for r in rows:
        if r["state"] != "complete":
            out["rows"].append({k: v for k, v in r.items() if k != "dir"})
            if r.get("peak_rss_kb") is not None:
                out["rows"][-1]["peak_rss_gib"] = gib(r["peak_rss_kb"])
            continue
        rr = r["result"]
        dep = rr["deployment"]
        out["rows"].append({
            "budget": r["budget"], "state": "complete",
            "mode": r.get("mode"), "toolchain": r.get("toolchain"),
            "split": rr.get("split"), "images": rr["K9"]["images"],
            "correct": rr["K9"]["correct"], "correct_ties_lowest_index": rr["K9"]["correct_ties_lowest_index"],
            "top1": rr["K9"]["top1"], "ci95": rr["K9"]["clopper_pearson_95"], "argmax_ties": rr["K9"]["argmax_ties"],
            "float_correct_same_images": rr["float_agreement"]["float_correct"],
            "agree_with_float": rr["float_agreement"]["agree"], "paired": rr["paired"],
            "paired_p": paired_p(rr["paired"]["float_right_deployed_wrong_b"],
                                 rr["paired"]["float_wrong_deployed_right_c"]),
            "declared_cores": dep["declared_count"], "physical_cores": dep["physical_cores"],
            "listed_cores": dep["listed_cores"], "passes_per_segment_max": dep["passes_per_segment_max"],
            "passes_summed_over_segments": dep["passes_summed_over_segments"],
            "record_pass_count": (rr["record"].get("schedule") or {}).get("pass_count"),
            "params_reloaded": (rr["record"].get("schedule") or {}).get("params_reloaded"),
            "host_mac_share": (rr.get("host") or {}).get("host_mac_share"),
            "host_operation_classes": (rr.get("host") or {}).get("host_operation_classes"),
            "pipeline_wall_s": dep["pipeline_wall_s"], "packed_s_per_image": rr["harvest"]["packed_fp64_s_per_image"],
            "golden_s_per_image": rr["harvest"]["golden_s_per_image"], "certificates": certificates(rr),
            "departures": rr.get("departures"), "faithfulness": rr["inputs"]["faithfulness_labels"]})
    structure_path = records / f"{group}.structure.json"
    structure = json.loads(structure_path.read_text()) if structure_path.exists() else None
    out["x"] = {"rule": configs.arms()["reporting"]["anchor_structure"]}
    complete = structure is not None and structure.get("status") == "complete"
    check = structure_provenance(structure, rows) if complete else None
    if complete and not check["passed"]:
        out["x"]["shape_only"] = {"state": "withheld: the structural pack is not on the rows' head or the head of "
                                           "record", "provenance": check, "file": structure_path.name,
                                  "sha256": c.sha256_file(structure_path)}
    elif complete:
        out["x"]["shape_only"] = {"F": structure["shape"]["packed_tiles_F"], "C": structure["shape"]["C"],
                                  "x": structure["shape"]["x_packed"], "x_fluid": structure["shape"]["x_fluid"],
                                  "passes": (structure.get("schedule") or {}).get("passes"),
                                  "reprogram_passes": (structure.get("schedule") or {}).get("reprogram_passes"),
                                  "reprogrammed_bytes_per_inference":
                                      (structure.get("schedule") or {}).get("reprogrammed_bytes_per_inference"),
                                  "host_mac_share": structure["host_census"]["host_mac_share"],
                                  "schedule_refusal": (structure.get("schedule") or {}).get("refusal"),
                                  "label": "shape-only plane", "file": structure_path.name,
                                  "sha256": c.sha256_file(structure_path), "provenance": check}
    else:
        out["x"]["shape_only"] = {"state": "absent" if structure is None else structure.get("status"),
                                  "error": None if structure is None else structure.get("error"),
                                  "refused_by_name": None if structure is None else preflight.named_refusal(structure),
                                  "stop": None if structure is None else preflight.stopped_pack(structure)}
    out["x"]["scheduled_multiple"] = _scheduled_multiple(records, group, rows)

    out["x"]["executed"] = _executed_x(rows, structure if complete else None, out["x"]["shape_only"])
    out["x"]["executed_passes"] = _executed_passes(records, group, rows)
    out["x"]["refusals_by_name"] = [
        {"budget": r["budget"], "refusal": name, "error": r.get("error")} for r in rows
        if r["state"] != "complete" and (name := preflight.named_refusal({"error": r.get("error"),
                                                                         "refusal": r.get("refusal")}))]
    c.atomic_write_json(records / f"{group}.group.json", out)
    lines = [f"# {group}", "", "Departures of the arm: " + "; ".join(out["departures"]), "",
             "| budget | state | images | correct (ties wrong) | top-1 | float correct | paired d, 95% (b, c) | "
             "physical cores | declared | passes max | host MAC share | pipeline wall s | K4 | K5 | K6 | K11 | K14 | "
             "NU |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in out["rows"]:
        if x["state"] != "complete":
            lines.append(f"| {x['budget']} | {x['state']} | | | | | | | | | | | | | | | | "
                         f"{_cell(x.get('error'))} |")
            continue
        k, p = x["certificates"], x["paired"]
        share = "" if x["host_mac_share"] is None else f"{x['host_mac_share']:.3f}"

        def v(name):
            return f"{k[name]['verdict']} ({k[name]['denominator']})"

        lines.append(f"| {x['budget']} | complete | {x['images']} | {x['correct']} | {x['top1']:.4f} | "
                     f"{x['float_correct_same_images']} | {p['difference_d']:+.4f} [{p['difference_ci95_conditional'][0]:+.4f}, "
                     f"{p['difference_ci95_conditional'][1]:+.4f}] ({p['float_right_deployed_wrong_b']}, "
                     f"{p['float_wrong_deployed_right_c']}) | {x['physical_cores']} | {x['declared_cores']} | "
                     f"{x['passes_per_segment_max']} | {share} | {x['pipeline_wall_s']:.0f} | {v('K4')} | {v('K5')} | "
                     f"{v('K6')} | {k['K11']['verdict']} | {k['K14']['verdict']} (difference {k['K14']['difference']}; "
                     f"{k['K14']['denominator']}) | {k['NU']['verdict']} |")
    lines += ["", "K4, K5 and K6 are sampled checks on the images named; NU is exact on every perceptron."]
    if "K10" in out:
        lines += ["", f"K10 against {out['K10']['reference_budget']}: " + "; ".join(
            f"{x['budget']} {x['outputs_differ']} of {x['outputs']} outputs differ ({x['images']} images)"
            for x in out["K10"]["comparisons"])]
    so = out["x"]["shape_only"]
    if "x" in so:
        lines += ["", f"x (shape-only plane, the value of record): F {so['F']} / C {so['C']} = {so['x']:.4f}; passes "
                      f"{so['passes']}, reprogrammed passes {so['reprogram_passes']}, bytes reprogrammed per inference "
                      f"{so['reprogrammed_bytes_per_inference']}; host MAC share {so['host_mac_share']}; schedule "
                      f"refusal {json.dumps(so['schedule_refusal'], default=str)}"]
    elif "provenance" in so:
        lines += ["", f"x (shape-only plane): withheld, not the value of record: "
                      f"{json.dumps(so['provenance']['differ'], default=str)[:2000]}"]
    else:
        lines += ["", f"x (shape-only plane): {so}"]
    km = out["x"]["scheduled_multiple"]
    if "passes" in km:
        lines += [f"passes, reloads ({km['label']}, {km['token']}, C {km['C']}): passes {km['passes']}, reprogrammed "
                  f"passes {km['reprogram_passes']}, bytes reprogrammed per inference "
                  f"{km['reprogrammed_bytes_per_inference']}; schedule refusal "
                  f"{json.dumps(km['schedule_refusal'], default=str)}"]
    elif km.get("token"):
        lines += [f"passes, reloads ({km['token']}): {km.get('state')} "
                  f"{json.dumps((km.get('provenance') or {}).get('differ') or km.get('error'), default=str)[:1500]}"]
    ex = out["x"]["executed"]
    if "x" in ex:
        lines += [f"x (executed pack, printed beside): F {ex['F']} / C {ex['C']} = {ex['x']:.4f}; equal F: "
                  f"{ex.get('equals_shape_only')}; equal x: {ex.get('x_equals_shape_only')}"]
    else:
        lines += [f"x (executed pack): {ex['state']}"]
    for e in out["x"]["executed_passes"]:
        q = e["equal"]
        nc = e["not_comparable"]["reprogrammed_bytes_per_inference"]
        lines += [f"passes (executed, {e['budget']}): passes {e['passes']}, reprogrammed passes {e['reprogram_passes']}, "
                  f"parameters reloaded (planned at Soft Core Mapping) {e['params_reloaded']}; " + (
                      f"equal to the {e['budget']} pack: passes {q['passes']}, reprogrammed passes "
                      f"{q['reprogram_passes']}; bytes reprogrammed per inference: not comparable ({nc['reason']})"
                      if q is not None else f"no shape-only pack at {e['budget']}")]
    ps = [r for r in out["rows"] if r.get("paired_p")]
    if ps:
        lines += ["McNemar exact p per complete row (a p below 10^-300 by its bound and its log10): " + "; ".join(
            f"{r['budget']} {r['paired_p']['text']}" for r in ps)]
    if out["x"]["refusals_by_name"]:
        lines += ["refusals by name of the group's rows, in full: " + "; ".join(
            f"{e['budget']} ({e['refusal']}): {_cell(e['error'])}" for e in out["x"]["refusals_by_name"])]
    (records / f"{group}.group.md").write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--budgets", default=None)
    ap.add_argument("--structure-only", action="store_true", help="the structure-only check of a pack (D-9 (i))")
    a = ap.parse_args()
    lock = lockcheck.verify()
    start = c.driver_code_sha256()
    if a.structure_only:
        rec = structure_check(Path(a.records), a.group)
        c.refuse_if_changed(start, "the structure-only check")
        print(json.dumps({"lock": lock["lock_sha256"], **{k: rec.get(k) for k in (
            "group", "structure_sha256", "structure_status", "passed", "reason", "refused_by_name", "stopped")}},
                         indent=1,
                         default=str))
        return 0 if rec["passed"] else 1
    if a.budgets is None:
        raise SystemExit("REFUSED: --budgets is required without --structure-only")
    budgets = [t for t, _ in configs.parse_budgets(a.budgets)]
    out = collect(Path(a.records), a.group, budgets)
    c.refuse_if_changed(start, "the collector")
    print(json.dumps({"lock": lock["lock_sha256"], **{k: out.get(k) for k in ("group", "K10", "x")}}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
