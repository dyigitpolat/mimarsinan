""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import common
import configs
import declarations
import inventory

RESULT_FORMAT = "e2e-row-result/1"
ROWS = common.ROWS
ALLOW_SMOKE = False
PLAN = None


def _toolchain_state(res: dict) -> tuple:
    ""
    t = res["inputs"]["toolchain"]
    return (t.get("commit"), t.get("diff_sha256"), json.dumps(t.get("untracked_sha256", {}), sort_keys=True))


def driver_files_differing(res: dict, current: dict) -> list:
    ""
    own = res["inputs"].get("session_driver_code_sha256", {})
    return sorted(k for k in set(own) | set(current) if own.get(k) != current.get(k))


def regeneration_refusal(res: dict, row_dir: Path, row: configs.Row) -> str | None:
    ""
    config = json.loads((row_dir / "config.json").read_text())
    try:
        again = configs.make_config(row, footprint=res["deployment"].get("footprint_input"),
                                    **configs.declared_inputs(config, row))
    except (SystemExit, KeyError, TypeError, ValueError) as exc:
        return f"config not regenerable by the current driver: {exc!r}"
    return None if again == config else "the current driver declares this row's config differently"


def check(res: dict, row_dir: Path, state: tuple | None, current: dict | None = None) -> str | None:
    ""
    if res.get("format") != RESULT_FORMAT:
        return f"foreign format {res.get('format')!r}"
    if res.get("status") != "complete" or "K9" not in res:
        return f"incomplete (status {res.get('status')!r})"
    if res.get("smoke") and not ALLOW_SMOKE:
        return "smoke row (the smoke provider's 64 E01 images): never in a table of record"
    if state is not None and _toolchain_state(res) != state:
        return "foreign code state (toolchain commit, diff or untracked files differ from the collection's)"
    if res["inputs"].get("driver_code_sha256") != res["inputs"].get("session_driver_code_sha256"):
        return "harvest code differs from the session's driver code"
    cfg = row_dir / "config.json"
    if not cfg.exists() or common.sha256_file(cfg) != res["inputs"]["config_sha256"]:
        return "config digest does not verify"
    try:
        row = configs.make_row(res["member"], res["target"], res["route"], res["budget"], res["geometry"])
    except SystemExit as exc:
        return f"row not buildable: {exc}"
    current = common.driver_code_sha256() if current is None else current
    if driver_files_differing(res, current):
        reason = regeneration_refusal(res, row_dir, row)
        if reason:
            return reason
    if res["inputs"]["checkpoint_sha256"] != configs.checkpoint_of(row).sha256:
        return "checkpoint digest differs from the inventory"
    pred = row_dir / "pred.npy"
    if not pred.exists() or common.sha256_file(pred) != res["K9"]["files"]["pred.npy"]:
        return "prediction file does not verify"
    if PLAN is not None and row.v3:
        import templates

        tc = templates.check_config(json.loads(cfg.read_text()), row, ROWS, PLAN)
        if not tc["passed"]:
            return f"the config does not map to its locked template: {tc['problems'][:3]}"
    return None


def budget_invariance(rows: dict[str, dict]) -> dict:
    ""
    groups: dict[tuple, dict[str, str]] = {}
    for key, res in rows.items():
        budgets = groups.setdefault((res["member"], res["target"], res["route"]), {})
        if res["budget"] == "B0" and "B0" in budgets:
            raise SystemExit(f"REFUSED: two B0 rows of one (member, target, route): {budgets['B0']} and {key}")
        budgets[res["budget"]] = key
    out = {}
    for group, budgets in sorted(groups.items()):
        b0_dir, b0_res, source = None, None, None
        if "B0" in budgets:
            b0_key, b0_dir = budgets["B0"], ROWS / budgets["B0"]
        else:

            if common.FOOTPRINT_ROOT is None and common.CARRIED_ROWS_ROOT is None:
                continue
            try:
                b0_key = configs.make_row(group[0], group[1], group[2], "B0").key
            except SystemExit:
                continue
            b0_dir, source = common.needed_row_dir(b0_key, ROWS)
            if source["read_from"] == "rows root" or not (b0_dir / "packed_fp64.npy").exists():
                continue
            b0_res = json.loads((b0_dir / "result.json").read_text()) if (b0_dir / "result.json").exists() else {}
            if b0_res.get("status") != "complete":
                continue
        base = np.load(b0_dir / "packed_fp64.npy")
        y = np.load(common.RECORDS / "wi2" / group[0] / "e01_labels.npy")[: len(base)] if (
            common.RECORDS / "wi2" / group[0] / "e01_labels.npy").exists() else np.load(
            common.RECORDS / "wi1" / group[0] / "e01_labels.npy")[: len(base)]
        for budget, key in sorted(budgets.items()):
            if budget == "B0":
                continue
            other = np.load(ROWS / key / "packed_fp64.npy")
            if other.shape != base.shape:
                out[key] = {"against": b0_key, "passed": False, "reason": "outputs of another shape"}
                continue
            pb, po = base.argmax(1), other.argmax(1)
            out[key] = {"against": b0_key, "outputs": int(other.size),
                        "outputs_differ": int((other != base).sum()), "max_abs_delta": float(np.abs(other - base).max()),
                        "pred_differ": int((pb != po).sum()),
                        "accuracy_difference_images": int((po == y).sum() - (pb == y).sum()),
                        "passed": bool(np.array_equal(other, base))}
            if source is not None:
                out[key].update(against_rows_root=source["rows_root"], against_read_from=source["read_from"],
                                b0_result_sha256=common.sha256_file(b0_dir / "result.json"))
            if group[2] in declarations.TUNED_ROUTES:
                mine = (rows[key].get("tuned_network") or {}).get("state_sha256")
                theirs = ((rows[b0_key] if b0_res is None else b0_res).get("tuned_network") or {}).get("state_sha256")
                same = bool(mine) and mine == theirs
                out[key].update(same_tuned_network=same,
                                reading="budget invariance" if same else "two training draws: not a budget-invariance "
                                                                          "statement")
    return out


def k14_census_count(k14: dict | None):
    ""
    k14 = k14 or {}
    for name in ("pipeline_census_count", "pipeline_fp32_census_count"):
        if name in k14:
            return k14[name]
    return None


def _verdict(entry: dict | None, key: str = "passed") -> str:
    if not entry:
        return "not run"
    if entry.get("applicable") is False:
        return "not applicable"
    return "pass" if entry.get(key) else "FAIL"


def table_rows(rows: dict[str, dict], k10: dict, current: dict | None = None) -> list[dict]:
    out = []
    current = common.driver_code_sha256() if current is None else current
    for key, res in sorted(rows.items()):
        member = inventory.member(res["member"])
        row_ = configs.make_row(res["member"], res["target"], res["route"], res["budget"], res["geometry"])
        k13 = res.get("K13") or {}
        k4k5 = res.get("K4_K5_pipeline", {})
        out.append({
            "row": key, "member": res["member"], "label": member.label, "target": res["target"], "route": res["route"],
            "budget": res["budget"], "geometry": res["geometry"],
            "activation_bits": res["deployment"].get("activation_bits", configs.ACTIVATION_BITS.get(res["route"])),
            "driver_listing_sha256": common.listing_sha256(res["inputs"].get("session_driver_code_sha256", {})),
            "driver_files_differing_from_collection": driver_files_differing(res, current),
            "declared_cores": res["deployment"]["declared_count"], "physical_cores": res["deployment"]["physical_cores"],
            "max_passes": res["deployment"]["max_passes"],
            "harness_count_recorded": configs.checkpoint_of(row_).e01_correct,
            "smoke": bool(res.get("smoke")), "images": res["K9"].get("images"),
            "K7": (res.get("K7_window_counts") or {}).get("verdict", "not run"),
            "K8": (res.get("K8_nevresim") or {}).get("verdict", "not run"),
            "K12": _verdict(res.get("K12")) if res.get("K12") else "not applicable",
            "design_Q_against_toolchain_passes": (res.get("design") or {}).get("Q_exploration_against_toolchain_passes"),
            "static_check_theorem_1_failed": (res.get("static_program_check") or {}).get("theorem_1_failed"),
            "toolchain_count_fp64": res["K9"]["correct"],
            "agree_harness_fp64": k13.get("agree_harness_fp64"), "agree_harness_fp32": k13.get("agree_harness_fp32"),
            "harness_cpu_fp32": k13.get("harness_fp32_correct"), "harness_cpu_fp64": k13.get("harness_fp64_correct"),
            "K4": (k4k5.get("K4_value_parity_gate") or {}).get("verdict", "not run"),
            "K4_samples": (k4k5.get("K4_value_parity_gate") or {}).get("samples"),
            "K5": (k4k5.get("K5_value_twin_certificate") or {}).get("verdict", "not run"),
            "K5_samples": (k4k5.get("K5_value_twin_certificate") or {}).get("samples"),
            "K6": _verdict(res.get("K6")), "K6_samples": (res.get("K6") or {}).get("samples"),
            "K10": _verdict(k10.get(key)) if res["budget"] != "B0" else "reference",
            "K10_reading": (k10.get(key) or {}).get("reading"),
            "device": (res.get("device") or {}).get("device"),
            "tuned_network_sha256": (res.get("tuned_network") or {}).get("state_sha256"),
            "K11": _verdict(res.get("K11")),
            "K14_difference": (res.get("K14") or {}).get("difference"),
            "K14_census_count": k14_census_count(res.get("K14")),
            "no_update": _verdict(res.get("no_update")),
            "test_count": "not read (section 8)",
        })
    return out


def write_markdown(path: Path, table: list[dict], refused: dict, k10: dict) -> None:
    head = ["row", "activation bits", "declared cores", "physical cores", "passes", "harness count (recorded)",
            "toolchain count (fp64)", "agreement with harness fp64", "K4", "K5", "K6", "K10", "K11", "K14 difference",
            "no update", "test count"]
    lines = ["# E2E Rows", "", "Generated by `driver/collect.py` from `records/rows/*/result.json`; counts are of the "
             "5,000 E01 images (MEASURED); agreement is the number of the 5,000 E01 images whose prediction equals "
             "the harness's CPU fp64 prediction (WI-2); no update is the check that the deployed weights are the "
             "checkpoint's rate-1 projection with no training step (`driver/no_update.py`).", "",
             "| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]
    for r in table:
        cells = [r["row"], r["activation_bits"], r["declared_cores"], r["physical_cores"], r["max_passes"],
                 r["harness_count_recorded"],
                 r["toolchain_count_fp64"], r["agree_harness_fp64"], f"{r['K4']} ({r['K4_samples']})",
                 f"{r['K5']} ({r['K5_samples']})", f"{r['K6']} ({r['K6_samples']})", r["K10"], r["K11"],
                 r["K14_difference"], r["no_update"], r["test_count"]]
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    if refused:
        lines += ["", "## Refused Records", "", "| row | reason |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in sorted(refused.items())]
    path.write_text("\n".join(lines) + "\n")


def _wanted(key: str, routes: set | None, budgets: set | None, members: set | None = None,
            exclude: set | None = None) -> bool:
    parts = key.split(".")
    return (len(parts) == 5 and (routes is None or parts[2] in routes) and (budgets is None or parts[3] in budgets)
            and (members is None or parts[0] in members) and ".".join(parts[:3]) not in (exclude or set()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolchain-commit", default=None, help="refuse rows of any other toolchain commit")
    ap.add_argument("--routes", default=None, help="comma-separated routes to collect (default: every route)")
    ap.add_argument("--budgets", default=None, help="comma-separated budgets to collect (default: every budget)")
    ap.add_argument("--members", default=None, help="comma-separated members to collect (default: every member; "
                    "unit ur2, Q-6: one collection per host form of the tuned arm)")
    ap.add_argument("--out", default=str(common.RECORDS), help="folder of table.json and table.md")
    ap.add_argument("--smoke", action="store_true", help="a smoke collection (smoke rows admitted, labelled smoke)")
    ap.add_argument("--plan", default=None, help="routeb dir: refuse a row whose config is not its locked template")
    ap.add_argument("--exclude", default=None, help="[U-HOST] comma-separated <member>.<target>.<route> prefixes left "
                    "out (another host form's rows)")
    a = ap.parse_args()
    global ALLOW_SMOKE, PLAN
    ALLOW_SMOKE = bool(a.smoke)
    PLAN = Path(a.plan) if a.plan else None
    routes = set(a.routes.split(",")) if a.routes else None
    budgets = set(a.budgets.split(",")) if a.budgets else None
    members = set(filter(None, a.members.split(","))) if a.members is not None else None
    exclude = set(filter(None, a.exclude.split(","))) if a.exclude else None
    out_dir = Path(a.out)
    current = common.driver_code_sha256()
    results = {}
    for res_path in sorted(ROWS.glob("*/result.json")):
        if _wanted(res_path.parent.name, routes, budgets, members, exclude):
            results[res_path.parent.name] = json.loads(res_path.read_text())
    states = {_toolchain_state(r) for r in results.values() if r.get("format") == RESULT_FORMAT and "inputs" in r}
    state = next(iter(states)) if len(states) == 1 else None
    refused, accepted, discovery = {}, {}, {}
    for key, res in results.items():
        reason = ("rows of more than one toolchain state" if len(states) > 1
                  else check(res, ROWS / key, state, current))
        if reason is None and a.toolchain_commit and res["inputs"]["toolchain"]["commit"] != a.toolchain_commit:
            reason = f"foreign toolchain commit {res['inputs']['toolchain']['commit']}"
        if reason:
            refused[key] = reason
        elif res["budget"] == "disc" and res["route"] not in configs.V2_ROUTES:
            discovery[key] = {"physical_cores": res["deployment"]["physical_cores"],
                              "note": "a v3 discovery row measures F on 64 E01 images and never enters a table"}
        else:
            accepted[key] = res
    k10 = budget_invariance(accepted)
    table = table_rows(accepted, k10, current)
    filters = {"routes": a.routes, "budgets": a.budgets, "members": a.members}
    if a.exclude:
        filters["exclude"] = a.exclude
    common.atomic_write_json(out_dir / "table.json", {"rows": table, "K10": k10, "refused": refused,
                                                      "discovery_rows": discovery,
                                                      "toolchain_state": list(state) if state else None,
                                                      "collector_driver_listing_sha256": common.listing_sha256(current),
                                                      "filters": filters})
    write_markdown(out_dir / "table.md", table, refused, k10)
    print(json.dumps({"accepted": sorted(accepted), "refused": refused, "K10": k10}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
