""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("w0_price_e1", HERE / "w0_price.py")
W0 = importlib.util.module_from_spec(_spec)
sys.modules["w0_price_e1"] = W0
_spec.loader.exec_module(W0)
WP, W48 = W0.WP, W0.W48
VARIANTS = ("study:shipped", "study:sync100", "study:sync0")


def arm_of(budget: int, limit: str, buffer: bool, slots: int = 0, chip_slots=None):
    declare = {"max_schedule_passes": budget, "pipelined_multiplexing": True}
    name = f"B5m@{budget}+pipelined"
    if limit != "A":
        declare.update(layer_replication=True, wavefront_group_cores=int(limit))
        name += f"+copies@{int(limit)}"
    if slots:
        declare.update(batch_pipeline_slots=int(slots))
        name += f"+slots@{int(slots)}"
    if buffer:
        declare.update(WP.buffer_declaration()[0])
        name += "+buffer1MiB"
    if chip_slots is not None:
        declare.update(carry_input_slots=chip_slots)
    return name, declare


def plan_summary(softcores, plan) -> dict:
    groups = []
    for g in (plan.wavefront.groups if getattr(plan, "wavefront", None) is not None else ()):
        n = len(g.keys) if g.copies is None else int(sum(g.copies))
        groups.append({"levels": len(g.levels), "passes": g.end_pass - g.first_pass, "pipelined": bool(g.pipelined),
                       "keys": len(g.keys), "copies": n, "most": None if g.copies is None else int(max(g.copies))})
    first = softcores[0] if len(softcores) else None
    return {"segment": None if first is None else getattr(first, "segment_id", None), "specs": len(softcores),
            "passes": len(plan.pass_lists), "wavefront": getattr(plan, "wavefront", None) is not None,
            "groups": groups}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worktree", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cores", required=True, type=int)
    ap.add_argument("--budget", required=True, type=int)
    ap.add_argument("--limit", required=True)
    ap.add_argument("--buffer", required=True, type=int, choices=(0, 1))
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--ks", default=None)
    ap.add_argument("--point-budget-s", type=float, default=7200.0)
    args = ap.parse_args(argv)
    start = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log")

    def log(line: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.time() - start:8.1f} {line}\n")

    head = subprocess.run(["git", "-C", str(args.worktree), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    ks = tuple(int(k) for k in (args.ks or (str(args.slots) if args.slots else "1,2,4,8,16")).split(","))
    tc = W0.W0Toolchain(args.worktree)
    tc.count = args.cores
    from mimarsinan.deployment_record.introspection import scaling_ladder
    from mimarsinan.mapping.noc import collect_noc_fragments
    from mimarsinan.mapping.support.schedule import pass_planner as PP
    original, captured = PP.plan_segment_passes, []

    def recorded(softcores, *a, **kw):
        plan = original(softcores, *a, **kw)
        captured.append(plan_summary(softcores, plan))
        return plan
    for module in list(sys.modules.values()):
        if getattr(module, "plan_segment_passes", None) is original:
            setattr(module, "plan_segment_passes", recorded)
    chip_slots = tc.chip_values(W0.CHIP).get("carry_input_slots")
    arm, declare = arm_of(args.budget, args.limit, bool(args.buffer), args.slots, chip_slots)
    src = WP.with_pipelined_arm(W48.route_members())[W0.SOURCE_MEMBER]
    members = {W0.MEMBER: {**{k: v for k, v in src.items() if k not in ("arms", "declare", "not_run")},
                           "arms": {arm: src["arms"]["B5m"]}, "declare": {arm: declare}}}
    log(json.dumps({"start": True, "head": head, "cores": args.cores, "arm": arm, "declare": declare, "ks": ks}))
    priced = W48.price_members(tc, members, [W0.MEMBER], (arm,), ks, ("candidate",), VARIANTS,
                               W48.DEFAULT_RECORD_ROOT, args.point_budget_s, log, chips={W0.MEMBER: W0.CHIP})
    try:
        live = WP.live_peaks(tc, priced, members, arm)
    except Exception as exc:
        live = {"error": f"{type(exc).__name__}: {exc}"[:300]}
    base = tc.platform(members[W0.MEMBER]["arms"][arm], declare=declare)
    resolved = {k: base.get(k) for k in ("layer_replication", "wavefront_group_cores", "pipelined_multiplexing",
                                         "max_schedule_passes", "carry_buffer_bytes", "batch_pipeline_slots",
                                         "carry_input_slots")}
    plans = {}
    for p in captured:
        plans[f"{p['segment']}:{p['specs']}"] = p
    plans = list(plans.values())
    result = {"format": "thru-e2/1", "slots": args.slots, "branch": "tcad/throughput-replication", "head": head,
              "script_sha256": W48.sha256(Path(__file__).resolve()), "w0_price_sha256": W48.sha256(HERE / "w0_price.py"),
              "worktree": str(args.worktree), "chip": W0.CHIP, "cores": args.cores, "budget": args.budget,
              "limit": args.limit, "buffer": bool(args.buffer), "arm": arm, "declare": declare, "resolved": resolved,
              "ks": list(ks), "variants": list(VARIANTS), "planner_calls": len(captured), "plans": plans,
              "live_peaks": live, "members": priced, "wall_s": round(time.time() - start, 1)}
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    log(json.dumps({"done": True, "wall_s": result["wall_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
