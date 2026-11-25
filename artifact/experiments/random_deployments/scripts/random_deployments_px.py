""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MAX_WORKERS = 8
HERE = os.path.dirname(os.path.abspath(__file__))

TREE = Path(__import__("mimarsinan").__file__).resolve().parents[2]
TABLE = TREE / "tests" / "fixtures" / "theorem_box" / "hypothesis_table.json"


def _windowed(seed: int, arm_id: str):
    ""
    import mimarsinan.certification.randomized.executors as ex
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.runner import semantics_for

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import window_rule as W

    run = ex.RUNS[semantics_for(seed, ARMS[arm_id])]
    from mimarsinan.mapping.ir_mapping_class import IRMapping

    return W.building(IRMapping, "map", run.config)


def _one(task: Tuple[int, str]) -> Dict[str, Any]:
    import torch

    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.attribution import attribute
    from mimarsinan.certification.randomized.runner import (COUNTEREXAMPLE, REFERENCE_ERROR, REFUSED, REFUSED_UNNAMED,
                                                            RUN_ERROR, instance_for, run_instance)

    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import error_site as ES

    torch.set_num_threads(1)
    seed, arm_id = task
    arm = ARMS[arm_id]
    with _windowed(seed, arm_id):
        verdict = run_instance(seed, arm_id)
        row = verdict.to_dict()
        row["timesteps"] = instance_for(seed, arm).timesteps
        if verdict.outcome in (RUN_ERROR, REFERENCE_ERROR, REFUSED, REFUSED_UNNAMED) and verdict.detail:

            row["error_site"] = ES.site(lambda: run_instance(seed, arm_id), verdict.detail)
            if ES.unnamed(verdict.detail):
                sw = ES.switch((lambda: run_instance(seed, arm_id, control=True)) if arm.switch is not None else None,
                               verdict.detail)
                if sw is not None:
                    row["error_site"]["switch"] = sw
        if verdict.outcome in (COUNTEREXAMPLE, RUN_ERROR):
            row["classes"] = list(attribute(verdict))
        if verdict.outcome == COUNTEREXAMPLE and arm.changes_a_field:
            ctl = run_instance(seed, arm_id, control=True)
            row["control"] = ctl.outcome
            if ctl.detail:
                row["control_site"] = ES.site(lambda: run_instance(seed, arm_id, control=True), ctl.detail)
    return row


def _teeth(rows: List[Dict[str, Any]], arm_id: str, args) -> bool:
    ""
    from mimarsinan.certification.randomized.arms import ARMS

    arm = ARMS[arm_id]
    if not args.until_teeth or not arm.hypothesis:
        return True
    if arm.changes_a_field:
        return any(r.get("control") == "equal" for r in rows)
    return any(arm.mechanism in r.get("classes", ()) for r in rows)


def screen(args) -> int:
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.screen import ADMITTED

    workers = min(int(args.workers), MAX_WORKERS)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",") if args.arms else [a for a, arm in ARMS.items() if arm.exercisable]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=200) as pool:
        for arm_id in arms:
            rows: List[Dict[str, Any]] = []
            seed, admitted = int(args.seed0), 0
            while (admitted < int(args.target) or not _teeth(rows, arm_id, args)) \
                    and seed < int(args.seed0) + int(args.max_seeds):
                batch = [(s, arm_id) for s in range(seed, min(seed + 4 * workers,
                                                             int(args.seed0) + int(args.max_seeds)))]
                seed += len(batch)
                for row in pool.map(_one, batch):
                    if admitted >= int(args.target) and _teeth(rows, arm_id, args):
                        break
                    rows.append(row)
                    admitted += row["outcome"] in ADMITTED
            with open(out / f"{arm_id.replace('/', '_')}.jsonl", "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"{arm_id}: {len(rows)} seeds, {admitted} admitted", flush=True)
    return 0


def nevresim(args) -> int:
    import tempfile
    import time

    import mimarsinan.certification.randomized.executors as ex
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.cost_draws import platform_constraints
    from mimarsinan.certification.randomized.runner import (
        EQUAL, compare, deployment, instance_for, reference_nodes, run_instance, semantics_for)

    arm = ARMS[args.arm]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seed = [], int(args.seed0)
    while len(rows) < int(args.target) and seed < int(args.seed0) + int(args.max_seeds):
        seed += 1
        if semantics_for(seed - 1, arm) != args.semantics:
            continue
        with _windowed(seed - 1, args.arm):
            verdict = run_instance(seed - 1, args.arm)
            if verdict.outcome != EQUAL:
                continue
            instance = instance_for(seed - 1, arm)
            constraints = platform_constraints(verdict.declaration, hardware_bias=instance.hardware_bias)
            ref_ir, ref_map, ref_nodes = reference_nodes(instance, arm)
            ir, mapping = deployment(instance, constraints, seed - 1, arm)
        row = {"seed": seed - 1, "semantics": args.semantics, "stages": verdict.stages,
               "multipass": verdict.multipass, "hcm_wall_s": verdict.wall_s}
        started = time.perf_counter()
        try:
            nev_ref, compile_ref = ex.run_nevresim_nodes(
                ref_map, instance, tempfile.mkdtemp(prefix=f"x3a_{seed - 1}_ref_", dir=args.workdir or None))
            nev_dep, compile_dep = ex.run_nevresim_nodes(
                mapping, instance, tempfile.mkdtemp(prefix=f"x3a_{seed - 1}_dep_", dir=args.workdir or None))

            row["windows"], row["mismatches"], row["first"] = compare(ref_ir, nev_ref, ir, nev_dep)
            row["outcome"] = "equal" if row["mismatches"] == 0 else "counterexample"

            _w, row["parity_mismatches"], row["parity_first"] = compare(ref_ir, ref_nodes, ref_ir, nev_ref)
            row["compile_s"], row["compile_reference_s"] = round(compile_dep, 3), round(compile_ref, 3)
        except (RuntimeError, ValueError) as exc:
            row["outcome"], row["detail"] = "nevresim_error", f"{type(exc).__name__}: {str(exc)[:200]}"
        row["nevresim_wall_s"] = round(time.perf_counter() - started, 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(out / f"nevresim_{args.semantics}.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


def report(args) -> int:
    from mimarsinan.certification.randomized.screen import report as build

    rows = [json.loads(line) for path in sorted(Path(args.out).glob("*.jsonl")) for line in path.open()]
    model = json.loads(Path(args.model_level).read_text()) if args.model_level else None
    doc = build(rows, table=json.loads(Path(args.table).read_text()), model_level=model)
    target = Path(args.out) / "report.json"
    target.write_text(json.dumps(doc, indent=1, sort_keys=True, default=list))
    print(target)
    return 0


def instance(args) -> int:
    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.runner import instance_for

    print(instance_for(int(args.seed), ARMS[args.arm]).to_json())
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="")
    p.add_argument("--target", type=int, default=100)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=5000)
    p.add_argument("--until-teeth", action="store_true")
    p.add_argument("--workers", type=int, default=min(MAX_WORKERS, os.cpu_count() or 1))
    p.set_defaults(func=screen)
    p = sub.add_parser("nevresim")
    p.add_argument("--out", required=True)
    p.add_argument("--semantics", default="cascaded_ttfs")
    p.add_argument("--arm", default="on_scheduled")
    p.add_argument("--target", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--max-seeds", type=int, default=5000)
    p.add_argument("--workdir", default="")
    p.set_defaults(func=nevresim)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--model-level", default="")
    p.add_argument("--table", default=str(TABLE))
    p.set_defaults(func=report)
    p = sub.add_parser("instance")
    p.add_argument("--seed", required=True)
    p.add_argument("--arm", default="on")
    p.set_defaults(func=instance)
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) > MAX_WORKERS:
        print(f"--workers {args.workers} exceeds {MAX_WORKERS}; capped", file=sys.stderr)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
