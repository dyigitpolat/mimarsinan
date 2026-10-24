#!/usr/bin/env python3
"""X3a randomized deployments (plan section 10.2): the code-level screen, its report and the compile wall.

    random_deployments.py screen  --out DIR [--arms a,b] [--target N] [--seed0 S] [--max-seeds M] [--workers W]
    random_deployments.py report  --out DIR [--model-level FILE] [--table FILE]
    random_deployments.py nevresim --out DIR [--semantics S] [--target N] [--arm A] [--workdir W]
    random_deployments.py instance --seed S [--arm A]          (print one instance document, canonical JSON)

``nevresim`` sends the first ``--target`` admitted instances of one semantics to nevresim (one binary per
neural stage, compiled per instance: the connectivity-keyed compile cache never hits on random topologies)
and compares each against the identity reference per neuron; it records the compile wall apart.

``screen`` runs each arm over consecutive seeds until ``--target`` instances are admitted (built and run),
in seed order whatever the worker count, and writes ``DIR/<arm>.jsonl`` (one verdict per seed). Every
counterexample and run error is root-caused by ablation (``attribution``); an off arm's counterexample is
re-run as its paired control (the same instance with the arm's field on). Each verdict also carries the
identity program against the integer network N (Proposition 2's relation), the input dependence of the
identity program and the weight range of every layer; a value run (G2's integer readout law, from CL1) also
carries its identity program and deployment against the int64 golden at atol 0. At most 8 workers.

``report`` pools the hypothesis-on strata that run the toolchain as shipped, prints the harness-setting stratum
and the stratum beyond the declared window range (``on_long_window``, T in 9-64) apart, and adds the network
relation per semantics, the int64 golden relation of the value runs, the coverage report and the denominators
over every hypothesis of the theorem box (``--table``, default the in-tree fixture).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

MAX_WORKERS = 8
TABLE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "theorem_box" / "hypothesis_table.json"


def _one(task: Tuple[int, str]) -> Dict[str, Any]:
    import torch

    from mimarsinan.certification.randomized.arms import ARMS
    from mimarsinan.certification.randomized.attribution import attribute
    from mimarsinan.certification.randomized.runner import COUNTEREXAMPLE, RUN_ERROR, instance_for, run_instance

    torch.set_num_threads(1)
    seed, arm_id = task
    verdict = run_instance(seed, arm_id)
    row = verdict.to_dict()
    row["timesteps"] = instance_for(seed, ARMS[arm_id]).timesteps
    if verdict.outcome in (COUNTEREXAMPLE, RUN_ERROR):
        row["classes"] = list(attribute(verdict))
    if verdict.outcome == COUNTEREXAMPLE and ARMS[arm_id].changes_a_field:
        row["control"] = run_instance(seed, arm_id, control=True).outcome
    return row


def _teeth(rows: List[Dict[str, Any]], arm_id: str, args) -> bool:
    """With ``--until-teeth``, an off arm keeps drawing past its target until one counterexample has teeth
    (plan section 10.4 acceptance 5: the arm is rerun until it yields a counterexample, the count printed)."""
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
            # Decision invariance on nevresim: the deployment against the identity program, both on nevresim.
            row["windows"], row["mismatches"], row["first"] = compare(ref_ir, nev_ref, ir, nev_dep)
            row["outcome"] = "equal" if row["mismatches"] == 0 else "counterexample"
            # Backend parity, apart: the identity program on nevresim against the same program on HCM.
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
