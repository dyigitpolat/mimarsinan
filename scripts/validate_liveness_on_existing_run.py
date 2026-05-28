"""Smoke-test the new dead-cores-cleanly machinery against an existing
post-pruning IR pickle.

Loads ``Soft Core Mapping.ir_graph.pickle`` from a finished run, classifies
every NeuralCore via ``compute_liveness``, counts ``(1, 1)`` legacy
placeholders, runs ``IRGraph.remove_nodes`` for the ``DEAD`` set, and
verifies the surviving graph still validates.

The plan calls this out as the "Validation (after implementation)"
checklist:

    1. Confirm the IR pickle has 576 - 106 = 470 NeuralCores (or fewer).
    2. Confirm 0 cores with core_matrix.shape == (1, 1).
    3. Confirm 17 cores classified bias_only.

Usage::

    python scripts/validate_liveness_on_existing_run.py \
        generated/mnist_hard_all_lif_ca60_phased_deployment_run_20260522_030439
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import Counter

import numpy as np

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.ir_liveness import (
    NodeLiveness,
    compute_liveness,
)


def _count_placeholders(graph: IRGraph) -> int:
    n = 0
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        try:
            mat = node.get_core_matrix(graph)
        except Exception:
            continue
        if tuple(int(d) for d in mat.shape) == (1, 1):
            n += 1
    return n


def _shape_histogram(graph: IRGraph) -> Counter:
    counts: Counter = Counter()
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        try:
            mat = node.get_core_matrix(graph)
        except Exception:
            continue
        shape = tuple(int(d) for d in mat.shape)
        if shape[0] == 1:
            counts["1xN"] += 1
        elif shape[1] == 1:
            counts["Nx1"] += 1
        elif shape == (1, 1):
            counts["1x1"] += 1
        else:
            counts["other"] += 1
    return counts


def _load_spiking_mode(run_dir: str) -> str:
    import json

    for name in ("config.json", "pipeline_config.json"):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        dp = cfg.get("deployment_parameters", cfg)
        if isinstance(dp, dict) and "spiking_mode" in dp:
            return str(dp["spiking_mode"])
        if "spiking_mode" in cfg:
            return str(cfg["spiking_mode"])
    return "lif"


def main(run_dir: str, *, spiking_mode: str | None = None) -> int:
    pickle_path = os.path.join(run_dir, "Soft Core Mapping.ir_graph.pickle")
    if not os.path.exists(pickle_path):
        print(f"FAIL: {pickle_path!s} does not exist", file=sys.stderr)
        return 2

    with open(pickle_path, "rb") as fh:
        ir_graph: IRGraph = pickle.load(fh)

    n_total = sum(1 for n in ir_graph.nodes if isinstance(n, NeuralCore))
    n_placeholders = _count_placeholders(ir_graph)
    shapes = _shape_histogram(ir_graph)

    print(f"Loaded {pickle_path}")
    print(f"  NeuralCores total: {n_total}")
    print(f"  (1, 1) placeholders observed (legacy): {n_placeholders}")
    print(f"  Shape histogram: {dict(shapes)}")

    mode = spiking_mode or _load_spiking_mode(run_dir)
    sim_steps = 32
    for name in ("config.json", "pipeline_config.json"):
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        import json

        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        dp = cfg.get("deployment_parameters", cfg)
        if isinstance(dp, dict) and "simulation_steps" in dp:
            sim_steps = int(dp["simulation_steps"])
            break
    print(f"  spiking_mode={mode!r} simulation_steps={sim_steps}")
    liveness = compute_liveness(
        ir_graph,
        simulation_steps=sim_steps,
        spiking_mode=mode,
    )
    by_status: Counter = Counter()
    for nid, status in liveness.per_node.items():
        by_status[status.value] += 1
    print(f"  Liveness classification: {dict(by_status)}")

    # Show a few examples per class for quick eyeballing.
    examples = {NodeLiveness.DEAD: [], NodeLiveness.BIAS_ONLY: []}
    for nid, status in liveness.per_node.items():
        if status in examples and len(examples[status]) < 3:
            examples[status].append((nid, liveness.reasons[nid]))
    for status, sample in examples.items():
        if not sample:
            continue
        print(f"  {status.value} examples:")
        for nid, why in sample:
            print(f"    id={nid}: {why}")

    dead_ids = sorted(
        nid for nid, status in liveness.per_node.items()
        if status == NodeLiveness.DEAD
    )
    bias_only_count = sum(
        1 for status in liveness.per_node.values()
        if status == NodeLiveness.BIAS_ONLY
    )
    print(f"  DEAD count: {len(dead_ids)}  BIAS_ONLY count: {bias_only_count}")

    if dead_ids:
        try:
            ir_graph.remove_nodes(dead_ids)
        except ValueError as exc:
            print(f"  remove_nodes raised: {exc}", file=sys.stderr)
            return 1
        n_after = sum(1 for n in ir_graph.nodes if isinstance(n, NeuralCore))
        n_placeholders_after = _count_placeholders(ir_graph)
        print(
            f"  After remove_nodes: NeuralCores={n_after} "
            f"({n_total} - {len(dead_ids)} = {n_total - len(dead_ids)}); "
            f"(1, 1) placeholders={n_placeholders_after}"
        )
        errors = ir_graph.validate()
        if errors:
            print(f"  validate() errors after remove_nodes:", file=sys.stderr)
            for e in errors:
                print(f"    {e}", file=sys.stderr)
            return 1
        print("  validate() returns no errors")
    else:
        print("  no DEAD nodes detected; nothing to remove")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument(
        "--spiking-mode",
        default=None,
        help="Override spiking_mode (default: read from run config)",
    )
    args = parser.parse_args()
    sys.exit(main(args.run_dir, spiking_mode=args.spiking_mode))
