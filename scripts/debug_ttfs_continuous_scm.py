#!/usr/bin/env python3
"""Compare IR liveness status counts across spiking modes (SCM diagnostic).

Usage::

    python scripts/debug_ttfs_continuous_scm.py RUN_DIR
    python scripts/debug_ttfs_continuous_scm.py path/to/Soft\\ Core\\ Mapping.ir_graph.pickle
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.ir_liveness import NodeLiveness, compute_liveness


MODES = ("lif", "ttfs", "ttfs_quantized")


def _resolve_pickle(path: str) -> str:
    if path.endswith(".pickle"):
        return path
    candidate = os.path.join(path, "Soft Core Mapping.ir_graph.pickle")
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"No IR pickle at {candidate!r}")


def _load_sim_steps(run_dir: str, override: int | None) -> int:
    if override is not None:
        return override
    import json

    for name in ("config.json", "pipeline_config.json"):
        cfg_path = os.path.join(run_dir, name)
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        dp = cfg.get("deployment_parameters", cfg)
        if isinstance(dp, dict) and "simulation_steps" in dp:
            return int(dp["simulation_steps"])
    return 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="Run directory or path to Soft Core Mapping.ir_graph.pickle",
    )
    parser.add_argument("--simulation-steps", type=int, default=None)
    args = parser.parse_args()

    pickle_path = _resolve_pickle(args.path)
    run_dir = args.path if not args.path.endswith(".pickle") else os.path.dirname(
        pickle_path,
    )
    sim_steps = _load_sim_steps(run_dir, args.simulation_steps)

    with open(pickle_path, "rb") as fh:
        graph: IRGraph = pickle.load(fh)

    n_cores = sum(1 for n in graph.nodes if isinstance(n, NeuralCore))
    print(f"IR: {pickle_path}")
    print(f"  NeuralCores: {n_cores}  simulation_steps={sim_steps}")

    by_mode: dict[str, dict[int, NodeLiveness]] = {}
    for mode in MODES:
        result = compute_liveness(
            graph, simulation_steps=sim_steps, spiking_mode=mode,
        )
        by_mode[mode] = dict(result.per_node)
        counts = Counter(s.value for s in result.per_node.values())
        print(f"  [{mode}] {dict(counts)}")

    flipped: list[tuple[int, str]] = []
    lif_map = by_mode["lif"]
    ttfs_map = by_mode["ttfs"]
    for nid in sorted(set(lif_map) | set(ttfs_map)):
        if lif_map.get(nid) == NodeLiveness.DEAD and ttfs_map.get(nid) == NodeLiveness.BIAS_ONLY:
            flipped.append((nid, "DEAD->BIAS_ONLY (continuous TTFS)"))
        elif lif_map.get(nid) == NodeLiveness.BIAS_ONLY and ttfs_map.get(nid) == NodeLiveness.DEAD:
            flipped.append((nid, "BIAS_ONLY->DEAD"))

    print(f"  Nodes flipping DEAD <-> BIAS_ONLY (lif vs ttfs): {len(flipped)}")
    for nid, label in flipped[:20]:
        print(f"    id={nid}: {label}")
    if len(flipped) > 20:
        print(f"    ... and {len(flipped) - 20} more")

    output_ids = set()
    if graph.output_sources is not None:
        for src in graph.output_sources.flatten():
            if hasattr(src, "node_id") and src.node_id >= 0:
                output_ids.add(int(src.node_id))
    reachable_ttfs = sum(
        1
        for nid, st in ttfs_map.items()
        if st != NodeLiveness.DEAD or nid in output_ids
    )
    print(f"  Output-referenced node ids: {sorted(output_ids)}")
    print(f"  Non-DEAD cores under ttfs (incl. output contract): {reachable_ttfs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
