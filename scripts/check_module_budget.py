#!/usr/bin/env python3
"""Report files/directories exceeding modularization size budgets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MAX_LOC = 300
MAX_SIBLING_PY = 10

# Modules scheduled for split; exempt until refactored.
ALLOWLIST_FILES: frozenset[str] = frozenset({
    "search/optimizers/agent_evolve_optimizer.py",
    "gui/server.py",
    "visualization/search_viz/report_html.py",
    "models/hybrid_core_flow.py",
    "models/unified_core_flow.py",
    "chip_simulation/sanafe/runner.py",
    "chip_simulation/sanafe/runner_analysis.py",
    "chip_simulation/lava_loihi_runner.py",
    "gui/snapshot/ir_graph_snapshot.py",
    "mapping/verification/layout_verification_stats.py",
    "mapping/pruning/ir_pruning.py",
    "visualization/graphviz/ir.py",
    "visualization/graphviz/hybrid.py",
    "mapping/packing/hybrid_segment.py",
    "mapping/packing/hybrid_build.py",
    "mapping/layout/layout_ir_mapping_fc.py",
    "mapping/packing/softcore_mapping.py",
    "mapping/layout/layout_source_view.py",
    "search/problems/joint_arch_hw_problem.py",
})

# mapping/ root may keep orchestration modules until support/ extraction.
ALLOWLIST_DIRS: frozenset[str] = frozenset({
    "mapping",
    "mapping/packing",
    "pipelining/pipeline_steps",
    "chip_simulation",
    "gui",
    "gui/snapshot",
    "models",
    "models/builders",
    "tuning",
    "transformations",
    "pipelining",
})


def _loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.open(encoding="utf-8"))
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any violation")
    parser.add_argument(
        "--strict-prefix",
        action="append",
        default=[],
        help="With --strict, only enforce LOC budget under these path prefixes",
    )
    parser.add_argument("--max-loc", type=int, default=MAX_LOC)
    parser.add_argument("--max-sibling-py", type=int, default=MAX_SIBLING_PY)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1] / "src" / "mimarsinan"
    loc_violations: list[tuple[int, str]] = []
    dir_violations: list[tuple[int, str]] = []

    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.endswith("__init__.py"):
            continue
        n = _loc(path)
        if n > args.max_loc and rel not in ALLOWLIST_FILES:
            if args.strict_prefix:
                if not any(rel.startswith(p) for p in args.strict_prefix):
                    continue
            loc_violations.append((n, rel))

    for directory in sorted(root.rglob("*")):
        if not directory.is_dir():
            continue
        rel = directory.relative_to(root).as_posix()
        if rel in ALLOWLIST_DIRS or rel == "":
            continue
        siblings = [p for p in directory.iterdir() if p.suffix == ".py" and p.name != "__init__.py"]
        if len(siblings) > args.max_sibling_py:
            dir_violations.append((len(siblings), rel))

    print(f"# module budget ({root})\n")
    if loc_violations:
        print(f"Files > {args.max_loc} LOC:")
        for n, rel in sorted(loc_violations, reverse=True):
            print(f"  {n:5d}  {rel}")
    else:
        print(f"No non-allowlisted files > {args.max_loc} LOC.")

    if dir_violations:
        print(f"\nDirectories with > {args.max_sibling_py} sibling .py files:")
        for n, rel in sorted(dir_violations, reverse=True):
            print(f"  {n:5d}  {rel}/")
    else:
        print(f"\nNo non-allowlisted directories with > {args.max_sibling_py} sibling .py files.")

    if args.strict and (loc_violations or dir_violations):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
