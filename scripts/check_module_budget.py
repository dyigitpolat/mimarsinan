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
    "chip_simulation/certification.py",
    "chip_simulation/cost_extraction.py",
    "chip_simulation/pareto.py",
    "chip_simulation/sanafe/runner/segment_io.py",
    "chip_simulation/semantic_axis_screen.py",
    "chip_simulation/spiking_mode_policy.py",
    "chip_simulation/ttfs/ttfs_executor.py",
    "mapping/mappers/compute_op_mapper.py",
    "mapping/verification/onchip_fraction.py",
    "model_training/basic_trainer.py",
    "pipelining/core/nf_scm_parity.py",
    "pipelining/core/simulation_factory.py",
    "pipelining/pipeline_steps/mapping/soft_core_mapping_step.py",
    "tuning/orchestration/kd_blend_adaptation_tuner.py",
    "tuning/orchestration/smooth_adaptation_cycle.py",
    "tuning/orchestration/smooth_adaptation_run.py",
    "tuning/tuners/ttfs_cycle_adaptation_tuner.py",
})

# Directories scheduled for split; shrink this set as refactors land.
ALLOWLIST_DIRS: frozenset[str] = frozenset({
    "chip_simulation",
    "mapping/mappers",
    "mapping/packing",
    "mapping/support",
    "spiking",
    "tuning/orchestration",
})


def _loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.open(encoding="utf-8"))
    except OSError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any violation")
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
