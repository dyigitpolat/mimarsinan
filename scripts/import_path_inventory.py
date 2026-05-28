#!/usr/bin/env python3
"""List mimarsinan import paths and flag banned legacy shim paths."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(mimarsinan\.[^\s#]+)\s+import|import\s+(mimarsinan\.[^\s#]+))"
)

# Legacy paths removed in phase 2 (no compatibility shims).
LEGACY_BANNED: tuple[str, ...] = (
    "mimarsinan.mapping.core_packing",
    "mimarsinan.mapping.hybrid_hardcore_mapping",
    "mimarsinan.mapping.softcore_mapping",
    "mimarsinan.mapping.ir_pruning",
    "mimarsinan.mapping.chip_latency",
    "mimarsinan.mapping.ir_latency",
    "mimarsinan.mapping.pruning_propagation",
    "mimarsinan.mapping.pruning_graph_propagation",
    "mimarsinan.mapping.platform_constraints",
    "mimarsinan.mapping.mapping_structure",
    "mimarsinan.mapping.layout_verification_stats",
    "mimarsinan.mapping.mapping_verifier",
    "mimarsinan.visualization.mapping_graphviz",
    "mimarsinan.visualization.search_visualization",
    "mimarsinan.pipelining.pipeline_steps.sanafe_simulation_step",
    "mimarsinan.pipelining.pipeline_steps.hard_core_mapping_step",
    "mimarsinan.pipelining.pipeline_steps.soft_core_mapping_step",
    "mimarsinan.pipelining.pipeline_steps.architecture_search_step",
)

# Phase 3 top-level modules moved into subpackages (no shims).
PHASE3_BANNED: tuple[str, ...] = (
    "mimarsinan.gui.data_collector",
    "mimarsinan.gui.persistence",
    "mimarsinan.gui.process_manager",
    "mimarsinan.gui.active_run_stream",
    "mimarsinan.gui.composite_reporter",
    "mimarsinan.gui.snapshot_executor",
    "mimarsinan.gui.run_cache_seed",
    "mimarsinan.models.hybrid_core_flow",
    "mimarsinan.models.unified_core_flow",
    "mimarsinan.models.lif_core_step",
    "mimarsinan.models.ttfs_kernels",
    "mimarsinan.models.layers",
    "mimarsinan.models.decorators",
    "mimarsinan.pipelining.model_registry",
    "mimarsinan.pipelining.trainer_factory",
    "mimarsinan.pipelining.pipeline_helpers",
    "mimarsinan.pipelining.simulation_factory",
    "mimarsinan.mapping.spike_source_spans",
    "mimarsinan.mapping.schedule_partitioner",
    "mimarsinan.mapping.activation_scales",
    "mimarsinan.chip_simulation.ttfs_segment",
    "mimarsinan.chip_simulation.ttfs_kernels",
    "mimarsinan.chip_simulation.spike_recorder",
    "mimarsinan.chip_simulation.hybrid_stage_runner",
    "mimarsinan.mapping.packing.softcore_mapping",
    "mimarsinan.transformations.perceptron_transformer",
    "mimarsinan.search.optimizers.agent_evolve_optimizer",
    "mimarsinan.visualization.search_viz.report_html",
)

# Bare ``pipelining.pipeline`` without ``pipeline_steps`` or ``core.`` subpath.
_PIPELINE_BARE_RE = re.compile(
    r"mimarsinan\.pipelining\.pipeline(?!_steps|\.core|lines)"
)

_SKIP_FILES = frozenset({
    "migrate_phase2_imports.py",
    "migrate_phase3_imports.py",
    "import_path_inventory.py",
    "phase3_wave1_moves.sh",
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _package_root() -> Path:
    return _repo_root() / "src" / "mimarsinan"


def _collect_scan_paths() -> list[Path]:
    root = _repo_root()
    pkg = _package_root()
    paths: list[Path] = list(pkg.rglob("*.py"))
    tests = root / "tests"
    if tests.is_dir():
        paths.extend(tests.rglob("*.py"))
    scripts = root / "scripts"
    if scripts.is_dir():
        paths.extend(scripts.rglob("*.py"))
    for extra in (root / "run.py", root / "src" / "main.py"):
        if extra.is_file():
            paths.append(extra)
    return [
        p
        for p in paths
        if p.is_file()
        and "mimarsinan-baseline-test" not in str(p)
        and p.name not in _SKIP_FILES
    ]


def _imports_from_paths(paths: list[Path]) -> dict[str, list[tuple[str, int]]]:
    counts: dict[str, list[tuple[str, int]]] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            mod = (m.group(1) or m.group(2)).split(" as ")[0].strip()
            counts.setdefault(mod, []).append((str(path), i))
    return counts


def _module_path(mod: str, root: Path) -> Path | None:
    rel = mod.removeprefix("mimarsinan.").replace(".", "/")
    direct = root / f"{rel}.py"
    if direct.is_file():
        return direct
    init = root / rel / "__init__.py"
    if init.is_file():
        return init
    return None


def _check_banned_in_lines(
    paths: list[Path],
    banned: tuple[str, ...],
    *,
    extra_line_check: re.Pattern[str] | None = None,
) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for token in banned:
                if token in line:
                    hits.append((token, str(path), f"L{i}"))
            if extra_line_check and extra_line_check.search(line):
                hits.append((extra_line_check.pattern, str(path), f"L{i}"))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only show modules imported at least this many times",
    )
    parser.add_argument(
        "--check-legacy-path",
        action="store_true",
        help="Fail if banned phase-2 legacy import paths appear",
    )
    parser.add_argument(
        "--check-phase3-paths",
        action="store_true",
        help="Fail if banned phase-3 moved import paths appear",
    )
    parser.add_argument(
        "--check-resolved",
        action="store_true",
        help="Fail if any mimarsinan import path has no module file",
    )
    args = parser.parse_args(argv)

    root = _package_root()
    scan_paths = _collect_scan_paths()
    counts = _imports_from_paths(scan_paths)

    if args.check_legacy_path:
        hits = _check_banned_in_lines(scan_paths, LEGACY_BANNED)
        if hits:
            print("Banned legacy import paths found:", file=sys.stderr)
            for banned, path, loc in hits:
                print(f"  {banned}  {path}:{loc}", file=sys.stderr)
            return 1
        print("No banned legacy import paths.")
        return 0

    if args.check_phase3_paths:
        hits = _check_banned_in_lines(
            scan_paths,
            PHASE3_BANNED,
            extra_line_check=_PIPELINE_BARE_RE,
        )
        if hits:
            print("Banned phase-3 import paths found:", file=sys.stderr)
            for banned, path, loc in hits:
                print(f"  {banned}  {path}:{loc}", file=sys.stderr)
            return 1
        print("No banned phase-3 import paths.")
        return 0

    if args.check_resolved:
        missing: list[tuple[str, str, int]] = []
        for mod, sites in sorted(counts.items()):
            if _module_path(mod, root) is None:
                for path, lineno in sites:
                    missing.append((mod, path, lineno))
        if missing:
            print("Unresolved mimarsinan import paths:", file=sys.stderr)
            for mod, path, lineno in missing:
                print(f"  {mod}  {path}:L{lineno}", file=sys.stderr)
            return 1
        print(f"All {len(counts)} mimarsinan import paths resolve to modules.")
        return 0

    ranked = sorted(
        ((mod, len(sites)) for mod, sites in counts.items()),
        key=lambda x: (-x[1], x[0]),
    )
    print(f"# mimarsinan import inventory ({root})\n")
    for mod, n in ranked:
        if n < args.min_count:
            continue
        path = _module_path(mod, root)
        status = "ok" if path else "MISSING"
        print(f"{n:5d}  {status:7s}  {mod}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
