#!/usr/bin/env python3
"""List mimarsinan import paths and flag banned legacy shim paths."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(mimarsinan\.[^\s]+)\s+import|import\s+(mimarsinan\.[^\s]+))"
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


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "mimarsinan"


def _scan_imports(root: Path) -> dict[str, list[str]]:
    counts: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            m = _IMPORT_RE.match(line)
            if not m:
                continue
            mod = m.group(1) or m.group(2)
            mod = mod.split(" as ")[0].strip()
            counts.setdefault(mod, []).append(str(path))
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


def _check_legacy(paths: list[Path]) -> list[tuple[str, str, str]]:
    hits: list[tuple[str, str, str]] = []
    for path in paths:
        if "migrate_phase2_imports.py" in str(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for banned in LEGACY_BANNED:
                if banned in line:
                    hits.append((banned, str(path), f"L{i}"))
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
        help="Fail if banned legacy import paths appear in src/ or tests/",
    )
    args = parser.parse_args(argv)

    root = _package_root()
    pkg_parent = root.parent
    counts = _scan_imports(root)
    tests = pkg_parent.parent / "tests"
    scan_paths = list(root.rglob("*.py"))
    if tests.is_dir():
        scan_paths.extend(tests.rglob("*.py"))
        for path in tests.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                m = _IMPORT_RE.match(line)
                if not m:
                    continue
                mod = (m.group(1) or m.group(2)).split(" as ")[0].strip()
                counts.setdefault(mod, []).append(str(path))

    if args.check_legacy_path:
        hits = _check_legacy(scan_paths)
        if hits:
            print("Banned legacy import paths found:", file=sys.stderr)
            for banned, path, loc in hits:
                print(f"  {banned}  {path}:{loc}", file=sys.stderr)
            return 1
        print("No banned legacy import paths.")
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
