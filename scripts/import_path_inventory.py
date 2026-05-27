#!/usr/bin/env python3
"""List mimarsinan import paths and flag moves that lack compatibility shims."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+(mimarsinan\.[^\s]+)\s+import|import\s+(mimarsinan\.[^\s]+))"
)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "mimarsinan"


def _scan_imports(root: Path) -> dict[str, list[str]]:
    counts: dict[str, list[str]] = {}
    for path in root.rglob("*.py"):
        if path.name == "import_path_inventory.py":
            continue
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only show modules imported at least this many times",
    )
    parser.add_argument(
        "--check-shim",
        action="store_true",
        help="Report canonical paths that were moved but still have a .py shim at the old path",
    )
    args = parser.parse_args(argv)

    root = _package_root()
    pkg_parent = root.parent
    counts = _scan_imports(root)
    # also scan tests
    tests = pkg_parent.parent / "tests"
    if tests.is_dir():
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
        if args.check_shim and path and " # moved" in path.read_text(encoding="utf-8")[:200]:
            print(f"        shim at {path.relative_to(root.parent.parent)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
