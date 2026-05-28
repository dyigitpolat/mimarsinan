#!/usr/bin/env python3
"""Verify ``from mimarsinan.<mod> import <name>`` symbols resolve at import time."""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "mimarsinan"
SKIP_SUBSTR = "mimarsinan-baseline-test"


def _scan_paths() -> list[Path]:
    paths: list[Path] = []
    for base in (PKG_ROOT, ROOT / "tests", ROOT / "scripts"):
        if base.is_dir():
            paths.extend(sorted(base.rglob("*.py")))
    for extra in (ROOT / "run.py", ROOT / "src" / "main.py"):
        if extra.is_file():
            paths.append(extra)
    return [p for p in paths if SKIP_SUBSTR not in str(p)]


def _imports_in_file(path: Path) -> list[tuple[int, str, list[str]]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    out: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mimarsinan."):
            names = [a.name for a in node.names if a.name != "*"]
            if names:
                out.append((node.lineno, node.module, names))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod.startswith("mimarsinan."):
                    out.append((node.lineno, mod, []))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)

    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    failures: list[str] = []
    for path in _scan_paths():
        if path.name in ("check_import_symbols.py", "migrate_phase3_imports.py"):
            continue
        for lineno, mod, names in _imports_in_file(path):
            try:
                imported = importlib.import_module(mod)
            except Exception as exc:
                failures.append(f"{path}:{lineno}: import {mod!r} failed: {exc}")
                continue
            for name in names:
                if not hasattr(imported, name):
                    failures.append(
                        f"{path}:{lineno}: {mod}.{name} not found (module loaded from {getattr(imported, '__file__', '?')})"
                    )

    if failures:
        print("Import symbol check failed:\n", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1

    print(f"All mimarsinan import symbols resolved ({len(_scan_paths())} files scanned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
