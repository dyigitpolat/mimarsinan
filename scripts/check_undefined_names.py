#!/usr/bin/env python3
"""Fail when pyflakes reports new ``undefined name`` issues under src/mimarsinan.

Uses a checked-in baseline so legacy gaps do not block CI while refactors cannot
introduce fresh NameError-class bugs (e.g. ``SegmentSpikeRecord`` without import).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "src" / "mimarsinan"
BASELINE = Path(__file__).with_name("undefined_names_baseline.txt")
_SKIP = "mimarsinan-baseline-test"

_UNDEFINED_RE = re.compile(
    r"^(?P<path>.+):(?P<lineno>\d+):\d+: undefined name '(?P<name>[^']+)'$"
)


def _run_pyflakes() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pyflakes", str(PKG_ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
    hits: list[str] = []
    for line in lines:
        if _SKIP in line:
            continue
        m = _UNDEFINED_RE.match(line.strip())
        if m is None:
            continue
        path = Path(m.group("path"))
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        hits.append(f"{rel}:{m.group('lineno')}: {m.group('name')}")
    return sorted(set(hits))


def _load_baseline() -> set[str]:
    if not BASELINE.is_file():
        return set()
    rows: set[str] = set()
    for line in BASELINE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.add(line)
    return rows


def _write_baseline(rows: list[str]) -> None:
    header = (
        "# One entry per line: relative/path.py:lineno: SymbolName\n"
        "# Regenerate: python scripts/check_undefined_names.py --update-baseline\n"
    )
    BASELINE.write_text(header + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite undefined_names_baseline.txt from current pyflakes output.",
    )
    args = parser.parse_args(argv)

    try:
        import pyflakes  # noqa: F401
    except ImportError:
        print("pyflakes is required: pip install pyflakes", file=sys.stderr)
        return 1

    current = _run_pyflakes()
    if args.update_baseline:
        _write_baseline(current)
        print(f"Wrote {len(current)} baseline entries to {BASELINE.name}")
        return 0

    baseline = _load_baseline()
    current_set = set(current)
    new_hits = sorted(current_set - baseline)
    fixed_hits = sorted(baseline - current_set)

    if new_hits or fixed_hits:
        print("Undefined name check failed:\n", file=sys.stderr)
        if new_hits:
            print("  New undefined names (add import or fix refactor):", file=sys.stderr)
            for line in new_hits:
                print(f"    + {line}", file=sys.stderr)
        if fixed_hits:
            print("  Removed from baseline (run --update-baseline):", file=sys.stderr)
            for line in fixed_hits:
                print(f"    - {line}", file=sys.stderr)
        return 1

    print(
        f"No new undefined names ({len(current_set)} baselined, "
        f"{len(list(PKG_ROOT.rglob('*.py')))} files scanned)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
