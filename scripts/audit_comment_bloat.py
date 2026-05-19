#!/usr/bin/env python3
"""Flag Python files under src/mimarsinan with heavy inline comment bloat."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mimarsinan"

MIN_HASH_LINES = 30
MIN_CONSECUTIVE = 4
LONG_LINE = 60


def scan_file(path: Path) -> tuple[int, int, int]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    hash_count = sum(1 for ln in lines if re.match(r"^\s*#", ln))
    long_count = sum(
        1 for ln in lines if re.match(rf"^\s+# .{{{LONG_LINE},}}", ln)
    )
    max_run = 0
    run = 0
    for ln in lines:
        if re.match(r"^\s*#", ln):
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return hash_count, long_count, max_run


def main() -> int:
    issues: list[tuple[Path, int, int, int]] = []
    for path in sorted(SRC.rglob("*.py")):
        h, long_c, max_run = scan_file(path)
        if h >= MIN_HASH_LINES or max_run >= MIN_CONSECUTIVE:
            issues.append((path, h, long_c, max_run))

    issues.sort(key=lambda x: (-x[1], -x[3]))
    for path, h, long_c, max_run in issues:
        rel = path.relative_to(ROOT)
        flags = []
        if h >= MIN_HASH_LINES:
            flags.append(f"#lines={h}")
        if max_run >= MIN_CONSECUTIVE:
            flags.append(f"max_consecutive={max_run}")
        print(f"{rel}: {', '.join(flags)}, long_lines(>{LONG_LINE})={long_c}")

    print(f"\n{len(issues)} file(s) flagged under {SRC}")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
