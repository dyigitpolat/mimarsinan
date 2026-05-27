"""Every package directory under src/mimarsinan has at most 10 direct .py siblings."""

from __future__ import annotations

from pathlib import Path

MAX_SIBLING_PY = 10
ROOT = Path(__file__).resolve().parents[3] / "src" / "mimarsinan"


def test_package_flatness():
    violations: list[tuple[int, str]] = []
    for directory in sorted(ROOT.rglob("*")):
        if not directory.is_dir():
            continue
        siblings = [
            p for p in directory.iterdir()
            if p.suffix == ".py" and p.name != "__init__.py"
        ]
        if len(siblings) > MAX_SIBLING_PY:
            rel = directory.relative_to(ROOT).as_posix()
            violations.append((len(siblings), rel))

    assert not violations, (
        "Directories with >10 direct non-__init__ .py files:\n"
        + "\n".join(f"  {n:3d}  {rel}/" for n, rel in sorted(violations, reverse=True))
    )
