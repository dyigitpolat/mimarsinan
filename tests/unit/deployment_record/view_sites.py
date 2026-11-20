"""[U-ENERGY T-E2a] Every site of the toolchain's source that builds a view a pricer reads (a ``Quantities``), found by
the constructor it calls, never by a list of planes: a call of ``Quantities`` or of the planes' one four-state
constructor ``plane_view``. Each site is keyed by its module and enclosing function, so a line edit never moves it."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple

from unit.deployment_record.time_sites import Site, _scopes

CONSTRUCTORS = ("Quantities", "plane_view")


def view_sites_in(module: str, text: str) -> List[Site]:
    tree = ast.parse(text)
    owner = _scopes(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if func in CONSTRUCTORS:
                out.append(Site(func, f"{module}:{owner.get(node, '')}", f"{module}:{node.lineno}"))
    return out


def view_sites(root: Path, extra: Iterable[Tuple[str, str]] = ()) -> List[Site]:
    modules = [(str(p.relative_to(root)), p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.py"))]
    return [site for module, text in [*modules, *extra] for site in view_sites_in(module, text)]
