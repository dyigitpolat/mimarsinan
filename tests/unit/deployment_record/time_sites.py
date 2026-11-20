"""[U-ENERGY T-E1a] Every site of the toolchain's source that produces a time, found by its dimension, never by a list
of packages or modules.

A site is (1) a term constructor (``term``, ``CostTerm``, ``modeled``, ``derived``, ``measured``, the load law's
``_put``) whose unit is the literal "s" or a name (a unit given by a name is a site too, resolved by its table entry);
(2) a dict literal key that names a time (it ends in "_s"); (3) a quantity of the catalog whose dimension is a time.
Each site is keyed by its module and what it produces: a literal name by the name, a computed name by its enclosing
function and expression (a catalog quantity by its key), so a line edit never moves a key."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

#: Term constructors and the positions of (name, unit) among their arguments.
CONSTRUCTORS: Dict[str, Tuple[int, int]] = {"term": (0, 1), "CostTerm": (0, 1), "modeled": (0, 1), "derived": (0, 1),
                                            "measured": (0, 1), "_put": (1, 2)}
TIME_UNITS = {"s"}


@dataclass(frozen=True)
class Site:
    kind: str  # "term", "term (unit by name)", "field", "quantity"
    key: str
    where: str  # module:line, for the reader only


def _arg(call: ast.Call, position: int, keyword: str) -> Optional[ast.expr]:
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return call.args[position] if len(call.args) > position else None


def _scopes(tree: ast.AST) -> Dict[ast.AST, str]:
    """Each node's enclosing qualified function or class name."""
    owner: Dict[ast.AST, str] = {}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            owner[child] = inner
            visit(child, inner)

    visit(tree, "")
    return owner


def _local_non_terms(tree: ast.AST) -> set:
    """Names of the module's own functions that share a constructor's name but take no unit at its position (a
    module's own helper, such as a view builder's ``term`` that puts a value)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in CONSTRUCTORS:
            params = [a.arg for a in node.args.args]
            at_unit = CONSTRUCTORS[node.name][1] + (1 if params[:1] in (["self"], ["cls"]) else 0)
            if len(params) <= at_unit or params[at_unit] != "unit":
                out.add(node.name)
    return out


def sites_in(module: str, text: str) -> List[Site]:
    """The time sites of one module's source (``module`` its path under the package root)."""
    tree = ast.parse(text)
    owner = _scopes(tree)
    helpers = _local_non_terms(tree)
    out: List[Site] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if func not in CONSTRUCTORS or (isinstance(node.func, ast.Name) and func in helpers):
                continue
            at_name, at_unit = CONSTRUCTORS[func]
            unit, name = _arg(node, at_unit, "unit"), _arg(node, at_name, "name")
            if unit is None or name is None:
                continue
            literal = name.value if isinstance(name, ast.Constant) and isinstance(name.value, str) else None
            key = f"{module}:{literal}" if literal else f"{module}:{owner.get(node, '')}:{ast.unparse(name)}"
            if isinstance(unit, ast.Constant) and unit.value in TIME_UNITS:
                out.append(Site("term", key, f"{module}:{node.lineno}"))
            elif not isinstance(unit, ast.Constant):
                out.append(Site("term (unit by name)", f"{module}:{owner.get(node, '')}:{ast.unparse(name)} "
                                f"[unit {ast.unparse(unit)}]", f"{module}:{node.lineno}"))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.endswith("_s"):
                    out.append(Site("field", f"{module}:{key.value}", f"{module}:{key.lineno}"))
    return out


def source_sites(root: Path, extra: Iterable[Tuple[str, str]] = ()) -> List[Site]:
    """Every time site of every module under ``root`` (a package root, e.g. ``src/mimarsinan``), with ``extra``
    (module, text) pairs scanned beside them (a planted module in the ratchet's own test)."""
    modules = [(str(p.relative_to(root)), p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.py"))]
    return [site for module, text in [*modules, *extra] for site in sites_in(module, text)]


def catalog_sites(specs: Dict[str, object]) -> List[Site]:
    """Every catalog quantity whose dimension is a time."""
    return [Site("quantity", f"quantity:{key}", "quantities/spec.py") for key, spec in specs.items()
            if str(getattr(spec, "dimension", "")).lower() == "time" or str(getattr(spec, "unit", "")) == "s"]


def unclassified(sites: Iterable[Site], table: Dict[str, Tuple[str, str]]) -> List[Site]:
    return [site for site in sites if site.key not in table]


def stale(sites: Iterable[Site], table: Dict[str, Tuple[str, str]]) -> List[str]:
    keys = {site.key for site in sites}
    return sorted(key for key in table if key not in keys)
